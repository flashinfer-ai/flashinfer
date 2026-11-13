"""SM90 role geometry and shared-memory budgets for resident compute CTAs."""
from dataclasses import dataclass, replace
from functools import lru_cache

from ...layout.sm90_gc_moe import get_role_pipeline_geometry


def select_sm90_schedule_policy(total_sms: int) -> str:
    """Select from the device-visible SM resource before communication splitting.

    The 78-SM H20 keeps its legacy geometry. Full 132-SM H100/H200 devices
    share the pipelined resource class; smaller or partitioned devices retain
    the conservative schedule. Changing the dispatch split cannot change class.
    """
    if isinstance(total_sms, bool) or not isinstance(total_sms, int) or total_sms <= 0:
        raise ValueError('total_sms must be a positive integer')
    return 'pipelined' if total_sms >= 128 else 'legacy'


@dataclass(frozen=True)
class SM90RoleConfig:
    grids: tuple[int, int, int]
    tile_m: int = 32
    threads: tuple[int, int, int] = (256, 256, 256)
    # Dispatch allocates a dynamic shared uint32 counter; the graph must reserve it.
    smem: tuple[int, int, int] = (128, 65792, 0)
    shared_grid: int = 0
    policy: str = 'legacy'
    shared_scale_prefetch: bool = False
    routed_scale_preweighted: bool = False
    shared_tile_m: int = 0
    shared_smem: int = 0
    selected_combine: bool = False
    shared_cache_normal: bool = False
    split_shared: bool = False
    shared_l2_grid: int = 0
    compute_sms: int = 0


@lru_cache(maxsize=256)
def _get_sm90_role_config(config, dispatch_sms: int, compute_sms: int, policy: str):
    if policy not in ('legacy', 'pipelined'):
        raise ValueError(f'unknown SM90 schedule policy: {policy}')
    small_pipelined = 0 < config.bound <= (16 if policy == 'pipelined' else 2)
    pipelined = small_pipelined or config.bound == 0 or config.bound >= 32
    high_sm = policy == 'pipelined'
    has_shared = high_sm and config.shared_experts > 0
    # Common public bound controls geometry only. Every received row is still
    # processed by the kernel's complete M-chunk loop, including skewed loads.
    expected_rows = (config.bound or config.capacity) * config.ranks * config.topk / config.experts
    if pipelined:
        if small_pipelined:
            tile_m = 8
        elif expected_rows <= 12:
            tile_m = 24 if high_sm else 16
        else:
            tile_m = 32 if expected_rows <= 24 else 64
    else:
        tile_m = 8 if config.bound <= 16 else 16
    workers = (1 if tile_m == 64 else 2) if pipelined else (3 if tile_m == 8 else 2)
    # Small-bound routed decode (T=1/2) runs one CTA per SM so each CTA owns
    # the SM's whole TMA stream. Packing the two adjacent K128 weight records
    # of one task into a single stage halves the per-stage fixed cost; the
    # 198 912 B budget also makes two CTAs physically unable to co-reside, so
    # the grid size and the resource both enforce the single resident CTA.
    deepen_small_ring = (high_sm and small_pipelined and config.bound <= 2
                         and config.shared_experts == 0)
    small_shared = has_shared and small_pipelined
    if small_shared:
        # A shared branch uses the same compute SM budget as routed work.
        workers = 1
    elif has_shared and pipelined and tile_m == 24:
        # M24 shares the compute context with the complete shared branch.
        workers = 1
    if deepen_small_ring:
        workers = 1
    if pipelined:
        stage_bytes = (17408 + tile_m * 132 + 1023) // 1024 * 1024
        stages, epilogue_rows = get_role_pipeline_geometry(tile_m, policy)
        if deepen_small_ring:
            stage_bytes = (17408 * 2 + tile_m * 132 * 2 + 1023) // 1024 * 1024
            stages = 5
        compute_smem = (stages * stage_bytes + 1024 + epilogue_rows * 1024
                        + (0 if tile_m == 64 else 256))
    else:
        compute_smem = 53376
    direct_scan = 0 < config.bound <= 2
    combine_bound = config.bound or config.capacity
    combine_vector = 16 if high_sm and config.hidden % 512 == 0 else 8
    combine_elements = 256 * combine_vector
    combine_grid = min(compute_sms * 4, max(1, (combine_bound * config.hidden + combine_elements - 1) // combine_elements))
    if high_sm and pipelined and tile_m >= 16:
        combine_grid = min(combine_grid, compute_sms * 2)
    compute_grid = compute_sms * workers
    shared_grid = 0
    if small_shared:
        # Reserve room for routed work while covering the largest shared phase.
        m_blocks = (config.bound + tile_m - 1) // tile_m
        shared_tiles = max(config.intermediate * config.shared_experts // 128,
                           config.hidden // 256)
        shared_tasks = m_blocks * shared_tiles
        shared_floor = (compute_sms + 1) // 2
        shared_grid = min(compute_grid, max(shared_floor, shared_tasks))
    elif has_shared and pipelined and tile_m >= 16:
        # Every row is still dynamically queued; omit CTAs beyond either phase.
        shared_bound = config.bound or config.capacity
        m_blocks = (shared_bound + tile_m - 1) // tile_m
        shared_tasks = m_blocks * max(config.intermediate * config.shared_experts // 128,
                                      config.hidden // 256)
        shared_grid = min(compute_grid, max(1, shared_tasks))
    shared_tile_m, shared_smem = 0, 0
    if has_shared and tile_m == 64:
        # Dense local rows use their own M32 parallelism; routed stays M64.
        shared_tile_m = 32
        shared_stage_bytes = (17408 + shared_tile_m * 132 + 1023) // 1024 * 1024
        shared_smem = 3 * shared_stage_bytes + 1024 + shared_tile_m * 1024 + 256
        shared_bound = config.bound or config.capacity
        m_blocks = (shared_bound + shared_tile_m - 1) // shared_tile_m
        shared_tasks = m_blocks * max(config.intermediate * config.shared_experts // 128,
                                      config.hidden // 256)
        shared_grid = min(2 * compute_sms, max(1, shared_tasks))
    return SM90RoleConfig((config.experts // config.ranks if direct_scan else dispatch_sms * 4,
                           compute_grid, combine_grid),
                          shared_grid=shared_grid,
                          policy=policy,
                          # Geometry-derived shared budgets stay separate; this
                          # flag pairs the routed SF writer and consumer.
                          routed_scale_preweighted=(high_sm and pipelined
                                                     and tile_m in (16, 24, 32, 64)),
                          shared_scale_prefetch=(has_shared and pipelined and tile_m >= 16
                                                 and (shared_tile_m or tile_m) >= 32),
                          shared_tile_m=shared_tile_m,
                          shared_smem=shared_smem,
                          selected_combine=(high_sm and pipelined and tile_m >= 16
                                            and config.hidden % 256 == 0
                                            and 1 <= config.topk <= 32),
                          shared_cache_normal=(has_shared and (shared_tile_m or tile_m) >= 32),
                          split_shared=(has_shared and tile_m in (32, 64)),
                          shared_l2_grid=(max(1, compute_sms // 4)
                                          if has_shared and tile_m == 64 else 0),
                          compute_sms=compute_sms,
                          tile_m=tile_m,
                          threads=(128 if direct_scan else 256, 384 if pipelined else 256, 384),
                          smem=(max(128, ((2 * config.ranks + 2) * 4 + 127) // 128 * 128), compute_smem, 0))


def get_sm90_role_config(config, context):
    return _get_sm90_role_config(config, context.dispatch_sms, context.compute_sms,
                                 context.schedule_policy)


@dataclass(frozen=True)
class RoutedRoleConfig:
    schedule: SM90RoleConfig
    n_factor: int
    stage_count_override: int = 0
    k_pack: int = 1


@lru_cache(maxsize=256)
def get_routed_role_config(config, schedule):
    if (schedule.selected_combine and schedule.tile_m == 64
            and schedule.policy == 'pipelined' and schedule.threads[1] == 384
            and schedule.routed_scale_preweighted):
        stage_bytes = (17408 + schedule.tile_m * 132 + 1023) // 1024 * 1024
        routed_smem = 6 * stage_bytes + 1024 + schedule.tile_m * 1024
        return RoutedRoleConfig(replace(schedule,
                                       smem=(schedule.smem[0], routed_smem,
                                             schedule.smem[2])), 1, 6)

    if (config.shared_experts == 0 and 0 < config.bound <= 2
            and schedule.policy == 'pipelined' and schedule.tile_m == 8
            and schedule.threads[1] == 384):
        # Same small-bound class as the role geometry: one resident CTA per
        # SM, with two adjacent K128 weight records packed into every stage of
        # a five-slot ring (kernel stage indexing is generic).
        stage_bytes = (17408 * 2 + schedule.tile_m * 132 * 2 + 1023) // 1024 * 1024
        _, epilogue_rows = get_role_pipeline_geometry(schedule.tile_m, schedule.policy)
        routed_smem = 5 * stage_bytes + 1024 + epilogue_rows * 1024 + 256
        return RoutedRoleConfig(replace(schedule,
                                       smem=(schedule.smem[0], routed_smem,
                                             schedule.smem[2])), 1, 5, 2)

    if (config.shared_experts > 0 and schedule.threads[1] == 256
            and schedule.tile_m == 8 and config.hidden % 512 == 0
            and config.intermediate % 256 == 0):
        return RoutedRoleConfig(
            replace(schedule, smem=(schedule.smem[0], 73856, schedule.smem[2])), 2)
    return RoutedRoleConfig(schedule, 1)


@lru_cache(maxsize=256)
def get_shared_role_config(schedule):
    tile_m = schedule.shared_tile_m or schedule.tile_m
    smem = schedule.smem
    if schedule.shared_smem:
        smem = (smem[0], schedule.shared_smem, smem[2])
    elif schedule.threads[1] != 384:
        stage_stride = (36864 + tile_m * 144 + 1023) // 1024 * 1024
        shared_smem = (2 * stage_stride + 128 + 1023) // 1024 * 1024
        smem = (smem[0], shared_smem, smem[2])
    return replace(schedule, tile_m=tile_m,
                   threads=(schedule.threads[0], 256, schedule.threads[2]),
                   smem=smem,
                   grids=(schedule.grids[0], schedule.shared_grid or schedule.grids[1],
                          schedule.grids[2]))
