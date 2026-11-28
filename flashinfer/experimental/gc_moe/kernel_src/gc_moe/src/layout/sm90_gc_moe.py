"""SM90-owned workspace ABI, independent of the SM100 ring layout."""
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType


def align(n, a=128):
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class View:
    offset: int
    shape: tuple
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class SM90GCBufferLayout:
    ranks: int
    experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    shared_experts: int = 0

    def __post_init__(self):
        ranks, experts, capacity, topk = self.ranks, self.experts, self.capacity, self.topk
        hidden, intermediate, shared_experts = self.hidden, self.intermediate, self.shared_experts
        views = {}
        cursor = 0
        def reserve(name, shape, dtype, itemsize):
            nonlocal cursor
            cursor = align(cursor)
            size = itemsize
            for dim in shape:
                size *= dim
            views[name] = View(cursor, shape, dtype, size)
            object.__setattr__(self, name, cursor)
            cursor += size
        local = experts // ranks
        rows = ranks * capacity
        # Header contains the uint32 epoch and per-call publication/counter state.
        # Epoch equality checks remain valid across modulo-2**32 wraparound.
        reserve('header', (32,), 'int32', 4)
        reserve('peer_table', (ranks,), 'int64', 8)
        reserve('expert_ready', (local,), 'int32', 4)
        reserve('expert_count', (local,), 'int32', 4)
        reserve('l1_done', (local,), 'int32', 4)
        reserve('active_expert', (local,), 'int32', 4)
        reserve('active_ready', (local,), 'int32', 4)
        reserve('compute_ready', (ranks,), 'int32', 4)
        reserve('combine_block_done', (1024,), 'int32', 4)
        reserve('send_count', (experts,), 'int32', 4)
        reserve('source_index', (local, ranks, capacity), 'int32', 4)
        reserve('x', (capacity, hidden), 'float8_e4m3fn', 1)
        reserve('x_sf', (capacity, hidden // 128), 'float32', 4)
        reserve('topk_idx', (capacity, topk), 'int64', 8)
        reserve('topk_weights', (capacity, topk), 'float32', 4)
        reserve('route_meta', (local, rows, 3), 'int32', 4)
        reserve('route_weight', (local, rows), 'float32', 4)
        reserve('l1_acts', (local, rows, hidden), 'float8_e4m3fn', 1)
        reserve('l1_acts_sf', (local, hidden // 128, rows), 'float32', 4)
        reserve('l2_acts', (local, rows, intermediate), 'float8_e4m3fn', 1)
        reserve('l2_acts_sf', (local, intermediate // 128, rows), 'float32', 4)
        reserve('combine', (capacity, topk, hidden), 'bfloat16', 2)
        if shared_experts:
            reserve('shared_l1_acts', (capacity, hidden), 'float8_e4m3fn', 1)
            reserve('shared_l1_acts_sf', (capacity, hidden // 32), 'float32', 4)
            reserve('shared_l2_acts', (capacity, intermediate * shared_experts), 'float8_e4m3fn', 1)
            reserve('shared_l2_acts_sf', (capacity, intermediate * shared_experts // 32), 'float32', 4)
            reserve('shared_l1_done', ((capacity + 7) // 8,), 'int32', 4)
            reserve('shared_y', (capacity, hidden), 'bfloat16', 2)
        object.__setattr__(self, 'views', MappingProxyType(views))
        object.__setattr__(self, 'nbytes', align(cursor))


def get_role_pipeline_geometry(tile_m: int, policy: str) -> tuple[int, int]:
    """Pair pipeline depth and epilogue rows for host and kernel budgets."""
    if policy == 'legacy':
        return (5 if tile_m == 64 else (3 if tile_m == 32 else 4)), tile_m
    if policy == 'pipelined':
        return (5 if tile_m in (8, 64) else 4), (16 if tile_m == 32 else tile_m)
    raise ValueError(f'unknown SM90 schedule policy: {policy}')


@dataclass(frozen=True)
class SharedStorageLayout:
    routed_x_offset: int
    sf_stage_offset: int
    stage_bytes: int
    stages: int
    full_offset: int
    empty_offset: int
    task_full_offset: int
    task_empty_offset: int
    task_record_offset: int
    command_offset: int
    epilogue_rows: int


@lru_cache(maxsize=32)
def get_role_shared_layout(tile_m: int, warp_specialized: bool,
                           routed_n_factor: int, stage_count_override: int = 0,
                           routed_k_pack: int = 1,
                           *, policy: str) -> SharedStorageLayout:
    # Each N256 weight stream is 16384B FP4 + 512B 4-bit exponents.
    # Keep 512B padding per stream before the activation region: the wgmma
    # 128B-swizzle descriptor requires a 1024B-aligned matrix base.
    if routed_k_pack not in (1, 2):
        raise ValueError('routed K packing must cover one or two K128 records')
    routed_x = 17408 * routed_n_factor * routed_k_pack
    stage_m = 8 if routed_n_factor == 2 else (tile_m if warp_specialized else 64)
    sf_stage = routed_x + stage_m * 128 * routed_k_pack
    stage = align(sf_stage + stage_m * 4 * routed_k_pack, 1024)
    stages, epilogue_rows = get_role_pipeline_geometry(tile_m, policy)
    if routed_n_factor == 2:
        # The legacy N512 M8 branch shares a fixed 73 856 B budget.
        stages = 4
    if stage_count_override:
        qualified = ((stage_count_override == 6 and tile_m == 64
                      and warp_specialized and routed_n_factor == 1
                      and routed_k_pack == 1)
                     or (stage_count_override == 5 and tile_m == 8
                         and warp_specialized and routed_n_factor == 1
                         and routed_k_pack == 2))
        if not qualified:
            raise ValueError('stage override requires a qualified routed pipeline')
        stages = stage_count_override
    # Pipelined M32 trades half of its epilogue slab for a fourth weight stage;
    # legacy M32 retains three stages and the full 32-row epilogue.
    full = stages * stage
    empty = full + stages * 8
    task_full = empty + stages * 8
    return SharedStorageLayout(routed_x, sf_stage, stage, stages, full, empty,
                               task_full, task_full + 16, task_full + 64,
                               task_full + 160, epilogue_rows)
