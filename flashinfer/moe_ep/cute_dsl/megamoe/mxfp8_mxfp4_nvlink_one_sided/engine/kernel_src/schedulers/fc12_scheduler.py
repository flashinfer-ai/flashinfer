"""Composable grouped and phase-interleaved FC12 schedulers."""

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Integer,
    extract_mlir_values,
    new_from_mlir_values,
)

from ...api import (
    ImplDesc,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import spin_peek, spin_wait
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div
from .base import SchedulerBase, SchedulerWorkTileBase, WorkIdAcquisitionMode
from .fc12_mapping import (
    BlockPhase,
    NonSwapAbFc12WorkTileInfo,
    SharedExpertTiles,
    SwapAbFc12WorkTileInfo,
    count_fc12_tiles,
    count_fc12_tiles_from_sizes,
    create_fc12_task_mapping_state,
    create_phase_interleaved_fc12_mapping_state,
    load_fc12_token_count_cache,
    make_fc12_done_tile,
    map_fc12_linear_work_id,
    map_fc12_pipelined_work_id,
    map_phase_interleaved_fc12_work_id,
    map_shared_expert_work_id,
    resolve_phase_interleaved_fc1_claim_target,
)
from ...helpers.ptx_helpers import nanosleep
from ....comm import ptx as sentinel_ptx
from .non_clc_mixed_cga import NonClcMixedCgaConfig, NonClcMixedCgaSchedulerWorker


def _make_non_clc_mixed_cga_config(impl_desc: ImplDesc) -> NonClcMixedCgaConfig:
    return NonClcMixedCgaConfig(
        preferred_cluster_shape=impl_desc["cluster_shape_mn"],
        fallback_cluster_shape=impl_desc.get("fallback_cluster_shape_mn"),
        launch_cluster_count=impl_desc.get("launch_cluster_count"),
        preferred_cluster_count=impl_desc.get("preferred_cluster_count"),
        fallback_cluster_count=impl_desc.get("fallback_cluster_count"),
    )


def _mixed_cga_impl_requirements() -> dict:
    return {
        "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
        "launch_cluster_count": OptionalRequirement(Optional[int]),
        "preferred_cluster_count": OptionalRequirement(Optional[int]),
        "fallback_cluster_count": OptionalRequirement(Optional[int]),
    }


def minimum_phase_interleave_fc1_claims(
    *, blocks_fc1: int, blocks_fc2: int, launch_cluster_cnt_merge_as_preferred: int
) -> int:
    """Return the global FC1 claim watermark covering one canonical FC2 wave."""
    effective_wave_width = launch_cluster_cnt_merge_as_preferred
    max_dependent_token_blocks = ceil_div(
        effective_wave_width + blocks_fc2 - 1, blocks_fc2
    )
    return max_dependent_token_blocks * blocks_fc1


def minimum_phase_interleave_hint(
    *, blocks_fc1: int, blocks_fc2: int, launch_cluster_cnt_merge_as_preferred: int
) -> int:
    """Return the per-cluster FC1 prologue covering one canonical FC2 claim wave."""
    required_fc1_claims = minimum_phase_interleave_fc1_claims(
        blocks_fc1=blocks_fc1,
        blocks_fc2=blocks_fc2,
        launch_cluster_cnt_merge_as_preferred=launch_cluster_cnt_merge_as_preferred,
    )
    return max(1, ceil_div(required_fc1_claims, launch_cluster_cnt_merge_as_preferred))


def _to_fc12_mapping_cta_coord(
    cta_coord_in_preferred_cluster: cute.Coord, is_swap_ab: bool
) -> cute.Coord:
    if cutlass.const_expr(not is_swap_ab):
        return cta_coord_in_preferred_cluster
    return (
        cta_coord_in_preferred_cluster[1],
        cta_coord_in_preferred_cluster[0],
        cta_coord_in_preferred_cluster[2],
    )


class BlackwellFusedFc12Scheduler(SchedulerBase):
    """Compose work-ID claim, FC12 mapping, and SMEM tile transport.

    Work IDs map to tiles in schedule groups of about ``hint`` FC1 tiles. In the
    grouped order a group's FC2 tiles follow its own FC1 tiles; with
    ``pipelined_groups`` they are claimed together with the next group's FC1
    tiles (``map_fc12_pipelined_work_id``).

    With ``die_local_weights`` the first and second half of every expert's M
    tiles (FC1 and FC2 alike) are two claim streams, one per die, each with its
    own work counter. Both enumerate the same order over their half: the order
    above with half the M tiles per token tile and half the hint, so position
    ``p`` of either stream is the same phase, expert and token tile, and the
    second stream's tiles sit ``M tiles / 2`` higher. A cluster claims from its
    die's stream (``sm_die_table``), and from the other one once its own is
    exhausted or more than ``die_claim_lead_limit`` claims ahead of it.

    An FC2 tile reads the FC1 tiles of its token tile from both streams. Before
    a claimed FC2 tile is published, the other stream's counter must have passed
    the position below which the order puts those FC1 tiles (the end of the
    group's FC1 tiles, or under pipelined groups the start of the segment). FC1
    tiles wait on the communication kernel only, so every tile then waits only on
    tiles whose claims completed before its own, and neither the queues nor the
    gates can form a cycle. While one die's clusters wait there, the other die's
    claim from the stream behind, which waits on nothing, so the counter moves.

    With a shared expert (``shared_expert_tiles``) every claim stream starts with
    the shared FC1 tiles, then holds the shared FC2 tiles either right behind
    them or behind the routed tiles (``SharedExpertTiles``); the routed order
    follows unchanged, shifted by the tiles in front of it. Shared FC1 tiles
    wait on nothing, and a shared FC2 tile reads only shared FC1 tiles, which
    precede it in both streams.

    With the shared tiles' entry head (``SharedExpertTiles.entry_head``) the
    shared FC1 tiles, and the early shared FC2 tiles, are claimed instead from
    one more work counter, the head counter, which clusters drain from kernel
    entry (``gen_next_shared_head_work``) before they claim from the streams;
    the streams then hold the routed tiles and the shared FC2 tiles not in the
    head. A cluster leaves the head at its first claim past it, so every head
    tile has been claimed before any cluster claims from a stream, and a shared
    FC2 tile, in the head or behind the routed tiles, reads only head FC1 tiles.
    """

    pipeline_mbarriers_region = "blackwell.fc12.scheduler.pipeline_mbarriers"
    work_tiles_region = "blackwell.fc12.scheduler.work_tiles"
    # Die-local weights: words [0, slots) hold the die of each SM id, words
    # [slots, 2 * slots) its rank among its die's SMs, and the next two each
    # die's SM count; the host fills it (``die_local.SmDieMap.table``).
    sm_die_table_region = "blackwell.fc12.scheduler.sm_die_table"
    sm_die_table_slots = 256
    # Die-local weights: claims a die's stream may run ahead of the other's
    # before its clusters claim from the other one. The two dies' clusters
    # differ in number (the SM split, the communication kernel's CTAs), so a
    # fixed split of the work would leave the faster die idle or waiting on the
    # slower one's FC1 tiles; a few claims of lead absorb the claims in flight.
    die_claim_lead_limit = 16

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "hint": Optional[int],
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "is_swap_ab": bool,
            "pipelined_groups": OptionalRequirement(bool),
            "die_local_weights": OptionalRequirement(bool),
            "shared_expert_tiles": OptionalRequirement(Optional[SharedExpertTiles]),
            **_mixed_cga_impl_requirements(),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.hint = impl_desc["hint"]
        self.group_hint = self.hint
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.is_swap_ab = impl_desc["is_swap_ab"]
        self.pipelined_groups = impl_desc.get("pipelined_groups", False)
        self.shared_expert_tiles = impl_desc.get("shared_expert_tiles")
        self.non_clc_mixed_cga_config = _make_non_clc_mixed_cga_config(impl_desc)
        self.launch_cluster_cnt_merge_as_preferred = (
            self.non_clc_mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        )
        self.work_tile_type = (
            SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        )
        if self.hint is None:
            self.hint = self.launch_cluster_cnt_merge_as_preferred
            self.group_hint = self.hint
        self.die_local_weights = impl_desc.get("die_local_weights", False)
        # Each die's stream holds half of every token tile's M tiles, so half the
        # hint cuts it into groups of the same experts.
        self.mapping_group_hint = (
            max(1, self.group_hint // 2) if self.die_local_weights else self.group_hint
        )

        self._validate_configuration()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        launch_cta_tile_shape_mnk = (
            self.mma_tiler_mnk[0] // mma_cta_count,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        if self.is_swap_ab:
            self.mapping_cta_tile_shape_mnk = (
                launch_cta_tile_shape_mnk[1],
                launch_cta_tile_shape_mnk[0],
                launch_cta_tile_shape_mnk[2],
            )
            self.mapping_cluster_shape_mn = (
                self.cluster_shape_mn[1],
                self.cluster_shape_mn[0],
            )
        else:
            self.mapping_cta_tile_shape_mnk = launch_cta_tile_shape_mnk
            self.mapping_cluster_shape_mn = self.cluster_shape_mn

        if self.work_id_mode == "cluster_launch_control":
            raise NotImplementedError(
                "cluster_launch_control is not implemented for FC12."
            )
        # The head counter of a shared expert's entry head follows the claim streams' counters.
        self.shared_head_counter_index = 2 if self.die_local_weights else 1
        entry_head = (
            self.shared_expert_tiles is not None and self.shared_expert_tiles.entry_head
        )
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=self.non_clc_mixed_cga_config,
            work_id_mode=self.work_id_mode,
            stream_count=self.shared_head_counter_index + (1 if entry_head else 0),
        )
        if self.die_local_weights:
            # M tiles of each die's half, in CTA tiles (the work tile's tile_m_idx).
            cluster_tile_m = (
                self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
            )
            self.die_half_cta_tiles = tuple(
                size // 2 // self.mapping_cta_tile_shape_mnk[1]
                for size in (self.intermediate_gateup_size, self.hidden_size)
            )
            if (
                not self.is_swap_ab
                or self.work_id_mode != "atomic_counter"
                or self.non_clc_mixed_cga_config.is_mixed
                or not isinstance(self.intermediate_gateup_size, int)
                or self.intermediate_gateup_size % (2 * cluster_tile_m)
                or self.hidden_size % (2 * cluster_tile_m)
            ):
                raise ValueError(
                    "die_local_weights needs swap-AB atomic-counter claims on a uniform launch and static "
                    "dimensions that split into whole cluster tiles per die."
                )
        if self.shared_expert_tiles is not None and (
            not self.is_swap_ab
            or self.work_id_mode != "atomic_counter"
            or self.non_clc_mixed_cga_config.is_mixed
            or self.mapping_cluster_shape_mn[0] != 1
            or not all(
                isinstance(size, int)
                for size in (self.intermediate_gateup_size, self.hidden_size)
            )
        ):
            raise ValueError(
                "A shared expert needs swap-AB atomic-counter claims on a uniform launch, one token tile per "
                "cluster tile and static expert dimensions."
            )

    def _validate_configuration(self) -> None:
        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if self.group_hint is not None and self.group_hint <= 0:
            raise ValueError("group_hint must be positive.")
        if not isinstance(self.pipelined_groups, bool):
            raise TypeError("pipelined_groups must be a bool.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_cnt_merge_as_preferred <= 0:
            raise ValueError("launch_cluster_cnt_merge_as_preferred must be positive.")
        if self.work_id_mode not in (
            "grid_stride",
            "atomic_counter",
            "cluster_launch_control",
        ):
            raise ValueError(
                "work_id_mode must be 'grid_stride', 'atomic_counter', or "
                f"'cluster_launch_control', got {self.work_id_mode!r}."
            )
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if self.work_id_mode == "atomic_counter" and cluster_size > 32:
            raise ValueError(
                "The atomic broadcast protocol supports at most 32 CTAs per cluster."
            )
        fallback_cluster_shape = self.non_clc_mixed_cga_config.fallback_cluster_shape
        if fallback_cluster_shape is not None:
            fallback_cluster_size = (
                fallback_cluster_shape[0] * fallback_cluster_shape[1]
            )
            if self.work_id_mode == "atomic_counter" and fallback_cluster_size > 32:
                raise ValueError(
                    "The atomic broadcast protocol supports at most 32 CTAs per fallback cluster."
                )
        for field_name in ("expert_count", "intermediate_gateup_size", "hidden_size"):
            value = getattr(self, field_name)
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        static_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_dimensions) and not all(static_dimensions):
            raise ValueError(
                "FC12 expert dimensions must be either all static or all runtime."
            )

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register scheduler-owned SMEM transport and claim regions."""
        super().register_smem_regions(smem_workspace)
        self._work_id_worker.register_smem_regions(smem_workspace)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register work-ID counters and optional fixed-group fallback state."""
        self._work_id_worker.register_device_workspace(device_workspace)
        if self.die_local_weights:
            device_workspace.register(
                self.sm_die_table_region,
                cutlass.Int32,
                (2 * self.sm_die_table_slots + 2,),
                buffer_space="local",
            )

    @cute.jit
    def initialize_fallback_group(self) -> None:
        """Register this physical fallback cluster with its fixed logical group."""
        self._work_id_worker.initialize_fallback_group()

    def get_grid_shape(
        self, *, max_active_clusters: Optional[int] = None, problem_desc=None
    ) -> Tuple[int, int, int]:
        """Return the persistent launch grid in GEMM-domain orientation."""
        if (
            not self.non_clc_mixed_cga_config.is_mixed
            and max_active_clusters is not None
            and max_active_clusters < self.launch_cluster_cnt_merge_as_preferred
        ):
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least "
                "launch_cluster_cnt_merge_as_preferred "
                f"({self.launch_cluster_cnt_merge_as_preferred})."
            )
        return (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            self.launch_cluster_cnt_merge_as_preferred,
        )

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Optional[Tuple],
        block_idx: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
        is_fallback_cluster: Optional[Boolean] = None,
        cache_token_counts: bool = False,
    ) -> None:
        """Initialize all FC12 scheduler state rooted for one CTA lifetime.

        ``cache_token_counts`` gives the mapping a register copy of the
        per-expert token sizes, which ``load_token_count_cache`` fills.
        """
        if cutlass.const_expr(
            (expert_token_sizes is None) == (expert_token_prefix_sum is None)
        ):
            raise ValueError(
                "Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided."
            )
        needs_actual_shape = not all(
            isinstance(dimension, int)
            for dimension in (
                self.expert_count,
                self.intermediate_gateup_size,
                self.hidden_size,
            )
        )
        if cutlass.const_expr(needs_actual_shape and actual_expert_shape is None):
            raise ValueError("actual_expert_shape is required for runtime dimensions.")
        if cutlass.const_expr(isinstance(self.expert_count, int)):
            expert_count = self.expert_count
        else:
            expert_count = actual_expert_shape[0]
        if cutlass.const_expr(isinstance(self.intermediate_gateup_size, int)):
            intermediate_gateup_size = self.intermediate_gateup_size
        else:
            intermediate_gateup_size = actual_expert_shape[1]
        if cutlass.const_expr(isinstance(self.hidden_size, int)):
            hidden_size = self.hidden_size
        else:
            hidden_size = actual_expert_shape[2]

        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self._work_id_worker.assign_device_members(
            is_fallback_cluster=is_fallback_cluster,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
        )

        mapped_sizes = (intermediate_gateup_size, hidden_size)
        if cutlass.const_expr(self.die_local_weights):
            mapped_sizes = (intermediate_gateup_size // 2, hidden_size // 2)
        task_mapping_states = [
            create_fc12_task_mapping_state(
                expert_count=expert_count,
                intermediate_gateup_size=mapped_sizes[0],
                hidden_size=mapped_sizes[1],
                mapping_cta_tile_shape_mnk=(self.mapping_cta_tile_shape_mnk),
                mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
                group_hint=self.mapping_group_hint,
                token_padding_block=self.token_padding_block,
                sf_padding_block=self.sf_padding_block,
                is_swap_ab=self.is_swap_ab,
                expert_token_sizes=expert_token_sizes,
                expert_token_prefix_sum=expert_token_prefix_sum,
                cache_token_counts=cache_token_counts,
                pipelined_groups=self.pipelined_groups,
            )
            for _ in range(2 if self.die_local_weights else 1)
        ]

        self._task_mapping_state = task_mapping_states[0]
        if cutlass.const_expr(self.die_local_weights):
            # One mapping state per die stream, each monotonic in its own IDs.
            self._die_task_mapping_state = task_mapping_states[1]
            self._sm_die_table = device_workspace.ptr(self.sm_die_table_region)
            self._own_die = Int32(0)
            self._exhausted_streams = Int32(0)
        if cutlass.const_expr(self.shared_expert_tiles is not None):
            # Routed tiles of one claim stream; ``load_routed_tile_count`` fills it.
            self._routed_tiles = Int32(0)

    @cute.jit
    def load_sm_die(self) -> None:
        """Read this SM's die from ``sm_die_table``; the claiming warp calls it once the table is final."""
        if cutlass.const_expr(self.die_local_weights):
            self._own_die = cute.arch.load(
                self._sm_die_table + sentinel_ptx.smid(),
                Int32,
                sem="relaxed",
                scope="gpu",
            )

    @cute.jit
    def load_token_count_cache(self) -> None:
        """Copy the per-expert token sizes into the mapping's registers; call once they are final."""
        self._task_mapping_state = load_fc12_token_count_cache(self._task_mapping_state)
        if cutlass.const_expr(self.die_local_weights):
            self._die_task_mapping_state = load_fc12_token_count_cache(
                self._die_task_mapping_state
            )

    @cute.jit
    def count_work_tiles(self) -> Tuple[Int32, Int32]:
        """FC1 and FC2 cluster tiles of this launch, from the token count cache."""
        fc1_tiles, fc2_tiles = count_fc12_tiles(self._task_mapping_state)
        if cutlass.const_expr(self.shared_expert_tiles is not None):
            shared_fc1_tiles, shared_fc2_tiles = self._shared_tile_counts(
                self._task_mapping_state
            )
            fc1_tiles, fc2_tiles = (
                fc1_tiles + Int32(shared_fc1_tiles),
                fc2_tiles + Int32(shared_fc2_tiles),
            )
        if cutlass.const_expr(self.die_local_weights):
            fc1_tiles, fc2_tiles = fc1_tiles * Int32(2), fc2_tiles * Int32(2)
        return fc1_tiles, fc2_tiles

    def _shared_tile_counts(self, mapping_state) -> Tuple[int, int]:
        """Shared FC1 and FC2 cluster tiles of one claim stream (static)."""
        token_tiles = self.shared_expert_tiles.token_tiles
        return (
            token_tiles * mapping_state.num_fc1_intermediate_blocks,
            token_tiles * mapping_state.num_fc2_hidden_blocks,
        )

    def _shared_head_tile_counts(self) -> Tuple[int, int, int, int]:
        """Shared FC1 and FC2 tiles of the entry head and their output blocks per token tile (static).

        Head tiles span every output block of a token tile; the head holds the
        FC2 tiles only where they are early, and the FC1 tiles of the token
        tiles past ``head_token_tiles`` not at all.
        """
        shared = self.shared_expert_tiles
        cluster_tile_m = (
            self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
        )
        fc1_blocks = ceil_div(self.intermediate_gateup_size, cluster_tile_m)
        fc2_blocks = ceil_div(self.hidden_size, cluster_tile_m)
        fc2_tiles = shared.token_tiles * fc2_blocks if shared.fc2_early else 0
        head_token_tiles = shared.token_tiles - shared.streamed_token_tiles
        return head_token_tiles * fc1_blocks, fc2_tiles, fc1_blocks, fc2_blocks

    @cute.jit
    def gen_next_shared_head_work(self) -> SchedulerWorkTileBase:
        """Claim and map one tile of a shared expert's entry head; past its end, the terminal tile.

        Needs no plan: the head's tiles are static. A cluster calls it from
        kernel entry until it returns the terminal tile, then claims from the
        streams (``gen_next_work``).
        """
        if cutlass.const_expr(
            self.shared_expert_tiles is None or not self.shared_expert_tiles.entry_head
        ):
            raise ValueError(
                "gen_next_shared_head_work needs a shared expert's entry head."
            )
        shared = self.shared_expert_tiles
        fc1_tiles, fc2_tiles, fc1_blocks, fc2_blocks = self._shared_head_tile_counts()
        head_id = self._work_id_worker.claim_next_work(
            stream_index=self.shared_head_counter_index
        )
        cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
            self._work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
        )
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        if head_id < Int32(fc1_tiles):
            work_tile = map_shared_expert_work_id(
                head_id,
                BlockPhase.Linear1,
                cta_id_in_mapping_cluster,
                self._task_mapping_state,
                shared,
                fc1_blocks,
            )
        else:
            if cutlass.const_expr(fc2_tiles > 0):
                if head_id < Int32(fc1_tiles + fc2_tiles):
                    work_tile = map_shared_expert_work_id(
                        head_id - Int32(fc1_tiles),
                        BlockPhase.Linear2,
                        cta_id_in_mapping_cluster,
                        self._task_mapping_state,
                        shared,
                        fc2_blocks,
                    )
        return work_tile

    @cute.jit
    def load_routed_tile_count(self) -> None:
        """Count one claim stream's routed tiles, which the shared FC2 tiles behind them need; call once the sizes are final."""
        if cutlass.const_expr(self.shared_expert_tiles is not None):
            fc1_tiles, fc2_tiles = count_fc12_tiles_from_sizes(self._task_mapping_state)
            self._routed_tiles = fc1_tiles + fc2_tiles

    @cute.jit
    def _map_work_id_with_shared(
        self, work_id: Int32, cta_id_in_mapping_cluster: cute.Coord, mapping_state
    ):
        """Map one claim stream's work ID with the shared expert's tiles in front of (and behind) the routed order.

        Returns the tile, the mapping state and the position of the other claim
        stream below which the order puts the FC1 tiles an FC2 tile here reads
        (see ``_gen_next_die_local_work``).
        """
        shared = self.shared_expert_tiles
        shared_fc1_tiles, shared_fc2_tiles = self._shared_tile_counts(mapping_state)
        if cutlass.const_expr(shared.streamed_token_tiles > 0):
            return self._map_work_id_with_streamed_shared(
                work_id, cta_id_in_mapping_cluster, mapping_state
            )
        if cutlass.const_expr(shared.entry_head):
            # The head holds the shared FC1 tiles (and the early FC2 ones), all
            # claimed before any stream claim; nothing precedes the routed tiles.
            work_tile = make_fc12_done_tile(self.is_swap_ab)
            fc1_claims_needed = Int32(0)
            if work_id < self._routed_tiles:
                work_tile, mapping_state, fc1_claims_needed = self._map_routed_work_id(
                    work_id, cta_id_in_mapping_cluster, mapping_state, 0
                )
            else:
                if cutlass.const_expr(not shared.fc2_early):
                    tail_id = work_id - self._routed_tiles
                    if tail_id < Int32(shared_fc2_tiles):
                        work_tile = map_shared_expert_work_id(
                            tail_id,
                            BlockPhase.Linear2,
                            cta_id_in_mapping_cluster,
                            mapping_state,
                            shared,
                        )
            return work_tile, mapping_state, fc1_claims_needed
        prefix_tiles = shared_fc1_tiles + (shared_fc2_tiles if shared.fc2_early else 0)
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        fc1_claims_needed = Int32(shared_fc1_tiles)
        if work_id < Int32(shared_fc1_tiles):
            work_tile = map_shared_expert_work_id(
                work_id,
                BlockPhase.Linear1,
                cta_id_in_mapping_cluster,
                mapping_state,
                shared,
            )
        else:
            routed_id = work_id - Int32(prefix_tiles)
            if cutlass.const_expr(shared.fc2_early):
                if routed_id < Int32(0):
                    work_tile = map_shared_expert_work_id(
                        work_id - Int32(shared_fc1_tiles),
                        BlockPhase.Linear2,
                        cta_id_in_mapping_cluster,
                        mapping_state,
                        shared,
                    )
                elif routed_id < self._routed_tiles:
                    work_tile, mapping_state, fc1_claims_needed = (
                        self._map_routed_work_id(
                            routed_id,
                            cta_id_in_mapping_cluster,
                            mapping_state,
                            prefix_tiles,
                        )
                    )
            else:
                if routed_id < self._routed_tiles:
                    work_tile, mapping_state, fc1_claims_needed = (
                        self._map_routed_work_id(
                            routed_id,
                            cta_id_in_mapping_cluster,
                            mapping_state,
                            prefix_tiles,
                        )
                    )
                else:
                    tail_id = routed_id - self._routed_tiles
                    if tail_id < Int32(shared_fc2_tiles):
                        work_tile = map_shared_expert_work_id(
                            tail_id,
                            BlockPhase.Linear2,
                            cta_id_in_mapping_cluster,
                            mapping_state,
                            shared,
                        )
        return work_tile, mapping_state, fc1_claims_needed

    @cute.jit
    def _map_work_id_with_streamed_shared(
        self, work_id: Int32, cta_id_in_mapping_cluster: cute.Coord, mapping_state
    ):
        """Map one claim stream's work ID where the entry head leaves shared FC1 tiles to the streams.

        The stream's ``S`` streamed shared FC1 tiles (its output blocks of the
        token tiles past the head, token tile major) and its ``R`` routed tiles
        share the positions ``[0, S * (q + 1))``, ``q = R // S``: each run of
        ``q`` routed tiles is followed by one shared tile (all shared tiles
        first where ``q`` is zero). The other routed tiles and then the shared
        FC2 tiles follow. Returns what ``_map_work_id_with_shared`` returns; the
        other stream's position an FC2 tile needs is the position of the routed
        tile that bounds its FC1 tiles, or for a shared FC2 tile the end of the
        shared FC1 tiles.
        """
        shared = self.shared_expert_tiles
        _, shared_fc2_tiles = self._shared_tile_counts(mapping_state)
        blocks = mapping_state.num_fc1_intermediate_blocks
        streamed = Int32(shared.streamed_token_tiles * blocks)
        first_streamed = Int32(
            (shared.token_tiles - shared.streamed_token_tiles) * blocks
        )
        spacing = self._routed_tiles // streamed
        period = spacing + Int32(1)
        streamed_end = streamed * period
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        fc1_claims_needed = Int32(0)
        routed_id = work_id - streamed
        is_routed = Boolean(True)
        if work_id < streamed_end:
            run = work_id // period
            within = work_id - run * period
            routed_id = run * spacing + within
            if within == spacing:
                is_routed = Boolean(False)
                work_tile = map_shared_expert_work_id(
                    first_streamed + run,
                    BlockPhase.Linear1,
                    cta_id_in_mapping_cluster,
                    mapping_state,
                    shared,
                )
        if is_routed:
            if routed_id < self._routed_tiles:
                work_tile, mapping_state = self._map_work_id(
                    routed_id, cta_id_in_mapping_cluster, mapping_state
                )
                if cutlass.const_expr(self.die_local_weights):
                    if cutlass.const_expr(self.pipelined_groups):
                        bound = mapping_state.cursor_state.segment_start
                    else:
                        bound = (
                            mapping_state.cursor_state.current_group_fc1_subphase_end
                        )
                    shift = streamed
                    if spacing > Int32(0):
                        shift = cutlass.min(bound // spacing, streamed)
                    fc1_claims_needed = bound + shift
            else:
                tail_id = routed_id - self._routed_tiles
                if tail_id < Int32(shared_fc2_tiles):
                    work_tile = map_shared_expert_work_id(
                        tail_id,
                        BlockPhase.Linear2,
                        cta_id_in_mapping_cluster,
                        mapping_state,
                        shared,
                    )
                    fc1_claims_needed = streamed_end
        return work_tile, mapping_state, fc1_claims_needed

    @cute.jit
    def _map_routed_work_id(
        self,
        routed_id: Int32,
        cta_id_in_mapping_cluster: cute.Coord,
        mapping_state,
        prefix_tiles: int,
    ):
        """A routed tile behind ``prefix_tiles`` shared tiles, and the other stream's claims its FC2 tiles need."""
        work_tile, mapping_state = self._map_work_id(
            routed_id, cta_id_in_mapping_cluster, mapping_state
        )
        if cutlass.const_expr(self.pipelined_groups):
            fc1_claims_needed = mapping_state.cursor_state.segment_start + Int32(
                prefix_tiles
            )
        else:
            fc1_claims_needed = (
                mapping_state.cursor_state.current_group_fc1_subphase_end
                + Int32(prefix_tiles)
            )
        return work_tile, mapping_state, fc1_claims_needed

    @cute.jit
    def _map_work_id(
        self, work_id: Int32, cta_id_in_mapping_cluster: cute.Coord, mapping_state
    ):
        if cutlass.const_expr(self.pipelined_groups):
            return map_fc12_pipelined_work_id(
                work_id, cta_id_in_mapping_cluster, mapping_state
            )
        return map_fc12_linear_work_id(
            work_id, cta_id_in_mapping_cluster, mapping_state
        )

    @cute.jit
    def _gen_next_die_local_work(self) -> SchedulerWorkTileBase:
        """Claim and map one work tile from the die streams (see the class docstring)."""
        cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
            self._work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
        )
        work_id_worker = self._work_id_worker
        stream_states = (self._task_mapping_state, self._die_task_mapping_state)
        first_state, second_state = stream_states
        exhausted_streams = self._exhausted_streams
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        searching = exhausted_streams != Int32(3)
        while searching:
            stream, work_id = work_id_worker.claim_next_die_local_work(
                self._own_die, exhausted_streams, Int32(self.die_claim_lead_limit)
            )
            mapping_state = first_state
            if stream != Int32(0):
                mapping_state = second_state
            if cutlass.const_expr(self.shared_expert_tiles is not None):
                stream_tile, mapping_state, fc1_claims_needed = (
                    self._map_work_id_with_shared(
                        work_id, cta_id_in_mapping_cluster, mapping_state
                    )
                )
            else:
                stream_tile, mapping_state = self._map_work_id(
                    work_id, cta_id_in_mapping_cluster, mapping_state
                )
                # Every claim below this position of the other stream precedes the
                # FC1 tiles an FC2 tile here reads.
                if cutlass.const_expr(self.pipelined_groups):
                    fc1_claims_needed = mapping_state.cursor_state.segment_start
                else:
                    fc1_claims_needed = (
                        mapping_state.cursor_state.current_group_fc1_subphase_end
                    )
            if stream != Int32(0):
                second_state = mapping_state
            else:
                first_state = mapping_state
            if stream_tile.is_valid_tile:
                searching = Boolean(False)
                is_fc2 = stream_tile.phase == Int32(BlockPhase.Linear2)
                half_tiles = Int32(self.die_half_cta_tiles[0])
                if is_fc2:
                    half_tiles = Int32(self.die_half_cta_tiles[1])
                work_tile = SwapAbFc12WorkTileInfo(
                    expert_idx=stream_tile.expert_idx,
                    tile_m_idx=stream_tile.tile_m_idx + stream * half_tiles,
                    tile_n_idx=stream_tile.tile_n_idx,
                    cumulative_data_physical_row=stream_tile.cumulative_data_physical_row,
                    cumulative_sf_physical_row=stream_tile.cumulative_sf_physical_row,
                    cumulative_token_block_count=stream_tile.cumulative_token_block_count,
                    valid_tokens_in_cta_tile=stream_tile.valid_tokens_in_cta_tile,
                    phase_and_flags=stream_tile.phase_and_flags,
                )
                if is_fc2:
                    other_counter = work_id_worker.get_atomic_counter_pointer(
                        Int32(1) - stream
                    )
                    claims = cute.arch.load(
                        other_counter, Int32, sem="relaxed", scope="gpu"
                    )
                    while claims < fc1_claims_needed:
                        nanosleep(100)
                        claims = cute.arch.load(
                            other_counter, Int32, sem="relaxed", scope="gpu"
                        )
            else:
                exhausted_streams = exhausted_streams | (Int32(1) << stream)
                searching = exhausted_streams != Int32(3)
        self._work_id_worker = work_id_worker
        self._task_mapping_state = first_state
        self._die_task_mapping_state = second_state
        self._exhausted_streams = exhausted_streams
        return work_tile

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim and map one work tile without first-tile prefetch."""
        if cutlass.const_expr(self.die_local_weights):
            return self._gen_next_die_local_work()
        work_id = self._work_id_worker.claim_next_work()
        cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
            self._work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
        )
        if cutlass.const_expr(self.shared_expert_tiles is not None):
            work_tile, self._task_mapping_state, _ = self._map_work_id_with_shared(
                work_id, cta_id_in_mapping_cluster, self._task_mapping_state
            )
        elif cutlass.const_expr(self.pipelined_groups):
            work_tile, self._task_mapping_state = map_fc12_pipelined_work_id(
                work_id, cta_id_in_mapping_cluster, self._task_mapping_state
            )
        else:
            work_tile, self._task_mapping_state = map_fc12_linear_work_id(
                work_id, cta_id_in_mapping_cluster, self._task_mapping_state
            )
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self._work_id_worker))
        values.extend(extract_mlir_values(self._task_mapping_state))
        if self.die_local_weights:
            for field in self._die_local_fields():
                values.extend(extract_mlir_values(field))
        if self.shared_expert_tiles is not None:
            values.extend(extract_mlir_values(self._routed_tiles))
        return values

    def _die_local_fields(self) -> Tuple:
        return (
            self._die_task_mapping_state,
            self._sm_die_table,
            self._own_die,
            self._exhausted_streams,
        )

    def __new_from_mlir_values__(self, values: list) -> "BlackwellFusedFc12Scheduler":
        base_value_count = len(super().__extract_mlir_values__())
        if len(values) < base_value_count:
            raise ValueError(
                "BlackwellFusedFc12Scheduler MLIR value count is smaller than "
                f"its base state: expected at least {base_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            state_value_count = len(extract_mlir_values(state))
            rebuilt_state = new_from_mlir_values(
                state, values[value_index : value_index + state_value_count]
            )
            value_index += state_value_count
            return rebuilt_state

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        if self.die_local_weights:
            (
                result._die_task_mapping_state,
                result._sm_die_table,
                result._own_die,
                result._exhausted_streams,
            ) = (rebuild(field) for field in self._die_local_fields())
        if self.shared_expert_tiles is not None:
            result._routed_tiles = rebuild(self._routed_tiles)
        if value_index != len(values):
            raise ValueError(
                f"BlackwellFusedFc12Scheduler MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )

        for field_name in (
            "expert_count",
            "intermediate_gateup_size",
            "hidden_size",
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "use_2cta_instrs",
            "hint",
            "group_hint",
            "token_padding_block",
            "sf_padding_block",
            "work_id_mode",
            "is_swap_ab",
            "pipelined_groups",
            "launch_cluster_cnt_merge_as_preferred",
            "non_clc_mixed_cga_config",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
            "die_local_weights",
            "mapping_group_hint",
            "shared_expert_tiles",
            "shared_head_counter_index",
        ):
            setattr(result, field_name, getattr(self, field_name))
        if self.die_local_weights:
            result.die_half_cta_tiles = self.die_half_cta_tiles
        return result


class _PhaseInterleaveControlState:
    """Per-cluster phase cadence and stream exhaustion state."""

    def __init__(
        self,
        prologue_remaining: Int32,
        cycle_position: Int32,
        fc1_exhausted: Boolean,
        fc2_exhausted: Boolean,
        prologue_claims_synchronized: Boolean,
    ) -> None:
        self.prologue_remaining = prologue_remaining
        self.cycle_position = cycle_position
        self.fc1_exhausted = fc1_exhausted
        self.fc2_exhausted = fc2_exhausted
        self.prologue_claims_synchronized = prologue_claims_synchronized

    def _fields(self) -> Tuple:
        return (
            self.prologue_remaining,
            self.cycle_position,
            self.fc1_exhausted,
            self.fc2_exhausted,
            self.prologue_claims_synchronized,
        )

    def __extract_mlir_values__(self) -> list:
        values = []
        for field in self._fields():
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: list) -> "_PhaseInterleaveControlState":
        value_index = 0
        rebuilt_fields = []
        for field in self._fields():
            field_value_count = len(extract_mlir_values(field))
            rebuilt_fields.append(
                new_from_mlir_values(
                    field, values[value_index : value_index + field_value_count]
                )
            )
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(
                f"_PhaseInterleaveControlState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(*rebuilt_fields)


class PhaseInterleavedFc12Scheduler(SchedulerBase):
    """Schedule independent FC1 and FC2 streams with per-phase atomic counters."""

    pipeline_mbarriers_region = "fc12.phase_interleaved.scheduler.pipeline_mbarriers"
    work_tiles_region = "fc12.phase_interleaved.scheduler.work_tiles"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "intermediate_gateup_size": int,
            "hidden_size": int,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "hint": int,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "is_swap_ab": bool,
            "dfc1_m_group": OptionalRequirement(int),
            "defer_consumer_until_full_ready": OptionalRequirement(bool),
            "fuse_ready_probe_and_linear1_claim": OptionalRequirement(bool),
            **_mixed_cga_impl_requirements(),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.hint = impl_desc["hint"]
        self.fc1_prologue_tiles = self.hint
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.is_swap_ab = impl_desc["is_swap_ab"]
        self.dfc1_m_group = impl_desc.get("dfc1_m_group", 1)
        self.defer_consumer_until_full_ready = impl_desc.get(
            "defer_consumer_until_full_ready", False
        )
        self.fuse_ready_probe_and_linear1_claim = impl_desc.get(
            "fuse_ready_probe_and_linear1_claim", False
        )
        self.non_clc_mixed_cga_config = _make_non_clc_mixed_cga_config(impl_desc)
        self.launch_cluster_cnt_merge_as_preferred = (
            self.non_clc_mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        )
        self.work_tile_type = (
            SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        )

        self._validate_configuration()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        launch_cta_tile_shape_mnk = (
            self.mma_tiler_mnk[0] // mma_cta_count,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        if self.is_swap_ab:
            self.mapping_cta_tile_shape_mnk = (
                launch_cta_tile_shape_mnk[1],
                launch_cta_tile_shape_mnk[0],
                launch_cta_tile_shape_mnk[2],
            )
            self.mapping_cluster_shape_mn = (
                self.cluster_shape_mn[1],
                self.cluster_shape_mn[0],
            )
        else:
            self.mapping_cta_tile_shape_mnk = launch_cta_tile_shape_mnk
            self.mapping_cluster_shape_mn = self.cluster_shape_mn
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=self.non_clc_mixed_cga_config,
            work_id_mode=self.work_id_mode,
            stream_count=2,
        )

        mapping_cluster_tile_n = (
            self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
        )
        self.blocks_fc1 = (
            self.intermediate_gateup_size + mapping_cluster_tile_n - 1
        ) // mapping_cluster_tile_n
        self.blocks_fc2 = (
            self.hidden_size + mapping_cluster_tile_n - 1
        ) // mapping_cluster_tile_n
        self.fc2_claims_per_token_block = self.blocks_fc2
        interleave_gcd = math.gcd(self.blocks_fc1, self.fc2_claims_per_token_block)
        self.interleave_fc2_slots = self.fc2_claims_per_token_block // interleave_gcd
        self.interleave_cycle_length = (
            self.blocks_fc1 + self.fc2_claims_per_token_block
        ) // interleave_gcd
        self.minimum_global_fc1_claims = minimum_phase_interleave_fc1_claims(
            blocks_fc1=self.blocks_fc1,
            blocks_fc2=self.blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.launch_cluster_cnt_merge_as_preferred,
        )
        minimum_hint = minimum_phase_interleave_hint(
            blocks_fc1=self.blocks_fc1,
            blocks_fc2=self.blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.launch_cluster_cnt_merge_as_preferred,
        )
        if self.fc1_prologue_tiles < minimum_hint and not (
            self.fc1_prologue_tiles == 0 and self.defer_consumer_until_full_ready
        ):
            raise ValueError(
                f"phase_interleave hint {self.fc1_prologue_tiles} cannot cover a "
                f"{self.launch_cluster_cnt_merge_as_preferred}-cluster FC2 claim wave; "
                f"raise hint to at least {minimum_hint}."
            )

    def _validate_configuration(self) -> None:
        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in self.mma_tiler_mnk
        ):
            raise TypeError("mma_tiler_mnk dimensions must be Python ints.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in self.cluster_shape_mn
        ):
            raise TypeError("cluster_shape_mn dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if (
            isinstance(self.fc1_prologue_tiles, bool)
            or not isinstance(self.fc1_prologue_tiles, int)
            or self.fc1_prologue_tiles < 0
        ):
            raise ValueError(
                "fc1_prologue_tiles must be a non-negative Python int "
                "resolved by the kernel frontend."
            )
        if type(self.dfc1_m_group) is not int or self.dfc1_m_group not in (1, 16):
            raise ValueError("dfc1_m_group must be 1 or 16")
        if self.dfc1_m_group > 1 and (
            self.is_swap_ab or not self.defer_consumer_until_full_ready
        ):
            raise ValueError(
                "Grouped dFC1 M traversal requires non-swap singleton consumers, "
                "ordinary producer order and full-ready admission"
            )
        if not isinstance(self.defer_consumer_until_full_ready, bool):
            raise TypeError("defer_consumer_until_full_ready must be a bool.")
        if not isinstance(self.fuse_ready_probe_and_linear1_claim, bool):
            raise TypeError("fuse_ready_probe_and_linear1_claim must be a bool.")
        if (
            self.defer_consumer_until_full_ready
            and self.non_clc_mixed_cga_config.is_mixed
        ):
            raise NotImplementedError(
                "defer_consumer_until_full_ready does not yet support fixed "
                "preferred/fallback cluster groups."
            )
        if self.fuse_ready_probe_and_linear1_claim:
            if not self.defer_consumer_until_full_ready:
                raise ValueError(
                    "fuse_ready_probe_and_linear1_claim requires "
                    "defer_consumer_until_full_ready=True."
                )
        if self.fc1_prologue_tiles == 0 and not self.defer_consumer_until_full_ready:
            raise ValueError("fc1_prologue_tiles=0 requires deferred full-readiness.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_cnt_merge_as_preferred <= 0:
            raise ValueError("launch_cluster_cnt_merge_as_preferred must be positive.")
        if self.work_id_mode != "atomic_counter":
            raise ValueError(
                "Phase-interleaved FC12 scheduling currently requires work_id_mode='atomic_counter'."
            )
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if cluster_size > 32:
            raise ValueError(
                "The atomic broadcast protocol supports at most 32 CTAs per cluster."
            )
        fallback_cluster_shape = self.non_clc_mixed_cga_config.fallback_cluster_shape
        if fallback_cluster_shape is not None:
            fallback_cluster_size = (
                fallback_cluster_shape[0] * fallback_cluster_shape[1]
            )
            if fallback_cluster_size > 32:
                raise ValueError(
                    "The atomic broadcast protocol supports at most 32 CTAs per fallback cluster."
                )
        maximum_int32 = (1 << 31) - 1
        for field_name in ("expert_count", "intermediate_gateup_size", "hidden_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive Python int.")
            if value > maximum_int32:
                raise ValueError(
                    f"{field_name} must fit in a signed Int32, got {value}."
                )

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register work transport and the phase-counter broadcast channel."""
        super().register_smem_regions(smem_workspace)
        self._work_id_worker.register_smem_regions(smem_workspace)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register independently reset FC1 and FC2 work-ID streams."""
        self._work_id_worker.register_device_workspace(device_workspace)

    @cute.jit
    def initialize_fallback_group(self) -> None:
        """Register this physical fallback cluster with its fixed logical group."""
        self._work_id_worker.initialize_fallback_group()

    def get_grid_shape(
        self, *, max_active_clusters: Optional[int] = None, problem_desc=None
    ) -> Tuple[int, int, int]:
        """Return the statically configured persistent launch grid."""
        if (
            not self.non_clc_mixed_cga_config.is_mixed
            and max_active_clusters is not None
            and max_active_clusters < self.launch_cluster_cnt_merge_as_preferred
        ):
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least "
                "launch_cluster_cnt_merge_as_preferred "
                f"({self.launch_cluster_cnt_merge_as_preferred})."
            )
        return (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            self.launch_cluster_cnt_merge_as_preferred,
        )

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Optional[Tuple],
        block_idx: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
        is_fallback_cluster: Optional[Boolean] = None,
    ) -> None:
        """Initialize phase-local mapping, cadence, and counter state."""
        if cutlass.const_expr(
            (expert_token_sizes is None) == (expert_token_prefix_sum is None)
        ):
            raise ValueError(
                "Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided."
            )
        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self._work_id_worker.assign_device_members(
            is_fallback_cluster=is_fallback_cluster,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
        )
        self._task_mapping_state = create_phase_interleaved_fc12_mapping_state(
            expert_count=self.expert_count,
            intermediate_gateup_size=self.intermediate_gateup_size,
            hidden_size=self.hidden_size,
            mapping_cta_tile_shape_mnk=self.mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            dfc1_m_group=self.dfc1_m_group,
        )
        self._control_state = _PhaseInterleaveControlState(
            prologue_remaining=Int32(self.fc1_prologue_tiles),
            # Stagger the steady-state cadence by physical cluster.  Starting
            # every persistent cluster at slot zero turns an intended rolling
            # mix into chip-wide FC1/FC2 bursts; the phase offset keeps both
            # streams represented in every claim wave after the prologue.
            cycle_position=(Int32(block_idx[2]) % Int32(self.interleave_cycle_length)),
            fc1_exhausted=Boolean(False),
            fc2_exhausted=Boolean(False),
            prologue_claims_synchronized=Boolean(self.defer_consumer_until_full_ready),
        )

    @cute.jit
    def cluster_uniform_counter_ready(
        self,
        counter_pointer: cute.Pointer,
        ready_threshold: Int32,
        ready_is_mask: bool = False,
    ) -> Boolean:
        """Observe one counter on the cluster leader and broadcast readiness."""
        if cutlass.const_expr(not self.defer_consumer_until_full_ready):
            raise ValueError(
                "cluster readiness polling requires "
                "defer_consumer_until_full_ready=True."
            )
        work_id_worker = self._work_id_worker
        ready = work_id_worker.cluster_uniform_counter_ready(
            counter_pointer, ready_threshold, ready_is_mask
        )
        self._work_id_worker = work_id_worker
        return ready

    @cute.jit
    def gen_next_linear1_work_unless_counter_ready(
        self,
        counter_pointer: cute.Pointer,
        ready_threshold: Int32,
        ready_is_mask: bool = False,
    ) -> Tuple[Boolean, SchedulerWorkTileBase]:
        """Probe full readiness and otherwise claim one Linear1 tile.

        When Linear1 still has unclaimed work, the readiness load and atomic
        claim share one cluster broadcast transaction.  Once the stream is
        exhausted this falls back to a probe-only transaction, avoiding
        repeated tail over-claims while the held consumer drains.
        """
        if cutlass.const_expr(not self.fuse_ready_probe_and_linear1_claim):
            raise ValueError(
                "fused readiness/Linear1 claims require "
                "fuse_ready_probe_and_linear1_claim=True."
            )

        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        fc1_exhausted = control_state.fc1_exhausted
        ready = Boolean(False)

        if fc1_exhausted:
            ready = work_id_worker.cluster_uniform_counter_ready(
                counter_pointer, ready_threshold, ready_is_mask
            )
        else:
            ready, linear_work_id = (
                work_id_worker.cluster_uniform_counter_ready_or_claim(
                    counter_pointer,
                    ready_threshold,
                    claim_stream_index=Int32(0),
                    ready_is_mask=ready_is_mask,
                )
            )
            if not ready:
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster,
                    self.is_swap_ab,
                )
                work_tile, stream_has_work, task_mapping_state = (
                    map_phase_interleaved_fc12_work_id(
                        linear_work_id,
                        Int32(BlockPhase.Linear1),
                        cta_id_in_mapping_cluster,
                        task_mapping_state,
                    )
                )
                if not stream_has_work:
                    fc1_exhausted = Boolean(True)

        control_state.fc1_exhausted = fc1_exhausted
        self._work_id_worker = work_id_worker
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return ready, work_tile

    @cute.jit
    def gen_next_linear1_work(self) -> SchedulerWorkTileBase:
        """Force one dependency-free Linear1 claim while a Linear2 tile is held.

        This corrective claim is outside the normal rolling cadence: the
        deferred Linear2 claim already consumed its cadence slot.  The caller
        must publish a returned Linear1 tile before rechecking or publishing
        the held consumer.  Consequently no producer descriptor can become
        trapped behind an unready consumer.
        """
        if cutlass.const_expr(not self.defer_consumer_until_full_ready):
            raise ValueError(
                "forced Linear1 claims require defer_consumer_until_full_ready=True."
            )

        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        fc1_exhausted = control_state.fc1_exhausted

        if not fc1_exhausted:
            linear_work_id = work_id_worker.claim_next_work(Int32(0))
            cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                work_id_worker.cta_coord_in_preferred_cluster,
                self.is_swap_ab,
            )
            work_tile, stream_has_work, task_mapping_state = (
                map_phase_interleaved_fc12_work_id(
                    linear_work_id,
                    Int32(BlockPhase.Linear1),
                    cta_id_in_mapping_cluster,
                    task_mapping_state,
                )
            )
            if not stream_has_work:
                fc1_exhausted = Boolean(True)

        control_state.fc1_exhausted = fc1_exhausted
        self._work_id_worker = work_id_worker
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return work_tile

    @cute.jit
    def _wait_for_fc1_prologue_claims(
        self,
        fc1_exhausted: Boolean,
        work_id_worker: NonClcMixedCgaSchedulerWorker,
        task_mapping_state,
    ) -> Boolean:
        """Wait until the static FC1 watermark or the smaller runtime stream end is claimed."""
        if not fc1_exhausted:
            fc1_counter_pointer = work_id_worker.get_atomic_counter_pointer(Int32(0))
            minimum_claim_count = Int32(self.minimum_global_fc1_claims)
            if not spin_peek(
                fc1_counter_pointer, lambda value: value >= minimum_claim_count
            ):
                claim_target, stream_ends_before_target = (
                    resolve_phase_interleaved_fc1_claim_target(
                        minimum_claim_count, task_mapping_state
                    )
                )
                spin_wait(fc1_counter_pointer, lambda value: value >= claim_target)
                if stream_ends_before_target:
                    fc1_exhausted = Boolean(True)
        return fc1_exhausted

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim until one stream yields valid work or both streams terminate."""
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        prologue_remaining = control_state.prologue_remaining
        cycle_position = control_state.cycle_position
        fc1_exhausted = control_state.fc1_exhausted
        fc2_exhausted = control_state.fc2_exhausted
        stream_has_work = Boolean(False)
        prologue_claims_synchronized = control_state.prologue_claims_synchronized
        resolved = Boolean(False)

        while not resolved:
            if fc1_exhausted and fc2_exhausted:
                work_tile = make_fc12_done_tile(self.is_swap_ab)
                resolved = Boolean(True)

            if not resolved:
                want_fc1 = Boolean(True)
                if prologue_remaining <= Int32(0):
                    is_fc2_slot = (
                        cycle_position * Int32(self.interleave_fc2_slots)
                    ) % Int32(self.interleave_cycle_length) < Int32(
                        self.interleave_fc2_slots
                    )
                    want_fc1 = not is_fc2_slot
                if want_fc1 and fc1_exhausted:
                    want_fc1 = Boolean(False)
                if (not want_fc1) and fc2_exhausted:
                    want_fc1 = Boolean(True)

                atomic_counter_index = Int32(1)
                if want_fc1:
                    atomic_counter_index = Int32(0)
                linear_work_id = work_id_worker.claim_next_work(atomic_counter_index)
                want_fc1 = work_id_worker.claimed_stream_index == Int32(0)
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
                )
                phase = Int32(BlockPhase.Linear2)
                if want_fc1:
                    phase = Int32(BlockPhase.Linear1)
                work_tile, stream_has_work, task_mapping_state = (
                    map_phase_interleaved_fc12_work_id(
                        linear_work_id,
                        phase,
                        cta_id_in_mapping_cluster,
                        task_mapping_state,
                    )
                )
                if stream_has_work:
                    if (not want_fc1) and (not prologue_claims_synchronized):
                        fc1_exhausted = self._wait_for_fc1_prologue_claims(
                            fc1_exhausted, work_id_worker, task_mapping_state
                        )
                        prologue_claims_synchronized = Boolean(True)
                    if prologue_remaining > Int32(0):
                        prologue_remaining = prologue_remaining - Int32(1)
                    else:
                        cycle_position = (cycle_position + Int32(1)) % Int32(
                            self.interleave_cycle_length
                        )
                    resolved = Boolean(True)
                else:
                    if want_fc1:
                        fc1_exhausted = Boolean(True)
                    else:
                        fc2_exhausted = Boolean(True)

        control_state.prologue_remaining = prologue_remaining
        control_state.cycle_position = cycle_position
        control_state.fc1_exhausted = fc1_exhausted
        control_state.fc2_exhausted = fc2_exhausted
        control_state.prologue_claims_synchronized = prologue_claims_synchronized
        self._work_id_worker = work_id_worker
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for state in (
            self._work_id_worker,
            self._task_mapping_state,
            self._control_state,
        ):
            values.extend(extract_mlir_values(state))
        return values

    def __new_from_mlir_values__(self, values: list) -> "PhaseInterleavedFc12Scheduler":
        base_value_count = len(super().__extract_mlir_values__())
        if len(values) < base_value_count:
            raise ValueError(
                "PhaseInterleavedFc12Scheduler MLIR value count is smaller than "
                f"its base state: expected at least {base_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            state_value_count = len(extract_mlir_values(state))
            rebuilt_state = new_from_mlir_values(
                state, values[value_index : value_index + state_value_count]
            )
            value_index += state_value_count
            return rebuilt_state

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        result._control_state = rebuild(self._control_state)
        if value_index != len(values):
            raise ValueError(
                f"PhaseInterleavedFc12Scheduler MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )

        for field_name in (
            "expert_count",
            "intermediate_gateup_size",
            "hidden_size",
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "use_2cta_instrs",
            "hint",
            "fc1_prologue_tiles",
            "token_padding_block",
            "sf_padding_block",
            "work_id_mode",
            "is_swap_ab",
            "launch_cluster_cnt_merge_as_preferred",
            "non_clc_mixed_cga_config",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
            "blocks_fc1",
            "blocks_fc2",
            "dfc1_m_group",
            "defer_consumer_until_full_ready",
            "fuse_ready_probe_and_linear1_claim",
            "fc2_claims_per_token_block",
            "interleave_fc2_slots",
            "interleave_cycle_length",
            "minimum_global_fc1_claims",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result


__all__ = [
    "BlackwellFusedFc12Scheduler",
    "PhaseInterleavedFc12Scheduler",
    "minimum_phase_interleave_hint",
]
