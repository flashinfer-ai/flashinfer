"""Composable grouped and phase-interleaved FC12 schedulers."""

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Boolean, Int32, Integer, extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, ProblemDesc, StaticOrRuntimeIntegerType
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div
from .base import SchedulerBase, SchedulerWorkTileBase, WorkIdAcquisitionMode
from .fc12_mapping import (
    BlockPhase,
    NonSwapAbFc12WorkTileInfo,
    SwapAbFc12WorkTileInfo,
    create_fc12_task_mapping_state,
    create_phase_interleaved_fc12_mapping_state,
    make_fc12_done_tile,
    map_fc12_linear_work_id,
    map_phase_interleaved_fc12_work_id,
)
from .work_id_claim import AtomicCounterWorkIdState, GridStrideWorkIdState, claim_work_id


class BlackwellFusedFc12Scheduler(SchedulerBase):
    """Compose work-ID claim, FC12 mapping, and SMEM tile transport."""

    pipeline_mbarriers_region = "blackwell.fc12.scheduler.pipeline_mbarriers"
    work_tiles_region = "blackwell.fc12.scheduler.work_tiles"
    cluster_pipeline_mbarriers_region = "blackwell.fc12.scheduler.cluster_pipeline_mbarriers"
    cluster_broadcast_region = "blackwell.fc12.scheduler.cluster_broadcast"
    work_id_counter_region = "blackwell.fc12.scheduler.work_id_counter"

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
            "launch_cluster_count": int,
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
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.work_tile_type = SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        if self.hint is None:
            self.hint = self.launch_cluster_count
            self.group_hint = self.hint

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
            self.mapping_cluster_shape_mn = (self.cluster_shape_mn[1], self.cluster_shape_mn[0])
        else:
            self.mapping_cta_tile_shape_mnk = launch_cta_tile_shape_mnk
            self.mapping_cluster_shape_mn = self.cluster_shape_mn

        if self.work_id_mode == "cluster_launch_control":
            raise NotImplementedError("cluster_launch_control is not implemented for FC12.")

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
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive.")
        if self.work_id_mode not in ("grid_stride", "atomic_counter", "cluster_launch_control"):
            raise ValueError(
                "work_id_mode must be 'grid_stride', 'atomic_counter', or "
                f"'cluster_launch_control', got {self.work_id_mode!r}."
            )
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if self.work_id_mode == "atomic_counter" and cluster_size > 32:
            raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per cluster.")
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
            raise ValueError("FC12 expert dimensions must be either all static or all runtime.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register scheduler-owned SMEM transport and claim regions."""
        super().register_smem_regions(smem_workspace)
        if self.work_id_mode == "atomic_counter":
            smem_workspace.register_mbarrier(self.cluster_pipeline_mbarriers_region, 2)
            smem_workspace.register_tensor(self.cluster_broadcast_region, cutlass.Int32, (1,))

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register the optional atomic counter in device workspace."""
        if self.work_id_mode == "atomic_counter":
            device_workspace.register(
                self.work_id_counter_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )

    @cute.jit
    def create_scheduler_pipelines(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Create the common transport and optional atomic broadcast pipelines."""
        super().create_scheduler_pipelines(smem_workspace, smem_base)
        self._cluster_pipeline = None
        if cutlass.const_expr(self.work_id_mode == "atomic_counter"):
            cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
            self._cluster_pipeline = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * cluster_size),
                barrier_storage=smem_workspace.ptr(self.cluster_pipeline_mbarriers_region, smem_base),
                defer_sync=True,
            )

    def get_grid_shape(self, *, max_active_clusters: Optional[int] = None, problem_desc=None) -> Tuple[int, int, int]:
        """Return the persistent launch grid in GEMM-domain orientation."""
        if max_active_clusters is not None and max_active_clusters < self.launch_cluster_count:
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least launch_cluster_count "
                f"({self.launch_cluster_count})."
            )
        return (self.cluster_shape_mn[0], self.cluster_shape_mn[1], self.launch_cluster_count)

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Optional[Tuple],
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
    ) -> None:
        """Initialize all FC12 scheduler state rooted for one CTA lifetime."""
        if cutlass.const_expr((expert_token_sizes is None) == (expert_token_prefix_sum is None)):
            raise ValueError("Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided.")
        needs_actual_shape = not all(
            isinstance(dimension, int)
            for dimension in (self.expert_count, self.intermediate_gateup_size, self.hidden_size)
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

        block_x, block_y, block_z = block_idx
        if cutlass.const_expr(self.is_swap_ab):
            cta_id_in_mapping_cluster = (
                Int32(block_y % self.mapping_cluster_shape_mn[0]),
                Int32(block_x % self.mapping_cluster_shape_mn[1]),
                Int32(0),
            )
        else:
            cta_id_in_mapping_cluster = (
                Int32(block_x % self.mapping_cluster_shape_mn[0]),
                Int32(block_y % self.mapping_cluster_shape_mn[1]),
                Int32(0),
            )
        num_persistent_clusters = Int32(cute.size(grid_dim) // cute.size(self.cluster_shape_mn))
        self.create_scheduler_pipelines(smem_workspace, smem_base)

        if cutlass.const_expr(self.work_id_mode == "atomic_counter"):
            cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
            work_id_state = AtomicCounterWorkIdState(
                counter_pointer=device_workspace.ptr(self.work_id_counter_region),
                counter_count=1,
                broadcast_pointer=smem_workspace.ptr(self.cluster_broadcast_region, smem_base),
                is_leader_cta=(
                    cta_id_in_mapping_cluster[0] + cta_id_in_mapping_cluster[1] + cta_id_in_mapping_cluster[2]
                )
                == Int32(0),
                cluster_pipeline=self._cluster_pipeline,
                producer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1),
                consumer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1),
                cluster_size=cluster_size,
            )
        else:
            work_id_state = GridStrideWorkIdState(next_work_id=Int32(block_z), work_id_stride=num_persistent_clusters)

        task_mapping_state = create_fc12_task_mapping_state(
            expert_count=expert_count,
            intermediate_gateup_size=intermediate_gateup_size,
            hidden_size=hidden_size,
            mapping_cta_tile_shape_mnk=(self.mapping_cta_tile_shape_mnk),
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            cta_id_in_mapping_cluster=cta_id_in_mapping_cluster,
        )

        self._work_id_state = work_id_state
        self._task_mapping_state = task_mapping_state

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim and map one work tile without first-tile prefetch."""
        work_id, self._work_id_state = claim_work_id(self._work_id_state)
        work_tile, self._task_mapping_state = map_fc12_linear_work_id(work_id, self._task_mapping_state)
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for state in (self._work_id_state, self._task_mapping_state):
            values.extend(extract_mlir_values(state))
        return values

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
            rebuilt_state = new_from_mlir_values(state, values[value_index : value_index + state_value_count])
            value_index += state_value_count
            return rebuilt_state

        result._work_id_state = rebuild(self._work_id_state)
        result._task_mapping_state = rebuild(self._task_mapping_state)
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
            "launch_cluster_count",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
        ):
            setattr(result, field_name, getattr(self, field_name))
        result._cluster_pipeline = self._cluster_pipeline
        return result


class _PhaseInterleaveControlState:
    """Per-cluster phase cadence and stream exhaustion state."""

    def __init__(
        self, prologue_remaining: Int32, cycle_position: Int32, fc1_exhausted: Boolean, fc2_exhausted: Boolean
    ) -> None:
        self.prologue_remaining = prologue_remaining
        self.cycle_position = cycle_position
        self.fc1_exhausted = fc1_exhausted
        self.fc2_exhausted = fc2_exhausted

    def _fields(self) -> Tuple:
        return (self.prologue_remaining, self.cycle_position, self.fc1_exhausted, self.fc2_exhausted)

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
            rebuilt_fields.append(new_from_mlir_values(field, values[value_index : value_index + field_value_count]))
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
    cluster_pipeline_mbarriers_region = "fc12.phase_interleaved.scheduler.cluster_pipeline_mbarriers"
    cluster_broadcast_region = "fc12.phase_interleaved.scheduler.cluster_broadcast"
    work_id_counter_region = "fc12.phase_interleaved.scheduler.work_id_counters"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {"expert_count": int, "intermediate_gateup_size": int, "hidden_size": int}

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
            "launch_cluster_count": int,
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
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.work_tile_type = SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo

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
            self.mapping_cluster_shape_mn = (self.cluster_shape_mn[1], self.cluster_shape_mn[0])
        else:
            self.mapping_cta_tile_shape_mnk = launch_cta_tile_shape_mnk
            self.mapping_cluster_shape_mn = self.cluster_shape_mn

        mapping_cluster_tile_n = self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
        self.blocks_fc1 = (self.intermediate_gateup_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        self.blocks_fc2 = (self.hidden_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        interleave_gcd = math.gcd(self.blocks_fc1, self.blocks_fc2)
        self.interleave_fc2_slots = self.blocks_fc2 // interleave_gcd
        self.interleave_cycle_length = (self.blocks_fc1 + self.blocks_fc2) // interleave_gcd
        max_dependent_token_blocks = ceil_div(self.launch_cluster_count + self.blocks_fc2 - 1, self.blocks_fc2)
        required_fc1_work = max_dependent_token_blocks * self.blocks_fc1
        minimum_hint = max(1, ceil_div(required_fc1_work, self.launch_cluster_count))
        if self.fc1_prologue_tiles < minimum_hint:
            raise ValueError(
                f"phase_interleave hint {self.fc1_prologue_tiles} cannot cover the maximum "
                f"FC1 dependency ({required_fc1_work} work tiles) of one "
                f"{self.launch_cluster_count}-cluster FC2 claim wave; "
                f"raise hint to at least {minimum_hint}."
            )

    def _validate_configuration(self) -> None:
        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in self.mma_tiler_mnk):
            raise TypeError("mma_tiler_mnk dimensions must be Python ints.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in self.cluster_shape_mn
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
            or self.fc1_prologue_tiles <= 0
        ):
            raise ValueError("fc1_prologue_tiles must be a positive Python int resolved by the kernel frontend.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive.")
        if self.work_id_mode != "atomic_counter":
            raise ValueError("Phase-interleaved FC12 scheduling currently requires work_id_mode='atomic_counter'.")
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if cluster_size > 32:
            raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per cluster.")
        maximum_int32 = (1 << 31) - 1
        for field_name in ("expert_count", "intermediate_gateup_size", "hidden_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive Python int.")
            if value > maximum_int32:
                raise ValueError(f"{field_name} must fit in a signed Int32, got {value}.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register work transport and the phase-counter broadcast channel."""
        super().register_smem_regions(smem_workspace)
        smem_workspace.register_mbarrier(self.cluster_pipeline_mbarriers_region, 2)
        smem_workspace.register_tensor(self.cluster_broadcast_region, cutlass.Int32, (1,))

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register independently reset FC1 and FC2 work-ID counters."""
        device_workspace.register(
            self.work_id_counter_region, cutlass.Int32, (2,), buffer_space="local", reset="tail_reset"
        )

    @cute.jit
    def create_scheduler_pipelines(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Create work transport and the shared phase-counter broadcast pipeline."""
        super().create_scheduler_pipelines(smem_workspace, smem_base)
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self._cluster_pipeline = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * cluster_size),
            barrier_storage=smem_workspace.ptr(self.cluster_pipeline_mbarriers_region, smem_base),
            defer_sync=True,
        )

    def get_grid_shape(self, *, max_active_clusters: Optional[int] = None, problem_desc=None) -> Tuple[int, int, int]:
        """Return the statically configured persistent launch grid."""
        if max_active_clusters is not None and max_active_clusters < self.launch_cluster_count:
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least launch_cluster_count "
                f"({self.launch_cluster_count})."
            )
        return (self.cluster_shape_mn[0], self.cluster_shape_mn[1], self.launch_cluster_count)

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Optional[Tuple],
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
    ) -> None:
        """Initialize phase-local mapping, cadence, and counter state."""
        if cutlass.const_expr((expert_token_sizes is None) == (expert_token_prefix_sum is None)):
            raise ValueError("Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided.")
        block_x, block_y, _ = block_idx
        if cutlass.const_expr(self.is_swap_ab):
            cta_id_in_mapping_cluster = (
                Int32(block_y % self.mapping_cluster_shape_mn[0]),
                Int32(block_x % self.mapping_cluster_shape_mn[1]),
                Int32(0),
            )
        else:
            cta_id_in_mapping_cluster = (
                Int32(block_x % self.mapping_cluster_shape_mn[0]),
                Int32(block_y % self.mapping_cluster_shape_mn[1]),
                Int32(0),
            )

        self.create_scheduler_pipelines(smem_workspace, smem_base)
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self._work_id_state = AtomicCounterWorkIdState(
            counter_pointer=device_workspace.ptr(self.work_id_counter_region),
            counter_count=2,
            broadcast_pointer=smem_workspace.ptr(self.cluster_broadcast_region, smem_base),
            is_leader_cta=(cta_id_in_mapping_cluster[0] + cta_id_in_mapping_cluster[1] + cta_id_in_mapping_cluster[2])
            == Int32(0),
            cluster_pipeline=self._cluster_pipeline,
            producer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1),
            consumer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1),
            cluster_size=cluster_size,
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
            cta_id_in_mapping_cluster=cta_id_in_mapping_cluster,
        )
        self._control_state = _PhaseInterleaveControlState(
            prologue_remaining=Int32(self.fc1_prologue_tiles),
            cycle_position=Int32(0),
            fc1_exhausted=Boolean(False),
            fc2_exhausted=Boolean(False),
        )

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim until one stream yields valid work or both streams terminate."""
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_state = self._work_id_state
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        prologue_remaining = control_state.prologue_remaining
        cycle_position = control_state.cycle_position
        fc1_exhausted = control_state.fc1_exhausted
        fc2_exhausted = control_state.fc2_exhausted
        resolved = Boolean(False)

        while not resolved:
            if fc1_exhausted and fc2_exhausted:
                work_tile = make_fc12_done_tile(self.is_swap_ab)
                resolved = Boolean(True)
            else:
                want_fc1 = Boolean(True)
                if prologue_remaining <= Int32(0):
                    is_fc2_slot = (cycle_position * Int32(self.interleave_fc2_slots)) % Int32(
                        self.interleave_cycle_length
                    ) < Int32(self.interleave_fc2_slots)
                    want_fc1 = not is_fc2_slot
                if want_fc1 and fc1_exhausted:
                    want_fc1 = Boolean(False)
                if (not want_fc1) and fc2_exhausted:
                    want_fc1 = Boolean(True)

                phase = Int32(BlockPhase.Linear2)
                atomic_counter_index = Int32(1)
                if want_fc1:
                    phase = Int32(BlockPhase.Linear1)
                    atomic_counter_index = Int32(0)
                linear_work_id, work_id_state = claim_work_id(work_id_state, atomic_counter_index=atomic_counter_index)
                work_tile, stream_has_work, task_mapping_state = map_phase_interleaved_fc12_work_id(
                    linear_work_id, phase, task_mapping_state
                )
                if stream_has_work:
                    if prologue_remaining > Int32(0):
                        prologue_remaining = prologue_remaining - Int32(1)
                    else:
                        cycle_position = (cycle_position + Int32(1)) % Int32(self.interleave_cycle_length)
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
        self._work_id_state = work_id_state
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for state in (self._work_id_state, self._task_mapping_state, self._control_state):
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
            rebuilt_state = new_from_mlir_values(state, values[value_index : value_index + state_value_count])
            value_index += state_value_count
            return rebuilt_state

        result._work_id_state = rebuild(self._work_id_state)
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
            "launch_cluster_count",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
            "blocks_fc1",
            "blocks_fc2",
            "interleave_fc2_slots",
            "interleave_cycle_length",
        ):
            setattr(result, field_name, getattr(self, field_name))
        result._cluster_pipeline = self._cluster_pipeline
        return result


__all__ = ["BlackwellFusedFc12Scheduler", "PhaseInterleavedFc12Scheduler"]
