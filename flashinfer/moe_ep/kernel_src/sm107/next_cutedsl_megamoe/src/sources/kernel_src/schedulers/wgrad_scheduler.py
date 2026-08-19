"""Full-grid CLC scheduler for Blackwell MoE Wgrad kernels."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Boolean, Int32, Integer, extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, ProblemDesc, StaticOrRuntimeIntegerType
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace
from .base import SchedulerBase, SchedulerWorkTileBase
from .work_id_claim import ClusterLaunchControlWorkIdState, claim_work_id
from .wgrad_mapping import WgradWorkTileInfo, make_wgrad_done_tile, make_wgrad_work_tile


class BlackwellMoeWgradScheduler(SchedulerBase):
    """Schedule one full 2Dx2D Wgrad output grid through CLC."""

    pipeline_mbarriers_region = "blackwell.wgrad.scheduler.pipeline_mbarriers"
    work_tiles_region = "blackwell.wgrad.scheduler.work_tiles"
    clc_mbarriers_region = "blackwell.wgrad.scheduler.clc_mbarriers"
    clc_response_region = "blackwell.wgrad.scheduler.clc_response"
    grid_z_limit = 65535

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {"expert_count": int, "intermediate": StaticOrRuntimeIntegerType, "hidden": StaticOrRuntimeIntegerType}

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.expert_count = problem_desc["expert_count"]
        self.intermediate = problem_desc["intermediate"]
        self.hidden = problem_desc["hidden"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.work_tile_type = WgradWorkTileInfo

        self._validate_configuration()
        self.mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.cta_tile_shape_mnk = (
            self.mma_tiler_mnk[0] // self.mma_cta_count,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )

    def _validate_configuration(self) -> None:
        if isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be positive.")
        for field_name in ("intermediate", "hidden"):
            value = getattr(self, field_name)
            if isinstance(value, int) and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{field_name} must be positive.")
        if self.expert_count > self.grid_z_limit:
            raise ValueError(f"expert_count must not exceed grid Z limit {self.grid_z_limit}, got {self.expert_count}.")

        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if not all(isinstance(dimension, int) for dimension in self.mma_tiler_mnk):
            raise TypeError("mma_tiler_mnk dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")

        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(isinstance(dimension, int) for dimension in self.cluster_shape_mn):
            raise TypeError("cluster_shape_mn dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        if any(dimension & (dimension - 1) for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be powers of two.")
        if self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16:
            raise ValueError("A Blackwell CTA cluster may contain at most 16 CTAs.")

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if self.cluster_shape_mn[0] % mma_cta_count != 0:
            raise ValueError("cluster M must be divisible by the MMA CTA count.")

    def get_grid_shape(self, *, max_active_clusters: Optional[int] = None, problem_desc=None) -> Tuple[int, int, int]:
        """Return the cluster-aligned full Wgrad output grid."""
        if problem_desc is None:
            if not isinstance(self.intermediate, int) or not isinstance(self.hidden, int):
                raise ValueError("Dynamic Wgrad dimensions require an actual expert shape.")
            expert_count = self.expert_count
            intermediate = self.intermediate
            hidden = self.hidden
        else:
            expert_count, intermediate, hidden = problem_desc
        cta_tile_m, cta_tile_n, _ = self.cta_tile_shape_mnk
        cluster_m, cluster_n = self.cluster_shape_mn
        grid_m = (hidden + cta_tile_m * cluster_m - 1) // (cta_tile_m * cluster_m) * cluster_m
        grid_n = (intermediate + cta_tile_n * cluster_n - 1) // (cta_tile_n * cluster_n) * cluster_n
        return grid_m, grid_n, expert_count

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register common transport before the CLC regions."""
        super().register_smem_regions(smem_workspace)
        smem_workspace.register_mbarrier(self.clc_mbarriers_region, 2)
        smem_workspace.register_tensor(self.clc_response_region, cutlass.Int32, (4,), byte_alignment=16)

    @cute.jit
    def assign_device_members(
        self,
        *,
        offs: cute.Tensor,
        actual_expert_shape: Tuple,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: Optional[DeviceWorkspace] = None,
    ) -> None:
        """Create CTA-lifetime transport, CLC, and bootstrap state."""
        if cutlass.const_expr(not isinstance(self.intermediate, int)):
            self.intermediate = actual_expert_shape[1]
        if cutlass.const_expr(not isinstance(self.hidden, int)):
            self.hidden = actual_expert_shape[2]
        block_m, block_n, block_expert = block_idx
        cluster_m, cluster_n = self.cluster_shape_mn
        cta_coord_in_cluster = (Int32(block_m % cluster_m), Int32(block_n % cluster_n), Int32(0))
        is_leader_cta = (cta_coord_in_cluster[0] + cta_coord_in_cluster[1] + cta_coord_in_cluster[2]) == Int32(0)

        self.create_scheduler_pipelines(smem_workspace, smem_base)
        cluster_size = cluster_m * cluster_n
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=smem_workspace.ptr(self.clc_mbarriers_region, smem_base),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * cluster_size),
            tx_count=16,
            cta_layout_vmnk=cute.make_layout((1, cluster_m, cluster_n, 1)),
            defer_sync=True,
        )

        producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, 1)
        consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        self.offs = offs
        self.current_work = make_wgrad_done_tile()
        self._work_id_state = ClusterLaunchControlWorkIdState(
            response_pending=Boolean(True),
            grid_m=Int32(block_m),
            grid_n=Int32(block_n),
            grid_l=Int32(block_expert),
            response_is_valid=Boolean(True),
            cta_coord_in_cluster=cta_coord_in_cluster,
            cluster_pipeline=clc_pipeline,
            producer_state=producer_state,
            consumer_state=consumer_state,
            is_leader_cta=is_leader_cta,
            response_pointer=smem_workspace.ptr(self.clc_response_region, smem_base),
        )

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Consume bootstrap once, then fetch and map CLC responses."""
        work_id, self._work_id_state = claim_work_id(self._work_id_state)
        next_work = make_wgrad_done_tile()
        if work_id.is_valid:
            next_work = make_wgrad_work_tile(
                offs=self.offs,
                expert_idx=work_id.grid_l,
                tile_m_idx=work_id.grid_m,
                tile_n_idx=work_id.grid_n,
                cta_tile_k=self.cta_tile_shape_mnk[2],
            )
        else:
            next_work = make_wgrad_done_tile()
        self.current_work = next_work
        return self.current_work

    @cute.jit
    def produce_tail(self) -> None:
        """Drain work transport and the leader-owned CLC producer."""
        super().produce_tail()
        work_id_state = self._work_id_state
        if self._work_id_state.is_leader_cta:
            self._work_id_state.cluster_pipeline.producer_tail(self._work_id_state.producer_state)
        else:
            self._work_id_state = work_id_state

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for state in (self.offs, self.current_work, self._work_id_state):
            values.extend(extract_mlir_values(state))
        return values

    def __new_from_mlir_values__(self, values: list) -> "BlackwellMoeWgradScheduler":
        base_value_count = len(super().__extract_mlir_values__())
        if len(values) < base_value_count:
            raise ValueError(
                "BlackwellMoeWgradScheduler MLIR value count is smaller "
                f"than its base state: expected at least "
                f"{base_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            state_value_count = len(extract_mlir_values(state))
            rebuilt_state = new_from_mlir_values(state, values[value_index : value_index + state_value_count])
            value_index += state_value_count
            return rebuilt_state

        result.offs = rebuild(self.offs)
        result.current_work = rebuild(self.current_work)
        result._work_id_state = rebuild(self._work_id_state)
        if value_index != len(values):
            raise ValueError(f"BlackwellMoeWgradScheduler consumed {value_index} MLIR values, got {len(values)}.")

        for field_name in (
            "expert_count",
            "intermediate",
            "hidden",
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "use_2cta_instrs",
            "mma_cta_count",
            "cta_tile_shape_mnk",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result


__all__ = ["BlackwellMoeWgradScheduler"]
