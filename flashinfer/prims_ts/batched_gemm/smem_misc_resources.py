# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WorkQueue, work-throttle barrier, and cluster proxy resources."""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    WorkQueue,
)

from .batched_gemm_config import BatchedGemmConfig

Constexpr = cutlass.Constexpr


@dataclass(kw_only=True)
class WorkThrottleBarrierResource(MemoryResource):
    """One-shot per-work-tile throttle before issuing the next CLC work id.

    Clustered persistent kernels use a CpAsync pipeline
    signal from the active TMA load task to the WorkId task. This prevents the
    scheduler warp from running ahead of the mainloop start for each work tile.
    """

    is_barrier: Constexpr[bool] = True
    producer_state_owned_by_signaling_ctas_only: Constexpr[bool] = True


@dataclass(kw_only=True)
class BatchedGemmWorkQueue(WorkQueue):
    """WorkQueue with batched-GEMM CUDA-graph early-exit knowledge."""

    cfg: Constexpr[BatchedGemmConfig]
    num_non_exiting_ctas_tensor: Any = None
    num_non_exiting_ctas_value: Any = None

    def __init__(
        self,
        tile_scheduler_config,
        cfg,
        num_non_exiting_ctas_tensor=None,
        num_non_exiting_ctas_value=None,
        **kwargs,
    ) -> None:
        super().__init__(
            tile_scheduler_config=tile_scheduler_config,
            **kwargs,
        )
        self.cfg = cfg
        self.num_non_exiting_ctas_tensor = num_non_exiting_ctas_tensor
        self.num_non_exiting_ctas_value = num_non_exiting_ctas_value

    @cute.jit
    def should_skip_work_tile(self, work_tile) -> bool:
        # For CUDA-graph reuse, persistent tasks still fetch the padded
        # work ids, but skip the full task body for token CTAs past the active
        # batch limit and only run the WorkQueue tail to advance the scheduler.
        if cutlass.const_expr(
            (not self.cfg.use_early_exit)
            or (
                self.num_non_exiting_ctas_value is None
                and self.num_non_exiting_ctas_tensor is None
            )
        ):
            return cutlass.Boolean(False)

        tile_coord_m, tile_coord_n, _ = work_tile.tile_idx
        if cutlass.const_expr(self.cfg.is_swap_ab):
            token_cta_idx = tile_coord_n
        else:
            token_cta_idx = tile_coord_m

        if cutlass.const_expr(self.num_non_exiting_ctas_value is not None):
            num_non_exiting_ctas = self.num_non_exiting_ctas_value
        else:
            num_non_exiting_ctas_view = cutlass.make_array_view(
                self.num_non_exiting_ctas_tensor
            )
            num_non_exiting_ctas = num_non_exiting_ctas_view.load(
                idx=cutlass.Int32(0), vector_size=1
            )[0]
        return token_cta_idx >= num_non_exiting_ctas


@dataclass(kw_only=True)
class ProxyClusterBarrierResource(MemoryResource):
    """Virtual resource for cross-CTA data-readiness signaling.

    Carries no SMEM data — pure barrier. SyncTask is the producer
    (cross-CTA arrive). MmaTask is the consumer (waits for both CTAs'
    data to be ready, releases via tcgen05.commit CTA_2).

    Pipeline: TS ``AsyncUmma`` (CUTLASS ``PipelineAsyncUmma``)
      Full barrier: AsyncThread producer with 32*cluster_m arrivals into the
      leader CTA's full barrier.
      Empty barrier: TCGen05Mma consumer release with CTA_2 peer signaling.
    """

    cfg: Constexpr[BatchedGemmConfig]

    @cute.jit
    def producer_work(self, stage_info: StageInfo) -> None:
        """Publish SMEM writes before the cross-CTA proxy arrive.

        SyncTask reaches this point only after waiting on the gather/TMA
        producer barriers.  The proxy barrier is a second handoff to the UMMA
        task, so use the same cluster-scoped async proxy release fence as the
        generated low-latency kernels before ProducerCommit releases MMA.
        """
        prims.fence_proxy_async_release_sync_restrict()

    @cute.jit
    def consumer_work(self, stage_info: StageInfo) -> None:
        """Make cross-CTA SMEM writes visible to the leader MMA task."""
        prims.fence_proxy_async_acquire_sync_restrict()
