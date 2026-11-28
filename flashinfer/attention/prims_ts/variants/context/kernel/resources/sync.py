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

"""Ordering resources: softmax ping-pong and TMEM readiness handoffs."""

import cutlass
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
)
from cutlass.pipeline import PipelineAsync
from dataclasses import dataclass
from .....device.pipeline import (
    StructuredWaitPipelineAsync,
)


@dataclass(kw_only=True)
class S0S1SequenceResource(MemoryResource):
    """Sequence barrier between Softmax0 (producer) and Softmax1 (consumer).

    Paces peer iterations. The paired FP8 main loop hands the token back
    before computing P, allowing overlap without letting one peer run ahead.
    Each peer's own P handoff orders its P stores before the MMA reads them:
    the SP release (or TmemPResource) with P in TMEM, SmemPResource with P in
    SMEM.
    Other paths retain serialized P computation.

    Single resource instance shared across tasks:
      - Softmax0's dst_resource (ProducerAcquire/Commit)
      - Softmax1's src_resource (ConsumerWait/Release)
    """

    is_barrier: cutlass.Constexpr[bool] = True

    def create_pipeline(self, pipeline_config: PipelineConfig) -> object:
        base = super().create_pipeline(pipeline_config)
        assert isinstance(base, PipelineAsync)
        return StructuredWaitPipelineAsync(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
        )


# ---------------------------------------------------------------------------
# TmemPPrefixReadyResource -- leading keys of P stored
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemPPrefixReadyResource(MemoryResource):
    """Barrier: Softmax signals after storing the leading half of a P tile.

    Lets the MMA warp issue the PV MMAs for the first 64 keys while softmax
    is still computing the rest. There is no matching barrier for the rest of
    P; the existing SP stage guards the full tile and the S buffer reuse.

    Single resource instance shared across tasks:
      - Softmax's dst_resource (ProducerAcquire/Commit)
      - MMA's src_resource (ConsumerWait/Release)
    """

    is_barrier: cutlass.Constexpr[bool] = True


# ---------------------------------------------------------------------------
# TmemStatsDoneResource -- cross-tile stats-read notification
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemStatsDoneResource(MemoryResource):
    """Notification barrier: Correction signals after reading stats from TMEM.

    Prevents cross-tile aliasing race where next tile's QK→S UMMA can
    overwrite TMEM columns overlapping TmemStats0/1 before correction reads them.

    Single resource instance shared across tasks:
      - MMA's dst_resource (ProducerAcquire/Commit)
      - Correction's src_resource (ConsumerWait/Release)
    """

    is_barrier: cutlass.Constexpr[bool] = True
