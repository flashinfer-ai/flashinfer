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

"""Base resource and work throttling shared by the 2CTA MLA resources."""

import cutlass.cute as cute
import cutlass
from cutlass import Boolean
from cutlass.cutlass_dsl import dsl_user_op, if_generate
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
)
from dataclasses import dataclass, fields as dataclass_fields
from typing import ClassVar
from ...device.task_locals import install_task_local_specs


@dataclass(kw_only=True)
class HighThroughputMlaResource(MemoryResource):
    """Base class that binds captured-schedule task-local variables."""

    task_local_specs: ClassVar[tuple[tuple, ...]] = ()

    def __post_init__(self) -> None:
        install_task_local_specs(self, self.task_local_specs)


# =====================================================================
# WorkThrottleBarrierResource — Cluster-safe CLC scheduler pacing
# =====================================================================


@dataclass(kw_only=True)
class WorkThrottleBarrierResource(MemoryResource):
    """Pace CLC schedule token reuse against the leader CTA's active MMA task.

    The barrier has no payload.  Its producer task already runs only on CTA 0,
    so ordinary public pipeline ownership is sufficient for every stage and
    producer-tail operation.
    """

    is_barrier: cutlass.Constexpr[bool] = True


# Both CTAs publish converted payloads to the leader with a cluster release.
# The consumer must acquire at the same scope before reading either payload.
def with_cluster_acquire_waits(base: object) -> object:
    """Return ``base`` (a TmaUmma pipeline) with consumer waits/try-waits on the full barriers using acquire at cluster scope."""
    cls = type(base)

    class ClusterAcquirePipeline(cls):  # type: ignore[valid-type, misc]
        @dsl_user_op
        def consumer_try_wait(self, state, *, loc=None, ip=None) -> Boolean:
            """Adapt the pipeline try-wait hook to the cluster-scope payload acquire."""
            return self.try_cluster_acquire(state)

        @dsl_user_op
        def consumer_wait(
            self, state, try_wait_token=None, *, loc=None, ip=None
        ) -> None:
            """Complete the cluster acquire unless an earlier successful try-wait already did so."""
            if_generate(
                try_wait_token is None or try_wait_token == 0,
                lambda: self.wait_cluster_acquire(state),
                loc=loc,
                ip=ip,
            )

        @cute.jit
        def try_cluster_acquire(self, state) -> Boolean:
            """Try the full-stage barrier with cluster acquire ordering for both CTAs' payloads."""
            bar = self.sync_object_full.get_barrier(state.index)
            return prims.mbarrier_wait_parity(
                bar,
                state.phase,
                prims.MBarrierWait.TRY,
                scope=prims.MBarrierScope.CLUSTER,
                order=prims.MemOrder.ACQUIRE,
            )

        @cute.jit
        def wait_cluster_acquire(self, state) -> None:
            """Wait until the full stage is visible under the producer's cluster release."""
            while not self.try_cluster_acquire(state):
                pass

    values = {f.name: getattr(base, f.name) for f in dataclass_fields(base)}
    return ClusterAcquirePipeline(**values)
