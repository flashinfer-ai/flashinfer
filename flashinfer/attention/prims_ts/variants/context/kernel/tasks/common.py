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

"""Task base classes, work queues and schedule helpers shared by the context kernel tasks."""

import cutlass.cute as cute
import cutlass
from collections.abc import Callable, Generator
from contextlib import contextmanager
from cutlass import Int32
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import work_tile_loop
from cutlass.experimental.task_scheduling.task import Task
from dataclasses import dataclass, field
from ..config import FmhaConfig
from typing import Any


@dataclass(kw_only=True)
class PackedContextWorkQueue(WorkQueue):
    """Skip padded Q tiles, retaining empty KV rows when outputs must be written."""

    # Static configuration
    cfg: cutlass.Constexpr[FmhaConfig] = field(init=False, default=None)
    keep_empty_kv: cutlass.Constexpr[bool] = field(init=False, default=False)

    # Per-task state
    cum_seqlen_k: Any = field(init=False, default=None)
    cum_seqlen_q: Any = field(init=False, default=None)
    seq_lens_kv: Any = field(init=False, default=None)

    def __init__(
        self,
        cfg: FmhaConfig,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None = None,
        seq_lens_kv: cute.Pointer | None = None,
        keep_empty_kv: bool = False,
        **kwargs: Any,
    ) -> None:
        """Attach the live packed-Q metadata used by the skip predicate."""
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cum_seqlen_q = cum_seqlen_q
        self.cum_seqlen_k = cum_seqlen_k
        self.seq_lens_kv = seq_lens_kv
        self.keep_empty_kv = keep_empty_kv

    @cute.jit
    def skip_work_tile_if(self, work_tile: Any) -> cutlass.Boolean:
        """Skip a scheduler tile whose first Q row is outside its request."""
        seq_idx, _, batch_idx = self.cfg.work_tile_coord_indices
        seq_coord = Int32(work_tile.tile_idx[seq_idx])
        if cutlass.const_expr(self.cfg.uses_causal_reversed_head_batch_seq_tile_order):
            seq_coord = Int32(self.cfg.num_seq_tiles) - seq_coord - Int32(1)
        batch_coord = Int32(work_tile.tile_idx[batch_idx])
        skip = cutlass.Boolean(False)
        if cutlass.const_expr(
            self.cum_seqlen_q is not None and not self.cfg.has_uniform_varlen
        ):
            q_begin = Int32(self.cum_seqlen_q[batch_coord])
            q_end = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
            skip = seq_coord * Int32(self.cfg.cta_tiler[0]) >= q_end - q_begin
        if cutlass.const_expr(not self.keep_empty_kv and self.seq_lens_kv is not None):
            skip = skip or Int32(self.seq_lens_kv[batch_coord]) <= Int32(0)
        elif cutlass.const_expr(
            not self.keep_empty_kv
            and self.cum_seqlen_k is not None
            and not self.cfg.has_uniform_varlen
        ):
            skip = (
                skip
                or self.cum_seqlen_k[batch_coord + Int32(1)]
                <= self.cum_seqlen_k[batch_coord]
            )
        return skip


def captured_loop_bounds(
    task_class: type[Task],
    task_kwargs: dict[str, object],
) -> tuple[object, object, object]:
    """Infer ``(start, end, step)`` loop bounds for a captured schedule.

    Dense schedules pass a static ``domain``; causal schedules pass
    ``num_kv_tiles`` and use the task class's ``get_domain`` as a dynamic end.
    """
    loop_start = task_kwargs.pop("domain_start", 0)
    loop_step = task_kwargs.pop("step", 1)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None:
        if "num_kv_tiles" not in task_kwargs:
            raise ValueError(
                "create_*_task requires a 'domain' or 'num_kv_tiles' kwarg to "
                "determine the loop end."
            )
        loop_end = task_class.get_domain
    return loop_start, loop_end, loop_step


def packed_context_skip_predicate(
    work_queue: WorkQueue | None,
) -> Callable[..., object] | None:
    """Select the live-Q skip predicate before schedule capture creates proxies."""
    if isinstance(work_queue, PackedContextWorkQueue):
        return PackedContextWorkQueue.skip_work_tile_if
    return None


def persistent_tail(work_queue: WorkQueue) -> None:
    """Advance and release the persistent work tile after one task body."""
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


def schedule_with_work_queue(
    schedule: Callable[..., object],
    *resources: MemoryResource,
    work_queue: WorkQueue | None,
) -> object:
    """Invoke a captured schedule with the optional WorkQueue argument."""
    if work_queue is None:
        return schedule(*resources)
    return schedule(*resources, work_queue)


def src_resources(
    *resources: MemoryResource,
    work_queue: WorkQueue | None,
) -> list[MemoryResource]:
    """Build a task source-resource list, including WorkQueue when present."""
    src = list(resources)
    if work_queue is not None:
        src.append(work_queue)
    return src


@contextmanager
def work_tile_schedule_loop(
    work_queue: WorkQueue | None,
    *,
    skip_if: Callable[..., object] | None = None,
) -> Generator[object | None, None, None]:
    """Wrap a task body once per persistent work tile, or once for static schedules."""
    if skip_if is not None:
        assert work_queue is not None
        with work_tile_loop(
            work_queue,
            skip_if=skip_if,
        ) as work_tiles:
            with work_tiles.skippable():
                yield work_tiles
            # Every fetched tile, including a skipped one, must advance and
            # release the queue exactly once so persistent workers converge.
            persistent_tail(work_queue)
    elif work_queue is not None:
        with work_tile_loop(work_queue) as work_tile:
            yield work_tile
            persistent_tail(work_queue)
    else:
        yield None
