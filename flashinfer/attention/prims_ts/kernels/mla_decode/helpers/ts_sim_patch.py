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

"""Step-budget patch for the task-scheduling schedule simulator.

``TaskManager`` dry-runs every task's HEAD + LOOP x domain + TAIL schedule in
round-robin to prove the pipelines cannot deadlock.  Its step budget is the
longest per-task list plus 100 steps, which assumes the critical dependency
chain is about as long as the busiest task.  The native FP4 QK graph
serializes MMA -> correction (SFB token) -> MMA per tile through single-stage
TMEM pipelines, so its chain is longer than any single task list and the
simulation stops early with a false "ProducerCommit not consumed" report (it
passes with a 4-tile domain and fails from 8 tiles on).  This widens the
budget to a multiple of the longest list; the private surface is checked by
source so a changed nvidia-cutlass-dsl fails loudly instead of silently.
"""

from __future__ import annotations

import inspect
import textwrap

_NEEDLE = "max_steps = max((len(s) for s in schedule_lists), default=0) + 100"


def ensure_schedule_sim_step_budget(factor: int = 8) -> None:
    """Multiply the schedule simulator's step budget by ``factor`` (idempotent)."""
    from cutlass.experimental.task_scheduling import task_manager as TM

    if getattr(TM, "_flashinfer_step_budget_factor", 0) >= factor:
        return
    try:
        src = inspect.getsource(TM.print_schedule_list)
    except OSError as exc:
        raise RuntimeError(
            "cannot widen the task-scheduling simulator step budget: the source of "
            "print_schedule_list is unavailable (pyc-only install)"
        ) from exc
    src = textwrap.dedent(src)
    if _NEEDLE not in src:
        raise RuntimeError(
            "task-scheduling simulator step budget line not found; the "
            "nvidia-cutlass-dsl private surface changed (print_schedule_list)"
        )
    src = src.replace(
        _NEEDLE,
        f"max_steps = max((len(s) for s in schedule_lists), default=0) * {int(factor)} + 100",
    )
    exec(compile(src, TM.__file__, "exec"), TM.__dict__)
    TM._flashinfer_step_budget_factor = factor
