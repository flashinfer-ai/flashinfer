# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Complete credit verification for delayed-release MLA schedules."""

from collections import defaultdict

from cutlass.experimental.task_scheduling.enums import ScheduleStage
from cutlass.experimental.task_scheduling.resources import PipelineGroup, WorkQueue
from cutlass.experimental.task_scheduling.task_manager import TaskManager, expand_loop


def verify_complete_schedule(schedules, initial_credits):
    """Check single-producer/single-consumer pipelines until every task drains.

    A successful simulation step executes at least one operation. Thus the
    total number of operations bounds all progress, including schedules whose
    critical path is longer than the longest individual task. Multi-task
    merge/fork credit semantics are intentionally left to the DSL verifier.
    """
    owners = defaultdict(set)
    for task_id, schedule in enumerate(schedules):
        for resource, stage, *_ in schedule:
            if isinstance(resource, WorkQueue):
                continue
            if (
                isinstance(resource, PipelineGroup)
                or getattr(resource, "pipeline_group", None) is not None
            ):
                raise ValueError(
                    "complete MLA verifier does not handle pipeline groups"
                )
            if stage in (
                ScheduleStage.ProducerAcquire,
                ScheduleStage.ProducerCommit,
                ScheduleStage.ConsumerWait,
                ScheduleStage.ConsumerRelease,
            ):
                owners[id(resource), stage].add(task_id)
    if any(len(tasks) != 1 for tasks in owners.values()):
        raise ValueError("complete MLA verifier requires one task per pipeline role")

    available = {key: value[0] for key, value in initial_credits.items()}
    committed = defaultdict(int)
    positions = [0] * len(schedules)
    remaining = sum(map(len, schedules))
    while remaining:
        progressed = False
        for task_id, schedule in enumerate(schedules):
            if positions[task_id] == len(schedule):
                continue
            resource, stage, *_ = schedule[positions[task_id]]
            key = id(resource)
            if not isinstance(resource, WorkQueue):
                if stage == ScheduleStage.ProducerAcquire:
                    if available.get(key, 0) == 0:
                        continue
                    available[key] -= 1
                elif stage == ScheduleStage.ConsumerWait:
                    if committed[key] == 0:
                        continue
                    committed[key] -= 1
                elif stage == ScheduleStage.ProducerCommit:
                    committed[key] += 1
                elif stage == ScheduleStage.ConsumerRelease:
                    available[key] = available.get(key, 0) + 1
            positions[task_id] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise ValueError("complete MLA schedule verification found a deadlock")
    if any(committed.values()):
        raise ValueError("complete MLA schedule has unconsumed producer commits")
    if any(
        value != initial_credits.get(key, (0, None))[0]
        for key, value in available.items()
    ):
        raise ValueError("complete MLA schedule has unmatched acquires/releases")


class MlaTaskManager(TaskManager):
    """Keep DSL checks, retrying its older bounded credit simulation if needed.

    DSL 4.7 stops its schedule printer after max(task lengths) + 100 steps.
    BF16 KV reuse can exceed that bound while continuing to make progress.
    Retry only its end-of-simulation balance errors; genuine deadlocks and all
    subsequent structural, DMA, aliasing and interleaving checks remain active.
    No installed dependency or generated device code is modified.
    """

    def _print_task_schedules(self):
        try:
            super()._print_task_schedules()
        except ValueError as error:
            if not any(
                text in str(error)
                for text in (
                    "not consumed!",
                    "not matched by ConsumerRelease!",
                    "not matched by ProducerAcquire!",
                )
            ):
                raise
            fallback = (
                max(
                    (
                        task.domain_start
                        for task in self.tasks
                        if isinstance(task.domain_start, int)
                    ),
                    default=0,
                )
                + 1
            )
            schedules = [
                expand_loop(task, dynamic_domain_fallback=fallback)
                for task in self.tasks
            ]
            verify_complete_schedule(schedules, self._build_initial_cons_release())
