# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass", minversion="4.7.0")
from cutlass.experimental.task_scheduling.enums import ScheduleStage as S
from cutlass.experimental.task_scheduling.task_manager import print_schedule_list
from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.task_manager import (
    verify_complete_schedule,
)


def op(resource, stage):
    return resource, stage, 0, ""


def test_serial_pipeline_outlives_old_printer_bound():
    # Every producer acquires its storage before committing; each consumer
    # drains its input before producing the next stream. The critical path
    # exceeds the longest individual task by more than 100 steps.
    a = SimpleNamespace(name="a", pipeline_config=None)
    b = SimpleNamespace(name="b", pipeline_config=None)
    n = 128
    schedules = [
        [op(a, S.ProducerAcquire)] * n + [op(a, S.ProducerCommit)] * n,
        [op(a, S.ConsumerWait)] * n
        + [op(a, S.ConsumerRelease)] * n
        + [op(b, S.ProducerAcquire)] * n
        + [op(b, S.ProducerCommit)] * n,
        [op(b, S.ConsumerWait)] * n + [op(b, S.ConsumerRelease)] * n,
    ]
    initial = {id(a): (n, a), id(b): (n, b)}
    # This is the DSL 4.7 regression. A fixed newer printer may also pass.
    try:
        print_schedule_list(
            [None] * 3,
            "",
            {},
            dict(initial),
            initial,
            True,
            schedule_lists=schedules,
            verbose=False,
        )
    except ValueError as error:
        assert "not consumed" in str(error) or "not matched" in str(error)
    verify_complete_schedule(schedules, initial)


def test_complete_verifier_still_rejects_deadlock():
    resource = SimpleNamespace(name="empty", pipeline_config=None)
    with pytest.raises(ValueError, match="deadlock"):
        verify_complete_schedule([[op(resource, S.ConsumerWait)]], {})


@pytest.mark.parametrize("missing", [S.ConsumerWait, S.ConsumerRelease])
def test_complete_verifier_still_rejects_unbalanced_pipeline(missing):
    resource = SimpleNamespace(name="unbalanced", pipeline_config=None)
    schedule = [
        op(resource, s)
        for s in (
            S.ProducerAcquire,
            S.ProducerCommit,
            S.ConsumerWait,
            S.ConsumerRelease,
        )
        if s != missing
    ]
    with pytest.raises(ValueError, match="unconsumed|unmatched"):
        verify_complete_schedule([schedule], {id(resource): (1, resource)})
