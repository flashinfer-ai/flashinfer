# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared BF16 stages preserve tactic validation and graph-local identity."""

from unittest.mock import patch

import pytest

from flashinfer.fused_moe import cudnn_backend as backend


FIRST = (20400, ((26, 32), (1000, 0)))
SECOND = (20400, ((26, 32), (1000, 1)))
DECLINED = (20400, ((26, 64), (1000, 1)))


class Graph:
    def __init__(self, records):
        self.records = tuple(records)

    def get_execution_plan_count(self):
        return len(self.records)

    def get_engine_and_knobs_at_index(self, index):
        engine, knobs = self.records[index]
        return engine, dict(knobs)


def stage(records=(FIRST, DECLINED, SECOND)):
    result = object.__new__(backend._Stage)
    result.graph = Graph(records)
    result.tactic_indices = {
        record: index for index, record in enumerate(records) if record != DECLINED
    }
    return result


@pytest.mark.parametrize(
    "tactic,index", [(FIRST, 0), (SECOND, 2), (0, 0), (2, 2), (-1, -1)]
)
def test_prepared_tactics_do_not_query_graph(tactic, index):
    prepared = stage()
    with patch.object(
        backend, "_plan_index", side_effect=AssertionError("repeated graph lookup")
    ):
        assert prepared.plan_index(tactic) == index


@pytest.mark.parametrize(
    "tactic,index",
    [
        ((20400, ((1000, 1), (26, 32))), 2),
        (("20400", (("26", "32"), ("1000", "1"))), 2),
        ((20400.0, ((26.0, 32.0), (1000.0, 1.0))), 2),
        ((20400, [[26, 32], [1000, 1]]), 2),
    ],
)
def test_legacy_record_normalization_remains_supported(tactic, index):
    assert stage().plan_index(tactic) == index


@pytest.mark.parametrize(
    "tactic",
    [
        True,
        False,
        None,
        -2,
        3,
        1,
        0.0,
        [],
        (),
        (20400,),
        DECLINED,
        (99999, ()),
        (20400, ((26, 32), (1000, 2))),
    ],
)
def test_invalid_or_unprepared_tactics_are_rejected(tactic):
    with pytest.raises((TypeError, ValueError)):
        stage().plan_index(tactic)


def test_one_record_resolves_against_each_stages_prepared_domain():
    a, b = stage(), stage((SECOND, FIRST, DECLINED))
    assert a.plan_index(FIRST) == 0
    assert b.plan_index(FIRST) == 1
    assert a.plan_index(SECOND) == 2
    assert b.plan_index(SECOND) == 0


def test_mutable_legacy_knobs_are_revalidated_after_change():
    prepared = stage()
    knobs = [[26, 32], [1000, 0]]
    tactic = (20400, knobs)
    assert prepared.plan_index(tactic) == 0
    knobs[1][1] = 1
    assert prepared.plan_index(tactic) == 2
    knobs[0][1] = 64
    with pytest.raises(ValueError, match="unavailable"):
        prepared.plan_index(tactic)


def test_joint_record_rejects_an_unprepared_second_stage():
    state = {"fc1": stage(), "fc2": stage((FIRST, DECLINED))}
    with pytest.raises(ValueError, match="unavailable"):
        backend.CudnnMoeRunner._stage_indices(state, (FIRST, SECOND))
