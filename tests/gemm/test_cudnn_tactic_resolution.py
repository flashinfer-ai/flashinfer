"""Unit tests for cuDNN GEMM structured (engine_id, knobs) tactics.

The cuDNN GEMM runners use the plan's *structured identity* -- engine global
index + knob choices, read via the frontend's get_engine_and_knobs_at_index --
as the autotuner tactic, and replay it by pinning via create_execution_plan
(skipping the heuristics query).  This is a version-robust alternative to a
bare positional plan index, which silently mis-points after the plan list is
re-enumerated across cudnn-frontend / backend versions.

These tests cover the tactic enumeration / encode / classify helpers in
isolation with a stub graph, so they need neither a GPU nor a loadable cuDNN
backend.
"""

import json

import pytest

from flashinfer.gemm.gemm_base import (
    _cudnn_plan_tactics,
    _cudnn_is_structured_tactic,
    _cudnn_structured_pin,
)
from flashinfer.autotuner import _tactic_to_json, _json_to_tactic


class _StubGraph:
    """Stand-in for a cudnn-frontend pygraph after build_plans().

    ``plans`` is a list of (engine_id, {knob_int: value}) for each plan index.
    """

    def __init__(self, plans):
        self._plans = plans

    def get_execution_plan_count(self):
        return len(self._plans)

    def get_engine_and_knobs_at_index(self, i):
        engine_id, knobs = self._plans[i]
        # Frontend returns a dict keyed by knob_type; the stub keys by int and
        # the production code only does int(k), so plain ints are equivalent
        # for these tests.
        return engine_id, dict(knobs)


def test_plan_tactics_are_structured_and_int_encoded():
    g = _StubGraph([(0, {}), (7, {9: 41, 16: 1})])
    tactics = _cudnn_plan_tactics(g)
    # engine_id + sorted (knob_int, value) pairs; all plain ints (JSON-safe).
    assert tactics == [(0, ()), (7, ((9, 41), (16, 1)))]
    for t in tactics:
        assert _cudnn_is_structured_tactic(t)


def test_knobs_are_sorted_deterministically():
    # Same engine+knobs in different dict order must yield the same tactic.
    g1 = _StubGraph([(3, {16: 1, 9: 2})])
    g2 = _StubGraph([(3, {9: 2, 16: 1})])
    assert (
        _cudnn_plan_tactics(g1) == _cudnn_plan_tactics(g2) == [(3, ((9, 2), (16, 1)))]
    )


def test_is_structured_tactic_discrimination():
    assert _cudnn_is_structured_tactic((3, ((9, 2),)))
    assert _cudnn_is_structured_tactic((0, ()))  # engine with no knobs
    assert _cudnn_is_structured_tactic((3, [[9, 2]]))  # JSON list form
    # Not structured -- the autotuner fallback / disabled-autotune tactic:
    assert not _cudnn_is_structured_tactic(-1)
    assert not _cudnn_is_structured_tactic(5)
    assert not _cudnn_is_structured_tactic("eng3_k9=2")


def test_structured_pin_normalizes_to_int_pairs():
    engine_id, pin_knobs = _cudnn_structured_pin((3, ((9, 2), (16, 1))))
    assert engine_id == 3
    assert pin_knobs == ((9, 2), (16, 1))
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in pin_knobs)


def test_tactic_json_round_trip():
    # The autotuner persists tactics via _tactic_to_json / _json_to_tactic;
    # the structured tactic must survive a JSON round-trip unchanged so the
    # on-disk autotune cache replays the same pinned plan.
    tactic = (7, ((9, 41), (16, 1)))
    restored = _json_to_tactic(json.loads(json.dumps(_tactic_to_json(tactic))))
    assert restored == tactic
    assert _cudnn_is_structured_tactic(restored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
