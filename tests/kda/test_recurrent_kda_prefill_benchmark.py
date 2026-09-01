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

import runpy
from pathlib import Path

import pytest


_BENCHMARK = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "bench_recurrent_kda_prefill.py"
    )
)
_resolve_recorded_cake_route = _BENCHMARK["_resolve_recorded_cake_route"]
_timing_iteration_budget = _BENCHMARK["_timing_iteration_budget"]


def test_recorded_cake_route_serializes_legacy_single_module():
    assert _resolve_recorded_cake_route([("m128", "sm100f")]) == (
        "m128",
        "sm100f",
        ["m128"],
    )


def test_recorded_cake_route_serializes_combined_bt16_module():
    assert _resolve_recorded_cake_route([("bt16_prepare_chain_m64_s8", "sm100f")]) == (
        "bt16_prepare_chain_m64",
        "sm100f",
        ["bt16_prepare_chain_m64_s8"],
    )


def test_recorded_cake_route_serializes_bt16_physical_pair():
    assert _resolve_recorded_cake_route(
        [
            ("bt16_prepare_beta_tma", "sm100f"),
            ("bt16_chain_m64_s8", "sm100f"),
        ]
    ) == (
        "bt16_prepare_chain_m64",
        "sm100f",
        ["bt16_prepare_beta_tma", "bt16_chain_m64_s8"],
    )


@pytest.mark.parametrize(
    "routes",
    [
        [("bt16_prepare", "sm100f")],
        [
            ("bt16_chain_m64_s8", "sm100f"),
            ("bt16_prepare", "sm100f"),
        ],
        [
            ("bt16_prepare", "sm100a"),
            ("bt16_chain_m64_s8", "sm100f"),
        ],
    ],
)
def test_recorded_cake_route_rejects_malformed_bt16_pair(routes):
    with pytest.raises(RuntimeError, match="ordered BT16 prepare/chain pair"):
        _resolve_recorded_cake_route(routes)


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [
        (8, (1, 1)),
        (10, (1, 3)),
        (126, (20, 100)),
        (1024, (20, 100)),
    ],
)
def test_timing_iteration_budget_never_exhausts_rotating_state(capacity, expected):
    budget = _timing_iteration_budget(
        state_rotation_capacity=capacity,
        requested_dry_run_iters=20,
        requested_repeat_iters=100,
    )
    assert budget == expected
    assert 6 + sum(budget) <= capacity


def test_timing_iteration_budget_requires_one_dry_and_measured_call():
    with pytest.raises(ValueError, match="six CUPTI estimate calls"):
        _timing_iteration_budget(
            state_rotation_capacity=7,
            requested_dry_run_iters=20,
            requested_repeat_iters=100,
        )
