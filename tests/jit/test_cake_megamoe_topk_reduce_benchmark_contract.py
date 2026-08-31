# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only regression tests for the MegaMoE reducer benchmark contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "bench_cake_megamoe_topk_reduce.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bench_cake_megamoe_topk_reduce",
    _BENCHMARK_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


@pytest.mark.parametrize("num_tokens,capacity", _BENCHMARK._SHAPES)
def test_matched_cutedsl_comparison_has_identical_reduction_work(
    num_tokens: int,
    capacity: int,
):
    plan = _BENCHMARK._comparison_plan(
        num_tokens,
        "vendored-cutedsl-matched",
    )

    assert plan["comparison_kind"] == "matched_work_kernel"
    assert plan["same_token_extent"]
    assert plan["same_reduction_work"]
    assert plan["native_work_tokens"] == num_tokens
    assert plan["baseline_work_tokens"] == num_tokens
    assert plan["baseline_tensor_extent"] == num_tokens
    assert plan["native_grid_ctas"] == plan["baseline_grid_ctas"]
    assert plan["native_io_bytes"] == plan["baseline_io_bytes"]

    partials = _BENCHMARK.torch.empty(capacity, 6, 4)
    out = _BENCHMARK.torch.empty(capacity, 4)
    baseline_partials, baseline_out = _BENCHMARK._matched_cutedsl_views(
        partials,
        out,
        plan,
    )
    assert baseline_partials.shape == (num_tokens, 6, 4)
    assert baseline_out.shape == (num_tokens, 4)


@pytest.mark.parametrize("num_tokens,capacity", _BENCHMARK._SHAPES)
def test_fixed_capacity_comparison_is_labelled_as_a_serving_scenario(
    num_tokens: int,
    capacity: int,
):
    del capacity
    plan = _BENCHMARK._comparison_plan(
        num_tokens,
        "vendored-cutedsl-fixed-capacity",
    )

    assert plan["comparison_kind"] == "legacy_fixed_capacity_serving_scenario"
    assert plan["baseline_work_tokens"] == 4096
    assert plan["baseline_tensor_extent"] == 4096
    assert plan["same_token_extent"] is (num_tokens == 4096)
    assert plan["same_reduction_work"] is (num_tokens == 4096)
    assert plan["baseline_grid_ctas"] == 4 * 4096
    assert plan["native_grid_ctas"] == 4 * num_tokens


def test_ordered_pytorch_does_not_claim_single_kernel_work_equivalence():
    plan = _BENCHMARK._comparison_plan(64, "ordered-pytorch")

    assert plan["comparison_kind"] == "matched_live_tokens_semantic_reference"
    assert plan["same_token_extent"]
    assert plan["same_reduction_work"] is None
    assert plan["baseline_work_tokens"] is None
    assert plan["baseline_tensor_extent"] is None
    assert plan["baseline_grid_ctas"] is None
    assert plan["baseline_io_bytes"] is None
