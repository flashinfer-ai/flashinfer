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

"""CPU-only structural checks for the PrimTS context signoff matrix."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_attention_ts_context.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bench_attention_ts_context_for_test", _BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)


def test_context_performance_matrix_is_exact_cartesian_product():
    cases = _BENCHMARK.context_perf_cases()
    assert len(cases) == 48
    assert len({case.case_id for case in cases}) == 48
    assert {
        (
            case.head_dim,
            case.num_heads_kv,
            case.batch_size,
            case.max_seq_len_q,
            case.qkv_layout,
        )
        for case in cases
    } == set(
        itertools.product(
            (128, 256),
            (32, 4),
            (1, 4),
            (1024, 4096, 16384),
            ("separate_qkv", "paged_kv"),
        )
    )
    assert all(case.dtype == "float8_e4m3fn" for case in cases)
    assert all(case.num_heads_q == 32 for case in cases)
    assert all(case.max_seq_len_q == case.max_seq_len_kv for case in cases)
    assert all(case.mask_type == "dense" for case in cases)
    assert all(
        case.page_size == (32 if case.qkv_layout == "paged_kv" else None)
        for case in cases
    )


def test_context_performance_matrix_preserves_exact_timing_shapes():
    for case in _BENCHMARK.context_perf_cases():
        lengths = _BENCHMARK._sequence_lengths(case)
        assert len(lengths) == case.batch_size
        assert lengths == (case.max_seq_len_q,) * case.batch_size


def test_context_performance_layout_pairs_share_one_unique_logical_seed():
    cases = _BENCHMARK.context_perf_cases()
    seeds_by_shape: dict[tuple[int, int, int, int], set[int]] = {}
    for case in cases:
        shape = (
            case.head_dim,
            case.num_heads_kv,
            case.batch_size,
            case.max_seq_len_q,
        )
        seeds_by_shape.setdefault(shape, set()).add(
            _BENCHMARK._case_seed(case, base_seed=20260715)
        )
    assert len(seeds_by_shape) == 24
    assert all(len(seeds) == 1 for seeds in seeds_by_shape.values())
    assert len({next(iter(seeds)) for seeds in seeds_by_shape.values()}) == 24


def test_context_performance_paged_indices_are_unique_and_nonidentity():
    for num_used_pages in (32, 128, 512, 128, 512, 2048):
        indices = _BENCHMARK._permuted_page_ids(
            num_used_pages, num_used_pages + 4, seed=1111 + num_used_pages
        )
        assert len(indices) == num_used_pages
        assert len(set(indices)) == num_used_pages
        assert indices != tuple(range(num_used_pages))
        assert min(indices) >= 0
        assert max(indices) < num_used_pages + 4


def test_context_performance_accuracy_samples_are_unique_for_b1_and_b4():
    cases = _BENCHMARK.context_perf_cases()
    for batch_size in (1, 4):
        case = next(case for case in cases if case.batch_size == batch_size)
        inputs = type(
            "Inputs",
            (),
            {
                "lengths": (case.max_seq_len_q,) * batch_size,
                "case": case,
            },
        )()
        points = _BENCHMARK._accuracy_sample_points(inputs)
        assert len(points) == 8
        assert len(set(points)) == 8
        assert {point[0] for point in points} == set(range(batch_size))

    gqa_case = next(case for case in cases if case.num_heads_kv == 4)
    inputs = type(
        "Inputs",
        (),
        {
            "lengths": (gqa_case.max_seq_len_q,),
            "case": gqa_case,
        },
    )()
    points = _BENCHMARK._accuracy_sample_points(inputs)
    head_ratio = gqa_case.num_heads_q // gqa_case.num_heads_kv
    assert {query_head // head_ratio for _, _, query_head in points} == set(range(4))
