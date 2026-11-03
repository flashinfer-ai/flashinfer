# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regressions for native rounding, graph timing and context capacity."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/bench_fp8_mqa_selective_logits.py"
    )
    spec = importlib.util.spec_from_file_location("mqa_oracle_test_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("native_first", "failure"),
    [
        (155.81251525878906, None),
        (155.8125, "selective"),
        (156.0, "native producer FP32"),
    ],
)
def test_native_half_cutoff_and_independent_math(benchmark, native_first, failure):
    # At the half midpoint, valid FP32 reduction orders can differ by one ULP.
    # Native rounding makes keys 0 and 1 tie; exact native selection permits 0.
    q = torch.ones((1, 1, 1))
    keys = [torch.tensor([[155.8125], [155.82]])]
    groups = SimpleNamespace(q_prefix=[0, 1], real_ends=[2])
    options = dict(
        outputs={"full_gvr2": torch.tensor([[1]]), "selective": torch.tensor([[0]])},
        q=q,
        kv=keys,
        scales=[torch.ones(2)],
        weights=torch.ones((1, 1)),
        groups=groups,
        top_k=1,
        atol=1e-4,
        rtol=1e-4,
        logits_dtype=torch.float16,
        native_scores=torch.tensor([[native_first, 155.82]]),
    )
    if failure:
        with pytest.raises(AssertionError, match=failure):
            benchmark._check_reference(**options)
    else:
        result = benchmark._check_reference(**options)
        assert result["selective"] == 0
        assert 0 < result["native_producer_fp32"] < 1e-4


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("settle_seconds", [0.0, 2.0])
def test_same_route_settles_before_each_timed_block(
    benchmark, monkeypatch, reverse, settle_seconds
):
    state = {"time": 0.0, "timed": False}
    calls, samples = [], []

    def call(label):
        calls.append((label, state["timed"]))
        state["time"] += 0.25

    def full():
        call("a")

    def selective():
        call("b")

    def event(fn):
        state["timed"] = True
        samples.append(len(calls))
        fn()
        state["timed"] = False
        return 250000.0

    def sleep(seconds):
        state["time"] += seconds

    monkeypatch.setattr(
        benchmark,
        "time",
        SimpleNamespace(
            perf_counter=lambda: state["time"],
            time_ns=lambda: int((1700000000 + state["time"]) * 1e9),
            sleep=sleep,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    )
    monkeypatch.setattr(benchmark, "_record_event", event)
    a, b, pauses = benchmark._paired_events(
        full, selective, 20, 30, reverse=reverse, settle_seconds=settle_seconds
    )
    assert a == b == [250000.0] * 30
    assert len(samples) == 60
    order = ("b", "a") if reverse else ("a", "b")
    for block, label in enumerate(order):
        first = samples[block * 30]
        assert calls[first : first + 30] == [(label, True)] * 30
        if settle_seconds:
            assert calls[first - 8 : first] == [(label, False)] * 8
        warmup = pauses[block]["pre_block_warmup"]
        assert warmup["route"] == label
        assert warmup["replays"] == int(settle_seconds / 0.25)
        assert warmup["elapsed_s"] == settle_seconds
    for label in ("a", "b"):
        assert calls.count((label, False)) == 20 + int(settle_seconds / 0.25)
    benchmark._validate_pauses(pauses, 30, reverse, settle_seconds)


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_invalid_settling_duration_is_rejected_before_cuda(benchmark, duration):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        benchmark._paired_events(None, None, 20, 30, settle_seconds=duration)


@pytest.mark.parametrize("defect", ["wrong_route", "too_short", "after_block_start"])
def test_settling_evidence_must_precede_its_own_block(benchmark, defect):
    pauses = [
        dict(
            after=label,
            samples=30,
            elapsed_s=5.0,
            block_start_utc_ns=4000000000,
            block_end_utc_ns=5000000000,
            pre_block_warmup=dict(
                route=label,
                requested_s=2.0,
                elapsed_s=2.0,
                replays=100,
                start_utc_ns=1000000000,
                end_utc_ns=3000000000,
            ),
        )
        for label in ("a", "b")
    ]
    benchmark._validate_pauses(pauses, 30, settle_seconds=2.0)
    warmup = pauses[1]["pre_block_warmup"]
    if defect == "wrong_route":
        warmup["route"] = "a"
    elif defect == "too_short":
        warmup["elapsed_s"] = 1.9
    else:
        warmup["end_utc_ns"] = 4500000000
    with pytest.raises(ValueError, match="same-route warmup"):
        benchmark._validate_pauses(pauses, 30, settle_seconds=2.0)


@pytest.mark.parametrize(
    ("live", "capacity", "ratio", "expected"),
    [
        (8192, None, 1, (8192, 8192)),
        (8192, None, 4, (8192, 2048)),
        (8192, 1048576, 1, (1048576, 1048576)),
        (8192, 1048576, 4, (1048576, 262144)),
        (8193, 8195, 4, (8195, 2048)),
    ],
)
def test_context_capacity_uses_original_tokens(
    benchmark, live, capacity, ratio, expected
):
    assert benchmark._context_capacity(live, capacity, ratio) == expected


def test_zero_physical_capacity_is_rejected(benchmark):
    with pytest.raises(ValueError, match="at least one physical key"):
        benchmark._context_capacity(1, None, 4)


@pytest.mark.parametrize("capacity", [0, -1, 4096])
def test_context_capacity_rejected_before_cuda(benchmark, monkeypatch, capacity):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--q",
            "8192",
            "--kv",
            "8192",
            "--max-context-length",
            str(capacity),
        ],
    )
    with pytest.raises(ValueError, match="max-context-length"):
        benchmark.main()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("backend", "heads", "model", "named"),
    [
        ("v1", 4, "Qwen3.8-Flash-Next", None),
        ("v2", 4, "Qwen3.8-Flash-Next", None),
        ("v2", 4, "Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next"),
        ("v2", 64, "DS-V4-Flash", "DS-V4-Flash"),
    ],
)
def test_fixed_context_capacity_keeps_live_oracle_bounds(
    benchmark, backend, heads, model, named
):
    args = SimpleNamespace(
        q=64,
        batch_size=1,
        compression_ratio=4,
        max_context_length=65536,
        backend=backend,
        num_heads=heads,
        head_dim=128,
        page_size=128,
        top_k=512,
        qkv_dtype="fp8",
        logits_dtype="float32",
        input_distribution="model",
        model_profile=model,
        model_config=named,
        full_query_tile=None,
        full_epi_subtiles=1,
        full_w_cache_regs=160,
        seed=1,
        warmups=1,
        repeats=1,
        settle_seconds=0.0,
        layer_only=True,
    )
    result = benchmark.run_row(4096, args)
    assert result["kv"] == 4096
    assert result["physical_kv"] == 1024
    assert result["max_context_length"] == 65536
    assert result["physical_capacity"] == 16384
    assert result["capacity_pages_per_request"] == 128
    assert result["full_config"]["logical_score_width"] == 16384
    assert result["full_config"]["score_pitch"] >= 16384
    assert result["correctness"]["rows"] == 64
    assert result["correctness"]["exact_error"] == 0
    assert result["correctness"]["max_topk_value_error_before"]["selective"] < 1e-4
    assert result["correctness"]["max_topk_value_error_after"]["selective"] < 1e-4
