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
"""Public API and benchmark contract tests for selective TopK."""

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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("cutlass") is None
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100/SM103 and CuTe DSL",
)
def test_fixed_context_capacity_keeps_live_oracle_bounds(benchmark):
    args = SimpleNamespace(
        q=64,
        batch_size=1,
        compression_ratio=4,
        max_context_length=65536,
        backend="cutedsl",
        num_heads=64,
        head_dim=128,
        page_size=128,
        top_k=512,
        qkv_dtype="fp8",
        logits_dtype="float32",
        input_distribution="model",
        model_profile="DS-V4-Flash",
        model_config="DS-V4-Flash",
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
    assert result["correctness"]["rows"] == 64
    assert result["correctness"]["exact_error"] == 0
    assert result["correctness"]["max_topk_value_error_before"]["selective"] < 1e-4
    assert result["correctness"]["max_topk_value_error_after"]["selective"] < 1e-4


def test_summary_validates_identity_and_duplicates_without_cuda(benchmark, tmp_path):
    import json
    import subprocess

    pauses = [
        dict(
            after=route,
            samples=1,
            elapsed_s=5.0,
            block_start_utc_ns=4,
            pre_block_warmup=dict(
                route=route,
                requested_s=0.0,
                elapsed_s=0.0,
                replays=0,
                start_utc_ns=1,
                end_utc_ns=2,
            ),
        )
        for route in ("a", "b")
    ]
    samples = dict(raw_us=[10.0], median_us=10.0, p10_us=10.0, p90_us=10.0)
    row = dict(
        bs=1,
        q=8,
        num_heads=64,
        top_k=512,
        compression_ratio=1,
        kv=1024,
        correctness=dict(exact_error=0),
        full_gvr2=samples,
        selective=samples,
        step=dict(full_gvr2=samples, selective=samples, post_component_pauses=pauses),
        protocol=dict(
            order="AxN-5s-BxN-5s",
            samples_per_route=1,
            samples_per_block=1,
            sleep_between_samples=False,
            a="full_gvr2",
            b="selective",
            post_component_pauses=pauses,
            steady_state_warmup_requested_s=0.0,
        ),
    )
    result = dict(
        schema_version=21,
        status="partial",
        device=dict(name="B200"),
        environment=dict(torch="test"),
        rows=[row],
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    rendered = subprocess.run(
        [sys.executable, benchmark.__file__, "--summarize", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "Partial successful prefixes only" in rendered
    with pytest.raises(ValueError, match="duplicate comparison"):
        benchmark._render_results([("one", result), ("two", result)])
    other = {**result, "device": {"name": "different"}}
    other["rows"] = [{**row, "kv": 2048}]
    with pytest.raises(ValueError, match="one environment/GPU"):
        benchmark._render_results([("one", result), ("two", other)])


def test_continuous_public_dispatch_and_signatures(monkeypatch):
    import inspect
    import flashinfer
    from flashinfer.experimental import attn_scores

    seen = {}

    class Plan:
        def __init__(self, **kw):
            seen.update(kw)

        def plan(self, *args, **kw):
            return args, kw

        def run(self, *args, **kw):
            return args, kw

    monkeypatch.setattr(attn_scores, "_continuous_plan_type", lambda backend: Plan)
    public = flashinfer.FP8MQATopKWrapper
    assert public is flashinfer.attn_scores.FP8MQATopKWrapper
    assert (
        inspect.signature(public).parameters["num_heads"].default
        is inspect.Parameter.empty
    )
    assert tuple(inspect.signature(public.plan).parameters) == (
        "self",
        "qo_indptr",
        "kv_indptr",
        "num_query_rows",
        "next_n",
        "host_length_hints",
    )
    assert tuple(inspect.signature(public.run).parameters) == (
        "self",
        "q",
        "weights",
        "k",
        "k_scale",
        "out",
        "q_scale",
    )
    op = public(
        max_context_length=8192,
        max_chunk_size=64,
        max_batch_size=1,
        device="cuda",
        backend="cutedsl",
        num_heads=64,
        qkv_dtype="mxfp4",
        logits_dtype=torch.bfloat16,
    )
    assert seen["logits_dtype"] == torch.bfloat16
    assert op.plan(1, 2) == (
        (1, 2),
        {"num_query_rows": None, "next_n": None, "host_length_hints": None},
    )
    hints = (torch.tensor([3]), torch.tensor([4]))
    args, forwarded = op.plan(1, 2, num_query_rows=3, next_n=1, host_length_hints=hints)
    assert args == (1, 2)
    assert forwarded["num_query_rows"] == 3
    assert forwarded["next_n"] == 1
    assert forwarded["host_length_hints"] is hints
    assert op.run(1, 2, 3, k_scale=4, q_scale=5)[1]["k_scale"] == 4
    for name in ("page_size", "block_table", "kv_layout", "num_pages"):
        assert name not in inspect.signature(public).parameters


def test_selective_warning_and_lazy_import_in_fresh_process():
    import subprocess
    import textwrap

    program = r"""
        import sys
        import warnings
        import flashinfer
        from flashinfer.api_logging import ExperimentalWarning
        assert "flashinfer.experimental.attn_scores.selective_prefill" not in sys.modules
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalWarning)
            for _ in range(2):
                try:
                    flashinfer.FP8MQATopKWrapper(
                        num_heads=64, strategy="full", max_context_length=1024,
                        max_chunk_size=4, max_batch_size=1, device="cuda"
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError("full must be rejected")
        assert len([w for w in caught if issubclass(w.category, ExperimentalWarning)]) == 1
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], check=True, timeout=120
    )


def _supports_selective_gpu():
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("cutlass") is not None
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    )


@pytest.mark.skipif(
    not _supports_selective_gpu(), reason="requires SM100/SM103 and CuTe DSL"
)
def test_benchmark_cli_smoke(tmp_path):
    import json
    import os
    import subprocess

    script = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/bench_fp8_mqa_selective_logits.py"
    )
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--q",
            "64",
            "--kv",
            "4096",
            "--num-heads",
            "64",
            "--top-k",
            "512",
            "--warmups",
            "1",
            "--repeats",
            "2",
            "--settle-seconds",
            "0",
            "--math-mode",
            "relu_fast",
            "--weights-dtype",
            "float32",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    data = json.loads(output.read_text())
    assert data["status"] == "complete" and "PASS H64" in run.stdout
    assert len(data["rows"]) == 1 and data["rows"][0]["correctness"]["exact_error"] == 0
    summary = subprocess.run(
        [sys.executable, str(script), "--summarize", str(output)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert summary.returncode == 0, summary.stdout + summary.stderr
    assert "comparisons" in summary.stdout
