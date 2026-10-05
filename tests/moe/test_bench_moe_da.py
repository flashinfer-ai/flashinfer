"""End-to-end command-line contract for the backend-aware DA MoE benchmark."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from benchmarks.bench_moe_da import _time_graphs_counterbalanced
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("corruption", ["zeros", "scaled", "localized", "nonfinite"])
def test_numerical_gate_rejects_missing_or_corrupt_computation(corruption):
    from benchmarks.bench_moe_da import _check_output

    expected = torch.full((4096,), 1e-3)
    actual = expected.clone()
    if corruption == "zeros":
        actual.zero_()
    elif corruption == "scaled":
        actual.mul_(1.5)
    elif corruption == "localized":
        expected.fill_(1.0)
        actual.copy_(expected)
        actual[0] = 2.0
    else:
        actual[0] = float("nan")
    with pytest.raises(AssertionError):
        _check_output(actual, expected)


def test_numerical_gate_accepts_rounding_and_exact_zeros():
    from benchmarks.bench_moe_da import _check_output

    expected = torch.full((32,), 1e-3)
    assert _check_output(expected * 1.01, expected) == pytest.approx(0.01, rel=1e-4)
    assert _check_output(torch.zeros(32), torch.zeros(32)) == 0.0


def test_trtllm_fp8_logits_rejected_before_allocation(monkeypatch) -> None:
    from benchmarks import bench_moe_da as bench

    monkeypatch.setattr(
        bench, "_canonical_inputs", lambda *_: pytest.fail("allocated unsupported mode")
    )
    with pytest.raises(ValueError, match="requires routed inputs"):
        bench._prepare_precision("fp8_per_tensor", None, "trtllm", "logits")


@pytest.mark.parametrize("failure", ["acquire", "diagnostic", "policy", "flush"])
def test_benchmark_releases_graph_resources_on_setup_failure(monkeypatch, failure):
    from benchmarks import bench_moe_da as bench

    events = []
    graphs = [
        SimpleNamespace(reset=lambda: events.append("reset_noda")),
        SimpleNamespace(reset=lambda: events.append("reset_da")),
    ]
    captures = iter(graphs)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(
        bench,
        "_prepare_precision",
        lambda *_: SimpleNamespace(stage=lambda *_: None, invoke=lambda: None),
    )
    monkeypatch.setattr(bench, "_realization", lambda *_: (None, None))
    monkeypatch.setattr(bench, "_capture", lambda *_: next(captures))
    monkeypatch.setattr(bench, "autotune", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *_: nullcontext())
    monkeypatch.setattr(
        bench,
        "da_moe_acquire_graph_leases",
        fail
        if failure == "acquire"
        else lambda *_: (
            SimpleNamespace(release=lambda: events.append("release_lease")),
        ),
    )
    monkeypatch.setattr(
        bench,
        "_matching_diagnostic",
        fail
        if failure == "diagnostic"
        else lambda *_: {"policy": "invalid" if failure == "policy" else "da_switch"},
    )
    monkeypatch.setattr(bench, "_cold_l2_buffers", fail)
    monkeypatch.setattr(
        bench, "da_moe_release_resources", lambda: events.append("release_resources")
    )
    with pytest.raises(RuntimeError):
        bench._benchmark_precision(
            "nvfp4", SimpleNamespace(num_tokens=5), ("uniform",), None, False, 0, 2
        )
    assert events == ["reset_noda", "reset_da"] + (
        [] if failure == "acquire" else ["release_lease"]
    ) + ["release_resources"]


@pytest.mark.parametrize(
    "mode,backends",
    [("logits", ("prims-ts-nvfp4",)), ("routed", None), ("routed", ("cutedsl",))],
)
def test_deepseek_rejects_unconsumed_distribution_sweeps(mode, backends) -> None:
    from benchmarks.bench_moe_deepseek import run_benchmark

    with pytest.raises(ValueError, match="Distribution sweeps require"):
        run_benchmark(
            token_counts=[5],
            routing_input_mode=mode,
            backends=backends,
            distributions=("uniform", "ddist:4"),
        )


class _FakeGraph:
    def __init__(self, label: str, trace: list[str]) -> None:
        self.label = label
        self.trace = trace
        self.start = _FakeEvent()
        self.end = _FakeEvent()

    def replay(self) -> None:
        self.trace.append(self.label)


class _FakeFlushBuffer:
    def __init__(self, label: str, trace: list[str]) -> None:
        self.label = label
        self.trace = trace

    def zero_(self) -> None:
        self.trace.append(self.label)


class _FakeEvent:
    def record(self) -> None:
        pytest.fail("timing events must already be captured in the graph")

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, _other: object) -> float:
        return 1.0


def test_graph_timing_counterbalances_order_and_flush_buffers(monkeypatch) -> None:
    """Each graph must occupy both timing positions and see both eviction buffers."""
    trace: list[str] = []
    no_da_graph = _FakeGraph("noda", trace)
    da_graph = _FakeGraph("da", trace)
    buffers = (_FakeFlushBuffer("flush0", trace), _FakeFlushBuffer("flush1", trace))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda **_: pytest.fail("allocated an uncaptured event")
    )

    no_da_ms, da_ms = _time_graphs_counterbalanced(
        cast(Any, no_da_graph),
        cast(Any, da_graph),
        cast(Any, buffers),
        warmup=2,
        iterations=2,
    )

    assert no_da_ms == da_ms == 1.0
    assert trace == [
        "noda",
        "da",
        "da",
        "noda",
        "flush0",
        "noda",
        "flush1",
        "da",
        "flush0",
        "da",
        "flush1",
        "noda",
    ]


@pytest.mark.parametrize("iterations", (0, 1, -2))
def test_graph_timing_rejects_unbalanced_iteration_counts(iterations: int) -> None:
    """Exact counterbalancing requires a positive even number of samples per graph."""
    with pytest.raises(ValueError, match="positive, even"):
        _time_graphs_counterbalanced(
            cast(Any, None),
            cast(Any, None),
            cast(Any, ()),
            warmup=0,
            iterations=iterations,
        )


def _require_sm100() -> None:
    """Skip the CLI lifecycle test unless an SM100-family GPU is active."""
    if not torch.cuda.is_available():
        pytest.skip("production DA benchmark requires CUDA")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("production DA benchmark requires an SM100-family GPU")


def test_relu2_benchmark_preparation_uses_non_gated_squared_activation(
    monkeypatch,
):
    from benchmarks import bench_moe_da as bench

    _require_sm100()
    shape = bench.BenchmarkShape(
        num_tokens=8,
        num_experts=32,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=4,
        hidden_size=512,
        intermediate_size=512,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=8,
        activation="relu2",
    )
    hidden, w1, w2, ids, weights = bench._canonical_inputs(shape)
    assert w1.shape == (32, 512, 512)
    hidden.zero_()
    hidden[:, 0] = 0.5
    hidden[:, 1] = -0.5
    w1.zero_()
    w2.zero_()
    for channel in (0, 1):
        w1[:, channel, channel] = 1
        w2[:, channel, channel] = 1
    ids.copy_(torch.arange(4, device="cuda", dtype=torch.int32).expand_as(ids))
    weights.fill_(0.25)
    monkeypatch.setattr(
        bench, "_canonical_inputs", lambda _: (hidden, w1, w2, ids, weights)
    )
    # Use BF16 for this exact formula oracle; FP4's intermediate quantization
    # needs a quantized reference rather than this unquantized expected value.
    prepared = bench._prepare_precision("bf16", shape, backend="prims_ts")
    prepared.stage(ids, weights)
    with bench._temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        actual = prepared.invoke()
    expected = torch.zeros_like(actual)
    expected[:, 0] = (
        0.25  # ReLU alone would produce 0.5; a missing ReLU keeps channel 1.
    )
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)


def _benchmark_command(
    cache: Path,
    output: Path,
    *,
    cache_only: bool,
    backend: str = "trtllm",
    activation: str = "swiglu",
    precision: str = "nvfp4",
    swiglu_params: tuple[float, float, float] | None = None,
) -> list[str]:
    """Build one bounded public CLI invocation for tuning or cache-only replay."""
    # Keep one compact real NVFP4 shape while exercising two distinct selector distributions.
    command = [
        sys.executable,
        "benchmarks/bench_moe_da.py",
        "--backend",
        backend,
        "--activation",
        activation,
        "--precision",
        precision,
        "--distributions",
        "uniform,ddist:4",
        "--mixture",
        "uniform=50,ddist:4=50",
        "--num-tokens",
        "64",
        "--num-experts",
        "32",
        "--local-num-experts",
        "32",
        "--top-k",
        "4",
        "--hidden-size",
        "128",
        "--intermediate-size",
        "128",
        "--n-group",
        "4",
        "--topk-group",
        "2",
        "--tune-max-num-tokens",
        "64",
        "--warmup",
        "0",
        "--iters",
        "2",
        "--cache",
        str(cache),
        "--json-out",
        str(output),
        "--mixture-out",
        str(output.with_suffix(".mixtures.json")),
        "--table-out",
        str(output.with_suffix(".md")),
    ]
    # Cache-only replay changes only the public lifecycle flag and reuses the same operation key.
    if cache_only:
        command.append("--skip-autotune")
    if swiglu_params is not None:
        for parameter, value in zip(
            ("alpha", "beta", "limit"), swiglu_params, strict=True
        ):
            command.extend((f"--swiglu-{parameter}", str(value)))
    return command


def _assert_result_file(path: Path) -> None:
    """Validate finite, numerical, topology-aware benchmark result records."""
    rows = json.loads(path.read_text())
    mixtures = json.loads(path.with_suffix(".mixtures.json").read_text())
    assert len(mixtures) == 1
    assert mixtures[0]["noda_ms"] == pytest.approx(sum(r["noda_ms"] for r in rows) / 2)
    assert mixtures[0]["da_ms"] == pytest.approx(sum(r["da_ms"] for r in rows) / 2)
    assert f"| {rows[0]['precision']} |" in path.with_suffix(".md").read_text()
    assert len(rows) == 2
    assert {row["distribution"] for row in rows} == {"uniform", "ddist:4"}
    assert all(row["status"] == "pass" for row in rows)
    assert all(row["finite"] is True for row in rows)
    assert all(float(row["max_abs_difference"]) <= 3e-2 for row in rows)
    for field in ("noda_autotune_ms", "da_autotune_ms"):
        values = {float(row[field]) for row in rows}
        assert len(values) == 1
        assert all(math.isfinite(value) and value >= 0.0 for value in values)
    policies = {row["policy"] for row in rows}
    assert policies <= {"da_switch", "da_single_body"}
    assert len(policies) == 1
    capture_policies = {row["capture_policy"] for row in rows}
    assert len(capture_policies) == 1
    capture_policy = capture_policies.pop()
    if capture_policy == "da_switch":
        assert policies == {"da_switch"}
        assert all(row["conditional_nodes"] == 1 for row in rows)
        assert all(row["selected_body"] is not None for row in rows)
    elif capture_policy == "da_single_body":
        assert policies == {"da_single_body"}
        assert all(row["conditional_nodes"] in (None, 0) for row in rows)
        assert all(int(row["selected_body"]) == 0 for row in rows)
    else:
        assert capture_policy == "noda_capture_fallback"
        assert policies == {"da_switch"}
        assert all(row["capture_fallback_reason"] for row in rows)
        assert all(row["conditional_nodes"] in (None, 0) for row in rows)
        assert all(row["selected_body"] is None for row in rows)


@pytest.mark.parametrize(
    "backend,precision,activation,swiglu_params",
    [
        ("trtllm", "nvfp4", "swiglu", None),
        ("prims_ts", "bf16", "relu2", None),
        ("trtllm", "nvfp4", "relu2", None),
        ("prims_ts", "nvfp4", "swiglu", (1.0, 0.0, 10.0)),
        ("prims_ts", "nvfp4", "swiglu", (1.702, 1.0, 7.0)),
    ],
)
def test_cli_json_cache_restores_in_a_fresh_process(
    tmp_path: Path, backend: str, precision: str, activation: str, swiglu_params
) -> None:
    """The public JSON tuning cache must restore DA replay without profiling."""
    _require_sm100()
    cache = tmp_path / "tuning-cache.json"
    tuned = tmp_path / "tuned.json"
    restored = tmp_path / "restored.json"
    # Start from the user environment, then pin only public tuning/cache controls for this process.
    environment = os.environ.copy()
    python_path = [str(Path.cwd())]
    if inherited_python_path := environment.get("PYTHONPATH"):
        python_path.append(inherited_python_path)
    environment.update(
        {
            "FLASHINFER_DA_BASELINE_GUARD": "0",
            "FLASHINFER_WORKSPACE_BASE": str(Path.cwd() / ".cache"),
            "MAX_JOBS": "8",
            "PYTHONPATH": os.pathsep.join(python_path),
        }
    )
    for name in (
        "CUDA_LAUNCH_BLOCKING",
        "FLASHINFER_CUDA_ARCH_LIST",
        "FLASHINFER_JIT_DIR",
        "FLASHINFER_NVCC_THREADS",
    ):
        environment.pop(name, None)

    # First process must tune and persist one operation record before validating its public rows.
    subprocess.run(
        _benchmark_command(
            cache,
            tuned,
            cache_only=False,
            backend=backend,
            activation=activation,
            precision=precision,
            swiglu_params=swiglu_params,
        ),
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )
    cache_payload = json.loads(cache.read_text())
    da_records = cache_payload["_records"]["moe_da"]
    assert len(da_records) == 1
    operation_key, record = next(iter(da_records.items()))
    assert json.loads(operation_key)["backend"] == backend
    assert record["backend"] == backend
    _assert_result_file(tuned)

    # A second process proves cache-only replay can restore the same public result contract.
    subprocess.run(
        _benchmark_command(
            cache,
            restored,
            cache_only=True,
            backend=backend,
            activation=activation,
            precision=precision,
            swiglu_params=swiglu_params,
        ),
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )
    _assert_result_file(restored)
    tuned_rows = json.loads(tuned.read_text())
    restored_rows = json.loads(restored.read_text())
    for fresh, cached in zip(tuned_rows, restored_rows, strict=True):
        assert fresh["activation"] == cached["activation"] == activation
        if swiglu_params is not None:
            for parameter, value in zip(
                ("alpha", "beta", "limit"), swiglu_params, strict=True
            ):
                assert (
                    fresh[f"swiglu_{parameter}"]
                    == cached[f"swiglu_{parameter}"]
                    == value
                )
        for field in ("distribution", "routing_sha256", "policy", "selected_body"):
            assert fresh[field] == cached[field]


def _mixture_rows():
    # One profile gains 2x and the other loses 2x; latency weighting gives 0.8x,
    # whereas averaging speedups would incorrectly report a 1.25x improvement.
    common = dict(
        backend="prims_ts",
        precision="nvfp4",
        num_tokens=96,
        num_experts=32,
        local_num_experts=16,
        local_expert_offset=0,
        top_k=4,
        hidden_size=128,
        intermediate_size=256,
        activation="swiglu",
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
        execution_mode="graph",
        timing_protocol="counterbalanced_cold_l2",
        routing_input_mode="routed",
        baseline_guard_enabled=False,
        capture_policy="da_switch",
        selected_body=0,
        max_abs_difference=0.0,
        status="pass",
        finite=True,
    )
    return [
        dict(common, distribution="ddist:1.1", noda_ms=2.0, da_ms=1.0),
        dict(common, distribution="ddist:4", noda_ms=2.0, da_ms=4.0, selected_body=1),
    ]


def test_mixture_uses_weighted_latency_and_keeps_shape_groups_separate():
    from benchmarks import bench_moe_da as bench

    rows = _mixture_rows()
    rows += [dict(r, num_tokens=128, noda_ms=r["noda_ms"] * 2) for r in rows]
    mixtures = (bench._parse_mixture("ddist:1.1=50,ddist:4=50"),)
    results = bench._summarize_mixtures(rows, mixtures)
    assert [r["speedup_da_over_noda"] for r in results] == pytest.approx([0.8, 1.6])
    assert results[0]["noda_ms"] == 2.0
    assert results[0]["da_ms"] == 2.5
    assert results[0]["selected_bodies"] == {"ddist:1.1": 0, "ddist:4": 1}
    table = bench._mixture_tables(results)
    assert "| Dtype | 96 | 128 |" in table
    assert "| nvfp4 | 0.8000x | 1.6000x |" in table


def test_mixture_keeps_activation_groups_separate():
    from benchmarks import bench_moe_da as bench

    rows = _mixture_rows()
    rows += [dict(r, activation="relu2", da_ms=r["da_ms"] * 2) for r in rows]
    results = bench._summarize_mixtures(
        rows, (bench._parse_mixture("ddist:1.1=1,ddist:4=1"),)
    )
    assert [r["activation"] for r in results] == ["swiglu", "relu2"]
    assert [r["speedup_da_over_noda"] for r in results] == pytest.approx([0.8, 0.4])
    table = bench._mixture_tables(results)
    assert "activation=swiglu" in table and "activation=relu2" in table


@pytest.mark.parametrize(
    "field,value",
    [("swiglu_alpha", 1.702), ("swiglu_beta", 1.0), ("swiglu_limit", 7.0)],
)
def test_mixture_keeps_swiglu_variants_separate(field, value):
    from benchmarks import bench_moe_da as bench

    rows = _mixture_rows()
    rows += [dict(r, **{field: value}, da_ms=r["da_ms"] * 2) for r in rows]
    results = bench._summarize_mixtures(
        rows, (bench._parse_mixture("ddist:1.1=1,ddist:4=1"),)
    )
    assert len(results) == 2
    assert [r["speedup_da_over_noda"] for r in results] == pytest.approx([0.8, 0.4])


@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 0.0, 10.0), (1.702, 1.0, 7.0)])
def test_model_swiglu_benchmark_matches_formula(monkeypatch, alpha, beta, limit):
    from benchmarks import bench_moe_da as bench

    _require_sm100()
    shape = bench.BenchmarkShape(
        num_tokens=8,
        num_experts=32,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=4,
        hidden_size=512,
        intermediate_size=512,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=8,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
    )
    hidden, w1, w2, ids, weights = bench._canonical_inputs(shape)
    hidden.zero_()
    # Include unclamped, clamped, and negative gates plus both linear clamp edges.
    gates = torch.tensor([0.5, 20.0, -20.0, 2.0], device="cuda")
    linear = torch.tensor([0.0, 20.0, 2.0, -20.0], device="cuda")
    hidden[:, :4] = gates
    hidden[:, 4:8] = linear
    w1.zero_()
    w2.zero_()
    for channel in range(4):
        # The native preparation ABI takes [up; gate], not HF's [gate; up].
        w1[:, channel, channel + 4] = 1
        w1[:, shape.intermediate_size + channel, channel] = 1
        w2[:, channel, channel] = 1
    ids.copy_(torch.arange(4, device="cuda", dtype=torch.int32).expand_as(ids))
    weights.fill_(0.25)
    monkeypatch.setattr(
        bench, "_canonical_inputs", lambda _: (hidden, w1, w2, ids, weights)
    )
    prepared = bench._prepare_precision("bf16", shape, backend="prims_ts")
    prepared.stage(ids, weights)
    with bench._temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        actual = prepared.invoke()
    gate = gates.clamp(max=limit)
    expected = torch.zeros_like(actual)
    expected[:, :4] = (
        gate * torch.sigmoid(alpha * gate) * (linear.clamp(-limit, limit) + beta)
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ddist:2",
        "ddist:2=0",
        "ddist:2=-1",
        "ddist:2=nan",
        "ddist:2=inf",
        "ddist:2=1,ddist:2.0=2",
    ],
)
def test_invalid_mixture_rejected(value):
    import argparse
    from benchmarks import bench_moe_da as bench

    with pytest.raises(argparse.ArgumentTypeError):
        bench._parse_mixture(value)


def test_mixture_normalizes_weights_and_distribution_aliases():
    from benchmarks import bench_moe_da as bench

    assert bench._parse_mixture("ddist_1.1=20,ddist:2.0=30,ddist:4=50").weights == (
        ("ddist:1.1", 0.2),
        ("ddist:2", 0.3),
        ("ddist:4", 0.5),
    )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nonfinite", "failed"])
def test_mixture_rejects_incomplete_or_invalid_evidence(failure):
    from benchmarks import bench_moe_da as bench

    rows = _mixture_rows()
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "nonfinite":
        rows[0]["da_ms"] = float("nan")
    else:
        rows[0]["status"] = "fail"
    with pytest.raises(ValueError):
        bench._summarize_mixtures(
            rows, (bench._parse_mixture("ddist:1.1=1,ddist:4=1"),)
        )


def test_missing_mixture_component_rejected_before_gpu_work(monkeypatch):
    from benchmarks import bench_moe_da as bench

    monkeypatch.setattr(sys, "argv", ["bench_moe_da.py", "--distributions", "uniform"])
    monkeypatch.setattr(
        bench, "_benchmark_precision", lambda *_: pytest.fail("GPU work")
    )
    with pytest.raises(SystemExit, match="mixture components missing"):
        bench.main()


def test_cli_prints_mixture_table_and_saves_component_evidence(
    monkeypatch, tmp_path, capsys
):
    from benchmarks import bench_moe_da as bench

    raw = tmp_path / "raw.csv"
    summary = tmp_path / "mixtures.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_moe_da.py",
            "--backend",
            "prims_ts",
            "--num-tokens",
            "96",
            "--distributions",
            "ddist:1.1,ddist:4",
            "--mixture",
            "ddist:1.1=50,ddist:4=50",
            "--out",
            str(raw),
            "--mixture-out",
            str(summary),
        ],
    )
    monkeypatch.setattr(bench, "_benchmark_precision", lambda *_: _mixture_rows())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert bench.main() == 0
    assert "| nvfp4 | 0.8000x |" in capsys.readouterr().out
    assert "distribution" in raw.read_text().splitlines()[0]
    assert json.loads(summary.read_text())[0]["speedup_da_over_noda"] == 0.8


def test_nondefault_bucket_is_active_during_both_captures(monkeypatch):
    from benchmarks import bench_moe_da as bench

    active = []
    captured = []

    @contextmanager
    def tuning(mode, **kwargs):
        active.append((mode, kwargs["tuning_buckets"]))
        try:
            yield
        finally:
            active.pop()

    def capture(_invoke):
        captured.append(active[-1] if active else None)
        return SimpleNamespace(reset=lambda: None)

    monkeypatch.setattr(bench, "autotune", tuning)
    monkeypatch.setattr(bench, "_capture", capture)
    monkeypatch.setattr(
        bench,
        "_prepare_precision",
        lambda *_: SimpleNamespace(stage=lambda *_: None, invoke=lambda: None),
    )
    monkeypatch.setattr(bench, "_realization", lambda *_: (None, None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *_: nullcontext())
    monkeypatch.setattr(bench, "da_moe_acquire_graph_leases", lambda *_: ())
    monkeypatch.setattr(
        bench, "_matching_diagnostic", lambda *_: {"policy": "da_single_body"}
    )
    monkeypatch.setattr(bench, "da_moe_release_resources", lambda: None)

    def stop():
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(bench, "_cold_l2_buffers", stop)
    with pytest.raises(RuntimeError, match="stop after capture"):
        bench._benchmark_precision(
            "nvfp4", SimpleNamespace(num_tokens=96), ("uniform",), None, True, 0, 2
        )
    assert captured == [(False, (96,)), (False, (96,))]
    assert active == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_routes_ignore_tuning_rng_consumption():
    from benchmarks import bench_moe_da as bench

    shape = bench.BenchmarkShape(
        num_tokens=8,
        num_experts=32,
        local_num_experts=16,
        local_expert_offset=0,
        top_k=4,
        hidden_size=128,
        intermediate_size=256,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=8,
    )
    state = torch.cuda.get_rng_state()
    first = bench._realization(bench.RoutingRealizationFactory(), shape, "ddist:2")
    assert torch.equal(torch.cuda.get_rng_state(), state)
    # Simulate random draws made by fresh tuning but absent on cache restore.
    torch.rand(4096, device="cuda")
    state = torch.cuda.get_rng_state()
    second = bench._realization(bench.RoutingRealizationFactory(), shape, "ddist:2")
    assert torch.equal(torch.cuda.get_rng_state(), state)
    for expected, actual in zip(first, second, strict=True):
        assert torch.equal(expected, actual)


def test_nvfp4_tma_scale_padding_uses_input_token_extent(monkeypatch):
    """Routed scale storage is smaller than the padded output capacity."""
    from benchmarks import bench_moe_da as bench
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.prims_ts import is_prims_ts_device_supported
    from flashinfer.prims_ts.moe.runner import PrimsTsNvfp4MoERunner
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_nvfp4_moe_tactic
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import RouteImpl

    if not torch.cuda.is_available() or not is_prims_ts_device_supported(
        torch.device("cuda")
    ):
        pytest.skip("PrimsTS device support required")
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
    shape = bench.BenchmarkShape(
        num_tokens=64,
        num_experts=128,
        local_num_experts=16,
        local_expert_offset=0,
        top_k=4,
        hidden_size=6144,
        intermediate_size=3072,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=64,
        activation="swiglu",
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    )
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner(warmup=0, repeat=2))
    prepared = bench._prepare_precision("nvfp4", shape, backend="prims_ts")
    # Exercise the profiling allocation that originally exposed the invalid TMA
    # extent. Keep the test bounded to the routed-SF TMA candidate.
    monkeypatch.setattr(
        PrimsTsNvfp4MoERunner, "get_valid_tactics", lambda *a, **kw: [(8, 18)]
    )
    with autotune(tuning_buckets=(64,)):
        prepared.invoke()
    # Only three local assignments: the second gather4 group is all padding.
    ids = torch.tensor([16, 17, 18, 19], device="cuda", dtype=torch.int32).repeat(64, 1)
    ids[:3, 0] = 0
    weights = torch.full((64, 4), 0.25, device="cuda", dtype=torch.float32)
    prepared.stage(ids, weights)
    # These two tactics share FC2 and differ in the FC1 gather implementation.
    tma_tactic, ldgsts_tactic = (8, 18), (8, 12)
    pair = map_trtllm_nvfp4_moe_tactic(
        tma_tactic, num_tokens=64, top_k=4, num_local_experts=16
    )
    assert pair.fc1.cfg.build().route_sfs_act == int(RouteImpl.TMA)
    monkeypatch.setattr(AutoTuner, "choose_one", lambda *a, **kw: (0, ldgsts_tactic))
    expected = prepared.invoke().clone()
    monkeypatch.setattr(AutoTuner, "choose_one", lambda *a, **kw: (0, tma_tactic))
    actual = prepared.invoke()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepared.invoke()
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0.02, rtol=0.02)
