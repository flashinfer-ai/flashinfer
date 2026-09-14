"""End-to-end command-line contract for the production TRTLLM DA benchmark."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from benchmarks.bench_moe_da import _time_graphs_counterbalanced
from flashinfer.utils import get_compute_capability


class _FakeGraph:
    def __init__(self, label: str, trace: list[str]) -> None:
        self.label = label
        self.trace = trace

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
        pass

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
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: _FakeEvent())

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


def _benchmark_command(cache: Path, output: Path, *, cache_only: bool) -> list[str]:
    """Build one bounded public CLI invocation for tuning or cache-only replay."""
    # Keep one compact real NVFP4 shape while exercising two distinct selector distributions.
    command = [
        sys.executable,
        "benchmarks/bench_trtllm_moe_da.py",
        "--precision",
        "nvfp4",
        "--distributions",
        "uniform,ddist:4",
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
    ]
    # Cache-only replay changes only the public lifecycle flag and reuses the same operation key.
    if cache_only:
        command.append("--skip-autotune")
    return command


def _assert_result_file(path: Path) -> None:
    """Validate finite, numerical, topology-aware benchmark result records."""
    rows = json.loads(path.read_text())
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


def test_cli_json_cache_restores_in_a_fresh_process(tmp_path: Path) -> None:
    """The public JSON tuning cache must restore DA replay without profiling."""
    _require_sm100()
    cache = tmp_path / "tuning-cache.json"
    tuned = tmp_path / "tuned.json"
    restored = tmp_path / "restored.json"
    # Start from the user environment, then pin only public tuning/cache controls for this process.
    environment = os.environ.copy()
    environment.update(
        {
            "FLASHINFER_DA_BASELINE_GUARD": "0",
            "FLASHINFER_WORKSPACE_BASE": str(Path.cwd() / ".cache"),
            "MAX_JOBS": "8",
            "PYTHONPATH": str(Path.cwd()),
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
        _benchmark_command(cache, tuned, cache_only=False),
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )
    cache_payload = json.loads(cache.read_text())
    da_records = cache_payload["_records"]["moe_da"]
    assert len(da_records) == 1
    operation_key, record = next(iter(da_records.items()))
    assert json.loads(operation_key)["backend"] == "trtllm"
    assert record["backend"] == "trtllm"
    _assert_result_file(tuned)

    # A second process proves cache-only replay can restore the same public result contract.
    subprocess.run(
        _benchmark_command(cache, restored, cache_only=True),
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )
    _assert_result_file(restored)
