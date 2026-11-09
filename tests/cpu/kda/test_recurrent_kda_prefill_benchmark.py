# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Host timing accounts for synchronization without charging warmup."""

from pathlib import Path
import json
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch


_BENCHMARK = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "bench_recurrent_kda_prefill.py"
    )
)
_measure_host = _BENCHMARK["_measure_host"]


@pytest.mark.cpu_only
def test_host_timing_separates_return_from_completion(monkeypatch):
    clock = [0]
    calls = []

    def run():
        calls.append("run")
        clock[0] += 2_000_000

    def synchronize():
        calls.append("sync")
        clock[0] += 3_000_000

    monkeypatch.setattr(_BENCHMARK["time"], "perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(_BENCHMARK["torch"].cuda, "synchronize", synchronize)
    result = _measure_host(run, dry_run_iters=2, repeat_iters=3)

    assert calls == ["run", "run", "sync"] + ["run", "sync"] * 3
    assert result["host_return_samples_ms"] == [2.0] * 3
    assert result["completed_call_samples_ms"] == [5.0] * 3
    assert result["host_return_median_ms"] == 2.0
    assert result["completed_call_median_ms"] == 5.0
    assert result["stateful_calls"] == 5
    assert result["cold_l2"] is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("inference_mode", [False, True])
@pytest.mark.parametrize("host_timing", [False, True])
def test_main_serializes_timings_in_requested_inference_mode(
    monkeypatch, tmp_path, inference_mode, host_timing
):
    main = _BENCHMARK["main"]
    namespace = main.__globals__
    clock = [0]
    cursor = [0]
    inputs = []
    calls = []

    def candidate_run():
        assert torch.is_inference_mode_enabled() is inference_mode
        assert inputs[0].is_inference() is inference_mode
        cursor[0] += 1
        calls.append("run")
        clock[0] += 2_000_000

    def reset_state_pools():
        cursor[0] = 0

    def make_case(case, **kwargs):
        assert torch.is_inference_mode_enabled() is inference_mode
        inputs.append(torch.empty(1))
        return SimpleNamespace(
            candidate_run=candidate_run,
            peer_raw_run=None,
            reset_state_pools=reset_state_pools,
            state_cursors={"pr": cursor},
            metadata={"name": case.name, "resolved_backend": "mock", "variant": "cpu"},
        )

    def measure(run, *, dry_run_iters, repeat_iters):
        for _ in range(
            namespace["_CUPTI_ESTIMATE_CALLS_PER_BLOCK"] + dry_run_iters + repeat_iters
        ):
            run()
        return 1.0, [1.0] * repeat_iters

    def synchronize():
        clock[0] += 3_000_000

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(namespace["time"], "perf_counter_ns", lambda: clock[0])
    monkeypatch.setitem(namespace, "get_compute_capability", lambda _: (10, 0))
    monkeypatch.setitem(namespace, "_require_cupti", lambda: None)
    monkeypatch.setitem(
        namespace,
        "_hardware_metadata",
        lambda _: {
            "device_name": "mock",
            "compute_capability": [10, 0],
            "cuda_arch": "sm100a",
        },
    )
    monkeypatch.setitem(namespace, "H12_CASES", namespace["H12_CASES"][:1])
    monkeypatch.setitem(namespace, "_make_case", make_case)
    monkeypatch.setitem(namespace, "_measure", measure)
    output = tmp_path / "timings.json"
    argv = [
        "bench_recurrent_kda_prefill.py",
        "--case-set",
        "h12",
        "--dry-run-iters",
        "2",
        "--repeat-iters",
        "3",
        "--json",
        str(output),
    ]
    if host_timing:
        argv.append("--host-timing")
    if inference_mode:
        argv.append("--inference-mode")
    monkeypatch.setattr(sys, "argv", argv)

    # Exercise both enabling and disabling the caller's inference context.
    with torch.inference_mode(not inference_mode):
        main()
        assert torch.is_inference_mode_enabled() is not inference_mode

    results = json.loads(output.read_text())
    assert len(results) == 1
    result = results[0]
    assert result["inference_mode"] is inference_mode
    assert result["samples_ms"] == [1.0] * 3
    assert result["median_ms"] == 1.0
    assert len(calls) == 1 + 6 + 2 + 3 + (5 if host_timing else 0)
    if host_timing:
        assert result["host_timing"] == {
            "host_return_samples_ms": [2.0] * 3,
            "completed_call_samples_ms": [5.0] * 3,
            "host_return_median_ms": 2.0,
            "completed_call_median_ms": 5.0,
            "stateful_calls": 5,
            "cold_l2": False,
            "timing_scope": "eager_public_recurrent_kda_with_inplace_state_update",
        }
    else:
        assert "host_timing" not in result
