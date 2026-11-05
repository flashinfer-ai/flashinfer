# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Host timing accounts for synchronization without charging warmup."""

from pathlib import Path
import runpy

import pytest


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
