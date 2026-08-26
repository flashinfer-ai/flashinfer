"""Unit coverage for the Wan hybrid qualification report contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "bench_wan_hybrid_attention.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "flashinfer_wan_hybrid_benchmark_test_target", _BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_measure_order_reports_signed_and_absolute_deltas(monkeypatch) -> None:
    samples = iter(([2.0], [4.0], [6.0], [2.0]))
    monkeypatch.setattr(benchmark, "_measure_leg", lambda _fn: next(samples))
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=7),
    )

    result = benchmark._measure_order(
        ("C", "F", "F", "C"), lambda: None, lambda: None
    )

    assert result["order"] == "C/F/F/C"
    assert result["candidate_median_ms"] == 2.0
    assert result["production_fa4_median_ms"] == 5.0
    assert result["candidate_minus_production_fa4_ms"] == -3.0
    assert result["production_fa4_minus_candidate_ms"] == 3.0
    assert result["absolute_delta_ms"] == 3.0
    assert result["speedup"] == 2.5
    assert result["passed_speedup_ge_1"] is True


def test_baseline_quality_is_required_for_promotion() -> None:
    orders = [{"passed_speedup_ge_1": True}, {"passed_speedup_ge_1": True}]
    assert benchmark._qualification_passed(
        {"passed": True}, {"passed": True}, True, True, orders
    )
    assert not benchmark._qualification_passed(
        {"passed": True}, {"passed": False}, True, True, orders
    )


def test_callable_provenance_records_source_identity(tmp_path, monkeypatch) -> None:
    source = tmp_path / "provider.py"
    source.write_text("def provider():\n    return None\n", encoding="utf-8")
    module = types.ModuleType("wan_benchmark_test_provider")
    module.__file__ = str(source)

    def provider() -> None:
        return None

    provider.__module__ = module.__name__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        benchmark,
        "distribution_version",
        lambda name: "1.2.3" if name == "provider-dist" else pytest.fail(name),
    )

    result = benchmark._callable_provenance("provider-dist", provider)
    assert result == {
        "distribution": "provider-dist",
        "distribution_version": "1.2.3",
        "callable_module": module.__name__,
        "callable_qualified_name": provider.__qualname__,
        "module_source_path": str(source.resolve()),
        "module_source_sha256": benchmark.hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }


def test_callable_provenance_fails_without_source(monkeypatch) -> None:
    module = types.ModuleType("wan_benchmark_test_sourceless_provider")

    def provider() -> None:
        return None

    provider.__module__ = module.__name__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="no source file"):
        benchmark._callable_provenance("provider-dist", provider)
