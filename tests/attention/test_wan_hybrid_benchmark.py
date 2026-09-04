"""Unit coverage for the Wan hybrid qualification report contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from flashinfer.testing import utils as testing_utils


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_wan_hybrid_attention.py"
)
_QUANTIZATION_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "bench_wan_hybrid_quantization.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "flashinfer_wan_hybrid_benchmark_test_target", _BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_measure_order_reports_signed_and_absolute_deltas(monkeypatch) -> None:
    def measured(sample: float, kernel_names: list[str]) -> tuple[list[float], dict]:
        return [sample], {
            "sample_count": 1,
            "iterations": [
                {
                    "sample_index": 0,
                    "kernel_activity_count": len(kernel_names),
                    "kernel_activities": [
                        {"name": name, "duration_ms": sample}
                        for name in kernel_names
                    ],
                }
            ],
        }

    samples = iter(
        (
            measured(
                2.0,
                [
                    "void kernel_wan_hybrid_quantize_value<128>()",
                    "void kernel_wan_hybrid_attention<128>()",
                ],
            ),
            measured(4.0, ["production_fa4_kernel"]),
            measured(6.0, ["production_fa4_kernel"]),
            measured(
                2.0,
                [
                    "void kernel_wan_hybrid_quantize_value<128>()",
                    "void kernel_wan_hybrid_attention<128>()",
                ],
            ),
        )
    )
    monkeypatch.setattr(benchmark, "_measure_leg", lambda _fn: next(samples))
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=7),
    )

    result = benchmark._measure_order(("C", "F", "F", "C"), lambda: None, lambda: None)

    assert result["order"] == "C/F/F/C"
    assert result["candidate_median_ms"] == 2.0
    assert result["production_fa4_median_ms"] == 5.0
    assert result["candidate_minus_production_fa4_ms"] == -3.0
    assert result["production_fa4_minus_candidate_ms"] == 3.0
    assert result["absolute_delta_ms"] == 3.0
    assert result["speedup"] == 2.5
    assert result["passed_speedup_ge_1"] is True
    assert result["legs"][0]["cupti_activity_evidence"]["sample_count"] == 1
    assert [
        activity["name"]
        for activity in result["legs"][0]["cupti_activity_evidence"][
            "iterations"
        ][0]["kernel_activities"]
    ] == [
        "void kernel_wan_hybrid_quantize_value<128>()",
        "void kernel_wan_hybrid_attention<128>()",
    ]
    assert result["legs"][1]["cupti_activity_evidence"]["iterations"][0][
        "kernel_activity_count"
    ] == 1


@pytest.mark.parametrize(
    ("label", "kernel_names", "message"),
    [
        (
            "C",
            [
                "kernel_wan_hybrid_attention",
                "kernel_wan_hybrid_quantize_value",
            ],
            "quantization and attention kernels in order",
        ),
        (
            "F",
            ["production_fa4_kernel", "unexpected_second_kernel"],
            "exactly one kernel activity",
        ),
    ],
)
def test_validate_leg_activity_evidence_rejects_wrong_kernel_sequence(
    label, kernel_names, message
) -> None:
    activity_evidence = {
        "iterations": [
            {
                "kernel_activity_count": len(kernel_names),
                "kernel_activities": [
                    {"name": name, "duration_ms": 0.1}
                    for name in kernel_names
                ],
            }
        ]
    }

    with pytest.raises(RuntimeError, match=message):
        benchmark._validate_leg_activity_evidence(label, activity_evidence)


def test_measure_leg_requests_cupti_activity_evidence(monkeypatch) -> None:
    captured = {}

    def fake_bench_gpu_time(**kwargs):
        captured.update(kwargs)
        kwargs["cupti_activity_evidence"].append(
            {
                "sample_index": 0,
                "kernel_activity_count": 2,
                "kernel_activities": [
                    {"name": "quantize", "duration_ms": 0.1},
                    {"name": "attention", "duration_ms": 0.3},
                ],
                "kernel_sum_ms": 0.4,
                "gpu_span_ms": 0.401,
                "inter_kernel_gap_ms": 0.001,
            }
        )
        return [0.401]

    monkeypatch.setattr(benchmark, "bench_gpu_time", fake_bench_gpu_time)

    samples, evidence = benchmark._measure_leg(lambda: None)

    assert samples == [0.401]
    assert captured["enable_cupti"] is True
    assert captured["dry_run_iters"] == 2
    assert captured["repeat_iters"] == 5
    assert captured["cold_l2_cache"] is True
    assert evidence["sample_count"] == 1
    assert evidence["iterations"][0]["kernel_sum_ms"] == 0.4
    assert evidence["iterations"][0]["inter_kernel_gap_ms"] == 0.001


def test_cupti_activity_summary_preserves_kernel_durations_and_gap() -> None:
    kernel_kind = object()
    activities = [
        ("attention", 1_100_250, 1_400_250, 2, 0, 0, 0, kernel_kind),
        ("quantize", 1_000_000, 1_100_000, 1, 0, 0, 0, kernel_kind),
    ]

    result = testing_utils._summarize_cupti_iteration_activities(
        activities, kernel_kind
    )

    assert result["activity_count"] == 2
    assert result["kernel_activity_count"] == 2
    assert [item["name"] for item in result["kernel_activities"]] == [
        "quantize",
        "attention",
    ]
    assert [item["duration_ms"] for item in result["kernel_activities"]] == [
        pytest.approx(0.1),
        pytest.approx(0.3),
    ]
    assert result["kernel_sum_ms"] == pytest.approx(0.4)
    assert result["active_kernel_union_ms"] == pytest.approx(0.4)
    assert result["kernel_span_ms"] == pytest.approx(0.40025)
    assert result["inter_kernel_gap_ms"] == pytest.approx(0.00025)
    assert result["gpu_span_ms"] == pytest.approx(0.40025)


def test_baseline_quality_is_required_for_promotion() -> None:
    orders = [{"passed_speedup_ge_1": True}, {"passed_speedup_ge_1": True}]
    assert benchmark._qualification_passed(
        {"passed": True}, {"passed": True}, True, True, orders
    )
    assert not benchmark._qualification_passed(
        {"passed": True}, {"passed": False}, True, True, orders
    )
    assert not benchmark._qualification_passed(
        {"passed": True}, {"passed": True}, True, False, orders
    )


def test_benchmark_records_peak_temporary_allocations() -> None:
    source = _QUANTIZATION_BENCHMARK_PATH.read_text(encoding="utf-8")
    assert "torch.cuda.reset_peak_memory_stats(device)" in source
    assert "torch.cuda.max_memory_allocated(device)" in source
    assert '"peak_temporary_allocation_bytes"' in source
    assert "allocated_before={allocated_before}" in source
    assert "allocated_after={allocated_after}" in source
    assert "peak_temporary_allocation_bytes={peak_temporary_allocation_bytes}" in source


def test_load_production_fa4_uses_sglang_runtime_package(monkeypatch) -> None:
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__path__ = []
    cute = types.ModuleType("flash_attn.cute")
    cute.__path__ = []
    interface = types.ModuleType("flash_attn.cute.interface")

    def provider() -> None:
        return None

    interface._flash_attn_fwd = provider
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.delitem(sys.modules, "sglang", raising=False)

    assert benchmark._load_production_fa4() is provider
    assert "sglang" not in sys.modules


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


def test_production_fa4_provenance_records_sglang_route(tmp_path, monkeypatch) -> None:
    source = tmp_path / "interface.py"
    source.write_text("def provider():\n    return None\n", encoding="utf-8")
    module = types.ModuleType("flash_attn.cute.interface")
    module.__file__ = str(source)

    def provider() -> None:
        return None

    provider.__module__ = module.__name__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    versions = {"flash-attn-4": "4.0.0b15", "sglang": "0.5.14"}
    monkeypatch.setattr(benchmark, "distribution_version", versions.__getitem__)

    result = benchmark._production_fa4_provenance(provider)
    assert result["distribution"] == "flash-attn-4"
    assert result["distribution_version"] == "4.0.0b15"
    assert result["sglang_distribution_version"] == "0.5.14"
    assert result["sglang_backend"] == "FA4"
    assert result["callable_module"] == module.__name__
