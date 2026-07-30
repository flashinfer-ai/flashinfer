"""Contracts for the all-precision DA-versus-NoDA benchmark."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks import bench_trtllm_moe_da as benchmark


def test_autotune_timing_compares_fresh_noda_with_one_sweep_da(monkeypatch):
    """Default reporting times two independent caches and retains the DA one."""

    class FakeTuner:
        def __init__(self):
            self.profiling_cache = {}
            self.clear_count = 0
            self.reset_count = 0

        def clear_cache(self):
            self.clear_count += 1
            self.profiling_cache.clear()

        def reset_statistics(self):
            self.reset_count += 1

    tuner = FakeTuner()
    phase_modes = []
    phase_entry_cache_sizes = []
    call_modes = []
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "test-initial-state")

    @contextmanager
    def fake_autotune(*_args, **_kwargs):
        mode = os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"]
        phase_modes.append(mode)
        phase_entry_cache_sizes.append(len(tuner.profiling_cache))
        yield
        if mode == "0":
            tuner.profiling_cache["noda-profile"] = 1.0
        else:
            tuner.profiling_cache.update(
                {"da-default-profile": 1.0, "da-value-profile": 2.0}
            )
            benchmark.da_state.PER_BODY_TACTICS["runtime-key"] = [(64, 7)]
            benchmark.da_state.BASELINE_GUARD_DECISIONS["runtime-key"] = {
                "final_policy": "da_singleton"
            }

    monkeypatch.setattr(benchmark.AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(benchmark, "autotune", fake_autotune)
    monkeypatch.setattr(benchmark, "_make_routing_input", lambda *_args: object())
    monkeypatch.setattr(
        benchmark, "get_hybrid_num_tokens_buckets", lambda *_args: (64,)
    )
    monkeypatch.setattr(benchmark, "_make_da_context", lambda *_args: object())
    monkeypatch.setattr(benchmark.da_core, "upload_bucket", lambda *_args: 64)
    monkeypatch.setattr(benchmark.da_state, "cache_key", lambda *_args: "runtime-key")
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    times = iter((10.0, 12.0, 20.0, 27.0))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    for name in (
        "PER_TILE_TACTICS",
        "PER_BODY_TACTICS",
        "STATIC_FALLBACK_TACTICS",
        "BUNDLE_EAGER_TACTICS",
        "BASELINE_GUARD_DECISIONS",
    ):
        monkeypatch.setattr(benchmark.da_state, name, {})

    case = SimpleNamespace(
        name="nvfp4",
        call=lambda _routing: call_modes.append(
            os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"]
        ),
    )
    cfg = SimpleNamespace(num_tokens=64, tune_max_num_tokens=64)

    result = benchmark._run_real_autotune(case, cfg, torch.device("cpu"), "routed")

    assert call_modes == ["0", "0", "1"]  # untimed setup, NoDA, then DA
    assert phase_modes == ["0", "1"]
    assert phase_entry_cache_sizes == [0, 0]
    assert tuner.clear_count == tuner.reset_count == 2
    assert result["autotune_measurement_contract"] == (
        "independent_fresh_noda_then_one_sweep_da_v1"
    )
    assert result["noda_autotune_measured"] is True
    assert result["noda_profiles"] == result["static_profiles"] == 1
    assert result["da_profiles"] == result["total_profiles"] == 2
    assert result["noda_tune_seconds"] == result["static_tune_seconds"] == 2.0
    assert result["da_tune_seconds"] == 7.0
    assert result["tune_seconds"] == 9.0
    assert os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] == "1"


def test_benchmark_mode_table_and_all_selector_are_exact():
    expected = [
        "bf16",
        "fp8_per_tensor",
        "fp8_block",
        "mxfp8",
        "nvfp4",
        "mxfp4_mxfp8",
        "mxfp4_bf16",
        "mxint4",
    ]
    assert list(benchmark.BENCHMARK_MODES) == expected
    assert benchmark._parse_precision_modes("all") == expected
    assert benchmark._parse_precision_modes("bf16,mxint4") == ["bf16", "mxint4"]

    with pytest.raises(ValueError, match="unsupported precision"):
        benchmark._parse_precision_modes("bf16,unknown")
    with pytest.raises(ValueError, match="at least one"):
        benchmark._parse_precision_modes("")


def test_local_expert_topology_is_explicit_and_defaults_to_no_ep():
    parser = benchmark.build_parser()

    default_args = parser.parse_args(["--num-experts", "64"])
    assert not hasattr(default_args, "ep")
    assert (
        benchmark._resolve_local_num_experts(
            default_args.num_experts, default_args.local_num_experts
        )
        == 64
    )

    ep4_args = parser.parse_args(["--num-experts", "64", "--local-num-experts", "16"])
    assert (
        benchmark._resolve_local_num_experts(
            ep4_args.num_experts, ep4_args.local_num_experts
        )
        == 16
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["--ep", "4"])


def test_routing_input_mode_defaults_to_routed_and_accepts_logits():
    """The benchmark defaults to public routed APIs and keeps logits opt-in."""
    parser = benchmark.build_parser()

    assert parser.parse_args([]).routing_input_mode == "routed"
    assert (
        parser.parse_args(["--routing-input-mode", "logits"]).routing_input_mode
        == "logits"
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["--routing-input-mode", "packed"])


def test_routed_mode_rejects_precisions_without_a_public_routed_api():
    """Routed mode covers every existing routed wrapper, but invents none."""
    supported = [
        "bf16",
        "fp8_block",
        "mxfp8",
        "nvfp4",
        "mxfp4_mxfp8",
        "mxfp4_bf16",
        "mxint4",
    ]
    assert benchmark._validate_routing_input_mode("routed", supported) == supported
    all_precisions = list(benchmark.BENCHMARK_MODES)
    assert (
        benchmark._validate_routing_input_mode("logits", all_precisions)
        == all_precisions
    )

    with pytest.raises(ValueError, match="public routed MoE API"):
        benchmark._validate_routing_input_mode("routed", ["nvfp4", "fp8_per_tensor"])


def test_routed_inputs_follow_each_public_api_contract_and_report_routed():
    """FP4 gets an unpacked pair; other routed wrappers get packed int32."""
    cfg = benchmark.BenchConfig(
        num_tokens=4,
        num_experts=8,
        local_num_experts=8,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        tune_max_num_tokens=4,
    )

    topk_ids, topk_weights = benchmark._make_routing_input(
        "routed", "nvfp4", "uniform", cfg, torch.device("cpu")
    )

    assert topk_ids.shape == topk_weights.shape == (4, 2)
    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.bfloat16
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    for precision in ("bf16", "fp8_block", "mxfp8", "mxint4"):
        packed = benchmark._make_routing_input(
            "routed", precision, "uniform", cfg, torch.device("cpu")
        )
        assert isinstance(packed, torch.Tensor)
        assert packed.shape == (4, 2)
        assert packed.dtype == torch.int32
        packed_ids = packed >> 16
        assert torch.all((packed_ids >= 0) & (packed_ids < cfg.num_experts))
        assert torch.equal(
            (packed & 0xFFFF).to(torch.int16), topk_weights.view(torch.int16)
        )

    assert benchmark._reported_internal_routing_mode("routed") == "routed"
    assert benchmark._reported_internal_routing_mode("logits") == "packed"


def test_guarded_noda_requires_no_da_capture_dispatch():
    """A rejected DA plan must execute without DA graph mutation."""
    row = {
        "execution_mode": "graph",
        "da_policy": "noda_baseline_guard",
        "noda_capture_dispatch_count": 0,
        "da_capture_dispatch_count": 0,
        "noda_finite": 1.0,
        "da_finite": 1.0,
        "noda_match_ratio": 1.0,
        "da_match_ratio": 1.0,
        "match_ratio_threshold": 1.0,
    }
    assert benchmark._row_status(row) == "PASS"
    row["da_capture_dispatch_count"] = 1
    assert benchmark._row_status(row) == "FAIL_NODA_USED_DA_CAPTURE"


def test_guard_reporting_separates_candidate_and_final_plans():
    """CSV fields retain the switch that a baseline guard collapsed."""
    fields = benchmark._guard_row_fields(
        {
            "policy": "noda_baseline_guard",
            "candidate_policy": "da_switch",
            "candidate_tactics": [(32, 121), (8, 65)],
            "final_policy": "noda_baseline_guard",
            "final_tactics": [(32, 43)],
            "baseline_tactic": (32, 43),
            "singleton_tactic": (32, 43),
            "singleton_source": "noda_baseline",
            "control_overhead_source": "pre_recorded_calibration",
            "overhead_ms": 0.012,
            "admission_applied": True,
            "limitation": None,
        }
    )

    assert fields["da_policy"] == "noda_baseline_guard"
    assert fields["da_candidate_policy"] == "da_switch"
    assert fields["da_candidate_tactics"] == "[(32, 121), (8, 65)]"
    assert fields["da_final_policy"] == "noda_baseline_guard"
    assert fields["da_final_tactics"] == "[(32, 43)]"
    assert fields["da_overhead_ms"] == pytest.approx(0.012)
    assert fields["da_guard_admission_applied"] is True
    assert fields["da_guard_limitation"] is None


def test_noda_plan_requires_no_da_capture_dispatch():
    """The noda label is reserved for execution without the DA mechanism."""
    row = {
        "execution_mode": "graph",
        "da_policy": "noda",
        "noda_capture_dispatch_count": 0,
        "da_capture_dispatch_count": 0,
        "noda_finite": 1.0,
        "da_finite": 1.0,
        "noda_match_ratio": 1.0,
        "da_match_ratio": 1.0,
        "match_ratio_threshold": 1.0,
    }
    assert benchmark._row_status(row) == "PASS"
    row["da_capture_dispatch_count"] = 1
    assert benchmark._row_status(row) == "FAIL_NODA_USED_DA_CAPTURE"
