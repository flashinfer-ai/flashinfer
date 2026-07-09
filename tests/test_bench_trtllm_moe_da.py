"""Contracts for the all-precision DA-versus-NoDA benchmark."""

import pytest
import torch

from benchmarks import bench_trtllm_moe_da as benchmark


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
