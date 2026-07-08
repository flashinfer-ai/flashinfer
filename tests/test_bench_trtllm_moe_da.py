"""Contracts for the all-precision DA-versus-NoDA benchmark."""

import pytest

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
