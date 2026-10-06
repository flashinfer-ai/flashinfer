#!/usr/bin/env python3
"""Benchmark b12x dense_gemm with numerical reference and graph-replay checks.

The default profile uses the Nemotron 3 Super shared-expert down projection for
FP4, the per-rank dense-linear shape from the cached DeepSeek V4 Flash DSpark
checkpoint at TP=2 for MXFP8, and Qwen linear shapes for regular block FP8. The
Qwen3.8-27B profile uses its hidden=5120 and intermediate=17408 FFN checkpoint
projections for every quantization track. The super3-mamba profile isolates
Super3.5's input (K=4096, N=18560) and output (K=8192, N=4096) projections;
use --dtype fp4-a16 for their serving precision. End-to-end MXFP8 includes activation
quantization; weight quantization remains setup work. Correctness uses b12x
dequantization references and FP32 matrix multiplication outside timing.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import pathlib
import statistics
import sys
from typing import Callable, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x.gemm import block_fp8_linear
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.reference.helpers import dequantize_grouped_nvfp4
from b12x.gemm._shared.wo_mxfp8 import (
    quantize_mxfp8_rows_torch, dequantize_mxfp8_rows_torch,
)
from b12x.gemm._shared.block_fp8 import quantize_block_fp8_linear_input_mxfp8
from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from benchmarks.experimental.b12x.common import make_l2_flush_fn, resolve_l2_flush_bytes

# Nemotron 3 Super shared expert down projection from the released NVFP4
# checkpoint:
#   down: [M, 5376] x [5376, 4096]
NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE = 5376
NEMOTRON_HIDDEN_SIZE = 4096

FP4_GEMM_SPECS = [
    # (name, K, N, note)
    (
        "Nemotron shared expert down",
        NEMOTRON_SHARED_EXPERT_INTERMEDIATE_SIZE,
        NEMOTRON_HIDDEN_SIZE,
        "NVIDIA Nemotron 3 Super shared_experts.down_proj",
    ),
]

# DeepSeek-V4-Flash-DSpark, TP=2. The checkpoint stores wq_b as
# [32768, 1024]; vLLM's ColumnParallelLinear shards its output dimension to
# [16384, 1024] per rank. This is the representative generic FP8 dense linear.
# Routed experts are intentionally excluded because they execute through fused
# MoE, while WO is primarily covered by its specialized fused projection path.
FP8_GEMM_SPECS = [
    # (name, K, N, note)
    (
        "DSV4-DSpark TP2 q_b",
        1024,
        16384,
        "column-parallel half of checkpoint wq_b[32768,1024]",
    ),
]

QWEN38_27B_GEMM_SPECS = [
    # (name, K, N, note)
    (
        "Qwen3.8-27B gate/up",
        5120,
        17408,
        "checkpoint gate_proj/up_proj (hidden=5120, intermediate=17408)",
    ),
    (
        "Qwen3.8-27B down",
        17408,
        5120,
        "checkpoint down_proj (intermediate=17408, hidden=5120)",
    ),
]

FP8_BLOCK_GEMM_SPECS = QWEN38_27B_GEMM_SPECS

SUPER3_MAMBA_GEMM_SPECS = [
    # Each shape occurs in all 40 Mamba blocks of NVIDIA_Super3_5_VL_IQ2XXS-Packed.
    ("Super3.5 Mamba in_proj", 4096, 18560,
     "NVFP4 mixer.in_proj: BF16 activations, hidden=4096"),
    ("Super3.5 Mamba out_proj", 8192, 4096,
     "NVFP4 mixer.out_proj: BF16 activations, expanded hidden=8192"),
]

DEFAULT_PROFILE = "default"
QWEN38_27B_PROFILE = "qwen3.8-27b"
SUPER3_MAMBA_PROFILE = "super3-mamba"
GEMM_PROFILES = {
    DEFAULT_PROFILE: "mixed production shapes",
    QWEN38_27B_PROFILE: "Qwen3.8-27B FFN checkpoint projections",
    SUPER3_MAMBA_PROFILE: "Super3.5 NVFP4 Mamba input/output projections",
}


def gemm_specs_for_mode(mode: str, profile: str = DEFAULT_PROFILE):
    if profile == QWEN38_27B_PROFILE:
        return QWEN38_27B_GEMM_SPECS
    if profile == SUPER3_MAMBA_PROFILE:
        return SUPER3_MAMBA_GEMM_SPECS
    if profile != DEFAULT_PROFILE:
        raise ValueError(f"unknown dense GEMM profile: {profile}")
    if mode in ("fp4-a16", "fp8-a16"):
        return [*FP4_GEMM_SPECS, *FP8_GEMM_SPECS, *QWEN38_27B_GEMM_SPECS]
    if mode == "fp4":
        return FP4_GEMM_SPECS
    if mode == "fp8-block":
        return FP8_BLOCK_GEMM_SPECS
    return FP8_GEMM_SPECS

FP4_BATCH_SIZES = [2, 4, 8]
FP8_BATCH_SIZES = [1, 2, 4, 8, 4096]
FP8_BLOCK_BATCH_SIZES = [1, 2, 4, 8, 16, 128, 512, 4096]
COSINE_THRESHOLD = 0.9999
BLOCK_FP8_COSINE_THRESHOLD = 0.9999


class BenchmarkAbort(RuntimeError):
    """Fatal benchmark failure that should stop the run without a summary."""


class CorrectnessError(BenchmarkAbort):
    """Raised when replay outputs fail the correctness gate."""


def bench_events(fn, *, warmup, iters, l2_flush=None):
    from b12x.testing.benchmark import samples_ms
    return samples_ms(fn, warmup=warmup, iters=iters, l2_flush=l2_flush)


def fmt_us(times_ms: List[float]) -> str:
    med = statistics.median(times_ms) * 1000
    mn = min(times_ms) * 1000
    return f"{med:7.1f} us (min {mn:.1f})"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return F.cosine_similarity(a_f, b_f, dim=0).item()


def check_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
    cosine_threshold: float,
) -> None:
    cand_finite = bool(torch.isfinite(candidate).all().item())
    ref_finite = bool(torch.isfinite(reference).all().item())
    if not cand_finite or not ref_finite:
        raise CorrectnessError(
            f"non-finite output detected during correctness check vs {label}: "
            f"candidate_finite={cand_finite}, reference_finite={ref_finite}"
        )
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    rmse = diff.square().mean().sqrt().item()
    cos = cosine_similarity(candidate, reference)
    print(f"    check vs {label}: max_abs={max_abs:.8f} rmse={rmse:.8f} cos={cos:.10f}")
    if not math.isfinite(cos):
        raise CorrectnessError(
            f"cosine similarity vs {label} is non-finite: "
            f"max_abs={max_abs:.8f}, rmse={rmse:.8f}, cos={cos}"
        )
    if cos < cosine_threshold:
        raise CorrectnessError(
            f"cosine similarity vs {label} fell below threshold "
            f"{cosine_threshold:.6f}: got {cos:.10f}"
        )


def capture_graph_replay(fn: Callable[[], None]) -> Callable[[], None]:
    # Warm eager launch state before capture so compile/cache work does not leak
    # into the replay measurement.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
        g.replay()

    replay._b12x_benchmark_call = fn
    return replay


def make_quantized_operand(M: int, K: int):
    source = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16) / 4
    row_counts = torch.full((1,), M, dtype=torch.int32, device="cuda")
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed, scales, global_scale


def quantize_mxfp8_source(source: torch.Tensor):
    quantized = quantize_mxfp8_rows_torch(source)
    return quantized.values, quantized.scale_rows, quantized.scale_mma




def make_mxfp8_operand(M: int, K: int):
    source = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    return (*quantize_mxfp8_source(source), source)




def bench_one_fp4(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    """Benchmark one (M,N,K) problem with stream-gated event timing."""
    torch.manual_seed(42)
    a_packed, a_sf, a_gs = make_quantized_operand(M, K)
    b_packed, b_sf, b_gs = make_quantized_operand(N, K)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)

    results = {}

    # b12x FP4.
    try:
        b12x_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)

        def b12x_launch():
            dense_gemm(
                (a_packed, a_sf),
                (b_packed, b_sf),
                alpha=alpha,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
                out=b12x_out,
            )

        b12x_replay = capture_graph_replay(b12x_launch)
        results["b12x_replay"] = b12x_replay
        results["b12x_out"] = b12x_out
    except Exception as exc:
        raise BenchmarkAbort(f"b12x execution failed: {exc}") from exc

    if check:
        results["b12x_replay"]()
        a_ref = dequantize_grouped_nvfp4(a_packed.permute(2, 0, 1), a_sf, K, a_gs)[0]
        b_ref = dequantize_grouped_nvfp4(b_packed.permute(2, 0, 1), b_sf, K, b_gs)[0]
        reference = a_ref @ b_ref.T
        torch.cuda.synchronize()
        check_outputs(results["b12x_out"][:, :, 0], reference,
                      label="b12x dequantized reference", cosine_threshold=COSINE_THRESHOLD)
    results["b12x"] = bench_events(
        b12x_replay, warmup=warmup, iters=iters, l2_flush=l2_flush,
    )
    return results


def bench_one_fp8(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
    include_input_quant: bool = False,
):
    """Benchmark one MXFP8 (M,N,K) problem with stream-gated event timing.

    When ``include_input_quant`` is true, each replay starts from the BF16 A
    operand. The b12x launch uses caller-owned MXFP8 storage and the production
    ``quantize_block_fp8_linear_input_mxfp8(..., out=...)`` path, which launches
    ``_quantize_dense_tk_to_tk_kernel`` before ``dense_gemm``. B remains a
    prequantized model weight.
    """
    with ExitStack() as scopes:
        torch.manual_seed(42)
        a_quantized, a_scale, a_scale_mma, a_source = make_mxfp8_operand(M, K)
        b_quantized, b_scale, b_scale_mma, b_source = make_mxfp8_operand(N, K)

        results = {}

        # b12x MXFP8. Keep quantizer output allocation outside capture so the e2e
        # replay matches an allocation-stable serving path.
        try:
            b12x_out = torch.empty((M, N, 1), device="cuda", dtype=torch.bfloat16)
            a_quantized_b12x = None
            if include_input_quant:
                a_quantized_b12x = empty_mxfp8_rows_for_dense_gemm(
                    M,
                    K,
                    device=a_source.device,
                )
                quant_plan = block_fp8_linear.plan(block_fp8_linear.Caps(
                    device=a_source.device, max_tokens=M, in_features=K,
                    out_features=N, output_dtype=torch.bfloat16,
                ))
                session = scopes.enter_context(PreparationSession(
                    device=a_source.device, autotune=False, compile_workers=2,
                ))
                session.prepare((quant_plan.request(
                    name="activation-quantization",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.quantize_input(a_source, out=a_quantized_b12x),
                    ),
                ),))

            def b12x_launch():
                if a_quantized_b12x is not None:
                    quantize_block_fp8_linear_input_mxfp8(
                        a_source,
                        plan=quant_plan,
                        out=a_quantized_b12x,
                    )
                    a_values = a_quantized_b12x.values
                    a_scale_for_gemm = a_quantized_b12x.scale_mma
                else:
                    a_values = a_quantized
                    a_scale_for_gemm = a_scale_mma
                dense_gemm(
                    (a_values.view(M, K, 1), a_scale_for_gemm),
                    (b_quantized.view(N, K, 1), b_scale_mma),
                    ab_dtype="float8_e4m3fn",
                    sf_dtype="float8_e8m0fnu",
                    c_dtype="bfloat16",
                    sf_vec_size=32,
                    out=b12x_out,
                    # Match the production scaled-mm route: the graph shape is the
                    # regime hint, so 1024 stays on BK128 while 2048+ may select the
                    # separately keyed BK64 specialization.
                    expected_m=M,
                )

            if include_input_quant:
                b12x_launch()
                session.freeze()
            b12x_replay = capture_graph_replay(b12x_launch)
            results["b12x_replay"] = b12x_replay
            results["b12x_out"] = b12x_out
        except Exception as exc:
            raise BenchmarkAbort(f"b12x execution failed: {exc}") from exc

        if check:
            results["b12x_replay"]()
            a_ref = dequantize_mxfp8_rows_torch(a_quantized, a_scale)
            b_ref = dequantize_mxfp8_rows_torch(b_quantized, b_scale)
            reference = a_ref @ b_ref.T
            torch.cuda.synchronize()
            check_outputs(results["b12x_out"][:, :, 0], reference,
                          label="b12x dequantized reference", cosine_threshold=COSINE_THRESHOLD)
        results["b12x"] = bench_events(
            b12x_replay, warmup=warmup, iters=iters, l2_flush=l2_flush,
        )
        return results


def bench_one_fp8_e2e(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    return bench_one_fp8(
        M,
        N,
        K,
        warmup=warmup,
        iters=iters,
        check=check,
        l2_flush=l2_flush,
        include_input_quant=True,
    )


def bench_one_fp8_block(
    M: int,
    N: int,
    K: int,
    *,
    warmup: int,
    iters: int,
    check: bool,
    l2_flush: Callable[[], None] | None,
):
    """Benchmark compact FP32 K128 block scales with a dequantized reference."""
    if N % 128 or K % 128:
        raise BenchmarkAbort("fp8-block requires N and K divisible by 128")

    torch.manual_seed(42)
    a = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 4).to(
        torch.float8_e4m3fn
    )
    a_scale = (torch.rand(M, K // 128, device="cuda") * 0.01 + 0.005).contiguous()
    b_scale = (
        torch.rand(N // 128, K // 128, device="cuda") * 0.01 + 0.005
    ).contiguous()
    b12x_out = torch.empty(
        (M, N, 1), device="cuda", dtype=torch.bfloat16
    )

    def b12x_launch():
        dense_gemm(
            (a.view(M, K, 1), a_scale),
            (b.view(N, K, 1), b_scale),
            ab_dtype="float8_e4m3fn",
            sf_dtype="float32",
            c_dtype="bfloat16",
            sf_vec_size=128,
            block_fp8=True,
            expected_m=M,
            out=b12x_out,
        )

    results = {}
    try:
        b12x_replay = capture_graph_replay(b12x_launch)
        results["b12x_replay"] = b12x_replay
        results["b12x_out"] = b12x_out
    except Exception as exc:
        raise BenchmarkAbort(f"b12x execution failed: {exc}") from exc

    if check:
        results["b12x_replay"]()
        a_ref = a.float() * a_scale.repeat_interleave(128, dim=1)
        b_ref = b.float() * b_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        reference = a_ref @ b_ref.T
        torch.cuda.synchronize()
        check_outputs(results["b12x_out"][:, :, 0], reference,
                      label="b12x dequantized reference", cosine_threshold=COSINE_THRESHOLD)
    results["b12x"] = bench_events(
        b12x_replay, warmup=warmup, iters=iters, l2_flush=l2_flush,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(GEMM_PROFILES),
        default=DEFAULT_PROFILE,
        help=(
            "Model shape profile. qwen3.8-27b applies the Qwen3.8-27B FFN "
            "gate/up and down projection shapes to NVFP4, MXFP8, and regular "
            "K128 block-FP8 modes. super3-mamba isolates Super3.5 Mamba "
            "input/output projections; use fp4-a16 for the serving precision."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--evidence", type=pathlib.Path, help="JSONL evidence output for A16 comparisons.")
    parser.add_argument("--model-path", type=pathlib.Path, help="Safetensors checkpoint for checkpoint-a16.")
    parser.add_argument("--checkpoint-recipe", choices=("iq2_xs", "iq2_xxs", "q8_0", "nvfp4"),
                        help="Restrict checkpoint-a16 to one stored weight recipe.")
    parser.add_argument("--profile-graphs", action="store_true",
                        help="Expose one cold-L2 replay per checkpoint case to CUDA profiling.")
    a16_config = parser.add_mutually_exclusive_group()
    a16_config.add_argument("--tune-a16", action="store_true", help="Race the 16 A16 tile/split configurations.")
    a16_config.add_argument(
        "--a16-config", type=int, nargs=3, metavar=("TILE_N", "TILE_K", "SPLIT_K"),
        help="Pin the fp4-a16/fp8-a16 launch configuration instead of tuning.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "M values to benchmark. Defaults to 2/4/8 for FP4, "
            "1/2/4/8/4096 for MXFP8, and a decode-to-prefill sweep for "
            "regular block FP8; checkpoint-a16 uses 1/2/4/8/16/512."
        ),
    )
    parser.add_argument("--n", type=int, default=None, help="Override output width N.")
    parser.add_argument("--k", type=int, default=None, help="Override reduction width K.")
    parser.add_argument(
        "--shape-name",
        default="custom",
        help="Label used with --n/--k.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp4", "fp8", "fp8-block", "fp8-e2e", "fp4-a16", "fp8-a16", "checkpoint-a16", "all"),
        default="fp4",
        help=(
            "Benchmark NVFP4, prequantized MXFP8, regular K128 block FP8, "
            "end-to-end MXFP8 including BF16 input quantization, actual "
            "IQ2_XS/NVFP4 checkpoint weights with BF16 inputs, or all synthetic modes."
        ),
    )
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict GPU L2 before each measured invocation (default: enabled).",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    parser.set_defaults(check=True)
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Check against b12x numerical references and fail when cosine similarity falls below the threshold (default: enabled).",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable correctness checks before timing.",
    )
    args = parser.parse_args()

    if args.a16_config is not None and args.dtype not in ("fp4-a16", "fp8-a16"):
        parser.error("--a16-config requires --dtype fp4-a16 or fp8-a16")
    if args.dtype == "checkpoint-a16":
        from benchmarks.experimental.b12x.checkpoint_dense import run
        run(args)
        return
    if args.model_path is not None:
        parser.error("--model-path requires --dtype checkpoint-a16")
    if args.checkpoint_recipe is not None or args.profile_graphs:
        parser.error("--checkpoint-recipe and --profile-graphs require checkpoint-a16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (args.n is None) != (args.k is None):
        raise ValueError("--n and --k must be provided together")
    if args.n is not None and (args.n <= 0 or args.k <= 0):
        raise ValueError("--n and --k must be positive")

    if args.dtype in ("fp4-a16", "fp8-a16"):
        from benchmarks.experimental.b12x.benchmark_blockscaled_precision import run
        specs = ([(args.shape_name, args.k, args.n, "explicit CLI shape")]
                 if args.n is not None else
                 gemm_specs_for_mode(args.dtype, args.profile))
        run(args, specs)
        return

    custom_specs = None
    if args.n is not None:
        custom_specs = (
            (args.shape_name, args.k, args.n, "explicit CLI shape"),
        )

    def selected_specs(mode: str):
        return (
            custom_specs
            if custom_specs is not None
            else gemm_specs_for_mode(mode, args.profile)
        )

    torch.empty(1, device="cuda")
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    if args.dtype == "all":
        benchmark_modes = (
            ("fp4", bench_one_fp4),
            ("fp8", bench_one_fp8),
            ("fp8-block", bench_one_fp8_block),
            ("fp8-e2e", bench_one_fp8_e2e),
        )
    elif args.dtype == "fp4":
        benchmark_modes = (("fp4", bench_one_fp4),)
    elif args.dtype == "fp8":
        benchmark_modes = (("fp8", bench_one_fp8),)
    elif args.dtype == "fp8-block":
        benchmark_modes = (
            ("fp8-block", bench_one_fp8_block),
        )
    else:
        benchmark_modes = (("fp8-e2e", bench_one_fp8_e2e),)
    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    elif args.dtype == "fp4":
        batch_sizes = FP4_BATCH_SIZES
    elif args.dtype == "fp8-block":
        batch_sizes = FP8_BLOCK_BATCH_SIZES
    else:
        batch_sizes = FP8_BATCH_SIZES

    mode_desc = ", ".join(mode.upper() for mode, _ in benchmark_modes)
    print(f"Dense GEMM ({mode_desc}): b12x latency")
    print(f"Profile: {args.profile} ({GEMM_PROFILES[args.profile]})")
    if args.profile == QWEN38_27B_PROFILE:
        print("Qwen3.8-27B FFN gate/up and down projections")
    elif args.profile == SUPER3_MAMBA_PROFILE:
        print("Super3.5 Mamba in_proj and out_proj (40 blocks each)")
    elif args.dtype == "fp4":
        print("NVIDIA Nemotron 3 Super shared-expert down-proj")
    elif args.dtype in ("fp8", "fp8-e2e"):
        print("DeepSeek V4 Flash DSpark TP=2 q_b projection")
    elif args.dtype == "fp8-block":
        print("Qwen regular block-FP8 linear projections")
    else:
        print(
            "FP4: Nemotron shared down; MXFP8: DSV4-DSpark TP=2 q_b; "
            "block FP8: Qwen linears"
        )
    print("Timing mode: stream-gated events (graph qualification enabled)")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    print(f"b12x reference check: {'on' if args.check else 'off'} (cos >= {COSINE_THRESHOLD:.6f})")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print(f"M values: {batch_sizes}")
    print()

    # Collect all results for summary.
    # (mode, name, bs, M, N, K, b12x_med).
    all_results = []

    for mode, bench_fn in benchmark_modes:
        print(f"{'=' * 75}")
        print(f"  {mode.upper()} dense GEMM")
        print(f"{'=' * 75}")

        for name, K, N, note in selected_specs(mode):
            print(f"  {name}  K={K} N={N}  [{note}]")

            for bs in batch_sizes:
                M = bs
                try:
                    results = bench_fn(
                        M,
                        N,
                        K,
                        warmup=args.warmup,
                        iters=args.iters,
                        check=args.check,
                        l2_flush=l2_flush,
                    )
                except BenchmarkAbort as exc:
                    print(
                        f"ERROR: benchmark aborted for {mode} {name} "
                        f"(bs={bs}, M={M}, N={N}, K={K}): {exc}",
                        file=sys.stderr,
                    )
                    raise SystemExit(1) from None

                b12x_med = (
                    statistics.median(results["b12x"]) * 1000
                    if results.get("b12x")
                    else None
                )
                print(f"  {mode:<8} M={M:>5} b12x={b12x_med:7.2f} us")
                all_results.append((mode, name, bs, M, N, K, b12x_med))

            print()

        print()

    print(f"\n{'=' * 75}")
    print("  SUMMARY: b12x latency in microseconds (lower is better)")
    for mode, name, _bs, M, N, K, latency in all_results:
        print(f"  {mode:<9} {name:<30} M={M} N={N} K={K}: {latency:.2f} us")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
