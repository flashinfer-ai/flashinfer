"""
Benchmark: cuDNN grouped_mm (MoE) vs CUTLASS grouped_gemm_nt_masked.

Uses DeepSeek-V3 and Mixtral-8x7B model shapes. Supports both CUPTI and
delay-kernel timing methods for accurate GPU kernel measurement.

Usage:
    python benchmarks/bench_grouped_mm.py [--dtype nvfp4|fp8|all]
    python benchmarks/bench_grouped_mm.py --dtype fp8 --method delay
    python benchmarks/bench_grouped_mm.py --model deepseek-v3 --tokens 4096
"""

import argparse
import random

import cutlass
import cutlass.torch as cutlass_torch
import numpy as np
import torch
import triton
import triton.language as tl

import flashinfer
from flashinfer import SfLayout
from flashinfer.cute_dsl.utils import get_cutlass_dtype
from flashinfer.fp4_quantization import nvfp4_quantize
from flashinfer.gemm import create_scale_factor_tensor, grouped_gemm_nt_masked
from flashinfer.testing.utils import bench_gpu_time

MAX_M = 4096

MOE_MODELS = {
    "mixtral-8x7b": {
        "hidden": 4096,
        "intermediate": 14336,
        "num_experts": 8,
        "top_k": 2,
    },
    "deepseek-v3": {
        "hidden": 7168,
        "intermediate": 18432,
        "num_experts": 64,
        "top_k": 8,
    },
}

TOKEN_COUNTS = [1024, 4096, 16384]

BENCH_KWARGS = dict(
    dry_run_iters=10,
    repeat_iters=30,
    enable_cupti=True,
    use_cuda_graph=False,
    cold_l2_cache=True,
)


# ---------------------------------------------------------------------------
# Delay kernel for accurate GPU timing
# ---------------------------------------------------------------------------


@triton.jit
def _spin_kernel(target_ns):
    start = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - start < target_ns:
        pass


def _launch_delay(duration_us: int = 1000):
    _spin_kernel[(1,)](duration_us * 1000)


def bench_with_delay_kernel(fn, warmup=10, repeat=50):
    """Benchmark using a delay (spin) kernel to eliminate CPU launch overhead.

    A spin kernel keeps the GPU busy so that by the time it finishes, the CPU
    has already enqueued start-event, the target kernel, and end-event.  The
    CUDA events therefore measure pure kernel execution time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        _launch_delay(duration_us=500)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_tflops(total_m, n, k, ms):
    return 2.0 * total_m * n * k * 1e-9 / ms


def generate_expert_token_counts(num_experts, expected_tpe, max_m=MAX_M):
    """Generate per-expert token counts with ±30% random variation.

    Returns a list of ints, one per expert. Shared by both cuDNN and CuTe-DSL
    paths so both sides benchmark the same workload distribution.
    """
    return [
        min(max(1, int(expected_tpe * random.uniform(0.7, 1.3))), max_m)
        for _ in range(num_experts)
    ]


def _seg_lens_to_indptr(seg_lens):
    indptr = [0]
    for s in seg_lens:
        indptr.append(indptr[-1] + s)
    return torch.tensor(indptr, dtype=torch.int32, device="cuda")


def _seg_lens_to_masked_m(seg_lens):
    return torch.tensor(seg_lens, dtype=torch.int32, device="cuda")


def _bench(fn, method):
    if method == "delay":
        return np.median(bench_with_delay_kernel(fn))
    else:
        return np.median(bench_gpu_time(fn, **BENCH_KWARGS))


# ---------------------------------------------------------------------------
# Data creation for grouped_gemm_nt_masked
# ---------------------------------------------------------------------------


def _create_masked_data(num_groups, seg_lens, n, k, ab_dtype_str, sf_vec_size=None):
    if sf_vec_size is None:
        sf_vec_size = 16 if ab_dtype_str == "float4_e2m1fn" else 32
    sf_dtype_str = (
        "float8_e4m3fn" if ab_dtype_str == "float4_e2m1fn" else "float8_e8m0fnu"
    )
    c_dtype_str = "bfloat16"
    device = torch.device("cuda:0")
    l = num_groups
    m = max(seg_lens)

    a_ref = cutlass_torch.matrix(l, m, k, False, cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, False, cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, False, cutlass.Float32)

    _, a_torch = cutlass_torch.cute_tensor_like(
        a_ref,
        get_cutlass_dtype(ab_dtype_str),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    _, b_torch = cutlass_torch.cute_tensor_like(
        b_ref,
        get_cutlass_dtype(ab_dtype_str),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    _, c_torch = cutlass_torch.cute_tensor_like(
        c_ref,
        get_cutlass_dtype(c_dtype_str),
        is_dynamic_layout=True,
        assumed_align=16,
    )

    if ab_dtype_str == "float4_e2m1fn":
        m_dim, k_dim, l_dim = a_torch.shape
        n_dim, _, _ = b_torch.shape
        half_a = a_torch.numel() // 2
        half_b = b_torch.numel() // 2
        a_torch = (
            a_torch.permute(2, 0, 1)
            .flatten()[:half_a]
            .reshape(l, m_dim, k_dim // 2)
            .permute(1, 2, 0)
        )
        b_torch = (
            b_torch.permute(2, 0, 1)
            .flatten()[:half_b]
            .reshape(l, n_dim, k_dim // 2)
            .permute(1, 2, 0)
        )

    _, _, sfa_torch = create_scale_factor_tensor(
        l,
        m,
        k,
        sf_vec_size,
        get_cutlass_dtype(sf_dtype_str),
        device,
    )
    _, _, sfb_torch = create_scale_factor_tensor(
        l,
        n,
        k,
        sf_vec_size,
        get_cutlass_dtype(sf_dtype_str),
        device,
    )

    masked_m_tensor = _seg_lens_to_masked_m(seg_lens)

    return dict(
        a=(a_torch, sfa_torch),
        b=(b_torch, sfb_torch),
        c=c_torch,
        masked_m=masked_m_tensor,
    )


# ---------------------------------------------------------------------------
# NVFP4 benchmarks
# ---------------------------------------------------------------------------


def bench_cudnn_nvfp4(num_experts, seg_lens, n, k, method):
    m_indptr = _seg_lens_to_indptr(seg_lens)
    cum_m = sum(seg_lens)
    a_bf16 = torch.randn(cum_m, k, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")

    a_gsf = (448 * 6) / a_bf16.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(
        a_bf16,
        a_gsf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)

    b_2d = b_bf16.reshape(num_experts * n, k)
    b_gsf = (448 * 6) / b_2d.float().abs().nan_to_num().max()
    b_fp4, b_sf = nvfp4_quantize(
        b_2d,
        b_gsf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    b_fp4 = b_fp4.reshape(num_experts, n, k // 2)
    b_sf = b_sf.view(torch.float8_e4m3fn).reshape(num_experts, -1, k // 16)

    alpha = torch.tensor([1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device="cuda")
    out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")

    flashinfer.grouped_mm.grouped_mm_fp4(
        a_fp4,
        b_fp4,
        a_sf,
        b_sf,
        m_indptr,
        alpha=alpha,
        out=out,
        block_size=16,
    )

    return _bench(
        lambda: flashinfer.grouped_mm.grouped_mm_fp4(
            a_fp4,
            b_fp4,
            a_sf,
            b_sf,
            m_indptr,
            alpha=alpha,
            out=out,
            block_size=16,
        ),
        method,
    )


TILING_CONFIGS = [
    {"mma_tiler_mn": (128, 128), "cluster_shape_mn": (1, 1)},
    {"mma_tiler_mn": (128, 128), "cluster_shape_mn": (2, 1)},
    {"mma_tiler_mn": (256, 128), "cluster_shape_mn": (1, 1)},
    {"mma_tiler_mn": (256, 128), "cluster_shape_mn": (2, 1)},
]


def _bench_masked_best(
    data, ab_dtype, sf_dtype, sf_vec_size, method, extra_kwargs=None
):
    """Run grouped_gemm_nt_masked across tiling configs, return best time."""
    best_ms = float("inf")
    best_cfg = None
    last_error = None
    for cfg in TILING_CONFIGS:
        kwargs = dict(
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype="bfloat16",
            sf_vec_size=sf_vec_size,
            **cfg,
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        def fn(kw=kwargs):
            grouped_gemm_nt_masked(
                lhs=data["a"],
                rhs=data["b"],
                out=data["c"],
                masked_m=data["masked_m"],
                **kw,
            )

        try:
            ms = _bench(fn, method)
            if ms < best_ms:
                best_ms = ms
                best_cfg = cfg
        except Exception as e:
            last_error = e
            continue
    if best_cfg is None:
        raise RuntimeError(f"All tiling configs failed. Last error: {last_error}")
    return best_ms


def bench_masked_nvfp4(num_groups, seg_lens, n, k, method):
    data = _create_masked_data(num_groups, seg_lens, n, k, "float4_e2m1fn")
    ms = _bench_masked_best(
        data,
        "float4_e2m1fn",
        "float8_e4m3fn",
        16,
        method,
        extra_kwargs={"alpha_dtype": "float32"},
    )
    return ms


# ---------------------------------------------------------------------------
# FP8 benchmarks
# ---------------------------------------------------------------------------


def bench_cudnn_fp8(num_experts, seg_lens, n, k, method):
    m_indptr = _seg_lens_to_indptr(seg_lens)
    cum_m = sum(seg_lens)
    a = torch.randn(cum_m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    b = torch.randn(num_experts, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")

    flashinfer.grouped_mm.grouped_mm_fp8(a, b, m_indptr, alpha=None, out=out)

    return _bench(
        lambda: flashinfer.grouped_mm.grouped_mm_fp8(
            a, b, m_indptr, alpha=None, out=out
        ),
        method,
    )


def bench_masked_fp8(num_groups, seg_lens, n, k, method):
    data = _create_masked_data(num_groups, seg_lens, n, k, "float8_e4m3fn")
    ms = _bench_masked_best(
        data,
        "float8_e4m3fn",
        "float8_e8m0fnu",
        32,
        method,
    )
    return ms


# ---------------------------------------------------------------------------
# Runners (DeepSeek / Mixtral shapes)
# ---------------------------------------------------------------------------


def run_moe_benchmark(dtype, method, model_names, token_counts):
    for model_name in model_names:
        cfg = MOE_MODELS[model_name]
        hidden = cfg["hidden"]
        intermediate = cfg["intermediate"]
        num_experts = cfg["num_experts"]
        top_k = cfg["top_k"]

        print()
        print("=" * 115)
        print(
            f"  {model_name.upper()}  ({dtype.upper()})  "
            f"experts={num_experts}  top_k={top_k}  hidden={hidden}  intermediate={intermediate}  "
            f"method={method}"
        )
        print("=" * 115)
        header = (
            f"{'proj':<6} {'tokens':>7} {'E':>4} {'tpe':>6} {'n':>6} {'k':>6}"
            f" | {'cuDNN (ms)':>12} {'TFLOPS':>8}"
            f" | {'masked (ms)':>13} {'TFLOPS':>8}"
            f" | {'Speedup':>8}"
        )
        print(header)
        print("-" * len(header))

        for proj, k_dim, n_dim in [("fwd", hidden, intermediate)]:
            for n_tokens in token_counts:
                total = n_tokens * top_k
                tpe = max(1, total // num_experts)
                seg_lens = generate_expert_token_counts(num_experts, tpe)
                total_m = sum(seg_lens)

                cudnn_str = ""
                cudnn_ms = None
                try:
                    if dtype == "fp8":
                        cudnn_ms = bench_cudnn_fp8(
                            num_experts, seg_lens, n_dim, k_dim, method
                        )
                    else:
                        cudnn_ms = bench_cudnn_nvfp4(
                            num_experts, seg_lens, n_dim, k_dim, method
                        )
                    cudnn_tflops = compute_tflops(total_m, n_dim, k_dim, cudnn_ms)
                    cudnn_str = f"{cudnn_ms:>10.3f}ms {cudnn_tflops:>7.1f}T"
                except Exception as e:
                    cudnn_str = f"{'SKIP':>12} {'':>8}"
                    print(f"  [cuDNN err] {e}")

                masked_str = ""
                masked_ms = None
                try:
                    if dtype == "fp8":
                        masked_ms = bench_masked_fp8(
                            num_experts, seg_lens, n_dim, k_dim, method
                        )
                    else:
                        masked_ms = bench_masked_nvfp4(
                            num_experts, seg_lens, n_dim, k_dim, method
                        )
                    masked_tflops = compute_tflops(total_m, n_dim, k_dim, masked_ms)
                    masked_str = f"{masked_ms:>11.3f}ms {masked_tflops:>7.1f}T"
                except Exception as e:
                    masked_str = f"{'SKIP':>13} {'':>8}"
                    print(f"  [masked err] {e}")

                speedup_str = ""
                if cudnn_ms is not None and masked_ms is not None:
                    speedup_str = f"{cudnn_ms / masked_ms:>7.2f}x"

                print(
                    f"{proj:<6} {n_tokens:>7} {num_experts:>4} {tpe:>6} {n_dim:>6} {k_dim:>6}"
                    f" | {cudnn_str} | {masked_str} | {speedup_str}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark cuDNN grouped_mm vs CUTLASS grouped_gemm_nt_masked (MoE shapes)",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp8", "nvfp4", "all"],
        default="all",
        help="Data type to benchmark (default: all)",
    )
    parser.add_argument(
        "--method",
        choices=["cupti", "delay"],
        default="cupti",
        help="Timing method: 'cupti' (CUPTI/fallback) or 'delay' (delay-kernel + CUDA events)",
    )
    parser.add_argument(
        "--model",
        choices=list(MOE_MODELS.keys()) + ["all"],
        default="all",
        help="Model to benchmark (default: all)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=TOKEN_COUNTS,
        help=f"Token counts to test (default: {TOKEN_COUNTS})",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    print(f"PyTorch: {torch.__version__}, Device: {torch.cuda.get_device_name()}")
    print(f"Timing method: {args.method}")

    model_names = list(MOE_MODELS.keys()) if args.model == "all" else [args.model]
    dtypes = ["fp8", "nvfp4"] if args.dtype == "all" else [args.dtype]

    for dt in dtypes:
        run_moe_benchmark(dt, args.method, model_names, args.tokens)

    print("\nDone.")
