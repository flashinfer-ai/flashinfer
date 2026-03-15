"""
Standalone NVFP4 backend comparison: correctness, CPU overhead, and kernel time.

For various M values (random within 1-1024, N=K=4096), calls each backend
(cudnn, cutlass, trtllm) and compares:
  - Result correctness (cosine similarity)
  - CPU overhead (wall-clock minus GPU time)
  - GPU kernel time (via CUPTI or CUDA events)

Two modes:
  --mode warm  (default): Each M is called many times, measures steady-state.
  --mode cold : Clears cuDNN graph cache before each M, measures first-call
                cost which includes graph build. This simulates LLM serving
                where M varies per request and shows the real impact of
                dynamic shape support.

Usage:
    python tests/gemm/test_nvfp4_backend_comparison.py
    python tests/gemm/test_nvfp4_backend_comparison.py --mode cold
    python tests/gemm/test_nvfp4_backend_comparison.py --num_m_values 20 --use_cupti
    python tests/gemm/test_nvfp4_backend_comparison.py --n 7168 --k 2048
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.gemm import gemm_base as _gemm_base
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability


def _clear_cudnn_graph_cache():
    """Clear cuDNN's cached FP4 GEMM graphs so the next call rebuilds them."""
    # Static path: build_plans_cudnn_fp4_gemm_graph is @functools.cache'd
    if hasattr(_gemm_base, "build_plans_cudnn_fp4_gemm_graph"):
        fn = _gemm_base.build_plans_cudnn_fp4_gemm_graph
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    # Dynamic path (this branch): dict-based cache
    if hasattr(_gemm_base, "_dynamic_fp4_graph_cache"):
        _gemm_base._dynamic_fp4_graph_cache.clear()


BACKENDS = ["cudnn", "cutlass", "trtllm"]


def get_supported_backends(cc_number: int) -> list[str]:
    supported = []
    for b in BACKENDS:
        try:
            if mm_fp4.is_backend_supported(b, cc_number):
                supported.append(b)
        except Exception:
            pass
    return supported


def quantize_inputs(input_bf16, mat2_bf16, do_shuffle_b=False):
    """Quantize input tensors to NVFP4."""
    global_sf_input = (448 * 6) / input_bf16.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2_bf16.float().abs().nan_to_num().max()

    input_fp4, input_sf = nvfp4_quantize(
        input_bf16, global_sf_input, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    mat2_fp4, mat2_sf = nvfp4_quantize(
        mat2_bf16,
        global_sf_mat2,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=do_shuffle_b,
    )
    alpha = 1.0 / (global_sf_input * global_sf_mat2)
    return input_fp4, input_sf, mat2_fp4, mat2_sf, alpha


def run_mm_fp4(input_fp4, mat2_fp4, input_sf, mat2_sf, alpha, backend, res_dtype):
    """Run mm_fp4 with given backend."""
    return mm_fp4(
        a=input_fp4,
        b=mat2_fp4.T,
        a_descale=input_sf,
        b_descale=mat2_sf.T,
        alpha=alpha,
        out_dtype=res_dtype,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
    )


def measure_cpu_overhead(fn, num_iters=50):
    """Measure wall-clock time per call (includes CPU overhead + GPU time)."""
    torch.cuda.synchronize()

    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    # measure
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return np.median(times), np.std(times)


def main():
    parser = argparse.ArgumentParser(description="NVFP4 backend comparison benchmark")
    parser.add_argument("--n", type=int, default=4096, help="N dimension")
    parser.add_argument("--k", type=int, default=4096, help="K dimension")
    parser.add_argument(
        "--m_values",
        type=int,
        nargs="+",
        default=None,
        help="Explicit M values to test (overrides --num_m_values)",
    )
    parser.add_argument(
        "--num_m_values",
        type=int,
        default=10,
        help="Number of random M values to sample from [1, 1024]",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cupti", action="store_true", help="Use CUPTI timing")
    parser.add_argument(
        "--gpu_iters", type=int, default=30, help="Iterations for GPU timing"
    )
    parser.add_argument(
        "--cpu_iters", type=int, default=50, help="Iterations for CPU overhead timing"
    )
    parser.add_argument(
        "--res_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="warm",
        choices=["warm", "cold"],
        help="warm: steady-state timing (cached graphs). "
        "cold: clear cuDNN graph cache per M to measure graph build overhead.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    res_dtype = torch.bfloat16 if args.res_dtype == "bfloat16" else torch.float16

    # Check GPU
    cc = get_compute_capability(torch.device("cuda"))
    cc_number = cc[0] * 10 + cc[1]
    print(f"GPU: {torch.cuda.get_device_name()}, SM{cc_number}")

    backends = get_supported_backends(cc_number)
    # trtllm doesn't support fp16 output
    if res_dtype == torch.float16 and "trtllm" in backends:
        backends.remove("trtllm")
    # trtllm doesn't support SM110/SM120
    if cc[0] in [11, 12] and "trtllm" in backends:
        backends.remove("trtllm")

    if not backends:
        print("No supported backends on this GPU. Exiting.")
        return

    print(f"Backends: {backends}")
    print(f"N={args.n}, K={args.k}, dtype={args.res_dtype}")
    print(
        f"Mode: {args.mode}"
        + (
            " (clears cuDNN graph cache per M)"
            if args.mode == "cold"
            else " (steady-state, cached graphs)"
        )
    )
    print(
        f"GPU timing: {'CUPTI' if args.use_cupti else 'CUDA events'}, "
        f"{args.gpu_iters} iters"
    )
    print()

    # Generate M values
    if args.m_values:
        m_values = sorted(args.m_values)
    else:
        m_values = sorted(random.sample(range(1, 1025), args.num_m_values))
    print(f"M values ({len(m_values)}): {m_values}")
    print()

    n, k = args.n, args.k

    # Pre-quantize weight matrix (shared across M values, except trtllm needs shuffle)
    mat2_bf16 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    # Probe backends with a small run to trigger JIT + filter truly unsupported
    print("Probing backends (JIT warmup)...")
    probe_input = torch.randn([128, k], device="cuda", dtype=torch.bfloat16)
    probe_fp4, probe_sf, probe_mat2_fp4, probe_mat2_sf, probe_alpha = quantize_inputs(
        probe_input, mat2_bf16
    )
    valid_backends = []
    for b in backends:
        try:
            if b == "trtllm":
                _, _, trtllm_mat2_fp4, trtllm_mat2_sf, _ = quantize_inputs(
                    probe_input, mat2_bf16, do_shuffle_b=True
                )
                run_mm_fp4(
                    probe_fp4,
                    trtllm_mat2_fp4,
                    probe_sf,
                    trtllm_mat2_sf,
                    probe_alpha,
                    b,
                    res_dtype,
                )
            else:
                run_mm_fp4(
                    probe_fp4,
                    probe_mat2_fp4,
                    probe_sf,
                    probe_mat2_sf,
                    probe_alpha,
                    b,
                    res_dtype,
                )
            valid_backends.append(b)
            print(f"  {b}: OK")
        except Exception as e:
            print(f"  {b}: FAILED ({e})")
    backends = valid_backends
    if not backends:
        print("No backends passed probe. Exiting.")
        return
    print()

    # Pre-quantize weight for trtllm (shuffled)
    has_trtllm = "trtllm" in backends
    if has_trtllm:
        global_sf_mat2 = (448 * 6) / mat2_bf16.float().abs().nan_to_num().max()
        trtllm_mat2_fp4, trtllm_mat2_sf = nvfp4_quantize(
            mat2_bf16, global_sf_mat2, sfLayout=SfLayout.layout_128x4, do_shuffle=True
        )

    is_cold = args.mode == "cold"

    # Table header
    hdr_parts = ["  M  "]
    if is_cold:
        for b in backends:
            hdr_parts.append(f" {b:>8s}_1st_call(ms) ")
        hdr_parts.append(" cos_sim_check ")
    else:
        for b in backends:
            hdr_parts.append(f" {b:>8s}_gpu(ms) ")
            hdr_parts.append(f" {b:>8s}_wall(ms) ")
            hdr_parts.append(f" {b:>8s}_cpu_oh(ms) ")
        hdr_parts.append(" cos_sim_check ")
    header = "|".join(hdr_parts)
    sep = "-" * len(header)
    print(header)
    print(sep)

    for m in m_values:
        input_bf16 = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
        reference = torch.mm(input_bf16, mat2_bf16.T)

        # Quantize input
        global_sf_input = (448 * 6) / input_bf16.float().abs().nan_to_num().max()
        global_sf_mat2 = (448 * 6) / mat2_bf16.float().abs().nan_to_num().max()
        input_fp4, input_sf = nvfp4_quantize(
            input_bf16,
            global_sf_input,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        # Non-trtllm weight quantization (no shuffle)
        mat2_fp4, mat2_sf = nvfp4_quantize(
            mat2_bf16, global_sf_mat2, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        alpha = 1.0 / (global_sf_input * global_sf_mat2)

        row_parts = [f"{m:>5d}"]
        cos_sims = {}

        for b in backends:
            # Pick correct weight tensors
            if b == "trtllm":
                b_mat2_fp4 = trtllm_mat2_fp4
                b_mat2_sf = trtllm_mat2_sf
            else:
                b_mat2_fp4 = mat2_fp4
                b_mat2_sf = mat2_sf

            # Capture loop vars via default args to avoid late-binding closure bug
            def fn(
                _fp4=input_fp4,
                _mfp4=b_mat2_fp4,
                _sf=input_sf,
                _msf=b_mat2_sf,
                _a=alpha,
                _b=b,
                _dt=res_dtype,
            ):
                return run_mm_fp4(_fp4, _mfp4, _sf, _msf, _a, _b, _dt)

            if is_cold:
                # Cold mode: clear cache, then measure single first-call wall time
                if b == "cudnn":
                    _clear_cudnn_graph_cache()

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn()
                torch.cuda.synchronize()
                first_call_ms = (time.perf_counter() - t0) * 1000

                cos_sim = F.cosine_similarity(
                    reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
                ).item()
                cos_sims[b] = cos_sim
                row_parts.append(f" {first_call_ms:>20.2f} ")
            else:
                # Warm mode: steady-state timing
                # Correctness
                out = fn()
                cos_sim = F.cosine_similarity(
                    reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
                ).item()
                cos_sims[b] = cos_sim

                # GPU kernel time
                gpu_times = bench_gpu_time(
                    fn=run_mm_fp4,
                    repeat_iters=args.gpu_iters,
                    sleep_after_run=True,
                    enable_cupti=args.use_cupti,
                    cold_l2_cache=True,
                    input_args=(
                        input_fp4,
                        b_mat2_fp4,
                        input_sf,
                        b_mat2_sf,
                        alpha,
                        b,
                        res_dtype,
                    ),
                )
                gpu_median = np.median(gpu_times)

                # CPU wall-clock time (includes CPU overhead + GPU)
                wall_median, _ = measure_cpu_overhead(fn, num_iters=args.cpu_iters)

                cpu_overhead = max(wall_median - gpu_median, 0.0)

                row_parts.append(f" {gpu_median:>15.4f} ")
                row_parts.append(f" {wall_median:>16.4f} ")
                row_parts.append(f" {cpu_overhead:>17.4f} ")

        # Cos sim summary
        sim_strs = [f"{b}={v:.4f}" for b, v in cos_sims.items()]
        row_parts.append(f" {', '.join(sim_strs)} ")

        print("|".join(row_parts))

    print(sep)
    print()
    if is_cold:
        print("Legend:")
        print(
            "  *_1st_call(ms) = wall-clock time of first call with cleared graph cache"
        )
        print("                   For cudnn, this includes cuDNN graph build overhead")
        print("                   For cutlass/trtllm, graph build is not needed")
        print("  cos_sim        = cosine similarity vs bf16 reference (>0.97 = PASS)")
    else:
        print("Legend:")
        print("  *_gpu(ms)    = GPU kernel time (median)")
        print("  *_wall(ms)   = wall-clock time per call (median, includes sync)")
        print("  *_cpu_oh(ms) = CPU overhead estimate (wall - gpu)")
        print("  cos_sim      = cosine similarity vs bf16 reference (>0.97 = PASS)")


if __name__ == "__main__":
    main()
