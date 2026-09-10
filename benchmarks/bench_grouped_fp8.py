"""Reproduce grouped FP8 GEMM throughput and compile-cache reuse across M.

Run from a FlashInfer checkout with SM100/SM103 and nvidia-cutlass-dsl installed:
    python benchmarks/bench_grouped_fp8.py --cache-probe
    python benchmarks/bench_grouped_fp8.py --shape 4096 2048 4096 16
    python benchmarks/bench_grouped_fp8.py --production-shapes --repeats 2

Both backends use the same FP8 inputs and preallocated output. Warm GPU timing
uses CUDA graphs to exclude Python dispatch/compilation; cache-probe host timing
includes dispatch, output allocation, compilation (on misses), and synchronization.
JSON lines record the actual software/GPU environment and units.
"""

import argparse
import importlib.metadata
import json
import statistics
import time

import torch

from flashinfer.gemm import group_deepgemm_fp8_nt_groupwise


PRODUCTION_SHAPES = [
    (131072, 256, 4096, 512),
    (131072, 4096, 128, 512),
    (16384, 2048, 4096, 64),
    (16384, 4096, 1024, 64),
    (4096, 2048, 4096, 16),
    (4096, 4096, 1024, 16),
]


def environment():
    """Report the actual benchmark stack rather than a historical package pin."""
    packages = {}
    for name in ("flashinfer-python", "torch", "nvidia-cutlass-dsl"):
        packages[name] = importlib.metadata.version(name)
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "cuda": torch.version.cuda,
        "packages": packages,
    }


def make_inputs(m, n, k, groups):
    """Build aligned expert blocks, allowing a partial final expert."""
    rows = max(128, ((m // groups) // 128) * 128)
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(groups, n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(m, k // 128, device="cuda")
    sb = torch.ones(groups, n // 128, k // 128, device="cuda")
    indices = torch.arange(m, device="cuda", dtype=torch.int32)
    indices = (indices // max(rows, 1)).clamp(max=groups - 1)
    return a, b, sa, sb, indices


def reference(inputs):
    """Independent unit-scale reference for the compile-cost probe."""
    a, b, _, _, indices = inputs
    out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=torch.bfloat16)
    for group in range(b.shape[0]):
        mask = indices == group
        out[mask] = (a[mask].float() @ b[group].float().T).bfloat16()
    return out


def cache_probe():
    """Measure the review's cold/repeat/new-M sequence without changing the key."""
    import flashinfer.gemm.kernels.grouped_gemm_contiguous_blackwell as mod

    # Fresh benchmark process is required: do not clear a caller's live cache.
    if mod._COMPILED:
        raise ValueError("Run --cache-probe in a fresh process")
    for m in (256, 384, 512, 129, 257, 513):
        # Match the review's two-expert boundary at row 128 for every M.
        inputs = list(make_inputs(m, 128, 128, 2))
        inputs[-1] = (torch.arange(m, device="cuda") >= 128).to(torch.int32)
        expected = reference(inputs)
        for phase in ("new_m", "repeat"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = group_deepgemm_fp8_nt_groupwise(*inputs, backend="cute_dsl")
            torch.cuda.synchronize()
            host_ms = (time.perf_counter() - start) * 1000
            torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
            print(
                json.dumps(
                    {
                        "probe": "compile_cache",
                        "m": m,
                        "phase": phase,
                        "host_ms": host_ms,
                        "cache_entries": len(mod._COMPILED),
                        "max_abs_error": (out.float() - expected.float())
                        .abs()
                        .max()
                        .item(),
                    }
                ),
                flush=True,
            )


def benchmark(
    shape, repeats=2, warmup=10, iterations=50, backends=("deepgemm", "cute_dsl")
):
    """Compare warm GPU execution on identical preallocated inputs/outputs."""
    from flashinfer.testing import bench_gpu_time

    torch.manual_seed(0)
    inputs = make_inputs(*shape)
    outputs = {
        backend: torch.empty(shape[0], shape[1], device="cuda", dtype=torch.bfloat16)
        for backend in backends
    }
    for backend, out in outputs.items():
        group_deepgemm_fp8_nt_groupwise(*inputs, out=out, backend=backend)
    torch.cuda.synchronize()
    expected = outputs["deepgemm"] if "deepgemm" in outputs else reference(inputs)
    for out in outputs.values():
        torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
    for repeat in range(repeats):
        # Alternate order to reduce systematic bias from warmup/clock ramping.
        for backend in list(outputs) if repeat % 2 == 0 else list(reversed(outputs)):

            def call():
                return group_deepgemm_fp8_nt_groupwise(
                    *inputs, out=outputs[backend], backend=backend
                )

            times = bench_gpu_time(
                call,
                enable_cupti=False,
                use_cuda_graph=True,
                dry_run_iters=warmup,
                repeat_iters=iterations,
            )
            print(
                json.dumps(
                    {
                        "probe": "warm_gpu",
                        "shape": shape,
                        "backend": backend,
                        "repeat": repeat,
                        "mean_us": statistics.mean(times) * 1000,
                        "median_us": statistics.median(times) * 1000,
                        "timing": "cuda_graph_events",
                        "samples": len(times),
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-probe", action="store_true")
    parser.add_argument("--production-shapes", action="store_true")
    parser.add_argument(
        "--shape", nargs=4, type=int, action="append", metavar=("M", "N", "K", "G")
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("deepgemm", "cute_dsl"),
        default=["deepgemm", "cute_dsl"],
    )
    args = parser.parse_args()
    print(json.dumps({"environment": environment()}), flush=True)
    if args.cache_probe:
        cache_probe()
    shapes = PRODUCTION_SHAPES if args.production_shapes else args.shape or []
    for shape in shapes:
        benchmark(shape, repeats=args.repeats, backends=args.backends)


if __name__ == "__main__":
    main()
