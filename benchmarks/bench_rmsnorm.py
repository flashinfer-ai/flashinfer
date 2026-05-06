"""Benchmark weightless vs standard RMSNorm and FusedAddRMSNorm.

Usage:
    python benchmarks/bench_rmsnorm.py
    python benchmarks/bench_rmsnorm.py --csv results.csv
    python benchmarks/bench_rmsnorm.py --dtype float16
"""

import argparse
import csv
import os
import pathlib
import sys

# Ensure dev tree takes precedence over any installed flashinfer package
_repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
import statistics

import torch

import flashinfer


def _bench(fn, repeat_iters, dry_run_iters, loop_iters=100):
    """Return (median_ms, std_ms), using inner loop to amortize timer overhead."""
    for _ in range(dry_run_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat_iters):
        start.record()
        for _ in range(loop_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / loop_iters)

    return statistics.median(times), statistics.stdev(times)


DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# (model_name, hidden_size)
HIDDEN_CONFIGS = [
    ("Llama-3.1-8B", 4096),
    ("Qwen3-14B", 5120),
    ("DeepSeek-V3", 7168),
    ("Llama-3.1-70B", 8192),
]

# batch sizes covering decode (1-32) and prefill (128-2048) regimes
BATCH_SIZES = [1, 32, 128, 2048]

DRY_RUN_ITERS = 20
REPEAT_ITERS = 100


def benchmark_rmsnorm(batch_size: int, hidden_size: int, dtype: torch.dtype) -> dict:
    """Benchmark weighted and weightless RMSNorm."""
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    out = torch.empty_like(x)

    def run_with_weight():
        """Run RMSNorm with a weight tensor."""
        flashinfer.rmsnorm(x, w, out=out)

    def run_weightless():
        """Run RMSNorm without a weight tensor."""
        flashinfer.rmsnorm(x, None, out=out)

    t_w, std_w = _bench(run_with_weight, REPEAT_ITERS, DRY_RUN_ITERS)
    t_wl, std_wl = _bench(run_weightless, REPEAT_ITERS, DRY_RUN_ITERS)

    # memory bandwidth: 2 reads (input + weight or input only) + 1 write
    elem = batch_size * hidden_size
    w_elem = hidden_size
    bw_w = (2 * elem + w_elem) * x.element_size() / (t_w * 1e-3) / 1e9
    bw_wl = 2 * elem * x.element_size() / (t_wl * 1e-3) / 1e9

    return {
        "op": "rmsnorm",
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "dtype": str(dtype).split(".")[-1],
        "time_with_weight_us": t_w * 1e3,
        "std_with_weight_us": std_w * 1e3,
        "time_weightless_us": t_wl * 1e3,
        "std_weightless_us": std_wl * 1e3,
        "speedup": t_w / t_wl,
        "bw_with_weight_GBs": bw_w,
        "bw_weightless_GBs": bw_wl,
    }


def benchmark_fused_add_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype
) -> dict:
    """Benchmark weighted and weightless fused add RMSNorm."""
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    r = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    def run_with_weight():
        """Run fused add RMSNorm with a weight tensor."""
        flashinfer.fused_add_rmsnorm(x, r, w)

    def run_weightless():
        """Run fused add RMSNorm without a weight tensor."""
        flashinfer.fused_add_rmsnorm(x, r)

    t_w, std_w = _bench(run_with_weight, REPEAT_ITERS, DRY_RUN_ITERS)
    t_wl, std_wl = _bench(run_weightless, REPEAT_ITERS, DRY_RUN_ITERS)

    # reads: input + residual + weight(or not); writes: input + residual
    elem = batch_size * hidden_size
    w_elem = hidden_size
    bw_w = (4 * elem + w_elem) * x.element_size() / (t_w * 1e-3) / 1e9
    bw_wl = 4 * elem * x.element_size() / (t_wl * 1e-3) / 1e9

    return {
        "op": "fused_add_rmsnorm",
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "dtype": str(dtype).split(".")[-1],
        "time_with_weight_us": t_w * 1e3,
        "std_with_weight_us": std_w * 1e3,
        "time_weightless_us": t_wl * 1e3,
        "std_weightless_us": std_wl * 1e3,
        "speedup": t_w / t_wl,
        "bw_with_weight_GBs": bw_w,
        "bw_weightless_GBs": bw_wl,
    }


def print_row(r: dict, name_width: int = 14) -> None:
    """Print one benchmark result row."""
    print(
        f"  bs={r['batch_size']:>4}  h={r['hidden_size']:>5}  "
        f"with_w={r['time_with_weight_us']:>7.2f}±{r['std_with_weight_us']:>5.2f} µs  "
        f"weightless={r['time_weightless_us']:>7.2f}±{r['std_weightless_us']:>5.2f} µs  "
        f"speedup={r['speedup']:>5.3f}x"
    )


def main():
    """Run RMSNorm benchmarks and optionally save CSV results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16", choices=list(DTYPE_MAP))
    parser.add_argument("--csv", default=None, help="Path to save CSV results")
    args = parser.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")
    print(f"iters: {REPEAT_ITERS} measurement + {DRY_RUN_ITERS} warmup")
    print()

    all_rows = []

    for op_fn, op_name in [
        (benchmark_rmsnorm, "rmsnorm"),
        (benchmark_fused_add_rmsnorm, "fused_add_rmsnorm"),
    ]:
        print(f"── {op_name} ──────────────────────────────────────────────")
        for model_name, hidden_size in HIDDEN_CONFIGS:
            print(f"  [{model_name}, h={hidden_size}]")
            for batch_size in BATCH_SIZES:
                row = op_fn(batch_size, hidden_size, dtype)
                row["gpu"] = gpu_name
                row["model"] = model_name
                print_row(row)
                all_rows.append(row)
        print()

    if args.csv:
        fieldnames = [
            "gpu",
            "op",
            "model",
            "batch_size",
            "hidden_size",
            "dtype",
            "time_with_weight_us",
            "std_with_weight_us",
            "time_weightless_us",
            "std_weightless_us",
            "speedup",
            "bw_with_weight_GBs",
            "bw_weightless_GBs",
        ]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
