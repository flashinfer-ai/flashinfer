# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Time four dense PrimsTS GEMM projections at several M, in FP4 and FP8.

Each case is one process of ``benchmarks.bench_prims_ts_gemm``. Tuning and
timing stay in that script. Default M is 4096, 8192, 16384 (8192*2), and
32768 (8192*4).

  qkv       fused QK-norm and RoPE    K=3072   N=9216
  proj      linear                     K=3072   N=3072
  mlp_up    SwiGLU                     K=3072   N=18432
  mlp_down  linear                     K=9216   N=3072

Run from the repository root: python -m benchmarks.bench_prims_ts_gemm_sweep
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_M = (4096, 8192, 8192 * 2, 8192 * 4)
_KERNELS = (
    ("qkv", "qkv_qknorm_rope", 3072, 9216),
    ("proj", "linear", 3072, 3072),
    ("mlp_up", "swiglu", 3072, 18432),
    ("mlp_down", "linear", 9216, 3072),
)


def _command(dtype: str, epilogue: str, k: int, n: int, args) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.bench_prims_ts_gemm",
        "--dtype",
        dtype,
        "--epilogue",
        epilogue,
        "--k",
        str(k),
        "--n",
        str(n),
        "--m",
        *(str(m) for m in args.m),
        "--dry-run-iters",
        str(args.dry_run_iters),
        "--repeat-iters",
        str(args.repeat_iters),
    ]
    if args.tuning_buckets is not None:
        command.extend(
            ["--tuning-buckets", *(str(bucket) for bucket in args.tuning_buckets)]
        )
    if args.all_tactics:
        command.append("--all-tactics")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=list(_M),
        help="Token counts. Default is 4096, 8192, 16384, and 32768.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp4", "fp8"),
        nargs="+",
        default=["fp4", "fp8"],
    )
    parser.add_argument(
        "--kernel",
        choices=tuple(name for name, *_rest in _KERNELS),
        nargs="+",
        default=[name for name, *_rest in _KERNELS],
    )
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument("--tuning-buckets", type=int, nargs="+", default=None)
    parser.add_argument("--all-tactics", action="store_true")
    args = parser.parse_args()

    selected = set(args.kernel)
    failed = []
    for dtype in args.dtype:
        for name, epilogue, k, n in _KERNELS:
            if name not in selected:
                continue
            print(
                f"\n=== {name} dtype={dtype} epilogue={epilogue} K={k} N={n} ===",
                flush=True,
            )
            result = subprocess.run(
                _command(dtype, epilogue, k, n, args), cwd=_ROOT, check=False
            )
            if result.returncode != 0:
                failed.append(f"{name}/{dtype} (exit {result.returncode})")

    if failed:
        print("failed: " + ", ".join(failed), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
