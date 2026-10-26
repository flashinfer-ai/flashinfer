#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Collect checkpointing SSU benchmark data as a function of PNAT (previously
accepted tokens) and plot results — one line per (kernel, npredicted).

Usage:
  # Full run: sweep PNAT at npredicted={4,8}, collect + plot
  python benchmarks/collect_checkpointing_ssu_runs.py --batch-size 512 \
      --pnat 0,1,4,8,12,14 --npredicted 4,8 --max-window 16 --state-dtypes bf16 --cupti

  # Plot only from existing CSV
  python benchmarks/collect_checkpointing_ssu_runs.py --plot-only benchmarks/img/foo.csv

NOTE (2026-07-09): pre-07-09 f32 tags were collected under the launcher's
then-default stg2/cps4 regime; the default is now stg1/cps16 (tuned on the
production-shaped mixed-PNAT + conv1d workload), so fresh f32 large-batch 2k
rows in this UNIFORM-PNAT sweep read ~8-28 µs higher than those tags.  That is
the default change, not a kernel regression.  The FLASHINFER_SSU_MAIN_* envs
(inherited by the child bench subprocess) remain available as optional
overrides to pin any regime when isolating kernels.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
# RESULTS_DIR = SCRIPT_DIR
# Shared with bench_ssu_checkpoint_mixed.py: FLASHINFER_SSU_BENCH_OUTDIR else cwd.
RESULTS_DIR = Path(os.environ.get("FLASHINFER_SSU_BENCH_OUTDIR", ".")).expanduser()
IMG_DIR = RESULTS_DIR / "img"
BENCH_SCRIPT = SCRIPT_DIR / "bench_checkpointing_ssu.py"

DEFAULT_NPREDICTED = [4, 8]  # one line per value, per kernel
DEFAULT_PNAT = [0, 1, 4, 8, 12, 14]  # x-axis: previously accepted tokens (≤ max_window)
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def parse_bench_output(text: str) -> list[dict]:
    """Parse the table output from bench_checkpointing_ssu.py into rows."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "kernel" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # parts[0] and parts[-1] are empty from leading/trailing |
        parts = [p for p in parts if p]
        if len(parts) < 9:
            continue
        kernel = parts[0]
        batch = int(parts[1])
        mtp_len = int(parts[2])
        pnat = parts[3]
        state_dtype = parts[4]
        act_dtype = parts[5]
        median_us = float(parts[6])
        p95_us = float(parts[7])
        p99_us = float(parts[8])
        rows.append(
            {
                "kernel": kernel,
                "batch": batch,
                "mtp_len": mtp_len,
                "pnat": pnat,
                "state_dtype": state_dtype,
                "act_dtype": act_dtype,
                "median_us": median_us,
                "p95_us": p95_us,
                "p99_us": p99_us,
            }
        )
    return rows


def collect_data(args) -> list[dict]:
    """Run bench_checkpointing_ssu.py for each npredicted, sweeping absolute PNAT."""
    all_rows = []
    pnat_str = ",".join(str(p) for p in args.pnat)
    for mtp in args.npredicted:
        cmd = [
            sys.executable,
            str(BENCH_SCRIPT),
            "--batch-sizes",
            str(args.batch_size),
            "--mtp-lengths",
            str(mtp),
            "--max-window",
            str(args.max_window),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            # "--baseline", "flashinfer",
            "--no-flashinfer-dump",
            "--state-dtypes",
            args.state_dtypes,
            "--pnat",
            pnat_str,
            "--output",
            "-",
        ]
        if args.cupti:
            cmd.append("--cupti")
        print(f"Running npredicted={mtp}, pnat={args.pnat} ...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}):", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            continue
        print(result.stdout)
        rows = parse_bench_output(result.stdout)
        all_rows.extend(rows)
    return all_rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write collected rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kernel",
        "batch",
        "mtp_len",
        "pnat",
        "state_dtype",
        "act_dtype",
        "median_us",
        "p95_us",
        "p99_us",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {path}")


def read_csv(path: Path) -> list[dict]:
    """Read CSV back into row dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["batch"] = int(r["batch"])
            r["mtp_len"] = int(r["mtp_len"])
            r["median_us"] = float(r["median_us"])
            r["p95_us"] = float(r["p95_us"])
            r["p99_us"] = float(r["p99_us"])
            rows.append(r)
    return rows


def plot_results(rows: list[dict], png_path: Path) -> None:
    """Plot latency vs PNAT (previously accepted tokens), one line per (kernel, npredicted).

    Kernel → color, npredicted (mtp) → linestyle/marker.  Only the CUDA kernels
    are drawn; other impls in the CSV are skipped."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Kernel → (color, legend label).  Kernels not listed here are skipped.
    kernel_style = {
        "cuda-incr": ("#ff7f0e", "cuda-incr (mono)"),
        "cuda-incr-2k": ("#1f77b4", "cuda-incr-2k"),
    }
    # npredicted (mtp) → (linestyle, marker), assigned from the values actually present.
    ls_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D"]
    mtp_vals = sorted(set(r["mtp_len"] for r in rows))
    mtp_style = {
        m: (ls_cycle[i % 4], marker_cycle[i % 4]) for i, m in enumerate(mtp_vals)
    }

    # Build series: key = (kernel, mtp) -> sorted list of (pnat, median_us).
    series: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for r in rows:
        if r["pnat"] == "N/A":
            continue
        key = (r["kernel"], r["mtp_len"])
        series.setdefault(key, []).append((int(r["pnat"]), r["median_us"]))
    for v in series.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(10, 6))
    for (kernel, mtp), pts in sorted(series.items()):
        if kernel not in kernel_style or mtp not in mtp_style:
            continue
        color, klabel = kernel_style[kernel]
        ls, marker = mtp_style[mtp]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=ls,
            linewidth=1.8,
            marker=marker,
            markersize=5,
            label=f"{klabel} (mtp={mtp})",
        )

    batch = rows[0]["batch"] if rows else "?"
    state_dtypes = sorted(set(r["state_dtype"] for r in rows))
    state_dtype_str = ",".join(state_dtypes)
    ax.set_xlabel("PNAT (previously accepted tokens)")
    ax.set_ylabel("Median latency (us)")
    ax.set_title(
        f"Checkpointing SSU: latency vs PNAT  (batch={batch}, state={state_dtype_str})"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    pnat_vals = sorted(set(int(r["pnat"]) for r in rows if r["pnat"] != "N/A"))
    ax.set_xticks(pnat_vals)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect checkpointing SSU benchmarks and plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for benchmarks"
    )
    parser.add_argument(
        "--npredicted",
        type=str,
        default=",".join(str(m) for m in DEFAULT_NPREDICTED),
        help="Comma-separated NPREDICTED (mtp_len) values — one line per value, per kernel "
        "(e.g. --npredicted 4,8)",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=16,
        help="Cache capacity. CUDA checkpoints when PNAT + npredicted > max_window; "
        "Triton receives a matching write_checkpoint flag. Must be >= every PNAT value.",
    )
    parser.add_argument(
        "--pnat",
        type=str,
        default=",".join(str(p) for p in DEFAULT_PNAT),
        help="Comma-separated absolute PNAT (previously accepted tokens) values for the "
        "x-axis; each must be <= max-window (e.g. --pnat 0,1,4,8,12,14)",
    )
    parser.add_argument(
        "--state-dtypes",
        type=str,
        default="f32",
        help="Comma-separated state dtypes: f16, bf16, f32, int8, fp8 "
        "(i8/fp16/fp32/e4m3 aliases). "
        "int8 and fp8 (e4m3) use per-(head, dim) block scaling. "
        "Append '-philox-<N>' to f16/fp16/int8/i8/fp8/e4m3 for "
        "Philox-<N> stochastic rounding on gmem state writes "
        "(cuda-incr / Triton incr only). "
        "Example: 'fp16-philox-5', 'int8-philox-10', 'fp8-philox-5'.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Path to existing CSV; skip data collection, just plot",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: incremental_ssu_b<batch>_<state>)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Version tag appended to the output filename as _v<tag> "
        "(e.g. --tag 25 → ..._v25.csv/.png)",
    )
    parser.add_argument(
        "--cupti",
        action="store_true",
        default=False,
        help="Pass --cupti to bench_checkpointing_ssu.py for CUPTI timing",
    )
    args = parser.parse_args()

    args.npredicted = [int(x) for x in args.npredicted.split(",")]
    args.pnat = [int(x) for x in args.pnat.split(",")]
    if any(p > args.max_window for p in args.pnat):
        parser.error(
            f"--pnat values must be <= --max-window ({args.max_window}); got {args.pnat}"
        )

    prefix = (
        args.output_prefix or f"incremental_ssu_b{args.batch_size}_{args.state_dtypes}"
    )
    if args.tag is not None:
        prefix = f"{prefix}_v{args.tag}"
    csv_path = RESULTS_DIR / f"{prefix}.csv"
    png_path = IMG_DIR / f"{prefix}.png"

    if args.plot_only:
        csv_path = Path(args.plot_only)
        # Derive png path next to the csv
        png_path = csv_path.with_suffix(".png")
        rows = read_csv(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")
    else:
        rows = collect_data(args)
        if not rows:
            print("No data collected!", file=sys.stderr)
            sys.exit(1)
        write_csv(rows, csv_path)

    plot_results(rows, png_path)


if __name__ == "__main__":
    main()
