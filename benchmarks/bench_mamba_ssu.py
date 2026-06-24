#!/usr/bin/env python3
"""
Benchmark selective_state_update (varlen) for NVIDIA-Nemotron-3-Ultra-550B-A55B.

Model config (TP=8 slice of the full Ultra model):
  mamba_num_heads=256 → nheads=32 per TP rank
  mamba_head_dim=64
  n_groups=8 → ngroups=1 per TP rank
  ssm_state_size=128
  mamba_ssm_cache_dtype=float32  (state is fp32)
  io dtype: bfloat16

Usage:
  uv run python benchmarks/bench_mamba_ssu.py
  uv run python benchmarks/bench_mamba_ssu.py --backends flashinfer triton
  uv run python benchmarks/bench_mamba_ssu.py --mtp-len 4
"""

import argparse
import sys
import os

import torch

# routines/ lives next to this file
sys.path.insert(0, os.path.dirname(__file__))

from routines.mamba import parse_mamba_args, run_mamba_test

# ---------------------------------------------------------------------------
# Nemotron Ultra full-model config
# ---------------------------------------------------------------------------
NHEADS_FULL = 256
NGROUPS_FULL = 8
HEADDIM = 64
DSTATE = 128

BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512]


def make_args(
    batch_size,
    mtp_len,
    backends,
    no_cuda_graph,
    num_iters,
    dry_run_iters,
    nheads,
    ngroups,
    state_dtype="float32",
    philox_rounds=0,
):
    """Build the args namespace expected by run_mamba_test."""
    import argparse as _ap

    shared_parser = _ap.ArgumentParser(add_help=False)
    shared_parser.add_argument("--routine", type=str, default="selective_state_update")
    shared_parser.add_argument("--verbose", "-v", action="count", default=0)

    line = [
        "--routine",
        "selective_state_update",
        "--batch_size",
        str(batch_size),
        "--nheads",
        str(nheads),
        "--dim",
        str(HEADDIM),
        "--dstate",
        str(DSTATE),
        "--ngroups",
        str(ngroups),
        "--cache_steps",
        str(mtp_len),
        "--input_dtype",
        "bfloat16",
        "--state_dtype",
        state_dtype,
        "--weight_dtype",
        "float32",
        "--dt_softplus",
        "--varlen",
        "--philox-rounds",
        str(philox_rounds),
        "--backends",
    ] + backends

    args = parse_mamba_args(line, shared_parser)

    # Shared benchmark knobs (normally set by flashinfer_benchmark.py)
    args.no_cuda_graph = no_cuda_graph
    args.use_cupti = False
    args.use_cuda_events = True
    args.refcheck = False
    args.allow_output_mismatch = False
    args.random_seed = 42
    args.verbose = 0
    args.output_path = (
        "_collect"  # non-None triggers result collection in run_mamba_test
    )
    args.num_iters = num_iters
    args.dry_run_iters = dry_run_iters
    args.autotune_cache = None
    args.case_tag = f"nemotron_ultra_b{batch_size}_mtp{mtp_len}"
    args.generate_repro_command = False
    args.repro_command = ""

    return args


def _print_table(results, backends):
    """Print a summary table: rows = batch sizes, columns = backends."""
    # Gather: {batch -> {backend -> median_us}}
    from collections import defaultdict

    data = defaultdict(dict)
    for r in results:
        data[r["batch_size"]][r["backend"]] = r["median_time"] * 1e3  # ms → µs

    has_both = "flashinfer" in backends and "triton" in backends

    # Header
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    header = f"  {'batch':>6}"
    for b in backends:
        header += f"  {b + ' (µs)':>16}"
    if has_both:
        header += f"  {'speedup':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for batch in BATCH_SIZES:
        if batch not in data:
            continue
        row = f"  {batch:>6}"
        for b in backends:
            us = data[batch].get(b, float("nan"))
            row += f"  {us:>16.1f}"
        if has_both:
            fi = data[batch].get("flashinfer", float("nan"))
            tr = data[batch].get("triton", float("nan"))
            speedup = tr / fi if fi > 0 else float("nan")
            row += f"  {speedup:>7.2f}x"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Mamba SSU for Nemotron Ultra"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["flashinfer", "triton"],
        choices=["flashinfer", "triton"],
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Tensor parallel size (divides nheads and ngroups). Default: 8",
    )
    parser.add_argument(
        "--mtp-len",
        type=int,
        default=1,
        help="Tokens per sequence (max_seqlen / cache_steps). Default: 1",
    )
    parser.add_argument(
        "--state-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="State cache dtype. Default: float32",
    )
    parser.add_argument(
        "--philox-rounds",
        type=int,
        default=0,
        help="Philox stochastic rounding rounds (0=disabled, requires fp16 state)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph timing (off by default: as_strided "
        "tensors are incompatible with buffer rotation)",
    )
    parser.add_argument("-n", "--num-iters", type=int, default=30)
    parser.add_argument("-d", "--dry-run-iters", type=int, default=5)
    args = parser.parse_args()

    nheads = NHEADS_FULL // args.tp
    ngroups = max(1, NGROUPS_FULL // args.tp)

    print("=" * 70)
    print("Mamba selective_state_update — Nemotron Ultra")
    print("=" * 70)
    print(f"  Device  : {torch.cuda.get_device_name()}")
    print(f"  TP      : {args.tp}")
    print(
        f"  nheads  : {nheads}  headdim: {HEADDIM}  dstate: {DSTATE}  ngroups: {ngroups}"
    )
    print(f"  mtp_len : {args.mtp_len}  (tokens/seq)")
    philox_str = f" + philox-{args.philox_rounds}" if args.philox_rounds > 0 else ""
    print(f"  state   : {args.state_dtype}{philox_str}   io: bfloat16   varlen: yes")
    print(f"  backends: {args.backends}")
    print()

    all_results = []
    for batch in BATCH_SIZES:
        bench_args = make_args(
            batch_size=batch,
            mtp_len=args.mtp_len,
            backends=args.backends,
            no_cuda_graph=not args.cuda_graph,
            num_iters=args.num_iters,
            dry_run_iters=args.dry_run_iters,
            nheads=nheads,
            ngroups=ngroups,
            state_dtype=args.state_dtype,
            philox_rounds=args.philox_rounds,
        )
        print(f"batch={batch}")
        all_results.extend(run_mamba_test(bench_args))
        print()

    _print_table(all_results, args.backends)


if __name__ == "__main__":
    main()
