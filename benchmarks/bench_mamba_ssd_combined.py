#!/usr/bin/env python3
"""
Benchmark Mamba2 SSD chunk scan combined kernel.

Compares:
  - FlashInfer CuTe-DSL fused kernel (Blackwell SM100+)
  - Triton reference (5 separate kernels)

Usage:
    python benchmarks/bench_mamba_ssd_combined.py --varlen
    python benchmarks/bench_mamba_ssd_combined.py --batched
    python benchmarks/bench_mamba_ssd_combined.py --varlen --batched
    python benchmarks/bench_mamba_ssd_combined.py --varlen --batch 4 --nchunks 8
    python benchmarks/bench_mamba_ssd_combined.py --ncu --batch 4 --nchunks 8
    python benchmarks/bench_mamba_ssd_combined.py --prof --batch 4 --nchunks 8
"""

import argparse
import sys

import numpy as np
import torch

from flashinfer.mamba import SSDCombined
from flashinfer.testing.utils import bench_gpu_time

sys.path.insert(0, "tests/mamba")
from triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd


# ---------------------------------------------------------------------------
# Data creation
# ---------------------------------------------------------------------------


def compute_varlen_metadata(cu_seqlens, chunk_size):
    """Compute seq_idx, chunk_indices, chunk_offsets for varlen."""
    total_seqlen = cu_seqlens[-1].item()
    seq_idx = torch.zeros(1, total_seqlen, dtype=torch.int32, device=cu_seqlens.device)
    num_seqs = len(cu_seqlens) - 1
    for i in range(num_seqs):
        s = cu_seqlens[i].item()
        e = cu_seqlens[i + 1].item()
        seq_idx[0, s:e] = i

    nchunks = (total_seqlen + chunk_size - 1) // chunk_size
    chunk_indices_list = []
    chunk_offsets_list = []
    for phys_chunk in range(nchunks):
        chunk_start = phys_chunk * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_seqlen)
        chunk_seq_vals = seq_idx[0, chunk_start:chunk_end]
        prev_ids = torch.cat([chunk_seq_vals[:1] - 1, chunk_seq_vals[:-1]])
        transitions = (chunk_seq_vals != prev_ids).nonzero(as_tuple=True)[0]
        for offset in transitions:
            chunk_indices_list.append(phys_chunk)
            chunk_offsets_list.append(offset.item())

    chunk_indices = torch.tensor(
        chunk_indices_list, dtype=torch.int32, device=cu_seqlens.device
    )
    chunk_offsets = torch.tensor(
        chunk_offsets_list, dtype=torch.int32, device=cu_seqlens.device
    )
    return seq_idx, chunk_indices, chunk_offsets


def make_batched_inputs(
    batch, nchunks, nheads, headdim, dstate, ngroups, chunk_size, dtype
):
    """Create batched input tensors. Returns (kwargs, label_dict)."""
    seqlen = nchunks * chunk_size
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    D = torch.randn(nheads, dtype=dtype, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0
    initial_states = torch.randn(
        batch,
        nheads,
        headdim,
        dstate,
        dtype=dtype,
        device="cuda",
    )
    kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
    )
    label = dict(col0=batch, chunks_per_seq=nchunks, nchunks=nchunks, seqlen=seqlen)
    return kwargs, label


def make_varlen_inputs(
    num_seqs, chunks_per_seq, nheads, headdim, dstate, ngroups, chunk_size, dtype
):
    """Create packed varlen inputs. Returns (kwargs, label_dict).

    kwargs includes cu_seqlens (needed by triton only — callers strip it for flashinfer)
    and seq_chunk_cumsum (pre-computed so ssd.run() does zero allocations).
    """
    seq_len_each = chunks_per_seq * chunk_size
    total_seqlen = num_seqs * seq_len_each

    x = torch.randn(1, total_seqlen, nheads, headdim, dtype=dtype, device="cuda")
    dt = torch.randn(1, total_seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(1, total_seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    C = torch.randn(1, total_seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    D = torch.randn(nheads, dtype=dtype, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0
    initial_states = torch.randn(
        num_seqs,
        nheads,
        headdim,
        dstate,
        dtype=dtype,
        device="cuda",
    )

    cu_seqlens = torch.tensor(
        [i * seq_len_each for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device="cuda",
    )
    seq_idx, chunk_indices, chunk_offsets = compute_varlen_metadata(
        cu_seqlens, chunk_size
    )

    # Pre-compute seq_chunk_cumsum so ssd.run() does zero allocations
    from flashinfer.mamba.ssd_combined import _get_seq_chunk_cumsum_module

    module = _get_seq_chunk_cumsum_module()
    tile_state_bytes = module.seq_chunk_cumsum_tile_state_size(num_seqs)
    tile_state = (
        torch.empty(tile_state_bytes, dtype=torch.uint8, device="cuda")
        if tile_state_bytes > 0
        else None
    )
    seq_chunk_cumsum = torch.zeros(num_seqs + 1, dtype=torch.int32, device="cuda")
    module.seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        seq_chunk_cumsum,
        tile_state,
        chunk_size,
        len(chunk_indices),
        num_seqs,
    )

    kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        seq_chunk_cumsum=seq_chunk_cumsum,
    )
    nchunks = total_seqlen // chunk_size
    label = dict(
        col0=num_seqs,
        chunks_per_seq=chunks_per_seq,
        nchunks=nchunks,
        seqlen=total_seqlen,
    )
    return kwargs, label


# ---------------------------------------------------------------------------
# Layer 1: bench_one — create data, warmup, measure
# ---------------------------------------------------------------------------


def bench_one(
    fn,
    mode,
    config,
    model_params,
    strip_keys,
    warmup,
    repetitions,
    enable_cupti,
    use_cuda_graph,
):
    """Create inputs, warmup, and measure a single kernel on a single config.

    Args:
        fn: kernel callable
        mode: "batched" or "varlen"
        config: (a, b) tuple — (batch, nchunks) or (num_seqs, chunks_per_seq)
        model_params: dict with nheads, headdim, dstate, ngroups, chunk_size, dtype
        strip_keys: set of kwarg keys to remove before calling fn
        warmup: number of explicit warmup iterations
        repetitions: passed to bench_gpu_time
        enable_cupti: use CUPTI timing
        use_cuda_graph: use CUDA graph timing

    Returns:
        (label_dict, median_ms)
    """
    p = model_params
    a, b = config
    make_fn = make_varlen_inputs if mode == "varlen" else make_batched_inputs
    kwargs, label = make_fn(
        a,
        b,
        p["nheads"],
        p["headdim"],
        p["dstate"],
        p["ngroups"],
        p["chunk_size"],
        p["dtype"],
    )
    kw = (
        {k: v for k, v in kwargs.items() if k not in strip_keys}
        if strip_keys
        else kwargs
    )

    # Explicit warmup
    for _ in range(warmup):
        fn(**kw)
    torch.cuda.synchronize()

    times = bench_gpu_time(
        lambda **k: fn(**k),
        input_kwargs=kw,
        enable_cupti=enable_cupti,
        use_cuda_graph=use_cuda_graph,
        dry_run_iters=warmup,
        repeat_iters=repetitions,
    )
    return label, np.median(times)


# ---------------------------------------------------------------------------
# Layer 2: bench_mode — batched or varlen harness
# ---------------------------------------------------------------------------


def bench_mode(configs, mode, model_params, bench_params, title=None):
    """Run a list of configs in either 'batched' or 'varlen' mode.

    Args:
        configs: list of (a, b) tuples.
                 batched: (batch, nchunks), varlen: (num_seqs, chunks_per_seq)
        mode: "batched" or "varlen"
        model_params: dict with nheads, headdim, dstate, ngroups, chunk_size, dtype
        bench_params: dict with warmup, repetitions, enable_cupti, use_cuda_graph
    """
    if title:
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)

    p = model_params
    print()
    print(
        f"  chunk_size={p['chunk_size']}, nheads={p['nheads']}, headdim={p['headdim']}, "
        f"dstate={p['dstate']}, ngroups={p['ngroups']}"
    )

    col0_name = "num_seqs" if mode == "varlen" else "batch"
    print()
    print(
        f"  {col0_name:<10} {'chunks/sequence':<18} {'total chunks':<14} {'total seqlen':<14} "
        f"{'FlashInfer (ms)':<18} {'Triton (ms)':<18} {'Speedup':<10}"
    )
    print("  " + "-" * 108)

    # Construct SSDCombined once for this mode — uses class-based API directly
    has_init_states = mode == "varlen"
    ssd = SSDCombined(
        chunk_size=p["chunk_size"],
        nheads=p["nheads"],
        headdim=p["headdim"],
        dstate=p["dstate"],
        ngroups=p["ngroups"],
        io_dtype=p["dtype"],
        has_d=True,
        has_initial_states=has_init_states,
        has_varlen=(mode == "varlen"),
        seq_idx_dtype=torch.int32,  # varlen metadata uses int32
    )

    # Keys to strip per kernel per mode.
    # - FlashInfer (SSDCombined.run) doesn't accept chunk_size or cu_seqlens
    # - In batched mode, strip initial_states from both: Triton's _chunk_scan_fwd
    #   only supports initial_states with batch==1 + varlen metadata.
    # - Triton doesn't know about seq_chunk_cumsum
    strip_keys = {
        "batched": {
            "FlashInfer": {"chunk_size", "initial_states"},
            "Triton": {"initial_states"},
        },
        "varlen": {
            "FlashInfer": {"cu_seqlens", "chunk_size"},
            "Triton": {"seq_chunk_cumsum"},
        },
    }

    kernels = [
        ("FlashInfer", ssd.run),
        ("Triton", _mamba_chunk_scan_combined_fwd),
    ]

    for cfg in configs:
        results = {}
        label = None
        for name, fn in kernels:
            label, median_ms = bench_one(
                fn,
                mode,
                cfg,
                p,
                strip_keys[mode][name],
                **bench_params,
            )
            results[name] = median_ms

        fi_ms = results["FlashInfer"]
        tr_ms = results["Triton"]
        speedup = tr_ms / fi_ms

        print(
            f"  {label['col0']:<10} {label['chunks_per_seq']:<18} {label['nchunks']:<14} {label['seqlen']:<14} "
            f"{fi_ms:<18.4f} {tr_ms:<18.4f} {speedup:<10.2f}x"
        )


# ---------------------------------------------------------------------------
# Layer 3: sweep or single-point
# ---------------------------------------------------------------------------

VARLEN_CONFIGS = [
    (1, 1),
    (4, 1),
    (8, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    (4, 8),
    (8, 8),
    (16, 8),
    (32, 8),
    (64, 8),
    (32, 32),
    (64, 32),
    (128, 32),
]

BATCHED_CONFIGS = [
    (1, 1),
    (1, 4),
    (1, 16),
    (1, 64),
    (1, 256),
    (4, 1),
    (4, 4),
    (4, 16),
    (4, 64),
    (16, 1),
    (16, 4),
    (16, 16),
    (64, 1),
    (64, 4),
    (64, 16),
    (128, 1),
    (128, 4),
    (128, 16),
    (256, 1),
    (256, 4),
    (256, 16),
    (512, 2),
]


def run_benchmarks(args, model_params, bench_params):
    """Top-level: sweep or single-point, batched or varlen."""
    single_point = args.batch is not None and args.nchunks is not None

    if single_point:
        if args.varlen:
            bench_mode(
                [(args.batch, args.nchunks)],
                "varlen",
                model_params,
                bench_params,
                title="SINGLE POINT (varlen)",
            )
        if args.batched:
            bench_mode(
                [(args.batch, args.nchunks)],
                "batched",
                model_params,
                bench_params,
                title="SINGLE POINT (batched)",
            )
        # default to batched if neither flag given
        if not args.varlen and not args.batched:
            bench_mode(
                [(args.batch, args.nchunks)],
                "batched",
                model_params,
                bench_params,
                title="SINGLE POINT (batched)",
            )
    else:
        if args.varlen:
            bench_mode(
                VARLEN_CONFIGS,
                "varlen",
                model_params,
                bench_params,
                title="VARLEN: serving scenario — packed user sequences (batch=1)",
            )
        if args.batched:
            bench_mode(
                BATCHED_CONFIGS,
                "batched",
                model_params,
                bench_params,
                title="BATCHED: uniform sequence lengths (no varlen metadata, no init states)",
            )


# ---------------------------------------------------------------------------
# NCU profiling mode
# ---------------------------------------------------------------------------


def ncu_mode(args, model_params):
    """Single kernel launch for NCU profiling."""
    assert args.batch is not None and args.nchunks is not None, (
        "--ncu requires --batch and --nchunks"
    )
    p = model_params
    seqlen = args.nchunks * p["chunk_size"]
    x = torch.randn(
        args.batch, seqlen, p["nheads"], p["headdim"], dtype=p["dtype"], device="cuda"
    )
    dt = torch.randn(
        args.batch, seqlen, p["nheads"], dtype=torch.float32, device="cuda"
    )
    A = -torch.rand(p["nheads"], dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(
        args.batch, seqlen, p["ngroups"], p["dstate"], dtype=p["dtype"], device="cuda"
    )
    C = torch.randn(
        args.batch, seqlen, p["ngroups"], p["dstate"], dtype=p["dtype"], device="cuda"
    )
    D = torch.randn(p["nheads"], dtype=p["dtype"], device="cuda")
    dt_bias = torch.rand(p["nheads"], dtype=torch.float32, device="cuda") - 4.0

    torch.cuda.synchronize()
    print(
        f"  NCU mode: launching kernel once (batch={args.batch}, "
        f"nchunks={args.nchunks}, seqlen={seqlen})"
    )
    ssd = SSDCombined(
        chunk_size=p["chunk_size"],
        nheads=p["nheads"],
        headdim=p["headdim"],
        dstate=p["dstate"],
        ngroups=p["ngroups"],
        io_dtype=p["dtype"],
        has_d=True,
        has_initial_states=False,
    )
    ssd.run(x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize()
    print("  Done.")


# ---------------------------------------------------------------------------
# torch.profiler mode — host-side overhead analysis
# ---------------------------------------------------------------------------


def profile_mode(args, model_params):
    """Profile SSDCombined.run() with torch.profiler to find host-side bottlenecks.

    Writes traces to ./profiler_traces/ — view with:
      - Perfetto UI: https://ui.perfetto.dev/ (drag-drop the .json file)
      - TensorBoard: tensorboard --logdir=./profiler_traces
    """
    from torch.profiler import ProfilerActivity, profile, schedule

    assert args.batch is not None and args.nchunks is not None, (
        "--prof requires --batch and --nchunks"
    )
    p = model_params
    kwargs, _ = make_batched_inputs(
        args.batch,
        args.nchunks,
        p["nheads"],
        p["headdim"],
        p["dstate"],
        p["ngroups"],
        p["chunk_size"],
        p["dtype"],
    )
    # SSDCombined.run() doesn't take chunk_size (it's set in __init__)
    kw = {k: v for k, v in kwargs.items() if k != "chunk_size"}

    ssd = SSDCombined(
        chunk_size=p["chunk_size"],
        nheads=p["nheads"],
        headdim=p["headdim"],
        dstate=p["dstate"],
        ngroups=p["ngroups"],
        io_dtype=p["dtype"],
        has_d=True,
        has_initial_states=True,
    )

    warmup = args.warmup
    active = args.repetitions or 3

    print(
        f"  Profiler mode: batch={args.batch}, nchunks={args.nchunks}, "
        f"seqlen={args.nchunks * p['chunk_size']}"
    )
    print(f"  Warmup: {warmup}, Active (profiled) iterations: {active}")

    # Warmup (outside profiler to avoid capturing JIT compilation)
    for _ in range(warmup):
        ssd.run(**kw)
    torch.cuda.synchronize()
    print("  Warmup done, starting profiler...")

    # schedule: wait=0, warmup=0, active=N
    # (real warmup already done above, so all profiler steps are active)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=schedule(wait=0, warmup=0, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_traces"),
    ) as prof:
        for _ in range(active):
            ssd.run(**kw)
            torch.cuda.synchronize()
            prof.step()

    print()
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="cpu_time_total",
            row_limit=30,
        )
    )
    print()
    print("  Traces saved to ./profiler_traces/")
    print("  View with: https://ui.perfetto.dev/ (drag-drop the .json file)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba2 SSD combined kernel")
    parser.add_argument("--cupti", action="store_true", help="Use CUPTI timing")
    parser.add_argument(
        "--cuda-graph", action="store_true", help="Use CUDA graph timing"
    )
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batched", action="store_true", help="Run batched benchmark")
    parser.add_argument("--varlen", action="store_true", help="Run varlen benchmark")
    parser.add_argument("-w", "--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument(
        "-r", "--repetitions", type=int, default=None, help="Measurement iterations"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (single-point or varlen num_seqs)",
    )
    parser.add_argument(
        "--nchunks",
        type=int,
        default=None,
        help="Number of chunks (or chunks_per_seq for varlen)",
    )
    parser.add_argument(
        "--ncu", action="store_true", help="NCU profiling mode: single kernel launch"
    )
    parser.add_argument(
        "--prof",
        action="store_true",
        help="torch.profiler mode: host-side overhead analysis",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    model_params = dict(
        nheads=args.nheads,
        headdim=args.headdim,
        dstate=args.dstate,
        ngroups=args.ngroups,
        chunk_size=args.chunk_size,
        dtype=dtype,
    )
    bench_params = dict(
        warmup=args.warmup,
        repetitions=args.repetitions,
        enable_cupti=args.cupti,
        use_cuda_graph=args.cuda_graph,
    )

    # Header
    print("=" * 100)
    print("Mamba2 SSD Combined Kernel Benchmark")
    print("=" * 100)
    print(f"  Device:     {torch.cuda.get_device_name()}")
    timing = (
        "CUPTI" if args.cupti else "CUDA Graphs" if args.cuda_graph else "CUDA Events"
    )
    print(f"  Timing:     {timing}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Warmup:     {args.warmup}")
    print(f"  Reps:       {args.repetitions or 'auto'}")

    # NCU mode
    if args.ncu:
        ncu_mode(args, model_params)
        return

    # Profiler mode
    if args.prof:
        profile_mode(args, model_params)
        return

    # Determine which modes to run
    do_batched = args.batched
    do_varlen = args.varlen
    # Default: if neither flag given and no single-point, require at least one
    if not do_batched and not do_varlen:
        single_point = args.batch is not None and args.nchunks is not None
        if single_point:
            do_batched = True  # default single-point to batched
        else:
            print(
                "\n  Error: specify --batched and/or --varlen (or --batch/--nchunks for single-point)"
            )
            return

    run_benchmarks(args, model_params, bench_params)

    print()
    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
