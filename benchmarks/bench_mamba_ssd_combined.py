#!/usr/bin/env python3
"""
Benchmark Mamba2 SSD chunk scan combined kernel.

Compares:
  - FlashInfer CuTe-DSL fused kernel (Blackwell SM100+)
  - Triton reference (5 separate kernels)

Simulates a serving scenario: many users' variable-length sequences are
packed into a single batch=1 tensor (continuous batching). The sweep axis
is total packed sequence length.

The CuTe kernel is persistent — the grid is (batch * nheads) CTAs, each
looping over all chunks assigned to it. With batch=1 and nheads=8, that's
only 8 CTAs on a 160-SM B200, so the kernel has a ~34ms fixed cost from
pipeline fill/drain. Real throughput only shows at large total seqlen.

Usage:
    docker exec -w /home/scratch.ishovkun_gpu/code/flashinfer-dev \
        flashinfer-cu130-dev-ishovkun \
        python benchmarks/bench_mamba_ssd_combined.py

    # With CUPTI (more accurate, requires cupti-python):
    ... python benchmarks/bench_mamba_ssd_combined.py --cupti

    # Custom model dims:
    ... python benchmarks/bench_mamba_ssd_combined.py --nheads 64 --headdim 64
"""

import argparse
import sys

import numpy as np
import torch

# FlashInfer
from flashinfer.mamba import ssd_combined_fwd
from flashinfer.testing.utils import bench_gpu_time, bench_kineto

# Triton reference (lives in tests/)
sys.path.insert(0, "tests/mamba")
from triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd


# ---------------------------------------------------------------------------
# Helpers
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


def make_inputs(batch, seqlen, nheads, headdim, dstate, ngroups, dtype):
    """Create input tensors for benchmarking."""
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device="cuda")
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device="cuda")
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device="cuda")
    D = torch.randn(nheads, dtype=dtype, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0
    return x, dt, A, B, C, D, dt_bias


def make_varlen_inputs(
    num_seqs, chunks_per_seq, nheads, headdim, dstate, ngroups, chunk_size, dtype
):
    """Create packed varlen inputs simulating a serving batch."""
    seq_len_each = chunks_per_seq * chunk_size
    total_seqlen = num_seqs * seq_len_each

    x, dt, A, B, C, D, dt_bias = make_inputs(
        1,
        total_seqlen,
        nheads,
        headdim,
        dstate,
        ngroups,
        dtype,
    )
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
        cu_seqlens,
        chunk_size,
    )

    return dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        total_seqlen=total_seqlen,
        num_seqs=num_seqs,
    )


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def bench_configs(
    configs,
    nheads,
    headdim,
    dstate,
    ngroups,
    chunk_size,
    dtype,
    enable_cupti,
    use_cuda_graph,
    warmup_iters,
    repeat_iters,
    title=None,
):
    """Benchmark a list of configs.

    Each config is a dict with keys for make_inputs / make_varlen_inputs.
    Required: 'batch', 'nchunks' (or 'num_seqs' + 'chunks_per_seq' for varlen).
    """
    if title:
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)
    print()
    print(
        f"  chunk_size={chunk_size}, nheads={nheads}, headdim={headdim}, "
        f"dstate={dstate}, ngroups={ngroups}"
    )
    # Detect if all configs are varlen or batched for header
    is_varlen = any("num_seqs" in c for c in configs)
    col0 = "num_seqs" if is_varlen else "batch"
    print()
    print(
        f"  {col0:<10} {'nchunks':<10} {'seqlen':<10} "
        f"{'FlashInfer (ms)':<18} {'Triton (ms)':<18} {'Speedup':<10}"
    )
    print("  " + "-" * 76)

    for cfg in configs:
        is_varlen = "num_seqs" in cfg

        if is_varlen:
            inp = make_varlen_inputs(
                cfg["num_seqs"],
                cfg["chunks_per_seq"],
                nheads,
                headdim,
                dstate,
                ngroups,
                chunk_size,
                dtype,
            )
            batch = cfg["num_seqs"]
            seqlen = inp["total_seqlen"]
            nchunks = seqlen // chunk_size
            fi_kwargs = dict(
                x=inp["x"],
                dt=inp["dt"],
                A=inp["A"],
                B=inp["B"],
                C=inp["C"],
                chunk_size=chunk_size,
                D=inp["D"],
                dt_bias=inp["dt_bias"],
                dt_softplus=True,
                initial_states=inp["initial_states"],
                seq_idx=inp["seq_idx"],
                chunk_indices=inp["chunk_indices"],
                chunk_offsets=inp["chunk_offsets"],
            )
            tr_kwargs = dict(
                **fi_kwargs,
                cu_seqlens=inp["cu_seqlens"],
            )
        else:
            batch = cfg["batch"]
            nchunks = cfg["nchunks"]
            seqlen = nchunks * chunk_size
            x, dt, A, B, C, D, dt_bias = make_inputs(
                batch, seqlen, nheads, headdim, dstate, ngroups, dtype,
            )
            initial_states = torch.randn(
                batch, nheads, headdim, dstate, dtype=dtype, device="cuda",
            )
            fi_kwargs = dict(
                x=x, dt=dt, A=A, B=B, C=C, chunk_size=chunk_size,
                D=D, dt_bias=dt_bias, dt_softplus=True,
                initial_states=initial_states,
            )
            tr_kwargs = dict(
                x=x, dt=dt, A=A, B=B, C=C, chunk_size=chunk_size,
                D=D, dt_bias=dt_bias, dt_softplus=True,
            )

        # FlashInfer timing
        # Note: CUDA graphs can't capture ssd_combined_fwd because it does
        # host-side tensor prep (bincount, cumsum, etc.) inside the call.
        # Use CUDA events for FlashInfer when --cuda-graph is requested.
        if use_cuda_graph or enable_cupti:
            fi_times = bench_gpu_time(
                lambda: ssd_combined_fwd(**fi_kwargs),
                enable_cupti=enable_cupti,
                use_cuda_graph=use_cuda_graph,
                dry_run_iters=warmup_iters,
                repeat_iters=repeat_iters,
            )
            fi_med = np.median(fi_times)
        else:
            # bench_kineto measures only GPU kernel time via torch profiler
            fi_kernel_time = bench_kineto(
                lambda: ssd_combined_fwd(**fi_kwargs),
                "SSDKernel",
                num_tests=repeat_iters or 30,
                suppress_kineto_output=True,
            )
            fi_med = fi_kernel_time * 1e3  # bench_kineto returns seconds

        # Triton reference
        triton_times = bench_gpu_time(
            lambda: _mamba_chunk_scan_combined_fwd(**tr_kwargs),
            enable_cupti=enable_cupti,
            use_cuda_graph=use_cuda_graph,
            dry_run_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

        tr_med = np.median(triton_times)
        speedup = tr_med / fi_med

        print(
            f"  {batch:<10} {nchunks:<10} {seqlen:<10} "
            f"{fi_med:<18.4f} {tr_med:<18.4f} {speedup:<10.2f}x"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba2 SSD combined kernel")
    parser.add_argument(
        "--cupti", action="store_true", help="Use CUPTI timing (most accurate)"
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use CUDA graph timing (amortizes launch overhead)",
    )
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--skip-batched", action="store_true", help="Skip batched benchmark"
    )
    parser.add_argument(
        "--skip-varlen", action="store_true", help="Skip varlen benchmark"
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup iterations (default: auto)",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=None,
        help="Number of measurement iterations (default: auto)",
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Single-point: batch size"
    )
    parser.add_argument(
        "--nchunks", type=int, default=None, help="Single-point: number of chunks"
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="NCU profiling mode: run kernel exactly once (use with --batch/--nchunks)",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("=" * 100)
    print("Mamba2 SSD Combined Kernel Benchmark")
    print("=" * 100)
    print(f"  Device:     {torch.cuda.get_device_name()}")
    timing_method = "CUPTI" if args.cupti else "CUDA Graphs" if args.cuda_graph else "Kineto (FlashInfer) + CUDA Events (Triton)"
    print(f"  Timing:     {timing_method}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Warmup:     {args.warmup or 'auto'}")
    print(f"  Reps:       {args.repetitions or 'auto'}")

    # JIT warmup — first call compiles each kernel variant
    print()
    print("  Warming up JIT compilation...")
    warmup_x, warmup_dt, warmup_A, warmup_B, warmup_C, warmup_D, warmup_dt_bias = (
        make_inputs(
            1,
            args.chunk_size,
            args.nheads,
            args.headdim,
            args.dstate,
            args.ngroups,
            dtype,
        )
    )
    warmup_init = torch.randn(
        1,
        args.nheads,
        args.headdim,
        args.dstate,
        dtype=dtype,
        device="cuda",
    )

    if not args.skip_batched:
        ssd_combined_fwd(
            warmup_x,
            warmup_dt,
            warmup_A,
            warmup_B,
            warmup_C,
            args.chunk_size,
            D=warmup_D,
            dt_bias=warmup_dt_bias,
            dt_softplus=True,
            initial_states=warmup_init,
        )
    if not args.skip_varlen:
        warmup_seq_idx = torch.zeros(
            1, args.chunk_size, dtype=torch.int32, device="cuda"
        )
        warmup_ci = torch.tensor([0], dtype=torch.int32, device="cuda")
        warmup_co = torch.tensor([0], dtype=torch.int32, device="cuda")
        ssd_combined_fwd(
            warmup_x,
            warmup_dt,
            warmup_A,
            warmup_B,
            warmup_C,
            args.chunk_size,
            D=warmup_D,
            dt_bias=warmup_dt_bias,
            dt_softplus=True,
            initial_states=warmup_init,
            seq_idx=warmup_seq_idx,
            chunk_indices=warmup_ci,
            chunk_offsets=warmup_co,
        )
    del (
        warmup_x,
        warmup_dt,
        warmup_A,
        warmup_B,
        warmup_C,
        warmup_D,
        warmup_dt_bias,
        warmup_init,
    )
    torch.cuda.synchronize()
    print("  Done.")

    # -- NCU profiling mode: single kernel launch, no timing --
    if args.ncu:
        assert args.batch is not None and args.nchunks is not None, (
            "--ncu requires --batch and --nchunks"
        )
        seqlen = args.nchunks * args.chunk_size
        x, dt, A, B, C, D, dt_bias = make_inputs(
            args.batch,
            seqlen,
            args.nheads,
            args.headdim,
            args.dstate,
            args.ngroups,
            dtype,
        )
        torch.cuda.synchronize()
        print(
            f"  NCU mode: launching kernel once (batch={args.batch}, "
            f"nchunks={args.nchunks}, seqlen={seqlen})"
        )
        ssd_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            args.chunk_size,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        torch.cuda.synchronize()
        print("  Done.")
        return

    # -- Single-point mode --
    if args.batch is not None and args.nchunks is not None:
        print()
        print("=" * 100)
        print("SINGLE POINT (batched, no varlen)")
        print("=" * 100)
        bench_configs(
            [{"batch": args.batch, "nchunks": args.nchunks}],
            nheads=args.nheads,
            headdim=args.headdim,
            dstate=args.dstate,
            ngroups=args.ngroups,
            chunk_size=args.chunk_size,
            dtype=dtype,
            enable_cupti=args.cupti,
            use_cuda_graph=args.cuda_graph,
            warmup_iters=args.warmup,
            repeat_iters=args.repetitions,
        )
        print()
        print("=" * 100)
        print("Done.")
        print("=" * 100)
        return

    bench_kwargs = dict(
        nheads=args.nheads,
        headdim=args.headdim,
        dstate=args.dstate,
        ngroups=args.ngroups,
        chunk_size=args.chunk_size,
        dtype=dtype,
        enable_cupti=args.cupti,
        use_cuda_graph=args.cuda_graph,
        warmup_iters=args.warmup,
        repeat_iters=args.repetitions,
    )

    # Varlen: simulate serving with packed user sequences
    if not args.skip_varlen:
        varlen_configs = [
            (1, 1), (4, 1), (8, 1),
            (32, 1), (64, 1), (128, 1), (256, 1),
            (4, 8), (8, 8), (16, 8), (32, 8), (64, 8),
            (32, 32), (64, 32), (128, 32),
        ]
        bench_configs(
            [{"num_seqs": ns, "chunks_per_seq": cps} for ns, cps in varlen_configs],
            title="VARLEN: serving scenario — packed user sequences (batch=1)",
            **bench_kwargs,
        )

    # Batched: uniform sequence lengths
    if not args.skip_batched:
        batched_configs = [
            (1, 1), (1, 4), (1, 16), (1, 64), (1, 256),
            (4, 1), (4, 4), (4, 16), (4, 64),
            (16, 1), (16, 4), (16, 16),
        ]
        bench_configs(
            [{"batch": b, "nchunks": nc} for b, nc in batched_configs],
            title="BATCHED: uniform sequence lengths (no varlen metadata)",
            **bench_kwargs,
        )

    print()
    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
