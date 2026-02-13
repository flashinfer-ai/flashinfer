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


def bench_varlen(
    configs,
    nheads,
    headdim,
    dstate,
    ngroups,
    chunk_size,
    dtype,
    enable_cupti,
    warmup_iters,
    repeat_iters,
):
    """Sweep over (num_seqs, chunks_per_seq) configs in varlen mode.

    Each config is (num_seqs, chunks_per_seq). Total packed seqlen =
    num_seqs * chunks_per_seq * chunk_size.
    """
    print()
    print(
        f"  chunk_size={chunk_size}, nheads={nheads}, headdim={headdim}, "
        f"dstate={dstate}, ngroups={ngroups}"
    )
    print()
    print(
        f"  {'num_seqs':<10} {'chunks/seq':<12} {'total_seqlen':<14} {'nchunks':<10} "
        f"{'FlashInfer (ms)':<18} {'Triton (ms)':<18} {'Speedup':<10}"
    )
    print("  " + "-" * 92)

    for num_seqs, chunks_per_seq in configs:
        inp = make_varlen_inputs(
            num_seqs,
            chunks_per_seq,
            nheads,
            headdim,
            dstate,
            ngroups,
            chunk_size,
            dtype,
        )
        total_seqlen = inp["total_seqlen"]
        nchunks = total_seqlen // chunk_size

        # FlashInfer
        fi_times = bench_gpu_time(
            lambda: ssd_combined_fwd(
                inp["x"],
                inp["dt"],
                inp["A"],
                inp["B"],
                inp["C"],
                chunk_size,
                D=inp["D"],
                dt_bias=inp["dt_bias"],
                dt_softplus=True,
                initial_states=inp["initial_states"],
                seq_idx=inp["seq_idx"],
                chunk_indices=inp["chunk_indices"],
                chunk_offsets=inp["chunk_offsets"],
            ),
            enable_cupti=enable_cupti,
            dry_run_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

        # Triton reference
        triton_times = bench_gpu_time(
            lambda: _mamba_chunk_scan_combined_fwd(
                inp["x"],
                inp["dt"],
                inp["A"],
                inp["B"],
                inp["C"],
                chunk_size,
                D=inp["D"],
                dt_bias=inp["dt_bias"],
                dt_softplus=True,
                initial_states=inp["initial_states"],
                seq_idx=inp["seq_idx"],
                chunk_indices=inp["chunk_indices"],
                chunk_offsets=inp["chunk_offsets"],
                cu_seqlens=inp["cu_seqlens"],
            ),
            enable_cupti=enable_cupti,
            dry_run_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

        fi_med = np.median(fi_times)
        tr_med = np.median(triton_times)
        speedup = tr_med / fi_med

        print(
            f"  {num_seqs:<10} {chunks_per_seq:<12} {total_seqlen:<14} {nchunks:<10} "
            f"{fi_med:<18.4f} {tr_med:<18.4f} {speedup:<10.2f}x"
        )


def bench_batched(
    configs,
    nheads,
    headdim,
    dstate,
    ngroups,
    chunk_size,
    dtype,
    enable_cupti,
    warmup_iters,
    repeat_iters,
):
    """Sweep over (batch, nchunks) configs in batched mode."""
    print()
    print(
        f"  chunk_size={chunk_size}, nheads={nheads}, headdim={headdim}, "
        f"dstate={dstate}, ngroups={ngroups}"
    )
    print()
    print(
        f"  {'batch':<8} {'nchunks':<10} {'seqlen':<10} "
        f"{'FlashInfer (ms)':<18} {'Triton (ms)':<18} {'Speedup':<10}"
    )
    print("  " + "-" * 74)

    for batch, nchunks in configs:
        seqlen = nchunks * chunk_size

        x, dt, A, B, C, D, dt_bias = make_inputs(
            batch,
            seqlen,
            nheads,
            headdim,
            dstate,
            ngroups,
            dtype,
        )
        initial_states = torch.randn(
            batch, nheads, headdim, dstate, dtype=dtype, device="cuda"
        )

        # FlashInfer — use bench_kineto to measure only GPU kernel time
        def fi_fn():
            ssd_combined_fwd(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                initial_states=initial_states,
            )

        fi_kernel_time = bench_kineto(
            fi_fn,
            "SSDKernel",
            num_tests=repeat_iters or 30,
            suppress_kineto_output=True,
        )

        # Triton reference (no initial_states — the reference has a bug
        # where _chunk_scan_fwd expects chunk_indices when initial_states
        # is provided with batch > 1)
        triton_times = bench_gpu_time(
            lambda: _mamba_chunk_scan_combined_fwd(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
            ),
            enable_cupti=enable_cupti,
            dry_run_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

        fi_med = fi_kernel_time * 1e3  # bench_kineto returns seconds
        tr_med = np.median(triton_times)
        speedup = tr_med / fi_med

        print(
            f"  {batch:<8} {nchunks:<10} {seqlen:<10} "
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
    print(f"  Timing:     {'CUPTI' if args.cupti else 'CUDA Events'}")
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
        bench_batched(
            [(args.batch, args.nchunks)],
            nheads=args.nheads,
            headdim=args.headdim,
            dstate=args.dstate,
            ngroups=args.ngroups,
            chunk_size=args.chunk_size,
            dtype=dtype,
            enable_cupti=args.cupti,
            warmup_iters=args.warmup,
            repeat_iters=args.repetitions,
        )
        print()
        print("=" * 100)
        print("Done.")
        print("=" * 100)
        return

    # -- Varlen: simulate serving with packed user sequences --
    if not args.skip_varlen:
        print()
        print("=" * 100)
        print("VARLEN: serving scenario — packed user sequences (batch=1)")
        print("=" * 100)

        # Realistic serving configs: (num_users, chunks_per_user)
        # Total tokens = num_users * chunks_per_user * chunk_size
        varlen_configs = [
            # Few users, short sequences
            (1, 1),  # 128 tokens
            (4, 1),  # 512 tokens
            (8, 1),  # 1K tokens
            # More users, short sequences (decode-like prefill)
            (32, 1),  # 4K tokens
            (64, 1),  # 8K tokens
            (128, 1),  # 16K tokens
            (256, 1),  # 32K tokens
            # Fewer users, longer sequences (prefill-heavy)
            (4, 8),  # 4K tokens
            (8, 8),  # 8K tokens
            (16, 8),  # 16K tokens
            (32, 8),  # 32K tokens
            (64, 8),  # 64K tokens
            # Large serving batches
            (32, 32),  # 128K tokens
            (64, 32),  # 256K tokens
            (128, 32),  # 512K tokens
        ]

        bench_varlen(
            varlen_configs,
            nheads=args.nheads,
            headdim=args.headdim,
            dstate=args.dstate,
            ngroups=args.ngroups,
            chunk_size=args.chunk_size,
            dtype=dtype,
            enable_cupti=args.cupti,
            warmup_iters=args.warmup,
            repeat_iters=args.repetitions,
        )

    # -- Batched: uniform sequence lengths --
    if not args.skip_batched:
        print()
        print("=" * 100)
        print("BATCHED: uniform sequence lengths (no varlen metadata)")
        print("=" * 100)

        batched_configs = [
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
        ]

        bench_batched(
            batched_configs,
            nheads=args.nheads,
            headdim=args.headdim,
            dstate=args.dstate,
            ngroups=args.ngroups,
            chunk_size=args.chunk_size,
            dtype=dtype,
            enable_cupti=args.cupti,
            warmup_iters=args.warmup,
            repeat_iters=args.repetitions,
        )

    print()
    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
