#!/usr/bin/env python3
"""Stage timings for the Mamba2 SSD path used by vLLM prefill."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from flashinfer.mamba.musa_ssd_bmm import _bmm_chunk_fwd
from flashinfer.mamba.musa_ssd_chunk_scan import _chunk_scan_fwd
from flashinfer.mamba.musa_ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from flashinfer.mamba.musa_ssd_state_passing import _state_passing_fwd


def time_call(fn, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(values),
        "p90_ms": sorted(values)[max(0, int(0.9 * len(values)) - 1)],
    }


def build_metadata(lengths: list[int], chunk_size: int, device: torch.device):
    """Build packed chunk metadata used by the vLLM Mamba2 prefill path."""
    cu_seqlens = [0]
    cu_chunk_seqlens = [0]
    last_chunk_indices = []
    seq_idx = []
    offset = 0
    for sequence_id, length in enumerate(lengths):
        if length <= 0:
            raise ValueError("all sequence lengths must be positive")
        end = offset + length
        while cu_chunk_seqlens[-1] < end:
            cu_chunk_seqlens.append(min(cu_chunk_seqlens[-1] + chunk_size, end))
            seq_idx.append(sequence_id)
        last_chunk_indices.append(len(seq_idx) - 1)
        offset = end
        cu_seqlens.append(offset)
    make = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
    return tuple(
        make(values)
        for values in (cu_seqlens, cu_chunk_seqlens, last_chunk_indices, seq_idx)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--lengths", help="comma-separated packed sequence lengths")
    parser.add_argument("--nheads", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.musa.set_device(0)
    torch.manual_seed(41)
    lengths = (
        [int(value) for value in args.lengths.split(",")]
        if args.lengths
        else [args.seqlen]
    )
    t, h, p, n, g, c = (
        sum(lengths),
        args.nheads,
        args.headdim,
        args.dstate,
        args.ngroups,
        args.chunk_size,
    )
    device = torch.device("musa")
    dt = torch.randn((t, h), device=device, dtype=torch.float32)
    A = -torch.rand((h,), device=device, dtype=torch.float32) - 1
    bias = torch.randn((h,), device=device, dtype=torch.float32)
    x = torch.randn((t, h, p), device=device, dtype=torch.bfloat16)
    B = torch.randn((t, g, n), device=device, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    _, cu, last, seq_idx = build_metadata(lengths, c, device)
    nchunks = seq_idx.numel()
    initial = torch.randn((len(lengths), h, p, n), device=device, dtype=torch.float16)

    dA, dt_out = _chunk_cumsum_fwd(dt, A, c, cu, dt_bias=bias, dt_softplus=True)
    states4 = _chunk_state_fwd(B, x, dt_out, dA, cu, states_in_fp32=True)
    states3 = states4.reshape(nchunks, h, p * n)
    initial3 = initial.reshape(len(lengths), h, p * n)
    passed = _state_passing_fwd(
        states3, dA, last, initial_states=initial3, out_dtype=torch.float16
    )
    cb = _bmm_chunk_fwd(C, B, c, cu, output_dtype=torch.float32)
    out = torch.empty_like(x)

    calls = {
        "chunk_cumsum": lambda: _chunk_cumsum_fwd(
            dt, A, c, cu, dt_bias=bias, dt_softplus=True
        ),
        "chunk_state": lambda: _chunk_state_fwd(
            B, x, dt_out, dA, cu, states_in_fp32=True
        ),
        "state_passing": lambda: _state_passing_fwd(
            states3, dA, last, initial_states=initial3, out_dtype=torch.float16
        ),
        "bmm_chunk": lambda: _bmm_chunk_fwd(C, B, c, cu, output_dtype=torch.float32),
        "chunk_scan": lambda: _chunk_scan_fwd(
            cb,
            x,
            dt_out,
            dA,
            C,
            passed.reshape(nchunks, h, p, n),
            cu,
            out,
            seq_idx,
            initial_states=initial,
        ),
    }
    result = {
        "shape": {
            "lengths": lengths,
            "T": t,
            "H": h,
            "D": p,
            "N": n,
            "G": g,
            "chunk": c,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "stages": {
            name: time_call(fn, args.warmup, args.repeats) for name, fn in calls.items()
        },
        "finite": bool(torch.isfinite(out).all().item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
