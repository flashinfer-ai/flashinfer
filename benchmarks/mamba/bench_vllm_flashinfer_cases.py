#!/usr/bin/env python3
"""Small production-shaped FlashInfer Mamba2/SSD benchmarks.

These are intentionally smaller than a vLLM server run but preserve the
Nemotron tensor layouts, dtypes, slots, grouped B/C, dt_softplus and chunk
metadata. They are used to screen StormEye candidates before serving A/B.
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import torch
from flashinfer.mamba import (
    selective_state_update,
    ssd_combined_fwd,
    ssd_combined_fwd_varlen,
)
from flashinfer.mamba.musa_reference import (
    ssd_combined_fwd_musa_reference,
    ssd_combined_fwd_varlen_musa_reference,
)

H, D, N, G, CHUNK = 64, 64, 128, 8, 128


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(iters):
        s = torch.musa.Event(enable_timing=True)
        e = torch.musa.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        vals.append(s.elapsed_time(e))
    return {
        "median_ms": statistics.median(vals),
        "p90_ms": sorted(vals)[max(0, int(0.9 * len(vals)) - 1)],
        "samples_ms": vals,
    }


def _graph_time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        graph_result = fn()
    graph.replay()
    torch.musa.synchronize()
    vals = []
    for _ in range(iters):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(vals),
        "p90_ms": sorted(vals)[max(0, int(0.9 * len(vals)) - 1)],
        "samples_ms": vals,
    }, graph_result


def ssu_case(native: bool, warmup: int, iters: int, graph: bool = False):
    if native:
        os.environ["FLASHINFER_MUSA_SIMPLE_STP_NATIVE"] = "1"
    else:
        os.environ.pop("FLASHINFER_MUSA_SIMPLE_STP_NATIVE", None)
    torch.manual_seed(17)
    state = torch.randn((2, H, D, N), device="musa", dtype=torch.float16)
    x = torch.randn((1, H, D), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, H, 1), device="musa", dtype=torch.float32).expand(1, H, D)
    A = (-torch.rand((H, 1, 1), device="musa", dtype=torch.float32) - 1).expand(H, D, N)
    B = torch.randn((1, G, N), device="musa", dtype=torch.bfloat16)
    C = torch.randn_like(B)
    Dv = torch.randn((H, 1), device="musa", dtype=torch.float32).expand(H, D)
    slot = torch.tensor([0], device="musa", dtype=torch.int32)
    out = torch.empty_like(x)
    fn = lambda: selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        Dv,
        dt_softplus=True,
        state_batch_indices=slot,
        out=out,
        backend="flashinfer",
    )
    timing = _graph_time(fn, warmup, iters)[0] if graph else _time(fn, warmup, iters)
    return {
        "case": "ssu_decode",
        "backend": "native" if native else "triton",
        "graph": graph,
        "shape": {"B": 1, "H": H, "D": D, "N": N, "G": G},
        "timing": timing,
        "output_shape": list(out.shape),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def ssd_case(warmup: int, iters: int, seqlen: int, graph: bool = False):
    torch.manual_seed(19)
    batch = 1
    x = torch.randn((batch, seqlen, H, D), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((batch, seqlen, H), device="musa", dtype=torch.float32)
    A = -torch.rand((H,), device="musa", dtype=torch.float32) - 1
    B = torch.randn((batch, seqlen, G, N), device="musa", dtype=torch.bfloat16)
    C = torch.randn_like(B)
    Dv = torch.randn((H,), device="musa", dtype=torch.bfloat16)
    bias = torch.rand((H,), device="musa", dtype=torch.float32) - 4
    init = torch.randn((batch, H, D, N), device="musa", dtype=torch.float16)
    fn = lambda: ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        D=Dv,
        dt_bias=bias,
        dt_softplus=True,
        initial_states=init,
        return_final_states=True,
    )
    timing = _graph_time(fn, warmup, iters)[0] if graph else _time(fn, warmup, iters)
    out, final = fn()
    torch.musa.synchronize()
    ref, ref_final = ssd_combined_fwd_musa_reference(
        x,
        dt,
        A,
        B,
        C,
        D=Dv,
        dt_bias=bias,
        dt_softplus=True,
        initial_states=init,
        return_final_states=True,
    )
    err = float((out.float() - ref.float()).abs().max().item())
    return {
        "case": "ssd_prefill",
        "graph": graph,
        "shape": {
            "B": batch,
            "T": seqlen,
            "H": H,
            "D": D,
            "N": N,
            "G": G,
            "chunk": CHUNK,
        },
        "timing": timing,
        "output_shape": list(out.shape),
        "max_abs_err": err,
        "finite": bool(
            torch.isfinite(out).all().item() and torch.isfinite(final).all().item()
        ),
    }


def _build_varlen_metadata(lengths, chunk_size, device):
    """Build the packed metadata generated by vLLM's Mamba2 prefill path."""
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


def ssd_varlen_case(warmup: int, iters: int, lengths, graph: bool = False):
    torch.manual_seed(23)
    device = torch.device("musa")
    tokens = sum(lengths)
    x = torch.randn((tokens, H, D), device=device, dtype=torch.bfloat16)
    dt = torch.randn((tokens, H), device=device, dtype=torch.float32)
    A = -torch.rand((H,), device=device, dtype=torch.float32) - 1
    B = torch.randn((tokens, G, N), device=device, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    Dv = torch.randn((H,), device=device, dtype=torch.float32)
    bias = torch.rand((H,), device=device, dtype=torch.float32) - 4
    initial = torch.randn((len(lengths), H, D, N), device=device, dtype=torch.float16)
    metadata = _build_varlen_metadata(lengths, CHUNK, device)
    out = torch.empty_like(x)

    def fn():
        return ssd_combined_fwd_varlen(
            x,
            dt,
            A,
            B,
            C,
            CHUNK,
            *metadata,
            out,
            D=Dv,
            dt_bias=bias,
            dt_softplus=True,
            initial_states=initial,
            state_dtype=torch.float16,
        )

    timing = _graph_time(fn, warmup, iters)[0] if graph else _time(fn, warmup, iters)
    final_states = fn()
    reference_out = torch.empty_like(x)
    reference_states = ssd_combined_fwd_varlen_musa_reference(
        x,
        dt,
        A,
        B,
        C,
        CHUNK,
        *metadata,
        reference_out,
        D=Dv,
        dt_bias=bias,
        dt_softplus=True,
        initial_states=initial,
        state_dtype=torch.float16,
    )
    torch.musa.synchronize()
    return {
        "case": "ssd_varlen_prefill",
        "graph": graph,
        "shape": {
            "lengths": lengths,
            "T": tokens,
            "H": H,
            "D": D,
            "N": N,
            "G": G,
            "chunk": CHUNK,
        },
        "timing": timing,
        "output_max_abs_err": float(
            (out.float() - reference_out.float()).abs().max().item()
        ),
        "state_max_abs_err": float(
            (final_states.float() - reference_states.float()).abs().max().item()
        ),
        "finite": bool(
            torch.isfinite(out).all().item()
            and torch.isfinite(final_states).all().item()
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        choices=["ssu-native", "ssu-triton", "ssd", "ssd-varlen", "all"],
        default="all",
    )
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--lengths", default="2048,2048")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.musa.set_device(0)
    result = {"device": "musa", "cases": []}
    if args.case in ("ssu-native", "all"):
        result["cases"].append(ssu_case(True, args.warmup, args.iters, args.graph))
    if args.case in ("ssu-triton", "all"):
        result["cases"].append(ssu_case(False, args.warmup, args.iters, args.graph))
    if args.case in ("ssd", "all"):
        result["cases"].append(
            ssd_case(args.warmup, args.iters, args.seqlen, args.graph)
        )
    if args.case in ("ssd-varlen", "all"):
        result["cases"].append(
            ssd_varlen_case(
                args.warmup,
                args.iters,
                [int(value) for value in args.lengths.split(",")],
                args.graph,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
