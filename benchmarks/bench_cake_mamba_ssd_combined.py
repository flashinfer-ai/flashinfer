#!/usr/bin/env python3
"""Direct Cake-versus-CuTe SSDCombined parity and CUPTI benchmark."""

import argparse
import json
from importlib.metadata import version

import numpy as np
import torch

from flashinfer.mamba import SSDCombined
from flashinfer.testing.utils import bench_gpu_time


def _require_cupti() -> None:
    try:
        from cupti import cupti  # noqa: F401

        cupti_version = version("cupti-python")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "cupti-python >= 13 is required for this benchmark"
        ) from error
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python >= 13 is required, found {cupti_version}")


def _diagnostic(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    atol = rtol = 1e-2
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    abs_err = (actual_f32 - expected_f32).abs()
    rel_err = abs_err / expected_f32.abs().clamp_min(1e-12)
    mismatches = int((actual != expected).sum().item())
    return {
        "bitwise_equal": mismatches == 0,
        "tolerance_passed": bool(
            torch.isclose(actual_f32, expected_f32, atol=atol, rtol=rtol).all().item()
        ),
        "atol": atol,
        "rtol": rtol,
        "mismatch_count": mismatches,
        "numel": actual.numel(),
        "max_abs": float(abs_err.max().item()),
        "max_rel": float(rel_err.max().item()),
    }


def main() -> None:
    _require_cupti()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batched", "varlen"), default="batched")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--chunks-per-seq", type=int, default=1)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.mode == "batched":
        batch = args.batch
        nchunks = args.nchunks
        num_sequences = batch
        seq_idx = chunk_indices = chunk_offsets = seq_chunk_cumsum = None
    else:
        batch = 1
        nchunks = args.num_seqs * args.chunks_per_seq
        num_sequences = args.num_seqs
        sequence_ids = torch.arange(args.num_seqs, dtype=torch.int32, device="cuda")
        seq_idx = sequence_ids.repeat_interleave(args.chunks_per_seq * 128).reshape(
            1, -1
        )
        chunk_indices = torch.arange(nchunks, dtype=torch.int32, device="cuda")
        chunk_offsets = torch.zeros(nchunks, dtype=torch.int32, device="cuda")
        seq_chunk_cumsum = (
            torch.arange(args.num_seqs + 1, dtype=torch.int32, device="cuda")
            * args.chunks_per_seq
        )
    seqlen = nchunks * 128
    shape = (batch, seqlen, args.nheads)
    x = torch.randn(*shape, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(*shape, device="cuda", dtype=torch.float32)
    A = -torch.rand(args.nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, args.ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(args.nheads, device="cuda").to(torch.bfloat16)
    dt_bias = torch.rand(args.nheads, device="cuda") - 4.0
    initial_states = torch.randn(num_sequences, args.nheads, 64, 128, device="cuda").to(
        torch.bfloat16
    )
    inputs = (x, dt, A, B, C)
    backend_arguments = {"D": D, "z": None}

    constructor = dict(
        chunk_size=128,
        nheads=args.nheads,
        headdim=64,
        dstate=128,
        ngroups=args.ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=True,
        d_has_hdim=False,
        has_initial_states=True,
        has_varlen=args.mode == "varlen",
        has_z=False,
        seq_idx_dtype=torch.int32,
    )
    runners = {
        backend: SSDCombined(**constructor, backend=backend)
        for backend in ("cute", "cake")
    }
    outputs = {}
    timings = {}
    for backend, runner in runners.items():
        out = torch.empty(
            batch,
            args.nheads,
            64,
            nchunks,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )

        def invoke():
            return runner.run(
                *inputs,
                **backend_arguments,
                dt_bias=dt_bias,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                initial_states=initial_states,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                seq_chunk_cumsum=seq_chunk_cumsum,
                out=out,
                return_final_states=True,
            )

        outputs[backend] = invoke()
        samples = bench_gpu_time(
            invoke,
            enable_cupti=True,
            dry_run_iters=args.warmup,
            repeat_iters=args.repetitions,
        )
        timings[backend] = float(np.median(samples))

    report = {
        "shape": {
            "mode": args.mode,
            "batch": batch,
            "num_sequences": num_sequences,
            "chunks_per_sequence": (
                args.chunks_per_seq if args.mode == "varlen" else nchunks
            ),
            "seqlen": seqlen,
            "nheads": args.nheads,
            "ngroups": args.ngroups,
            "headdim": 64,
            "dstate": 128,
            "chunk_size": 128,
            "input_layout": "contiguous",
        },
        "out": _diagnostic(outputs["cake"][0], outputs["cute"][0]),
        "final_states": _diagnostic(outputs["cake"][1], outputs["cute"][1]),
        "flashinfer_ms": timings["cute"],
        "cake_ms": timings["cake"],
        "speedup": timings["cute"] / timings["cake"],
        "timing_backend": "cupti",
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
