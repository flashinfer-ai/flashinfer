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


def _validate_report(report: dict, *, require_qualified_row: bool) -> None:
    if not report["out"]["tolerance_passed"]:
        raise AssertionError("Cake output failed BF16 parity")
    if not report["final_states"]["tolerance_passed"]:
        raise AssertionError("Cake final state failed BF16 parity")
    if require_qualified_row and report["speedup"] <= 1.0:
        raise AssertionError("Cake must be faster than CuTe for a reported row")


def _packed_varlen_metadata(sequence_lengths: list[int]):
    total_seqlen = sum(sequence_lengths)
    if total_seqlen % 128 != 0:
        raise ValueError("packed sequence lengths must sum to a multiple of 128")
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise ValueError("packed sequence lengths must all be positive")

    seq_idx = torch.empty((1, total_seqlen), dtype=torch.int32, device="cuda")
    seq_chunk_cumsum = [0]
    start = 0
    for sequence, length in enumerate(sequence_lengths):
        end = start + length
        seq_idx[0, start:end] = sequence
        segments = (end + 127) // 128 - start // 128
        seq_chunk_cumsum.append(seq_chunk_cumsum[-1] + segments)
        start = end

    chunk_indices = []
    chunk_offsets = []
    for chunk in range(total_seqlen // 128):
        values = seq_idx[0, chunk * 128 : (chunk + 1) * 128]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)

    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda"),
        torch.tensor(seq_chunk_cumsum, dtype=torch.int32, device="cuda"),
    )


def main() -> None:
    _require_cupti()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batched", "varlen"), default="batched")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--chunks-per-seq", type=int, default=1)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        help="packed varlen lengths; overrides --num-seqs/--chunks-per-seq",
    )
    parser.add_argument("--zero-initial-states", action="store_true")
    parser.add_argument("--has-z", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-qualified-row", action="store_true")
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.mode == "batched":
        if args.sequence_lengths is not None:
            raise ValueError("--sequence-lengths requires --mode varlen")
        batch = args.batch
        nchunks = args.nchunks
        num_sequences = batch
        sequence_lengths = None
        seq_idx = chunk_indices = chunk_offsets = seq_chunk_cumsum = None
    else:
        batch = 1
        sequence_lengths = (
            args.sequence_lengths or [args.chunks_per_seq * 128] * args.num_seqs
        )
        num_sequences = len(sequence_lengths)
        nchunks = sum(sequence_lengths) // 128
        seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = (
            _packed_varlen_metadata(sequence_lengths)
        )
    seqlen = nchunks * 128
    shape = (batch, seqlen, args.nheads)
    x = torch.randn(*shape, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(*shape, device="cuda", dtype=torch.float32)
    A = -torch.rand(args.nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, args.ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(args.nheads, device="cuda").to(torch.bfloat16)
    z = torch.randn_like(x) if args.has_z else None
    dt_bias = torch.rand(args.nheads, device="cuda") - 4.0
    initial_states = torch.randn(num_sequences, args.nheads, 64, 128, device="cuda").to(
        torch.bfloat16
    )
    if args.zero_initial_states:
        initial_states.zero_()
    inputs = (x, dt, A, B, C)
    backend_arguments = {"D": D, "z": z}

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
        has_z=args.has_z,
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
            "sequence_lengths": sequence_lengths,
            "initial_states": "zero" if args.zero_initial_states else "random",
            "has_z": args.has_z,
            "seed": args.seed,
        },
        "out": _diagnostic(outputs["cake"][0], outputs["cute"][0]),
        "final_states": _diagnostic(outputs["cake"][1], outputs["cute"][1]),
        "flashinfer_ms": timings["cute"],
        "cake_ms": timings["cake"],
        "speedup": timings["cute"] / timings["cake"],
        "timing_backend": "cupti",
    }
    print(json.dumps(report, sort_keys=True))
    _validate_report(report, require_qualified_row=args.require_qualified_row)


if __name__ == "__main__":
    main()
