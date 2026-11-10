#!/usr/bin/env python3
"""Cake-versus-CuTe SSDCombined accuracy and CUPTI benchmark.

Both backends are measured against an fp64 token-by-token recurrence: the Cake
programs carry the per-token delta in FP16 and CuTe in BF16, so their sparse
outside-tolerance outliers no longer coincide and elementwise Cake-vs-CuTe
parity is not a correctness oracle (CuTe itself is outside atol=rtol=1e-2 of
the recurrence on 0.04-1.4 % of its outputs).  A row is valid when Cake's
output is finite, at most 1 % of its entries are outside 1e-2 of the
recurrence, and Cake has no more outliers and no larger maximum error than
CuTe on the same inputs.  The Cake-vs-CuTe statistics are still reported.
"""

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


def _fp64_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor,
    initial_states: torch.Tensor | None,
    sequence_lengths: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 token-by-token SSM recurrence on the unpadded token-major inputs.

    ``dt' = clamp(softplus(dt + dt_bias), 0, inf)``;
    ``state = exp(dt' * A) * state + dt' * (x (x) B)``; ``y = C . state + D * x``;
    ``y *= z * sigmoid(z)`` when ``z`` is given.  Sequences advance in lockstep
    (one iteration per token position, masked by each sequence's length), so a
    batched call or a packed varlen stream costs ``max(length)`` iterations.
    Returns the fp64 output with ``x``'s shape and the ``[num_seqs, nheads,
    headdim, dstate]`` final states (zero initial state when ``initial_states``
    is ``None``).
    """

    f64 = torch.float64
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2], B.shape[3]
    rep = nheads // ngroups
    lengths = (
        list(sequence_lengths) if sequence_lengths is not None else [seqlen] * batch
    )
    assert sum(lengths) == batch * seqlen, (lengths, batch, seqlen)
    starts = [sum(lengths[:index]) for index in range(len(lengths))]
    xf = x.reshape(-1, nheads, headdim).to(f64)
    dtp = dt.reshape(-1, nheads).to(f64) + dt_bias.to(f64)
    dtp = torch.nn.functional.softplus(dtp).clamp_min(0.0)
    Bf = B.reshape(-1, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    Cf = C.reshape(-1, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    Af = A.to(f64)
    Df = D.to(f64).reshape(nheads, 1) if D.dim() == 1 else D.to(f64)
    zf = z.reshape(-1, nheads, headdim).to(f64) if z is not None else None
    num_seqs = len(lengths)
    state = (
        initial_states[:num_seqs].to(f64).clone()
        if initial_states is not None
        else torch.zeros(num_seqs, nheads, headdim, dstate, dtype=f64, device=x.device)
    )
    out = torch.empty_like(xf)
    start_index = torch.tensor(starts, device=x.device)
    length_index = torch.tensor(lengths, device=x.device)
    for position in range(max(lengths)):
        active = (length_index > position).nonzero(as_tuple=True)[0]
        tokens = start_index[active] + position
        step = dtp[tokens]  # [n, H]
        decay = torch.exp(step * Af)
        update = step[:, :, None, None] * (
            xf[tokens][:, :, :, None] * Bf[tokens][:, :, None, :]
        )
        state[active] = decay[:, :, None, None] * state[active] + update
        y = torch.einsum("nhpd,nhd->nhp", state[active], Cf[tokens]) + Df * xf[tokens]
        if zf is not None:
            zt = zf[tokens]
            y = y * (zt * torch.sigmoid(zt))
        out[tokens] = y
    return out.reshape(x.shape), state


def _accuracy(cake: torch.Tensor, cute: torch.Tensor, reference: torch.Tensor) -> dict:
    """Outside-tolerance counts and maximum errors of both backends vs the fp64 oracle."""

    atol = rtol = 1e-2
    ref = reference.float()
    tolerance = atol + rtol * ref.abs()

    def stats(value: torch.Tensor) -> tuple[int, float, float, bool]:
        value_f32 = value.float()
        error = (value_f32 - ref).abs()
        worst = int(error.argmax().item())
        return (
            int((error > tolerance).sum().item()),
            float(error.max().item()),
            float(ref.flatten()[worst].abs().item()),
            bool(torch.isfinite(value_f32).all().item()),
        )

    cake_outside, cake_max, cake_worst_ref, cake_finite = stats(cake)
    cute_outside, cute_max, _cute_worst_ref, cute_finite = stats(cute)
    # one bf16 ulp at the magnitude of Cake's worst entry (bf16 keeps 8 significand bits)
    exponent = int(np.floor(np.log2(cake_worst_ref))) if cake_worst_ref > 0 else -126
    return {
        "atol": atol,
        "rtol": rtol,
        "numel": int(ref.numel()),
        "cake_outside": cake_outside,
        "cute_outside": cute_outside,
        "cake_outside_fraction": cake_outside / max(1, ref.numel()),
        "cute_outside_fraction": cute_outside / max(1, ref.numel()),
        "cake_max_abs": cake_max,
        "cute_max_abs": cute_max,
        "cake_finite": cake_finite,
        "cute_finite": cute_finite,
        "bf16_ulp_at_cake_worst": float(2.0 ** (exponent - 7)),
    }


def _validate_report(report: dict, *, require_qualified_row: bool) -> None:
    for name in ("out", "final_states"):
        accuracy = report["accuracy"][name]
        if not accuracy["cake_finite"]:
            raise AssertionError(f"Cake {name} is not finite")
        if accuracy["cake_outside_fraction"] > 0.01:
            raise AssertionError(
                f"Cake {name}: {accuracy['cake_outside']} of {accuracy['numel']} entries "
                "outside atol=rtol=1e-2 of the fp64 recurrence (limit 1 %)"
            )
        # Poisson slack on CuTe's count: the two outlier sets are independent samples.
        allowed = accuracy["cute_outside"] + 2.0 * np.sqrt(accuracy["cute_outside"])
        if accuracy["cake_outside"] > allowed:
            raise AssertionError(
                f"Cake {name} has more entries outside 1e-2 of the fp64 recurrence than CuTe: "
                f"{accuracy['cake_outside']} vs {accuracy['cute_outside']}"
            )
        if (
            accuracy["cake_max_abs"]
            > accuracy["cute_max_abs"] + accuracy["bf16_ulp_at_cake_worst"]
        ):
            raise AssertionError(
                f"Cake {name} max abs error {accuracy['cake_max_abs']:.4g} exceeds CuTe's "
                f"{accuracy['cute_max_abs']:.4g} by more than one bf16 ulp"
            )
    if require_qualified_row and report["speedup"] <= 1.0:
        raise AssertionError("Cake must be faster than CuTe for a reported row")


def _packed_varlen_metadata(sequence_lengths: list[int]):
    """Packed-varlen metadata; the stream may end inside a physical chunk."""

    total_seqlen = sum(sequence_lengths)
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
    for chunk in range(-(-total_seqlen // 128)):
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


def _pad_stream(value: torch.Tensor, pad: int) -> torch.Tensor:
    """Append ``pad`` zero tokens to a packed ``[1, T, ...]`` stream."""

    if pad == 0:
        return value
    padding = torch.zeros(
        (value.shape[0], pad, *value.shape[2:]), dtype=value.dtype, device=value.device
    )
    return torch.cat((value, padding), dim=1).contiguous()


def main() -> None:
    _require_cupti()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batched", "varlen"), default="batched")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument(
        "--seqlen",
        type=int,
        help="batched sequence length (any positive value; default nchunks * 128)",
    )
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--chunks-per-seq", type=int, default=1)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        help="packed varlen lengths (any positive values); overrides "
        "--num-seqs/--chunks-per-seq",
    )
    parser.add_argument("--zero-initial-states", action="store_true")
    parser.add_argument(
        "--no-initial-states",
        action="store_true",
        help="run Cake with initial_states=None (zero state); the CuTe arm gets "
        "explicit zero states since it requires them in varlen mode",
    )
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
        seqlen = args.seqlen if args.seqlen is not None else args.nchunks * 128
        num_sequences = batch
        sequence_lengths = None
    else:
        if args.seqlen is not None:
            raise ValueError("--seqlen requires --mode batched")
        batch = 1
        sequence_lengths = (
            args.sequence_lengths or [args.chunks_per_seq * 128] * args.num_seqs
        )
        num_sequences = len(sequence_lengths)
        seqlen = sum(sequence_lengths)
    if seqlen <= 0:
        raise ValueError("the sequence length must be positive")
    nchunks = -(-seqlen // 128)
    # The CuTe backend needs seqlen % 128 == 0: pad its stream with zero tokens
    # (varlen: one extra packed sequence) and compare the real-token slice.
    cute_pad = nchunks * 128 - seqlen
    if cute_pad and args.mode == "batched" and batch != 1:
        raise ValueError("an unaligned batched --seqlen requires --batch 1")
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
    if args.zero_initial_states or args.no_initial_states:
        initial_states.zero_()

    arms = {}
    for backend in ("cute", "cake"):
        pad = cute_pad if backend == "cute" else 0
        lengths = sequence_lengths
        if pad and args.mode == "varlen":
            lengths = [*sequence_lengths, pad]
        elif pad:
            # Batched 1 x seqlen: CuTe runs the padded stream as packed varlen
            # [seqlen, pad] so the padding never reaches the real sequence.
            lengths = [seqlen, pad]
        varlen = args.mode == "varlen" or (pad > 0)
        states = initial_states
        if pad:
            states = torch.cat((states, torch.zeros_like(states[:1])), dim=0)
        # CuTe needs explicit states in varlen mode; zero states are the
        # semantic equivalent of Cake's initial_states=None.
        has_initial_states = not args.no_initial_states or (
            backend == "cute" and varlen
        )
        if varlen:
            seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = (
                _packed_varlen_metadata(lengths)
            )
        else:
            seq_idx = chunk_indices = chunk_offsets = seq_chunk_cumsum = None
        tensors = tuple(_pad_stream(value, pad) for value in (x, dt, B, C))
        arms[backend] = dict(
            pad=pad,
            varlen=varlen,
            has_initial_states=has_initial_states,
            tensors=(tensors[0], tensors[1], A, tensors[2], tensors[3]),
            arguments=dict(
                D=D,
                z=_pad_stream(z, pad) if z is not None else None,
                dt_bias=dt_bias,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                initial_states=states if has_initial_states else None,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                seq_chunk_cumsum=seq_chunk_cumsum,
                return_final_states=True,
                num_seqs=None if has_initial_states or not varlen else num_sequences,
            ),
        )

    outputs = {}
    timings = {}
    for backend, arm in arms.items():
        runner = SSDCombined(
            chunk_size=128,
            nheads=args.nheads,
            headdim=64,
            dstate=128,
            ngroups=args.ngroups,
            io_dtype=torch.bfloat16,
            state_dtype=torch.bfloat16,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=arm["has_initial_states"],
            has_varlen=arm["varlen"],
            has_z=args.has_z,
            seq_idx_dtype=torch.int32,
            backend=backend,
        )
        padded_seqlen = seqlen + arm["pad"]
        # Each backend's kernel output layout: CuTe chunked, Cake token-major.
        out = (
            torch.empty(
                batch, seqlen, args.nheads, 64, dtype=torch.bfloat16, device="cuda"
            )
            if backend == "cake"
            else torch.empty(
                batch,
                args.nheads,
                64,
                padded_seqlen // 128,
                128,
                dtype=torch.bfloat16,
                device="cuda",
            )
        )

        def invoke(runner=runner, arm=arm, out=out):
            return runner.run(*arm["tensors"], **arm["arguments"], out=out)

        result_out, result_states = invoke()
        outputs[backend] = (
            result_out[:, :seqlen],
            result_states[:num_sequences],
        )
        samples = bench_gpu_time(
            invoke,
            enable_cupti=True,
            dry_run_iters=args.warmup,
            repeat_iters=args.repetitions,
        )
        timings[backend] = float(np.median(samples))

    reference_out, reference_states = _fp64_reference(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        None if args.no_initial_states else initial_states,
        sequence_lengths,
    )

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
            "initial_states": (
                "none"
                if args.no_initial_states
                else "zero"
                if args.zero_initial_states
                else "random"
            ),
            "has_z": args.has_z,
            "seed": args.seed,
            "cute_padded_tokens": cute_pad,
        },
        "out": _diagnostic(outputs["cake"][0], outputs["cute"][0]),
        "final_states": _diagnostic(outputs["cake"][1], outputs["cute"][1]),
        "accuracy": {
            "out": _accuracy(outputs["cake"][0], outputs["cute"][0], reference_out),
            "final_states": _accuracy(
                outputs["cake"][1], outputs["cute"][1], reference_states
            ),
        },
        "flashinfer_ms": timings["cute"],
        "cake_ms": timings["cake"],
        "speedup": timings["cute"] / timings["cake"],
        "timing_backend": "cupti",
    }
    print(json.dumps(report, sort_keys=True))
    _validate_report(report, require_qualified_row=args.require_qualified_row)


if __name__ == "__main__":
    main()
