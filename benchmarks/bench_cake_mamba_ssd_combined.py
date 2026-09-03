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


def _diagnostic(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
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


def _packed_varlen_metadata(sequence_lengths: list[int], seq_idx_dtype):
    total_seqlen = sum(sequence_lengths)
    if total_seqlen % 128 != 0:
        raise ValueError("packed sequence lengths must sum to a multiple of 128")
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise ValueError("packed sequence lengths must all be positive")

    seq_idx = torch.empty((1, total_seqlen), dtype=seq_idx_dtype, device="cuda")
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


def _fp64_reference(
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    initial_states,
    seq_idx,
    dt_limit,
):
    """Sequential fp64 ground truth for the bench workload.

    The bench's cake-vs-cute diagnostic compares two implementations that
    share one computation graph, so near-bitwise parity there is expected
    and does not by itself prove accuracy.  This independent reference
    evaluates the plain per-token recurrence in float64:

        delta    = clamp(softplus(dt + dt_bias), dt_lo, dt_hi)
        state_t  = state_{t-1} * exp(delta_t * A) + delta_t * x_t x B_t
        y_t      = (state_t . C_t).sum(dstate) + D * x_t
        y_t     *= z_t * sigmoid(z_t)            (when z is given)

    with state reset to ``initial_states[s]`` at each sequence boundary for
    packed varlen layouts (``seq_idx is not None``).
    """
    dt_lo, dt_hi = dt_limit
    xf = x.double()
    delta = torch.nn.functional.softplus(dt.double() + dt_bias.double())
    delta = delta.clamp(min=dt_lo, max=dt_hi)
    Af = A.double()
    Bf = B.double()
    Cf = C.double()
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    hpg = nheads // ngroups

    if seq_idx is not None:
        seq_of_token = seq_idx.reshape(-1).long()
    else:
        seq_of_token = (
            torch.arange(batch, device=x.device)
            .view(batch, 1)
            .expand(batch, seqlen)
            .reshape(-1)
        )
    xf = xf.reshape(batch * seqlen, nheads, headdim)
    delta = delta.reshape(batch * seqlen, nheads)
    Bf = Bf.reshape(batch * seqlen, ngroups, -1)
    Cf = Cf.reshape(batch * seqlen, ngroups, -1)

    init_flat = initial_states.double().reshape(-1, nheads, headdim, B.shape[-1])
    dstate = B.shape[-1]
    final = torch.zeros_like(init_flat)
    y_ref = torch.zeros(
        batch * seqlen, nheads, headdim, dtype=torch.float64, device=x.device
    )
    prev_seq = -1
    cur = None  # group-major [ngroups, hpg, headdim, dstate]
    for t in range(batch * seqlen):
        s = int(seq_of_token[t].item())
        if s != prev_seq:
            if prev_seq >= 0:
                final[prev_seq] = cur.reshape(nheads, headdim, dstate)
            cur = init_flat[s].view(ngroups, hpg, headdim, dstate).clone()
            prev_seq = s
        dt_t = delta[t]  # [H]
        cur = cur * torch.exp(dt_t * Af).view(ngroups, hpg, 1, 1)
        cur = cur + (
            dt_t.view(ngroups, hpg, 1, 1) * xf[t].view(ngroups, hpg, headdim, 1)
        ) * Bf[t].view(ngroups, 1, 1, dstate)
        y_ref[t] = (
            (cur * Cf[t].view(ngroups, 1, 1, dstate)).sum(-1).reshape(nheads, headdim)
        )
    final[prev_seq] = cur.reshape(nheads, headdim, dstate)
    if D.ndim == 2:
        y_ref = y_ref + D.double().view(1, nheads, headdim) * xf
    else:
        y_ref = y_ref + D.double().view(1, nheads, 1) * xf
    if z is not None:
        zf = z.double().reshape(batch * seqlen, nheads, headdim)
        y_ref = y_ref * (zf * torch.sigmoid(zf))
    return y_ref.view(batch, seqlen, nheads, headdim), final


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-qualified-row", action="store_true")
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument(
        "--state-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--seq-idx-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument(
        "--preprocess-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--d-has-hdim", action="store_true")
    parser.add_argument("--unbounded-dt", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--vibecuda",
        action="store_true",
        help="also benchmark the vibecuda backend and report its speedup "
        "against the cake baseline",
    )
    parser.add_argument(
        "--no-truth-check",
        action="store_true",
        help="skip the independent fp64 sequential-reference diagnostics "
        "(enabled by default whenever --vibecuda is requested, so both the "
        "cake denominator and the candidate are validated against ground "
        "truth instead of only against each other's rounding)",
    )
    parser.add_argument(
        "--reference",
        choices=("cute", "cake", "none"),
        default="cute",
        help="parity reference backend instantiated for the report; 'none' "
        "skips parity diagnostics entirely (cake stays the speedup "
        "denominator either way)",
    )
    return parser


def run_workload(args) -> dict:
    """Run one workload row and return the report dict.

    Candidate-side gates (vibecuda fp64 truth, NaN-sentinel full write) raise
    ``AssertionError`` here; the cake-vs-cute self-gate stays in :func:`main`
    so an in-process matrix driver can treat it as informational.
    """
    torch.manual_seed(args.seed)
    state_dtype = getattr(torch, args.state_dtype)
    seq_idx_dtype = getattr(torch, args.seq_idx_dtype)
    preprocess_dtype = getattr(torch, args.preprocess_dtype)
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
            _packed_varlen_metadata(sequence_lengths, seq_idx_dtype)
        )
    seqlen = nchunks * 128
    shape = (batch, seqlen, args.nheads)
    x = torch.randn(*shape, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(*shape, device="cuda", dtype=preprocess_dtype)
    A = -torch.rand(args.nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, args.ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    d_shape = (args.nheads, 64) if args.d_has_hdim else (args.nheads,)
    D = torch.randn(*d_shape, device="cuda").to(torch.bfloat16)
    z = torch.randn_like(x) if args.has_z else None
    dt_bias = (torch.rand(args.nheads, device="cuda") - 4.0).to(preprocess_dtype)
    initial_states = torch.randn(num_sequences, args.nheads, 64, 128, device="cuda").to(
        state_dtype
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
        state_dtype=state_dtype,
        has_d=True,
        d_has_hdim=args.d_has_hdim,
        has_initial_states=True,
        has_varlen=args.mode == "varlen",
        has_z=args.has_z,
        seq_idx_dtype=seq_idx_dtype,
    )
    dt_limit = (0.0, float("inf")) if args.unbounded_dt else (0.001, 0.1)
    backends = ["cake"]
    if args.reference == "cute":
        backends.insert(0, "cute")
    if args.vibecuda:
        backends.append("vibecuda")
    runners = {
        backend: SSDCombined(**constructor, backend=backend) for backend in backends
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
                dt_limit=dt_limit,
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
            "state_dtype": args.state_dtype,
            "seq_idx_dtype": args.seq_idx_dtype,
            "preprocess_dtype": args.preprocess_dtype,
            "d_has_hdim": args.d_has_hdim,
            "dt_limit": "unbounded" if args.unbounded_dt else [0.001, 0.1],
        },
        "cake_ms": timings["cake"],
        "timing_backend": "cupti",
    }
    if args.reference == "cute":
        report["out"] = _diagnostic(outputs["cake"][0], outputs["cute"][0])
        report["final_states"] = _diagnostic(outputs["cake"][1], outputs["cute"][1])
        report["flashinfer_ms"] = timings["cute"]
        report["speedup"] = timings["cute"] / timings["cake"]
    if args.vibecuda:
        # Primary comparison for the vibecuda port: the nailed-down Cake
        # baseline stays the speedup denominator.
        report["vibecuda_ms"] = timings["vibecuda"]
        report["vibecuda_speedup_vs_cake"] = timings["cake"] / timings["vibecuda"]
        if args.reference != "none":
            parity = outputs[args.reference]
            report["vibecuda_out"] = _diagnostic(outputs["vibecuda"][0], parity[0])
            report["vibecuda_final_states"] = _diagnostic(
                outputs["vibecuda"][1], parity[1]
            )
    if args.vibecuda:
        # Independent full-write proof on the caller-owned ``out``: prefill a
        # fresh buffer with a NaN sentinel, run the vibecuda backend once
        # (untimed), and require every element to be overwritten.  Mirrors the
        # NaN-sentinel test in tests/mamba/test_vibecuda_ssd_combined.py and
        # guards against a kernel silently writing only part of the output.
        out_sentinel = torch.full(
            (batch, args.nheads, 64, nchunks, 128),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        )
        runners["vibecuda"].run(
            *inputs,
            **backend_arguments,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=dt_limit,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            seq_chunk_cumsum=seq_chunk_cumsum,
            out=out_sentinel,
            return_final_states=True,
        )
        unwritten = int(torch.isnan(out_sentinel).sum().item())
        report["full_write"] = {
            "sentinel": "NaN",
            "out_numel": int(out_sentinel.numel()),
            "unwritten_elements": unwritten,
            "fully_written": unwritten == 0,
        }
        if unwritten:
            raise AssertionError(
                "vibecuda left "
                f"{unwritten}/{out_sentinel.numel()} NaN-sentinel elements "
                "unwritten in the caller-owned out tensor"
            )
    if args.vibecuda and not args.no_truth_check:
        # Independent accuracy proof: validate BOTH the cake denominator and
        # the vibecuda candidate against an fp64 sequential reference.  The
        # cake-vs-cute diagnostic alone cannot show this, because cake and
        # cute share one computation graph and agree near-bitwise regardless
        # of their joint distance to the true result.
        y_ref, fs_ref = _fp64_reference(
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            initial_states,
            seq_idx,
            dt_limit=dt_limit,
        )
        report["truth_reference"] = "fp64_sequential_recurrence"
        report["cake_truth_out"] = _diagnostic(
            outputs["cake"][0], y_ref, atol=6e-2, rtol=6e-2
        )
        report["cake_truth_final_states"] = _diagnostic(
            outputs["cake"][1], fs_ref, atol=6e-2, rtol=6e-2
        )
        report["vibecuda_truth_out"] = _diagnostic(
            outputs["vibecuda"][0], y_ref, atol=5.9e-2, rtol=5.9e-2
        )
        report["vibecuda_truth_final_states"] = _diagnostic(
            outputs["vibecuda"][1], fs_ref, atol=5.9e-2, rtol=5.9e-2
        )
        report["candidate_no_worse_than_cake"] = {
            "baseline_contract": "allclose:6e-2,6e-2",
            "candidate_contract": "allclose:5.9e-2,5.9e-2",
            "out": report["vibecuda_truth_out"]["tolerance_passed"],
            "final_states": report["vibecuda_truth_final_states"]["tolerance_passed"],
        }
        # The candidate must independently beat the fixed fast-baseline
        # allclose:6e-2,6e-2 contract against ground truth. Cake's own truth
        # diagnostics are reported (not asserted):
        # cake and cute are bitwise-matched twins whose joint rounding can
        # exceed 1e-2 vs fp64 at these unbounded-dt magnitudes.
        if not (
            report["vibecuda_truth_out"]["tolerance_passed"]
            and report["vibecuda_truth_final_states"]["tolerance_passed"]
            and report["candidate_no_worse_than_cake"]["out"]
            and report["candidate_no_worse_than_cake"]["final_states"]
        ):
            raise AssertionError(
                "vibecuda failed fp64 ground-truth validation: "
                f"out={report['vibecuda_truth_out']['tolerance_passed']} "
                f"final_states={report['vibecuda_truth_final_states']['tolerance_passed']}"
            )
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_workload(args)
    print(json.dumps(report, sort_keys=True))
    if args.reference == "cute":
        _validate_report(report, require_qualified_row=args.require_qualified_row)


if __name__ == "__main__":
    main()
