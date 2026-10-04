"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Accuracy probe: Cake SSDCombined vs CuTe vs an fp64 sequential reference.

Runs ``SSDCombined(backend="cake")`` and ``SSDCombined(backend="cute")`` on
identical inputs for the configurations whose CuTe-parity tests failed after
the FP16-delta change (CAKE-956 D1) and measures each backend against an
independent fp64 token-by-token recurrence (``fp64_reference`` below).  Per
case it reports the number of outputs / final-state entries outside
``atol = rtol = 1e-2`` of the reference for Cake, for CuTe and between the two
backends, the maxima, and where the Cake outliers sit (token / head
histograms), so an argument-binding or metadata defect (dense or structured
errors: whole rows, whole heads, one sequence, one batch) can be told apart
from the sparse rounding-amplification class.  Two emulated references
(``delta`` rounded to fp16 / bf16 before the ``delta * (x (x) B)`` term, the
decay kept exact) isolate the delta-rounding difference between the kernels.

Usage (GPU; FlashInfer repo root on ``PYTHONPATH``)::

    python tests/mamba/cake_ssd_accuracy_probe.py --out probe.json [--only NAME ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from flashinfer.mamba import SSDCombined


def _load_test_module():
    path = Path(__file__).resolve().with_name("test_cake_ssd_combined.py")
    spec = importlib.util.spec_from_file_location("_cake_ssd_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fp64_reference(tensors, arguments, lengths, *, delta_dtype=None):
    """fp64 token-by-token SSM recurrence.

    ``dt' = clamp(softplus(dt + dt_bias), dt_limit)`` (softplus only when
    requested); ``state = exp(dt' * A) * state + delta * (x (x) B)``;
    ``y = C . state + D * x``; ``y *= z * sigmoid(z)`` when ``z`` is given.
    ``delta`` is ``dt'`` rounded to ``delta_dtype`` (``None`` = exact) so the
    kernels' FP16 (Cake) / BF16 (CuTe) ``delta`` rounding can be emulated; the
    decay always uses the exact ``dt'`` (both kernels scan the fp32 ``dt * A``).
    Batched ``[B, S]`` inputs are treated as ``B`` sequences of ``S`` tokens;
    packed varlen ``[1, T]`` inputs follow ``lengths``.  Returns the token-major
    fp64 output with ``x``'s shape and the ``[num_seqs, H, 64, 128]`` final
    states.
    """

    x, dt, A, B, C = tensors
    batch, seqlen, nheads, headdim = x.shape
    total = batch * seqlen
    assert sum(lengths) == total, (lengths, total)
    ngroups, dstate = B.shape[2], B.shape[3]
    rep = nheads // ngroups
    f64 = torch.float64
    xf = x.reshape(total, nheads, headdim).to(f64)
    dtf = dt.reshape(total, nheads).to(f64)
    dt_bias = arguments.get("dt_bias")
    if dt_bias is not None:
        dtf = dtf + dt_bias.to(f64)
    if arguments.get("dt_softplus", False):
        dtf = torch.nn.functional.softplus(dtf)
    dt_min, dt_max = arguments.get("dt_limit", (0.0, float("inf")))
    dtf = dtf.clamp(min=float(dt_min), max=float(dt_max))
    delta = dtf if delta_dtype is None else dtf.to(delta_dtype).to(f64)
    decay = torch.exp(A.to(f64)[None, :] * dtf)
    Bf = B.reshape(total, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    Cf = C.reshape(total, ngroups, dstate).to(f64).repeat_interleave(rep, dim=1)
    D = arguments.get("D")
    if D is None:
        Df = torch.zeros((nheads, 1), dtype=f64, device=x.device)
    else:
        Df = D.to(f64)
        Df = Df[:, None] if Df.ndim == 1 else Df
        if Df.shape[1] != headdim:
            # A 2D D on a per-head constructor consumes its first column.
            Df = Df[:, :1]
    initial = arguments.get("initial_states")
    y = torch.empty((total, nheads, headdim), dtype=f64, device=x.device)
    states = torch.empty(
        (len(lengths), nheads, headdim, dstate), dtype=f64, device=x.device
    )
    start = 0
    for sequence, length in enumerate(lengths):
        if initial is None:
            state = torch.zeros((nheads, headdim, dstate), dtype=f64, device=x.device)
        else:
            state = initial[sequence].to(f64).clone()
        for token in range(start, start + length):
            state = state * decay[token][:, None, None] + (
                (delta[token][:, None] * xf[token])[:, :, None] * Bf[token][:, None, :]
            )
            y[token] = (
                torch.einsum("hdn,hn->hd", state, Cf[token]) + Df * xf[token]
            )
        states[sequence] = state
        start += length
    z = arguments.get("z")
    if z is not None:
        zf = z.reshape(total, nheads, headdim).to(f64)
        y = y * (zf * torch.sigmoid(zf))
    return y.reshape(x.shape), states


def tol_stats(reference, value, *, atol=1e-2, rtol=1e-2):
    """Elementwise ``|value - reference| > atol + rtol * |reference|`` statistics."""

    ref = reference.to(torch.float64)
    val = value.to(torch.float64)
    diff = (val - ref).abs()
    outside = diff > atol + rtol * ref.abs()
    n_outside = int(outside.sum())
    stats = {
        "n": int(diff.numel()),
        "n_outside": n_outside,
        "frac_outside": n_outside / max(1, diff.numel()),
        "max_abs": float(diff.max()),
        "finite": bool(torch.isfinite(val).all()),
    }
    if n_outside:
        index = int(diff.argmax())
        stats["argmax_abs"] = list(
            int(v) for v in torch.unravel_index(torch.tensor(index), diff.shape)
        )
        stats["ref_at_argmax"] = float(ref.flatten()[index])
        stats["val_at_argmax"] = float(val.flatten()[index])
        rel = diff[outside] / ref.abs()[outside].clamp(min=1e-30)
        stats["max_rel_outside"] = float(rel.max())
        stats["median_abs_outside"] = float(diff[outside].median())
    return stats, outside


OUT_AXES = ("batch", "token", "head", "dim")
STATE_AXES = ("seq", "head", "dim", "dstate")


def structure(mask, names=OUT_AXES):
    """Where a ``[B, S, H, D]`` output (or ``[N, H, D, S]`` state) outlier mask sits."""

    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    dims = mask.ndim
    out = {"n": n}
    for axis, name in enumerate(names):
        reduce_axes = tuple(a for a in range(dims) if a != axis)
        per = mask.sum(dim=reduce_axes)
        hit = int((per > 0).sum())
        top = per.topk(min(5, per.numel()))
        out[name] = {
            "hit": hit,
            "total": int(per.numel()),
            "top": [(int(i), int(c)) for i, c in zip(top.indices, top.values)],
        }
    # The largest cluster in one (batch, token, head) row (max 64 = whole row).
    row = mask.sum(dim=-1)
    out["max_per_row"] = int(row.max())
    out["rows_hit"] = int((row > 0).sum())
    out["rows_full"] = int((row == mask.shape[-1]).sum())
    return out


def bf16_ulp_stats(a, b):
    """How far apart two bf16 tensors are in bf16 ulps."""

    assert a.dtype == b.dtype == torch.bfloat16
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    # Map sign-magnitude to a monotone integer line so ulp distance is |ia - ib|.
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    ulps = (ia - ib).abs()
    return {
        "n_diff": int((ulps > 0).sum()),
        "n_gt1": int((ulps > 1).sum()),
        "n_gt2": int((ulps > 2).sum()),
        "max_ulp": int(ulps.max()),
        "n": int(ulps.numel()),
    }


def _lengths_of(tensors, arguments, constructor, lengths):
    x = tensors[0]
    if constructor["has_varlen"]:
        assert lengths is not None
        return list(lengths)
    return [x.shape[1]] * x.shape[0]


def build_cases(tests):
    """name -> (constructor, tensors, arguments, lengths, cute_mode)."""

    def nemotron(constructor, tensors, arguments):
        arguments["initial_states"].zero_()
        arguments["dt_limit"] = (0.0, float("inf"))
        return constructor, tensors, arguments

    cases = {}

    def add(name, built, *, lengths=None, cute="direct"):
        constructor, tensors, arguments = built
        cases[name] = (
            constructor,
            tensors,
            arguments,
            _lengths_of(tensors, arguments, constructor, lengths),
            cute,
        )

    add("rm_h8g8_bf16_batched_dhdim", tests._case())
    add("rm_h8g8_f16_batched", tests._case(state_dtype=torch.float16, d_has_hdim=False))
    add("rm_h8g8_bf16_varlen_96_160", tests._case(varlen=True), lengths=(96, 160))
    add(
        "rm_h8g8_bf16_batched_dt_bf16",
        tests._case(preprocess_dtype=torch.bfloat16, d_has_hdim=False),
    )
    add("rm_h1g1_batched", tests._case(nheads=1, ngroups=1, d_has_hdim=False))
    add("rm_h128g128_batched", tests._case(nheads=128, ngroups=128, d_has_hdim=False))
    add(
        "rm_h128g8_batched_s128",
        nemotron(*tests._case(nheads=128, ngroups=8, d_has_hdim=False)),
    )
    add(
        "rm_h128g8_varlen_96_160",
        nemotron(*tests._case(nheads=128, ngroups=8, varlen=True, d_has_hdim=False)),
        lengths=(96, 160),
    )
    add(
        "h128g8_batched_s1024",
        nemotron(*tests._case(nheads=128, ngroups=8, seqlen=1024, d_has_hdim=False)),
    )
    add(
        "realistic_varlen_8x128_seed0",
        tests._realistic_decay_inputs((128,) * 8, 0),
        lengths=(128,) * 8,
    )
    add(
        "realistic_batched_2x128_seed1",
        tests._realistic_decay_inputs((128, 128), 1, varlen=False),
    )
    add(
        "unaligned_varlen_300_300",
        tests._case(varlen=True, lengths=(300, 300)),
        lengths=(300, 300),
        cute="padded",
    )
    add("unaligned_batched_2x1000", tests._case(seqlen=1000), cute="padded")
    add(
        "noinit_varlen_1x1000",
        tests._case(varlen=True, lengths=(1000,), initial_states=False),
        lengths=(1000,),
        cute="padded",
    )
    # f32 state: bf16-representable initial states so CuTe (bf16 state) sees
    # exactly the same values.
    constructor, tensors, arguments = tests._case(state_dtype=torch.float32)
    arguments["initial_states"] = (
        arguments["initial_states"].to(torch.bfloat16).to(torch.float32)
    )
    add("f32_state_batched", (constructor, tensors, arguments), cute="f32")
    return cases


def run_cute(tests, constructor, tensors, arguments, lengths, mode):
    if mode == "direct":
        return SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    if mode == "padded":
        return tests._cute_padded_reference(constructor, tensors, arguments, lengths)
    if mode == "f32":
        cute_constructor = {**constructor, "state_dtype": torch.bfloat16}
        cute_arguments = {
            **arguments,
            "initial_states": arguments["initial_states"].to(torch.bfloat16),
        }
        return SSDCombined(**cute_constructor, backend="cute").run(
            *tensors, **cute_arguments
        )
    raise ValueError(mode)


def probe_case(tests, name, constructor, tensors, arguments, lengths, cute_mode):
    started = time.time()
    cake = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    cute = run_cute(tests, constructor, tensors, arguments, lengths, cute_mode)
    torch.cuda.synchronize()
    ref_out, ref_state = fp64_reference(tensors, arguments, lengths)
    ref16_out, ref16_state = fp64_reference(
        tensors, arguments, lengths, delta_dtype=torch.float16
    )
    refb_out, refb_state = fp64_reference(
        tensors, arguments, lengths, delta_dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    row = {"case": name, "shape": list(tensors[0].shape), "lengths": list(lengths)}
    row["ref_out_absmax"] = float(ref_out.abs().max())
    row["ref_out_absmean"] = float(ref_out.abs().mean())
    cake_out, cake_state = cake
    cute_out, cute_state = cute
    assert tuple(cake_out.shape) == tuple(cute_out.shape), (cake_out.shape, cute_out.shape)
    for label, (actual, ref) in {
        "out": (cake_out, ref_out),
        "state": (cake_state, ref_state),
    }.items():
        cake_stats, cake_mask = tol_stats(ref, actual)
        cute_stats, cute_mask = tol_stats(ref, cute_out if label == "out" else cute_state)
        both_stats, _ = tol_stats(
            cute_out if label == "out" else cute_state, actual
        )
        cake16_stats, _ = tol_stats(ref16_out if label == "out" else ref16_state, actual)
        cuteb_stats, _ = tol_stats(
            refb_out if label == "out" else refb_state,
            cute_out if label == "out" else cute_state,
        )
        cake_b_stats, _ = tol_stats(refb_out if label == "out" else refb_state, actual)
        cute16_stats, _ = tol_stats(
            ref16_out if label == "out" else ref16_state,
            cute_out if label == "out" else cute_state,
        )
        overlap = int((cake_mask & cute_mask).sum())
        row[label] = {
            "cake_vs_ref": cake_stats,
            "cute_vs_ref": cute_stats,
            "cake_vs_cute": both_stats,
            "cake_vs_ref_f16delta": cake16_stats,
            "cake_vs_ref_bf16delta": cake_b_stats,
            "cute_vs_ref_bf16delta": cuteb_stats,
            "cute_vs_ref_f16delta": cute16_stats,
            "outlier_overlap_cake_cute": overlap,
            "cake_outlier_structure": structure(
                cake_mask, OUT_AXES if label == "out" else STATE_AXES
            ),
            "cute_outlier_structure": structure(
                cute_mask, OUT_AXES if label == "out" else STATE_AXES
            ),
        }
        if actual.dtype == torch.bfloat16 and (
            cute_out if label == "out" else cute_state
        ).dtype == torch.bfloat16:
            row[label]["cake_vs_cute_ulps"] = bf16_ulp_stats(
                actual, cute_out if label == "out" else cute_state
            )
    row["seconds"] = round(time.time() - started, 1)
    return row


def summarize(row):
    def fmt(stats):
        return (
            f"n_out={stats['n_outside']}/{stats['n']} ({stats['frac_outside']:.3%}) "
            f"max_abs={stats['max_abs']:.4g}"
        )

    lines = []
    for label in ("out", "state"):
        block = row[label]
        lines.append(
            f"RESULT {row['case']} {label}: cake~ref {fmt(block['cake_vs_ref'])} | "
            f"cute~ref {fmt(block['cute_vs_ref'])} | cake~cute {fmt(block['cake_vs_cute'])} | "
            f"cake~ref_f16d {fmt(block['cake_vs_ref_f16delta'])} | "
            f"cute~ref_bf16d {fmt(block['cute_vs_ref_bf16delta'])} | "
            f"overlap={block['outlier_overlap_cake_cute']}"
        )
        if "cake_vs_cute_ulps" in block:
            lines.append(f"RESULT {row['case']} {label} ulps(cake,cute): {block['cake_vs_cute_ulps']}")
        lines.append(
            f"RESULT {row['case']} {label} cake outliers: {json.dumps(block['cake_outlier_structure'])}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="JSON report path")
    parser.add_argument("--only", nargs="*", default=None, help="case names to run")
    parser.add_argument("--list", action="store_true", help="list case names and exit")
    args = parser.parse_args(argv)
    tests = _load_test_module()
    cases = build_cases(tests)
    if args.list:
        print("\n".join(cases))
        return 0
    selected = args.only or list(cases)
    rows = []
    failures = []
    for name in selected:
        print(f"LAUNCH {name}", flush=True)
        try:
            row = probe_case(tests, name, *cases[name])
        except Exception as error:  # noqa: BLE001 - report every case
            failures.append(name)
            print(f"RESULT {name} ERROR {error!r}"[:600], flush=True)
            rows.append({"case": name, "error": repr(error)[:600]})
            continue
        rows.append(row)
        print(summarize(row), flush=True)
        if args.out is not None:
            args.out.write_text(json.dumps(rows, indent=1))
    print(f"SUMMARY cases={len(selected)} errors={len(failures)} {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
