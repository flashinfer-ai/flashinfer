# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU models of ``lse.row``.

``lse_reference(x, temperature, token_ids, rows)``: per row, the maximum and the sum through the canonical trees, the
chains through :mod:`flashinfer.batch_invariant.fp_interface.exact` for the per-row scalars and torch float32 CPU ops
for the per-element values (each a single IEEE rounding: ``torch.add / mul`` and the RNE ``.to(dtype)``), and the two
opaque primitives (``ex2.approx.ftz``, ``lg2.approx.ftz``) evaluated on the device with the kernels' own instructions
(``mufu_device.py``). The dataflow is fully declared, so this is exact, and it is cross-checked against the
all-symbolic evaluation (:func:`exact_rows`) in the GPU suite.

Outside every kernel cache closure (no kernel module imports it).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..fp_interface import exact as fp32
from . import sterm as term
from .mufu_device import ex2_device, lg2_device
from .schedule import BLOCK, LANES, V, padded_blocks
from .spec import LN2_F32, LOG2E_F32, LSESpec, scale_value

_FMT = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


# ---------------------------------------------------------------------------------------------------------- the trees
def row_max(xr: torch.Tensor) -> float:
    """The ``fmaxc`` tree's value over one f32 row: the largest non-NaN value (``fmaxc`` drops NaN; a row of NaN gives
    the fold's ``-inf``), and for a zero maximum ``+0`` iff some zero of the row is ``+0``."""
    fin = xr[~torch.isnan(xr)]
    if fin.numel() == 0:
        return float("-inf")
    m = float(fin.max())
    if m == 0.0:
        return 0.0 if bool(((fin == 0) & ~torch.signbit(fin)).any()) else -0.0
    return m


def row_sum(e: torch.Tensor, N: int) -> float:
    """The ``add`` tree over one row of ``e_j`` (f32): the elem fold from ``+0``, the pairwise block tree, the pairwise
    lane tree -- with torch f32 CPU adds (one rounding each). Absent elements are ``+0`` (the digits' identity)."""
    Kp = padded_blocks(N)
    t = torch.zeros(Kp * BLOCK, dtype=torch.float32)
    t[:N] = e
    t = t.view(Kp, LANES, V)
    acc = torch.zeros(Kp, LANES, dtype=torch.float32)
    for j in range(V):
        acc = torch.add(acc, t[:, :, j])
    while acc.shape[0] > 1:
        acc = torch.add(acc[0::2], acc[1::2])
    v = acc[0]
    while v.shape[0] > 1:
        v = torch.add(v[0::2], v[1::2])
    return float(v[0])


def _scales(temperature, M: int) -> List[float]:
    """The per-row ``c`` as the API forms it: ``LOG2E`` (none), ``div(LOG2E, f32(T))`` for a scalar, per row for a tensor
    (16-bit temperatures widened exactly, f64 rounded RNE to f32 as the API does)."""
    if temperature is None:
        return [LOG2E_F32] * M
    if isinstance(temperature, torch.Tensor):
        ts = temperature.detach().to("cpu", torch.float32).reshape(-1).tolist()
        if len(ts) == 1 and M != 1:
            ts = ts * M
        assert len(ts) == M, (len(ts), M)
        return [fp32.div(LOG2E_F32, t) for t in ts]
    return [scale_value(float(temperature))] * M


def lse_reference(
    x: torch.Tensor,
    temperature=None,
    token_ids: Optional[torch.Tensor] = None,
    rows: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """The exact CPU model for ``rows`` of ``x`` (default: all). Returns ``{"m", "s", "lse", "logp"}`` as f32 ``(n,)``
    tensors (``logp`` NaN where no token id is given or it is out of range), ``"probs"`` ``(n, N)`` in the input dtype
    and ``"logprobs"`` ``(n, N)`` f32."""
    x = x.detach().cpu()
    M, N = x.shape
    rows = list(range(M)) if rows is None else list(rows)
    cs = _scales(temperature, M)
    xf = x.to(torch.float32)
    ids = (
        token_ids.detach().cpu().reshape(-1).tolist() if token_ids is not None else None
    )
    out: Dict[str, list] = {
        k: [] for k in ("m", "s", "lse", "logp", "probs", "logprobs")
    }
    for r in rows:
        xr = xf[r]
        m = row_max(xr)
        nm, c = fp32.neg(m), cs[r]
        z = torch.mul(
            torch.add(xr, torch.tensor(nm, dtype=torch.float32)),
            torch.tensor(c, dtype=torch.float32),
        )
        e = ex2_device(z)
        s = row_sum(e, N)
        inv_s = fp32.div(1.0, s)
        p = torch.mul(e, torch.tensor(inv_s, dtype=torch.float32)).to(x.dtype)
        lg2s = float(lg2_device(torch.tensor([s], dtype=torch.float32))[0])
        lse = fp32.mul(fp32.add(fp32.mul(m, c), lg2s), LN2_F32)
        nlg = fp32.neg(lg2s)
        lp = torch.mul(
            torch.add(z, torch.tensor(nlg, dtype=torch.float32)),
            torch.tensor(LN2_F32, dtype=torch.float32),
        )
        logp_t = float("nan")
        if ids is not None and 0 <= ids[r] < N:
            zt = fp32.mul(fp32.add(float(xr[ids[r]]), nm), c)
            logp_t = fp32.mul(fp32.add(zt, nlg), LN2_F32)
        out["m"].append(m)
        out["s"].append(s)
        out["lse"].append(lse)
        out["logp"].append(logp_t)
        out["probs"].append(p)
        out["logprobs"].append(lp)
    res = {
        k: torch.tensor(out[k], dtype=torch.float64).to(torch.float32)
        for k in ("m", "s", "lse", "logp")
    }
    res["probs"] = (
        torch.stack(out["probs"]) if out["probs"] else torch.empty(0, N, dtype=x.dtype)
    )
    res["logprobs"] = (
        torch.stack(out["logprobs"])
        if out["logprobs"]
        else torch.empty(0, N, dtype=torch.float32)
    )
    return res


# -------------------------------------------------------------------------------------------- the all-symbolic oracle
def _table_fn(table: Dict[int, float], missing: Dict[int, float], default: float):
    def fn(v: float) -> float:
        b = fp32.f32_bits(v)
        if b in table:
            return table[b]
        missing[b] = v
        return default

    return fn


def exact_rows(
    x: torch.Tensor,
    temperature=None,
    rows: Sequence[int] = (0,),
    token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Every op of the symbolic spec through the exact backend (the opaque ``ex2`` / ``lg2`` from device tables collected
    in two passes). Slow; the oracle of :func:`lse_reference` for short rows."""
    x = x.detach().cpu()
    M, N = x.shape
    in_fmt = _FMT[x.dtype]
    spec = LSESpec(N, in_fmt=in_fmt, temperature=temperature is not None)
    xf = x.to(torch.float32).double()
    ids = (
        token_ids.detach().cpu().reshape(-1).tolist() if token_ids is not None else None
    )
    ts = None
    if temperature is not None:
        ts = (
            temperature.detach().to("cpu", torch.float32).reshape(-1).tolist()
            if isinstance(temperature, torch.Tensor)
            else [fp32.to_f32(float(temperature))] * M
        )
        if len(ts) == 1 and M != 1:
            ts = ts * M
        assert len(ts) == M, (len(ts), M)
    out: Dict[str, list] = {
        k: [] for k in ("m", "s", "lse", "logp", "probs", "logprobs")
    }
    for r in rows:
        xr = xf[r].tolist()
        terms = spec.outputs(r)
        want = (
            list(terms["probs"])
            + list(terms["logprobs"])
            + [terms["m"], terms["s"], terms["lse"]]
        )
        if ids is not None and 0 <= ids[r] < N:
            want.append(spec.logp(r, ids[r]))
        ex2_tab: Dict[int, float] = {}
        lg2_tab: Dict[int, float] = {}

        def run(ex2_missing, lg2_missing, xr=xr, want=want, r=r):
            memo: Dict[int, float] = {}
            syms = {
                "ex2": _table_fn(ex2_tab, ex2_missing, 1.0),
                "op:lg2": _table_fn(lg2_tab, lg2_missing, 0.0),
            }
            if ts is not None:
                syms["T"] = ts[r]
            return [term.evaluate(t, lambda rr, i: xr[i], memo, syms) for t in want]

        miss_e: Dict[int, float] = {}
        miss_l: Dict[int, float] = {}
        run(miss_e, miss_l)  # pass 1: the ex2 arguments (functions of x and m only)
        ex2_tab.update(
            zip(
                miss_e.keys(),
                ex2_device(
                    torch.tensor(list(miss_e.values()), dtype=torch.float32)
                ).tolist(),
                strict=True,
            )
        )
        miss_e, miss_l = {}, {}
        run(miss_e, miss_l)  # pass 2: the lg2 argument (the exact sum)
        assert not miss_e, "every ex2 argument was collected in pass 1"
        lg2_tab.update(
            zip(
                miss_l.keys(),
                lg2_device(
                    torch.tensor(list(miss_l.values()), dtype=torch.float32)
                ).tolist(),
                strict=True,
            )
        )
        miss_e, miss_l = {}, {}
        vals = run(miss_e, miss_l)
        assert not miss_e and not miss_l
        out["probs"].append(torch.tensor(vals[:N], dtype=torch.float64).to(x.dtype))
        out["logprobs"].append(
            torch.tensor(vals[N : 2 * N], dtype=torch.float64).to(torch.float32)
        )
        out["m"].append(vals[2 * N])
        out["s"].append(vals[2 * N + 1])
        out["lse"].append(vals[2 * N + 2])
        out["logp"].append(vals[2 * N + 3] if len(vals) > 2 * N + 3 else float("nan"))
    res = {
        k: torch.tensor(out[k], dtype=torch.float64).to(torch.float32)
        for k in ("m", "s", "lse", "logp")
    }
    res["probs"] = (
        torch.stack(out["probs"]) if out["probs"] else torch.empty(0, N, dtype=x.dtype)
    )
    res["logprobs"] = (
        torch.stack(out["logprobs"])
        if out["logprobs"]
        else torch.empty(0, N, dtype=torch.float32)
    )
    return res
