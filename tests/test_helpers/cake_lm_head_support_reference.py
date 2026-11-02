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

Seeded inputs, the masked FP64 oracle and the masked PyTorch baseline (B0) of the candidate-support
LM-head problem (flashinfer-ai/flashinfer#7636): ``X [T, H]`` BF16, ``W [V, H]`` BF16, a ragged support
``support_ids [M]`` / ``support_offsets [T + 1]``, optional int64 labels (``-100`` = ignored), an optional
detached FP32 teacher ``[M]`` and a temperature (``None``, a scalar or an FP32 ``[T]`` tensor).

Contract reproduced here (independently of the package):

* ``q_tv = z_tv / tau_t`` with ``z = X @ W^T``; ``logp_tv = q_tv - logsumexp_{u in S_t} q_tu`` for
  ``v in S_t``; rows with ``K_t <= 1`` give ``logp = 0`` and ``KL = 0``.
* Selected: ``logp_t[y_t]`` for ``y_t != -100`` (a member of ``S_t``), 0 otherwise.
* ``KL_t = sum_{v in S_t, Q > 0} Q_tv (lq_tv - logp_tv)`` with ``Q = exp(lq)``.
* The downstream loss of the gradients is ``sum g * packed + sum d * selected + sum e * KL`` for the
  outputs a call forms (an upstream on a constant output entry never contributes).

:func:`reference_support_fp64` evaluates it in float64 from the BF16 input values (no rounding of ``z``)
with PyTorch autograd, chunked over rows; :func:`reference_support_b0` is the eager PyTorch path at the
contract's rounding boundaries (BF16 ``z`` promoted to FP32, a dense ``[T, V]`` logits tensor with the
support as a mask, autograd), the "dense logits + support mask" baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from tests.test_helpers.cake_lm_head_loss_reference import (
    IGNORE_INDEX,
    _randn_rows,
    make_weight,
    rel_l2,
)

OUTPUTS = ("packed", "selected", "kl")
TAU_FORMS = ("none", "scalar", "constant", "hetero")
SUPPORT_KINDS = ("ind", "shr", "heavy", "full")
# Accuracy gates against the FP64 oracle: at most 1.05x the error of B0 on the same inputs plus a floor
# per metric (the values of tests/experimental/test_cake_lm_head_det_loss.py, logp-scale floors for KL).
GATE_MARGIN = 1.05
GATE_TINY = dict(
    packed_max_abs=1e-5,
    selected_max_abs=1e-5,
    kl_max_abs=1e-5,
    dX_rel_l2=1e-3,
    dW_rel_l2=1e-3,
)
_TAU_SEED_OFFSET = (
    2000003  # the temperature's own generator (the #7637 rule): other draws do not move
)
_TEACHER_SEED_OFFSET = 3000017


@dataclass
class SupportInputs:
    """One candidate-support problem plus default upstream gradients ``g [M]``, ``d [T]``, ``e [T]``."""

    X: torch.Tensor
    W: torch.Tensor
    support_ids: torch.Tensor
    support_offsets: torch.Tensor
    labels: Optional[torch.Tensor]
    teacher: Optional[torch.Tensor]
    temperature: Union[None, float, torch.Tensor]
    k_cap: int
    g: torch.Tensor
    d: torch.Tensor
    e: torch.Tensor
    seed: int

    @property
    def T(self) -> int:
        return int(self.X.shape[0])

    @property
    def H(self) -> int:
        return int(self.X.shape[1])

    @property
    def V(self) -> int:
        return int(self.W.shape[0])

    @property
    def M(self) -> int:
        return int(self.support_ids.numel())

    @property
    def counts(self) -> torch.Tensor:
        off = self.support_offsets.to(torch.int64)
        return off[1:] - off[:-1]

    def tau_rows(self) -> torch.Tensor:
        """FP32 ``[T]``: the temperature of every row."""
        return support_tau_rows(self.temperature, self.T, self.X.device)

    def kwargs(self) -> dict:
        """Keyword arguments of the package entry points for this problem."""
        return dict(k_cap=self.k_cap, temperature=self.temperature)


def support_tau_rows(temperature, T: int, device) -> torch.Tensor:
    if temperature is None:
        return torch.ones(T, dtype=torch.float32, device=device)
    if isinstance(temperature, torch.Tensor):
        return temperature.to(device=device, dtype=torch.float32).reshape(T)
    return torch.full((T,), float(temperature), dtype=torch.float32, device=device)


def make_temperature(
    form: str, T: int, *, seed: int, device, scalar: float = 0.7
) -> Union[None, float, torch.Tensor]:
    """``none`` -> ``None``; ``scalar`` -> ``scalar``; ``constant`` -> ``full((T,), scalar)``; ``hetero`` ->
    log-uniform in ``[0.25, 4]`` with about ten percent of the rows exactly 1.0, from its own generator."""
    if form not in TAU_FORMS:
        raise ValueError(f"form must be one of {TAU_FORMS}")
    if form == "none":
        return None
    if form == "scalar":
        return float(scalar)
    if form == "constant":
        return torch.full((T,), float(scalar), dtype=torch.float32, device=device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + _TAU_SEED_OFFSET)
    tau = torch.exp(
        torch.empty(T, dtype=torch.float64).uniform_(
            math.log(0.25), math.log(4.0), generator=gen
        )
    )
    ones = torch.rand(T, generator=gen) < 0.1
    tau = torch.where(ones, torch.ones_like(tau), tau).float()
    return tau.to(device)


def make_support(
    T: int,
    V: int,
    k_cap: int,
    *,
    kind: str = "ind",
    seed: int,
    edge_rows: bool = True,
    k_max: Optional[int] = None,
    group: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU int64 ``(support_ids [M], support_offsets [T + 1])``, ids in random order inside each row.

    ``K_t`` is uniform in ``[0, min(k_cap, V, k_max)]``; with ``edge_rows`` the first rows get ``K`` = 0, 1, 2
    and the maximum.  ``ind``: independent ids per row; ``shr``: rows in groups of ``group`` draw from one pool
    of ``2 * K_max`` ids; ``heavy``: like ``ind`` with id ``V // 2 + 1`` in every nonempty row; ``full``: every
    row holds all ``V`` ids (``k_cap >= V``)."""
    if kind not in SUPPORT_KINDS:
        raise ValueError(f"kind must be one of {SUPPORT_KINDS}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    top = (
        min(int(k_cap), int(V))
        if k_max is None
        else min(int(k_cap), int(V), int(k_max))
    )
    if kind == "full":
        if k_cap < V:
            raise ValueError("kind='full' needs k_cap >= V")
        counts = torch.full((T,), V, dtype=torch.int64)
    else:
        counts = torch.randint(0, top + 1, (T,), generator=gen, dtype=torch.int64)
        if edge_rows:
            for t, k in enumerate((0, 1, min(2, top), top)[:T]):
                counts[t] = k
    heavy = V // 2 + 1
    rows = []
    pool = None
    for t in range(T):
        k = int(counts[t])
        if kind == "shr":
            if t % group == 0:
                pool = torch.randperm(V, generator=gen)[: min(V, 2 * max(top, 1))]
            ids = pool[torch.randperm(int(pool.numel()), generator=gen)[:k]]
        elif kind == "heavy" and k:
            others = torch.randperm(V - 1, generator=gen)[: k - 1]
            others = others + (others >= heavy).long()  # skip the heavy id
            ids = torch.cat([torch.tensor([heavy]), others])[
                torch.randperm(k, generator=gen)
            ]
        else:
            ids = torch.randperm(V, generator=gen)[:k]
        rows.append(ids.to(torch.int64))
    ids = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.int64)
    offsets = torch.zeros(T + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, 0)
    return ids, offsets


def _padded(offsets: torch.Tensor, r0: int, r1: int):
    """CPU ``(rows, local, kmax, counts, o0, o1)`` of rows ``[r0, r1)``: each pair's row (from ``r0``) and index
    inside its row, the largest count, the counts and the rows' pair range ``[o0, o1)``."""
    off = offsets.to(torch.int64).cpu()
    counts = off[r0 + 1 : r1 + 1] - off[r0:r1]
    o0, o1 = int(off[r0]), int(off[r1])
    rows = torch.repeat_interleave(torch.arange(r1 - r0), counts)
    local = torch.arange(o1 - o0) - (off[r0:r1] - o0)[rows]
    kmax = int(counts.max()) if r1 > r0 and o1 > o0 else 0
    return rows, local, kmax, counts, o0, o1


def label_positions(
    ids: torch.Tensor, offsets: torch.Tensor, labels: Optional[torch.Tensor]
) -> torch.Tensor:
    """int64 ``[T]``: the label's index inside its row's candidates (``-1`` for ``-100``); raises when a label
    is not a candidate (the helper builds valid inputs only)."""
    T = int(offsets.numel()) - 1
    out = torch.full((T,), -1, dtype=torch.int64)
    if labels is None or T == 0:
        return out.to(ids.device)
    rows, local, kmax, counts, o0, o1 = _padded(offsets, 0, T)
    lab = labels.to(torch.int64).cpu()
    if kmax:
        pad = torch.full((T, kmax), -1, dtype=torch.int64)
        pad[rows, local] = ids.to(torch.int64).cpu()
        hit = pad == lab[:, None]
        found = hit.any(1)
        out = torch.where(found, hit.int().argmax(1), out)
    else:
        found = torch.zeros(T, dtype=torch.bool)
    need = lab != IGNORE_INDEX
    if bool((need & ~found).any()):
        raise ValueError("a label is not one of its row's candidates")
    return out.to(ids.device)


def make_support_inputs(
    T: int,
    *,
    seed: int,
    H: int = 256,
    V: int = 512,
    k_cap: int = 8,
    kind: str = "ind",
    device="cpu",
    labels: bool = False,
    ignore_frac: float = 0.1,
    teacher: bool = False,
    teacher_inf_frac: float = 0.05,
    tau_form: str = "none",
    tau_scalar: float = 0.7,
    x_scale: float = 0.5,
    ids_dtype: torch.dtype = torch.int64,
    offsets_dtype: torch.dtype = torch.int64,
    edge_rows: bool = True,
    k_max: Optional[int] = None,
    W: Optional[torch.Tensor] = None,
) -> SupportInputs:
    """Seeded inputs of one problem.  ``X = bf16(x_scale * N(0, 1))``, ``W ~ N(0, 1/H)`` from its own generator
    (``seed + 1000003``) unless passed in.  Labels: a random candidate of each nonempty row, ``-100`` on empty
    rows and on about ``ignore_frac`` of the others.  Teacher: the FP64 log-softmax over each row of
    ``N(0, 1)`` teacher logits, with about ``teacher_inf_frac`` of the entries of rows with ``K >= 3`` at
    ``-inf`` (at least two finite entries stay), as FP32.  Upstreams ``g, d, e = 0.05 N(0, 1)`` FP32."""
    device = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    X = _randn_rows((T, H), gen, torch.device("cpu"), x_scale).to(device)
    if W is None:
        W = make_weight(V, H, seed=seed + 1000003, device="cpu").to(device)
    ids, offsets = make_support(
        T, V, k_cap, kind=kind, seed=seed + 17, edge_rows=edge_rows, k_max=k_max
    )
    M = int(ids.numel())
    counts = offsets[1:] - offsets[:-1]
    lab = None
    if labels:
        lab = torch.full((T,), IGNORE_INDEX, dtype=torch.int64)
        u = torch.rand(T, generator=gen)
        pick = torch.floor(torch.rand(T, generator=gen) * counts.clamp_min(1)).long()
        take = (counts > 0) & (u >= ignore_frac)
        idx = (offsets[:-1] + pick).clamp_max(max(M - 1, 0))
        if M:
            lab = torch.where(take, ids[idx], lab)
        lab = lab.to(device)
    lq = None
    if teacher:
        tg = torch.Generator(device="cpu")
        tg.manual_seed(seed + _TEACHER_SEED_OFFSET)
        logits = torch.randn(M, generator=tg, dtype=torch.float64)
        rows = torch.repeat_interleave(torch.arange(T), counts)
        local = torch.arange(M) - offsets[:-1][rows]
        drop = (
            (torch.rand(M, generator=tg) < teacher_inf_frac)
            & (counts[rows] >= 3)
            & (local >= 2)
        )
        logits = torch.where(drop, torch.full_like(logits, -math.inf), logits)
        mx = torch.full((T,), -math.inf, dtype=torch.float64).scatter_reduce(
            0, rows, logits, "amax", include_self=True
        )
        ssum = torch.zeros(T, dtype=torch.float64).index_add_(
            0, rows, torch.exp(logits - mx[rows])
        )
        lq = (logits - mx[rows] - torch.log(ssum)[rows]).float().to(device)
    ug = torch.Generator(device="cpu")
    ug.manual_seed(seed + 7)
    g = (torch.randn(M, generator=ug) * 0.05).to(device)
    d = (torch.randn(T, generator=ug) * 0.05).to(device)
    e = (torch.randn(T, generator=ug) * 0.05).to(device)
    return SupportInputs(
        X=X,
        W=W,
        support_ids=ids.to(ids_dtype).to(device),
        support_offsets=offsets.to(offsets_dtype).to(device),
        labels=lab,
        teacher=lq,
        temperature=make_temperature(
            tau_form, T, seed=seed, device=device, scalar=tau_scalar
        ),
        k_cap=int(k_cap),
        g=g,
        d=d,
        e=e,
        seed=int(seed),
    )


@torch.no_grad()
def topp_support(
    X: torch.Tensor,
    W: torch.Tensor,
    *,
    p: float,
    k_cap: int,
    temperature=None,
    chunk_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-p sampling mask of every row under ``W`` (float64 from the BF16 values): the smallest set of the
    most probable ids whose probability reaches ``p``, capped at ``k_cap``, in descending probability.
    Returns CPU int64 ``(support_ids, support_offsets)``."""
    T = int(X.shape[0])
    tau = support_tau_rows(temperature, T, X.device).double()
    W64 = W.double()
    rows = []
    counts = torch.zeros(T, dtype=torch.int64)
    for r0 in range(0, T, chunk_rows):
        r1 = min(T, r0 + chunk_rows)
        prob = torch.softmax((X[r0:r1].double() @ W64.t()) / tau[r0:r1, None], dim=1)
        sp, order = torch.sort(prob, dim=1, descending=True)
        before = torch.cumsum(sp, 1) - sp
        k = ((before < p).sum(1)).clamp(1, int(k_cap)).cpu()
        counts[r0:r1] = k
        order = order.cpu()
        for i in range(r1 - r0):
            rows.append(order[i, : int(k[i])])
    ids = torch.cat(rows).to(torch.int64)
    offsets = torch.zeros(T + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, 0)
    return ids, offsets


def _upstreams(
    inp: SupportInputs, outputs: Sequence[str], upstream: Optional[dict]
) -> dict:
    up = (
        dict(packed=inp.g, selected=inp.d, kl=inp.e)
        if upstream is None
        else dict(upstream)
    )
    return {k: up.get(k) for k in outputs}


def reference_support_fp64(
    inp: SupportInputs,
    *,
    outputs: Sequence[str],
    upstream: Optional[dict] = None,
    need_dx: bool = True,
    need_dw: bool = True,
    chunk_rows: int = 256,
) -> dict:
    """Float64 oracle from the BF16 input values (``z`` at the support only, chunked over rows, PyTorch autograd).

    Returns FP64 ``packed [M]`` / ``selected [T]`` / ``kl [T]`` for the requested ``outputs``, ``lse [T]`` (0 on
    empty rows) and the FP64 gradients ``dX [T, H]`` / ``dW [V, H]`` (``None`` when not requested) of
    ``sum g * packed + sum d * selected + sum e * kl`` (``upstream``: ``{name: tensor or None}``, default the
    input's ``g`` / ``d`` / ``e``)."""
    for name in outputs:
        if name not in OUTPUTS:
            raise ValueError(f"unknown output {name!r}")
    up = _upstreams(inp, outputs, upstream)
    X, W, T, H, V, M = inp.X, inp.W, inp.T, inp.H, inp.V, inp.M
    device = X.device
    ids = inp.support_ids.to(torch.int64)
    offsets = inp.support_offsets.to(torch.int64).cpu()
    tau = inp.tau_rows().double()
    lpos = label_positions(inp.support_ids, inp.support_offsets, inp.labels).to(device)
    W64 = W.detach().double().requires_grad_(need_dw)
    f64 = dict(dtype=torch.float64, device=device)
    res = dict(
        packed=torch.zeros(M, **f64) if "packed" in outputs else None,
        selected=torch.zeros(T, **f64) if "selected" in outputs else None,
        kl=torch.zeros(T, **f64) if "kl" in outputs else None,
        lse=torch.zeros(T, **f64),
        dX=torch.zeros(T, H, **f64) if need_dx else None,
        dW=torch.zeros(V, H, **f64) if need_dw else None,
    )
    for r0 in range(0, T, chunk_rows):
        r1 = min(T, r0 + chunk_rows)
        rows, local, kmax, counts, o0, o1 = _padded(offsets, r0, r1)
        if kmax == 0:
            continue
        n = r1 - r0
        rows, local, counts = rows.to(device), local.to(device), counts.to(device)
        ids_pad = torch.zeros(n, kmax, dtype=torch.int64, device=device)
        ids_pad[rows, local] = ids[o0:o1]
        mask = torch.zeros(n, kmax, dtype=torch.bool, device=device)
        mask[rows, local] = True
        multi = (counts >= 2)[:, None]
        x = X[r0:r1].detach().double().requires_grad_(need_dx)
        with torch.enable_grad():
            q = (x @ W64.t()).gather(1, ids_pad) / tau[r0:r1, None]
            qm = torch.where(mask, q, torch.full_like(q, -math.inf))
            qm = torch.where((counts > 0)[:, None], qm, torch.zeros_like(qm))
            lse = torch.logsumexp(qm, dim=1)
            logp = torch.where(mask & multi, q - lse[:, None], torch.zeros_like(q))
            loss = torch.zeros((), **f64)
            if "packed" in outputs:
                pk = logp[mask]
                res["packed"][o0:o1] = pk.detach()
                if up["packed"] is not None:
                    loss = loss + (pk * up["packed"][o0:o1].double()).sum()
            if "selected" in outputs:
                lp = lpos[r0:r1]
                sel = torch.where(
                    lp >= 0,
                    logp.gather(1, lp.clamp_min(0)[:, None]).squeeze(1),
                    torch.zeros_like(lse),
                )
                res["selected"][r0:r1] = sel.detach()
                if up["selected"] is not None:
                    dd = torch.where(
                        lp >= 0, up["selected"][r0:r1].double(), torch.zeros_like(lse)
                    )
                    loss = loss + (sel * dd).sum()
            if "kl" in outputs:
                lq = torch.full((n, kmax), -math.inf, **f64)
                lq[rows, local] = inp.teacher[o0:o1].double()
                Q = torch.exp(lq)
                term = torch.where(mask & (Q > 0), Q * (lq - logp), torch.zeros_like(q))
                klr = torch.where(multi[:, 0], term.sum(1), torch.zeros_like(lse))
                res["kl"][r0:r1] = klr.detach()
                if up["kl"] is not None:
                    ee = torch.where(
                        multi[:, 0], up["kl"][r0:r1].double(), torch.zeros_like(lse)
                    )
                    loss = loss + (klr * ee).sum()
            res["lse"][r0:r1] = torch.where(
                counts > 0, lse, torch.zeros_like(lse)
            ).detach()
            wrt = [t for t, need in ((x, need_dx), (W64, need_dw)) if need]
            if wrt and loss.requires_grad:
                grads = torch.autograd.grad(loss, wrt, allow_unused=True)
                gi = iter(grads)
                if need_dx:
                    gx = next(gi)
                    if gx is not None:
                        res["dX"][r0:r1] = gx
                if need_dw:
                    gw = next(gi)
                    if gw is not None:
                        res["dW"] += gw
    return res


def reference_support_b0(
    inp: SupportInputs,
    *,
    outputs: Sequence[str],
    upstream: Optional[dict] = None,
    need_dx: bool = True,
    need_dw: bool = True,
) -> dict:
    """The eager PyTorch baseline at the contract's rounding boundaries: BF16 ``z = X @ W^T`` promoted to FP32,
    ``q = z / tau`` (tensor by tensor), the support as a mask of the dense ``[T, V]`` logits, ``logsumexp``,
    gathers, autograd.  Returns FP32 outputs and the BF16 gradients (``None`` when not requested)."""
    up = _upstreams(inp, outputs, upstream)
    T, V, M = inp.T, inp.V, inp.M
    device = inp.X.device
    X = inp.X.detach().requires_grad_(bool(need_dx))
    W = inp.W.detach().requires_grad_(bool(need_dw))
    ids = inp.support_ids.to(torch.int64)
    counts = inp.counts.to(device)
    rows = torch.repeat_interleave(torch.arange(T, device=device), counts)
    lpos = label_positions(inp.support_ids, inp.support_offsets, inp.labels).to(device)
    offsets = inp.support_offsets.to(torch.int64).to(device)
    multi = counts >= 2
    zero_t = torch.zeros(T, dtype=torch.float32, device=device)
    res: dict = dict(packed=None, selected=None, kl=None)
    with torch.enable_grad():
        z = (X @ W.t()).float()
        q = z / inp.tau_rows()[:, None]
        mask = torch.zeros(T, V, dtype=torch.bool, device=device)
        mask[rows, ids] = True
        qm = q.masked_fill(~mask, -math.inf)
        qm = torch.where((counts > 0)[:, None], qm, torch.zeros_like(qm))
        lse = torch.logsumexp(qm, dim=1)
        logp_full = q - lse[:, None]
        packed = torch.where(
            multi[rows], logp_full[rows, ids], torch.zeros(M, device=device)
        )
        loss = torch.zeros((), dtype=torch.float32, device=device)
        if "packed" in outputs:
            res["packed"] = packed
            if up["packed"] is not None:
                loss = loss + (packed * up["packed"]).sum()
        if "selected" in outputs:
            j = (offsets[:-1] + lpos.clamp_min(0)).clamp_max(max(M - 1, 0))
            sel = torch.where(lpos >= 0, packed[j], zero_t) if M else zero_t
            res["selected"] = sel
            if up["selected"] is not None:
                loss = (
                    loss + (sel * torch.where(lpos >= 0, up["selected"], zero_t)).sum()
                )
        if "kl" in outputs:
            lq_full = torch.full((T, V), -math.inf, dtype=torch.float32, device=device)
            lq_full[rows, ids] = inp.teacher
            Q = torch.exp(lq_full)
            lpk = torch.zeros(T, V, dtype=torch.float32, device=device)
            lpk = lpk.index_put((rows, ids), packed)
            term = torch.where(Q > 0, Q * (lq_full - lpk), torch.zeros_like(Q))
            kl = torch.where(multi, term.sum(1), zero_t)
            res["kl"] = kl
            if up["kl"] is not None:
                loss = loss + (kl * torch.where(multi, up["kl"], zero_t)).sum()
        if (need_dx or need_dw) and loss.requires_grad:
            loss.backward()
    for name in OUTPUTS:
        if res[name] is not None:
            res[name] = res[name].detach()
    res["dX"] = (
        X.grad.detach()
        if (need_dx and X.grad is not None)
        else (torch.zeros_like(inp.X) if need_dx else None)
    )
    res["dW"] = (
        W.grad.detach()
        if (need_dw and W.grad is not None)
        else (torch.zeros_like(inp.W) if need_dw else None)
    )
    return res


def support_error_report(result: dict, oracle: dict, inp: SupportInputs) -> dict:
    """Errors of ``result`` against the FP64 ``oracle``: ``packed_max_abs``, ``selected_max_abs`` (labelled rows),
    ``kl_max_abs``, ``dX_rel_l2`` / ``dW_rel_l2`` where both sides carry them, plus ``selected_ignored_zero`` and
    ``nan`` (any non-finite value)."""
    report: dict = {}
    nan = False
    for name in ("packed", "selected", "kl"):
        got, ref = result.get(name), oracle.get(name)
        if got is None or ref is None:
            continue
        got = got.detach().double()
        nan |= not bool(torch.isfinite(got).all())
        diff = (got - ref).abs()
        if name == "selected" and inp.labels is not None:
            lab = inp.labels.to(diff.device) != IGNORE_INDEX
            report["selected_ignored_zero"] = bool((got[~lab] == 0).all())
            diff = diff[lab]
        report[f"{name}_max_abs"] = float(diff.max()) if diff.numel() else 0.0
    for name in ("dX", "dW"):
        got, ref = result.get(name), oracle.get(name)
        if got is not None and ref is not None:
            report[f"{name}_rel_l2"] = rel_l2(got, ref)
            nan |= not bool(torch.isfinite(got.detach().float()).all())
    report["nan"] = bool(nan)
    return report


def gate_failures(errors: dict, b0_errors: dict) -> list[str]:
    """Metrics whose error exceeds ``GATE_MARGIN * B0 + GATE_TINY`` (empty: every gate holds)."""
    bad = []
    for key, tiny in GATE_TINY.items():
        if key in errors and key in b0_errors:
            if not errors[key] <= GATE_MARGIN * b0_errors[key] + tiny:
                bad.append(
                    f"{key}={errors[key]:.3e} > {GATE_MARGIN} x {b0_errors[key]:.3e} + {tiny:.0e}"
                )
    return bad
