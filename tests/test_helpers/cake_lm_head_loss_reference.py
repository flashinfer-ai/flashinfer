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

Seeded inputs, the unchunked PyTorch baseline and the chunked FP64 oracle of
the chunked LM-head + loss training problem (``X [T, H]`` BF16 hidden states,
``W [V, H]`` BF16 output weight, int64 labels with ``-100`` = ignored), shared
by ``tests/experimental/test_cake_lm_head_loss.py`` and
``benchmarks/bench_cake_lm_head_loss.py``.

Contract reproduced here (see ``flashinfer.chunked_lm_head``):

* ``z = X @ W^T`` is a BF16 GEMM (BF16 output of an FP32 accumulation),
  promoted to FP32; ``lse_t = logsumexp_v z[t, v]``; ``logp_t = z[t, y_t] -
  lse_t`` on valid rows and 0 on ignored rows.
* Cross-entropy: ``loss = -sum(logp[valid]) / loss_div`` (``loss_div`` a
  caller-supplied positive scalar); ``d_t = -1 / loss_div`` on valid rows.
* Policy: ``ratio_t = exp(logp_t - infer_logp_t)``; ``loss = -sum(w_t *
  min(ratio_t, 2))`` over valid rows; ``d_t = -w_t * ratio_t`` when ``ratio_t
  <= 2`` and 0 above the clip.
* Log-probability entry: the downstream loss is ``(logp * dlogp)[valid].sum()``
  with the incoming per-row gradient ``dlogp``; ``d_t = dlogp_t``.
* ``dz[t, v] = d_t * (1[v = y_t] - softmax(z_t)_v)`` rounded to BF16;
  ``dX = dz @ W`` (FP32 accumulation, one BF16 cast); ``dW = dz^T @ X``
  accumulated in FP32 and cast once (BF16 by default, FP32 on request).

The FP64 oracle applies the analytic gradient to the same BF16 input values
in float64 without any intermediate rounding; the unchunked baseline is the
plain PyTorch autograd path at the contract's rounding boundaries.  The
policy inputs keep every valid row at least ``KNEE_MARGIN`` (0.05) away from
the clip knee in ``logp - infer_logp`` (a BF16 implementation moves the ratio
by about ``+-1e-3`` through the selected logit's rounding, so a row at the
knee would fall on either side of the clip); the knee itself belongs to a
dedicated test built around the implementation's own FP32 ``logp``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

IGNORE_INDEX = -100
DEFAULT_H = 6144
DEFAULT_V = 154880
DEFAULT_LOSS_DIV = 12345.0
RATIO_CLIP = 2.0
LN2 = math.log(2.0)  # ratio 2 <=> logp - infer_logp = ln 2
KNEE_MARGIN = 0.05  # every valid policy row: |logp - infer_logp - ln 2| >= KNEE_MARGIN

_ROW_CHUNK = 4096  # rows per random draw (bounds the FP32 staging of the BF16 inputs)
_BOUNDARY_FRACTION = 0.1  # of the valid policy rows forced below / above the clip each


@dataclass
class Inputs:
    """One problem: BF16 ``X [T, H]`` (any leading stride), BF16 ``W [V, H]``,
    int64 ``labels [T]`` and the objective operands."""

    X: torch.Tensor
    W: torch.Tensor
    labels: torch.Tensor
    objective: str
    loss_div: float
    infer_logp: Optional[torch.Tensor]
    loss_weights: Optional[torch.Tensor]
    dlogp: torch.Tensor  # FP32 [T] incoming gradient of the log-probability entry (0 on ignored rows)
    regime: Optional[torch.Tensor]  # policy: int8 [T]; -1 ignored, 0 below the clip, 2 above, 3 random (with margin)
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
    def valid(self) -> torch.Tensor:
        return self.labels >= 0

    def bytes_inputs(self) -> int:
        tensors = [self.X, self.W, self.labels, self.dlogp, self.infer_logp, self.loss_weights]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def _randn_rows(shape, gen, device, scale: float) -> torch.Tensor:
    out = torch.empty(*shape, dtype=torch.bfloat16, device=device)
    for r0 in range(0, shape[0], _ROW_CHUNK):
        r1 = min(shape[0], r0 + _ROW_CHUNK)
        out[r0:r1].copy_(torch.randn(r1 - r0, *shape[1:], device=device, generator=gen) * scale)
    return out


def make_weight(V: int = DEFAULT_V, H: int = DEFAULT_H, *, seed: int, device) -> torch.Tensor:
    """``W [V, H]`` BF16 ~ ``N(0, 1/H)`` from its own generator (shareable between problems)."""
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return _randn_rows((V, H), gen, device, 1.0 / math.sqrt(H))


def make_inputs(
    T: int,
    *,
    objective: str = "ce",
    seed: int,
    H: int = DEFAULT_H,
    V: int = DEFAULT_V,
    device,
    ignore_frac: float = 0.05,
    loss_div: float = DEFAULT_LOSS_DIV,
    W: Optional[torch.Tensor] = None,
    ld_pad: int = 0,
) -> Inputs:
    """Seeded inputs of one problem with ``T`` rows.

    ``X = bf16(0.5 * N(0, 1))``; with ``ld_pad > 0`` the returned ``X`` is the
    ``[:, :H]`` view of a ``[T, H + ld_pad]`` buffer (leading stride ``H +
    ld_pad``).  ``W`` comes from a separate generator seeded ``seed + 1000003``
    unless passed in.  ``round(ignore_frac * T)`` rows are ignored
    (``ignore_frac = 1.0``: every row).  ``dlogp = FP32 0.05 *
    N(0, 1)``, 0 on ignored rows.  Policy: ``infer_logp`` is the FP64 oracle
    ``logp`` (as FP32) plus ``0.5 * N(0, 1)`` noise, with about ten percent of
    the valid rows each forced below the clip (``infer = logp + 1``, ratio
    ``e^-1``) and above it (``infer = logp - 1``, ratio ``e``), recorded in
    ``regime`` (0 / 2; 3 = random); a random row whose ``logp - infer`` falls
    within ``KNEE_MARGIN`` of ``ln 2`` is pushed to the edge of that band on
    its own side, so no row sits at the clip knee.  ``loss_weights ~ N(0, 1)``
    FP32, 0 on ignored rows.  Every draw comes from one generator in a fixed
    order, so ``(T, seed)`` fixes the problem.
    """
    if objective not in ("ce", "policy"):
        raise ValueError(f"objective must be 'ce' or 'policy', got {objective!r}")
    T, H, V = int(T), int(H), int(V)
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    full = _randn_rows((T, H + int(ld_pad)), gen, device, 0.5)
    X = full[:, :H] if ld_pad else full
    labels = torch.randint(0, V, (T,), generator=gen, device=device, dtype=torch.int64)
    num_ignored = int(round(float(ignore_frac) * T))
    if num_ignored:
        order = torch.randperm(T, generator=gen, device=device)
        labels[order[:num_ignored]] = IGNORE_INDEX
    valid = labels >= 0
    zero = torch.zeros((T,), dtype=torch.float32, device=device)
    dlogp = torch.where(valid, torch.randn(T, generator=gen, device=device) * 0.05, zero)
    if W is None:
        W = make_weight(V, H, seed=seed + 1000003, device=device)
    elif tuple(W.shape) != (V, H) or W.dtype != torch.bfloat16 or W.device != device:
        raise ValueError(f"W must be a BF16 [{V}, {H}] tensor on {device}")
    infer_logp = loss_weights = regime = None
    if objective == "policy":
        logp32 = _logp_fp64(X, W, labels).float()
        # random rows: delta = logp - infer ~ 0.5 N(0, 1), pushed out of the band |delta - ln 2| < KNEE_MARGIN
        delta = torch.randn(T, generator=gen, device=device) * 0.5
        edge = LN2 + torch.where(delta >= LN2, torch.full_like(delta, KNEE_MARGIN), torch.full_like(delta, -KNEE_MARGIN))
        delta = torch.where((delta - LN2).abs() < KNEE_MARGIN, edge, delta)
        u = torch.rand(T, generator=gen, device=device)
        below = valid & (u < _BOUNDARY_FRACTION)  # ratio e^-1
        above = valid & (u >= _BOUNDARY_FRACTION) & (u < 2 * _BOUNDARY_FRACTION)  # ratio e
        delta = torch.where(below, torch.full_like(delta, -1.0), delta)
        delta = torch.where(above, torch.full_like(delta, 1.0), delta)
        infer_logp = torch.where(valid, logp32 - delta, zero).contiguous()
        regime = torch.full((T,), 3, dtype=torch.int8, device=device)
        regime[below], regime[above], regime[~valid] = 0, 2, -1
        loss_weights = torch.where(valid, torch.randn(T, generator=gen, device=device), zero).contiguous()
    return Inputs(
        X=X, W=W, labels=labels, objective=objective, loss_div=float(loss_div), infer_logp=infer_logp,
        loss_weights=loss_weights, dlogp=dlogp.contiguous(), regime=regime, seed=int(seed),
    )


# ---------------------------------------------------------------------------
# FP64 oracle
# ---------------------------------------------------------------------------


def _row_stats_fp64(x64: torch.Tensor, W64: torch.Tensor, labels: torch.Tensor):
    """``(z, lse, valid, index, logp)`` of one row chunk in float64."""
    z = x64 @ W64.t()
    lse = torch.logsumexp(z, dim=-1)
    valid = labels >= 0
    index = torch.where(valid, labels, torch.zeros_like(labels))
    zy = z.gather(1, index[:, None]).squeeze(1)
    logp = torch.where(valid, zy - lse, torch.zeros_like(lse))
    return z, lse, valid, index, logp


@torch.no_grad()
def _logp_fp64(X: torch.Tensor, W: torch.Tensor, labels: torch.Tensor, chunk_rows: int = 512) -> torch.Tensor:
    T = int(X.shape[0])
    logp = torch.empty((T,), dtype=torch.float64, device=X.device)
    W64 = W.double()
    for r0 in range(0, T, chunk_rows):
        r1 = min(T, r0 + chunk_rows)
        logp[r0:r1] = _row_stats_fp64(X[r0:r1].double(), W64, labels[r0:r1])[4]
    return logp


@torch.no_grad()
def reference_fp64(
    inp: Inputs,
    *,
    entry: str = "loss",
    need_dx: bool = True,
    need_dw: bool = True,
    chunk_rows: int = 512,
) -> dict:
    """Analytic float64 oracle from the BF16 input values, chunked over rows.

    Returns ``loss`` (FP64 0-dim), ``logp`` / ``lse`` (FP64 ``[T]``), ``dX``
    (FP64 ``[T, H]`` or ``None``) and ``dW`` (FP64 ``[V, H]`` or ``None``).
    ``entry="loss"`` applies ``inp.objective``; ``entry="logprob"`` the
    downstream loss ``(logp * dlogp)[valid].sum()``.  No ``[T, V]`` tensor is
    materialized: each chunk of ``chunk_rows`` rows forms ``dz`` in place and
    accumulates ``dW`` in float64.  ``T == 0`` and all-ignored batches give
    zeros.
    """
    if entry not in ("loss", "logprob"):
        raise ValueError("entry must be 'loss' or 'logprob'")
    X, W, labels = inp.X, inp.W, inp.labels
    T, H, V = inp.T, inp.H, inp.V
    device = X.device
    W64 = W.double()
    total = torch.zeros((), dtype=torch.float64, device=device)
    logp = torch.empty((T,), dtype=torch.float64, device=device)
    lse = torch.empty((T,), dtype=torch.float64, device=device)
    dX = torch.empty((T, H), dtype=torch.float64, device=device) if need_dx else None
    dW = torch.zeros((V, H), dtype=torch.float64, device=device) if need_dw else None
    for r0 in range(0, T, chunk_rows):
        r1 = min(T, r0 + chunk_rows)
        x64 = X[r0:r1].double()
        z, l, valid, index, lp = _row_stats_fp64(x64, W64, labels[r0:r1])
        logp[r0:r1], lse[r0:r1] = lp, l
        zero = torch.zeros_like(lp)
        if entry == "logprob":
            g = inp.dlogp[r0:r1].double()
            total += (lp * g)[valid].sum()
            d = torch.where(valid, g, zero)
        elif inp.objective == "ce":
            total += -lp[valid].sum()
            d = torch.where(valid, torch.full_like(lp, -1.0 / inp.loss_div), zero)
        else:
            ratio = torch.exp(lp - inp.infer_logp[r0:r1].double())
            w = inp.loss_weights[r0:r1].double()
            total += -(w * torch.clamp_max(ratio, RATIO_CLIP))[valid].sum()
            d = torch.where(valid & (ratio <= RATIO_CLIP), -w * ratio, zero)
        if need_dx or need_dw:
            dz = torch.exp(z - l[:, None]).neg_()  # -softmax
            dz.scatter_add_(1, index[:, None], torch.ones((r1 - r0, 1), dtype=torch.float64, device=device))
            dz.mul_(d[:, None])  # ignored rows: d = 0
            if need_dx:
                dX[r0:r1] = dz @ W64
            if need_dw:
                dW.add_(dz.t() @ x64)
            del dz
        del z
    loss = total / inp.loss_div if (entry == "loss" and inp.objective == "ce") else total
    return dict(loss=loss, logp=logp, lse=lse, dX=dX, dW=dW)


# ---------------------------------------------------------------------------
# Unchunked PyTorch baseline (B0)
# ---------------------------------------------------------------------------


def reference_unchunked(
    inp: Inputs,
    *,
    entry: str = "loss",
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    need_dx: bool = True,
    need_dw: bool = True,
) -> dict:
    """The plain autograd path at the contract's rounding boundaries: BF16
    ``z = X @ W^T`` promoted to FP32, ``torch.logsumexp``, gather, the objective,
    ``loss.backward()``.  Returns ``loss`` (FP32 0-dim), ``logp`` (FP32 ``[T]``),
    ``dX`` (BF16 or ``None``) and ``dW`` (BF16 by construction, upcast to
    ``grad_weight_dtype`` when FP32 is requested, or ``None``)."""
    if entry not in ("loss", "logprob"):
        raise ValueError("entry must be 'loss' or 'logprob'")
    X = inp.X.detach().requires_grad_(bool(need_dx))
    W = inp.W.detach().requires_grad_(bool(need_dw))
    labels = inp.labels
    valid = labels >= 0
    with torch.enable_grad():
        z = (X @ W.t()).float()
        lse = torch.logsumexp(z, dim=-1)
        index = torch.where(valid, labels, torch.zeros_like(labels))
        zy = z.gather(1, index[:, None]).squeeze(1)
        logp = torch.where(valid, zy - lse, torch.zeros_like(lse))
        if entry == "logprob":
            loss = (logp * inp.dlogp)[valid].sum()
        elif inp.objective == "ce":
            loss = -logp[valid].sum() / inp.loss_div
        else:
            ratio = torch.exp(logp - inp.infer_logp)
            loss = -(inp.loss_weights * torch.clamp_max(ratio, RATIO_CLIP))[valid].sum()
        if need_dx or need_dw:
            loss.backward()
    dW = None if W.grad is None else W.grad.detach().to(grad_weight_dtype)
    return dict(loss=loss.detach().float(), logp=logp.detach(), dX=None if X.grad is None else X.grad.detach(), dW=dW)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    """``||a - ref||_2 / max(||ref||_2, 1e-30)`` in float64."""
    a = a.detach().double().reshape(-1)
    ref = ref.detach().double().reshape(-1)
    return (a - ref).norm().item() / max(ref.norm().item(), 1e-30)


def _finite(t: Optional[torch.Tensor]) -> bool:
    return t is None or bool(torch.isfinite(t.detach().float()).all())


def error_report(result: dict, oracle: dict, labels: torch.Tensor) -> dict:
    """Errors of ``result`` (``loss`` / ``logp`` / ``dX`` / ``dW``) against the
    FP64 ``oracle``: ``loss_rel``, ``logp_max_abs`` and ``logp_mean_abs`` over the
    valid rows, ``logp_ignored_zero``, ``dX_rel_l2`` / ``dW_rel_l2`` when both
    sides carry them, ``dX_ignored_zero`` and ``nan`` (any non-finite value)."""
    valid = labels >= 0
    report: dict = {}
    nan = False
    loss, loss64 = result.get("loss"), oracle.get("loss")
    if loss is not None and loss64 is not None:
        l, l64 = float(loss), float(loss64)
        report["loss_rel"] = abs(l - l64) / max(abs(l64), 1e-30)
        nan |= not math.isfinite(l)
    logp = result["logp"].detach().double()
    diff = (logp - oracle["logp"]).abs()
    report["logp_max_abs"] = float(diff[valid].max()) if bool(valid.any()) else 0.0
    report["logp_mean_abs"] = float(diff[valid].mean()) if bool(valid.any()) else 0.0
    report["logp_ignored_zero"] = bool((logp[~valid] == 0).all())
    nan |= not _finite(result["logp"])
    for name in ("dX", "dW"):
        got, ref = result.get(name), oracle.get(name)
        if got is not None and ref is not None:
            report[f"{name}_rel_l2"] = rel_l2(got, ref)
        nan |= not _finite(got)
    if result.get("dX") is not None:
        report["dX_ignored_zero"] = bool((result["dX"].detach()[~valid] == 0).all())
    report["nan"] = bool(nan)
    return report


def gate_ratios(errors: dict, b0_errors: dict) -> dict:
    """Per numeric metric ``errors[k] / b0_errors[k]`` (``inf`` when the baseline
    is exact and the result is not, 0 when both are exact)."""
    ratios = {}
    for key, value in errors.items():
        base = b0_errors.get(key)
        if isinstance(value, bool) or base is None or isinstance(base, bool):
            continue
        if value == 0 and base == 0:
            ratios[key] = 0.0
        elif base == 0:
            ratios[key] = math.inf
        else:
            ratios[key] = float(value) / float(base)
    return ratios
