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

Seeded inputs and the exact per-document reference of the packed KDA chunked
training problem (GLM-5.3-Flash geometry: head dimension 128, natural-log
decay gate, sigmoid beta), shared by ``tests/experimental/test_cake_kda_det_train.py``,
``tests/experimental/test_cake_kda_det_train_api.py``,
``tests/experimental/test_cake_kda_det_train_raw_gate_accuracy.py`` and
``benchmarks/bench_cake_kda_det_train.py`` (and, through the previous module path
``tests/test_helpers/cake_kda_train_reference.py``, by the KCP suites).

The reference is the token recurrence of one document, in FP64 with autograd,
state ``S`` in ``R^{K x V}`` starting from zero::

    S   = S * exp(g_t)[:, None]                   # per-key-dimension decay
    S   = S + (beta_t * k_t) (x) (v_t - k_t^T S)  # delta rule
    o_t = scale * q_t^T S

with the BF16 ``q``, ``k``, ``v``, ``beta`` and the FP32 ``g`` taken as exact
FP64 values.  The recurrence is checkpointed in 64-token blocks so the
backward of a few thousand tokens fits in memory.

The raw-gate form (``use_gate_in_kernel=True``) takes the BF16 raw gate, the
FP32 ``A_log [H]`` and ``dt_bias [H * 128]`` and the lower bound ``lb``, and
activates the gate in the same FP64 graph::

    g = lb * sigmoid(exp(A_log)[h] * (raw + dt_bias[h]))

so autograd gives the gradients of the raw gate and of both parameters too
(:func:`kda_exact_raw_with_grads`).
"""

from __future__ import annotations

import inspect
import math
import os
import subprocess
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

D = 128
LOWER_BOUND = -5.0  # the gate's lower bound: g = -5 * sigmoid(...)
GRAD_NAMES = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd.grad order of the leaves (q, k, v, g, beta)
# the raw-gate form: autograd.grad order of the leaves (q, k, v, raw, beta, A_log, dt_bias); "dg" is the gradient
# of the BF16 raw gate there
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")


def cu_seqlens_of(lengths: Sequence[int]) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def _draw_operands(lengths: Sequence[int], H: int, seed: int | None) -> dict:
    """The FP32 draws of one problem in their fixed order (``q, k, v, a_log, dt_bias, raw, beta logits, dout``) from
    a CPU generator seeded from ``seed`` (default: a CRC of the shape), so :func:`make_inputs` and
    :func:`make_raw_inputs` of the same shape and seed see the same numbers."""
    S = int(sum(lengths))
    if seed is None:
        seed = zlib.crc32(f"{list(lengths)}:{H}".encode())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    out = {}
    out["q"] = l2norm_bf16(randn(S, H, D))
    out["k"] = l2norm_bf16(randn(S, H, D))
    out["v"] = (randn(S, H, D) * 0.5).to(torch.bfloat16)
    out["a_log"] = torch.log(
        torch.rand(H, generator=gen, dtype=torch.float32) + 1.0
    )  # decay rates in [1, 2)
    out["dt_bias"] = randn(H, D) * 0.5
    out["raw"] = randn(S, H, D)
    out["beta"] = torch.sigmoid(randn(S, H)).to(torch.bfloat16)
    out["dout"] = lambda: (randn(S, H, D) * 0.5).to(
        torch.bfloat16
    )  # drawn last, on demand
    return out


def make_inputs(
    lengths: Sequence[int],
    H: int,
    device: str | torch.device = "cuda",
    *,
    seed: int | None = None,
    with_dout: bool = True,
    batched: bool = False,
) -> dict[str, torch.Tensor]:
    """The packed GLM operands of documents of ``lengths`` tokens with ``H`` heads: ``q, k, v [S, H, 128]`` BF16
    (``q`` / ``k`` L2-normalized in FP32 with eps 1e-6 before the BF16 store), ``g [S, H, 128]`` FP32
    ``= -5 * sigmoid(exp(A_log) * (raw + dt_bias))``, ``beta [S, H]`` BF16 ``= sigmoid(logits)``, ``cu_seqlens``
    int32 on the device and (with ``with_dout``) ``dout [S, H, 128]`` BF16.  ``batched`` adds the leading batch
    axis of 1.  Device-independent: a CPU generator seeded from ``seed`` (default: a CRC of the shape)."""
    d = _draw_operands(lengths, H, seed)
    a_log, dt_bias, raw = d["a_log"], d["dt_bias"], d["raw"]
    g = LOWER_BOUND * torch.sigmoid(
        torch.exp(a_log)[None, :, None] * (raw + dt_bias[None])
    )
    cu = torch.tensor(cu_seqlens_of(lengths), dtype=torch.int32)
    out = {
        "q": d["q"],
        "k": d["k"],
        "v": d["v"],
        "g": g.float(),
        "beta": d["beta"],
        "cu_seqlens": cu,
    }
    if with_dout:
        out["dout"] = d["dout"]()
    tensors = {name: t.to(device) for name, t in out.items()}
    if batched:
        for name in ("q", "k", "v", "g", "beta", "dout"):
            if name in tensors:
                tensors[name] = tensors[name].unsqueeze(0)
    return tensors


def make_raw_inputs(
    lengths: Sequence[int],
    H: int,
    device: str | torch.device = "cuda",
    *,
    seed: int | None = None,
    with_dout: bool = True,
    batched: bool = False,
    dt_bias_mean: float = 0.0,
    dt_bias_2d: bool = False,
) -> dict[str, torch.Tensor]:
    """The raw-gate sibling of :func:`make_inputs` (the same draws in the same order, so ``q, k, v, beta, dout`` of
    the same shape and seed are bitwise :func:`make_inputs`'s): ``g`` is the raw gate ``raw.bfloat16()`` ``[S, H,
    128]`` (the BF16 operand of ``use_gate_in_kernel=True``, not an activated gate), ``A_log`` FP32 ``[H]`` and
    ``dt_bias`` FP32 ``[H * 128]`` (``[H, 128]`` with ``dt_bias_2d``), the draw shifted by ``dt_bias_mean`` (``+6``:
    the fast-decay rows, ``s = sigmoid(...) -> 1``; ``-6``: the slow-decay rows, ``g ~ -0.012``).  The activated
    gate of these operands is :func:`raw_gate` (``lower_bound`` :data:`LOWER_BOUND`)."""
    d = _draw_operands(lengths, H, seed)
    dt_bias = d["dt_bias"]
    if dt_bias_mean != 0.0:
        dt_bias = dt_bias + float(dt_bias_mean)
    cu = torch.tensor(cu_seqlens_of(lengths), dtype=torch.int32)
    out = {
        "q": d["q"],
        "k": d["k"],
        "v": d["v"],
        "g": d["raw"].to(torch.bfloat16),
        "beta": d["beta"],
        "cu_seqlens": cu,
        "A_log": d["a_log"].contiguous(),
        "dt_bias": (dt_bias if dt_bias_2d else dt_bias.flatten()).contiguous(),
    }
    if with_dout:
        out["dout"] = d["dout"]()
    tensors = {name: t.to(device) for name, t in out.items()}
    if batched:
        for name in ("q", "k", "v", "g", "beta", "dout"):
            if name in tensors:
                tensors[name] = tensors[name].unsqueeze(0)
    return tensors


def raw_gate(
    raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = LOWER_BOUND,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The activated gate ``lower_bound * sigmoid(exp(A_log)[h] * (raw + dt_bias[h]))`` of a raw gate ``[..., H,
    128]`` in ``dtype`` (``float32``: the explicit torch gate of the training stack; ``float64``: the reference),
    differentiable in ``raw``, ``A_log`` and ``dt_bias`` (``[H * 128]`` or ``[H, 128]``)."""
    H = raw.shape[-2]
    x = raw.to(dtype)
    a = torch.exp(A_log.to(dtype)).reshape(H, 1)
    b = dt_bias.to(dtype).reshape(H, D)
    return lower_bound * torch.sigmoid(a * (x + b))


GateFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
FLA_GATE_ATOL = 1e-5  # the external gate against raw_gate in FP32: the same formula, a few FP32 ulps apart


def fla_kda_gate(lower_bound: float = LOWER_BOUND) -> tuple[GateFn | None, dict]:
    """flash-linear-attention's KDA gate with the lower bound (``fla.ops.kda.gate.fused_kda_gate``), adapted to
    the ``[S, H, 128]`` raw gate, FP32 ``A_log [H]`` and ``dt_bias [H * 128]``: ``(gate, info)``, or ``(None,
    info)`` with the reason when it does not import or has no lower-bound form.  ``info`` records the module's
    provenance (version, file, the git commit of its checkout when it is one, ``FLA_COMMIT``, the signature).
    Check the adapter with :func:`check_fla_gate` before using it."""
    info: dict = {"commit_env": os.environ.get("FLA_COMMIT")}
    try:
        import fla
        from fla.ops.kda import gate as fla_gate_module
    except Exception as exc:  # the arm is optional
        info["unavailable"] = f"import: {exc!r}"
        return None, info
    info["version"] = getattr(fla, "__version__", None)
    info["file"] = getattr(fla, "__file__", None)
    info["git_commit"] = None
    if info["file"]:
        # only a checkout whose root holds the package (repo/fla/__init__.py), not an enclosing repository
        pkg = Path(info["file"]).resolve().parent
        try:
            res = subprocess.run(
                ["git", "-C", str(pkg), "rev-parse", "--show-toplevel", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            top, _, head = res.stdout.strip().partition("\n")
            if res.returncode == 0 and Path(top).resolve() == pkg.parent:
                info["git_commit"] = head.strip() or None
        except Exception:
            pass
    fn = getattr(fla_gate_module, "fused_kda_gate", None)
    if fn is None:
        info["unavailable"] = "fla.ops.kda.gate has no fused_kda_gate"
        return None, info
    params = inspect.signature(fn).parameters
    info["signature"] = str(inspect.signature(fn))
    if "lower_bound" not in params:
        info["unavailable"] = (
            "fused_kda_gate takes no lower_bound (no bounded sigmoid gate)"
        )
        return None, info
    bias_name = next((n for n in ("dt_bias", "g_bias", "bias") if n in params), None)
    if bias_name is None:
        info["unavailable"] = f"fused_kda_gate has no bias parameter: {list(params)}"
        return None, info

    def gate(raw, a_log, dt_bias):
        H, D_ = raw.shape[-2], raw.shape[-1]
        kwargs = {bias_name: dt_bias.reshape(H * D_), "lower_bound": lower_bound}
        if "head_k_dim" in params:
            kwargs["head_k_dim"] = D_
        if "output_dtype" in params:
            kwargs["output_dtype"] = torch.float32
        g = fn(raw.unsqueeze(0), a_log, **kwargs)
        return g.reshape(raw.shape).float()

    return gate, info


def check_fla_gate(
    gate: GateFn, inputs: dict, lower_bound: float = LOWER_BOUND
) -> str | None:
    """``None`` when ``gate`` gives :func:`raw_gate`'s FP32 values on ``inputs`` (``g`` the raw gate, ``A_log``,
    ``dt_bias``) within :data:`FLA_GATE_ATOL`, else the reason the adapted gate is not that computation."""
    try:
        with torch.no_grad():
            got = gate(inputs["g"], inputs["A_log"], inputs["dt_bias"])
            want = raw_gate(
                inputs["g"],
                inputs["A_log"],
                inputs["dt_bias"],
                lower_bound,
                torch.float32,
            )
    except Exception as exc:
        return f"call: {exc!r}"
    if got.shape != want.shape:
        return f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    err = (
        (got.float() - want).abs().max().item()
    )  # FP32: the values are within a factor 2, exact
    if not err <= FLA_GATE_ATOL:
        return f"values differ from lb * sigmoid(exp(A_log) * (raw + dt_bias)) by {err}"
    return None


def _block(S0, q, k, v, g, beta, scale):
    # S0 [H, K, V]; q, k, v, g [T, H, D]; beta [T, H]  (all FP64)
    St = S0
    outs = []
    for t in range(q.shape[0]):
        St = St * torch.exp(g[t])[:, :, None]
        kt = k[t]
        pred = torch.einsum("hk,hkv->hv", kt, St)
        St = St + torch.einsum("hk,hv->hkv", beta[t][:, None] * kt, v[t] - pred)
        outs.append(scale * torch.einsum("hk,hkv->hv", q[t], St))
    return torch.stack(outs, 0), St


def kda_exact_forward(
    q, k, v, g, beta, cu_seqlens, scale, block=64, dtype=torch.float64
):
    """``q, k, v, g [S, H, D]``, ``beta [S, H]``, ``cu_seqlens`` 1-D int (device or host): ``out [S, H, D]`` in
    ``dtype`` from the exact per-document recurrence."""
    S, H, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(S, H, V, dtype=dtype, device=q.device)
    cu = [int(x) for x in cu_seqlens.tolist()]
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        St = torch.zeros(H, K, V, dtype=dtype, device=q.device)
        for b0 in range(s0, s1, block):
            b1 = min(b0 + block, s1)
            ob, St = checkpoint(
                _block,
                St,
                q[b0:b1],
                k[b0:b1],
                v[b0:b1],
                g[b0:b1],
                beta[b0:b1],
                scale,
                use_reentrant=False,
            )
            o[b0:b1] = ob
    return o


def kda_exact_with_grads(
    q, k, v, g, beta, dout, cu_seqlens, scale, dtype=torch.float64
):
    """The exact forward output and the five input gradients (``GRAD_NAMES`` order), all in ``dtype``, from the
    GLM operands (BF16 ``q, k, v, beta, dout``, FP32 ``g``) of shape ``[S, H, D]`` / ``[S, H]``."""
    leaves = [x.detach().to(dtype).requires_grad_(True) for x in (q, k, v, g, beta)]
    o = kda_exact_forward(*leaves, cu_seqlens, scale, dtype=dtype)
    grads = torch.autograd.grad(o, leaves, dout.to(dtype))
    return o.detach(), dict(zip(GRAD_NAMES, (x.detach() for x in grads), strict=True))


def kda_exact_raw_with_grads(
    q,
    k,
    v,
    raw,
    beta,
    A_log,
    dt_bias,
    dout,
    cu_seqlens,
    scale,
    lower_bound: float = LOWER_BOUND,
    dtype=torch.float64,
):
    """The exact forward output and the seven input gradients (:data:`RAW_GRAD_NAMES` order: ``dq, dk, dv, dg``
    (the raw gate's), ``dbeta, dA_log, ddt_bias``), all in ``dtype``, from the raw-gate operands (BF16 ``q, k, v,
    raw, beta, dout``, FP32 ``A_log`` / ``dt_bias``) taken as exact ``dtype`` leaves; the gate is activated in the
    same graph (:func:`raw_gate`), so the parameter gradients are the exact sums.  ``ddt_bias`` has ``dt_bias``'s
    shape."""
    leaves = [
        x.detach().to(dtype).requires_grad_(True)
        for x in (q, k, v, raw, beta, A_log, dt_bias)
    ]
    q64, k64, v64, raw64, beta64, a64, b64 = leaves
    g64 = raw_gate(raw64, a64, b64, lower_bound, dtype)
    o = kda_exact_forward(q64, k64, v64, g64, beta64, cu_seqlens, scale, dtype=dtype)
    grads = torch.autograd.grad(o, leaves, dout.to(dtype))
    return o.detach(), dict(
        zip(RAW_GRAD_NAMES, (x.detach() for x in grads), strict=True)
    )


def error_metrics(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """max |err|, RMS err, relative RMS (err RMS / ref RMS) and the reference RMS, computed in FP64."""
    c = cand.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    err = c - r
    rms_ref = torch.sqrt(torch.mean(r * r)).item() if r.numel() else 0.0
    rms_err = torch.sqrt(torch.mean(err * err)).item() if err.numel() else 0.0
    return {
        "max_abs": err.abs().max().item() if err.numel() else 0.0,
        "rms": rms_err,
        "rel_rms": (rms_err / rms_ref)
        if rms_ref > 0
        else float("inf")
        if rms_err > 0
        else 0.0,
        "ref_rms": rms_ref,
    }


def elementwise_gate(
    cand: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float
) -> dict:
    """``|cand - ref| <= atol + rtol * |ref|`` elementwise in FP32; the mismatch and non-finite counts."""
    c = cand.detach().to(torch.float32)
    r = ref.detach().to(torch.float32)
    bad = (c - r).abs() > (atol + rtol * r.abs())
    nonfinite = int((~torch.isfinite(c)).sum().item())
    mismatch = int(bad.sum().item())
    return {
        "mismatch": mismatch,
        "numel": int(c.numel()),
        "nonfinite": nonfinite,
        "pass": mismatch == 0 and nonfinite == 0,
    }


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """The CDF of Student's t distribution with ``df`` degrees of freedom."""
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: int) -> float:
    """The ``p`` quantile of Student's t distribution (bisection on :func:`t_cdf`, to double precision)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_quantile: p must be in (0, 1), got {p}")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def upper_t_bound(values: Sequence[float], level: float = 0.975) -> float:
    """``mean + t_{level, N - 1} * sd / sqrt(N)`` of ``values`` (``sd`` with ``N - 1``): the upper end of the
    two-sided ``2 * level - 1`` t-interval of their mean (``level=0.975``: the 95 % interval)."""
    n = len(values)
    if n < 2:
        raise ValueError("upper_t_bound: needs at least two values")
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    return mean + t_quantile(level, n - 1) * sd / math.sqrt(n)
