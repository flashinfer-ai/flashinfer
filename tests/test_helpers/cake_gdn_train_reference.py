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

Seeded operands and the exact FP64 reference of the packed GDN (gated delta
rule) training problem with grouped Q/K heads, a scalar natural-log gate and an
FP32 ``beta``, shared by ``tests/experimental/test_cake_gdn_train*.py`` and the
kernel tests that cite this text as their oracle.

Semantics (FLA ``naive_recurrent_gated_delta_rule`` convention): per token
``t`` of ONE document, per value head ``hv`` with state ``S`` in ``R^{K x V}``::

    S   = exp(g_t[hv]) * S                          # scalar decay of the whole state
    S   = S + beta_t[hv] * k_t (x) (v_t - S^T k_t)  # delta rule; S^T k = sum_K S[K, V] k[K]
    o_t = scale * S^T q_t                           # scale = 128 ** -0.5

The Q/K head of value head ``hv`` is ``hv // (HV // H_qk)`` (grouped value
heads); the BF16 ``q`` / ``k`` (L2-normalised rows as stored, not re-normalised)
/ ``v`` / ``dout`` and the FP32 ``g`` / ``beta`` are taken as exact FP64 values.
Every document starts from the zero state unless an initial state ``h0`` is
given for a token range that starts inside a document.  Autograd in FP64 gives
``dq``, ``dk`` (the value-head group summed in FP64), ``dv``, ``dg``, ``dbeta``
and the cotangent ``dh0`` of the initial state; the recurrence is checkpointed
in 64-token blocks so a 131,071-token document fits.

API (operands with or without the batch axis of 1; ``q/k [T, H_qk, 128]``,
``v/dout [T, HV, 128]``, ``g/beta [T, HV]``; the tensors may be the global pack
or exactly the requested ``token_range``):

* ``make_inputs(cu_seqlens, H_qk, HV, *, regime, seed, with_conv, token_range)``
  -> seeded CPU operands (regimes ``normal`` and the adversarial ``A1``-``A4``);
  a token's operands depend only on (seed, heads, document, 64-token block), so
  ``token_range`` slices are bitwise the slices of the global pack.
* ``gdn_exact_forward(...)`` -> ``(out, final_state)``;
  ``gdn_exact_with_grads(...)`` -> ``out, final_state, dq, dk, dv, dg, dbeta, dh0``
  (``h0`` / ``dfinal`` give any fragment's forward and adjoint oracle).
* ``cut_states`` (state carried out of the tokens before each cut),
  ``fragment_map`` (``M`` of the affine map ``S_out = M S_in + B`` of a range),
  ``carry_diagnostic`` (``||M_f||_2`` and the carried fraction per fragment).
* ``conv_exact`` / ``conv_exact_with_grads``: per-document width-4 causal
  depthwise conv + SiLU in FP64 with history rows and ``dhistory``.
* ``error_metrics`` / ``elementwise_mismatch`` / ``ulp_storage`` /
  ``judge_tensor`` / ``compare_to_reference`` (rel-RMS, max-abs, the one
  storage-ulp max-abs clause) and ``reference_pack`` (whole pack, per document).

Provenance: port of the calibration-harness modules ``gdn_inputs.py``
(sha256 6bd0f61e3e02287645244256b94e3fc4b54249c8b8d9377635f2f491f808ee1e) and
``gdn_reference.py`` (sha256 7a59fac9c390fff4539cf1d0eb2c674a04f2add3236656ce079235c512506498);
the function bodies are verbatim and every output is bitwise identical to the
harness (checked by the lane self-test); the docstrings and comments are
reworded to be self-contained, the only code edits are lint-only
(``zip(..., strict=True)``, ``off >= 0``, an unused loop variable renamed), and
the harness disk-cache helpers are dropped.
"""

from __future__ import annotations

import hashlib
import math
import time

import torch
from torch.utils.checkpoint import checkpoint

# ----------------------------------------------------------------------------------------------------------------
# seeded operands

SEED = 20261023
D = 128
C_CONV = 10240
CONV_WIDTH = 4
CONV_W_STD = 0.5
BLOCK = 64
EPS = 1e-6
REGIMES = ("normal", "A1", "A2", "A3", "A4")
NORMAL_A_MAX = math.log(16.0)  # A ~ U(0, log 16): decay rate exp(A) in [1, 16)
ADV_GMAX = 1e-5  # A1/A2/A4: g = -ADV_GMAX * U[0, 1)
ADV_BETA0 = {"A1": 0.02, "A2": 0.02, "A4": 0.008}
ADV_BETA_SPREAD = 0.2  # beta = sigmoid(logit(b0) + 0.2 N)
A2_NOISE = 0.02  # k = k_raw * mask + 0.02 * noise (before L2 normalisation)
A2_DIMS = 4  # key energy concentrated on 4 fixed dims per Q/K head
A2_GZERO_FRACTION = 0.10  # 10 % of tokens with g == 0 exactly
GDN_KEYS = ("q", "k", "v", "g", "beta", "dout")
CONV_KEYS = ("conv_x", "conv_w", "conv_dy")
CONV_TOKEN_KEYS = ("conv_x", "conv_dy")


def _seed64(*parts) -> int:
    key = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def _gen(*parts) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_seed64(*parts))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def doc_regime(regime: str, doc: int) -> str:
    """The regime family applied to document `doc`: normal | A1 | A2 | A4 (A3 -> A1 for document 0, normal otherwise)."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if regime == "A3":
        return "A1" if doc == 0 else "normal"
    return regime


def head_params(H_qk: int, HV: int, seed: int = SEED) -> dict[str, torch.Tensor]:
    """Per-head constants shared by every block: a_log [HV] (normal regime decay rates, A ~ U(0, log 16)) and
    aniso_dims [H_qk, 4] (A2 key dimensions).  Seeded from (seed, H_qk, HV) only."""
    g = _gen(seed, "heads", H_qk, HV)
    a_log = torch.rand(HV, generator=g, dtype=torch.float32) * NORMAL_A_MAX
    dims = torch.stack(
        [torch.randperm(D, generator=g)[:A2_DIMS] for _ in range(H_qk)]
    )  # [H_qk, 4]
    return {"a_log": a_log, "aniso_dims": dims}


def conv_weight(seed: int = SEED, C: int = C_CONV) -> torch.Tensor:
    """FP32 taps [C, 4] ~ N(0, 0.5^2); seeded from (seed, 'conv_w', C) only (independent of the pack and heads)."""
    g = _gen(seed, "conv_w", C)
    return torch.randn(C, CONV_WIDTH, generator=g, dtype=torch.float32) * CONV_W_STD


def _l2norm(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)).to(dtype)


def doc_block(
    regime: str,
    H_qk: int,
    HV: int,
    doc: int,
    block: int,
    n: int,
    params: dict[str, torch.Tensor],
    *,
    seed: int = SEED,
    dtype_qk: torch.dtype = torch.bfloat16,
    with_conv: bool = False,
) -> dict[str, torch.Tensor]:
    """Operands of tokens [block * 64, block * 64 + n) of document `doc` (CPU, no batch axis): q, k [n, H_qk, 128],
    v, dout [n, HV, 128], g, beta [n, HV], (conv_x [n, 10240]).  `regime` is the pack regime (A3 is resolved here)."""
    fam = doc_regime(regime, doc)
    gen = _gen(seed, H_qk, HV, "doc", doc, "block", block)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, dtype=torch.float32)

    # fixed draw order for every regime
    q_raw = randn(n, H_qk, D)
    k_raw = randn(n, H_qk, D)
    k_noise = randn(n, H_qk, D)
    v = randn(n, HV, D)
    x_gate = randn(n, HV)
    u_gate = rand(n, HV)
    u_zero = rand(n)
    logits = randn(n, HV)
    dout = randn(n, HV, D)
    conv_x = randn(n, C_CONV) if with_conv else None
    conv_dy = randn(n, C_CONV) if with_conv else None

    if fam == "A2":
        mask = torch.zeros(H_qk, D, dtype=torch.float32)
        mask.scatter_(1, params["aniso_dims"], 1.0)
        k_raw = k_raw * mask[None] + A2_NOISE * k_noise
    q = _l2norm(q_raw, dtype_qk)
    k = _l2norm(k_raw, dtype_qk)
    if fam == "normal":
        g = -torch.exp(params["a_log"])[None, :] * torch.nn.functional.softplus(x_gate)
        beta = torch.sigmoid(logits)
    else:
        g = -ADV_GMAX * u_gate
        if fam == "A2":
            g[u_zero < A2_GZERO_FRACTION] = 0.0
        beta = torch.sigmoid(_logit(ADV_BETA0[fam]) + ADV_BETA_SPREAD * logits)
    out = {
        "q": q,
        "k": k,
        "v": v.to(torch.bfloat16),
        "g": g.float().contiguous(),
        "beta": beta.float().contiguous(),
        "dout": dout.to(torch.bfloat16),
    }
    if with_conv:
        out["conv_x"] = conv_x.to(torch.bfloat16)
        out["conv_dy"] = conv_dy.to(torch.bfloat16)
    return out


def _empty(
    H_qk: int, HV: int, dtype_qk: torch.dtype, with_conv: bool
) -> dict[str, torch.Tensor]:
    out = {
        "q": torch.empty(0, H_qk, D, dtype=dtype_qk),
        "k": torch.empty(0, H_qk, D, dtype=dtype_qk),
        "v": torch.empty(0, HV, D, dtype=torch.bfloat16),
        "g": torch.empty(0, HV, dtype=torch.float32),
        "beta": torch.empty(0, HV, dtype=torch.float32),
        "dout": torch.empty(0, HV, D, dtype=torch.bfloat16),
    }
    if with_conv:
        out["conv_x"] = torch.empty(0, C_CONV, dtype=torch.bfloat16)
        out["conv_dy"] = torch.empty(0, C_CONV, dtype=torch.bfloat16)
    return out


def doc_range(
    regime: str,
    H_qk: int,
    HV: int,
    doc: int,
    off: int,
    length: int,
    doc_len: int,
    params: dict[str, torch.Tensor],
    *,
    seed: int = SEED,
    dtype_qk: torch.dtype = torch.bfloat16,
    with_conv: bool = False,
) -> dict[str, torch.Tensor]:
    """Operands of document `doc` tokens [off, off + length) (CPU, no batch axis), assembled from 64-token blocks."""
    if length <= 0:
        return _empty(H_qk, HV, dtype_qk, with_conv)
    if not (off >= 0 and off + length <= doc_len):
        raise ValueError(
            f"doc {doc}: range [{off}, {off + length}) outside the document of {doc_len} tokens"
        )
    parts: dict[str, list[torch.Tensor]] = {}
    b0, b1 = off // BLOCK, (off + length - 1) // BLOCK
    for b in range(b0, b1 + 1):
        n = min(BLOCK, doc_len - b * BLOCK)
        blk = doc_block(
            regime,
            H_qk,
            HV,
            doc,
            b,
            n,
            params,
            seed=seed,
            dtype_qk=dtype_qk,
            with_conv=with_conv,
        )
        lo = max(off, b * BLOCK) - b * BLOCK
        hi = min(off + length, b * BLOCK + n) - b * BLOCK
        for name, t in blk.items():
            parts.setdefault(name, []).append(t[lo:hi])
    return {name: torch.cat(ts, 0) for name, ts in parts.items()}


def make_inputs(
    cu_seqlens,
    H_qk: int,
    HV: int,
    *,
    regime: str = "normal",
    seed: int = SEED,
    dtype_qk: torch.dtype = torch.bfloat16,
    with_conv: bool = False,
    token_range=None,
) -> dict:
    """The packed operands of the documents `cu_seqlens` (list of ints or 1-D int tensor, global order) in the contract
    layout (batch axis of 1 on q/k/v/g/beta/dout; conv_x [T, 10240], conv_w [10240, 4]); `token_range=(a, b)` returns
    exactly the global tokens [a, b) (bitwise the slice of the full pack)."""
    cu = [
        int(x)
        for x in (
            cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens
        )
    ]
    if not cu or cu[0] != 0 or any(cu[i + 1] < cu[i] for i in range(len(cu) - 1)):
        raise ValueError(
            f"cu_seqlens must start at 0 and be non-decreasing, got {cu[:8]}..."
        )
    if HV % H_qk != 0:
        raise ValueError(f"HV ({HV}) must be a multiple of H_qk ({H_qk})")
    total = cu[-1]
    a, b = (
        (0, total)
        if token_range is None
        else (int(token_range[0]), int(token_range[1]))
    )
    if not (0 <= a <= b <= total):
        raise ValueError(f"token_range {token_range} outside [0, {total}]")
    params = head_params(H_qk, HV, seed)
    parts: dict[str, list[torch.Tensor]] = {}
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        lo, hi = max(a, s0), min(b, s1)
        if hi <= lo:
            continue
        ops = doc_range(
            regime,
            H_qk,
            HV,
            d,
            lo - s0,
            hi - lo,
            s1 - s0,
            params,
            seed=seed,
            dtype_qk=dtype_qk,
            with_conv=with_conv,
        )
        for name, t in ops.items():
            parts.setdefault(name, []).append(t)
    if parts:
        out = {name: torch.cat(ts, 0) for name, ts in parts.items()}
    else:
        out = _empty(H_qk, HV, dtype_qk, with_conv)
    for name in GDN_KEYS:
        out[name] = out[name].unsqueeze(0).contiguous()
    if with_conv:
        out["conv_x"] = out["conv_x"].contiguous()
        out["conv_dy"] = out["conv_dy"].contiguous()
        out["conv_w"] = conv_weight(seed)
    out["cu_seqlens"] = torch.tensor(cu, dtype=torch.int64)
    out["token_range"] = (a, b)
    out["regime"] = regime
    out["seed"] = seed
    return out


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of the raw little-endian bytes of a CPU tensor (any dtype, BF16 included)."""
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def digests(inputs: dict) -> dict[str, str]:
    return {
        name: tensor_sha256(inputs[name])
        for name in (*GDN_KEYS, *CONV_KEYS)
        if name in inputs
    }


# ----------------------------------------------------------------------------------------------------------------
# FP64 reference

SCALE = D**-0.5
GRAD_NAMES = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd.grad order of the leaves (q, k, v, g, beta)
TENSORS = ("out",) + GRAD_NAMES
STORAGE_OF = {
    "out": "bf16",
    "dq": "bf16",
    "dk": "bf16",
    "dv": "bf16",
    "dg": "fp32",
    "dbeta": "fp32",
}  # storage format in which each tensor is compared (out/dq/dk/dv as stored BF16, dg/dbeta as FP32)
STORAGE_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}
FRACTION_BITS = {"bf16": 7, "fp32": 23}
MIN_EXP = -126
PROGRESS_BLOCKS = 64
F64 = torch.float64


# ----------------------------------------------------------------------------------------------------------------
# shapes


def _tok(t: torch.Tensor, nd: int, name: str) -> torch.Tensor:
    """Strip the batch axis of 1 from a [1, T, ...] tensor; `nd` = the rank without the batch axis."""
    if t.dim() == nd + 1:
        if t.shape[0] != 1:
            raise ValueError(
                f"{name}: batch axis must be 1, got shape {tuple(t.shape)}"
            )
        return t[0]
    if t.dim() != nd:
        raise ValueError(
            f"{name}: expected rank {nd} or {nd + 1}, got shape {tuple(t.shape)}"
        )
    return t


def _cu_list(cu_seqlens) -> list[int]:
    cu = [
        int(x)
        for x in (
            cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens
        )
    ]
    if not cu or cu[0] != 0 or any(cu[i + 1] < cu[i] for i in range(len(cu) - 1)):
        raise ValueError(f"cu_seqlens must start at 0 and be non-decreasing: {cu[:8]}")
    return cu


def _range(cu: list[int], token_range) -> tuple[int, int]:
    a, b = (
        (0, cu[-1])
        if token_range is None
        else (int(token_range[0]), int(token_range[1]))
    )
    if not (0 <= a <= b <= cu[-1]):
        raise ValueError(f"token_range {token_range} outside [0, {cu[-1]}]")
    return a, b


def _segments(cu: list[int], a: int, b: int) -> list[tuple[int, int, int]]:
    """(doc, global_lo, global_hi) of every non-empty document portion inside [a, b)."""
    segs = []
    for d in range(len(cu) - 1):
        lo, hi = max(a, cu[d]), min(b, cu[d + 1])
        if hi > lo:
            segs.append((d, lo, hi))
    return segs


def _doc_start_of(cu: list[int], a: int) -> int:
    """Global start of the document containing token a (a itself when a is a document boundary)."""
    for d in range(len(cu) - 1):
        if cu[d] <= a < cu[d + 1]:
            return cu[d]
    return a


def _slice_to_range(
    t: torch.Tensor, total: int, a: int, b: int, name: str
) -> torch.Tensor:
    """`t` is either exactly the range [a, b) (b - a tokens) or the global pack (total tokens, sliced here)."""
    if t.shape[0] == b - a:
        return t
    if t.shape[0] == total:
        return t[a:b]
    raise ValueError(
        f"{name}: {t.shape[0]} tokens given; expected the global pack ({total}) or the range ({b - a})"
    )


# ----------------------------------------------------------------------------------------------------------------
# GDN recurrence


def _block(S0, qe, ke, v, g, beta, scale, want_out: bool = True):
    # S0 [HV, K, V]; qe, ke [n, HV, K] (expanded to value heads); v [n, HV, V]; g, beta [n, HV]  (all FP64)
    St = S0
    outs = []
    for t in range(ke.shape[0]):
        St = St * torch.exp(g[t])[:, None, None]
        kt = ke[t]
        pred = (kt[:, :, None] * St).sum(1)  # S^T k -> [HV, V]
        St = St + (beta[t][:, None] * kt)[:, :, None] * (v[t] - pred)[:, None, :]
        if want_out:
            outs.append(scale * (qe[t][:, :, None] * St).sum(1))  # S^T q -> [HV, V]
    if want_out:
        return torch.stack(outs, 0), St
    return St


def _expand_heads(x: torch.Tensor, HV: int) -> torch.Tensor:
    H = x.shape[1]
    if HV == H:
        return x
    if HV % H != 0:
        raise ValueError(f"HV ({HV}) must be a multiple of H_qk ({H})")
    return x.repeat_interleave(HV // H, dim=1)


def _recur(S0, qe, ke, v, g, beta, scale, block, progress, tag):
    T, HV = ke.shape[0], ke.shape[1]
    St = S0
    outs = []
    t0 = time.time()
    nb = -(-T // block)
    for bi, b0 in enumerate(range(0, T, block)):
        b1 = min(b0 + block, T)
        ob, St = checkpoint(
            _block,
            St,
            qe[b0:b1],
            ke[b0:b1],
            v[b0:b1],
            g[b0:b1],
            beta[b0:b1],
            scale,
            use_reentrant=False,
        )
        outs.append(ob)
        if progress is not None and bi % PROGRESS_BLOCKS == PROGRESS_BLOCKS - 1:
            progress(
                f"{tag} fwd block {bi + 1}/{nb} tokens {b1}/{T} elapsed {time.time() - t0:.1f}s"
            )
            if St.requires_grad:
                St.register_hook(
                    lambda grad, bi=bi, tag=tag, nb=nb, t0=t0: progress(
                        f"{tag} bwd reached block {bi + 1}/{nb} elapsed {time.time() - t0:.1f}s"
                    )
                )
    o = torch.cat(outs, 0) if outs else ke.new_zeros(0, HV, v.shape[-1])
    return o, St


def _prepare(q, k, v, g, beta, cu_seqlens, token_range):
    q = _tok(q, 3, "q")
    k = _tok(k, 3, "k")
    v = _tok(v, 3, "v")
    g = _tok(g, 2, "g")
    beta = _tok(beta, 2, "beta")
    cu = _cu_list(cu_seqlens)
    a, b = _range(cu, token_range)
    total = cu[-1]
    q, k, v, g, beta = (
        _slice_to_range(t, total, a, b, n)
        for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g"), (beta, "beta"))
    )
    HV = v.shape[1]
    if g.shape[1] != HV or beta.shape[1] != HV:
        raise ValueError(
            f"g / beta must have HV = {HV} heads, got {g.shape[1]} / {beta.shape[1]}"
        )
    if q.shape[1] != k.shape[1]:
        raise ValueError("q and k must have the same number of heads")
    return q, k, v, g, beta, cu, a, b


def gdn_exact_forward(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    scale: float = SCALE,
    h0=None,
    token_range=None,
    block: int = BLOCK,
    progress=None,
    tag: str = "ref",
):
    """FP64 forward of global tokens [a, b) (see the module docstring): (out [b - a, HV, V], final_state [HV, K, V])."""
    q, k, v, g, beta, cu, a, b = _prepare(q, k, v, g, beta, cu_seqlens, token_range)
    HV, K, V = v.shape[1], q.shape[2], v.shape[2]
    dev = q.device
    qe, ke, v, g, beta = (
        _expand_heads(q.to(F64), HV),
        _expand_heads(k.to(F64), HV),
        v.to(F64),
        g.to(F64),
        beta.to(F64),
    )
    a_is_start = a in cu
    if h0 is not None:
        h0 = h0.to(F64)
        if tuple(h0.shape) != (HV, K, V):
            raise ValueError(
                f"h0 must be [HV, K, V] = {(HV, K, V)}, got {tuple(h0.shape)}"
            )
        if a_is_start and bool((h0 != 0).any()):
            raise ValueError(f"token {a} starts a document: h0 must be None or zero")
    S = torch.zeros(HV, K, V, dtype=F64, device=dev)
    if h0 is not None and not a_is_start:
        S = h0
    outs = []
    for d, lo, hi in _segments(cu, a, b):
        S_init = S if lo == a else torch.zeros(HV, K, V, dtype=F64, device=dev)
        o, S = _recur(
            S_init,
            qe[lo - a : hi - a],
            ke[lo - a : hi - a],
            v[lo - a : hi - a],
            g[lo - a : hi - a],
            beta[lo - a : hi - a],
            scale,
            block,
            progress,
            f"{tag} doc {d} [{lo},{hi})",
        )
        outs.append(o)
    out = torch.cat(outs, 0) if outs else torch.zeros(0, HV, V, dtype=F64, device=dev)
    return out, S


def gdn_exact_with_grads(
    q,
    k,
    v,
    g,
    beta,
    dout,
    cu_seqlens,
    *,
    scale: float = SCALE,
    h0=None,
    dfinal=None,
    token_range=None,
    block: int = BLOCK,
    progress=None,
    tag: str = "ref",
) -> dict:
    """FP64 forward + autograd of global tokens [a, b): out, final_state, dq, dk, dv, dg, dbeta, dh0 (all FP64)."""
    dout = _tok(dout, 3, "dout")
    cu = _cu_list(cu_seqlens)
    a, b = _range(cu, token_range)
    dout = _slice_to_range(dout, cu[-1], a, b, "dout").to(F64)
    qq, kk, vv, gg, bb, _, _, _ = _prepare(q, k, v, g, beta, cu_seqlens, token_range)
    leaves = [x.detach().to(F64).requires_grad_(True) for x in (qq, kk, vv, gg, bb)]
    HV, K, V = vv.shape[1], qq.shape[2], vv.shape[2]
    a_is_start = a in cu
    h0_leaf = None
    if not a_is_start:
        h0_leaf = (
            torch.zeros(HV, K, V, dtype=F64, device=qq.device)
            if h0 is None
            else h0.detach().to(F64)
        ).requires_grad_(True)
    out, S = gdn_exact_forward(
        *leaves,
        cu,
        scale=scale,
        h0=h0_leaf,
        token_range=(a, b),
        block=block,
        progress=progress,
        tag=tag,
    )
    outputs, cots = [out], [dout]
    if dfinal is not None:
        outputs.append(S)
        cots.append(dfinal.to(F64))
    all_leaves = leaves + ([h0_leaf] if h0_leaf is not None else [])
    grads = torch.autograd.grad(outputs, all_leaves, cots, allow_unused=True)
    grads = [
        torch.zeros_like(l) if gr is None else gr
        for l, gr in zip(all_leaves, grads, strict=True)
    ]
    res = {"out": out.detach(), "final_state": S.detach()}
    res.update({n: gr.detach() for n, gr in zip(GRAD_NAMES, grads[:5], strict=True)})
    res["dh0"] = (
        grads[5].detach()
        if h0_leaf is not None
        else torch.zeros(HV, K, V, dtype=F64, device=qq.device)
    )
    return res


@torch.no_grad()
def cut_states(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    cuts,
    *,
    scale: float = SCALE,
    block: int = BLOCK,
    progress=None,
) -> dict[int, torch.Tensor]:
    """The FP64 state carried OUT of the tokens before every global cut: the state after the tokens < cut of the document
    containing token cut - 1 (zeros for cut 0).  At a document boundary this is the END state of the finished document; the
    next document starts from zero (callers know a fragment with off == 0 has S_in = 0).  One pass over [0, max(cuts))
    chaining h0 between consecutive cuts (a segment that starts at a document boundary starts from zero)."""
    cu = _cu_list(cu_seqlens)
    cuts = sorted({int(c) for c in cuts})
    qq, kk, vv, gg, bb, _, _, _ = _prepare(q, k, v, g, beta, cu_seqlens, None)
    HV, K, V = vv.shape[1], qq.shape[2], vv.shape[2]
    out: dict[int, torch.Tensor] = {}
    S = torch.zeros(HV, K, V, dtype=F64, device=qq.device)
    prev = 0
    for c in cuts:
        if c > prev:
            _, S = gdn_exact_forward(
                qq,
                kk,
                vv,
                gg,
                bb,
                cu,
                scale=scale,
                h0=(None if prev in cu else S),
                token_range=(prev, c),
                block=block,
                progress=progress,
                tag=f"cut[{prev},{c})",
            )
        out[c] = S.clone()
        prev = c
    return out


@torch.no_grad()
def fragment_map(
    k, g, beta, cu_seqlens, token_range, *, block: int = BLOCK
) -> torch.Tensor:
    """The homogeneous part M [HV, K, K] of the affine map S_out = M S_in + B of one in-document token range
    (M = prod_t exp(g_t) (I - beta_t k_t k_t^T), applied as M @ S): the recurrence from S_in = I with v = 0."""
    k = _tok(k, 3, "k")
    g = _tok(g, 2, "g")
    beta = _tok(beta, 2, "beta")
    cu = _cu_list(cu_seqlens)
    a, b = _range(cu, token_range)
    if len(_segments(cu, a, b)) > 1:
        raise ValueError(f"fragment_map: range [{a}, {b}) spans several documents")
    k = _slice_to_range(k, cu[-1], a, b, "k")
    g = _slice_to_range(g, cu[-1], a, b, "g")
    beta = _slice_to_range(beta, cu[-1], a, b, "beta")
    HV, K = g.shape[1], k.shape[2]
    ke = _expand_heads(k.to(F64), HV)
    g = g.to(F64)
    beta = beta.to(F64)
    S = torch.eye(K, dtype=F64, device=k.device).expand(HV, K, K).clone()
    zeros_v = torch.zeros(0, HV, K, dtype=F64, device=k.device)
    for b0 in range(0, b - a, block):
        b1 = min(b0 + block, b - a)
        S = _block(
            S,
            zeros_v,
            ke[b0:b1],
            torch.zeros(b1 - b0, HV, K, dtype=F64, device=k.device),
            g[b0:b1],
            beta[b0:b1],
            SCALE,
            want_out=False,
        )
    return S


@torch.no_grad()
def carry_diagnostic(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    fragments,
    *,
    scale: float = SCALE,
    block: int = BLOCK,
    progress=None,
) -> list[dict]:
    """Carry-gain diagnostic: for every fragment (doc, off, length) with off > 0 (continuation
    fragment) the exact FP64 map M_f, its spectral norm per head (max / median), ||S_in||, ||S_out|| and the carried
    fraction ||M_f S_in|| / ||S_out|| (Frobenius over all heads).  Fragments of length 0 are skipped."""
    cu = _cu_list(cu_seqlens)
    frs = [(int(d), int(o), int(n)) for d, o, n in fragments if int(n) > 0]
    cuts = sorted({cu[d] + o for d, o, n in frs} | {cu[d] + o + n for d, o, n in frs})
    states = cut_states(
        q, k, v, g, beta, cu, cuts, scale=scale, block=block, progress=progress
    )
    out = []
    for d, o, n in sorted(frs):
        if o == 0:
            continue
        lo, hi = cu[d] + o, cu[d] + o + n
        S_in, S_out = states[lo], states[hi]
        M = fragment_map(k, g, beta, cu, (lo, hi), block=block)
        sn = torch.linalg.matrix_norm(M, ord=2)
        carried = (M @ S_in).norm().item()
        tot = S_out.norm().item()
        out.append(
            {
                "doc": d,
                "off": o,
                "length": n,
                "global": [lo, hi],
                "chunks": -(-n // 64),
                "spectral_norm_M_max": sn.max().item(),
                "spectral_norm_M_median": sn.median().item(),
                "spectral_norm_M_min": sn.min().item(),
                "S_in_norm": S_in.norm().item(),
                "S_out_norm": tot,
                "carried_norm": carried,
                "carried_fraction": (carried / tot) if tot > 0 else None,
            }
        )
        if progress is not None:
            progress(
                f"carry doc {d} [{lo},{hi}) ||M||2 max {sn.max().item():.3e} carried fraction {(carried / tot) if tot > 0 else float('nan'):.3e}"
            )
    return out


# ----------------------------------------------------------------------------------------------------------------
# conv reference


def _conv_segment(xp: torch.Tensor, w: torch.Tensor, n: int, slab: int) -> torch.Tensor:
    """pre[t] = sum_k w[:, k] * xp[t + k], t in [0, n), xp = [3 history/zero rows; x_seg] (FP64)."""
    parts = []
    for s0 in range(0, n, slab):
        s1 = min(s0 + slab, n)
        pre = w[:, 0][None] * xp[s0:s1]
        for kk in range(1, CONV_WIDTH):
            pre = pre + w[:, kk][None] * xp[s0 + kk : s1 + kk]
        parts.append(pre * torch.sigmoid(pre))
    return torch.cat(parts, 0)


def _conv_prepare(x, w, cu_seqlens, token_range, history):
    x = _tok(x, 2, "x")
    cu = _cu_list(cu_seqlens)
    a, b = _range(cu, token_range)
    total = cu[-1]
    C = x.shape[1]
    w = w.to(F64)
    if tuple(w.shape) != (C, CONV_WIDTH):
        raise ValueError(f"w must be [C, 4] = {(C, CONV_WIDTH)}, got {tuple(w.shape)}")
    need = min(
        CONV_WIDTH - 1, a - _doc_start_of(cu, a)
    )  # history rows the first segment needs
    empty = torch.zeros(0, C, dtype=F64, device=x.device)
    if x.shape[0] == b - a:  # range form (or the whole pack when (a, b) == (0, total))
        xr = x
        if need > 0:
            if history is None or history.shape[0] != need:
                raise ValueError(
                    f"token {a} continues a document: history must have exactly {need} rows (got {None if history is None else tuple(history.shape)})"
                )
            hist = history.to(F64)
        else:
            hist = empty
    elif x.shape[0] == total:  # global pack: the history rows come from x itself
        xr = x[a:b]
        hist = x[a - need : a].to(F64) if need > 0 else empty
    else:
        raise ValueError(
            f"x: {x.shape[0]} tokens; expected the global pack ({total}) or the range ({b - a})"
        )
    return xr, w, hist, cu, a, b, C


def conv_exact(
    x, w, cu_seqlens, *, token_range=None, history=None, slab: int = 4096
) -> torch.Tensor:
    """FP64 per-document causal depthwise width-4 conv + SiLU of global tokens [a, b) (module docstring); y [b - a, C]."""
    xr, w, hist, cu, a, b, C = _conv_prepare(x, w, cu_seqlens, token_range, history)
    xr = xr.to(F64)
    outs = []
    for _d, lo, hi in _segments(cu, a, b):
        seg = xr[lo - a : hi - a]
        h = hist if lo == a else xr.new_zeros(0, C)
        pad = xr.new_zeros(CONV_WIDTH - 1 - h.shape[0], C)
        xp = torch.cat([pad, h, seg], 0)
        outs.append(_conv_segment(xp, w, hi - lo, slab))
    return torch.cat(outs, 0) if outs else xr.new_zeros(0, C)


def conv_exact_with_grads(
    x, w, cu_seqlens, dy, *, token_range=None, history=None, slab: int = 4096
) -> dict:
    """FP64 conv forward + autograd: y, dx [b - a, C], dw [C, 4], dhistory [n_hist, C] (the halo rows' gradient)."""
    xr, w64, hist, cu, a, b, C = _conv_prepare(x, w, cu_seqlens, token_range, history)
    dy = _tok(dy, 2, "dy")
    dy = _slice_to_range(dy, cu[-1], a, b, "dy").to(F64)
    x_leaf = (
        xr.detach().to(F64).requires_grad_(True)
    )  # range form: conv_exact takes the range branch below
    w_leaf = w64.detach().clone().requires_grad_(True)
    h_leaf = hist.detach().clone().requires_grad_(True)
    y = conv_exact(
        x_leaf,
        w_leaf,
        cu,
        token_range=(a, b),
        history=(h_leaf if h_leaf.shape[0] > 0 else None),
        slab=slab,
    )
    leaves = [x_leaf, w_leaf] + ([h_leaf] if h_leaf.shape[0] > 0 else [])
    grads = torch.autograd.grad(y, leaves, dy, allow_unused=True)
    grads = [
        torch.zeros_like(l) if gr is None else gr
        for l, gr in zip(leaves, grads, strict=True)
    ]
    return {
        "y": y.detach(),
        "dx": grads[0].detach(),
        "dw": grads[1].detach(),
        "dhistory": grads[2].detach() if len(grads) > 2 else hist.new_zeros(0, C),
    }


# ----------------------------------------------------------------------------------------------------------------
# error metrics and the per-tensor verdict (dbeta is compared in its FP32 storage)

_BLOCK_ELEMS = 1 << 26


def _blocks(cand: torch.Tensor, ref: torch.Tensor):
    c = cand.detach().reshape(-1)
    r = ref.detach().reshape(-1)
    if c.numel() != r.numel():
        raise ValueError(f"shape mismatch {tuple(cand.shape)} vs {tuple(ref.shape)}")
    dev = c.device if c.is_cuda else r.device
    for i in range(0, c.numel(), _BLOCK_ELEMS):
        yield i, c[i : i + _BLOCK_ELEMS].to(dev), r[i : i + _BLOCK_ELEMS].to(dev)


def error_metrics(cand: torch.Tensor, ref: torch.Tensor) -> dict:
    """rel_rms = RMS(cand - ref) / RMS(ref), max_abs, rms, ref_rms, ref_max_abs, numel, nonfinite, and the worst element
    (flat index, |err|, ref and cand values there) -- the candidate is compared AS STORED, everything computed in FP64."""
    n = int(cand.numel())
    res = {
        "rel_rms": 0.0,
        "max_abs": 0.0,
        "rms": 0.0,
        "ref_rms": 0.0,
        "ref_max_abs": 0.0,
        "numel": n,
        "nonfinite": 0,
        "worst": None,
        "dtype": str(cand.dtype).replace("torch.", ""),
    }
    if n == 0:
        return res
    sq_err = sq_ref = 0.0
    max_abs = ref_max = 0.0
    nonfinite = 0
    worst = None
    for base, c, r in _blocks(cand, ref):
        c64 = c.to(F64)
        r64 = r.to(F64)
        nonfinite += int((~torch.isfinite(c64)).sum().item())
        err = c64 - r64
        sq_err += float((err * err).sum().item())
        sq_ref += float((r64 * r64).sum().item())
        ae = err.abs()
        j = int(torch.argmax(torch.nan_to_num(ae, nan=float("inf"))).item())
        m = float(ae[j].item())
        if worst is None or m > worst["err"] or math.isnan(m):
            worst = {
                "index": base + j,
                "err": m,
                "ref": float(r64[j].item()),
                "cand": float(c64[j].item()),
            }
        max_abs = max(max_abs, m)
        ref_max = max(ref_max, float(r64.abs().max().item()))
        del c64, r64, err, ae
    rms_ref = math.sqrt(sq_ref / n)
    rms_err = math.sqrt(sq_err / n)
    res.update(
        rel_rms=(rms_err / rms_ref)
        if rms_ref > 0
        else (float("inf") if rms_err > 0 else 0.0),
        max_abs=max_abs,
        rms=rms_err,
        ref_rms=rms_ref,
        ref_max_abs=ref_max,
        nonfinite=nonfinite,
        worst=worst,
    )
    return res


def elementwise_mismatch(
    cand: torch.Tensor, ref: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2
) -> dict:
    mismatch = nonfinite = 0
    worst = 0.0
    for _, c, r in _blocks(cand, ref):
        c64 = c.to(F64)
        r64 = r.to(F64)
        excess = (c64 - r64).abs() - (atol + rtol * r64.abs())
        bad = excess > 0
        mismatch += int(bad.sum().item())
        if mismatch:
            worst = max(worst, float(excess.max().item()))
        nonfinite += int((~torch.isfinite(c64)).sum().item())
    return {
        "mismatch": mismatch,
        "numel": int(cand.numel()),
        "nonfinite": nonfinite,
        "pass": bool(mismatch == 0 and nonfinite == 0),
        "worst_excess": worst,
        "atol": atol,
        "rtol": rtol,
    }


def worst_element(cand: torch.Tensor, ref: torch.Tensor) -> dict | None:
    return error_metrics(cand, ref)["worst"]


def ulp_storage(magnitude: float, storage: str) -> float:
    """Unit in the last place of `storage` ('bf16' | 'fp32') in the binade of |magnitude| (form:
    2 ** (floor(log2 m) - p), p = 7 / 23, exponent clamped at the minimum normal exponent -126)."""
    m = abs(float(magnitude))
    if math.isnan(m) or math.isinf(m):
        raise ValueError(f"non-finite magnitude {magnitude!r}")
    if m == 0.0:
        e = MIN_EXP
    else:
        _, exp = math.frexp(m)
        e = max(exp - 1, MIN_EXP)
    return math.ldexp(1.0, e - FRACTION_BITS[storage])


def judge_tensor(
    metrics: dict, budget_rel_rms: float, budget_max_abs: float, storage: str
) -> dict:
    """G1 verdict of one tensor: pass_pre_u1 (rel-RMS and max-abs within budget, finite) and pass_u1 (max-abs may exceed
    the budget by one storage ulp of |ref| at the candidate's worst element: the one-storage-ulp slack of the
    acceptance rule)."""
    rel, mx, nf = (
        float(metrics["rel_rms"]),
        float(metrics["max_abs"]),
        int(metrics["nonfinite"]),
    )
    brel, bmax = float(budget_rel_rms), float(budget_max_abs)
    rel_ok, finite_ok = rel <= brel, nf == 0
    pass_pre = bool(rel_ok and mx <= bmax and finite_ok)
    out = {
        "storage": storage,
        "rel_rms": rel,
        "max_abs": mx,
        "nonfinite": nf,
        "budget_rel_rms": brel,
        "budget_max_abs": bmax,
        "ratio_rel": (rel / brel) if brel > 0 else float("inf"),
        "ratio_max": (mx / bmax) if bmax > 0 else float("inf"),
        "pass_pre_u1": pass_pre,
    }
    w = metrics.get("worst")
    if w is None:
        out.update(
            ulp_at_worst=None,
            budget_max_abs_u1=None,
            pass_u1=(True if pass_pre else None),
        )
        return out
    ulp = ulp_storage(w["ref"], storage)
    bmax_u1 = bmax + ulp
    out.update(
        ulp_at_worst=ulp,
        ref_at_worst=w["ref"],
        budget_max_abs_u1=bmax_u1,
        ratio_max_u1=(mx / bmax_u1) if bmax_u1 > 0 else float("inf"),
        pass_u1=bool(rel_ok and mx <= bmax_u1 and finite_ok),
    )
    out["u1_decisive"] = bool(out["pass_u1"] and not pass_pre)
    return out


def compare_to_reference(
    cand: dict, ref: dict, *, elementwise: bool = True, names=TENSORS
) -> dict:
    """Per-tensor error_metrics (+ elementwise 1e-2 clause) of a candidate dict vs the FP64 reference dict."""
    res = {}
    for n in names:
        if n not in cand or n not in ref:
            continue
        m = error_metrics(cand[n], ref[n])
        m["storage"] = STORAGE_OF.get(n)
        if elementwise:
            m["elementwise_1e-2"] = elementwise_mismatch(cand[n], ref[n], 1e-2, 1e-2)
        res[n] = m
    return res


# ----------------------------------------------------------------------------------------------------------------
# whole-pack reference (per document)


def reference_pack(
    inputs: dict, device, *, scale: float = SCALE, block: int = BLOCK, progress=None
) -> tuple[dict, dict]:
    """The FP64 reference of the whole pack (gdn_inputs.make_inputs output): out / dq / dk / dv / dg / dbeta as CPU FP64
    tensors without the batch axis, document by document (autograd per document bounds memory), plus timings."""
    cu = _cu_list(inputs["cu_seqlens"])
    q, k, v, g, beta, dout = (inputs[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    q, k, v, dout = (
        _tok(t, 3, n) for t, n in ((q, "q"), (k, "k"), (v, "v"), (dout, "dout"))
    )
    g, beta = _tok(g, 2, "g"), _tok(beta, 2, "beta")
    T = cu[-1]
    res = {
        "out": torch.zeros(T, v.shape[1], v.shape[2], dtype=F64),
        "dq": torch.zeros(q.shape, dtype=F64),
        "dk": torch.zeros(k.shape, dtype=F64),
        "dv": torch.zeros(v.shape, dtype=F64),
        "dg": torch.zeros(g.shape, dtype=F64),
        "dbeta": torch.zeros(beta.shape, dtype=F64),
    }
    meta = {"docs": [], "elapsed_s": 0.0}
    t_all = time.time()
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            meta["docs"].append({"doc": d, "tokens": 0, "s": 0.0})
            continue
        td = time.time()
        ops = [t[s0:s1].to(device) for t in (q, k, v, g, beta, dout)]
        r = gdn_exact_with_grads(
            *ops[:5],
            ops[5],
            [0, s1 - s0],
            scale=scale,
            block=block,
            progress=progress,
            tag=f"doc {d} ({s1 - s0} tok)",
        )
        for n in TENSORS:
            res[n][s0:s1] = r[n].cpu()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - td
        meta["docs"].append(
            {
                "doc": d,
                "tokens": s1 - s0,
                "s": round(dt, 2),
                "ms_per_token": round(dt / (s1 - s0) * 1e3, 3),
            }
        )
        if progress is not None:
            progress(
                f"doc {d}/{len(cu) - 1} done: {s1 - s0} tokens in {dt:.1f}s ({dt / (s1 - s0) * 1e3:.2f} ms/token), total {time.time() - t_all:.1f}s"
            )
        del ops, r
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    meta["elapsed_s"] = round(time.time() - t_all, 2)
    return res, meta
