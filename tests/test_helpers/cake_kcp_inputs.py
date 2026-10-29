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

Operand generation of the packed KDA context-parallel (KCP) tests, deterministic per (row, document, 64-token
block) of the rows in ``cake_kcp_shapes``.

Every rank generates only its own fragments; the FP64 reference generates the whole pack.  Both call `doc_block()`
with the same arguments, so a token's operands do not depend on the partition, the device or the process.
Generation runs on the CPU with a seeded torch.Generator (CRC32 of "SEED:row:doc:block"), then moves to `device`.

Contract: q, k BF16 [T, H, 128] L2-normalized in FP32 (eps 1e-6) before the BF16 store; v BF16; g FP32 in (-5, 0)
(natural-log decay, safe gate); beta BF16 = sigmoid(logits); dout BF16.  Regimes (Row.data, parameters parsed by
parse_regime):
  default                           : g = -5 * sigmoid(exp(A_log) * (raw + dt_bias)), A_log ~ log(U[1,2)) per head,
                                      beta = sigmoid(N(0,1))
  iso_weak:b0=B:gmax=G:gzero=Z      : isotropic keys, beta = sigmoid(logit(B) + 0.2 N) ~ B (weak erasure),
                                      g = -G * sigmoid(raw + dt_bias) in (-G, 0), heads 0..Z-1 with g == 0 exactly
  aniso_weak:noise=E:gmax=G:gzero=Z : k on 4 fixed dims + E * isotropic noise, beta = sigmoid(3.9 + 0.2 N) ~ 0.98,
                                      g as in iso_weak
  mixed_v2:b0=B:gmax=G              : document 0 iso_weak(B, G), document 1 fast_decay
                                      (g = -5 + 4 * sigmoid(raw + dt_bias) in (-5, -1))
The draw order (q, k, [noise], v, raw, logits, dout), seeds, dtypes and scalings are identical for every regime, so a
token's operands depend only on (row name, document, block) and the regime string.
"""

from __future__ import annotations

import zlib

import torch

from tests.test_helpers.cake_kcp_shapes import SEED, D, Row

BLOCK = 64
EPS = 1e-6


def _gen(*parts) -> torch.Generator:
    key = ":".join(str(p) for p in parts)
    return torch.Generator(device="cpu").manual_seed(
        zlib.crc32(f"{SEED}:{key}".encode())
    )


def head_params(row: Row, H: int) -> dict[str, torch.Tensor]:
    """Per-row, per-head parameters shared by every block: A_log [H], dt_bias [H, 128], aniso dims [H, 4]."""
    g = _gen(row.name, "heads", H)
    a_log = torch.log(torch.rand(H, generator=g, dtype=torch.float32) + 1.0)
    dt_bias = torch.randn(H, D, generator=g, dtype=torch.float32) * 0.5
    dims = torch.stack([torch.randperm(D, generator=g)[:4] for _ in range(H)])  # [H, 4]
    return {"a_log": a_log, "dt_bias": dt_bias, "dims": dims}


def _l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
    return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)).to(torch.bfloat16)


def parse_regime(data: str) -> tuple[str, dict[str, float]]:
    """'iso_weak:b0=0.02:gmax=1e-5:gzero=0' -> ('iso_weak', {'b0': 0.02, 'gmax': 1e-5, 'gzero': 0.0}); 'default' -> ('default', {})."""
    parts = data.split(":")
    params: dict[str, float] = {}
    for p in parts[1:]:
        k, v = p.split("=")
        params[k] = float(v)
    return parts[0], params


def regime_of(row: Row, doc: int) -> str:
    """Regime family of document `doc`: default | iso_weak | aniso_weak | fast_decay (document 1 of mixed_v2)."""
    fam, _ = parse_regime(row.data)
    if fam == "mixed_v2":
        return "iso_weak" if doc == 0 else "fast_decay"
    return fam


def regime_params(row: Row, doc: int) -> dict[str, float]:
    """Parameters of the regime of document `doc` (empty for the default regime and for mixed_v2's fast-decay document)."""
    fam, params = parse_regime(row.data)
    if fam == "mixed_v2":
        return params if doc == 0 else {}
    return params


def _logit(p: float) -> float:
    return float(torch.logit(torch.tensor(p, dtype=torch.float64)))


def doc_block(
    row: Row,
    H: int,
    doc: int,
    block: int,
    params: dict[str, torch.Tensor],
    n: int = BLOCK,
) -> dict[str, torch.Tensor]:
    """Operands of tokens [block*64, block*64 + n) of document `doc` (CPU tensors): q, k, v, dout BF16 [n, H, 128],
    g FP32 [n, H, 128], beta BF16 [n, H]."""
    gen = _gen(row.name, "doc", doc, "block", block)
    regime = regime_of(row, doc)
    rp = regime_params(row, doc)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = _l2norm_bf16(randn(n, H, D))
    k_raw = randn(n, H, D)
    if regime == "aniso_weak":
        mask = torch.zeros(H, D, dtype=torch.float32)
        mask.scatter_(1, params["dims"], 1.0)
        noise = float(rp.get("noise", 0.02))
        k_raw = k_raw * mask[None] + noise * randn(n, H, D)
    k = _l2norm_bf16(k_raw)
    v = (randn(n, H, D) * 0.5).to(torch.bfloat16)
    raw = randn(n, H, D)
    pre = raw + params["dt_bias"][None]
    if regime == "default":
        g = -5.0 * torch.sigmoid(torch.exp(params["a_log"])[None, :, None] * pre)
    elif regime == "fast_decay":
        g = -5.0 + 4.0 * torch.sigmoid(pre)
    elif regime in ("iso_weak", "aniso_weak"):
        gmax = float(rp.get("gmax", 1e-5))
        g = -gmax * torch.sigmoid(pre)
        gzero = int(rp.get("gzero", 0))
        if gzero > 0:
            g[:, :gzero, :] = 0.0
    else:
        raise ValueError(regime)
    logits = randn(n, H)
    if regime == "aniso_weak":
        logits = 3.9 + 0.2 * logits
    elif regime == "iso_weak":
        logits = _logit(float(rp.get("b0", 0.02))) + 0.2 * logits
    beta = torch.sigmoid(logits).to(torch.bfloat16)
    dout = (randn(n, H, D) * 0.5).to(torch.bfloat16)
    return {"q": q, "k": k, "v": v, "g": g.float(), "beta": beta, "dout": dout}


def doc_range(
    row: Row,
    H: int,
    doc: int,
    off: int,
    length: int,
    params: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Operands of document `doc` tokens [off, off + length) (CPU), assembled from 64-token blocks."""
    params = params or head_params(row, H)
    if length == 0:
        return {
            "q": torch.empty(0, H, D, dtype=torch.bfloat16),
            "k": torch.empty(0, H, D, dtype=torch.bfloat16),
            "v": torch.empty(0, H, D, dtype=torch.bfloat16),
            "g": torch.empty(0, H, D, dtype=torch.float32),
            "beta": torch.empty(0, H, dtype=torch.bfloat16),
            "dout": torch.empty(0, H, D, dtype=torch.bfloat16),
        }
    doc_len = row.lengths[doc]
    assert off >= 0 and off + length <= doc_len, (row.name, doc, off, length, doc_len)
    parts: dict[str, list[torch.Tensor]] = {}
    b0, b1 = off // BLOCK, (off + length - 1) // BLOCK
    for b in range(b0, b1 + 1):
        n = min(BLOCK, doc_len - b * BLOCK)
        blk = doc_block(row, H, doc, b, params, n)
        lo = max(off, b * BLOCK) - b * BLOCK
        hi = min(off + length, b * BLOCK + n) - b * BLOCK
        for name, t in blk.items():
            parts.setdefault(name, []).append(t[lo:hi])
    return {name: torch.cat(ts, 0) for name, ts in parts.items()}


def global_operands(
    row: Row, H: int, device, docs: list[int] | None = None
) -> dict[str, torch.Tensor]:
    """The whole pack in global order (or the given documents only), with cu_seqlens int32 on `device`."""
    params = head_params(row, H)
    docs = list(range(len(row.lengths))) if docs is None else docs
    parts: dict[str, list[torch.Tensor]] = {}
    cu = [0]
    for d in docs:
        ops = doc_range(row, H, d, 0, row.lengths[d], params)
        for name, t in ops.items():
            parts.setdefault(name, []).append(t)
        cu.append(cu[-1] + row.lengths[d])
    out = {name: torch.cat(ts, 0).to(device) for name, ts in parts.items()}
    out["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device=device)
    return out
