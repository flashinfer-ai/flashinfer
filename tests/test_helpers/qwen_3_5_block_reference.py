# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp32 torch oracle for the Qwen 3.5 gated attention block tests.

A trimmed port of cuDNN Frontend's ``test/python/gated_attention_block/cutedsl/gated_block_reference.py`` (the block's
correctness oracle): the forward order -- project, QK-RMSNorm (V unnormed), partial RoPE, SDPA, ``* sigmoid(GATE)``,
out-project -- with fp32 accumulation, chunked over query tiles, returning every intermediate so a mismatch localizes to a
stage.  Kept independent of the vendored kernels on purpose: a reference that imports the thing it validates can agree
with a bug.  The quantized and backward oracles of the original are not ported (the bf16 serving forms do not use them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class RefGeometry:
    """Mirror of the block geometry, kept independent on purpose."""

    d_model: int
    h_q: int
    h_kv: int
    d_head: int
    rope_dim: int
    qk_norm_eps: float = 1e-6
    attn_scale: Optional[float] = None
    is_causal: bool = True
    rope_base: float = 1_000_000.0
    qk_norm: bool = True
    window_left: int = -1
    window_right: int = -1
    causal_bottom_right: bool = False

    @property
    def scale(self) -> float:
        return (
            self.attn_scale
            if self.attn_scale is not None
            else float(self.d_head) ** -0.5
        )

    @property
    def n_qkvg(self) -> int:
        return (2 * self.h_q + 2 * self.h_kv) * self.d_head

    @property
    def offsets(self) -> Tuple[int, int, int, int]:
        q = 0
        g = q + self.h_q * self.d_head
        k = g + self.h_q * self.d_head
        v = k + self.h_kv * self.d_head
        return (q, g, k, v)


def build_rope_tables(
    seq_len: int,
    rope_dim: int,
    *,
    base: float = 1_000_000.0,
    batch: int = 1,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos, sin`` of shape ``[batch, seq_len, rope_dim]`` with duplicated halves -- byte-for-byte what HF
    ``apply_rotary_pos_emb`` and vLLM's default ``RotaryEmbedding`` consume.  ``positions`` (``[batch, seq_len]``) lets a
    caller supply arbitrary positions (a prefix offset, mRoPE sections)."""
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    if positions is None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).expand(
            batch, seq_len
        )
    positions = positions.to(device=device, dtype=torch.float32)
    freqs = positions[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def make_inputs(
    geom: RefGeometry,
    batch: int,
    seq_len: int,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> dict:
    """Random inputs in the block's declared layouts (weights ``nn.Linear``-shaped, ``[out, in]``)."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, std=0.02):
        return (
            torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * std
        ).to(dtype)

    cos, sin = build_rope_tables(
        seq_len,
        geom.rope_dim,
        base=geom.rope_base,
        batch=batch,
        device=device,
        dtype=dtype,
    )
    norm_w = (
        (lambda: torch.ones(geom.d_head, device=device, dtype=dtype))
        if geom.qk_norm
        else (lambda: None)
    )
    return {
        "h": randn(batch, seq_len, geom.d_model, std=1.0),
        "w_qkvg": randn(geom.n_qkvg, geom.d_model),
        "w_q_norm": norm_w(),
        "w_k_norm": norm_w(),
        "cos": cos,
        "sin": sin,
        "w_o": randn(geom.d_model, geom.h_q * geom.d_head),
    }


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qk_norm_rope_reference(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    eps: float,
    *,
    qk_norm: bool = True,
    acc_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """QK-RMSNorm then partial RoPE, fused: fp32 norm, fp32 rotation, ONE cast at the end (the fused kernel's rounding)."""
    x32 = x.to(acc_dtype)
    if qk_norm:
        if w is None:
            raise ValueError(
                "qk_norm=True needs a [D] norm weight; pass qk_norm=False for RoPE-only Q/K"
            )
        rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = x32 * rstd * w.to(acc_dtype)
    else:
        rstd = None
        y = x32
    if rope_dim:
        c = cos[:, :, None, :rope_dim].to(acc_dtype)
        s = sin[:, :, None, :rope_dim].to(acc_dtype)
        rot, passthrough = y[..., :rope_dim], y[..., rope_dim:]
        rot = rot * c + _rotate_half(rot) * s
        y = torch.cat((rot, passthrough), dim=-1) if passthrough.shape[-1] else rot
    return y.to(x.dtype), (None if rstd is None else rstd.squeeze(-1).to(acc_dtype))


def split_qkvg(
    proj: torch.Tensor, geom: RefGeometry
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[B, S, N] -> (Q, GATE, K, V)`` in the block's contract column order."""
    b, s, _ = proj.shape
    o_q, o_g, o_k, o_v = geom.offsets
    hq, hkv, d = geom.h_q, geom.h_kv, geom.d_head
    q = proj[..., o_q:o_g].view(b, s, hq, d)
    gate = proj[..., o_g:o_k].view(b, s, hq, d)
    k = proj[..., o_k:o_v].view(b, s, hkv, d)
    v = proj[..., o_v:].view(b, s, hkv, d)
    return q, gate, k, v


def _key_padding_and_causal_mask(
    s_q: int,
    s_kv: int,
    *,
    is_causal: bool,
    seq_lens: Optional[torch.Tensor],
    batch_index: int,
    q_lo: int,
    device,
    window_left: int = -1,
    window_right: int = -1,
    causal_bottom_right: bool = False,
    s_q_total: Optional[int] = None,
) -> torch.Tensor:
    """``[q_chunk, s_kv]`` bool mask, True where a column is ALLOWED (rows ``[q_lo, q_lo + s_q)`` of the full mask)."""
    if int(window_right) >= 0 and not is_causal:
        raise ValueError("window_right requires is_causal=True")
    if causal_bottom_right and not is_causal:
        raise ValueError("causal_bottom_right=True requires is_causal=True")
    q_idx = torch.arange(q_lo, q_lo + s_q, device=device)
    k_idx = torch.arange(s_kv, device=device)
    allowed = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
    diag = (
        (s_kv - (s_kv if s_q_total is None else int(s_q_total)))
        if causal_bottom_right
        else 0
    )
    if is_causal:
        allowed &= k_idx[None, :] <= q_idx[:, None] + diag + max(int(window_right), 0)
    if window_left >= 0:
        allowed &= k_idx[None, :] >= q_idx[:, None] + diag - int(window_left)
    if seq_lens is not None:
        allowed &= k_idx[None, :] < int(seq_lens[batch_index])
    return allowed


@dataclass
class RefOutputs:
    """Everything the oracle computed, so a mismatch localizes to a stage."""

    out: torch.Tensor  # [B, S, d_model]
    q_pre: torch.Tensor  # [B, S, H_q, D]
    k_pre: torch.Tensor  # [B, S, H_kv, D]
    gate: torch.Tensor  # [B, S, H_q, D] pre-sigmoid
    v: torch.Tensor  # [B, S, H_kv, D]
    q: torch.Tensor  # [B, S, H_q, D] post-norm, post-RoPE
    k: torch.Tensor  # [B, S, H_kv, D]
    rstd_q: Optional[torch.Tensor]
    rstd_k: Optional[torch.Tensor]
    o: torch.Tensor  # [B, S, H_q, D] SDPA output, PRE-gate
    o_gated: torch.Tensor  # [B, S, H_q, D]
    lse: torch.Tensor  # [B, H_q, S] fp32, natural log


def gated_attention_block_reference(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geom: RefGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    q_chunk: int = 512,
    acc_dtype: torch.dtype = torch.float32,
) -> RefOutputs:
    """FP32 oracle for the whole block over a full sequence, chunked over query tiles.  A row with NO allowed column yields
    ``O = 0`` and ``LSE = -inf`` exactly (a select, never a floored denominator)."""
    b, s, d_model = h.shape
    if d_model != geom.d_model:
        raise ValueError(f"h last dim {d_model} != geometry d_model {geom.d_model}")
    if w_qkvg.shape != (geom.n_qkvg, geom.d_model):
        raise ValueError(
            f"w_qkvg must be [{geom.n_qkvg}, {geom.d_model}], got {tuple(w_qkvg.shape)}"
        )
    if w_o.shape != (geom.d_model, geom.h_q * geom.d_head):
        raise ValueError(
            f"w_o must be [{geom.d_model}, {geom.h_q * geom.d_head}], got {tuple(w_o.shape)}"
        )

    dev = h.device
    out_dtype = h.dtype
    d = geom.d_head
    rep = geom.h_q // geom.h_kv

    proj = (h.to(acc_dtype) @ w_qkvg.to(acc_dtype).t()).to(out_dtype)
    q_pre, gate, k_pre, v = split_qkvg(proj, geom)
    q, rstd_q = qk_norm_rope_reference(
        q_pre,
        w_q_norm,
        cos,
        sin,
        geom.rope_dim,
        geom.qk_norm_eps,
        qk_norm=geom.qk_norm,
        acc_dtype=acc_dtype,
    )
    k, rstd_k = qk_norm_rope_reference(
        k_pre,
        w_k_norm,
        cos,
        sin,
        geom.rope_dim,
        geom.qk_norm_eps,
        qk_norm=geom.qk_norm,
        acc_dtype=acc_dtype,
    )

    o = torch.zeros(b, s, geom.h_q, d, device=dev, dtype=acc_dtype)
    lse = torch.full((b, geom.h_q, s), float("-inf"), device=dev, dtype=acc_dtype)
    k_b = k.to(acc_dtype).repeat_interleave(rep, dim=2)
    v_b = v.to(acc_dtype).repeat_interleave(rep, dim=2)
    for bi in range(b):
        for lo in range(0, s, q_chunk):
            hi = min(lo + q_chunk, s)
            qc = q[bi, lo:hi].to(acc_dtype).transpose(0, 1)
            kc = k_b[bi].transpose(0, 1)
            vc = v_b[bi].transpose(0, 1)
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * geom.scale
            allowed = _key_padding_and_causal_mask(
                hi - lo,
                s,
                is_causal=geom.is_causal,
                seq_lens=seq_lens,
                batch_index=bi,
                q_lo=lo,
                device=dev,
                window_left=geom.window_left,
                window_right=geom.window_right,
                causal_bottom_right=geom.causal_bottom_right,
                s_q_total=s,
            )
            scores = scores.masked_fill(~allowed[None], float("-inf"))
            row_max = scores.amax(dim=-1)
            dead = torch.isinf(row_max) & (row_max < 0)
            safe_max = torch.where(dead, torch.zeros_like(row_max), row_max)
            p = torch.exp(scores - safe_max[..., None])
            p = torch.where(allowed[None], p, torch.zeros_like(p))
            denom = p.sum(dim=-1)
            o_chunk = (
                torch.matmul(p, vc)
                / torch.where(dead, torch.ones_like(denom), denom)[..., None]
            )
            o_chunk = torch.where(dead[..., None], torch.zeros_like(o_chunk), o_chunk)
            o[bi, lo:hi] = o_chunk.transpose(0, 1)
            lse_chunk = safe_max + torch.log(denom)
            lse[bi, :, lo:hi] = torch.where(
                dead, torch.full_like(lse_chunk, float("-inf")), lse_chunk
            )
    o = o.to(out_dtype)
    o_gated = (o.to(acc_dtype) * torch.sigmoid(gate.to(acc_dtype))).to(out_dtype)
    out = (
        o_gated.reshape(b, s, geom.h_q * d).to(acc_dtype) @ w_o.to(acc_dtype).t()
    ).to(out_dtype)
    return RefOutputs(
        out=out,
        q_pre=q_pre,
        k_pre=k_pre,
        gate=gate,
        v=v,
        q=q,
        k=k,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        o=o,
        o_gated=o_gated,
        lse=lse,
    )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two tensors as flat fp32 vectors."""
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))
