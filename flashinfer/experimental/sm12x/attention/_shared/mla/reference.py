# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/reference.py @ e9a11f2c (2026-06-28) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Simple PyTorch MLA references for the NSA packed-cache contract."""

from __future__ import annotations

import math

import torch


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_E4M3_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_LN2 = math.log(2.0)
_MLA_NOPE_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_GROUP_SIZE = 128
_MLA_SCALE_BYTES = (_MLA_NOPE_DIM // _MLA_GROUP_SIZE) * 4
_MLA_PACKED_DIM = 656


def _as_2d_cache(x: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(f"{name} middle dimension must be 1, got {tuple(x.shape)}")
        x = x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or rank-3, got {tuple(x.shape)}")
    if x.shape[1] != expected_dim:
        raise ValueError(
            f"{name} last dimension must be {expected_dim}, got {x.shape[1]}"
        )
    return x.contiguous()


def pack_mla_kv_cache_reference(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    group_size: int = _MLA_GROUP_SIZE,
) -> torch.Tensor:
    """Pack MLA KV cache into the FP8+scale+rope byte layout used by NSA."""

    if group_size != _MLA_GROUP_SIZE:
        raise ValueError(
            f"Only group_size={_MLA_GROUP_SIZE} is supported in the reference."
        )

    k_nope_2d = _as_2d_cache(k_nope, _MLA_NOPE_DIM, "k_nope")
    k_rope_2d = _as_2d_cache(k_rope, _MLA_ROPE_DIM, "k_rope")
    if k_nope_2d.shape[0] != k_rope_2d.shape[0]:
        raise ValueError("k_nope and k_rope must have the same token count")
    if k_rope_2d.dtype != torch.bfloat16:
        raise ValueError(
            "k_rope must have dtype torch.bfloat16 because the packed MLA layout "
            f"stores raw BF16 rope bytes, got {k_rope_2d.dtype}"
        )

    quant_bytes: list[torch.Tensor] = []
    scale_bytes: list[torch.Tensor] = []
    for block_start in range(0, _MLA_NOPE_DIM, group_size):
        block = k_nope_2d[:, block_start : block_start + group_size].to(torch.float32)
        scale = block.abs().amax(dim=1) / _FP8_E4M3_MAX
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quant = (block / scale.unsqueeze(1)).clamp(_FP8_E4M3_MIN, _FP8_E4M3_MAX)
        quant = quant.to(torch.float8_e4m3fn)
        quant_bytes.append(quant.view(torch.uint8).reshape(block.shape[0], group_size))
        scale_bytes.append(scale.view(torch.uint8).reshape(block.shape[0], 4))

    rope_bytes = k_rope_2d.view(torch.uint8).reshape(
        k_rope_2d.shape[0], _MLA_ROPE_DIM * 2
    )
    packed = torch.cat(
        [torch.cat(quant_bytes, dim=1), torch.cat(scale_bytes, dim=1), rope_bytes],
        dim=1,
    )
    return packed.unsqueeze(1).contiguous()


def unpack_mla_kv_cache_reference(
    kv_cache: torch.Tensor,
    *,
    group_size: int = _MLA_GROUP_SIZE,
) -> torch.Tensor:
    """Unpack the NSA MLA byte layout back into dequantized K tensors."""

    if group_size != _MLA_GROUP_SIZE:
        raise ValueError(
            f"Only group_size={_MLA_GROUP_SIZE} is supported in the reference."
        )

    packed = _as_2d_cache(kv_cache, _MLA_PACKED_DIM, "kv_cache").view(torch.uint8)
    num_tokens = packed.shape[0]
    num_groups = _MLA_NOPE_DIM // group_size

    nope_q = packed[:, :_MLA_NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
    nope_q = nope_q.reshape(num_tokens, _MLA_NOPE_DIM).to(torch.float32)
    scales = packed[:, _MLA_NOPE_DIM : _MLA_NOPE_DIM + num_groups * 4].contiguous()
    scales = scales.view(torch.float32).reshape(num_tokens, num_groups)
    rope = packed[:, _MLA_NOPE_DIM + num_groups * 4 :].contiguous().view(torch.bfloat16)
    rope = rope.reshape(num_tokens, _MLA_ROPE_DIM).to(torch.float32)

    nope = nope_q.reshape(num_tokens, num_groups, group_size) * scales.unsqueeze(-1)
    nope = nope.reshape(num_tokens, _MLA_NOPE_DIM)
    return torch.cat([nope, rope], dim=1).unsqueeze(1).contiguous()


def dense_mla_reference(
    *,
    q_all: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    """Reference attention using the unquantized MLA cache tensors."""

    k_nope_2d = _as_2d_cache(k_nope, _MLA_NOPE_DIM, "k_nope").to(torch.float32)
    k_rope_2d = _as_2d_cache(k_rope, _MLA_ROPE_DIM, "k_rope").to(torch.float32)
    k_all = torch.cat([k_nope_2d, k_rope_2d], dim=1)
    return _sparse_attention_reference(
        q_all=q_all,
        k_all=k_all,
        v_all=k_nope_2d[:, :v_head_dim],
        page_table_1=page_table_1,
        sm_scale=sm_scale,
    )


def sparse_mla_reference(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor | None = None,
    sm_scale: float,
    v_head_dim: int,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference attention using the packed NSA MLA cache layout."""

    kv = unpack_mla_kv_cache_reference(kv_cache).squeeze(1).to(torch.float32)
    return _sparse_attention_reference(
        q_all=q_all,
        k_all=kv,
        v_all=kv[:, :v_head_dim],
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        return_lse=return_lse,
    )


def _sparse_attention_reference(
    *,
    q_all: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor | None = None,
    sm_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if q_all.ndim != 3:
        raise ValueError(f"q_all must be rank-3, got {tuple(q_all.shape)}")
    if page_table_1.ndim != 2:
        raise ValueError(
            f"page_table_1 must be rank-2, got {tuple(page_table_1.shape)}"
        )
    if page_table_1.shape[0] != q_all.shape[0]:
        raise ValueError(
            f"page_table_1 rows {page_table_1.shape[0]} do not match q rows {q_all.shape[0]}"
        )
    if active_token_counts is not None:
        if (
            active_token_counts.ndim != 1
            or active_token_counts.shape[0] != q_all.shape[0]
        ):
            raise ValueError(
                "active_token_counts must be rank-1 with one entry per query row, "
                f"got {tuple(active_token_counts.shape)}"
            )

    out = torch.zeros(
        (q_all.shape[0], q_all.shape[1], v_all.shape[1]),
        dtype=torch.float32,
        device=q_all.device,
    )
    lse_base2 = torch.full(
        (q_all.shape[0], q_all.shape[1]),
        float("-inf"),
        dtype=torch.float32,
        device=q_all.device,
    )
    q_all_f = q_all.to(torch.float32)
    num_kv = k_all.shape[0]
    width = page_table_1.shape[1]
    if num_kv == 0 or width == 0:
        output = out.to(q_all.dtype)
        return (output, lse_base2) if return_lse else output

    positions = torch.arange(width, dtype=torch.long, device=q_all.device)
    tiny = torch.finfo(torch.float32).tiny

    for row in range(q_all.shape[0]):
        selected = page_table_1[row].to(torch.long)
        active_mask = torch.ones((width,), dtype=torch.bool, device=q_all.device)
        if active_token_counts is not None:
            token_end = active_token_counts[row].to(torch.long).clamp(min=0, max=width)
            active_mask = positions < token_end
        valid_mask = active_mask & (selected >= 0) & (selected < num_kv)
        safe_selected = selected.clamp(0, num_kv - 1)

        k_sel = k_all.index_select(0, safe_selected)
        v_sel = v_all.index_select(0, safe_selected)
        scores = torch.matmul(q_all_f[row], k_sel.transpose(0, 1)) * float(sm_scale)
        scores = torch.where(
            valid_mask.unsqueeze(0),
            scores,
            torch.full_like(scores, float("-inf")),
        )
        row_max = scores.amax(dim=-1, keepdim=True)
        finite_row = torch.isfinite(row_max)
        safe_row_max = torch.where(finite_row, row_max, torch.zeros_like(row_max))
        exp_scores = torch.where(
            valid_mask.unsqueeze(0),
            torch.exp(scores - safe_row_max),
            torch.zeros_like(scores),
        )
        denom = exp_scores.sum(dim=-1, keepdim=True)
        probs = exp_scores / denom.clamp_min(tiny)
        out[row] = torch.matmul(probs, v_sel)
        if return_lse:
            row_lse = torch.log(denom) + safe_row_max
            row_lse = torch.where(
                finite_row,
                row_lse,
                torch.full_like(row_lse, float("-inf")),
            )
            lse_base2[row] = row_lse.squeeze(-1) / _LN2

    output = out.to(q_all.dtype)
    return (output, lse_base2) if return_lse else output
