# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time weight packing: a serving stack's merged QKV (+ indexer) linear -> the block's ``W_qkvg`` bands and the four norm weights.

Pure torch, CPU-safe, a row PERMUTATION of the merged weight (same byte count: a layer may replace its merged weight in place).
The arithmetic mirrors cuDNN Frontend's ``qkvg_from_vllm(..., index_qk_proj_weight=)`` exactly (the vendored
``_cudnn_frost.gated_attention_block.api``) and the tests pin the five results bitwise; this twin exists so the loader never needs
the CuTe DSL importable, which the vendored tree does at import time.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import torch

from .support import Qwen38BlockGeometry


class Qwen38BlockWeights(NamedTuple):
    """The block's per-layer weights: ``w_qkvg [n_qkvg, d_model]`` with the ``Q | GATE | K | V | INDEX`` bands (the indexer band = its
    query heads then its raw key head, the checkpoint's row order), the two QK-RMSNorm weights ``[head_dim]`` (``None`` under
    ``geometry.qk_norm=False``) and the two indexer RMSNorm weights ``[index_head_dim]`` (the queries', the compressed keys'), every
    norm in the form the kernels multiply by (``(1 + w)``)."""

    w_qkvg: torch.Tensor
    w_q_norm: Optional[torch.Tensor]
    w_k_norm: Optional[torch.Tensor]
    w_iq_norm: torch.Tensor
    w_ik_norm: torch.Tensor


def _norm(name: str, w, width: int, act_dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(w, torch.Tensor):
        raise ValueError(
            f"{name} is required (a zero-centered [{width}] RMSNorm weight vector), got {type(w).__name__}"
        )
    if tuple(w.shape) != (width,):
        raise ValueError(
            f"{name} must be [{width}] (one RMSNorm weight per head dim), got {tuple(w.shape)}"
        )
    # The checkpoint's RMSNorm is x * rsqrt(mean(x^2) + eps) * (1 + w); the kernels multiply by the vector they are given:
    # (1 + w) formed in fp32 and rounded ONCE into the activation dtype (at most 2^-8 relative per channel in bf16).
    return (1.0 + w.float()).to(act_dtype).contiguous()


def qwen_3_8_block_pack_weights(
    qkv_proj_weight: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    geometry: Qwen38BlockGeometry,
    *,
    act_dtype: torch.dtype = torch.bfloat16,
    index_qk_proj_weight: Optional[torch.Tensor] = None,
    index_q_norm_weight: Optional[torch.Tensor] = None,
    index_k_norm_weight: Optional[torch.Tensor] = None,
) -> Qwen38BlockWeights:
    """See :func:`flashinfer.qwen_3_8_block.qwen_3_8_block_pack_weights` (the public, decorated entry)."""
    if not isinstance(geometry, Qwen38BlockGeometry):
        raise TypeError(
            f"geometry must be a Qwen38BlockGeometry, got {type(geometry).__name__}"
        )
    if not isinstance(act_dtype, torch.dtype) or not act_dtype.is_floating_point:
        raise ValueError(
            f"act_dtype must be a floating-point torch dtype (the block's activation dtype), got {act_dtype!r}"
        )
    if not isinstance(qkv_proj_weight, torch.Tensor) or qkv_proj_weight.dim() != 2:
        got = (
            tuple(qkv_proj_weight.shape)
            if isinstance(qkv_proj_weight, torch.Tensor)
            else type(qkv_proj_weight).__name__
        )
        raise ValueError(
            f"qkv_proj_weight must be a 2-D [rows, d_model] weight (a merged QKV linear's weight at this rank), got {got}"
        )
    d, hq, hkv, dm = (
        geometry.head_dim,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
        geometry.d_model,
    )
    rows_q, rows_kv = 2 * hq * d, hkv * d
    rows_qkv = rows_q + 2 * rows_kv
    rows_band = geometry.index_band_cols
    n_rows = int(qkv_proj_weight.shape[0])
    if qkv_proj_weight.shape[1] != dm:
        raise ValueError(
            f"qkv_proj_weight must have {dm} columns (d_model), got {qkv_proj_weight.shape[1]}"
        )
    layout = (
        f"[{rows_q} q_proj rows (per-head [q_h | gate_h]) | {rows_kv} k_proj | {rows_kv} v_proj] at this rank's geometry "
        f"(num_qo_heads={hq}, num_kv_heads={hkv}, head_dim={d})"
    )
    if n_rows == rows_qkv + rows_band:
        # the packed form: the indexer projection's rows ride the merged weight, below v_proj (a stack that packs the replicated
        # indexer projection into the same GEMM)
        if index_qk_proj_weight is not None:
            raise ValueError(
                f"index_qk_proj_weight: the indexer band's {rows_band} rows were given twice -- appended to qkv_proj_weight "
                f"({n_rows} = {rows_qkv} + {rows_band} rows) AND as index_qk_proj_weight; pass one of the two"
            )
        index_qk_proj_weight = qkv_proj_weight[rows_qkv:]
    elif n_rows != rows_qkv:
        raise ValueError(
            f"qkv_proj_weight must have {rows_qkv} rows = {layout}, or {rows_qkv + rows_band} with the indexer band's {rows_band} rows "
            f"appended; got {n_rows}"
        )
    elif index_qk_proj_weight is None:
        raise ValueError(
            f"index_qk_proj_weight is required: the Qwen 3.8 block's W_qkvg carries a fifth band of {rows_band} indexer columns below V "
            f"(the indexer's {geometry.index_heads} query heads then its {geometry.index_kv_heads} raw key head, {geometry.index_head_dim} "
            f"rows each); pass the indexer projection, or hand qkv_proj_weight with its {rows_band} rows appended ({rows_qkv + rows_band} rows)"
        )
    if not isinstance(index_qk_proj_weight, torch.Tensor) or tuple(
        index_qk_proj_weight.shape
    ) != (rows_band, dm):
        got = (
            tuple(index_qk_proj_weight.shape)
            if isinstance(index_qk_proj_weight, torch.Tensor)
            else type(index_qk_proj_weight).__name__
        )
        raise ValueError(
            f"index_qk_proj_weight must be [{rows_band}, {dm}] (the indexer's {geometry.index_heads} query heads then its "
            f"{geometry.index_kv_heads} raw key head, {geometry.index_head_dim} rows each, over d_model), got {got}"
        )
    if index_qk_proj_weight.dtype != qkv_proj_weight.dtype:
        raise ValueError(
            f"index_qk_proj_weight must have qkv_proj_weight's dtype {qkv_proj_weight.dtype}, got {index_qk_proj_weight.dtype}"
        )
    q_gate = qkv_proj_weight[:rows_q].reshape(hq, 2 * d, dm)
    w_k = qkv_proj_weight[rows_q : rows_q + rows_kv]
    w_v = qkv_proj_weight[rows_q + rows_kv : rows_qkv]
    # the Q band, the GATE band (the per-head [q_h | gate_h] interleave undone), K and V as they are, the indexer band as the
    # checkpoint stores it (query heads then the raw key head)
    w_qkvg = (
        torch.cat(
            [
                q_gate[:, :d].reshape(hq * d, dm),
                q_gate[:, d:].reshape(hq * d, dm),
                w_k,
                w_v,
                index_qk_proj_weight,
            ],
            dim=0,
        )
        .to(act_dtype)
        .contiguous()
    )
    norms: List[Optional[torch.Tensor]] = []
    for name, w in (("q_norm_weight", q_norm_weight), ("k_norm_weight", k_norm_weight)):
        if not geometry.qk_norm:
            if w is not None:
                raise ValueError(
                    f"{name}: geometry.qk_norm=False -- the block applies no QK-RMSNorm and takes no norm weights; pass None"
                )
            norms.append(None)
            continue
        if w is None:
            raise ValueError(
                f"{name} is required when geometry.qk_norm=True (a zero-centered [head_dim] vector; Qwen initialises it to zeros)"
            )
        norms.append(_norm(name, w, d, act_dtype))
    # the two indexer norms are REQUIRED: the geometry always declares the in-block indexer
    w_iq = _norm(
        "index_q_norm_weight", index_q_norm_weight, geometry.index_head_dim, act_dtype
    )
    w_ik = _norm(
        "index_k_norm_weight", index_k_norm_weight, geometry.index_head_dim, act_dtype
    )
    return Qwen38BlockWeights(w_qkvg, norms[0], norms[1], w_iq, w_ik)
