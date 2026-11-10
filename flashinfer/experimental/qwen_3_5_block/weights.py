# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time weight packing: a serving stack's merged QKV linear -> the block's ``W_qkvg`` bands and norm weights.

Pure torch, CPU-safe, a row PERMUTATION of the merged weight (same byte count: a layer may replace its merged weight in
place).  The arithmetic mirrors cuDNN Frontend's ``qkvg_from_vllm`` / ``qkvg_from_hf`` exactly (the vendored
``_cudnn_frost.gated_attention_block.api``) and the tests pin the two bitwise; this twin exists so the loader never needs
the CuTe DSL importable, which the vendored tree does at import time.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import torch

from .support import Qwen35BlockGeometry


class Qwen35BlockWeights(NamedTuple):
    """The block's per-layer weights: ``w_qkvg [n_qkvg, d_model]`` with the ``Q | GATE | K | V`` bands, and the two QK-RMSNorm
    weights ``[head_dim]`` in the form the kernels multiply by (``None`` under ``geometry.qk_norm=False``)."""

    w_qkvg: torch.Tensor
    w_q_norm: Optional[torch.Tensor]
    w_k_norm: Optional[torch.Tensor]


def qwen_3_5_block_pack_weights(
    qkv_proj_weight: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    geometry: Qwen35BlockGeometry,
    *,
    act_dtype: torch.dtype = torch.bfloat16,
) -> Qwen35BlockWeights:
    """See :func:`flashinfer.qwen_3_5_block.qwen_3_5_block_pack_weights` (the public, decorated entry)."""
    if not isinstance(geometry, Qwen35BlockGeometry):
        raise TypeError(
            f"geometry must be a Qwen35BlockGeometry, got {type(geometry).__name__}"
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
    rows = rows_q + 2 * rows_kv
    if qkv_proj_weight.shape[1] != dm:
        raise ValueError(
            f"qkv_proj_weight must have {dm} columns (d_model), got {qkv_proj_weight.shape[1]}"
        )
    if qkv_proj_weight.shape[0] != rows:
        raise ValueError(
            f"qkv_proj_weight must have {rows} rows = [{rows_q} q_proj rows (per-head [q_h | gate_h]) | {rows_kv} k_proj | {rows_kv} v_proj] "
            f"at this rank's geometry (num_qo_heads={hq}, num_kv_heads={hkv}, head_dim={d}); got {qkv_proj_weight.shape[0]}"
        )
    q_gate = qkv_proj_weight[:rows_q].reshape(hq, 2 * d, dm)
    w_k = qkv_proj_weight[rows_q : rows_q + rows_kv]
    w_v = qkv_proj_weight[rows_q + rows_kv : rows]
    # the Q band, the GATE band (the per-head [q_h | gate_h] interleave undone), then K and V as they are
    w_qkvg = (
        torch.cat(
            [
                q_gate[:, :d].reshape(hq * d, dm),
                q_gate[:, d:].reshape(hq * d, dm),
                w_k,
                w_v,
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
                f"{name} is required when geometry.qk_norm=True (a zero-centered [head_dim] vector; Qwen3.5 initialises it to zeros)"
            )
        if tuple(w.shape) != (d,):
            raise ValueError(
                f"{name} must be [head_dim={d}] (one RMSNorm weight per head dim), got {tuple(w.shape)}"
            )
        # The checkpoint's RMSNorm is x * rsqrt(mean(x^2) + eps) * (1 + w); the kernels multiply by the vector they are given:
        # (1 + w) formed in fp32 and rounded ONCE into the activation dtype (at most 2^-8 relative per channel in bf16).
        norms.append((1.0 + w.float()).to(act_dtype).contiguous())
    return Qwen35BlockWeights(w_qkvg, norms[0], norms[1])
