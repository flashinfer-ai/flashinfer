# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Score Recompute Interface (SM100 Cute-DSL wrapper).

Orchestrates compilation, caching, and kernel dispatch for:
  - sparse_indexer_score_recompute: sparse index scores + softmax (predict)
  - sparse_attn_score_recompute: sparse attention scores + L1-norm (target)
  - dense_indexer_score_recompute: dense index scores + LSE denom
  - dense_attn_score_recompute: dense attention scores + L1-norm denom
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute

import cuda.bindings.driver as cuda

from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100
from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100
from .dense_score_recompute_sm100_mxfp8 import BwdDenseAttnScoreSm100Mxfp8
from .indexer_score_unified_sm100 import IndexerScoreUnifiedSm100
from ._block_scaled import (
    BLOCK_SCALED_FORMATS,
    BLOCK_SCALED_PRECISION_NAMES,
    kernel_qk_view,
    logical_head_dim,
    validate_indexer_precision,
    validate_sf_vec_size,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.compiler import compile_options
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import (
    ceil_div as _ceil_div,
    device_major as _get_device_capability,
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
    validate_q_causal_offsets,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# CUDA exposes 227 KiB (232448 bytes) as B200's maximum opt-in dynamic shared
# memory per block.  Budgeting the nominal 228 KiB asks the driver to launch an
# impossible 233472-byte kernel at H128/D512.
_SM100_SMEM_BYTES = 227 * 1024


# Execute-path contract checks (Rule 1 / R5): raise, never .contiguous() / .to(dtype).
def _require_unit_innermost(t: Optional[torch.Tensor], name: str) -> None:
    # A size-1 innermost dim has no observable stride (e.g. the SM90 (B, H, 1) per-head view).
    if t is not None and t.shape[-1] != 1 and t.stride(-1) != 1:
        raise ValueError(f"{name} must have a unit innermost stride, got strides {tuple(t.stride())}")


def _require_int32_contiguous(t: torch.Tensor, name: str) -> None:
    if t.dtype != torch.int32 or not t.is_contiguous() or not t.is_cuda:
        raise ValueError(f"{name} must be a contiguous int32 CUDA tensor, got dtype {t.dtype} strides {tuple(t.stride())} device {t.device}")


def _require_fp32_output(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(t.stride())}")
    if t.dtype != torch.float32 or tuple(t.shape) != tuple(shape) or not t.is_cuda:
        raise ValueError(f"{name} must be a float32 CUDA tensor of shape {tuple(shape)}, got dtype {t.dtype} shape {tuple(t.shape)}")


def _normalize_dense_precision(
    precision: str,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
) -> str:
    precision = precision.lower()
    if precision not in ("bf16", "mxfp8"):
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")
    # validate_indexer_precision reports a format dtype missing from the installed torch by name, not as a dtype mismatch.
    precision = validate_indexer_precision(precision)
    if precision == "bf16" and (q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None):
        raise ValueError("MXFP8 scales and scale padded cu_seqlens require precision='mxfp8'")
    return precision


def _normalize_dense_indexer_precision(
    precision: str,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
) -> str:
    """Normalize a dense indexer score ``precision`` and reject block scales passed with ``precision='bf16'``."""
    precision = validate_indexer_precision(precision)
    if precision == "bf16" and (q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None):
        raise ValueError(f"Block scales and scale padded cu_seqlens require precision={BLOCK_SCALED_PRECISION_NAMES}")
    return precision


def _validate_dense_block_scaled_inputs(
    *,
    precision: str,
    q: torch.Tensor,
    k: torch.Tensor,
    per_head: torch.Tensor,
    per_head_name: str,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    sf_vec_size: int,
) -> None:
    """Check the dtype, device and sf_vec_size contract of a block-scaled dense score's Q/K, per-head tensor and scales.

    ``precision`` is a validated block-scaled precision. Run it before any shape check, so Q/K of another format
    are reported as a dtype mismatch rather than as a wrong head_dim.
    """
    fmt = BLOCK_SCALED_FORMATS[precision]
    if q.dtype != fmt.data_dtype or k.dtype != fmt.data_dtype:
        raise TypeError(f"precision={precision!r} requires q and k to be {fmt.data_dtype}")
    if per_head_name == "weights" and per_head.dtype != torch.bfloat16:
        raise TypeError(f"precision={precision!r} requires weights to be torch.bfloat16")
    if per_head_name == "lse" and per_head.dtype != torch.float32:
        raise TypeError(f"precision={precision!r} requires lse to be torch.float32")
    if q_scale is None or k_scale is None:
        raise ValueError(f"precision={precision!r} requires q_scale and k_scale")
    if q_scale.dtype != fmt.scale_dtype or k_scale.dtype != fmt.scale_dtype:
        raise TypeError(f"precision={precision!r} requires q_scale and k_scale to be {fmt.scale_dtype}")
    for scale, reference, name in ((q_scale, q, "q_scale"), (k_scale, k, "k_scale")):
        if scale.device != reference.device:
            raise ValueError(f"{name} must be on the same device as its data tensor")
    validate_sf_vec_size(precision, sf_vec_size)


def _validate_dense_block_scaled_contract(
    *,
    precision: str,
    q: torch.Tensor,
    per_head_name: str,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor],
    cu_seqlens_k_scale_padded: Optional[torch.Tensor],
    qhead_per_kv_head: int,
    head_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    bs: int,
    n_heads_q: int,
    n_heads_kv: int,
    sf_vec_size: int,
    is_varlen: bool,
) -> None:
    """Check the head and packed-scale-shape contract of a block-scaled dense score.

    ``q_scale`` / ``k_scale`` are present and dtype-checked. ``head_dim`` is the logical head dim (twice the
    stored last dim for packed FP4).
    """
    label = precision.upper()
    if head_dim != 128 and not (per_head_name == "lse" and head_dim == 512):
        raise ValueError(f"precision={precision!r} currently supports head_dim=128, " "or attention head_dim=512")
    if n_heads_kv != 1:
        raise ValueError(f"precision={precision!r} dense score currently requires n_heads_kv=1")
    if n_heads_q != n_heads_kv * qhead_per_kv_head:
        raise ValueError(f"n_heads_q must equal n_heads_kv * qhead_per_kv_head for precision={precision!r}")

    sf_groups = _ceil_div(head_dim, sf_vec_size)
    sf_groups_padded = _ceil_div(sf_groups, 4) * 4
    if is_varlen:
        if cu_seqlens_q_scale_padded is None or cu_seqlens_k_scale_padded is None:
            raise ValueError(f"THD {label} requires Q/K scale padded cu_seqlens")
        for prefix, name in (
            (cu_seqlens_q_scale_padded, "cu_seqlens_q_scale_padded"),
            (cu_seqlens_k_scale_padded, "cu_seqlens_k_scale_padded"),
        ):
            if prefix.dtype != torch.int32 or prefix.ndim != 1 or prefix.shape[0] != bs + 1 or not prefix.is_contiguous() or prefix.device != q.device:
                raise ValueError(f"{name} must be contiguous int32 shape ({bs + 1},) on {q.device}")
        for scale, name in ((q_scale, "q_scale"), (k_scale, "k_scale")):
            if scale.ndim != 3 or scale.shape[0] != n_heads_kv or scale.shape[1] % 128 != 0 or scale.shape[2] != sf_groups_padded:
                raise ValueError(f"THD {name} must have shape ({n_heads_kv}, multiple_of_128, " f"{sf_groups_padded}), got {tuple(scale.shape)}")
    else:
        if cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
            raise ValueError(f"BSHD {label} requires scale padded cu_seqlens to be None")
        q_mn = seqlen_q * qhead_per_kv_head
        q_shape = (
            bs * n_heads_kv,
            _ceil_div(q_mn, 128) * 128,
            sf_groups_padded,
        )
        k_shape = (
            bs * n_heads_kv,
            _ceil_div(seqlen_k, 128) * 128,
            sf_groups_padded,
        )
        if tuple(q_scale.shape) != q_shape:
            raise ValueError(f"q_scale packed shape must be {q_shape}, got {tuple(q_scale.shape)}")
        if tuple(k_scale.shape) != k_shape:
            raise ValueError(f"k_scale packed shape must be {k_shape}, got {tuple(k_scale.shape)}")


def _sparse_indexer_score_recompute(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    *,
    out: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 128,
    sm_scale: float = 1.0,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Compute sparse index scores + softmax (predict) for backward pass.

    Args:
        q_indexer: (bs, seqlen_q, n_heads_q, head_dim) [BF16], unit innermost stride
        k_indexer: (bs, seqlen_k, head_dim) [BF16] — no head dim (MQA), unit innermost stride
        weights: (bs, seqlen_q, n_heads_q) [BF16], unit innermost stride
        topk_indices: (bs, seqlen_q, topk) [INT32], contiguous
        qhead_per_kv_head: auto inferred if None
        out: required output tensor (bs, seqlen_q, topk) [FP32], contiguous
        topk_length: (bs, seqlen_q) [INT32], contiguous — per-q valid topk count (compact layout), optional
        m_block_size: defaults to qhead_per_kv_head (must equal qhead_per_kv_head)

    Layout/dtype mismatches raise ``ValueError``; nothing is copied or cast here.

    Returns:
        predict: (bs, seqlen_q, topk) [FP32] — softmax of sparse index scores
    """
    current_stream = _resolve_stream(current_stream)
    for t, name in ((q_indexer, "q_indexer"), (k_indexer, "k_indexer"), (weights, "weights")):
        _require_unit_innermost(t, name)
    _require_int32_contiguous(topk_indices, "topk_indices")

    if qhead_per_kv_head is None:
        qhead_per_kv_head = q_indexer.shape[2]

    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    bs, seqlen_q, n_heads_q, head_dim = q_indexer.shape
    _, seqlen_k, _ = k_indexer.shape
    topk = topk_indices.shape[2]

    have_topk_length = topk_length is not None
    if have_topk_length:
        _require_int32_contiguous(topk_length, "topk_length")
    _require_fp32_output(out, "out", (bs, seqlen_q, topk))

    # Compute kv_stage and topk_in_smem from the usable SM100 SMEM budget.
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    sK_per_stage = n_block_size * head_dim_padded * 2  # BF16
    sQ_size = m_block_size * head_dim_padded * 2  # BF16
    sTopkIdx_bytes = topk * 2 * 4  # double-buffer, INT32
    sPerHead_bytes = m_block_size * 2 * 2  # double-buffer, BF16
    smem_fixed = sPerHead_bytes + 2048  # barriers, sScoreAll, alignment

    topk_in_smem = True
    smem_overhead = sTopkIdx_bytes + smem_fixed
    kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
    total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
    if total_smem_est > _SM100_SMEM_BYTES:
        topk_in_smem = False
        smem_overhead = smem_fixed
        kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
        total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
        assert total_smem_est <= _SM100_SMEM_BYTES, (
            f"SMEM overflow ({total_smem_est} > {_SM100_SMEM_BYTES}) even without sTopkIdx: "
            f"topk={topk}, head_dim={head_dim}(padded={head_dim_padded}), "
            f"m_block={m_block_size}, n_block={n_block_size}, kv_stage={kv_stage}."
        )

    compute_capability = _get_device_capability()

    if compute_capability >= 10:
        # sm_scale is a runtime arg (passed as cutlass.Float32), so it is
        # excluded from compile_key.
        # topk_indices_global bakes in (controls compile-time global→local
        # decode of topk ids).
        compile_key = (
            "indexer",
            q_indexer.dtype,
            head_dim,
            qhead_per_kv_head,
            topk,
            m_block_size,
            n_block_size,
            kv_stage,
            have_topk_length,
            topk_in_smem,
            topk_indices_global,
        )

        if compile_key not in _sparse_indexer_score_recompute.compile_cache:
            q_cute = to_cute_tensor(q_indexer)
            k_cute = to_cute_tensor(k_indexer)
            w_cute = to_cute_tensor(weights)
            topk_cute = to_cute_tensor(topk_indices)
            out_cute = to_cute_tensor(out)
            # R3: the dead mTopkLength slot is compiled out (None at compile AND launch).
            topk_length_cute = to_cute_tensor(topk_length) if have_topk_length else None

            kernel_obj = SparseScoreRecomputeSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                topk=topk,
                kv_stage=kv_stage,
                score_type="indexer",
                have_topk_length=have_topk_length,
                topk_in_smem=topk_in_smem,
                topk_indices_global=topk_indices_global,
            )

            scale_arg = cutlass.Float32(sm_scale)

            _sparse_indexer_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                topk_cute,
                out_cute,
                topk_length_cute,
                scale_arg,
                current_stream,
                options=compile_options(),
            )

        scale_arg = cutlass.Float32(sm_scale)
        with torch.cuda.nvtx.range("sparse_indexer_score_recompute"):
            _sparse_indexer_score_recompute.compile_cache[compile_key](
                q_indexer,
                k_indexer,
                weights,
                topk_indices,
                out,
                topk_length,
                scale_arg,
                current_stream,
            )
        return out

    raise NotImplementedError(f"Sparse indexer backward score requires SM100+ (got compute capability {compute_capability}).")


_sparse_indexer_score_recompute.compile_cache = {}


def sparse_indexer_score_recompute(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    *,
    out: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Public entry point for sparse indexer backward predict computation.

    Computes sparse scores and applies softmax:
      S[b,q,i] = sm_scale * sum_h [ReLU(Q_h · K_{topk[b,q,i]}^T) · W_{b,q,h}]
      predict = softmax(S, dim=-1)  (invalid positions masked as -inf)

    sm_scale is applied to the fp32 head-reduced score inside the kernel
    (preserves precision vs pre-multiplying onto bf16 weights on the host).

    Args:
        q_indexer: (bs, seqlen_q, n_heads_q, head_dim) [BF16], unit innermost stride
        k_indexer: (bs, seqlen_k, head_dim) [BF16], unit innermost stride
        weights: (bs, seqlen_q, n_heads_q) [BF16], unit innermost stride
        topk_indices: (bs, seqlen_q, topk) [INT32], contiguous
        qhead_per_kv_head: auto inferred if None
        out: required pre-allocated output (bs, seqlen_q, topk) [FP32], contiguous
        topk_length: (bs, seqlen_q) [INT32], contiguous — compact layout, optional
        sm_scale: scalar applied to fp32 score post head-reduce; default 1.0
        topk_indices_global: when True (default, matches public fwd output),
            ``topk_indices`` are global ids ``b * seqlen_k + local`` and the
            kernel decodes back to local. When False, ids are already local.

    Returns:
        predict: (bs, seqlen_q, topk) [FP32]
    """
    return _sparse_indexer_score_recompute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        qhead_per_kv_head,
        out=out,
        topk_length=topk_length,
        sm_scale=sm_scale,
        topk_indices_global=topk_indices_global,
        current_stream=current_stream,
    )


# =============================================================================
# Sparse attention score: exp(QK*scale - LSE) -> head reduce -> L1-norm
# =============================================================================


def _sparse_attn_score_recompute(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    *,
    out: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 64,
    k_block_size: Optional[int] = None,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Compute sparse attention target (L1-normalized head-summed softmax) for backward.

    Args:
        q_attn: (bs, seqlen_q, n_heads_q, head_dim) [BF16], unit innermost stride
        k_attn: (bs, seqlen_k, head_dim) [BF16], unit innermost stride
        lse: (bs, seqlen_q, n_heads_q) [FP32], unit innermost stride — logsumexp from forward softmax
        topk_indices: (bs, seqlen_q, topk) [INT32], contiguous
        softmax_scale: float — attention softmax scale
        qhead_per_kv_head: auto inferred if None
        out: required output tensor (bs, seqlen_q, topk) [FP32], contiguous
        topk_length: (bs, seqlen_q) [INT32], contiguous — per-q valid topk count (compact layout), optional
        m_block_size: defaults to qhead_per_kv_head (must equal qhead_per_kv_head)

    Layout/dtype mismatches raise ``ValueError``; nothing is copied or cast here.

    Returns:
        target: (bs, seqlen_q, topk) [FP32] — L1-normalized attention scores
    """

    current_stream = _resolve_stream(current_stream)
    for t, name in ((q_attn, "q_attn"), (k_attn, "k_attn"), (lse, "lse")):
        _require_unit_innermost(t, name)
    _require_int32_contiguous(topk_indices, "topk_indices")

    if qhead_per_kv_head is None:
        qhead_per_kv_head = q_attn.shape[2]

    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    bs, seqlen_q, n_heads_q, head_dim = q_attn.shape
    _, seqlen_k, _ = k_attn.shape
    topk = topk_indices.shape[2]

    have_topk_length = topk_length is not None
    if have_topk_length:
        _require_int32_contiguous(topk_length, "topk_length")
    _require_fp32_output(out, "out", (bs, seqlen_q, topk))

    # Compute kv_stage and topk_in_smem from the usable SM100 SMEM budget.
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    k_block_size_eff = k_block_size if k_block_size is not None else head_dim_padded
    sK_per_stage = n_block_size * k_block_size_eff * 2  # BF16
    sQ_size = m_block_size * head_dim_padded * 2  # BF16 (Q total unchanged)
    sTopkIdx_bytes = topk * 2 * 4  # double-buffer, INT32
    sPerHead_bytes = m_block_size * 2 * 4  # double-buffer, FP32
    smem_fixed = sPerHead_bytes + 2048  # barriers, sScoreAll, alignment

    topk_in_smem = True
    smem_overhead = sTopkIdx_bytes + smem_fixed
    kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
    total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
    if total_smem_est > _SM100_SMEM_BYTES:
        topk_in_smem = False
        smem_overhead = smem_fixed
        kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
        total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
        assert total_smem_est <= _SM100_SMEM_BYTES, (
            f"SMEM overflow ({total_smem_est} > {_SM100_SMEM_BYTES}) even without sTopkIdx: "
            f"topk={topk}, head_dim={head_dim}(padded={head_dim_padded}), "
            f"m_block={m_block_size}, n_block={n_block_size}, kv_stage={kv_stage}."
        )

    compute_capability = _get_device_capability()

    if compute_capability >= 10:
        compile_key = (
            "attention",
            q_attn.dtype,
            head_dim,
            qhead_per_kv_head,
            topk,
            m_block_size,
            n_block_size,
            k_block_size,
            kv_stage,
            have_topk_length,
            topk_in_smem,
            topk_indices_global,
        )

        if compile_key not in _sparse_attn_score_recompute.compile_cache:
            q_cute = to_cute_tensor(q_attn)
            k_cute = to_cute_tensor(k_attn)
            lse_cute = to_cute_tensor(lse)
            topk_cute = to_cute_tensor(topk_indices)
            out_cute = to_cute_tensor(out)
            # R3: the dead mTopkLength slot is compiled out (None at compile AND launch).
            topk_length_cute = to_cute_tensor(topk_length) if have_topk_length else None

            kernel_obj = SparseScoreRecomputeSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                topk=topk,
                kv_stage=kv_stage,
                score_type="attention",
                have_topk_length=have_topk_length,
                topk_in_smem=topk_in_smem,
                topk_indices_global=topk_indices_global,
            )

            _sparse_attn_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                lse_cute,
                topk_cute,
                out_cute,
                topk_length_cute,
                cutlass.Float32(softmax_scale),
                current_stream,
                # O2 avoids the n128 attention kernel's register-spill
                # regression in CuTeDSL 4.6.2. Keep the default optimization
                # for n64 configurations, which can regress at O2.
                options=compile_options("--opt-level 2" if n_block_size == 128 else ""),
            )

        with torch.cuda.nvtx.range("sparse_attn_score_recompute"):
            _sparse_attn_score_recompute.compile_cache[compile_key](
                q_attn,
                k_attn,
                lse,
                topk_indices,
                out,
                topk_length,
                cutlass.Float32(softmax_scale),
                current_stream,
            )
        return out

    raise NotImplementedError(f"Sparse attention backward score requires SM100+ (got compute capability {compute_capability}).")


_sparse_attn_score_recompute.compile_cache = {}


def _dispatch_sparse_attn_tile_params(
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    compact: bool = False,
):
    """Select (m_block_size, n_block_size, k_block_size) for sparse attention backward.

    Returns (m_block_size, n_block_size, k_block_size) where k_block_size=None
    means no head_dim splitting.

    Compact (have_topk_length) and non-compact have different optimal configs:
    compact benefits from finer-grained early termination (smaller n, less
    k-split overhead), non-compact benefits from n=128 + k-split for better
    pipeline utilization.

    Rules tuned on B200 via tests/sweep_tile_params.py (--compact / default).
    """
    m = qhead_per_kv_head

    # H96 cannot use the default n64 attention epilogue: Ld16x64bOp would
    # require Repetition(96 // 2), but 48 is not a supported TMEM repetition.
    # The existing n128 epilogue uses its fixed Repetition(8) load and covers
    # all 96 heads natively.  Split D576 into three K192 chunks to retain two
    # KV pipeline stages within the SM100 shared-memory budget.
    if head_dim == 576 and qhead_per_kv_head == 96:
        return m, 128, 192

    if compact:
        # --- Compact tuned configs ---
        # hd=512
        if head_dim == 512 and qhead_per_kv_head == 64:
            return m, 64, None  # kvs=2, ~789 TFLOPS
        if head_dim == 512 and qhead_per_kv_head == 128:
            return m, 128, 128  # kvs=2, ~844/1004 TFLOPS
        # hd=576
        if head_dim == 576 and qhead_per_kv_head == 64:
            return m, 64, None  # kvs=2, ~827 TFLOPS
        if head_dim == 576 and qhead_per_kv_head == 128:
            return m, 64, 192  # kvs=2~3, ~677/1054 TFLOPS
        return m, 64, None

    # --- Non-compact tuned configs ---
    # hd=512
    if head_dim == 512 and qhead_per_kv_head == 64:
        return m, 128, 256  # kvs=2, ~800 TFLOPS
    if head_dim == 512 and qhead_per_kv_head == 128 and topk <= 512:
        return m, 128, 128  # kvs=2, ~1000 TFLOPS
    if head_dim == 512 and qhead_per_kv_head == 128:
        return m, 128, 256  # kvs=1, ~910 TFLOPS
    # hd=576
    if head_dim == 576 and qhead_per_kv_head == 64:
        return m, 128, 192  # kvs=3, ~770 TFLOPS
    if head_dim == 576 and qhead_per_kv_head == 128:
        return m, 128, 192  # kvs=1, ~830 TFLOPS

    return m, 64, None


def sparse_attn_score_recompute(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    *,
    out: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Public entry point for sparse attention backward target computation.

    Recovers per-head softmax from LSE, sums across heads, L1-normalizes:
      P[b,q,h,i] = exp(Q_h · K_{topk[b,q,i]}^T · scale - LSE[b,q,h])
      S[b,q,i] = sum_h P[b,q,h,i]
      target = S / sum(S)  (L1-norm over topk dim)

    Args:
        q_attn: (bs, seqlen_q, n_heads_q, head_dim) [BF16], unit innermost stride
        k_attn: (bs, seqlen_k, head_dim) [BF16], unit innermost stride
        lse: (bs, seqlen_q, n_heads_q) [FP32], unit innermost stride
        topk_indices: (bs, seqlen_q, topk) [INT32], contiguous
        softmax_scale: float
        qhead_per_kv_head: auto inferred if None
        out: required pre-allocated output (bs, seqlen_q, topk) [FP32], contiguous
        topk_length: (bs, seqlen_q) [INT32], contiguous — compact layout, optional

    Returns:
        target: (bs, seqlen_q, topk) [FP32]
    """
    if qhead_per_kv_head is None:
        qhead_per_kv_head = q_attn.shape[2]
    head_dim = q_attn.shape[3]
    topk = topk_indices.shape[2]

    compact = topk_length is not None
    m_block_size, n_block_size, k_block_size = _dispatch_sparse_attn_tile_params(
        head_dim,
        qhead_per_kv_head,
        topk,
        compact=compact,
    )

    # Optimization: when compact and non-compact tile configs match,
    # use non-compact kernel path (have_topk_length=False) for better
    # codegen — static loop bounds avoid dynamic branch overhead that
    # causes 18-28% regression in compact attention.
    # Invalid positions (topk_idx=-1) are skipped in K load and masked
    # in epilogue, preserving correctness.
    if compact:
        nc_params = _dispatch_sparse_attn_tile_params(
            head_dim,
            qhead_per_kv_head,
            topk,
            compact=False,
        )
        if (m_block_size, n_block_size, k_block_size) == nc_params:
            topk_length = None  # use non-compact kernel path

    return _sparse_attn_score_recompute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        softmax_scale,
        qhead_per_kv_head,
        out=out,
        topk_length=topk_length,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
        topk_indices_global=topk_indices_global,
        current_stream=current_stream,
    )


# =============================================================================
# Dense backward: full KV via TMA, no topk
# =============================================================================


def _select_dense_k_block_size(head_dim_padded, m_block_size, n_block_size, per_head_elem_bytes):
    """Auto-select k_block_size so that sQ + sK(1 stage) + overhead fits in SMEM.

    When head_dim_padded is too large (e.g. m_block=128, hd=512 -> sQ=128KB,
    sK=128KB > 228KB), we reduce k_block_size to shrink sK per stage.
    sQ total = m_block * head_dim_padded * 2 (fixed, since Q has num_k_chunks stages).

    k_block_size must be a multiple of 64 and divide head_dim_padded evenly.
    """
    sQ_size = m_block_size * head_dim_padded * 2
    sPerHead_bytes = m_block_size * 2 * per_head_elem_bytes
    smem_overhead = sPerHead_bytes + 2048

    avail_for_sK = _SM100_SMEM_BYTES - sQ_size - smem_overhead
    max_kbs = max(64, avail_for_sK // (n_block_size * 2))
    max_kbs = (max_kbs // 64) * 64

    kbs = head_dim_padded
    while kbs > max_kbs:
        # Try common divisors of head_dim_padded in descending order
        found = False
        for candidate in range(kbs - 64, 63, -64):
            if candidate > 0 and head_dim_padded % candidate == 0:
                kbs = candidate
                found = True
                break
        if not found:
            kbs = 64
            break

    assert head_dim_padded % kbs == 0, f"Cannot find valid k_block_size: head_dim_padded={head_dim_padded}, " f"m_block={m_block_size}, n_block={n_block_size}"
    return kbs


def _compute_dense_kv_stage(head_dim_padded, m_block_size, n_block_size, k_block_size_eff, per_head_elem_bytes):
    sK_per_stage = n_block_size * k_block_size_eff * 2
    sQ_size = m_block_size * head_dim_padded * 2
    sPerHead_bytes = m_block_size * 2 * per_head_elem_bytes
    smem_overhead = sPerHead_bytes + 2048
    return min(4, max(1, (_SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))


# ---- Dispatch helpers (mirror _dispatch_sparse_attn_tile_params) -------------


def _dense_m_block_with_smem_check(qhead_per_kv_head, head_dim, per_head_elem_bytes):
    """Return m=qhpkv*2 if SMEM fits, else fall back to m=qhpkv."""
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    n = 128
    min_kbs = 64
    smem_overhead = 4096

    m2 = qhead_per_kv_head * 2
    sQ_m2 = m2 * head_dim_padded * 2
    sK_min = n * min_kbs * 2
    sPerHead_m2 = m2 * 2 * per_head_elem_bytes
    if sQ_m2 + sK_min + sPerHead_m2 + smem_overhead <= _SM100_SMEM_BYTES:
        return m2

    return qhead_per_kv_head


def _dispatch_dense_attn_tile_params(
    head_dim,
    qhead_per_kv_head,
    *,
    precision: str = "bf16",
    seqlen_k_hint: Optional[int] = None,
    k_block_size: Optional[int] = None,
):
    """Select (m_block_size, n_block_size, k_block_size) for dense attention backward.

    n_block_size is always 128 (required for Ld32x32bOp TMEM path).
    m_block_size = qhpkv * 2 (2 q_tokens per tile) when SMEM allows,
    falling back to qhpkv when it doesn't (e.g. nh_q=128, hd=512).

    Returns (m_block_size, n_block_size, k_block_size) where k_block_size=None
    means no head_dim splitting.

    Rules tuned on B200 via dense score sweeps.
    """
    precision = precision.lower()
    if precision == "mxfp8":
        if head_dim == 512:
            default_k = 256 if seqlen_k_hint is not None and seqlen_k_hint >= 4096 else 128
        else:
            default_k = 64
        return 128, 128, default_k if k_block_size is None else k_block_size
    if precision != "bf16":
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")

    m = _dense_m_block_with_smem_check(qhead_per_kv_head, head_dim, per_head_elem_bytes=4)
    n = 128

    if head_dim == 512:
        if qhead_per_kv_head == 64:
            return m, n, 128 if k_block_size is None else k_block_size
        if qhead_per_kv_head == 128:
            return m, n, 128 if k_block_size is None else k_block_size
        return m, n, 64 if k_block_size is None else k_block_size
    if head_dim == 576:
        return m, n, 64 if k_block_size is None else k_block_size

    # Fallback: auto-select from SMEM budget
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    k = _select_dense_k_block_size(head_dim_padded, m, n, per_head_elem_bytes=4)
    if k == head_dim_padded:
        k = None
    if k_block_size is not None:
        k = k_block_size
    return m, n, k


def _dispatch_dense_indexer_tile_params(
    head_dim,
    qhead_per_kv_head,
    *,
    precision: str = "bf16",
    k_block_size: Optional[int] = None,
):
    """Select (m_block_size, n_block_size, k_block_size) for indexer score recompute.

    M128 packs four query tokens for 32 heads or two for 64 heads.
    ``head_dim`` is the logical head dim.
    """
    assert head_dim == 128, f"Indexer score requires head_dim=128, got {head_dim}"
    validate_indexer_precision(precision)
    return 128, 128, 64 if k_block_size is None else k_block_size


# ---- Internal dense functions (explicit tile params) ------------------------


def _dense_indexer_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 128,
    k_block_size: Optional[int] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    out: torch.Tensor,
    denom_out: torch.Tensor,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal: dense indexer backward with explicit tile params.

    BSHD layout (default):
        q: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k: (bs, seqlen_k, n_heads_kv, head_dim) [BF16]
        weights: (bs, seqlen_q, n_heads_q) [BF16]
        out: required (bs, seqlen_q, seqlen_k) [FP32], contiguous
        denom_out: required (bs, seqlen_q) [FP32], contiguous — LogSumExp denom

    THD layout (when cu_seqlens_q/k supplied):
        q: (total_q, n_heads_q, head_dim) [BF16]
        k: (total_k, n_heads_kv, head_dim) [BF16]
        weights: (total_q, n_heads_q) [BF16]
        out: required (total_q, max_seqlen_k) [FP32], contiguous
        denom_out: required (total_q,) [FP32], contiguous

    Block-scaled precisions (``BLOCK_SCALED_FORMATS``) take Q/K in the
    format's dtype (packed FP4 has last dim ``head_dim // 2``) plus packed
    ``q_scale`` / ``k_scale``.

    Inputs need a unit innermost stride; layout/dtype mismatches raise
    ``ValueError`` (nothing is copied or cast here).

    Both dense-indexer implementations fill causal-invalid score entries
    with -inf, matching raw-logit mask semantics. Denom/LSE semantics are
    unchanged.

    Returns:
        (out, denom_out)
    """
    current_stream = _resolve_stream(current_stream)
    for t, name in ((q, "q"), (k, "k"), (weights, "weights"), (q_scale, "q_scale"), (k_scale, "k_scale")):
        _require_unit_innermost(t, name)
    precision = _normalize_dense_indexer_precision(
        precision,
        q_scale,
        k_scale,
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    is_block_scaled = precision in BLOCK_SCALED_FORMATS
    if is_block_scaled:
        _validate_dense_block_scaled_inputs(
            precision=precision,
            q=q,
            k=k,
            per_head=weights,
            per_head_name="weights",
            q_scale=q_scale,
            k_scale=k_scale,
            sf_vec_size=sf_vec_size,
        )
    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    assert is_varlen_q == is_varlen_k, "THD input requires both cu_seqlens_q and cu_seqlens_k"
    is_varlen = is_varlen_q

    if is_varlen:
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1 and t.is_cuda, f"{name} must be 1D CUDA tensor"
            assert t.stride(0) == 1, f"{name} must be contiguous"
        assert q.ndim == 3 and k.ndim == 3 and weights.ndim == 2, "THD dense indexer expects q=(total_q,H,D), k=(total_k,H_kv,D), w=(total_q,H)"
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD dense indexer requires max_seqlen_q and max_seqlen_k")
        total_q, n_heads_q, head_dim = q.shape
        total_k, n_heads_kv, head_dim_k = k.shape
        bs = cu_seqlens_q.shape[0] - 1
        seqlen_q = int(max_seqlen_q)
        seqlen_k = int(max_seqlen_k)
        if head_dim != head_dim_k:
            raise ValueError(f"q last dim ({head_dim}) != k last dim ({head_dim_k})")
        assert weights.shape == (total_q, n_heads_q)
        assert cu_seqlens_q.shape == (bs + 1,) and cu_seqlens_k.shape == (bs + 1,)
    else:
        bs, seqlen_q, n_heads_q, head_dim = q.shape
        _, seqlen_k, n_heads_kv, head_dim_k = k.shape
        if head_dim != head_dim_k:
            raise ValueError(f"q last dim ({head_dim}) != k last dim ({head_dim_k})")
    head_dim = logical_head_dim(precision, head_dim)
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    if is_block_scaled:
        if qhead_per_kv_head not in (32, 64):
            raise ValueError(f"precision={precision!r} dense indexer requires qhead_per_kv_head=32 or 64")
        if m_block_size is not None and m_block_size != 128:
            raise ValueError(f"precision={precision!r} dense indexer requires m_block_size=128")
        if n_block_size != 128:
            raise ValueError(f"precision={precision!r} dense indexer requires n_block_size=128")
        m_block_size = 128
        # k_block_size is locked to 64 (the D128 two-chunk plan): kv_stage is
        # hardcoded to 24 below, which keeps K-SMEM within the SM100 budget only
        # at k_block=64 (k_block>64 overflows SMEM; k_block<64 under-pipelines).
        k_block_size = 64 if k_block_size is None else k_block_size
        if k_block_size != 64:
            raise ValueError(f"precision={precision!r} dense indexer requires k_block_size=64; got {k_block_size}")
    elif m_block_size is None:
        m_block_size = qhead_per_kv_head

    if is_block_scaled:
        _validate_dense_block_scaled_contract(
            precision=precision,
            q=q,
            per_head_name="weights",
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            qhead_per_kv_head=qhead_per_kv_head,
            head_dim=head_dim,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            bs=bs,
            n_heads_q=n_heads_q,
            n_heads_kv=n_heads_kv,
            sf_vec_size=sf_vec_size,
            is_varlen=is_varlen,
        )

    _require_fp32_output(out, "out", (q.shape[0], seqlen_k) if is_varlen else (bs, seqlen_q, seqlen_k))
    _require_fp32_output(denom_out, "denom_out", (q.shape[0],) if is_varlen else (bs, seqlen_q))
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    if k_block_size is None:
        k_block_size = _select_dense_k_block_size(head_dim_padded, m_block_size, n_block_size, per_head_elem_bytes=2)
    kv_stage = _compute_dense_kv_stage(head_dim_padded, m_block_size, n_block_size, k_block_size, per_head_elem_bytes=2)
    if is_block_scaled:
        # K-SMEM = n_block * k_block_size(=64) * kv_stage at <= 1 byte per element;
        # 24 fits the SM100 SMEM budget, and k_block is locked to 64 above.
        kv_stage = 24

    compute_capability = _get_device_capability()
    if compute_capability < 10:
        raise NotImplementedError(f"Dense indexer backward requires SM100+ (got {compute_capability}).")

    if is_block_scaled:
        assert q_scale is not None and k_scale is not None
        compile_key = (
            "dense_indexer",
            precision,
            q.dtype,
            k.dtype,
            weights.dtype,
            q_scale.dtype,
            k_scale.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            head_dim_padded // k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
            sf_vec_size,
        )
    else:
        base_compile_key = (
            q.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
        )
        compile_key = ("dense_indexer",) + base_compile_key
    q_kernel = kernel_qk_view(precision, q)
    k_kernel = kernel_qk_view(precision, k)
    if compile_key not in _dense_indexer_score_recompute.compile_cache:
        q_cute = to_cute_tensor(q_kernel)
        k_cute = to_cute_tensor(k_kernel)
        w_cute = to_cute_tensor(weights)
        out_cute = to_cute_tensor(out)
        denom_cute = to_cute_tensor(denom_out)
        cu_q_cute = to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        cu_q_scale_cute = to_cute_tensor(cu_seqlens_q_scale_padded, leading_dim=0) if is_varlen and is_block_scaled else None
        cu_k_scale_cute = to_cute_tensor(cu_seqlens_k_scale_padded, leading_dim=0) if is_varlen and is_block_scaled else None
        q_offsets_cute = to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

        if is_block_scaled:
            assert q_scale is not None and k_scale is not None
            q_scale_cute = to_cute_tensor(q_scale)
            k_scale_cute = to_cute_tensor(k_scale)
            kernel_obj = BLOCK_SCALED_FORMATS[precision].kernel_cls(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                ratio=ratio,
                is_varlen=is_varlen,
                sf_vec_size=sf_vec_size,
                compute_lse=True,
                is_compressed_logits=False,
            )
        else:
            q_scale_cute = None
            k_scale_cute = None
            kernel_obj = IndexerScoreUnifiedSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                ratio=ratio,
                is_varlen=is_varlen,
                compute_lse=True,
                is_compressed_logits=False,
            )
        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q)
        max_k_arg = cutlass.Int32(seqlen_k)

        if is_block_scaled:
            _dense_indexer_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                q_scale_cute,
                k_scale_cute,
                out_cute,
                denom_cute,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                cu_q_scale_cute,
                cu_k_scale_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )
        else:
            _dense_indexer_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                out_cute,
                denom_cute,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )

    # Dense indexer score is a raw-logit buffer, so masked/skipped positions
    # must remain outside the softmax/logsumexp domain.
    with _torch_stream_context(current_stream):
        out.fill_(float("-inf"))
    scale_arg = cutlass.Float32(sm_scale)
    max_q_arg = cutlass.Int32(seqlen_q)
    max_k_arg = cutlass.Int32(seqlen_k)
    with torch.cuda.nvtx.range("dense_indexer_score_recompute"):
        if is_block_scaled:
            _dense_indexer_score_recompute.compile_cache[compile_key](
                q_kernel,
                k_kernel,
                weights,
                q_scale,
                k_scale,
                out,
                denom_out,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                cu_seqlens_q_scale_padded if is_varlen else None,
                cu_seqlens_k_scale_padded if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
        else:
            _dense_indexer_score_recompute.compile_cache[compile_key](
                q,
                k,
                weights,
                out,
                denom_out,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
    return out, denom_out


_dense_indexer_score_recompute.compile_cache = {}


def _dense_attn_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 128,
    k_block_size: Optional[int] = None,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    out: torch.Tensor,
    denom_out: torch.Tensor,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal: dense attention backward with explicit tile params.

    BSHD layout (default):
        q: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k: (bs, seqlen_k, n_heads_kv, head_dim) [BF16]
        lse: (bs, seqlen_q, n_heads_q) [FP32]
        out: required (bs, seqlen_q, seqlen_k) [FP32], contiguous
        denom_out: required (bs, seqlen_q) [FP32], contiguous — L1-norm denom

    THD layout (when cu_seqlens_q/k supplied):
        q: (total_q, n_heads_q, head_dim) [BF16]
        k: (total_k, n_heads_kv, head_dim) [BF16]
        lse: (total_q, n_heads_q) [FP32]
        out: required (total_q, max_seqlen_k) [FP32], contiguous
        denom_out: required (total_q,) [FP32], contiguous

    Inputs need a unit innermost stride; layout/dtype mismatches raise
    ``ValueError`` (nothing is copied or cast here).

    Returns:
        (out, denom_out)
    """
    current_stream = _resolve_stream(current_stream)
    for t, name in ((q, "q"), (k, "k"), (lse, "lse"), (q_scale, "q_scale"), (k_scale, "k_scale")):
        _require_unit_innermost(t, name)
    precision = _normalize_dense_precision(
        precision,
        q_scale,
        k_scale,
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    if precision == "mxfp8":
        _validate_dense_block_scaled_inputs(
            precision=precision,
            q=q,
            k=k,
            per_head=lse,
            per_head_name="lse",
            q_scale=q_scale,
            k_scale=k_scale,
            sf_vec_size=sf_vec_size,
        )

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    assert is_varlen_q == is_varlen_k, "THD input requires both cu_seqlens_q and cu_seqlens_k"
    is_varlen = is_varlen_q

    if is_varlen:
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1 and t.is_cuda, f"{name} must be 1D CUDA tensor"
            assert t.stride(0) == 1, f"{name} must be contiguous"
        assert q.ndim == 3 and k.ndim == 3 and lse.ndim == 2, "THD dense attn expects q=(total_q,H,D), k=(total_k,H_kv,D), lse=(total_q,H)"
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD dense attn requires max_seqlen_q and max_seqlen_k")
        total_q, n_heads_q, head_dim = q.shape
        total_k, n_heads_kv, head_dim_k = k.shape
        bs = cu_seqlens_q.shape[0] - 1
        seqlen_q = int(max_seqlen_q)
        seqlen_k = int(max_seqlen_k)
        assert head_dim == head_dim_k
        assert lse.shape == (total_q, n_heads_q)
        assert cu_seqlens_q.shape == (bs + 1,) and cu_seqlens_k.shape == (bs + 1,)
    else:
        bs, seqlen_q, n_heads_q, head_dim = q.shape
        _, seqlen_k, n_heads_kv, _ = k.shape
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    if precision == "mxfp8":
        _validate_dense_block_scaled_contract(
            precision=precision,
            q=q,
            per_head_name="lse",
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            qhead_per_kv_head=qhead_per_kv_head,
            head_dim=head_dim,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            bs=bs,
            n_heads_q=n_heads_q,
            n_heads_kv=n_heads_kv,
            sf_vec_size=sf_vec_size,
            is_varlen=is_varlen,
        )

    _require_fp32_output(out, "out", (q.shape[0], seqlen_k) if is_varlen else (bs, seqlen_q, seqlen_k))
    _require_fp32_output(denom_out, "denom_out", (q.shape[0],) if is_varlen else (bs, seqlen_q))
    if precision == "mxfp8" and ((head_dim == 512 and qhead_per_kv_head >= 128) or (head_dim == 128 and qhead_per_kv_head == 64)):
        with _torch_stream_context(current_stream):
            out.zero_()
            denom_out.zero_()

    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    if k_block_size is None:
        k_block_size = _select_dense_k_block_size(head_dim_padded, m_block_size, n_block_size, per_head_elem_bytes=4)
    kv_stage = _compute_dense_kv_stage(head_dim_padded, m_block_size, n_block_size, k_block_size, per_head_elem_bytes=4)
    if precision == "mxfp8":
        if head_dim == 512:
            if k_block_size == 64:
                kv_stage = 7
            elif k_block_size == 256:
                kv_stage = 3
            elif qhead_per_kv_head >= 128:
                kv_stage = 9
            else:
                kv_stage = 7
        else:
            kv_stage = 4

    compute_capability = _get_device_capability()
    if compute_capability < 10:
        raise NotImplementedError(f"Dense attention backward requires SM100+ (got {compute_capability}).")

    if precision == "mxfp8":
        assert q_scale is not None and k_scale is not None
        compile_key = (
            "dense_attention",
            "mxfp8",
            q.dtype,
            k.dtype,
            lse.dtype,
            q_scale.dtype,
            k_scale.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            head_dim_padded // k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
            sf_vec_size,
        )
    else:
        compile_key = (
            "dense_attention",
            q.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
        )

    if compile_key not in _dense_attn_score_recompute.compile_cache:
        q_cute = to_cute_tensor(q)
        k_cute = to_cute_tensor(k)
        lse_cute = to_cute_tensor(lse)
        out_cute = to_cute_tensor(out)
        denom_cute = to_cute_tensor(denom_out)
        cu_q_cute = to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        cu_q_scale_cute = to_cute_tensor(cu_seqlens_q_scale_padded, leading_dim=0) if is_varlen and precision == "mxfp8" else None
        cu_k_scale_cute = to_cute_tensor(cu_seqlens_k_scale_padded, leading_dim=0) if is_varlen and precision == "mxfp8" else None
        q_offsets_cute = to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

        if precision == "mxfp8":
            assert q_scale is not None and k_scale is not None
            q_scale_cute = to_cute_tensor(q_scale)
            k_scale_cute = to_cute_tensor(k_scale)
            kernel_obj = BwdDenseAttnScoreSm100Mxfp8(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                ratio=ratio,
                is_varlen=is_varlen,
                sf_vec_size=sf_vec_size,
            )
        else:
            q_scale_cute = None
            k_scale_cute = None
            kernel_obj = DenseScoreRecomputeSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                score_type="attention",
                ratio=ratio,
                is_varlen=is_varlen,
            )

        max_q_arg = cutlass.Int32(seqlen_q)
        max_k_arg = cutlass.Int32(seqlen_k)

        if precision == "mxfp8":
            _dense_attn_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                lse_cute,
                q_scale_cute,
                k_scale_cute,
                out_cute,
                denom_cute,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                cu_q_scale_cute,
                cu_k_scale_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )
        else:
            _dense_attn_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                lse_cute,
                out_cute,
                denom_cute,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )

    max_q_arg = cutlass.Int32(seqlen_q)
    max_k_arg = cutlass.Int32(seqlen_k)
    # Dense score kernels skip K blocks that are wholly outside the ratio-causal
    # region, so skipped masked/padding columns must be pre-filled. MXFP8
    # dense-attention uses mOut as an atomic-add accumulation buffer for valid
    # scores in some paths; those kernels patch invalid/skipped scores to -inf.
    with _torch_stream_context(current_stream):
        if precision == "mxfp8":
            out.zero_()
        else:
            out.fill_(float("-inf"))
    with torch.cuda.nvtx.range("dense_attn_score_recompute"):
        if precision == "mxfp8":
            _dense_attn_score_recompute.compile_cache[compile_key](
                q,
                k,
                lse,
                q_scale,
                k_scale,
                out,
                denom_out,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                cu_seqlens_q_scale_padded if is_varlen else None,
                cu_seqlens_k_scale_padded if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
        else:
            _dense_attn_score_recompute.compile_cache[compile_key](
                q,
                k,
                lse,
                out,
                denom_out,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
    return out, denom_out


_dense_attn_score_recompute.compile_cache = {}


# ---- Public dense entry points (auto-dispatch tile params) ------------------


def dense_indexer_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    out: torch.Tensor,
    denom_out: torch.Tensor,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    k_block_size: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense indexer backward score: sm_scale * ReLU(QK)*W head reduce over full KV.

    Tile params auto-selected via _dispatch_dense_indexer_tile_params. sm_scale
    is applied to the fp32 head-reduced score inside the kernel (preserves
    precision vs pre-multiplying onto bf16 weights on the host).
    The shared unified indexer kernel runs with ``compute_lse=True``;
    causal-invalid entries in ``out`` remain -inf and ``denom_out`` stores
    LogSumExp.

    Args:
        q: BSHD ``(bs, seqlen_q, n_heads_q, head_dim)`` or THD
           ``(total_q, n_heads_q, head_dim)`` [BF16, or the block-scaled
           precision's dtype; packed FP4 has last dim ``head_dim // 2``]
        k: BSHD ``(bs, seqlen_k, n_heads_kv, head_dim)`` or THD
           ``(total_k, n_heads_kv, head_dim)`` [same dtype as ``q``]
        weights: BSH ``(bs, seqlen_q, n_heads_q)`` or TH ``(total_q, n_heads_q)`` [BF16]
        out: required, contiguous. BSHD ``(bs, seqlen_q, seqlen_k)``;
             THD ``(total_q, max_seqlen_k)`` [FP32]
        denom_out: required, contiguous. BSHD ``(bs, seqlen_q)`` ; THD ``(total_q,)`` [FP32]
        sm_scale: scalar applied to fp32 score post head-reduce; default 1.0
        ratio, cu_seqlens_*, max_seqlen_*: see causal mask + THD docs.
        precision: ``"bf16"``, ``"mxfp8"`` (E4M3 Q/K) or ``"mxfp4"`` (packed
            E2M1 Q/K), the block-scaled two with one E8M0 scale per 32
            elements. Block-scaled precisions need packed ``q_scale`` /
            ``k_scale`` (and, for THD, ``cu_seqlens_*_scale_padded``).

    Returns:
        (out, denom_out)
    """
    if qhead_per_kv_head is None:
        qhead_per_kv_head = (q.shape[-2] // k.shape[-2]) if q.ndim == 3 else (q.shape[2] // k.shape[2])

    precision = _normalize_dense_indexer_precision(precision, q_scale, k_scale)
    fmt = BLOCK_SCALED_FORMATS.get(precision)
    # Check the Q/K dtype first: logical_head_dim trusts ``precision`` and would report a mismatch as a wrong head_dim.
    if fmt is not None and (q.dtype != fmt.data_dtype or k.dtype != fmt.data_dtype):
        raise TypeError(f"precision={precision!r} requires q and k to be {fmt.data_dtype}")
    head_dim = logical_head_dim(precision, q.shape[-1])
    m, n, kbs = _dispatch_dense_indexer_tile_params(
        head_dim,
        qhead_per_kv_head,
        precision=precision,
        k_block_size=k_block_size,
    )
    return _dense_indexer_score_recompute(
        q,
        k,
        weights,
        qhead_per_kv_head,
        out=out,
        denom_out=denom_out,
        m_block_size=m,
        n_block_size=n,
        k_block_size=kbs,
        sm_scale=sm_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        current_stream=current_stream,
    )


def dense_attn_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    out: torch.Tensor,
    denom_out: torch.Tensor,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    k_block_size: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense attention backward score: exp(QK*scale - LSE) head reduce over full KV.

    Tile params auto-selected via _dispatch_dense_attn_tile_params.

    Args:
        q: BSHD ``(bs, seqlen_q, n_heads_q, head_dim)`` or THD
           ``(total_q, n_heads_q, head_dim)`` [BF16]
        k: BSHD ``(bs, seqlen_k, n_heads_kv, head_dim)`` or THD
           ``(total_k, n_heads_kv, head_dim)`` [BF16]
        lse: BSH ``(bs, seqlen_q, n_heads_q)`` or TH ``(total_q, n_heads_q)`` [FP32]
        softmax_scale: float
        out: required, contiguous. BSHD ``(bs, seqlen_q, seqlen_k)``;
             THD ``(total_q, max_seqlen_k)`` [FP32]
        denom_out: required, contiguous. BSHD ``(bs, seqlen_q)`` ; THD ``(total_q,)`` [FP32]
        ratio, cu_seqlens_*, max_seqlen_*: see causal mask + THD docs.

    Returns:
        (out, denom_out)
    """
    if qhead_per_kv_head is None:
        qhead_per_kv_head = (q.shape[-2] // k.shape[-2]) if q.ndim == 3 else (q.shape[2] // k.shape[2])
    head_dim = q.shape[-1]

    precision = _normalize_dense_precision(precision, q_scale, k_scale)
    seqlen_k_hint = int(max_seqlen_k) if max_seqlen_k is not None else (k.shape[0] if q.ndim == 3 else k.shape[1])
    m, n, kbs = _dispatch_dense_attn_tile_params(
        head_dim,
        qhead_per_kv_head,
        precision=precision,
        seqlen_k_hint=seqlen_k_hint,
        k_block_size=k_block_size,
    )

    return _dense_attn_score_recompute(
        q,
        k,
        lse,
        softmax_scale,
        qhead_per_kv_head,
        out=out,
        denom_out=denom_out,
        m_block_size=m,
        n_block_size=n,
        k_block_size=kbs,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        current_stream=current_stream,
    )
