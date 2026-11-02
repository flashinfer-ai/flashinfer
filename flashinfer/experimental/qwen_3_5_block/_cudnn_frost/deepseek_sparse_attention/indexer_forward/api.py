# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper and dispatcher for indexer forward CuTe DSL score kernels.

``indexer_forward_wrapper`` produces dense indexer scores Q @ K^T with
per-head ReLU, weighted head reduction, and a ratio causal mask.
``indexer_forward_top_k_wrapper`` is the SM100 combined path that generates
compact scores, selects Top-K, and optionally computes Top-K softmax without
materializing the dense score tensor.
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import APIBase, TupleDict

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import device_major

from ._interface import TMA_ALIGN_ELEMS, _indexer_fwd_bound, indexer_fwd as indexer_fwd_sm100
from ._interface_sm90 import indexer_fwd as indexer_fwd_sm90
from ._support import SUPPORTED_QHEAD_PER_KV_HEAD_BF16


class IndexerForward(APIBase):
    """SM100+ BF16 Q/K APIBase shell with BF16 or FP32 per-head weights.

    The backend interface owns lazy compilation and its kernel cache. Hopper
    dispatch uses the direct SM90 wrapper in ``_interface_sm90.py``.

    ``sample_out`` / the ``out`` passed to ``execute()`` is the contiguous
    padded ``(B, S_q, S_k_padded)`` allocation (``S_k_padded % 4 == 0``);
    ``execute()`` binds ``out[..., :S_k]`` to the kernel directly, so the row
    stride stays ``S_k_padded`` and columns ``[S_k, S_k_padded)`` are never
    written. Q/K/W must have a unit innermost stride; other layouts are
    declined in ``check_support()`` (Rule 2 / R5) rather than copied.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k: torch.Tensor,  # (B, S_k, H_kv, D) BF16
        sample_w: torch.Tensor,  # (B, S_q, H_q) BF16 or FP32
        sample_out: torch.Tensor,  # (B, S_q, S_k_padded) FP32, contiguous
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 4,
        sm_scale: float = 1.0,
    ):
        super().__init__()

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.o_desc = self._make_tensor_desc(sample_out, name="sample_out")

        self.ratio = int(ratio)
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.q_stage = int(q_stage)
        self.kv_stage = int(kv_stage)
        self.sm_scale = float(sm_scale)
        self.qhead_per_kv_head = qhead_per_kv_head

        self.batch_size = None
        self.s_q = None
        self.s_k = None
        self.s_k_padded = None
        self.h_q = None
        self.h_kv = None
        self.head_dim = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")
        self._value_error_if(
            self.q_desc.ndim != 4,
            f"Q must be 4-D (B, S_q, H_q, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.k_desc.ndim != 4,
            f"K must be 4-D (B, S_k, H_kv, D), got {self.k_desc.shape}",
        )
        self._value_error_if(
            self.w_desc.ndim != 3,
            f"W must be 3-D (B, S_q, H_q), got {self.w_desc.shape}",
        )
        self._value_error_if(
            self.o_desc.ndim != 3,
            f"Out must be 3-D (B, S_q, S_k_padded), got {self.o_desc.shape}",
        )

        b, s_q, h_q, d = self.q_desc.shape
        b_k, s_k, h_kv, d_k = self.k_desc.shape
        b_o, s_q_out, s_k_padded_from_out = self.o_desc.shape
        self._value_error_if(b != b_k, f"Batch size mismatch Q={b} vs K={b_k}")
        self._value_error_if(b != b_o, f"Batch size mismatch Q={b} vs Out={b_o}")
        self._value_error_if(s_q != s_q_out, f"S_q mismatch Q={s_q} vs Out={s_q_out}")
        self._value_error_if(d != d_k, f"Head dim mismatch Q={d} vs K={d_k}")
        self._value_error_if(
            d != 128,
            f"IndexerForward is tuned for head_dim=128 only, got {d}",
        )

        qhpkv = self.qhead_per_kv_head if self.qhead_per_kv_head is not None else (h_q // h_kv)
        self._value_error_if(
            qhpkv * h_kv != h_q,
            f"qhead_per_kv_head * h_kv != h_q ({qhpkv} * {h_kv} != {h_q})",
        )
        self._value_error_if(
            qhpkv not in SUPPORTED_QHEAD_PER_KV_HEAD_BF16,
            f"qhead_per_kv_head must be one of {SUPPORTED_QHEAD_PER_KV_HEAD_BF16}, got {qhpkv}",
        )
        self.qhead_per_kv_head = qhpkv

        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.w_desc, [torch.bfloat16, torch.float32], name="W")
        if self.w_desc.dtype == torch.float32 and self.w_desc.stride[-1] != 1:
            raise NotImplementedError(f"FP32 W requires unit last stride, got strides {self.w_desc.stride}")
        self._check_dtype(self.o_desc, torch.float32, name="Out")

        for desc, name in ((self.q_desc, "Q"), (self.k_desc, "K"), (self.w_desc, "W")):
            self._not_implemented_error_if(
                desc.stride[-1] != 1,
                f"IndexerForward reads {name} with a unit innermost stride natively; got strides {desc.stride}",
            )
        # The padded allocation; execute() binds its [..., :S_k] view directly.
        self._not_implemented_error_if(
            not self.o_desc.is_contiguous(),
            f"IndexerForward writes Out through a contiguous (B, S_q, S_k_padded) buffer; got strides {self.o_desc.stride}",
        )
        self._value_error_if(
            s_k_padded_from_out % TMA_ALIGN_ELEMS != 0,
            f"Out seqlen_k dim must be a multiple of {TMA_ALIGN_ELEMS}, got {s_k_padded_from_out}",
        )
        self._value_error_if(
            s_k_padded_from_out < s_k,
            f"Out seqlen_k dim ({s_k_padded_from_out}) must cover logical K length ({s_k})",
        )

        major = device_major()
        self._runtime_error_if(
            major < 10,
            f"IndexerForward requires SM100+ compute capability, found SM{major}",
        )

        self.batch_size = b
        self.s_q = s_q
        self.s_k = s_k
        self.s_k_padded = s_k_padded_from_out
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        # The direct interface owns its compile cache and needs real tensors
        # to preserve runtime layouts. Mark this APIBase object ready and let
        # the interface compile lazily on first execute().
        self._compiled_kernel = True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("IndexerForward kernel not compiled")
        self._check_dtype(w, self.w_desc.dtype, name="W")

        # `out` is the padded (B, S_q, S_k_padded) allocation the descriptor
        # promised; the kernel binds the [..., :S_k] view with that row stride
        # (no staging, no copy-back). _indexer_fwd_bound re-checks the live strides.
        padded_shape = (self.batch_size, self.s_q, self.s_k_padded)
        if tuple(out.shape) != padded_shape or not out.is_contiguous():
            raise ValueError(
                f"out must be the contiguous {padded_shape} fp32 allocation declared to IndexerForward; got shape {tuple(out.shape)}, strides {tuple(out.stride())}"
            )
        logical_out = out[..., : self.s_k]
        _indexer_fwd_bound(
            q,
            k,
            w,
            ratio=self.ratio,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=logical_out,
            m_block_size=self.m_block_size,
            n_block_size=self.n_block_size,
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
            sm_scale=self.sm_scale,
            precision="bf16",
            current_stream=current_stream,
        )


def indexer_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> TupleDict:
    """High-level wrapper. Allocates the output buffer with TMA padding on S_k.

    With ``precision='bf16'``, Q/K are BF16 and per-head W may be BF16 or FP32.
    The block-scaled SM100 precisions take BF16 W and blockscaled-packed E8M0
    ``q_scale`` / ``k_scale``, one scale per 32 head_dim values (``head_dim // 32``
    per row): ``'mxfp8'`` (E4M3 Q/K) or ``'mxfp4'`` (``float4_e2m1fn_x2`` Q/K with
    last dim ``head_dim // 2``, still ``head_dim // 32`` scales per row).
    SM90 offers ``'fp8'`` with per-token scales instead.
    Returns ``{'scores': (B, S_q, S_k) FP32}``. The ratio causal mask marks
    positions outside the valid KV range with -inf. ``q_causal_offsets`` may
    specify the global uncompressed token index for each batch/THD segment's
    local q[0].

    ``block_table`` ((B, max_pages) int32) with ``kv_lens`` ((B,) int32) reads
    ``k`` as a page pool ``(num_pages, page_size, 1, D)`` or
    ``(num_pages, page_size, D)`` (SM100, BF16, ``page_size`` a multiple of 8
    that divides 128 or is a multiple of it): sequence ``b`` owns pages
    ``block_table[b, :ceil(kv_lens[b] / page_size)]`` and scores its first
    ``kv_lens[b]`` keys; ``cu_seqlens_k`` must be ``None`` and the score width
    is ``max_seqlen_k`` (default ``max_pages * page_size``).

    On SM100 the kernel writes a ``(B, S_q, ceil4(S_k))`` (THD:
    ``(total_q, ceil4(max_seqlen_k))``) padded buffer through its ``[..., :S_k]``
    view. The wrapper keeps the torch-op contract on top of that: a returned
    ``scores`` is contiguous (one copy when ``S_k % 4 != 0``), and a
    caller-provided ``out`` of the logical shape is filled whatever its layout
    (a padded-stride view is bound directly; anything else is staged and copied
    back, so the caller's tensor keeps its identity). Strided ``q``/``k``/``w``/
    scales are made contiguous here on the launch stream as a torch-op
    convenience (one copy kernel each). ``IndexerForward`` itself declines all
    of that in ``check_support()`` and binds only a padded-stride ``out``.
    """
    if device_major() == 9:
        if cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
            raise NotImplementedError("Block-scale padded cu_seqlens are only supported on SM100+")
        if block_table is not None or kv_lens is not None:
            raise NotImplementedError("paged K is only supported on SM100+")
        unsupported = []
        if m_block_size != 128:
            unsupported.append(f"m_block_size={m_block_size}")
        if n_block_size != 128:
            unsupported.append(f"n_block_size={n_block_size}")
        if q_stage != 2:
            unsupported.append(f"q_stage={q_stage}")
        if kv_stage != 4:
            unsupported.append(f"kv_stage={kv_stage}")
        if unsupported:
            raise ValueError(
                "SM90 indexer_forward_wrapper only supports default tuning parameters "
                "(m_block_size=128, n_block_size=128, q_stage=2, kv_stage=4); got " + ", ".join(unsupported)
            )
        # Both arches route through their own indexer_fwd wrapper (which owns
        # output allocation + TMA padding); Hopper uses the SM90 variant.
        result = indexer_fwd_sm90(
            q,
            k,
            w,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            return_lse=return_lse,
            lse_out=lse_out,
            current_stream=stream,
        )
        if return_lse:
            scores, lse = result
            return TupleDict(scores=scores, lse=lse)
        return TupleDict(scores=result)

    if return_lse or lse_out is not None:
        raise NotImplementedError("SM100 dense indexer forward does not expose LSE; " "LSE is produced by dense_indexer_score_recompute_wrapper")

    # indexer_fwd is the dense convenience entry: it normalises strided inputs,
    # allocates / stages `out` and returns the torch-op layout, all on `stream`.
    # BSHD and THD both go through it (it branches on cu_seqlens internally).
    scores = indexer_fwd_sm100(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        out=out,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=384,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        block_table=block_table,
        kv_lens=kv_lens,
        current_stream=stream,
    )
    return TupleDict(scores=scores)


def indexer_forward_top_k_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    top_k: int,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    stream: Optional[cuda.CUstream] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    return_softmax: Optional[bool] = None,
    softmax_out: Optional[torch.Tensor] = None,
    microbatch_rows: int = -1,
    topk_indices_global: bool = True,
    cand_buffer: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    block_table: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Combined SM100 indexer score generation and Top-K selection API.

    With ``precision='bf16'``, Q/K are BF16 and per-head W may be BF16 or FP32;
    ``'mxfp8'`` and ``'mxfp4'`` take block-scaled Q/K and BF16 W as in
    :func:`indexer_forward_wrapper`.
    Returns ``{'indices', 'logits', 'softmax'}`` by default, plus ``'lse'``
    when requested. Set ``return_softmax=False`` to return only indices and
    logits. Unlike :func:`indexer_forward_wrapper`, this path never
    materializes the dense score tensor.

    BSHD outputs have shape ``(B, S_q, top_k)`` and THD outputs have shape
    ``(total_q, top_k)``. Caller-owned candidate and output buffers may be
    supplied to avoid per-call allocation. Indices are global KV ids by
    default; set ``topk_indices_global=False`` for local ids matching
    ``indexer_top_k_wrapper``, and use the same convention downstream. Local
    ids also avoid conversion temporaries when a fully preallocated CUDA-graph
    path must have zero extra allocations; the default global conversion is
    capture-safe but uses the graph's private pool. Leave ``microbatch_rows``
    at ``-1`` for the compatible BSHD long-sequence auto policy, or pass ``0``
    to force a single launch.
    Set ``deterministic=True`` to resolve exact-value ties at the K-th boundary
    toward the smallest local KV indices. This makes the selected set
    reproducible, although output slot order remains unspecified.

    ``block_table`` ((B, max_pages) int32) with ``kv_lens`` ((B,) int32) reads
    ``k`` as a page pool ``(num_pages, page_size, 1, D)`` or
    ``(num_pages, page_size, D)`` (BF16, ``page_size`` a multiple of 8 that
    divides 128 or is a multiple of it): sequence ``b`` owns pages
    ``block_table[b, :ceil(kv_lens[b] / page_size)]`` and scores its first
    ``kv_lens[b]`` keys. ``cu_seqlens_k`` must be ``None``, and the ids are
    local, so pass ``topk_indices_global=False``.
    """
    if device_major() < 10:
        raise NotImplementedError("compressed indexer forward is SM100-only; standalone IndexerTopK remains SM90+")

    result = indexer_fwd_sm100(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=384,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        is_compressed_logits=True,
        topk=top_k,
        microbatch_rows=microbatch_rows,
        topk_indices_global=topk_indices_global,
        cand_buffer=cand_buffer,
        out_indices=out_indices,
        out_logits=out_logits,
        cand_batch_offsets=cand_batch_offsets,
        return_lse=return_lse,
        lse_out=lse_out,
        return_softmax=return_softmax,
        softmax_out=softmax_out,
        deterministic=deterministic,
        block_table=block_table,
        kv_lens=kv_lens,
        current_stream=stream,
    )

    want_softmax = (return_softmax is not False) or softmax_out is not None
    want_lse = return_lse or lse_out is not None
    result_iter = iter(result)
    values = {
        "indices": next(result_iter),
        "logits": next(result_iter),
    }
    if want_softmax:
        values["softmax"] = next(result_iter)
    if want_lse:
        values["lse"] = next(result_iter)
    return TupleDict(**values)
