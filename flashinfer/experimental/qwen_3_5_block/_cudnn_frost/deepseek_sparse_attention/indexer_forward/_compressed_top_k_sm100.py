# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compressed-logits + stage-2 Top-K orchestration for SM100.

The shared score kernels remain owned by score_recompute. This module only
coordinates compact candidate storage, stage-2 selection, optional LSE, and
BSHD/THD buffer management for indexer_forward.
"""

from __future__ import annotations

from typing import Optional

import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_top_k.local_to_global_dsl import _local_to_global_inplace

import cutlass
import cutlass.cute as cute

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.score_recompute.indexer_score_unified_sm100 import (
    IndexerScoreUnifiedSm100,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.score_recompute._block_scaled import (
    BLOCK_SCALED_FORMATS,
    BLOCK_SCALED_PRECISION_NAMES,
    bshd_packed_scale_shape,
    kernel_qk_view,
    logical_head_dim,
    validate_block_scaled_inputs,
    validate_indexer_precision,
    validate_sf_vec_size,
    validate_thd_scale_contract,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.compiler import compile_options
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import (
    ceil_div as _ceil_div,
    maybe_contiguous as _maybe_contiguous,
    resolve_stream,
    validate_q_causal_offsets,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.tensor_conversion import (
    to_cute_tensor as _to_cute_tensor,
)

from ._support import (
    check_paged_args as _check_paged_args,
    check_q_covered_by_k as _check_q_covered_by_k,
    resolve_m_block_size as _resolve_m_block_size,
    resolve_paged_k as _resolve_paged_k,
    validate_qhead_per_kv_head as _validate_indexer_qhead_per_kv_head,
)

_compile_cache: dict = {}

# Stage-2 fuses softmax over the selected logits. It is returned by default
# because the indexer backward KL-loss path consumes this probability tensor.
_RETURN_SOFTMAX_DEFAULT = True


def _make_i64_cand_buffer_compile_tensor():
    """Create the 1D candidate-buffer ABI with a 64-bit dynamic extent."""
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.Float32,
        shape=(cute.sym_int64(symbol="cand_buffer_numel"),),
        stride=(1,),
        assumed_align=16,
    )


def _compress_local_to_global_bshd_(idx: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    """In-place ``idx += b * seqlen_k`` on valid (>= 0) slots; -1 padding kept, so
    a caller-provided output buffer is converted in place (no realloc).  -1 padding
    is preserved; global ids fit int32 by design (int64 intermediate keeps
    ``b * seqlen_k`` exact)."""
    return _local_to_global_inplace(idx, seqlen_k)


def _select_microbatch_rows(seqlen_q: int, bs: int, ratio: int) -> int:
    """Long-sequence auto-microbatch window size.  Returns 0 ⇒ no microbatching.

    Only triggers for very long sequences (seqlen_q >= 32768); the window is
    rounded down to a multiple of ``ratio`` so each window's KV bound
    ``(q_global_start + rs + mb) / ratio`` stays integral.  Requires
    ``seqlen_q % ratio == 0`` (otherwise the windowed sub-problem geometry is
    not exact, so we fall back to a single launch).
    """
    if seqlen_q < 32768 or seqlen_q % ratio != 0:
        return 0
    if bs >= 8:
        mb = 2048
    elif bs >= 4:
        mb = 4096
    else:
        mb = 8192
    mb = (mb // ratio) * ratio
    if mb <= 0 or mb >= seqlen_q:
        return 0
    return mb


def _select_topk_block_threads(rows: int, seqlen_k: int) -> int:
    """Auto-pick the stage-2 radix block size (one block per (row, batch)).

    The radix is latency/barrier-bound, so its speed is governed by occupancy
    (concurrent blocks/SM), not by per-row parallelism:
      * prefill (many rows): the grid is far larger than the GPU, so SMALLER
        blocks win — 256 threads → 8 blocks/SM (vs 512 → 4) hides the per-block
        barrier latency across more concurrent rows.
      * decode (few rows, seqlen_q==1): the grid is tiny so occupancy is not the
        limit; each block wants MORE threads to chew through its long segment, so
        512, or 1024 for very long KV.
    block_threads must divide both 2048 and 1024 (256 / 512 / 1024 all do).
    Measured on B200/SM100: prefill stage-2 is ~7-17% faster at 256 than 512.
    """
    if rows >= 512:
        return 256
    return 1024 if seqlen_k >= 8192 else 512


def _use_cand_2d(precision: str, pbf: int) -> bool:
    """Whether the compress GEMM stores through a 2D (bs, pbf) cand view.

    The 2D store writes ``mOut[batch_idx, row_offset(q)+pos]`` with an Int32 column
    index, instead of the 1D flat Int64 offset — lower epilogue register pressure,
    the SAME compact layout (the result is bit-identical; only the addressing
    differs).  The gate is per-batch ``pbf < 2^31``, NOT the whole ``bs*pbf``
    buffer: the column index ``row_offset(q)+pos < pbf`` must fit Int32, while the
    batch base ``batch_idx*pbf`` is computed in 64-bit by CuTe's crd2idx, so
    ``bs*pbf`` may exceed 2^31 with no overflow (verified at ``bs*pbf=2.2e9`` by the
    opt-in ``test_indexer_fwd_dsl_compress_cand_2d_high_offset_64bit``).  bf16 BSHD only:
    the block-scaled (MXFP8/MXFP4) epilogue and varlen use the 1D store.
    """
    return precision == "bf16" and pbf < (1 << 31)


def compress_topk_cand_buffer_size(
    bs: int,
    seqlen_q: int,
    seqlen_k: int,
    ratio: int,
    microbatch_rows: int = -1,
    return_lse: bool = False,
    q_causal_offsets: torch.Tensor | None = None,
    precision: str = "bf16",
) -> int:
    """Number of float32 elements a caller-provided ``cand_buffer`` must hold for
    ``indexer_fwd_compress_topk`` at this BSHD shape.  It is ``bs * per_batch_floats``; for a
    microbatched launch it is the largest window's compact buffer.  The default
    ``microbatch_rows=-1`` MATCHES the kernel's default (long-seq auto policy), so
    sizing with the default and launching with the default agree — pass the same
    explicit value if you override it on the kernel.

    Pass ``return_lse=True`` when the launch will request LSE (``return_lse`` /
    ``lse_out``): an LSE request forces the SINGLE-LAUNCH path (microbatch is disabled
    under LSE), so this returns the FULL ``bs * per_batch_floats`` buffer.  Sizing a
    long sequence with the auto default (-1) but launching with LSE would otherwise
    UNDER-size the buffer and the launch would raise "cand_buffer too small".  To
    match the launch contract, ``return_lse=True`` with an explicit
    ``microbatch_rows > 0`` raises here too (microbatch + LSE is unsupported).

    Pass the launch's ``precision``: block-scaled precisions (``"mxfp8"``, ``"mxfp4"``)
    always run a single launch, so this returns the full buffer and an explicit
    ``microbatch_rows > 0`` raises, as at launch."""
    from ..indexer_top_k.compress_top_k_sm100 import per_batch_floats

    _check_q_covered_by_k(seqlen_q, seqlen_k, ratio)
    precision = validate_indexer_precision(precision)
    if precision in BLOCK_SCALED_FORMATS:
        if microbatch_rows > 0:
            raise NotImplementedError(f"microbatch_rows is not supported with precision={precision!r} " "(per-window scale repacking is not implemented)")
        microbatch_rows = 0
    if q_causal_offsets is not None:
        if not q_causal_offsets.is_cuda or q_causal_offsets.dtype != torch.int32 or q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != bs:
            raise ValueError(f"q_causal_offsets must be a 1D CUDA int32 tensor of shape ({bs},)")
        # Explicit per-batch offsets ⇒ per_batch_floats is non-uniform; the buffer is the
        # per-batch sum (microbatch is disabled under offsets; LSE is per-row, unaffected).
        return int(_bshd_cand_batch_offsets(int(bs), int(seqlen_q), int(ratio), q_causal_offsets)[bs].item())
    if return_lse:
        # Mirror the launch contract (indexer_fwd_compress_topk): an explicit
        # microbatch_rows > 0 with LSE is rejected; auto (-1) / 0 force single-launch.
        if microbatch_rows > 0:
            raise NotImplementedError("microbatch_rows > 0 is not supported with LSE output (return_lse / lse_out); " "pass microbatch_rows=0")
        # LSE → single-launch path, so the buffer is the full bs * per_batch_floats.
        return bs * per_batch_floats(seqlen_q, seqlen_k, ratio)
    if microbatch_rows < 0:
        microbatch_rows = _select_microbatch_rows(seqlen_q, bs, ratio)
    if 0 < microbatch_rows < seqlen_q:
        # Offset-0 windowing: window [rs, rs+mb) runs with per-window causal offset rs
        # (see indexer_fwd_compress_topk); size the shared buffer for the largest window.
        max_pbf, rs = 0, 0
        while rs < seqlen_q:
            mb = min(microbatch_rows, seqlen_q - rs)
            kw = min((rs + mb) // ratio, seqlen_k)
            max_pbf = max(max_pbf, per_batch_floats(mb, kw, ratio, rs))
            rs += mb
        return bs * max_pbf
    return bs * per_batch_floats(seqlen_q, seqlen_k, ratio)


def _resolve_cand_buffer(
    cand_buffer: Optional[torch.Tensor],
    required_floats: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate & slice a caller-provided ``cand_buffer`` to ``required_floats``
    float32 elements, or allocate one.  A caller-provided buffer lets callers reuse
    a single allocation across calls."""
    if cand_buffer is None:
        return torch.empty(required_floats, dtype=torch.float32, device=device)
    if not cand_buffer.is_cuda or cand_buffer.dtype != torch.float32:
        raise ValueError("cand_buffer must be a CUDA float32 tensor")
    if cand_buffer.device != device:
        raise ValueError(f"cand_buffer device {cand_buffer.device} != input device {device}")
    if cand_buffer.numel() < required_floats:
        raise ValueError(
            f"cand_buffer too small: need >= {required_floats} float32 elements "
            f"(size it with compress_topk_cand_buffer_size(...), passing this launch's precision), got {cand_buffer.numel()}"
        )
    if not cand_buffer.is_contiguous():
        # A caller-provided buffer is written in place; reshape()-ing a
        # non-contiguous tensor would silently copy (a fresh allocation the
        # kernel writes into instead of the caller's buffer), defeating the
        # "caller-provided" contract.  Require contiguous and view (no copy).
        raise ValueError("cand_buffer must be contiguous (it is written in place)")
    return cand_buffer.view(-1)[:required_floats]


def _bshd_cand_batch_offsets(bs: int, seqlen_q: int, ratio: int, q_causal_offsets: torch.Tensor) -> torch.Tensor:
    """BSHD per-batch compact slab offsets (int64 ``(bs+1,)``) under per-batch
    ``q_causal_offsets[b]`` (uniform ``seqlen_q``): pbf_b = G(offset_b + sq) - G(offset_b).
    The BSHD analog of ``_compress_cand_batch_offsets`` (which derives sq from cu_seqlens);
    used when explicit offsets make per_batch_floats non-uniform across the batch."""
    qgs = q_causal_offsets.to(torch.int64)
    sq = torch.full_like(qgs, int(seqlen_q))

    def _G(n):
        Q = n // ratio
        S = n - Q * ratio
        return ratio * Q * (Q - 1) // 2 + Q * (S + 1)

    pbf = _G(qgs + sq) - _G(qgs)
    offsets = torch.zeros(bs + 1, dtype=torch.int64, device=q_causal_offsets.device)
    torch.cumsum(pbf, dim=0, out=offsets[1:])
    return offsets


def indexer_fwd_compress_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    sm_scale: float = 1.0,
    microbatch_rows: int = -1,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    topk_block_threads: int = -1,
    topk_indices_global: bool = False,
    cand_buffer: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    return_softmax: Optional[bool] = None,
    softmax_out: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    _cand_total_floats: Optional[int] = None,
    _cand_batch_offsets: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Compress-logits + top-k (CuTe DSL, BSHD).

    ``precision`` is ``"bf16"`` (default), ``"mxfp8"``, or ``"mxfp4"``.  The
    block-scaled precisions take blockscaled-packed E8M0 ``q_scale``/``k_scale``
    (same packing as ``indexer_fwd``).  Microbatching is bf16-only.

    ``block_table`` ((bs, max_pages) int32) + ``kv_lens`` ((bs,) int32) read K
    from a page pool ``(num_pages, page_size, 1, head_dim)`` instead of a dense
    ``(bs, seqlen_k, 1, head_dim)`` tensor (BF16, single launch, local ids);
    ``max_seqlen_k`` is then the key extent (default ``max_pages * page_size``).

    Runs the logits GEMM in *compress* mode — the epilogue writes only the valid
    ratio-causal scores into a compact per-row variable-length cand_buffer
    (about half the dense (bs, sq, sk) footprint; no full score tensor is
    materialised) — then the CuTeDSL stage-2 radix top-k over that buffer.

    Args:
        q: BSHD ``(bs, seqlen_q, n_heads_q, head_dim)`` [BF16; E4M3 with
            precision="mxfp8"; packed ``float4_e2m1fn_x2`` with last dim
            ``head_dim // 2`` with precision="mxfp4"]
        k: BSHD ``(bs, seqlen_k, n_heads_kv, head_dim)`` [same dtype rules as q]
        w: BSH  ``(bs, seqlen_q, n_heads_q)`` BF16 or FP32 for BF16 Q/K;
            BF16 for block-scaled Q/K.
        topk: top-k width K
        ratio: compression ratio
        sm_scale: scalar applied to the fp32 head-reduced score (same as
            ``indexer_fwd``); the returned logits are post-scale.
        microbatch_rows: when > 0 and < seqlen_q, process the query rows in
            windows of this many rows, reusing one small cand_buffer per window
            instead of allocating the full ``bs * sq^2/(2*ratio)`` buffer — the
            *microbatch* memory optimisation.  Each window ``[rs, rs+mb)`` is an
            independent ratio-causal sub-problem (``seqlen_q=mb``,
            ``seqlen_k=(q_global_start+rs+mb)/ratio``), so it returns the **same
            top-k set** as the single-launch path (only the order of equal slots
            within the K outputs may differ, as for any two radix launches).
            Requires
            ``microbatch_rows % ratio == 0`` and ``seqlen_q % ratio == 0`` so the
            per-window KV bound stays integral. Default ``-1`` enables the
            long-sequence auto policy; pass ``0`` to force a single launch.
        cand_buffer: optional caller-provided float32 CUDA scratch buffer for the
            stage-1 compact logits.  Size it with ``compress_topk_cand_buffer_size(..., precision=precision)``;
            lets callers reuse one allocation across calls.  ``None`` = allocate
            internally.
        out_indices / out_logits: optional caller-provided ``(bs, seqlen_q, topk)``
            int32 / fp32 CUDA output tensors, written in place (allocation-free /
            CUDA-graph friendly; ``None`` = allocate).  With microbatch, each window
            writes directly into its slice of these (no per-window temp + copy).
            When ``topk_indices_global`` the local→global add is applied in place to
            ``out_indices``.
        topk_indices_global: when True, return public global KV ids
            (``b * seqlen_k + local``) to match indexer_topk / indexer_bwd; when
            False (default for this low-level entry), return raw local KV ids.
        deterministic: when true, exact-value ties at the K-th boundary select
            the smallest local KV indices, making the selected set reproducible.
            The within-row slot order remains unspecified. Default false keeps
            the faster scheduling-dependent tie-break.

    Returns:
        (topk_indices, topk_logits):
          * topk_indices ``(bs, seqlen_q, topk)`` INT32 — **local** KV ids in
            ``[0, seqlen_k)``; padded slots (row has < K candidates) are -1.
          * topk_logits  ``(bs, seqlen_q, topk)`` FP32 — the selected scores
            (post sm_scale); padded slots are -inf.

    Softmax over the selected logits is fused in stage-2 and returned by default as
    ``topk_softmax`` FP32 (padding slots are zero). Pass ``return_softmax=False``
    to omit it. With LSE enabled, the tuple order is
    ``(indices, logits, softmax, lse)``.

    When ``return_lse=True`` (or a caller-provided ``lse_out`` is given), the
    stage-1 epilogue also returns the per-row LSE (all precisions). An explicit
    ``microbatch_rows > 0`` remains unsupported with LSE.
    """
    from ..indexer_top_k.compress_top_k_sm100 import (
        compress_stage2_topk,
        per_batch_floats,
    )

    if return_softmax is None:
        return_softmax = _RETURN_SOFTMAX_DEFAULT
    precision = validate_indexer_precision(precision)
    is_block_scaled = precision in BLOCK_SCALED_FORMATS
    if precision == "bf16" and w.dtype == torch.float32 and w.stride(-1) != 1:
        raise NotImplementedError(f"FP32 w requires unit last stride, got strides {w.stride()}")
    q, k, w = [_maybe_contiguous(t) for t in (q, k, w)]
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 3:
        raise ValueError("compress-topk expects BSHD q (bs,sq,Hq,D), k (bs,sk,Hkv,D), w (bs,sq,Hq)")
    if precision == "bf16" and (q_scale is not None or k_scale is not None):
        # Stray scales would be silently ignored and would pollute the compile cache key.
        raise ValueError(f"q_scale and k_scale are only valid with precision={BLOCK_SCALED_PRECISION_NAMES}")
    if precision == "bf16" and not (q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and w.dtype in (torch.bfloat16, torch.float32)):
        raise TypeError("precision='bf16' requires bfloat16 q/k and bfloat16 or float32 w")
    if is_block_scaled:
        # A dtype/device mismatch would otherwise fail cryptically at compile or run time.
        validate_block_scaled_inputs(precision, q, k, w, q_scale, k_scale)
    validate_sf_vec_size(precision, sf_vec_size)

    bs, seqlen_q, n_heads_q, head_dim = q.shape
    paged = _check_paged_args(block_table, kv_lens)
    if paged:
        if precision != "bf16":
            raise NotImplementedError("paged K is served by the BF16 indexer only")
        page_size, max_pages = _resolve_paged_k(k, block_table, kv_lens, None, n_block_size=n_block_size, batch=int(bs), device=q.device)
        seqlen_k = max_pages * page_size if max_seqlen_k is None else int(max_seqlen_k)
        n_heads_kv = int(k.shape[2])
    else:
        page_size = 0
        _, seqlen_k, n_heads_kv, _ = k.shape
    # Auto stage-2 block size (topk_block_threads<=0): resolved ONCE from the full
    # shape and applied to every microbatch window, NOT re-derived per window.
    # block_threads is occupancy-only (correctness-neutral — it just must divide
    # 2048 & 1024), so the full-shape pick is already optimal per window and a
    # per-window pick would only add compile-cache entries.  Pass an explicit
    # topk_block_threads to override.
    if topk_block_threads <= 0:
        topk_block_threads = _select_topk_block_threads(bs * seqlen_q, seqlen_k)
    # Cross-tensor shape / device consistency (the compressed-logits path is public; reject
    # malformed inputs here instead of as a kernel OOB / cryptic failure).
    if (not paged and k.shape[0] != bs) or k.shape[3] != head_dim:
        raise ValueError(f"k must be (bs={bs}, seqlen_k, n_heads_kv, last dim={head_dim}), " f"got {tuple(k.shape)}")
    head_dim = logical_head_dim(precision, head_dim)
    if tuple(w.shape) != (bs, seqlen_q, n_heads_q):
        raise ValueError(f"w shape must be (bs={bs}, seqlen_q={seqlen_q}, n_heads_q={n_heads_q}), " f"got {tuple(w.shape)}")
    if k.device != q.device or w.device != q.device:
        raise ValueError("q, k, w must be on the same device")
    # qhead_per_kv_head (GQA group size) contract: a
    # wrong head mapping corrupts head-reduce, tile mapping and the block-scale
    # layout, so n_heads_q must be divisible by n_heads_kv and an explicit value
    # must equal the derived ratio.
    if n_heads_q % n_heads_kv != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})")
    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    elif qhead_per_kv_head != n_heads_q // n_heads_kv:
        raise ValueError(f"qhead_per_kv_head ({qhead_per_kv_head}) must equal n_heads_q // " f"n_heads_kv ({n_heads_q // n_heads_kv})")
    _validate_indexer_qhead_per_kv_head(qhead_per_kv_head, precision)
    if n_heads_kv != 1:
        raise ValueError("compress-logits top-k currently requires n_heads_kv=1 (MQA); " f"got n_heads_kv={n_heads_kv}. The stage-1 GEMM reads only KV head 0.")
    if precision == "bf16":
        m_block_size = _resolve_m_block_size(m_block_size, qhead_per_kv_head, head_dim, compressed=True, path="SM100 compressed indexer forward")
    if m_block_size % qhead_per_kv_head != 0:
        raise ValueError(f"m_block_size ({m_block_size}) must be divisible by " f"qhead_per_kv_head ({qhead_per_kv_head})")
    _check_q_covered_by_k(seqlen_q, seqlen_k, ratio)
    device = q.device

    # Per-batch causal offsets (q_causal_offsets, (bs,) int32; default None = 0 = top-left,
    # matching the dense indexer).  Supported for every precision: each epilogue has a
    # cand_batch_offsets branch, so an explicit per-batch offset routes the BSHD path
    # through the tight per-batch cand_batch_offsets slab (like THD) and the flat 1D store
    # (the 2D compact store needs a uniform pbf; block-scaled is 1D-only regardless).  Microbatch
    # windowing assumes the default offset-0 and is rejected with explicit offsets below.
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), device)

    out_shape = (bs, seqlen_q, topk)
    if out_indices is not None and (
        tuple(out_indices.shape) != out_shape
        or out_indices.dtype != torch.int32
        or not out_indices.is_cuda
        or out_indices.device != device
        or out_indices.stride(-1) != 1
    ):
        raise ValueError(f"out_indices must be a CUDA int32 tensor of shape {out_shape} on {device} " f"with a contiguous last (topk) dim")
    if out_logits is not None and (
        tuple(out_logits.shape) != out_shape
        or out_logits.dtype != torch.float32
        or not out_logits.is_cuda
        or out_logits.device != device
        or out_logits.stride(-1) != 1
    ):
        raise ValueError(f"out_logits must be a CUDA float32 tensor of shape {out_shape} on {device} " f"with a contiguous last (topk) dim")

    want_softmax = return_softmax or softmax_out is not None
    if softmax_out is not None and (
        tuple(softmax_out.shape) != out_shape
        or softmax_out.dtype != torch.float32
        or not softmax_out.is_cuda
        or softmax_out.device != device
        or softmax_out.stride(-1) != 1
    ):
        raise ValueError(f"softmax_out must be a CUDA float32 tensor of shape {out_shape} on " f"{device} with a contiguous last (topk) dim")

    want_lse = return_lse or lse_out is not None
    if lse_out is not None and (
        tuple(lse_out.shape) != (bs, seqlen_q)
        or lse_out.dtype != torch.float32
        or not lse_out.is_cuda
        or lse_out.device != device
        or not lse_out.is_contiguous()
    ):
        raise ValueError(f"lse_out must be a contiguous CUDA float32 tensor of shape {(bs, seqlen_q)} " f"on {device}")
    # microbatch + LSE is unsupported (the windowing path does not plumb the per-row
    # LSE).  Reject an EXPLICIT microbatch request instead of silently downgrading to a
    # single launch (which could unexpectedly raise peak memory on long sequences);
    # the auto policy (microbatch_rows < 0) simply stays single-launch under LSE.
    if want_lse and microbatch_rows > 0:
        raise NotImplementedError("microbatch_rows > 0 is not supported with LSE output (return_lse / lse_out); " "pass microbatch_rows=0")
    # Explicit q_causal_offsets + microbatch is unsupported (the windowing recomputes a
    # bottom-right per-window geometry that does not compose with arbitrary offsets).
    if q_causal_offsets is not None and microbatch_rows > 0:
        raise NotImplementedError("microbatch_rows > 0 is not supported with explicit q_causal_offsets; " "pass microbatch_rows=0")
    if paged:
        # Microbatch windows slice k[:, :kw], which has no meaning on a page pool, and the
        # global-id convention (b * seqlen_k + local) has no pool-row counterpart yet.
        if microbatch_rows > 0:
            raise NotImplementedError("microbatch_rows > 0 is not supported with paged K; pass microbatch_rows=0")
        if topk_indices_global:
            raise NotImplementedError("paged K returns local key ids only; pass topk_indices_global=False")
        microbatch_rows = 0

    # Resolve auto microbatch (microbatch_rows < 0).  Long-sequence auto-policy
    # applies to bf16 only; block-scaled microbatch is unsupported (per-window
    # scale repacking is not implemented), LSE needs a single launch, and explicit
    # q_causal_offsets are not composable with the windowing — all stay single-launch.
    if microbatch_rows < 0:
        microbatch_rows = 0 if (is_block_scaled or want_lse or q_causal_offsets is not None) else _select_microbatch_rows(seqlen_q, bs, ratio)
    if is_block_scaled and microbatch_rows > 0:
        raise NotImplementedError(f"microbatch_rows is not supported with precision={precision!r} " "(per-window scale repacking is not implemented)")

    # --- Microbatch: process query rows in windows, REUSING ONE cand buffer ---
    # Each window [rs, rs+mb) is an independent ratio-causal sub-problem
    # (seqlen_q=mb, seqlen_k=kw=(true_qgs+rs+mb)/ratio): its window-local
    # q_global_start == kw*ratio - mb == true_qgs+rs reproduces the absolute
    # geometry and local KV ids == absolute KV ids (same top-k set as one launch).
    # q/k/w are passed as zero-copy strided VIEWS (no per-window contiguous copy
    # of q/w, no repeated copy of k[:, :kw]); a SINGLE cand buffer sized for the
    # largest (last) window is reused across windows, so peak transient memory is
    # one window's compact buffer, not the full bs*sq^2/(2*ratio).
    if 0 < microbatch_rows < seqlen_q:
        if microbatch_rows % ratio != 0 or seqlen_q % ratio != 0:
            raise ValueError(f"microbatching requires microbatch_rows ({microbatch_rows}) and " f"seqlen_q ({seqlen_q}) to be multiples of ratio ({ratio})")
        # Offset-0 windowing: the global problem is top-left (default offset 0), so
        # window [rs, rs+mb) reproduces global rows rs..rs+mb-1 with a per-window causal
        # offset == rs (NOT bottom-right).  Each window sees up to kw = (rs+mb)//ratio KV
        # columns and runs the per-window compress with q_causal_offsets == rs.
        windows = []
        max_pbf = 0
        rs = 0
        while rs < seqlen_q:
            mb = min(microbatch_rows, seqlen_q - rs)
            kw = min((rs + mb) // ratio, seqlen_k)
            max_pbf = max(max_pbf, per_batch_floats(mb, kw, ratio, rs))
            windows.append((rs, mb, kw))
            rs += mb
        cand_shared = _resolve_cand_buffer(cand_buffer, bs * max_pbf, device)
        idx_out = out_indices if out_indices is not None else torch.empty((bs, seqlen_q, topk), dtype=torch.int32, device=device)
        val_out = out_logits if out_logits is not None else torch.empty((bs, seqlen_q, topk), dtype=torch.float32, device=device)
        sm_out = (softmax_out if softmax_out is not None else torch.empty((bs, seqlen_q, topk), dtype=torch.float32, device=device)) if want_softmax else None
        # Per-window slab bases are [0, pbf, 2*pbf, ..., bs*pbf] (uniform offset ⇒ uniform
        # pbf), i.e. arange(bs+1)*pbf.  Build the ramp ONCE here so each window only does a
        # single scalar mul, instead of re-running _bshd_cand_batch_offsets' ~24-op device
        # chain (int64 cast + the _G polynomial twice + cumsum) per window — that chain is
        # what otherwise dominates compress's "others" GPU time at long seqlen (many windows).
        _slab_ramp = torch.arange(bs + 1, dtype=torch.int64, device=device)
        for rs, mb, kw in windows:
            # Each window's stage-2 writes DIRECTLY into its slice of the full
            # output (a strided view) — no per-window temp tensor + copy-back, so
            # the path is allocation-free / CUDA-graph friendly.  The per-window causal
            # offset rs (uniform across the batch) makes the window reproduce global rows
            # rs..rs+mb-1; the returned local ids are absolute KV columns (offset rs sees
            # cols [0, kw)), so the b*seqlen_k+local conversion below is correct.
            wco = torch.full((bs,), int(rs), dtype=torch.int32, device=device)
            # Uniform per-window offset rs ⇒ uniform per-batch float count pbf, so both the
            # buffer total (bs*pbf) and the per-batch slab bases (_slab_ramp*pbf) are host-known.
            # Pass both so the window launch skips the .item() sync AND the _bshd_cand_batch_offsets
            # device-arithmetic chain; stays CUDA-graph-capturable.
            pbf = per_batch_floats(mb, kw, ratio, rs)
            indexer_fwd_compress_topk(
                q[:, rs : rs + mb],
                k[:, :kw],
                w[:, rs : rs + mb],
                topk,
                ratio=ratio,
                qhead_per_kv_head=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                sm_scale=sm_scale,
                microbatch_rows=0,
                precision=precision,
                topk_block_threads=topk_block_threads,
                topk_indices_global=False,  # windows return local; convert once below
                cand_buffer=cand_shared,
                out_indices=idx_out[:, rs : rs + mb, :],
                out_logits=val_out[:, rs : rs + mb, :],
                return_softmax=want_softmax,
                softmax_out=(sm_out[:, rs : rs + mb, :] if want_softmax else None),
                q_causal_offsets=wco,
                deterministic=deterministic,
                _cand_total_floats=bs * pbf,
                _cand_batch_offsets=_slab_ramp * pbf,
            )
        # Convert with the FULL seqlen_k (each window's local id is the absolute KV
        # column, so b*seqlen_k+local is correct across windows).  In place so a
        # caller-provided idx_out stays the same tensor.
        if topk_indices_global:
            _compress_local_to_global_bshd_(idx_out, seqlen_k)
        return (idx_out, val_out, sm_out) if want_softmax else (idx_out, val_out)

    # Compact stage-1 buffer.  Two layouts:
    #   * default (q_causal_offsets is None ⇒ uniform offset 0): bs * per_batch_floats,
    #     2D compact store eligible (Int32 column + 64-bit batch base; see _use_cand_2d).
    #   * explicit per-batch q_causal_offsets: per_batch_floats varies per batch, so use a
    #     tight per-batch slab (cand_batch_offsets prefix sum, like THD) and the flat 1D
    #     store (2D would need a uniform pbf).
    if q_causal_offsets is None:
        cand_batch_offsets = None
        pbf = per_batch_floats(seqlen_q, seqlen_k, ratio)  # offset-0
        cand = _resolve_cand_buffer(cand_buffer, bs * pbf, device)
        cand_2d = _use_cand_2d(precision, pbf)
        cand_gemm = cand[: bs * pbf].view(bs, pbf) if cand_2d else cand
    else:
        # cand_batch_offsets (int64 (bs+1,) prefix-sum slab bases) is handed to the kernel.
        # The microbatch loop drives this path with a UNIFORM per-window offset known on the
        # host, so it precomputes the bases cheaply (arange*pbf) and the total (bs*pbf) and
        # passes them in — skipping BOTH the .item() host sync (illegal during CUDA-graph
        # capture → cudaErrorStreamCaptureInvalidated) AND _bshd_cand_batch_offsets' ~24-op
        # device-arithmetic chain (int64 cast + the _G polynomial twice + cumsum), which
        # otherwise dominated compress's "others" GPU time at long seqlen (one chain/window).
        # The passed bases are bit-identical to the _bshd output for a uniform offset.
        cand_batch_offsets = _cand_batch_offsets if _cand_batch_offsets is not None else _bshd_cand_batch_offsets(bs, seqlen_q, ratio, q_causal_offsets)
        if _cand_total_floats is not None:
            cand = _resolve_cand_buffer(cand_buffer, _cand_total_floats, device)
        elif cand_buffer is None:
            # Internal allocation requires a host-visible size.
            cand = _resolve_cand_buffer(None, int(cand_batch_offsets[bs].item()), device)
        else:
            # The required size depends on device-resident offsets. Validate
            # only host-visible scratch metadata and trust the caller's sizing.
            cand = _resolve_cand_buffer(cand_buffer, cand_buffer.numel(), device)
        cand_2d = False
        cand_gemm = cand

    head_dim_padded = (head_dim + 15) // 16 * 16
    unified_k_block_size = 64 if head_dim_padded % 64 == 0 else head_dim_padded
    unified_kv_stage = 4
    # Without LSE the denom slot is compiled out: None at compile and launch (R3).
    lse_buf = (lse_out if lse_out is not None else torch.empty((bs, seqlen_q), dtype=torch.float32, device=device)) if want_lse else None
    denom_tmp = lse_buf

    if is_block_scaled:
        sf_groups = _ceil_div(head_dim, sf_vec_size)
        q_shape = bshd_packed_scale_shape(
            bs=bs,
            seqlen=seqlen_q,
            n_heads_kv=n_heads_kv,
            sf_groups=sf_groups,
            pack_q_heads=qhead_per_kv_head,
        )
        k_shape = bshd_packed_scale_shape(
            bs=bs,
            seqlen=seqlen_k,
            n_heads_kv=n_heads_kv,
            sf_groups=sf_groups,
        )
        if tuple(q_scale.shape) != q_shape:
            raise ValueError(f"q_scale packed shape must be {q_shape}, got {tuple(q_scale.shape)}")
        if tuple(k_scale.shape) != k_shape:
            raise ValueError(f"k_scale packed shape must be {k_shape}, got {tuple(k_scale.shape)}")

    compile_key = (
        precision,
        "is_compressed_logits",
        bool(want_lse),
        bool(cand_2d),
        q_causal_offsets is not None,  # per-batch offset path (mQCausalOffsets/cand_batch_offsets)
        q.dtype,
        # Block-scaled kernels bake scale/TMA layout from bs*n_heads_kv and the
        # q/k/w/scale dtypes; bf16 ignores the scale dtypes
        # (None) but the extra fields are harmless.
        k.dtype,
        w.dtype,
        getattr(q_scale, "dtype", None),
        getattr(k_scale, "dtype", None),
        bs,
        n_heads_kv,
        head_dim,
        qhead_per_kv_head,
        ratio,
        m_block_size,
        n_block_size,
        unified_k_block_size,
        unified_kv_stage,
        sf_vec_size,
        float(sm_scale),
        seqlen_q,
        None if paged else seqlen_k,  # paged: the key extent is a runtime bound
        page_size,
    )
    q_kernel = kernel_qk_view(precision, q)
    k_kernel = kernel_qk_view(precision, k)

    if compile_key not in _compile_cache:
        q_cute = _to_cute_tensor(q_kernel)
        k_cute = _to_cute_tensor(k_kernel)
        w_cute = _to_cute_tensor(w)
        cand_cute = _to_cute_tensor(cand_gemm, leading_dim=1 if cand_2d else 0)
        denom_cute = _to_cute_tensor(denom_tmp) if denom_tmp is not None else None
        qco_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None
        cbo_cute = _to_cute_tensor(cand_batch_offsets, leading_dim=0) if cand_batch_offsets is not None else None
        bt_cute = _to_cute_tensor(block_table, assumed_align=4, leading_dim=1) if paged else None
        kvl_cute = _to_cute_tensor(kv_lens, assumed_align=4, leading_dim=0) if paged else None
        current_stream = resolve_stream(None)
        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q)
        max_k_arg = cutlass.Int32(seqlen_k)

        if is_block_scaled:
            q_scale_cute = _to_cute_tensor(q_scale)
            k_scale_cute = _to_cute_tensor(k_scale)
            kernel_obj = BLOCK_SCALED_FORMATS[precision].kernel_cls(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=64,
                kv_stage=24,
                ratio=ratio,
                is_varlen=False,
                sf_vec_size=sf_vec_size,
                compute_lse=want_lse,
                is_compressed_logits=True,
            )
            _compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                q_scale_cute,
                k_scale_cute,
                cand_cute,
                denom_cute,
                scale_arg,
                max_q_arg,
                max_k_arg,
                None,
                None,
                None,
                None,
                qco_cute,  # mQCausalOffsets (None ⇒ offset 0; per-batch tensor otherwise)
                current_stream,
                cbo_cute,  # mCandBatchOffsets (per-batch tight slab base; None ⇒ uniform)
                options=compile_options(),
            )
        else:
            kernel_obj = IndexerScoreUnifiedSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=unified_k_block_size,
                kv_stage=unified_kv_stage,
                ratio=ratio,
                is_varlen=False,
                compute_lse=want_lse,
                is_compressed_logits=True,
                cand_2d=cand_2d,
                page_size=page_size,
            )
            _compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                cand_cute,
                denom_cute,
                scale_arg,
                max_q_arg,
                max_k_arg,
                None,
                None,
                qco_cute,  # mQCausalOffsets (None ⇒ offset 0; per-batch tensor otherwise)
                current_stream,
                cbo_cute,
                bt_cute,
                kvl_cute,
                options=compile_options(),
            )

    current_stream = resolve_stream(None)
    scale_arg = cutlass.Float32(sm_scale)
    max_q_arg = cutlass.Int32(seqlen_q)
    max_k_arg = cutlass.Int32(seqlen_k)
    # No out.fill_(-inf): every compact slot in [0, per_batch_floats) is a valid
    # ratio-causal position and is written exactly once by the GEMM epilogue.
    with torch.cuda.nvtx.range("indexer_fwd_compress_gemm"):
        if is_block_scaled:
            _compile_cache[compile_key](
                q_kernel,
                k_kernel,
                w,
                q_scale,
                k_scale,
                cand,
                denom_tmp,
                scale_arg,
                max_q_arg,
                max_k_arg,
                None,
                None,
                None,
                None,
                q_causal_offsets,  # mQCausalOffsets (None ⇒ offset 0; per-batch otherwise)
                current_stream,
                cand_batch_offsets,  # mCandBatchOffsets (per-batch tight slab base)
            )
        else:
            _compile_cache[compile_key](
                q,
                k,
                w,
                cand_gemm,
                denom_tmp,
                scale_arg,
                max_q_arg,
                max_k_arg,
                None,
                None,
                q_causal_offsets,  # mQCausalOffsets (None ⇒ offset 0)
                current_stream,
                cand_batch_offsets,
                block_table,
                kv_lens,
            )

    with torch.cuda.nvtx.range("indexer_fwd_compress_stage2_topk"):
        stage2_out = compress_stage2_topk(
            cand,
            bs,
            seqlen_q,
            seqlen_k,
            topk,
            ratio,
            block_threads=topk_block_threads,
            stream=current_stream,
            out_indices=out_indices,
            out_logits=out_logits,
            cand_batch_offsets=cand_batch_offsets,
            q_causal_offsets=q_causal_offsets,
            out_softmax=softmax_out,
            return_softmax=want_softmax,
            deterministic=deterministic,
            kv_lens=kv_lens,
        )
    if want_softmax:
        topk_indices, topk_logits, topk_softmax = stage2_out
    else:
        topk_indices, topk_logits = stage2_out
    if topk_indices_global:
        _compress_local_to_global_bshd_(topk_indices, seqlen_k)
    result = (topk_indices, topk_logits)
    if want_softmax:
        result += (topk_softmax,)
    if want_lse:
        result += (lse_buf,)
    return result


def _compress_cand_batch_offsets(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None,
    ratio: int,
    q_causal_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPU int64 ``(bs+1,)`` exclusive-prefix offsets of the per-batch compact
    sizes (``offsets[b]`` = start of batch b's slab; ``offsets[bs]`` = total).
    Computed on-device (no CPU sync) so the native varlen path stays graph-able.

    Per-batch causal offset ``q_causal_offsets[b]`` (default 0 = top-left; matches the
    dense indexer convention).
    Mirrors host ``per_batch_floats``: pbf_b = G(offset_b + sq_b) - G(offset_b)."""
    cu_q = cu_seqlens_q.to(torch.int64)
    sq = cu_q[1:] - cu_q[:-1]
    if q_causal_offsets is None:
        qgs = torch.zeros_like(sq)
    else:
        qgs = q_causal_offsets.to(torch.int64)

    def _G(n):  # sum_{m=0}^{n} floor(m/ratio) = ratio*Q*(Q-1)/2 + Q*(S+1)
        Q = n // ratio
        S = n - Q * ratio
        return ratio * Q * (Q - 1) // 2 + Q * (S + 1)

    pbf = _G(qgs + sq) - _G(qgs)
    offsets = torch.zeros(sq.numel() + 1, dtype=torch.int64, device=cu_q.device)
    torch.cumsum(pbf, dim=0, out=offsets[1:])
    return offsets


def compress_topk_cand_buffer_size_thd(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None,
    ratio: int,
    q_causal_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """THD/varlen pre-allocation helper for the compressed-logits top-k (the BSHD
    analog is ``compress_topk_cand_buffer_size``).

    Returns ``(cand_batch_offsets, total_floats)``:
      * ``cand_batch_offsets`` — int64 ``(bs+1,)`` GPU exclusive-prefix slab
        offsets (``[b]`` = batch b's compact start; ``[bs]`` = total).
      * ``total_floats`` — float32 element count the ``cand_buffer`` must hold.

    ``cu_seqlens_k`` is ``None`` for paged K (``block_table`` + ``kv_lens``): the
    compact sizes depend on the query lengths and offsets only.

    Device values are consumed as provided; callers own the per-batch
    ratio-causal and non-negative offset contracts. Feed the returned
    ``cand_batch_offsets`` (plus a float32 ``cand_buffer`` with
    ``total_floats`` elements and ``out_indices``/``out_logits``) back into
    ``indexer_forward_top_k_wrapper(...)`` to make the THD path
    CUDA-graph capturable with no
    internal offset compute + ``.item()`` sync per call.  (Strictly zero extra
    transient allocation additionally needs ``topk_indices_global=False``; the
    default global-id conversion records transients into the graph pool — no sync.)"""
    if (
        cu_seqlens_q.ndim != 1
        or cu_seqlens_q.numel() < 2
        or (cu_seqlens_k is not None and (cu_seqlens_k.ndim != 1 or cu_seqlens_k.numel() != cu_seqlens_q.numel()))
    ):
        raise ValueError("cu_seqlens_q/k must be 1D of equal length (bs+1) >= 2")
    bs = cu_seqlens_q.numel() - 1
    if q_causal_offsets is not None:
        if (
            not q_causal_offsets.is_cuda
            or q_causal_offsets.dtype != torch.int32
            or q_causal_offsets.ndim != 1
            or q_causal_offsets.numel() != bs
            or q_causal_offsets.device != cu_seqlens_q.device
        ):
            raise ValueError("q_causal_offsets must be a 1D CUDA int32 tensor of shape (bs,) " "on the same device as cu_seqlens_q")
    offsets = _compress_cand_batch_offsets(cu_seqlens_q, cu_seqlens_k, ratio, q_causal_offsets)
    total_floats = int(offsets[-1].item())
    return offsets, total_floats


def _run_compress_gemm_varlen(
    q,
    k,
    w,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    ratio,
    sm_scale,
    qhead_per_kv_head,
    m_block_size,
    n_block_size,
    cand_buffer=None,
    cand_batch_offsets=None,
    q_causal_offsets=None,
    *,
    precision="bf16",
    q_scale=None,
    k_scale=None,
    cu_seqlens_q_scale_padded=None,
    cu_seqlens_k_scale_padded=None,
    sf_vec_size=32,
    want_lse=False,
    lse_out=None,
    page_size=0,
    block_table=None,
    kv_lens=None,
):
    """Native varlen compress-logits GEMM (BF16 or block-scaled): one launch writes the
    per-batch tight compact cand_buffer (slab b at ``cand_batch_offsets[b]``). Returns
    ``(cand, cand_batch_offsets)``.  When both ``cand_buffer`` and
    ``cand_batch_offsets`` are caller-supplied (from
    ``compress_topk_cand_buffer_size_thd``) the path does NO ``.item()`` sync.

    ``page_size > 0`` reads K from a page pool through ``block_table`` / ``kv_lens``
    (BF16 only); ``cu_seqlens_k`` is then ``None``.

    A block-scaled ``precision`` (``'mxfp8'`` or ``'mxfp4'``) runs the blockscaled
    logits GEMM on the format's Q/K and packed scales (see ``indexer_fwd_compress_topk``).
    THD concatenates per-sequence padded scale regions along ``MN`` and keeps
    only KV heads in ``L``. The user-provided Q/K scale prefixes are consumed
    verbatim. Each Q span times ``qhead_per_kv_head`` and each K span must be a
    multiple of 128 MN rows. The Q/K data remains compact under the original
    ``cu_seqlens``.
    Block-scaled precisions keep the tight 1D compact store (single-launch).

    ``q_causal_offsets`` ((bs,) int32, default None=0 = top-left, matches dense): the
    per-batch causal offset; the caller-supplied ``cand_batch_offsets`` must have been
    sized with the SAME offsets (``compress_topk_cand_buffer_size_thd``)."""
    device = q.device
    total_q, n_heads_q, head_dim = q.shape
    head_dim = logical_head_dim(precision, head_dim)
    n_heads_kv = k.shape[-2]
    bs = cu_seqlens_q.numel() - 1
    if cand_batch_offsets is None:
        offsets = _compress_cand_batch_offsets(cu_seqlens_q, cu_seqlens_k, ratio, q_causal_offsets)
    else:
        offsets = cand_batch_offsets.to(torch.int64)
    if cand_buffer is None:
        # Internal alloc: one setup-time .item() for the total size.
        cand = _resolve_cand_buffer(None, int(offsets[bs].item()), device)
    else:
        if not cand_buffer.is_cuda or cand_buffer.dtype != torch.float32 or not cand_buffer.is_contiguous() or cand_buffer.device != device:
            raise ValueError("cand_buffer must be a contiguous CUDA float32 tensor on the input device")
        # The required size depends on device-resident offsets. Avoid reading
        # offsets back merely to validate caller-owned scratch; the caller
        # guarantees that the buffer was sized for the supplied sequence data.
        cand = cand_buffer.view(-1)
    denom_tmp = lse_out if want_lse else None  # dead slot compiled out (R3)

    head_dim_padded = (head_dim + 15) // 16 * 16
    unified_k_block_size = 64 if head_dim_padded % 64 == 0 else head_dim_padded
    unified_kv_stage = 4

    if precision in BLOCK_SCALED_FORMATS:
        # Data uses compact cu_seqlens offsets. Scale regions are independently padded
        # per sequence, concatenated along MN, and addressed by the two required
        # scale-padded prefixes; L contains only the KV head.
        # No host-side max_seqlen slab is part of the THD scale ABI. The sole
        # caller validates the complete scale contract before entering here.
        compile_key = (
            f"{precision}_varlen_compress",
            q.dtype,
            k.dtype,
            w.dtype,
            q_scale.dtype,
            k_scale.dtype,
            bs,
            n_heads_kv,
            head_dim,
            qhead_per_kv_head,
            ratio,
            m_block_size,
            n_block_size,
            sf_vec_size,
            float(sm_scale),
            int(max_seqlen_q),
            int(max_seqlen_k),
            q_causal_offsets is not None,
            bool(want_lse),
        )
        q_kernel = kernel_qk_view(precision, q)
        k_kernel = kernel_qk_view(precision, k)
        if compile_key not in _compile_cache:
            q_cute = _to_cute_tensor(q_kernel)
            k_cute = _to_cute_tensor(k_kernel)
            w_cute = _to_cute_tensor(w)
            q_scale_cute = _to_cute_tensor(q_scale)
            k_scale_cute = _to_cute_tensor(k_scale)
            cand_cute = _make_i64_cand_buffer_compile_tensor()
            denom_cute = _to_cute_tensor(denom_tmp) if denom_tmp is not None else None
            cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0)
            cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0)
            cu_q_scale_cute = _to_cute_tensor(cu_seqlens_q_scale_padded, leading_dim=0)
            cu_k_scale_cute = _to_cute_tensor(cu_seqlens_k_scale_padded, leading_dim=0)
            offs_cute = _to_cute_tensor(offsets, leading_dim=0)
            qco_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None
            stream = resolve_stream(None)
            kernel_obj = BLOCK_SCALED_FORMATS[precision].kernel_cls(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=64,
                kv_stage=24,
                ratio=ratio,
                is_varlen=True,
                sf_vec_size=sf_vec_size,
                compute_lse=want_lse,
                is_compressed_logits=True,
            )
            _compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                q_scale_cute,
                k_scale_cute,
                cand_cute,
                denom_cute,
                cutlass.Float32(sm_scale),
                cutlass.Int32(int(max_seqlen_q)),
                cutlass.Int32(int(max_seqlen_k)),
                cu_q_cute,
                cu_k_cute,
                cu_q_scale_cute,
                cu_k_scale_cute,
                qco_cute,
                stream,
                offs_cute,
                options=compile_options(),
            )
        stream = resolve_stream(None)
        with torch.cuda.nvtx.range(f"indexer_fwd_compress_gemm_varlen_{precision}"):
            _compile_cache[compile_key](
                q_kernel,
                k_kernel,
                w,
                q_scale,
                k_scale,
                cand,
                denom_tmp,
                cutlass.Float32(sm_scale),
                cutlass.Int32(int(max_seqlen_q)),
                cutlass.Int32(int(max_seqlen_k)),
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_q_scale_padded,
                cu_seqlens_k_scale_padded,
                q_causal_offsets,
                stream,
                offsets,
            )
        return cand, offsets

    compile_key = (
        "bf16_varlen_compress",
        q.dtype,
        w.dtype,
        bs,
        n_heads_kv,
        head_dim,
        qhead_per_kv_head,
        ratio,
        m_block_size,
        n_block_size,
        unified_k_block_size,
        unified_kv_stage,
        float(sm_scale),
        int(max_seqlen_q),
        None if page_size else int(max_seqlen_k),  # paged: a runtime bound, not a layout parameter
        q_causal_offsets is not None,
        bool(want_lse),
        page_size,
    )
    if compile_key not in _compile_cache:
        q_cute = _to_cute_tensor(q)
        k_cute = _to_cute_tensor(k)
        w_cute = _to_cute_tensor(w)
        cand_cute = _make_i64_cand_buffer_compile_tensor()
        denom_cute = _to_cute_tensor(denom_tmp) if denom_tmp is not None else None
        cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0)
        cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if cu_seqlens_k is not None else None
        offs_cute = _to_cute_tensor(offsets, leading_dim=0)
        qco_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None
        bt_cute = _to_cute_tensor(block_table, assumed_align=4, leading_dim=1) if page_size else None
        kvl_cute = _to_cute_tensor(kv_lens, assumed_align=4, leading_dim=0) if page_size else None
        stream = resolve_stream(None)
        kernel_obj = IndexerScoreUnifiedSm100(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv_head,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            k_block_size=unified_k_block_size,
            kv_stage=unified_kv_stage,
            ratio=ratio,
            is_varlen=True,
            compute_lse=want_lse,
            is_compressed_logits=True,
            page_size=page_size,
        )
        _compile_cache[compile_key] = cute.compile(
            kernel_obj,
            q_cute,
            k_cute,
            w_cute,
            cand_cute,
            denom_cute,
            cutlass.Float32(sm_scale),
            cutlass.Int32(int(max_seqlen_q)),
            cutlass.Int32(int(max_seqlen_k)),
            cu_q_cute,
            cu_k_cute,
            qco_cute,
            stream,
            offs_cute,
            bt_cute,
            kvl_cute,
            options=compile_options(),
        )
    stream = resolve_stream(None)
    with torch.cuda.nvtx.range("indexer_fwd_compress_gemm_varlen"):
        _compile_cache[compile_key](
            q,
            k,
            w,
            cand,
            denom_tmp,
            cutlass.Float32(sm_scale),
            cutlass.Int32(int(max_seqlen_q)),
            cutlass.Int32(int(max_seqlen_k)),
            cu_seqlens_q,
            cu_seqlens_k,
            q_causal_offsets,
            stream,
            offsets,
            block_table,
            kv_lens,
        )
    return cand, offsets


def _indexer_fwd_compress_topk_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    ratio: int,
    qhead_per_kv_head: Optional[int],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    sm_scale: float = 1.0,
    microbatch_rows: int = -1,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    topk_block_threads: int = -1,
    topk_indices_global: bool = True,
    cand_buffer: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    return_softmax: Optional[bool] = None,
    softmax_out: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    block_table: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    """THD/varlen compressed-logits top-k, supporting BF16, MXFP8, and MXFP4.

    ``block_table`` ((bs, max_pages) int32) + ``kv_lens`` ((bs,) int32) read K from a
    page pool ``(num_pages, page_size, 1, head_dim)`` (BF16, local ids); ``cu_seqlens_k``
    is then ``None`` and ``max_seqlen_k`` defaults to ``max_pages * page_size``.

    A block-scaled ``precision`` runs the blockscaled logits GEMM with
    blockscaled-packed E8M0 scales: for ``'mxfp8'``, ``q``/``k`` are
    ``float8_e4m3fn``; for ``'mxfp4'``, they are packed ``float4_e2m1fn_x2``
    (last dim ``head_dim // 2``).
    THD requires both scale-padded prefixes and concatenates per-sequence scale regions
    along ``MN``. The user-provided prefix values are consumed verbatim; spans may
    exceed the logical lengths, but each Q span times ``qhead_per_kv_head`` and
    each K span must be a multiple of 128 MN rows. The Q/K data remains compact
    under ``cu_seqlens_q``/``cu_seqlens_k``.

    Caller pre-allocation (CUDA-graph friendly): ``out_indices``/``out_logits``
    ``(total_q, topk)`` are written in place; passing ``cand_buffer`` +
    ``cand_batch_offsets`` (from ``compress_topk_cand_buffer_size_thd``) skips the
    internal per-call offset compute + ``.item()`` sync (and its per-batch shape
    check, which the helper already did).

    One compress-logits GEMM writes the per-batch *tight* compact cand_buffer
    (slab b at ``cand_batch_offsets[b]``, a GPU-computed prefix sum) and one
    stage-2 launch (grid=(total_q, 1)) selects the per-token top-k.  No per-batch
    CPU sync and no B kernel launches — CUDA-graph friendly.  ``microbatch_rows`` is
    unused here (the tight layout already keeps
    peak memory at the exact total).  Output is ``(total_q, topk)``; global ids
    follow the THD convention ``cu_seqlens_k[b] + local`` (matches the dispatcher).

    CUDA-graph capture contract: pass explicit ``max_seqlen_q``/``max_seqlen_k`` and
    pre-allocated ``cand_buffer`` + ``cand_batch_offsets`` (from
    ``compress_topk_cand_buffer_size_thd``) + ``out_indices``/``out_logits``.
    Since fused softmax is on by default, also pass ``softmax_out`` or set
    ``return_softmax=False``. LSE capture additionally requires ``lse_out``.
    Omitting ``max_seqlen_*`` (derived via ``.max().item()``) or ``cand_buffer`` /
    ``cand_batch_offsets`` (internal scratch sizing via ``.item()``) needs a host
    sync that is illegal mid-capture; omitting ``out_indices`` / ``out_logits``
    would allocate the output mid-capture. Each raises a clear ValueError here.
    Device-resident prefix and offset values are always trusted; only their
    host-visible metadata is validated.

    Scope of "allocation-free": the above makes the path graph-capturable with NO
    host sync, and — with ``topk_indices_global=False`` (local KV ids) — with no
    extra transient allocation either.  The DEFAULT ``topk_indices_global=True``
    still creates transients in the local->global id conversion (batch ids /
    per-batch offsets / the broadcast add); these are graph-capturable (no sync) but
    are recorded into the graph's private pool, so they are NOT zero-extra-allocation.
    Use ``topk_indices_global=False`` to strictly avoid the conversion temporaries.
    ``deterministic=True`` makes exact-value ties at the K-th boundary select the
    smallest local KV indices; output slot order is still unspecified.
    """
    from ..indexer_top_k.compress_top_k_sm100 import compress_stage2_topk_varlen

    if return_softmax is None:
        return_softmax = _RETURN_SOFTMAX_DEFAULT
    precision = validate_indexer_precision(precision)
    is_block_scaled = precision in BLOCK_SCALED_FORMATS
    if precision == "bf16" and (q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None):
        raise ValueError(f"Block scales and scale padded cu_seqlens require precision={BLOCK_SCALED_PRECISION_NAMES}")
    # The THD scale contract (regions concatenated along MN, addressed by the padded
    # prefixes, L = KV head only) is validated below once bs/head metadata is known.
    validate_sf_vec_size(precision, sf_vec_size)
    paged = _check_paged_args(block_table, kv_lens)
    if q.ndim != 3 or k.ndim != (4 if paged else 3) or w.ndim != 2:
        raise ValueError("THD compressed-logits top-k expects q (T_q,H,D), k (T_k,Hkv,D) or a page pool (num_pages,page,Hkv,D), w (T_q,H)")
    device = q.device
    total_q, n_heads_q, head_dim = q.shape
    # Shape / dtype / device consistency: reject malformed inputs here rather than
    # a kernel OOB / cryptic failure.
    if k.shape[-1] != head_dim:
        raise ValueError(f"q last dim ({head_dim}) != k last dim ({k.shape[-1]})")
    if tuple(w.shape) != (total_q, n_heads_q):
        raise ValueError(f"w shape must be (total_q={total_q}, n_heads_q={n_heads_q}), " f"got {tuple(w.shape)}")
    if k.device != device or w.device != device:
        raise ValueError("q, k, w must be on the same device")
    if precision == "bf16":
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("THD compressed-logits top-k requires bfloat16 q/k and bfloat16 or float32 w")
        if w.dtype == torch.float32 and w.stride(-1) != 1:
            raise NotImplementedError(f"FP32 w requires unit last stride, got strides {w.stride()}")
    else:
        validate_block_scaled_inputs(precision, q, k, w, q_scale, k_scale)
    # After the dtype checks, so a wrong Q/K dtype is reported as such, not as a bad head_dim.
    head_dim = logical_head_dim(precision, head_dim)
    # cu_seqlens are consumed directly by the kernel: enforce the same int32 /
    # contiguous / same-device contract as the dense THD path (avoids a silent
    # .to() copy or a wrong-device kernel arg).
    if cu_seqlens_k is None and not paged:
        raise ValueError("THD compressed-logits top-k requires cu_seqlens_k (or block_table + kv_lens for paged K)")
    for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
        if t is not None and (not t.is_cuda or t.ndim != 1 or t.dtype != torch.int32 or t.stride(0) != 1 or t.device != device):
            raise ValueError(f"{name} must be a contiguous 1D int32 CUDA tensor on the input device")

    bs = cu_seqlens_q.numel() - 1
    if bs <= 0 or (cu_seqlens_k is not None and cu_seqlens_k.numel() != bs + 1):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have matching bs + 1 length")
    n_heads_kv = int(k.shape[-2])
    # qhead_per_kv_head (GQA group size) contract: a
    # wrong head mapping corrupts head-reduce / tile mapping, so n_heads_q must be
    # divisible by n_heads_kv and an explicit value must equal the derived ratio.
    if n_heads_q % n_heads_kv != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})")
    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    elif qhead_per_kv_head != n_heads_q // n_heads_kv:
        raise ValueError(f"qhead_per_kv_head ({qhead_per_kv_head}) must equal n_heads_q // " f"n_heads_kv ({n_heads_q // n_heads_kv})")
    _validate_indexer_qhead_per_kv_head(qhead_per_kv_head, precision)
    if n_heads_kv != 1:
        raise ValueError(
            "THD compress-logits top-k currently requires n_heads_kv=1 (MQA); " f"got n_heads_kv={n_heads_kv}. The stage-1 GEMM reads only KV head 0."
        )
    if paged:
        if precision != "bf16":
            raise NotImplementedError("paged K is served by the BF16 indexer only")
        if topk_indices_global:
            raise NotImplementedError("paged K returns local key ids only; pass topk_indices_global=False")
        page_size, max_pages = _resolve_paged_k(k, block_table, kv_lens, cu_seqlens_k, n_block_size=n_block_size, batch=int(bs), device=device)
        if max_seqlen_k is None:
            max_seqlen_k = max_pages * page_size
    else:
        page_size = 0
    if is_block_scaled:
        validate_thd_scale_contract(
            q_scale,
            k_scale,
            cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded,
            precision=precision,
            bs=bs,
            n_heads_kv=n_heads_kv,
            sf_groups=_ceil_div(head_dim, sf_vec_size),
            device=device,
        )
    if precision == "bf16":
        m_block_size = _resolve_m_block_size(m_block_size, qhead_per_kv_head, head_dim, compressed=True, path="SM100 compressed indexer forward")
    if m_block_size % qhead_per_kv_head != 0:
        raise ValueError(f"m_block_size ({m_block_size}) must be divisible by " f"qhead_per_kv_head ({qhead_per_kv_head})")
    cu_q32 = cu_seqlens_q.to(torch.int32)
    cu_k32 = cu_seqlens_k.to(torch.int32) if cu_seqlens_k is not None else None
    capturing = torch.cuda.is_current_stream_capturing()
    # Deriving max_seqlen from cu_seqlens needs a .max().item() host sync, which is
    # ILLEGAL during CUDA-graph capture.  Graph contract: the caller MUST pass
    # explicit max_seqlen_q/k when capturing (both are known at capture time) —
    # raise a clear error instead of letting the implicit sync crash the capture
    # cryptically.  (Eager / warmup may still omit them and derive here.)
    if max_seqlen_q is None or max_seqlen_k is None:
        if capturing:
            raise ValueError(
                "max_seqlen_q and max_seqlen_k must be passed explicitly during " "CUDA-graph capture (deriving them from cu_seqlens needs a host sync)"
            )
        if max_seqlen_q is None:
            max_seqlen_q = int((cu_q32[1:] - cu_q32[:-1]).max().item())
        if max_seqlen_k is None:
            max_seqlen_k = int((cu_k32[1:] - cu_k32[:-1]).max().item())
    if topk_block_threads <= 0:
        # one stage-2 block per global query token (grid = (total_q, 1)).
        topk_block_threads = _select_topk_block_threads(total_q, int(max_seqlen_k))
    # Same graph contract: omitting cand_buffer or cand_batch_offsets makes
    # _run_compress_gemm_varlen size/compute the scratch via int(offsets[bs].item()) —
    # a host sync illegal mid-capture.  Require BOTH to be pre-passed when capturing
    # (raise a clear error instead of a cryptic sync crash).
    if capturing and (cand_buffer is None or cand_batch_offsets is None):
        raise ValueError(
            "cand_buffer and cand_batch_offsets must be passed explicitly during "
            "CUDA-graph capture (from compress_topk_cand_buffer_size_thd); sizing the "
            "scratch internally needs a host sync"
        )
    # Allocation-free graph contract: a missing out_indices/out_logits is allocated
    # by stage-2 (torch.empty), so the captured output would live in the graph's
    # private pool instead of a caller buffer.  Require both during capture — unlike
    # a sync this would not crash, but it breaks the pre-allocated contract.
    if capturing and (out_indices is None or out_logits is None):
        raise ValueError(
            "out_indices and out_logits must be passed explicitly during CUDA-graph "
            "capture (the allocation-free contract); stage-2 would otherwise allocate "
            "the output mid-capture"
        )
    want_softmax = return_softmax or softmax_out is not None
    if capturing and want_softmax and softmax_out is None:
        raise ValueError(
            "softmax over the top-k is computed by default; during CUDA-graph "
            "capture pass a pre-allocated softmax_out ((total_q, topk) fp32) or "
            "set return_softmax=False"
        )
    want_lse = return_lse or lse_out is not None
    if lse_out is not None and (
        tuple(lse_out.shape) != (total_q,) or lse_out.dtype != torch.float32 or not lse_out.is_cuda or lse_out.device != device or not lse_out.is_contiguous()
    ):
        raise ValueError(f"lse_out must be a contiguous CUDA float32 tensor of shape " f"({total_q},) on {device}")
    if capturing and want_lse and lse_out is None:
        raise ValueError("return_lse=True during CUDA-graph capture requires a pre-allocated " "lse_out ((total_q,) fp32)")
    lse_buf = None
    if want_lse:
        lse_buf = lse_out if lse_out is not None else torch.empty((total_q,), dtype=torch.float32, device=device)
    _check_q_covered_by_k(int(max_seqlen_q), int(max_seqlen_k), ratio, what_q="max_seqlen_q", what_k="max_seqlen_k")
    if cand_batch_offsets is not None:
        # Structural check (host metadata, no sync): must be the exact tensor the
        # helper produces — int64, contiguous, (bs+1,), on the input device — so
        # the downstream .to(torch.int64) is a guaranteed no-op (no extra alloc /
        # copy).  The VALUES ([0]=0, [bs]=total,
        # monotonic) are guaranteed by compress_topk_cand_buffer_size_thd — pass
        # offsets ONLY from that helper.
        if (
            not cand_batch_offsets.is_cuda
            or cand_batch_offsets.ndim != 1
            or cand_batch_offsets.numel() != bs + 1
            or cand_batch_offsets.dtype != torch.int64
            or not cand_batch_offsets.is_contiguous()
            or cand_batch_offsets.device != device
        ):
            raise ValueError(
                "cand_batch_offsets must be a contiguous 1D int64 CUDA tensor of " "length bs+1 on the input device (use " "compress_topk_cand_buffer_size_thd)"
            )

    # Only needed for the optional GPU-side local-to-global conversion. Device
    # prefix values are caller-owned and are not copied to the host for
    # validation.
    sq_b = None

    # Caller output buffers: validate up front (same contract as the stage-2
    # kernel: shape (total_q, topk), dtype, same device, contiguous last/topk dim).
    if out_indices is not None or out_logits is not None or softmax_out is not None:
        _out_shape = (total_q, topk)
        for _t, _nm, _dt in (
            (out_indices, "out_indices", torch.int32),
            (out_logits, "out_logits", torch.float32),
            (softmax_out, "softmax_out", torch.float32),
        ):
            if _t is not None and (tuple(_t.shape) != _out_shape or _t.dtype != _dt or not _t.is_cuda or _t.device != device or _t.stride(-1) != 1):
                raise ValueError(f"{_nm} must be a CUDA {_dt} tensor of shape {_out_shape} on " f"{device} with a contiguous last (topk) dim")

    # Per-batch causal offsets (default None = 0 = top-left, matching the dense indexer).
    # When given, _run_compress_gemm_varlen sizes cand_batch_offsets with these offsets;
    # if the caller pre-supplied cand_batch_offsets it must have been sized with the SAME
    # offsets (compress_topk_cand_buffer_size_thd(..., q_causal_offsets=...)).
    if q_causal_offsets is not None and (
        not q_causal_offsets.is_cuda
        or q_causal_offsets.ndim != 1
        or q_causal_offsets.shape[0] != bs
        or q_causal_offsets.dtype != torch.int32
        or q_causal_offsets.device != device
    ):
        raise ValueError(f"q_causal_offsets must be a contiguous 1D int32 CUDA tensor of shape ({bs},) " f"on {device}")

    # Stage 1: native varlen compress GEMM → tight per-batch compact buffer.
    cand, offsets = _run_compress_gemm_varlen(
        q,
        k,
        w,
        cu_q32,
        cu_k32,
        max_seqlen_q,
        max_seqlen_k,
        ratio,
        sm_scale,
        qhead_per_kv_head,
        m_block_size,
        n_block_size,
        cand_buffer=cand_buffer,
        cand_batch_offsets=cand_batch_offsets,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        want_lse=want_lse,
        lse_out=lse_buf,
        page_size=page_size,
        block_table=block_table,
        kv_lens=kv_lens,
    )
    # Stage 2: single launch over all total_q query tokens (writes the caller's
    # out_indices/out_logits in place when provided).
    current_stream = resolve_stream(None)
    with torch.cuda.nvtx.range("indexer_fwd_compress_stage2_topk_varlen"):
        stage2_out = compress_stage2_topk_varlen(
            cand,
            cu_q32,
            cu_k32,
            offsets,
            total_q,
            max_seqlen_q,
            max_seqlen_k,
            topk,
            ratio,
            block_threads=topk_block_threads,
            stream=current_stream,
            out_indices=out_indices,
            out_logits=out_logits,
            q_causal_offsets=q_causal_offsets,
            out_softmax=softmax_out,
            return_softmax=want_softmax,
            deterministic=deterministic,
            kv_lens=kv_lens,
        )
    if want_softmax:
        idx_out, val_out, sm_out = stage2_out
    else:
        idx_out, val_out = stage2_out
    if topk_indices_global:
        # global = cu_seqlens_k[b] + local, per query token's batch b (GPU-only),
        # applied IN PLACE so a caller-provided out_indices stays the same tensor.
        _local_to_global_inplace(idx_out, 0, cu_q32, cu_k32, current_stream)
    result = (idx_out, val_out)
    if want_softmax:
        result += (sm_out,)
    if want_lse:
        result += (lse_buf,)
    return result
