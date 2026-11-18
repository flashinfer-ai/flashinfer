# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Packed-contiguous PyTorch APIs for the SM120 FP8 FMHA kernel.

Public entry point
------------------
``sm120_fmha_fp8_ragged_prefill``: packed contiguous Q/K/V.
``sm120_fmha_fp8_paged_prefill``: packed Q with paged K/V pools.

Causal mask conventions
-----------------------
- Both paths use bottom-right alignment: query i attends to key j when
  ``j <= i + (kv_len - q_len)``.

Sequence lengths are runtime metadata. ``max_seqlen_q`` only sizes the launch
grid and is not part of the compiled kernel cache key.

Causal kernels use load-balanced scheduling by default. Set
``PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1`` before planning or launching to
disable it.

Optional dependency
-------------------
Requires ``nvidia-cutlass-dsl>=4.7.0`` and ``cutlass.experimental``. The
version is checked only when the SM120 backend is used.
"""

import math
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import Optional, Union

import torch
from cutlass.cute.typing import Float32, Int32
from packaging.version import Version

from flashinfer.utils import ceil_div


_MIN_CUTLASS_DSL_VERSION = Version("4.7.0")


@lru_cache(maxsize=1)
def _check_cutlass_dsl_version() -> None:
    """Require the CuTe DSL version used by the SM120 FMHA kernels."""
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "SM120 FMHA requires nvidia-cutlass-dsl>=4.7.0, but the package "
            "is not installed"
        ) from exc
    if Version(installed_version) < _MIN_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            f"SM120 FMHA requires nvidia-cutlass-dsl>=4.7.0; found {installed_version}"
        )


def _cutlass_dtype(torch_dtype: torch.dtype):
    import cutlass

    return {
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[torch_dtype]


def _check_sm120(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    if not (major == 12 and minor == 0):
        raise RuntimeError(
            f"SM120 FMHA kernel requires SM120 GPU (compute capability 12.0), "
            f"got {major}.{minor}"
        )


def _use_balanced_scheduler(is_causal: bool) -> bool:
    return is_causal and os.environ.get("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING") != "1"


def _validate_sm120_prims_plan_options(
    *,
    kv_layout: str,
    required_kv_layout: str,
    custom_mask: Optional[torch.Tensor],
    packed_custom_mask: Optional[torch.Tensor],
    pos_encoding_mode: str,
    use_fp16_qk_reduction: bool,
    window_left: int,
    logits_soft_cap: float,
    prefix_len_ptr: Optional[torch.Tensor],
    token_pos_in_items_ptr: Optional[torch.Tensor],
    max_item_len_ptr: Optional[torch.Tensor],
    max_sequence_kv: Optional[int],
) -> None:
    """Fail fast for features outside the first PRIMS backend contract."""
    backend = "backend='cute-dsl-prims'"
    if kv_layout != required_kv_layout:
        raise ValueError(
            f"{backend} requires kv_layout={required_kv_layout!r}; got {kv_layout!r}"
        )
    if custom_mask is not None or packed_custom_mask is not None:
        raise NotImplementedError(f"{backend} does not support custom masks")
    if pos_encoding_mode != "NONE":
        raise NotImplementedError(
            f"{backend} requires pos_encoding_mode='NONE'; got "
            f"{pos_encoding_mode!r}. Apply RoPE to Q/K before attention."
        )
    if use_fp16_qk_reduction:
        raise NotImplementedError(
            f"{backend} uses FP32 accumulation and does not support "
            "use_fp16_qk_reduction=True"
        )
    if window_left >= 0:
        raise NotImplementedError(
            f"{backend} does not support sliding window; got window_left={window_left}"
        )
    if logits_soft_cap > 0:
        raise NotImplementedError(
            f"{backend} does not support logits_soft_cap={logits_soft_cap}"
        )
    if any(
        value is not None
        for value in (prefix_len_ptr, token_pos_in_items_ptr, max_item_len_ptr)
    ):
        raise NotImplementedError(
            f"{backend} does not support multi-item scoring/prefix metadata"
        )
    if max_sequence_kv is not None:
        raise NotImplementedError(
            f"{backend} does not support max_sequence_kv={max_sequence_kv}; "
            "this option is only supported by the cuDNN backend"
        )


def _validate_lse(q: torch.Tensor, lse: Optional[torch.Tensor]) -> None:
    """Validate the optional packed log2 LSE output tensor."""
    if lse is None:
        return
    expected_shape = (q.shape[0], q.shape[1])
    if tuple(lse.shape) != expected_shape:
        raise ValueError(
            f"lse must have shape {expected_shape}, got {tuple(lse.shape)}"
        )
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must have dtype torch.float32, got {lse.dtype}")
    if lse.device != q.device:
        raise ValueError("lse and q must be on the same device")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous")


def _prepare_skip_softmax_threshold(
    skip_softmax_threshold: Optional[Union[float, torch.Tensor]],
    q: torch.Tensor,
    batch_size: int,
) -> Optional[torch.Tensor]:
    """Validate or expand the per-request e-based skip threshold."""
    if skip_softmax_threshold is None:
        return None
    if isinstance(skip_softmax_threshold, torch.Tensor):
        if skip_softmax_threshold.dtype != torch.float32:
            raise ValueError(
                "skip_softmax_threshold must have dtype torch.float32, got "
                f"{skip_softmax_threshold.dtype}"
            )
        if not skip_softmax_threshold.is_cuda:
            raise ValueError("skip_softmax_threshold must be a CUDA tensor")
        if skip_softmax_threshold.device != q.device:
            raise ValueError("skip_softmax_threshold and q must be on the same device")
        expected_shape = (batch_size,)
        if tuple(skip_softmax_threshold.shape) != expected_shape:
            raise ValueError(
                f"skip_softmax_threshold must have shape {expected_shape}, got "
                f"{tuple(skip_softmax_threshold.shape)}"
            )
        if not skip_softmax_threshold.is_contiguous():
            raise ValueError("skip_softmax_threshold must be contiguous")
        return skip_softmax_threshold

    try:
        value = float(skip_softmax_threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "skip_softmax_threshold must be None, a Python float, or a torch.Tensor"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"skip_softmax_threshold must be finite and >= 0; got {value!r}"
        )
    return torch.full((batch_size,), value, dtype=torch.float32, device=q.device)


def _validate_hnd_paged_pool(name: str, pool: torch.Tensor) -> None:
    """Validate the HND inner layout while allowing combined-cache views."""
    if pool.ndim != 4:
        raise ValueError(
            f"{name} must have HND shape "
            f"(num_pages, Hkv, page_size, D), got {tuple(pool.shape)}"
        )
    _, num_heads, page_size, head_dim = pool.shape
    expected_inner_strides = (page_size * head_dim, head_dim, 1)
    min_page_stride = num_heads * page_size * head_dim
    if (
        tuple(pool.stride()[1:]) != expected_inner_strides
        or pool.stride(0) < min_page_stride
    ):
        raise ValueError(
            f"{name} must use HND storage with compact [Hkv, page_size, D] "
            "planes; a standalone pool or a K/V plane view from a combined "
            f"cache is supported, got strides {pool.stride()}"
        )


def sm120_fmha_fp8_ragged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    kv_tile: Optional[int] = None,
    q_tile: Optional[int] = None,
    lse: Optional[torch.Tensor] = None,
    v_scale: Optional[float] = None,
    enable_pdl: bool = False,
    skip_softmax_threshold: Optional[Union[float, torch.Tensor]] = None,
) -> None:
    """Run SM120 FP8 FMHA on packed contiguous ragged Q/K/V.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(total_q, Hq, D)``.
        dtype: ``float8_e4m3fn``.
    k : torch.Tensor
        Key tensor, shape ``(total_k, Hkv, D)``. Same dtype as ``q``.
    v : torch.Tensor
        Value tensor, shape ``(total_k, Hkv, D)``. Same dtype as ``q``.
    o : torch.Tensor
        Output tensor, shape ``(total_q, Hq, D)``, written in-place.
        dtype: ``float16`` or ``bfloat16``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        Runtime cumulative sequence offsets, both shape ``(B + 1,)`` int32.
    max_seqlen_q : int, optional
        Runtime launch-grid bound. Derived from ``cu_seqlens_q`` when omitted.
    is_causal : bool
        Bottom-right aligned causal mask. Causal kernels use balanced
        scheduling unless ``PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1``.
    sm_scale : float, optional
        Softmax scale. Defaults to ``1 / sqrt(D)``.
    kv_tile : int, optional
        K/V tile size (64 or 128). Defaults to 128.
    q_tile : int, optional
        Q tile size (64 or 128). Defaults to 128.
    lse : torch.Tensor, optional
        Preallocated float32 log2 LSE output, shape ``(total_q, Hq)``.
    v_scale : float, optional
        Scalar multiplier folded into the normalized output. Defaults to 1.
    enable_pdl : bool
        Whether to enable Programmatic Dependent Launch.
    skip_softmax_threshold : float or torch.Tensor, optional
        Skip a K/V tile when all rows owned by a compute warp satisfy
        ``exp((tile_max - running_max) * sm_scale) < threshold``. A Python
        float applies one static threshold to the whole batch. During CUDA
        Graph capture its value is fixed in the captured graph. A tensor is
        the dynamic form: it must be a contiguous CUDA float32 tensor on the
        same device as ``q`` with shape ``(batch_size,)``. The caller owns its
        allocation and lifetime, must keep its address stable, and may update
        values in place between graph replays. Tensor values are required to
        be finite and non-negative; this content contract is not checked to
        avoid a device-to-host synchronization. ``None`` selects the dense
        specialization; every supplied value, including zero, selects the
        skip-enabled specialization.
    Raises
    ------
    RuntimeError
        If the GPU is not SM120 or the configuration is not supported.
    """
    _check_cutlass_dsl_version()

    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
        compile_sm120_fmha_fp8_ragged_kernel,
    )

    assert q.ndim == 3, f"q must be (total_q, Hq, D), got {q.shape}"
    assert k.ndim == 3 and v.ndim == 3 and o.ndim == 3

    _check_sm120(q.device)
    _validate_lse(q, lse)

    batch_size = cu_seqlens_q.numel() - 1
    threshold = _prepare_skip_softmax_threshold(skip_softmax_threshold, q, batch_size)

    _, Hq, D = q.shape
    _, Hkv, D_k = k.shape
    if D_k != D:
        raise ValueError(f"head_dim mismatch: q={D}, k={D_k}")
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have the same shape, got {k.shape} and {v.shape}"
        )
    # CUDA tensor maps cannot describe a zero-extent K/V tensor. Handle the
    # all-empty ragged batch before TMA descriptor construction; requests with
    # empty K/V inside a non-empty batch still use the kernel's empty-KV path.
    if k.shape[0] == 0:
        o.zero_()
        if lse is not None:
            lse.fill_(-float("inf"))
        return
    cu_seqlens_q_i32 = cu_seqlens_q.to(torch.int32)
    cu_seqlens_k_i32 = cu_seqlens_k.to(torch.int32)
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q_i32[1:] - cu_seqlens_q_i32[:-1]).max().item())

    kv_tile = kv_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_KV_TILES[0]
    q_tile = q_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_Q_TILES[0]
    if (
        q_tile not in SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_Q_TILES
        or kv_tile not in SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_KV_TILES
        or D not in SM120FusedMultiHeadAttentionFP8ForwardTMA.SUPPORTED_HEAD_TILES
        or Hq % Hkv != 0
    ):
        raise RuntimeError(
            f"SM120 FP8 FMHA cannot implement config: "
            f"q={q.shape} k={k.shape} in={q.dtype} out={o.dtype} "
            f"kv_tile={kv_tile} q_tile={q_tile}"
        )

    kernel_fn = compile_sm120_fmha_fp8_ragged_kernel(
        in_dtype=q.dtype,
        out_dtype=o.dtype,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        is_causal=is_causal,
        kv_tile=kv_tile,
        q_tile=q_tile,
        device=q.device,
        with_lse=lse is not None,
        balanced_scheduler=_use_balanced_scheduler(is_causal),
        enable_skip_softmax=threshold is not None,
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scale_log2 = Float32(sm_scale * math.log2(math.e))
    output_scale = Float32(1.0 if v_scale is None else float(v_scale))

    kernel_fn(
        q,
        k,
        v,
        o,
        lse,
        scale_log2,
        output_scale,
        threshold,
        None,
        cu_seqlens_q_i32,
        None,
        cu_seqlens_k_i32,
        Int32(max_seqlen_q),
        enable_pdl,
        None,  # kv_tile_ranges
    )


# =============================================================================
# Paged KV prefill — packed Q only
# =============================================================================


def sm120_fmha_fp8_paged_prefill(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    o: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens_kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    max_seqlen_q: Optional[int] = None,
    kv_tile: Optional[int] = None,
    q_tile: Optional[int] = None,
    lse: Optional[torch.Tensor] = None,
    v_scale: Optional[float] = None,
    enable_pdl: bool = False,
    skip_softmax_threshold: Optional[Union[float, torch.Tensor]] = None,
    kv_tile_ranges: Optional[torch.Tensor] = None,
    gqa_pack_size: int = 1,
) -> None:
    """Run SM120 FP8 FMHA prefill with paged K/V cache.

    Q/O use packed contiguous storage ``(total_q, Hq, D)``. K and V are
    stored in separate paged pools;
    ``block_tables`` maps each batch item's logical K/V pages to shared
    physical page IDs, matching the shared block-table paged-KV format.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(total_q, Hq, D)``.
        dtype: ``float8_e4m3fn``.
    k_pool : torch.Tensor
        HND paged K pool, shape ``(num_pages, Hkv, num_tokens_per_page, D)``.
        Same dtype as ``q``. Every slot, including unused page padding, must
        contain a finite value. A plane view from a combined
        ``(num_pages, 2, Hkv, num_tokens_per_page, D)`` cache is accepted.
    v_pool : torch.Tensor
        HND paged V pool, shape ``(num_pages, Hkv, num_tokens_per_page, D)``.
        Same dtype and finite-value contract as ``k_pool``.
    o : torch.Tensor
        Output tensor, same shape as ``q``, written in-place.
        dtype: ``float16`` or ``bfloat16``.
    block_tables : torch.Tensor
        Shared K/V page index table, shape
        ``(B, max_num_pages_per_seq_kv)`` int32.
    seqlens_kv : torch.Tensor
        Actual K/V sequence length for each batch item, shape ``(B,)`` int32.
        Required for paged attention.
    cu_seqlens_q : torch.Tensor
        Runtime cumulative Q sequence offsets, shape ``(B+1,)`` int32.
    is_causal : bool
        Bottom-right aligned causal mask. Causal kernels use balanced
        scheduling unless ``PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1``.
    sm_scale : float, optional
        Softmax scale. Defaults to ``1 / sqrt(D)``.
    max_seqlen_q : int, optional
        Runtime launch-grid bound. Derived from ``cu_seqlens_q`` if omitted.
    kv_tile : int, optional
        K/V tile size (64 or 128).  Auto-selected if ``None``.
    q_tile : int, optional
        Q tile size (64 or 128).  Auto-selected if ``None``.
    lse : torch.Tensor, optional
        Preallocated float32 log2 LSE output, shape ``(total_q, Hq)``.
    v_scale : float, optional
        Scalar multiplier folded into the normalized output. Defaults to 1.
    enable_pdl : bool
        Whether to enable Programmatic Dependent Launch.
    skip_softmax_threshold : float or torch.Tensor, optional
        Skip a K/V tile when all rows owned by a compute warp satisfy
        ``exp((tile_max - running_max) * sm_scale) < threshold``. A Python
        float applies one static threshold to the whole batch. During CUDA
        Graph capture its value is fixed in the captured graph. A tensor is
        the dynamic form: it must be a contiguous CUDA float32 tensor on the
        same device as ``q`` with shape ``(batch_size,)``. The caller owns its
        allocation and lifetime, must keep its address stable, and may update
        values in place between graph replays. Tensor values are required to
        be finite and non-negative; this content contract is not checked to
        avoid a device-to-host synchronization. ``None`` selects the dense
        specialization; every supplied value, including zero, selects the
        skip-enabled specialization.
    kv_tile_ranges : torch.Tensor, optional
        Split KV: int32 ``(num_chunks * B, 2)`` K/V tile ranges ``[begin, end)``, row ``c * B + r`` for chunk
        ``c`` of request ``r``. ``o``/``lse`` then hold row ``p * num_chunks + c`` for chunk ``c`` of token
        ``p``, the layout ``merge_states`` reads.
    gqa_pack_size : int
        Q heads sharing one K/V head packed into each Q tile. Must divide ``Hq // Hkv`` and ``q_tile``.
    Raises
    ------
    RuntimeError
        If the GPU is not SM120 or the configuration is unsupported.
    ValueError
        If required arguments are missing or shapes are inconsistent.
    """
    _check_cutlass_dsl_version()

    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
        compile_sm120_fmha_fp8_paged_kernel,
    )

    _check_sm120(q.device)
    _validate_lse(o, lse)

    assert q.ndim == 3, f"q must be packed (total_q, Hq, D), got {q.shape}"
    _, Hq, D = q.shape

    _validate_hnd_paged_pool("k_pool", k_pool)
    _validate_hnd_paged_pool("v_pool", v_pool)
    if tuple(v_pool.shape) != tuple(k_pool.shape):
        raise ValueError(
            f"v_pool must have the same HND shape as k_pool, got "
            f"{v_pool.shape} and {k_pool.shape}"
        )
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("q, k_pool, and v_pool must have the same FP8 dtype")
    if k_pool.device != q.device or v_pool.device != q.device:
        raise ValueError("q, k_pool, and v_pool must be on the same device")
    _, Hkv, page_size, D_k = k_pool.shape
    assert D_k == D, f"head_dim mismatch: q={D}, k_pool={D_k}"

    assert block_tables.ndim == 2, (
        f"block_tables must be (B, max_pages), got {block_tables.shape}"
    )
    B = block_tables.shape[0]
    threshold = _prepare_skip_softmax_threshold(skip_softmax_threshold, q, B)
    num_chunks = 1 if kv_tile_ranges is None else kv_tile_ranges.shape[0] // B
    if kv_tile_ranges is not None and tuple(kv_tile_ranges.shape) != (
        num_chunks * B,
        2,
    ):
        raise ValueError(
            f"kv_tile_ranges must have shape (num_chunks * {B}, 2), got {tuple(kv_tile_ranges.shape)}"
        )
    if o.shape[0] != q.shape[0] * num_chunks:
        raise ValueError(
            f"o must have {num_chunks} row(s) per Q token, got {o.shape[0]} rows for {q.shape[0]} tokens"
        )

    in_ct = _cutlass_dtype(q.dtype)
    out_ct = _cutlass_dtype(o.dtype)

    seqlens_kv_i32 = seqlens_kv.to(torch.int32)
    cu_seqlens_q_i32 = cu_seqlens_q.to(torch.int32)
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q_i32[1:] - cu_seqlens_q_i32[:-1]).max().item())
    kv_tile = kv_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_KV_TILES[0]
    q_tile = q_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_Q_TILES[0]

    # Only structural properties are checked here. Runtime lengths are bounded
    # by cu_seqlens_q, seqlens_kv, and block_tables capacity.
    if (
        not SM120FusedMultiHeadAttentionFP8ForwardTMA.can_implement_paged(
            in_ct,
            out_ct,
            q_shape=(B, 1, Hq, D),
            k_shape=(B, 1, Hkv, D),
            num_tokens_per_page=page_size,
            kv_tile=kv_tile,
            q_tile=q_tile,
        )
        or (Hq // Hkv) % gqa_pack_size != 0
        or q_tile % gqa_pack_size != 0
    ):
        raise RuntimeError(
            f"SM120 FP8 paged FMHA cannot implement config: "
            f"q={q.shape} k_pool={k_pool.shape} in={q.dtype} out={o.dtype} "
            f"page_size={page_size} kv_tile={kv_tile} q_tile={q_tile} "
            f"gqa_pack_size={gqa_pack_size}"
        )

    kernel_fn = compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=q.dtype,
        out_dtype=o.dtype,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        is_causal=is_causal,
        kv_tile=kv_tile,
        q_tile=q_tile,
        num_tokens_per_page=page_size,
        device=q.device,
        with_lse=lse is not None,
        balanced_scheduler=_use_balanced_scheduler(is_causal),
        enable_skip_softmax=threshold is not None,
        **({"with_kv_tile_ranges": True} if kv_tile_ranges is not None else {}),
        **({"gqa_pack_size": gqa_pack_size} if gqa_pack_size > 1 else {}),
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scale_log2 = Float32(sm_scale * math.log2(math.e))
    output_scale = Float32(1.0 if v_scale is None else float(v_scale))

    # TVM-FFI ABI (env stream):
    # kernel_fn(
    #     q, k_pool, v_pool, o, lse, scale_log2, output_scale,
    #     seqlens_kv, cu_seqlens_q, block_tables, max_seqlen_q
    # )
    kernel_fn(
        q,
        k_pool,
        v_pool,
        o,
        lse,
        scale_log2,
        output_scale,
        threshold,
        seqlens_kv_i32,
        cu_seqlens_q_i32,
        block_tables,
        None,
        Int32(max_seqlen_q),
        enable_pdl,
        kv_tile_ranges,
    )


def _paged_kv_split_bounds(
    q_lens: list,
    kv_lens: list,
    num_qo_heads: int,
    causal: bool,
    tile: int,
    num_ctas_per_wave: int,
    fixed_split_tokens: int,
    max_chunks: int,
    gqa_pack_size: int = 1,
) -> Optional[list]:
    """K/V tile bounds per request and chunk, or ``None`` to run unsplit.

    With ``fixed_split_tokens``, each request is cut at multiples of that length within its own visible
    prefix, so its bounds do not depend on the rest of the batch. Otherwise the chunk count minimizes the
    estimated time in K/V tiles: the larger of the longest CTAs' waves and the total CTA work spread over
    ``num_ctas_per_wave`` SMs.
    """
    overhead = 4  # Fixed cost of one CTA (Q load, epilogue, merge share), in K/V tiles.
    kv_tiles = [ceil_div(kv, tile) for kv in kv_lens]
    visible = [
        (kv - q) // tile if causal else t
        for q, kv, t in zip(q_lens, kv_lens, kv_tiles, strict=True)
    ]
    if fixed_split_tokens > 0:
        step = ceil_div(fixed_split_tokens, tile)
        cuts = [
            list(range(step, min(v, t - 1) + 1, step))
            for t, v in zip(kv_tiles, visible, strict=True)
        ]
        num_chunks = max(len(c) for c in cuts) + 1
        if num_chunks < 2:
            return None
        if num_chunks > max_chunks:
            raise ValueError(
                f"fixed_split_size needs {num_chunks} K/V chunks, but the float workspace buffer holds "
                f"the partial outputs of only {max_chunks}"
            )
        # Requests with fewer chunks get empty [t, t) ranges, which add nothing in the merge.
        return [
            [0, *c, t] + [t] * (num_chunks - 1 - len(c))
            for c, t in zip(cuts, kv_tiles, strict=True)
        ]
    max_chunks = min([128, max_chunks] + [v + 1 for v in visible])
    if max_chunks < 2:
        return None
    q_tile_tokens = tile // gqa_pack_size
    ctas = [
        ceil_div(q, q_tile_tokens) * (num_qo_heads // gqa_pack_size) for q in q_lens
    ]
    longest_tiles = max(kv_tiles)
    longest_ctas = sum(
        c for c, t in zip(ctas, kv_tiles, strict=True) if t == longest_tiles
    )
    total_ctas = sum(ctas)
    total_work = sum(c * t for c, t in zip(ctas, kv_tiles, strict=True))

    def cost(n: int) -> float:
        waves = ceil_div(n * longest_ctas, num_ctas_per_wave)
        longest = waves * (ceil_div(longest_tiles, n) + overhead)
        return max(
            longest, (total_work + n * total_ctas * overhead) / num_ctas_per_wave
        )

    num_chunks = min(range(1, max_chunks + 1), key=cost)
    if num_chunks < 2:
        return None
    bounds = []
    for t, v in zip(kv_tiles, visible, strict=True):
        b = [0]
        for c in range(1, num_chunks):
            target = round(c * t / num_chunks)
            b.append(min(max(target, b[-1] + 1), v - (num_chunks - 1 - c)))
        b.append(t)
        bounds.append(b)
    return bounds


def _kv_split_scratch(
    total_q: int, heads: int, head_dim: int, o_dtype: torch.dtype, num_chunks: int
) -> tuple:
    """Shape, dtype and byte offset of the split's partial O, partial LSE and merged LSE in the float
    workspace, and the bytes they span."""
    rows = num_chunks * total_q
    layout, offset = [], 0
    for shape, dtype in (
        ((rows, heads, head_dim), o_dtype),
        ((rows, heads), torch.float32),
        ((total_q, heads), torch.float32),
    ):
        layout.append((shape, dtype, offset))
        offset += ceil_div(math.prod(shape) * dtype.itemsize, 256) * 256
    return layout, offset


def _gqa_pack_size(num_qo_heads: int, num_kv_heads: int, q_tile: int) -> int:
    """Largest power of two dividing both the GQA group and ``q_tile``."""
    group = num_qo_heads // num_kv_heads
    return min(group & -group, q_tile)


class SM120PrimsBatchPrefillBackend:
    """Plan/run adapter used by the public batch-prefill wrappers.

    Host-derived metadata and the base kernel are prepared in ``plan_*``.
    Runtime LSE specializations are compiled lazily and cached before dispatch.
    PDL is a runtime launch option shared by every compiled kernel.
    """

    _FP8_DTYPES = (torch.float8_e4m3fn,)
    _OUT_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self, device: torch.device) -> None:
        _check_cutlass_dsl_version()
        self.device = torch.device(device)
        self._split_buffers: dict = {}
        self._gqa_pack_size = 1
        self._mode: Optional[str] = None
        self._compiled_lse_variants: set[bool] = set()

    @staticmethod
    def _scalar_scale(name: str, value: Optional[float]) -> float:
        if value is None:
            return 1.0
        if isinstance(value, torch.Tensor):
            raise NotImplementedError(
                f"backend='cute-dsl-prims' only supports Python scalar {name}; "
                f"got tensor with shape {tuple(value.shape)}"
            )
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"backend='cute-dsl-prims' requires scalar {name}, got {value!r}"
            ) from exc

    def _validate_config(
        self,
        *,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
    ) -> None:
        _check_sm120(self.device)
        if q_dtype not in self._FP8_DTYPES or kv_dtype != q_dtype:
            raise ValueError(
                "backend='cute-dsl-prims' requires Q/K/V to have the same "
                f"FP8 dtype (float8_e4m3fn); got q={q_dtype}, "
                f"kv={kv_dtype}"
            )
        if o_dtype not in self._OUT_DTYPES:
            raise ValueError(
                "backend='cute-dsl-prims' requires output dtype float16 or "
                f"bfloat16; got {o_dtype}"
            )
        if head_dim_qk != head_dim_vo or head_dim_qk not in (32, 64, 128, 256):
            raise ValueError(
                "backend='cute-dsl-prims' requires equal QK/VO head dimensions "
                f"in {{32, 64, 128, 256}}; got {head_dim_qk}/{head_dim_vo}"
            )
        if num_kv_heads <= 0 or num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                "backend='cute-dsl-prims' requires num_qo_heads to be divisible "
                f"by num_kv_heads; got {num_qo_heads}/{num_kv_heads}"
            )

    def plan_ragged(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        causal: bool,
        sm_scale: Optional[float],
    ) -> None:
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_ragged_kernel,
        )

        self._validate_config(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
        )
        q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
        kv_lens = kv_indptr_host[1:] - kv_indptr_host[:-1]
        if causal and bool(torch.any(q_lens > kv_lens)):
            raise ValueError(
                "backend='cute-dsl-prims' causal attention requires q_len <= "
                "kv_len for every request"
            )
        self._mode = "ragged"
        self._qo_indptr = qo_indptr
        self._kv_indptr = kv_indptr
        self._max_seqlen_q = int(q_lens.max().item())
        self._q_dtype = q_dtype
        self._kv_dtype = kv_dtype
        self._o_dtype = o_dtype
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim_qk
        self._causal = causal
        self._sm_scale = sm_scale
        self._page_size: Optional[int] = None
        compile_sm120_fmha_fp8_ragged_kernel(
            in_dtype=q_dtype,
            out_dtype=o_dtype,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_qk,
            is_causal=causal,
            kv_tile=128,
            q_tile=128,
            device=self.device,
            with_lse=False,
            balanced_scheduler=_use_balanced_scheduler(causal),
            enable_skip_softmax=False,
        )
        self._compiled_lse_variants = {False}

    def plan_paged(
        self,
        *,
        qo_indptr: torch.Tensor,
        qo_indptr_host: torch.Tensor,
        seqlens_kv: torch.Tensor,
        seqlens_kv_host: torch.Tensor,
        block_tables: torch.Tensor,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool,
        sm_scale: Optional[float],
        float_workspace_buffer: Optional[torch.Tensor] = None,
        fixed_split_size: int = -1,
        disable_split_kv: bool = False,
        use_cuda_graph: bool = False,
    ) -> None:
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_paged_kernel,
        )

        self._validate_config(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
        )
        if page_size not in (16, 32, 64, 128):
            raise ValueError(
                "backend='cute-dsl-prims' supports page_size in "
                f"{{16, 32, 64, 128}}; got {page_size}"
            )
        q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
        if causal and bool(torch.any(q_lens > seqlens_kv_host)):
            raise ValueError(
                "backend='cute-dsl-prims' causal attention requires q_len <= "
                "kv_len for every request"
            )
        self._mode = "paged"
        self._qo_indptr = qo_indptr
        self._seqlens_kv = seqlens_kv
        self._block_tables = block_tables
        self._max_seqlen_q = int(q_lens.max().item())
        self._q_dtype = q_dtype
        self._kv_dtype = kv_dtype
        self._o_dtype = o_dtype
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim_qk
        self._causal = causal
        self._sm_scale = sm_scale
        self._page_size = page_size
        self._gqa_pack_size = _gqa_pack_size(num_qo_heads, num_kv_heads, 128)
        compile_sm120_fmha_fp8_paged_kernel(
            in_dtype=q_dtype,
            out_dtype=o_dtype,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim_qk,
            is_causal=causal,
            kv_tile=128,
            q_tile=128,
            num_tokens_per_page=page_size,
            device=self.device,
            with_lse=False,
            balanced_scheduler=_use_balanced_scheduler(causal),
            enable_skip_softmax=False,
            **self._pack_kwargs(),
        )
        self._compiled_lse_variants = {False}
        self._plan_kv_split(
            qo_indptr_host=qo_indptr_host,
            seqlens_kv_host=seqlens_kv_host,
            float_workspace_buffer=float_workspace_buffer,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            use_cuda_graph=use_cuda_graph,
        )

    def paged_workspace_size(
        self,
        *,
        qo_indptr_host: torch.Tensor,
        seqlens_kv_host: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        o_dtype: torch.dtype,
        page_size: int,
        causal: bool,
        fixed_split_size: int,
        disable_split_kv: bool,
        use_cuda_graph: bool,
    ) -> int:
        """Float workspace bytes the paged plan's K/V split needs; 0 when it does not split."""
        if disable_split_kv or use_cuda_graph or head_dim < 64:
            return 0
        bounds = _paged_kv_split_bounds(
            (qo_indptr_host[1:] - qo_indptr_host[:-1]).tolist(),
            seqlens_kv_host.tolist(),
            num_qo_heads,
            causal,
            128,
            torch.cuda.get_device_properties(self.device).multi_processor_count,
            fixed_split_size * page_size if fixed_split_size > 0 else 0,
            1 << 30,
            _gqa_pack_size(num_qo_heads, num_kv_heads, 128),
        )
        if bounds is None:
            return 0
        total_q = int(qo_indptr_host[-1].item())
        return _kv_split_scratch(
            total_q, num_qo_heads, head_dim, o_dtype, len(bounds[0]) - 1
        )[1]

    def _plan_kv_split(
        self,
        *,
        qo_indptr_host: torch.Tensor,
        seqlens_kv_host: torch.Tensor,
        float_workspace_buffer: Optional[torch.Tensor],
        fixed_split_size: int,
        disable_split_kv: bool,
        use_cuda_graph: bool,
    ) -> None:
        """Choose the K/V split for the paged plan; its partial O/LSE live in the float workspace."""
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_paged_kernel,
        )

        self._kv_split = None
        # merge_states needs head_dim >= 64.
        if (
            disable_split_kv
            or use_cuda_graph
            or self._head_dim < 64
            or float_workspace_buffer is None
        ):
            return
        tile = 128
        q_lens = (qo_indptr_host[1:] - qo_indptr_host[:-1]).tolist()
        kv_lens = seqlens_kv_host.tolist()
        total_q = int(qo_indptr_host[-1].item())
        heads, dim = self._num_qo_heads, self._head_dim
        # Per chunk: one partial O row and one LSE row per Q token; plus the merged LSE.
        chunk_bytes = total_q * heads * (dim * self._o_dtype.itemsize + 4)
        max_chunks = max(
            (float_workspace_buffer.nbytes - total_q * heads * 4)
            // max(chunk_bytes, 1),
            1,
        )
        while (
            max_chunks > 1
            and _kv_split_scratch(total_q, heads, dim, self._o_dtype, max_chunks)[1]
            > float_workspace_buffer.nbytes
        ):
            max_chunks -= 1
        bounds = _paged_kv_split_bounds(
            q_lens,
            kv_lens,
            self._num_qo_heads,
            self._causal,
            tile,
            torch.cuda.get_device_properties(self.device).multi_processor_count,
            fixed_split_size * self._page_size if fixed_split_size > 0 else 0,
            max_chunks,
            self._gqa_pack_size,
        )
        if bounds is None:
            return
        num_chunks = len(bounds[0]) - 1
        batch_size = len(q_lens)
        ranges = torch.tensor(
            [
                [bounds[r][c], bounds[r][c + 1]]
                for c in range(num_chunks)
                for r in range(batch_size)
            ],
            dtype=torch.int32,
            pin_memory=True,
        )
        kv_tile_ranges = self._split_buffer(
            "kv_tile_ranges", tuple(ranges.shape), torch.int32
        )
        kv_tile_ranges.copy_(ranges, non_blocking=True)
        workspace = float_workspace_buffer.reshape(-1).view(torch.uint8)
        layout, _ = _kv_split_scratch(total_q, heads, dim, self._o_dtype, num_chunks)
        scratch = [
            workspace[offset : offset + math.prod(shape) * dtype.itemsize]
            .view(dtype)
            .view(shape)
            for shape, dtype, offset in layout
        ]
        self._kv_split = SimpleNamespace(
            num_chunks=num_chunks,
            total_q=total_q,
            kv_tile_ranges=kv_tile_ranges,
            o=scratch[0],
            lse=scratch[1],
            merged_lse=scratch[2],
        )
        compile_sm120_fmha_fp8_paged_kernel(
            in_dtype=self._q_dtype,
            out_dtype=self._o_dtype,
            num_qo_heads=self._num_qo_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            is_causal=self._causal,
            kv_tile=tile,
            q_tile=128,
            num_tokens_per_page=self._page_size,
            device=self.device,
            with_lse=True,
            balanced_scheduler=_use_balanced_scheduler(self._causal),
            enable_skip_softmax=False,
            with_kv_tile_ranges=True,
            **self._pack_kwargs(),
        )

    def _pack_kwargs(self) -> dict:
        # Keep the default kernel's compile cache key when not packing.
        return {"gqa_pack_size": self._gqa_pack_size} if self._gqa_pack_size > 1 else {}

    def _split_buffer(
        self, name: str, shape: tuple, dtype: torch.dtype
    ) -> torch.Tensor:
        numel = math.prod(shape)
        buf = self._split_buffers.get(name)
        if buf is None or buf.dtype != dtype or buf.numel() < numel:
            buf = torch.empty(numel, dtype=dtype, device=self.device)
            self._split_buffers[name] = buf
        return buf[:numel].view(shape)

    def _validate_run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        if self._mode is None:
            raise RuntimeError("backend='cute-dsl-prims' must be planned before run")
        for name, tensor, dtype in (
            ("q", q, self._q_dtype),
            ("k", k, self._kv_dtype),
            ("v", v, self._kv_dtype),
            ("out", out, self._o_dtype),
        ):
            if tensor.device != self.device or tensor.dtype != dtype:
                raise ValueError(
                    f"backend='cute-dsl-prims' expected {name} on {self.device} "
                    f"with dtype {dtype}; got {tensor.device}/{tensor.dtype}"
                )
            if self._mode == "paged" and name in ("k", "v"):
                _validate_hnd_paged_pool(name, tensor)
            elif not tensor.is_contiguous():
                raise ValueError(f"backend='cute-dsl-prims' requires contiguous {name}")
        if self._mode == "paged":
            expected_pool_shape = (
                self._num_kv_heads,
                self._page_size,
                self._head_dim,
            )
            if tuple(k.shape[1:]) != expected_pool_shape or v.shape != k.shape:
                raise ValueError(
                    "backend='cute-dsl-prims' expected HND K/V pools with "
                    f"shape [num_pages, {self._num_kv_heads}, "
                    f"{self._page_size}, {self._head_dim}]; got "
                    f"{tuple(k.shape)} and {tuple(v.shape)}"
                )

    def _ensure_kernel(self, *, with_lse: bool) -> None:
        if with_lse in self._compiled_lse_variants:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "backend='cute-dsl-prims' kernel specialization "
                f"(return_lse={with_lse}) was not compiled before CUDA Graph "
                "capture; warm up the same run configuration before capture"
            )
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_paged_kernel,
            compile_sm120_fmha_fp8_ragged_kernel,
        )

        if self._mode == "ragged":
            compile_sm120_fmha_fp8_ragged_kernel(
                in_dtype=self._q_dtype,
                out_dtype=self._o_dtype,
                num_qo_heads=self._num_qo_heads,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                is_causal=self._causal,
                kv_tile=128,
                q_tile=128,
                device=self.device,
                with_lse=with_lse,
                balanced_scheduler=_use_balanced_scheduler(self._causal),
                enable_skip_softmax=False,
            )
        else:
            compile_sm120_fmha_fp8_paged_kernel(
                in_dtype=self._q_dtype,
                out_dtype=self._o_dtype,
                num_qo_heads=self._num_qo_heads,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                is_causal=self._causal,
                kv_tile=128,
                q_tile=128,
                num_tokens_per_page=self._page_size,
                device=self.device,
                with_lse=with_lse,
                balanced_scheduler=_use_balanced_scheduler(self._causal),
                enable_skip_softmax=False,
                **self._pack_kwargs(),
            )
        self._compiled_lse_variants.add(with_lse)

    def run_ragged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        lse: Optional[torch.Tensor],
        enable_pdl: bool,
        q_scale: Optional[float],
        k_scale: Optional[float],
        v_scale: Optional[float],
    ) -> None:
        self._validate_run(q, k, v, out)
        if self._mode != "ragged":
            raise RuntimeError("SM120 PRIMS backend was not planned for ragged KV")
        self._ensure_kernel(with_lse=lse is not None)
        sm_scale = (
            self._sm_scale
            if self._sm_scale is not None
            else 1.0 / math.sqrt(self._head_dim)
        )
        sm_scale *= self._scalar_scale("q_scale", q_scale)
        sm_scale *= self._scalar_scale("k_scale", k_scale)
        scale_v = self._scalar_scale("v_scale", v_scale)
        sm120_fmha_fp8_ragged_prefill(
            q,
            k,
            v,
            out,
            self._qo_indptr,
            self._kv_indptr,
            max_seqlen_q=self._max_seqlen_q,
            is_causal=self._causal,
            sm_scale=sm_scale,
            v_scale=scale_v,
            lse=lse,
            enable_pdl=enable_pdl,
        )

    def run_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        return_lse: bool,
        lse: Optional[torch.Tensor],
        enable_pdl: bool,
        q_scale: Optional[float],
        k_scale: Optional[float],
        v_scale: Optional[float],
    ):
        self._validate_run(q, k, v, out)
        if self._mode != "paged":
            raise RuntimeError("SM120 PRIMS backend was not planned for paged KV")
        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)),
                    dtype=torch.float32,
                    device=q.device,
                )
        else:
            lse = None
        self._ensure_kernel(with_lse=return_lse)
        sm_scale = (
            self._sm_scale
            if self._sm_scale is not None
            else 1.0 / math.sqrt(self._head_dim)
        )
        sm_scale *= self._scalar_scale("q_scale", q_scale)
        sm_scale *= self._scalar_scale("k_scale", k_scale)
        scale_v = self._scalar_scale("v_scale", v_scale)
        split = self._kv_split
        if split is not None:
            from flashinfer.cascade import get_cascade_module

            sm120_fmha_fp8_paged_prefill(
                q,
                k,
                v,
                split.o,
                self._block_tables,
                self._seqlens_kv,
                self._qo_indptr,
                is_causal=self._causal,
                sm_scale=sm_scale,
                v_scale=scale_v,
                max_seqlen_q=self._max_seqlen_q,
                lse=split.lse,
                enable_pdl=enable_pdl,
                kv_tile_ranges=split.kv_tile_ranges,
                gqa_pack_size=self._gqa_pack_size,
            )
            rows, n = split.total_q, split.num_chunks
            get_cascade_module().merge_states(
                split.o.view(rows, n, *out.shape[1:]),
                split.lse.view(rows, n, -1),
                out,
                lse if return_lse else split.merged_lse,
            )
            if return_lse:
                return out, lse
            return out
        sm120_fmha_fp8_paged_prefill(
            q,
            k,
            v,
            out,
            self._block_tables,
            self._seqlens_kv,
            self._qo_indptr,
            is_causal=self._causal,
            sm_scale=sm_scale,
            v_scale=scale_v,
            max_seqlen_q=self._max_seqlen_q,
            lse=lse,
            enable_pdl=enable_pdl,
            gqa_pack_size=self._gqa_pack_size,
        )
        if return_lse:
            return out, lse
        return out
