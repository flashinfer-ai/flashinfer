# Copyright (c) 2026 by FlashInfer team.
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

"""NVFP4 cache primitives for DeepSeek-V4 sparse MLA on SM120."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import torch

from ..api_logging import flashinfer_api
from ..jit.mla import (
    gen_sparse_mla_nvfp4_sm120_module,
    gen_sparse_mla_nvfp4_sm120_tile_module,
)
from ..utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)


_D_NOPE = 448
_D_ROPE = 64
_D_LATENT = _D_NOPE + _D_ROPE
_PACKED_NOPE_BYTES = _D_NOPE // 2
_ROPE_BYTES = _D_ROPE * 2
_DATA_BYTES_PER_TOKEN = _PACKED_NOPE_BYTES + _ROPE_BYTES
_SCALE_BYTES_PER_TOKEN = 32
_BYTES_PER_TOKEN = _DATA_BYTES_PER_TOKEN + _SCALE_BYTES_PER_TOKEN


@functools.cache
def get_sparse_mla_nvfp4_sm120_module():
    module = gen_sparse_mla_nvfp4_sm120_module().build_and_load()

    @register_custom_op(
        "flashinfer::sparse_mla_nvfp4_sm120_paged_attention",
        mutates_args=(
            "mid_out",
            "mid_lse",
            "output",
            "out_lse",
        ),
    )
    def _paged_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        mid_out: torch.Tensor | None,
        mid_lse: torch.Tensor | None,
        output: torch.Tensor,
        out_lse: torch.Tensor,
        sm_scale: float,
        topk_length: torch.Tensor | None,
        attn_sink: torch.Tensor | None,
        extra_kv_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_topk_length: torch.Tensor | None,
        use_prefill: bool,
        chunks_per_block_override: int,
    ) -> None:
        if not use_prefill:
            if mid_out is None or mid_lse is None:
                raise ValueError("NVFP4 decode requires mid_out and mid_lse workspace")
            num_splits = (indices.shape[1] + 63) // 64
            if extra_indices is not None:
                num_splits += (extra_indices.shape[1] + 63) // 64
            module.sparse_mla_sm120_nvfp4_decode(
                q,
                kv_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                num_splits,
                sm_scale,
                topk_length,
                attn_sink,
                extra_kv_cache,
                extra_indices,
                extra_topk_length,
                chunks_per_block_override,
                False,
            )
        else:
            module.sparse_mla_sm120_nvfp4_prefill(
                q,
                kv_cache,
                indices,
                output,
                out_lse,
                sm_scale,
                topk_length,
                attn_sink,
                extra_kv_cache,
                extra_indices,
                extra_topk_length,
            )

    @register_fake_op("flashinfer::sparse_mla_nvfp4_sm120_paged_attention")
    def _fake_paged_attention(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(
        paged_attention=_paged_attention,
        sparse_mla_sm120_nvfp4_quantize_pack=module.sparse_mla_sm120_nvfp4_quantize_pack,
        sparse_mla_sm120_nvfp4_quantize_append=module.sparse_mla_sm120_nvfp4_quantize_append,
        sparse_mla_sm120_nvfp4_decode=module.sparse_mla_sm120_nvfp4_decode,
        sparse_mla_sm120_nvfp4_prefill=module.sparse_mla_sm120_nvfp4_prefill,
    )


@functools.cache
def get_sparse_mla_nvfp4_sm120_tile_module():
    """Build the test-only MMA-layout validation module."""
    return gen_sparse_mla_nvfp4_sm120_tile_module().build_and_load()


@supported_compute_capability([120, 121])
def _sparse_mla_nvfp4_sm120_paged_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    use_prefill: bool,
    chunks_per_block_override: int = 0,
) -> None:
    """Run the allocation-free NVFP4 sparse-MLA custom op."""
    get_sparse_mla_nvfp4_sm120_module().paged_attention(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        use_prefill,
        chunks_per_block_override,
    )


def _check_latent_kv(latent_kv: torch.Tensor, *, expected_rows: int | None) -> int:
    if not latent_kv.is_cuda:
        raise ValueError(f"latent_kv must be a CUDA tensor, got {latent_kv.device}")
    if latent_kv.dtype != torch.bfloat16:
        raise ValueError(
            f"DeepSeek-V4 latent_kv must have dtype torch.bfloat16, got {latent_kv.dtype}"
        )
    if latent_kv.ndim not in (2, 3, 4):
        raise ValueError(
            f"latent_kv must be 2D, 3D, or 4D, got shape={tuple(latent_kv.shape)}"
        )
    if latent_kv.shape[-1] != _D_LATENT:
        raise ValueError(
            f"latent_kv last dimension must be {_D_LATENT}, got {latent_kv.shape[-1]}"
        )
    if not latent_kv.is_contiguous():
        raise ValueError("latent_kv must be contiguous")
    rows = latent_kv.numel() // _D_LATENT
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"latent_kv contains {rows} rows, expected {expected_rows} rows"
        )
    return rows


def _cache_shape(cache: torch.Tensor) -> tuple[int, int, str]:
    if not cache.is_cuda:
        raise ValueError(f"cache must be a CUDA tensor, got {cache.device}")
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must have dtype torch.uint8, got {cache.dtype}")
    if cache.ndim not in (3, 4) or cache.shape[-1] != _BYTES_PER_TOKEN:
        raise ValueError(
            "cache must be [num_pages, page_size, 384], HND "
            "[num_pages, 1, page_size, 384], or NHD "
            f"[num_pages, page_size, 1, 384], got shape={tuple(cache.shape)}"
        )
    if cache.ndim == 3:
        num_pages, page_size, layout = cache.shape[0], cache.shape[1], "NHD"
    elif cache.shape[1] == 1:
        num_pages, page_size, layout = cache.shape[0], cache.shape[2], "HND"
    elif cache.shape[2] == 1:
        num_pages, page_size, layout = cache.shape[0], cache.shape[1], "NHD"
    else:
        raise ValueError(
            "cache must have a singleton latent-head dimension at axis 1 or 2"
        )
    page_dim = 1 if cache.ndim == 3 or layout == "NHD" else 2
    if cache.stride(-1) != 1 or cache.stride(page_dim) != _BYTES_PER_TOKEN:
        raise ValueError(
            "cache entries must be contiguous inside each page with strides "
            f"(..., {_BYTES_PER_TOKEN}, 1), got {cache.stride()}"
        )
    if cache.stride(0) < page_size * _BYTES_PER_TOKEN:
        raise ValueError(
            "cache page stride must cover the logical page payload, got "
            f"stride(0)={cache.stride(0)} for page_size={page_size}"
        )
    return int(num_pages), int(page_size), layout


def _validate_unique_slots(slot_mapping: torch.Tensor, capacity: int) -> None:
    """Reject write races when defensive MLA input checks are enabled."""
    from ._core import _validate_dsv4_sync_checks

    if slot_mapping.numel() < 2 or not _validate_dsv4_sync_checks(slot_mapping.device):
        return
    ordered = torch.sort(slot_mapping).values
    duplicate_valid = (
        (ordered[1:] == ordered[:-1]) & (ordered[1:] >= 0) & (ordered[1:] < capacity)
    )
    if duplicate_valid.any().item():
        raise ValueError("valid slot_mapping entries must be unique")


@supported_compute_capability([120, 121])
@flashinfer_api
def nvfp4_quantize_pack_sparse_mla_cache(
    latent_kv: torch.Tensor,
    *,
    kv_layout: str = "HND",
) -> torch.Tensor:
    r"""Quantize complete DeepSeek-V4 latent-KV pages to the NVFP4 cache ABI.

    Parameters
    ----------
    latent_kv : torch.Tensor
        Contiguous CUDA BF16 tensor with shape
        ``[num_pages, page_size, 512]``. A singleton latent-head axis is also
        accepted in HND or NHD position. The first 448 values are quantized in
        groups of 16 to packed E2M1 with E4M3 scales; the final 64 BF16 RoPE
        values are copied bit-for-bit.
    kv_layout : str
        Output layout, either ``"HND"`` or ``"NHD"``.

    Returns
    -------
    torch.Tensor
        Opaque uint8 paged cache with logical shape
        ``[num_pages, 1, page_size, 384]`` for HND or
        ``[num_pages, page_size, 1, 384]`` for NHD. Within each physical page
        it stores ``page_size * 352`` data bytes followed by
        ``page_size * 32`` scale bytes. Consumers must not interpret the last
        dimension as a contiguous per-token record.
    """
    if kv_layout not in ("HND", "NHD"):
        raise ValueError(f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}")
    _check_latent_kv(latent_kv, expected_rows=None)

    if latent_kv.ndim == 2:
        raise ValueError(
            "full-page pack requires a page dimension; use shape "
            "[num_pages, page_size, 512]"
        )
    if latent_kv.ndim == 3:
        num_pages, page_size = latent_kv.shape[:2]
    elif latent_kv.shape[1] == 1:
        num_pages, page_size = latent_kv.shape[0], latent_kv.shape[2]
    elif latent_kv.shape[2] == 1:
        num_pages, page_size = latent_kv.shape[0], latent_kv.shape[1]
    else:
        raise ValueError(
            "4D latent_kv must have a singleton latent-head dimension at axis 1 or 2"
        )

    _check_latent_kv(latent_kv, expected_rows=int(num_pages) * int(page_size))
    cache_shape = (
        (num_pages, 1, page_size, _BYTES_PER_TOKEN)
        if kv_layout == "HND"
        else (num_pages, page_size, 1, _BYTES_PER_TOKEN)
    )
    cache = torch.empty(cache_shape, dtype=torch.uint8, device=latent_kv.device)
    if int(num_pages) == 0 or int(page_size) == 0:
        return cache
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_quantize_pack(
        latent_kv, cache
    )
    return cache


@supported_compute_capability([120, 121])
@flashinfer_api
def nvfp4_quantize_append_sparse_mla_cache(
    latent_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    r"""Quantize and append DeepSeek-V4 latent KV by physical cache slot.

    Parameters
    ----------
    latent_kv : torch.Tensor
        Contiguous CUDA BF16 tensor with one 512-element latent-KV row per
        entry in ``slot_mapping``.
    slot_mapping : torch.Tensor
        Contiguous 1D CUDA int32 or int64 tensor. ``slot_mapping[i]`` is
        ``page_id * page_size + entry_id``. Negative and out-of-range slots
        are padding and are ignored. Valid slots must be unique; set
        ``FLASHINFER_VALIDATE_INPUTS=1`` to check that invariant eagerly
        outside CUDA Graph capture.
    cache : torch.Tensor
        Destination opaque uint8 paged cache. The 3D shorthand
        ``[num_pages, page_size, 384]`` and public 4D HND/NHD layouts are
        accepted. Page-strided views may place the cache inside vLLM's packed
        physical block allocation. Only addressed data rows and scale slots
        are written, so prefill history is reused directly by decode.
    """
    num_pages, page_size, _ = _cache_shape(cache)
    if not slot_mapping.is_cuda:
        raise ValueError(
            f"slot_mapping must be a CUDA tensor, got {slot_mapping.device}"
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"slot_mapping must have dtype torch.int32 or torch.int64, got {slot_mapping.dtype}"
        )
    if slot_mapping.ndim != 1 or not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be a contiguous 1D tensor")
    if latent_kv.device != cache.device or slot_mapping.device != cache.device:
        raise ValueError(
            "latent_kv, slot_mapping, and cache must be on the same device"
        )
    _check_latent_kv(latent_kv, expected_rows=slot_mapping.numel())
    if num_pages * page_size == 0 and slot_mapping.numel() != 0:
        raise ValueError("cannot append to an empty cache")
    _validate_unique_slots(slot_mapping, num_pages * page_size)

    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_quantize_append(
        latent_kv, slot_mapping, cache
    )


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_m16n32k64(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    iterations: int = 1,
) -> torch.Tensor:
    """Run the internal M16N32K64 NVFP4 validation tile."""
    output = torch.empty((16, 32), dtype=torch.float32, device=a.device)
    get_sparse_mla_nvfp4_sm120_tile_module().sparse_mla_sm120_nvfp4_m16n32k64(
        a, b, sfa, sfb, output, iterations
    )
    return output


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_m16n8k64_candidate_major(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    iterations: int = 1,
) -> torch.Tensor:
    """Run the internal candidate-major PV validation tile."""
    output = torch.empty((16, 8), dtype=torch.float32, device=a.device)
    get_sparse_mla_nvfp4_sm120_tile_module().sparse_mla_sm120_nvfp4_m16n8k64_candidate_major(
        a, b, sfa, sfb, output, iterations
    )
    return output


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    chunks_per_block_override: int = 0,
    stage1_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the internal DeepSeek-V4 NVFP4 sparse-MLA decode prototype."""
    if q.ndim != 3 or q.shape[-1] != _D_LATENT:
        raise ValueError(f"q must be [T, H, {_D_LATENT}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("indices must be a 2D int32 tensor")
    num_tokens, num_heads, _ = q.shape
    topk = indices.shape[1]
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    extra_topk = extra_indices.shape[1] if extra_indices is not None else 0
    num_splits = (topk + 63) // 64 + (extra_topk + 63) // 64
    mid_out = torch.empty(
        (num_tokens, num_heads, num_splits, _D_LATENT),
        dtype=torch.bfloat16,
        device=q.device,
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, num_splits),
        dtype=torch.float32,
        device=q.device,
    )
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_decode(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        num_splits,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        chunks_per_block_override,
        stage1_only,
    )
    return output, out_lse


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the single-launch streaming DeepSeek-V4 NVFP4 prefill kernel."""
    if q.ndim != 3 or q.shape[-1] != _D_LATENT:
        raise ValueError(f"q must be [T, H, {_D_LATENT}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("indices must be a 2D int32 tensor")
    num_tokens, num_heads, _ = q.shape
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_prefill(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
    )
    return output, out_lse


__all__ = [
    "nvfp4_quantize_append_sparse_mla_cache",
    "nvfp4_quantize_pack_sparse_mla_cache",
]
