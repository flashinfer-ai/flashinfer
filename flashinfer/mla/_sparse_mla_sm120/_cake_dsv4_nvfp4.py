# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake SM120 DeepSeek-V4 NVFP4 sparse-MLA decode (``backend="cake"`` on SM120/SM121).

The device code is generated from the Cake kernel schedules into
``csrc/cake_dsv4/sm_120a``; this module owns the host side: cache geometry
(runtime page size and page stride for the 3-D / HND / NHD layouts, padded
pools), the split planner, caller-owned split scratch, the public-API
workspace carve and the ``SparseMLASm120Wrapper`` route.

Contract (shared with the SM120 NVFP4 ``"sparse"`` backend):

* ``q`` ``[T, H, 512]`` BF16; ``output`` ``[T, H, 512]`` BF16; ``out_lse``
  ``[T, H]`` fp32 base-2 (times ``lse_scale``).
* ``kv_cache`` / ``extra_kv_cache``: packed NVFP4 pages (``page_size * 352``
  data bytes followed by ``page_size * 32`` scale bytes) as ``[P, page, 384]``,
  HND ``[P, 1, page, 384]`` or NHD ``[P, page, 1, 384]`` uint8 views; any
  positive page size; the page stride may exceed the payload (16-byte multiple).
* ``indices`` / ``extra_indices`` ``[T, topk]`` (or ``[T, 1, topk]``) int32,
  ``-1`` masks a slot; ``topk_length`` / ``extra_topk_length`` ``[T]`` int32;
  ``attn_sink`` ``[H]`` fp32 (sigmoid gate of the output, logaddexp into LSE).
* An empty row (no valid slot in either cache) writes zeros and ``-inf`` LSE,
  or the sink-only LSE when ``attn_sink`` is given.
* Split-K: ``num_splits`` CTAs per (token, 16-head block) write BF16 /
  fp32 partials to caller-owned ``mid_out`` ``[T, H, S, 512]`` /
  ``mid_lse`` ``[T, H, S]`` (``S >= num_splits``); a merge launch combines
  them.  ``num_splits == 1`` writes the output directly.  Allocation-free
  and CUDA-graph safe when the scratch is caller-owned.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from ...jit.cake_sparse_mla_sm120_dsv4_nvfp4 import (
    cake_sparse_mla_sm120_dsv4_nvfp4_manifest,
    gen_cake_sparse_mla_sm120_dsv4_nvfp4_module,
)
from ...utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

_D_QK = 512
_D_V = 512
_BYTES_PER_TOKEN = 384
_CHUNK = 64


def _manifest() -> dict:
    return cake_sparse_mla_sm120_dsv4_nvfp4_manifest()


def cake_sparse_mla_sm120_dsv4_nvfp4_format_info() -> dict:
    """Static facts of the Cake SM120 NVFP4 route (mirrors ``dsv4_nvfp4_format_info``)."""

    manifest = _manifest()
    return {
        "query_dim": int(manifest["head_dim"]),
        "value_dim": int(manifest["value_dim"]),
        "bytes_per_token": int(manifest["bytes_per_token"]),
        "chunk_width": int(manifest["candidates_per_chunk"]),
        "heads_per_block": int(manifest["heads_per_block"]),
        "max_chunks_per_block": int(manifest["max_chunks_per_block"]),
        "heads": tuple(int(h) for h in manifest["head_counts"]),
        "runtime_page": True,
        "runtime_extra_page": True,
        "kernel_commit": manifest["kernel_commit"],
    }


def cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads() -> Tuple[int, ...]:
    return tuple(int(h) for h in _manifest()["head_counts"])


def cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk: int, extra_topk: int = 0) -> int:
    """64-candidate chunks of one token: the upper bound on ``num_splits``."""

    return (int(topk) + _CHUNK - 1) // _CHUNK + (int(extra_topk) + _CHUNK - 1) // _CHUNK


def cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int = 0,
    num_sms: int,
    max_splits: int = 16,
) -> Tuple[int, int]:
    """Return ``(num_splits, chunks_per_block)`` for one decode call.

    Measured split rules (RTX PRO 6000 Blackwell / RTX 5090):

    * two chunks never split: one CTA pipelining both chunks beats two CTAs
      plus a merge;
    * while the unsplit grid covers at most 40 % of the SMs, take the largest
      power-of-two split that keeps the grid within 80 % of the SMs and leaves
      at least two chunks per CTA (one chunk per CTA only pays off for a lone
      CTA);
    * once the unsplit grid fills the SMs, split in two only for 16-chunk work
      whose doubled grid ends in a wave at least half full;
    * everything else runs unsplit.

    The CTA index table holds at most ``max_chunks_per_block`` chunks, so
    longer candidate lists always split.
    """

    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    hpb = info["heads_per_block"]
    max_cpb = info["max_chunks_per_block"]
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    if chunks < 1:
        raise ValueError("topk must be at least 1")
    head_blocks = (int(num_heads) + hpb - 1) // hpb
    base_ctas = int(num_tokens) * head_blocks
    splits = 1
    if chunks > 2 and max_splits > 1:
        grid_cap = int(num_sms) * 4 // 5
        if base_ctas * 2 <= grid_cap:
            min_cpb = 1 if base_ctas == 1 else 2
            want = 1
            while (
                want * 2 <= max_splits
                and base_ctas * want * 2 <= grid_cap
                and -(-chunks // (want * 2)) >= min_cpb
            ):
                want *= 2
            splits = want
        elif base_ctas >= num_sms and chunks >= 16:
            tail = (base_ctas * 2) % int(num_sms)
            if tail == 0 or tail * 2 >= int(num_sms):
                splits = 2
    splits = max(splits, -(-chunks // max_cpb))
    cpb = -(-chunks // splits)
    splits = -(-chunks // cpb)
    return splits, cpb


def _resolve_splits(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    device: torch.device,
    num_splits: Optional[int],
    max_splits: int,
) -> Tuple[int, int]:
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    if num_splits is None:
        return cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits(
            num_tokens=num_tokens,
            num_heads=num_heads,
            topk=topk,
            extra_topk=extra_topk,
            num_sms=_num_sms(device),
            max_splits=max_splits,
        )
    num_splits = int(num_splits)
    if num_splits < 1:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    cpb = -(-chunks // num_splits)
    max_cpb = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()["max_chunks_per_block"]
    if cpb > max_cpb:
        raise ValueError(
            f"num_splits={num_splits} leaves {cpb} chunks per block; the kernel holds at most {max_cpb}"
        )
    return -(-chunks // cpb), cpb


def cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0
) -> int:
    """Workspace bytes that cover every split plan of this shape (partials + LSE + alignment slack)."""

    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    rows = int(num_tokens) * int(num_heads)
    return rows * chunks * (_D_V * 2 + 4) + rows * 4 + 3 * 16


@functools.cache
def _num_sms(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _cache_geometry(cache: torch.Tensor, name: str) -> Tuple[torch.Tensor, int, int]:
    """Flatten a paged NVFP4 cache view to ``(flat uint8 storage span, page_size, page_stride_bytes)``."""

    if not cache.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {cache.device}")
    if cache.dtype != torch.uint8:
        raise ValueError(f"{name} must have dtype torch.uint8, got {cache.dtype}")
    if cache.ndim not in (3, 4) or cache.shape[-1] != _BYTES_PER_TOKEN:
        raise ValueError(
            f"{name} must be [num_pages, page_size, {_BYTES_PER_TOKEN}], HND "
            f"[num_pages, 1, page_size, {_BYTES_PER_TOKEN}] or NHD "
            f"[num_pages, page_size, 1, {_BYTES_PER_TOKEN}], got shape={tuple(cache.shape)}"
        )
    if cache.ndim == 3:
        page_dim = 1
    elif cache.shape[1] == 1:
        page_dim = 2
    elif cache.shape[2] == 1:
        page_dim = 1
    else:
        raise ValueError(
            f"{name} must have a singleton latent-head dimension at axis 1 or 2"
        )
    num_pages = int(cache.shape[0])
    page_size = int(cache.shape[page_dim])
    if num_pages < 1 or page_size < 1:
        raise ValueError(f"{name} must hold at least one page with at least one row")
    if cache.stride(-1) != 1 or cache.stride(page_dim) != _BYTES_PER_TOKEN:
        raise ValueError(
            f"{name} entries must be contiguous inside each page with strides "
            f"(..., {_BYTES_PER_TOKEN}, 1), got {cache.stride()}"
        )
    page_stride = int(cache.stride(0))
    if page_stride < page_size * _BYTES_PER_TOKEN:
        raise ValueError(
            f"{name} page stride {page_stride} is smaller than the logical "
            f"{page_size * _BYTES_PER_TOKEN}-byte page"
        )
    if page_stride % 16:
        raise ValueError(
            f"{name} page stride must be a multiple of 16 bytes, got {page_stride}"
        )
    if cache.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    span = (num_pages - 1) * page_stride + page_size * _BYTES_PER_TOKEN
    flat = cache.as_strided((span,), (1,), cache.storage_offset())
    return flat, page_size, page_stride


def _normalize_indices(
    indices: torch.Tensor, name: str, num_tokens: int
) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices.squeeze(1)
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError(
            f"{name} must be a [num_tokens, topk] or [num_tokens, 1, topk] int32 tensor"
        )
    if indices.shape[0] != num_tokens:
        raise ValueError(f"{name} must have {num_tokens} rows, got {indices.shape[0]}")
    if indices.shape[1] < 1:
        raise ValueError(f"{name} must select at least one slot per token")
    return indices.contiguous()


def _normalize_length(
    length: Optional[torch.Tensor], name: str, num_tokens: int
) -> Optional[torch.Tensor]:
    if length is None:
        return None
    if length.ndim != 1 or length.dtype != torch.int32 or length.shape[0] < num_tokens:
        raise ValueError(
            f"{name} must be a 1-D int32 tensor with at least {num_tokens} entries"
        )
    return length.contiguous()


@functools.cache
def get_cake_sparse_mla_sm120_dsv4_nvfp4_module():
    module = gen_cake_sparse_mla_sm120_dsv4_nvfp4_module().build_and_load()
    entry = getattr(module, _manifest()["entry"])

    @register_custom_op(
        "flashinfer::cake_sparse_mla_sm120_dsv4_nvfp4_decode",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )
    def _decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        output: torch.Tensor,
        out_lse: torch.Tensor,
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
        page_size: int,
        page_stride_bytes: int,
        extra_page_size: int,
        extra_page_stride_bytes: int,
        num_splits: int,
        chunks_per_block: int,
        sm_scale: float,
        lse_scale: float,
    ) -> None:
        entry(
            q,
            kv_cache,
            indices,
            extra_kv_cache,
            extra_indices,
            topk_length,
            extra_topk_length,
            attn_sink,
            output,
            out_lse,
            mid_out,
            mid_lse,
            page_size,
            page_stride_bytes,
            extra_page_size,
            extra_page_stride_bytes,
            num_splits,
            chunks_per_block,
            sm_scale,
            lse_scale,
        )

    @register_fake_op("flashinfer::cake_sparse_mla_sm120_dsv4_nvfp4_decode")
    def _fake_decode(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(decode=_decode, raw_decode=entry)


@supported_compute_capability([120, 121])
def cake_sparse_mla_sm120_dsv4_nvfp4_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    max_splits: int = 16,
) -> Tuple[int, int]:
    """Run the allocation-free Cake SM120 NVFP4 sparse-MLA decode.

    Writes ``output`` and ``out_lse`` in place and returns the resolved
    ``(num_splits, chunks_per_block)``.  ``mid_out`` / ``mid_lse`` are
    required when the plan splits (``num_splits > 1``); size them with
    ``cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks`` splits to cover every plan.
    """

    if q.ndim != 3 or q.shape[-1] != _D_QK:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    heads = cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads()
    if num_heads not in heads:
        raise ValueError(
            f"Cake SM120 DSv4 NVFP4 sparse MLA supports {heads} query heads, got {num_heads}"
        )
    if (
        output.shape != q.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be a contiguous bfloat16 tensor of shape {tuple(q.shape)}"
        )
    if (
        out_lse.shape != (num_tokens, num_heads)
        or out_lse.dtype != torch.float32
        or not out_lse.is_contiguous()
    ):
        raise ValueError(
            f"out_lse must be a contiguous float32 tensor of shape {(num_tokens, num_heads)}"
        )
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    indices = _normalize_indices(indices, "indices", num_tokens)
    topk = int(indices.shape[1])
    kv_flat, page_size, page_stride = _cache_geometry(kv_cache, "kv_cache")
    extra_flat = None
    extra_page_size = 0
    extra_page_stride = 0
    extra_topk = 0
    if extra_kv_cache is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
        extra_topk = int(extra_indices.shape[1])
        extra_flat, extra_page_size, extra_page_stride = _cache_geometry(
            extra_kv_cache, "extra_kv_cache"
        )
    topk_length = _normalize_length(topk_length, "topk_length", num_tokens)
    extra_topk_length = _normalize_length(
        extra_topk_length, "extra_topk_length", num_tokens
    )
    if attn_sink is not None:
        if (
            attn_sink.ndim != 1
            or attn_sink.dtype != torch.float32
            or attn_sink.shape[0] < num_heads
        ):
            raise ValueError(
                f"attn_sink must be a 1-D float32 tensor with at least {num_heads} entries"
            )
        attn_sink = attn_sink.contiguous()
    splits, cpb = _resolve_splits(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        device=q.device,
        num_splits=num_splits,
        max_splits=max_splits,
    )
    if splits > 1:
        if mid_out is None or mid_lse is None:
            raise ValueError(
                f"this shape splits into {splits} CTAs per head block; pass caller-owned mid_out / mid_lse scratch"
            )
        if (
            mid_out.ndim != 4
            or mid_out.shape[0] != num_tokens
            or mid_out.shape[1] != num_heads
            or mid_out.shape[2] < splits
            or mid_out.shape[3] != _D_V
            or mid_out.dtype != torch.bfloat16
            or not mid_out.is_contiguous()
        ):
            raise ValueError(
                f"mid_out must be a contiguous bfloat16 [{num_tokens}, {num_heads}, >= {splits}, {_D_V}] tensor, "
                f"got {tuple(mid_out.shape)} {mid_out.dtype}"
            )
        if (
            mid_lse.ndim != 3
            or tuple(mid_lse.shape) != tuple(mid_out.shape[:3])
            or mid_lse.dtype != torch.float32
            or not mid_lse.is_contiguous()
        ):
            raise ValueError(
                f"mid_lse must be a contiguous float32 {tuple(mid_out.shape[:3])} tensor, got {tuple(mid_lse.shape)}"
            )
    else:
        mid_out = None
        mid_lse = None
    get_cake_sparse_mla_sm120_dsv4_nvfp4_module().decode(
        q,
        kv_flat,
        indices,
        extra_flat,
        extra_indices if extra_flat is not None else None,
        topk_length,
        extra_topk_length,
        attn_sink,
        output,
        out_lse,
        mid_out,
        mid_lse,
        page_size,
        page_stride,
        extra_page_size,
        extra_page_stride,
        splits,
        cpb,
        float(sm_scale),
        float(lse_scale),
    )
    return splits, cpb


@supported_compute_capability([120, 121])
def _cake_nvfp4_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocating convenience entry (tests / benchmarks): returns ``(output, out_lse)``."""

    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, {_D_QK}], got {tuple(q.shape)}")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    topk = int(indices.shape[-1])
    extra_topk = int(extra_indices.shape[-1]) if extra_indices is not None else 0
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    mid_out = torch.empty(
        (num_tokens, num_heads, chunks, _D_V), dtype=torch.bfloat16, device=q.device
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, chunks), dtype=torch.float32, device=q.device
    )
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=num_splits,
    )
    return output, out_lse


def functional_run(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
    *,
    lengths: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    extra: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lengths: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
) -> torch.Tensor:
    """Public-API route: carve the split scratch (and the LSE when the caller passes none) from ``workspace``."""

    from ._prepared import _workspace_tensor_view

    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    indices = _normalize_indices(indices, "indices", num_tokens)
    if extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
    topk = int(indices.shape[1])
    extra_topk = int(extra_indices.shape[1]) if extra_indices is not None else 0
    splits, _ = _resolve_splits(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        device=q.device,
        num_splits=None,
        max_splits=16,
    )
    requirements = []
    if splits > 1:
        requirements.append(((num_tokens, num_heads, splits, _D_V), torch.bfloat16))
        requirements.append(((num_tokens, num_heads, splits), torch.float32))
    if lse is None:
        requirements.append(((num_tokens, num_heads), torch.float32))
    views = []
    offset = 0
    for shape, dtype in requirements:
        view, offset = _workspace_tensor_view(
            workspace, byte_offset=offset, shape=shape, dtype=dtype, alignment=16
        )
        if view is None:
            raise ValueError(
                "attention workspace insufficient for the resolved Cake plan: need at least "
                f"{cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(num_tokens, num_heads, topk, extra_topk)} "
                f"bytes for {num_tokens} tokens x {num_heads} heads x topk {topk}"
                + (f" + extra topk {extra_topk}" if extra_topk else "")
            )
        views.append(view)
    mid_out = views[0] if splits > 1 else None
    mid_lse = views[1] if splits > 1 else None
    result = lse if lse is not None else views[-1]
    cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        cache,
        indices,
        output,
        result,
        scale,
        topk_length=lengths,
        attn_sink=sink,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=splits,
    )
    return result


def _arena(
    wrapper, name: str, numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    arenas: Dict[str, torch.Tensor] = wrapper.__dict__.setdefault("_cake_arenas", {})
    current = arenas.get(name)
    if current is None or current.numel() < numel or current.device != device:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("warm up this attention shape before CUDA graph capture")
        current = torch.empty(max(numel, 1), dtype=dtype, device=device)
        arenas[name] = current
    return current


def wrapper_run(
    wrapper,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    prefill_impl: Optional[str] = None,
    return_lse: bool = False,
    lse_scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """``SparseMLASm120Wrapper.run`` for ``backend="cake"``: wrapper-owned grow-only scratch."""

    if q.ndim == 4 and q.shape[1] == 1:
        q = q.squeeze(1)
    if output.ndim == 4 and output.shape[1] == 1:
        output = output.squeeze(1)
    if q.ndim != 3:
        raise ValueError("q must be [T,H,D] or [T,1,H,D]")
    if q.device != wrapper._device:
        raise ValueError("tensors must be on the Wrapper device")
    num_tokens, num_heads = int(q.shape[0]), int(q.shape[1])
    if (
        wrapper._max_num_tokens is not None and num_tokens > wrapper._max_num_tokens
    ) or (wrapper._max_num_heads is not None and num_heads > wrapper._max_num_heads):
        raise ValueError("query exceeds max_num_tokens/max_num_heads")
    if prefill_impl not in (None, "auto", "mg"):
        raise ValueError("NVFP4 prefill_impl must be None, auto, or mg")
    indices = _normalize_indices(indices, "indices", num_tokens)
    if extra_indices is not None:
        extra_indices = _normalize_indices(extra_indices, "extra_indices", num_tokens)
    topk = int(indices.shape[1])
    extra_topk = int(extra_indices.shape[1]) if extra_indices is not None else 0
    splits, _ = _resolve_splits(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        device=q.device,
        num_splits=None,
        max_splits=16,
    )
    if (mid_out is None) != (mid_lse is None):
        raise ValueError("mid_out and mid_lse must be provided together")
    if splits > 1 and mid_out is None:
        rows = num_tokens * num_heads * splits
        mid_out = _arena(wrapper, "mid_out", rows * _D_V, torch.bfloat16, q.device)[
            : rows * _D_V
        ].view(num_tokens, num_heads, splits, _D_V)
        mid_lse = _arena(wrapper, "mid_lse", rows, torch.float32, q.device)[:rows].view(
            num_tokens, num_heads, splits
        )
    if out_lse is None:
        lse = _arena(wrapper, "lse", num_tokens * num_heads, torch.float32, q.device)[
            : num_tokens * num_heads
        ].view(num_tokens, num_heads)
    else:
        lse = out_lse[:num_tokens, :num_heads]
        if not lse.is_contiguous():
            raise ValueError(
                "out_lse must be a contiguous [num_tokens, num_heads] float32 buffer"
            )
    cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        kv_cache,
        indices,
        output,
        lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=splits,
    )
    return lse if return_lse else None


__all__ = [
    "cake_sparse_mla_sm120_dsv4_nvfp4_decode",
    "cake_sparse_mla_sm120_dsv4_nvfp4_format_info",
    "cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks",
    "cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits",
    "cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes",
    "cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads",
    "get_cake_sparse_mla_sm120_dsv4_nvfp4_module",
]
