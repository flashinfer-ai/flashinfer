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

"""Paged K/V cache views, page tables and NVFP4 scale factors."""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .tensors import MAX_INT32, validate_16byte_alignment

PagedKVCache = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
PagedKVScaleFactors = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class NativePagedKVCache:
    """Zero-copy logical HND views and TensorMap strides for native decode."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    num_physical_pages: int
    num_kv_heads: int
    storage_page_size: int
    head_dim: int
    k_page_stride: int
    k_head_stride: int
    k_token_stride: int
    v_page_stride: int
    v_head_stride: int
    v_token_stride: int


def normalize_native_paged_kv_cache(
    paged_kv_cache: PagedKVCache,
    *,
    expected_device: torch.device,
    kv_layout: str = "HND",
) -> NativePagedKVCache:
    """Return native decode views supporting HND- or NHD-physical storage."""

    (
        k_cache,
        v_cache,
        num_pages,
        num_kv_heads,
        storage_page_size,
        head_dim,
    ) = normalize_paged_kv_cache_views(
        paged_kv_cache,
        expected_device=expected_device,
        kv_layout=kv_layout,
    )
    k_page_stride, k_head_stride, k_token_stride = validate_native_hnd_tma_strides(
        k_cache, "K cache"
    )
    v_page_stride, v_head_stride, v_token_stride = validate_native_hnd_tma_strides(
        v_cache, "V cache"
    )
    return NativePagedKVCache(
        k_cache=k_cache,
        v_cache=v_cache,
        num_physical_pages=num_pages,
        num_kv_heads=num_kv_heads,
        storage_page_size=storage_page_size,
        head_dim=head_dim,
        k_page_stride=k_page_stride,
        k_head_stride=k_head_stride,
        k_token_stride=k_token_stride,
        v_page_stride=v_page_stride,
        v_head_stride=v_head_stride,
        v_token_stride=v_token_stride,
    )


def normalize_paged_kv_cache(
    paged_kv_cache: PagedKVCache,
    *,
    expected_device: torch.device,
    logical_head_dim: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int]:
    """Return compact HND views for block-sparse and legacy paged paths."""

    views = normalize_paged_kv_cache_views(
        paged_kv_cache,
        expected_device=expected_device,
    )
    k_cache, v_cache = views[:2]
    inferred_head_dim = views[5]
    if logical_head_dim is not None and inferred_head_dim != logical_head_dim:
        raise ValueError(
            "paged_kv_cache storage head dimension does not match the logical "
            f"head dimension {logical_head_dim}: got {int(k_cache.shape[-1])}"
        )
    k_page_stride = validate_hnd_inner_strides(k_cache, "K cache")
    v_page_stride = validate_hnd_inner_strides(v_cache, "V cache")
    return (*views, k_page_stride, v_page_stride)


def normalize_paged_kv_cache_views(
    paged_kv_cache: PagedKVCache,
    *,
    expected_device: torch.device,
    kv_layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """Return validated zero-copy HND views and the logical head dimension."""

    if isinstance(paged_kv_cache, torch.Tensor):
        if paged_kv_cache.ndim != 5 or paged_kv_cache.shape[1] != 2:
            raise ValueError(
                "combined paged_kv_cache must have shape "
                "[num_pages, 2, Hkv, page_size, head_dim]"
            )
        if paged_kv_cache.device != expected_device:
            raise ValueError(
                f"paged_kv_cache must be on {expected_device}, "
                f"got {paged_kv_cache.device}"
            )
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]
    elif isinstance(paged_kv_cache, tuple):
        if len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache tuple must contain exactly (K, V)")
        k_cache, v_cache = paged_kv_cache
        if not isinstance(k_cache, torch.Tensor) or not isinstance(
            v_cache, torch.Tensor
        ):
            raise TypeError("paged_kv_cache tuple members must be torch.Tensor")
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                "tuple K/V caches must each have shape "
                "[num_pages, Hkv, page_size, head_dim]"
            )
        if k_cache.device != expected_device or v_cache.device != expected_device:
            raise ValueError(f"tuple K/V caches must be on {expected_device}")
    else:
        raise TypeError(
            "paged_kv_cache must be a combined torch.Tensor or a (K, V) tuple"
        )

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("K/V cache views must be rank-4 tensors")
    if kv_layout == "NHD":
        k_cache = k_cache.transpose(1, 2)
        v_cache = v_cache.transpose(1, 2)
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache views must have identical logical shapes")
    if k_cache.device != v_cache.device:
        raise ValueError("K and V cache views must be on the same device")

    num_pages, num_kv_heads, page_size, storage_head_dim = map(int, k_cache.shape)
    if min(num_pages, num_kv_heads, page_size, storage_head_dim) <= 0:
        raise ValueError("paged_kv_cache dimensions must be positive")
    # This API reserves uint8 K/V storage for two packed NVFP4 values per byte.
    head_dim = (
        storage_head_dim * 2 if k_cache.dtype == torch.uint8 else storage_head_dim
    )
    return (
        k_cache,
        v_cache,
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
    )


def normalize_paged_kv_scale_factors(
    kv_scale_factors: Optional[PagedKVScaleFactors],
    *,
    k_cache: torch.Tensor,
    logical_head_dim: int,
    interleaved: bool = False,
    strided_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    """Validate NVFP4 scale tensors or create homogeneous-mode placeholders.

    Returns the K and V scale tensors and their (page, head, token) element
    strides. With ``strided_rows`` (the QToken sparse route), scale rows of
    at least 16 bytes may be strided HND views like the packed K/V (for
    example transposed token-major storage); otherwise they need compact HND
    rows. Interleaved NVFP4 rows carry their own scales and use placeholders.
    """

    if k_cache.dtype != torch.uint8 or interleaved:
        if kv_scale_factors is not None:
            raise ValueError(
                "kv_scale_factors are accepted only with packed NVFP4 torch.uint8 K/V"
            )
        placeholder = k_cache[0, 0, 0, :1].view(torch.uint8)[:1]
        return placeholder, placeholder, (0, 0, 0), (0, 0, 0)
    if kv_scale_factors is None:
        raise ValueError(
            "packed NVFP4 torch.uint8 K/V requires kv_scale_factors=(K_SF, V_SF)"
        )
    if not isinstance(kv_scale_factors, tuple) or len(kv_scale_factors) != 2:
        raise TypeError("kv_scale_factors must be a (K_SF, V_SF) tensor tuple")
    k_sf_cache, v_sf_cache = kv_scale_factors
    if not isinstance(k_sf_cache, torch.Tensor) or not isinstance(
        v_sf_cache, torch.Tensor
    ):
        raise TypeError("kv_scale_factors tuple members must be torch.Tensor")
    expected_shape = (*k_cache.shape[:-1], logical_head_dim // 16)
    strides = []
    for scale, name in (
        (k_sf_cache, "K scale factors"),
        (v_sf_cache, "V scale factors"),
    ):
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(scale.shape)}"
            )
        if scale.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"{name} must have dtype torch.float8_e4m3fn, got {scale.dtype}"
            )
        if scale.device != k_cache.device:
            raise ValueError(f"{name} must be on {k_cache.device}, got {scale.device}")
        sf_per_token = logical_head_dim // 16
        if sf_per_token < 16 or not strided_rows:
            # Folded scale rows (FmhaDecodeConfig.sf_tma_reshape_factor) copy
            # several tokens per box row, which needs compact HND rows.
            try:
                page_stride = validate_hnd_inner_strides(scale, name)
            except ValueError as error:
                raise ValueError(
                    f"{name} of head_dim={logical_head_dim} need token-contiguous "
                    f"compact HND rows on this route: {error}"
                ) from error
            strides.append((page_stride, *scale.stride()[1:3]))
        else:
            strides.append(validate_native_hnd_tma_strides(scale, name))
    return k_sf_cache, v_sf_cache, strides[0], strides[1]


def validate_block_table_metadata(
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
) -> tuple[torch.device, int, int]:
    """Validate fixed page-table and sequence-length tensor structure."""

    block_table_device, block_table_batch_size, table_capacity = validate_block_tables(
        block_tables
    )

    if not isinstance(seq_lens_kv, torch.Tensor):
        raise TypeError("seq_lens_kv must be a torch.Tensor")
    if seq_lens_kv.ndim != 1:
        raise ValueError("seq_lens_kv must be one-dimensional")
    if seq_lens_kv.dtype != torch.int32:
        raise TypeError("seq_lens_kv must have dtype torch.int32")
    if seq_lens_kv.device.type != "cuda":
        raise ValueError("seq_lens_kv must be a CUDA tensor")
    if not seq_lens_kv.is_contiguous():
        raise ValueError("seq_lens_kv must be contiguous")
    if seq_lens_kv.data_ptr() % 4 != 0:
        raise ValueError("seq_lens_kv data pointer must be 4-byte aligned")
    if block_table_device != seq_lens_kv.device:
        raise ValueError("block_tables and seq_lens_kv must be on the same device")

    batch_size = int(seq_lens_kv.numel())
    if batch_size <= 0:
        raise ValueError("seq_lens_kv must contain at least one request")
    if block_table_batch_size != batch_size:
        raise ValueError(
            "block_tables must have one row per request: expected "
            f"{batch_size}, got {block_table_batch_size}"
        )
    return seq_lens_kv.device, batch_size, table_capacity


def validate_block_tables(
    block_tables: torch.Tensor,
) -> tuple[torch.device, int, int]:
    """Validate a fixed page table without requiring sequence-length storage."""

    if not isinstance(block_tables, torch.Tensor):
        raise TypeError("block_tables must be a torch.Tensor")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [B, C]")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must have dtype torch.int32")
    if block_tables.device.type != "cuda":
        raise ValueError("block_tables must be a CUDA tensor")
    if block_tables.data_ptr() % 4 != 0:
        raise ValueError("block_tables data pointer must be 4-byte aligned")

    batch_size = int(block_tables.shape[0])
    if batch_size <= 0:
        raise ValueError("block_tables must contain at least one request row")
    table_capacity = int(block_tables.shape[1])
    if table_capacity <= 0:
        raise ValueError("block_tables must contain at least one page column")
    if table_capacity > MAX_INT32:
        raise ValueError("block_tables column count exceeds signed int32")
    if block_tables.stride(1) != 1:
        raise ValueError("block_tables must be contiguous within each row")
    if block_tables.stride(0) < table_capacity:
        raise ValueError(
            "block_tables rows must not overlap: row stride must be at least "
            f"the column count ({table_capacity}), got {block_tables.stride(0)}"
        )
    return block_tables.device, batch_size, table_capacity


def validate_hnd_inner_strides(tensor: torch.Tensor, name: str) -> int:
    _, num_kv_heads, page_size, head_dim = tensor.shape
    expected_inner = (page_size * head_dim, head_dim, 1)
    if tensor.stride()[1:] != expected_inner:
        raise ValueError(
            f"{name} must have compact HND inner strides {expected_inner}, "
            f"got {tensor.stride()[1:]}"
        )
    page_stride = int(tensor.stride(0))
    compact_page_elements = num_kv_heads * page_size * head_dim
    if page_stride < compact_page_elements:
        raise ValueError(
            f"{name} pages overlap: outer stride {page_stride} is smaller than "
            f"{compact_page_elements}"
        )
    if page_stride > 2**63 - 1:
        raise ValueError(f"{name} outer page stride exceeds signed int64")
    validate_16byte_alignment(tensor, name)
    if page_stride * tensor.element_size() % 16 != 0:
        raise ValueError(f"{name} outer page stride must be 16-byte aligned")
    return page_stride


def validate_native_hnd_tma_strides(
    tensor: torch.Tensor,
    name: str,
) -> tuple[int, int, int]:
    """Validate a non-overlapping logical HND view for TensorMap addressing."""

    num_pages, num_kv_heads, storage_page_size, head_dim = map(int, tensor.shape)
    page_stride, head_stride, token_stride, dim_stride = map(int, tensor.stride())
    if dim_stride != 1:
        raise ValueError(f"{name} head-dimension stride must be one")
    named_strides = (
        ("page", page_stride),
        ("head", head_stride),
        ("token", token_stride),
    )
    if any(stride <= 0 for _, stride in named_strides):
        raise ValueError(f"{name} TensorMap strides must be positive")
    max_int64 = 2**63 - 1
    if any(stride > max_int64 for _, stride in named_strides):
        raise ValueError(f"{name} TensorMap stride exceeds signed int64")

    element_size = tensor.element_size()
    for stride_name, stride in named_strides:
        if stride * element_size % 16 != 0:
            raise ValueError(f"{name} {stride_name} stride must be 16-byte aligned")
    validate_16byte_alignment(tensor, name)

    # A gapped K or V slice of a packed 2D cache is not dense, so PyTorch's
    # dense-layout predicates cannot establish safety. Prove non-overlap from
    # the actual logical dimensions and strides instead.
    span = 1
    dimensions = sorted(
        (
            (1, head_dim, "head dimension"),
            (token_stride, storage_page_size, "token"),
            (head_stride, num_kv_heads, "head"),
            (page_stride, num_pages, "page"),
        ),
        key=lambda item: item[0],
    )
    for stride, extent, dimension_name in dimensions:
        if extent <= 1:
            continue
        if stride < span:
            raise ValueError(
                f"{name} {dimension_name} dimension overlaps another dimension"
            )
        span += (extent - 1) * stride
        if span - 1 > max_int64:
            raise ValueError(f"{name} address span exceeds signed int64")
    return page_stride, head_stride, token_stride
