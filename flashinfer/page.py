"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .trace.templates.page import (
    append_paged_kv_cache_trace,
    append_paged_mla_kv_cache_trace,
    nvfp4_quantize_append_paged_kv_cache_trace,
    nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_trace,
)
from .jit.page import gen_page_module
from .utils import (
    TensorLayout,
    _check_kv_layout,
    check_shape_dtype_device,
    _unpack_paged_kv_cache,
    register_custom_op,
    register_fake_op,
)


@functools.cache
def get_page_module():
    return gen_page_module().build_and_load()


@register_custom_op(
    "flashinfer::append_paged_mla_kv_cache",
    mutates_args=("ckv_cache", "kpe_cache"),
)
def _append_paged_mla_kv_cache_kernel(
    append_ckv: torch.Tensor,
    append_kpe: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    ckv_cache: Optional[torch.Tensor],
    kpe_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().append_paged_mla_kv_cache(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


@register_custom_op(
    "flashinfer::append_paged_kv_cache",
    mutates_args=("paged_k_cache", "paged_v_cache"),
)
def _append_paged_kv_cache_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: Optional[torch.Tensor],
    paged_v_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    layout: int,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout,
    )


@register_fake_op("flashinfer::append_paged_kv_cache")
def _fake_append_paged_kv_cache_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: Optional[torch.Tensor],
    paged_v_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    layout: int,
) -> None:
    pass


@register_custom_op(
    "flashinfer::nvfp4_quantize_append_paged_kv_cache",
    mutates_args=(
        "paged_k_cache",
        "paged_v_cache",
        "k_scale_cache",
        "v_scale_cache",
    ),
)
def _nvfp4_quantize_append_paged_kv_cache_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    k_scale: float,
    v_scale: float,
    layout: int,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().nvfp4_quantize_append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        k_scale,
        v_scale,
        layout,
    )


@register_fake_op("flashinfer::nvfp4_quantize_append_paged_kv_cache")
def _fake_nvfp4_quantize_append_paged_kv_cache_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    k_scale: float,
    v_scale: float,
    layout: int,
) -> None:
    pass


@register_custom_op(
    "flashinfer::nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
    mutates_args=(
        "paged_k_cache",
        "paged_v_cache",
        "k_scale_cache",
        "v_scale_cache",
    ),
)
def _nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    layout: int,
) -> None:
    slot_mapping = slot_mapping.contiguous()
    get_page_module().nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        append_key,
        append_value,
        slot_mapping,
        paged_k_cache,
        paged_v_cache,
        k_scale_cache,
        v_scale_cache,
        k_scale,
        v_scale,
        layout,
    )


@register_fake_op("flashinfer::nvfp4_quantize_append_paged_kv_cache_with_slot_mapping")
def _fake_nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    layout: int,
) -> None:
    pass


@flashinfer_api
def get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int,
    batch_indices: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert append indptr and sequence lengths to batch indices and positions.

    Parameters
    ----------
    append_indptr : torch.Tensor
        The indptr of the ragged tensor, shape: ``[batch_size + 1]``.
    seq_lens: torch.Tensor
        The sequence lengths of each request in the KV-Cache, shape: ``[batch_size]``.
    nnz : int
        The number of entries in the ragged tensor.
    batch_indices : Optional[torch.Tensor]
        Pre-allocated output tensor for batch_indices. If ``None``, allocated automatically.
    positions : Optional[torch.Tensor]
        Pre-allocated output tensor for positions. If ``None``, allocated automatically.

    Returns
    -------
    batch_indices: torch.Tensor
        The batch indices of each entry in the ragged tensor, shape: ``[nnz]``.
    positions: torch.Tensor
        The positions of each entry in the ragged tensor, shape: ``[nnz]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> nnz_kv = 10
    >>> append_indptr = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32, device="cuda:0")
    >>> seq_lens = torch.tensor([5, 5, 5, 5])
    >>> batch_indices, positions = flashinfer.get_batch_indices_positions(append_indptr, seq_lens, nnz_kv)
    >>> batch_indices
    tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], device='cuda:0', dtype=torch.int32)
    >>> positions  # the rightmost column index of each row
    tensor([4, 3, 4, 2, 3, 4, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function is similar to `CSR2COO <https://docs.nvidia.com/cuda/cusparse/#csr2coo>`_
    conversion in cuSPARSE library, with the difference that we are converting from a ragged
    tensor (which doesn't require a column indices array) to a COO format.

    See Also
    --------
    append_paged_kv_cache
    """
    batch_size = append_indptr.size(0) - 1
    device = append_indptr.device
    dtype = torch.int32

    if batch_indices is None:
        batch_indices = torch.full((nnz,), -1, device=device, dtype=dtype)
    else:
        check_shape_dtype_device(batch_indices, (nnz,), dtype, device, "batch_indices")
        batch_indices.fill_(-1)

    if positions is None:
        positions = torch.zeros((nnz,), device=device, dtype=dtype)
    else:
        check_shape_dtype_device(positions, (nnz,), dtype, device, "positions")

    from .triton.page import get_batch_indices_positions_kernel

    get_batch_indices_positions_kernel[(batch_size,)](
        append_indptr, seq_lens, batch_indices, positions, num_stages=2
    )
    return batch_indices, positions


def get_seq_lens(
    kv_indptr: torch.Tensor, kv_last_page_len: torch.Tensor, page_size: int
) -> torch.Tensor:
    r"""Convert KV indptr and last page length to sequence lengths.

    Parameters
    ----------
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    page_size : int
        The size of a page in the paged kv-cache.

    Returns
    -------
    seq_lens: torch.Tensor
        The sequence lengths of each request in the paged kv-cache, shape: ``[batch_size]``.
    """
    return (
        torch.clamp(kv_indptr[1:] - kv_indptr[:-1] - 1, min=0) * page_size
        + kv_last_page_len
    )


@flashinfer_api(trace=append_paged_mla_kv_cache_trace)
def append_paged_mla_kv_cache(
    append_ckv: torch.Tensor,
    append_kpe: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    ckv_cache: Optional[torch.Tensor],
    kpe_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    r"""Append a batch of key-value pairs to a paged key-value cache,
    Note: current only support ckv=512 and kpe=64

    Parameters
    ----------
    append_ckv : torch.Tensor
        The compressed kv tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], ckv_dim]``.
    append_kpe : torch.Tensor
        The value tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], kpe_dim]``.
    batch_indices : torch.Tensor
        The batch indices of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    positions : torch.Tensor
        The positions of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    ckv_cache : cache for compressed kv, torch.Tensor, shape: [page_num, page_size, ckv_dim]
    kpe_cache : cache for key position embedding, torch.Tensor, shape: [page_num, page_size, kpe_dim]
    kv_indices : torch.Tensor
        The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    """
    _append_paged_mla_kv_cache_kernel(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


@flashinfer_api(trace=append_paged_kv_cache_trace)
def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
) -> None:
    r"""Append a batch of key-value pairs to a paged key-value cache.

    Parameters
    ----------
    append_key : torch.Tensor
        The key tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], num_kv_heads, head_dim]``.
    append_value : torch.Tensor
        The value tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], num_kv_heads, head_dim]``.
    batch_indices : torch.Tensor
        The batch indices of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    positions : torch.Tensor
        The positions of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        The paged KV-Cache stored as a tuple of tensors or a single tensor:

        * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
          ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
          and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

        * a single 5-D tensor with shape:
          ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
          :attr:`kv_layout` is ``NHD``, and
          ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
          :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
          ``paged_kv_cache[:, 1]`` is the value-cache.

    kv_indices : torch.Tensor
        The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    kv_layout : str
        The layout of the paged kv-cache, either ``NHD`` or ``HND``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> nnz_kv = 100
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    >>> v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    >>> # 45 + 8 + 25 + 22 = nnz_kv
    >>> kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    >>> kv_append_indptr = torch.cat(
    ...     [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ... ).int()  # [0, 45, 53, 78, 100]
    >>> max_num_pages = 1000
    >>> page_size = 16
    >>> paged_kv_cache = torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)
    >>> num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
    >>> kv_page_indptr = torch.cat(
    ...     [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ... ).int()
    >>> # use first 8 pages in the paged-kv
    >>> kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
    >>> # 45 = (3 - 1) * 16 + 13
    >>> # 8 = (1 - 1) * 16 + 8
    >>> # 25 = (2 - 1) * 16 + 9
    >>> # 22 = (2 - 1) * 16 + 6
    >>> kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
    >>> batch_indices, positions = flashinfer.get_batch_indices_positions(
    ...     kv_append_indptr, flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size), nnz_kv
    ... )
    >>> batch_indices
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3], device='cuda:0', dtype=torch.int32)
    >>> positions
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44,  0,  1,  2,  3,  4,  5,  6,  7,  0,
            1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21], device='cuda:0',
        dtype=torch.int32)
    >>> flashinfer.append_paged_kv_cache(
    ...     k_append,
    ...     v_append,
    ...     batch_indices,
    ...     positions,
    ...     paged_kv_cache,
    ...     kv_page_indices,
    ...     kv_page_indptr,
    ...     kv_last_page_len
    ... )

    Note
    ----
    The function assumes that the space for appended k/v has already been allocated,
    which means :attr:`kv_indices`, :attr:`kv_indptr`, :attr:`kv_last_page_len` has
    incorporated appended k/v.

    See Also
    --------
    get_batch_indices_positions
    """
    _check_kv_layout(kv_layout)
    _append_paged_kv_cache_kernel(
        append_key,
        append_value,
        batch_indices,
        positions,
        *_unpack_paged_kv_cache(paged_kv_cache, kv_layout),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        TensorLayout[kv_layout].value,
    )


@flashinfer_api(trace=nvfp4_quantize_append_paged_kv_cache_trace)
def nvfp4_quantize_append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_sf: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    k_scale: float,
    v_scale: float,
    kv_layout: str = "NHD",
) -> None:
    r"""Quantize and append K/V rows into an NVFP4 paged KV cache.

    ``append_key`` and ``append_value`` must be fp16/bf16 tensors with shape
    ``[nnz, num_kv_heads, head_dim]``. The function writes packed E2M1 data
    into uint8 paged K/V cache tensors with last dimension ``head_dim // 2``
    and writes FP8 E4M3 block scales into ``kv_cache_sf`` tensors with last
    dimension ``head_dim // 16``.

    ``k_scale`` and ``v_scale`` are the global decode scales consumed by the
    NVFP4 attention kernels, i.e. dequantization reconstructs values as
    ``e2m1_value * block_scale * global_scale``.

    Parameters
    ----------
    append_key : torch.Tensor
        The key tensor to quantize and append, shape
        ``[nnz, num_kv_heads, head_dim]`` with dtype ``torch.float16`` or
        ``torch.bfloat16``.
    append_value : torch.Tensor
        The value tensor to quantize and append, with the same shape and dtype
        as ``append_key``.
    batch_indices : torch.Tensor
        The batch index for each appended row, shape ``[nnz]``.
    positions : torch.Tensor
        The logical token position for each appended row, shape ``[nnz]``.
    paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Caller-owned packed NVFP4 K/V cache. For tuple input, each tensor has
        shape ``[max_num_pages, page_size, num_kv_heads, head_dim // 2]`` when
        ``kv_layout="NHD"`` and ``[max_num_pages, num_kv_heads, page_size,
        head_dim // 2]`` when ``kv_layout="HND"``. A stacked 5-D cache is also
        accepted with K/V on the second dimension.
    kv_cache_sf : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Caller-owned FP8 E4M3 scale cache with the same tuple or stacked cache
        format as ``paged_kv_cache``, replacing ``head_dim // 2`` with
        ``head_dim // 16``.
    kv_indices : torch.Tensor
        The page indices of the paged KV cache, shape ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged KV cache, shape ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request, shape
        ``[batch_size]``.
    k_scale : float
        Positive finite global decode scale for K.
    v_scale : float
        Positive finite global decode scale for V.
    kv_layout : str
        Layout of the paged KV cache, either ``"NHD"`` or ``"HND"``.

    Returns
    -------
    None
        This function updates ``paged_kv_cache`` and ``kv_cache_sf`` in place.

    Note
    ----
    The function assumes that the space for appended K/V rows has already been
    allocated and described by ``kv_indices``, ``kv_indptr``, and
    ``kv_last_page_len``.
    """
    _check_kv_layout(kv_layout)
    if append_key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"append_key must be float16 or bfloat16, got {append_key.dtype}"
        )
    if append_value.dtype != append_key.dtype:
        raise ValueError(
            f"append_key and append_value must have the same dtype, got "
            f"{append_key.dtype} and {append_value.dtype}"
        )
    paged_k_cache, paged_v_cache = _unpack_paged_kv_cache(paged_kv_cache, kv_layout)
    k_scale_cache, v_scale_cache = _unpack_paged_kv_cache(kv_cache_sf, kv_layout)
    if paged_k_cache.dtype != torch.uint8 or paged_v_cache.dtype != torch.uint8:
        raise ValueError("NVFP4 paged K/V cache tensors must have dtype torch.uint8")
    if (
        k_scale_cache.dtype != torch.float8_e4m3fn
        or v_scale_cache.dtype != torch.float8_e4m3fn
    ):
        raise ValueError(
            "NVFP4 scale cache tensors must have dtype torch.float8_e4m3fn"
        )
    k_scale = float(k_scale)
    v_scale = float(v_scale)
    if (
        not math.isfinite(k_scale)
        or not math.isfinite(v_scale)
        or k_scale <= 0.0
        or v_scale <= 0.0
    ):
        raise ValueError(
            "k_scale and v_scale must be positive finite global decode scales"
        )

    _nvfp4_quantize_append_paged_kv_cache_kernel(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        k_scale,
        v_scale,
        TensorLayout[kv_layout].value,
    )


def _as_float32_scalar_tensors(
    values: Tuple[Tuple[str, Union[float, torch.Tensor]], ...],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    stream_capturing = _is_stream_capturing_on_device(device)

    tensors = []
    cuda_tensors_to_validate = []
    for name, value in values:
        if not isinstance(value, torch.Tensor):
            if stream_capturing:
                raise ValueError(
                    f"{name} must be a contiguous float32 CUDA tensor on {device} "
                    "during CUDA graph capture"
                )
            scalar = float(value)
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(
                    f"{name} must be a positive finite global decode scale"
                )
            tensors.append(torch.tensor([scalar], dtype=torch.float32, device=device))
            continue

        if value.numel() != 1:
            raise ValueError(
                f"{name} must have exactly one element, got {value.numel()}"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be torch.float32, got {value.dtype}")
        is_cuda_tensor = value.is_cuda
        is_capturing = stream_capturing
        if is_capturing and (
            not is_cuda_tensor or value.device != device or not value.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous float32 CUDA tensor on {device} "
                "during CUDA graph capture"
            )
        if not is_cuda_tensor:
            scalar = float(value.item())
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(
                    f"{name} must be a positive finite global decode scale"
                )
        if value.device != device:
            value = value.to(device=device)
        value = value.contiguous()
        tensors.append(value)
        if is_capturing:
            continue
        if is_cuda_tensor:
            cuda_tensors_to_validate.append((name, value))

    # Validate CUDA scalar tensors with one device-to-host transfer instead of
    # a separate implicit sync for each tensor.
    if cuda_tensors_to_validate:
        host_values = (
            torch.stack([value.reshape(()) for _, value in cuda_tensors_to_validate])
            .detach()
            .cpu()
            .tolist()
        )
        for (name, _), scalar in zip(
            cuda_tensors_to_validate, host_values, strict=True
        ):
            scalar = float(scalar)
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(
                    f"{name} must be a positive finite global decode scale"
                )

    return tuple(tensors)


def _is_stream_capturing_on_device(device: torch.device) -> bool:
    if not hasattr(torch.cuda, "is_current_stream_capturing"):
        return False
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


@flashinfer_api(trace=nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_trace)
def nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_sf: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    k_scale: Union[float, torch.Tensor],
    v_scale: Union[float, torch.Tensor],
    kv_layout: str = "NHD",
) -> None:
    r"""Quantize and write K/V rows into an NVFP4 paged KV cache by slot mapping.

    This variant is intended for runtimes that already assign each token to a
    flat cache slot. ``slot_mapping[i]`` is interpreted as
    ``page_id * page_size + entry_idx``. Negative slots are padding and are
    ignored. ``append_key`` and ``append_value`` may contain additional padded
    rows; only ``slot_mapping.shape[0]`` rows are considered.

    Parameters
    ----------
    append_key : torch.Tensor
        The key tensor to quantize and write, shape
        ``[num_rows, num_kv_heads, head_dim]`` with dtype ``torch.float16`` or
        ``torch.bfloat16``. ``num_rows`` must be at least
        ``slot_mapping.shape[0]``.
    append_value : torch.Tensor
        The value tensor to quantize and write, with the same shape and dtype
        as ``append_key``.
    slot_mapping : torch.Tensor
        Flat cache slot for each row to write, shape ``[nnz]`` with dtype
        ``torch.int32`` or ``torch.int64``. Negative entries are ignored.
    paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Caller-owned packed NVFP4 K/V cache. For tuple input, each tensor has
        shape ``[max_num_pages, page_size, num_kv_heads, head_dim // 2]`` when
        ``kv_layout="NHD"`` and ``[max_num_pages, num_kv_heads, page_size,
        head_dim // 2]`` when ``kv_layout="HND"``. A stacked 5-D cache is also
        accepted with K/V on the second dimension.
    kv_cache_sf : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Caller-owned FP8 E4M3 scale cache with the same tuple or stacked cache
        format as ``paged_kv_cache``, replacing ``head_dim // 2`` with
        ``head_dim // 16``.
    k_scale : Union[float, torch.Tensor]
        Positive finite global decode scale for K. During CUDA graph capture,
        this must be a contiguous scalar ``torch.float32`` CUDA tensor on the
        same device as ``append_key``.
    v_scale : Union[float, torch.Tensor]
        Positive finite global decode scale for V. During CUDA graph capture,
        this must be a contiguous scalar ``torch.float32`` CUDA tensor on the
        same device as ``append_key``.
    kv_layout : str
        Layout of the paged KV cache, either ``"NHD"`` or ``"HND"``.

    Returns
    -------
    None
        This function updates ``paged_kv_cache`` and ``kv_cache_sf`` in place.
    """
    _check_kv_layout(kv_layout)
    if append_key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"append_key must be float16 or bfloat16, got {append_key.dtype}"
        )
    if append_value.dtype != append_key.dtype:
        raise ValueError(
            f"append_key and append_value must have the same dtype, got "
            f"{append_key.dtype} and {append_value.dtype}"
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"slot_mapping must be int32 or int64, got {slot_mapping.dtype}"
        )
    if (
        append_key.shape[0] < slot_mapping.shape[0]
        or append_value.shape[0] < slot_mapping.shape[0]
    ):
        raise ValueError(
            "append_key and append_value must have at least slot_mapping.shape[0] rows"
        )

    paged_k_cache, paged_v_cache = _unpack_paged_kv_cache(paged_kv_cache, kv_layout)
    k_scale_cache, v_scale_cache = _unpack_paged_kv_cache(kv_cache_sf, kv_layout)
    if paged_k_cache.dtype != torch.uint8 or paged_v_cache.dtype != torch.uint8:
        raise ValueError("NVFP4 paged K/V cache tensors must have dtype torch.uint8")
    if (
        k_scale_cache.dtype != torch.float8_e4m3fn
        or v_scale_cache.dtype != torch.float8_e4m3fn
    ):
        raise ValueError(
            "NVFP4 scale cache tensors must have dtype torch.float8_e4m3fn"
        )

    k_scale_tensor, v_scale_tensor = _as_float32_scalar_tensors(
        (("k_scale", k_scale), ("v_scale", v_scale)),
        device=append_key.device,
    )

    _nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_kernel(
        append_key,
        append_value,
        slot_mapping,
        paged_k_cache,
        paged_v_cache,
        k_scale_cache,
        v_scale_cache,
        k_scale_tensor,
        v_scale_tensor,
        TensorLayout[kv_layout].value,
    )
