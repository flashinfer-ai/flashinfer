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

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import (
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
    get_cuda_stream,
    register_custom_op,
    register_fake_op,
)

_page_module = None


def get_page_module():
    global _page_module
    if _page_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            _page_module = _kernels
        else:
            _page_module = load_cuda_ops(
                "page",
                [
                    FLASHINFER_CSRC_DIR / "page.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_page_ops.cu",
                ],
            )
    return _page_module


def block_sparse_indices_to_vector_sparse_offsets(
    block_sparse_indices: torch.Tensor,
    block_sparse_indptr: torch.Tensor,
    vector_sparse_offsets: torch.Tensor,
    vector_sparse_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    stride_block: int,
    stride_n: int,
    block_size: int,
) -> torch.Tensor:
    if block_size == 1:
        if stride_block == 1:
            return block_sparse_indices
        else:
            return block_sparse_indices * stride_block

    with block_sparse_indices.device as device:
        assert block_sparse_indices.dtype == torch.int32
        assert block_sparse_indptr.dtype == torch.int32
        assert vector_sparse_offsets.dtype == torch.int32
        assert vector_sparse_indptr.dtype == torch.int32
        assert kv_lens.dtype == torch.int32
        batch_size = block_sparse_indptr.size(0) - 1
        get_page_module().block_sparse_indices_to_vector_sparse_offsets(
            block_sparse_indices,
            block_sparse_indptr,
            vector_sparse_offsets,
            vector_sparse_indptr,
            kv_lens,
            stride_block,
            stride_n,
            batch_size,
            block_size,
            get_cuda_stream(device),
        )
    return vector_sparse_offsets


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
    with append_key.device as device:
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
            get_cuda_stream(device),
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


@triton.jit
def get_batch_indices_positions_kernel(
    append_indptr,
    seq_lens_ptr,
    batch_indices_ptr,
    positions_ptr,
    num_stages: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    batch_start = tl.load(append_indptr + batch_idx)
    batch_end = tl.load(append_indptr + batch_idx + 1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)

    for i in tl.range(batch_start, batch_end, 128, num_stages=num_stages):
        offsets = tl.arange(0, 128) + i
        mask = offsets < batch_end
        tl.store(batch_indices_ptr + offsets, batch_idx, mask)
        tl.store(positions_ptr + offsets, offsets + seq_len - batch_end, mask)


def get_batch_indices_positions(
    append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
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

    Returns
    -------
    batch_indices: torch.Tensor
        The batch indices of the each entry in the ragged tensor, shape: ``[nnz]``.
    positions: torch.Tensor
        The positions of the each entry in the ragged tensor, shape: ``[nnz]``.

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
    tensor (which don't require a column indices array) to a COO format.

    See Also
    --------
    append_paged_kv_cache
    """
    batch_size = append_indptr.size(0) - 1
    batch_indices = torch.empty((nnz,), device=append_indptr.device, dtype=torch.int32)
    positions = torch.empty((nnz,), device=append_indptr.device, dtype=torch.int32)
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
    The function assumes that the space for appended k/v have already been allocated,
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
