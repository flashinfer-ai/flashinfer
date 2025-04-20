"""
Copyright (c) 2025 by FlashInfer team.

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

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _compute_padded_indptr(
    indptr_ptr, padded_indptr_ptr, n_rows, multiple_of, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows

    # Load row lengths
    row_start = tl.load(indptr_ptr + offsets, mask=mask, other=0)
    row_end = tl.load(indptr_ptr + offsets + 1, mask=mask, other=0)
    row_lengths = row_end - row_start

    # Compute padded lengths (round up to multiple_of)
    padded_lengths = ((row_lengths + multiple_of - 1) // multiple_of) * multiple_of

    # Compute cumulative sum for padded indptr
    if pid == 0:
        # First element is always 0
        tl.store(padded_indptr_ptr + 0, 0)

    # Store the padded lengths at the correct positions
    tl.store(padded_indptr_ptr + offsets + 1, padded_lengths, mask=mask)


@triton.jit
def _pad_ragged_tensor(
    ragged_tensor_ptr,
    padded_tensor_ptr,
    indptr_ptr,
    padded_indptr_ptr,
    n_rows,
    dim,
    BLOCK_SIZE: tl.constexpr,
    fill_zeros: tl.constexpr,
):
    pid = tl.program_id(0)

    # Process one row per program
    if pid >= n_rows:
        return

    # Get original and padded row information
    row_start = tl.load(indptr_ptr + pid)
    row_end = tl.load(indptr_ptr + pid + 1)
    row_length = row_end - row_start

    padded_row_start = tl.load(padded_indptr_ptr + pid)
    padded_row_end = tl.load(padded_indptr_ptr + pid + 1)
    padded_row_length = padded_row_end - padded_row_start

    # Copy the original data
    for i in range(0, row_length):
        col_idx = i
        src_offset = (row_start + i) * dim
        dst_offset = (padded_row_start + i) * dim

        # Copy the entire feature vector for this position
        for j in range(0, dim, BLOCK_SIZE):
            j_offsets = j + tl.arange(0, BLOCK_SIZE)
            j_mask = j_offsets < dim
            values = tl.load(ragged_tensor_ptr + src_offset + j_offsets, mask=j_mask)
            tl.store(padded_tensor_ptr + dst_offset + j_offsets, values, mask=j_mask)

    # Zero-pad the remaining positions
    if fill_zeros:
        for i in range(row_length, padded_row_length):
            col_idx = i
            dst_offset = (padded_row_start + i) * dim

            # Zero out the entire feature vector for this position
            for j in range(0, dim, BLOCK_SIZE):
                j_offsets = j + tl.arange(0, BLOCK_SIZE)
                j_mask = j_offsets < dim
                tl.store(padded_tensor_ptr + dst_offset + j_offsets, 0.0, mask=j_mask)

def max_power_of_2_leq(x: int) -> int:
    r"""Return the maximum power of 2 less than or equal to x."""
    return 1 << (x - 1).bit_length()


def pad_ragged_tensor_to_multiple_of(
    ragged_tensor: torch.Tensor,
    indptr: torch.Tensor,
    multiple_of: int,
    fill_zeros: bool = False,
    output_ragged_tensor: Optional[torch.Tensor] = None,
    output_indptr: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Pad each row of ragged tensor to a multiple of ``multiple_of``.

    Suppose the ragged tensor has shape (150, 1024), and the indptr is [0, 100, 150] (which means there are 2 rows,
    the first row has 100 columns, the second row has 50 columns), and the multiple_of is 16.
    We will pad the first row to 112 columns, and the second row to 64 columns.
    The padded ragged tensor will have shape (176, 1024), and the returned indptr will be [0, 112, 176].

    Parameters
    ----------
    ragged_tensor: torch.Tensor
        The ragged tensor to pad, expected shape: (nnz, D)
    indptr: torch.Tensor
        The indptr of the ragged tensor, expected shape: (n_rows + 1,)
    multiple_of: int
        The multiple of to pad to, e.g. 256
    fill_zeros: bool
        If True, the padded positions will be filled with zeros, otherwise they will be random values,
        default is False.
    output_ragged_tensor: Optional[torch.Tensor]
        If provided, the padded ragged tensor will be stored in this tensor,
        otherwise a new tensor will be allocated.
    output_indptr: Optional[torch.Tensor]
        If provided, the padded indptr will be stored in this tensor,
        otherwise a new tensor will be allocated.

    Returns
    -------
    padded_ragged_tensor: torch.Tensor
        The padded ragged tensor, expected shape: (n_rows, padded_nnz, D)
    padded_indptr: torch.Tensor
        The padded indptr, expected shape: (n_rows + 1,)
    """
    # Get dimensions
    n_rows = indptr.shape[0] - 1
    nnz = ragged_tensor.shape[0]
    dim = ragged_tensor.shape[1]

    # First compute padded indptr
    if output_indptr is None:
        padded_indptr = torch.zeros_like(indptr)
    else:
        padded_indptr = output_indptr

    grid_size = triton.cdiv(n_rows, 128)
    _compute_padded_indptr[(grid_size,)](
        indptr, padded_indptr, n_rows, multiple_of, BLOCK_SIZE=128
    )

    # Perform exclusive scan to get final padded_indptr
    padded_indptr[1:] = torch.cumsum(padded_indptr[1:], dim=0)

    # Allocate padded tensor
    if output_ragged_tensor is None:
        total_padded_length = padded_indptr[-1].item()
        padded_ragged_tensor = torch.empty(
            (total_padded_length, dim),
            dtype=ragged_tensor.dtype,
            device=ragged_tensor.device,
        )
    else:
        padded_ragged_tensor = output_ragged_tensor

    # Pad the tensor
    _pad_ragged_tensor[(n_rows,)](
        ragged_tensor,
        padded_ragged_tensor,
        indptr,
        padded_indptr,
        n_rows,
        dim,
        BLOCK_SIZE=min(max_power_of_2_leq(dim), 16384),
        num_stages=2,
        fill_zeros=fill_zeros,
    )

    return padded_ragged_tensor, padded_indptr
