"""
Copyright (c) 2024 by FlashInfer team.

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
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.topk import gen_topk_module
from .utils import _get_cache_buf, register_custom_op, register_fake_op


def _resolve_deterministic_mode(deterministic: bool) -> int:
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"`deterministic` must be bool, but got {type(deterministic).__name__}."
        )
    return 1 if deterministic else 0


@functools.cache
def get_topk_module():
    module = gen_topk_module().build_and_load()

    @register_custom_op(
        "flashinfer::radix_topk", mutates_args=("row_states_buffer", "output_values")
    )
    def radix_topk(
        input: torch.Tensor,
        top_k: int,
        deterministic_mode: int,
        row_states_buffer: Optional[torch.Tensor],
        output_values: torch.Tensor,
    ) -> torch.Tensor:
        device = input.device
        # Supports float32, float16, bfloat16
        assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {input.dtype}, expected float32, float16, or bfloat16"
        )
        batch_size = input.size(0)
        output_indices = torch.empty(
            batch_size, top_k, dtype=torch.int32, device=device
        )
        module.radix_topk(
            input,
            output_indices,
            output_values,
            row_states_buffer,
            top_k,
            deterministic_mode,
        )
        return output_indices

    @register_fake_op("flashinfer::radix_topk")
    def _fake_radix_topk(
        input: torch.Tensor,
        top_k: int,
        deterministic_mode: int,
        row_states_buffer: Optional[torch.Tensor],
        output_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input.size(0)
        return torch.empty(batch_size, top_k, dtype=torch.int32, device=input.device)

    @register_custom_op(
        "flashinfer::radix_topk_page_table_transform",
        mutates_args=("row_states_buffer", "output_page_table"),
    )
    def radix_topk_page_table_transform(
        input: torch.Tensor,
        output_page_table: torch.Tensor,
        src_page_table: torch.Tensor,
        row_to_batch: Optional[torch.Tensor],
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic_mode: int,
    ) -> None:
        assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {input.dtype}, expected float32, float16, or bfloat16"
        )
        module.radix_topk_page_table_transform(
            input,
            output_page_table,
            src_page_table,
            row_to_batch,
            lengths,
            row_states_buffer,
            top_k,
            deterministic_mode,
        )

    @register_fake_op("flashinfer::radix_topk_page_table_transform")
    def _fake_radix_topk_page_table_transform(
        input: torch.Tensor,
        output_page_table: torch.Tensor,
        src_page_table: torch.Tensor,
        row_to_batch: Optional[torch.Tensor],
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic_mode: int,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::radix_topk_ragged_transform",
        mutates_args=("row_states_buffer", "output_indices"),
    )
    def radix_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic_mode: int,
    ) -> None:
        assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {input.dtype}, expected float32, float16, or bfloat16"
        )
        module.radix_topk_ragged_transform(
            input,
            output_indices,
            offsets,
            lengths,
            row_states_buffer,
            top_k,
            deterministic_mode,
        )

    @register_fake_op("flashinfer::radix_topk_ragged_transform")
    def _fake_radix_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic_mode: int,
    ) -> None:
        pass

    return SimpleNamespace(
        radix_topk=radix_topk,
        radix_topk_page_table_transform=radix_topk_page_table_transform,
        radix_topk_ragged_transform=radix_topk_ragged_transform,
        can_implement_filtered_topk=module.can_implement_filtered_topk,
    )


def can_implement_filtered_topk() -> bool:
    r"""Check if the GPU supports enough shared memory for FilteredTopK algorithm.

    FilteredTopK requires 128KB dynamic shared memory. This function checks if the
    current GPU's max shared memory per SM is sufficient.

    Returns
    -------
    bool
        True if GPU supports FilteredTopK, False otherwise.
    """
    return get_topk_module().can_implement_filtered_topk()


@flashinfer_api
def top_k(
    input: torch.Tensor,
    k: int,
    sorted: bool = False,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Radix-based Top-K selection.

    This function selects the top-k largest elements from each row of the input
    tensor. It uses an efficient radix-based selection algorithm that is
    particularly fast for large vocabularies.

    This is designed as a drop-in replacement for ``torch.topk`` with better
    performance for large tensors (vocab_size > 10000).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(batch_size, d)`` containing the values to select from.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    k : int
        Number of top elements to select from each row.
    sorted : bool, optional
        If True, the returned top-k elements will be sorted in descending order.
        Default is False (unsorted, which is faster).
    deterministic : bool, optional
        If ``True``, uses deterministic mode.
        If ``False``, uses non-deterministic mode for best performance.

    Returns
    -------
    values : torch.Tensor
        Tensor of shape ``(batch_size, k)`` containing the top-k values.
        Same dtype as input.
    indices : torch.Tensor
        Tensor of shape ``(batch_size, k)`` with int64 dtype containing the
        indices of the top-k elements.

    Note
    ----
    - Unlike ``torch.topk``, the default behavior returns unsorted results for
      better performance. Set ``sorted=True`` if you need sorted output.
    - The radix-based algorithm is O(n) in vocabulary size, compared to O(n log k)
      for heap-based methods, making it faster for large vocabularies.
    - For small vocabularies (< 1000), ``torch.topk`` may be faster.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 32000
    >>> k = 256
    >>> logits = torch.randn(batch_size, vocab_size, device="cuda")
    >>> values, indices = flashinfer.top_k(logits, k)
    >>> values.shape, indices.shape
    (torch.Size([4, 256]), torch.Size([4, 256]))

    With sorting enabled (for compatibility with torch.topk):

    >>> values_sorted, indices_sorted = flashinfer.top_k(logits, k, sorted=True)
    >>> # Values are now in descending order within each row

    See Also
    --------
    torch.topk : PyTorch's built-in top-k function
    sampling.top_k_mask_logits : Top-k masking for logits (sets non-top-k to -inf)
    sampling.top_k_renorm_probs : Top-k filtering and renormalization for probabilities
    """
    batch_size = input.size(0)
    device = input.device
    mode = _resolve_deterministic_mode(deterministic)

    # Allocate row_states buffer for multi-CTA path
    # 1MB is enough for any reasonable GPU (covers up to ~500 groups)
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{input.device}",
        1024 * 1024,  # 1MB
        input.device,
        zero_init=True,
    )

    # Allocate output_values for kernel to write directly
    output_values = torch.empty(batch_size, k, dtype=input.dtype, device=device)

    # Get indices using radix-based selection
    indices_int32 = get_topk_module().radix_topk(
        input, k, int(mode), row_states_buffer, output_values
    )

    # Convert to int64 for compatibility
    indices = indices_int32.long()

    if sorted:
        if deterministic:
            # Keep deterministic ordering for ties based on kernel output order.
            sorted_values, sort_indices = torch.sort(
                output_values, dim=-1, descending=True, stable=True
            )
            sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
        else:
            # Sort within each row by value (descending)
            sorted_values, sort_indices = torch.sort(
                output_values, dim=-1, descending=True
            )
            sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
        return sorted_values, sorted_indices

    return output_values, indices


# Alias for compatibility
topk = top_k


@flashinfer_api
def top_k_page_table_transform(
    input: torch.Tensor,
    src_page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_to_batch: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    r"""Fused Top-K selection + Page Table Transform for sparse attention.

    This function performs top-k selection on input scores and transforms the
    selected indices through a page table lookup in a single fused kernel.
    Used in sparse attention's second stage where selected KV cache positions
    need to be mapped through page tables.

    For each row i:
        output_page_table[i, j] = src_page_table[batch_idx, topk_indices[j]]

    where batch_idx is determined by row_to_batch[i] if provided, otherwise i.

    Parameters
    ----------
    input : torch.Tensor
        Input scores tensor of shape ``(num_rows, max_len)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    src_page_table : torch.Tensor
        Source page table of shape ``(batch_size, max_len)`` with dtype ``int32``.
    lengths : torch.Tensor
        Actual KV lengths per row of shape ``(num_rows,)`` with dtype ``int32``.
    k : int
        Number of top elements to select from each row.
    row_to_batch : Optional[torch.Tensor], optional
        Mapping from row index to batch index of shape ``(num_rows,)`` with
        dtype ``int32``. If None, uses 1:1 mapping (row_idx == batch_idx).
        Default is None.
    deterministic : bool, optional
        If ``True``, uses deterministic mode.
        If ``False``, uses non-deterministic mode for best performance.

    Returns
    -------
    output_page_table : torch.Tensor
        Output page table entries of shape ``(num_rows, k)`` with dtype ``int32``.
        Contains the gathered page table entries for the top-k indices.
        Positions beyond actual length are set to -1.

    Note
    ----
    - This is specifically designed for sparse attention's second stage.
    - If lengths[i] <= k, the output simply contains src_page_table[batch_idx, 0:lengths[i]]
      with remaining positions set to -1.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_rows = 8
    >>> max_len = 4096
    >>> k = 256
    >>> scores = torch.randn(num_rows, max_len, device="cuda", dtype=torch.float16)
    >>> src_page_table = torch.randint(0, 1000, (num_rows, max_len), device="cuda", dtype=torch.int32)
    >>> lengths = torch.full((num_rows,), max_len, device="cuda", dtype=torch.int32)
    >>> output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)
    >>> output.shape
    torch.Size([8, 256])
    """
    device = input.device
    num_rows = input.size(0)
    mode = _resolve_deterministic_mode(deterministic)

    # Allocate row_states buffer for multi-CTA path
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{device}",
        1024 * 1024,  # 1MB
        device,
        zero_init=True,
    )

    # Allocate output
    output_page_table = torch.empty(num_rows, k, dtype=torch.int32, device=device)

    get_topk_module().radix_topk_page_table_transform(
        input,
        output_page_table,
        src_page_table,
        row_to_batch,
        lengths,
        row_states_buffer,
        k,
        int(mode),
    )

    return output_page_table


@flashinfer_api
def top_k_ragged_transform(
    input: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    deterministic: bool = False,
) -> torch.Tensor:
    r"""Fused Top-K selection + Ragged Index Transform for sparse attention.

    This function performs top-k selection on input scores and transforms the
    selected indices by adding an offset in a single fused kernel.
    Used in sparse attention's second stage with ragged/variable-length KV cache.

    For each row i:
        output_indices[i, j] = topk_indices[j] + offsets[i]

    Parameters
    ----------
    input : torch.Tensor
        Input scores tensor of shape ``(num_rows, max_len)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    offsets : torch.Tensor
        Offset to add per row of shape ``(num_rows,)`` with dtype ``int32``.
    lengths : torch.Tensor
        Actual KV lengths per row of shape ``(num_rows,)`` with dtype ``int32``.
    k : int
        Number of top elements to select from each row.
    deterministic : bool, optional
        If ``True``, uses deterministic mode.
        If ``False``, uses non-deterministic mode for best performance.

    Returns
    -------
    output_indices : torch.Tensor
        Output indices of shape ``(num_rows, k)`` with dtype ``int32``.
        Contains the top-k indices plus offsets.
        Positions beyond actual length are set to -1.

    Note
    ----
    - This is specifically designed for sparse attention's second stage with
      ragged KV cache layout.
    - If lengths[i] <= k, the output contains [offsets[i], offsets[i]+1, ..., offsets[i]+lengths[i]-1]
      with remaining positions set to -1.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_rows = 8
    >>> max_len = 4096
    >>> k = 256
    >>> scores = torch.randn(num_rows, max_len, device="cuda", dtype=torch.float16)
    >>> offsets = torch.arange(0, num_rows * max_len, max_len, device="cuda", dtype=torch.int32)
    >>> lengths = torch.full((num_rows,), max_len, device="cuda", dtype=torch.int32)
    >>> output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)
    >>> output.shape
    torch.Size([8, 256])
    """
    device = input.device
    num_rows = input.size(0)
    mode = _resolve_deterministic_mode(deterministic)

    # Allocate row_states buffer for multi-CTA path
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{device}",
        1024 * 1024,  # 1MB
        device,
        zero_init=True,
    )

    # Allocate output
    output_indices = torch.empty(num_rows, k, dtype=torch.int32, device=device)

    get_topk_module().radix_topk_ragged_transform(
        input,
        output_indices,
        offsets,
        lengths,
        row_states_buffer,
        k,
        int(mode),
    )

    return output_indices
