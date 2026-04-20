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
import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.topk import gen_topk_module
from .utils import (
    _get_cache_buf,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
    get_shared_bytes_per_block_optin,
)


@functools.cache
def get_topk_module():
    module = gen_topk_module().build_and_load()

    @register_custom_op(
        "flashinfer::radix_topk", mutates_args=("row_states_buffer", "output_values")
    )
    def radix_topk(
        input: torch.Tensor,
        top_k: int,
        sorted_output: bool,
        deterministic: bool,
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
            sorted_output,
            deterministic,
        )
        return output_indices

    @register_fake_op("flashinfer::radix_topk")
    def _fake_radix_topk(
        input: torch.Tensor,
        top_k: int,
        sorted_output: bool,
        deterministic: bool,
        row_states_buffer: Optional[torch.Tensor],
        output_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input.size(0)
        return torch.empty(batch_size, top_k, dtype=torch.int32, device=input.device)

    @register_custom_op(
        "flashinfer::fast_topk_clusters_exact",
        mutates_args=("indices", "output_values", "cached_overflow"),
    )
    def _fast_topk_clusters_exact(
        logits: torch.Tensor,
        indices: torch.Tensor,
        output_values: Optional[torch.Tensor],
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_exact(
            logits,
            indices,
            output_values,
            histogram,
            cached_overflow,
            top_k,
            num_cached,
            num_clusters,
            pdl_enabled,
        )

    @register_fake_op("flashinfer::fast_topk_clusters_exact")
    def _fake_fast_topk_clusters_exact(
        logits: torch.Tensor,
        indices: torch.Tensor,
        output_values: Optional[torch.Tensor],
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::fast_topk_clusters_exact_page_table_transform",
        mutates_args=("indices", "cached_overflow"),
    )
    def _fast_topk_clusters_exact_page_table_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_exact_page_table_transform(
            logits,
            indices,
            seq_lens,
            page_table,
            histogram,
            cached_overflow,
            top_k,
            num_cached,
            num_clusters,
            pdl_enabled,
        )

    @register_fake_op("flashinfer::fast_topk_clusters_exact_page_table_transform")
    def _fake_fast_topk_clusters_exact_page_table_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::fast_topk_clusters_exact_ragged_transform",
        mutates_args=("indices", "cached_overflow"),
    )
    def _fast_topk_clusters_exact_ragged_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_exact_ragged_transform(
            logits,
            indices,
            seq_lens,
            offsets,
            histogram,
            cached_overflow,
            top_k,
            num_cached,
            num_clusters,
            pdl_enabled,
        )

    @register_fake_op("flashinfer::fast_topk_clusters_exact_ragged_transform")
    def _fake_fast_topk_clusters_exact_ragged_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        pass

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
        deterministic: bool,
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
            deterministic,
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
        deterministic: bool,
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
        deterministic: bool,
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
            deterministic,
        )

    @register_fake_op("flashinfer::radix_topk_ragged_transform")
    def _fake_radix_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic: bool,
    ) -> None:
        pass

    return SimpleNamespace(
        radix_topk=radix_topk,
        radix_topk_page_table_transform=radix_topk_page_table_transform,
        radix_topk_ragged_transform=radix_topk_ragged_transform,
        can_implement_filtered_topk=module.can_implement_filtered_topk,
        fast_topk_clusters_exact=_fast_topk_clusters_exact,
        fast_topk_clusters_exact_page_table_transform=_fast_topk_clusters_exact_page_table_transform,
        fast_topk_clusters_exact_ragged_transform=_fast_topk_clusters_exact_ragged_transform,
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


def roundup_kbyte(x):
    return (x + 1023) // 1024 * 1024


@functools.cache
def get_num_cached_for_topk(device, k):
    regs_per_thread = 32
    threads_per_block = 1024
    blocks_per_sm = 65536 // (threads_per_block * regs_per_thread)

    shared_per_block = (
        get_shared_bytes_per_block_optin(device) // blocks_per_sm
    )  # SMEM_CARVEOUT // blocks_per_sm

    buffers_used = (k + 5 + 3 * 256 + 8) * 4  # other shared memory for buffers
    # num_bytes = 2 * 2 * sizeof(int) * num_cached, double buffer on indices and values cache
    return (shared_per_block - buffers_used - 1024) // 16


def get_fast_topk_clusters(batch_size: int) -> int:
    # low batch size, allocate more clusters to get more parallelism
    # high batch size, more parallelism available per row
    if batch_size <= 32:
        return 8
    elif batch_size < 128:
        return 4
    elif batch_size < 256:
        return 2
    else:
        return 1


def topk_clusters_exact(
    logits, top_k, output_values=False, out_dtype=torch.int32, pdl=False
):
    assert out_dtype in (torch.int32, torch.int64), (
        "out_dtype must be torch.int32 or torch.int64"
    )
    batch_size, max_model_len = logits.shape
    indices = torch.empty(batch_size, top_k, dtype=out_dtype, device=logits.device)
    num_clusters = get_fast_topk_clusters(batch_size)
    if max_model_len < 8192:
        num_clusters = 1
    topk_global_overflow = max_model_len // num_clusters
    overflow_buf = torch.empty(
        batch_size,
        4 * topk_global_overflow * num_clusters,
        device=logits.device,
        dtype=torch.int32,
    )
    output_vals = None
    if output_values:
        output_vals = torch.empty(
            batch_size, top_k, dtype=logits.dtype, device=logits.device
        )

    num_cached = get_num_cached_for_topk(logits.device, top_k)
    get_topk_module().fast_topk_clusters_exact(
        logits,
        indices,
        output_vals,
        None,  # histogram
        overflow_buf,
        top_k,
        num_cached,  # num_cached
        num_clusters,
        pdl,
    )
    return indices, output_vals


def topk_clusters_page_table_transform(
    logits, seq_lens, src_page_table, top_k, pdl=False
):
    batch_size, max_model_len = logits.shape
    indices = torch.empty(batch_size, top_k, dtype=torch.int32, device=logits.device)
    num_clusters = get_fast_topk_clusters(batch_size)
    if max_model_len < 8192:
        num_clusters = 1
    topk_global_overflow = max_model_len // num_clusters
    overflow_buf = torch.empty(
        batch_size,
        4 * topk_global_overflow * num_clusters,
        device=logits.device,
        dtype=torch.int32,
    )
    num_cached = get_num_cached_for_topk(logits.device, top_k)
    get_topk_module().fast_topk_clusters_exact_page_table_transform(
        logits,
        indices,
        seq_lens,
        src_page_table,
        None,  # histogram
        overflow_buf,
        top_k,
        num_cached,  # num_cached
        num_clusters,
        pdl,
    )
    return indices


def topk_clusters_ragged_transform(logits, seq_lens, offsets, top_k, pdl=False):
    batch_size, max_model_len = logits.shape
    indices = torch.empty(batch_size, top_k, dtype=torch.int32, device=logits.device)
    num_clusters = get_fast_topk_clusters(batch_size)
    if max_model_len < 8192:
        num_clusters = 1
    topk_global_overflow = max_model_len // num_clusters
    overflow_buf = torch.empty(
        batch_size,
        4 * topk_global_overflow * num_clusters,
        device=logits.device,
        dtype=torch.int32,
    )
    num_cached = get_num_cached_for_topk(logits.device, top_k)
    get_topk_module().fast_topk_clusters_exact_ragged_transform(
        logits,
        indices,
        seq_lens,
        offsets,
        None,  # histogram
        overflow_buf,
        top_k,
        num_cached,  # num_cached
        num_clusters,
        pdl,
    )
    return indices


def can_use_clusters_topk(device, deterministic):
    algo = os.environ.get("FLASHINFER_TOPK_ALGO")
    cap = get_compute_capability(device)
    return (algo is None or algo == "clusters") and not deterministic and cap[0] == 10


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
        If True, uses deterministic mode.
        Default is False (non-deterministic, which is faster).

        Deterministic mode guarantees repeatable FlashInfer output ordering for
        the selected top-k set on a fixed input and system.

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

    Deterministic mode (bitwise-reproducible output):

    >>> values, indices = flashinfer.top_k(logits, k, deterministic=True)

    See Also
    --------
    torch.topk : PyTorch's built-in top-k function
    sampling.top_k_mask_logits : Top-k masking for logits (sets non-top-k to -inf)
    sampling.top_k_renorm_probs : Top-k filtering and renormalization for probabilities
    """
    batch_size = input.size(0)
    device = input.device

    if can_use_clusters_topk(input.device, deterministic):
        indices, output_values = topk_clusters_exact(
            input, k, output_values=True, out_dtype=torch.int64
        )
        if sorted:
            sorted_values, sort_indices = torch.sort(
                output_values, dim=-1, descending=True
            )
            sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
            return sorted_values, sorted_indices
        return output_values, indices

    # Allocate row_states buffer for multi-CTA path
    # 1MB is enough for any reasonable GPU (covers up to ~200 groups for deterministic
    # mode and ~300 groups for non-deterministic mode)
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{input.device}",
        1024 * 1024,  # 1MB
        input.device,
        zero_init=True,
    )

    # Allocate output_values for kernel to write directly
    output_values = torch.empty(batch_size, k, dtype=input.dtype, device=device)

    # For deterministic + sorted + k <= 2048: CUDA handles the stable value sort on device.
    sorted_cuda = sorted and deterministic and k <= 2048
    indices_int32 = get_topk_module().radix_topk(
        input, k, sorted_cuda, deterministic, row_states_buffer, output_values
    )

    # Convert to int64 for compatibility
    indices = indices_int32.long()

    if sorted and not sorted_cuda:
        # Sort within each row by value (descending)
        sorted_values, sort_indices = torch.sort(
            output_values, dim=-1, descending=True, stable=deterministic
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
        If True, uses deterministic mode.
        Default is False (non-deterministic, which is faster).

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

    if can_use_clusters_topk(input.device, deterministic) and row_to_batch is None:
        return topk_clusters_page_table_transform(input, lengths, src_page_table, k)

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
        deterministic,
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
        If True, uses deterministic mode.
        Default is False (non-deterministic, which is faster).

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

    if can_use_clusters_topk(input.device, deterministic):
        return topk_clusters_ragged_transform(input, lengths, offsets, k)

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
        deterministic,
    )

    return output_indices
