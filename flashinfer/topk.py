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
from typing import Optional, Tuple, Union

import torch

from .jit.topk import gen_topk_module
from .utils import _get_cache_buf, register_custom_op, register_fake_op

# RadixRowState size (histogram[2][256] + remaining_k + prefix + arrival_counter + output_counter)
# = 2*256*4 + 4 + 4 + 4 + 4 = 2064 bytes
RADIX_ROW_STATE_SIZE = 2064


@functools.cache
def get_topk_module():
    module = gen_topk_module().build_and_load()

    @register_custom_op("flashinfer::radix_topk", mutates_args=("row_states_buffer",))
    def radix_topk(
        input: torch.Tensor,
        top_k: int,
        row_states_buffer: Optional[torch.Tensor],
        output_values: Optional[torch.Tensor] = None,
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
            input, output_indices, output_values, row_states_buffer, top_k
        )
        return output_indices

    @register_fake_op("flashinfer::radix_topk")
    def _fake_radix_topk(
        input: torch.Tensor,
        top_k: int,
        row_states_buffer: Optional[torch.Tensor],
        output_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = input.size(0)
        return torch.empty(batch_size, top_k, dtype=torch.int32, device=input.device)

    return SimpleNamespace(
        radix_topk=radix_topk,
    )


def top_k(
    input: torch.Tensor,
    k: int,
    sorted: bool = False,
    return_values: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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
    return_values : bool, optional
        If True (default), return both values and indices.
        If False, return only indices (faster, avoids gather operation).

    Returns
    -------
    If return_values=True (default):
        values : torch.Tensor
            Tensor of shape ``(batch_size, k)`` containing the top-k values.
            Same dtype as input.
        indices : torch.Tensor
            Tensor of shape ``(batch_size, k)`` with int64 dtype containing the
            indices of the top-k elements.

    If return_values=False:
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
    - Setting ``return_values=False`` is faster when you only need indices,
      as it avoids the gather operation for values.

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

    Getting only indices (faster):

    >>> indices_only = flashinfer.top_k(logits, k, return_values=False)
    >>> indices_only.shape
    torch.Size([4, 256])

    See Also
    --------
    torch.topk : PyTorch's built-in top-k function
    """
    input.size(1)
    batch_size = input.size(0)
    device = input.device

    # Allocate row_states buffer for multi-CTA path
    # For single-CTA path this buffer is not used but we always allocate for simplicity
    # 1MB is enough for any reasonable GPU (covers up to ~500 groups)
    # zero_init=True ensures arrival_counter starts at 0 on first use
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{input.device}",
        1024 * 1024,  # 1MB
        input.device,
        zero_init=True,
    )

    # Allocate output_values for kernel to write directly
    output_values: Optional[torch.Tensor] = None
    if return_values:
        output_values = torch.empty(batch_size, k, dtype=input.dtype, device=device)

    # Get indices using radix-based selection (kernel writes values if output_values provided)
    indices_int32 = get_topk_module().radix_topk(
        input, k, row_states_buffer, output_values
    )

    # Convert to int64 for compatibility
    indices = indices_int32.long()

    if not return_values:
        return indices

    values = output_values

    if sorted:
        # Sort within each row by value (descending)
        sorted_values, sort_indices = torch.sort(values, dim=-1, descending=True)
        sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
        return sorted_values, sorted_indices

    return values, indices


# Alias for compatibility
topk = top_k
