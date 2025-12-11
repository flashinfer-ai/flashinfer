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
from typing import Optional

import torch

from .jit.topk import gen_topk_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_topk_module():
    module = gen_topk_module().build_and_load()

    @register_custom_op("flashinfer::radix_topk", mutates_args=())
    def radix_topk(
        input: torch.Tensor,
        top_k: int,
        starts: Optional[torch.Tensor],
        ends: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = input.device
        input = input.float()
        batch_size = input.size(0)
        output_indices = torch.empty(
            batch_size, top_k, dtype=torch.int32, device=device
        )
        module.radix_topk(input, output_indices, starts, ends, top_k)
        return output_indices

    @register_fake_op("flashinfer::radix_topk")
    def _fake_radix_topk(
        input: torch.Tensor,
        top_k: int,
        starts: Optional[torch.Tensor],
        ends: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = input.size(0)
        return torch.empty(batch_size, top_k, dtype=torch.int32, device=input.device)

    return SimpleNamespace(
        radix_topk=radix_topk,
    )


def radix_topk(
    input: torch.Tensor,
    top_k: int,
    starts: Optional[torch.Tensor] = None,
    ends: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Radix-based Top-K selection using a multi-pass histogram algorithm.

    This function efficiently selects the indices of the top-k largest elements
    from each row of the input tensor using a radix-based selection algorithm.
    The algorithm uses multiple passes with 8-bit radix buckets to progressively
    filter candidates.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(batch_size, d)`` containing the values to select from.
        Currently only float32 is supported.
    top_k : int
        Number of top elements to select from each row.
    starts : Optional[torch.Tensor]
        Optional tensor of shape ``(batch_size,)`` with int32 dtype specifying the
        start index for each row. If None, defaults to 0 for all rows.
    ends : Optional[torch.Tensor]
        Optional tensor of shape ``(batch_size,)`` with int32 dtype specifying the
        end index (exclusive) for each row. If None, defaults to d for all rows.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, top_k)`` with int32 dtype containing the
        indices of the top-k largest elements in each row. The indices are not
        guaranteed to be sorted.

    Note
    ----
    - The algorithm uses shared memory for intermediate storage, with a maximum
      of 4096 candidates per round. For very large top_k values, accuracy may
      be slightly reduced.
    - This implementation is particularly efficient for large vocabularies
      (d > 10000) and moderate top_k values (256-2048).

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 32000
    >>> top_k = 256
    >>> logits = torch.randn(batch_size, vocab_size, device="cuda")
    >>> indices = flashinfer.topk.radix_topk(logits, top_k)
    >>> indices.shape
    torch.Size([4, 256])

    With custom start/end indices:

    >>> starts = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    >>> ends = torch.full((batch_size,), vocab_size // 2, dtype=torch.int32, device="cuda")
    >>> indices = flashinfer.topk.radix_topk(logits, top_k, starts=starts, ends=ends)
    """
    return get_topk_module().radix_topk(input, top_k, starts, ends)
