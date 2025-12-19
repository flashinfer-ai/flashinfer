"""
Copyright (c) 2024-2025 by FlashInfer team.

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
from typing import Tuple

import torch

from .jit.moe_topk import gen_moe_warp_topk_module
from .utils import register_custom_op


@functools.cache
def get_moe_warp_topk_module():
    module = gen_moe_warp_topk_module().build_and_load()

    @register_custom_op(
        "flashinfer::moe_warp_topk", mutates_args=("output_values", "output_indices")
    )
    def moe_warp_topk(
        input: torch.Tensor,
        k: int,
        output_values: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> None:
        module.moe_warp_topk(input, output_values, output_indices, k)

    return SimpleNamespace(
        moe_warp_topk=moe_warp_topk,
    )


def moe_top_k(
    input: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Warp-level Top-K selection optimized for MoE routing.

    This function selects the top-k largest elements from each row of the input
    tensor using warp-level reduction. It is optimized for small vocabulary sizes
    typical in MoE routing (num_experts <= 512).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(num_tokens, num_experts)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
        num_experts must be <= 512.
    k : int
        Number of top elements to select from each row.
        Must be <= 16.

    Returns
    -------
    values : torch.Tensor
        Tensor of shape ``(num_tokens, k)`` containing the top-k values.
        Same dtype as input.
    indices : torch.Tensor
        Tensor of shape ``(num_tokens, k)`` with int32 dtype containing the
        indices of the top-k elements.

    Note
    ----
    - This kernel uses warp-level reduction and is optimized for small N (MoE routing).
    - For large vocabularies (N > 10000), use ``flashinfer.top_k`` instead.
    - Results are returned in descending order by value.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_tokens = 4096
    >>> num_experts = 256
    >>> k = 8
    >>> router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.bfloat16)
    >>> values, indices = flashinfer.moe_top_k(router_logits, k)
    >>> values.shape, indices.shape
    (torch.Size([4096, 8]), torch.Size([4096, 8]))
    """
    num_tokens, num_experts = input.shape
    device = input.device

    if num_experts > 512:
        raise ValueError(
            f"moe_top_k: num_experts ({num_experts}) must be <= 512. "
            "Use flashinfer.top_k for larger vocabularies."
        )
    if k > 16:
        raise ValueError(f"moe_top_k: k ({k}) must be <= 16. ")

    output_values = torch.empty(num_tokens, k, dtype=input.dtype, device=device)
    output_indices = torch.empty(num_tokens, k, dtype=torch.int32, device=device)

    get_moe_warp_topk_module().moe_warp_topk(input, k, output_values, output_indices)

    return output_values, output_indices
