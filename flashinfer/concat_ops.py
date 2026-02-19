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

import functools

import torch

from .api_logging import flashinfer_api


@functools.cache
def get_concat_mla_module():
    from .jit.dsv3_optimizations import gen_concat_mla_module

    return gen_concat_mla_module().build_and_load()


@flashinfer_api
def concat_mla_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> None:
    r"""Concatenate k_nope and k_rope tensors for MLA attention.



    This function efficiently concatenates:
      - k_nope: per-head nope values
      - k_rope: shared rope values (broadcast to all heads)

    Key optimizations:
      - Warp-based processing with software pipelining
      - Vectorized memory access (int2 for nope, int for rope)
      - L2 prefetching for next row while processing current
      - Register reuse for rope values across all heads in a chunk

    Parameters
    ----------
    k : torch.Tensor
        Output tensor, shape: ``[num_tokens, num_heads, nope_dim + rope_dim]``.
        Modified in-place.
    k_nope : torch.Tensor
        The nope part of k, shape: ``[num_tokens, num_heads, nope_dim]``.
    k_rope : torch.Tensor
        The rope part of k (shared), shape: ``[num_tokens, 1, rope_dim]``.
        This is broadcast to all heads.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_tokens = 2048
    >>> num_heads = 128
    >>> nope_dim = 128
    >>> rope_dim = 64
    >>> k = torch.empty(num_tokens, num_heads, nope_dim + rope_dim, dtype=torch.bfloat16, device="cuda")
    >>> k_nope = torch.randn(num_tokens, num_heads, nope_dim, dtype=torch.bfloat16, device="cuda")
    >>> k_rope = torch.randn(num_tokens, 1, rope_dim, dtype=torch.bfloat16, device="cuda")
    >>> flashinfer.concat_ops.concat_mla_k(k, k_nope, k_rope)

    Note
    ----
    This kernel is specifically optimized for:
    - ``num_heads = 128``
    - ``nope_dim = 128``
    - ``rope_dim = 64``
    """
    get_concat_mla_module().concat_mla_k(k, k_nope, k_rope)
