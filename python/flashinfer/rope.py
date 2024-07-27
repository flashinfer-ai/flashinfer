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

import torch

# mypy: disable-error-code="attr-defined"
try:
    from . import _kernels
except ImportError as e:
    import os
    import logging

    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.
    """
    return _kernels.apply_rope_inplace(q, k, indptr, offsets, rope_scale, rope_theta)


def apply_llama31_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rope_scale: float = 8,
    rope_theta: float = 1e4,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default.
    """
    return _kernels.apply_llama31_rope_inplace(
        q,
        k,
        indptr,
        offsets,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
