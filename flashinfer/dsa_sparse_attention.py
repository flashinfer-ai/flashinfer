"""
Copyright (c) 2026 by FlashInfer team.

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

from typing import Optional

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the backend contract, the
# workspace, launch binding, autograd wrapper and JIT registration live in
# flashinfer.experimental.cake_dsa_train.

_FEATURE = "DSA sparse-attention training (64 query heads, SM100/SM103)"


def _backend(backend: str):
    if backend != "cake":
        raise ValueError("DSA sparse-attention training currently supports backend='cake'")
    from .experimental.cake_dsa_train import cake_backend

    return cake_backend


@flashinfer_experimental_api(feature=_FEATURE)
def dsa_sparse_attention(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "cake",
):
    r"""Differentiable top-k sparse MLA attention with absorbed queries (DeepSeek
    Sparse Attention) for training, 64 query heads, SM100 / SM103 only.

    ``O = softmax(softmax_scale * (q_rope . k_rope^T + q_latent . kv_latent^T)) . kv_latent``
    over the keys a row's ``indices`` select (K = V = the latent).

    Parameters
    ----------
    q_latent : torch.Tensor
        BF16 ``[T, 64, 512]`` absorbed query latents; a view of a packed
        ``[T, 64, 576]`` query (row stride 576) is accepted.
    q_rope : torch.Tensor
        BF16 ``[T, 64, 64]`` query rope part (same packing rule).
    kv_latent : torch.Tensor
        BF16 ``[S, 512]`` key/value latent (shared K = V); a view of a packed
        ``[S, 576]`` cache is accepted.
    k_rope : torch.Tensor
        BF16 ``[S, 64]`` key rope part.
    indices : torch.Tensor
        int32 ``[T, topk]`` **global** key rows into ``kv_latent`` / ``k_rope``.
        ``-1`` or a value ``>= S`` marks an invalid slot; invalid slots may
        appear anywhere in the row.  Any positive ``topk``.
    topk_length : Optional[torch.Tensor]
        int32 ``[T]``; slots ``>= topk_length[t]`` are invalid regardless of
        their content.
    softmax_scale : Optional[float]
        Defaults to ``576 ** -0.5``.
    return_lse : bool
        Also return the natural-log logsumexp ``[T, 64]`` FP32 of the scaled
        scores over the valid keys (``-inf`` for fully masked rows).
    backend : str
        Only ``"cake"``.

    Returns
    -------
    out : torch.Tensor
        BF16 ``[T, 64, 512]``; zero for fully masked rows.
    lse : torch.Tensor
        Only with ``return_lse=True``.

    Gradients flow to ``q_latent``, ``q_rope``, ``kv_latent`` and ``k_rope``
    (``dq`` bitwise deterministic; ``dkv`` accumulated in FP32 with atomics,
    then cast to BF16).  BF16 operands into the tensor cores, FP32
    accumulation; the backward recomputes the scores from the BF16 inputs and
    forms the exact ``delta`` from the saved output residual.  The kernels
    apply no positional mask: the index rows define the key set.
    """
    return _backend(backend).dsa_sparse_attention(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def dsa_sparse_attention_varlen(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "cake",
):
    r"""Packed multi-document form of :func:`dsa_sparse_attention`.

    Documents are packed along the first dimension of the query tensors
    (``T = cu_seqlens_q[-1]``) and of the key tensors (``S = cu_seqlens_k[-1]``);
    query and key lengths may differ per document.  ``gather_kv_indices``
    ``[T, topk]`` int32 hold key positions relative to the row's document
    (``-1`` or ``>= seqlen_k[d]`` invalid) and are offset by ``cu_seqlens_k`` on
    device before the flat kernels run (this glue counts in the step time).
    ``max_seqlen_q`` / ``max_seqlen_k`` are accepted for signature parity with
    other varlen attention entry points and are not read on the host.  Other
    arguments and returns as in :func:`dsa_sparse_attention`.
    """
    return _backend(backend).dsa_sparse_attention_varlen(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
    )
