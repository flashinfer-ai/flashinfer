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

_FEATURE = "DSA sparse-attention training (64 query heads, SM100/SM103/SM107)"


def _backend(backend: str):
    if backend != "cake":
        raise ValueError(
            "DSA sparse-attention training currently supports backend='cake'"
        )
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
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
):
    r"""Differentiable top-k sparse MLA attention with absorbed queries (DeepSeek
    Sparse Attention) for training, 64 query heads, SM100 / SM103 / SM107 only.

    ``O = softmax(softmax_scale * (q_rope . k_rope^T + q_latent . kv_latent^T)) . kv_latent``
    over the keys a row's ``indices`` select (K = V = the latent).

    Strided inputs are consumed in place (no host copy): the kernels read the
    query operands through TMA descriptors encoded from the tensor's own head
    and token strides and gather the key operands by row stride.  Admitted
    layouts: unit innermost stride; head / token / key-row strides that are
    positive multiples of 8 elements (16 bytes, the TMA descriptor stride
    granule); head stride and key row stride at least the slice width; a
    16-byte-aligned base address.  Outputs equal the contiguous path bitwise
    (``out``, ``lse``, ``dq``) or within the FP32 reduction spread (``dkv``).

    Parameters
    ----------
    q_latent : torch.Tensor
        BF16 ``[T, 64, 512]`` absorbed query latents; contiguous or a view of
        a packed ``[T, 64, 576]`` query (head stride 576).
    q_rope : torch.Tensor
        BF16 ``[T, 64, 64]`` query rope part; contiguous, a view of the packed
        ``[T, 64, 576]`` query, or the ``192:256`` channel slice of the
        pre-absorption ``[T, 64, 256]`` query (head stride 256, token stride
        16384, storage offset 192 elements).
    kv_latent : torch.Tensor
        BF16 ``[S, 512]`` key/value latent (shared K = V); columns ``0:512`` of
        a packed ``[S, 576]`` or ``[S, 704]`` row (a 128-channel indexer key
        stored alongside) are accepted.
    k_rope : torch.Tensor
        BF16 ``[S, 64]`` key rope part; columns ``512:576`` of the same packed
        row are accepted.
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
    key_passes : Optional[int]
        Number of key-range passes of the backward's main stage.  ``None``
        (default) applies the registered policy: the key range is split into
        ``ceil(S * 2304 B / 100 MiB)`` passes when that exceeds one and the
        whole row fits the pass workspace budget (``T <= 4224`` at top-k 2048),
        so that each pass's FP32 dK/dV accumulator slice stays L2-resident;
        otherwise a single pass runs.  ``1`` forces the single pass; a larger
        value forces that many passes (``<= S``).  The passes carry dQ through
        an FP32 partial in the workspace (``T * 147,456`` B plus ``T * (4 *
        topk + 4)`` B, ~608 MiB at ``T = 4096``, top-k 2048); dQ stays bitwise
        deterministic and the dK/dV accumulation is unchanged.
    backend : str
        Only ``"cake"``.
    dkv_acc : Optional[torch.Tensor]
        Caller-owned FP32 ``[S_dst, >= 576]`` buffer (row stride a multiple of
        4 elements, 16-byte aligned) into which the backward accumulates the
        key gradients in place (``+=``; the caller zeroes it): latent
        dimensions in columns ``0:512``, rope dimensions in ``512:576``.  When
        given, the ``kv_latent`` / ``k_rope`` gradients of the autograd call
        are ``None`` and no BF16 cast runs.
    dkv_dst_map : Optional[torch.Tensor]
        int32 ``[S]`` destination row of ``dkv_acc`` for every key row
        (default identity).  Repeated destinations are allowed (the rows of a
        context-parallel window that map onto one parameter row) and are
        combined with FP32 atomics.  Every value must lie in ``[0, S_dst)``;
        the kernel does not range-check the map.  Setting
        ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP=1`` validates the values on every
        call (one device synchronization) and raises ``ValueError`` otherwise.

    Returns
    -------
    out : torch.Tensor
        BF16 ``[T, 64, 512]``; zero for fully masked rows.
    lse : torch.Tensor
        Only with ``return_lse=True``.

    Gradients flow to ``q_latent``, ``q_rope``, ``kv_latent`` and ``k_rope``
    (``dq`` bitwise deterministic; ``dkv`` accumulated in FP32 with atomics,
    then cast to BF16, or accumulated into ``dkv_acc``).  BF16 operands into the tensor cores, FP32
    accumulation; the backward recomputes the scores from the BF16 inputs and
    forms the exact ``delta`` from the saved output residual.  The kernels
    apply no positional mask: the index rows define the key set.

    Host side: the inputs are validated and bound on the first call for an
    input binding (``data_ptr``, shape, stride and dtype of every input plus
    the scale); later calls with the same binding launch from the remembered
    argument plans with freshly allocated outputs
    (``flashinfer.experimental.cake_dsa_train.cake_backend.BINDING_CACHE``;
    ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE=0`` disables it).
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
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
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
    causal: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
):
    r"""Packed multi-document form of :func:`dsa_sparse_attention` (SM100 /
    SM103 / SM107).

    Documents are packed along the first dimension of the query tensors
    (``T = cu_seqlens_q[-1]``) and of the key tensors (``S = cu_seqlens_k[-1]``);
    ``cu_seqlens_q`` and ``cu_seqlens_k`` are independent and query and key
    lengths may differ per document: the query segment is the tail of its key
    prefix, so query ``local_q`` of document ``d`` sits at key position
    ``(seqlen_k[d] - seqlen_q[d]) + local_q`` (a context-parallel tail or a
    GLM-style packed batch).  ``gather_kv_indices`` ``[T, topk]`` int32 hold
    key positions relative to the row's document (``-1`` or ``>= seqlen_k[d]``
    invalid) and are offset by ``cu_seqlens_k`` on device before the flat
    kernels run (this glue counts in the step time).  Zero-length query
    segments contribute no rows; a row whose every slot is invalid gives
    ``out = 0``, ``lse = -inf`` and zero gradients.  The kernels apply no
    positional mask of their own; the offset index rows define the key sets.
    ``max_seqlen_q`` / ``max_seqlen_k`` are accepted for signature parity with
    other varlen attention entry points and are not read on the host.  Other
    arguments and returns as in :func:`dsa_sparse_attention`; the strided
    query slices and packed key rows described there are accepted here too.

    Parameters
    ----------
    causal : bool
        With ``False`` (default) the index row is taken as is -- only ``-1`` /
        out-of-range slots are dropped -- exactly as in the first release.
        With ``True`` a slot that selects a key after the query's own position
        (``idx > (seqlen_k[d] - seqlen_q[d]) + local_q``) is invalid too, so a
        top-k selector need not enforce causality itself (the rule GLM-style
        trainers with query segments that are tails of their key prefix rely
        on).
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
        causal=causal,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
    )
