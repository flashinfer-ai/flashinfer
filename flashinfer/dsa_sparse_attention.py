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

import functools
import os
from typing import Optional

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the backend contract, the
# workspace, launch binding, autograd wrapper and JIT registration live in
# flashinfer.experimental.cake_dsa_train.  The backend takes every switch as
# a keyword argument and reads no environment variable; the variables below
# are this module's A/B knobs, resolved here and passed down.

_FEATURE = "DSA sparse-attention training (64 query heads, SM100/SM103/SM107)"

# ``auto`` (default): the registered record's size rule decides whether a
# backward with ``dkv_acc`` accumulates directly from the main stage; ``1``
# forces the direct path, ``0`` the accumulate-and-cast path.
DKV_DIRECT_ENV = "FLASHINFER_CAKE_DSA_DKV_DIRECT"
# Unset, empty or ``0`` (default): the values of ``dkv_dst_map`` are not
# checked; anything else validates them on every backward (one device sync).
CHECK_DST_MAP_ENV = "FLASHINFER_CAKE_DSA_CHECK_DST_MAP"
# A positive integer: the capacity of the backend's binding cache (64 by
# default).  Read when this module is imported, applied when it first reaches
# the backend.
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY"


def _dkv_direct_from_environment() -> Optional[bool]:
    """``FLASHINFER_CAKE_DSA_DKV_DIRECT`` as the backend's ``dkv_direct`` keyword:
    ``auto`` (default) is ``None`` (the record's size rule), ``1`` is ``True``
    (direct whenever the program registers the natural stages), ``0`` is
    ``False`` (never).  Any other value raises ``ValueError``."""
    mode = os.environ.get(DKV_DIRECT_ENV, "auto").strip() or "auto"
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"{DKV_DIRECT_ENV} must be auto, 0 or 1, got {mode!r}")
    return None if mode == "auto" else mode == "1"


def _check_dst_map_from_environment() -> bool:
    """``FLASHINFER_CAKE_DSA_CHECK_DST_MAP`` as the backend's ``check_dst_map``
    keyword: unset, empty or ``0`` is ``False`` (default), anything else
    ``True``."""
    return os.environ.get(CHECK_DST_MAP_ENV, "0") not in ("", "0")


def _binding_cache_capacity_from_environment() -> Optional[int]:
    """``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY`` as a positive integer,
    or ``None`` when unset or blank (the backend's configured capacity stays).
    Anything else raises ``ValueError``."""
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        capacity = int(raw)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(
            f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
        )
    return capacity


@functools.cache
def _cake_backend():
    # The backend is imported on first use (experimental containment: core
    # reaches kernels and their dependencies through a deferred import) and
    # the binding-cache capacity read at import is applied to it once.
    from .experimental.cake_dsa_train import cake_backend

    if _BINDING_CACHE_CAPACITY is not None:
        cake_backend.BINDING_CACHE.configure(capacity=_BINDING_CACHE_CAPACITY)
    return cake_backend


def _backend(backend: str):
    if backend != "cake":
        raise ValueError(
            "DSA sparse-attention training currently supports backend='cake'"
        )
    return _cake_backend()


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
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
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
        the kernel does not range-check the map (see ``check_dst_map``).
    dkv_direct : Optional[bool]
        How the backward accumulates dK / dV into ``dkv_acc`` when the
        registered program has the natural-layout main stages.  ``None``
        (default) reads ``FLASHINFER_CAKE_DSA_DKV_DIRECT``: ``auto`` (its
        default) applies the record's size rule -- the main stage adds into
        the caller's packed rows directly (no workspace accumulators, zero
        fill or cast launch) when the row has at least
        ``dkv_direct.min_keys_per_query`` (4) keys per query token --, ``1``
        forces the direct path whenever the stages are registered, ``0`` always
        takes the accumulate-and-cast path; any other value raises
        ``ValueError``.  ``True`` / ``False`` select the path explicitly and
        ignore the variable.  Both paths add the same FP32 contributions into
        the same rows (another summation order); without ``dkv_acc`` or the
        natural stages the setting has no effect.
    check_dst_map : Optional[bool]
        Validate the values of ``dkv_dst_map`` (every destination row in
        ``[0, S_dst)``) on every backward, at the cost of one device
        synchronization, raising ``ValueError`` on a violation.  ``None``
        (default) reads ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP``: unset, empty or
        ``0`` (its default) means no check, anything else enables it.  ``True``
        / ``False`` decide explicitly and ignore the variable.  Leave it off in
        production.

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
    (``flashinfer.experimental.cake_dsa_train.cake_backend.BINDING_CACHE``, 64
    bindings by default, least recently used first out;
    ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY`` -- a positive integer,
    read when this module is imported -- sets the capacity through
    ``BINDING_CACHE.configure(capacity=...)``).
    """
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
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
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
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
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
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
    dkv_direct : Optional[bool]
        As in :func:`dsa_sparse_attention`: ``None`` (default) reads
        ``FLASHINFER_CAKE_DSA_DKV_DIRECT`` (``auto`` = the record's size rule,
        ``1`` / ``0`` = force / disable the direct accumulation into
        ``dkv_acc``); ``True`` / ``False`` decide explicitly.
    check_dst_map : Optional[bool]
        As in :func:`dsa_sparse_attention`: ``None`` (default) reads
        ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP`` (unset, empty or ``0`` = no
        check); ``True`` validates the values of ``dkv_dst_map`` on every
        backward, ``False`` never does.
    """
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
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
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


# Read once at import: a malformed value fails here, naming the variable, not
# on the first call.  Applied to the backend's cache by ``_cake_backend``.
_BINDING_CACHE_CAPACITY = _binding_cache_capacity_from_environment()
