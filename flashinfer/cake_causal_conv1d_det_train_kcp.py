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

Deterministic context-parallel (KCP) packed width-4 depthwise causal convolution + SiLU training
(SM100 / SM103 / SM107), the Cake backend.

Every rank convolves its fragments of the packed batch; the first tokens of a fragment that continues a
document read the history rows of the preceding fragment, exchanged through the halo collective of the
KCP context the calling family's KCP context builder made (the context its own KCP step uses).
The per-rank launch sequences, the autograd Functions and the kernels of the ``kcp_jit`` registry live in
``flashinfer.experimental.cake.cake_causal_conv1d_det_train.kcp`` (imported on first use: importing this module
loads none of it); the single-rank entry points are in ``flashinfer.cake_causal_conv1d_det_train``::

    from flashinfer.cake_causal_conv1d_det_train_kcp import (
        cake_causal_conv1d_det_kcp,
        cake_causal_conv1d_det_kcp_multi,
    )

    # this rank's fragments, with the history rows of the preceding fragments exchanged
    # through the KCP context (FP32 taps, no bias)
    y_local = cake_causal_conv1d_det_kcp(x_local, weight, ctx)

    # a layer's q / k / v convolutions with one halo exchange per direction
    # (bitwise three single calls)
    q, k, v = cake_causal_conv1d_det_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional, Sequence

import torch

from .api_logging import flashinfer_api


def _kcp() -> ModuleType:
    """The KCP host module of the det backend package, imported on first use (torch.distributed, the KCP loader)."""
    from .experimental.cake.cake_causal_conv1d_det_train import kcp

    return kcp


@flashinfer_api
def cake_causal_conv1d_det_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
) -> torch.Tensor:
    """Autograd-enabled context-parallel packed width-4 depthwise causal convolution + SiLU of this rank's fragments.

    The first tokens of a fragment that continues a document read the history rows of the preceding fragment
    (exchanged through the context's halo collective) instead of zeros, so the result equals the single-rank
    :func:`flashinfer.cake_causal_conv1d_det_train.cake_causal_conv1d_det` of the whole batch on this rank's tokens.  The KCP path takes FP32 taps and no
    bias (the package's pre-round-6 contract).

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[S_local, C]`` or ``[1, S_local, C]`` contiguous packed activations of this rank's fragments, in the
        partition's order; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps.
    ctx : KCP context
        From the calling family's KCP context builder (the same context that family's KCP step of the rank uses).
    recompute_handle : CakeCausalConv1dDetKcpRecomputeHandle, optional
        Keyword-only, default ``None`` (unchanged behavior).  A handle (any object with an ``entries`` dict, such as
        ``flashinfer.cake_causal_conv1d_det_train_kcp.CakeCausalConv1dDetKcpRecomputeHandle``) makes the checkpoint recompute of
        this call run without the forward halo exchange: the original forward exchanges as always and stores its
        history rows (the saved set's own tensor, key ``"conv"``) in the handle; torch's recompute of the forward
        inside the backward reads them back, so ``y`` and the gradients are bitwise those of the same checkpoint
        without a handle.  The backward still exchanges.  Create one handle per convolution call and micro-step,
        outside the outermost checkpoint region, and pass it as an argument of the checkpointed function.  A
        forward outside the backward that finds an entry raises (a handle reused across micro-steps).  The forward
        halo exchange is a collective: every rank of ``ctx.group`` passes a handle for the same call or none, created
        at the same checkpoint nesting level.  A mismatch pairs one rank's recompute exchange with another rank's
        backward exchange and hangs or corrupts data; no check is made.  Never share one handle between two
        convolution calls (also not in nested regions), and never ``clear()`` or reuse a handle while a backward
        that needs it is pending.
    cooperative : bool, optional
        Keyword-only, default ``None`` (the package default,
        ``flashinfer.experimental.cake.cake_causal_conv1d_det_train.kcp.COOPERATIVE_DEFAULT``, ``False``).  The launch
        form of the backward's main kernel; the forward is the same either way.  ``False``: an ordinary launch one
        SM short of the co-resident CTA count, then a fixed-order reduce kernel on the same stream; it starts beside
        a small kernel of another stream (for example a communication kernel).  ``True``: one cooperative launch,
        which starts only when all its CTAs fit on the GPU at once, so it waits for any kernel of another stream
        that occupies an SM.  ``dx`` is bitwise the same in both forms.  ``dweight`` is reproducible and within the
        same FP64-oracle bound in each form, but its bits differ between the forms, so keep one form for runs whose
        ``dweight`` bits must match.  Every rank of ``ctx.group`` passes the same value: no collective depends on
        the form, but the ranks' ``dweight`` partials would otherwise mix two reduction orders.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward runs the head program, the backward halo exchange and the
        main ``dx`` + deterministic FP32 ``dweight`` program of this rank's fragments: with ``cooperative=False``
        (the default) an ordinary launch followed by a fixed-order reduce program on the same stream, with
        ``True`` one cooperative kernel.  With ``cooperative=False`` at more than one rank the main program runs
        while the exchange is in flight and a tail-row program writes the last rows of the fragments that receive
        rows after the wait (``dx`` and ``dweight`` keep the bits of the waiting launch).
    """

    return _kcp().cake_causal_conv1d_det_kcp(
        x_local,
        weight,
        ctx,
        recompute_handle=recompute_handle,
        cooperative=cooperative,
    )


@flashinfer_api
def cake_causal_conv1d_det_kcp_multi(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
) -> tuple[torch.Tensor, ...]:
    """Autograd-enabled context-parallel convolutions of several tensors of one layer (for example its q, k and v)
    with one halo exchange per direction.

    The result is bitwise ``tuple(cake_causal_conv1d_det_kcp(x, w, ctx) for x, w in zip(xs, weights))``: outputs and
    every gradient, ``dweight`` included (each tensor keeps its own kernels and fixed-order reduction).  The forward
    issues one halo collective instead of one per tensor (the history rows of all tensors travel side by side and are
    copied out after the wait), and so does the backward, so a layer has fewer global synchronization points.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[S_local, C_i]`` or ``[1, S_local, C_i]`` contiguous packed activations of this rank's
        fragments, in the partition's order; each ``C_i`` is a multiple of 8 (they may differ).
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : KCP context
        As in :func:`cake_causal_conv1d_det_kcp`.  Every rank of the group calls this function once per layer, in the
        same order, with the same tensor count and channel counts.  The ranks are not compared (that would cost a
        collective): a different count, channel count or order hangs the exchange or silently corrupts the history
        rows, as a mismatch of the single entry's per-rank contract does.
    recompute_handle : CakeCausalConv1dDetKcpRecomputeHandle, optional
        Keyword-only, default ``None`` (unchanged behavior).  One handle for the whole call: the original forward
        stores the ``n`` history pools under ``"conv"`` and torch's recompute inside the backward reads them back,
        so the recompute issues no collective; the backward still exchanges once.  The rules of
        :func:`cake_causal_conv1d_det_kcp` apply (one handle per call and micro-step, created outside the outermost
        checkpoint region; every rank passes one or none).  The same handle may carry the layer's ``"kda"`` entry;
        its ``"conv"`` entry serves one multi call or one single call, never both.
    cooperative : bool, optional
        As in :func:`cake_causal_conv1d_det_kcp` (default ``False``, the same value on every rank), one form for
        every tensor of the call; the result is bitwise the single calls with the same ``cooperative``.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    return _kcp().cake_causal_conv1d_det_kcp_multi(
        xs,
        weights,
        ctx,
        recompute_handle=recompute_handle,
        cooperative=cooperative,
    )


# The KCP route's types (the checkpoint recompute handle and the autograd Functions) resolve on first access, so
# importing this module does not import the KCP host module (torch.distributed, the KCP loader), like the entry
# points above.
_LAZY_TYPES = frozenset(
    {
        "CakeCausalConv1dDetKcpFunction",
        "CakeCausalConv1dDetKcpMultiFunction",
        "CakeCausalConv1dDetKcpRecomputeHandle",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_kcp(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_TYPES)


__all__ = [
    "CakeCausalConv1dDetKcpFunction",  # noqa: F822 - resolved by __getattr__
    "CakeCausalConv1dDetKcpMultiFunction",  # noqa: F822
    "CakeCausalConv1dDetKcpRecomputeHandle",  # noqa: F822
    "cake_causal_conv1d_det_kcp",
    "cake_causal_conv1d_det_kcp_multi",
]
