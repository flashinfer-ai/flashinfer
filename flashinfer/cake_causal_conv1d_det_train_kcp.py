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
(SM100 / SM103 / SM107), the Cake backend.  ``flashinfer.cake_causal_conv1d_det_train`` keeps the
names these entry points had before (``cake_causal_conv1d_kcp``, ``cake_causal_conv1d_kcp_multi``,
``KcpRecomputeHandle``) as compatibility aliases of the objects defined here.

Every rank convolves its fragments of the packed batch; the first tokens of a fragment that continues a
document read the history rows of the preceding fragment, exchanged through the halo collective of the
context-parallel context of the KDA step (``flashinfer.cake_kda_det_train_kcp.build_context_det_kcp``).
The per-rank launch sequences, the autograd Functions and the kernels of the ``kcp_jit`` registry live in
``flashinfer.cake_causal_conv1d_det_train_backend.kcp``; the single-rank entry points are in
``flashinfer.cake_causal_conv1d_det_train``::

    from flashinfer.cake_causal_conv1d_det_train_kcp import (
        cake_causal_conv1d_det_kcp,
        cake_causal_conv1d_det_multi_kcp,
    )

    # this rank's fragments, with the history rows of the preceding fragments exchanged
    # through the KCP context (FP32 taps, no bias)
    y_local = cake_causal_conv1d_det_kcp(x_local, weight, ctx)

    # a layer's q / k / v convolutions with one halo exchange per direction
    # (bitwise three single calls)
    q, k, v = cake_causal_conv1d_det_multi_kcp((xq, xk, xv), (wq, wk, wv), ctx)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .api_logging import flashinfer_api


@flashinfer_api
def cake_causal_conv1d_det_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
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
    ctx : ContextDetKcp
        From ``flashinfer.cake_kda_det_train_kcp.build_context_det_kcp`` (the same context the KDA step of the rank uses).
    recompute_handle : RecomputeHandleDetKcp, optional
        Keyword-only, default ``None`` (unchanged behavior).  A handle (any object with an ``entries`` dict, such as
        ``flashinfer.cake_causal_conv1d_det_train_kcp.RecomputeHandleDetKcp``) makes the checkpoint recompute of
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

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward runs the head program, the backward halo exchange and the
        fused ``dx`` + deterministic FP32 ``dweight`` kernel of this rank's fragments.
    """

    from .cake_causal_conv1d_det_train_backend import kcp as _kcp  # noqa: PLC0415

    return _kcp.cake_causal_conv1d_det_kcp(
        x_local, weight, ctx, recompute_handle=recompute_handle
    )


@flashinfer_api
def cake_causal_conv1d_det_multi_kcp(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
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
    ctx : ContextDetKcp
        As in :func:`cake_causal_conv1d_det_kcp`.  Every rank of the group calls this function once per layer, in the
        same order, with the same tensor count and channel counts.  The ranks are not compared (that would cost a
        collective): a different count, channel count or order hangs the exchange or silently corrupts the history
        rows, as a mismatch of the single entry's per-rank contract does.
    recompute_handle : RecomputeHandleDetKcp, optional
        Keyword-only, default ``None`` (unchanged behavior).  One handle for the whole call: the original forward
        stores the ``n`` history pools under ``"conv"`` and torch's recompute inside the backward reads them back,
        so the recompute issues no collective; the backward still exchanges once.  The rules of
        :func:`cake_causal_conv1d_det_kcp` apply (one handle per call and micro-step, created outside the outermost
        checkpoint region; every rank passes one or none).  The same handle may carry the layer's ``"kda"`` entry;
        its ``"conv"`` entry serves one multi call or one single call, never both.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    from .cake_causal_conv1d_det_train_backend import kcp as _kcp  # noqa: PLC0415

    return _kcp.cake_causal_conv1d_det_multi_kcp(
        xs, weights, ctx, recompute_handle=recompute_handle
    )


# ``RecomputeHandleDetKcp`` (the KCP route's checkpoint recompute handle) resolves on first access, so importing this
# module does not import the KCP host module (torch.distributed, the KCP loader), like the entry points above.
def __getattr__(name: str) -> Any:
    if name == "RecomputeHandleDetKcp":
        from .cake_causal_conv1d_det_train_backend.kcp import RecomputeHandleDetKcp  # noqa: PLC0415

        return RecomputeHandleDetKcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"RecomputeHandleDetKcp"})


__all__ = [
    "RecomputeHandleDetKcp",  # noqa: F822 - resolved by __getattr__
    "cake_causal_conv1d_det_kcp",
    "cake_causal_conv1d_det_multi_kcp",
]
