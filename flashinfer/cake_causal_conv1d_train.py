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

Cake packed width-4 depthwise causal convolution + SiLU training forward /
backward (SM100 / SM103 / SM107).

The input is one packed BF16 ``[S, C]`` (or ``[1, S, C]``) activation whose
documents ``cu_seqlens`` (``int32 [num_docs + 1]`` device prefix sums)
delimit; every document is convolved on its own, with zero history before its
first token.  With taps ``w[C, 4]`` (``w[c, 3]`` multiplies the current token)
and an optional per-channel bias ``b[C]``, the forward is, in FP32 with one
BF16 rounding,

    pre[t, c] = sum_k w[c, k] * x[t - 3 + k, c] + b[c]    # x of another document is 0
    y[t, c]   = bf16(pre * sigmoid(pre))

The taps and the bias are FP32 or BF16, in any combination (BF16 is widened to
FP32 on load; the arithmetic is FP32 either way); the bias is added last, after
the four taps, which is flash-linear-attention's order.  The backward
recomputes ``pre`` in FP32 from ``x`` and evaluates the SiLU derivative in
FP32,

    dpre[t, c]    = dy[t, c] * s * (1 + p * (1 - s)),  s = sigmoid(p)
    dx[t, c]      = bf16(sum_k w[c, k] * dpre[t + 3 - k, c])   # dpre of another document is 0
    dweight[c, k] = sum_t dpre[t, c] * x[t - 3 + k, c]         # fixed order
    dbias[c]      = sum_t dpre[t, c]                            # with a bias

at ``p`` chosen by ``silu_grad_preactivation``: ``"bf16"`` (default), ``p =
bf16(pre)``, the rounded value the forward stores and the boundary
flash-linear-attention's backward uses (``dweight`` lands within about 3e-5
rel-L2 of the exact FP64 gradient); ``"fp32"``, ``p = pre`` unrounded, with
the final ``dweight`` reduction in FP64 (within about 2e-7; a slower program).
``dx`` is within one BF16 ulp of exact under both.  The backward is two
ordinary launches on one stream: a persistent main program writes ``dx`` and
one FP32 partial row of ``dweight`` (and ``dbias``) per CTA into a workspace,
and a reduce program launched programmatically dependent on it sums the rows
in a fixed order (no floating-point atomics), so the gradients are bitwise
reproducible for a given ``(S, C, device)`` and route.  Neither launch is
cooperative: the backward keeps its speed while kernels of other streams
(NCCL collectives) are resident, which a whole-device cooperative grid had to
wait for.  Nothing is saved from the forward but ``x``, ``w``, ``b`` and
``cu_seqlens``.

The forward is one ordinary launch: the co-resident CTAs of each channel tile
take the tile's tokens in chunks from int32 counters on the device (the CTAs
that finish first take the tail), so ``y`` is the same bits whatever CTA
computes a chunk.  The counters live in a forward workspace of
``cake_causal_conv1d_train_forward_workspace_bytes(S, C, device)`` bytes that is
zero before its first launch and that every launch leaves zero, so one buffer
serves every later forward call of one stream, CUDA Graph replays included.  A
caller may own it (``workspace=``); otherwise the package keeps one zeroed
buffer per (device, current stream), and a call inside a CUDA Graph capture
allocates a zeroed buffer of its own.  Two launches that may run at once need
two buffers.

Inputs and outputs are contiguous (``C`` a multiple of 8); ``dweight`` and
``dbias`` are FP32 or BF16 (the parameter's dtype when not preallocated).
``S``, ``C`` and the document count are runtime scalars: they may change
between calls, the host never reads ``cu_seqlens`` (no synchronization), ``S
== 0`` returns empty outputs without launching, and the calls are CUDA Graph
capturable once the libraries are built.

A call's arguments (bias, BF16 taps or gradient, the SiLU' policy) select its
route, and within a route the channel count selects one of the architecture's
programs through a host-side rule of the backend's registry (never the data),
so a given ``(S, C, device)`` and route always run the same program.  The
workspace size is the maximum over every backward program of every route, so
one buffer of ``cake_causal_conv1d_train_backward_workspace_bytes(S, C,
device)`` serves whichever program a call selects.  Before capturing a CUDA
Graph, build every library of the device's architecture (the backend's
``cake_jit.specs_for_arch``) so that no JIT build happens inside the capture;
the captured graph holds the program selected for its shape and route.

Usage::

    y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens)
    dx, dweight = cake_causal_conv1d_train_backward(x, weight, cu_seqlens, dy)

    # a bias (FP32 or BF16, like the taps) adds a third output
    y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens, bias=bias)
    dx, dweight, dbias = cake_causal_conv1d_train_backward(x, weight, cu_seqlens, dy, bias=bias)

    # SiLU' at the unrounded FP32 preactivation for a ~2e-7 rel-L2 weight gradient
    dx, dweight = cake_causal_conv1d_train_backward(
        x, weight, cu_seqlens, dy, silu_grad_preactivation="fp32"
    )

    # autograd (saves x, weight, bias and cu_seqlens only)
    y = cake_causal_conv1d(x, weight, cu_seqlens, bias)

    # caller-owned backward workspace (per-CTA FP32 partials, no zeroing needed)
    ws = cake_causal_conv1d_train_backward_workspace(S, C, x.device)
    dx, dweight = cake_causal_conv1d_train_backward(x, weight, cu_seqlens, dy, workspace=ws)

    # caller-owned forward workspace (zeroed int32 token-chunk counters, left zero
    # by every launch: keep one per stream and reuse it for every later call)
    fws = cake_causal_conv1d_train_forward_workspace(S, C, x.device)
    y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens, workspace=fws)

    # context-parallel training: this rank's fragments of the packed batch, with the
    # history rows of the preceding fragments exchanged through the KCP context
    # (FP32 taps, no bias)
    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from . import cake_causal_conv1d_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_causal_conv1d_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeCausalConv1dFunction,
    cake_causal_conv1d_train_backward_workspace,
    cake_causal_conv1d_train_backward_workspace_bytes,
    cake_causal_conv1d_train_forward_workspace,
    cake_causal_conv1d_train_forward_workspace_bytes,
    is_cake_causal_conv1d_train_supported,
)

# The three stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.  The signatures mirror the
# backend's exactly (a test asserts it).


@flashinfer_api
def cake_causal_conv1d_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the
        current token, ``weight[c, 0]`` the token three positions back.  Any
        other width is rejected.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths (``cu_seqlens[0] == 0``, ``cu_seqlens[-1] == S``; a repeated
        value is an empty document).  The host reads its metadata only: no
        element is copied to the host and no stream is synchronized.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias, added after the four
        taps; its dtype is independent of ``weight``'s.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous.
    workspace : torch.Tensor, optional
        Caller-owned ``int32`` buffer from
        :func:`cake_causal_conv1d_train_forward_workspace` (the token-chunk
        counters: zero before its first launch, left zero by every launch, so
        one buffer serves every later call of one stream, CUDA Graph replays
        included; two launches that may run at once need two buffers).  Its
        size (:func:`cake_causal_conv1d_train_forward_workspace_bytes`) is the
        most over every forward route.  When omitted the package uses a zeroed
        buffer it keeps per (device, current stream), or a zeroed buffer of the
        call's own inside a CUDA Graph capture.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape: ``y[t, c] = bf16(silu(sum_k weight[c, k] *
        x[t - 3 + k, c] + bias[c]))`` with zero history before each document's
        first token.  ``S == 0`` returns the empty output without launching.
    """

    return _backend.cake_causal_conv1d_train_forward(
        x, weight, cu_seqlens, bias=bias, out=out, workspace=workspace
    )


@flashinfer_api
def cake_causal_conv1d_train_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dy: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    silu_grad_preactivation: str = "bf16",
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    dbias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward.

    Recomputes the FP32 preactivation from ``x``, evaluates the SiLU derivative
    at it -- rounded to BF16 (``silu_grad_preactivation="bf16"``, the default)
    or unrounded (``"fp32"``) -- and produces ``dx``, the weight gradient and,
    with a bias, the bias gradient in two ordinary launches on the current
    stream: the main program writes ``dx`` and per-CTA partial rows of the
    gradients, and a reduce program launched programmatically dependent on it
    sums the rows in a fixed order (no cooperative grid, so kernels resident
    on other streams never stall the call).

    The call's arguments select the route (bias, BF16 among ``weight`` /
    ``dweight``, the SiLU' policy) and the channel count selects the route's
    program through a host-side rule of the backend's registry (never the
    data), so a given ``(S, C, device)`` and route always run the same program.
    Before capturing a CUDA Graph, build every library of the device's
    architecture (the backend's ``cake_jit.specs_for_arch``) so that no JIT
    build happens inside the capture; the captured graph holds the program
    selected for its shape and route.

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs (same contracts as
        :func:`cake_causal_conv1d_train_forward`); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    silu_grad_preactivation : {"bf16", "fp32"}
        ``"bf16"``: SiLU' at the BF16-rounded recomputed preactivation (the
        boundary flash-linear-attention's backward uses; ``dweight`` within
        about 3e-5 rel-L2 of the exact FP64 gradient).  ``"fp32"``: SiLU' at
        the unrounded FP32 preactivation with an FP64 final reduction (about
        2e-7; a slower program).  ``dx`` is within one BF16 ulp of exact under
        both.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_causal_conv1d_train_backward_workspace` (the per-CTA FP32
        ``dweight`` / ``dbias`` partials; fully written by the kernel, so it
        needs no zeroing); allocated when omitted.  Its size
        (:func:`cake_causal_conv1d_train_backward_workspace_bytes`) is the
        maximum over every backward program of every route, so one buffer
        serves whichever program a call selects.
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (each FP32 or BF16; the parameter's dtype when omitted), contiguous.
        ``dbias`` needs ``bias``.

    Returns
    -------
    ``(dx, dweight)``, with a third element ``dbias`` only when ``bias`` is
    given: BF16 ``dx`` of ``x``'s shape, ``dx[t, c] = bf16(sum_k weight[c, k] *
    dpre[t + 3 - k, c])`` over the tokens of the same document, the ``[C, 4]``
    ``dweight[c, k] = sum_t dpre[t, c] * x[t - 3 + k, c]`` and the ``[C]``
    ``dbias[c] = sum_t dpre[t, c]``, each reduced in a fixed order, so they are
    bitwise reproducible for a given ``(S, C, device)`` and route.  ``S == 0``
    returns the empty ``dx`` and zero gradients without launching.
    """

    return _backend.cake_causal_conv1d_train_backward(
        x,
        weight,
        cu_seqlens,
        dy,
        bias=bias,
        silu_grad_preactivation=silu_grad_preactivation,
        workspace=workspace,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
    )


@flashinfer_api
def cake_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations on a
        supported CUDA device.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias.
    silu_grad_preactivation : {"bf16", "fp32"}
        The weight gradient's SiLU' policy (see
        :func:`cake_causal_conv1d_train_backward`).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.  The backward saves only ``x``, ``weight``,
        ``bias`` and ``cu_seqlens`` and launches the fused ``dx`` +
        deterministic ``dweight`` (+ ``dbias``) kernel of
        :func:`cake_causal_conv1d_train_backward`; the parameters' gradients
        come in the parameters' dtypes.
    """

    return _backend.cake_causal_conv1d(
        x, weight, cu_seqlens, bias, silu_grad_preactivation=silu_grad_preactivation
    )


@flashinfer_api
def cake_causal_conv1d_kcp(
    x_local: torch.Tensor, weight: torch.Tensor, ctx: Any
) -> torch.Tensor:
    """Autograd-enabled context-parallel packed width-4 depthwise causal convolution + SiLU of this rank's fragments.

    The first tokens of a fragment that continues a document read the history rows of the preceding fragment
    (exchanged through the context's halo collective) instead of zeros, so the result equals the single-rank
    :func:`cake_causal_conv1d` of the whole batch on this rank's tokens.  The KCP path takes FP32 taps and no
    bias (the package's pre-round-6 contract).

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[S_local, C]`` or ``[1, S_local, C]`` contiguous packed activations of this rank's fragments, in the
        partition's order; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps.
    ctx : KcpContext
        From ``flashinfer.kda_train.build_kcp_context`` (the same context the KDA step of the rank uses).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward runs the head program, the backward halo exchange and the
        fused ``dx`` + deterministic FP32 ``dweight`` kernel of this rank's fragments.
    """

    from .cake_causal_conv1d_train_backend import kcp as _kcp

    return _kcp.cake_causal_conv1d_kcp(x_local, weight, ctx)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d",
    "cake_causal_conv1d_kcp",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "is_cake_causal_conv1d_train_supported",
]
