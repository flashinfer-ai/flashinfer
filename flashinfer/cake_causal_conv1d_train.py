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
first token.  With FP32 taps ``w[C, 4]`` (``w[c, 3]`` multiplies the current
token) the forward is, in FP32 with one BF16 rounding,

    pre[t, c] = sum_k w[c, k] * x[t - 3 + k, c]      # x of another document is 0
    y[t, c]   = bf16(pre * sigmoid(pre))

The backward recomputes ``pre`` in FP32 from ``x``, rounds it to BF16 and
evaluates the SiLU derivative in FP32 from the rounded value ``p`` (policy A),

    dpre[t, c]    = dy[t, c] * s * (1 + p * (1 - s)),  s = sigmoid(p)
    dx[t, c]      = bf16(sum_k w[c, k] * dpre[t + 3 - k, c])   # dpre of another document is 0
    dweight[c, k] = sum_t dpre[t, c] * x[t - 3 + k, c]         # FP32, fixed order

in one cooperative launch: ``dweight`` is reduced in a fixed order through a
per-CTA FP32 partial workspace (no floating-point atomics), so it is bitwise
reproducible for a given ``(S, C, device)``.  Nothing is saved from the
forward but ``x``, ``w`` and ``cu_seqlens``.

Inputs and outputs are contiguous (``C`` a multiple of 8); ``weight`` and
``dweight`` are FP32 ``[C, 4]``.  ``S``, ``C`` and the document count are
runtime scalars: they may change between calls, the host never reads
``cu_seqlens`` (no synchronization), ``S == 0`` returns empty outputs without
launching, and the calls are CUDA Graph capturable once the libraries are
built.

An architecture's backward may ship several programs of the kernel; every
call runs exactly one, chosen by ``C`` through a host-side rule of the
backend's registry (never the data), so a given ``(S, C, device)`` always
runs the same program.  The workspace size is the maximum over the
architecture's backward programs, so one buffer of
``cake_causal_conv1d_train_backward_workspace_bytes(S, C, device)`` serves
whichever program ``C`` selects.  Before capturing a CUDA Graph, build every
library of the device's architecture (the backend's
``cake_jit.specs_for_arch``) so that no JIT build happens inside the capture;
the captured graph holds the program selected for its shape.

Usage::

    y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens)
    dx, dweight = cake_causal_conv1d_train_backward(x, weight, cu_seqlens, dy)

    # autograd (saves x, weight and cu_seqlens only)
    y = cake_causal_conv1d(x, weight, cu_seqlens)

    # caller-owned backward workspace (per-CTA FP32 partials, no zeroing needed)
    ws = cake_causal_conv1d_train_backward_workspace(S, C, x.device)
    dx, dweight = cake_causal_conv1d_train_backward(x, weight, cu_seqlens, dy, workspace=ws)
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_causal_conv1d_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_causal_conv1d_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeCausalConv1dFunction,
    cake_causal_conv1d_train_backward_workspace,
    cake_causal_conv1d_train_backward_workspace_bytes,
    is_cake_causal_conv1d_train_supported,
)

# The three stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_causal_conv1d_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current
        token, ``weight[c, 0]`` the token three positions back.  Any other width
        is rejected.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths (``cu_seqlens[0] == 0``, ``cu_seqlens[-1] == S``; a repeated
        value is an empty document).  The host reads its metadata only: no
        element is copied to the host and no stream is synchronized.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape: ``y[t, c] = bf16(silu(sum_k weight[c, k] *
        x[t - 3 + k, c]))`` with zero history before each document's first
        token.  ``S == 0`` returns the empty output without launching.
    """

    return _backend.cake_causal_conv1d_train_forward(x, weight, cu_seqlens, out=out)


@flashinfer_api
def cake_causal_conv1d_train_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dy: torch.Tensor,
    *,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward.

    Recomputes the FP32 preactivation from ``x``, rounds it to BF16 and takes
    the SiLU derivative of the rounded value (policy A), then produces ``dx``
    and the FP32 weight gradient in one cooperative launch.

    Parameters
    ----------
    x, weight, cu_seqlens
        The forward's inputs (same contracts as
        :func:`cake_causal_conv1d_train_forward`); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_causal_conv1d_train_backward_workspace` (the per-CTA FP32
        ``dweight`` partials; fully written by the kernel, so it needs no
        zeroing); allocated when omitted.
    dx, dweight : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape and FP32 ``[C, 4]``, both
        contiguous.

    Returns
    -------
    ``(dx, dweight)``: BF16 ``dx`` of ``x``'s shape, ``dx[t, c] =
    bf16(sum_k weight[c, k] * dpre[t + 3 - k, c])`` over the tokens of the
    same document, and the FP32 ``[C, 4]`` ``dweight[c, k] = sum_t dpre[t, c]
    * x[t - 3 + k, c]`` reduced in a fixed order, so it is bitwise
    reproducible for a given ``(S, C, device)``.  ``S == 0`` returns the empty
    ``dx`` and a zero ``dweight`` without launching.
    """

    return _backend.cake_causal_conv1d_train_backward(
        x, weight, cu_seqlens, dy, workspace=workspace, dx=dx, dweight=dweight
    )


@flashinfer_api
def cake_causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations on a
        supported CUDA device.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.  The backward saves only ``x``, ``weight``
        and ``cu_seqlens`` and launches the fused ``dx`` + deterministic FP32
        ``dweight`` kernel of :func:`cake_causal_conv1d_train_backward`.
    """

    return _backend.cake_causal_conv1d(x, weight, cu_seqlens)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "cake_causal_conv1d_train_forward",
    "is_cake_causal_conv1d_train_supported",
]
