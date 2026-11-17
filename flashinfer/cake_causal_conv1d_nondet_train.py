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
backward (SM100 / SM103 / SM107), the non-deterministic package: the stable
entry points over the experimental backend package
``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train``.
The backward is ONE ordinary launch whose CTAs add their ``dweight`` /
``dbias`` rows into zeroed accumulators with unordered global reductions (the
backward of the deterministic package
(:mod:`flashinfer.cake_causal_conv1d_det_train`) is a main launch plus a
fixed-order reduce launch).  This module never runs the deterministic package.

Numerical mode (the same rule as every non-deterministic training package):
the module name chooses the package, and no entry point takes a mode keyword.
"torch deterministic mode" is ``torch.use_deterministic_algorithms(True)``,
with or without ``warn_only=True``::

    package                                        torch deterministic mode off   torch deterministic mode on
    the non-deterministic package                  non-deterministic              the forward runs; the backward
      (flashinfer.cake_causal_conv1d_nondet_                                        raises RuntimeError before it
      train[_kcp], this module)                                                     launches (warn_only=True: a
                                                                                    UserWarning, then it runs); it
                                                                                    never falls back
    the deterministic package                      deterministic                  deterministic; no warning
      (flashinfer.cake_causal_conv1d_det_
      train[_kcp])

The forward is fixed-order (the deterministic package's programs) and never
alerts; only the backward's unordered ``dweight`` / ``dbias`` reduction does,
and it reads torch's deterministic setting when it runs (the autograd backward
included).  The package is chosen by name, so a checkpoint recompute
(``torch.utils.checkpoint``, reentrant or not) runs the same package.

Numerical contract: only the run-to-run bitwise reproducibility of
``dweight`` / ``dbias`` is relaxed.  ``y`` is bitwise the deterministic
forward's (the same programs).  ``dx`` is bitwise the deterministic backward's
for the same route and channel count (the same per-thread FP32 arithmetic and
one BF16 rounding).  ``dweight`` / ``dbias`` accumulate in FP32 where the
deterministic reduce sums in FP32 (the default
``silu_grad_preactivation="bf16"``: within about 3e-5 rel-L2 of the exact FP64
gradient) and in FP64 on the ``silu_grad_preactivation="fp32"`` routes (the
deterministic FP64 reduction's accuracy, about 2e-7 rel-L2), are rounded once
into the requested dtype, and stay within the deterministic package's error
budgets; the accumulation order across CTAs varies run to run.  The inputs,
output dtypes, masks, layouts, dynamic shapes and parameter-gradient ownership
are the deterministic package's.

The backward workspace holds zeroed accumulators and counters that every
launch leaves zero (:func:`cake_causal_conv1d_nondet_train_backward_workspace`
returns it zeroed; size depends on ``C`` and the route, not on ``S``; never
share one between two backward launches that may run at once, e.g. on two
streams); when omitted the package keeps one zeroed buffer per (device,
current stream), or inside a CUDA Graph capture one zeroed buffer per capture
and stream, shared by its backward calls on that stream: the capture records
one fill node for it, and another for a larger buffer when a call needs more
bytes.  That buffer stays alive after its graph is freed, until a later
capture on that stream replaces it.  With capture id 0 (also when
``cuda.bindings`` is not importable) each captured backward call allocates
and zeroes a buffer of its own.  The forward workspace (zeroed int32
token-chunk counters) has the deterministic package's size and rules.

The backward entry points and the backward workspace helpers take the
deterministic package's opt-in ``bwd_grid_mult: int = 1`` (one of
``BWD_GRID_MULTS``: 1, 2, 3 or 4), with its validation (``TypeError`` for a
non-int, ``ValueError`` for another value): the backward launches
``bwd_grid_mult`` times its default untrimmed CTA count, a finer static
partition of the token axis for a backward that overlaps kernels of other
streams.  The CTAs of each channel tile split its tokens among themselves at
any grid size, so ``dx`` is still bitwise the deterministic one and
``dweight`` / ``dbias`` keep their budgets; the workspace size does not depend
on it (one accumulator set and one counter per channel tile, whatever the
grid).

Name map (the deterministic package -> the non-deterministic package)::

    flashinfer.cake_causal_conv1d_det_train.             flashinfer.cake_causal_conv1d_nondet_train.
      cake_causal_conv1d_det                               cake_causal_conv1d_nondet
      cake_causal_conv1d_det_train_forward                 cake_causal_conv1d_nondet_train_forward
      cake_causal_conv1d_det_train_backward                cake_causal_conv1d_nondet_train_backward
      cake_causal_conv1d_det_train_{forward,backward}_     cake_causal_conv1d_nondet_train_{forward,backward}_
        workspace{,_bytes}                                   workspace{,_bytes}
      CakeCausalConv1dDetFunction                          CakeCausalConv1dNondetFunction
      is_cake_causal_conv1d_det_train_supported            is_cake_causal_conv1d_nondet_train_supported
    flashinfer.cake_causal_conv1d_det_train_kcp          flashinfer.cake_causal_conv1d_nondet_train_kcp
                                                           (cake_causal_conv1d_nondet_kcp,
                                                           cake_causal_conv1d_nondet_kcp_multi,
                                                           CakeCausalConv1dNondetKcpRecomputeHandle)

Usage::

    from flashinfer.cake_causal_conv1d_nondet_train import (
        cake_causal_conv1d_nondet,
        cake_causal_conv1d_nondet_train_backward,
        cake_causal_conv1d_nondet_train_forward,
    )

    y = cake_causal_conv1d_nondet(x, weight, cu_seqlens, bias)  # one-launch backward

    y = cake_causal_conv1d_nondet_train_forward(x, weight, cu_seqlens)
    dx, dweight = cake_causal_conv1d_nondet_train_backward(x, weight, cu_seqlens, dy)

    # opt-in finer backward partition (1, 2, 3 or 4; default 1)
    dx, dweight = cake_causal_conv1d_nondet_train_backward(
        x, weight, cu_seqlens, dy, bwd_grid_mult=3
    )
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_api

# The package lives under flashinfer.experimental; this core module imports it on first use only (the containment
# rules of flashinfer/experimental/README.md), so importing this module loads nothing from the experimental tree.
# Its constants and its autograd Function resolve through the module ``__getattr__`` below.  The entry points are stable
# APIs (``@flashinfer_api``): real ``def``s whose signatures mirror the package's, forwarding each call to it.
_FROM_BACKEND = (
    "ARCHES",
    "BWD_GRID_MULTS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dNondetFunction",
)


def _backend():
    from .experimental.cake import cake_causal_conv1d_nondet_train  # noqa: PLC0415

    return cake_causal_conv1d_nondet_train


@flashinfer_api
def is_cake_causal_conv1d_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` supports the non-deterministic causal Conv1d package.

    Parameters
    ----------
    device : torch.device
        CUDA device to query.

    Returns
    -------
    bool
        ``True`` when ``device`` runs the package's base forward and backward
        routes and the rendered ``build.json`` is available.
    """

    return _backend().is_cake_causal_conv1d_nondet_train_supported(device)


@flashinfer_api
def cake_causal_conv1d_nondet_train_backward_workspace_bytes(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
) -> int:
    """Return the backward workspace size in bytes.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.
    bwd_grid_mult : int, default=1
        Backward grid multiplier to validate for the selected backend.

    Returns
    -------
    int
        Bytes of the caller-owned backward workspace. The size depends on
        ``C`` and the target device, not on ``S``.
    """

    return _backend().cake_causal_conv1d_nondet_train_backward_workspace_bytes(
        S, C, device, bwd_grid_mult=bwd_grid_mult
    )


@flashinfer_api
def cake_causal_conv1d_nondet_train_backward_workspace(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
) -> torch.Tensor:
    """Allocate a zeroed backward workspace tensor.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.
    bwd_grid_mult : int, default=1
        Backward grid multiplier to validate for the selected backend.

    Returns
    -------
    torch.Tensor
        Zeroed caller-owned workspace for reuse on a single stream. Never share it
        between two backward launches that may run at once (e.g. two CUDA graphs
        replayed on different streams).
    """

    return _backend().cake_causal_conv1d_nondet_train_backward_workspace(
        S, C, device, bwd_grid_mult=bwd_grid_mult
    )


@flashinfer_api
def cake_causal_conv1d_nondet_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device
) -> int:
    """Return the forward workspace size in bytes.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.

    Returns
    -------
    int
        Bytes of the zeroed int32 forward workspace.
    """

    return _backend().cake_causal_conv1d_nondet_train_forward_workspace_bytes(
        S, C, device
    )


@flashinfer_api
def cake_causal_conv1d_nondet_train_forward_workspace(
    S: int, C: int, device: torch.device
) -> torch.Tensor:
    """Allocate a zeroed forward workspace tensor.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.

    Returns
    -------
    torch.Tensor
        Zeroed int32 forward workspace for reuse on a single stream.
    """

    return _backend().cake_causal_conv1d_nondet_train_forward_workspace(S, C, device)


@flashinfer_api
def cake_causal_conv1d_nondet_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    The package's copy of the deterministic forward programs: ``y`` is bitwise the deterministic package's, and
    the forward runs (without an alert) under torch's deterministic mode.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the
        current token.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths; the host reads its metadata only.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias, added after the taps.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous.
    workspace : torch.Tensor, optional
        Caller-owned zeroed int32 buffer from
        :func:`cake_causal_conv1d_nondet_train_forward_workspace`.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    return _backend().cake_causal_conv1d_nondet_train_forward(
        x,
        weight,
        cu_seqlens,
        bias=bias,
        out=out,
        workspace=workspace,
    )


@flashinfer_api
def cake_causal_conv1d_nondet_train_backward(
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
    bwd_grid_mult: int = 1,
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward (one launch, unordered
    ``dweight`` / ``dbias`` reductions).

    Under ``torch.use_deterministic_algorithms(True)`` the backward raises PyTorch's ``RuntimeError`` before it
    launches (a ``UserWarning``, then it runs, with ``warn_only=True``); it never falls back to the deterministic
    package.

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs; nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    silu_grad_preactivation : {"bf16", "fp32"}
        ``"bf16"``: SiLU' at the BF16-rounded recomputed preactivation (FP32
        accumulators).  ``"fp32"``: at the unrounded FP32 preactivation with
        FP64 accumulators.
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_causal_conv1d_nondet_train_backward_workspace`.
    dx : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape.
    dweight : torch.Tensor, optional
        Preallocated ``[C, 4]`` output (FP32 or BF16; the parameter's dtype
        when omitted).  An FP32 ``dweight`` must be 16-byte aligned (one
        16-byte store per channel; ``ValueError`` otherwise).
    dbias : torch.Tensor, optional
        Preallocated ``[C]`` output (FP32 or BF16; the parameter's dtype
        when omitted).
    bwd_grid_mult : {1, 2, 3, 4}
        Opt-in finer static partition of the token axis (default 1, the
        default grid): the backward launches ``bwd_grid_mult`` times its
        default untrimmed CTA count, so each CTA streams that much fewer tokens
        (for a backward that overlaps kernels of other streams: a CTA that
        starts late trails by a shorter stream).  ``dx`` is bitwise the
        default's; more CTAs add into the same accumulators (same budgets, same
        workspace size).  ``TypeError`` for a non-int, ``ValueError`` for
        another value.

    Returns
    -------
    ``(dx, dweight)``, with a third element ``dbias`` only when ``bias`` is
    given.  ``S == 0`` returns the empty ``dx`` and zero gradients without
    launching.
    """

    return _backend().cake_causal_conv1d_nondet_train_backward(
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
        bwd_grid_mult=bwd_grid_mult,
    )


@flashinfer_api
def cake_causal_conv1d_nondet(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
    bwd_grid_mult: int = 1,
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU, non-deterministic backward.

    The package's autograd Function :class:`CakeCausalConv1dNondetFunction`, whose backward is one launch with
    unordered ``dweight`` / ``dbias`` reductions.  When torch's deterministic mode is on as the backward runs, the
    backward raises a ``RuntimeError`` (a ``UserWarning``, then it runs, with ``warn_only=True``).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias.
    silu_grad_preactivation : {"bf16", "fp32"}
        The weight gradient's SiLU' policy.
    bwd_grid_mult : {1, 2, 3, 4}
        The backward's opt-in finer partition (default 1; see
        :func:`cake_causal_conv1d_nondet_train_backward`); checked at the forward
        call, used by the backward.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape; the backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns the gradients in the
        parameters' dtypes.
    """

    return _backend().cake_causal_conv1d_nondet(
        x,
        weight,
        cu_seqlens,
        bias,
        silu_grad_preactivation=silu_grad_preactivation,
        bwd_grid_mult=bwd_grid_mult,
    )


def __getattr__(name: str) -> Any:
    if name in _FROM_BACKEND:
        return getattr(_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FROM_BACKEND))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "BWD_GRID_MULTS",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822 - resolved by __getattr__
    "CakeCausalConv1dNondetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_causal_conv1d_nondet",
    "cake_causal_conv1d_nondet_train_backward",
    "cake_causal_conv1d_nondet_train_backward_workspace",
    "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_forward",
    "cake_causal_conv1d_nondet_train_forward_workspace",
    "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    "is_cake_causal_conv1d_nondet_train_supported",
]
