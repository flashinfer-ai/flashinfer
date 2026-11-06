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
backward (SM100 / SM103 / SM107), the non-deterministic package: experimental
entry points of ``flashinfer.experimental.cake.cake_causal_conv1d_train_backend``.
The backward is ONE ordinary launch whose CTAs add their ``dweight`` /
``dbias`` rows into zeroed accumulators with unordered global reductions (the
deterministic backward of :mod:`flashinfer.cake_causal_conv1d_det_train` is a
main launch plus a fixed-order reduce launch).  This module never runs the
deterministic package.

Numerical mode (the same rule as every non-deterministic training package;
"torch deterministic mode" is ``torch.use_deterministic_algorithms(True)``,
with or without ``warn_only=True``)::

    call                                  torch deterministic mode off   torch deterministic mode on
    this module, deterministic=False      non-deterministic              the forward runs; the backward
                                                                           raises RuntimeError before it
                                                                           launches (warn_only=True: a
                                                                           UserWarning, then it runs); it
                                                                           never falls back
    this module, any other deterministic  ValueError naming the          ValueError
      value (the default True included)     deterministic entry point
    the deterministic names (*_det*)      deterministic; no              deterministic; no warning
                                            deterministic keyword
                                            (TypeError)

Every entry point, the workspace helpers and
``is_cake_causal_conv1d_train_supported`` included, takes keyword-only
``deterministic: bool = True`` and runs only with ``deterministic=False``, so
a call states the non-deterministic mode explicitly and a call that relied on
a deterministic default fails with an error that names the deterministic entry
point instead of silently changing mode.  The package is chosen by name, so a
checkpoint recompute (``torch.utils.checkpoint``, reentrant or not) runs the
same package; keep torch's deterministic setting the same from a forward to
its backward.

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
launch leaves zero (:func:`cake_causal_conv1d_train_backward_workspace`
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

Name map (deterministic -> this module)::

    flashinfer.cake_causal_conv1d_det_train.             flashinfer.experimental.cake.cake_causal_conv1d_train.
      cake_causal_conv1d_det                               cake_causal_conv1d
      cake_causal_conv1d_det_train_forward                 cake_causal_conv1d_train_forward
      cake_causal_conv1d_det_train_backward                cake_causal_conv1d_train_backward
      cake_causal_conv1d_det_train_{forward,backward}_     cake_causal_conv1d_train_{forward,backward}_
        workspace{,_bytes}                                   workspace{,_bytes}
      CakeCausalConv1dDetFunction                          CakeCausalConv1dFunction
      is_cake_causal_conv1d_det_train_supported            is_cake_causal_conv1d_train_supported
    flashinfer.cake_causal_conv1d_det_train_kcp.         flashinfer.cake_causal_conv1d_train_kcp.
      cake_causal_conv1d_det_kcp                           cake_causal_conv1d_kcp
      cake_causal_conv1d_det_multi_kcp                     cake_causal_conv1d_multi_kcp
      RecomputeHandleDetKcp                                RecomputeHandleKcp

Usage::

    from flashinfer.experimental.cake.cake_causal_conv1d_train import (
        cake_causal_conv1d,
        cake_causal_conv1d_train_backward,
        cake_causal_conv1d_train_forward,
    )

    y = cake_causal_conv1d(x, weight, cu_seqlens, bias, deterministic=False)  # one-launch backward

    y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens, deterministic=False)
    dx, dweight = cake_causal_conv1d_train_backward(
        x, weight, cu_seqlens, dy, deterministic=False
    )

    # opt-in finer backward partition (1, 2, 3 or 4; default 1)
    dx, dweight = cake_causal_conv1d_train_backward(
        x, weight, cu_seqlens, dy, bwd_grid_mult=3, deterministic=False
    )
"""

from __future__ import annotations

from typing import Optional

import torch

from ...api_logging import flashinfer_experimental_api
from .cake_causal_conv1d_train_backend import (  # noqa: F401
    ARCHES,
    BWD_GRID_MULTS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeCausalConv1dFunction,
)

_FEATURE = (
    "Cake causal conv1d + SiLU training, non-deterministic package (SM100/SM103/SM107)"
)


def _nondeterministic_only(deterministic: object, entry: str, det_entry: str) -> None:
    """Refuse every ``deterministic`` value but ``False`` (the default ``True`` included) with a ``ValueError`` that
    names the deterministic entry point: this module runs the non-deterministic package only."""

    if deterministic is not False:
        raise ValueError(
            f"{entry} runs the non-deterministic package (unordered dweight / dbias reductions) and needs "
            f"deterministic=False, got deterministic={deterministic!r}; the deterministic entry point is "
            f"flashinfer.cake_causal_conv1d_det_train.{det_entry} (pass deterministic=False here for the "
            "non-deterministic path)"
        )


def _backend():
    from . import cake_causal_conv1d_train_backend  # noqa: PLC0415

    return cake_causal_conv1d_train_backend


@flashinfer_experimental_api(feature=_FEATURE)
def is_cake_causal_conv1d_train_supported(
    device: torch.device, *, deterministic: bool = True
) -> bool:
    """Whether ``device`` runs the non-deterministic package's base routes (``fwd`` and ``bwd``; the package's
    rendered ``build.json`` is required).  Needs ``deterministic=False`` (``ValueError`` otherwise, naming
    ``is_cake_causal_conv1d_det_train_supported``)."""

    _nondeterministic_only(
        deterministic,
        "is_cake_causal_conv1d_train_supported",
        "is_cake_causal_conv1d_det_train_supported",
    )
    return _backend().is_cake_causal_conv1d_train_supported(device)


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_train_backward_workspace_bytes(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
    deterministic: bool = True,
) -> int:
    """Bytes of the caller-owned backward workspace (accumulators and counters) for ``C`` channels on ``device``:
    the size depends on ``C``, not on ``S`` or ``bwd_grid_mult`` (which is validated); it must be zero before its
    first launch.  Needs ``deterministic=False``."""

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_det_train_backward_workspace_bytes",
    )
    return _backend().cake_causal_conv1d_train_backward_workspace_bytes(
        S, C, device, bwd_grid_mult=bwd_grid_mult, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_train_backward_workspace(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
    deterministic: bool = True,
) -> torch.Tensor:
    """Allocate a ZEROED backward workspace, kept zero by every launch and the same size for every
    ``bwd_grid_mult``; reuse it for every later backward call of one stream, never for two launches that may run
    at once.  Needs ``deterministic=False``."""

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_train_backward_workspace",
        "cake_causal_conv1d_det_train_backward_workspace",
    )
    return _backend().cake_causal_conv1d_train_backward_workspace(
        S, C, device, bwd_grid_mult=bwd_grid_mult, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned forward workspace (zeroed int32 token-chunk counters) for ``C`` channels on
    ``device`` (the deterministic package's size: the same forward programs).  Needs ``deterministic=False``."""

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_det_train_forward_workspace_bytes",
    )
    return _backend().cake_causal_conv1d_train_forward_workspace_bytes(
        S, C, device, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_train_forward_workspace(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """Allocate a zeroed int32 forward workspace (left zero by every launch; keep one per stream).  Needs
    ``deterministic=False``."""

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_train_forward_workspace",
        "cake_causal_conv1d_det_train_forward_workspace",
    )
    return _backend().cake_causal_conv1d_train_forward_workspace(
        S, C, device, deterministic=False
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

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
        :func:`cake_causal_conv1d_train_forward_workspace`.
    deterministic : bool
        Keyword-only, default ``True``, which raises: pass ``False`` (the
        package's copy of the deterministic forward programs, bitwise the same
        ``y``).  Any value but ``False`` raises ``ValueError`` naming
        ``cake_causal_conv1d_det_train_forward``.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_train_forward",
        "cake_causal_conv1d_det_train_forward",
    )
    return _backend().cake_causal_conv1d_train_forward(
        x,
        weight,
        cu_seqlens,
        bias=bias,
        out=out,
        workspace=workspace,
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
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
    bwd_grid_mult: int = 1,
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward (one launch, unordered
    ``dweight`` / ``dbias`` reductions).

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
        Caller-owned buffer from :func:`cake_causal_conv1d_train_backward_workspace`.
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (FP32 or BF16 each; the parameter's dtype when omitted).  An FP32
        ``dweight`` must be 16-byte aligned (one 16-byte store per channel;
        ``ValueError`` otherwise).
    bwd_grid_mult : {1, 2, 3, 4}
        Opt-in finer static partition of the token axis (default 1, the
        default grid): the backward launches ``bwd_grid_mult`` times its
        default untrimmed CTA count, so each CTA streams that much fewer tokens
        (for a backward that overlaps kernels of other streams: a CTA that
        starts late trails by a shorter stream).  ``dx`` is bitwise the
        default's; more CTAs add into the same accumulators (same budgets, same
        workspace size).  ``TypeError`` for a non-int, ``ValueError`` for
        another value.
    deterministic : bool
        Keyword-only, default ``True``, which raises: pass ``False``.  Any value
        but ``False`` raises ``ValueError`` naming
        ``cake_causal_conv1d_det_train_backward``.  Under
        ``torch.use_deterministic_algorithms(True)`` the backward raises
        PyTorch's ``RuntimeError`` before it launches (a ``UserWarning``, then
        it runs, with ``warn_only=True``).

    Returns
    -------
    ``(dx, dweight)``, with a third element ``dbias`` only when ``bias`` is
    given.  ``S == 0`` returns the empty ``dx`` and zero gradients without
    launching.
    """

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_train_backward",
        "cake_causal_conv1d_det_train_backward",
    )
    return _backend().cake_causal_conv1d_train_backward(
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
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
    bwd_grid_mult: int = 1,
    deterministic: bool = True,
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU, non-deterministic backward.

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
        :func:`cake_causal_conv1d_train_backward`); checked at the forward
        call, used by the backward.
    deterministic : bool
        Keyword-only, default ``True``, which raises: pass ``False`` (the
        package's autograd Function :class:`CakeCausalConv1dFunction`, whose
        backward is one launch with unordered ``dweight`` / ``dbias``
        reductions).  Any value but ``False`` raises ``ValueError`` naming
        ``cake_causal_conv1d_det``.  When torch's deterministic mode is on at
        the backward, the backward raises a ``RuntimeError`` (a
        ``UserWarning``, then it runs, with ``warn_only=True``).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape; the backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns the gradients in the
        parameters' dtypes.
    """

    _nondeterministic_only(
        deterministic, "cake_causal_conv1d", "cake_causal_conv1d_det"
    )
    return _backend().cake_causal_conv1d(
        x,
        weight,
        cu_seqlens,
        bias,
        silu_grad_preactivation=silu_grad_preactivation,
        bwd_grid_mult=bwd_grid_mult,
        deterministic=False,
    )


__all__ = [
    "ARCHES",
    "BWD_GRID_MULTS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "is_cake_causal_conv1d_train_supported",
]
