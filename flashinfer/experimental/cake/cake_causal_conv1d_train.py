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
backward (SM100 / SM103 / SM107) with a selectable numerical mode.

Every entry point takes ``deterministic: bool = True``:

* ``deterministic=True`` (the default) runs the deterministic package
  :mod:`flashinfer.cake_causal_conv1d_det_train` unchanged: ``dweight`` /
  ``dbias`` are reduced in a fixed order and are bitwise reproducible for a
  given ``(S, C, device)`` and route.
* ``deterministic=False`` runs the non-deterministic package
  ``flashinfer.experimental.cake.cake_causal_conv1d_train_backend``: the
  backward is ONE ordinary launch whose CTAs add their ``dweight`` / ``dbias``
  rows into zeroed accumulators with unordered global reductions (the
  deterministic backward is a main launch plus a fixed-order reduce launch).
* With ``torch.use_deterministic_algorithms(True)`` a call with
  ``deterministic=False`` runs the deterministic package and warns once
  (``UserWarning``): determinism requested anywhere is never silently dropped.
  Under that fallback a caller-owned backward ``workspace`` (a
  ``deterministic=False`` buffer) is not passed on; the deterministic package
  allocates its own.

Numerical contract of ``deterministic=False``: only the run-to-run bitwise
reproducibility of ``dweight`` / ``dbias`` is relaxed.  ``y`` is bitwise the
deterministic forward's (the same programs).  ``dx`` is bitwise the
deterministic backward's for the same route and channel count (the same
per-thread FP32 arithmetic and one BF16 rounding).  ``dweight`` / ``dbias``
accumulate in FP32 where the deterministic reduce sums in FP32 (the default
``silu_grad_preactivation="bf16"``: within about 3e-5 rel-L2 of the exact
FP64 gradient) and in FP64 on the ``silu_grad_preactivation="fp32"`` routes
(the deterministic FP64 reduction's accuracy, about 2e-7 rel-L2), are rounded
once into the requested dtype, and stay within the deterministic package's
error budgets; the accumulation order across CTAs varies run to run.  The
inputs, output dtypes, masks, layouts, dynamic shapes and parameter-gradient
ownership are the deterministic package's.

The forward and backward workspaces are the selected package's: with
``deterministic=False`` the backward workspace holds zeroed accumulators and
counters that every launch leaves zero
(:func:`cake_causal_conv1d_train_backward_workspace` returns it zeroed; size
depends on ``C`` and the route, not on ``S``; never share one between two
backward launches that may run at once, e.g. on two streams); when omitted
the package keeps one zeroed buffer per (device, current stream), or a fresh
one inside a CUDA Graph capture.  The forward workspace (zeroed int32
token-chunk counters) has the same size and rules in both packages.

Name map (the deterministic names and their compatibility aliases are
unchanged and always deterministic)::

    deterministic package (unchanged)                    this module (deterministic=...)
    flashinfer.cake_causal_conv1d_det_train.             flashinfer.experimental.cake.cake_causal_conv1d_train.
      cake_causal_conv1d_det                               cake_causal_conv1d
      cake_causal_conv1d_det_train_forward                 cake_causal_conv1d_train_forward
      cake_causal_conv1d_det_train_backward                cake_causal_conv1d_train_backward
      cake_causal_conv1d_det_train_{forward,backward}_     cake_causal_conv1d_train_{forward,backward}_
        workspace{,_bytes}                                   workspace{,_bytes}
      CakeCausalConv1dDetFunction                          CakeCausalConv1dFunction (non-deterministic)
      is_cake_causal_conv1d_det_train_supported            is_cake_causal_conv1d_train_supported
    flashinfer.cake_causal_conv1d_det_train_kcp.         flashinfer.cake_causal_conv1d_train_kcp.
      cake_causal_conv1d_det_kcp                           cake_causal_conv1d_kcp
      cake_causal_conv1d_det_multi_kcp                     cake_causal_conv1d_multi_kcp
      RecomputeHandleDetKcp                                RecomputeHandleKcp

The legacy module ``flashinfer.cake_causal_conv1d_train`` and the legacy names
``flashinfer.cake_causal_conv1d``, ``flashinfer.cake_causal_conv1d_train_forward``
/ ``_backward``, ``flashinfer.CakeCausalConv1dFunction`` and
``flashinfer.cake_causal_conv1d_det_train.cake_causal_conv1d_kcp`` /
``cake_causal_conv1d_kcp_multi`` / ``KcpRecomputeHandle`` remain aliases of the
deterministic objects; this module is the explicit namespace of the selectable
mode.

Usage::

    from flashinfer.experimental.cake.cake_causal_conv1d_train import (
        cake_causal_conv1d,
        cake_causal_conv1d_train_backward,
        cake_causal_conv1d_train_forward,
    )

    y = cake_causal_conv1d(x, weight, cu_seqlens, bias)                      # deterministic (default)
    y = cake_causal_conv1d(x, weight, cu_seqlens, bias, deterministic=False)  # one-launch backward

    y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens, deterministic=False)
    dx, dweight = cake_causal_conv1d_train_backward(
        x, weight, cu_seqlens, dy, deterministic=False
    )
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from ...api_logging import flashinfer_experimental_api
from .cake_causal_conv1d_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeCausalConv1dFunction,
)

_FEATURE = (
    "Cake causal conv1d + SiLU training with a selectable numerical mode "
    "(SM100/SM103/SM107)"
)
# The entry points that already warned about the deterministic fallback under torch's deterministic mode.
_FALLBACK_WARNED: set[str] = set()


def _deterministic(deterministic: object, entry: str) -> bool:
    """Whether a call runs the deterministic package: ``deterministic`` itself, or torch's deterministic mode
    (``torch.are_deterministic_algorithms_enabled()``) overriding ``deterministic=False`` with a one-time
    warning per entry point."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"{entry}: deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return True
    if torch.are_deterministic_algorithms_enabled():
        if entry not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(entry)
            warnings.warn(
                f"{entry}(deterministic=False) runs the deterministic implementation because "
                "torch.use_deterministic_algorithms(True) is in effect",
                UserWarning,
                stacklevel=3,
            )
        return True
    return False


def _det():
    from ... import cake_causal_conv1d_det_train  # noqa: PLC0415

    return cake_causal_conv1d_det_train


def _backend():
    from . import cake_causal_conv1d_train_backend  # noqa: PLC0415

    return cake_causal_conv1d_train_backend


def is_cake_causal_conv1d_train_supported(
    device: torch.device, *, deterministic: bool = True
) -> bool:
    """Whether ``device`` runs the selected package's base routes (``fwd`` and ``bwd``): the deterministic
    package's support with ``deterministic=True``, the non-deterministic package's (also requiring its rendered
    ``build.json``) with ``deterministic=False``."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return _det().is_cake_causal_conv1d_det_train_supported(device)
    return _backend().is_cake_causal_conv1d_train_supported(device)


def cake_causal_conv1d_train_backward_workspace_bytes(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned backward workspace of the selected package for ``C`` channels on ``device``.

    ``deterministic=True``: the deterministic package's per-CTA partial rows
    (no zeroing needed).  ``deterministic=False``: the non-deterministic
    package's accumulators and counters (size depends on ``C``, not on ``S``;
    must be zero before its first launch).  The size is of the package named by
    ``deterministic`` (torch's deterministic mode does not change it).
    """

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return _det().cake_causal_conv1d_det_train_backward_workspace_bytes(
            S, C, device
        )
    return _backend().cake_causal_conv1d_train_backward_workspace_bytes(
        S, C, device, deterministic=False
    )


def cake_causal_conv1d_train_backward_workspace(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """Allocate the selected package's backward workspace (``deterministic=False``: ZEROED, kept zero by every
    launch; reuse it for every later backward call of one stream, never for two launches that may run at once)."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return _det().cake_causal_conv1d_det_train_backward_workspace(S, C, device)
    return _backend().cake_causal_conv1d_train_backward_workspace(
        S, C, device, deterministic=False
    )


def cake_causal_conv1d_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned forward workspace (zeroed int32 token-chunk counters) for ``C`` channels on
    ``device``; both packages run the same forward programs, so the size is the same either way."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return _det().cake_causal_conv1d_det_train_forward_workspace_bytes(
            S, C, device
        )
    return _backend().cake_causal_conv1d_train_forward_workspace_bytes(
        S, C, device, deterministic=False
    )


def cake_causal_conv1d_train_forward_workspace(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """Allocate a zeroed int32 forward workspace (left zero by every launch; keep one per stream)."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return _det().cake_causal_conv1d_det_train_forward_workspace(S, C, device)
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
        :func:`cake_causal_conv1d_train_forward_workspace` (the same in both
        modes).
    deterministic : bool
        ``True`` (default): the deterministic package's forward.  ``False``:
        the non-deterministic package's copy of the same programs (bitwise the
        same ``y``).  torch's deterministic mode turns ``False`` into ``True``
        with a one-time warning.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    if _deterministic(deterministic, "cake_causal_conv1d_train_forward"):
        return _det().cake_causal_conv1d_det_train_forward(
            x, weight, cu_seqlens, bias=bias, out=out, workspace=workspace
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
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward.

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs; nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    silu_grad_preactivation : {"bf16", "fp32"}
        ``"bf16"``: SiLU' at the BF16-rounded recomputed preactivation (FP32
        accumulation).  ``"fp32"``: at the unrounded FP32 preactivation with an
        FP64 reduction (deterministic) / FP64 accumulators (non-deterministic).
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_causal_conv1d_train_backward_workspace`
        with the same ``deterministic`` value.  Ignored under the
        torch-deterministic-mode fallback of ``deterministic=False``.
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (FP32 or BF16 each; the parameter's dtype when omitted).
    deterministic : bool
        ``True`` (default): main launch + fixed-order reduce launch, bitwise
        reproducible gradients.  ``False``: one launch with unordered
        cross-CTA reductions; ``dx`` bitwise the deterministic one,
        ``dweight`` / ``dbias`` within the same error budgets but not bitwise
        reproducible run to run.  torch's deterministic mode turns ``False``
        into ``True`` with a one-time warning.

    Returns
    -------
    ``(dx, dweight)``, with a third element ``dbias`` only when ``bias`` is
    given.  ``S == 0`` returns the empty ``dx`` and zero gradients without
    launching.
    """

    if _deterministic(deterministic, "cake_causal_conv1d_train_backward"):
        return _det().cake_causal_conv1d_det_train_backward(
            x,
            weight,
            cu_seqlens,
            dy,
            bias=bias,
            silu_grad_preactivation=silu_grad_preactivation,
            # A deterministic=False workspace is never handed to the deterministic package.
            workspace=workspace if deterministic else None,
            dx=dx,
            dweight=dweight,
            dbias=dbias,
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
    deterministic: bool = True,
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU.

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
    deterministic : bool
        ``True`` (default): ``flashinfer.cake_causal_conv1d_det_train.cake_causal_conv1d_det``.
        ``False``: the non-deterministic autograd Function
        (:class:`CakeCausalConv1dFunction`), whose backward is one launch with
        unordered ``dweight`` / ``dbias`` reductions.  The mode is chosen at the
        forward call (torch's deterministic mode then turns ``False`` into
        ``True`` with a one-time warning) and the backward runs the same package.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape; the backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns the gradients in the
        parameters' dtypes.
    """

    if _deterministic(deterministic, "cake_causal_conv1d"):
        return _det().cake_causal_conv1d_det(
            x, weight, cu_seqlens, bias, silu_grad_preactivation=silu_grad_preactivation
        )
    return _backend().cake_causal_conv1d(
        x,
        weight,
        cu_seqlens,
        bias,
        silu_grad_preactivation=silu_grad_preactivation,
        deterministic=False,
    )


__all__ = [
    "ARCHES",
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
