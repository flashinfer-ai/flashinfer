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

Cake fused BF16 RMSNorm training forward / backward (SM100 / SM103 / SM107)
with a deterministic default and an opt-in non-deterministic weight gradient.

Forward: ``r_t = (mean_j x_tj^2 + eps)^(-1/2)`` in FP32, ``y = BF16((x * r) * w)``.
The only tensor saved for the backward besides the inputs is the FP32 per-row
reciprocal RMS ``rstd`` (``[T]``).  Backward: one fused pass produces
``dx = BF16(r * g * w - x * r^3 * mean(g * w * x))`` and the FP32 weight
gradient ``dw = sum_t g * x * r``.  The optional residual-add fusion computes
``h_new = BF16(h + u)`` first, normalizes ``h_new``, and in the backward adds
the gradient flowing directly through ``h_new`` so that one ``dx`` tensor is
the gradient of both ``h`` and ``u``.

Every entry point takes ``deterministic`` (default ``True``):

* ``deterministic=True`` runs :mod:`flashinfer.cake_rmsnorm_det_train` with the
  same arguments: ``dw`` through a fixed row-chunk partition and a fixed-order
  reduction (a main kernel and a reduce kernel), bitwise reproducible for a
  given token count on a given architecture.  The default call returns exactly
  that package's results.
* ``deterministic=False`` runs this package's kernels: the same forward programs
  (the forward has no cross-row reduction) and, per token-count window, the
  backward form this build's registry lists for the window
  (``cake_jit.program_windows``; the backward docstring's notes are generated
  from it).  The single-kernel form, in the windows that ship it: persistent
  CTAs claim row chunks, write ``dx`` and add their per-column ``dw`` sums into
  FP64 sums in the workspace in the order they finish; the CTA that finishes
  last rounds the sums into ``dw``.  ``dx`` is bitwise reproducible and within
  one BF16 ulp of max(|dx|, rms(dx)) of the deterministic backward's ``dx``
  (not always bitwise equal to it); ``dw`` is not guaranteed bitwise
  reproducible (the order of the FP64 additions depends on scheduling).  In
  the windows the backward docstring's notes name, a twosum single kernel keeps
  each thread's ``dw`` column sums exact (error-free FP32 products, TwoSum
  compensation) before the FP64 additions, so its ``dw`` error does not grow
  with ``T``.  Every other window runs
  the deterministic package's backward pair (main kernel and fixed-order
  reduce), rendered into this package.  This path also accepts ``[..., H]`` inputs whose leading dimensions collapse into
  rows (for example a ``[T, heads, 256]`` per-head view sharing one ``w[256]``)
  and an FP32 weight on the routes that ship the FP32-weight programs.
* ``deterministic=False`` while ``torch.use_deterministic_algorithms(True)`` is
  in effect raises ``RuntimeError``.  With ``warn_only=True`` it warns, the
  forward runs this package's forward programs (deterministic) and the backward
  runs the deterministic package with a workspace of its own; a call that
  package cannot serve (``[..., H]`` inputs, an FP32 weight, a width or residual
  route it does not export) raises one ``RuntimeError`` naming the feature
  instead.  A call that asks for determinism never runs the non-deterministic
  kernels.  ``deterministic`` must be a ``bool`` (``TypeError`` otherwise).

All kernels use FP32 statistics and accumulation (the non-deterministic
single kernel's ``dw`` sums across CTAs in FP64) with IEEE division and square root (no
fast-math).  Inputs and outputs are BF16; ``rstd`` and ``dw``
are FP32.  Row-strided inputs (``x``, ``residual``, ``g``, ``g_residual``)
are accepted when ``stride(1) == 1``, ``stride(0)`` is a multiple of eight
elements and the storage is aligned (32 bytes for the forward inputs, 16 for
the backward); outputs are contiguous.  The token count ``T`` is a runtime
scalar: it may change between calls, ``T == 1`` is supported and ``T == 0``
returns empty outputs (a zero ``dw``) without launching.

The non-deterministic backward's workspace
(``cake_rmsnorm_train_backward_workspace_bytes(T, H, ..., deterministic=False)``
bytes) has one layout, with offsets fixed by the package, for every call: the
deterministic pair's counters, the single kernel's counters and FP64 column
sums (``cake_jit.SUMS_BYTES``, the largest of the package's single kernels)
and, when a window up to ``T`` runs the deterministic pair, its FP32 partial
rows.  Zero the buffer once: after every launch the kernels leave the counters
at rest and the sums at zero, so with no host write one buffer serves any
sequence of ``deterministic=False`` backward calls it is large enough for --
hidden sizes, residual flags, weight dtypes, token counts, windows and CUDA
Graph replays.  A buffer serves calls of one mode: a ``deterministic=True``
call takes the deterministic package's workspace.  Zeroing it again is
harmless.  Give each stream that may run a backward at the same time its own
buffer.  A call with a workspace issues no memset, copy or synchronization; a
call without one allocates and zero-fills one per call (a caching-allocator
call, one fill kernel of about 3 us and about 35 us of host time on B200).
The backward writes all of ``dw``.

Beside kernels resident on other streams (``deterministic=False``): the backward
kernel needs part of its SM's shared memory, registers for its CTA and one CTA
slot, and an SM serves it only once it can switch to that shared-memory
carveout.  A kernel on another stream whose CTAs hold every SM with a small
carveout therefore delays the backward's start until that kernel exits; kernels
that hold only some SMs cost it no more than the SMs they hold.  Setting
``cudaFuncAttributePreferredSharedMemoryCarveout`` to
``cudaSharedmemCarveoutMaxShared`` on such a resident kernel lets the backward
start beside it.  The windows that run the deterministic pair start and wait
exactly as the deterministic backward does, including its whole-SM programs
(see :mod:`flashinfer.cake_rmsnorm_det_train`).  A resident kernel that holds a
large dynamic shared-memory allocation on every SM delays the windows whose
programs need more shared memory than it leaves, deterministic-pair windows
included: ``cake_windows.SMEM_WAITS`` and the backward's notes list, per
architecture, route and window, the measured allocation from which the window
waits (``cake_windows.SMEM_SIDE_KB``: the sizes measured).  Programs that need a
whole SM wait beside any kernel resident on every SM, whatever its carveout: the
deterministic package's whole-SM programs where this package runs them -- the
residual-fused hidden 6144 backward in its deterministic-pair windows, and on
R200 (``sm_107a``) the backward of hidden 1536 and 4096 above 1024 tokens and of
hidden 2048 above 256 tokens in deterministic-pair windows (FP32-weight twins
included) and the forward of hidden 6144 above 4096 tokens.
``cake_windows.WHOLE_SM`` and the notes of the forward and backward list them per
architecture, route and token range; no single-kernel window of this package
needs a whole SM at the token counts measured (256 to 16384).  Beside those,
leave some SMs free of the resident kernel.  The forward programs are the
deterministic package's; see :mod:`flashinfer.cake_rmsnorm_det_train`.

Usage::

    y, rstd = cake_rmsnorm_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd)                  # deterministic
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd, deterministic=False)

    # autograd (dw is returned in w.dtype; the kernels accumulate in FP32)
    y = cake_rmsnorm(x, w, eps)                                          # deterministic
    y = cake_rmsnorm(x, w, eps, deterministic=False)
    y, h_new = cake_rmsnorm(h, w, eps, residual=u, deterministic=False)  # fused add

    # caller-owned non-deterministic workspace (counters, FP64 column sums, partial rows)
    ws = cake_rmsnorm_train_backward_workspace(T, H, x.device, deterministic=False)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd, deterministic=False, workspace=ws)
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch

from . import cake_rmsnorm_train_backend as _backend
from .api_logging import flashinfer_api
from .cake_rmsnorm_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeRMSNormFunction,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    is_cake_rmsnorm_train_supported,
    route_key,
)

# The three stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_rmsnorm_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    """See :func:`flashinfer.cake_rmsnorm_train_backend.cake_rmsnorm_train_forward`."""

    return _backend.cake_rmsnorm_train_forward(
        x,
        w,
        eps,
        residual=residual,
        out=out,
        rstd=rstd,
        residual_out=residual_out,
        deterministic=deterministic,
    )


@flashinfer_api
def cake_rmsnorm_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    deterministic: bool = True,
    g_residual: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """See :func:`flashinfer.cake_rmsnorm_train_backend.cake_rmsnorm_train_backward`."""

    return _backend.cake_rmsnorm_train_backward(
        g,
        x,
        w,
        rstd,
        deterministic=deterministic,
        g_residual=g_residual,
        workspace=workspace,
        dx=dx,
        dw=dw,
    )


@flashinfer_api
def cake_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    deterministic: bool = True,
):
    """See :func:`flashinfer.cake_rmsnorm_train_backend.cake_rmsnorm`."""

    return _backend.cake_rmsnorm(
        x, w, eps, residual=residual, deterministic=deterministic
    )


# Sphinx renders the backend docstrings; the one-line delegations satisfy the checker.
cake_rmsnorm_train_forward.__doc__ = _backend.cake_rmsnorm_train_forward.__doc__
cake_rmsnorm_train_backward.__doc__ = _backend.cake_rmsnorm_train_backward.__doc__
cake_rmsnorm.__doc__ = _backend.cake_rmsnorm.__doc__


# The deterministic names of the previous ``flashinfer.cake_rmsnorm_train`` (before
# this package it was the deterministic facade itself): each resolves on first
# access to the deterministic facade's object of that name (not in ``__all__``).
_PREVIOUS_DET_NAMES = frozenset(
    (
        "CakeRMSNormDetFunction",
        "cake_rmsnorm_det",
        "cake_rmsnorm_det_train_backward",
        "cake_rmsnorm_det_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
        "cake_rmsnorm_det_train_forward",
        "is_cake_rmsnorm_det_train_supported",
    )
)


def __getattr__(name: str):
    """A deterministic name of the previous module (:data:`_PREVIOUS_DET_NAMES`): the
    deterministic facade's object of that name."""
    if name in _PREVIOUS_DET_NAMES:
        det = importlib.import_module(f"{__package__}.cake_rmsnorm_det_train")
        return getattr(det, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
    "is_cake_rmsnorm_train_supported",
    "route_key",
]
