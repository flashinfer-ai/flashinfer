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
with the non-deterministic weight gradient.

Mode and dispatch rules:

* The mode is part of the name, and only of the name.  This is the
  non-deterministic package: every public name carries ``nondet`` (the
  constants ``ARCHES`` and ``SUPPORTED_COMPUTE_CAPABILITIES`` excepted).  It
  runs its own kernels only and never the deterministic package
  :mod:`flashinfer.cake_rmsnorm_det_train`, whose weight gradient ``dw`` is
  bitwise reproducible and whose names carry ``det``.  No entry point of
  either package takes a ``deterministic`` keyword: passing one raises
  Python's ``TypeError`` for an unexpected keyword argument.
* ``torch.use_deterministic_algorithms(True)`` alerts only before work whose
  result can vary from run to run (PyTorch's rule).
  :func:`cake_rmsnorm_nondet_train_backward` always alerts (whatever window
  form the call runs): ``RuntimeError`` naming
  ``flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward``; with
  ``warn_only=True`` one ``UserWarning`` naming it, then this package's
  backward runs (no fallback to the deterministic package).
  :func:`cake_rmsnorm_nondet_train_forward` never alerts (no cross-row
  reduction; its results are bitwise reproducible).  The autograd
  :func:`cake_rmsnorm_nondet` / :class:`CakeRMSNormNondetFunction`: the
  forward never alerts; the backward alerts when it runs, by the same rule,
  naming ``flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det``.  The
  workspace helpers, :func:`cake_rmsnorm_nondet_train_route_key` and
  :func:`is_cake_rmsnorm_nondet_train_supported` never consult the setting.
* The ``flashinfer`` top level binds none of this package's names; reach it
  as the module ``flashinfer.cake_rmsnorm_nondet_train``.
* There are no aliases: no name here resolves to an object of the
  deterministic package, and the names of the former ``cake_rmsnorm_train``
  module do not exist.
* A backward does not check which package produced its inputs: both packages
  run the same forward programs and save the same ``rstd``, and a workspace is
  an untagged raw buffer (size it with the helper of the package that uses it).

Forward: ``r_t = (mean_j x_tj^2 + eps)^(-1/2)`` in FP32, ``y = BF16((x * r) * w)``.
The only tensor saved for the backward besides the inputs is the FP32 per-row
reciprocal RMS ``rstd`` (``[T]``).  Backward: one fused pass produces
``dx = BF16(r * g * w - x * r^3 * mean(g * w * x))`` and the FP32 weight
gradient ``dw = sum_t g * x * r``.  The optional residual-add fusion computes
``h_new = BF16(h + u)`` first, normalizes ``h_new``, and in the backward adds
the gradient flowing directly through ``h_new`` so that one ``dx`` tensor is
the gradient of both ``h`` and ``u``.

The forward runs the deterministic package's forward programs, rendered into
this package (the forward has no cross-row reduction).  The backward runs, per
token-count window, the form this build's registry lists for the window
(``cake_jit.program_windows``; the backend's backward notes are generated
from it).  The single-kernel form, in the windows that ship it: persistent
CTAs claim row chunks, write ``dx`` and add their per-column ``dw`` sums into
FP64 sums in the workspace in the order they finish; the CTA that finishes
last rounds the sums into ``dw``.  ``dx`` is bitwise reproducible and within
one BF16 ulp of ``max(|dx|, rms(dx))`` of the deterministic backward's ``dx``
(not always bitwise equal to it); ``dw`` is not guaranteed bitwise
reproducible (the order of the FP64 additions depends on scheduling).  In the
windows those notes name, a twosum single kernel keeps each
thread's ``dw`` column sums exact (error-free FP32 products, TwoSum
compensation) before the FP64 additions, so its ``dw`` error does not grow
with ``T``; a kblock single kernel adds the ``dw`` products of each block of 8
rows in FP32 and keeps the block sums as FP32 TwoSum pairs before the FP64
additions, so its ``dw`` error grows slowly with ``T``; a pair-row single
kernel keeps the twosum sums as FP32 pairs, writes one row of pairs per CTA
into scratch in the workspace and folds the rows in helper CTAs once every row
is in, with no FP64 additions (the order of its rows depends on scheduling, so
its ``dw`` is not guaranteed bitwise reproducible either).  Every other window
runs the deterministic package's backward pair (main kernel and fixed-order
reduce), rendered into this package.  The package also accepts ``[..., H]``
inputs whose leading dimensions collapse into rows (for example a
``[T, heads, 256]`` per-head view sharing one ``w[256]``) and an FP32 weight
on the routes that ship the FP32-weight programs.

All kernels use FP32 statistics and accumulation (the non-deterministic
single kernel's ``dw`` sums across CTAs in FP64, the pair-row single kernel's
in FP32 pairs) with IEEE division and square root (no
fast-math).  Inputs and outputs are BF16; ``rstd`` and ``dw``
are FP32.  Row-strided inputs (``x``, ``residual``, ``g``, ``g_residual``)
are accepted when ``stride(1) == 1``, ``stride(0)`` is a multiple of eight
elements and the storage is aligned (32 bytes for the forward inputs, 16 for
the backward); outputs are contiguous.  The token count ``T`` is a runtime
scalar: it may change between calls, ``T == 1`` is supported and ``T == 0``
returns empty outputs (a zero ``dw``) without launching.

The backward's workspace (``cake_rmsnorm_nondet_train_backward_workspace_bytes(T,
H, device)`` bytes) has one layout, with offsets fixed by the
package, for every call: the deterministic pair's counters, the single kernel's
counters and FP64 column sums (``cake_jit.SUMS_BYTES``, the largest of the
package's single kernels) and, when a window up to ``T`` runs the deterministic
pair or a pair-row single kernel, its FP32 partial rows or the pair-row
scratch (one region after the sums; it needs no rest state).  Zero the buffer
once: after every launch the
kernels leave the counters at rest and the sums at zero, so with no host write
one buffer serves any sequence of backward calls it is large enough for --
hidden sizes, residual flags, weight dtypes, token counts, windows and CUDA
Graph replays.
The deterministic package's workspace has another layout; do not pass this
buffer to it.  Zeroing it again is harmless.  Give each stream that may run a
backward at the same time its own buffer.  A call with a workspace issues no
memset, copy or synchronization; a call without one allocates and zero-fills
one per call (a caching-allocator call, one fill kernel of about 3 us and
about 35 us of host time on B200).  The backward writes all of ``dw``.

Beside kernels resident on other streams: the backward
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
included: ``cake_windows.SMEM_WAITS`` and the backend's backward notes list, per
architecture, route and window, the measured allocation from which the window
waits (``cake_windows.SMEM_SIDE_KB``: the sizes measured).  Programs that need a
whole SM wait beside any kernel resident on every SM, whatever its carveout: the
deterministic package's whole-SM programs where this package runs them -- the
residual-fused hidden 6144 backward in its deterministic-pair windows, and on
R200 (``sm_107a``) the backward of hidden 1536 and 4096 above 1024 tokens and of
hidden 2048 above 256 tokens in deterministic-pair windows (FP32-weight twins
included) and the forward of hidden 6144 above 4096 tokens.
``cake_windows.WHOLE_SM`` and the backend's forward and backward notes list them
per architecture, route and token range; no single-kernel window of this
package needs a whole SM at the token counts measured (256 to 16384).  Beside
those, leave some SMs free of the resident kernel.  The forward programs are
the deterministic package's; see :mod:`flashinfer.cake_rmsnorm_det_train`.

Usage::

    from flashinfer.cake_rmsnorm_nondet_train import (
        cake_rmsnorm_nondet,
        cake_rmsnorm_nondet_train_backward,
        cake_rmsnorm_nondet_train_backward_workspace,
        cake_rmsnorm_nondet_train_forward,
    )

    y, rstd = cake_rmsnorm_nondet_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_nondet_train_backward(g, x, w, rstd)  # dw is FP32 [H]

    # autograd (dw is returned in w.dtype; the kernels accumulate in FP32)
    y = cake_rmsnorm_nondet(x, w, eps)
    y, h_new = cake_rmsnorm_nondet(h, w, eps, residual=u)  # fused add

    # caller-owned workspace (counters, FP64 column sums, partial rows)
    ws = cake_rmsnorm_nondet_train_backward_workspace(T, H, x.device)
    dx, dw = cake_rmsnorm_nondet_train_backward(g, x, w, rstd, workspace=ws)

    # the bitwise-reproducible weight gradient: the deterministic package
    from flashinfer.cake_rmsnorm_det_train import cake_rmsnorm_det
    y = cake_rmsnorm_det(x, w, eps)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


# The backend package is imported on first use and cached: importing this module does not
# import ``flashinfer.experimental``, and a call does not repeat the import.
@functools.cache
def _backend():
    from .experimental.cake import cake_rmsnorm_nondet_train as backend  # noqa: PLC0415 - the backend package, imported on first use

    return backend


# The three stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_rmsnorm_nondet_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Fused BF16 RMSNorm forward with FP32 statistics.

    This package's forward programs: the deterministic package's programs
    rendered here (the forward has no cross-row reduction, so its outputs are
    bitwise reproducible).  ``torch.use_deterministic_algorithms`` does not
    affect the forward.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input of any layout.  Row-strided views with a contiguous
        feature dimension, 32-byte aligned storage and a row stride that is a
        multiple of 32 bytes are read in place; other layouts (odd column offset,
        odd row stride, transposed storage) are materialized once.  A tensor of
        more dimensions, ``[..., H]``, is normalized row by row when its feature
        dimension is contiguous and its leading dimensions are contiguous with
        one another in row-major order (``stride(i) == stride(i + 1) * size(i +
        1)``; a ``[T, heads, H]`` slice of a wider ``[T, heads, W]`` buffer
        qualifies): the leading dimensions collapse into ``prod(...)`` rows, the
        outputs take the input's leading shape (``rstd`` is ``[...]``), ``w`` is
        shared by every row and the backward's ``dw`` sums them all; any other
        N-D layout raises ``ValueError`` naming the rule.  With ``residual`` this
        is the residual stream ``h`` and the normalized value is
        ``h_new = BF16(h + residual)``.
    w : torch.Tensor
        BF16 ``[H]`` weight, read in place when contiguous and 32-byte aligned,
        else copied once.  An FP32 ``[H]`` weight -- the effective weight
        ``1 + w`` a zero-centred model computes in FP32 -- is accepted on the
        routes that ship the FP32-weight programs (``cake_jit.weight_dtypes``),
        else ``ValueError``.
    eps : float
        Variance epsilon (FP32 inside the kernel).
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization (same
        layout rules as ``x``).  Enables the fused residual-add route.
    out : torch.Tensor, optional
        Preallocated BF16 ``y`` of ``x``'s shape. Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.
    rstd : torch.Tensor, optional
        Preallocated FP32 reciprocal RMS of ``x``'s leading shape (``[...]``
        for ``x`` of shape ``[..., H]``). Contiguous aligned buffers are written
        directly; any other layout receives a copy after the launch.
    residual_out : torch.Tensor, optional
        Preallocated BF16 ``h_new`` of ``x``'s shape, used only with ``residual``.
        Contiguous aligned buffers are written directly; any other layout
        receives a copy after the launch.

    Returns
    -------
    ``(y, rstd)`` or ``(y, rstd, h_new)`` when ``residual`` is given.
    ``rstd`` is the FP32 per-row ``1 / sqrt(mean(h_new^2) + eps)`` the backward
    needs; no other ``[T, H]`` temporary is produced.

    Notes
    -----
    The program windows of every architecture and route, and the program
    each window runs, are listed in the docstring of
    :func:`flashinfer.experimental.cake.cake_rmsnorm_nondet_train.cake_rmsnorm_nondet_train_forward`.
    """

    return _backend().cake_rmsnorm_nondet_train_forward(
        x,
        w,
        eps,
        residual=residual,
        out=out,
        rstd=rstd,
        residual_out=residual_out,
    )


@flashinfer_api
def cake_rmsnorm_nondet_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    g_residual: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused BF16 RMSNorm backward: ``dx`` and the FP32 weight gradient ``dw``.

    The call runs, per window of ``T``, the form the "Program selection by
    token count" notes list for that window: in the windows that ship it, one
    kernel that writes ``dx`` and adds the ``dw`` sums of its CTAs into FP64
    sums in the workspace, which the CTA that finishes last rounds into ``dw``
    (``dx`` is bitwise reproducible and within one BF16 ulp of
    ``max(|dx|, rms(dx))`` of the deterministic kernel's ``dx``, not always
    bitwise equal to it; the order of the FP64 additions depends on
    scheduling, so ``dw`` is not guaranteed bitwise reproducible); in the
    others, the deterministic package's backward pair rendered into this
    package.

    While ``torch.use_deterministic_algorithms(True)`` is in effect every call
    raises ``RuntimeError`` naming
    :func:`flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward`,
    the deterministic package's backward (``dw`` through a fixed row-chunk
    partition and a fixed-order reduction, bitwise reproducible for a given
    token count on devices with the same SM count), whatever window form it
    would run; with ``warn_only=True`` it warns once and runs this package's
    backward (never the deterministic package's).

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[T, H]`` gradient of ``y`` of any layout.  Views with a contiguous
        feature dimension, 16-byte aligned storage and a row stride that is a
        multiple of 16 bytes (expanded or sliced autograd gradients included) are
        read in place; other layouts are materialized once.  An N-D gradient,
        ``[..., H]``, takes ``x``'s shape and collapses its leading dimensions
        into rows under the forward's rule.
    x : torch.Tensor
        The BF16 tensor that was normalized (``x`` of the plain forward,
        ``h_new`` of the residual forward); same layout rules as ``g``.
    w : torch.Tensor
        The forward's weight: BF16 ``[H]`` (read in place when contiguous and
        16-byte aligned), or, on the FP32-weight routes, FP32 ``[H]`` (in place
        when contiguous and 32-byte aligned); any other layout is copied once.
    rstd : torch.Tensor
        FP32 ``[T]`` reciprocal RMS returned by the forward (of the leading
        shape ``[...]`` for an N-D call, or flat ``[prod(...)]``).
    g_residual : torch.Tensor, optional
        BF16 gradient of ``g``'s shape flowing directly through ``h_new``
        (residual forward), any layout.  It is added once so that the returned
        ``dx`` is the gradient of both residual-add inputs.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_rmsnorm_nondet_train_backward_workspace`, at least as large as that
        helper returns for this call (one buffer serves calls of any hidden
        size, residual flag and weight dtype it is large enough for; see
        :func:`cake_rmsnorm_nondet_train_backward_workspace_bytes`).  The call issues
        no memset, copy or synchronization on it.  When omitted, the call
        allocates and zero-fills one: a caching-allocator call, one fill kernel
        (about 3 us on B200) and about 35 us of host time per call.  One buffer
        per backward that may run concurrently (on another stream).
    dx : torch.Tensor, optional
        Preallocated BF16 ``dx`` of ``x``'s shape. Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.
        Its previous contents are ignored.
    dw : torch.Tensor, optional
        Preallocated FP32 ``[H]`` weight gradient. Contiguous aligned buffers
        are written directly; any other layout receives a copy after the launch.
        Its previous contents are ignored.

    Returns
    -------
    ``(dx, dw)`` with ``dw`` in FP32; the caller decides whether to cast it.
    ``T == 0`` launches nothing and returns an empty ``dx`` and a zero ``dw``.

    Notes
    -----
    The program windows of every architecture and route, and the program
    each window runs, are listed in the docstring of
    :func:`flashinfer.experimental.cake.cake_rmsnorm_nondet_train.cake_rmsnorm_nondet_train_backward`.
    """

    return _backend().cake_rmsnorm_nondet_train_backward(
        g,
        x,
        w,
        rstd,
        g_residual=g_residual,
        workspace=workspace,
        dx=dx,
        dw=dw,
    )


@flashinfer_api
def cake_rmsnorm_nondet(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
):
    """Autograd-enabled fused BF16 RMSNorm (optionally with a fused residual add).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input on a supported CUDA device, or ``[..., H]`` under
        the layout rule of :func:`cake_rmsnorm_nondet_train_forward`.
    w : torch.Tensor
        BF16 ``[H]`` RMSNorm weight (FP32 on the routes that ship the
        FP32-weight programs); the weight gradient is returned in ``w.dtype``.
    eps : float
        Variance epsilon evaluated in FP32 by the kernel.
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization.  When
        provided, the normalized input ``h_new`` is returned with the output.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 ``y`` of ``x``'s shape, or ``(y, h_new)`` when ``residual`` is
        given.  Backward launches the kernels of
        :func:`cake_rmsnorm_nondet_train_backward`.

    Notes
    -----
    This package's kernels, whose weight gradient is not guaranteed bitwise
    reproducible (some token-count windows sum it over the CTAs in a
    scheduling-dependent order); the deterministic package's autograd entry
    point is :func:`flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det`.  The
    forward does not consult ``torch.use_deterministic_algorithms``; while it
    is ``True`` the backward raises ``RuntimeError`` naming that entry point
    when it runs, or, with ``warn_only=True``, warns once and runs this
    package's backward.
    """

    return _backend().cake_rmsnorm_nondet(x, w, eps, residual=residual)


# The backend's own objects this module exports (the autograd class, the constants and
# the helpers) resolve on first access and are cached in this module; each is the
# backend's object itself.
_BACKEND_NAMES = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormNondetFunction",
    "cake_rmsnorm_nondet_train_backward_workspace",
    "cake_rmsnorm_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_nondet_train_route_key",
    "is_cake_rmsnorm_nondet_train_supported",
)


def __getattr__(name: str) -> Any:
    if name not in _BACKEND_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_backend(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BACKEND_NAMES))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeRMSNormNondetFunction",  # noqa: F822
    "cake_rmsnorm_nondet",
    "cake_rmsnorm_nondet_train_backward",
    "cake_rmsnorm_nondet_train_backward_workspace",  # noqa: F822
    "cake_rmsnorm_nondet_train_backward_workspace_bytes",  # noqa: F822
    "cake_rmsnorm_nondet_train_forward",
    "cake_rmsnorm_nondet_train_route_key",  # noqa: F822
    "is_cake_rmsnorm_nondet_train_supported",  # noqa: F822
]
