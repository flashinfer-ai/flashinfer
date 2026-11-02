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

Cake ragged BF16 MoE grouped GEMM (SM100 / SM103 / SM107) with a selectable
numerical mode.

Every entry point takes ``deterministic: bool = True``:

* ``deterministic=True`` (the default) runs the deterministic package
  :mod:`flashinfer.experimental.cake_moe_grouped_det_gemm` unchanged: every
  output is reduced in a fixed order and is bitwise reproducible for a given
  shape, group distribution and device.
* ``deterministic=False`` runs the package
  ``flashinfer.experimental.cake.cake_moe_grouped_gemm``: the weight gradient
  with the 512-wide k tile adds its split tail (the tiles of its last partial
  wave, cut into k-step units over the clusters) in place with unordered fp32
  reductions, in one launch; the deterministic weight gradient sums that tail's
  fp32 partials in a fixed order in a second launch.
* With ``torch.use_deterministic_algorithms(True)`` a call with
  ``deterministic=False`` runs the deterministic package and warns once per
  entry point (``UserWarning``): determinism requested anywhere is never
  silently dropped.

Numerical contract of ``deterministic=False``: only the run-to-run bitwise
reproducibility of the weight gradient ``dW`` is relaxed.  ``Y`` and ``dX`` are
bitwise the deterministic package's (copies of the same programs), and so is
``dW`` wherever the 256-wide k tile runs or the 512-wide k tile does not split
its tail.  Every whole tile of ``dW`` is reduced by one CTA pair over the
group's rows in a fixed order; only the units of a split tail tile are added
in an order the launch does not fix.  The operands (bf16), the accumulation
(fp32), the one round-to-nearest rounding to bf16 (or the fp32 store), the
layouts, the device-resident ``offs`` contract (no host synchronisation, no
padding), the exact zeros of empty groups, the error budgets and the fixes the
deterministic package carries are unchanged.  The split tail's fp32 sums and
tickets are workspaces kept per (device, CUDA stream) at rest at zero (every
launch leaves them zero); a CUDA-graph capture takes zeroed buffers of its own.

Dispatch (one row per call; "warns" is a ``UserWarning`` once per entry point)::

    deterministic   torch.are_deterministic_algorithms_enabled()   runs
    True            any                                             deterministic package
    False           False                                           cake.cake_moe_grouped_gemm
    False           True                                            deterministic package, warns

The mode of :func:`cake_grouped_mm` is fixed at its forward call: the backward
runs the package the forward chose (a non-reentrant activation-checkpoint
replay that recomputes the forward gives the same bits; with early stop on,
the non-deterministic package's replay may skip its forward GEMM, as a
built-in op's does).  Called directly, the package
``flashinfer.experimental.cake.cake_moe_grouped_gemm`` accepts only
``deterministic=False`` and, under torch's deterministic mode, its weight
gradient raises PyTorch's non-determinism alert (a ``RuntimeError``, or a
``UserWarning`` under ``warn_only=True``) and never falls back.

Name map (the deterministic names and their compatibility aliases are
unchanged and always deterministic)::

    deterministic (unchanged)                            selectable mode (deterministic=...)
    flashinfer.grouped_mm.                               flashinfer.experimental.cake.grouped_mm.
      grouped_det_gemm_fwd                                 grouped_gemm_fwd
      grouped_det_gemm_dgrad                               grouped_gemm_dgrad
      grouped_det_gemm_wgrad                               grouped_gemm_wgrad
      cake_grouped_det_mm                                  cake_grouped_mm
    flashinfer.experimental.cake_moe_grouped_det_gemm.   flashinfer.experimental.cake.cake_moe_grouped_gemm.
      grouped_det_gemm_{fwd,dgrad,wgrad}                   grouped_gemm_{fwd,dgrad,wgrad}
      prepare_grouped_det_gemm_{fwd,dgrad,wgrad}           prepare_grouped_gemm_{fwd,dgrad,wgrad}
      GroupedDetGemmLaunch                                 GroupedGemmLaunch
      CakeGroupedDetMm, cake_grouped_det_mm                CakeGroupedMm, cake_grouped_mm

The legacy names without ``det`` -- the package
``flashinfer.experimental.cake_moe_grouped_gemm`` with its submodules and
``grouped_gemm_*`` / ``prepare_grouped_gemm_*`` / ``GroupedGemmLaunch`` /
``CakeGroupedMm`` / ``cake_grouped_mm`` names, and
``flashinfer.grouped_mm.grouped_gemm_{fwd,dgrad,wgrad}`` /
``cake_grouped_mm`` -- remain aliases of the deterministic objects, as does the
stable ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")``; this
module is the explicit namespace of the selectable mode.

This is a local operator: on a context-parallel rank it runs on that rank's
local rows; it adds no collective, and the reduction of a parameter gradient
across ranks stays with the caller.

Usage::

    from flashinfer.experimental.cake.grouped_mm import (
        cake_grouped_mm,
        grouped_gemm_wgrad,
    )

    y = cake_grouped_mm(x, w, offs)                        # deterministic (default)
    y = cake_grouped_mm(x, w, offs, deterministic=False)   # unordered dW split tail
    dw = grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32, deterministic=False)

The kernels, the host planning, the autograd wrappers and the JIT registration
live in the two packages, which every function imports inside its body (a
deferred import), so importing this module loads neither.
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import Optional

import torch

from ...api_logging import flashinfer_experimental_api

__all__ = [
    "cake_grouped_mm",
    "grouped_gemm_dgrad",
    "grouped_gemm_fwd",
    "grouped_gemm_wgrad",
]

# The entry points that already warned about the deterministic fallback under torch's deterministic mode.
_FALLBACK_WARNED: set[str] = set()

# The flashinfer package directory: frames below it are the facade, its decorators and the packages.
_PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside the flashinfer package, for a warning
    issued by the function that calls this one.

    The decorators between the caller and the facade (the experimental-API wrapper, and more
    frames when API logging is enabled) vary, so a fixed ``stacklevel`` would name a frame of
    flashinfer instead of the call the caller has to change.
    """

    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None and not os.path.abspath(
            parent.f_code.co_filename
        ).startswith(_PACKAGE_ROOT + os.sep):
            return level
        frame = parent
    return 3


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
                stacklevel=_caller_stacklevel(),
            )
        return True
    return False


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_fwd")
def grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T`` for every group ``e``.

    Parameters
    ----------
    x : torch.Tensor
        ``[sum_m, K]`` bfloat16 rows of all groups, group after group.
    w : torch.Tensor
        ``[E, N, K]`` bfloat16 weights, one ``[N, K]`` matrix per group.
    offs : torch.Tensor
        ``[E]`` int32 cumulative end offsets on the device (``offs[-1] == sum_m``),
        or ``m_indptr`` ``[E + 1]`` with a leading 0; read by the kernel, never
        on the host.
    out : Optional[torch.Tensor]
        ``[sum_m, N]`` bfloat16 output (allocated when omitted); may be row-padded.
    deterministic : bool
        ``True`` (default): the deterministic package's forward.  ``False``: the
        non-deterministic package's copy of the same programs (bitwise the same
        ``Y``).  torch's deterministic mode turns ``False`` into ``True`` with a
        one-time warning.

    Returns
    -------
    torch.Tensor
        ``out``.  ``N % 256 == 0`` and ``K % 64 == 0`` are required.
    """
    if _deterministic(deterministic, "grouped_gemm_fwd"):
        from ..cake_moe_grouped_det_gemm import grouped_det_gemm_fwd

        return grouped_det_gemm_fwd(x, w, offs, out=out)
    from .cake_moe_grouped_gemm import grouped_gemm_fwd as _impl

    return _impl(x, w, offs, out=out, deterministic=False)


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_dgrad")
def grouped_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.
    ``K % 256 == 0`` and ``N % 64 == 0`` are required.  ``deterministic=True``
    (default) runs the deterministic package; ``False`` the non-deterministic
    package's copy of the same programs (bitwise the same ``dX``); torch's
    deterministic mode turns ``False`` into ``True`` with a one-time warning.
    """
    if _deterministic(deterministic, "grouped_gemm_dgrad"):
        from ..cake_moe_grouped_det_gemm import grouped_det_gemm_dgrad

        return grouped_det_gemm_dgrad(g, w, offs, out=out)
    from .cake_moe_grouped_gemm import grouped_gemm_dgrad as _impl

    return _impl(g, w, offs, out=out, deterministic=False)


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_wgrad")
def grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, one rounding at the
    store, exact zeros for empty groups.  ``N % 256 == 0`` and ``K`` a multiple
    of the selected k tile (256, or 512 when chosen) are required.

    ``deterministic=True`` (default): the deterministic package, bitwise
    reproducible.  ``False``: with the 512-wide k tile the split tail is added
    in place with unordered fp32 reductions (one launch instead of two), within
    the same error budgets but not bitwise reproducible run to run; with the
    256-wide k tile the deterministic programs run.  torch's deterministic mode
    turns ``False`` into ``True`` with a one-time warning.
    """
    if _deterministic(deterministic, "grouped_gemm_wgrad"):
        from ..cake_moe_grouped_det_gemm import grouped_det_gemm_wgrad

        return grouped_det_gemm_wgrad(
            g, x, offs, out=out, out_dtype=out_dtype, num_groups=num_groups
        )
    from .cake_moe_grouped_gemm import grouped_gemm_wgrad as _impl

    return _impl(
        g,
        x,
        offs,
        out=out,
        out_dtype=out_dtype,
        num_groups=num_groups,
        deterministic=False,
    )


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.cake_grouped_mm")
def cake_grouped_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    deterministic: bool = True,
) -> torch.Tensor:
    """Differentiable ragged grouped GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``.

    Parameters
    ----------
    x : torch.Tensor
        ``[sum_m, K]`` bfloat16 rows of all groups, group after group.
    w : torch.Tensor
        ``[E, N, K]`` bfloat16 weights (``[N, K]`` per group).
    offs : torch.Tensor
        int32 device tensor: ``[E]`` cumulative end offsets or ``m_indptr``
        ``[E + 1]``.  Read by the kernels, never on the host.
    deterministic : bool
        ``True`` (default): the deterministic autograd wrapper
        (``cake_grouped_det_mm``), bitwise reproducible gradients.  ``False``:
        ``CakeGroupedMm`` of the non-deterministic package, whose backward runs
        the deterministic data-gradient programs and the weight gradient with
        the unordered split tail.  The mode is chosen at the forward call
        (torch's deterministic mode then turns ``False`` into ``True`` with a
        one-time warning) and the backward runs the same package.

    Returns
    -------
    torch.Tensor
        ``[sum_m, N]`` bfloat16 (the same bits in both modes).  ``backward``
        produces ``x.grad`` ``[sum_m, K]`` bfloat16 and ``w.grad`` ``[E, N, K]``
        in ``w``'s dtype; use :func:`grouped_gemm_wgrad` for an fp32 weight
        gradient.
    """
    if _deterministic(deterministic, "cake_grouped_mm"):
        from ..cake_moe_grouped_det_gemm.cake_autograd import cake_grouped_det_mm

        return cake_grouped_det_mm(x, w, offs, deterministic=True)
    from .cake_moe_grouped_gemm.cake_autograd import cake_grouped_mm as _impl

    return _impl(x, w, offs, deterministic=False)
