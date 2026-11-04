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
"""

# Experimental Cake backend: the non-deterministic ragged BF16 MoE grouped GEMM
# for SM100 / SM103 / SM107, the ``deterministic=False`` counterpart of
# ``flashinfer.experimental.cake_moe_grouped_det_gemm``.  The same three
# operations over groups of rows described by an int32 device tensor of
# cumulative end offsets (never read on the host): the row projection
# ``Y = X @ W[e].T`` (``grouped_gemm_fwd``), the activation gradient
# ``dX = G @ W[e]`` (``grouped_gemm_dgrad``) and the weight gradient
# ``dW[e] = G[e].T @ X[e]`` with bf16 or fp32 output (``grouped_gemm_wgrad``).
#
# Every entry point takes ``deterministic: bool = True`` and raises
# ``ValueError`` unless it is ``False``.  The forward pass, the activation
# gradient and the 256-wide k tile's weight gradient run copies of the
# deterministic programs (the same bits); the 512-wide k tile's weight
# gradient adds its split tail in place with unordered fp32 reductions, so its
# ``dW`` is not bitwise reproducible run to run (``cake_backend``, numerical
# mode).  Under ``torch.use_deterministic_algorithms(True)`` the weight
# gradient's entry points raise the non-determinism alert (a ``RuntimeError``,
# or a ``UserWarning`` under ``warn_only=True``).
#
# The public (experimental) entry points are the facade functions
# ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_fwd`` /
# ``grouped_gemm_dgrad`` / ``grouped_gemm_wgrad`` / ``cake_grouped_mm``, whose
# ``deterministic=True`` (the default) runs the deterministic package and
# ``deterministic=False`` this one; they carry the experimental-API marking and
# the once-only warnings and hand off to the plain functions of this package.
# The package holds those plain functions (the three one-shot helpers below,
# which prepare and launch in one call), the ``torch.autograd`` wrapper
# ``cake_grouped_mm`` / ``CakeGroupedMm`` of ``cake_autograd``, the host
# planning and the prepare-once / launch-many surface
# (``prepare_grouped_gemm_*`` returning a ``GroupedGemmLaunch``) of
# ``cake_backend``, and the JIT registration of the generated sources in
# ``cake_jit``.  The stable ``flashinfer.grouped_mm.grouped_mm_bf16(...,
# backend="cake")`` stays on the deterministic package.  The submodules are
# re-exported lazily and stay unimported until first use, so importing this
# package is cheap.  The package imports nothing from the rest of FlashInfer:
# only itself, ``torch`` and ``tvm_ffi`` (the JIT fallback of ``cake_build``
# imports ``flashinfer`` inside ``try/except ImportError``).
# ``offs`` may be given as ``[E]`` end offsets or as ``m_indptr`` ``[E + 1]``.

from __future__ import annotations

from typing import Any, Optional

import torch

__all__ = [
    "CakeGroupedMm",
    "GroupedGemmLaunch",
    "cake_grouped_mm",
    "grouped_gemm_dgrad",
    "grouped_gemm_fwd",
    "grouped_gemm_wgrad",
    "prepare_grouped_gemm_dgrad",
    "prepare_grouped_gemm_fwd",
    "prepare_grouped_gemm_wgrad",
]

_BACKEND_EXPORTS = frozenset(
    {
        "GroupedGemmLaunch",
        "prepare_grouped_gemm_dgrad",
        "prepare_grouped_gemm_fwd",
        "prepare_grouped_gemm_wgrad",
    }
)
_AUTOGRAD_EXPORTS = frozenset({"CakeGroupedMm", "cake_grouped_mm"})


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
        Must be ``False`` (``ValueError`` otherwise).  The forward programs are
        copies of the deterministic package's and give its bits.

    Returns
    -------
    torch.Tensor
        ``out``.  ``N % 256 == 0`` and ``K % 64 == 0`` are required.

    Notes
    -----
    Prepares and launches in one call.  For repeated launches over the same
    tensors (or CUDA Graph capture) use ``prepare_grouped_gemm_fwd`` once and
    call ``launch()`` on the returned object.
    """
    from .cake_backend import _require_deterministic_false, prepare_grouped_gemm_fwd

    _require_deterministic_false(deterministic, "grouped_gemm_fwd")
    return prepare_grouped_gemm_fwd(x, w, offs, out=out, deterministic=False).launch()


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
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.  ``K % 256 == 0`` and
    ``N % 64 == 0`` are required.  ``deterministic`` must be ``False``; the
    programs are copies of the deterministic package's and give its bits.  See
    ``grouped_gemm_fwd`` for the prepared form.
    """
    from .cake_backend import _require_deterministic_false, prepare_grouped_gemm_dgrad

    _require_deterministic_false(deterministic, "grouped_gemm_dgrad")
    return prepare_grouped_gemm_dgrad(g, w, offs, out=out, deterministic=False).launch()


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
    store, exact zeros for empty groups.  ``deterministic`` must be ``False``:
    with the 512-wide k tile the split tail is added with unordered fp32
    reductions, so ``dW`` is not bitwise reproducible run to run, and under
    ``torch.use_deterministic_algorithms(True)`` this function raises the
    non-determinism alert.  ``N % 256 == 0`` and ``K`` a
    multiple of the selected k tile (256, or 512 when chosen) are required.
    See ``grouped_gemm_fwd`` for the prepared form.
    """
    from .cake_backend import (
        _alert_not_deterministic,
        _prepare_wgrad,
        _require_deterministic_false,
    )

    _require_deterministic_false(deterministic, "grouped_gemm_wgrad")
    _alert_not_deterministic("grouped_gemm_wgrad")
    return _prepare_wgrad(g, x, offs, out, out_dtype, num_groups)._run()


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        from . import cake_backend

        return getattr(cake_backend, name)
    if name in _AUTOGRAD_EXPORTS:
        from . import cake_autograd

        return getattr(cake_autograd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
