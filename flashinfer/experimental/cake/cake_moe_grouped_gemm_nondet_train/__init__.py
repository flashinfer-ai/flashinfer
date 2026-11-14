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

# Experimental Cake backend of ``flashinfer.cake_moe_grouped_gemm_nondet_train``: the
# non-deterministic ragged BF16 MoE grouped GEMM for SM100 / SM103 / SM107, the
# counterpart of ``flashinfer.cake_moe_grouped_gemm_det_train``.  The same three
# operations over groups of rows described by an int32 device tensor of
# cumulative end offsets (never read on the host): the row projection
# ``Y = X @ W[e].T`` (``cake_moe_grouped_gemm_nondet_train_forward``), the
# activation gradient ``dX = G @ W[e]`` (``cake_moe_grouped_gemm_nondet_train_dgrad``)
# and the weight gradient ``dW[e] = G[e].T @ X[e]`` with bf16 or fp32 output
# (``cake_moe_grouped_gemm_nondet_train_wgrad``).
#
# The module is the mode: no entry point takes a ``deterministic`` argument, and
# nothing here routes to the deterministic package.  The forward pass, the
# activation gradient and the 256-wide k tile's weight gradient run copies of the
# deterministic programs (the same bits); the 512-wide k tile's weight gradient
# finishes its split tail in the same launch: each unit stores its fp32 partial
# to a slot and the tile's last unit sums the slots in slot order
# (``cake_backend``, numerical mode).  Every output repeats bit for bit for the
# same inputs, shapes, device SM count and configuration, and no launch alerts under
# ``torch.use_deterministic_algorithms(True)``.  The alert stays for an
# order-varying launch, which only an architecture listed in the host plan's
# ``unordered_tail_archs`` (empty) plans: a ``RuntimeError``, or a
# ``UserWarning`` under ``warn_only=True``, after which it runs.
#
# The public entry points are the functions of the top-level module
# ``flashinfer.cake_moe_grouped_gemm_nondet_train`` (stable APIs), which take the
# same arguments and hand off to the plain functions of this package, which have
# the same names.
# The package holds those plain functions (the three one-shot helpers below,
# which prepare and launch in one call), the ``torch.autograd`` wrapper
# ``cake_moe_grouped_gemm_nondet`` / ``CakeMoEGroupedGemmNondetFunction`` of
# ``cake_autograd``, the host planning and the prepare-once / launch-many surface
# (``cake_moe_grouped_gemm_nondet_train_prepare_{forward,dgrad,wgrad}`` returning
# a ``CakeMoEGroupedGemmNondetLaunch``) of ``cake_backend``, and the JIT
# registration of the generated sources in ``cake_jit``.  The stable
# ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` stays on the
# deterministic package.  The submodules are re-exported lazily and stay
# unimported until first use, so importing this package is cheap.  The package
# imports nothing from the rest of FlashInfer: only itself, ``torch`` and
# ``tvm_ffi`` (the JIT fallback of ``cake_build`` imports ``flashinfer`` inside
# ``try/except ImportError``).
# ``offs`` may be given as ``[E]`` end offsets or as ``m_indptr`` ``[E + 1]``.

from __future__ import annotations

from typing import Any, Optional

import torch

__all__ = [
    "CakeMoEGroupedGemmNondetFunction",
    "CakeMoEGroupedGemmNondetLaunch",
    "cake_moe_grouped_gemm_nondet",
    "cake_moe_grouped_gemm_nondet_train_dgrad",
    "cake_moe_grouped_gemm_nondet_train_forward",
    "cake_moe_grouped_gemm_nondet_train_prepare_dgrad",
    "cake_moe_grouped_gemm_nondet_train_prepare_forward",
    "cake_moe_grouped_gemm_nondet_train_prepare_wgrad",
    "cake_moe_grouped_gemm_nondet_train_wgrad",
]

_BACKEND_EXPORTS = frozenset(
    {
        "CakeMoEGroupedGemmNondetLaunch",
        "cake_moe_grouped_gemm_nondet_train_prepare_dgrad",
        "cake_moe_grouped_gemm_nondet_train_prepare_forward",
        "cake_moe_grouped_gemm_nondet_train_prepare_wgrad",
    }
)
_AUTOGRAD_EXPORTS = frozenset(
    {"CakeMoEGroupedGemmNondetFunction", "cake_moe_grouped_gemm_nondet"}
)


def cake_moe_grouped_gemm_nondet_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T`` for every group ``e``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

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

    Returns
    -------
    torch.Tensor
        ``out``.  ``N % 256 == 0`` and ``K % 64 == 0`` are required.

    Notes
    -----
    The forward programs are copies of the deterministic package's and give its
    bits; they never alert under torch's deterministic mode.
    Prepares and launches in one call.  For repeated launches over the same
    tensors (or CUDA Graph capture) use ``cake_moe_grouped_gemm_nondet_train_prepare_forward`` once and
    call ``launch()`` on the returned object.
    """
    from .cake_backend import cake_moe_grouped_gemm_nondet_train_prepare_forward

    return cake_moe_grouped_gemm_nondet_train_prepare_forward(
        x, w, offs, out=out
    ).launch()


def cake_moe_grouped_gemm_nondet_train_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.  ``K % 256 == 0`` and
    ``N % 64 == 0`` are required.  The programs are copies of the deterministic
    package's and give its bits (no alert under torch's deterministic mode).  See
    ``cake_moe_grouped_gemm_nondet_train_forward`` for the prepared form.
    """
    from .cake_backend import cake_moe_grouped_gemm_nondet_train_prepare_dgrad

    return cake_moe_grouped_gemm_nondet_train_prepare_dgrad(
        g, w, offs, out=out
    ).launch()


def cake_moe_grouped_gemm_nondet_train_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    accumulate: bool = False,
) -> torch.Tensor:
    """``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` for every group ``e``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, one rounding at the
    store, exact zeros for empty groups.  With the 512-wide k tile a split tail
    is summed in slot order in the same launch, so ``dW`` repeats bit for bit
    for the same inputs, shapes, device SM count and configuration, and no
    launch alerts under ``torch.use_deterministic_algorithms(True)``.
    ``N % 256 == 0`` and ``K`` a
    multiple of the selected k tile (256, or 512 when chosen) are required.
    See ``cake_moe_grouped_gemm_nondet_train_forward`` for the prepared form.

    ``accumulate=True`` adds into ``out`` instead, ``out[e] += G[e].T @ X[e]``
    (Megatron's gradient-accumulation fusion; ``out`` required, bf16 or fp32,
    ``out_dtype`` None or its dtype), and returns ``out``: fused into the
    512-wide k tile's epilogue where the route serves that stage's accumulate
    build, else the overwrite into a scratch followed by ``out.add_``
    (``cake_moe_grouped_gemm_nondet_train_prepare_wgrad`` and the package README give the routes and
    the numerical contract).
    """
    from .cake_backend import _prepare_wgrad

    return _prepare_wgrad(
        g, x, offs, out, out_dtype, num_groups, accumulate=accumulate
    )._run("cake_moe_grouped_gemm_nondet_train_wgrad")


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        from . import cake_backend

        return getattr(cake_backend, name)
    if name in _AUTOGRAD_EXPORTS:
        from . import cake_autograd

        return getattr(cake_autograd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
