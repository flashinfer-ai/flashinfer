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

# Experimental Cake backend of ``flashinfer.cake_moe_grouped_gemm_det_train``: the
# deterministic ragged BF16 MoE grouped GEMM for SM100 / SM103 / SM107.  Three
# operations over groups of rows described by an int32 device tensor of
# cumulative end offsets (never read on the host): the row projection
# ``Y = X @ W[e].T`` (``cake_moe_grouped_gemm_det_train_forward``), the activation
# gradient ``dX = G @ W[e]`` (``cake_moe_grouped_gemm_det_train_dgrad``) and the
# weight gradient ``dW[e] = G[e].T @ X[e]`` with bf16 or fp32 output
# (``cake_moe_grouped_gemm_det_train_wgrad``), also added into a caller's fp32 /
# bf16 buffer (``accumulate=True``; the autograd wrapper's
# ``fuse_wgrad_accumulation`` adds into ``w.main_grad``).
#
# The public entry points are the functions of the top-level module
# ``flashinfer.cake_moe_grouped_gemm_det_train`` (stable APIs) and
# ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")``: they hand off
# to the plain functions of this package, which have the same names.  The package holds
# those plain functions (the three one-shot helpers below, which prepare and
# launch in one call), the ``torch.autograd`` wrapper
# ``cake_moe_grouped_gemm_det`` / ``CakeMoEGroupedGemmDetFunction`` of
# ``cake_autograd``, the host planning and the prepare-once / launch-many surface
# (``cake_moe_grouped_gemm_det_train_prepare_{forward,dgrad,wgrad}`` returning a
# ``CakeMoEGroupedGemmDetLaunch``) of ``cake_backend`` together with the
# stable-API handoff ``cake_backend.grouped_det_mm_bf16_cake``, and the JIT
# registration of the generated sources in ``cake_jit``.  The submodules are
# re-exported lazily and stay unimported until first use, so importing this
# package is cheap.  The package imports nothing from the rest of FlashInfer:
# only itself, ``torch`` and ``tvm_ffi`` (the JIT fallback of ``cake_build``
# imports ``flashinfer`` inside ``try/except ImportError``).
# ``offs`` may be given as ``[E]`` end offsets or as ``m_indptr`` ``[E + 1]``.

from __future__ import annotations

from typing import Any, Optional

import torch

__all__ = [
    "CakeMoEGroupedGemmDetFunction",
    "CakeMoEGroupedGemmDetLaunch",
    "cake_moe_grouped_gemm_det",
    "cake_moe_grouped_gemm_det_train_dgrad",
    "cake_moe_grouped_gemm_det_train_forward",
    "cake_moe_grouped_gemm_det_train_prepare_dgrad",
    "cake_moe_grouped_gemm_det_train_prepare_forward",
    "cake_moe_grouped_gemm_det_train_prepare_wgrad",
    "cake_moe_grouped_gemm_det_train_wgrad",
]

_BACKEND_EXPORTS = frozenset(
    {
        "CakeMoEGroupedGemmDetLaunch",
        "cake_moe_grouped_gemm_det_train_prepare_dgrad",
        "cake_moe_grouped_gemm_det_train_prepare_forward",
        "cake_moe_grouped_gemm_det_train_prepare_wgrad",
    }
)
_AUTOGRAD_EXPORTS = frozenset(
    {"CakeMoEGroupedGemmDetFunction", "cake_moe_grouped_gemm_det"}
)


def cake_moe_grouped_gemm_det_train_forward(
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
    Prepares and launches in one call.  For repeated launches over the same
    tensors (or CUDA Graph capture) use ``cake_moe_grouped_gemm_det_train_prepare_forward`` once and
    call ``launch()`` on the returned object.
    """
    from .cake_backend import cake_moe_grouped_gemm_det_train_prepare_forward

    return cake_moe_grouped_gemm_det_train_prepare_forward(x, w, offs, out=out).launch()


def cake_moe_grouped_gemm_det_train_dgrad(
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
    ``N % 64 == 0`` are required.  See ``cake_moe_grouped_gemm_det_train_forward`` for the prepared form.
    """
    from .cake_backend import cake_moe_grouped_gemm_det_train_prepare_dgrad

    return cake_moe_grouped_gemm_det_train_prepare_dgrad(g, w, offs, out=out).launch()


def cake_moe_grouped_gemm_det_train_wgrad(
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
    dtype of ``out``, else bfloat16).  fp32 accumulation, bitwise deterministic, exact zeros for empty
    groups.  ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256,
    or 512 when chosen) are required.  See ``cake_moe_grouped_gemm_det_train_forward`` for the
    prepared form.

    ``accumulate=True`` (keyword-only, default ``False``) adds into ``out``
    instead, ``out[e] += G[e].T @ X[e]`` (Megatron's gradient-accumulation
    fusion into an fp32 or bf16 ``main_grad``; ``out`` required, bfloat16 or
    float32, ``out_dtype`` None or its dtype, no overlap of ``out`` with itself,
    ``g``, ``x`` or ``offs``: ``ValueError``), and returns ``out``, which holds
    AccumulateGrad's bits of the deterministic weight gradient in its dtype,
    bitwise reproducible run to run: fused into the 512-wide k tile's programs
    (the main stage and the ordered tail reduce add into ``out`` in place)
    where the route serves their accumulate builds for ``out``'s dtype, else
    the overwrite into a scratch of ``out``'s dtype (``E * N * K`` elements,
    ``4 * E * N * K`` bytes for fp32) added with ``out.add_`` on the same
    stream; both give the same bits
    (``cake_moe_grouped_gemm_det_train_prepare_wgrad`` and the package README
    give the routes; a prepared launch is CUDA-graph capturable and every
    launch and replay adds again).  ``sum_m == 0`` leaves ``out`` unchanged.
    """
    from .cake_backend import cake_moe_grouped_gemm_det_train_prepare_wgrad

    return cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g,
        x,
        offs,
        out=out,
        out_dtype=out_dtype,
        num_groups=num_groups,
        accumulate=accumulate,
    ).launch()


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        from . import cake_backend

        return getattr(cake_backend, name)
    if name in _AUTOGRAD_EXPORTS:
        from . import cake_autograd

        return getattr(cake_autograd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
