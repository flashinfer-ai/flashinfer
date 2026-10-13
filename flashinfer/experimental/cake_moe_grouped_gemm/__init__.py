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

# Experimental Cake backend: ragged BF16 MoE grouped GEMM for SM100 / SM103 /
# SM107.  Three operations over groups of rows described by an int32 device
# tensor of cumulative end offsets (never read on the host): the row
# projection ``Y = X @ W[e].T`` (``grouped_gemm_fwd``), the activation gradient
# ``dX = G @ W[e]`` (``grouped_gemm_dgrad``) and the weight gradient
# ``dW[e] = G[e].T @ X[e]`` with bf16 or fp32 output (``grouped_gemm_wgrad``).
#
# The one-shot helpers below are the public opt-in (they warn once as an
# experimental API).  The prepare-once / launch-many surface
# (``prepare_grouped_gemm_*`` returning a ``GroupedGemmLaunch``) lives in
# ``cake_backend`` and is re-exported lazily, as are the ``torch.autograd``
# wrapper ``cake_grouped_mm`` / ``CakeGroupedMm`` of ``cake_autograd``; the
# JIT registry filled by the generated-program export lives in ``cake_jit``.
# All stay unimported until first use so importing this package is cheap.
# The stable API ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")``
# routes the forward projection to ``cake_backend.grouped_mm_bf16_cake``.
# ``offs`` may be given as ``[E]`` end offsets or as ``m_indptr`` ``[E + 1]``.

from __future__ import annotations

from typing import Any, Optional

import torch

from ...api_logging import flashinfer_experimental_api

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


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_fwd")
def grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
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
    from .cake_backend import prepare_grouped_gemm_fwd

    return prepare_grouped_gemm_fwd(x, w, offs, out=out).launch()


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_dgrad")
def grouped_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.  ``K % 256 == 0`` and
    ``N % 64 == 0`` are required.  See ``grouped_gemm_fwd`` for the prepared form.
    """
    from .cake_backend import prepare_grouped_gemm_dgrad

    return prepare_grouped_gemm_dgrad(g, w, offs, out=out).launch()


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_wgrad")
def grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, bitwise deterministic, exact zeros for empty
    groups.  ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256,
    or 512 when chosen) are required.  See ``grouped_gemm_fwd`` for the
    prepared form.
    """
    from .cake_backend import prepare_grouped_gemm_wgrad

    return prepare_grouped_gemm_wgrad(
        g, x, offs, out=out, out_dtype=out_dtype, num_groups=num_groups
    ).launch()


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        from . import cake_backend

        return getattr(cake_backend, name)
    if name in _AUTOGRAD_EXPORTS:
        from . import cake_autograd

        return getattr(cake_autograd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
