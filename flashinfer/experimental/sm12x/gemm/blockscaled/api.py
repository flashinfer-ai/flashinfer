# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for gemm.blockscaled (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.dense_gemm import (
    dense_gemm as mm,
)
from ..._lib.dense_gemm import (
    dense_gemm_fused_quant_a as mm_fused_quant_a,
)
from ..._lib.dense_gemm import (
    dense_gemm_fused_quant_a_grouped as mm_fused_quant_a_grouped,
)
from ..._lib.gating import default_is_supported
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = ["mm", "mm_fused_quant_a", "mm_fused_quant_a_grouped", "is_supported"]
