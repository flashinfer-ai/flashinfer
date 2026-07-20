# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for quantization.mxfp8 (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ..._lib.quant.mxfp8_rows import (
    quantize_mxfp8_rows_cute as quantize_rows,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = ["quantize_rows", "is_supported"]
