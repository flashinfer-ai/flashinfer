# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for gemm.mxfp8_linear (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._kernel import (
    MXFP8LinearWeight as Weight,
)
from ._kernel import (
    is_mxfp8_linear_supported as _kernel_is_supported,
)
from ._kernel import (
    mxfp8_linear as mm,
)
from ._kernel import (
    pack_mxfp8_linear_weight as pack_weight,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0, triton, and
    the kernel's own capability checks."""
    return default_is_supported(device, requires=META.requires) and bool(
        _kernel_is_supported()
    )


__all__ = ["Weight", "mm", "pack_weight", "is_supported"]
