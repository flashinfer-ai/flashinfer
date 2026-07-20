# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for gemm.block_fp8_linear (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.block_fp8 import (
    BlockFP8LinearBinding as Binding,
)
from .._shared.block_fp8 import (
    BlockFP8LinearScratchCaps as Caps,
)
from .._shared.block_fp8 import (
    BlockFP8LinearScratchPlan as Plan,
)
from .._shared.block_fp8 import (
    BlockFP8LinearWeight as Weight,
)
from .._shared.block_fp8 import (
    block_fp8_linear_mxfp8 as run,
)
from .._shared.block_fp8 import (
    pack_block_fp8_linear_weight_mxfp8 as pack_weight,
)
from .._shared.block_fp8 import (
    plan_block_fp8_linear_scratch as plan,
)
from .._shared.block_fp8 import (
    prewarm_block_fp8_linear_mxfp8 as prewarm,
)
from .._shared.block_fp8 import (
    quantize_block_fp8_linear_input_mxfp8 as quantize_input,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "Weight",
    "plan",
    "bind",
    "run",
    "pack_weight",
    "quantize_input",
    "prewarm",
    "is_supported",
]
