# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for gemm.wo_projection (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.wo_mxfp8 import (
    MXFP8Rows,
)
from .._shared.wo_mxfp8 import (
    WOProjectionBinding as Binding,
)
from .._shared.wo_mxfp8 import (
    WOProjectionInvRopeBinding as InvRopeBinding,
)
from .._shared.wo_mxfp8 import (
    WOProjectionMXFP8Weights as Weights,
)
from .._shared.wo_mxfp8 import (
    WOProjectionScratchCaps as Caps,
)
from .._shared.wo_mxfp8 import (
    WOProjectionScratchPlan as Plan,
)
from .._shared.wo_mxfp8 import (
    pack_wo_projection_fp8_block_scaled_weights_mxfp8 as pack_weights,
)
from .._shared.wo_mxfp8 import (
    plan_wo_projection_scratch as plan,
)
from .._shared.wo_mxfp8 import (
    quantize_wo_a_input_inv_rope_mxfp8 as quantize_input_inv_rope,
)
from .._shared.wo_mxfp8 import (
    quantize_wo_a_input_mxfp8 as quantize_input,
)
from .._shared.wo_mxfp8 import (
    quantize_wo_b_input_mxfp8 as quantize_input_b,
)
from .._shared.wo_mxfp8 import (
    wo_projection_inv_rope_mxfp8 as run_inv_rope,
)
from .._shared.wo_mxfp8 import (
    wo_projection_mxfp8 as run,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def bind_inv_rope(plan: Plan, **kwargs) -> InvRopeBinding:
    """Bind the inverse-RoPE variant (views only; capture safe)."""
    return plan.bind_inv_rope(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "InvRopeBinding",
    "Weights",
    "MXFP8Rows",
    "plan",
    "bind",
    "bind_inv_rope",
    "run",
    "run_inv_rope",
    "pack_weights",
    "quantize_input",
    "quantize_input_inv_rope",
    "quantize_input_b",
    "is_supported",
]
