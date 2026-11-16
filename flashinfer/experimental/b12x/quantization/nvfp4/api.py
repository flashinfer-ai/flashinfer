"""Public surface for quantization.nvfp4 (docs in the op ``__init__``)."""

from __future__ import annotations


import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._impl import (
    BF16ToFP4TMAOutputs as Outputs,
)
from ._impl import (
    allocate_bf16_to_fp4_tma_outputs as _allocate,
)
from ._preparation import plan
from ._tuning import TUNING, Nvfp4QuantizationConfig, Nvfp4QuantizationQuery


def allocate_outputs(plan: Plan, *, device: torch.device | str = "cuda") -> Outputs:
    """Allocate the packed-FP4 + MMA-layout-scale pair before preparation."""
    if not isinstance(plan, Plan) or plan.contract is not TUNING:
        raise TypeError("outputs require an NVFP4 quantization declaration")
    return _allocate(plan.query.rows, plan.query.columns, device=torch.device(device))


def run(
    *,
    plan: Plan,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    outputs: Outputs,
) -> None:
    """Quantize into caller-owned outputs using the admitted launch callable."""
    state = require_prepared(plan, "quantization.nvfp4", x.device)
    state.run(x, global_scale, outputs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Outputs",
    "Plan",
    "Nvfp4QuantizationConfig",
    "Nvfp4QuantizationQuery",
    "plan",
    "allocate_outputs",
    "run",
    "is_supported",
]
