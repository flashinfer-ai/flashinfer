"""Prepared public surface for serialized block-FP8 linear."""
from __future__ import annotations

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from .._shared.block_fp8 import BlockFP8LinearBinding as Binding
from .._shared.block_fp8 import BlockFP8LinearScratchCaps as Caps
from .._shared.block_fp8 import BlockFP8LinearWeight as Weight
from .._shared.block_fp8 import block_fp8_linear_mxfp8 as run
from .._shared.block_fp8 import pack_block_fp8_linear_weight_mxfp8 as pack_weight
from .._shared.block_fp8 import quantize_block_fp8_linear_input_mxfp8 as quantize_input
from ._preparation import plan
from .._tuning import DenseGemmConfig
from ._tuning import BlockFp8LinearQuery
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Map caller-owned scratch after the declaration is session-prepared."""
    state = require_prepared(plan, "gemm.block_fp8_linear")
    return state.bind(plan=plan, **kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps", "Plan", "Binding", "Weight", "DenseGemmConfig",
    "BlockFp8LinearQuery", "plan", "bind", "run", "pack_weight",
    "quantize_input", "is_supported",
]
