"""Public surface for gemm.blockscaled (docs in the op ``__init__``)."""

from __future__ import annotations

import torch
from b12x.preparation import Plan
from b12x.preparation.types import require_prepared
from ._preparation import plan, plan_regimes, query_from_call
from ._tuning import BlockscaledQuery, BlockscaledConfig, FixedBlockscaledQuery

from ..._lib.gating import default_is_supported
from ._linear import (
    Weight,
    blockscaled_mm as mm,
    pack_weight,
)
from . import META
from ._a16 import NVFP4LinearWeight, w4a16, w8a16
from ._iq2_xs import IQ2XSLinearWeight, BlockQuantLinearWeight


def workspace_size(plan: Plan) -> int:
    return require_prepared(plan, "gemm.blockscaled_precision").required_workspace


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"block-scaled output must be bf16/fp16, got {dtype}")


def mm_mxfp4(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    *,
    plan: Plan | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> torch.Tensor:
    """Run serialized MXFP4 operands through ``blockscaled.mm``."""
    return mm(
        (lhs_values, lhs_scale_storage),
        (rhs_values, rhs_scale_storage),
        plan=plan,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=32,
        stream=stream,
    )


def mm_nvfp4(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor,
    *,
    plan: Plan | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> torch.Tensor:
    """Run serialized NVFP4 operands through ``blockscaled.mm``."""
    return mm(
        (lhs_values, lhs_scale_storage),
        (rhs_values, rhs_scale_storage),
        plan=plan,
        alpha=alpha.reshape(1),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=16,
        stream=stream,
    )


def mm_block_fp8(
    lhs_values: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale: torch.Tensor,
    *,
    plan: Plan | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> torch.Tensor:
    """Run compact 128x128 block-FP8 operands."""
    return mm(
        (lhs_values, lhs_scale),
        (rhs_values, rhs_scale),
        plan=plan,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype=_output_dtype_name(out_dtype),
        sf_vec_size=128,
        block_fp8=True,
        stream=stream,
    )


__all__ = [
    "BlockscaledQuery",
    "BlockscaledConfig",
    "FixedBlockscaledQuery",
    "plan",
    "plan_regimes",
    "query_from_call",
    "Weight",
    "NVFP4LinearWeight",
    "IQ2XSLinearWeight",
    "BlockQuantLinearWeight",
    "is_supported",
    "mm",
    "mm_block_fp8",
    "mm_mxfp4",
    "mm_nvfp4",
    "pack_weight",
    "w4a16",
    "w8a16",
    "workspace_size",
]
