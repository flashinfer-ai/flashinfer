"""Public surface for gemm.bf16_gemv (docs in the op ``__init__``)."""

from __future__ import annotations

import os

import torch

from ..._lib.gating import default_is_supported
from ...preparation import Plan
from . import META
from ._kernel import SMALL_M_MAX
from ._preparation import bf16_gemv_small_n, bf16_gemv_small_n_out  # noqa: F401
from ._preparation import plan, query_from_call
from ._tuning import GemvConfig, GemvQuery

# Routing thresholds for integrations (canonical home; formerly the vLLM
# plugin's constants). Unquantized bf16 linears with N <= MAX_OUT and
# K >= MIN_IN are worth routing through this small-N GEMV: catches narrow
# projections like the GDN ``in_proj_ba`` while excluding lm_head (N = vocab)
# and anything wide enough that cuBLAS tiles efficiently.
SMALL_N_GEMV_MAX_OUT = 1024
SMALL_N_GEMV_MIN_IN = 1024


def is_disabled() -> bool:
    """True when ``B12X_DISABLE_BF16_GEMV`` turns the routing off
    entirely (debug isolation switch)."""
    return os.environ.get("B12X_DISABLE_BF16_GEMV", "").lower() in (
        "1",
        "true",
        "yes",
    )


def mm(
    x: torch.Tensor, weight: torch.Tensor, *, plan: Plan,
    bias: torch.Tensor | None = None, out: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the selected native projection; caller-owned output supports replay."""
    if output_dtype is None:
        output_dtype = getattr(torch, plan.query.output_dtype)
    if out is None:
        return torch.ops.b12x.bf16_gemv_small_n(x, weight, plan.handle, bias, output_dtype)
    if output_dtype is not None and out.dtype != output_dtype:
        raise ValueError("output_dtype disagrees with caller-owned output")
    torch.ops.b12x.bf16_gemv_small_n_out(x, weight, out, plan.handle, bias)
    return out


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0, unless disabled
    via ``B12X_DISABLE_BF16_GEMV``."""
    if is_disabled():
        return False
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Plan",
    "GemvQuery",
    "GemvConfig",
    "plan",
    "query_from_call",
    "mm",
    "is_supported",
    "is_disabled",
    "SMALL_M_MAX",
    "SMALL_N_GEMV_MAX_OUT",
    "SMALL_N_GEMV_MIN_IN",
]
