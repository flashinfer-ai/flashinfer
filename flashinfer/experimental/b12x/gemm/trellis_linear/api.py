"""Public prepared API for :mod:`b12x.gemm.trellis_linear`."""
from __future__ import annotations

from typing import Optional

import torch

from ..._lib.gating import default_is_supported
from ...moe._shared.kernels.w4a16.kernel import clear_w4a16_kernel_cache
from ...moe._shared.kernels.w4a16.prepare import (
    PreparedTrellis256DenseWeight,
    prepare_trellis256_dense_weight,
    prepare_trellis256_pair_dense_weight,
)
from ...preparation import Plan
from ...preparation.types import require_prepared
from . import META
from ._preparation import plan, query_from_weight
from ._tuning import TrellisQuery

PreparedWeight = PreparedTrellis256DenseWeight


def prepare_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Validate one native EXL3 dense weight and retain zero-copy views."""
    return prepare_trellis256_dense_weight(
        trellis, suh, svh, mcg=mcg, mul1_e4m3=mul1_e4m3, codebook=codebook,
        params_dtype=params_dtype, dummy_scale=dummy_scale,
    )


def prepare_pair_weight(
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    pair_kind: str,
    rate_axis: str,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Prepare one compact TP12 P24/P33 pair for the SM12x decoder."""
    return prepare_trellis256_pair_dense_weight(
        payload, suh, svh, pair_kind=pair_kind, rate_axis=rate_axis, mcg=mcg,
        mul1_e4m3=mul1_e4m3, codebook=codebook, params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def run(
    x: torch.Tensor,
    *,
    plan: Plan,
    output: Optional[torch.Tensor] = None,
    gemm_output: Optional[torch.Tensor] = None,
    input_f16: Optional[torch.Tensor] = None,
    rotated_f16: Optional[torch.Tensor] = None,
    rotated_compute: Optional[torch.Tensor] = None,
    gemm_output_f16: Optional[torch.Tensor] = None,
    output_f16: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a session-prepared exact-M Trellis declaration."""
    state = require_prepared(plan, "gemm.trellis_linear", x.device)
    return state.run(
        x, output=output, gemm_output=gemm_output, input_f16=input_f16,
        rotated_f16=rotated_f16, rotated_compute=rotated_compute,
        gemm_output_f16=gemm_output_f16, output_f16=output_f16,
    )


def is_supported(device=None) -> bool:
    """True when the SM120/SM121 Trellis kernel stack is available."""
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    """Clear unretained compiled W4A16 specializations."""
    clear_w4a16_kernel_cache()


__all__ = list(META.entry_points)
