# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Public GEMM argument checks for the cuDNN batch-invariant entries."""

from __future__ import annotations

from typing import Optional

import torch

from ...batch_invariant.support import unsupported
from .api import (
    CLASS,
    CLASS_FP8,
    CLASS_MXFP8,
    CLASS_FP4,
    CLASS_FP8BS,
    GEMM_FP8BS_API_OUT_DTYPES,
    MM_MXFP8_API_OUT_DTYPES,
    MM_FP4_API_OUT_DTYPES,
    bmm_fp8_batch_invariant,
    gemm_fp8_nt_groupwise_batch_invariant,
    mm_bf16_batch_invariant,
    mm_fp4_batch_invariant,
    mm_mxfp8_batch_invariant,
)


def _require_auto(cls, backend):
    if backend != "auto":
        raise unsupported(
            cls,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" for cuDNN batch-invariant plan selection',
        )


def mm_bf16_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    pdl: Optional[bool],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    _require_auto(CLASS, backend)
    if out_dtype != torch.bfloat16:
        raise unsupported(CLASS, f"out_dtype={out_dtype}", "bf16 output")
    return mm_bf16_batch_invariant(a, b, out, bias=bias)


def bmm_fp8_batch_invariant_auto(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor],
    backend: str,
) -> torch.Tensor:
    _require_auto(CLASS_FP8, backend)
    return bmm_fp8_batch_invariant(A, B, A_scale, B_scale, dtype, out)


def mm_mxfp8_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    use_8x4_sf_layout: bool,
    backend: str,
) -> torch.Tensor:
    _require_auto(CLASS_MXFP8, backend)
    if use_8x4_sf_layout:
        raise unsupported(
            CLASS_MXFP8,
            "the 8x4 scale-factor layout (use_8x4_sf_layout=True)",
            "the 128x4 swizzled layout (mxfp8_quantize(..., is_sf_swizzled_layout=True))",
        )
    if out_dtype not in MM_MXFP8_API_OUT_DTYPES:
        raise unsupported(CLASS_MXFP8, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    return mm_mxfp8_batch_invariant(
        a, b, a_descale, b_descale, out, out_dtype=out_dtype
    )


def mm_fp4_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    use_8x4_sf_layout: bool,
    backend: str,
    use_nvfp4: bool,
) -> torch.Tensor:
    _require_auto(CLASS_FP4, backend)
    if use_8x4_sf_layout:
        raise unsupported(
            CLASS_FP4,
            "the 8x4 scale-factor layout (use_8x4_sf_layout=True)",
            "the 128x4 swizzled layout (nvfp4_quantize(..., sfLayout=SfLayout.layout_128x4, do_shuffle=False) / mxfp4_quantize)",
        )
    if out_dtype not in MM_FP4_API_OUT_DTYPES:
        raise unsupported(CLASS_FP4, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    return mm_fp4_batch_invariant(
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size=block_size,
        use_nvfp4=use_nvfp4,
    )


def gemm_fp8_nt_groupwise_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[str],
    mma_sm: int,
    scale_granularity_mnk: tuple,
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    backend: str,
) -> torch.Tensor:
    _require_auto(CLASS_FP8BS, backend)
    if tuple(scale_granularity_mnk) != (1, 128, 128):
        raise unsupported(
            CLASS_FP8BS,
            f"scale_granularity_mnk={tuple(scale_granularity_mnk)}",
            "scale_granularity_mnk = (1, 128, 128) (the class's block sizes)",
        )
    if out is not None:
        out_dtype = out.dtype
    out_dtype = out_dtype or torch.bfloat16
    if out_dtype not in GEMM_FP8BS_API_OUT_DTYPES:
        raise unsupported(CLASS_FP8BS, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    return gemm_fp8_nt_groupwise_batch_invariant(
        a, b, a_scale, b_scale, scale_major_mode, out, out_dtype=out_dtype
    )
