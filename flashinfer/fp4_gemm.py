# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""Experimental native FP4 GEMM using prepacked E2M1 operands and UE8M0 scales."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp4_gemm(
    a,
    b,
    a_scales,
    b_scales,
    *,
    m=None,
    alpha=1.0,
    out=None,
    num_stages=None,
    block_n=128,
    epilogue_store_n=32,
    descriptor_workspace=None,
):
    """Prepare ``out[:m] = alpha * A @ B.T`` with FP32 accumulation and BF16 output.

    A is ``[rows, K/2]`` and B ``[N, K/2]`` packed E2M1 bytes (even K element
    in the low nibble). Scales are prepacked UE8M0 words: four bytes per
    int32/uint32 word, one byte per 32 K elements, packed-K major with the
    4-aligned MN extent contiguous (``[K/128, align(MN, 4)]``). ``m`` is the
    logical output M (default: the rows of A); any ``m`` in ``[1, rows]`` is
    accepted and K must be a multiple of 256. ``alpha`` multiplies the FP32
    accumulators before the BF16 conversion. The schedule is selected per
    problem shape and device; ``num_stages`` may name the selected schedule's
    stage count, ``block_n``/``epilogue_store_n`` accept their defaults, and
    ``descriptor_workspace`` is accepted for signature stability (tensor maps
    are passed by value). No data conversion occurs during ``run()``.
    """
    from .experimental.deepgemm_fp4_gemm.fp4_gemm import Fp4GemmPlan

    return Fp4GemmPlan(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        alpha=alpha,
        out=out,
        num_stages=num_stages,
        block_n=block_n,
        epilogue_store_n=epilogue_store_n,
        descriptor_workspace=descriptor_workspace,
    )
