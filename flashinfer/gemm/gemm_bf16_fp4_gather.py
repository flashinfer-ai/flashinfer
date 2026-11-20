# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Gathered-row BF16 x NVFP4 GEMV with a scatter epilogue (SM12x, CuTe DSL)."""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.gemm import gemv_bf16_fp4_gather_trace
from ..utils import backend_requirement, supported_compute_capability

_KERNEL_CACHE: dict = {}
_ALPHA_ONE: dict = {}

# Within 2% of the best measured tile at K in {2048, 2560}, J = 16384,
# M in {1, 4, 5, 8, 32} on GB10.
_ROWS_PER_HALF_WARP = 2
_UNROLL = 2
_NUM_WARPS = 4

_OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_ID_DTYPES = (torch.int32, torch.int64)


def _get_kernel(
    hidden: int,
    ids_dtype: torch.dtype,
    out_dtype: torch.dtype,
    rows_per_hw: int,
    unroll: int,
    num_warps: int,
    enable_pdl: bool,
):
    key = (hidden, ids_dtype, out_dtype, rows_per_hw, unroll, num_warps, enable_pdl)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32

    from .kernels.cute_dsl.gather_gemv_bf16_fp4_sm12x import GatherGemvBf16Fp4Sm12x

    dtypes = {
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }
    m, r, j, n = (cute.sym_int() for _ in range(4))
    x = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (m, hidden),
        (cute.sym_int64(divisibility=8), 1),
        assumed_align=16,
    )
    b = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (r, hidden // 2), stride_order=(1, 0), assumed_align=16
    )
    sf = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (r, hidden // 16), stride_order=(1, 0), assumed_align=4
    )
    alpha = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    ids = cute.runtime.make_fake_compact_tensor(
        dtypes[ids_dtype], (m, j), stride_order=(1, 0)
    )
    out = cute.runtime.make_fake_tensor(
        dtypes[out_dtype], (m, n), (cute.sym_int64(), 1)
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        GatherGemvBf16Fp4Sm12x(
            hidden,
            rows_per_hw=rows_per_hw,
            unroll=unroll,
            num_warps=num_warps,
            enable_pdl=enable_pdl,
        ),
        x,
        b,
        sf,
        alpha,
        ids,
        out,
        Int32(0),
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


@supported_compute_capability([120, 121])
def _check_gemv_bf16_fp4_gather(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    row_ids: torch.Tensor,
    out: torch.Tensor,
    *,
    out_col_offset: int = 0,
    enable_pdl: bool = True,
) -> bool:
    if a.dtype != torch.bfloat16 or a.dim() != 2 or a.stride(1) != 1:
        raise ValueError("a must be a (M, K) bfloat16 tensor with unit inner stride.")
    m, k = a.shape
    if k % 32 != 0:
        raise ValueError(f"K must be a multiple of 32, got {k}.")
    if b.dtype != torch.uint8 or b.dim() != 2 or b.shape[1] != k // 2:
        raise ValueError("b must be a (R, K/2) uint8 tensor of packed E2M1.")
    if b_descale.dim() != 2 or b_descale.shape != (b.shape[0], k // 16):
        raise ValueError("b_descale must be (R, K/16) E4M3 block scales.")
    if b_descale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise ValueError("b_descale must be float8_e4m3fn (or its uint8 bytes).")
    if not (b.is_contiguous() and b_descale.is_contiguous()):
        raise ValueError("b and b_descale must be contiguous.")
    if a.data_ptr() % 16 or b.data_ptr() % 16 or b_descale.data_ptr() % 2:
        raise ValueError("a and b must be 16B aligned, b_descale 2B aligned.")
    if alpha is not None and (alpha.dtype != torch.float32 or alpha.numel() != 1):
        raise ValueError("alpha must be a (1,) float32 tensor.")
    if row_ids.dtype not in _ID_DTYPES or row_ids.dim() != 2:
        raise ValueError("row_ids must be a (M, J) int32 or int64 tensor.")
    if row_ids.shape[0] != m or not row_ids.is_contiguous():
        raise ValueError("row_ids must be contiguous with one row per token of a.")
    if out.dtype not in _OUT_DTYPES or out.dim() != 2 or out.stride(1) != 1:
        raise ValueError(
            "out must be a (M, N) bf16/fp16/fp32 tensor, unit inner stride."
        )
    if out.shape[0] != m or out_col_offset < 0:
        raise ValueError("out must have one row per token; out_col_offset >= 0.")
    if out_col_offset + b.shape[0] > out.shape[1]:
        raise ValueError("out_col_offset + R must not exceed out.shape[1].")
    return True


@backend_requirement({}, common_check=_check_gemv_bf16_fp4_gather)
@flashinfer_api(trace=gemv_bf16_fp4_gather_trace)
def gemv_bf16_fp4_gather(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    row_ids: torch.Tensor,
    out: torch.Tensor,
    *,
    out_col_offset: int = 0,
    enable_pdl: bool = True,
) -> torch.Tensor:
    r"""Per-token GEMV over gathered NVFP4 rows, scattered into ``out``.

    For every token ``m`` and slot ``j`` with ``row_ids[m, j] >= 0``::

        out[m, out_col_offset + row_ids[m, j]] = alpha * (a[m] @ dequant(b[row_ids[m, j]]))

    Columns of ``out`` that no slot names are left untouched, so the caller
    pre-fills them (e.g. ``-inf`` logits, or a dense GEMM over other rows).
    This is the dynamic-row half of a sparse LM head for speculative drafters.

    Args:
        a: ``(M, K)`` bfloat16 activations.
        b: ``(R, K/2)`` uint8, row-major packed E2M1 (low nibble = even column).
        b_descale: ``(R, K/16)`` E4M3 block scales (block size 16), row-major,
            not swizzled.
        alpha: ``(1,)`` float32 global dequantization scale, or ``None`` for 1.
        row_ids: ``(M, J)`` int32/int64 row indices into ``b``, in ``[0, R)``;
            negative entries are skipped.  Duplicates within a row write the
            same value.
        out: ``(M, N)`` output, unit inner stride, ``N >= out_col_offset + R``.
        out_col_offset: Column of ``out`` that row 0 of ``b`` maps to.
        enable_pdl: Launch with programmatic dependent launch.

    Returns:
        ``out``.
    """
    if alpha is None:
        alpha = _ALPHA_ONE.get(a.device)
        if alpha is None:
            alpha = torch.ones(1, dtype=torch.float32, device=a.device)
            _ALPHA_ONE[a.device] = alpha
    if a.shape[0] == 0 or row_ids.shape[1] == 0:
        return out
    kernel = _get_kernel(
        a.shape[1],
        row_ids.dtype,
        out.dtype,
        _ROWS_PER_HALF_WARP,
        _UNROLL,
        _NUM_WARPS,
        bool(enable_pdl),
    )
    kernel(
        a,
        b,
        b_descale.view(torch.uint8),
        alpha,
        row_ids,
        out,
        out_col_offset,
    )
    return out
