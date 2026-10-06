"""Inline MX-FP6 warp MMA helpers for dense block-scaled GEMM."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from b12x._lib.fp6 import (
    mxfp6_mma_m16n8k32_f32_e2m3_e2m3,
    mxfp6_mma_m16n8k32_f32_e2m3_e3m2,
    mxfp6_mma_m16n8k32_f32_e2m3_e4m3,
    mxfp6_mma_m16n8k32_f32_e3m2_e2m3,
    mxfp6_mma_m16n8k32_f32_e3m2_e3m2,
    mxfp6_mma_m16n8k32_f32_e3m2_e4m3,
    mxfp6_mma_m16n8k32_f32_e4m3_e2m3,
    mxfp6_mma_m16n8k32_f32_e4m3_e3m2,
)


@cute.jit
def _mxfp6_scale_u32_from_sf_frag(sf_frag: cute.Tensor) -> Uint32:
    """Pack the first UE8M0 scale lane from an SF fragment into a single u32 operand.

    The fragment element is ``Float8E8M0FNU``; the MMA wants the raw exponent byte,
    so reinterpret the bits as Uint8 (a zero-extending integer cast) rather than a
    numeric float->uint conversion.
    """
    sf_u8 = cute.recast_tensor(sf_frag, cutlass.Uint8)
    return Uint32(sf_u8[0])


@cute.jit
def emit_mxfp6_dense_mma_k_block(
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrSFA: cute.Tensor,
    tCrSFB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
    fmt_a: cutlass.Constexpr,
    fmt_b: cutlass.Constexpr,
) -> None:
    """Emit one MX-FP6 ``m16n8k32`` block-scaled MMA for a single (M,N) output tile.

    ``fmt_a`` / ``fmt_b`` are the FP6 sub-formats (``"e3m2"`` or ``"e2m3"``) of the
    A / B operands. The operand *fragments* are byte-containers (one FP6 code per
    byte, produced by cutlass's native 8-bit LdMatrix on the shared ``mxf8f6f4``
    atom); only the emitted MMA instruction's type qualifier distinguishes FP6 from
    FP8, so the format must be passed explicitly rather than inferred from a dtype.
    """
    acc = accumulators[None, mt, nt]
    # Operand fragments are byte-containers (Float8E4M3FN); reinterpret the raw
    # register bits as Uint32 (4 bytes -> 1 register) for the inline MMA rather
    # than numerically converting each element (which would emit fptoui).
    a_frag = cute.flatten(cute.recast_tensor(tCrA[None, mt, k_block_idx], cutlass.Uint32))
    b_frag = cute.flatten(cute.recast_tensor(tCrB[None, nt, k_block_idx], cutlass.Uint32))
    sfa_frag = tCrSFA[None, mt, k_block_idx]
    sfb_frag = tCrSFB[None, nt, k_block_idx]
    sfa = _mxfp6_scale_u32_from_sf_frag(sfa_frag)
    sfb = _mxfp6_scale_u32_from_sf_frag(sfb_frag)

    if cutlass.const_expr(fmt_a == "e3m2" and fmt_b == "e3m2"):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(fmt_a == "e2m3" and fmt_b == "e2m3"):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(fmt_a == "e2m3" and fmt_b == "e3m2"):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e3m2(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(fmt_a == "e4m3" and fmt_b == "e2m3"):
        # W6A8: A = FP8 activations, B = FP6 weights (dense and MoE).
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e4m3_e2m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(fmt_a == "e4m3" and fmt_b == "e3m2"):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e4m3_e3m2(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(fmt_a == "e2m3" and fmt_b == "e4m3"):
        # Cross-format (A=FP6, B=FP8) — not used by the dense W6A8 path.
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e4m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(fmt_a == "e3m2" and fmt_b == "e4m3"):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e4m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    else:
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e2m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    acc[0] = d0
    acc[1] = d1
    acc[2] = d2
    acc[3] = d3
