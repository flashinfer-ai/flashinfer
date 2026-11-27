# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 MXFP8 warp-MMA helpers for the swap-AB MegaMoE kernel.

This module owns the SM120 register-accumulator MXFP8 QMMA path. The target
instruction is PTX:

    mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
      .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0

The instruction issue path emits PTX directly with scalar E8M0 scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import Float32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


MMA_M = 16
MMA_N = 8
MMA_K = 32
MXFP8_BLOCK = 32
SWAP_AB_INTERLEAVE = 8
CTA_TOKEN_TILE = 64
INTERMEDIATE_ALIGNMENT = 2 * MXFP8_BLOCK


@dataclass(frozen=True)
class Mma16832LaneCoords:
    """Accumulator element coordinates owned by one lane for m16n8k32."""

    m0: int
    n0: int
    m1: int
    n1: int
    m2: int
    n2: int
    m3: int
    n3: int


@dataclass(frozen=True)
class Sm120Mxfp8ScaleSelectors:
    """PTX scale selector attributes for one SM120 MXFP8 MMA issue."""

    byte_id_a: int = 0
    byte_id_b: int = 0
    thread_id_a: int = 0
    thread_id_b: int = 0


def accumulator_coords(lane_id: int) -> Mma16832LaneCoords:
    """Return Figure-92 C/D coordinates for c0,c1,c2,c3.

    c0 = C[g,     2*t]
    c1 = C[g,     2*t + 1]
    c2 = C[g + 8, 2*t]
    c3 = C[g + 8, 2*t + 1]
    """

    g = lane_id >> 2
    t = lane_id & 3
    return Mma16832LaneCoords(
        g,
        2 * t,
        g,
        2 * t + 1,
        g + 8,
        2 * t,
        g + 8,
        2 * t + 1,
    )


def swap_ab_swiglu_coords(lane_id: int, m_group: int = 0) -> Tuple[int, int, int, int]:
    """Return (token0_n, token1_n, downproj0_m, downproj1_m) for Swap A/B.

    For one M16xN8 atom with interleave=8:

      c0/c2 -> token0, same downproj
      c1/c3 -> token1, same downproj

    ``m_group`` selects which M16 group inside M64 is being handled by the
    warp.  Swap AB 1 uses one group per warp; Swap AB 2 uses m_group=0..3
    inside the same warp.
    """

    g = lane_id >> 2
    t = lane_id & 3
    dp = m_group * SWAP_AB_INTERLEAVE + g
    return 2 * t, 2 * t + 1, dp, dp


def non_swap_swiglu_coords(lane_id: int, n8_group: int = 0) -> Tuple[int, int, int, int]:
    """Return (token_low_m, token_high_m, downproj_low, downproj_high).

    Non-swap uses C[M=token,N=gateup] and gate/up interleave=1:

      c0/c1 -> token_low,  downproj
      c2/c3 -> token_high, downproj

    ``n8_group`` selects which N8 group inside N64 is being handled.
    """

    g = lane_id >> 2
    t = lane_id & 3
    dp = n8_group * 4 + t
    return g, g + 8, dp, dp


MXFP8_MMA_PTX_E4M3 = (
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
    ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"
)


REQUIRED_SM120_DSL_SYMBOLS = (
    "cutlass.utils.blockscaled_layout.sm120_make_smem_layout_sfa",
)

def missing_sm120_dsl_symbols() -> Tuple[str, ...]:
    """Return missing low-level SM120 MXFP8 CUTE DSL helpers.

    This checks the shared-memory scale layout helper used by the SM120 path.
    """

    try:
        import cutlass.utils.blockscaled_layout as blockscaled_layout
    except Exception:
        return REQUIRED_SM120_DSL_SYMBOLS
    if not hasattr(blockscaled_layout, "sm120_make_smem_layout_sfa"):
        return REQUIRED_SM120_DSL_SYMBOLS

    return ()


def missing_high_level_dsl_symbols() -> Tuple[str, ...]:
    """Backward-compatible alias for older probes."""

    return missing_sm120_dsl_symbols()


def _sfa_atom_layout_tv() -> cute.Layout:
    return cute.make_layout(((2, 2, 8), 32), stride=((8, 0, 1), 16))


def _sfb_atom_layout_tv() -> cute.Layout:
    return cute.make_layout(((4, 8), 32), stride=((0, 1), 8))


def make_swapab_m64n8k128_tiled_mma(
    *,
    ab_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
) -> cute.TiledMma:
    """Create the warp-MMA tiler used by the SM120 Swap A/B mainloop.

    Four compute warps cover the four M16 atoms in a CTA ``M64 x N8`` MMA
    slice.  The kernel walks the CTA ``N64`` tile as eight explicit N8 groups.
    ``MmaMXF8Op`` is used here only to construct CuTe tiling/fragment layouts;
    the actual MMA issue is emitted by ``_arch_mma_m16n8k32_mxfp8`` below.
    """

    op = warp_mma.MmaMXF8Op(ab_dtype, acc_dtype, sf_dtype)
    permutation_mnk = sm120_utils.get_permutation_mnk(
        (64, 64, 128),
        MXFP8_BLOCK,
        True,
    )
    return cute.make_tiled_mma(
        op,
        cute.make_layout((4, 1, 1)),
        permutation_mnk=permutation_mnk,
    )


def make_sm120_ldmatrix_atom(
    operand_dtype: Type[cutlass.Numeric],
    *,
    transpose: bool,
) -> cute.CopyAtom:
    """Build the SM120 FP8 SMEM->RMEM ldmatrix copy atom."""

    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=4,
        ),
        operand_dtype,
    )


@dsl_user_op
def _arch_mma_m16n8k32_mxfp8(
    acc: cute.Tensor,
    a_reg: cute.Tensor,
    b_reg: cute.Tensor,
    sfa_scalar,
    sfb_scalar,
    *,
    ab_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    thread_id_a: int = 0,
    thread_id_b: int = 0,
    loc=None,
    ip=None,
) -> None:
    """Issue one QMMA ``m16n8k32`` MXFP8 block-scaled instruction.

    Per lane:
      A = 4 x b32, B = 2 x b32, C/D = 4 x f32.
    """

    a_i32 = a_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    b_i32 = b_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    sfa_u8 = sfa_scalar.bitcast(cutlass.Uint8, loc=loc, ip=ip)
    sfb_u8 = sfb_scalar.bitcast(cutlass.Uint8, loc=loc, ip=ip)

    if acc_dtype != cutlass.Float32 or sf_dtype != cutlass.Float8E8M0FNU:
        raise ValueError("m16n8k32 MXFP8 requires F32 accumulators and E8M0 scales")
    ptx_type = {cutlass.Float8E4M3FN: "e4m3", cutlass.Float8E5M2: "e5m2"}[ab_dtype]
    mma_ptx = (
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
        f".m16n8k32.row.col.f32.{ptx_type}.{ptx_type}.f32.ue8m0"
    )
    # Encode byte selector 0 in PTX: each scalar occupies the low byte.
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a_i32[0]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[1]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[2]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[3]).ir_value(loc=loc, ip=ip),
            Uint32(b_i32[0]).ir_value(loc=loc, ip=ip),
            Uint32(b_i32[1]).ir_value(loc=loc, ip=ip),
            Uint32(sfa_u8).ir_value(loc=loc, ip=ip),
            cutlass.Int16(thread_id_a).ir_value(loc=loc, ip=ip),
            Uint32(sfb_u8).ir_value(loc=loc, ip=ip),
            cutlass.Int16(thread_id_b).ir_value(loc=loc, ip=ip),
            Float32(acc[0]).ir_value(loc=loc, ip=ip),
            Float32(acc[1]).ir_value(loc=loc, ip=ip),
            Float32(acc[2]).ir_value(loc=loc, ip=ip),
            Float32(acc[3]).ir_value(loc=loc, ip=ip),
        ],
        mma_ptx + """
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$0, $1, $2, $3},
        {$10},
        {0, $11},
        {$12},
        {0, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,r,h,0,1,2,3",
        # All lanes must issue the MMA even when only some consume the result.
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    acc[0] = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    acc[1] = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    acc[2] = Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    acc[3] = Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))


@cute.jit
def partition_sfa_for_sm120_mma(
    tiled_mma: cute.TiledMma,
    sfa_stage: cute.Tensor,
    tidx,
) -> cute.Tensor:
    """Return per-thread SFA source view: ``(32, (RestM, RestK))``.

    This is the Python DSL equivalent of CUTLASS C++ ``thrfrg_SFA`` plus
    slicing by the current thread id for SM120 MXFP8 block-scaled MMA.
    """

    t_tensor = cute.logical_divide(
        sfa_stage, (tiled_mma.permutation_mnk[0], tiled_mma.permutation_mnk[2])
    )
    a_tensor = cute.zipped_divide(
        t_tensor,
        (
            cute.make_layout(tiled_mma.shape_mnk[0]),
            cute.make_layout(tiled_mma.shape_mnk[2]),
        ),
    )
    tv_tensor = cute.composition(a_tensor, (_sfa_atom_layout_tv(), None))
    thr_tensor = cute.zipped_divide(
        tv_tensor,
        (
            None,
            (
                cute.make_layout(tiled_mma.thr_layout_vmnk.shape[1]),
                cute.make_layout(tiled_mma.thr_layout_vmnk.shape[3]),
            ),
        ),
    )
    return cute.slice_(thr_tensor, (tidx, (None, None)))


@cute.jit
def partition_sfb_for_sm120_mma(
    tiled_mma: cute.TiledMma,
    sfb_stage: cute.Tensor,
    tidx,
) -> cute.Tensor:
    """Return per-thread SFB source view: ``(32, (RestN, RestK))``."""

    t_tensor = cute.logical_divide(
        sfb_stage, (tiled_mma.permutation_mnk[1], tiled_mma.permutation_mnk[2])
    )
    a_tensor = cute.zipped_divide(
        t_tensor,
        (
            cute.make_layout(tiled_mma.shape_mnk[1]),
            cute.make_layout(tiled_mma.shape_mnk[2]),
        ),
    )
    tv_tensor = cute.composition(a_tensor, (_sfb_atom_layout_tv(), None))
    thr_tensor = cute.zipped_divide(
        tv_tensor,
        (
            None,
            (
                cute.make_layout(tiled_mma.thr_layout_vmnk.shape[2]),
                cute.make_layout(tiled_mma.thr_layout_vmnk.shape[3]),
            ),
        ),
    )
    return cute.slice_(thr_tensor, (tidx, (None, None)))


@cute.jit
def issue_m64n8k32_mxfp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag,
    b_frag,
    sfa_frag,
    sfb_frag,
    *,
    n_group: int,
    active_n_groups: int,
    sfa_m_group,
    k_inner: int,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one SM120 MXFP8 ``M64 x N8 x K32`` warp-MMA slice."""

    if cutlass.const_expr(n_group >= active_n_groups):
        return

    a_reg = a_frag[(None, 0, k_inner)]
    b_reg = b_frag[(None, n_group, k_inner)]

    sfa_scalar = sfa_frag[((0, 0), sfa_m_group, k_inner)]
    sfb_scalar = sfb_frag[
        (
            (0, 0),
            (n_group % 2, (n_group // 2) % 2),
            k_inner,
            n_group // 4,
        )
    ]

    _arch_mma_m16n8k32_mxfp8(
        acc,
        a_reg,
        b_reg,
        sfa_scalar,
        sfb_scalar,
        ab_dtype=ab_dtype,
        acc_dtype=cutlass.Float32,
        sf_dtype=sf_dtype,
    )


@cute.jit
def issue_m64n8k128_mxfp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag,
    b_frag,
    sfa_frag,
    sfb_frag,
    *,
    n_group: int,
    active_n_groups: int,
    sfa_m_group,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one SM120 MXFP8 ``M64 x N8 x K128`` warp-MMA slice.

    ``a_frag`` / ``b_frag`` are register fragments already loaded from SMEM
    by the SM120 ldmatrix tiled copies.  ``sfa_frag`` / ``sfb_frag`` are
    register scale fragments copied with the SM120 scale TV layouts. Each K32
    slice issues the native block-scaled MMA instruction with scalar scales.
    """

    for k_inner in cutlass.range_constexpr(0, 4):
        issue_m64n8k32_mxfp8(
            tiled_mma,
            acc,
            a_frag,
            b_frag,
            sfa_frag,
            sfb_frag,
            n_group=n_group,
            active_n_groups=active_n_groups,
            sfa_m_group=sfa_m_group,
            k_inner=k_inner,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
        )
