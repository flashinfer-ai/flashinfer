# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 native-NVFP4 warp-MMA helpers for swap-AB MegaMoE.

The data operands are both packed E2M1 and every 16 consecutive K elements
share one E4M3 scale.  The target instruction is the public SM120 NVFP4 op::

    mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
      .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3

Unlike the mixed MXFP4 x MXFP8 instruction, native NVFP4 consumes the packed
FP4 fragment directly: no ``ldsm.b4x16_p64`` unpack or fixed ``<< 2`` shift is
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu.warp.mma import _pack_shape


MMA_M = 16
MMA_N = 8
MMA_K = 64
NVFP4_BLOCK = 16
SWAP_AB_INTERLEAVE = 8
CTA_TOKEN_TILE = 64
INTERMEDIATE_ALIGNMENT = 2 * NVFP4_BLOCK


@dataclass(frozen=True)
class Mma16864LaneCoords:
    """Accumulator element coordinates owned by one lane for m16n8k64."""

    m0: int
    n0: int
    m1: int
    n1: int
    m2: int
    n2: int
    m3: int
    n3: int


@dataclass(frozen=True)
class Sm120Nvfp4ScaleSelectors:
    """PTX scale selector attributes for one SM120 NVFP4 MMA issue."""

    byte_id_a: int = 0
    byte_id_b: int = 0
    thread_id_a: int = 0
    thread_id_b: int = 0


def accumulator_coords(lane_id: int) -> Mma16864LaneCoords:
    """Return Figure-92 C/D coordinates for c0,c1,c2,c3.

    c0 = C[g,     2*t]
    c1 = C[g,     2*t + 1]
    c2 = C[g + 8, 2*t]
    c3 = C[g + 8, 2*t + 1]
    """

    g = lane_id >> 2
    t = lane_id & 3
    return Mma16864LaneCoords(
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


NVFP4_MMA_PTX_E2M1 = (
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
    ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
)


REQUIRED_SM120_DSL_SYMBOLS = (
    "cutlass._mlir.dialects.cute_nvgpu.arch_mma_SM120_block_scaled",
    "cutlass.utils.blockscaled_layout.sm120_make_smem_layout_sfa",
    "cutlass.cute.nvgpu.warp.mma.MmaMXF4NVF4Op",
)

def missing_sm120_dsl_symbols() -> Tuple[str, ...]:
    """Return missing public SM120 NVFP4 CuTe DSL helpers.

    This is a quick sanity check for the API surface needed by the SM120 path.
    It does not prove that the installed compiler backend can lower the op.
    """

    missing = []
    try:
        from cutlass._mlir.dialects import cute_nvgpu as cute_nvgpu_ir
    except Exception:
        cute_nvgpu_ir = None
    if cute_nvgpu_ir is None or not hasattr(
        cute_nvgpu_ir, "arch_mma_SM120_block_scaled"
    ):
        missing.append(REQUIRED_SM120_DSL_SYMBOLS[0])

    try:
        import cutlass.utils.blockscaled_layout as blockscaled_layout
    except Exception:
        blockscaled_layout = None
    if (
        blockscaled_layout is None
        or not hasattr(blockscaled_layout, "sm120_make_smem_layout_sfa")
    ):
        missing.append(REQUIRED_SM120_DSL_SYMBOLS[1])

    if not hasattr(warp_mma, "MmaMXF4NVF4Op"):
        missing.append(REQUIRED_SM120_DSL_SYMBOLS[2])

    return tuple(missing)


def _sfa_atom_layout_tv() -> cute.Layout:
    return cute.make_layout(((2, 2, 8), 64), stride=((8, 0, 1), 16))


def _sfb_atom_layout_tv() -> cute.Layout:
    return cute.make_layout(((4, 8), 64), stride=((0, 1), 8))


def make_swapab_m64n8k128_tiled_mma(
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
) -> cute.TiledMma:
    """Create the warp-MMA tiler used by the SM120 Swap A/B mainloop.

    Four compute warps cover the four M16 atoms in one ``M64 x N8`` MMA
    slice.

    The public op constructs the native FP4 fragment and scale layouts.  The
    actual issue is emitted by :func:`_arch_mma_m16n8k64_nvfp4` below so the
    existing hand-written M64 scheduling stays unchanged.
    """

    if a_dtype is not cutlass.Float4E2M1FN or b_dtype is not cutlass.Float4E2M1FN:
        raise ValueError("native NVFP4 requires E2M1 x E2M1 operands")
    if sf_dtype is not cutlass.Float8E4M3FN:
        raise ValueError("native NVFP4 requires E4M3 block scales")
    op = warp_mma.MmaMXF4NVF4Op(a_dtype, acc_dtype, sf_dtype)
    permutation_mnk = sm120_utils.get_permutation_mnk(
        (64, 64, 128),
        NVFP4_BLOCK,
        False,
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
    mixed_mode: bool = False,
) -> cute.CopyAtom:
    """Build the native SM120 FP4 SMEM-to-register copy atom."""

    if mixed_mode:
        raise ValueError("native NVFP4 must not use the mixed FP4 unpack path")
    if operand_dtype is not cutlass.Float4E2M1FN:
        raise ValueError("native NVFP4 ldmatrix expects packed E2M1")

    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=4,
        ),
        operand_dtype,
    )


@dsl_user_op
def pack_e2m1x2(
    even,
    odd,
    *,
    loc=None,
    ip=None,
):
    """RTNE/satfinite-convert two FP32 values to one packed E2M1 byte."""
    packed = llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [odd.ir_value(loc=loc, ip=ip), even.ir_value(loc=loc, ip=ip)],
        "{\n"
        "  .reg .b8 r;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;\n"
        "  mov.b32 $0, {r, r, r, r};\n"
        "}",
        "=r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint8(cutlass.Int32(packed) & cutlass.Int32(0xFF))


@dsl_user_op
def _arch_mma_m16n8k64_nvfp4(
    acc: cute.Tensor,
    a_reg: cute.Tensor,
    b_reg: cute.Tensor,
    sfa_reg: cute.Tensor,
    sfb_reg: cute.Tensor,
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    byte_id_a: int = 0,
    byte_id_b: int = 0,
    thread_id_a: int = 0,
    thread_id_b: int = 0,
    loc=None,
    ip=None,
) -> None:
    """Issue one native-NVFP4 QMMA ``m16n8k64`` instruction.

    Per lane:
      A = 4 x b32, B = 2 x b32, C/D = 4 x f32.
    """

    a_i32 = a_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    b_i32 = b_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    sfa_i32 = sfa_reg.load(loc=loc, ip=ip).bitcast(
        cutlass.Int32, loc=loc, ip=ip
    )
    sfb_i32 = sfb_reg.load(loc=loc, ip=ip).bitcast(
        cutlass.Int32, loc=loc, ip=ip
    )

    shape_attr = _pack_shape((MMA_M, MMA_N, MMA_K), loc=loc, ip=ip).type.attribute
    res = _cute_nvgpu_ir.arch_mma_SM120_block_scaled(
        [acc_dtype.mlir_type] * 4,
        shape_attr,
        NVFP4_BLOCK,
        ir.TypeAttr.get(a_dtype.mlir_type),
        ir.TypeAttr.get(b_dtype.mlir_type),
        ir.TypeAttr.get(sf_dtype.mlir_type),
        [
            a_i32[0].ir_value(loc=loc, ip=ip),
            a_i32[1].ir_value(loc=loc, ip=ip),
            a_i32[2].ir_value(loc=loc, ip=ip),
            a_i32[3].ir_value(loc=loc, ip=ip),
        ],
        [
            b_i32[0].ir_value(loc=loc, ip=ip),
            b_i32[1].ir_value(loc=loc, ip=ip),
        ],
        [
            acc[0].ir_value(loc=loc, ip=ip),
            acc[1].ir_value(loc=loc, ip=ip),
            acc[2].ir_value(loc=loc, ip=ip),
            acc[3].ir_value(loc=loc, ip=ip),
        ],
        sfa_i32[0].ir_value(loc=loc, ip=ip),
        sfb_i32[0].ir_value(loc=loc, ip=ip),
        thread_id_a=thread_id_a,
        thread_id_b=thread_id_b,
        byte_id_a=cutlass.Int16(byte_id_a).ir_value(loc=loc, ip=ip),
        byte_id_b=cutlass.Int16(byte_id_b).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )

    acc[0] = acc_dtype(res[0])
    acc[1] = acc_dtype(res[1])
    acc[2] = acc_dtype(res[2])
    acc[3] = acc_dtype(res[3])


def _thrfrg_sfa(scale_layout: cute.Layout, tiled_mma: cute.TiledMma):
    """Build the public SM120 NVFP4 SFA thread-fragment layout."""
    t_tensor = cute.logical_divide(
        scale_layout,
        (tiled_mma.permutation_mnk[0], tiled_mma.permutation_mnk[2]),
    )
    a_tensor = cute.zipped_divide(
        t_tensor,
        (
            cute.make_layout(tiled_mma.shape_mnk[0]),
            cute.make_layout(tiled_mma.shape_mnk[2]),
        ),
    )
    tv_tensor = cute.composition(a_tensor, (_sfa_atom_layout_tv(), None))
    thr_tile = (
        None,
        (
            cute.make_layout(cute.size(tiled_mma.thr_layout_vmnk[1])),
            cute.make_layout(cute.size(tiled_mma.thr_layout_vmnk[3])),
        ),
    )
    return cute.zipped_divide(tv_tensor, thr_tile)


def _thrfrg_sfb(scale_layout: cute.Layout, tiled_mma: cute.TiledMma):
    """Build the public SM120 NVFP4 SFB thread-fragment layout."""
    t_tensor = cute.logical_divide(
        scale_layout,
        (tiled_mma.permutation_mnk[1], tiled_mma.permutation_mnk[2]),
    )
    a_tensor = cute.zipped_divide(
        t_tensor,
        (
            cute.make_layout(tiled_mma.shape_mnk[1]),
            cute.make_layout(tiled_mma.shape_mnk[2]),
        ),
    )
    tv_tensor = cute.composition(a_tensor, (_sfb_atom_layout_tv(), None))
    thr_tile = (
        None,
        (
            cute.make_layout(cute.size(tiled_mma.thr_layout_vmnk[2])),
            cute.make_layout(cute.size(tiled_mma.thr_layout_vmnk[3])),
        ),
    )
    return cute.zipped_divide(tv_tensor, thr_tile)


@cute.jit
def partition_fragment_sfa_for_sm120_mma(
    tiled_mma: cute.TiledMma,
    sfa_stage: cute.Tensor,
    tidx,
) -> cute.Tensor:
    """Allocate the per-thread SFA register fragment for native NVFP4."""
    thr_tensor = cute.make_tensor(
        sfa_stage.iterator, _thrfrg_sfa(sfa_stage.layout, tiled_mma)
    )
    thr_vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
    thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
    fragment = thr_tensor[thr_vmk, (None, None)]
    fragment = cute.group_modes(cute.flatten(fragment), 0, 2)
    return cute.make_fragment_like(fragment)


@cute.jit
def partition_fragment_sfb_for_sm120_mma(
    tiled_mma: cute.TiledMma,
    sfb_stage: cute.Tensor,
    tidx,
) -> cute.Tensor:
    """Allocate the per-thread SFB register fragment for native NVFP4."""
    thr_tensor = cute.make_tensor(
        sfb_stage.iterator, _thrfrg_sfb(sfb_stage.layout, tiled_mma)
    )
    thr_vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
    thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
    fragment = thr_tensor[thr_vnk, (None, None)]
    fragment = cute.group_modes(cute.flatten(fragment), 0, 2)
    fragment = cute.group_modes(fragment, 1, 3)
    return cute.make_fragment_like(fragment)


def get_layout_sfa_tv(tiled_mma: cute.TiledMma) -> cute.Layout:
    """Return the SM120 NVFP4 SFA copy thread/value layout."""
    perm_m = tiled_mma.permutation_mnk[0]
    perm_k = tiled_mma.permutation_mnk[2]
    ref = cute.make_layout((cute.size(perm_m), cute.size(perm_k)))
    thr_layout = tiled_mma.thr_layout_vmnk
    atile = (
        None,
        (
            cute.make_layout(
                (cute.size(thr_layout[1]), cute.size(thr_layout[2])),
                stride=(1, 0),
            ),
            None,
        ),
    )
    result = cute.composition(_thrfrg_sfa(ref, tiled_mma), (atile, None))
    return cute.composition(result, (cute.right_inverse(thr_layout), None))


def get_layout_sfb_tv(tiled_mma: cute.TiledMma) -> cute.Layout:
    """Return the SM120 NVFP4 SFB copy thread/value layout."""
    perm_n = tiled_mma.permutation_mnk[1]
    perm_k = tiled_mma.permutation_mnk[2]
    ref = cute.make_layout((cute.size(perm_n), cute.size(perm_k)))
    thr_layout = tiled_mma.thr_layout_vmnk
    atile = (
        None,
        (
            cute.make_layout(
                (cute.size(thr_layout[1]), cute.size(thr_layout[2])),
                stride=(0, 1),
            ),
            None,
        ),
    )
    result = cute.composition(_thrfrg_sfb(ref, tiled_mma), (atile, None))
    return cute.composition(result, (cute.right_inverse(thr_layout), None))


@cute.jit
def issue_m64n8k64_nvfp4(
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
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one public SM120 NVFP4 ``M64 x N8 x K64`` warp-MMA slice."""

    if cutlass.const_expr(n_group >= active_n_groups):
        return

    a_reg = a_frag[None, 0, k_inner]
    b_reg = b_frag[None, n_group, k_inner]
    sfa_reg = cute.make_tensor(
        sfa_frag[None, sfa_m_group, k_inner].iterator,
        cute.make_layout(4),
    )
    sfb_reg = cute.make_tensor(
        sfb_frag[
            None,
            (n_group % 2, (n_group // 2) % 2),
            n_group // 4,
            k_inner,
        ].iterator,
        cute.make_layout(4),
    )
    _arch_mma_m16n8k64_nvfp4(
        acc,
        a_reg,
        b_reg,
        sfa_reg,
        sfb_reg,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        acc_dtype=cutlass.Float32,
        sf_dtype=sf_dtype,
    )


@cute.jit
def issue_m64n8k128_nvfp4(
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
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one SM120 NVFP4 ``M64 x N8 x K128`` warp-MMA slice.

    ``a_frag`` / ``b_frag`` are register fragments already loaded from SMEM
    by the SM120 ldmatrix tiled copies.  ``sfa_frag`` / ``sfb_frag`` are
    register scale fragments copied with the SM120 scale TV layouts.  This helper is
    intentionally the only place
    in the SM120 mainloop that emits the block-scaled MMA operation.  The
    public ``cute.gemm`` lowering calls the same
    ``cute_nvgpu.arch.mma.SM120.block_scaled`` op that
    :func:`emit_mma_m16n8k32_mxfp8_e4m3` wraps directly; keeping the call here
    lets us swap to the explicit low-level form once fragment packing is
    stable.
    """

    for k_inner in cutlass.range_constexpr(0, 2):
        issue_m64n8k64_nvfp4(
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
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            sf_dtype=sf_dtype,
        )
