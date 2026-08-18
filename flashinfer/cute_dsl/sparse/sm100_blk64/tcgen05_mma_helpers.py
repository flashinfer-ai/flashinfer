# Copyright (c) 2025, Tri Dao.
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm

from . import mma_sm100_desc as sm100_desc


def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if isinstance(op, tcgen05.mma.MmaF16BF16Op):
        return "f16"
    if isinstance(op, tcgen05.mma.MmaTF32Op):
        return "tf32"
    if isinstance(op, tcgen05.mma.MmaI8Op):
        return "i8"
    # CUTLASS 4.4 renamed the ordinary (non block-scaled) FP8 op to
    # MmaFP8Op.  Keep the old spelling as well so this repository remains
    # usable with the CUTLASS revision it originally targeted.
    fp8_op_types = tuple(
        op_type
        for op_type in (
            getattr(tcgen05.mma, "MmaFP8Op", None),
            getattr(tcgen05.mma, "MmaF8F6F4Op", None),
        )
        if op_type is not None
    )
    if isinstance(op, fp8_op_types):
        return "f8f6f4"
    mxfp8_op_types = tuple(
        op_type
        for op_type in (
            getattr(tcgen05.mma, "MmaMXF8Op", None),
            getattr(tcgen05.mma, "MmaMXF8F6F4Op", None),
        )
        if op_type is not None
    )
    if isinstance(op, mxfp8_op_types):
        return "mxf8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF4Op):
        return "mxf4"
    if isinstance(op, tcgen05.mma.MmaMXF4NVF4Op):
        return "mxf4nvf4"
    raise TypeError(f"Unsupported tcgen05 MMA op kind: {type(op).__name__}")


@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool = False,
    swap_AB: bool = False,
    num_unroll_groups: int = 1,
) -> None:
    # `not zero_init` below is a Python-level negation, which would silently bake a
    # constant predicate for a runtime Boolean; use gemm_ptx_w_idx for dynamic flags.
    assert isinstance(zero_init, bool), "gemm_w_idx only supports a compile-time bool zero_init"
    if const_expr(swap_AB):
        return gemm_w_idx(tiled_mma, acc, tCrB, tCrA, B_idx, A_idx, zero_init=zero_init, swap_AB=False)
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]

        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k in cutlass.range(cute.size(tCrA.shape[2]), unroll=cute.size(tCrA.shape[2]) // num_unroll_groups):
            mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
            cute.gemm(mma_atom, acc, rA[None, None, k], rB[None, None, k], acc)


@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    **kwargs,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sA_cur = None
    if const_expr(sA is not None):
        sA_cur = sA if const_expr(A_idx is None) else sA[None, None, None, A_idx]
    sB_cur = sB if const_expr(B_idx is None) else sB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    acc_tmem_addr = acc.iterator.toint()
    gemm_ptx_partial(
        mma_atom.op,
        acc_tmem_addr,
        rA,
        rB,
        sA_cur,
        sB_cur,
        zero_init=zero_init,
        **kwargs,
    )


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K if const_expr(op.a_major_mode == tcgen05.OperandMajorMode.K) else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K if const_expr(op.b_major_mode == tcgen05.OperandMajorMode.K) else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = tCrA.layout if const_expr(not is_ts) else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    # zero_init may be a runtime Boolean (e.g. loop-carried accumulate flags); Python
    # `not` on it would bake a wrong constant predicate at trace time, so pass the raw
    # value through and flip the setp comparison instead.
    zero_init_is_dynamic = isinstance(zero_init, Boolean)
    pred_str = "p" if zero_init_is_dynamic else "0" if zero_init else "1"
    pred_input = zero_init if zero_init_is_dynamic else not zero_init
    pred_setp = "setp.eq.b32" if zero_init_is_dynamic else "setp.ne.b32"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(pred_input).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            f"{pred_setp} p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(pred_input).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            assert split_arrive is not None, "split_arrive must be provided when mbar_ptr is not None"
            assert split_arrive % op.shape_mnk[2] == 0, "split_arrive must be a multiple of the MMA K extent"
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            assert 1 <= split_arrive_idx <= cute.size(tCrA.shape[2]), "split_arrive must map to a K-tile index within [1, num_k_tiles]"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            # The post-wait loop below updates smem_desc_b_lo incrementally, and the
            # pre-wait loop that would otherwise seed it is empty when
            # split_arrive_idx == 1, so initialize it from the base descriptor.
            + ("mov.b32 smem_desc_b_lo, smem_desc_b_lo_start;\n\t" if mbar_ptr is not None else "") + f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            f"{pred_setp} p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2]) if const_expr(mbar_ptr is None) else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
