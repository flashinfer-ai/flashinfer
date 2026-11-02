# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 GEMM helpers shared by DSA CuTe DSL kernels."""

import re
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import OperandMajorMode


from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.sm100 import mma_desc as sm100_desc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_swizzle_from_pointer(ptr: cute.Pointer) -> cute.Swizzle:
    """Extract swizzle parameters from a pointer's swizzle_type."""
    swizzle_str = str(ptr.type.swizzle_type)
    match = re.search(r"S<(\d+),(\d+),(\d+)>", swizzle_str)
    if match:
        b, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return cute.make_swizzle(b, m, s)
    else:
        raise ValueError(f"Could not parse swizzle_type: {swizzle_str}")


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


# ---------------------------------------------------------------------------
# gemm_ptx_partial  (SS and TS paths)
# ---------------------------------------------------------------------------


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
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    if const_expr(not is_ts):
        sA_swizzle = parse_swizzle_from_pointer(sA.iterator)
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K if const_expr(op.a_major_mode == OperandMajorMode.K) else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = parse_swizzle_from_pointer(sB.iterator)
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K if const_expr(op.b_major_mode == OperandMajorMode.K) else sm100_desc.Major.MN,
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
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
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
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
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
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
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
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2]) if const_expr(mbar_ptr is None) else cute.size(tCrA.shape[2]) // 4 * 3,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(cute.size(tCrA.shape[2]) // 4 * 3, cute.size(tCrA.shape[2]))
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


@cute.jit
def gemm_ptx_partial_mxfp8(
    op: cute.nvgpu.tcgen05.mma.BlockScaledMmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    tSFA: cute.Tensor,
    tSFB: cute.Tensor,
    zero_init: bool | Boolean = False,
    kphase_offset: cutlass.Constexpr[int] = 0,
) -> None:
    """Issue blockscaled MXFP8 or MXFP4 QK MMA with explicit scale TMEM operands.

    The Indexer forward path uses K as operand A and packed Q as operand B, so
    K scales are SFA and Q scales are SFB.  All of a row's E8M0 scales share one
    TMEM column, and the instruction descriptor's scale factor id selects the
    ones each MMA reads: ``kind::mxf8f6f4`` (K=32) reads one scale, at id
    ``kphase``; ``kind::mxf4`` (packed E2M1, K=64) reads two, at id
    ``2 * kphase``.  ``kphase_offset`` is the index of this chunk's first MMA
    along K.  This helper has no inline mbarrier wait path; callers must pass
    warp-uniform, fully populated SFA/SFB TMEM tensors for the current MMA.
    """
    sA_layout = sA.layout
    sB_layout = sB.layout
    sA_swizzle = parse_swizzle_from_pointer(sA.iterator)
    smem_desc_base_a: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
            sA_swizzle,
            sm100_desc.Major.K if const_expr(op.a_major_mode == OperandMajorMode.K) else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
    smem_desc_a_hi = const_expr(smem_desc_a_hi)

    sB_swizzle = parse_swizzle_from_pointer(sB.iterator)
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K if const_expr(op.b_major_mode == OperandMajorMode.K) else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    offset_a = [cute.crd2idx((0, 0, k), tCrA.layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    # kind::mxf4 (packed E2M1, K=64) reads two E8M0 scales per row per MMA, kind::mxf8f6f4 (K=32) one.
    is_mxf4 = isinstance(op, cute.nvgpu.tcgen05.mma.MmaMXF4Op)
    mma_kind = "kind::mxf4.block_scale.scale_vec::2X" if const_expr(is_mxf4) else "kind::mxf8f6f4.block_scale.scale_vec::1X"
    sf_ids_per_mma = 2 if const_expr(is_mxf4) else 1
    idescs = [const_expr(sm100_desc.blockscaled_mma_op_to_idesc(op, (k + kphase_offset) * sf_ids_per_mma)) for k in range(cute.size(tCrA.shape[2]))]

    smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    tmem_sfa = tSFA[None, None, kphase_offset].iterator.toint()
    tmem_sfb = tSFB[None, None, kphase_offset].iterator.toint()
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"

    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(tmem_sfa)).ir_value(),
            Int32(cute.arch.make_warp_uniform(tmem_sfb)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 idesc;\n\t"
        ".reg .b32 tmem_acc;\n\t"
        ".reg .b32 tmem_sfa;\n\t"
        ".reg .b32 tmem_sfb;\n\t"
        ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
        ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
        ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
        ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        "mov.b32 tmem_acc, $3;\n\t"
        "mov.b32 tmem_sfa, $4;\n\t"
        "mov.b32 tmem_sfb, $5;\n\t"
        "mov.b32 smem_desc_a_lo_start, $0;\n\t"
        "mov.b32 smem_desc_b_lo_start, $1;\n\t"
        f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
        f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
        f"mov.b32 idesc, {hex(idescs[0])};\n\t"
        "mov.b64 smem_desc_a, {smem_desc_a_lo_start, smem_desc_a_hi};\n\t"
        "mov.b64 smem_desc_b, {smem_desc_b_lo_start, smem_desc_b_hi};\n\t"
        "setp.ne.b32 p, $2, 0;\n\t"
        f"@leader_thread tcgen05.mma.cta_group::1.{mma_kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, [tmem_sfa], [tmem_sfb], {pred_str};\n\t"
        + "".join(
            (
                f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                f"mov.b32 idesc, {hex(idescs[k])};\n\t"
                f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                f"@leader_thread tcgen05.mma.cta_group::1.{mma_kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, [tmem_sfa], [tmem_sfb], 1;\n\t"
            )
            for k in range(1, cute.size(tCrA.shape[2]))
        )
        + "}\n",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
