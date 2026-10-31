# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Conversion helpers for the SM100 W8A16 (e4m3 weight x 16-bit activation) GEMM.

The transform warps of ``w8a16_fp8_gemm_sm100.py`` run these on the weight
fragments between the TMA load and ``tcgen05.mma``.  Numerical contract (same as
``include/flashinfer/gemm/w8a16_convert.cuh``): exact for every finite e4m3 code,
denormals included.  The two e4m3fn NaN encodings (0x7F, 0xFF) are not handled: the
ALU bf16 path turns them into +-480, the native ``cvt`` paths into NaN.  Callers
hand the kernel a finite weight; the Python op documents this.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import Uint16, Uint32, dsl_user_op

# ``cvt.rn.bf16x2.e4m3x2`` (PTX ISA 9.2, sm_100+) converts a pair in one instruction;
# it needs CUDA 13.2 or newer.  Older toolkits take the four-instruction ALU chain.
_HAS_NATIVE_E4M3_BF16_CVT = cutlass.CUDA_VERSION.major > 13 or (
    cutlass.CUDA_VERSION.major == 13 and cutlass.CUDA_VERSION.minor >= 2
)

# Per 32-bit word of four e4m3 bytes: two 16-bit pairs, value order kept (byte 0 lands in
# the low half of the first output).  $0/$1 outputs, $2 input.
_E4M3X4_TO_BF16X4_NATIVE = """
{
    .reg .b16 l, h;
    mov.b32 {l, h}, $2;
    cvt.rn.bf16x2.e4m3x2 $0, l;
    cvt.rn.bf16x2.e4m3x2 $1, h;
}
"""

# Without a packed e4m3 -> bf16 instruction: prmt selector nibbles 0x9180 / 0xB3A2 put
# byte k in the low byte of lane k with its sign replicated into the high byte (selector
# bit 3); (v << 4) puts the 7 magnitude bits at [10:4] and the replicated sign at [15],
# mask 0x87F0 keeps exactly those.  The bf16 exponent field then holds e4m3's raw
# exponent, so one bf16x2 multiply by 2^(127 - 7) = 2^120 (bf16 0x7B80) fixes the bias;
# an e4m3 denormal becomes a bf16 denormal that the same multiply scales exactly.
_E4M3X4_TO_BF16X4_ALU = """
{
    .reg .b32 lo_s, hi_s, k;
    mov.b32 k, 0x7B807B80;
    prmt.b32 lo_s, $2, 0, 0x9180;
    prmt.b32 hi_s, $2, 0, 0xB3A2;
    shl.b32 lo_s, lo_s, 4;
    shl.b32 hi_s, hi_s, 4;
    and.b32 lo_s, lo_s, 0x87F087F0;
    and.b32 hi_s, hi_s, 0x87F087F0;
    mul.rn.bf16x2 $0, lo_s, k;
    mul.rn.bf16x2 $1, hi_s, k;
}
"""

# ``cvt.rn.f16x2.e4m3x2`` exists since sm_89 (PTX ISA 7.8).
_E4M3X4_TO_F16X4 = """
{
    .reg .b16 l, h;
    mov.b32 {l, h}, $2;
    cvt.rn.f16x2.e4m3x2 $0, l;
    cvt.rn.f16x2.e4m3x2 $1, h;
}
"""


def _asm_for(mma_dtype: type[cutlass.Numeric], native_cvt: bool) -> str:
    if mma_dtype is cutlass.BFloat16:
        return (
            _E4M3X4_TO_BF16X4_NATIVE
            if native_cvt and _HAS_NATIVE_E4M3_BF16_CVT
            else _E4M3X4_TO_BF16X4_ALU
        )
    if mma_dtype is cutlass.Float16:
        return _E4M3X4_TO_F16X4
    raise ValueError(f"W8A16 converts e4m3 to bf16 or fp16, not {mma_dtype}")


@dsl_user_op
def convert_e4m3_fragment(
    packed_fragment: cute.TensorSSA,
    mma_dtype: type[cutlass.Numeric],
    native_cvt: bool = True,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    """Convert a fragment of e4m3 bytes (``Uint8`` SSA, a multiple of 4 long) to
    ``mma_dtype`` values in the same order.  ``native_cvt=False`` forces the ALU bf16
    path (``cvt.rn.bf16x2.e4m3x2`` is sm_100+; the Hopper kernel passes it)."""
    count = cute.size(packed_fragment.shape)
    assert count % 4 == 0, "e4m3 fragments convert four bytes (one word) at a time"
    num_words = count // 4
    words_type = ir.VectorType.get([num_words], Uint32.mlir_type, loc=loc)
    words_vector = llvm.bitcast(
        words_type, packed_fragment.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    asm = _asm_for(mma_dtype, native_cvt)
    pair_type = llvm.StructType.get_literal([Uint32.mlir_type] * 2)
    out_words: list[ir.Value] = []
    for idx in range(num_words):
        word = vector.extract(words_vector, [], [idx], loc=loc, ip=ip)
        pair = llvm.inline_asm(
            pair_type,
            [word],
            asm,
            "=r,=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        out_words.append(llvm.extractvalue(Uint32.mlir_type, pair, [0], loc=loc, ip=ip))
        out_words.append(llvm.extractvalue(Uint32.mlir_type, pair, [1], loc=loc, ip=ip))
    out_words_type = ir.VectorType.get([2 * num_words], Uint32.mlir_type, loc=loc)
    out_words_vector = vector.from_elements(out_words_type, out_words, loc=loc, ip=ip)
    result_type = ir.VectorType.get([count], mma_dtype.mlir_type, loc=loc)
    result_vector = llvm.bitcast(result_type, out_words_vector, loc=loc, ip=ip)
    return cute.TensorSSA(result_vector, (count,), mma_dtype)


# One e4m3 pair, zero-extended into a 32-bit register ($1, bytes 0 and 1) -> one bf16x2
# / f16x2 word ($0): the low-pair half of the 4-byte converter above.  Taking the pair
# as a word lets the compiler feed the 16-bit smem load's zero-extended register in
# directly (no repacking instruction).
_E4M3X2_TO_BF16X2_ALU_W = """
{
    .reg .b32 s, k;
    mov.b32 k, 0x7B807B80;
    prmt.b32 s, $1, 0, 0x9180;
    shl.b32 s, s, 4;
    and.b32 s, s, 0x87F087F0;
    mul.rn.bf16x2 $0, s, k;
}
"""
_E4M3X2_TO_BF16X2_NATIVE_W = """
{
    .reg .b16 l, h;
    mov.b32 {l, h}, $1;
    cvt.rn.bf16x2.e4m3x2 $0, l;
}
"""
_E4M3X2_TO_F16X2_W = """
{
    .reg .b16 l, h;
    mov.b32 {l, h}, $1;
    cvt.rn.f16x2.e4m3x2 $0, l;
}
"""


def _asm_for_pair(mma_dtype: type[cutlass.Numeric], native_cvt: bool) -> str:
    if mma_dtype is cutlass.BFloat16:
        return (
            _E4M3X2_TO_BF16X2_NATIVE_W
            if native_cvt and _HAS_NATIVE_E4M3_BF16_CVT
            else _E4M3X2_TO_BF16X2_ALU_W
        )
    if mma_dtype is cutlass.Float16:
        return _E4M3X2_TO_F16X2_W
    raise ValueError(f"W8A16 converts e4m3 to bf16 or fp16, not {mma_dtype}")


@dsl_user_op
def convert_e4m3_pairs(
    pairs: cute.TensorSSA,
    mma_dtype: type[cutlass.Numeric],
    native_cvt: bool = True,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    """Convert e4m3 pairs (``Uint16`` SSA, each holding two consecutive weights) to
    ``mma_dtype`` values in the same order, one pair per asm statement, straight from
    the 16-bit register the pair was loaded into.  The Hopper mainloop loads its A
    fragment as 16-bit units (two consecutive k of one row); converting whole 32-bit
    words would first have the compiler pack bytes into words, which doubled the ALU
    work of the conversion."""
    count = cute.size(pairs.shape)
    half_type = ir.VectorType.get([count], Uint16.mlir_type, loc=loc)
    halves_vector = llvm.bitcast(
        half_type, pairs.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    asm = _asm_for_pair(mma_dtype, native_cvt)
    out_words: list[ir.Value] = []
    for idx in range(count):
        half = vector.extract(halves_vector, [], [idx], loc=loc, ip=ip)
        half32 = llvm.zext(Uint32.mlir_type, half, loc=loc, ip=ip)
        word = llvm.inline_asm(
            Uint32.mlir_type,
            [half32],
            asm,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        out_words.append(word)
    out_words_type = ir.VectorType.get([count], Uint32.mlir_type, loc=loc)
    out_words_vector = vector.from_elements(out_words_type, out_words, loc=loc, ip=ip)
    result_type = ir.VectorType.get([2 * count], mma_dtype.mlir_type, loc=loc)
    result_vector = llvm.bitcast(result_type, out_words_vector, loc=loc, ip=ip)
    return cute.TensorSSA(result_vector, (2 * count,), mma_dtype)


# Load one e4m3 pair (a 16-bit unit at shared address $1) and convert it to one bf16x2
# / f16x2 word ($0) in a single statement.
_E4M3X2_SMEM_TO_BF16X2_ALU = """
{
    .reg .b32 s, k;
    ld.shared.u16 s, [$1];
    mov.b32 k, 0x7B807B80;
    prmt.b32 s, s, 0, 0x9180;
    shl.b32 s, s, 4;
    and.b32 s, s, 0x87F087F0;
    mul.rn.bf16x2 $0, s, k;
}
"""
_E4M3X2_SMEM_TO_BF16X2_NATIVE = """
{
    .reg .b16 h;
    ld.shared.u16 h, [$1];
    cvt.rn.bf16x2.e4m3x2 $0, h;
}
"""
_E4M3X2_SMEM_TO_F16X2 = """
{
    .reg .b16 h;
    ld.shared.u16 h, [$1];
    cvt.rn.f16x2.e4m3x2 $0, h;
}
"""


def _asm_for_pair_smem(mma_dtype: type[cutlass.Numeric], native_cvt: bool) -> str:
    if mma_dtype is cutlass.BFloat16:
        return (
            _E4M3X2_SMEM_TO_BF16X2_NATIVE
            if native_cvt and _HAS_NATIVE_E4M3_BF16_CVT
            else _E4M3X2_SMEM_TO_BF16X2_ALU
        )
    if mma_dtype is cutlass.Float16:
        return _E4M3X2_SMEM_TO_F16X2
    raise ValueError(f"W8A16 converts e4m3 to bf16 or fp16, not {mma_dtype}")


@dsl_user_op
def load_convert_e4m3_units(
    addrs: list,
    mma_dtype: type[cutlass.Numeric],
    native_cvt: bool = True,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    """Load the e4m3 pairs at the shared-memory addresses ``addrs`` (``Int32`` each, a
    16-bit unit of two consecutive weights) and convert them to ``mma_dtype``; the
    result holds ``2 * len(addrs)`` values in order.  The Hopper mainloop computes the
    swizzled unit addresses of its fragment once per thread and loads through this, one
    load-and-convert statement per unit, instead of a copy whose per-load swizzle
    arithmetic the compiler could not hoist."""
    asm = _asm_for_pair_smem(mma_dtype, native_cvt)
    count = len(addrs)
    out_words: list[ir.Value] = []
    for addr in addrs:
        word = llvm.inline_asm(
            Uint32.mlir_type,
            [addr.ir_value(loc=loc, ip=ip)],
            asm,
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        out_words.append(word)
    out_words_type = ir.VectorType.get([count], Uint32.mlir_type, loc=loc)
    out_words_vector = vector.from_elements(out_words_type, out_words, loc=loc, ip=ip)
    result_type = ir.VectorType.get([2 * count], mma_dtype.mlir_type, loc=loc)
    result_vector = llvm.bitcast(result_type, out_words_vector, loc=loc, ip=ip)
    return cute.TensorSSA(result_vector, (2 * count,), mma_dtype)


@dsl_user_op
def load_e4m3_unit(addr: cutlass.Int32, *, loc=None, ip=None) -> cutlass.Uint32:
    """``ld.shared.u16`` of the e4m3 pair at shared address ``addr`` into a zero-extended
    32-bit register (the pair converter's input)."""
    word = llvm.inline_asm(
        Uint32.mlir_type,
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.u16 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(word)


@dsl_user_op
def convert_e4m3_words(
    words: list,
    mma_dtype: type[cutlass.Numeric],
    native_cvt: bool = True,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    """Convert e4m3 pairs held zero-extended in 32-bit registers (``Uint32`` each, from
    ``load_e4m3_unit``) to ``mma_dtype``; ``2 * len(words)`` values in order."""
    asm = _asm_for_pair(mma_dtype, native_cvt)
    count = len(words)
    out_words: list[ir.Value] = []
    for w in words:
        out = llvm.inline_asm(
            Uint32.mlir_type,
            [w.ir_value(loc=loc, ip=ip)],
            asm,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        out_words.append(out)
    out_words_type = ir.VectorType.get([count], Uint32.mlir_type, loc=loc)
    out_words_vector = vector.from_elements(out_words_type, out_words, loc=loc, ip=ip)
    result_type = ir.VectorType.get([2 * count], mma_dtype.mlir_type, loc=loc)
    result_vector = llvm.bitcast(result_type, out_words_vector, loc=loc, ip=ip)
    return cute.TensorSSA(result_vector, (2 * count,), mma_dtype)


@dsl_user_op
def global_timer_lo(tag: int, *, loc=None, ip=None) -> cutlass.Int32:
    """Low 32 bits of ``%globaltimer`` (nanoseconds); debug-mode timelines only.
    ``tag`` is unused by the asm: it makes every call site a distinct op (a no-argument
    user op is reused across call sites, and a value then escapes its region)."""
    val = llvm.inline_asm(
        Uint32.mlir_type,
        [],
        "{ .reg .u64 t; mov.u64 t, %globaltimer; cvt.u32.u64 $0, t; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(val)


__all__ = [
    "convert_e4m3_fragment",
    "convert_e4m3_pairs",
    "convert_e4m3_words",
    "global_timer_lo",
    "load_convert_e4m3_units",
    "load_e4m3_unit",
]
