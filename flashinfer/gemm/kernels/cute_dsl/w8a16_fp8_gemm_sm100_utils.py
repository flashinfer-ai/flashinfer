# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Device helpers of the W8A16 (e4m3 weight x 16-bit activation) GEMM kernels.

e4m3 -> bf16 / fp16 conversion for the weight fragments (the Blackwell transform
warps convert whole 32-bit words between the TMA load and ``tcgen05.mma``; the
Hopper consumers load pairs from shared memory and convert them into the MMA
operand registers), and the 64-bit acquire / release counter operations and
fences of the Hopper split tail.  Numerical contract of the conversion: exact for
every finite e4m3 code, denormals included.  The two e4m3fn NaN encodings (0x7F, 0xFF) are not handled: the
ALU bf16 path turns them into +-480, the native ``cvt`` paths into NaN.  Callers
hand the kernel a finite weight; the Python op documents this.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import Uint32, Uint64, dsl_user_op

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


def _select_asm(
    mma_dtype: type[cutlass.Numeric],
    native_cvt: bool,
    bf16_native: str,
    bf16_alu: str,
    f16: str,
) -> str:
    """The PTX for ``mma_dtype``: the native bf16 ``cvt`` when the toolkit has it
    and the caller allows it (sm_100+), else the ALU chain; fp16 always converts
    natively."""
    if mma_dtype is cutlass.BFloat16:
        return bf16_native if native_cvt and _HAS_NATIVE_E4M3_BF16_CVT else bf16_alu
    if mma_dtype is cutlass.Float16:
        return f16
    raise ValueError(f"W8A16 converts e4m3 to bf16 or fp16, not {mma_dtype}")


def _pack_result(out_words: list, mma_dtype: type[cutlass.Numeric], *, loc, ip):
    """The converted 32-bit words (two ``mma_dtype`` values each) as one TensorSSA."""
    out_words_type = ir.VectorType.get([len(out_words)], Uint32.mlir_type, loc=loc)
    out_words_vector = vector.from_elements(out_words_type, out_words, loc=loc, ip=ip)
    count = 2 * len(out_words)
    result_type = ir.VectorType.get([count], mma_dtype.mlir_type, loc=loc)
    result_vector = llvm.bitcast(result_type, out_words_vector, loc=loc, ip=ip)
    return cute.TensorSSA(result_vector, (count,), mma_dtype)


def _asm_for(mma_dtype: type[cutlass.Numeric], native_cvt: bool) -> str:
    return _select_asm(
        mma_dtype,
        native_cvt,
        _E4M3X4_TO_BF16X4_NATIVE,
        _E4M3X4_TO_BF16X4_ALU,
        _E4M3X4_TO_F16X4,
    )


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
    ``mma_dtype`` values in the same order.  ``native_cvt=False`` forces the ALU
    bf16 path: the native ``cvt.rn.bf16x2.e4m3x2`` exists on sm_100+ only, so a
    kernel compiled for Hopper must pass False for bf16 (fp16 converts natively on
    both).  The Blackwell kernel is the caller of this fragment form."""
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
    return _pack_result(out_words, mma_dtype, loc=loc, ip=ip)


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
    return _select_asm(
        mma_dtype,
        native_cvt,
        _E4M3X2_TO_BF16X2_NATIVE_W,
        _E4M3X2_TO_BF16X2_ALU_W,
        _E4M3X2_TO_F16X2_W,
    )


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
    ``load_e4m3_unit``) to ``mma_dtype``; ``2 * len(words)`` values in order.
    ``native_cvt`` as in :func:`convert_e4m3_fragment`: the Hopper kernel, the
    caller of this word form, passes False since its bf16 ``cvt`` is sm_100+."""
    asm = _asm_for_pair(mma_dtype, native_cvt)
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
    return _pack_result(out_words, mma_dtype, loc=loc, ip=ip)


def _asm_op(result_type, operands, asm, constraints, *, loc, ip):
    return llvm.inline_asm(
        result_type,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_store_complete_fence(*, loc=None, ip=None) -> None:
    """Wait for every bulk (TMA) store of this thread to complete, then order the
    async-proxy global writes before the generic proxy and release them at GPU scope.
    The tail fix-up's partial tiles are published this way before the counter add."""
    _asm_op(
        None,
        [],
        "cp.async.bulk.wait_group 0;\nfence.proxy.async.global;\nfence.acq_rel.gpu;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_acq_rel_gpu_u64(
    addr: cutlass.Int64, value: cutlass.Int64, *, loc=None, ip=None
) -> cutlass.Int64:
    """``atom.acq_rel.gpu.global.add.u64`` on the global 64-bit word at ``addr``;
    returns the old value."""
    old = _asm_op(
        Uint64.mlir_type,
        [addr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "atom.acq_rel.gpu.global.add.u64 $0, [$1], $2;",
        "=l,l,l",
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(old)


@dsl_user_op
def ld_acquire_gpu_u64(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int64:
    val = _asm_op(
        Uint64.mlir_type,
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.acquire.gpu.global.u64 $0, [$1];",
        "=l,l",
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(val)


@dsl_user_op
def atomic_cas_acq_rel_gpu_u64(
    addr: cutlass.Int64,
    expected: cutlass.Int64,
    desired: cutlass.Int64,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int64:
    """``atom.acq_rel.gpu.global.cas.b64`` on the global 64-bit word at ``addr``:
    stores ``desired`` if the word equals ``expected``; returns the value found."""
    found = _asm_op(
        Uint64.mlir_type,
        [
            addr.ir_value(loc=loc, ip=ip),
            expected.ir_value(loc=loc, ip=ip),
            desired.ir_value(loc=loc, ip=ip),
        ],
        "atom.acq_rel.gpu.global.cas.b64 $0, [$1], $2, $3;",
        "=l,l,l,l",
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(found)


@dsl_user_op
def popc_u64(value: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int32:
    """The number of set bits of ``value`` (``popc.b64``)."""
    count = _asm_op(
        Uint32.mlir_type,
        [value.ir_value(loc=loc, ip=ip)],
        "popc.b64 $0, $1;",
        "=r,l",
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(count)


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    _asm_op(None, [], f"nanosleep.u32 {int(ns)};", "", loc=loc, ip=ip)


__all__ = [
    "atomic_add_acq_rel_gpu_u64",
    "bulk_store_complete_fence",
    "convert_e4m3_fragment",
    "convert_e4m3_words",
    "ld_acquire_gpu_u64",
    "load_e4m3_unit",
    "nanosleep",
]
