# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Device-side helpers shared by the two SM120 KDA prefill variants.

PTX wrappers (``ldmatrix``/``stmatrix``/``mma``/``movmatrix``, TMA loads and
stores, the tensormap fence), the S128 shared-memory images and the fragment
helpers the two kernels build on.  The kernels themselves, the layouts specific
to one schedule and TMA descriptor construction stay with their variant.

This module imports the CuTe DSL, so the package facade must not import it at
module scope.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from .runtime import DK


BT = 16
KEY_BLOCKS = DK // BT  # 8

# raw_bf16_s128: 2 segments x 128 bytes, 8-element (16 byte) groups.
BF16_SEGMENT_ELEMS = 64
BF16_GROUP_ELEMS = 8
BF16_SEGMENTS = DK // BF16_SEGMENT_ELEMS
BF16_SEGMENT_STRIDE = BT * BF16_SEGMENT_ELEMS  # 1024 elements
BF16_ROW_XOR_MASK = BF16_SEGMENT_ELEMS // BF16_GROUP_ELEMS - 1  # 7

# raw_f32_s128: 4 segments x 128 bytes, 4-element (16 byte) groups.
F32_SEGMENT_ELEMS = 32
F32_GROUP_ELEMS = 4
F32_SEGMENTS = DK // F32_SEGMENT_ELEMS
F32_SEGMENT_STRIDE = BT * F32_SEGMENT_ELEMS  # 512 elements
F32_ROW_XOR_MASK = F32_SEGMENT_ELEMS // F32_GROUP_ELEMS - 1  # 7

#: Row stride and column permutation of the 16x16 pairwise image.
PAIRWISE_ROW_STRIDE = BT
PAIRWISE_COL_XOR = 8

#: Physical rows of one value in the state image, by element width.  The state
#: is stored ``[V, K]`` and read logically as ``[K, V]``, so one value spans
#: 128 keys = 2 BF16 or 4 FP32 128-byte segments.
STATE_BF16_ROWS_PER_VALUE = DK // BF16_SEGMENT_ELEMS  # 2
STATE_F32_ROWS_PER_VALUE = DK // F32_SEGMENT_ELEMS  # 4


def _stmatrix(count: str, trans: str, smem_ptr, regs, *, loc=None, ip=None):
    ins = ", ".join(f"${i + 1}" for i in range(len(regs)))
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            *[cutlass.Int32(r).ir_value(loc=loc, ip=ip) for r in regs],
        ],
        f"stmatrix.sync.aligned.m8n8{count}{trans}.shared.b16 [$0], {{{ins}}};",
        ",".join(["r"] * (len(regs) + 1)),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmatrix_x2(smem_ptr, r0, r1, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x2.shared.b16``: store two 8x8 b16 tiles."""
    _stmatrix(".x2", "", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x2_trans(smem_ptr, r0, r1, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x2.trans.shared.b16``: transposed x2 store.

    Exactly inverts :func:`ldmatrix_x2_trans` against the same pointer map.
    """
    _stmatrix(".x2", ".trans", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def fence_tensormap_acquire(desc_addr, *, loc=None, ip=None):
    """Publish a host-written tensor map to the tensormap proxy.

    The descriptor is written by the host and read by the TMA unit through a
    different proxy, so the kernel has to acquire it before first use.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip)],
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_load_3d(smem_ptr, desc_addr, mbar_ptr, c0, c1, c2, *, loc=None, ip=None):
    """One ``cp.async.bulk.tensor.3d`` box, global to shared.

    ``shared::cta`` rather than ``shared::cluster``: sm_120 has no thread block
    clusters.  Completion is reported to ``mbar_ptr`` as transaction bytes, so
    the consumer waits on the mbarrier rather than on a commit group.
    """
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [$0], [$1, {$3, $4, $5}], [$2];",
        "r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_3d(desc_addr, smem_ptr, c0, c1, c2, *, loc=None, ip=None):
    """One ``cp.async.bulk.tensor.3d`` box, shared to global.

    Stores have no mbarrier; they are tracked with bulk commit groups, and the
    ``read`` wait below is what releases the source SMEM for reuse.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [$0, {$2, $3, $4}], [$1];",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_commit_group(*, loc=None, ip=None):
    """Close the current bulk-store group."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_wait_read(keep: int, *, loc=None, ip=None):
    """Wait until at most ``keep`` bulk-store groups still hold their source.

    ``.read`` is a source-SMEM reuse guarantee, not a claim that the store is
    globally visible.
    """
    llvm.inline_asm(
        None,
        [],
        f"cp.async.bulk.wait_group.read {keep};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def warp_arrive(mbar, lane):
    """One arrival per warp on an mbarrier whose arrival count is in warps.

    ``mbarrier.arrive`` counts per thread, so a whole warp arriving would
    overshoot a per-warp arrival count.  The warp synchronization before the
    elected arrival is what makes the other 31 lanes' shared-memory stores
    visible to whoever observes the arrival; the arrival itself carries release
    semantics at CTA scope for the electing lane.
    """
    cute.arch.sync_warp()
    if lane == 0:
        cute.arch.mbarrier_arrive(mbar)


@cute.jit
def vec_at(ptr, idx, elems):
    """``ptr + idx`` as an ``elems``-long tensor, keeping the pointer alignment.

    ``Pointer.__add__`` lowers the pointer's ``alignment`` attribute to one
    element whenever the offset is dynamic -- even for ``ptr + 8 * dyn``, since
    it does not reason about the multiplier.  ``autovec_copy`` honours that
    attribute, so without this every 8-element BF16 access lowers to eight
    ``STG.E.U16`` / ``STS.U16`` instructions instead of one 128-bit access.

    Callers must pass an index that is a multiple of ``elems``; the ``assume``
    states that divisor as a compile-time constraint, not a round-up, so it
    costs no instructions.
    """
    return cute.make_tensor(
        ptr + cute.assume(cutlass.Int32(idx), divby=elems), cute.make_layout(elems)
    )


def _ldmatrix(count: str, trans: str, smem_ptr, num: int, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    outs = ", ".join(f"${i}" for i in range(num))
    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.IntegerType.get_signless(32)] * num),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8{count}{trans}.shared.b16 {{{outs}}}, [${num}];",
        ",".join(["=r"] * num) + ",r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            llvm.extractvalue(
                _T.IntegerType.get_signless(32), struct, [i], loc=loc, ip=ip
            )
        )
        for i in range(num)
    )


@dsl_user_op
def ldmatrix_x2(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x2.shared.b16`` -> two b32 registers.

    Only lanes 0-15 supply addresses.
    """
    return _ldmatrix(".x2", "", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x2_trans(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`` -> two b32 registers.

    Same addresses as :func:`ldmatrix_x2`, read down the memory columns instead
    of across its rows; this is how the physical ``[V, K]`` state is read as a
    logical ``[K, V]`` tile.
    """
    return _ldmatrix(".x2", ".trans", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4_trans(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`` -> four b32 registers.

    Four transposed 8x8 tiles; all 32 lanes supply addresses.
    """
    return _ldmatrix(".x4", ".trans", smem_ptr, 4, loc=loc, ip=ip)


@dsl_user_op
def movmatrix_b16(value, *, loc=None, ip=None):
    """``movmatrix.sync.aligned.m8n8.trans.b16``: transpose one 8x8 b16 tile."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _mma_m16n8k16(
    kind: str, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None
):
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.F32Type.get()] * 4),
        [
            cutlass.Int32(a0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a3).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c3).ir_value(loc=loc, ip=ip),
        ],
        f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(
            llvm.extractvalue(_T.F32Type.get(), struct, [i], loc=loc, ip=ip)
        )
        for i in range(4)
    )


@dsl_user_op
def mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``."""

    return _mma_m16n8k16("bf16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def mma_m16n8k16_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``."""

    return _mma_m16n8k16("f16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def unpack_bf16x2(value, *, loc=None, ip=None):
    """Widen a packed BF16 pair to two FP32 (``cvt.f32.bf16``), low half first."""
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.F32Type.get()] * 2),
        [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi;"
        "  mov.b32 {lo, hi}, $2;"
        "  cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(
            llvm.extractvalue(_T.F32Type.get(), struct, [i], loc=loc, ip=ip)
        )
        for i in range(2)
    )


@cute.jit
def mma_n8(a, b, c):
    """One native ``m16n8k16``: A is four registers, B two, C four."""
    return mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])


def _s128_index(row, column, segment_elems, group_elems):
    """S128 element index of ``column`` within 128-byte row ``row``.

    ``row`` here is the *segment* row -- the unit the swizzle XOR keys on -- not
    a logical matrix row.  The three recurrence images differ only in what they
    call a segment row and how they split a logical coordinate into one.
    """
    group = column // group_elems
    inner = column - group * group_elems
    return (
        row * segment_elems
        + (group ^ (row & (segment_elems // group_elems - 1))) * group_elems
        + inner
    )


def raw_bf16_s128(token, dim):
    """Physical BF16 element index of logical ``(token, dim)``.

    Used for raw Q, raw K, ``Ki``, ``Kd`` and ``Qd``.  ``Kd``/``Qd`` carry no
    feature permutation, so they share this image with the
    raw stages and with ``Ki``.
    """
    segment = dim // BF16_SEGMENT_ELEMS
    local = dim - segment * BF16_SEGMENT_ELEMS
    group = local // BF16_GROUP_ELEMS
    inner = local - group * BF16_GROUP_ELEMS
    return (
        segment * BF16_SEGMENT_STRIDE
        + token * BF16_SEGMENT_ELEMS
        + (group ^ (token & BF16_ROW_XOR_MASK)) * BF16_GROUP_ELEMS
        + inner
    )


def raw_f32_s128(token, dim):
    """Physical FP32 element index of logical ``(token, dim)``.

    The single FP32 image in the kernel: TMA destination for raw ``G`` and,
    after the gate scan, in-place storage for FP32 ``exp_g`` at the very same
    addresses.
    """
    segment = dim // F32_SEGMENT_ELEMS
    local = dim - segment * F32_SEGMENT_ELEMS
    group = local // F32_GROUP_ELEMS
    inner = local - group * F32_GROUP_ELEMS
    return (
        segment * F32_SEGMENT_STRIDE
        + token * F32_SEGMENT_ELEMS
        + (group ^ (token & F32_ROW_XOR_MASK)) * F32_GROUP_ELEMS
        + inner
    )


@cute.jit
def store_vec8_bf16(ptr, idx, frag):
    """Store an 8-element BF16 fragment as one 16-byte access."""
    cute.autovec_copy(frag, vec_at(ptr, idx, 8))


@cute.jit
def store_vec4_f32(ptr, idx, frag):
    """Store a 4-element FP32 fragment as one 16-byte access."""
    cute.autovec_copy(frag, vec_at(ptr, idx, 4))


def _pack(instr: str, lo, hi, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            f"{instr} $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """``cvt.rn.bf16x2.f32``; ``lo`` lands in the low half."""
    return _pack("cvt.rn.bf16x2.f32", lo, hi, loc=loc, ip=ip)


@dsl_user_op
def pack_f16x2(lo, hi, *, loc=None, ip=None):
    """``cvt.rn.f16x2.f32``; ``lo`` lands in the low half.

    The inverse chain's pack: it must not go through BF16 first -- the three
    extra significand bits are the point.
    """
    return _pack("cvt.rn.f16x2.f32", lo, hi, loc=loc, ip=ip)


def _binary_packed(instr: str, a, b, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            f"{instr} $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_bf16x2(a, b, *, loc=None, ip=None):
    """``mul.rn.bf16x2``: two BF16 products, each rounded once."""
    return _binary_packed("mul.rn.bf16x2", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_bf16x2(a, b, *, loc=None, ip=None):
    """``sub.rn.bf16x2``: the residual ``BF16(V - X)`` for two tokens at once."""
    return _binary_packed("sub.rn.bf16x2", a, b, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16`` -> four b32.

    Qd/Kd A operands, the Ki B operand and the pairwise tiles.  Which of those
    it produces is entirely a property of the pointer map it is given.
    """
    return _ldmatrix(".x4", "", smem_ptr, 4, loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x4(smem_ptr, r0, r1, r2, r3, *, loc=None, ip=None):
    """``stmatrix...x4``: Qd/Kd/Ki publication, AinvBeta, Aq and Ak.T."""
    _stmatrix(".x4", "", smem_ptr, (r0, r1, r2, r3), loc=loc, ip=ip)


@cute.jit
def bf16_round(x):
    """R16: ``cvt.rn.bf16.f32`` widened back to FP32."""
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


@cute.jit
def f16_round(x):
    """R_F16: FP32 -> FP16 -> FP32, the inverse chain's boundary."""
    return x.to(cutlass.Float16).to(cutlass.Float32)


@cute.jit
def a_to_b(f):
    """A-layout -> B-layout of the same matrix; register order unchanged."""
    return (
        movmatrix_b16(f[0]),
        movmatrix_b16(f[1]),
        movmatrix_b16(f[2]),
        movmatrix_b16(f[3]),
    )


@cute.jit
def mma_16x16(a, b, c):
    """One logical ``m16n16k16``: the N=8 low half, then the high half.

    That order is fixed, and no rounding may happen between the two
    halves or inside the K reduction.
    """
    n0 = mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[2], b[3], c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


@cute.jit
def mma_16x16_f16(a, b, c):
    """:func:`mma_16x16` with FP16 operands: the twelve inverse MMAs."""
    n0 = mma_m16n8k16_f16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_f16(a[0], a[1], a[2], a[3], b[2], b[3], c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


def mma_c_coord(lane, reg, n_base=0):
    """Logical ``(row, n)`` of FP32 accumulator register ``reg``."""
    g = lane >> 2
    q = lane & 3
    return (g + 8 * (reg >> 1), n_base + 2 * q + (reg & 1))


@cute.jit
def vec8_bf16(ptr, idx):
    """Load eight adjacent BF16 at ``ptr + idx`` as one 16-byte access."""
    frag = cute.make_rmem_tensor(8, cutlass.BFloat16)
    cute.autovec_copy(vec_at(ptr, idx, 8), frag)
    return frag


@cute.jit
def zero_vec8_bf16(ptr, idx):
    """Zero eight adjacent BF16 at ``ptr + idx`` with one 16-byte store."""
    zeros = cute.make_rmem_tensor(8, cutlass.BFloat16)
    for i in cutlass.range_constexpr(8):
        zeros[i] = cutlass.BFloat16(0.0)
    cute.autovec_copy(zeros, vec_at(ptr, idx, 8))


@cute.jit
def zero_vec4_f32(ptr, idx):
    """Zero four adjacent FP32 at ``ptr + idx`` with one 16-byte store."""
    zeros = cute.make_rmem_tensor(4, cutlass.Float32)
    for i in cutlass.range_constexpr(4):
        zeros[i] = cutlass.Float32(0.0)
    cute.autovec_copy(zeros, vec_at(ptr, idx, 4))


def state_bf16_idx(v, k):
    """BF16 element index of external state ``[V, K]`` element ``(v, k)``.

    ``v`` is the CTA-global value in ``[0, 128)``: the two
    16 KiB halves are main slots 0 and 1, which are adjacent, so one index
    function spans both.  The unswizzled address is ``v * 128 + k`` -- physical
    ``[V, K]`` row-major and logical ``[K, V]`` column-major at once, which is
    what lets an external ``[V, K]`` state land by TMA with no transpose and
    still be read as ``H[K, V]``.
    """
    segment = k // BF16_SEGMENT_ELEMS
    local = k - segment * BF16_SEGMENT_ELEMS
    line = STATE_BF16_ROWS_PER_VALUE * v + segment
    group = local // BF16_GROUP_ELEMS
    inner = local - group * BF16_GROUP_ELEMS
    return line * BF16_SEGMENT_ELEMS + (group ^ (line & 7)) * BF16_GROUP_ELEMS + inner


def state_x2_ptr(lane, kb, value_base):
    """Boundary-state ``ldmatrix.x2.trans`` / ``stmatrix.x2.trans`` map.

    The state image is physically ``[V, K]``; ``.trans`` turns those 16 rows
    into the ``[K, V]`` C tile the registers hold, so the prologue load and the
    epilogue store use one map and differ only in instruction direction.  Lanes
    16-31 are ignored by an x2 copy.
    """
    matrix = (lane // 8) & 1
    v = value_base + (lane - (lane // 8) * 8)
    key = BT * kb + 8 * matrix
    return state_bf16_idx(v, key)


def vo_x2_ptr(lane, value_base):
    """V ``ldmatrix.x2`` / O ``stmatrix.x2`` over a ``[16, 8]`` value block.

    Non-transposed in both directions: the stage is token-major and the C
    tile's rows are tokens, so the load and the store share this map.
    """
    matrix = (lane // 8) & 1
    token = (lane - (lane // 8) * 8) + 8 * matrix
    return raw_bf16_s128(token, value_base)


def pairwise_sw32(row, col):
    """BF16 element index of a ``[16, 16]`` pairwise tile.

    Carries ``AinvBeta`` and ``Aq``.  The ``col ^ 8`` term is a coordinate
    permutation applied before the SW32 swizzle.
    """
    storage_col = col ^ PAIRWISE_COL_XOR
    byte0 = 2 * (PAIRWISE_ROW_STRIDE * row + storage_col)
    return (byte0 ^ (((byte0 >> 7) & 1) << 4)) // 2


@cute.jit
def clear_tail_rows(
    p_q,
    p_k,
    p_g_raw,
    valid_rows,
    tidx,
    G_FP32: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """Zero the invalid rows of the raw stages.

    TMA only zero-fills coordinates outside the *tensor*, and a short chunk
    sits mid-tensor: the rows past ``valid_rows`` hold the next sequence's
    tokens, so the copy faithfully loaded real data there.  The task map
    matches the loads' 16-byte width so the writes stay one vector wide.
    """
    for rep in cutlass.range_constexpr(2):
        task = tidx + rep * threads
        row = task // BT
        d0 = (task - row * BT) * 8
        if row >= valid_rows:
            idx = raw_bf16_s128(row, d0)
            zero_vec8_bf16(p_q, idx)
            zero_vec8_bf16(p_k, idx)

    if cutlass.const_expr(G_FP32):
        for rep in cutlass.range_constexpr(4):
            task = tidx + rep * threads
            row = task // 32
            d0 = (task - row * 32) * 4
            if row >= valid_rows:
                zero_vec4_f32(p_g_raw, raw_f32_s128(row, d0))
    else:
        for rep in cutlass.range_constexpr(2):
            task = tidx + rep * threads
            row = task // BT
            d0 = (task - row * BT) * 8
            if row >= valid_rows:
                zero_vec8_bf16(p_g_raw, raw_bf16_s128(row, d0))
