"""Inline-PTX wrappers (TMA 1D load/store, fns.b32, raw-int64 peer ops) for the cuTeDSL kernels."""

from typing import Optional, Tuple

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cutlass_dsl import Float32, Int32, Int64


# CUTLASS ``cute::TMA::CacheHintSm100`` policy descriptors
# (`copy_sm90_desc.hpp:193-196`). At large per-launch transfer sizes
# (e.g. 1+ GB/launch for T=32k batches) the L2 (~128 MB on B200/GB200) is
# heavily over-subscribed, so cache hints matter. mega_moe defaults TMA
# loads to EVICT_FIRST (single-use peer data) and TMA stores to
# EVICT_NORMAL (just-pulled data will be re-read shortly by the GEMM
# consumer). cuTeDSL was passing hint=0 (undefined policy) on loads and
# omitting the cache_hint operand entirely on stores -- both lead to L2
# pollution at large batch.
_TMA_CACHE_HINT_EVICT_NORMAL = 0x1000000000000000
_TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000


@dsl_user_op
def tma_load_1d(dst_smem, src_gmem, mbar_smem, num_bytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_load_1d_raw(dst_smem, src_gmem_addr: Int64, mbar_smem, num_bytes, *, loc=None, ip=None):
    """Variant of ``tma_load_1d`` that takes a raw int64 GMEM byte address.

    Used for cross-rank TMA load via ``peer_rank_ptr_mapper.map`` style: source
    address is computed dynamically as ``peer_rank_ptr_mapper.map(local_iter.toint(),
    dst_rank, element_offset)``, bypassing per-tensor constexpr fanout.
    """
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_gmem_addr.ir_value(),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_1d(dst_gmem, src_smem, num_bytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(),
            src_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
            Int64(_TMA_CACHE_HINT_EVICT_NORMAL).ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
        "[$0], [$1], $2, $3;",
        "l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_s2g(dst_gmem, src_smem, size_bytes, *, loc=None, ip=None) -> None:
    """Issue descriptor-free ``cp.async.bulk`` SMEM->GMEM.

    The caller owns ``cp_async_bulk_commit_group`` so copy and reduce bulk
    paths can share the same group boundary.
    """
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_reduce_async_bulk_add_noftz_bf16_s2g(
    dst_gmem,
    src_smem,
    size_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue descriptor-free ``cp.reduce.async.bulk`` for BF16 add."""
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def max_abs_bf16x2(lhs: Int32, rhs: Int32, *, loc=None, ip=None) -> Int32:
    """Per-lane abs-max of two packed bf16x2 words (``max.xorsign.abs.bf16x2``).

    Returns ``max(|lhs|, |rhs|)`` for each of the two bf16 lanes. The result's
    sign per lane is the xor of the input signs (junk), so callers mask 0x7FFF
    before reading a lane back. Stays in bf16 -- no fp32 convert needed to find
    an amax.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [lhs.ir_value(), rhs.ir_value()],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fns_b32(mask: Int32, base: Int32, n: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [mask.ir_value(), base.ir_value(), n.ir_value()],
            "fns.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unpack_bf16x2_to_f32x2(
    packed: Int32, *, loc=None, ip=None,
) -> Tuple[Float32, Float32]:
    """Bitcast packed BF16x2 bits and widen both lanes to FP32."""
    bf16_pair_type = ir.Type.parse("vector<2xbf16>")
    bf16_pair = llvm.bitcast(
        bf16_pair_type, packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip,
    )
    lo = vector.extract(bf16_pair, [], [0], loc=loc, ip=ip)
    hi = vector.extract(bf16_pair, [], [1], loc=loc, ip=ip)
    return (
        Float32(arith.extf(Float32.mlir_type, lo, loc=loc, ip=ip)),
        Float32(arith.extf(Float32.mlir_type, hi, loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f32x2_to_bf16x2(
    lo: Float32, hi: Float32, *, loc=None, ip=None,
) -> Int32:
    """Round two FP32 lanes to one packed BF16x2 word."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            # PTX places its first FP32 source in the high 16 bits and the
            # second in the low 16 bits.  Global memory is little-endian, so
            # reverse the operands to store logical [lo, hi] in address order.
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


# -----------------------------------------------------------------------------
# Raw-int64-pointer GMEM ops for the cross-rank ``peer_rank_ptr_mapper.map`` pattern.
# -----------------------------------------------------------------------------
# Each helper takes a 64-bit byte address (local iterator's ``.toint()`` +
# peer-rank offset + element offset) and emits the corresponding PTX op
# without requiring a cute.Tensor / iterator wrap. Mirrors mega_moe's pattern
# of computing ``peer_rank_ptr_mapper.map(local_ptr, dst_rank)`` and dereferencing it
# directly (sym_buffer.cuh:34-37).


@dsl_user_op
def ldg_b32_raw(addr: Int64, *, loc=None, ip=None) -> Int32:
    """``ld.global.u32`` via raw int64 byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value()],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_b64_raw(addr: Int64, *, loc=None, ip=None) -> Int64:
    """``ld.global.u64`` via raw int64 byte address."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [addr.ir_value()],
            "ld.global.u64 $0, [$1];",
            "=l,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_b128_raw(addr: Int64, *, loc=None, ip=None):
    """Diagnostic logical-16B load lowered as two proven scalar b64 ops."""
    packed_lo = ldg_b64_raw(addr, loc=loc, ip=ip)
    packed_hi = ldg_b64_raw(addr + Int64(8), loc=loc, ip=ip)
    return (
        Int32(packed_lo & Int64(0xFFFFFFFF)),
        Int32((packed_lo >> Int64(32)) & Int64(0xFFFFFFFF)),
        Int32(packed_hi & Int64(0xFFFFFFFF)),
        Int32((packed_hi >> Int64(32)) & Int64(0xFFFFFFFF)),
    )

@dsl_user_op
def ldg_f32_raw(addr: Int64, *, loc=None, ip=None) -> Float32:
    """``ld.global.f32`` via raw int64 byte address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [addr.ir_value()],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stg_b32_raw(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    """``st.global.u32`` via raw int64 byte address."""
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b64_raw(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """``st.global.u64`` via raw int64 byte address."""
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b128_raw(
    addr: Int64,
    val_0: Int32,
    val_1: Int32,
    val_2: Int32,
    val_3: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Diagnostic logical-16B store lowered as two proven scalar b64 ops."""
    packed_lo = (
        Int64(val_1) << Int64(32)
    ) | (Int64(val_0) & Int64(0xFFFFFFFF))
    packed_hi = (
        Int64(val_3) << Int64(32)
    ) | (Int64(val_2) & Int64(0xFFFFFFFF))
    stg_b64_raw(addr, packed_lo, loc=loc, ip=ip)
    stg_b64_raw(
        addr + Int64(8), packed_hi, loc=loc, ip=ip,
    )


@dsl_user_op
def stg_e8m0_from_f32(addr: Int64, fp32_val: Float32, *, loc=None, ip=None) -> None:
    """Convert ``fp32_val`` to E8M0 via PTX and store the 1-byte result to global memory.

    Uses ``cvt.rp.satfinite.ue8m0x2.f32`` which is the correct PTX path for
    fp32 → E8M0.  Avoids CuTe DSL's generic ``.to(Float8E8M0FNU)`` which does
    not lower correctly for the non-IEEE-754 E8M0 type.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), fp32_val.ir_value()],
        "{\n"
        "  .reg .b16 bf_lo;\n"
        "  .reg .u32 tmp;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 bf_lo, 0f00000000, $1;\n"
        "  cvt.u32.u16 tmp, bf_lo;\n"
        "  st.global.b8 [$0], tmp;\n"
        "}",
        "l,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_e8m0x8_from_f32(
    addr: Int64,
    v0: Float32, v1: Float32, v2: Float32, v3: Float32,
    v4: Float32, v5: Float32, v6: Float32, v7: Float32,
    *, loc=None, ip=None,
) -> None:
    """Convert 8 fp32 values to E8M0 and store them as 8 contiguous bytes in one shot.

    Batched form of ``stg_e8m0_from_f32``: four ``cvt.rp.satfinite.ue8m0x2.f32``
    each pack two E8M0 bytes (``d.lo = cvt(b)``, ``d.hi = cvt(a)``), the four
    ``.b16`` results are assembled into two ``.b32`` words, and a single
    ``st.global.v2.u32`` writes all 8 bytes.  Halves the cvt count and turns 8
    one-byte stores into one 64-bit store.  ``addr`` must be 8-byte aligned;
    output byte ``k`` holds E8M0(``v{k}``).
    """
    llvm.inline_asm(
        None,
        [
            addr.ir_value(),
            v0.ir_value(), v1.ir_value(), v2.ir_value(), v3.ir_value(),
            v4.ir_value(), v5.ir_value(), v6.ir_value(), v7.ir_value(),
        ],
        "{\n"
        "  .reg .b16 p0, p1, p2, p3;\n"
        "  .reg .b32 w0, w1;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p0, $2, $1;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p1, $4, $3;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p2, $6, $5;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p3, $8, $7;\n"
        "  mov.b32 w0, {p0, p1};\n"
        "  mov.b32 w1, {p2, p3};\n"
        "  st.global.v2.u32 [$0], {w0, w1};\n"
        "}",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_sys_u64_raw(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """``red.release.sys.global.add.u64`` via raw int64 byte address.

    Fire-and-forget atomic add (no return value) -- mega_moe uses this
    pattern for ``expert_recv_count_sum`` cross-rank publish where the
    fetched-old is unused (sm100_fp8_fp4_mega_moe.cuh:511-513).
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.release.sys.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_u64_raw(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """``red.relaxed.sys.global.add.u64`` via raw int64 byte address.

    Relaxed (no release fence) sibling of ``red_add_release_sys_u64_raw``.  Use
    when an enclosing ``bar.sync`` + a later release fence (e.g. the trailing
    ``nvlink_barrier`` publish) already provides cross-rank ordering, so the
    per-op release fence would just emit a redundant ``membar.sys`` drain.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.relaxed.sys.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_s32_raw(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    """``red.relaxed.sys.global.add.s32`` via raw int64 byte address.

    Relaxed sibling of ``red_add_release_sys_s32_raw`` for bulk peer
    redistribution: the caller publishes the batch once with a trailing
    ``fence.acq_rel.sys`` plus the enclosing barrier instead of paying one
    ``membar.sys`` drain per element.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.relaxed.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_sys_s32_raw(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    """``red.release.sys.global.add.s32`` via raw int64 byte address.

    Used for the NVLink barrier signal cross-rank fan-out (matches
    mega_moe ``ptx::red_add_rel_sys`` at barrier.cuh:50).
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.release.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_f32(address, value: Float32, *, loc=None, ip=None) -> None:
    """System-scope relaxed reduction add for a remote float32 destination."""
    llvm.inline_asm(
        None,
        [address.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        "red.relaxed.sys.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_v2_bf16x2(
    addr,
    val0_packed_bf16x2,
    val1_packed_bf16x2,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.relaxed.sys.global.add.v2.bf16x2 [addr], {v0, v1};``."""
    llvm.inline_asm(
        None,
        [
            addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            val0_packed_bf16x2.ir_value(loc=loc, ip=ip),
            val1_packed_bf16x2.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.sys.global.add.noftz.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_gpu_s32(
    counter_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.release.gpu.global.add.s32`` to a GMEM int32 location."""
    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_min_relaxed_gpu_u32_from_f32_raw(
    addr: Int64, value: Float32, *, loc=None, ip=None
) -> None:
    """``red.relaxed.gpu.global.min.u32`` on the bit pattern of a non-negative f32.

    PTX has no float min-reduce; for non-negative floats the u32 bit order
    matches the float order, so a u32 min on the reinterpreted bits realises a
    float min. The target slot must be seeded with a large positive sentinel
    (e.g. ``Fp32Max``) and only fed non-negative candidates. Relaxed/GPU scope
    suffices: a kernel boundary separates the reduction from its reader.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), value.ir_value()],
        "{ .reg .b32 t; mov.b32 t, $1; red.relaxed.gpu.global.min.u32 [$0], t; }",
        "l,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_add_release_sys_u32_raw(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    """``red.async.release.sys.global.add.u32`` via raw int64 byte address.

    sm_90+ async reduction — fire-and-forget; the issuing SM does NOT
    wait for the L2/HBM round-trip. Same architectural release/sys
    semantics as the synchronous form, but the SM can continue issuing
    instructions immediately. Used in cuTeDSL dispatch_pull V2 to
    replace the synchronous ``atom.release.gpu.global.add.u32`` for
    l1_arrival_count, eliminating the per-token atomic-wait stall.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.async.release.sys.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def read_clock64(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the GPU-wide nanosecond timer, comparable across SMs on one GPU."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def device_trap(*, loc=None, ip=None) -> None:
    """Abort the current kernel after a fail-closed device-side check."""
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_rel_sys(
    *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> None:
    """
    Fence operation with acquire-release semantics at system scope.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    llvm.fence(llvm.AtomicOrdering.release, loc=loc, ip=ip)

@dsl_user_op
def _fence_rel_gpu(
    *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> None:
    """
    Fence operation with acquire-release semantics at device (gpu) scope.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    llvm.fence(llvm.AtomicOrdering.release, syncscope="device", loc=loc, ip=ip)


@dsl_user_op
def copy_b128_raw(
    dst_addr: Int64, src_addr: Int64, *, loc=None, ip=None
) -> None:
    """One 128-bit global load plus store without an SSA result aggregate."""
    llvm.inline_asm(
        None,
        [dst_addr.ir_value(), src_addr.ir_value()],
        "{\n\t.reg .b64 lo, hi;\n"
        "\tld.global.v2.b64 {lo, hi}, [$1];\n"
        "\tst.global.v2.b64 [$0], {lo, hi};\n}",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_route_experts_b128_raw(
    addr: Int64, *, loc=None, ip=None
) -> Int64:
    """Load four route words once and return their low 16-bit expert IDs."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [addr.ir_value()],
            "{\n\t.reg .b32 r0, r1, r2, r3;\n"
            "\t.reg .b64 lo, hi;\n"
            "\tld.global.v4.u32 {r0, r1, r2, r3}, [$1];\n"
            "\tand.b32 r0, r0, 0xffff;\n"
            "\tand.b32 r1, r1, 0xffff;\n"
            "\tand.b32 r2, r2, 0xffff;\n"
            "\tand.b32 r3, r3, 0xffff;\n"
            "\tshl.b32 r1, r1, 16;\n"
            "\tshl.b32 r3, r3, 16;\n"
            "\tor.b32 r0, r0, r1;\n"
            "\tor.b32 r2, r2, r3;\n"
            "\tcvt.u64.u32 lo, r0;\n"
            "\tcvt.u64.u32 hi, r2;\n"
            "\tshl.b64 hi, hi, 32;\n"
            "\tor.b64 $0, lo, hi;\n}",
            "=l,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_route_match_b128_raw(
    addr: Int64, target_expert: Int32, *, loc=None, ip=None
) -> Int32:
    """Load four routes once; return the last matching top-k slot or -1."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(), target_expert.ir_value()],
            "{\n\t.reg .b32 r0, r1, r2, r3, tmp, result;\n"
            "\t.reg .pred p;\n"
            "\tld.global.v4.u32 {r0, r1, r2, r3}, [$1];\n"
            "\tmov.u32 result, 0xffffffff;\n"
            "\tand.b32 tmp, r0, 0xffff;\n"
            "\tsetp.eq.u32 p, tmp, $2;\n"
            "\tshr.u32 tmp, r0, 16;\n"
            "\tand.b32 tmp, tmp, 0xff;\n"
            "\t@p mov.b32 result, tmp;\n"
            "\tand.b32 tmp, r1, 0xffff;\n"
            "\tsetp.eq.u32 p, tmp, $2;\n"
            "\tshr.u32 tmp, r1, 16;\n"
            "\tand.b32 tmp, tmp, 0xff;\n"
            "\t@p mov.b32 result, tmp;\n"
            "\tand.b32 tmp, r2, 0xffff;\n"
            "\tsetp.eq.u32 p, tmp, $2;\n"
            "\tshr.u32 tmp, r2, 16;\n"
            "\tand.b32 tmp, tmp, 0xff;\n"
            "\t@p mov.b32 result, tmp;\n"
            "\tand.b32 tmp, r3, 0xffff;\n"
            "\tsetp.eq.u32 p, tmp, $2;\n"
            "\tshr.u32 tmp, r3, 16;\n"
            "\tand.b32 tmp, tmp, 0xff;\n"
            "\t@p mov.b32 result, tmp;\n"
            "\tmov.b32 $0, result;\n}",
            "=r,l,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def copy_b128x4_strided_raw(
    dst_addr: Int64,
    src_addr: Int64,
    count: Int32,
    *,
    stride_bytes: int,
    loc=None,
    ip=None,
) -> None:
    """Copy up to four 16-byte vectors at ``k * stride_bytes`` (``k < count``).

    All predicated loads are issued before the first store, so one warp moves
    four 512-byte rows per dependent round trip instead of one.
    """
    s1, s2, s3 = stride_bytes, 2 * stride_bytes, 3 * stride_bytes
    asm = f"{{\n\t.reg .b64 lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;\n\t.reg .pred p0, p1, p2, p3;\n\tsetp.gt.s32 p0, $2, 0;\n\tsetp.gt.s32 p1, $2, 1;\n\tsetp.gt.s32 p2, $2, 2;\n\tsetp.gt.s32 p3, $2, 3;\n\t@p0 ld.global.v2.b64 {{lo0, hi0}}, [$1];\n\t@p1 ld.global.v2.b64 {{lo1, hi1}}, [$1+{s1}];\n\t@p2 ld.global.v2.b64 {{lo2, hi2}}, [$1+{s2}];\n\t@p3 ld.global.v2.b64 {{lo3, hi3}}, [$1+{s3}];\n\t@p0 st.global.v2.b64 [$0], {{lo0, hi0}};\n\t@p1 st.global.v2.b64 [$0+{s1}], {{lo1, hi1}};\n\t@p2 st.global.v2.b64 [$0+{s2}], {{lo2, hi2}};\n\t@p3 st.global.v2.b64 [$0+{s3}], {{lo3, hi3}};\n}}"
    llvm.inline_asm(
        None,
        [dst_addr.ir_value(), src_addr.ir_value(), count.ir_value()],
        asm,
        "l,l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_b64_predicated_raw(
    addr: Int64, valid: Int32, *, loc=None, ip=None
) -> Int64:
    """``ld.global.u64`` when ``valid != 0``, else 0, without a branch."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [addr.ir_value(), valid.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n"
            "\tmov.u64 $0, 0;\n\t@p ld.global.u64 $0, [$1];\n}",
            "=l,l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_b32_predicated_raw(
    addr: Int64, valid: Int32, *, loc=None, ip=None
) -> Int32:
    """``ld.global.u32`` when ``valid != 0``, else 0, without a branch."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(), valid.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n"
            "\tmov.u32 $0, 0;\n\t@p ld.global.u32 $0, [$1];\n}",
            "=r,l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cp_async_16_smem_raw(
    dst_smem_addr: Int32, src_addr: Int64, valid: Int32, *, loc=None, ip=None
) -> None:
    """One 16-byte ``cp.async.cg`` global->shared copy.

    ``valid == 0`` reads nothing and zero-fills the 16 destination bytes
    (``src-size`` 0). Completion is tracked per issuing thread through
    ``cp_async_commit_group_raw`` / ``cp_async_wait_group_raw``; only that
    thread may read the landed bytes.
    """
    llvm.inline_asm(
        None,
        [dst_smem_addr.ir_value(), src_addr.ir_value(), valid.ir_value()],
        "{\n\t.reg .pred p;\n\t.reg .b32 sz;\n"
        "\tsetp.ne.s32 p, $2, 0;\n"
        "\tselp.b32 sz, 16, 0, p;\n"
        "\tcp.async.cg.shared.global [$0], [$1], 16, sz;\n}",
        "r,l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_commit_group_raw(*, loc=None, ip=None) -> None:
    """Close the issuing thread's current ``cp.async`` group."""
    llvm.inline_asm(
        None, [], "cp.async.commit_group;", "",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def cp_async_wait_group_raw(pending: int, *, loc=None, ip=None) -> None:
    """Wait until at most ``pending`` of this thread's ``cp.async`` groups are incomplete."""
    llvm.inline_asm(
        None, [], f"cp.async.wait_group {int(pending)};", "",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def fence_proxy_async_shared_cta_raw(*, loc=None, ip=None) -> None:
    """Order async-proxy (TMA, UMMA) and generic-proxy accesses to CTA shared memory."""
    llvm.inline_asm(
        None, [], "fence.proxy.async.shared::cta;", "",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def st_shared_b32_raw(addr: Int32, val: Int32, *, loc=None, ip=None) -> None:
    """Plain 32-bit shared store through a shared-window address."""
    llvm.inline_asm(
        None, [addr.ir_value(), val.ir_value()], "st.shared.b32 [$0], $1;", "r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def st_shared_release_cta_b32_raw(addr: Int32, val: Int32, *, loc=None, ip=None) -> None:
    """CTA-scope release store of a 32-bit shared flag."""
    llvm.inline_asm(
        None, [addr.ir_value(), val.ir_value()], "st.release.cta.shared.b32 [$0], $1;", "r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def ld_shared_acquire_cta_b32_raw(addr: Int32, *, loc=None, ip=None) -> Int32:
    """CTA-scope acquire load of a 32-bit shared flag."""
    return Int32(
        llvm.inline_asm(
            T.i32(), [addr.ir_value()], "ld.acquire.cta.shared.b32 $0, [$1];", "=r,r",
            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def lds_b32_addr_raw(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Ordered 32-bit shared load through a shared-window address."""
    return Int32(
        llvm.inline_asm(
            T.i32(), [addr.ir_value()], "ld.shared.u32 $0, [$1];", "=r,r",
            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def ldg_b128_to_smem_raw(
    dst_smem, src_addr: Int64, *, loc=None, ip=None
) -> None:
    """Load one global 16B vector into thread-private shared scratch."""
    llvm.inline_asm(
        None,
        [dst_smem.toint(loc=loc, ip=ip).ir_value(), src_addr.ir_value()],
        "{\n\t.reg .b64 lo, hi;\n"
        "\tld.global.v2.b64 {lo, hi}, [$1];\n"
        "\tst.shared.v2.b64 [$0], {lo, hi};\n}",
        "r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_b128x4_to_smem_predicated_raw(
    dst_smem,
    src_addr_0: Int64,
    src_addr_1: Int64,
    src_addr_2: Int64,
    src_addr_3: Int64,
    valid_0: Int32,
    valid_1: Int32,
    valid_2: Int32,
    valid_3: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue up to four predicated global 16B loads before shared stores."""
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            src_addr_0.ir_value(),
            src_addr_1.ir_value(),
            src_addr_2.ir_value(),
            src_addr_3.ir_value(),
            valid_0.ir_value(),
            valid_1.ir_value(),
            valid_2.ir_value(),
            valid_3.ir_value(),
        ],
        "{\n\t.reg .b64 lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;\n"
        "\t.reg .pred p0, p1, p2, p3;\n"
        "\tsetp.ne.s32 p0, $5, 0;\n"
        "\tsetp.ne.s32 p1, $6, 0;\n"
        "\tsetp.ne.s32 p2, $7, 0;\n"
        "\tsetp.ne.s32 p3, $8, 0;\n"
        "\t@p0 ld.global.v2.b64 {lo0, hi0}, [$1];\n"
        "\t@p1 ld.global.v2.b64 {lo1, hi1}, [$2];\n"
        "\t@p2 ld.global.v2.b64 {lo2, hi2}, [$3];\n"
        "\t@p3 ld.global.v2.b64 {lo3, hi3}, [$4];\n"
        "\t@p0 st.shared.v2.b64 [$0], {lo0, hi0};\n"
        "\t@p1 st.shared.v2.b64 [$0+16], {lo1, hi1};\n"
        "\t@p2 st.shared.v2.b64 [$0+32], {lo2, hi2};\n"
        "\t@p3 st.shared.v2.b64 [$0+48], {lo3, hi3};\n}",
        "r,l,l,l,l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldsm_b32_raw(src_smem, *, loc=None, ip=None) -> Int32:
    """Ordered scalar read from thread-private shared scratch."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [src_smem.toint(loc=loc, ip=ip).ir_value()],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stg_b128_words_raw(
    addr: Int64,
    val_0: Int32,
    val_1: Int32,
    val_2: Int32,
    val_3: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Pack four scalar words inside PTX and issue one global 16B store."""
    llvm.inline_asm(
        None,
        [
            addr.ir_value(),
            val_0.ir_value(),
            val_1.ir_value(),
            val_2.ir_value(),
            val_3.ir_value(),
        ],
        "{\n\t.reg .b64 lo, hi, tmp;\n"
        "\tcvt.u64.u32 lo, $1;\n"
        "\tcvt.u64.u32 tmp, $2;\n"
        "\tshl.b64 tmp, tmp, 32;\n"
        "\tor.b64 lo, lo, tmp;\n"
        "\tcvt.u64.u32 hi, $3;\n"
        "\tcvt.u64.u32 tmp, $4;\n"
        "\tshl.b64 tmp, tmp, 32;\n"
        "\tor.b64 hi, hi, tmp;\n"
        "\tst.global.v2.b64 [$0], {lo, hi};\n}",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
