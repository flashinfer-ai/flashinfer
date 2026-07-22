"""Inline-PTX wrappers (TMA 1D load/store, fns.b32, raw-int64 peer ops) for the cuTeDSL kernels."""

from typing import Optional

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
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
def tma_load_1d_raw(
    dst_smem, src_gmem_addr: Int64, mbar_smem, num_bytes, *, loc=None, ip=None
):
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
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
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
            v0.ir_value(),
            v1.ir_value(),
            v2.ir_value(),
            v3.ir_value(),
            v4.ir_value(),
            v5.ir_value(),
            v6.ir_value(),
            v7.ir_value(),
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
def red_async_add_release_sys_u32_raw(
    addr: Int64, val: Int32, *, loc=None, ip=None
) -> None:
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
    Fence operation with acquire-release semantics at system scope.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    llvm.fence(llvm.AtomicOrdering.release, syncscope="device", loc=loc, ip=ip)
