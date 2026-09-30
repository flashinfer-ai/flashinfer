"""PTX intrinsics shared by the CuTe DSL PCIe collectives.

The communication kernels specify the scope and ordering of every protocol
record access. CUDA's default global-memory operations are not a cross-device
visibility contract; the explicit ``.sys`` operations below are. Keeping these
operations isolated also makes generated-code inspection deterministic while
the surrounding kernels remain in Python/CuTe DSL.
"""

from __future__ import annotations

from typing import Tuple

from cutlass import Float32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def f32_as_u32(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Bitcast float32 to uint32 without numeric conversion."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_u64(addr: Int64, *, loc=None, ip=None) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_nc_u64(addr: Int64, *, loc=None, ip=None) -> Uint64:
    """Read an immutable device table entry through the read-only cache."""

    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_generic_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """Load an f32 through a generic/UVA address."""

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_relaxed_sys_global_v4_u32(
    addr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load one NCCL-style PCIe data-and-epoch line at system scope."""

    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.relaxed.sys.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def st_relaxed_sys_global_v4_u32(
    addr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one NCCL-style PCIe data-and-epoch line at system scope."""

    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def add_f16x2(left: Uint32, right: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two packed FP16 lanes with round-to-nearest semantics."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(left).ir_value(loc=loc, ip=ip),
                Uint32(right).ir_value(loc=loc, ip=ip),
            ],
            "add.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_bf16x2(left: Uint32, right: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two packed BF16 lanes with round-to-nearest semantics."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(left).ir_value(loc=loc, ip=ip),
                Uint32(right).ir_value(loc=loc, ip=ip),
            ],
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_f32(addr: Int64, value: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def unpack_f16x2(value: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def unpack_bf16x2(value: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f32x2_to_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_f32x2_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Match two scalar ``__float2bfloat16`` conversions without saturation."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 blo, bhi;
                cvt.rn.bf16.f32 blo, $1;
                cvt.rn.bf16.f32 bhi, $2;
                mov.b32 $0, {blo, bhi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pcie_arrive_and_wait(
    self_counter_addr: Int64,
    peer_counter_slot0_addr: Int64,
    wait_counter_slot0_addr: Int64,
    slot_stride_bytes: Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """One lane of the sense-alternating, system-scope PCIe barrier."""
    llvm.inline_asm(
        None,
        [
            Int64(self_counter_addr).ir_value(loc=loc, ip=ip),
            Int64(peer_counter_slot0_addr).ir_value(loc=loc, ip=ip),
            Int64(wait_counter_slot0_addr).ir_value(loc=loc, ip=ip),
            Int64(slot_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred done;
            .reg .b32 val, seen, parity;
            .reg .b64 slot_off, peer_addr, wait_addr;
            fence.sc.sys;
            ld.global.u32 val, [$0];
            add.u32 val, val, 1;
            st.global.u32 [$0], val;
            and.b32 parity, val, 1;
            cvt.u64.u32 slot_off, parity;
            mul.lo.u64 slot_off, slot_off, $3;
            add.u64 peer_addr, $1, slot_off;
            add.u64 wait_addr, $2, slot_off;
            st.relaxed.sys.global.u32 [peer_addr], val;
        wait_again:
            ld.relaxed.sys.global.u32 seen, [wait_addr];
            setp.eq.u32 done, seen, val;
            @!done bra wait_again;
        }
        """,
        "l,l,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_relaxed_gpu_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_add_global_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def graph_epoch_arrive(
    epoch_addr: Int64,
    arrived_addr: Int64,
    blocks: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Advance a graph slot epoch after every CTA has observed it.

    Call this exactly once per CTA, after every thread that uses the current
    epoch has loaded it.  Unlike a software grid barrier, non-final CTAs do
    not spin: the final arrival resets the counter and advances the epoch.
    Same-stream kernel ordering then guarantees that the following collective
    cannot observe a partially completed transition.  The scheme is safe for
    grids larger than the resident CTA capacity and for changing grid sizes.
    """
    llvm.inline_asm(
        None,
        [
            Int64(epoch_addr).ir_value(loc=loc, ip=ip),
            Int64(arrived_addr).ir_value(loc=loc, ip=ip),
            Uint32(blocks).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred not_last;
            .reg .b32 prior, last, discarded;
            sub.u32 last, $2, 1;
            atom.global.add.u32 prior, [$1], 1;
            setp.ne.u32 not_last, prior, last;
            @not_last bra epoch_arrive_done;
            st.global.u32 [$1], 0;
            fence.sc.gpu;
            atom.global.add.u32 discarded, [$0], 1;
        epoch_arrive_done:
        }
        """,
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def graph_epoch_arrive_serialized(
    epoch_addr: Int64,
    arrived_addr: Int64,
    blocks: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Advance an epoch for execution-ordered kernels on one local GPU.

    Every CTA must call this exactly once after all of its threads have loaded
    the current epoch.  ``atom.inc`` resets the arrival count on the final CTA;
    that CTA is the sole epoch writer.  Kernel execution ordering publishes the
    plain epoch store before the next launch, so no payload fence is required.
    """

    llvm.inline_asm(
        None,
        [
            Int64(epoch_addr).ir_value(loc=loc, ip=ip),
            Int64(arrived_addr).ir_value(loc=loc, ip=ip),
            Uint32(blocks).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred not_last;
            .reg .b32 prior, last, generation;
            sub.u32 last, $2, 1;
            atom.relaxed.gpu.global.inc.u32 prior, [$1], last;
            setp.ne.u32 not_last, prior, last;
            @not_last bra serialized_epoch_arrive_done;
            ld.relaxed.gpu.global.u32 generation, [$0];
            add.u32 generation, generation, 1;
            st.global.u32 [$0], generation;
        serialized_epoch_arrive_done:
        }
        """,
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def plain_graph_epoch_arrive(
    epoch_addr: Int64,
    arrived_addr: Int64,
    blocks: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Advance a plain peer-push epoch, reserving wire tags zero and one.

    The final epoch's kernel drains peers and clears both incoming slots before
    returning. Pull and fused collectives use separate counters.
    """

    llvm.inline_asm(
        None,
        [
            Int64(epoch_addr).ir_value(loc=loc, ip=ip),
            Int64(arrived_addr).ir_value(loc=loc, ip=ip),
            Uint32(blocks).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred not_last, rollover;
            .reg .b32 prior, last, generation, next_generation;
            sub.u32 last, $2, 1;
            atom.relaxed.gpu.global.inc.u32 prior, [$1], last;
            setp.ne.u32 not_last, prior, last;
            @not_last bra plain_epoch_arrive_done;
            ld.relaxed.gpu.global.u32 generation, [$0];
            setp.eq.u32 rollover, generation, 0xfffffffd;
            add.u32 next_generation, generation, 1;
            selp.u32 next_generation, 0, next_generation, rollover;
            st.global.u32 [$0], next_generation;
        plain_epoch_arrive_done:
        }
        """,
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

@dsl_user_op
def threadfence_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.sc.gpu;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def spin_until_changed_acquire_gpu(
    addr: Int64, generation: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(generation).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred same;
            .reg .b32 seen;
        generation_wait:
            ld.acquire.gpu.global.u32 seen, [$0];
            setp.eq.u32 same, seen, $1;
            @same bra generation_wait;
        }
        """,
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def rsqrt_approx_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def float_order_key(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Map a float onto a u32 whose unsigned order matches float order.

    Lets a (score, expert) pair be packed into one integer key, so "better
    candidate" becomes a plain integer max and the tie-break stops costing a
    second comparison.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .u32 bits, mask;
                mov.b32 bits, $1;
                shr.u32 mask, bits, 31;
                mul.lo.u32 mask, mask, 0x7fffffff;
                or.b32 mask, mask, 0x80000000;
                xor.b32 $0, bits, mask;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def redux_max_u32(value: Uint32, *, loc=None, ip=None) -> Uint32:
    """Warp-wide unsigned max in one instruction.

    ``cute.arch.warp_redux_sync`` exposes the float variant, which libNVVM
    rejects for sm_120a; the integer one is what the kernel actually needs.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
