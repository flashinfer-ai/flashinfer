"""SM120-only raw-address PTX helpers."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op


# Implemented by nvshmem_compat.cu and linked beside the validated NVSHMEM
# device bitcode. Keeping the field access in a header-compiled translation
# unit means an NVSHMEM layout change fails its static assertions instead of
# silently changing an immediate offset in inline PTX.
nvshmem_disable_p2p_peer = cute.ffi(
    name="megamoe_nvshmem_disable_p2p_peer",
    params_types=[cutlass.Int32],
)


@dsl_user_op
def ldg_b32_cv_raw(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Load a u32 with ``.cv`` so peer/RDMA writes cannot be hidden by L2."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "ld.global.cv.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def spin_wait_i32_ge_cv_raw(
    addr: Int64,
    threshold: Int32,
    sleep_cycles: int = 500,
    *,
    loc=None,
    ip=None,
) -> None:
    """Poll a peer/RDMA-updated word using cache-volatile global loads."""
    llvm.inline_asm(
        None,
        [
            addr.ir_value(loc=loc, ip=ip),
            threshold.ir_value(loc=loc, ip=ip),
            Int32(sleep_cycles).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .b32 %cur; .reg .pred %ready;\n\t"
            "WAIT_CV: \n\t"
            "ld.global.cv.u32 %cur, [$0];\n\t"
            "setp.ge.u32 %ready, %cur, $1;\n\t"
            "@%ready bra READY_CV;\n\t"
            "nanosleep.u32 $2;\n\t"
            "bra WAIT_CV;\n\t"
            "READY_CV: \n\t"
            "}"
        ),
        "l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_b32_raw(addr, *, loc=None, ip=None) -> Int32:
    """Load one shared-memory u32 through a raw shared address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_add_relaxed_sys_v2_bf16x2_raw(
    addr: Int64,
    val0_packed_bf16x2,
    val1_packed_bf16x2,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue a system-scope bf16x2 vector reduction via a raw address."""
    llvm.inline_asm(
        None,
        [
            addr.ir_value(loc=loc, ip=ip),
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
