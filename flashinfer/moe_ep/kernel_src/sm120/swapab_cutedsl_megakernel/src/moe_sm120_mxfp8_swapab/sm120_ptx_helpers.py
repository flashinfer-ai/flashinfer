"""SM120-only raw-address PTX helpers."""

from typing import Optional

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op


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
