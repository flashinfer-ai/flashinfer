"""The declared fp8 conversion primitive, shared by the kernels that quantize to e4m3 / e5m2:
``cvt.rn.satfinite.{e4m3,e5m2}x2.f32`` (RNE, saturating to the largest finite value; assumption A2 of
docs/requirements.md; exact CPU model ``flashinfer.batch_invariant.fp_interface.exact.to_fp8``). The row-norm kernel (``rlk/rmsnorm/kernel.py``)
carries the same inline-PTX primitive; it is kept there unchanged so that this file's existence does not
touch the row-norm cache closure (consolidating the two is a lead's refactor: it recompiles the row norms)."""

from __future__ import annotations

import cutlass
from cutlass import Float32, Uint16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def cvt_f32x2_to_fp8x2(
    a: Float32, b: Float32, kind: cutlass.Constexpr[str], *, loc=None, ip=None
) -> Uint16:
    """``cvt.rn.satfinite.{e4m3,e5m2}x2.f32``: a -> low byte, b -> high byte."""
    res = llvm.inline_asm(
        T.i16(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite.{kind}x2.f32 $0, $2, $1;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint16(res)
