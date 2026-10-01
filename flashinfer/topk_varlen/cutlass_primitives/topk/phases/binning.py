"""The survivor bin: one function, used by the filter pass and by the emit.

A survivor with value ``v`` above the walk threshold ``bar`` lands in bin ``(v - bar) *
scale`` truncated to [0, 255], NaN in bin 255 (NaN ranks on top, as torch does).  The filter
pass builds the survivor histogram with this function and the emit classifies each staged
candidate with it; the rank-k bin found on the histogram is only meaningful if both agree
bit for bit, so there is exactly one implementation, an inline PTX sequence whose rounding
cannot drift between call sites.
"""

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["survivor_bin"]

_PTX = (
    "{\n.reg .f32 d, m;\n.reg .pred q;\n"
    "sub.rn.f32 d, $1, $2;\n"
    "mul.rn.f32 m, d, $3;\n"
    "cvt.rzi.u32.f32 $0, m;\n"  # saturating: negatives and small values -> 0
    "min.u32 $0, $0, 255;\n"
    "setp.neu.f32 q, $1, $1;\n"  # NaN
    "@q mov.u32 $0, 255;\n"
    "}"
)


@dsl_user_op
def survivor_bin(
    value: cutlass.Float32,
    bar: cutlass.Float32,
    scale: cutlass.Float32,
    *,
    loc=None,
    ip=None,
):
    """Bin in [0, 255] of ``value`` relative to ``bar`` with ``scale`` bins per unit; NaN -> 255.

    Pure; 6 instructions.  Monotone in ``value`` for non-NaN inputs.
    """
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            cutlass.Float32(bar).ir_value(loc=loc, ip=ip),
            cutlass.Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        _PTX,
        "=r,f,f,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)
