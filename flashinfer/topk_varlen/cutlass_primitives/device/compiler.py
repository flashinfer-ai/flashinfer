"""Opaque register pins: identity moves the optimizer cannot see through.

These change no value.  They exist because NVVM and ptxas sometimes recompute a cheap
expression at every use instead of keeping it in a register, and inside a divergent loop that
recomputation is a measurable cost.  An inline-asm ``mov`` is a value the optimizer must treat
as opaque, so it is materialized exactly once and kept.  Use only where SASS shows the
rematerialization; the docstrings name the case each one fixed.
"""

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["pin_i32", "pin_i64", "pin_shared_address"]


def _identity(asm, out_type, wrap, val_ir, constraint, loc, ip):
    return wrap(
        llvm.inline_asm(
            out_type,
            [val_ir],
            asm,
            f"={constraint},{constraint}",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pin_i64(val: cutlass.Int64, *, loc=None, ip=None):
    """Return ``val`` unchanged through an opaque ``mov.b64``.

    Case: a row's global base address (parameter load + block index + multiply-add) was
    rematerialized inside every loop region of the streaming kernel.  Pinning it once at the
    top kept it in two registers.
    """
    return _identity(
        "mov.b64 $0, $1;",
        T.i64(),
        cutlass.Int64,
        cutlass.Int64(val).ir_value(loc=loc, ip=ip),
        "l",
        loc,
        ip,
    )


@dsl_user_op
def pin_i32(val: cutlass.Int32, *, loc=None, ip=None):
    """Return ``val`` unchanged through an opaque ``mov.b32``.  Int32 twin of ``pin_i64``."""
    return _identity(
        "mov.b32 $0, $1;",
        T.i32(),
        cutlass.Int32,
        cutlass.Int32(val).ir_value(loc=loc, ip=ip),
        "r",
        loc,
        ip,
    )


@dsl_user_op
def pin_shared_address(address: cutlass.Int32, *, loc=None, ip=None):
    """Return a shared byte ``address`` unchanged through an opaque ``mov.u32``.

    Case: the dynamic shared-memory base symbol was re-folded into every histogram increment
    inside the divergent emit loop (one extra move per survivor).  Pinned, the base is
    materialized once and each increment is a single address add.
    """
    return _identity(
        "mov.u32 $0, $1;",
        T.i32(),
        cutlass.Int32,
        cutlass.Int32(address).ir_value(loc=loc, ip=ip),
        "r",
        loc,
        ip,
    )
