"""On-device clocks for phase telemetry.

Telemetry is how every optimization in this library is explained: a kernel records the
clock at phase boundaries and the host turns deltas into a per-phase table.  The rule from
the DSL section of the conventions applies: keep one running mark and write each delta as
soon as it is known, because unused timestamps still cost registers.
"""

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["read_clock64", "read_globaltimer"]


@dsl_user_op
def read_clock64(*, loc=None, ip=None):
    """The SM's 64-bit cycle counter (``%clock64``) as Int64.

    Cycle resolution; the right clock for phases inside one CTA (a barrier is ~250 ns, well
    below the global timer's tick).  Not comparable across SMs: each SM counts on its own.
    Has side effects so the compiler cannot move it across the code being timed.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None):
    """The device-wide nanosecond timer (``%globaltimer``) as Int64.

    Comparable across SMs, so it is the clock for launch-to-first-instruction and
    across-CTA skew.  Ticks coarsely (about 0.3 to 1 us on B200 and L40S), too coarse for
    phases inside a CTA.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
