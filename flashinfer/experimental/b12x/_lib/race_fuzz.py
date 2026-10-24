"""Opt-in warp-delay instrumentation for b12x race tests.

When ``B12X_RACE_FUZZ_DELAY`` is set as a process starts (to the maximum
delay in nanoseconds, or ``1`` for 4096 ns), every kernel that process
compiles inserts a warp-uniform ``nanosleep`` after each
``cute.arch.sync_threads``, ``cute.arch.barrier`` and
``cute.arch.mbarrier_wait`` for about a quarter of (barrier instance, warp)
pairs, chosen from ``%globaltimer`` and ``%warpid``. Correctly synchronized
kernels only slow down; a kernel that relies on warps finishing in a
particular order (for example, publishing results through one thread before
the rest of its CTA has written them) sees that order change.

Any ``B12X_`` variable outside the compiler's operational set is part of the
compile cache key, so instrumented kernels never share cache entries with
normal ones, and spawned compile workers inherit the setting. The key's
environment part is fixed once a process computes it, so the variable must be
set before the first compile; changing it later has no effect.
"""

from __future__ import annotations

import os

ENV = "B12X_RACE_FUZZ_DELAY"
_installed_ns: int | None = None


def delay_ns() -> int | None:
    raw = os.environ.get(ENV, "")
    if raw in ("", "0"):
        return None
    value = 4096 if raw == "1" else int(raw)
    if value <= 0:
        raise ValueError(f"{ENV} must be a positive delay in nanoseconds")
    return value


def install_from_environment() -> None:
    """Wrap the CuTe synchronization primitives once if ``B12X_RACE_FUZZ_DELAY`` is set."""
    global _installed_ns
    value = delay_ns()
    if value is None or _installed_ns is not None:
        return
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import dsl_user_op

    mask = (1 << (value - 1).bit_length()) - 1
    asm = (
        "{\n"
        ".reg .u64 %rf_t;\n"
        ".reg .u32 %rf_h, %rf_w, %rf_d;\n"
        ".reg .pred %rf_p;\n"
        "mov.u64 %rf_t, %globaltimer;\n"
        "cvt.u32.u64 %rf_h, %rf_t;\n"
        "mov.u32 %rf_w, %warpid;\n"
        "mad.lo.u32 %rf_h, %rf_w, 0x9E3779B1, %rf_h;\n"
        "shr.u32 %rf_d, %rf_h, 11;\n"
        f"and.b32 %rf_d, %rf_d, {mask};\n"
        "and.b32 %rf_h, %rf_h, 0x600;\n"
        "setp.eq.u32 %rf_p, %rf_h, 0;\n"
        "@%rf_p nanosleep.u32 %rf_d;\n"
        "}"
    )

    @dsl_user_op
    def delay_some_warps(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            asm,
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    def delayed(original):
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            delay_some_warps()
            return result

        wrapper.__wrapped__ = original
        return wrapper

    for name in ("sync_threads", "barrier", "mbarrier_wait"):
        setattr(cute.arch, name, delayed(getattr(cute.arch, name)))
    _installed_ns = value
