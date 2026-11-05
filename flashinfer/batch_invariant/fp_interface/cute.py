# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The only floating-point primitives a batch-invariant CuTe-DSL kernel may use.

Each one lowers to an NVVM intrinsic with an explicit ``.rn`` rounding mode (``add.rn.f32``, ``mul.rn.f32``). Unlike
``a + b`` on ``Float32`` (which lowers to ``arith`` ops that the NVPTX backend is free to contract into
FMAs), these are opaque to the compiler: no contraction, no reassociation, no flush-to-zero. The GEMM's
tensor-core step is not here: it is the kernel's opaque ``cute.gemm`` on the accumulator, below which
nothing is reordered.

The class mirrors the interface of :class:`flashinfer.batch_invariant.fp_interface.term.TermAlgebra` (symbolic), so
the specification's chains can be written once and run under both backends.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.math import RoundingMode

RN = RoundingMode.NEAREST_EVEN


class CuteF32:
    @staticmethod
    def const(value: float):
        from .term import to_f32

        return Float32(to_f32(value))

    @staticmethod
    def add(a, b):
        """``add.rn.f32``."""
        return cute.math.add(a, b, rounding=RN)

    @staticmethod
    def mul(a, b):
        """``mul.rn.f32``."""
        return cute.math.mul(a, b, rounding=RN)

    @staticmethod
    def cvt_out(a, fmt_name: str):
        """``cvt.rn.<fmt>.f32`` (``arith.truncf``, RNE); the identity for ``f32``."""
        return a if fmt_name == "f32" else a.to(get_cutlass_dtype(fmt_name))


# Name <-> CuTe dtype table shared by the kernels (kept out of the API modules so that compiled-kernel cache keys do not depend
# on them).
def get_cutlass_dtype(name: str):
    import cutlass

    return {
        "bf16": cutlass.BFloat16,
        "f16": cutlass.Float16,
        "f32": cutlass.Float32,
        "e4m3": cutlass.Float8E4M3FN,
        "e5m2": cutlass.Float8E5M2,
    }[name]
