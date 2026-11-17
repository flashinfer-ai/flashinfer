# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The kernels' store of the LSE member (``spec.lse_term``): the thread that holds a row's final ``(m, l)`` writes
``add(mul(m, c), LG2(l))`` (``lse_mode`` 1, base 2) or its ``mul`` by ``LN2`` (``lse_mode`` 2, base e) to the row's fp32
element of ``mLse``; ``lse_mode`` 0 stores nothing. ``LG2`` is the sampling class's pinned ``lg2.approx.ftz.f32``."""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Int32

from ...batch_invariant.fp_interface.cute import CuteF32
from ...batch_invariant.sampling.cute_ext import lg2_approx_ftz
from .spec import LN2, State, lse_term


class LseF32(CuteF32):
    @staticmethod
    def lg2(a):
        """``lg2.approx.ftz.f32``, pinned as inline asm."""
        return lg2_approx_ftz(a)


@cute.jit
def store_lse(
    mLse: cute.Tensor,
    out_row: Int32,
    m: Float32,
    l: Float32,
    c: Float32,
    lse_mode: Int32,
):
    lse = lse_term(LseF32, State(m, l, Float32(0.0)), c, "log2")
    if lse_mode == 2:
        lse = LseF32.mul(lse, Float32(LN2))
    mLse[out_row] = lse
