# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""What the kernels do with a row's final ``(m, l)``, at every site that holds it: the normaliser of the epilogue
(``row_normaliser``: ``div(1, l)``, or in a kernel compiled for attention sinks the sink step ``spec.sink_merge`` and
``mul(a, div(1, l'))``) and the store of the LSE member (``store_lse``: ``spec.lse_term`` or, in a sink kernel,
``spec.lse_term_sink`` of the merged statistics; the run-time ``lse_mode`` 1 base 2, 2 its ``mul`` by ``LN2``, 0 no
store). ``LG2`` is the sampling class's pinned ``lg2.approx.ftz.f32``. Sinks are a compile-time variant of every kernel
(``sinks``, part of its name): a kernel without them holds no sink code and no sink operand in its loops."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ...batch_invariant.fp_interface.cute import CuteF32
from ...batch_invariant.sampling.cute_ext import lg2_approx_ftz
from .spec import (
    LN2,
    SinkState,
    State,
    lse_term,
    lse_term_sink,
    sink_merge,
    sink_normaliser,
)


class LseF32(CuteF32):
    @staticmethod
    def lg2(a):
        """``lg2.approx.ftz.f32``, pinned as inline asm."""
        return lg2_approx_ftz(a)


def _sink_row(m, l, sink, c):
    sk = sink_merge(CuteF32, State(m, l, Float32(0.0)), sink, c)
    return sk.m, sk.l, sink_normaliser(CuteF32, sk)


@cute.jit
def row_normaliser(
    mSink: cute.Tensor,
    head: Int32,
    m: Float32,
    l: Float32,
    c: Float32,
    sinks: cutlass.Constexpr,
    partial=None,
):
    """``(inv, m, l)`` of a row from its final ``(m, l)``: ``div(1, l)`` and the pair as given in a kernel without sinks
    (``sinks`` False: no sink code is emitted) and for a row stored as a partial (``partial``, a run-time flag; ``None`` at
    a site that only completes rows); otherwise the sink step with ``mSink[head]`` -- ``mul(a, div(1, l'))`` and the merged
    ``(m', l')`` that the LSE member reads."""
    m_s = m
    l_s = l
    inv = Float32(0.0)
    if cutlass.const_expr(not sinks):
        inv = CuteF32.div(Float32(1.0), l)
    else:
        if cutlass.const_expr(partial is None):
            m_s, l_s, inv = _sink_row(m, l, mSink[head], c)
        else:
            if partial:
                inv = CuteF32.div(Float32(1.0), l)
            else:
                m_s, l_s, inv = _sink_row(m, l, mSink[head], c)
    return inv, m_s, l_s


@cute.jit
def store_lse(
    mLse: cute.Tensor,
    out_row: Int32,
    m: Float32,
    l: Float32,
    c: Float32,
    lse_mode: Int32,
    sinks: cutlass.Constexpr,
):
    """``(m, l)`` of a completed row as ``row_normaliser`` returned them."""
    if cutlass.const_expr(sinks):
        lse = lse_term_sink(LseF32, SinkState(m, l, Float32(1.0)), "log2")
    else:
        lse = lse_term(LseF32, State(m, l, Float32(0.0)), c, "log2")
    if lse_mode == 2:
        lse = LseF32.mul(lse, Float32(LN2))
    mLse[out_row] = lse
