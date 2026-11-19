# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""What the kernels do with a row's final ``(m, l)``, at every site that holds it: the normaliser of the epilogue
(``row_normaliser``: ``div(1, l)``, or under attention sinks the sink step ``spec.sink_merge`` and ``mul(a, div(1, l'))``) and
the store of the LSE member (``store_lse``: ``spec.lse_term`` or, under sinks, ``spec.lse_term_sink`` of the merged
statistics; ``lse_mode`` 1 base 2, 2 its ``mul`` by ``LN2``, 0 no store). ``LG2`` is the sampling class's pinned
``lg2.approx.ftz.f32``."""

from __future__ import annotations

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
    sink_on,
):
    """``(inv, m, l)`` of a row from its final ``(m, l)``: ``div(1, l)`` and the pair as given, or, with ``sink_on``, the sink
    step with ``mSink[head]`` -- ``mul(a, div(1, l'))`` and the merged ``(m', l')`` that the LSE member reads."""
    m_s = m
    l_s = l
    inv = Float32(0.0)
    if sink_on:
        m_s, l_s, inv = _sink_row(m, l, mSink[head], c)
    else:
        inv = CuteF32.div(Float32(1.0), l)
    return inv, m_s, l_s


@cute.jit
def store_lse(
    mLse: cute.Tensor,
    out_row: Int32,
    m: Float32,
    l: Float32,
    c: Float32,
    lse_mode: Int32,
    sink_on,
):
    """``(m, l)`` as ``row_normaliser`` returned them."""
    lse = Float32(0.0)
    if sink_on:
        lse = lse_term_sink(LseF32, SinkState(m, l, Float32(1.0)), "log2")
    else:
        lse = lse_term(LseF32, State(m, l, Float32(0.0)), c, "log2")
    if lse_mode == 2:
        lse = LseF32.mul(lse, Float32(LN2))
    mLse[out_row] = lse
