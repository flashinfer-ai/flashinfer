# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""debug instrumentation: per-CTA wait accounting by warp role.

Behind ``FLASHINFER_MXFP8_MXFP4_WAIT_STATS=1`` the kernel records
``%globaltimer`` deltas around each pipeline wait and accumulates them, per
CTA, into a stats area: appended to the FC1-done counter array on the lean
path, and a separate ``wait_stats`` workspace region on the mega path (see
:mod:`.wait_stats_config`).  Each event is one relaxed ``red.global.add``
from a single lane, so the knob costs global atomics per wait and must stay
off in timed or shipped runs.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, T, dsl_user_op

from .wait_stats_config import (  # noqa: F401  (re-exported)
    MAX_CTAS,
    SLOTS,
    STATS_WORDS,
    WAIT_STATS_ENV,
    Slot,
    mega_stats_word_offset,
    read_mega_wait_stats,
    summarize,
    wait_stats_enabled,
)


@dsl_user_op
def globaltimer_lo(
    *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> Int32:
    """Low 32 bits of ``%globaltimer`` (ns); wrapping deltas are exact < 4 s."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %globaltimer_lo;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def clock_lo(
    *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> Int32:
    """Low 32 bits of the SM cycle counter."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %clock;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stat_add(
    slot_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Fire-and-forget relaxed add into one stats word."""

    llvm.inline_asm(
        None,
        [
            slot_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def stats_slot_ptr(stats, slot: int, word_offset=None):
    """This CTA's stats word ``slot``.

    Lean: the stats area ends the FC1-done counter array.  Mega: it is the
    ``wait_stats`` workspace region, ``word_offset`` int32 words past the
    counter array in the same local-workspace allocation.
    """

    bx, by, bz = cute.arch.block_idx()
    gx, gy, _ = cute.arch.grid_dim()
    cta = bx + by * gx + bz * gx * gy
    if cutlass.const_expr(word_offset is None):
        base = cute.size(stats) - Int32(STATS_WORDS)
    else:
        base = Int32(word_offset)
    return stats.iterator + base + cta * Int32(SLOTS) + Int32(slot)


__all__ = [
    "MAX_CTAS",
    "SLOTS",
    "STATS_WORDS",
    "Slot",
    "WAIT_STATS_ENV",
    "clock_lo",
    "globaltimer_lo",
    "mega_stats_word_offset",
    "read_mega_wait_stats",
    "stat_add",
    "stats_slot_ptr",
    "summarize",
    "wait_stats_enabled",
]
