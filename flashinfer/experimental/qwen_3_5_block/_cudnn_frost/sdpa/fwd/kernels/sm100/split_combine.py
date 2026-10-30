# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-split combine (reduction) pass for the SM100 DSL prefill SDPA kernels.

Each split writes ``O_s`` (normalized by its own running sum) and ``lse_s`` into
a split-major workspace at batch coord ``b + s*B``; this pass reduces over ``s``.

A split whose range came out empty ends with total_sum == 0, which the epilogue
turns into ``O := 0 / lse := -inf`` — the identity here, so empty splits need no
special case.

Attention sinks: a split kernel compiled with SPLIT_KV > 1 writes SINK-FREE
partials (its epilogue fold is compiled out), and this pass folds the per-head
sink logit in exactly once: ``M = max(max_s lse_s, sink)``, ``den = sum_s
exp(lse_s - M) + exp(sink - M)``, ``O = sum_s w_s O_s / den`` (the sink column
has no V row) and ``LSE = M + log(den)``.  A keyless row (every split dead)
therefore lands on the unsplit kernels' contract, ``O := 0 / LSE := sink``.  A PADDED query row of
the dense padded-Q trim (a row at or past its batch's Q length) keeps the unsplit contract too, ``O := 0 / LSE := -inf`` whatever the sink, but its
partials are the same as a valid keyless row's, so the dense sink entry takes the per-batch Q lengths (``compile_ptr(has_q_lens=True)``) and drops the
sink for those rows.

One block of four warps per four (q_row, head, batch) rows, a lane owning four
columns of d_v and walking the split axis; rows shorter than a block (decode,
S_q < 4) take the decode form, whose four warps share one row's split axis.

The DECODE-GRID kernel (``_combine_decode_grid_kernel``), a fourth form of the
lean fp32-partial path: one block per (q_row, head, batch,
``DECODE_CHUNK_COLS``-column chunk of d_v) -- four times the blocks at d 256 --
with the split axis spread over the block's eight 16-lane chains (two per warp;
chain ``c`` owns splits ``16 c .. 16 c + 15`` of every round of 128), every chain
issuing all of its loads before any math, each chain keeping an online-softmax
(max, sum, four columns) in registers, the eight chains meeting ONCE in a 2 KiB
SMEM exchange behind the kernel's single CTA barrier (every one of the 128
threads arrives exactly once; nothing exits or diverges around it) and warp 0
merging them in chain order and storing the row.  So a thread's dependent-load
chain is one DRAM round trip for up to 128 splits, and a decode step (B 1 x 24 Q
heads = 24 lean blocks on 148 .. 212 SMs) fills 96 SMs instead of 24.  Which
kernel runs is decided per launch in ``_launch_combine`` from the runtime
geometry: the decode-grid kernel when the entry's ``decode_grid`` form selector
(``compile_ptr(decode_grid=)``, ``DECODE_GRID_OFF`` unless named; the cc 10.7 adapter pins it per plan through
``heuristics.decode_combine_grid_selected``) is not ``DECODE_GRID_OFF``, the
launch is a sink-free, unquantized fp32-partial one whose lean geometry holds
(``d_v % 4 == 0``, a 16-byte aligned slab) and -- under ``DECODE_GRID_AUTO`` --
the lean grid holds at most ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks;
every other launch takes the three-way rule above (the decode form past G_S
splits at S_q <= 2, the wide lean group within it, the serial form from S_q 3),
which is also what ``DECODE_GRID_OFF`` means; ``DECODE_GRID_LEAN_WALK`` names the
fifth form below.  The quantized and packed entries
keep their own kernels.  Order statement: every reduction of every form is in a
FIXED order (no atomics on O / LSE), so a launch is deterministic; for
``n_splits <= 16`` only chain 0 of the decode-grid kernel holds splits and its
arithmetic is the lean group's own sequence (and the serial walk's: the same
weights and products added in split order from +0), so the decode-grid output is
BITWISE the lean / serial forms'; past 16 splits the decode-grid kernel
associates the sum differently (eight chain partials merged with ``exp(m_c - m)``
factors), as the decode form does with its warp-strided lanes, and the forms
agree to fp32 rounding.  The SMEM exchange of the decode-grid kernel (fp32
words: ``red_m[8]``, ``red_den[8]``, then ``red_acc[8 chains][16 column
groups][4]``) is lane-written and lane-read at the same per-thread 16-byte
stride -- 16 threads x 16 B = two full 128-byte bank cycles per chain, no
descriptor reads it, so it carries no swizzle.  Programmatic dependent launch
(``compile_ptr(use_pdl=True)``, the sink-free dense half entries): every form
carries the griddepcontrol pair under one constexpr -- the wait before its
first global read (the partial slabs are the split tile's output), the release
right after (the pass has no prologue to overlap) -- and the launch carries the
programmatic-stream-serialization attribute; the same math, bitwise.

The LEAN GROUP WALK (``_combine_lean_walk_kernel``), a fifth form of the lean
fp32-partial path: the module's FORMER lean kernel, restored for the lean-grid
OVERFLOW launches of the cc 10.7 line -- one warp per (q_row, head, batch), four
rows per block, a lane owning four columns of every split; the LSE and partial-O
rows of the first ``LEAN_WALK_G_S`` (4) splits are loaded up front, the rest walked
in groups of ``LEAN_WALK_G_T`` (16) loads per memory round trip (the max extended
group by group before any weight is formed), a clamped slot re-reading the last
split and SELECTED out, the row accumulated in split order from +0.  So its O /
LSE are BITWISE the serial walk's at every split count and the wide lean group's
within that group's window (<= 16 splits); the decode form (past 16 splits at
S_q <= 2) associates the sum differently and agrees to fp32 rounding.  It carries
the ragged placement, the gate, the amax / scale arms and the griddepcontrol pair
of the kernel it was, and no sink fold: ``_launch_combine`` takes it only on a
sink-free launch (``sinks is None and q_lens is None``; a sink entry keeps the
three-way rule under every selector value), and only where the lean geometry
holds (``d_v % 4 == 0``, a 16-byte aligned slab: the same runtime test as the
decode-grid kernel's), else the three-way rule -- never the decode-grid kernel.
It is reached through ``compile_ptr(decode_grid=DECODE_GRID_LEAN_WALK)`` alone:
the cc 10.7 adapter names it per plan where
``heuristics.decode_combine_lean_walk_selected`` says so -- a lean grid PAST
``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` (where the decode-grid kernel is
never eligible) at a split of at most ``LEAN_WALK_G_S``
(``heuristics.DECODE_COMBINE_LEAN_WALK_MAX_SPLIT``: the whole split axis in the
up-front group, one memory round trip -- the measured regime; a deeper split
keeps the module's own forms until a measurement widens the bound), measured on
the Rubin performance class: at B 32 / B 128 x 24
heads (768 / 3072 lean blocks) the walk costs 3.3-3.4 / 5.9 us at split 2 where
the wide lean group costs 5.9 / 12.4, and 4.0 / 8.9 us at S_q 4 (split 2-3) where
the serial walk costs 5.3-5.5 / 11.7; on the small grids inside the bound the
module's forms are the faster ones (the wide group 4.1 us against the walk's 5.6
at B 1 x 4K x 16 splits; the decode form 5.8 / 8.1 / 6.1 against 8.4 / 14.0 / 8.5
past 16 splits) and stay.  The Blackwell line's adapter never names the walk; the
quantized entries drop every selector value (``compile_ptr``: their form is
``()``, their launch rule the three-way one), so the walk's amax / scale arms are
reachable only by a direct ``_launch_combine`` call.

Gate in the combine (``compile_ptr(gate=True)``, the dense half entry only): a
split launch of a kernel that has no epilogue-gate seams -- the d256 decode
tile -- carries the graph's ``mul(O_v, sigmoid(G))`` tail HERE.  The gate is
applied to the fp32 MERGED value, before the single cast to the output dtype,
with the arithmetic the gated prefill kernels use in their epilogue
(``_common_blackwell.gate_epilogue_pairs``): ``h = acc * (inv_den / 2)`` and
``O = h * tanh(g / 2) + h`` (one FMUL2, one MUFU.TANH per element, one FFMA2),
the dead-row SELECT per element AFTER the fma (``sdpa-invariants.md`` section 2:
never a multiply by zero, a gate value is never trusted on a dead row).  So the
rounding convention is FROST's: ONE rounding, of ``O32 * sigmoid(G)``, exactly
what the fused epilogue does -- a gated split plan and a gated unsplit plan of
the same graph differ only by the summation order of the attention itself.  The
unfused references round the merged O to the output dtype BEFORE the fp32 gate
and round again after it (vLLM's split merge, and the gated attention block's
torch reference, which gates the output-dtype O): two roundings, so they differ
from this pass by up to one output ulp plus the approximate tanh.  LSE / Stats
never see the gate (the partial LSEs are gate-free and the final one is their
log-sum-exp).  The gate is read element-wise through the caller's BSHD strides
(no layout constraint of its own; the engine rows keep the zero-copy rule the
fused kernels need so one G declaration serves both paths).  Every form carries
the gate -- the decode and decode-grid forms apply the same arithmetic at their
store -- so a gated and an ungated launch of one shape run one kernel; the gated
entry takes no sink (a gated split with a sink is declined ahead of this pass).
"""

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from typing import Callable, Optional, Tuple

from functools import lru_cache
import hashlib
from pathlib import Path

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels._quantized import _unscale_amax_kernel
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import DECODE_COMBINE_MAX_LEAN_BLOCKS as _DECODE_COMBINE_MAX_LEAN_BLOCKS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import SmemAllocator

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import ffma2, fmul2, opaque_f32_zero
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global_v4, st_global_v2
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

# This helper is imported normally, outside the parameterized template loader.
# Supply its source identity so template_key can persist its pointer entry points;
# the cache environment manifest also covers all transitive Python sources.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

# One WARP per (q_row, head, batch) and ROWS_PER_BLOCK rows per block: each lane
# owns four consecutive columns (a 16-byte load per split, so a warp reads a
# row's 128 columns as one 512-byte coalesced request), and the wider
# d256/d512 flavors loop the warp over 128-column chunks.  The previous
# one-block-per-row form (128 lanes, one scalar load each, split loop rolled)
# was DRAM-latency bound: measured 15.5 us (bf16) / 28.7 us (fp8) for
# 985 x 9 rows x 4 splits on B300 against ~18 MB of traffic.
THREADS = 128
ROWS_PER_BLOCK = THREADS // 32
# Splits whose LSE and partial-O rows the lean form loads up front, before any math.  The lean form serves a launch whose whole split axis fits in
# this group (n_splits <= G_S): every load goes out at once, one memory round trip, which beats the serial form there (B300, d256 S_q = 4, 6-8 splits:
# 2.9 us against 4.7) and the decode form's three dependent trips (S_q = 1, 16 splits: 4.7 us against 2.9).  Past it the serial / decode forms' walks are
# far ahead.  A row with at most G_LEAN_SMALL splits loads only that many (the wide group's extra clamped slots cost ~0.4 us at 8 splits).
G_S = 16
# The narrow group: rows with at most this many splits load only this many (every extra clamped slot costs a load pair and an exp: ~0.4 us at 8 splits).
G_LEAN_SMALL = 8
# Unroll depth of the decode form's O pass: this many splits' partial-O loads of one warp are in flight together (B200, S_q = 1, b = 1: 4 -> 8 takes the
# combine at 32 / 64 / 128 splits from 7.2 / 10.7 / 17.4 us to 5.4 / 7.2 / 10.2; 16 gains nothing at 32 splits, whose warps own only 8 of them).
_DECODE_UNROLL = 8
# Unroll depth of the serial combine's split walks (one warp per row): the
# loads of this many splits are issued together, so the latency of the
# dependent walk is paid once per group rather than once per split.  Each walk
# runs in two legs -- the splits up to the last full group with this unroll,
# the remaining ``n_splits % _SERIAL_UNROLL`` with the four-wide one -- because
# a compiler-unrolled loop whose trip count is below its unroll degrades to a
# serial remainder loop (n_splits = 8 measured 4.2 -> 7.0 us on B300 at a
# plain unroll of 16), and an explicitly predicated group (clamped index, zero
# weight past the end) issued its loads serially (d256 x 128 splits 47 -> 80
# us).  Each unrolled split costs a lane four fp32 loads plus its LSE, inside
# the serial kernel's register budget.  B300, d128 S_q=16 x 32 splits: 11.0 ->
# 5.5 us; d256 S_q=4 x 64 splits: 25.0 -> 12.2 us; x 128 splits: 47 -> 21.
_SERIAL_UNROLL = 16
_SERIAL_TAIL_UNROLL = 4

# The DECODE-GRID form of the lean fp32-partial path (``_combine_decode_grid_kernel``):
# one block per (q_row, head, batch, DECODE_CHUNK_COLS-column chunk of d_v) -- 4x
# the blocks of the lean forms at d 256 -- and the split axis spread over the
# block's DECODE_CHAINS chains of DECODE_LANES_PER_CHAIN lanes (two half-warps per
# warp; a lane owns four consecutive columns, one 16-byte load per split).  Every
# chain issues the LSE and partial-O loads of its DECODE_SPLITS_PER_CHAIN slots
# before any math, so a lane's dependent-load chain is ONE memory round trip for
# up to DECODE_SPLITS_PER_ROUND splits; the chains' running (max, sum, acc) meet
# once in SMEM and are merged in a FIXED order (chain 0 .. 7) by warp 0, which
# also stores the row.  The selection between this kernel and the module's own
# forms is ``_launch_combine``'s (the entry's ``decode_grid`` selector and, under
# AUTO, the lean grid against heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS): a decode
# step's tens of blocks take this form, a prefill's thousands keep the module's.
DECODE_THREADS = 128
DECODE_CHUNK_COLS = 64
DECODE_LANES_PER_CHAIN = DECODE_CHUNK_COLS // 4
DECODE_CHAINS = DECODE_THREADS // DECODE_LANES_PER_CHAIN
# Splits per chain and per round: 16 was the group width (G_T) of the former lean walk, which this kernel's first round
# reproduces in split order -- it is also the bitwise window against the lean group and the serial walk (<= 16 splits).
DECODE_SPLITS_PER_CHAIN = 16
DECODE_SPLITS_PER_ROUND = DECODE_CHAINS * DECODE_SPLITS_PER_CHAIN
# The SMEM exchange: red_m[chain], red_den[chain], then red_acc[chain][lane][4].
DECODE_RED_DEN_OFF = DECODE_CHAINS
DECODE_RED_ACC_OFF = 2 * DECODE_CHAINS
DECODE_RED_WORDS = DECODE_RED_ACC_OFF + DECODE_CHAINS * DECODE_LANES_PER_CHAIN * 4
# ``compile_ptr(decode_grid=...)``: which form a pointer entry launches.
DECODE_GRID_AUTO = 0  # the grid rule (the cc 10.7 adapters' entry): the decode-grid kernel inside the lean-grid bound, else the module's own forms
DECODE_GRID_ON = 1  # always the decode-grid kernel where its geometry holds (tests; an explicit decode flag)
DECODE_GRID_OFF = 2  # never the decode-grid kernel: the module's own three-way rule (compile_ptr's default; the Blackwell line's entry; tests; a kill switch)
# Appended 2026-10-16: the LEAN GROUP WALK (the cc 10.7 adapter's entry on a lean-grid-OVERFLOW plan at a split of at most LEAN_WALK_G_S,
# heuristics.decode_combine_lean_walk_selected): the module's former lean kernel on a sink-free fp32-partial launch whose lean geometry holds, else the
# three-way rule; never the decode-grid kernel.
DECODE_GRID_LEAN_WALK = 3
# The lean group walk's two group widths (the former lean kernel's G_S / G_T).  LEAN_WALK_G_S splits' LSE and partial-O rows are loaded up front,
# before any math; splits past them are walked in GROUPS of LEAN_WALK_G_T: every group's rows are loaded before any of its math, so a warp
# pays one memory round trip per group instead of one per split (the one-split-per-step walk read 40 us for 64 splits at batch 1 where 2
# splits cost 3.3 us: ~0.3 us, one DRAM round trip, per split per pass).  A slot past ``n_splits`` re-reads the last split and contributes
# nothing, so the pass is bitwise what the one-split-per-step walk was for every split count.  LEAN_WALK_G_T is also the decode-grid
# kernel's per-chain slot count and the wide lean group's width (DECODE_SPLITS_PER_CHAIN == G_S == 16): the bitwise window of the forms.
LEAN_WALK_G_S = 4
LEAN_WALK_G_T = 16


def lean_grid_blocks(b: int, h: int, s_q: int) -> int:
    """The lean forms' grid in blocks: ``ceil(S_q / ROWS_PER_BLOCK) x H_q x B`` (the quantity the decode-grid and lean-walk selection rules read)."""
    return -(-int(s_q) // ROWS_PER_BLOCK) * int(h) * int(b)


def decode_grid_selected(b: int, h: int, s_q: int, d_v: int, has_sink: bool = False) -> bool:
    """The host-side twin of ``_launch_combine``'s selection under ``DECODE_GRID_AUTO``: whether a dense half launch with fp32
    partials (16-byte aligned, the lean path's own precondition) of this geometry takes the decode-grid kernel --
    ``d_v % 4 == 0``, the lean grid at most ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks, and no sink (``has_sink``,
    appended: the sink entries fold the sink in the module's own forms).  Quantized entries (an amax or an output scale) and
    half partials always take the module's own forms."""
    return not has_sink and int(d_v) % 4 == 0 and lean_grid_blocks(b, h, s_q) <= _DECODE_COMBINE_MAX_LEAN_BLOCKS


def _serial_legs(n_splits):
    """``(n_main, n_splits)``: the end of the last full _SERIAL_UNROLL group."""
    return n_splits - n_splits % cutlass.Int32(_SERIAL_UNROLL), n_splits


@cute.jit
def _accumulate_splits(
    lo,
    hi,
    unroll: cutlass.Constexpr[int],
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_base,
    vec_ok,
    batch,
    n_batch,
    head,
    q_row,
    d0,
    d_v,
    m_safe,
    neg_inf,
    zero,
    acc0,
    acc1,
    acc2,
    acc3,
):
    """One leg of the serial combine's O accumulation over splits ``[lo, hi)``
    for this lane's four columns ``d0..d0+3``; returns the four accumulators."""
    op = cutlass.make_array_view(o_partial)
    lp = cutlass.make_array_view(lse_partial)
    for s in cutlass.range(lo, hi, 1, unroll=unroll):
        lse_row = lp[batch + s * n_batch, head, :]
        lse_s = cutlass.Float32(lse_row[q_row])
        live = lse_s > neg_inf
        e = cute.math.exp(lse_s - m_safe, fastmath=True)
        w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
        # Finite LSE can still underflow against a dominant split or sink.
        # Its payload is dead too: never multiply zero by a NaN/Inf partial.
        live = w > zero
        e0 = zero
        e1 = zero
        e2 = zero
        e3 = zero
        # Element loads through the view, each bounded by the row, serve the
        # half partials (flavors whose split epilogue keeps the staged TMA-store
        # O path, e.g. d512) and any d_v / base that cannot take aligned 16-byte
        # loads (legal through the runtime-shape pointer entries); aligned fp32
        # partials take one 16-byte load per split.
        if cutlass.const_expr(o_partial.element_type == cutlass.Float32):
            if vec_ok:
                idx = cute.crd2idx((batch + s * n_batch, q_row, head, d0), o_partial.layout)
                v = ld_global_v4(o_base + cutlass.Int64(idx) * 4, cutlass.Float32)
                e0 = cutlass.Float32(v[0])
                e1 = cutlass.Float32(v[1])
                e2 = cutlass.Float32(v[2])
                e3 = cutlass.Float32(v[3])
            else:
                o_row = op[batch + s * n_batch, q_row, head, :]
                e0 = cutlass.Float32(o_row[d0])
                if d0 + cutlass.Int32(1) < d_v:
                    e1 = cutlass.Float32(o_row[d0 + cutlass.Int32(1)])
                if d0 + cutlass.Int32(2) < d_v:
                    e2 = cutlass.Float32(o_row[d0 + cutlass.Int32(2)])
                if d0 + cutlass.Int32(3) < d_v:
                    e3 = cutlass.Float32(o_row[d0 + cutlass.Int32(3)])
        else:
            o_row = op[batch + s * n_batch, q_row, head, :]
            e0 = cutlass.Float32(o_row[d0])
            if d0 + cutlass.Int32(1) < d_v:
                e1 = cutlass.Float32(o_row[d0 + cutlass.Int32(1)])
            if d0 + cutlass.Int32(2) < d_v:
                e2 = cutlass.Float32(o_row[d0 + cutlass.Int32(2)])
            if d0 + cutlass.Int32(3) < d_v:
                e3 = cutlass.Float32(o_row[d0 + cutlass.Int32(3)])
        # a dead slot may hold anything, including non-finite values
        v0 = cutlass.Float32(arith.select(live.ir_value(), e0.ir_value(), zero.ir_value()))
        v1 = cutlass.Float32(arith.select(live.ir_value(), e1.ir_value(), zero.ir_value()))
        v2 = cutlass.Float32(arith.select(live.ir_value(), e2.ir_value(), zero.ir_value()))
        v3 = cutlass.Float32(arith.select(live.ir_value(), e3.ir_value(), zero.ir_value()))
        acc0 = acc0 + w * v0
        acc1 = acc1 + w * v1
        acc2 = acc2 + w * v2
        acc3 = acc3 + w * v3
    return acc0, acc1, acc2, acc3


# Shared memory of the decode-form combine: the per-warp (max | sum) exchange and
# the per-lane four-column partial sums of the warps that share a row.
SMEM_BYTES = (2 * ROWS_PER_BLOCK + ROWS_PER_BLOCK * 32 * 4) * 4 + 64


def _min_i32(a, b):
    """min of two Int32 DSL values (usable inside a kernel and in a jit host)."""
    return cutlass.Int32(arith.select((a < b).ir_value(), a.ir_value(), b.ir_value()))


NEG_INF = float("-inf")

# The FP8 entries are reachable only as the OUTPUT type, where this pass
# performs the single cast.  "f32" is a PARTIAL-only type: the split epilogue can
# write its fp32 accumulator straight to global (no SMEM staging), so the
# reduction reads exactly what the mainloop produced with no intermediate
# rounding.
_ELEM = {
    "f32": cutlass.Float32,
    "f16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
    "e4m3": cutlass.Float8E4M3FN,
    "e5m2": cutlass.Float8E5M2,
}


def _ld_global_f32(addr):
    """32-bit global load of an fp32 at ``addr``."""
    r = nvvm.inline_ptx(
        "ld.global.f32 $0, [$1];",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr],
    )
    return cutlass.Float32(r[0] if isinstance(r, (tuple, list)) else r)


def _row_sink_logit(sinks, q_lens, head, batch, q_row):
    """The sink logit one row folds: its head's, or -inf (no sink at all) on a PADDED row -- a query row at or past its batch's Q length
    ``q_lens[batch]`` (the dense padded-Q trim), whose O / Stats stay 0 / -inf whatever the sink.  The partials of such a row and of a valid row whose
    every key is masked are the same (every split dead), so the reduction cannot tell them apart without the length."""
    sink_logit = cutlass.Float32(cutlass.make_array_view(sinks)[head])
    if cutlass.const_expr(q_lens is not None):
        padded = cutlass.Int32(q_row) >= cutlass.Int32(cutlass.make_array_view(q_lens)[batch])
        sink_logit = cutlass.Float32(arith.select(padded.ir_value(), cutlass.Float32(NEG_INF).ir_value(), sink_logit.ir_value()))
    return sink_logit


def _fold_sink(m, sink_logit):
    """(all_dead, m_safe, sink_term) of a row whose partial max is ``m``: the reference max is max(m, sink) -- finite on a keyless row, which lands on
    O = 0 / LSE = sink, and -inf only where there is neither a live split nor a sink (a padded row), which stays dead.  m_safe only feeds the
    exponentials, and the sink column's weight is dropped with a select on a row without a sink (exp(-inf - -inf) would be NaN)."""
    neg_inf = cutlass.Float32(NEG_INF)
    zero = cutlass.Float32(0.0)
    m_all = cute.math.max(m, sink_logit)
    all_dead = m_all == neg_inf
    m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m_all.ir_value()))
    sink_term = cutlass.Float32(arith.select((sink_logit > neg_inf).ir_value(), cute.math.exp(sink_logit - m_safe, fastmath=True).ir_value(), zero.ir_value()))
    return all_dead, m_safe, sink_term


def _lean_accumulate(G, lse_base, lse_step, o_row_base, o_step, s_last, n_splits, sink_logit):
    """The lean combine's load + weighted accumulate for ONE lane (columns 4l..4l+3) over the first ``G`` splits, as a trace-time function so the kernel
    can instantiate two group widths and pick one per row: the first group (``G`` = G_LEAN_SMALL) is the cheap one-round-trip form for the common few
    splits; the wide group (``G`` = G_S) keeps one round trip up to G_S splits at the price of the extra clamped loads.  Returns the dead-row flag, the
    denominator, the four weighted O sums and the (0-substituted) max the LSE is built from.  ``sink_logit`` is the row's sink logit (:func:`_row_sink_logit`)
    or None without a sink."""
    neg_inf = cutlass.Float32(NEG_INF)
    zero = cutlass.Float32(0.0)
    # Every load of the row goes out first -- the LSE of the first G splits and their partial-O
    # rows (a clamped split index re-reads the last split for slots past n_splits; those get
    # weight 0) -- so a warp pays ONE memory round trip instead of one per pass.
    lse_v = []
    o_v = []
    for s in range(G):
        s_eff64 = cutlass.Int64(cute.math.min(cutlass.Int32(s), s_last))
        lse_v.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
        o_v.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
    m = neg_inf
    for s in range(G):
        m = cute.math.max(m, lse_v[s])
    # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather than
    # exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
    all_dead = m == neg_inf
    m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m.ir_value()))
    # With a sink the reference max is max(m, sink), which is finite even on a
    # keyless row, and the denominator starts with the sink column's weight
    # exp(sink - M): that row then yields O = 0 and LSE = sink + log(1) = sink,
    # the unsplit kernels' keyless-row contract (a padded row has no sink: it stays dead).
    sink_term = zero
    if cutlass.const_expr(sink_logit is not None):
        all_dead, m_safe, sink_term = _fold_sink(m, sink_logit)
    acc0 = zero
    acc1 = zero
    acc2 = zero
    acc3 = zero
    den = sink_term
    for s in range(G):
        lse_s = lse_v[s]
        v = o_v[s]
        live = (lse_s > neg_inf) & (cutlass.Int32(s) < n_splits)
        e = cute.math.exp(lse_s - m_safe, fastmath=True)
        w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
        # A finite LSE can still underflow to zero weight; 0 * NaN is not zero.
        live = w > zero
        v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
        v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
        v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
        v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
        den = den + w
        acc0 = acc0 + w * v0
        acc1 = acc1 + w * v1
        acc2 = acc2 + w * v2
        acc3 = acc3 + w * v3
    return all_dead, den, acc0, acc1, acc2, acc3, m_safe


@cute.kernel
def _combine_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] — split-major partial O
    lse_partial: cute.Tensor,  # [S*B, H, S_q]   — split-major partial LSE
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    # 1-element fp32 amax of the RECOMBINED O.  The per-split epilogues cannot
    # compute this: each sees only its own partial, and O is a convex
    # combination of those, so a max over partials over-reports (~2.9x at 8
    # splits).  The FP8 kernels therefore skip their in-kernel amax write when
    # SPLIT_KV > 1 and leave it to this pass.  None-specialized off otherwise.
    amax_o: Optional[cute.Tensor],
    # 1-element fp32 output scale, present only when O is stored quantized.
    # The split kernels leave it out of their epilogue and write UNSCALED
    # partials; this pass applies it at the single cast, so the FP8 rounding
    # happens once, on the recombined value.
    scale_o: Optional[cute.Tensor],
    # fp32 [H] per-head attention-sink logits, folded ONCE here (the split
    # partials are sink-free).  None-specialized off without a sink.
    sinks: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    s_q: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],  # write the FINAL LSE in base 2 (stats_use_log2)
    # Ragged final rows (the decode tile's RAGGED_Q leg): (B+1,) int32 / int64 ragged
    # offsets of Q, O and Stats in ELEMENTS and their elements-per-token
    # divisors.  The partials stay dense; only the final O / LSE rows move to
    # ``offset[b] / div + q_row`` on the packed [1, T, ...] outputs, and a row
    # at or past this sequence's Q length (offset[b+1] - offset[b]) / div --
    # every row of a zero-length sequence -- is not written at all.  None
    # (None-specialized) keeps the dense (batch, q_row) placement.
    ragged_q: Optional[cute.Tensor] = None,
    ragged_o: Optional[cute.Tensor] = None,
    ragged_lse: Optional[cute.Tensor] = None,
    ragged_q_div: cutlass.Int64 = 1,
    ragged_o_div: cutlass.Int64 = 1,
    ragged_lse_div: cutlass.Int64 = 1,
    # Packed token capacities of the final O and Stats buffers (ragged only): a
    # row whose ragged placement lands at or past them is not written, so a
    # short buffer or a bad offset can never store outside the caller's bytes.
    ragged_o_cap: cutlass.Int32 = 0,
    ragged_lse_cap: cutlass.Int32 = 0,
    # The gate G of the graph's ``mul(O_v, sigmoid(G))`` tail, [B, S_q, H, D] in
    # the caller's strides (the dense placement only), applied to the fp32 merged
    # value before the single cast -- see the module docstring.  None
    # (None-specialized) is the plain reduction; its traced code is unchanged.
    gate: Optional[cute.Tensor] = None,
    # int32 (B,) per-batch Q lengths of the dense padded-Q trim, present only on the sink entry that carries them: a row at or past its batch's
    # length folds no sink (see _row_sink_logit).  None (None-specialized) folds the head's sink on every row.
    q_lens: Optional[cute.Tensor] = None,
    # The wide lean group (G_S up-front splits) is its own specialization: its register footprint (the kernel reaches the 128-register cap and spills)
    # would slow the serial walk this kernel also runs, so only the few-split decode launch asks for it.
    lean_wide: cutlass.Constexpr[bool] = False,
    # Programmatic dependent launch (the launch attribute is set by _launch_combine under the same flag): the partial slabs are
    # the split tile's output, so nothing is read above the wait; the pass has no prologue to overlap, so the dependent grid (the
    # out projection in the gated block) is released right after it and stages its own while this pass runs.
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(32)
    q_row = cute.arch.block_idx()[0] * cutlass.Int32(ROWS_PER_BLOCK) + tidx // cutlass.Int32(32)
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    # A warp past S_q (last block only) has no row; there is no block-level
    # barrier below, so it simply falls through. Everything inside is
    # warp-uniform except the per-lane column guard.
    if q_row < s_q:
        op = cutlass.make_array_view(o_partial)
        lp = cutlass.make_array_view(lse_partial)
        oo = cutlass.make_array_view(o_out)

        # Where the recombined row lands: dense (batch, q_row), or the ragged
        # placement on the packed outputs (batch coord 0).
        o_batch = batch
        o_tok = q_row
        lse_batch = batch
        lse_tok = q_row
        row_live = cutlass.Int32(1) == cutlass.Int32(1)
        lse_live = row_live
        if cutlass.const_expr(ragged_q is not None):
            # Offsets are int32 or int64 elements; divide in 64 bits (a large packed
            # buffer's element offset can exceed 2^31) and keep the token index in 32.
            # Bounds are checked in 64 bits BEFORE narrowing: a negative or wrapped
            # offset must fail the capacity test, not alias a valid token after the cast.
            rq = cutlass.make_array_view(ragged_q)
            q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
            q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
            in_seq = cutlass.Int64(q_row) < q_len64
            o_batch = cutlass.Int32(0)
            o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
            row_live = in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
            o_tok = cutlass.Int32(o_tok64)
            lse_live = row_live
            if cutlass.const_expr(ragged_lse is not None):
                lse_batch = cutlass.Int32(0)
                lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
                lse_live = in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
                lse_tok = cutlass.Int32(lse_tok64)

        neg_inf = cutlass.Float32(NEG_INF)
        zero = cutlass.Float32(0.0)
        # Pre-declared: both paths assign them and the tail below reads them after the join.
        amax_local = cutlass.Float32(0.0)
        m_safe = zero
        den = zero
        all_dead = zero > zero
        o_base = o_partial.iterator.toint()
        # fp32 partials with 16-byte rows: the lean path.  Lane l owns columns 4l..4l+3 of
        # every split; each (b, h, q_row) base offset is computed once and the per-split
        # address is a single 64-bit add (the split axis is the leading, batch-stacked mode).
        lean = zero > zero
        if cutlass.const_expr(o_partial.element_type == cutlass.Float32):
            lean = (
                ((d_v & cutlass.Int32(3)) == cutlass.Int32(0))
                & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
                & (n_splits <= cutlass.Int32(G_S if lean_wide else G_LEAN_SMALL))
            )
        if lean:
            lse_base = lse_partial.iterator.toint() + cutlass.Int64(cute.crd2idx((batch, head, q_row), lse_partial.layout)) * 4
            lse_step = cutlass.Int64(lse_partial.stride[0]) * cutlass.Int64(n_batch) * 4
            o_step = cutlass.Int64(o_partial.stride[0]) * cutlass.Int64(n_batch) * 4
            s_last = n_splits - cutlass.Int32(1)
            sink_row = None
            if cutlass.const_expr(sinks is not None):
                sink_row = _row_sink_logit(sinks, q_lens, head, batch, q_row)
            out_base = o_out.iterator.toint()
            st_ok = zero > zero
            if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
                _st_strides = cutlass.Int64(o_out.stride[0]) | cutlass.Int64(o_out.stride[1]) | cutlass.Int64(o_out.stride[2])
                st_ok = (
                    ((out_base & cutlass.Int64(7)) == cutlass.Int64(0))
                    & (cutlass.Int64(o_out.stride[3]) == cutlass.Int64(1))
                    & ((_st_strides & cutlass.Int64(3)) == cutlass.Int64(0))
                )
            q_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(scale_o is not None):
                q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
            # Gate in the combine: the 0.5 handed to the packed multiply must be opaque (a constant
            # float into inline_ptx ICEs libNVVM).  Hoisted once per row, like the fused kernels do
            # per tile; traced only on the gated entry, so the plain entries' code is unchanged.
            if cutlass.const_expr(gate is not None):
                gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
            for cbase in cutlass.range(0, d_v, 128, unroll=1):
                d0 = cbase + lane * cutlass.Int32(4)
                if d0 < d_v:
                    o_row_base = o_base + cutlass.Int64(cute.crd2idx((batch, q_row, head, d0), o_partial.layout)) * 4
                    den = zero
                    acc0 = zero
                    acc1 = zero
                    acc2 = zero
                    acc3 = zero
                    # Two group widths on the wide specialization, one per row of splits: the narrow group's loads are the whole axis for the common few
                    # splits, the wide one covers up to G_S.  The plain specialization has the narrow group only.
                    if cutlass.const_expr(lean_wide):
                        if n_splits <= cutlass.Int32(G_LEAN_SMALL):
                            all_dead, den, acc0, acc1, acc2, acc3, m_safe = _lean_accumulate(
                                G_LEAN_SMALL, lse_base, lse_step, o_row_base, o_step, s_last, n_splits, sink_row
                            )
                        else:
                            all_dead, den, acc0, acc1, acc2, acc3, m_safe = _lean_accumulate(
                                G_S, lse_base, lse_step, o_row_base, o_step, s_last, n_splits, sink_row
                            )
                    else:
                        all_dead, den, acc0, acc1, acc2, acc3, m_safe = _lean_accumulate(
                            G_LEAN_SMALL, lse_base, lse_step, o_row_base, o_step, s_last, n_splits, sink_row
                        )
                    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
                    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))
                    outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
                    if cutlass.const_expr(gate is not None):
                        # The gated prefill kernels' epilogue arithmetic, verbatim
                        # (_common_blackwell.gate_epilogue_pairs): per pair one
                        # FMUL2 (g / 2), two MUFU.TANH, one FFMA2 (h * t + h), on the
                        # fp32 merged value; then the dead-row SELECT per element,
                        # AFTER the fma -- a dead row is exactly 0 whatever G holds
                        # (a NaN gate times the zero h would be NaN; a select is not).
                        # sigmoid's 0.5 folds into the scale (``h = acc * (inv_den / 2)``:
                        # exact, a power-of-two scaling of the plain output).
                        inv_den_half = inv_den * cutlass.Float32(0.5)
                        gv = cutlass.make_array_view(gate)
                        g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                        g1 = zero
                        g2 = zero
                        g3 = zero
                        if d0 + cutlass.Int32(1) < d_v:
                            g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                        if d0 + cutlass.Int32(2) < d_v:
                            g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                        if d0 + cutlass.Int32(3) < d_v:
                            g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
                        h0 = acc0 * inv_den_half
                        h1 = acc1 * inv_den_half
                        h2 = acc2 * inv_den_half
                        h3 = acc3 * inv_den_half
                        s0, s1 = fmul2(g0, g1, gate_half, gate_half)
                        s2, s3 = fmul2(g2, g3, gate_half, gate_half)
                        t0 = cute.math.tanh(s0, approx=True)
                        t1 = cute.math.tanh(s1, approx=True)
                        t2 = cute.math.tanh(s2, approx=True)
                        t3 = cute.math.tanh(s3, approx=True)
                        y0, y1 = ffma2(t0, t1, h0, h1, h0, h1)
                        y2, y3 = ffma2(t2, t3, h2, h3, h2, h3)
                        outs = (
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
                        )
                    vals = []
                    for i in cutlass.range_constexpr(4):
                        o_val = outs[i]
                        if cutlass.const_expr(amax_o is not None):
                            # Measured before scale_o, so this is already the pre-quant
                            # amax and needs no post-hoc divide.
                            amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
                        if cutlass.const_expr(scale_o is not None):
                            o_val = o_val * q_scale
                        vals.append(o_val)
                    if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
                        if st_ok & row_live & (d0 + cutlass.Int32(3) < d_v):
                            _cvt = "bf16x2" if cutlass.const_expr(o_out.element_type == cutlass.BFloat16) else "f16x2"
                            p01, p23 = nvvm.inline_ptx(
                                f"cvt.rn.{_cvt}.f32 $0, $3, $2;\ncvt.rn.{_cvt}.f32 $1, $5, $4;",
                                write_only_types=[cutlass.Int32, cutlass.Int32],
                                read_only_args=[vals[0], vals[1], vals[2], vals[3]],
                            )
                            o_idx = cute.crd2idx((o_batch, o_tok, head, d0), o_out.layout)
                            st_global_v2(out_base + cutlass.Int64(o_idx) * 2, (p01, p23), cutlass.Int32)
                        else:
                            for i in cutlass.range_constexpr(4):
                                if row_live & (d0 + cutlass.Int32(i) < d_v):
                                    oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
                    else:
                        for i in cutlass.range_constexpr(4):
                            # Index all modes: ArrayView's row slice is a pointer and drops
                            # the final mode's stride, so a subsequent [d0] would assume
                            # contiguous D.
                            if row_live & (d0 + cutlass.Int32(i) < d_v):
                                oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        else:
            # --- pass 1: M = max_s lse_s, then den = sum_s exp(lse_s - M) ---
            # Every lane redundantly walks the (very short) split axis; the values
            # are warp-uniform and hit L1, which is cheaper than staging them.
            neg_inf = cutlass.Float32(NEG_INF)
            zero = cutlass.Float32(0.0)
            n_main, n_all = _serial_legs(n_splits)
            m = cutlass.Float32(NEG_INF)
            for s in cutlass.range(0, n_main, 1, unroll=_SERIAL_UNROLL):
                lse_row = lp[batch + s * n_batch, head, :]
                m = cute.math.max(m, cutlass.Float32(lse_row[q_row]))
            for s in cutlass.range(n_main, n_all, 1, unroll=_SERIAL_TAIL_UNROLL):
                lse_row = lp[batch + s * n_batch, head, :]
                m = cute.math.max(m, cutlass.Float32(lse_row[q_row]))

            # All splits dead (every row fully masked): emit O := 0 / lse := -inf
            # rather than exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
            # With a sink the reference max is max(m, sink), which is finite even on a
            # keyless row, and the denominator starts with the sink column's weight
            # exp(sink - M): that row then yields O = 0 and LSE = sink + log(1) = sink,
            # the unsplit kernels' keyless-row contract (a padded row has no sink: it stays dead).
            all_dead = m == cutlass.Float32(NEG_INF)
            m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))
            den = cutlass.Float32(0.0)
            if cutlass.const_expr(sinks is not None):
                all_dead, m_safe, den = _fold_sink(m, _row_sink_logit(sinks, q_lens, head, batch, q_row))

            # A dead split (empty KV range) carries lse_s = -inf and must contribute
            # nothing.  Its weight is dropped with a SELECT, not by trusting the
            # arithmetic: 0 * x is NaN for a non-finite x and a fastmath exp(-inf)
            # is only approximately zero.  (Observed: d512 with 5 KV tiles over 8
            # splits produced NaN without the guard.)
            for s in cutlass.range(0, n_main, 1, unroll=_SERIAL_UNROLL):
                lse_row = lp[batch + s * n_batch, head, :]
                lse_s = cutlass.Float32(lse_row[q_row])
                live = lse_s > neg_inf
                e = cute.math.exp(lse_s - m_safe, fastmath=True)
                den = den + cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
            for s in cutlass.range(n_main, n_all, 1, unroll=_SERIAL_TAIL_UNROLL):
                lse_row = lp[batch + s * n_batch, head, :]
                lse_s = cutlass.Float32(lse_row[q_row])
                live = lse_s > neg_inf
                e = cute.math.exp(lse_s - m_safe, fastmath=True)
                den = den + cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))

            inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
            inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))

            # --- pass 2: O = sum_s w_s O_s / den, accumulated in fp32 ---
            # The partial slab is compact [S*B, S_q, H, d_v]; lane l owns four
            # contiguous columns of every split (one 16-byte load for fp32 partials
            # when all four are inside the row); the split walk is branch-free and
            # unrolled _SERIAL_UNROLL deep (two legs, see _serial_legs) so that many
            # splits' loads are in flight together: four at a time cost one L2
            # round trip per four splits (0.27 us a split on B300 d128, 11 us at 32).
            o_base = o_partial.iterator.toint()
            # The 16-byte loads need d_v % 4 == 0 (row starts stay 16-byte aligned)
            # and a 16-byte aligned slab base: the pointer entries only promise the
            # element alignment and take a runtime d_v, so decide at runtime (one
            # warp-uniform test); otherwise every lane takes the element path.
            vec_ok = ((d_v & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
            amax_local = cutlass.Float32(0.0)
            q_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(scale_o is not None):
                q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
            # Gate in the combine: sigmoid's two constants fold into the scale
            # (``h = acc * (inv_den / 2)``: exact, a power-of-two scaling of the
            # plain output) and into the ``h * t + h`` fma; the 0.5 handed to the
            # packed multiply must be opaque (a constant float into inline_ptx ICEs
            # libNVVM).  Hoisted once per row, like the fused kernels do per tile;
            # traced only on the gated entry, so the plain entries' code is unchanged.
            if cutlass.const_expr(gate is not None):
                inv_den_half = inv_den * cutlass.Float32(0.5)
                gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
            for cbase in cutlass.range(0, d_v, 128, unroll=1):
                d0 = cbase + lane * cutlass.Int32(4)
                if d0 < d_v:
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    acc0, acc1, acc2, acc3 = _accumulate_splits(
                        cutlass.Int32(0),
                        n_main,
                        _SERIAL_UNROLL,
                        o_partial,
                        lse_partial,
                        o_base,
                        vec_ok,
                        batch,
                        n_batch,
                        head,
                        q_row,
                        d0,
                        d_v,
                        m_safe,
                        neg_inf,
                        zero,
                        acc0,
                        acc1,
                        acc2,
                        acc3,
                    )
                    acc0, acc1, acc2, acc3 = _accumulate_splits(
                        n_main,
                        n_all,
                        _SERIAL_TAIL_UNROLL,
                        o_partial,
                        lse_partial,
                        o_base,
                        vec_ok,
                        batch,
                        n_batch,
                        head,
                        q_row,
                        d0,
                        d_v,
                        m_safe,
                        neg_inf,
                        zero,
                        acc0,
                        acc1,
                        acc2,
                        acc3,
                    )
                    outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
                    if cutlass.const_expr(gate is not None):
                        # The gated prefill kernels' epilogue arithmetic, verbatim
                        # (_common_blackwell.gate_epilogue_pairs): per pair one
                        # FMUL2 (g / 2), two MUFU.TANH, one FFMA2 (h * t + h), on the
                        # fp32 merged value; then the dead-row SELECT per element,
                        # AFTER the fma -- a dead row is exactly 0 whatever G holds
                        # (a NaN gate times the zero h would be NaN; a select is not).
                        gv = cutlass.make_array_view(gate)
                        g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                        g1 = zero
                        g2 = zero
                        g3 = zero
                        if d0 + cutlass.Int32(1) < d_v:
                            g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                        if d0 + cutlass.Int32(2) < d_v:
                            g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                        if d0 + cutlass.Int32(3) < d_v:
                            g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
                        h0 = acc0 * inv_den_half
                        h1 = acc1 * inv_den_half
                        h2 = acc2 * inv_den_half
                        h3 = acc3 * inv_den_half
                        s0, s1 = fmul2(g0, g1, gate_half, gate_half)
                        s2, s3 = fmul2(g2, g3, gate_half, gate_half)
                        t0 = cute.math.tanh(s0, approx=True)
                        t1 = cute.math.tanh(s1, approx=True)
                        t2 = cute.math.tanh(s2, approx=True)
                        t3 = cute.math.tanh(s3, approx=True)
                        y0, y1 = ffma2(t0, t1, h0, h1, h0, h1)
                        y2, y3 = ffma2(t2, t3, h2, h3, h2, h3)
                        outs = (
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
                        )
                    for i in cutlass.range_constexpr(4):
                        o_val = outs[i]
                        if cutlass.const_expr(amax_o is not None):
                            # Measured before scale_o, so this is already the pre-quant
                            # amax and needs no post-hoc divide.
                            amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
                        if cutlass.const_expr(scale_o is not None):
                            o_val = o_val * q_scale
                        # Index all modes: ArrayView's row slice is a pointer and drops
                        # the final mode's stride, so a subsequent [d0] would assume
                        # contiguous D.
                        if row_live & (d0 + cutlass.Int32(i) < d_v):
                            oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = o_val.to(o_out.element_type)

        # amax: reduce across the warp, then ONE atomic per warp.  The value is
        # non-negative, so its fp32 bit pattern orders the same as the float and
        # an integer atomicMax is exact -- the same trick the kernels' own
        # epilogues use.
        if cutlass.const_expr(amax_o is not None):
            for off in cutlass.range_constexpr(5):
                amax_local = cute.math.max(amax_local, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_local, 16 >> off, 31, kind=nvvm.Shfl.BFLY)))
            if lane == cutlass.Int32(0):
                _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))

        # --- the recombined LSE (only when the caller asked for Stats) ---
        if cutlass.const_expr(lse_out is not None):
            if (lane == cutlass.Int32(0)) & lse_live:
                lo = cutlass.make_array_view(lse_out)
                lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
                lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(NEG_INF).ir_value(), lse_val.ir_value()))
                # Base-2 Stats (stats_use_log2): the partials stay natural (the
                # merge above needs them); only the final value converts.
                if cutlass.const_expr(stats_log2):
                    lse_val = lse_val * cutlass.Float32(1.4426950408889634)
                lo[lse_batch, head, lse_tok] = lse_val


_combine_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _combine_decode_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] — split-major partial O
    lse_partial: cute.Tensor,  # [S*B, H, S_q]   — split-major partial LSE
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    # 1-element fp32 amax of the RECOMBINED O.  The per-split epilogues cannot
    # compute this: each sees only its own partial, and O is a convex
    # combination of those, so a max over partials over-reports (~2.9x at 8
    # splits).  The FP8 kernels therefore skip their in-kernel amax write when
    # SPLIT_KV > 1 and leave it to this pass.  None-specialized off otherwise.
    amax_o: Optional[cute.Tensor],
    # 1-element fp32 output scale, present only when O is stored quantized.
    # The split kernels leave it out of their epilogue and write UNSCALED
    # partials; this pass applies it at the single cast, so the FP8 rounding
    # happens once, on the recombined value.
    scale_o: Optional[cute.Tensor],
    # fp32 [H] per-head attention-sink logits, folded ONCE here (the split
    # partials are sink-free).  None-specialized off without a sink.
    sinks: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    s_q: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],  # write the FINAL LSE in base 2 (stats_use_log2)
    # Ragged final rows (the decode tile's RAGGED_Q leg): (B+1,) int32 / int64 ragged
    # offsets of Q, O and Stats in ELEMENTS and their elements-per-token
    # divisors.  The partials stay dense; only the final O / LSE rows move to
    # ``offset[b] / div + q_row`` on the packed [1, T, ...] outputs, and a row
    # at or past this sequence's Q length (offset[b+1] - offset[b]) / div --
    # every row of a zero-length sequence -- is not written at all.  None
    # (None-specialized) keeps the dense (batch, q_row) placement.
    ragged_q: Optional[cute.Tensor] = None,
    ragged_o: Optional[cute.Tensor] = None,
    ragged_lse: Optional[cute.Tensor] = None,
    ragged_q_div: cutlass.Int64 = 1,
    ragged_o_div: cutlass.Int64 = 1,
    ragged_lse_div: cutlass.Int64 = 1,
    # Packed token capacities of the final O and Stats buffers (ragged only): a
    # row whose ragged placement lands at or past them is not written, so a
    # short buffer or a bad offset can never store outside the caller's bytes.
    ragged_o_cap: cutlass.Int32 = 0,
    ragged_lse_cap: cutlass.Int32 = 0,
    # The gate G of the graph's ``mul(O_v, sigmoid(G))`` tail, [B, S_q, H, D] in the
    # caller's strides (dense placement only): the serial form's arithmetic, applied by
    # the storing warp to the fp32 merged value.  None (None-specialized) is the plain reduction.
    gate: Optional[cute.Tensor] = None,
    # The per-batch Q lengths of the sink entry that carries them (see _combine_kernel).
    q_lens: Optional[cute.Tensor] = None,
    # Programmatic dependent launch: _combine_kernel's pair, traced under the same flag the launch attribute is set by -- the
    # wait before the first global read (the ragged offsets and the partial slabs alike), the release right after it.
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """The decode form (S_q < 4): one block of four warps per row set.

    Row mapping: a block serves ``rows_blk = min(4, S_q)`` consecutive q rows,
    and the ``4 // rows_blk`` warps that share a row split the SPLIT AXIS
    between them (warp w of a row owns splits w, w + k, w + 2k, ...), so a
    single-token decode row has four warps pulling its partials instead of
    one.  Within a warp the split axis is lane-strided for the LSE passes
    (one load per lane per 32 splits, warp-shuffle max / sum) and the per-split
    weights reach the O pass through shuffles, so no pass walks the splits one
    dependent global load at a time.  The warps of a row then merge their
    (max, sum, acc) through shared memory and warp 0 of the row stores.

    Every warp executes every barrier (rows past S_q and lanes past d_v only
    skip loads and stores), so the block-uniform barriers below are safe.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    rows_blk = _min_i32(cutlass.Int32(ROWS_PER_BLOCK), s_q)
    warps_per_row = cutlass.Int32(ROWS_PER_BLOCK) // rows_blk
    row_in_blk = warp // warps_per_row
    warp_in_row = warp % warps_per_row
    q_row = cute.arch.block_idx()[0] * rows_blk + row_in_blk
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]
    active = (q_row < s_q) & (row_in_blk < rows_blk)
    shared_row = warps_per_row > cutlass.Int32(1)

    smem = SmemAllocator()
    # per-warp row max / row sum exchange, then the per-lane 4-column partial sums
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2 * ROWS_PER_BLOCK,)), byte_alignment=16)
    s_acc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((ROWS_PER_BLOCK * 32 * 4,)), byte_alignment=16)

    op = cutlass.make_array_view(o_partial)
    lp = cutlass.make_array_view(lse_partial)
    oo = cutlass.make_array_view(o_out)
    neg_inf = cutlass.Float32(NEG_INF)
    zero = cutlass.Float32(0.0)

    # Where the recombined row lands: dense (batch, q_row), or the ragged
    # placement on the packed outputs (batch coord 0).
    o_batch = batch
    o_tok = q_row
    lse_batch = batch
    lse_tok = q_row
    row_live = active
    lse_live = active
    if cutlass.const_expr(ragged_q is not None):
        # Offsets are int32 or int64 elements; divide in 64 bits (a large packed
        # buffer's element offset can exceed 2^31) and keep the token index in 32.
        # Bounds are checked in 64 bits BEFORE narrowing: a negative or wrapped
        # offset must fail the capacity test, not alias a valid token after the cast.
        rq = cutlass.make_array_view(ragged_q)
        q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
        q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
        in_seq = cutlass.Int64(q_row) < q_len64
        o_batch = cutlass.Int32(0)
        o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
        row_live = active & in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
        o_tok = cutlass.Int32(o_tok64)
        lse_live = row_live
        if cutlass.const_expr(ragged_lse is not None):
            lse_batch = cutlass.Int32(0)
            lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
            lse_live = active & in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
            lse_tok = cutlass.Int32(lse_tok64)

    # This warp's share of the split axis: s = warp_in_row + warps_per_row * i,
    # i = chunk * 32 + lane.  (All of it when the row has one warp.)
    n_sub = (n_splits - warp_in_row + warps_per_row - cutlass.Int32(1)) // warps_per_row

    # --- pass 1: the row max over the splits (lane-strided, then warp max) ---
    m_lane = neg_inf
    for c in cutlass.range(0, n_sub, 32, unroll=1):
        i = c + lane
        if active & (i < n_sub):
            s = warp_in_row + warps_per_row * i
            m_lane = cute.math.max(m_lane, cutlass.Float32(lp[batch + s * n_batch, head, q_row]))
    m = m_lane
    for bit in cutlass.range_constexpr(5):
        m = cute.math.max(m, cute.arch.shuffle_sync_bfly(m, 1 << bit))
    if shared_row:
        if lane == cutlass.Int32(0):
            s_red[warp] = m
        cute.arch.barrier()
        m = neg_inf
        for k in cutlass.range(0, warps_per_row, 1, unroll=1):
            m = cute.math.max(m, cutlass.Float32(s_red[row_in_blk * warps_per_row + k]))
        cute.arch.barrier()  # s_red is reused for the sums below

    # All splits dead (every row fully masked): emit O := 0 / lse := -inf
    # rather than exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
    # With a sink the reference max is max(m, sink), which is finite even on a
    # keyless row, and the denominator starts with the sink column's weight
    # exp(sink - M): that row then yields O = 0 and LSE = sink + log(1) = sink,
    # the unsplit kernels' keyless-row contract (a padded row has no sink: it stays dead).
    all_dead = m == neg_inf
    m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m.ir_value()))
    sink_term = zero
    if cutlass.const_expr(sinks is not None):
        all_dead, m_safe, sink_term = _fold_sink(m, _row_sink_logit(sinks, q_lens, head, batch, q_row))

    # --- pass 2: den = sum_s exp(lse_s - M) (lane-strided, warp sum, row sum) ---
    # A dead split (empty KV range) carries lse_s = -inf and must contribute
    # nothing.  Its weight is dropped with a SELECT, not by trusting the
    # arithmetic: 0 * x is NaN for a non-finite x and a fastmath exp(-inf)
    # is only approximately zero.  (Observed: d512 with 5 KV tiles over 8
    # splits produced NaN without the guard.)
    den_lane = zero
    for c in cutlass.range(0, n_sub, 32, unroll=1):
        i = c + lane
        if active & (i < n_sub):
            s = warp_in_row + warps_per_row * i
            lse_s = cutlass.Float32(lp[batch + s * n_batch, head, q_row])
            e = cute.math.exp(lse_s - m_safe, fastmath=True)
            den_lane = den_lane + cutlass.Float32(arith.select((lse_s > neg_inf).ir_value(), e.ir_value(), zero.ir_value()))
    den = den_lane
    for bit in cutlass.range_constexpr(5):
        den = den + cute.arch.shuffle_sync_bfly(den, 1 << bit)
    if shared_row:
        if lane == cutlass.Int32(0):
            s_red[warp] = den
        cute.arch.barrier()
        den = zero
        for k in cutlass.range(0, warps_per_row, 1, unroll=1):
            den = den + cutlass.Float32(s_red[row_in_blk * warps_per_row + k])
    den = den + sink_term
    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))
    # Gate in the combine: the serial form's constants (see _combine_kernel), hoisted once per row.
    if cutlass.const_expr(gate is not None):
        inv_den_half = inv_den * cutlass.Float32(0.5)
        gate_half = opaque_f32_zero() + cutlass.Float32(0.5)

    # --- pass 3: O = sum_s w_s O_s / den, accumulated in fp32 ---
    # The partial slab is compact [S*B, S_q, H, d_v]; lane l owns four
    # contiguous columns of every split (one 16-byte load for fp32 partials
    # when all four are inside the row).  The per-split weight comes from the
    # lane that computed it (one shuffle), and the split loop skips a dead split
    # entirely (its payload may be non-finite); w is warp-uniform, so the test is.
    o_base = o_partial.iterator.toint()
    # The 16-byte loads need d_v % 4 == 0 (row starts stay 16-byte aligned)
    # and a 16-byte aligned slab base: the pointer entries only promise the
    # element alignment and take a runtime d_v, so decide at runtime (one
    # warp-uniform test); otherwise every lane takes the element path.
    vec_ok = ((d_v & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
    amax_local = zero
    q_scale = cutlass.Float32(1.0)
    if cutlass.const_expr(scale_o is not None):
        q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
    for cbase in cutlass.range(0, d_v, 128, unroll=1):
        d0 = cbase + lane * cutlass.Int32(4)
        col_ok = active & (d0 < d_v)
        acc0 = zero
        acc1 = zero
        acc2 = zero
        acc3 = zero
        for c in cutlass.range(0, n_sub, 32, unroll=1):
            i_lane = c + lane
            w_lane = zero
            if active & (i_lane < n_sub):
                s_l = warp_in_row + warps_per_row * i_lane
                lse_l = cutlass.Float32(lp[batch + s_l * n_batch, head, q_row])
                e_l = cute.math.exp(lse_l - m_safe, fastmath=True)
                w_lane = cutlass.Float32(arith.select((lse_l > neg_inf).ir_value(), e_l.ir_value(), zero.ir_value()))
            n_here = _min_i32(cutlass.Int32(32), n_sub - c)
            for j in cutlass.range(0, n_here, 1, unroll=_DECODE_UNROLL):
                # The weight is warp-uniform (one shuffle).  The loads are NOT
                # predicated on it: a branch per split would serialize the
                # unrolled loads behind each other's latency (measured: the
                # branched form ran 128 splits in 12.6 us, this form in less);
                # a dead split's payload (possibly non-finite) is dropped by the
                # select below, exactly as in the serial kernel.
                w = cute.arch.shuffle_sync(w_lane, j)
                live = w > zero
                if cutlass.const_expr(True):
                    s = warp_in_row + warps_per_row * (c + j)
                    e0 = zero
                    e1 = zero
                    e2 = zero
                    e3 = zero
                    if col_ok:
                        # Element loads through the view, each bounded by the row, serve
                        # the half partials (flavors whose split epilogue keeps the staged
                        # TMA-store O path, e.g. d512) and any d_v / base that cannot take
                        # aligned 16-byte loads (legal through the runtime-shape pointer
                        # entries); aligned fp32 partials take one 16-byte load per split.
                        if cutlass.const_expr(o_partial.element_type == cutlass.Float32):
                            if vec_ok:
                                idx = cute.crd2idx((batch + s * n_batch, q_row, head, d0), o_partial.layout)
                                v = ld_global_v4(o_base + cutlass.Int64(idx) * 4, cutlass.Float32)
                                e0 = cutlass.Float32(v[0])
                                e1 = cutlass.Float32(v[1])
                                e2 = cutlass.Float32(v[2])
                                e3 = cutlass.Float32(v[3])
                            else:
                                o_row = op[batch + s * n_batch, q_row, head, :]
                                e0 = cutlass.Float32(o_row[d0])
                                if d0 + cutlass.Int32(1) < d_v:
                                    e1 = cutlass.Float32(o_row[d0 + cutlass.Int32(1)])
                                if d0 + cutlass.Int32(2) < d_v:
                                    e2 = cutlass.Float32(o_row[d0 + cutlass.Int32(2)])
                                if d0 + cutlass.Int32(3) < d_v:
                                    e3 = cutlass.Float32(o_row[d0 + cutlass.Int32(3)])
                        else:
                            o_row = op[batch + s * n_batch, q_row, head, :]
                            e0 = cutlass.Float32(o_row[d0])
                            if d0 + cutlass.Int32(1) < d_v:
                                e1 = cutlass.Float32(o_row[d0 + cutlass.Int32(1)])
                            if d0 + cutlass.Int32(2) < d_v:
                                e2 = cutlass.Float32(o_row[d0 + cutlass.Int32(2)])
                            if d0 + cutlass.Int32(3) < d_v:
                                e3 = cutlass.Float32(o_row[d0 + cutlass.Int32(3)])
                    v0 = cutlass.Float32(arith.select(live.ir_value(), e0.ir_value(), zero.ir_value()))
                    v1 = cutlass.Float32(arith.select(live.ir_value(), e1.ir_value(), zero.ir_value()))
                    v2 = cutlass.Float32(arith.select(live.ir_value(), e2.ir_value(), zero.ir_value()))
                    v3 = cutlass.Float32(arith.select(live.ir_value(), e3.ir_value(), zero.ir_value()))
                    acc0 = acc0 + w * v0
                    acc1 = acc1 + w * v1
                    acc2 = acc2 + w * v2
                    acc3 = acc3 + w * v3
        if shared_row:
            # merge the warps of this row: lane-column partial sums through SMEM
            base_w = (warp * cutlass.Int32(32) + lane) * cutlass.Int32(4)
            s_acc[base_w] = acc0
            s_acc[base_w + cutlass.Int32(1)] = acc1
            s_acc[base_w + cutlass.Int32(2)] = acc2
            s_acc[base_w + cutlass.Int32(3)] = acc3
            cute.arch.barrier()
            if warp_in_row == cutlass.Int32(0):
                acc0 = zero
                acc1 = zero
                acc2 = zero
                acc3 = zero
                for k in cutlass.range(0, warps_per_row, 1, unroll=1):
                    base_k = ((row_in_blk * warps_per_row + k) * cutlass.Int32(32) + lane) * cutlass.Int32(4)
                    acc0 = acc0 + cutlass.Float32(s_acc[base_k])
                    acc1 = acc1 + cutlass.Float32(s_acc[base_k + cutlass.Int32(1)])
                    acc2 = acc2 + cutlass.Float32(s_acc[base_k + cutlass.Int32(2)])
                    acc3 = acc3 + cutlass.Float32(s_acc[base_k + cutlass.Int32(3)])
            cute.arch.barrier()  # s_acc is reused by the next column chunk
        if warp_in_row == cutlass.Int32(0):
            outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
            if cutlass.const_expr(gate is not None):
                # The serial form's gate arithmetic, verbatim (one FMUL2, two MUFU.TANH, one FFMA2 per
                # pair on the fp32 merged value; the dead-row SELECT after the fma); the loads are
                # bounded by this lane's columns, like the stores below.
                gv = cutlass.make_array_view(gate)
                g0 = zero
                g1 = zero
                g2 = zero
                g3 = zero
                if col_ok:
                    g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                    if d0 + cutlass.Int32(1) < d_v:
                        g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                    if d0 + cutlass.Int32(2) < d_v:
                        g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                    if d0 + cutlass.Int32(3) < d_v:
                        g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
                h0 = acc0 * inv_den_half
                h1 = acc1 * inv_den_half
                h2 = acc2 * inv_den_half
                h3 = acc3 * inv_den_half
                s0, s1 = fmul2(g0, g1, gate_half, gate_half)
                s2, s3 = fmul2(g2, g3, gate_half, gate_half)
                t0 = cute.math.tanh(s0, approx=True)
                t1 = cute.math.tanh(s1, approx=True)
                t2 = cute.math.tanh(s2, approx=True)
                t3 = cute.math.tanh(s3, approx=True)
                y0, y1 = ffma2(t0, t1, h0, h1, h0, h1)
                y2, y3 = ffma2(t2, t3, h2, h3, h2, h3)
                outs = (
                    cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                    cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                    cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                    cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
                )
            for i in cutlass.range_constexpr(4):
                o_val = outs[i]
                if cutlass.const_expr(amax_o is not None):
                    # Measured before scale_o, so this is already the pre-quant
                    # amax and needs no post-hoc divide.
                    if col_ok & (d0 + cutlass.Int32(i) < d_v):
                        amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
                if cutlass.const_expr(scale_o is not None):
                    o_val = o_val * q_scale
                # Index all modes: ArrayView's row slice is a pointer and drops
                # the final mode's stride, so a subsequent [d0] would assume
                # contiguous D.
                if row_live & (d0 + cutlass.Int32(i) < d_v):
                    oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = o_val.to(o_out.element_type)

    # amax: reduce across the warp, then ONE atomic per warp.  The value is
    # non-negative, so its fp32 bit pattern orders the same as the float and
    # an integer atomicMax is exact -- the same trick the kernels' own
    # epilogues use.  Only a row's storing warp holds a non-zero value.
    if cutlass.const_expr(amax_o is not None):
        for off in cutlass.range_constexpr(5):
            amax_local = cute.math.max(amax_local, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_local, 16 >> off, 31, kind=nvvm.Shfl.BFLY)))
        if (lane == cutlass.Int32(0)) & (warp_in_row == cutlass.Int32(0)) & active:
            _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))

    # --- the recombined LSE (only when the caller asked for Stats) ---
    if cutlass.const_expr(lse_out is not None):
        if (lane == cutlass.Int32(0)) & (warp_in_row == cutlass.Int32(0)) & lse_live:
            lo = cutlass.make_array_view(lse_out)
            lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
            lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(NEG_INF).ir_value(), lse_val.ir_value()))
            # Base-2 Stats (stats_use_log2): the partials stay natural (the
            # merge above needs them); only the final value converts.
            if cutlass.const_expr(stats_log2):
                lse_val = lse_val * cutlass.Float32(1.4426950408889634)
            lo[lse_batch, head, lse_tok] = lse_val


_combine_decode_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _combine_decode_grid_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] fp32 split-major partial O (the lean path's precondition: fp32, 16-byte rows)
    lse_partial: cute.Tensor,  # [S*B, H, S_q] fp32 split-major partial LSE (natural log)
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    amax_o: Optional[cute.Tensor],  # see _combine_kernel; None on every entry that takes this kernel today
    scale_o: Optional[cute.Tensor],  # see _combine_kernel; None on every entry that takes this kernel today
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    n_chunks: cutlass.Int32,  # ceil(d_v / DECODE_CHUNK_COLS): blockIdx.x = q_row * n_chunks + chunk
    stats_log2: cutlass.Constexpr[bool],
    ragged_q: Optional[cute.Tensor],
    ragged_o: Optional[cute.Tensor],
    ragged_lse: Optional[cute.Tensor],
    ragged_q_div: cutlass.Int64,
    ragged_o_div: cutlass.Int64,
    ragged_lse_div: cutlass.Int64,
    ragged_o_cap: cutlass.Int32,
    ragged_lse_cap: cutlass.Int32,
    gate: Optional[cute.Tensor],
    # Programmatic dependent launch (the launch attribute is set by _launch_combine under the same flag): the partial slabs are
    # the split tile's output, so nothing is read above the wait; the pass has no prologue to overlap, so the dependent grid (the
    # out projection in the gated block) is released right after it -- the lean kernel's pair, on the kernel the served decode
    # plans run (a chain whose combine lacked the sites would silently serialize here).
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """The DECODE-GRID form of the lean fp32-partial combine (module constants ``DECODE_*``; the layout
    + barrier / SMEM notes are in the module docstring; the cc 10.7 adapter's form on a small split
    launch, beside the module's lean / serial / decode forms).  One block per (q_row, head, batch,
    64-column chunk); chain ``c = tidx // 16`` reduces splits ``r * 128 + 16 c .. + 15`` of every round
    ``r`` with its own running max (the online-softmax merge across rounds; a single round for
    ``n_splits <= 128``), lane ``g = tidx % 16`` owns columns ``chunk * 64 + 4 g .. + 3``; the eight
    chains' (max, sum, acc) meet once in SMEM behind one CTA barrier and warp 0 merges them in chain
    order and stores the row.  The same arithmetic as ``_combine_kernel``'s lean group and serial walk
    per split (``w = SELECT(live, exp(lse - m_safe), 0)``, ``v = SELECT(live, v, 0)``, ``den += w``,
    ``acc += w * v`` in split order from +0): for ``n_splits <= 16`` only chain 0 holds splits and the
    output is bitwise those forms'; past 16 the sum is associated differently (fp32-rounding-level
    differences).  Sink-free by construction: a launch with a sink (``sinks`` / ``q_lens``) never
    takes this kernel (``_launch_combine``)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    chain = tidx // cutlass.Int32(DECODE_LANES_PER_CHAIN)
    g = tidx % cutlass.Int32(DECODE_LANES_PER_CHAIN)
    bx = cute.arch.block_idx()[0]
    q_row = bx // n_chunks
    chunk = bx - q_row * n_chunks
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    op = cutlass.make_array_view(o_partial)
    oo = cutlass.make_array_view(o_out)

    # Where the recombined row lands: dense (batch, q_row), or the ragged placement on the packed
    # outputs (batch coord 0) -- _combine_kernel's placement, verbatim.
    o_batch = batch
    o_tok = q_row
    lse_batch = batch
    lse_tok = q_row
    row_live = cutlass.Int32(1) == cutlass.Int32(1)
    lse_live = row_live
    if cutlass.const_expr(ragged_q is not None):
        rq = cutlass.make_array_view(ragged_q)
        q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
        q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
        in_seq = cutlass.Int64(q_row) < q_len64
        o_batch = cutlass.Int32(0)
        o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
        row_live = in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
        o_tok = cutlass.Int32(o_tok64)
        lse_live = row_live
        if cutlass.const_expr(ragged_lse is not None):
            lse_batch = cutlass.Int32(0)
            lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
            lse_live = in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
            lse_tok = cutlass.Int32(lse_tok64)

    neg_inf = cutlass.Float32(NEG_INF)
    zero = cutlass.Float32(0.0)
    # A thread's four columns.  A thread whose columns start past the row (the last chunk of a d_v that
    # is not a multiple of 64) LOADS a valid row start instead (d_v - 4: d_v % 4 == 0 and the slab is
    # 16-byte aligned on this path) and never stores: it still runs the chain so the chain's den
    # stays uniform and it still reaches the barrier.
    d0 = chunk * cutlass.Int32(DECODE_CHUNK_COLS) + g * cutlass.Int32(4)
    col_ok = d0 < d_v
    d_ld = cutlass.Int32(arith.select(col_ok.ir_value(), d0.ir_value(), (d_v - cutlass.Int32(4)).ir_value()))
    o_base = o_partial.iterator.toint()
    lse_base = lse_partial.iterator.toint() + cutlass.Int64(cute.crd2idx((batch, head, q_row), lse_partial.layout)) * 4
    lse_step = cutlass.Int64(lse_partial.stride[0]) * cutlass.Int64(n_batch) * 4
    o_step = cutlass.Int64(o_partial.stride[0]) * cutlass.Int64(n_batch) * 4
    o_row_base = o_base + cutlass.Int64(cute.crd2idx((batch, q_row, head, d_ld), o_partial.layout)) * 4
    s_last = n_splits - cutlass.Int32(1)
    chain_base = chain * cutlass.Int32(DECODE_SPLITS_PER_CHAIN)

    # The chain's running state over its slots: the max over its IN-RANGE splits, the sum of the
    # weights and the four weighted columns -- all +0 / -inf until a live split arrives.
    m_c = neg_inf
    den_c = zero
    acc0 = zero
    acc1 = zero
    acc2 = zero
    acc3 = zero
    for r0 in cutlass.range(0, n_splits, DECODE_SPLITS_PER_ROUND, unroll=1):
        s_base = r0 + chain_base
        # Every load of the round goes out first: the 16 LSEs and the 16 partial-O rows of the chain's
        # slots (a slot past n_splits re-reads the last split; its weight AND its payload are SELECTED to
        # zero below, and it is masked out of the max so an empty chain stays -inf).
        lse_v = []
        o_v = []
        for j in cutlass.range_constexpr(DECODE_SPLITS_PER_CHAIN):
            s_eff64 = cutlass.Int64(cute.math.min(s_base + cutlass.Int32(j), s_last))
            lse_v.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
            o_v.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
        m_r = neg_inf
        for j in cutlass.range_constexpr(DECODE_SPLITS_PER_CHAIN):
            in_range = (s_base + cutlass.Int32(j)) < n_splits
            m_r = cute.math.max(m_r, cutlass.Float32(arith.select(in_range.ir_value(), lse_v[j].ir_value(), neg_inf.ir_value())))
        m_new = cute.math.max(m_c, m_r)
        # m_safe only feeds the exponentials: a chain with no live split so far keeps exp(lse - 0),
        # which the live SELECT below zeroes anyway.
        m_safe_c = cutlass.Float32(arith.select((m_new == neg_inf).ir_value(), zero.ir_value(), m_new.ir_value()))
        if r0 > cutlass.Int32(0):
            # A later round (n_splits > 128 only): the online-softmax rescale of what the chain holds
            # onto its new max.  Never on the first round, so the first round's arithmetic is the lean
            # kernel's own sequence (no multiply ahead of the first add).
            alpha = cutlass.Float32(arith.select((m_c > neg_inf).ir_value(), cute.math.exp(m_c - m_safe_c, fastmath=True).ir_value(), zero.ir_value()))
            den_c = den_c * alpha
            acc0 = acc0 * alpha
            acc1 = acc1 * alpha
            acc2 = acc2 * alpha
            acc3 = acc3 * alpha
        m_c = m_new
        for j in cutlass.range_constexpr(DECODE_SPLITS_PER_CHAIN):
            lse_s = lse_v[j]
            v = o_v[j]
            live = (lse_s > neg_inf) & ((s_base + cutlass.Int32(j)) < n_splits)
            e = cute.math.exp(lse_s - m_safe_c, fastmath=True)
            w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
            v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
            v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
            v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
            v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
            den_c = den_c + w
            acc0 = acc0 + w * v0
            acc1 = acc1 + w * v1
            acc2 = acc2 + w * v2
            acc3 = acc3 + w * v3

    # The SMEM exchange (the module's SMEM table): red_m[c] / red_den[c] by lane 0 of each chain, the
    # four columns by every thread at its own (c, g) slot; the per-thread mapping spreads the banks (16 threads x
    # 16 B = two full 128-B cycles per chain), no descriptor reads it.
    red = cutlass.Array(cutlass.Float32, DECODE_RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    red_ptr = Pointer(red.data_ptr(), dtype=cutlass.Float32)
    if g == cutlass.Int32(0):
        (red_ptr + chain).store(m_c)
        (red_ptr + (chain + cutlass.Int32(DECODE_RED_DEN_OFF))).store(den_c)
    acc_off = cutlass.Int32(DECODE_RED_ACC_OFF) + (chain * cutlass.Int32(DECODE_LANES_PER_CHAIN) + g) * cutlass.Int32(4)
    (red_ptr + acc_off).store(acc0)
    (red_ptr + (acc_off + cutlass.Int32(1))).store(acc1)
    (red_ptr + (acc_off + cutlass.Int32(2))).store(acc2)
    (red_ptr + (acc_off + cutlass.Int32(3))).store(acc3)
    # The ONE barrier of this kernel: every one of the 128 threads arrives exactly once (nothing above
    # exits or diverges around it); the merge below reads what the chains published.
    nvvm.barrier_cta_sync()

    if warp == cutlass.Int32(0):
        # Warp 0 merges the eight chains in chain order and stores the row: lanes 0..15 own the chunk's
        # sixteen column groups; lanes 16..31 run the same arithmetic on a duplicate column group (they
        # keep the warp converged for the amax shuffles) and store nothing.
        lo = lane < cutlass.Int32(DECODE_LANES_PER_CHAIN)
        gm = lane % cutlass.Int32(DECODE_LANES_PER_CHAIN)
        d0 = chunk * cutlass.Int32(DECODE_CHUNK_COLS) + gm * cutlass.Int32(4)
        m = neg_inf
        for c in cutlass.range_constexpr(DECODE_CHAINS):
            m = cute.math.max(m, cutlass.Float32((red_ptr + cutlass.Int32(c)).load()))
        # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather than
        # exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
        all_dead = m == neg_inf
        m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m.ir_value()))
        den = zero
        acc0 = zero
        acc1 = zero
        acc2 = zero
        acc3 = zero
        for c in cutlass.range_constexpr(DECODE_CHAINS):
            m_cc = cutlass.Float32((red_ptr + cutlass.Int32(c)).load())
            den_cc = cutlass.Float32((red_ptr + cutlass.Int32(c + DECODE_RED_DEN_OFF)).load())
            # A chain with no live split (m_cc == -inf) contributes exactly 0 by SELECT; the chain that
            # holds the max contributes with the factor exp(0) == 1 exactly.
            f = cutlass.Float32(arith.select((m_cc > neg_inf).ir_value(), cute.math.exp(m_cc - m_safe, fastmath=True).ir_value(), zero.ir_value()))
            off = cutlass.Int32(DECODE_RED_ACC_OFF + c * DECODE_LANES_PER_CHAIN * 4) + gm * cutlass.Int32(4)
            a0 = cutlass.Float32((red_ptr + off).load())
            a1 = cutlass.Float32((red_ptr + (off + cutlass.Int32(1))).load())
            a2 = cutlass.Float32((red_ptr + (off + cutlass.Int32(2))).load())
            a3 = cutlass.Float32((red_ptr + (off + cutlass.Int32(3))).load())
            den = den + f * den_cc
            acc0 = acc0 + f * a0
            acc1 = acc1 + f * a1
            acc2 = acc2 + f * a2
            acc3 = acc3 + f * a3
        inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
        inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))
        outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
        if cutlass.const_expr(gate is not None):
            # _combine_kernel's gate-in-combine arithmetic, verbatim (the gated prefill kernels'
            # epilogue: FMUL2, MUFU.TANH, FFMA2 on the fp32 merged value, the dead-row SELECT AFTER
            # the fma); the opaque 0.5 is hoisted here, once per row, like there.
            gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
            inv_den_half = inv_den * cutlass.Float32(0.5)
            gv = cutlass.make_array_view(gate)
            g0 = zero
            g1 = zero
            g2 = zero
            g3 = zero
            if lo & col_ok:
                g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                if d0 + cutlass.Int32(1) < d_v:
                    g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                if d0 + cutlass.Int32(2) < d_v:
                    g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                if d0 + cutlass.Int32(3) < d_v:
                    g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
            hg0 = acc0 * inv_den_half
            hg1 = acc1 * inv_den_half
            hg2 = acc2 * inv_den_half
            hg3 = acc3 * inv_den_half
            s0, s1 = fmul2(g0, g1, gate_half, gate_half)
            s2, s3 = fmul2(g2, g3, gate_half, gate_half)
            t0 = cute.math.tanh(s0, approx=True)
            t1 = cute.math.tanh(s1, approx=True)
            t2 = cute.math.tanh(s2, approx=True)
            t3 = cute.math.tanh(s3, approx=True)
            y0, y1 = ffma2(t0, t1, hg0, hg1, hg0, hg1)
            y2, y3 = ffma2(t2, t3, hg2, hg3, hg2, hg3)
            outs = (
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
            )
        q_scale = cutlass.Float32(1.0)
        if cutlass.const_expr(scale_o is not None):
            q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
        amax_local = zero
        vals = []
        for i in cutlass.range_constexpr(4):
            o_val = outs[i]
            if cutlass.const_expr(amax_o is not None):
                # Measured before scale_o (the pre-quant amax); a duplicate or past-the-row column
                # group folds nothing.
                fold = lo & (d0 + cutlass.Int32(i) < d_v)
                amax_local = cutlass.Float32(
                    arith.select(fold.ir_value(), cute.math.max(amax_local, cute.math.max(o_val, -o_val)).ir_value(), amax_local.ir_value())
                )
            if cutlass.const_expr(scale_o is not None):
                o_val = o_val * q_scale
            vals.append(o_val)
        store_row = row_live & lo & col_ok
        if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
            out_base = o_out.iterator.toint()
            _st_strides = cutlass.Int64(o_out.stride[0]) | cutlass.Int64(o_out.stride[1]) | cutlass.Int64(o_out.stride[2])
            st_ok = (
                ((out_base & cutlass.Int64(7)) == cutlass.Int64(0))
                & (cutlass.Int64(o_out.stride[3]) == cutlass.Int64(1))
                & ((_st_strides & cutlass.Int64(3)) == cutlass.Int64(0))
            )
            if st_ok & store_row & (d0 + cutlass.Int32(3) < d_v):
                _cvt = "bf16x2" if cutlass.const_expr(o_out.element_type == cutlass.BFloat16) else "f16x2"
                p01, p23 = nvvm.inline_ptx(
                    f"cvt.rn.{_cvt}.f32 $0, $3, $2;\ncvt.rn.{_cvt}.f32 $1, $5, $4;",
                    write_only_types=[cutlass.Int32, cutlass.Int32],
                    read_only_args=[vals[0], vals[1], vals[2], vals[3]],
                )
                o_idx = cute.crd2idx((o_batch, o_tok, head, d0), o_out.layout)
                st_global_v2(out_base + cutlass.Int64(o_idx) * 2, (p01, p23), cutlass.Int32)
            else:
                for i in cutlass.range_constexpr(4):
                    if store_row & (d0 + cutlass.Int32(i) < d_v):
                        oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        else:
            for i in cutlass.range_constexpr(4):
                # Index all modes: ArrayView's row slice is a pointer and drops the final mode's stride.
                if store_row & (d0 + cutlass.Int32(i) < d_v):
                    oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        if cutlass.const_expr(amax_o is not None):
            # ONE atomic per block over the chunk's columns (an integer max on the non-negative fp32
            # bit pattern is exact and order-independent); all 32 lanes run the shuffles.
            for off_s in cutlass.range_constexpr(5):
                amax_local = cute.math.max(amax_local, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_local, 16 >> off_s, 31, kind=nvvm.Shfl.BFLY)))
            if lane == cutlass.Int32(0):
                _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))
        # The recombined LSE (when the caller asked for Stats): the chunk-0 block of the row writes it.
        if cutlass.const_expr(lse_out is not None):
            if (lane == cutlass.Int32(0)) & (chunk == cutlass.Int32(0)) & lse_live:
                lo_view = cutlass.make_array_view(lse_out)
                lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
                lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), neg_inf.ir_value(), lse_val.ir_value()))
                if cutlass.const_expr(stats_log2):
                    lse_val = lse_val * cutlass.Float32(1.4426950408889634)
                lo_view[lse_batch, head, lse_tok] = lse_val


_combine_decode_grid_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _combine_packed_kernel(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    total_q: cute.Tensor,
    n_splits: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],
    sinks: Optional[cute.Tensor] = None,
    prefetch_group: cutlass.Constexpr[int] = 0,
    cu_q: Optional[cute.Tensor] = None,
    origin_offsets=None,
    origin_specs: cutlass.Constexpr = None,
) -> None:
    """Four packed rows per CTA; each warp reuses split weights across D.

    Packed partials have no per-sequence padding or batch coordinate. The
    device total guards their unwritten tail before any partial is read.
    Final O/Stats retain the caller's strides and partial Stats stay in ln.
    """
    thread = cute.arch.thread_idx()[0]
    lane = thread % 32
    row = cute.arch.block_idx()[0] * 4 + thread // 32
    head = cute.arch.block_idx()[1]
    if (row >= o_partial.shape[1]) | (row >= cutlass.make_array_view(total_q)[0]):
        # Warp-uniform, with no CTA barriers in this kernel.
        nvvm.exit()
    o_row, stats_row = cutlass.Int64(row), cutlass.Int64(row)
    if cutlass.const_expr(cu_q is not None):
        prefix = cutlass.make_array_view(cu_q)
        lo, hi = cutlass.Int32(0), cutlass.Int32(cu_q.shape[0] - 1)
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix[mid + 1] <= row:
                lo = mid + 1
            else:
                hi = mid
        local_row = cutlass.Int64(row) - cutlass.Int64(prefix[lo])
        if cutlass.const_expr(origin_specs[3] is not None):
            o_row = (
                cutlass.Int64(cutlass.make_array_view(origin_offsets[3])[lo]) * cutlass.Int64(origin_specs[3][1]) // cutlass.Int64(o_out.stride[1]) + local_row
            )
        if cutlass.const_expr(origin_specs[4] is not None):
            stats_row = (
                cutlass.Int64(cutlass.make_array_view(origin_offsets[4])[lo]) * cutlass.Int64(origin_specs[4][1]) // cutlass.Int64(origin_specs[4][2])
                + local_row
            )
    op = cutlass.make_array_view(o_partial)
    lp = cutlass.make_array_view(lse_partial)
    oo = cutlass.make_array_view(o_out)
    neg_inf = cutlass.Float32(NEG_INF)
    # Sink-free partials carry only real keys. Add the per-head virtual key
    # exactly once, in the same stable normalizer as the real split weights.
    sink = neg_inf
    if cutlass.const_expr(sinks is not None):
        sink = cutlass.Float32(cutlass.make_array_view(sinks)[head])
    if cutlass.const_expr(prefetch_group > 0):
        # The existing small-split reducer prefetches one group before waiting
        # on its loads. Packed rows use the same partial slab geometry with B=1.
        lean = cutlass.Int32(0) > cutlass.Int32(1)
        if cutlass.const_expr(o_partial.element_type == cutlass.Float32):
            address_bits = o_partial.iterator.toint()
            for axis in cutlass.range_constexpr(3):
                address_bits = address_bits | (cutlass.Int64(o_partial.stride[axis]) * 4)
            lean = ((address_bits & 15) == 0) & ((o_partial.shape[3] & 3) == 0) & (n_splits <= prefetch_group)
        if lean:
            lse_base = lp.data_ptr((0, head, row)).toint(cutlass.Int64)
            lse_step = cutlass.Int64(lse_partial.stride[0]) * 4
            o_step = cutlass.Int64(o_partial.stride[0]) * 4
            sink_row = None
            if cutlass.const_expr(sinks is not None):
                sink_row = sink
            for d_base in cutlass.range(0, o_partial.shape[3], 128, unroll=1):
                d0 = d_base + lane * 4
                if d0 < o_partial.shape[3]:
                    o_row_base = op.data_ptr((0, row, head, d0)).toint(cutlass.Int64)
                    all_dead, den, a0, a1, a2, a3, m_safe = _lean_accumulate(
                        prefetch_group, lse_base, lse_step, o_row_base, o_step, n_splits - 1, n_splits, sink_row
                    )
                    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
                    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_den.ir_value()))
                    values = (a0, a1, a2, a3)
                    for slot in cutlass.range_constexpr(4):
                        oo[0, o_row, head, d0 + slot] = (values[slot] * inv_den).to(o_out.element_type)
                    if cutlass.const_expr(lse_out is not None):
                        if (lane == 0) & (d_base == 0):
                            value = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
                            value = cutlass.Float32(arith.select(all_dead.ir_value(), neg_inf.ir_value(), value.ir_value()))
                            if cutlass.const_expr(stats_log2):
                                value = value * cutlass.Float32(1.4426950408889634)
                            cutlass.make_array_view(lse_out)[0, head, stats_row] = value
            # Warp-uniform branch, no CTA synchronization in either reducer.
            nvvm.exit()
    parallel = n_splits <= 32
    m = sink
    m_safe = cutlass.Float32(0.0)
    all_dead = m == neg_inf
    den = cutlass.Float32(0.0)
    lane_weight = cutlass.Float32(0.0)
    if parallel:
        lane_lse = neg_inf
        if lane < n_splits:
            lane_lse = cutlass.Float32(lp[lane, head, row])
        m = lane_lse
        for bit in cutlass.range_constexpr(5):
            m = cute.math.max(m, cute.arch.shuffle_sync_bfly(m, 1 << bit))
        if cutlass.const_expr(sinks is not None):
            m = cute.math.max(m, sink)
        all_dead = m == neg_inf
        m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))
        if lane_lse > neg_inf:
            lane_weight = cute.math.exp(lane_lse - m_safe, fastmath=True)
        den = lane_weight
        for bit in cutlass.range_constexpr(5):
            den = den + cute.arch.shuffle_sync_bfly(den, 1 << bit)
    else:
        # Preserve arbitrary runtime split counts for internal prepared users.
        for split in cutlass.range(0, n_splits, 1, unroll=1):
            m = cute.math.max(m, cutlass.Float32(lp[split, head, row]))
        all_dead = m == neg_inf
        m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))
        for split in cutlass.range(0, n_splits, 1, unroll=1):
            partial_lse = cutlass.Float32(lp[split, head, row])
            if partial_lse > neg_inf:
                den = den + cute.math.exp(partial_lse - m_safe, fastmath=True)
    if cutlass.const_expr(sinks is not None):
        if sink > neg_inf:
            den = den + cute.math.exp(sink - m_safe, fastmath=True)
    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_den.ir_value()))
    for d_base in cutlass.range(0, o_partial.shape[3], 128, unroll=1):
        acc = cute.make_rmem_tensor((4,), cutlass.Float32)
        acc.fill(0.0)
        for split in cutlass.range(0, n_splits, 1, unroll=1):
            weight = cutlass.Float32(0.0)
            if parallel:
                weight = cute.arch.shuffle_sync(lane_weight, split)
            else:
                partial_lse = cutlass.Float32(lp[split, head, row])
                if partial_lse > neg_inf:
                    weight = cute.math.exp(partial_lse - m_safe, fastmath=True)
            # Dead splits may have non-finite payloads: skip their loads;
            # multiplication by zero would not remove a NaN contribution.
            if weight > cutlass.Float32(0.0):
                for slot in cutlass.range_constexpr(4):
                    d = d_base + lane + slot * 32
                    if d < o_partial.shape[3]:
                        acc[slot] = acc[slot] + weight * cutlass.Float32(op[split, row, head, d])
        for slot in cutlass.range_constexpr(4):
            d = d_base + lane + slot * 32
            if d < o_partial.shape[3]:
                oo[0, o_row, head, d] = (acc[slot] * inv_den).to(o_out.element_type)
    if cutlass.const_expr(lse_out is not None):
        if lane == 0:
            value = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
            value = cutlass.Float32(arith.select(all_dead.ir_value(), neg_inf.ir_value(), value.ir_value()))
            if cutlass.const_expr(stats_log2):
                value = value * cutlass.Float32(1.4426950408889634)
            cutlass.make_array_view(lse_out)[0, head, stats_row] = value


_combine_packed_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _combine_lean_walk_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] — split-major partial O
    lse_partial: cute.Tensor,  # [S*B, H, S_q]   — split-major partial LSE
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    # 1-element fp32 amax of the RECOMBINED O.  The per-split epilogues cannot
    # compute this: each sees only its own partial, and O is a convex
    # combination of those, so a max over partials over-reports (~2.9x at 8
    # splits).  The FP8 kernels therefore skip their in-kernel amax write when
    # SPLIT_KV > 1 and leave it to this pass.  None-specialized off otherwise.
    amax_o: Optional[cute.Tensor],
    # 1-element fp32 output scale, present only when O is stored quantized.
    # The split kernels leave it out of their epilogue and write UNSCALED
    # partials; this pass applies it at the single cast, so the FP8 rounding
    # happens once, on the recombined value.
    scale_o: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    s_q: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],  # write the FINAL LSE in base 2 (stats_use_log2)
    # Ragged final rows (the decode tile's RAGGED_Q leg): (B+1,) int32 / int64 ragged
    # offsets of Q, O and Stats in ELEMENTS and their elements-per-token
    # divisors.  The partials stay dense; only the final O / LSE rows move to
    # ``offset[b] / div + q_row`` on the packed [1, T, ...] outputs, and a row
    # at or past this sequence's Q length (offset[b+1] - offset[b]) / div --
    # every row of a zero-length sequence -- is not written at all.  None
    # (None-specialized) keeps the dense (batch, q_row) placement.
    ragged_q: Optional[cute.Tensor] = None,
    ragged_o: Optional[cute.Tensor] = None,
    ragged_lse: Optional[cute.Tensor] = None,
    ragged_q_div: cutlass.Int64 = 1,
    ragged_o_div: cutlass.Int64 = 1,
    ragged_lse_div: cutlass.Int64 = 1,
    # Packed token capacities of the final O and Stats buffers (ragged only): a
    # row whose ragged placement lands at or past them is not written, so a
    # short buffer or a bad offset can never store outside the caller's bytes.
    ragged_o_cap: cutlass.Int32 = 0,
    ragged_lse_cap: cutlass.Int32 = 0,
    # The gate G of the graph's ``mul(O_v, sigmoid(G))`` tail, [B, S_q, H, D] in
    # the caller's strides (the dense placement only), applied to the fp32 merged
    # value before the single cast -- see the module docstring.  None
    # (None-specialized) is the plain reduction; its traced code is unchanged.
    gate: Optional[cute.Tensor] = None,
    # Programmatic dependent launch (the launch attribute is set by _launch_combine under the same flag): the partial slabs are
    # the split tile's output, so nothing is read above the wait; the pass has no prologue to overlap, so the dependent grid (the
    # out projection in the gated block) is released right after it and stages its own while this pass runs.
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """The LEAN GROUP WALK: the module's former lean kernel, restored as the fifth form for the lean-grid-overflow launches of the cc 10.7
    line (the module docstring).  One warp per (q_row, head, batch), ROWS_PER_BLOCK rows per block, a lane owning four columns of every
    split: LEAN_WALK_G_S splits loaded up front, the rest in groups of LEAN_WALK_G_T per memory round trip, the row accumulated in split
    order from +0 after a full max pass (bitwise the serial walk at every split count, the wide lean group within 16 splits).  Precondition,
    the launch rule's (``_launch_combine`` under DECODE_GRID_LEAN_WALK): fp32 partials, ``d_v % 4 == 0``, a 16-byte aligned slab -- the
    kernel it was tested that at runtime and served half partials on an element path; that path is develop's ``_combine_kernel``'s now
    and is not duplicated here.  No sink fold (the selector takes a sink-free launch only); no CTA barrier; the griddepcontrol pair sits
    before the first global read."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(32)
    q_row = cute.arch.block_idx()[0] * cutlass.Int32(ROWS_PER_BLOCK) + tidx // cutlass.Int32(32)
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    # A warp past S_q (last block only) has no row; there is no block-level
    # barrier below, so it simply falls through. Everything inside is
    # warp-uniform except the per-lane column guard.
    if q_row < s_q:
        oo = cutlass.make_array_view(o_out)

        # Where the recombined row lands: dense (batch, q_row), or the ragged
        # placement on the packed outputs (batch coord 0).
        o_batch = batch
        o_tok = q_row
        lse_batch = batch
        lse_tok = q_row
        row_live = cutlass.Int32(1) == cutlass.Int32(1)
        lse_live = row_live
        if cutlass.const_expr(ragged_q is not None):
            # Offsets are int32 or int64 elements; divide in 64 bits (a large packed
            # buffer's element offset can exceed 2^31) and keep the token index in 32.
            # Bounds are checked in 64 bits BEFORE narrowing: a negative or wrapped
            # offset must fail the capacity test, not alias a valid token after the cast.
            rq = cutlass.make_array_view(ragged_q)
            q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
            q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
            in_seq = cutlass.Int64(q_row) < q_len64
            o_batch = cutlass.Int32(0)
            o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
            row_live = in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
            o_tok = cutlass.Int32(o_tok64)
            lse_live = row_live
            if cutlass.const_expr(ragged_lse is not None):
                lse_batch = cutlass.Int32(0)
                lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
                lse_live = in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
                lse_tok = cutlass.Int32(lse_tok64)

        neg_inf = cutlass.Float32(NEG_INF)
        zero = cutlass.Float32(0.0)
        # Pre-declared: assigned per 128-column chunk inside the loop below; the LSE tail reads them after the join.
        amax_local = cutlass.Float32(0.0)
        m_safe = zero
        den = zero
        all_dead = zero > zero
        o_base = o_partial.iterator.toint()
        # fp32 partials with 16-byte rows: the lean path.  Lane l owns columns 4l..4l+3 of
        # every split; each (b, h, q_row) base offset is computed once and the per-split
        # address is a single 64-bit add (the split axis is the leading, batch-stacked mode).
        lse_base = lse_partial.iterator.toint() + cutlass.Int64(cute.crd2idx((batch, head, q_row), lse_partial.layout)) * 4
        lse_step = cutlass.Int64(lse_partial.stride[0]) * cutlass.Int64(n_batch) * 4
        o_step = cutlass.Int64(o_partial.stride[0]) * cutlass.Int64(n_batch) * 4
        s_last = n_splits - cutlass.Int32(1)
        out_base = o_out.iterator.toint()
        st_ok = zero > zero
        if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
            _st_strides = cutlass.Int64(o_out.stride[0]) | cutlass.Int64(o_out.stride[1]) | cutlass.Int64(o_out.stride[2])
            st_ok = (
                ((out_base & cutlass.Int64(7)) == cutlass.Int64(0))
                & (cutlass.Int64(o_out.stride[3]) == cutlass.Int64(1))
                & ((_st_strides & cutlass.Int64(3)) == cutlass.Int64(0))
            )
        q_scale = cutlass.Float32(1.0)
        if cutlass.const_expr(scale_o is not None):
            q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
        # Gate in the combine: the 0.5 handed to the packed multiply must be opaque (a constant
        # float into inline_ptx ICEs libNVVM).  Hoisted once per row, like the fused kernels do
        # per tile; traced only on the gated entry, so the plain entries' code is unchanged.
        if cutlass.const_expr(gate is not None):
            gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
        for cbase in cutlass.range(0, d_v, 128, unroll=1):
            d0 = cbase + lane * cutlass.Int32(4)
            if d0 < d_v:
                o_row_base = o_base + cutlass.Int64(cute.crd2idx((batch, q_row, head, d0), o_partial.layout)) * 4
                # Every load of the row goes out first -- the LSE of the first LEAN_WALK_G_S splits and their partial-O
                # rows (a clamped split index re-reads the last split for slots past n_splits; those get
                # weight 0) -- so a warp pays ONE memory round trip instead of one per pass.
                lse_v = []
                o_v = []
                for s in cutlass.range_constexpr(LEAN_WALK_G_S):
                    s_eff64 = cutlass.Int64(cute.math.min(cutlass.Int32(s), s_last))
                    lse_v.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
                    o_v.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
                m = neg_inf
                for s in cutlass.range_constexpr(LEAN_WALK_G_S):
                    m = cute.math.max(m, lse_v[s])
                # Splits past LEAN_WALK_G_S extend the max before any weight is formed: LEAN_WALK_G_T LSEs per
                # round trip (a clamped slot re-reads the last split; max is idempotent).
                if n_splits > cutlass.Int32(LEAN_WALK_G_S):
                    for s0 in cutlass.range(LEAN_WALK_G_S, n_splits, LEAN_WALK_G_T, unroll=1):
                        lse_t = []
                        for j in cutlass.range_constexpr(LEAN_WALK_G_T):
                            s_eff64 = cutlass.Int64(cute.math.min(s0 + cutlass.Int32(j), s_last))
                            lse_t.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
                        for j in cutlass.range_constexpr(LEAN_WALK_G_T):
                            m = cute.math.max(m, lse_t[j])
                # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather than
                # exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
                all_dead = m == neg_inf
                m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m.ir_value()))
                acc0 = zero
                acc1 = zero
                acc2 = zero
                acc3 = zero
                den = zero
                for s in cutlass.range_constexpr(LEAN_WALK_G_S):
                    lse_s = lse_v[s]
                    v = o_v[s]
                    live = (lse_s > neg_inf) & (cutlass.Int32(s) < n_splits)
                    e = cute.math.exp(lse_s - m_safe, fastmath=True)
                    w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
                    v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
                    v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
                    v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
                    v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
                    den = den + w
                    acc0 = acc0 + w * v0
                    acc1 = acc1 + w * v1
                    acc2 = acc2 + w * v2
                    acc3 = acc3 + w * v3
                if n_splits > cutlass.Int32(LEAN_WALK_G_S):
                    # LEAN_WALK_G_T splits per round trip: every LSE and partial-O row of the group
                    # is loaded first, then the group is accumulated in split order.  A
                    # slot past n_splits (the last group only) re-reads the last split
                    # and is SELECTED out of the sums, so den / acc hold exactly what the
                    # one-split-per-step walk left in them.
                    for s0 in cutlass.range(LEAN_WALK_G_S, n_splits, LEAN_WALK_G_T, unroll=1):
                        lse_t = []
                        o_t = []
                        for j in cutlass.range_constexpr(LEAN_WALK_G_T):
                            s_eff64 = cutlass.Int64(cute.math.min(s0 + cutlass.Int32(j), s_last))
                            lse_t.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
                            o_t.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
                        for j in cutlass.range_constexpr(LEAN_WALK_G_T):
                            lse_s = lse_t[j]
                            v = o_t[j]
                            in_range = (s0 + cutlass.Int32(j)) < n_splits
                            live = (lse_s > neg_inf) & in_range
                            e = cute.math.exp(lse_s - m_safe, fastmath=True)
                            w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
                            v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
                            v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
                            v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
                            v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
                            den = cutlass.Float32(arith.select(in_range.ir_value(), (den + w).ir_value(), den.ir_value()))
                            acc0 = cutlass.Float32(arith.select(in_range.ir_value(), (acc0 + w * v0).ir_value(), acc0.ir_value()))
                            acc1 = cutlass.Float32(arith.select(in_range.ir_value(), (acc1 + w * v1).ir_value(), acc1.ir_value()))
                            acc2 = cutlass.Float32(arith.select(in_range.ir_value(), (acc2 + w * v2).ir_value(), acc2.ir_value()))
                            acc3 = cutlass.Float32(arith.select(in_range.ir_value(), (acc3 + w * v3).ir_value(), acc3.ir_value()))
                inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
                inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))
                outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
                if cutlass.const_expr(gate is not None):
                    # The gated prefill kernels' epilogue arithmetic, verbatim
                    # (_common_blackwell.gate_epilogue_pairs): per pair one
                    # FMUL2 (g / 2), two MUFU.TANH, one FFMA2 (h * t + h), on the
                    # fp32 merged value; then the dead-row SELECT per element,
                    # AFTER the fma -- a dead row is exactly 0 whatever G holds
                    # (a NaN gate times the zero h would be NaN; a select is not).
                    # sigmoid's 0.5 folds into the scale (``h = acc * (inv_den / 2)``:
                    # exact, a power-of-two scaling of the plain output).
                    inv_den_half = inv_den * cutlass.Float32(0.5)
                    gv = cutlass.make_array_view(gate)
                    g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                    g1 = zero
                    g2 = zero
                    g3 = zero
                    if d0 + cutlass.Int32(1) < d_v:
                        g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                    if d0 + cutlass.Int32(2) < d_v:
                        g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                    if d0 + cutlass.Int32(3) < d_v:
                        g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
                    h0 = acc0 * inv_den_half
                    h1 = acc1 * inv_den_half
                    h2 = acc2 * inv_den_half
                    h3 = acc3 * inv_den_half
                    s0, s1 = fmul2(g0, g1, gate_half, gate_half)
                    s2, s3 = fmul2(g2, g3, gate_half, gate_half)
                    t0 = cute.math.tanh(s0, approx=True)
                    t1 = cute.math.tanh(s1, approx=True)
                    t2 = cute.math.tanh(s2, approx=True)
                    t3 = cute.math.tanh(s3, approx=True)
                    y0, y1 = ffma2(t0, t1, h0, h1, h0, h1)
                    y2, y3 = ffma2(t2, t3, h2, h3, h2, h3)
                    outs = (
                        cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                        cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                        cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                        cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
                    )
                vals = []
                for i in cutlass.range_constexpr(4):
                    o_val = outs[i]
                    if cutlass.const_expr(amax_o is not None):
                        # Measured before scale_o, so this is already the pre-quant
                        # amax and needs no post-hoc divide.
                        amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
                    if cutlass.const_expr(scale_o is not None):
                        o_val = o_val * q_scale
                    vals.append(o_val)
                if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
                    if st_ok & row_live & (d0 + cutlass.Int32(3) < d_v):
                        _cvt = "bf16x2" if cutlass.const_expr(o_out.element_type == cutlass.BFloat16) else "f16x2"
                        p01, p23 = nvvm.inline_ptx(
                            f"cvt.rn.{_cvt}.f32 $0, $3, $2;\ncvt.rn.{_cvt}.f32 $1, $5, $4;",
                            write_only_types=[cutlass.Int32, cutlass.Int32],
                            read_only_args=[vals[0], vals[1], vals[2], vals[3]],
                        )
                        o_idx = cute.crd2idx((o_batch, o_tok, head, d0), o_out.layout)
                        st_global_v2(out_base + cutlass.Int64(o_idx) * 2, (p01, p23), cutlass.Int32)
                    else:
                        for i in cutlass.range_constexpr(4):
                            if row_live & (d0 + cutlass.Int32(i) < d_v):
                                oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
                else:
                    for i in cutlass.range_constexpr(4):
                        # Index all modes: ArrayView's row slice is a pointer and drops
                        # the final mode's stride, so a subsequent [d0] would assume
                        # contiguous D.
                        if row_live & (d0 + cutlass.Int32(i) < d_v):
                            oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        # amax: reduce across the warp, then ONE atomic per warp.  The value is
        # non-negative, so its fp32 bit pattern orders the same as the float and
        # an integer atomicMax is exact -- the same trick the kernels' own
        # epilogues use.
        if cutlass.const_expr(amax_o is not None):
            for off in cutlass.range_constexpr(5):
                amax_local = cute.math.max(amax_local, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_local, 16 >> off, 31, kind=nvvm.Shfl.BFLY)))
            if lane == cutlass.Int32(0):
                _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))

        # --- the recombined LSE (only when the caller asked for Stats) ---
        if cutlass.const_expr(lse_out is not None):
            if (lane == cutlass.Int32(0)) & lse_live:
                lo = cutlass.make_array_view(lse_out)
                lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
                lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(NEG_INF).ir_value(), lse_val.ir_value()))
                # Base-2 Stats (stats_use_log2): the partials stay natural (the
                # merge above needs them); only the final value converts.
                if cutlass.const_expr(stats_log2):
                    lse_val = lse_val * cutlass.Float32(1.4426950408889634)
                lo[lse_batch, head, lse_tok] = lse_val


_combine_lean_walk_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_own_forms(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    amax_o: Optional[cute.Tensor],
    scale_o: Optional[cute.Tensor],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],
    ragged_q: Optional[cute.Tensor],
    ragged_o: Optional[cute.Tensor],
    ragged_lse: Optional[cute.Tensor],
    ragged_divs: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    gate: Optional[cute.Tensor],
    stream: _cuda_driver.CUstream = None,
    sinks: Optional[cute.Tensor] = None,
    q_lens: Optional[cute.Tensor] = None,
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """The module's own THREE-WAY rule -- what ``DECODE_GRID_OFF`` means, and what every launch that takes neither the
    decode-grid kernel nor the lean group walk runs: one of three kernels, chosen per launch from the runtime geometry
    (the comment below).  The positional frame is ``_launch_combine``'s without its form selector."""
    B, H, SQ, D = problem_size
    args = (
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        amax_o,
        scale_o,
        sinks,
        cutlass.Int32(B),
        n_splits,
        cutlass.Int32(D),
        cutlass.Int32(SQ),
        stats_log2,
        ragged_q,
        ragged_o,
        ragged_lse,
        cutlass.Int64(ragged_divs[0]),
        cutlass.Int64(ragged_divs[1]),
        cutlass.Int64(ragged_divs[2]),
        cutlass.Int32(ragged_caps[0]),
        cutlass.Int32(ragged_caps[1]),
        gate,
        q_lens,
    )
    # Three kernels, chosen per launch: decode rows (S_q <= 2, where the block has
    # at least two warps per row) with more splits than the lean form's up-front
    # group hand the block's spare warps to the split axis of the same row
    # (_combine_decode_kernel: B300 128 splits 32.5 -> 12.6 us, 32 splits 10.5
    # -> 5.8); S_q = 3 and up keep the one-warp-per-row form, whose lower
    # register footprint matters where the combine is a throughput problem
    # (b=128 x 16 rows x 2 splits: 41 us against 61 for the decode form) and
    # which beats the single-warp lane-strided form at S_q = 3.  A decode row
    # whose whole split axis fits the lean group (n_splits <= G_S) also takes
    # the one-warp-per-row form: every load goes out at once, one memory round
    # trip, where the decode form pays three dependent ones (B300, d64 / d128
    # S_q = 1, 2-8 splits: 3.4-4.4 us -> 2.1-2.9 us).
    if cutlass.Int32(SQ) <= cutlass.Int32(ROWS_PER_BLOCK // 2):
        if n_splits > cutlass.Int32(G_S):
            _combine_decode_kernel(*args, use_pdl=use_pdl).launch(
                grid=(1, H, B),
                block=[THREADS, 1, 1],
                smem=SMEM_BYTES,
                stream=stream,
                use_pdl=use_pdl,
            )
        else:
            _combine_kernel(*args, lean_wide=True, use_pdl=use_pdl).launch(
                grid=((SQ + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, H, B),
                block=[THREADS, 1, 1],
                stream=stream,
                use_pdl=use_pdl,
            )
    else:
        _combine_kernel(*args, use_pdl=use_pdl).launch(
            grid=((SQ + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, H, B),
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=use_pdl,
        )


@cute.jit
def _launch_combine(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    amax_o: Optional[cute.Tensor],
    scale_o: Optional[cute.Tensor],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],
    ragged_q: Optional[cute.Tensor],
    ragged_o: Optional[cute.Tensor],
    ragged_lse: Optional[cute.Tensor],
    ragged_divs: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    gate: Optional[cute.Tensor],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
    sinks: Optional[cute.Tensor] = None,
    q_lens: Optional[cute.Tensor] = None,
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """One launch for every ABI: dense placement (ragged tensors None) or the
    ragged-Q leg's placement at the offsets, bounded by the (O, Stats) packed
    token capacities; ``gate`` (dense placement only) is the gated entry's G,
    ``sinks`` (fp32 [H]) folds the attention sink once and ``q_lens`` (int32 [B],
    with a sink, dense placement only) keeps the padded rows of the Q trim dead;
    ``use_pdl`` launches with the programmatic-stream-serialization attribute
    (every form's griddepcontrol pair is traced under the same flag).

    ``decode_grid`` picks between the DECODE-GRID kernel and the module's own
    forms: ``DECODE_GRID_AUTO`` launches the decode-grid kernel when the launch
    is a sink-free, unquantized fp32-partial one whose lean geometry holds
    (``d_v % 4 == 0``, a 16-byte aligned slab; a runtime test, like the lean
    path's own) and the lean grid holds at most
    ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks; ``DECODE_GRID_ON``
    drops the grid bound; ``DECODE_GRID_OFF`` never launches it;
    ``DECODE_GRID_LEAN_WALK`` (appended) launches the LEAN GROUP WALK
    (``_combine_lean_walk_kernel``) when the launch is a sink-free fp32-partial
    one whose lean geometry holds -- an amax or an output scale included (the
    kernel's own arms; ``compile_ptr``'s quantized entries name no form, so only
    a direct call reaches them) -- and never the decode-grid kernel.  Every
    other launch takes the module's own three-way rule (``_launch_own_forms``):
    under AUTO / ON / OFF a sink, an amax or an output scale, half partials, an
    unaligned slab, a grid past the bound (AUTO) or OFF itself; under LEAN_WALK
    a sink, half partials or an unaligned slab."""
    B, H, SQ, D = problem_size
    if cutlass.const_expr(decode_grid == DECODE_GRID_LEAN_WALK and o_partial.element_type == cutlass.Float32 and sinks is None and q_lens is None):
        # The LEAN GROUP WALK entry (the cc 10.7 adapter's on a lean-grid-overflow plan): the walk where the lean geometry holds -- the
        # same runtime test as the decode-grid kernel's and the lean path's own -- else the module's own three-way rule.  Never the
        # decode-grid kernel: a LEAN_WALK entry with fp32 sink-free partials reaches this branch only, and one with half partials or
        # a sink fails the fp32 / sink-free conjuncts of the branch below.  The walk's amax / scale arms are its own (the kernel it
        # was served the quantized entries); compile_ptr's quantized entry names no form, so they are reached by a direct call only.
        d32 = cutlass.Int32(D)
        o_base = o_partial.iterator.toint()
        lean_ok = ((d32 & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
        if lean_ok:
            _combine_lean_walk_kernel(
                o_partial,
                lse_partial,
                o_out,
                lse_out,
                amax_o,
                scale_o,
                cutlass.Int32(B),
                n_splits,
                cutlass.Int32(D),
                cutlass.Int32(SQ),
                stats_log2,
                ragged_q,
                ragged_o,
                ragged_lse,
                cutlass.Int64(ragged_divs[0]),
                cutlass.Int64(ragged_divs[1]),
                cutlass.Int64(ragged_divs[2]),
                cutlass.Int32(ragged_caps[0]),
                cutlass.Int32(ragged_caps[1]),
                gate,
                use_pdl,
            ).launch(
                grid=((SQ + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, H, B),
                block=[THREADS, 1, 1],
                stream=stream,
                use_pdl=use_pdl,
            )
        else:
            _launch_own_forms(
                o_partial,
                lse_partial,
                o_out,
                lse_out,
                amax_o,
                scale_o,
                problem_size,
                n_splits,
                stats_log2,
                ragged_q,
                ragged_o,
                ragged_lse,
                ragged_divs,
                ragged_caps,
                gate,
                stream,
                sinks,
                q_lens,
                use_pdl,
            )
    else:
        # The decode-grid kernel's geometry test is the lean path's own: lane l reads columns 4l..4l+3 of the compact fp32 partial
        # rows as one 16-byte load when d_v % 4 == 0 and the slab is 16-byte aligned; any other d_v / base (legal through this
        # runtime-shape entry) takes the module's forms, whose bounded element loads and stores serve it.
        use_decode_grid = cutlass.Int32(0) == cutlass.Int32(1)
        if cutlass.const_expr(
            decode_grid != DECODE_GRID_OFF
            and o_partial.element_type == cutlass.Float32
            and amax_o is None
            and scale_o is None
            and sinks is None
            and q_lens is None
        ):
            d32 = cutlass.Int32(D)
            o_base = o_partial.iterator.toint()
            lean_ok = ((d32 & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
            if cutlass.const_expr(decode_grid == DECODE_GRID_ON):
                use_decode_grid = lean_ok
            else:
                lean_blocks = ((cutlass.Int32(SQ) + cutlass.Int32(ROWS_PER_BLOCK - 1)) // cutlass.Int32(ROWS_PER_BLOCK)) * cutlass.Int32(H) * cutlass.Int32(B)
                use_decode_grid = lean_ok & (lean_blocks <= cutlass.Int32(_DECODE_COMBINE_MAX_LEAN_BLOCKS))
        if use_decode_grid:
            n_chunks = (cutlass.Int32(D) + cutlass.Int32(DECODE_CHUNK_COLS - 1)) // cutlass.Int32(DECODE_CHUNK_COLS)
            _combine_decode_grid_kernel(
                o_partial,
                lse_partial,
                o_out,
                lse_out,
                amax_o,
                scale_o,
                cutlass.Int32(B),
                n_splits,
                cutlass.Int32(D),
                n_chunks,
                stats_log2,
                ragged_q,
                ragged_o,
                ragged_lse,
                cutlass.Int64(ragged_divs[0]),
                cutlass.Int64(ragged_divs[1]),
                cutlass.Int64(ragged_divs[2]),
                cutlass.Int32(ragged_caps[0]),
                cutlass.Int32(ragged_caps[1]),
                gate,
                use_pdl,
            ).launch(
                grid=(cutlass.Int32(SQ) * n_chunks, H, B),
                block=[DECODE_THREADS, 1, 1],
                stream=stream,
                use_pdl=use_pdl,
            )
        else:
            _launch_own_forms(
                o_partial,
                lse_partial,
                o_out,
                lse_out,
                amax_o,
                scale_o,
                problem_size,
                n_splits,
                stats_log2,
                ragged_q,
                ragged_o,
                ragged_lse,
                ragged_divs,
                ragged_caps,
                gate,
                stream,
                sinks,
                q_lens,
                use_pdl,
            )


@cute.jit
def _host_ptr(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """Bind prepared f16/bf16 pointers to the same reduction as tensor callers.

    Partial workspaces are compact; final O and Stats use the actual caller
    strides. Widen before computing compact strides, including split batches.
    This is the dense placement's positional ABI (unchanged); the ragged-Q
    decode leg compiles :func:`_host_ptr_ragged` instead.
    """
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        None,
        None,
        None,
        (1, 1, 1),
        (0, 0),
        None,
        decode_grid,
        stream,
        None,
        None,
        use_pdl,
    )


@cute.jit
def _host_ptr_gated(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    gate_ptr: cute.Pointer,
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    """:func:`_host_ptr` plus the gate: ``gate_ptr`` / ``gate_strides`` describe G
    as a ``[B, S_q, H, D_v]`` view in BSHD stride order (like ``o_strides``), read
    element-wise -- the dense placement's gate-in-combine ABI (a split launch of
    a kernel without epilogue-gate seams, the d256 decode tile).  The dense
    entry's positional ABI stays exactly what its callers pass."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    B, H, SQ, D = problem_size
    gate = cute.make_tensor(gate_ptr, cute.make_layout((B, SQ, H, D), stride=gate_strides))
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        None,
        None,
        None,
        (1, 1, 1),
        (0, 0),
        gate,
        decode_grid,
        stream,
        None,
        None,
        use_pdl,
    )


@cute.jit
def _host_ptr_packed(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    total_q_ptr: cute.Pointer,
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
    sinks_ptr: Optional[cute.Pointer] = None,
    prefetch_group: cutlass.Constexpr[int] = 0,
    cu_q_ptr: Optional[cute.Pointer] = None,
    origin_ptrs=None,
    origin_specs: cutlass.Constexpr = None,
    n_batch: cutlass.Int32 = 0,
) -> None:
    """Combine compact [split, packed token, head, D] partials (B must be 1)."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    total_q = cute.make_tensor(total_q_ptr, cute.make_layout((1,), stride=(1,)))
    sinks = cute.make_tensor(sinks_ptr, cute.make_layout((problem_size[1],), stride=(1,))) if cutlass.const_expr(sinks_ptr is not None) else None
    args = (o_partial, lse_partial, o_out, lse_out, total_q, n_splits, stats_log2, sinks)
    grid = ((problem_size[2] + 3) // 4, problem_size[1], 1)
    # The prepared attention host already specializes on split count. Compile
    # just that reducer; standalone runtime-split callers keep the generic one.
    cu_q = cute.make_tensor(cu_q_ptr, cute.make_layout((n_batch + 1,), stride=(1,))) if cutlass.const_expr(cu_q_ptr is not None) else None
    offsets = (
        tuple(cute.make_tensor(ptr, cute.make_layout((n_batch + 1,), stride=(1,))) if ptr is not None else None for ptr in origin_ptrs)
        if cutlass.const_expr(origin_ptrs is not None)
        else None
    )
    _combine_packed_kernel(*args, prefetch_group=prefetch_group, cu_q=cu_q, origin_offsets=offsets, origin_specs=origin_specs).launch(
        grid=grid, block=[THREADS, 1, 1], stream=stream
    )


@cute.jit
def _host_ptr_quantized(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    amax_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
    has_scale_o: cutlass.Constexpr[bool],
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Preserve the sink-free quantized entry's original positional ABI."""
    _host_ptr_quantized_sink(
        o_partial_ptr,
        lse_partial_ptr,
        o_out_ptr,
        lse_out_ptr,
        problem_size,
        n_splits,
        o_strides,
        lse_strides,
        amax_o_ptr,
        scale_o_ptr,
        has_scale_o,
        stats_log2,
        stream,
    )


@cute.jit
def _host_ptr_quantized_sink(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    amax_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
    has_scale_o: cutlass.Constexpr[bool],
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
    sinks_ptr: Optional[cute.Pointer] = None,
    q_lens_ptr: Optional[cute.Pointer] = None,
) -> None:
    """Reduce FP8-input partials, then scale/cast and report the recombined Amax.

    Quantized outputs keep unscaled partials and apply scale_o here. Half outputs
    retain their existing scaled partials; normalize their requested Amax after
    reduction. The preceding split-attention kernel initializes Amax, ordered
    before this reduction on the same stream. MXFP8 has no scalar output
    scale: its None-specialized pointer also removes Amax unscaling.
    Optional sink/Q-length pointers trail the original stream slot. The sink
    participates in the final normalizer once; padded Q rows remain dead.
    """
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    amax_o = None
    if cutlass.const_expr(amax_o_ptr is not None):
        amax_o = cute.make_tensor(amax_o_ptr, cute.make_layout((1,), stride=(1,)))
    scale_o = None
    if cutlass.const_expr(has_scale_o):
        scale_o = cute.make_tensor(scale_o_ptr, cute.make_layout((1,), stride=(1,)))
    sinks = None
    if cutlass.const_expr(sinks_ptr is not None):
        sinks = _sink_operand(sinks_ptr, problem_size)
    q_lens = None
    if cutlass.const_expr(q_lens_ptr is not None):
        q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((problem_size[0],), stride=(1,)))
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        amax_o,
        scale_o,
        problem_size,
        n_splits,
        stats_log2,
        None,
        None,
        None,
        (1, 1, 1),
        (0, 0),
        None,
        DECODE_GRID_OFF,
        stream,
        sinks,
        q_lens,
    )
    if cutlass.const_expr(amax_o_ptr is not None and not has_scale_o and scale_o_ptr is not None):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@cute.jit
def _host_ptr_ragged(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_q_ptr: cute.Pointer,
    ragged_o_ptr: cute.Pointer,
    ragged_lse_ptr: Optional[cute.Pointer],
    ragged_divs: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """The ragged-Q decode leg's pointer ABI: :func:`_host_ptr` plus the (B+1,)
    ragged offsets of Q / O / Stats (int32 or int64 elements), their
    elements-per-token divisors and the (O, Stats) packed token capacities; the
    final O / Stats are the packed outputs addressed at ``offset[b] / div +
    q_row`` with batch coord 0 (``o_strides[0]`` / ``lse_strides[0]`` are never
    stepped), bounded by the capacities."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    B = problem_size[0]
    n_off = cutlass.Int32(B) + cutlass.Int32(1)
    ragged_q = cute.make_tensor(ragged_q_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_o = cute.make_tensor(ragged_o_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_lse = None
    if cutlass.const_expr(ragged_lse_ptr is not None):
        ragged_lse = cute.make_tensor(ragged_lse_ptr, cute.make_layout((n_off,), stride=(1,)))
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        ragged_q,
        ragged_o,
        ragged_lse,
        ragged_divs,
        ragged_caps,
        None,
        decode_grid,
        stream,
    )


@cute.jit
def _sink_operand(sinks_ptr: cute.Pointer, problem_size: Tuple[int, int, int, int]) -> cute.Tensor:
    """The fp32 [H] sink-logit view of a sink pointer entry."""
    return cute.make_tensor(sinks_ptr, cute.make_layout((problem_size[1],), stride=(1,)))


@cute.jit
def _host_ptr_sink(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    sinks_ptr: cute.Pointer,
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """:func:`_host_ptr` plus the fp32 [H] attention-sink logits, folded once by
    the reduction (the split partials are sink-free).  A separate entry so the
    sink-free dense ABI stays exactly what its (native) callers pass."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    sinks = _sink_operand(sinks_ptr, problem_size)
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        None,
        None,
        None,
        (1, 1, 1),
        (0, 0),
        None,
        decode_grid,
        stream,
        sinks,
    )


@cute.jit
def _host_ptr_sink_q_lens(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    sinks_ptr: cute.Pointer,
    q_lens_ptr: cute.Pointer,
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """:func:`_host_ptr_sink` plus the (B,) int32 per-batch Q lengths of the dense padded-Q trim: a query row at or past its batch's length is a
    PADDED row and keeps ``O := 0 / LSE := -inf`` whatever the sink, which the reduction cannot tell from a valid row whose every key is masked (the
    partials of both are dead).  A separate entry so the sink entry's ABI stays what its callers pass."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    sinks = _sink_operand(sinks_ptr, problem_size)
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((problem_size[0],), stride=(1,)))
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        None,
        None,
        None,
        (1, 1, 1),
        (0, 0),
        None,
        decode_grid,
        stream,
        sinks,
        q_lens,
    )


@cute.jit
def _host_ptr_ragged_sink(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_q_ptr: cute.Pointer,
    ragged_o_ptr: cute.Pointer,
    ragged_lse_ptr: Optional[cute.Pointer],
    ragged_divs: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    sinks_ptr: cute.Pointer,
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """:func:`_host_ptr_ragged` plus the fp32 [H] attention-sink logits."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    B = problem_size[0]
    n_off = cutlass.Int32(B) + cutlass.Int32(1)
    ragged_q = cute.make_tensor(ragged_q_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_o = cute.make_tensor(ragged_o_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_lse = None
    if cutlass.const_expr(ragged_lse_ptr is not None):
        ragged_lse = cute.make_tensor(ragged_lse_ptr, cute.make_layout((n_off,), stride=(1,)))
    sinks = _sink_operand(sinks_ptr, problem_size)
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        ragged_q,
        ragged_o,
        ragged_lse,
        ragged_divs,
        ragged_caps,
        None,
        decode_grid,
        stream,
        sinks,
    )


@cute.jit
def _ptr_operands(o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides):
    """The compact split-major partial slabs and the strided final O / Stats views of a pointer entry."""
    B, H, SQ, D = problem_size
    b64, h64, sq64, d64 = cutlass.Int64(B), cutlass.Int64(H), cutlass.Int64(SQ), cutlass.Int64(D)
    split_batches = b64 * cutlass.Int64(n_splits)
    o_partial = cute.make_tensor(o_partial_ptr, cute.make_layout((split_batches, SQ, H, D), stride=(sq64 * h64 * d64, h64 * d64, d64, 1)))
    lse_partial = cute.make_tensor(lse_partial_ptr, cute.make_layout((split_batches, H, SQ), stride=(h64 * sq64, sq64, 1)))
    o_out = cute.make_tensor(o_out_ptr, cute.make_layout((B, SQ, H, D), stride=o_strides))
    lse_out = None
    if cutlass.const_expr(lse_out_ptr is not None):
        lse_out = cute.make_tensor(lse_out_ptr, cute.make_layout((B, H, SQ), stride=lse_strides))
    return o_partial, lse_partial, o_out, lse_out


@lru_cache(maxsize=None)
def compile_ptr(
    dtype_o: str = "f16",
    dtype_partial: str = "f32",
    has_lse: bool = False,
    stats_log2: bool = False,
    ragged: bool = False,
    ragged_i64: bool = False,
    quantized: bool = False,
    has_amax: bool = False,
    has_scale_o: bool = False,
    has_scale_o_input: bool = True,
    packed: bool = False,
    has_sink: bool = False,
    gate: bool = False,
    dtype_gate: Optional[str] = None,
    has_q_lens: bool = False,
    decode_grid: int = DECODE_GRID_OFF,
    use_pdl: bool = False,
) -> Callable:
    """Compile a shape-generic pointer entry for prepared split execution.

    The runtime arguments are partial-O/partial-LSE/O/optional-LSE pointers,
    ``(B, H, S_q, D_v)``, split count, O strides in BSHD order, LSE strides
    in BHS order, the three ragged-offset pointers (``ragged``: (B+1,) Q / O /
    Stats offsets, int32 or -- ``ragged_i64`` -- int64; None-specialized off
    otherwise) and their elements-per-token divisors. Every stride is Int64,
    including singleton dimensions. Packed partials use a row-per-warp reduction
    and append the device-total-Q pointer; other entries share the existing
    reduction and launch. The quantized dense entry appends Amax/scale
    pointers; half and ragged entries keep their existing positional ABI.
    The quantized sink entry preserves that prefix and appends sink/Q-length
    pointers after the stream, with an explicit None Q-length slot when absent.
    ``has_scale_o_input=False`` removes the scalar input and Amax unscale
    launch for MXFP8. Per-tensor FP8 retains both by default.
    ``gate=True`` (dense half launches only; ``dtype_gate`` "f16" / "bf16" names
    G's dtype) is the gate-in-combine entry: the dense ABI plus G's pointer and
    its four BSHD strides appended -- see the module docstring for the
    numerics it fixes.
    ``has_sink`` selects the sink entries (half or quantized outputs): the reduction
    folds the attention sink once (the partials are sink-free).  The dense and
    ragged entries append the fp32 [H] sink-logit pointer right before
    ``stats_log2``; the packed entry takes it as its trailing ``sinks_ptr``.
    ``has_q_lens`` (the dense sink entry only) appends the int32 (B,) per-batch Q-length pointer after the sink pointer: the rows of the padded-Q trim
    past their batch's length keep ``O := 0 / LSE := -inf`` instead of folding the sink.
    ``decode_grid`` (appended; ``DECODE_GRID_OFF`` by default: an entry that names no
    form is the module's own three-way rule on every device, and the cc 10.7 adapter
    names AUTO / OFF per plan from its split-aware rule) selects the kernel
    FORM of the half / gated / ragged / sink entries, a compile-time specialization
    that leaves every positional ABI unchanged: AUTO = the DECODE-GRID kernel when
    the lean grid holds at most ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS``
    blocks on a sink-free launch (the grid rule, decided per launch from the
    runtime geometry), else the module's own forms; ``DECODE_GRID_ON`` forces the
    decode-grid kernel where its geometry holds, ``DECODE_GRID_OFF`` the module's
    own three-way rule (tests, an explicit decode flag, a kill switch; the sink
    entries take the module's own forms under every value); ``DECODE_GRID_LEAN_WALK``
    (appended) = the module's LEAN GROUP WALK (``_combine_lean_walk_kernel``) on a
    sink-free fp32-partial launch whose lean geometry holds, else the three-way
    rule, never the decode-grid kernel -- the cc 10.7 adapter names it on a
    lean-grid-overflow plan at a split of at most ``LEAN_WALK_G_S``
    (``heuristics.decode_combine_lean_walk_selected``).
    The quantized and packed entries ignore every value (always their own kernel).
    ``use_pdl=True`` (appended; in the key; the sink-free dense half entries
    ``_host_ptr`` / ``_host_ptr_gated`` only) traces the programmatic-dependent-launch
    form of the same reduction: the launch attribute plus every form's
    ``griddepcontrol`` wait before its first read and release right after -- the
    same math, bitwise.  A sink entry declines it (a typed error) until that
    combination is run on a device.
    """
    if dtype_o not in ("f16", "bf16", "e4m3", "e5m2") or dtype_partial not in ("f16", "bf16", "f32"):
        raise ValueError("prepared split combine requires half/FP8 O and f16/bf16/f32 partials")
    if (has_amax or has_scale_o or dtype_o in ("e4m3", "e5m2")) and not quantized:
        raise ValueError("FP8 outputs and scalar operands require the quantized pointer entry")
    if has_scale_o and not has_scale_o_input:
        raise ValueError("scaled combine requires a scale_o input")
    if quantized and ragged:
        raise ValueError("the quantized pointer entry serves dense split launches")
    if packed and (ragged or quantized):
        raise ValueError("packed THD partials require the half pointer entry without ragged final placement")
    if ragged_i64 and not ragged:
        raise ValueError("ragged_i64 is a ragged specialization")
    if gate and (ragged or quantized or packed):
        raise ValueError("the gate-in-combine entry serves dense half split launches (no ragged placement, no quantized O, no packed partials)")
    if gate and dtype_gate not in ("f16", "bf16"):
        raise ValueError("the gate-in-combine entry needs dtype_gate 'f16' or 'bf16' (the gate G is a half tensor of O's shape)")
    if dtype_gate is not None and not gate:
        raise ValueError("dtype_gate is a gate=True specialization")
    if has_sink and gate:
        raise ValueError("the gate-in-combine entry carries no sink fold (a gated split with a sink is declined)")
    if has_q_lens and not has_sink:
        raise ValueError("has_q_lens is a sink specialization: the Q lengths only decide which rows fold the sink")
    if has_q_lens and (ragged or packed):
        raise ValueError("has_q_lens serves the dense sink entry (the ragged and packed placements have no padded rows)")
    if decode_grid not in (DECODE_GRID_AUTO, DECODE_GRID_ON, DECODE_GRID_OFF, DECODE_GRID_LEAN_WALK):
        raise ValueError("decode_grid must be DECODE_GRID_AUTO, DECODE_GRID_ON, DECODE_GRID_OFF or DECODE_GRID_LEAN_WALK")
    if use_pdl and (packed or quantized or ragged):
        raise ValueError("use_pdl=True is served by the dense half entries only (no packed partials, no quantized O, no ragged placement)")
    if use_pdl and has_sink:
        raise ValueError("use_pdl=True is served by the sink-free dense half entries only (the sink entries' programmatic-dependent-launch form is not wired)")
    _cache_key = _template_key(globals(), locals(), "compile_ptr")
    gmem = cute.AddressSpace.gmem

    def P(dtype):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=dtype.width // 8)

    common = (
        P(_ELEM[dtype_partial]),
        P(cutlass.Float32),
        P(_ELEM[dtype_o]),
        P(cutlass.Float32) if has_lse else None,
        (0, 0, 0, 0),
        cutlass.Int32(0),
        (cutlass.Int64(0),) * 4,
        (cutlass.Int64(0),) * 3,
    )
    # The kernel-form selector is a trailing Constexpr of the half / gated / ragged / sink entries (not a
    # runtime slot: the positional frame the native binder passes is unchanged).
    form = (int(decode_grid),)
    if packed:
        entry, extra, form = _host_ptr_packed, (P(cutlass.Int32),), ()
    elif quantized:
        entry, extra, form = (
            _host_ptr_quantized_sink if has_sink else _host_ptr_quantized,
            (P(cutlass.Float32) if has_amax else None, P(cutlass.Float32) if has_scale_o_input else None, bool(has_scale_o)),
            (),
        )
    elif ragged:
        # The ragged-Q leg's entry appends the offsets / divisors / capacities; the
        # dense entry's positional ABI stays exactly what its callers pass.
        off_t = cutlass.Int64 if ragged_i64 else cutlass.Int32
        entry, extra = (_host_ptr_ragged_sink if has_sink else _host_ptr_ragged), (
            P(off_t),
            P(off_t),
            P(off_t) if has_lse else None,
            (cutlass.Int64(1),) * 3,
            (cutlass.Int32(0),) * 2,
        )
    elif gate:
        # The gate-in-combine entry appends G's pointer and BSHD strides; the
        # dense entry's positional ABI stays exactly what its callers pass.
        entry, extra = _host_ptr_gated, (P(_ELEM[dtype_gate]), (cutlass.Int64(0),) * 4)
    else:
        entry, extra = (_host_ptr_sink_q_lens if has_q_lens else _host_ptr_sink if has_sink else _host_ptr), ()
    if has_sink and not packed and not quantized:
        extra = (*extra, P(cutlass.Float32))  # the fp32 [H] sink logits, right before stats_log2 (the packed entry's ride as its trailing sinks_ptr)
        if has_q_lens:
            extra = (*extra, P(cutlass.Int32))  # the int32 (B,) per-batch Q lengths of the padded-Q trim, right after the sinks
    return _compile_cached(
        entry,
        *common,
        *extra,
        bool(stats_log2),
        *form,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_combine_ptr",
        **({"sinks_ptr": P(cutlass.Float32)} if (has_sink and (packed or quantized)) else {}),
        **({"q_lens_ptr": P(cutlass.Int32)} if (has_q_lens and quantized) else {}),
        **({"use_pdl": True} if use_pdl else {}),
    )
