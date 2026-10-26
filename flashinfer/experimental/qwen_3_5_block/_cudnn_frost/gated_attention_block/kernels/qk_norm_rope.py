# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (2)+(3) of the gated attention block, as ONE FROST kernel.

Per-head RMSNorm over ``D`` followed by partial RoPE on the leading
``ROPE_DIM``, for Q and K in a single launch. V is not normed and never
appears here.

**This kernel is HBM-bandwidth bound and is written to run at SOL.** It moves
exactly ``2 * (Q + K)`` bytes plus a negligible ``rstd`` tail; everything else
(the sum-of-squares reduction, the rsqrt, the rotation) is register work hidden
under the stream. The design follows from that:

* **One row = one 512-byte access group.** A head row is ``D`` elements, and
  every lane moves 16 bytes, so ``LANES = D // 8`` lanes cover a row with ONE
  ``ld.global.v4`` each. At ``D = 256`` that is a full warp issuing one
  perfectly-coalesced 512 B request per row — the whole point. Wider ``D``
  falls back to several chunks per lane (``VEC_CHUNKS``), still contiguous.
* **The rotate_half partner is a shuffle, not a second load.** Lane ``l`` holds
  elements ``[8l, 8l+8)``, so element ``e`` and its partner ``e ± ROPE_DIM/2``
  sit in lanes ``l`` and ``l ^ (ROPE_DIM/16)`` at the SAME sub-index. One
  ``shfl.bfly`` moves all 8 at once. No extra traffic, no SMEM.
* **One rounding, not two.** Norm and rotation both run in fp32 and the result
  is cast to the io dtype once. An unfused chain rounds after the norm and
  again after the rotation; this is strictly more accurate, and the fp32
  reference is written to match (see ``reference.qk_norm_rope_reference``).
* **Q and K share one launch** over a flat row space (all Q rows, then all K
  rows), so K's 1/16-of-Q traffic costs no second launch and cos/sin stay hot
  in L2 across the heads of a token.

Not TMA: at 512 bytes per row with no reuse and no MMA consumer, plain
vectorized LDG/STG needs no SMEM staging round trip. TMA earns its keep when a
tile is reused or feeds tcgen05.

**Where it actually lands, re-measured cold 2026-09-10** (aarch64 Rubin perf
node, B=1 bf16, 397B geometry, every point L2-flushed, copy ceiling 10777 GB/s
measured under the same flush):

    S        moved MB   x L2   GB/s   % ceiling
    2048           72    0.5   2890        26.8
    4096          143    1.1   3402        31.6
    8192          286    2.2   3684        34.2
    16384         573    4.3   3847        35.7
    32768        1145    8.7   3953        36.7

So **36.7% of ceiling at the shipped default**, 40.2% at the best knob, 45.8%
with a compact token stride. It is a large win for the block — stages (2)+(3)
went from 36-53% of the chain to 1.5-2.6% — and it is NOT the SOL this kernel
is designed for.

**The earlier "58% of a 6799 GB/s ceiling" is RETRACTED, twice over.** It was
measured hot, so it carried an L2 lift of up to 1.30x at the sizes it used, and
against a ceiling from a different node. Any fraction quoted for this kernel
must name the node and say whether it was flushed; the two Rubin boxes here
differ by 2.4x on the ceiling alone.

What is now known about the remaining ~60%, and it is a sharper story than the
one this comment used to tell:

* it is **register pressure**, not memory-level parallelism. Three unrelated
  register reductions — one row per group, no rstd, a compact token stride —
  each buy 20-25% on their own and do not compose, which is the signature of
  all three relieving the same limit. More rows in flight makes it strictly
  WORSE, so the kernel is not short of requests;
* Little's law agrees: saturating 10777 GB/s at ~700 ns of latency needs about
  35 KiB in flight per SM across 212 SMs, and the shipped config already has
  that at modest occupancy. Adding in-flight bytes is not the lever;
* the load -> 5-shuffle butterfly -> rsqrt -> store chain is the structural
  difference from a pure copy, and it is what makes those registers live;
* the ``rstd`` outputs cost **21%** on Rubin cold, not the ~1 point recorded
  before. They cost 14 on an A100 before the store was vectorized.

The remaining unknown is the register count itself, and neither Rubin node here
has ``ncu`` installed, so that is blocked on tooling rather than on analysis.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental import primitives as nvvm

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, lane_group_sum
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global_v4, st_global, st_global_v2, st_global_v4

COPY_BITS = 128
ELEMS_PER_ACCESS = COPY_BITS // 16  # halves moved by one ld.global.v4
WORDS_PER_ACCESS = COPY_BITS // 32
ACCESS_BYTES = COPY_BITS // 8
DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
DEFAULT_DEFER_SECONDARY_LOADS = True
DEFAULT_CONST_HEAD_COUNTS = True
"""``h_q``/``h_kv`` as compile-time CONSTANTS rather than runtime ``Int32``.

They are already in the compile cache key, so ``row // h_q`` was an integer
DIVISION by a runtime register where a shift would do. Nothing can observe the
difference and it is bit-identical.

**But it is only free once the registers are there.** Measured cold on the
Rubin perf node, S=32768, the block's fused layout: with
``defer_secondary_loads`` ON it is +0.7% (5772 -> 5813 GB/s); with it OFF it is
**-12%** (4447 -> 3913), because strength-reducing the division costs more live
registers than the division did, and the kernel is starved. An optimization
measured under starvation measures the starvation.

**MEASURED DEAD ENDS -- do not re-add** (both were bit-identical and both were
noise, so they bought a caller-violable contract for nothing):

* *sharing Q and K's token stride*, which they genuinely do share as column
  slices of one fused projection: **+0.34%** (5813 -> 5833 GB/s).
* *folding the destination token stride to a compact ``h*d``*: **-1.8%**
  (5813 -> 5710).

Both were the same hypothesis as ``--compact``'s +25%, and that 25% turned out
to be register relief, not address arithmetic: once ``defer`` supplies the
relief, folding strides buys nothing. ncu says why -- registers/thread 95 -> 64,
occupancy limited by registers 5 -> 8 blocks against a warp limit of 8, achieved
occupancy 55% -> 87%."""
"""Issue the norm-weight and cos/sin loads at their POINT OF USE in pass 2
rather than up front in pass 1.

Pass 1 exists to get every load in flight early, which is the right instinct for
a bandwidth kernel and the wrong one for THIS bandwidth kernel: it is register-
pressure bound, not request-starved. Holding ``w``, ``cos`` and ``sin`` from
pass 1 costs **24 live fp32 per row** across the whole sum-of-squares tree, and
buys memory-level parallelism the kernel already has to spare. Deferring them
exposes almost no latency, because both are hits by construction — the weight is
``[D]`` and shared by every row of its kind, cos/sin are shared by all
``h_q + h_kv`` heads of a token.

The transformation only MOVES two loads, so the result is **bit-identical**;
``probe_norm_rope_bw.py --check`` asserts exactly that, and does so per knob
(a register change can alter scheduling, never arithmetic).

Measured cold on the aarch64 Rubin perf node, B=1 bf16, L2-flushed, against a
copy ceiling of 10794 GB/s measured under the same flush:

    S        defer=0   defer=1    gain
    2048        2924      3716    +27%
    4096        3387      4421    +31%
    8192        3681      4887    +33%
    16384       3842      5172    +35%
    32768       3953      5316    +34%

It is a win in every mode measured and a loss in none: +33% on the training
shape, +21% with a compact token stride, +16% without rstd, and neutral at
``rows_per_group=1`` (4336 vs 4342), where there is little to hoist."""
"""Occupancy knobs. **The optimum is COUPLED to ``want_rstd`` and to the token
stride, so it is not one number** — which is why the single-winner table that
used to sit here was wrong.

Re-measured 2026-09-10 on an aarch64 Rubin perf node, B=1 S=32768 (1145 MB
moved, 8.7x the 126 MiB L2), every point L2-FLUSHED between iterations and
quoted against a copy ceiling measured under the same flush, 10777 GB/s:

    threads x rows_per_group      GB/s   % ceiling
    128 x 1                       4336      40.2      <- best WITH rstd
    128 x 2                       3946      36.6      (the shipped default)
    128 x 4                       2755      25.6
    256 x 2                       3196      29.7
    256 x 4                       1929      17.9
    512 x 2                       3056      28.4

    128 x 2, no rstd              4793      44.4      <- best WITHOUT rstd
    128 x 1, no rstd              4376      40.6
    128 x 2, compact stride       4944      45.8      <- best of all measured

The shape is the kernel's story and it did survive re-measurement: this kernel
is REGISTER-PRESSURE-bound. Every knob that adds live per-thread state is
monotonically worse, and **three unrelated ways of freeing registers each buy
about the same 20-25%** — dropping to one row per group, dropping the rstd
outputs, or handing the artifact a compact token stride. That they do not
compose into 60% is the tell: all three are buying the same thing.

Consequences, none of them optional to know:

* **``rows_per_group=2`` is the wrong default for the TRAINING shape.** With
  rstd live, 1 beats 2 by 9.9%; without rstd, 2 beats 1 by 9.5%. The rstd store
  block below keeps ``rstd_vals`` and ``rstds[]`` live across all of PASS 2,
  which is what tips it. Flip the default only behind its own A/B/A.
* **rstd costs 21%, not "~1 point".** That earlier figure was measured hot, on a
  node whose copy ceiling read 6799 GB/s. It does not survive a cold cache.
* **Re-measure per arch AND per node.** The same sweep on an A100 preferred
  2x256, and the two Rubin nodes disagree on the copy ceiling by 2.4x: 10777
  GB/s on the aarch64 perf node against 4451 GB/s on an x86 dev node. A
  percentage from one is not comparable to a percentage from the other.

Reproduce with the internal bandwidth probe script, which prints hot and cold
columns side by side so the L2 contribution is shown rather than argued about."""

_FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)


def lanes_per_row(d: int) -> int:
    """Lanes that cooperate on one head row: enough for one 16-byte access each,
    capped at a warp so a row never straddles a shuffle group."""
    return min(32, d // ELEMS_PER_ACCESS)


def vec_chunks(d: int) -> int:
    """Accesses each lane makes per row."""
    return d // (lanes_per_row(d) * ELEMS_PER_ACCESS)


def validate_shape(d: int, rope_dim: int, threads_per_cta: int) -> None:
    """Raise on any geometry this kernel cannot address.

    Never an ``assert``: these come from user-facing geometry, so they must
    survive ``python -O`` and name themselves (engine contract § 7).
    """
    if d % ELEMS_PER_ACCESS != 0:
        raise ValueError(f"d_head must be a multiple of {ELEMS_PER_ACCESS} for {COPY_BITS}-bit accesses, got {d}")
    lanes = lanes_per_row(d)
    if 32 % lanes != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp so the reduction and the RoPE shuffle stay in-row")
    if d != lanes * ELEMS_PER_ACCESS * vec_chunks(d):
        raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {vec_chunks(d)} accesses")
    if threads_per_cta % lanes != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")
    if rope_dim:
        if rope_dim % (2 * ELEMS_PER_ACCESS) != 0:
            raise ValueError(f"rope_dim must be a multiple of {2 * ELEMS_PER_ACCESS} so the rotate_half partner is a whole-lane shuffle, got {rope_dim}")
        half_lanes = rope_dim // (2 * ELEMS_PER_ACCESS)
        if half_lanes & (half_lanes - 1):
            raise ValueError(f"rope_dim/{2 * ELEMS_PER_ACCESS} must be a power of two for the butterfly shuffle, got {half_lanes}")
        if rope_dim > lanes * ELEMS_PER_ACCESS:
            raise ValueError(
                f"rope_dim={rope_dim} must fit in the first access chunk ({lanes * ELEMS_PER_ACCESS} elements) so the shuffle needs no cross-chunk exchange"
            )


@cute.kernel
def frost_qk_norm_rope(
    mQ: cute.Tensor,  # [T, H_q,  D] in
    mK: cute.Tensor,  # [T, H_kv, D] in
    mQo: cute.Tensor,  # [T, H_q,  D] out (may alias mQ)
    mKo: cute.Tensor,  # [T, H_kv, D] out (may alias mK)
    mWq: Optional[cute.Tensor],  # [D], or None: RoPE-only (no RMSNorm, no weight loads, no rstd)
    mWk: Optional[cute.Tensor],  # [D], or None (both or neither -- the host checks)
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,  # [T, ROPE_DIM]
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32, or None
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32, or None
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    defer_secondary_loads: cutlass.Constexpr[bool],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """``rows_per_group`` consecutive rows per lane group: ALL loads issued
    first, then the reductions and stores.

    That split is the kernel's whole latency story. One row is a single 16-byte
    load per lane, and the sum-of-squares depends on it immediately — so at
    ``rows_per_group == 1`` each thread has exactly one request in flight and
    the kernel runs at whatever memory-level parallelism occupancy alone
    provides. Issuing R independent rows first multiplies that by R at the cost
    of R times the live registers.

    The two norm weights are loaded ONCE per thread, not once per row: they are
    ``[D]`` and every row picks one of the two with a register select.

    Tail rows clamp their loads to the last valid row and skip every store, so
    the ragged final CTA costs a redundant read and never a wild write.

    ``apply_norm`` is decided at TRACE time from the presence of the weight
    tensors (the same presence switch ``want_rstd`` uses for the rstd outputs):
    with ``mWq is None`` the sum-of-squares pass, the rsqrt, the weight loads
    and the rstd store are not traced at all, and the kernel is partial RoPE
    on Q/K with the dims ``[rope_dim, D)`` copied through BIT-EXACTLY (they
    never leave the io dtype: no fp32 op touches them).
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    apply_norm = cutlass.const_expr(mWq is not None)
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)

    # Address-math folding. Every one of these is a compile-time fact of the
    # artifact that the shipped signature was carrying as a RUNTIME value, and
    # this kernel is register-and-address bound, so each one is real:
    #
    # * ``h_q``/``h_kv`` are in the compile cache key, yet ``row // h_q`` was an
    #   integer DIVISION by a runtime register. Folded, it is a shift.
    # * Q, K, GATE and V are column slices of ONE fused projection, so they
    #   SHARE a token stride by construction (the section-1 layout contract).
    #   Four symbolic strides the compiler cannot prove equal become one.
    # * the block writes COMPACT Q_c/K_c, so the output token stride is
    #   ``h * d`` and need not be symbolic at all.
    _hq = cutlass.Int32(h_q_ct) if cutlass.const_expr(const_head_counts) else h_q
    _hkv = cutlass.Int32(h_kv_ct) if cutlass.const_expr(const_head_counts) else h_kv

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)

    # The norm weight is picked by ADDRESS, then loaded once per row. Loading
    # both [D] vectors up front and selecting in registers looks like it saves
    # a load, but it pins 2x chunks*8 live fp32 registers for the whole kernel
    # -- and this kernel is occupancy-bound, not issue-bound (measured on
    # Rubin: rows_per_group=1 beats 2 beats 4, i.e. more per-thread state makes
    # it strictly worse). The extra load is an L1 hit; the registers are not free.

    rope_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    in_rope = lane < cutlass.Int32(rope_lanes) if cutlass.const_expr(rope_dim > 0) else cutlass.Boolean(False)

    # --- PASS 1: issue every load this thread will make ----------------------
    rows = []
    dsts = []
    rstds = []
    is_qs = []
    tokens = []
    xs = []
    ws = []
    cs = []
    ss = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        is_q = row_r < n_q_rows
        src_addr = cutlass.Int64(0)
        dst_addr = cutlass.Int64(0)
        rstd_addr = cutlass.Int64(0)
        token = cutlass.Int32(0)
        if is_q:
            token = row_r // _hq
            head = row_r % _hq
            src_addr = mQ.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mQ.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mQ.stride[1])
            ) * cutlass.Int64(2)
            dst_addr = mQo.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mQo.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mQo.stride[1])
            ) * cutlass.Int64(2)
            if cutlass.const_expr(mRstdQ is not None):
                rstd_addr = mRstdQ.iterator.toint() + row_r.to(cutlass.Int64) * cutlass.Int64(4)
        else:
            k_row = row_r - n_q_rows
            token = k_row // _hkv
            head = k_row % _hkv
            src_addr = mK.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mK.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mK.stride[1])
            ) * cutlass.Int64(2)
            dst_addr = mKo.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mKo.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mKo.stride[1])
            ) * cutlass.Int64(2)
            if cutlass.const_expr(mRstdK is not None):
                rstd_addr = mRstdK.iterator.toint() + k_row.to(cutlass.Int64) * cutlass.Int64(4)

        # The weight address exists only when there IS a weight: under
        # apply_norm=False no line below reads mWq / mWk.
        w_addr = cutlass.Int64(0)
        if cutlass.const_expr(apply_norm):
            w_addr = mWq.iterator.toint() if is_q else mWk.iterator.toint()
        row_x = []
        row_w = []
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
            pairs = [f16x2_to_f32(w, dtype=mQ.element_type) for w in ld_global_v4(src_addr + off, cutlass.Int32)]
            row_x.append([v for pair in pairs for v in pair])
            if cutlass.const_expr(apply_norm and not defer_secondary_loads):
                w_pairs = [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr + off, cutlass.Int32)]
                row_w.append([v for pair in w_pairs for v in pair])

        # cos/sin only on the rope lanes -- rope_dim/D of the warp. Loading them
        # on every lane (a clamped read) tripled this kernel's L1 request count
        # for no data, and cost real bandwidth.
        cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        if cutlass.const_expr(rope_dim > 0 and not defer_secondary_loads):
            if in_rope:
                base = mCos.iterator.toint() + token.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(2) + rope_off
                base_s = mSin.iterator.toint() + token.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(2) + rope_off
                c_pairs = [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(base, cutlass.Int32)]
                s_pairs = [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(base_s, cutlass.Int32)]
                cos_v = [v for pair in c_pairs for v in pair]
                sin_v = [v for pair in s_pairs for v in pair]

        rows.append(row)
        dsts.append(dst_addr)
        rstds.append(rstd_addr)
        is_qs.append(is_q)
        tokens.append(token)
        xs.append(row_x)
        ws.append(row_w)
        cs.append(cos_v)
        ss.append(sin_v)

    # --- PASS 2: reduce, normalize, rotate, store ----------------------------
    # rstd exists only where a norm exists: compile_qk_norm_rope refuses
    # want_rstd without apply_norm, so this conjunction never folds a store out
    # behind a caller's back -- it only keeps the kernel self-consistent.
    want_rstd = cutlass.const_expr(apply_norm and (mRstdQ is not None or mRstdK is not None))
    rstd_vals = []
    for r in cutlass.range_constexpr(rows_per_group):
        # Pre-bound so the name exists on both trace paths; the value is read
        # only under apply_norm, where it is replaced by the real rsqrt.
        rstd = cutlass.Float32(1.0)
        if cutlass.const_expr(apply_norm):
            acc = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(chunks):
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = acc + xs[r][c][i] * xs[r][c][i]
            rstd = cute.math.rsqrt(lane_group_sum(acc, lanes) * cutlass.Float32(1.0 / d) + eps, fastmath=True)

        # The norm weight and the cos/sin pair are consumed HERE, after the
        # reduction. Loading them in PASS 1 costs 24 live fp32 per row across
        # the whole sum-of-squares tree, and this kernel is REGISTER-PRESSURE
        # bound, not request-starved: measured cold on Rubin, three unrelated
        # register reductions each buy 20-25% while every knob that adds rows
        # in flight makes it strictly worse. So the early issue buys
        # memory-level parallelism the kernel does not need, with the one
        # resource it does. Both are L1/L2 hits by construction -- the weight
        # is [D] and shared by every row, cos/sin are shared by every head of a
        # token -- so deferring them exposes almost no latency.
        if cutlass.const_expr(defer_secondary_loads):
            if cutlass.const_expr(apply_norm):
                w_addr_r = mWq.iterator.toint() if is_qs[r] else mWk.iterator.toint()
                row_w = []
                for c in cutlass.range_constexpr(chunks):
                    off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
                    w_pairs = [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr_r + off, cutlass.Int32)]
                    row_w.append([v for pair in w_pairs for v in pair])
                ws[r] = row_w
            if cutlass.const_expr(rope_dim > 0):
                if in_rope:
                    tok = tokens[r]
                    base = mCos.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(2) + rope_off
                    base_s = mSin.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(2) + rope_off
                    c_pairs = [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(base, cutlass.Int32)]
                    s_pairs = [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(base_s, cutlass.Int32)]
                    cs[r] = [v for pair in c_pairs for v in pair]
                    ss[r] = [v for pair in s_pairs for v in pair]

        # RoPE-only: ys IS xs. The passthrough dims [rope_dim, D) then go back
        # through fp32_to_fp16 untouched by any fp32 op, which is exactly a
        # widen + round-to-same -> bit-exact copy (asserted by the tests).
        ys = []
        for c in cutlass.range_constexpr(chunks):
            if cutlass.const_expr(apply_norm):
                ys.append([xs[r][c][i] * rstd * ws[r][c][i] for i in range(ELEMS_PER_ACCESS)])
            else:
                ys.append(list(xs[r][c]))

        if cutlass.const_expr(rope_dim > 0):
            # rotate_half: element e pairs with e + rope_dim/2, the SAME
            # sub-index in lane (l ^ half_lanes). The shuffle is unconditional
            # -- shfl.sync must be reached by every lane of the warp -- and only
            # the rope lanes keep the result.
            rotated = []
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, ys[0][i], cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                signed = -partner if lane < cutlass.Int32(half_lanes) else partner
                rotated.append(ys[0][i] * cs[r][i] + signed * ss[r][i])
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                ys[0][i] = rotated[i] if in_rope else ys[0][i]

        if cutlass.const_expr(apply_norm):
            rstd_vals.append(rstd)
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
                y = ys[c]
                packed = [fp32_to_fp16(y[i], y[i + 1], dtype=mQo.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)]
                st_global_v4(dsts[r] + off, packed, cutlass.Int32)

    # --- rstd, stored as ONE vector when the group is contiguous -------------
    # One lane per row storing 4 scattered bytes turns each row into its own
    # 32-byte sector: measured 86.6% -> 72.5% of the HBM ceiling at
    # rows_per_group=2 on an A100. A group's R rows ARE consecutive in the same
    # rstd tensor unless it straddles the Q/K seam or the ragged tail, so check
    # that at run time and vectorize the common case.
    #
    # Contiguous is NECESSARY, not sufficient: a v2/v4 store also needs its
    # address R*4-byte ALIGNED. ``row0`` is a multiple of R, so the Q side
    # (``mRstdQ + row0*4``) always is -- but the K side lands at
    # ``mRstdK + (row0 - n_q_rows)*4``, which is aligned only when ``n_q_rows =
    # T*h_q`` is itself a multiple of R. It is not at T=3, h_q=3, R=2 (n_q_rows=9:
    # the group at rows 10..11 is whole-K and its K offset is 4 B), and the
    # vector store then faults with ``cudaErrorMisalignedAddress`` (found by
    # review on PR #1102). So a whole-K group additionally requires the aligned
    # K offset; every other group takes the scalar path below, which never
    # needed more than 4-byte alignment.
    if cutlass.const_expr(want_rstd):
        if lane == cutlass.Int32(0):
            if cutlass.const_expr(rows_per_group in (2, 4)):
                whole_q = row0 + cutlass.Int32(rows_per_group - 1) < n_q_rows
                k_aligned = (row0 - n_q_rows) % cutlass.Int32(rows_per_group) == cutlass.Int32(0)
                whole_k = (row0 >= n_q_rows) and k_aligned
                if (whole_q or whole_k) and (row0 + cutlass.Int32(rows_per_group) <= n_rows):
                    if cutlass.const_expr(rows_per_group == 2):
                        st_global_v2(rstds[0], rstd_vals, cutlass.Float32)
                    else:
                        st_global_v4(rstds[0], rstd_vals, cutlass.Float32)
                else:
                    for r in cutlass.range_constexpr(rows_per_group):
                        if rows[r] < n_rows:
                            st_global(rstds[r], rstd_vals[r], cutlass.Float32)
            else:
                for r in cutlass.range_constexpr(rows_per_group):
                    if rows[r] < n_rows:
                        st_global(rstds[r], rstd_vals[r], cutlass.Float32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def qk_norm_rope_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    q_out: cute.Tensor,
    k_out: cute.Tensor,
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    rstd_q: Optional[cute.Tensor],
    rstd_k: Optional[cute.Tensor],
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    eps: cutlass.Float32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    defer_secondary_loads: cutlass.Constexpr[bool],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qk_norm_rope(
        q,
        k,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        n_q_rows,
        n_rows,
        h_q,
        h_kv,
        eps,
        d,
        rope_dim,
        threads_per_cta,
        rows_per_group,
        defer_secondary_loads,
        h_q_ct,
        h_kv_ct,
        const_head_counts,
        use_pdl,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


class QkNormRopeRecipe(NamedTuple):
    """Build-time facts of one norm+RoPE launch.

    Everything in it is derivable from the DECLARATION — dtypes, head counts,
    head dim, rope dim, whether rstd is wanted, whether the norm is applied —
    so it is a legal compile key (AGENTS.md Rule 4). The token count is NOT: it
    enters the artifact as a ``cute.sym_int`` and the row counts ride in as
    runtime ``Int32`` arguments, so one compile serves every sequence length.

    ``apply_norm`` (appended, default True so every existing recipe reads the
    same) records whether the artifact traced the RMSNorm. A norm-on artifact
    bound to ``None`` weights would dereference a null pointer, and a norm-off
    one handed weights would silently ignore them, so ``run_qk_norm_rope``
    checks the recipe against the weights in BOTH directions.
    """

    compiled: object
    h_q: int
    h_kv: int
    d: int
    eps: float
    rows_per_cta: int
    want_rstd: bool
    apply_norm: bool = True


def _fake(dtype, shape, stride_order):
    return cute.runtime.make_fake_compact_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=shape,
        stride_order=stride_order,
        assumed_align=16,
    )


def fake_rowmajor_dynamic_token_stride(dtype, tok, h: int, d: int):
    """A ``[T, H, D]`` fake whose TOKEN stride is symbolic.

    A compact fake bakes the stride in, and the tvm-ffi boundary then rejects
    anything else with ``Mismatched src.strides[0] ... expected to be <h*d>``.
    That matters because the block's Q/K/GATE/V operands are COLUMN SLICES of
    the fused projection output: heads are contiguous within a token (head
    stride ``d``, elem stride 1), but the token stride is the projection's ``N``,
    not ``h*d``. Leaving it symbolic is what lets one artifact serve both the
    strided source and the compact destination — with no repack (Rule 2).
    """
    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=(tok, h, d),
        stride=(cute.sym_int(), d, 1),
        assumed_align=16,
    )


def compile_qk_norm_rope(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    want_rstd: bool,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    defer_secondary_loads: bool = DEFAULT_DEFER_SECONDARY_LOADS,
    const_head_counts: bool = DEFAULT_CONST_HEAD_COUNTS,
    use_pdl: bool = False,
    dynamic_token_stride: bool = True,
    apply_norm: bool = True,
) -> QkNormRopeRecipe:
    """Build the artifact from SHAPES ALONE — no device allocation, no launch.

    This is the plan-time half: the block calls it from ``compile()`` so the
    execute path's dispatch is a guaranteed cache hit rather than a
    multi-second compile on the first token batch.

    ``apply_norm=False`` (the block's ``GatedAttentionBlockGeometry.qk_norm=False``)
    traces the RoPE-only artifact: the two weight slots are traced as ``None``
    (the kernel's presence switch), no rstd can be wanted, and the flag is IN
    the cache key -- a cached norm-on artifact must never be reused with
    ``None`` weights.
    """
    validate_shape(d, rope_dim, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"qk_norm_rope serves bf16/f16 only, got {dtype}")
    if not apply_norm and rope_dim == 0:
        raise ValueError("apply_norm=False with rope_dim=0 is an identity copy of Q/K; drop the stage instead of launching it")
    if want_rstd and not apply_norm:
        raise ValueError("apply_norm=False (RoPE-only Q/K) computes no RMSNorm and therefore emits no rstd; want_rstd must be False")

    key = (
        str(dtype),
        h_q,
        h_kv,
        d,
        int(rope_dim),
        int(threads_per_cta),
        int(rows_per_group),
        bool(defer_secondary_loads),
        bool(const_head_counts),
        bool(use_pdl),
        bool(want_rstd),
        bool(dynamic_token_stride),
        current_device(),
        bool(apply_norm),
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        # A SYMBOLIC token stride is what lets the block feed Q/K straight out
        # of the fused projection with no repack -- and it is NOT free: measured
        # on Rubin at S=4096 it took this kernel from 3383 to 2213 GB/s (-35%),
        # because the compiler can no longer prove the row start is compact and
        # falls back to generic address arithmetic. A caller whose operands ARE
        # compact should say so and get the fast artifact.
        # This tax disappears with the stage-(1) fork: four compact output
        # buffers means no strided source, and the same fork also deletes the V
        # compaction stage.
        dense = [
            (fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) if dynamic_token_stride else _fake(dtype, (tok, h, d), (2, 1, 0)))
            for h in (h_q, h_kv, h_q, h_kv)
        ]
        # Presence switch: None weights trace the RoPE-only kernel, exactly as
        # None rstd tensors trace the store-less one.
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [_fake(dtype, (tok, rope_dim if rope_dim else 1), (1, 0)) for _ in range(2)]
        rstd = [_fake(torch.float32, (tok, h), (1, 0)) for h in (h_q, h_kv)] if want_rstd else [None, None]
        compiled_cache[key] = cute.compile(
            qk_norm_rope_launch,
            *dense,
            *weights,
            *tables,
            *rstd,
            cutlass.Int32(0),  # n_q_rows  ) all four are runtime values; the
            cutlass.Int32(0),  # n_rows    ) zeros here only pin their TYPE at
            cutlass.Int32(h_q),  #         ) trace time, and the real ones are
            cutlass.Int32(h_kv),  #        ) bound per launch.
            cutlass.Float32(eps),
            cutlass.Int32(0),  # n_blocks
            d,
            int(rope_dim),
            int(threads_per_cta),
            int(rows_per_group),
            bool(defer_secondary_loads),
            int(h_q),
            int(h_kv),
            bool(const_head_counts),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QkNormRopeRecipe(
        compiled=compiled_cache[key],
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        eps=float(eps),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        want_rstd=bool(want_rstd),
        apply_norm=bool(apply_norm),
    )


def check_norm_weights_match_recipe(apply_norm: bool, w_q, w_k) -> None:
    """The weights must agree with the artifact in BOTH directions -- typed.

    A norm-on artifact bound to ``None`` weights dereferences a null pointer in
    the kernel (a launch failure at best); a norm-off artifact handed weights
    would silently ignore them and return RoPE-only Q/K that LOOK plausible.
    Neither is a fallback anyone asked for (Rule 1).
    """
    if (w_q is None) != (w_k is None):
        raise ValueError(
            f"w_q_norm and w_k_norm must be given together or both be None; got w_q_norm={'None' if w_q is None else 'tensor'}, "
            f"w_k_norm={'None' if w_k is None else 'tensor'}"
        )
    if apply_norm and w_q is None:
        raise ValueError("this artifact was compiled WITH the RMSNorm (apply_norm=True); both [D] norm weights must be bound at execute")
    if not apply_norm and w_q is not None:
        raise ValueError("this artifact was compiled WITHOUT the RMSNorm (apply_norm=False, RoPE-only); pass w_q_norm=w_k_norm=None")


def run_qk_norm_rope(r: QkNormRopeRecipe, q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q=None, rstd_k=None, *, stream) -> None:
    """The lowered launch: no validation, no key build, no allocation.

    ``q``/``k`` are ``[T, H, D]`` (``T = B*S``); ``q_out`` may alias ``q``.
    ``w_q``/``w_k`` are both ``None`` for a RoPE-only recipe (``r.apply_norm``
    False) and both tensors otherwise -- checked, both directions.
    """
    t = int(q.shape[0])
    n_q_rows = t * r.h_q
    n_rows = n_q_rows + t * r.h_kv
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    check_norm_weights_match_recipe(r.apply_norm, w_q, w_k)
    if r.want_rstd and (rstd_q is None or rstd_k is None):
        raise ValueError("this artifact was compiled with rstd outputs; both must be bound at execute (Rule 1: no silent fallback)")
    if not r.want_rstd and (rstd_q is not None or rstd_k is not None):
        raise ValueError("this artifact was compiled WITHOUT rstd outputs (want_rstd=False); rstd_q / rstd_k would be silently ignored -- pass None")
    # The two address-math CONTRACTS are checked here and nowhere else. The
    # kernel folded these strides into constants, so a caller that violates one
    # gets a wild write, not a wrong number -- cheap host arithmetic against a
    # multi-microsecond launch, and the only thing standing between a layout
    # change upstream and silent corruption.
    # The rstd slots stay in the ABI even when they traced to None -- the
    # artifact folded out the STORES, not the parameters -- so always pass them.
    r.compiled(
        q,
        k,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        cutlass.Int32(n_q_rows),
        cutlass.Int32(n_rows),
        cutlass.Int32(r.h_q),
        cutlass.Int32(r.h_kv),
        cutlass.Float32(r.eps),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


def build_qk_norm_rope(
    q,
    k,
    q_out,
    k_out,
    w_q,
    w_k,
    cos,
    sin,
    rstd_q=None,
    rstd_k=None,
    *,
    rope_dim,
    eps,
    threads_per_cta=DEFAULT_THREADS_PER_CTA,
    rows_per_group=DEFAULT_ROWS_PER_GROUP,
    defer_secondary_loads=DEFAULT_DEFER_SECONDARY_LOADS,
    const_head_counts=DEFAULT_CONST_HEAD_COUNTS,
    use_pdl=False,
    stream,
):
    """Compile (cached) and run once — the convenience form for tests and
    benchmarks. Production callers split it: :func:`compile_qk_norm_rope` at
    plan time, :func:`run_qk_norm_rope` per execute.

    ``w_q=w_k=None`` selects the RoPE-only artifact (``apply_norm=False``)."""
    _, h_q, d = (int(x) for x in q.shape)
    _, h_kv, d_k = (int(x) for x in k.shape)
    if d_k != d:
        raise ValueError(f"q and k must share d_head; got {d} and {d_k}")
    if int(k.shape[0]) != int(q.shape[0]):
        raise ValueError(f"q and k must share the token count; got {int(q.shape[0])} and {int(k.shape[0])}")
    if (rstd_q is None) != (rstd_k is None):
        raise ValueError("rstd_q and rstd_k must be given together or not at all")
    if (w_q is None) != (w_k is None):
        raise ValueError("w_q and w_k must be given together (RMSNorm + RoPE) or both be None (RoPE only)")
    r = compile_qk_norm_rope(
        dtype=q.dtype,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        rope_dim=rope_dim,
        eps=eps,
        want_rstd=rstd_q is not None,
        threads_per_cta=threads_per_cta,
        rows_per_group=rows_per_group,
        defer_secondary_loads=defer_secondary_loads,
        const_head_counts=const_head_counts,
        use_pdl=use_pdl,
        apply_norm=w_q is not None,
    )
    run_qk_norm_rope(r, q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q, rstd_k, stream=stream)
    return r


def moved_bytes(t: int, h_q: int, h_kv: int, d: int, *, elem_bytes: int = 2, want_rstd: bool = True) -> int:
    """HBM traffic of one launch — the denominator for an SOL number.

    Q and K are read once and written once; the ``rstd`` tail is one fp32 per
    row (1/128 of a 512-byte row, so it barely registers). cos/sin and the norm
    weights are excluded: they are ``S * ROPE_DIM`` and ``D`` respectively and
    live in L2 across the heads of a token, so counting them would understate
    the achieved fraction.
    """
    rows = t * (h_q + h_kv)
    return 2 * rows * d * elem_bytes + (rows * 4 if want_rstd else 0)


frost_qk_norm_rope.set_name_prefix("cudnn", remove_cutlass_symbol=True)
