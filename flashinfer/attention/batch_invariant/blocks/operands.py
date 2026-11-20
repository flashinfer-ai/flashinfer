# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host-side operand block of the attention kernel: everything that depends only on the static
parameters (head dim ``D``, GQA group ``g``, page size ``P``, ring depth). None of it decides a
numeric order; the numerically visible choices are the two MMA instruction variants (the A2
surface: ``tcgen05.mma.cta_group::1.kind::f16 128x128x16`` for QK^T and ``128xDx16`` for PV, the
latter with its A operand in TMEM).

The budgets are functions of ``D``: a Q stage and a KV ring slot
are ``128 x D`` bf16 (16 / 32 / 64 KB at D = 64 / 128 / 256), so the ring takes up to ``512 / D`` slots (8 / 4 / 2:
128 KB of K/V in flight, the same bytes at every D) next to ``q_stages(D)`` Q stages (2, or 1 at D = 256); TMEM holds
``S (128) | P (64) | O_c (D) | O_m (D)`` = 320 / 448 columns at D = 64 / 128, and at D = 256 ``S | P | O_c`` = 448 with
the merged chunk state ``O_m`` in a per-CTA global scratch (``om_in_tmem(D)``: 128 + 64 + 2 D <= 512), as the TS body's
two-tile build places it. The same blocks compute every D: the numerics never see the placement.

Built at trace time (inside the ``@cute.jit`` entry point) because the CuTe layout algebra needs
the MLIR context; the results are plain attributes the kernel reads as constants.

Shared memory tiles come from the library (``make_smem_layout_a/b``: the UMMA canonical SW128
layouts); the producer addresses the same bytes through a second *view* in which a page slab of
``P`` keys in one 64-wide half is a contiguous ``(P, 64)`` block (element
``(key, d)`` of a K or V stage sits at ``(d % 64) + 64 key + 8192 (d // 64)``).
"""

from __future__ import annotations


import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode, OperandSource

from ..plan import om_in_tmem as _om_in_tmem
from ..plan import ws_row
from ..spec import MMA_K, TILE_KEYS, tiles_per_chunk

ACC_DTYPE = cutlass.Float32
IN_DTYPE = cutlass.BFloat16
HALF = 64  # d elements per TMA box / per SW128 half of a tile (128 bytes of bf16)
Q_ROWS = 128  # query rows per CTA tile (the MMA M)
TMEM_COLS = 512
HEAD_DIMS_BODY = (64, 128, 256)  # head dims this body implements (``spec.HEAD_DIMS``)
RING_BYTES = (
    128 << 10
)  # K/V bytes in flight per CTA at every D: 512 / D ring slots of 128 x D bf16
SMEM_CAP = 227 << 10  # the opt-in dynamic smem of one CTA on sm_100 / sm_107


def slot_bytes(D: int) -> int:
    """One KV ring slot (a K or a V tile) or one Q stage: ``128 x D`` bf16."""
    return Q_ROWS * D * 2


def max_kv_slots(D: int) -> int:
    """The deepest ring at ``D``: ``RING_BYTES`` of K/V in flight (8 / 4 / 2 slots at D = 64 / 128 / 256); ``RLK_RING_BYTES`` raises the
    budget where the part's shared memory allows it (sm_107: 328 KB per CTA)."""

    return RING_BYTES // slot_bytes(D)


def valid_kv_slots(D: int) -> tuple:
    return tuple(range(2, max_kv_slots(D) + 1))


def q_stages_for(D: int, q2: bool = False) -> int:
    """Q stages in smem: 2 (the load warp prefetches the next unit's Q) when the budget allows, 1 at D = 256
    (a 64 KB stage: ``Q + ring + payloads`` must stay under ``SMEM_CAP``) and for the paired placement at D = 128 (a stage is two
    32 KB Q tiles next to the 128 KB smem ``O_m`` of the two tiles; the stage is released after the unit's last QK, so the next
    unit's Q still lands under the last PVs and the epilogue)."""
    if q2 and D == 128:
        return 1
    return 2 if 2 * slot_bytes(D) + RING_BYTES + (24 << 10) <= SMEM_CAP else 1


def arch_smem_cap(device=None) -> int:
    """The dynamic shared memory one CTA may allocate on the compile target of ``device`` (``swaps.operands.arch_smem_cap``)."""
    from ..swaps.operands import arch_smem_cap as _cap

    return _cap(device)


om_in_tmem = _om_in_tmem  # the TMEM rule for the merged chunk state lives in plan.py (the planner reads it too); re-exported here
OM_VEC = 4  # the O_m scratch layout (D = 256): element (row, d) of a CTA's 128 x D block at (d // 4) * 512 + row * 4 + d % 4 fp32 --
# a warp's 32 lanes (rows) read / write 512 contiguous bytes per 16-byte-per-lane instruction (4 cache lines) instead of
# 32 lines of a row-major block (the merge is LSU-line bound, not latency bound)


def om_scratch_blocks(D: int) -> int:
    """Per-CTA ``128 x D`` fp32 blocks of the global scratch: one (``O_m``) when the merged chunk state is not in TMEM, else none."""
    return 0 if om_in_tmem(D) else 1


def tmem_plan_for(D: int) -> str:
    """The TMEM plan of the body at head dim ``D`` (a placement):

    * ``"om"`` (D <= 128): ``S (T) | P (T/2) | O_c (D) | O_m (D)`` -- one S stage, one P stage, the merged chunk state in TMEM;
    * ``"s2"`` (D = 256): ``O_c (D) | S_0 (T) | S_1 (T)`` = 512 columns -- TWO S stages with each tile's ``P`` written over the first
      ``T/2`` columns of its OWN S stage once the softmax has copied S out (the Prims TS context layout), so the MMA warp issues
      ``QK(n + 2)`` into the stage ``PV(n)`` has just consumed and the tensor core runs a full tile ahead of the softmax; ``O_m`` in the
      per-CTA global scratch as before."""
    return "s2" if not om_in_tmem(D) else "om"


# named barrier ids (1 is free; 0 is the CTA barrier)
BAR_TMEM_ALLOC = 2
BAR_CORRECTION = 3
BAR_XCHG = 4  # (+ warpgroup index, up to 5) the softmax warps' P-gather barrier (the replicated-row placement)
BAR_FOLD = 6  # the correction warpgroup's arrival barrier of the in-kernel chunk fold
BAR_SOFTMAX_ALL = 7  # every softmax thread has copied S out (P over S with two softmax warpgroups sharing TMEM lanes)
REP_ROWS = 32  # units with at most this many live rows take the replicated-row placement (one warp's lanes)

REGS_SOFTMAX_W2 = 160  # per softmax thread with two softmax warpgroups (64 keys per thread): 2 x 160 x 128 + 160 x 128 + 32 x 128 = 65536
REGS_CORRECTION_W2 = 160
STATS_SLOTS = 4  # softmax -> correction payload ring depth (one entry per tile: alpha, m', r_slices)
STATS_SLOTS_PP = 8  # the same ring under the ping-pong placement: 2 entries per tile x 4 tiles (both warpgroups' tiles in tile order)
SLOT_KEYS_256 = 128  # keys per ring slot at D = 256: whole tiles (two 64 KB slots in the 128 KB ring). Half-tile slots (four of 32 KB)
# let the producer run further ahead, but made QK^T two 128 x 64 x 16 instructions per K-step instead of one 128 x 128 x 16: the
# MMA warp issues one tcgen05.mma per fixed interval whatever its N, so the tile's 48 instructions cost more issue time than
# its tensor work; with whole slots the K / V loads of a tile still overlap the previous tile's MMAs (each slot's load is
# issued when its predecessor's MMAs complete, a tile ahead of its use)
Q_STAGES = 2  # Q tiles in smem: the load warp prefetches the next unit's Q while the MMA finishes this one
REGS_SOFTMAX, REGS_OTHER = 192, 32
# the one-warpgroup body at these head dims: 200 per softmax thread and 96 for the correction (16-column pieces). The softmax's
# exponential stream schedules better with the headroom and the correction's per-tile loop stays spill-free at 96; measured faster at
# every prefill shape on SM107 (the paired placement keeps 192 / 96: its budget is full)
REGS_1WG_DIMS = (64,)
REGS_SOFTMAX_1WG, REGS_CORRECTION_1WG = 200, 96
REGS_Q2_DIMS = (128,)
REGS_Q2_128 = (184, 80, 64)


class Roles:
    """Warp layout for ``n_wg`` softmax warpgroups (a placement knob): the softmax warpgroups first (all of
    them work on every tile, warpgroup k holding key slice k of the row: the warps of the warpgroups share
    their TMEM lanes), then the correction warpgroup, then MMA, load and empty warps completing the last
    warpgroup. Register budget: 65536 >= REGS_SOFTMAX x 128 x n_wg + REGS_CORRECTION x 128 + 32 x 128."""

    def __init__(self, n_wg: int, D: int = 128, q2: bool = False):
        assert n_wg in (1, 2), n_wg
        self.n_wg = n_wg
        self.softmax_warps = tuple(range(4 * n_wg))
        self.correction_warps = tuple(range(4 * n_wg, 4 * n_wg + 4))
        self.mma_warp = 4 * n_wg + 4
        self.load_warp = self.mma_warp + 1
        self.empty_warps = tuple(range(self.mma_warp + 2, self.mma_warp + 4))
        self.threads = 32 * (self.mma_warp + 4)
        # register budgets: one softmax warpgroup -- 192 per softmax thread (a whole row of scores), the correction 128 at D <= 128 and 208
        # when the chunk merge runs through the global scratch (O_m pieces in flight); two softmax warpgroups on key slices -- 160 each
        # (half a row of scores) and 160 for the correction; the paired placement (q2: each warpgroup a whole row) -- 192 and 96 (the
        # paired placement's MMA warp keeps both Q tiles' descriptors at 32). The launch allocates 65536 / threads (168 at 384 threads)
        # and the roles move above or below it
        two_slices = n_wg == 2 and not q2
        self.regs_softmax = REGS_SOFTMAX_W2 if two_slices else REGS_SOFTMAX
        self.regs_other = REGS_OTHER
        if two_slices:
            self.regs_correction = REGS_CORRECTION_W2
        elif q2 and D in REGS_Q2_DIMS:
            # the paired placement at D = 128: the MMA / load warpgroup at 64 so the MMA warp's two Q tiles' descriptors and pipeline
            # state stay in registers (its spills miss L1 next to the 300 KB smem plan and sit on every tile's chain), the correction
            # at 80 with 8-column pieces, the softmax at 184 with the streamed exponential form (StaticPlan.streamed)
            self.regs_softmax, self.regs_correction, self.regs_other = REGS_Q2_128
        elif q2:
            self.regs_correction = 96
        elif D in REGS_1WG_DIMS:
            self.regs_softmax, self.regs_correction = (
                REGS_SOFTMAX_1WG,
                REGS_CORRECTION_1WG,
            )
        else:
            self.regs_correction = 208 if not om_in_tmem(D) else 128
        self.n_tmem_users = 32 * (4 * n_wg + 4 + 1)
        self.regs_launch = (65536 // self.threads) // 8 * 8
        assert (
            self.regs_softmax * 128 * n_wg
            + self.regs_correction * 128
            + self.regs_other * 32 * 4
            <= 65536
        ), (n_wg, self.regs_softmax, self.regs_correction, self.regs_other)


class StaticPlan:
    """Tiled MMAs, tilers, smem/TMEM layouts, stage counts and the producer views."""

    def __init__(
        self,
        D: int,
        g: int,
        page: int,
        kv_slots: int,
        n_wg: int = 1,
        q2: bool = False,
        ragged: bool = False,
        hw_rowmax: bool = False,
        cap: int = 0,
    ):
        """``hw_rowmax``: the softmax takes a tile's row maximum from the TMEM load itself (``tcgen05.ld.red``, the parts that
        have it) instead of the register tree -- the same exact maximum; part of the kernel's identity (``compile.kernel_name``)."""
        self.hw_rowmax = hw_rowmax
        assert D % HALF == 0 and D in HEAD_DIMS_BODY, D
        # q2: the paired placement -- a unit is TWO Q tiles (A, B) of one request and KV head over one chunk range; softmax warpgroup k owns
        # Q tile k whole (every key of its rows, the one-warpgroup tile step), so the two tiles' softmax chains are independent and overlap on
        # the SM; the MMA warp issues both tiles' folds off one K / V stream; the correction warpgroup steps both tiles' statistics, A then B
        # per KV tile. TMEM at D = 64: S_A | S_B | P_A | P_B | O_c,A | O_c,B = 3 T + 2 D = 512 -- P in its own columns, so the MMA warp issues
        # QK(t + 1) as soon as the softmax has copied S(t) out; at D = 128: S_A | S_B | O_c,A | O_c,B = 2 T + 2 D = 512 with P_k over the
        # first T/2 columns of S_k (the D = 256 plan's P-over-S: the softmax stores P after its own copy of S, the MMA warp issues QK_k(t + 1)
        # behind PV_k(t)). O_m of both tiles in smem either way (the row-interleaved OM_VEC layout of the global-scratch merge, 2 x 128 x D
        # fp32). Every tile's computation is the one-warpgroup body's (a placement)
        assert not q2 or (n_wg == 2 and D in (64, 128)), (q2, n_wg, D)
        self.q2 = q2
        # the shared-memory capacity the smem plan is checked against (the execution device's; the current device's when 0): the paired
        # placement at D = 128 fits 328 KB parts only, so the capacity is part of the kernel's identity (compile.kernel_name)
        self.cap = cap or arch_smem_cap()
        # ragged: the K / V of every request packed along the token dimension, viewed as ONE page of total_kv tokens; the producer's
        # box starts at the request's token offset (producer.issue_slot). The box geometry is the paged one at page = T (128 keys per
        # box), so the compile passes page = T; a placement (the keys reach the same smem bytes)
        self.ragged = ragged
        assert not ragged or page == TILE_KEYS, (ragged, page)
        self.n_q = 2 if q2 else 1  # Q tiles per unit
        assert 1 <= g <= Q_ROWS, g
        # any power-of-two page size: a page slab is one TMA box of min(page, slot_keys) keys; the SW128 swizzle is applied by the TMA from the
        # smem address bits, so a slab of fewer than 8 keys (a sub-atom destination) lands where the MMA layout expects it (verified bit
        # for bit against page 128 at every head dim)
        assert page >= 1 and (
            page <= TILE_KEYS and TILE_KEYS % page == 0 or page % TILE_KEYS == 0
        ), page
        assert 2 <= kv_slots <= max_kv_slots(D), (
            kv_slots,
            D,
        )  # smem: q_stages(D) Q stages + ring + payloads <= 227 KB
        self.D, self.g, self.page, self.kv_slots = D, g, page, kv_slots
        self.roles = Roles(n_wg, D, q2)
        self.n_wg = n_wg
        self.softmax_threads = 128 * n_wg
        self.ct = (
            32 if self.roles.regs_correction >= 128 else (8 if q2 else 16)
        )  # the correction's TMEM piece (columns): 32 from 128 registers up, 16 below; 8 under the paired placement, whose correction
        # carries two Q tiles' chains and smem O_m pointers at 96 registers (16-column pieces spilled its tile loop in the ragged build)
        self.T = TILE_KEYS
        self.tokens_per_tile = (
            Q_ROWS // g
        )  # Q tile = tokens_per_tile tokens x g heads <= 128 rows
        self.rows_per_tile = (
            self.tokens_per_tile * g
        )  # 126 at g = 6, 120 at g = 12: the rows the full-tile Q box transfers
        # key slices: warpgroup k holds keys [k T/n_wg, (k+1) T/n_wg) of every row; in the replicated-row placement
        # (units of <= rep_rows live rows) the first rep_rows = (32 // g) g Q rows are loaded into all four 32-row quadrants, every
        # warp holds the live rows, warp wq of warpgroup k takes slice 4k + wq of 4 n_wg key slices and the correction reads rows
        # 0..rep_rows-1 (g <= 32; 30 rows at g = 6, 24 at g = 12). The Q pipeline's transaction count is armed per unit with the
        # bytes its load pattern moves (rows_per_tile or 4 rep_rows rows): with the full-tile count the barrier never completed
        # for g not dividing 128 (24 / 4 and 24 / 2 heads hung)
        # the replicated-row placement is off under the paired placement: units of at most 32 live rows take the full-row path there
        # (the same thread-per-row term; a placement). Its instantiation next to the full-row loop in the same softmax code region made
        # ptxas schedule the full-row exponential stream with its MUFU issue stalls exposed, which cost the paired softmax a fifth of its
        # tile whenever one warpgroup worked alone; at D = 128 its P exchange and four-slice stats payload would also not leave room for
        # the third ring slot next to the two Q tiles and the smem O_m
        self.rep_ok = g <= REP_ROWS and not q2
        # the softmax's exponential form: streamed (the P words take the scores' registers) for the one-warpgroup body and the paired
        # placement at D = 128, where the register split needs the smaller live set; packed for the paired placement at D = 64
        self.streamed = not q2 or D in (128,)
        self.tokens_rep = REP_ROWS // g if self.rep_ok else 1
        self.rep_rows = self.tokens_rep * g if self.rep_ok else 0
        self.n_slices = (
            1 if q2 else n_wg
        )  # key slices of a tile's row (the correction adds their row-sum partials)
        # the row-sum partial slots per stats entry (4 n_slices when the replicated placement can run) and the P exchange buffer
        # (the replicated placement's warps 1..3 -> warp 0 hand-off); a 16-byte stub when no unit of the build takes that placement
        self.r_slices = 4 * self.n_slices if self.rep_ok else self.n_slices
        self.s_p_elems = n_wg * 3 * 32 * 16 if self.rep_ok else 4
        self.kpt = (
            self.T // self.n_slices
        )  # keys per thread (full rows): warpgroup k holds keys [k kpt, (k+1) kpt)
        self.kpt_rep = (
            REP_ROWS // self.n_slices
        )  # keys per thread (replicated rows): warp wq of warpgroup k holds slice 4k + wq
        self.slices_rep = 4 * self.n_slices
        self.stats_slots = STATS_SLOTS_PP if q2 else STATS_SLOTS
        # ring slots: a whole KV tile (T keys) at every D (SLOT_KEYS_256: the half-tile slots of D = 256 were a placement that cost more
        # MMA issue than they saved in load latency); the code keeps the general slots_per_tile form
        self.slot_keys = SLOT_KEYS_256 if D == 256 else self.T
        self.slots_per_tile = self.T // self.slot_keys
        self.n_slots = (
            kv_slots * self.slots_per_tile
        )  # physical ring slots (kv_slots counts whole-tile equivalents)
        self.box_rows = min(page, self.slot_keys)  # keys per TMA box
        self.boxes_per_slot = self.slot_keys // self.box_rows  # per 64-wide half
        self.boxes_per_tile = self.T // self.box_rows
        self.halves = D // HALF
        self.d_steps = D // MMA_K
        self.k_steps = self.T // MMA_K
        self.k_steps_per_slot = self.slot_keys // MMA_K
        self.tpc = tiles_per_chunk(D)  # tiles per chunk
        cta_group = tcgen05.CtaGroup.ONE
        # QK^T: A = Q (K-major, smem), B = K (K-major, smem), (M, N) = (128, T), K = D -- the S accumulator's (128, T) shape
        self.qk_tiled_mma = utils.sm100.make_trivial_tiled_mma(
            IN_DTYPE,
            IN_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            ACC_DTYPE,
            cta_group,
            (Q_ROWS, self.T),
        )
        # the QK^T instruction actually issued: N = slot_keys (the same as above unless the ring holds half tiles)
        self.qks_tiled_mma = (
            self.qk_tiled_mma
            if self.slots_per_tile == 1
            else utils.sm100.make_trivial_tiled_mma(
                IN_DTYPE,
                IN_DTYPE,
                OperandMajorMode.K,
                OperandMajorMode.K,
                ACC_DTYPE,
                cta_group,
                (Q_ROWS, self.slot_keys),
            )
        )
        # PV: A = P (K-major, TMEM), B = V (MN-major, smem), (M, N) = (128, D), K = T
        self.pv_tiled_mma = utils.sm100.make_trivial_tiled_mma(
            IN_DTYPE,
            IN_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            ACC_DTYPE,
            cta_group,
            (Q_ROWS, D),
            OperandSource.TMEM,
        )
        self.qk_tiler = (Q_ROWS, self.T, D)
        self.qks_tiler = (Q_ROWS, self.slot_keys, D)
        self.pv_tiler = (Q_ROWS, D, self.T)
        self.pvs_tiler = (Q_ROWS, D, self.slot_keys)
        self.q_stages = q_stages_for(D, q2)
        self.q_tiles = (
            self.q_stages * self.n_q
        )  # Q tiles in smem: stage s holds tiles n_q s .. n_q s + n_q - 1
        self.ws_row = ws_row(D)  # fp32 per partial row of the split workspace
        self.q_smem_layout = utils.sm100.make_smem_layout_a(
            self.qks_tiled_mma, self.qks_tiler, IN_DTYPE, self.q_tiles
        )
        self.k_smem_layout = utils.sm100.make_smem_layout_b(
            self.qks_tiled_mma, self.qks_tiler, IN_DTYPE, self.n_slots
        )  # (slot_keys x D) per slot
        self.v_smem_layout = utils.sm100.make_smem_layout_b(
            self.pv_tiled_mma, self.pvs_tiler, IN_DTYPE, self.n_slots
        )  # (D x slot_keys) per slot
        self.p_tmem_layout = utils.sm100.make_smem_layout_a(
            self.pv_tiled_mma, self.pv_tiler, IN_DTYPE, 1
        )
        k_one = cute.select(self.k_smem_layout, mode=[0, 1, 2])
        v_one = cute.select(self.v_smem_layout, mode=[0, 1, 2])
        self.kv_slot_elems = cute.cosize(k_one)
        assert cute.cosize(v_one) == self.kv_slot_elems == self.slot_keys * D, (
            cute.cosize(v_one),
            self.kv_slot_elems,
        )
        self.kv_tile_elems = self.kv_slot_elems * self.slots_per_tile
        self.swizzle = k_one.inner
        # the producer's view of the ring: (d in half, key in box, 1, 1, box, half, slot) -> element offset; the two
        # unit modes mirror the 4-D (D, page, Hkv, num_pages) gmem tiler (64, box_rows, 1, 1)
        self.slab_layout = cute.make_layout(
            (HALF, self.box_rows, 1, 1), stride=(1, HALF, 0, 0)
        )
        self.ring_view_layout = cute.make_composed_layout(
            self.swizzle,
            0,
            cute.make_layout(
                (
                    HALF,
                    self.box_rows,
                    1,
                    1,
                    self.boxes_per_slot,
                    self.halves,
                    self.n_slots,
                ),
                stride=(
                    1,
                    HALF,
                    0,
                    0,
                    HALF * self.box_rows,
                    self.slot_keys * HALF,
                    self.kv_slot_elems,
                ),
            ),
        )
        # the producer's view of the Q tile: (d in half, head, token, half, stage)
        self.q_box = (HALF, g, self.tokens_per_tile)
        self.q_tile_elems = Q_ROWS * D
        self.q_view_layout = cute.make_composed_layout(
            self.swizzle,
            0,
            cute.make_layout(
                (HALF, g, self.tokens_per_tile, self.halves, self.q_tiles),
                stride=(1, HALF, HALF * g, Q_ROWS * HALF, self.q_tile_elems),
            ),
        )
        self.q_slab_layout = cute.make_composed_layout(
            self.swizzle, 0, cute.make_layout(self.q_box, stride=(1, HALF, HALF * g))
        )
        # the replicated load: box (64 d, g heads, 32/g tokens) = one 32-row quadrant; view (d, head, token, quadrant, half, stage)
        self.q_box_rep = (HALF, g, self.tokens_rep)
        self.q_slab_rep_layout = cute.make_composed_layout(
            self.swizzle,
            0,
            cute.make_layout(self.q_box_rep, stride=(1, HALF, HALF * g)),
        )
        self.q_view_rep_layout = cute.make_composed_layout(
            self.swizzle,
            0,
            cute.make_layout(
                (
                    HALF,
                    g,
                    self.tokens_rep,
                    Q_ROWS // REP_ROWS,
                    self.halves,
                    self.q_tiles,
                ),
                stride=(
                    1,
                    HALF,
                    HALF * g,
                    HALF * REP_ROWS,
                    Q_ROWS * HALF,
                    self.q_tile_elems,
                ),
            ),
        )
        self.kv_slab_layout = cute.make_composed_layout(
            self.swizzle, 0, self.slab_layout
        )
        # TMEM (tmem_plan_for): "om" at D <= 128 -- S (T cols) | P (T/2 cols, bf16 pairs) | O_c (D cols) | O_m (D cols) = 320 / 448
        # columns (one S / P stage; O_m in TMEM); "s2" at D = 256 -- O_c (D) | S_0 (T) | S_1 (T) = 512: two S stages, P over the first
        # T/2 columns of its own S stage (the softmax copies S out before it writes P), O_m in the per-CTA global scratch (om_tmem False)
        self.tmem_plan = tmem_plan_for(D)
        self.om_tmem = om_in_tmem(D)
        self.tmem_q2_stride = 0  # TMEM columns from Q tile A's O_c to Q tile B's
        self.tmem_p_stride = (
            None  # columns between the P stages (None: the S stride, P over its own S)
        )
        if q2:
            self.tmem_plan = "q2"
            self.s_stages = 2  # S / P stage k = Q tile k
            self.qk_ahead = 1
            self.tmem_s = 0
            self.tmem_s_stride = self.T
            if D == 64:
                # S_A | S_B | P_A | P_B | O_c,A | O_c,B: P in its own columns
                self.tmem_p = 2 * self.T
                self.tmem_p_stride = self.T // 2
                self.tmem_oc = self.tmem_p + self.T
            else:
                # S_A | S_B | O_c,A | O_c,B: P_k over the first T/2 columns of S_k (p_over_s; the MMA warp's issue order keeps
                # QK_k(t + 1) behind PV_k(t): mma.mma_loop_q2)
                self.tmem_p = self.tmem_s
                self.tmem_p_stride = self.T
                self.tmem_oc = 2 * self.T
            self.tmem_om = None
            self.tmem_q2_stride = D
            self.om_tmem = False  # O_m in smem (StaticPlan.om_smem)
            assert self.tmem_oc + 2 * D <= TMEM_COLS
        elif self.tmem_plan == "s2":
            self.s_stages = 2
            self.qk_ahead = 2  # the MMA warp issues QK two tiles ahead of PV (K two tiles ahead in the ring)
            self.tmem_oc = 0
            self.tmem_s = D
            self.tmem_s_stride = self.T  # columns between the S stages (P_k = S_k + 0)
            self.tmem_p = self.tmem_s
            self.tmem_om = None
            assert self.tmem_s + 2 * self.T <= TMEM_COLS
        else:
            self.s_stages = 1
            self.qk_ahead = 1
            self.tmem_s = 0
            self.tmem_p = self.T
            self.tmem_oc = self.T + self.T // 2
            self.tmem_s_stride = 0
            self.tmem_om = self.tmem_oc + D if self.om_tmem else None
            assert self.tmem_oc + D <= TMEM_COLS
        if self.tmem_p_stride is None:
            self.tmem_p_stride = self.tmem_s_stride
        self.om_smem = (
            q2  # O_m in smem (the paired placement), else TMEM or the global scratch
        )
        self.p_over_s = (
            self.tmem_p == self.tmem_s
        )  # P shares its S stage's columns: the P store must follow EVERY S load of the tile
        assert (
            cute.cosize(cute.select(self.q_smem_layout, mode=[0, 1, 2]))
            == self.q_tile_elems
        )
        self.q_bytes = cute.size_in_bytes(
            IN_DTYPE, cute.select(self.q_smem_layout, mode=[0, 1, 2])
        )  # the whole stage (128 rows)
        self.q_bytes_full = (
            self.rows_per_tile * D * 2
        )  # what the full-tile load moves (= q_bytes when g divides 128)
        self.q_bytes_rep = (
            4 * self.rep_rows * D * 2
        )  # what the replicated load moves (= q_bytes when g divides 32)
        self.kv_tile_bytes = self.kv_tile_elems * 2
        self.kv_slot_bytes = (
            self.kv_slot_elems * 2
        )  # the ring pipeline's transaction count (one slot)
        # the exact smem footprint against the target's capacity (the ring depth is a placement: a shallower ring on a smaller part
        # gives the same bits); the paired placement at D = 128 (two Q tiles, the 128 KB smem O_m, the ring) needs a 328 KB part
        self.smem_bytes = shared_storage_struct(self).size_in_bytes()
        assert self.smem_bytes <= max(SMEM_CAP, self.cap), (
            f"smem plan of {self.smem_bytes} bytes exceeds the {self.cap}-byte capacity of the target"
            + (
                " (the paired placement at D = 128 fits 328 KB parts only)"
                if q2 and D == 128
                else ""
            ),
            D,
            kv_slots,
            q2,
        )


def shared_storage_struct(st: StaticPlan):
    """The kernel's ``SharedStorage`` for the plan ``st``: the pipeline barriers, the softmax -> correction payload (stats ring, the
    replicated placement's P exchange), the paired placement's smem ``O_m``, then the 1 KB-aligned Q tiles and the K/V ring. Built here
    so the plan measures its exact footprint against the part's capacity instead of estimating the payload."""
    i64, i32, f32, mr = (
        cutlass.Int64,
        cutlass.Int32,
        cutlass.Float32,
        cute.struct.MemRange,
    )
    # built from an annotations dict of real types: this module's postponed annotations would hand ``cute.struct`` strings
    fields = dict(
        q_mbar=mr[i64, 2 * st.q_stages],
        kv_mbar=mr[i64, 2 * st.n_slots],
        s_mbar=mr[i64, 2 * st.s_stages],
        p_mbar=mr[i64, 2 * max(st.s_stages, st.n_q)],
        o_mbar=mr[i64, 2 * st.n_q],
        stats_mbar=mr[i64, 2 * st.stats_slots],
        tmem_dealloc_mbar=i64,
        tmem_holding_buf=i32,
        s_fold=i32,
        s_alpha=mr[f32, st.stats_slots * Q_ROWS],
        s_m=mr[f32, st.stats_slots * Q_ROWS],
        s_r=mr[f32, st.stats_slots * st.r_slices * Q_ROWS],
        s_p=cute.struct.Align[mr[f32, st.s_p_elems], 16],
        s_mx=mr[f32, 2 * st.n_slices * Q_ROWS],
        sOm=cute.struct.Align[mr[f32, st.n_q * Q_ROWS * st.D if st.om_smem else 4], 16],
        sQ=cute.struct.Align[mr[IN_DTYPE, st.q_tiles * st.q_tile_elems], 1024],
        sKV=cute.struct.Align[mr[IN_DTYPE, st.n_slots * st.kv_slot_elems], 1024],
    )
    return cute.struct(type("SharedStorage", (), {"__annotations__": fields}))


class TmaSetup:
    """TMA atoms over the producer's slab views: Q boxes ``(64 d, g heads, tokens_per_tile tokens)``
    from the (D, Hq, total_q) view; K and V boxes ``(64 d, box_rows keys)`` from the
    (D, page, Hkv, num_pages) views (NHD and HND pools differ only in runtime strides)."""

    def __init__(
        self, st: StaticPlan, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor
    ):
        op = cpasync.CopyBulkTensorTileG2SOp()
        self.atom_q, self.tensor_q = cpasync.make_tiled_tma_atom(
            op, mQ, st.q_slab_layout, st.q_box
        )
        self.atom_q_rep, self.tensor_q_rep = cpasync.make_tiled_tma_atom(
            op, mQ, st.q_slab_rep_layout, st.q_box_rep
        )
        self.atom_k, self.tensor_k = cpasync.make_tiled_tma_atom(
            op, mK, st.kv_slab_layout, (HALF, st.box_rows, 1, 1)
        )
        self.atom_v, self.tensor_v = cpasync.make_tiled_tma_atom(
            op, mV, st.kv_slab_layout, (HALF, st.box_rows, 1, 1)
        )
