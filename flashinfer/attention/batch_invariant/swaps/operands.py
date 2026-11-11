# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host-side operand block of the Swaps-AB decode body: the class's term for work units of at most 32
live rows (decode and spec-decode: ``g x q_len`` rows per (request, kv head, chunk range)) with the MMA roles swapped --

* ``S^T = K Q^T``: A = the K tile (M = 128 keys, K-major, smem), B = the unit's Q rows (N = ``N_pad`` in {8, 16, 32}, K-major, smem), the
  same ``D / 16`` ascending K-steps per score (the GEMM's ``kfold``: the opaque ``MMA16`` step of element ``(k, n)`` is the step of element
  ``(n, k)`` under the row body -- the A2 surface at these instruction shapes, tested bit for bit against the row body);
* ``O^T = V^T P^T``: A = the V tile read as ``(dims x keys)`` (MN-major, the same smem bytes as the row body's V operand), B = ``P^T``
  (``N_pad`` rows x 128 keys, K-major, written by the softmax to smem), M = 128 dims per slot (64 at D = 64), the same 8 ascending
  key-steps per tile with ``ACCUMULATE`` off on a chunk's first step.

Keys sit on TMEM lanes: the softmax thread ``t`` owns key ``t`` of every tile and ``N_pad`` scores; the tile's row maximum and row sum are the
engine's pairwise trees over the key digit placed ``key: [(32, lane), (4, warp)]`` (xor shuffles, then the four warp partials in warp order).
Per CTA: Q ``N_pad x D`` (2-16 KB, two stages), ``P^T`` ``N_pad x 128`` bf16 (2-8 KB, two stages), the K / V ring of 128-key slots of
``min(D, 128)`` dims (16 / 32 / 32 KB at D = 64 / 128 / 256; two slots per tile at D = 256) as deep as 227 KB allows (12 / 6 / 5 tiles at
N_pad = 32); TMEM ``S^T`` (2 stages x N_pad) | ``O^T`` (slots x N_pad) | ``O_m^T`` (slots x N_pad): the merged chunk state is in TMEM at every D.
Two softmax warpgroups work on alternate tiles (ping-pong): the per-tile chain of one warpgroup is latency-bound (two five-level shuffle
reductions, the exchange barrier, the exponentials), so two of them overlap; the running row maximum crosses between them through a
one-slot smem hand-off per direction (the online softmax's `m` is a serial state across tiles).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

from ..plan import ws_row
from ..spec import MMA_K, TILE_KEYS, tiles_per_chunk
from ..blocks.operands import (
    ACC_DTYPE,
    HALF,
    IN_DTYPE,
    SMEM_CAP,
)

N_PADS = (
    8,
    16,
    32,
)  # the B-operand N of the swapped MMAs: the smallest holding the batch's widest unit (rows = g x q_len <= 32)
Q_STAGES = 2
P_STAGES = 2
S_STAGES = 2  # TMEM S^T stages: one per softmax warpgroup (tile n in stage n % 2; QK(n + 1) runs while S^T(n) is read)
N_SOFTMAX_WGS = 2  # softmax warpgroups on alternate tiles (ping-pong): warpgroup k takes the tiles of parity k
MAX_RING_SLOTS = 16  # physical ring slots (mbarrier pairs): 12 fit the 228 KB parts at D = 64; sm_107's 328 KB reach 16
PAYLOAD_BYTES = (
    10 << 10
)  # stats rings, exchange buffers, coefficient arrays, barriers (SharedStorage below the tiles)
TMEM_COLS_SWAPS = 256  # allocated columns (a power of two >= the plan's columns)
REGS_CORRECTION_SWAPS = 128
REGS_SOFTMAX_SWAPS = 152  # per softmax warpgroup (n_pad scores + the row chains); the launch allocation is 65536 / 512 = 128
REGS_OTHER_SWAPS = 56  # the MMA / load / empty warpgroup: 2 x 152 + 128 + 56 = 488 <= 512 per lane-slot

# named barriers (0 = the CTA barrier, 1 free; the row body's ids are not shared with this kernel)
BAR_TMEM_ALLOC = 2
BAR_CORRECTION = 3
BAR_SOFTMAX = 4  # (+ warpgroup index: 4, 5) the softmax warpgroup's row-maximum exchange (the top two levels of the key tree)
BAR_FOLD = 6  # the correction warpgroup's arrival barrier of the in-kernel chunk fold (fold_group)
BAR_COEF = 7  # the correction warpgroup: the per-row coefficients (alpha / a1, a2 / inv) are in smem


def n_pad_for(rows: int) -> int:
    """The B-operand N for a unit of ``rows`` live rows (``rows <= 32``)."""
    assert 1 <= rows <= 32, rows
    return 8 if rows <= 8 else (16 if rows <= 16 else 32)


def slot_dims(D: int) -> int:
    return min(D, 128)


def slot_bytes(D: int) -> int:
    """One ring slot: 128 keys x ``slot_dims(D)`` bf16 (16 / 32 / 32 KB)."""
    return TILE_KEYS * slot_dims(D) * 2


def q_bytes(D: int, n_pad: int) -> int:
    return n_pad * D * 2


def p_bytes(n_pad: int) -> int:
    return n_pad * TILE_KEYS * 2


STATS_SLOTS_PP = 8  # the stats ring: 2 entries per tile x 4 tiles (both warpgroups' tiles in tile order)


def shared_storage_struct(
    q_stages: int,
    n_slots: int,
    s_stages: int,
    p_stages: int,
    q_stage_elems: int,
    p_stage_elems: int,
    kv_slot_elems: int,
):
    """The kernel's ``SharedStorage`` for a ring of ``n_slots`` (the barriers, the stats and exchange payload, then the 1 KB-aligned Q, P
    and K/V tiles); built here so the ring budget below can measure the exact footprint instead of estimating the payload."""

    i64, i32, f32, mr = (
        cutlass.Int64,
        cutlass.Int32,
        cutlass.Float32,
        cute.struct.MemRange,
    )
    # built from an annotations dict of real types: this module's postponed annotations would hand ``cute.struct`` strings
    fields = dict(
        q_mbar=mr[i64, 2 * q_stages],
        kv_mbar=mr[i64, 2 * n_slots],
        s_mbar=mr[i64, 2 * s_stages],
        p_mbar=mr[i64, 2 * p_stages],
        o_mbar=mr[i64, 2],
        stats_mbar=mr[i64, 2 * STATS_SLOTS_PP],
        hm_mbar=mr[i64, 2 * N_SOFTMAX_WGS],
        tmem_dealloc_mbar=i64,
        tmem_holding_buf=i32,
        s_fold=i32,
        s_alpha=mr[f32, STATS_SLOTS_PP * 128],
        s_m=mr[f32, STATS_SLOTS_PP * 128],
        s_r=mr[f32, STATS_SLOTS_PP * 4 * 128],
        s_x=mr[f32, N_SOFTMAX_WGS * 2 * 4 * 32],
        s_hm=mr[f32, N_SOFTMAX_WGS * 32],
        s_ab=mr[f32, 2 * 128],
        s_inv=mr[f32, 128],
        sQ=cute.struct.Align[mr[IN_DTYPE, q_stages * q_stage_elems], 1024],
        sP=cute.struct.Align[mr[IN_DTYPE, p_stages * p_stage_elems], 1024],
        sKV=cute.struct.Align[mr[IN_DTYPE, n_slots * kv_slot_elems], 1024],
    )
    return cute.struct(type("SharedStorage", (), {"__annotations__": fields}))


def arch_smem_cap(device=None) -> int:
    """The dynamic shared memory one CTA may allocate on the compile target of ``device`` (default: the current device): the DSL's
    per-architecture table (328 KB on sm_107, 228 KB on sm_100 / sm_103; the CUDA device attribute under-reports sm_107),
    ``SMEM_CAP`` when the target is not in the table."""
    import torch

    from ....cute_dsl.availability import cute_dsl_compile_arch
    from ....utils import get_compute_capability

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    major, minor = get_compute_capability(torch.device(device))
    arch = cute_dsl_compile_arch(int(major), int(minor))  # "sm_107a", "sm_100f", ...
    key = "sm_" + arch[3:].rstrip("af")
    try:
        from cutlass.cutlass_dsl.cutlass import SMEM_CAPACITY_MAP

        return int(SMEM_CAPACITY_MAP.get(key, SMEM_CAP))
    except ImportError:
        return {"sm_107": 334848}.get(key, SMEM_CAP)


def ring_slots(D: int, n_pad: int, cap: int | None = None) -> int:
    """Physical ring slots: everything left under ``SMEM_CAP`` after the Q and P stages and the payload, then trimmed a tile at a time until
    the exact ``SharedStorage`` fits the compile target's capacity ``cap`` (``arch_smem_cap()``). The ring depth is a placement: it buffers
    the same tiles in the same order, so a shallower ring on a smaller part gives the same bits."""
    cap = SMEM_CAP if cap is None else cap
    free = (
        max(SMEM_CAP, cap)
        - Q_STAGES * q_bytes(D, n_pad)
        - P_STAGES * p_bytes(n_pad)
        - PAYLOAD_BYTES
    )
    n = free // slot_bytes(D)
    spt = D // slot_dims(D)
    n = (n // spt) * spt  # whole tiles
    n = min(n, MAX_RING_SLOTS - MAX_RING_SLOTS % spt)
    while (
        n > 2 * spt
        and shared_storage_struct(
            Q_STAGES,
            n,
            S_STAGES,
            P_STAGES,
            n_pad * D,
            n_pad * TILE_KEYS,
            TILE_KEYS * slot_dims(D),
        ).size_in_bytes()
        > cap
    ):
        n -= spt
    assert n >= 2 * spt, (D, n_pad, n)
    return n


class Roles:
    """Warp layout: two softmax warpgroups (warps 0-3 and 4-7: thread = key; warpgroup k takes the tiles of parity k), the correction
    warpgroup (8-11: thread = TMEM lane = a dim of ``O^T``), the MMA warp (12), the load warp (13) and two empty warps. Registers
    152 / 152 / 128 / 56 (the launch allocation is 128 per thread at 512 threads; the last warpgroup's decrease funds the softmax increase)."""

    softmax_warps = tuple(range(8))
    correction_warps = (8, 9, 10, 11)
    mma_warp = 12
    load_warp = 13
    empty_warps = (14, 15)
    threads = 512
    regs_softmax = REGS_SOFTMAX_SWAPS
    regs_correction = REGS_CORRECTION_SWAPS
    regs_other = REGS_OTHER_SWAPS
    n_tmem_users = 32 * 13
    regs_launch = (65536 // 512) // 8 * 8
    assert (
        4 * regs_launch
        >= 2 * REGS_SOFTMAX_SWAPS + REGS_CORRECTION_SWAPS + REGS_OTHER_SWAPS
    )


class SwapsPlan:
    """Tiled MMAs, tilers, smem / TMEM layouts, stage counts and the producer views of the swapped body."""

    def __init__(
        self, D: int, g: int, page: int, n_pad: int, ragged: bool = False, cap: int = 0
    ):
        assert D in (64, 128, 256) and n_pad in N_PADS and 1 <= g <= 32, (D, n_pad, g)
        assert page >= 1 and (
            page <= TILE_KEYS and TILE_KEYS % page == 0 or page % TILE_KEYS == 0
        ), page
        self.D, self.g, self.page, self.n_pad = D, g, page, n_pad
        self.ragged = (
            ragged  # the packed-token KV view (blocks/operands.StaticPlan.ragged)
        )
        assert not ragged or page == TILE_KEYS, (ragged, page)
        self.roles = Roles
        self.n_wg = 1  # one softmax warpgroup (fold_group's prefetch depth reads it)
        self.T = TILE_KEYS
        self.tpc = tiles_per_chunk(D)
        self.slot_dims = slot_dims(D)
        self.slots_per_tile = D // self.slot_dims  # 1, 1, 2
        self.halves = self.slot_dims // HALF  # 64-wide d halves per slot: 1, 2, 2
        self.halves_tile = D // HALF
        self.slot_keys = self.T
        self.n_slots = ring_slots(D, n_pad, cap or arch_smem_cap())
        self.kv_slots = self.n_slots // self.slots_per_tile  # tiles in flight
        self.box_rows = min(page, self.T)  # keys per TMA box
        self.boxes_per_slot = self.T // self.box_rows
        self.d_steps = D // MMA_K  # QK^T K-steps per tile (over the slots)
        self.d_steps_slot = self.slot_dims // MMA_K
        self.k_steps = self.T // MMA_K  # PV^T K-steps per tile
        self.M_pv = self.slot_dims  # 128, or 64 at D = 64
        self.q_stages, self.p_stages, self.s_stages = Q_STAGES, P_STAGES, S_STAGES
        self.ws_row = ws_row(D)
        # the Q box: (64 d, g heads, tokens_box tokens) per 64-wide half; tokens_box x g = rows_box <= n_pad rows of the B operand
        self.tokens_box = n_pad // g
        self.rows_box = self.tokens_box * g
        assert 1 <= self.rows_box <= n_pad
        cta_group = tcgen05.CtaGroup.ONE
        # S^T = K Q^T: A = K slot (M = 128 keys, K-major), B = Q (N = n_pad rows, K-major); the instruction's K extent is one slot's dims
        self.qk_tiled_mma = utils.sm100.make_trivial_tiled_mma(
            IN_DTYPE,
            IN_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            ACC_DTYPE,
            cta_group,
            (self.T, n_pad),
        )
        # O^T = V^T P^T: A = V slot as (M = slot_dims dims, K = 128 keys) MN-major, B = P^T (N = n_pad rows, K = 128 keys) K-major
        self.pv_tiled_mma = utils.sm100.make_trivial_tiled_mma(
            IN_DTYPE,
            IN_DTYPE,
            OperandMajorMode.MN,
            OperandMajorMode.K,
            ACC_DTYPE,
            cta_group,
            (self.M_pv, n_pad),
        )
        self.qk_tiler = (self.T, n_pad, self.slot_dims)  # per K slot
        self.q_tiler = (self.T, n_pad, D)  # the whole Q (all slots' K-steps)
        self.pv_tiler = (self.M_pv, n_pad, self.T)
        self.k_smem_layout = utils.sm100.make_smem_layout_a(
            self.qk_tiled_mma, self.qk_tiler, IN_DTYPE, self.n_slots
        )  # (128 keys x slot_dims) per slot
        self.q_smem_layout = utils.sm100.make_smem_layout_b(
            self.qk_tiled_mma, self.q_tiler, IN_DTYPE, self.q_stages
        )  # (n_pad x D) per stage
        self.v_smem_layout = utils.sm100.make_smem_layout_a(
            self.pv_tiled_mma, self.pv_tiler, IN_DTYPE, self.n_slots
        )  # (slot_dims x 128 keys) per slot
        self.p_smem_layout = utils.sm100.make_smem_layout_b(
            self.pv_tiled_mma, self.pv_tiler, IN_DTYPE, self.p_stages
        )  # (n_pad x 128 keys) per stage
        k_one = cute.select(self.k_smem_layout, mode=[0, 1, 2])
        v_one = cute.select(self.v_smem_layout, mode=[0, 1, 2])
        self.kv_slot_elems = cute.cosize(k_one)
        assert cute.cosize(v_one) == self.kv_slot_elems == self.T * self.slot_dims, (
            cute.cosize(v_one),
            self.kv_slot_elems,
        )
        self.swizzle = k_one.inner
        q_one = cute.select(self.q_smem_layout, mode=[0, 1, 2])
        self.q_stage_elems = cute.cosize(q_one)
        assert self.q_stage_elems == n_pad * D, (self.q_stage_elems, n_pad, D)
        p_one = cute.select(self.p_smem_layout, mode=[0, 1, 2])
        self.p_stage_elems = cute.cosize(p_one)
        assert self.p_stage_elems == n_pad * self.T, (self.p_stage_elems, n_pad)
        # the producer's view of the ring: (d in half, key in box, 1, 1, box, half, slot) -> element; a page slab of box_rows keys in one
        # 64-wide half is contiguous ((d % 64) + 64 key + 64 T (d // 64) inside a slot: the row body's probed form at slot_dims = D)
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
                    self.T * HALF,
                    self.kv_slot_elems,
                ),
            ),
        )
        self.kv_slab_layout = cute.make_composed_layout(
            self.swizzle, 0, self.slab_layout
        )
        # the producer's view of Q: (d in half, head, token, half, stage): element (n, d) of the B operand at (d % 64) + 64 n + 64 n_pad (d // 64)
        self.q_box = (HALF, g, self.tokens_box)
        self.q_view_layout = cute.make_composed_layout(
            self.swizzle,
            0,
            cute.make_layout(
                (HALF, g, self.tokens_box, self.halves_tile, self.q_stages),
                stride=(1, HALF, HALF * g, HALF * n_pad, self.q_stage_elems),
            ),
        )
        self.q_slab_layout = cute.make_composed_layout(
            self.swizzle, 0, cute.make_layout(self.q_box, stride=(1, HALF, HALF * g))
        )
        self.q_tx_bytes = (
            self.rows_box * D * 2
        )  # what the Q boxes move (the barrier's transaction count)
        self.kv_slot_bytes = self.kv_slot_elems * 2
        # TMEM: S^T stages | O^T per slot | O_m^T per slot (all n_pad columns wide)
        self.tmem_s = 0
        self.tmem_s_stride = n_pad
        self.tmem_o = self.s_stages * n_pad
        self.tmem_om = self.tmem_o + self.slots_per_tile * n_pad
        self.tmem_cols = self.tmem_om + self.slots_per_tile * n_pad
        assert self.tmem_cols <= TMEM_COLS_SWAPS, self.tmem_cols
        total = (
            self.q_stages * q_bytes(D, n_pad)
            + self.p_stages * p_bytes(n_pad)
            + self.n_slots * slot_bytes(D)
            + PAYLOAD_BYTES
        )
        assert total <= max(SMEM_CAP, arch_smem_cap()), (D, n_pad, total)


class SwapsTma:
    """TMA atoms: Q boxes ``(64 d, g heads, tokens_box tokens)`` from the (D, Hq, total_q) view; K and V boxes ``(64 d, box_rows keys)``
    from the (D, page, Hkv, num_pages) views."""

    def __init__(
        self, st: SwapsPlan, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor
    ):
        op = cpasync.CopyBulkTensorTileG2SOp()
        self.atom_q, self.tensor_q = cpasync.make_tiled_tma_atom(
            op, mQ, st.q_slab_layout, st.q_box
        )
        self.atom_k, self.tensor_k = cpasync.make_tiled_tma_atom(
            op, mK, st.kv_slab_layout, (HALF, st.box_rows, 1, 1)
        )
        self.atom_v, self.tensor_v = cpasync.make_tiled_tma_atom(
            op, mV, st.kv_slab_layout, (HALF, st.box_rows, 1, 1)
        )
