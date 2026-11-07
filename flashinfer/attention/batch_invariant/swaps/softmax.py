# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The softmax block of the Swaps-AB body: the spec's tile step with the KEYS on the threads. Softmax thread ``t`` (= TMEM lane ``t``) owns
key ``t`` of every tile and the ``n_pad`` scores ``S^T[t, n]`` of the unit's rows. Per tile a thread

* copies its ``n_pad`` scores out of TMEM (one ``Ld32x32b x n_pad``), masks them by absolute key per row (``select``; only on boundary
  tiles of live rows),
* reduces the tile's row maximum over the 128 keys: a warp-wide ``redux.sync`` ``fmax`` over the warp's 32 adjacent keys (the same bits as the
  engine's ``rowmax`` plan's pairwise shuffle tree: ``max`` is exact under the declared total order) in every warp, then the four
  warp partials through smem in warp order -- the top two levels of the 128-leaf tree;
* forms ``m' = fmax(m, m_t)`` and ``alpha`` per row exactly as ``spec.alpha_term`` (every thread redundantly: the 128 threads hold identical
  ``m`` chains), hands ``alpha`` to the correction warpgroup at once (its ``O^T`` rescale overlaps the exponentials),
* forms the probabilities ``p_n = EX2(fma(s_n, c, mul(m', nc)))`` of its key for every row (``fma2`` pairs of rows), reduces them over the
  warp's keys with the ``rowsum`` plan under the same lane placement and hands ``(m', r_warp)`` to the correction warpgroup, whose
  ``row_sum`` adds the four warp partials in warp order (the tree's top levels) and owns the ``l`` chain, the chunk merges and the normaliser,
* converts its ``n_pad`` probabilities to bf16 and stores them into ``P^T`` in smem (the K-major SW128 B operand of ``O^T = V^T P^T``:
  element ``(n, k)`` at ``(k % 64 ^ 8 (n % 8)) + 64 n + 64 n_pad (k // 64)``), fences the generic writes for the async proxy and commits.

A tile with no visible key for a row is a *null step* for that row (``alpha = 1``, ``P^T`` column ``+0``, ``m`` untouched). Padding rows
(``n >= rows``) are never masked and never stored: their columns of ``S^T`` / ``P^T`` / ``O^T`` are their own.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import tcgen05

from ....batch_invariant import canon
from ....batch_invariant.fp_interface.cute import CuteF32
from .. import spec
from ..blocks.operands import ACC_DTYPE, IN_DTYPE
from .operands import BAR_SOFTMAX

_NEG_INF = float("-inf")
# the warp-wide ``fmax`` reduction (``warp_redux_sync``) returns the same bits as the shuffle fmax tree, including on
# subnormals, +-0, one NaN and all-NaN
LANES = 32


class _OneLoader:
    """``canon.init_state`` leaf loader of a lane-only plan: the thread's single value (a plain callable object, not a closure)."""

    def __init__(self, v):
        self.v = v

    def __call__(self, key):
        return self.v


class _LanePeers(canon.Peers):
    """The engine's ``Peers`` for the lane factor: ``shuffle_xor`` is ``shfl.sync.bfly`` (every lane ends with the warp's value)."""

    def shuffle_xor(self, key, value, mask):
        return cute.arch.shuffle_sync_bfly(value, offset=mask)

    def exchange(
        self, step, state
    ):  # pragma: no cover - the warp factor's top levels run through smem in warp order (unit_loop)
        raise NotImplementedError


def _pairwise(vals, combine):
    """Balanced pairwise tree over a static list (plain Python: trace-time structure; the engine's ``_pairwise`` order)."""
    while len(vals) > 1:
        vals = [combine(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    return vals[0]


def lane_masks(plan) -> list:
    """The xor masks of the plan's lane steps, in the engine's order (asserted: five lane steps, nothing else)."""
    masks = [s.lane_mask for s in plan.steps]
    assert all(s.level == "lane" for s in plan.steps) and masks == [1, 2, 4, 8, 16], (
        plan.describe()
    )
    return masks


@cute.jit
def warp_reduce_rows(plan, vals, combine, peers):
    """Every row's reduction over the warp's 32 keys at once: the plan's lane steps applied level by level to ALL rows (each level's
    shuffles issued together, then the combines -- the rows' chains interleave in the schedule), each row combining
    ``combine(own, partner)`` exactly as ``canon.apply_step`` does for its lane steps. Returns the rows' values (every lane holds them all)."""
    for mask in lane_masks(plan):
        partners = []
        for v in vals:
            partners.append(peers.shuffle_xor((), v, mask))
        nxt = []
        for v, q in zip(vals, partners, strict=True):
            nxt.append(combine(v, q))
        vals = nxt
    return vals


@cute.jit
def row_of_lane(vals, lane: Int32) -> Float32:
    """``vals[lane]`` (``vals[0]`` for lanes past the rows): a static chain of selects -- lane ``n`` publishes row ``n`` with one
    predicated store instead of one divergent branch per row."""
    v = vals[0]
    for n in cutlass.range_constexpr(1, len(vals)):
        if lane == n:
            v = vals[n]
    return v


@cute.jit
def unit_loop(
    st,
    k: Int32,
    qk_thr_mma: cute.TiledMma,
    tStS: cute.Tensor,
    s_cons,
    p_prod,
    stats_prod,
    hm_prod,
    hm_cons,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    s_x: cute.Tensor,
    s_hm: cute.Tensor,
    sP_e: cute.Tensor,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    rows_valid: Int32,
    q_tok0: Int32,
    kv_len: Int32,
    q_len: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
):
    """One unit for softmax warpgroup ``k`` (dynamic: one code copy): the tiles ``n_t = k, k + 2, ...`` of the unit. The running row maximum
    after tile ``n_t - 1`` arrives from the other warpgroup through the hand-off ring (``hm_cons``: the entry it wrote), and this
    warpgroup's ``m'`` goes back the same way when a next tile exists (``hm_prod``). The S / P / stats pipelines advance past the other
    warpgroup's stages (``advance``), and after a unit with an odd tile count warpgroup 1 skips the phantom tile's stages too."""
    T, g, TPC, NP = st.T, st.g, st.tpc, st.n_pad
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % 128  # = TMEM lane = key within the tile
    wq = cute.arch.make_warp_uniform(thread_idx // 32)
    lane = thread_idx % 32
    peers = _LanePeers()
    # TMEM -> registers: this key's n_pad scores (the S^T accumulator's (128, n_pad) C partition, one 32x32b load of n_pad columns)
    cS = cute.make_identity_tensor((T, NP))
    tScS = qk_thr_mma.partition_C(cS)
    ld = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(NP)), ACC_DTYPE
    )
    tiled = tcgen05.make_tmem_copy(ld, tStS)
    thr = tiled.get_slice(thread_idx)
    tFtS = thr.partition_S(tStS)
    tFcS = thr.partition_D(tScS)
    rS = cute.make_rmem_tensor(tFcS.shape, ACC_DTYPE)
    assert cute.size(rS) == NP
    col_of = []  # register i holds row (column) col_of[i]
    for i in cutlass.range_constexpr(NP):
        col_of.append(int(tFcS[i][1]))
    reg_of = {}
    for i in cutlass.range_constexpr(NP):
        reg_of[col_of[i]] = i
    assert len(reg_of) == NP, "every row must be owned exactly once by the thread"
    plan_sum = canon.plan(
        spec.rowsum_reduction(LANES, None),
        canon.Placement.make({"key": [(LANES, "lane")]}),
    )
    # the rows' visible ranges: row n is token n // g of the unit (bottom-right causal), head n % g
    L_row = []
    lo_row = []
    for n in cutlass.range_constexpr(NP):
        pos = kv_len - q_len + q_tok0 + n // g
        L_row.append(pos + 1)
        lo = Int32(0)
        if window_left >= 0:
            lo = pos - window_left
            if lo < 0:
                lo = Int32(0)
        lo_row.append(lo)
    neg_inf = Float32(_NEG_INF)
    N = tile_hi - tile_lo
    for n_t in range(k, N, 2):
        t = tile_lo + n_t
        chunk_first = ((t % TPC) == 0) | (t == tile_lo)
        par = (
            (n_t // 2) % 2
        )  # this warpgroup's exchange-buffer parity (its tiles alternate the two halves)
        sh = s_cons.wait_and_advance()
        s_cons.advance(1)  # the other warpgroup's stage
        cute.copy(
            tiled,
            cute.make_tensor(tFtS.iterator + sh.index * st.tmem_s_stride, tFtS.layout),
            rS,
        )
        cute.arch.fence_view_async_tmem_load()
        sh.release()
        key = t * T + thread_idx
        k0 = t * T
        # mask only the tiles that need it: a boundary of a live row's visible range inside the tile (a warp-uniform decision)
        need = cutlass.Boolean(False)
        for n in cutlass.range_constexpr(NP):
            if n < rows_valid:
                need = need | ((k0 < lo_row[n]) | (k0 + T > L_row[n]))
        if need:
            for n in cutlass.range_constexpr(NP):
                if (key < lo_row[n]) | (key >= L_row[n]):
                    rS[reg_of[n]] = neg_inf
        # ---- the tile's row maximum: the warp's 32 keys (the engine's lane plan, all rows in lockstep), then the four warps in warp
        # order through smem (lane n publishes row n)
        svals = []
        for n in cutlass.range_constexpr(NP):
            svals.append(rS[reg_of[n]])
        ws = [cute.arch.warp_redux_sync(v, "fmax") for v in svals]
        if lane < NP:
            s_x[(par, wq, lane)] = row_of_lane(ws, lane)
        cute.arch.barrier(barrier_id=BAR_SOFTMAX + k, number_of_threads=128)
        m_t = []
        for n in cutlass.range_constexpr(NP):
            m_t.append(
                _pairwise(
                    [
                        s_x[(par, 0, n)],
                        s_x[(par, 1, n)],
                        s_x[(par, 2, n)],
                        s_x[(par, 3, n)],
                    ],
                    CuteF32.fmaxc,
                )
            )
        # ---- the running maximum after the previous tile, from the other warpgroup (none before the unit's first tile): taken only now,
        # after this tile's own maximum, so the warpgroups' serial chain is the hand-off and one fmax per tile (a schedule, not a value)
        m = []
        for _n in cutlass.range_constexpr(NP):
            m.append(Float32(_NEG_INF))
        if n_t > 0:
            hh = hm_cons.wait_and_advance()
            hm_cons.advance(1)
            for n in cutlass.range_constexpr(NP):
                m[n] = s_hm[(1 - k, n)]
            hh.release()
        alpha = []
        nms = []
        for n in cutlass.range_constexpr(NP):
            m_old = m[n]
            if chunk_first:
                m_old = neg_inf
            null = m_t[n] == neg_inf  # no visible key for this row: a null step
            m_new = CuteF32.fmaxc(m_old, m_t[n])
            m_base = m_new
            if null:
                m_base = Float32(0.0)
            a = spec.alpha_term(
                CuteF32, m_old, m_base, c, nc
            )  # EX2(add(mul(m, c), mul(m_new, nc)))
            if null:
                a = Float32(1.0)
            alpha.append(a)
            nms.append(CuteF32.mul(m_base, nc))
            m[n] = m_new
        # ---- m' to the other warpgroup for the next tile (when there is one): lane n publishes row n
        if n_t + 1 < N:
            hh = hm_prod.acquire_and_advance()
            hm_prod.advance(1)
            if lane < NP:
                s_hm[(k, lane)] = row_of_lane(m, lane)
            hh.commit()
        # ---- alpha to the correction warpgroup at once (warp 0's lane n carries row n)
        sth = stats_prod.acquire_and_advance()
        if (wq == 0) & (lane < NP):
            s_alpha[(sth.index, lane)] = row_of_lane(alpha, lane)
        sth.commit()
        # ---- this key's probabilities for every row: p_n = EX2(fma(s_n, c, mul(m_n', nc))), two rows per fma.rn.f32x2 (a packing placement)
        ps = [None] * NP
        for n in cutlass.range_constexpr(0, NP, 2):
            x0, x1 = CuteF32.fma2(
                (rS[reg_of[n]], rS[reg_of[n + 1]]), (c, c), (nms[n], nms[n + 1])
            )
            ps[n] = CuteF32.ex2(x0)
            ps[n + 1] = CuteF32.ex2(x1)
        # ---- the row sums over the warp's keys (the lane plan); the correction adds the four warp partials in warp order
        sth = stats_prod.acquire_and_advance()
        rws = warp_reduce_rows(plan_sum, ps, CuteF32.add, peers)
        if lane < NP:
            s_r[(sth.index, wq, lane)] = row_of_lane(rws, lane)
            if wq == 0:
                s_m[(sth.index, lane)] = row_of_lane(m, lane)
        sth.commit()
        stats_prod.advance(2)  # the other warpgroup's two entries of the next tile
        # ---- P^T to smem (bf16, cvt.rn per element): element (n, key) of the K-major SW128 B operand of this P stage
        ph = p_prod.acquire_and_advance()
        p_prod.advance(1)
        kk = thread_idx  # the key within the tile
        for n in cutlass.range_constexpr(NP):
            # row n: + 64 n, the 16-byte granule of the key XORed with n % 8 (SW128 from the smem address bits)
            off = (
                ph.index * st.p_stage_elems
                + (((kk % 64) >> 3) ^ (n % 8)) * 8
                + (kk % 8)
                + 64 * n
                + 64 * NP * (kk // 64)
            )
            sP_e[off] = ps[n].to(IN_DTYPE)
        cute.arch.fence_view_async_shared()
        ph.commit()
    # an odd unit: warpgroup 1's phantom tile -- every pipeline step of a tile without the work (its S^T stage waited for and released, two
    # empty stats entries and an empty P^T stage committed), so the barriers' phases stay matched with arrivals (a parity wait is mod 2: a
    # skipped phase would let a later wait pass on the previous completion) and the next unit's tile 0 is warpgroup 0's stage 0 again
    if (k == 1) & (N % 2 == 1):
        sh = s_cons.wait_and_advance()
        s_cons.advance(1)
        sh.release()
        sth = stats_prod.acquire_and_advance()
        sth.commit()
        sth = stats_prod.acquire_and_advance()
        sth.commit()
        stats_prod.advance(2)
        ph = p_prod.acquire_and_advance()
        p_prod.advance(1)
        ph.commit()
    return s_cons, p_prod, stats_prod, hm_prod, hm_cons
