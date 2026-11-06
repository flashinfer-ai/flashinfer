# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The tensor-parallel kernels of ``lse.row`` (``README.md``, Sampling: the tensor-parallel protocol): a rank's partials over its shard, the
merge of every rank's partials into the row statistic, and the outputs of a shard from the global statistics.

* :class:`TPPartialsKernel` -- one row group per CTA group over the shard's cover (``shards.TPStructure``): the slot folds
  through ``canon.init_state`` (the engine's fused first step), then the shard's pieces -- the in-chunk nodes
  (``shards.candidate_nodes_chunk``: the adjacent-pairs tree over the thread's aligned chunk), the in-CTA nodes over the
  warps' chunk roots gathered by the engine's warp exchange (``_CutePeers.exchange``), the cluster nodes over the CTAs'
  roots gathered by the cluster exchange -- each stored at its slot when it is a maximal in-shard sub-tree (a runtime
  predicate on the lane's block range, the slot from the layout's table). Round 1 folds the logits with ``fmaxc``, round
  2 the weights ``ex2((x - m) c)`` of the merged global maximum with ``add``. Out-of-shard elements are never loaded (the
  copy predicate), their fragments hold the identity, and no piece contains them.
* :class:`TPMergeKernel` -- one warp per row, lane ``l`` rebuilds lane ``l``'s block root by its lane class's straight-line
  program (``shards.merge_ops``: the ranks' pieces combined in the canonical association, absent children dropped), then
  the engine's lane tree (``canon.run`` over the five butterfly steps) gives the row statistic.
* :class:`TPOutputsKernel` -- the class's chains (``spec.shifted / weight / prob / logp / lse_nat``) on the shard's columns
  from the global ``(m, s)``; the token log-probability for the rows whose id lies in the shard, ``+0`` for the other
  in-vocabulary ids (so the ranks' SUM is exact), NaN for an id outside ``[0, N)``.

No arithmetic is written here: the trees are the spec's digits' combines over the engine's / the layout's node lists, the
chains are the spec's functions on ``CuteF32``.
"""

from __future__ import annotations

import functools
from typing import Sequence, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .. import canon, jit
from .cute_ext import CuteF32
from . import shards
from .kernel import _CutePeers, _select, _tiled_copy
from .schedule import BLOCK, LANES, V
from .shards import ShardLayout
from .spec import (
    inv_sum,
    lg2,
    logp,
    lse_nat,
    max_reduction,
    prob,
    scale,
    shifted,
    sum_reduction,
)
from .tp_schedule import (
    MERGE_LANE_MAPPED_ROWS,
    OUT_ROWS_PER_CTA,
    TPConfig,
    merge_rows_per_cta,
)

NEG_INF = float("-inf")
NAN = float("nan")
KINDS = ("max", "sum")


def _tv_layout(rows: int, vec: int):
    shape = ((LANES, rows), (vec, V // vec, 1))
    stride = ((V * rows, 1), (rows, vec * rows, V * rows))
    return cute.make_layout(shape, stride=stride)


@cute.jit
def _shard_predicate(
    tXcX: cute.Tensor,
    M: Int64,
    col_lo: cutlass.Constexpr[int],
    col_hi: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Per-vector predicate in the cover's coordinates: ``row < M`` and ``col_lo <= col < col_hi`` (the shard's columns; the
    vector width divides the shard width, so a vector is in or out as a whole)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            crd = tXcX[(0, rest_v), 0, rest_k]
            ok = cute.elem_less(crd, (M, col_hi))
            if cutlass.const_expr(col_lo > 0):
                ok = ok & (crd[1] >= col_lo)
            tXpX[rest_v, 0, rest_k] = ok
    return tXpX


@cute.jit
def _store_if(
    pred: cutlass.Boolean, mOut: cute.Tensor, row: Int64, slot: Int32, v: Float32
):
    """A predicated store of one partial (the dynamic region lives in DSL-transformed code)."""
    if pred:
        mOut[row, slot] = v


def _ordered_vals(state, ri):
    """The thread's slot folds in block order (plain Python: the state keys of the fused first step)."""
    keys = sorted(state, key=lambda kk: (kk[ri] if ri is not None else 0))
    return keys, [state[kk] for kk in keys]


def _emit(k, cands, who, t, b_lo, b_hi, mTab, mOut, row):
    """Store every candidate node that is a maximal in-shard sub-tree of lane ``t``'s block tree (plain Python over the
    static candidate list; the predicates are runtime): node ``(L, gj)`` is in shard iff its block range lies in
    ``[b_lo, b_hi)``; it is maximal iff its parent is not (or ``L == kb``); its slot is the layout table's entry at
    ``(lane, L, side)`` with ``side = 1 - (gj & 1)``."""
    kb = k.lay.kb
    tab_stride = 2 * (kb + 1)
    for L, gj, v in cands:
        if kb < L or v is None:
            continue
        gj = Int32(gj) if isinstance(gj, int) else gj
        a0 = gj * (1 << L)
        e0 = a0 + (1 << L)
        pred = (a0 >= b_lo) & (e0 <= b_hi)
        if kb > L:
            pa = (gj // 2) * (1 << (L + 1))
            pe = pa + (1 << (L + 1))
            pred = pred & ((pa < b_lo) | (pe > b_hi))
        side = Int32(1) - (gj - (gj // 2) * 2)
        slot = mTab[t * tab_stride + 2 * L + side]
        _store_if(pred & who, mOut, row, slot, v)


class _ShardLoader:
    """``canon.init_state`` leaf loader over the shard: the fp32 logit of element ``e`` of the thread's block ``kk`` (an exact
    widening), the identity (``-inf``) for the elements the copy predicate excludes."""

    def __init__(self, k, plan, coords, mXv, idX, tiler_mn, rg, M, thr_x, atom_x):
        self.k, self.plan, self.coords = k, plan, coords
        self.mXv, self.idX, self.tiler_mn, self.rg, self.M = mXv, idX, tiler_mn, rg, M
        self.thr_x, self.atom_x = thr_x, atom_x
        self.ri_elem = plan.reg_index("elem")
        self.ri_block = plan.reg_index("block")
        self.frags = {}
        self.nm = self.c = None

    def key_for_block(self, kk):
        key = [0] * self.plan.n_reg
        if self.ri_block is not None:
            key[self.ri_block] = kk
        return tuple(key)

    def k_virtual(self, kk):
        """The block of the cover (relative to ``A_lo``) this thread holds under register value ``kk``: the engine's ownership."""
        return self.plan.digit_value("block", self.coords, self.key_for_block(kk))

    def block(self, kk):
        if kk not in self.frags:
            kv = self.k_virtual(kk)
            cX = cute.local_tile(self.idX, self.tiler_mn, (self.rg, kv))
            pred = _shard_predicate(
                self.thr_x.partition_S(cX), self.M, self.k.col_lo, self.k.col_hi
            )
            gX = self.thr_x.partition_S(
                cute.local_tile(self.mXv, self.tiler_mn, (self.rg, kv))
            )
            fx = cute.make_fragment_like(gX)
            fx.fill(self.mXv.element_type(NEG_INF))
            cute.copy(self.atom_x, gX, fx, pred=pred)
            self.frags[kk] = fx
        return self.frags[kk]

    def x(self, kk, e):
        return self.block(kk)[e].to(Float32)

    def max_leaf(self, key):
        return self.x(
            key[self.ri_block] if self.ri_block is not None else 0, key[self.ri_elem]
        )

    def sum_leaf(self, key):
        fp = CuteF32
        return fp.ex2(shifted(fp, self.max_leaf(key), self.nm, self.c))


class TPPartialsKernel:
    """The partials of one shard layout, one configuration, one round (``kind``) and one temperature form."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        lay: ShardLayout,
        cfg: TPConfig,
        kind: str,
        temp_rows: bool,
    ):
        assert kind in KINDS, kind
        self.dtype, self.lay, self.cfg, self.kind, self.temp_rows = (
            dtype,
            lay,
            cfg,
            kind,
            temp_rows,
        )
        self.st = st = cfg.structure(lay)
        self.W, self.C, self.kpw = st.W, st.C, st.kpw
        self.rows = cfg.rows_per_cta
        self.vec = cfg.vec_elems
        self.num_threads = cfg.num_threads
        # the virtual row: the cover's Kv blocks (the engine's placement of the block digit on reg x warp x cta)
        Nv = st.Kv * BLOCK
        red = (max_reduction if kind == "max" else sum_reduction)(Nv, None)
        self.red = red
        self.digit = red.digit("block")
        pl = canon.Placement.make(
            {
                "elem": [(V, "reg")],
                "block": [(st.kpw, "reg"), (st.W, "warp"), (st.C, "cta")],
                "lane": [(LANES, "lane")],
            },
            rows_per_cta=self.rows,
            vec_elems=self.vec,
        )
        self.plan = canon.plan(red, pl)
        assert self.plan.units["warp"] == st.W and self.plan.units["cta"] == st.C
        self.warp_step = next((s for s in self.plan.steps if s.level == "warp"), None)
        self.cta_step = next((s for s in self.plan.steps if s.level == "cta"), None)
        # the cover's column 0 is the global column A_lo * BLOCK; the shard tensor's column 0 is the global column lo
        self.col_off = (
            st.A_lo * BLOCK - lay.lo
        )  # <= 0: the shard tensor viewed from the cover's origin
        self.col_lo, self.col_hi = lay.lo - st.A_lo * BLOCK, lay.hi - st.A_lo * BLOCK
        self.s_lo, self.s_hi = lay.lo // V, -(-lay.hi // V)
        self.elem_bytes = dtype.width // 8

    def _smem_bytes(self) -> int:
        b = self.rows * self.W * LANES * 4 if self.W > 1 else 0
        if self.C > 1:
            b += self.rows * LANES * 4
        return b

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mM: cute.Tensor,
        mTemp: cute.Tensor,
        mTab: cute.Tensor,
        c_const: Float32,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        k = self.kernel(
            mX,
            mOut,
            mM,
            mTemp,
            mTab,
            c_const,
            M,
            enable_pdl,
            _tv_layout(self.rows, self.vec),
            (self.rows, BLOCK),
        )
        if cutlass.const_expr(self.C > 1):
            k.launch(
                grid=[cute.ceil_div(M, self.rows) * self.C, 1, 1],
                block=[self.num_threads, 1, 1],
                cluster=[self.C, 1, 1],
                smem=self._smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )
        else:
            k.launch(
                grid=[cute.ceil_div(M, self.rows), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mM: cute.Tensor,
        mTemp: cute.Tensor,
        mTab: cute.Tensor,
        c_const: Float32,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        fp = CuteF32
        st = self.st
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        W, rows = self.W, self.rows
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        if cutlass.const_expr(self.C > 1):
            c_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            rg = bidx // self.C
        else:
            c_rank = Int32(0)
            rg = bidx
        row_in_cta = warp_idx // W
        w = warp_idx % W
        t = lane_idx
        vtid = t + LANES * row_in_cta
        coords = {"lane": t, "warp": w, "cta": c_rank, "grid": 0}
        row = rg.to(Int64) * rows + row_in_cta
        row_ok = row < M
        row_c = _select(row_ok, row, M - 1)

        smem = cutlass.utils.SmemAllocator()
        xbuf = cbuf = None
        if cutlass.const_expr(W > 1):
            xbuf = smem.allocate_tensor(
                Float32, cute.make_layout((rows, 1, W, LANES)), byte_alignment=16
            )
        if cutlass.const_expr(self.C > 1):
            cbuf = smem.allocate_tensor(
                Float32, cute.make_layout((rows, 1, LANES)), byte_alignment=16
            )

        # the shard tensor seen from the cover's origin (column 0 = global column A_lo * BLOCK; loads stay inside [lo, hi))
        mXv = (
            cute.domain_offset((0, self.col_off), mX)
            if cutlass.const_expr(self.col_off != 0)
            else mX
        )
        idX = cute.make_identity_tensor((mX.shape[0], st.Kv * BLOCK))
        thr_x, atom_x = _tiled_copy(
            mX, self.vec * mX.element_type.width, tv_layout, tiler_mn, vtid
        )

        if cutlass.const_expr(self.temp_rows):
            c = scale(fp, mTemp[row_c])
        else:
            c = c_const

        loader = _ShardLoader(
            self, self.plan, coords, mXv, idX, tiler_mn, rg, M, thr_x, atom_x
        )
        peers = _CutePeers(xbuf, row_in_cta, w, lane_idx, cbuf, c_rank)
        if cutlass.const_expr(self.kind == "sum"):
            loader.nm, loader.c = fp.neg(mM[row_c]), c
            state, _ = canon.init_state(self.plan, fp, loader.sum_leaf)
        else:
            state, _ = canon.init_state(self.plan, fp, loader.max_leaf)
        keys, vals = _ordered_vals(state, loader.ri_block)

        # the lane's in-shard block range (b_lo, b_hi): slot 32 b + t in [s_lo, s_hi)
        b_lo = (Int32(self.s_lo + LANES - 1) - t) // LANES
        b_hi = (Int32(self.s_hi + LANES - 1) - t) // LANES

        chunk = st.chunk_index(c_rank, w)
        cands, root = shards.candidate_nodes_chunk(st, self.digit, fp, vals, chunk)
        _emit(self, cands, row_ok, t, b_lo, b_hi, mTab, mOut, row)
        if cutlass.const_expr(W > 1):
            sib = peers.exchange(self.warp_step, {keys[0]: root})[keys[0]]
            cands, cta_root = shards.candidate_nodes_cta(
                st, self.digit, fp, sib, c_rank
            )
            _emit(self, cands, row_ok & (w == 0), t, b_lo, b_hi, mTab, mOut, row)
        else:
            cta_root = root
        if cutlass.const_expr(self.C > 1):
            sib = peers.exchange(self.cta_step, {keys[0]: cta_root})[keys[0]]
            cands = shards.candidate_nodes_cluster(st, self.digit, fp, sib)
            _emit(
                self,
                cands,
                row_ok & (w == 0) & (c_rank == 0),
                t,
                b_lo,
                b_hi,
                mTab,
                mOut,
                row,
            )
        peers.finish()
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# ------------------------------------------------------------------------------------------------------------ the merge
def _lane_class(cls_of_lane, t):
    """The lane's class from the baked table (plain Python over 32 selects)."""
    cls = Int32(cls_of_lane[0])
    for l in range(1, LANES):
        if cls_of_lane[l] != cls_of_lane[0]:
            cls = _select(t == l, Int32(cls_of_lane[l]), cls)
    return cls


def _rank_bases(R, P, mBase, t):
    """Column of rank ``r``'s first piece of lane ``t`` in the gathered ``(M, R P)`` tensor."""
    return [Int32(r) * P + mBase[r * LANES + t] for r in range(R)]


def _run_class_program(k, cls_index, sP, t, bases, fp):
    """The lane class's straight-line program on the staged partials of row ``t`` (``sP[t, off_r + base_r + kk]``)."""
    ops, res = k.progs[cls_index]
    return shards.run_merge_ops(
        ops, res, k.digit, fp, lambda r, kk: sP[t, bases[r] + kk]
    )


def _stage_rows(k, mP, sP, bidx, tidx, P, M):
    """Copy the CTA's rows' partials (every rank's ``P_r`` columns) into shared memory as one flat, coalesced sweep: element
    ``e`` of the ``rows x sum_P`` staged block is row ``e // sum_P``, staged column ``f = e % sum_P``, which lies in rank
    ``r``'s segment (``rank_off[r] <= f``) at the gathered column ``r P + (f - rank_off[r])``. Every thread stores every
    iteration (the tail re-stores the last element; rows beyond ``M`` read the clamped last row into their own smem row,
    which no output uses), so there are no predicated regions. The first version strode each (row, rank) segment with
    ``P_r`` of the 1024 threads active and cost ~8 us at 32768 / 8 (2026-11-06)."""
    sum_P = k.sum_P
    total = k.rows * sum_P
    for j in range(-(-total // k.num_threads)):
        e = tidx + j * k.num_threads
        if j == -(-total // k.num_threads) - 1 and total % k.num_threads:
            e = _select(e < total, e, Int32(total - 1))
        row_i = e // sum_P
        f = e - row_i * sum_P
        row = bidx.to(Int64) * k.rows + row_i
        row_c = _select(row < M, row, M - 1)
        col = f + (Int32(0) * P - Int32(k.rank_off[0]))
        for r in range(1, k.R):
            col = _select(
                f >= k.rank_off[r], f + (Int32(r) * P - Int32(k.rank_off[r])), col
            )
        sP[row_i, f] = mP[row_c, col]


class TPMergeKernel:
    """The merge of ``R`` ranks' partials for one tiling of the vocabulary and one round.

    Thread mapping (2026-11-06): a CTA of 32 warps handles ``rows`` rows. Phase 1 stages the rows' partials in shared
    memory (coalesced copies of each rank's ``P_r`` columns). Phase 2: warp ``w`` is tree lane ``w`` and its lanes index
    the rows, so the lane class's program is a warp-UNIFORM branch (the lanes of a row have up to ``R + 1`` classes; with
    lane = tree lane the class branches diverged and the merge cost one program per class: 1.1 us per class per round at
    small M on sm_107). Phase 3: warp ``r`` is row ``r``: the engine's lane butterfly over the 32 staged roots."""

    def __init__(self, bounds: Sequence[Tuple[int, int]], N: int, kind: str):
        assert kind in KINDS, kind
        self.bounds, self.N, self.kind = tuple(bounds), N, kind
        self.layouts, self.cls, self.progs = shards.merge_programs(bounds, N)
        self.R = len(bounds)
        red = (max_reduction if kind == "max" else sum_reduction)(N, None)
        self.digit = red.digit("block")
        self.identity = NEG_INF if kind == "max" else 0.0
        # the lane tree alone: the engine's five butterfly steps over the 32 lane roots
        lane = red.digit("lane")
        self.lane_red = canon.Reduction(
            f"{red.name}.lanes", (lane,), lambda fp, idx: None
        )
        self.lane_plan = canon.plan(
            self.lane_red, canon.Placement.make({"lane": [(LANES, "lane")]})
        )
        # the staged row: rank r's P_r columns at rank_off[r]; one float of padding keeps the row stride odd (bank-conflict free
        # when the lanes of a warp read the same column of different rows)
        offs, acc = [], 0
        for lay in self.layouts:
            offs.append(acc)
            acc += lay.P
        self.rank_off, self.sum_P = tuple(offs), acc
        self.row_stride = acc + 1 if (acc + 1) % 2 else acc + 2
        # the mapping (a placement: both run the same programs and the same lane tree): with one lane class the class branch
        # is uniform in any mapping, so the lane-mapped form (warp = row, no staging) is the cheaper one (1.8 vs 2.6 us at
        # M = 1 on sm_107); with several classes the warp-mapped form avoids the per-class serialisation
        self.warp_mapped = len(self.progs) > 1
        if self.warp_mapped:
            self.rows = merge_rows_per_cta(self.row_stride)
            self.num_threads = LANES * LANES
        else:
            self.rows = MERGE_LANE_MAPPED_ROWS
            self.num_threads = LANES * self.rows

    def _smem_bytes(self) -> int:
        if not self.warp_mapped:
            return 0
        return self.rows * self.row_stride * 4 + self.rows * (LANES + 1) * 4

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mBase: cute.Tensor,
        mOut: cute.Tensor,
        P: Int32,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        if cutlass.const_expr(self.warp_mapped):
            self.kernel(mP, mBase, mOut, P, M, enable_pdl).launch(
                grid=[cute.ceil_div(M, self.rows), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )
        else:
            self.kernel_lanes(mP, mBase, mOut, P, M, enable_pdl).launch(
                grid=[cute.ceil_div(M, self.rows), 1, 1],
                block=[self.num_threads, 1, 1],
                stream=stream,
                use_pdl=enable_pdl,
            )

    @cute.kernel
    def kernel_lanes(
        self,
        mP: cute.Tensor,
        mBase: cute.Tensor,
        mOut: cute.Tensor,
        P: Int32,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        """The lane-mapped form (one lane class): warp = row, lane = tree lane, the program's loads straight from the gathered
        tensor, then the engine's lane tree."""
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        warp_idx = cute.arch.warp_idx()
        t = cute.arch.lane_idx()
        row = bidx.to(Int64) * self.rows + warp_idx
        row_ok = row < M
        row_c = _select(row_ok, row, M - 1)
        bases = _rank_bases(self.R, P, mBase, t)
        root = Float32(self.identity)
        if cutlass.const_expr(self.progs[0][1] >= 0):
            root = shards.run_merge_ops(
                self.progs[0][0],
                self.progs[0][1],
                self.digit,
                fp,
                lambda r, kk: mP[row_c, bases[r] + kk],
            )
        peers = _CutePeers(None, 0, 0, t)
        state, start = canon.init_state(self.lane_plan, fp, lambda key: root)
        stat = canon.run(self.lane_plan, state, start, peers, fp)
        if t == 0:
            if row_ok:
                mOut[row] = stat
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mBase: cute.Tensor,
        mOut: cute.Tensor,
        P: Int32,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        """The warp-mapped form (several lane classes). ``mP`` is ``(M, R * P)``: rank ``r``'s partials of a row at columns
        ``[r P, r P + P_r)``."""
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        warp_idx = cute.arch.warp_idx()
        t = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        sP = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.rows, self.row_stride), stride=(self.row_stride, 1)),
            byte_alignment=16,
        )
        sRoots = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.rows, LANES + 1), stride=(LANES + 1, 1)),
            byte_alignment=16,
        )
        # ---- phase 1: stage the CTA's rows ----------------------------------------------------------------------------
        _stage_rows(self, mP, sP, bidx, tidx, P, M)
        cute.arch.barrier()
        # ---- phase 2: warp w = tree lane w, lane t = row t: the class program (warp-uniform branch) --------------------
        w = cute.arch.make_warp_uniform(warp_idx)
        cls = _lane_class(self.cls, w)
        bases = [Int32(self.rank_off[r]) + mBase[r * LANES + w] for r in range(self.R)]
        root = Float32(self.identity)
        row_t = _select(t < self.rows, t, Int32(0))
        for k in cutlass.range_constexpr(len(self.progs)):
            if cutlass.const_expr(self.progs[k][1] >= 0):
                if cls == k:
                    root = _run_class_program(self, k, sP, row_t, bases, fp)
        if t < self.rows:
            sRoots[t, w] = root
        cute.arch.barrier()
        # ---- phase 3: warp r = row r: the engine's lane tree over the 32 roots ------------------------------------------
        if warp_idx < self.rows:
            row = bidx.to(Int64) * self.rows + warp_idx
            peers = _CutePeers(None, 0, 0, t)
            lane_root = sRoots[warp_idx, t]
            state, start = canon.init_state(self.lane_plan, fp, lambda key: lane_root)
            stat = canon.run(self.lane_plan, state, start, peers, fp)
            if t == 0:
                if row < M:
                    mOut[row] = stat
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------------------------------- the sharded outputs
class TPOutputsKernel:
    """The outputs of a shard from the global statistics: ``mode`` ``"logp"`` (the token log-probability and the ``lse`` per
    row) or a row output over the shard's columns (``"probs"`` in the input dtype, ``"logprobs"`` in fp32)."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        lay: ShardLayout,
        mode: str,
        temp_rows: bool,
        vec: int,
    ):
        assert mode in ("logp", "probs", "logprobs"), mode
        self.dtype, self.lay, self.mode, self.temp_rows, self.vec = (
            dtype,
            lay,
            mode,
            temp_rows,
            vec,
        )
        self.cols = lay.hi - lay.lo
        self.out_dtype = dtype if mode == "probs" else Float32
        self.rows = OUT_ROWS_PER_CTA
        self.vec_out = min(vec, max(1, 128 // self.out_dtype.width))
        self.num_threads = 128 if mode == "logp" else LANES * self.rows

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mM: cute.Tensor,
        mS: cute.Tensor,
        mTok: cute.Tensor,
        mLogp: cute.Tensor,
        mLse: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        M: Int64,
        N: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        if cutlass.const_expr(self.mode == "logp"):
            self.kernel_logp(
                mX, mM, mS, mTok, mLogp, mLse, mTemp, c_const, M, N, enable_pdl
            ).launch(
                grid=[cute.ceil_div(M, self.num_threads), 1, 1],
                block=[self.num_threads, 1, 1],
                stream=stream,
                use_pdl=enable_pdl,
            )
        else:
            self.kernel_rows(
                mX,
                mY,
                mM,
                mS,
                mTemp,
                c_const,
                M,
                enable_pdl,
                _tv_layout(self.rows, self.vec),
                _tv_layout(self.rows, self.vec_out),
                (self.rows, BLOCK),
            ).launch(
                grid=[cute.ceil_div(M, self.rows), -(-self.cols // BLOCK), 1],
                block=[self.num_threads, 1, 1],
                stream=stream,
                use_pdl=enable_pdl,
            )

    @cute.kernel
    def kernel_logp(
        self,
        mX: cute.Tensor,
        mM: cute.Tensor,
        mS: cute.Tensor,
        mTok: cute.Tensor,
        mLogp: cute.Tensor,
        mLse: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        M: Int64,
        N: Int64,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        row = bidx.to(Int64) * self.num_threads + tidx
        row_ok = row < M
        row_c = _select(row_ok, row, M - 1)
        if cutlass.const_expr(self.temp_rows):
            c = scale(fp, mTemp[row_c])
        else:
            c = c_const
        m, s = mM[row_c], mS[row_c]
        lg2s = lg2(fp, s)
        lse_v = lse_nat(fp, m, c, lg2s)
        tok = mTok[row_c]
        in_shard = (tok >= Int64(self.lay.lo)) & (tok < Int64(self.lay.hi))
        tok_ok = (tok >= 0) & (tok < N)
        tok_c = _select(in_shard, tok - Int64(self.lay.lo), Int64(0))
        xt = mX[row_c, tok_c].to(Float32)
        lp = logp(fp, shifted(fp, xt, fp.neg(m), c), fp.neg(lg2s))
        lp = _select(in_shard, lp, _select(tok_ok, Float32(0.0), Float32(NAN)))
        if row_ok:
            mLogp[row] = lp
            mLse[row] = lse_v
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel_rows(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mM: cute.Tensor,
        mS: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tv_layout_out: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        fp = CuteF32
        bidx, kg, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        row_in_cta = tidx // LANES
        row = bidx.to(Int64) * self.rows + row_in_cta
        row_ok = row < M
        row_c = _select(row_ok, row, M - 1)
        if cutlass.const_expr(self.temp_rows):
            c = scale(fp, mTemp[row_c])
        else:
            c = c_const
        m, s = mM[row_c], mS[row_c]
        nm = fp.neg(m)
        inv_s = inv_sum(fp, s) if cutlass.const_expr(self.mode == "probs") else None
        nlg = (
            fp.neg(lg2(fp, s)) if cutlass.const_expr(self.mode == "logprobs") else None
        )
        thr_x, atom_x = _tiled_copy(
            mX, self.vec * mX.element_type.width, tv_layout, tiler_mn, tidx
        )
        thr_y, atom_y = _tiled_copy(
            mY, self.vec_out * mY.element_type.width, tv_layout_out, tiler_mn, tidx
        )
        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, tiler_mn, (bidx, kg))
        pred = _shard_predicate(thr_x.partition_S(cX), M, 0, self.cols)
        gX = thr_x.partition_S(cute.local_tile(mX, tiler_mn, (bidx, kg)))
        fx = cute.make_fragment_like(gX)
        fx.fill(mX.element_type(NEG_INF))
        cute.copy(atom_x, gX, fx, pred=pred)
        pred_y = _shard_predicate(thr_y.partition_S(cX), M, 0, self.cols)
        gY = thr_y.partition_S(cute.local_tile(mY, tiler_mn, (bidx, kg)))
        fy = cute.make_fragment_like(gY)
        _shard_row_outputs(self, fx, fy, nm, c, inv_s, nlg, mY.element_type, fp)
        cute.copy(atom_y, fy, gY, pred=pred_y)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _shard_row_outputs(k, fx, fy, nm, c, inv_s, nlg, out_dtype, fp):
    """The output elements of one block tile (plain Python at trace time): the class's chains on the global statistics."""
    for e in range(V):
        z = shifted(fp, fx[e].to(Float32), nm, c)
        if k.mode == "probs":
            fy[e] = prob(fp, fp.ex2(z), inv_s).to(out_dtype)
        else:
            fy[e] = logp(fp, z, nlg)


# ------------------------------------------------------------------------------------------------- names, compile, cache
MODULE = "batch_invariant_sampling_tp"  # the on-disk cache module of the TP kernels (registered in sampling/__init__.py)


def partials_kernel_name(
    kind: str,
    dtype_str: str,
    lay: ShardLayout,
    cfg: TPConfig,
    temp_rows: bool,
    enable_pdl: bool,
) -> str:
    return jit.sanitize_symbol_name(
        f"tp_{kind}_{dtype_str}_{lay.describe()}_{cfg.describe()}_{'trow' if temp_rows else 'tconst'}_pdl{int(enable_pdl)}"
    )


def merge_kernel_name(
    kind: str, bounds: Sequence[Tuple[int, int]], N: int, enable_pdl: bool
) -> str:
    b = "_".join(f"{lo}-{hi}" for lo, hi in bounds)
    return jit.sanitize_symbol_name(
        f"tpmerge_{kind}_N{N}_R{len(bounds)}_{b}_pdl{int(enable_pdl)}"
    )


def outputs_kernel_name(
    mode: str,
    dtype_str: str,
    lay: ShardLayout,
    temp_rows: bool,
    vec: int,
    enable_pdl: bool,
) -> str:
    return jit.sanitize_symbol_name(
        f"tpout_{mode}_{dtype_str}_{lay.describe()}_V{vec}_{'trow' if temp_rows else 'tconst'}_pdl{int(enable_pdl)}"
    )


def _rows_fake(dt, vec, shape):
    return cute.runtime.make_fake_tensor(
        dt,
        shape,
        (cute.sym_int64(divisibility=vec), 1),
        assumed_align=max(vec * (dt.width // 8), 1),
    )


def _vector_fake(dt, align):
    return cute.runtime.make_fake_tensor(
        dt, (cute.sym_int(32),), (1,), assumed_align=align
    )


@functools.cache
def get_partials_kernel(
    dtype_str: str,
    lay: ShardLayout,
    cfg: TPConfig,
    kind: str,
    temp_rows: bool,
    enable_pdl: bool,
):
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = partials_kernel_name(kind, dtype_str, lay, cfg, temp_rows, enable_pdl)

    def compile_fn():
        obj = TPPartialsKernel(dtype, lay, cfg, kind, temp_rows)
        sym_m = cute.sym_int(64)
        args = (
            _rows_fake(dtype, cfg.vec_elems, (sym_m, cute.sym_int(32))),
            _rows_fake(Float32, 1, (sym_m, cute.sym_int(32))),
            _vector_fake(Float32, 4),
            _vector_fake(Float32, 4),
            _vector_fake(Int32, 4),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            obj,
            *args,
            Float32(1.0),
            Int64(1),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)


@functools.cache
def get_merge_kernel(
    bounds: Tuple[Tuple[int, int], ...], N: int, kind: str, enable_pdl: bool
):
    name = merge_kernel_name(kind, bounds, N, enable_pdl)

    def compile_fn():
        obj = TPMergeKernel(bounds, N, kind)
        sym_m = cute.sym_int(64)
        args = (
            _rows_fake(Float32, 1, (sym_m, cute.sym_int(32))),
            _vector_fake(Int32, 4),
            _vector_fake(Float32, 4),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            obj,
            *args,
            Int32(1),
            Int64(1),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)


@functools.cache
def get_outputs_kernel(
    dtype_str: str,
    lay: ShardLayout,
    mode: str,
    temp_rows: bool,
    vec: int,
    enable_pdl: bool,
):
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = outputs_kernel_name(mode, dtype_str, lay, temp_rows, vec, enable_pdl)

    def compile_fn():
        obj = TPOutputsKernel(dtype, lay, mode, temp_rows, vec)
        sym_m, sym_n = cute.sym_int(64), cute.sym_int(32)
        y_shape = (
            (sym_m, sym_n) if mode != "logp" else (cute.sym_int(64), cute.sym_int(32))
        )
        args = (
            _rows_fake(dtype, vec, (sym_m, sym_n)),
            _rows_fake(obj.out_dtype, obj.vec_out, y_shape),
            _vector_fake(Float32, 4),
            _vector_fake(Float32, 4),
            _vector_fake(Int64, 8),
            _vector_fake(Float32, 4),
            _vector_fake(Float32, 4),
            _vector_fake(Float32, 4),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            obj,
            *args,
            Float32(1.0),
            Int64(1),
            Int64(1),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
