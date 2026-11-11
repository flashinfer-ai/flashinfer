# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The ``sample.icdf.v1`` kernel (``draw_spec.py``): the row class's passes, then the draw as a reading of the canonical
sum tree.

For logits: the max pass and the sum pass of ``lse.row`` (the same engine plans, the same chains); for probabilities: the
sum pass over the given weights (absent elements read ``+0``). During the sum pass the peers *record* what the engine
already moves -- the butterfly partner at every lane step, the exchanged warp and CTA siblings -- and the loader's block
totals are kept, so every left-sibling node of the tree is at hand. The crossing pass then forms, per slot, ``outer`` (the
add chain over the set bits of the slot's tree address of those nodes, ascending, from ``+0``), walks the slot's fold and
finds the first present element with a positive weight and ``add(outer, fold_e) >= t`` (the positivity is part of the
rule: a zero-weight slot head's prefix is another add chain than the previous slot's last, so it could otherwise be the
first crossing); the row's token is the minimum tree address over threads (an exact integer reduction), or the last
present element with a positive weight when none crosses, or ``-1`` when the row's sum is not a positive finite number
or its gather index lies outside the source rows. The uniform is the fp32 lane of the request's stream, formed in the
kernel from ``(seed, offset)`` of the row (``stream_kernel.py``'s Philox helpers).

The kernel is built on the row kernel's plumbing (``kernel.py``: the loader, the peers, the tiled copies, the
predicates); its own cache module (``stream_kernel.DRAW_MODULE``), so an edit here never recompiles the row kernels.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import canon, jit
from .cute_ext import CuteF32
from .kernel import (
    NAN,
    NEG_INF,
    _CutePeers,
    _map_shared_rank,
    _predicate,
    _RowLoader,
    _select,
    _tiled_copy,
)
from .schedule import BLOCK, LANES, SMEM_BUDGET, V, LSEConfig
from .spec import lg2, logp, max_reduction, scale, shifted, sum_reduction
from .stream_kernel import (
    DRAW_MODULE,
    philox_block,
    request_key,
    u64_lo32,
    u64_shr,
    uniform32,
)

#: A register-resident share longer than this is staged in shared memory instead (the crossing pass unrolls per element
#: and spilled at 256 elements per thread; the dispatcher never picks such a share, user configs may).
STAGE_ABOVE_ELEMS = 128
INT_MAX = 0x7FFFFFFF


@dsl_user_op
def _ld_shared_cluster_i32(addr: Int32, *, loc=None, ip=None) -> Int32:
    """``ld.shared::cluster.b32``: one 32-bit word from a peer CTA's shared memory (bits move; no arithmetic)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _imin(a, b):
    """The integer minimum as a select (exact)."""
    return _select(a < b, a, b)


def _imax(a, b):
    """The integer maximum as a select (exact)."""
    return _select(a > b, a, b)


def _pick(values, idx):
    """``values[idx]`` for a runtime ``idx`` over a static list (a chain of selects; the values are tree nodes)."""
    acc = values[0]
    for i in range(1, len(values)):
        acc = _select(idx == i, values[i], acc)
    return acc


def _pairwise_levels(fp, vals):
    """All levels of the pairwise tree over ``vals`` (low index first, as ``canon._pairwise``): ``levels[0] = vals``."""
    levels = [list(vals)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([fp.add(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)])
    return levels


class _DrawLoader(_RowLoader):
    """The row kernel's loader with the draw's two placements of its reads: the pad an absent element reads (``-inf``
    for logits: the max identity, whose weight is ``+0``; ``+0`` for a weight row: the sum identity), and the staging
    of a long register-resident share into the CTA's smem tile through the predicated register loads
    (:meth:`stage_from_regs`), after which every block is served from the tile."""

    def __init__(self, *args, pad: float = NEG_INF, **kw):
        """The row loader with the draw's pad value (``-inf`` for logits, ``+0`` for weights)."""
        super().__init__(*args, **kw)
        self.pad = pad
        self.from_smem = False

    def stage_from_regs(self, smem_x, cta_base):
        """Stage this warp's blocks into the CTA's smem tile through the predicated register loads (any vector width,
        any alignment; absent elements carry the pad value), then serve every later read from the tile. The caller
        places a CTA barrier after it."""
        self.smem_x, self.cta_base = smem_x, cta_base
        self.thr_async = self.thr_x
        for kk in range(self.k.kpw):
            pred, fx = self.block(kk)
            kg = self.k_global(kk)
            dT = self.thr_x.partition_D(
                cute.local_tile(smem_x, self.tiler_mn, (0, kg - cta_base))
            )
            cute.autovec_copy(fx, dT)
        self.frags.clear()
        self._staged = True
        self.from_smem = True

    def block(self, kk):
        """The predicate and fragment of block ``kk``: from the smem tile when staged, else the predicated
        register load over the pad fill."""
        if kk not in self.frags:
            if not self.k.cfg.keep_regs or self.from_smem:
                self.frags.clear()
            kg = self.k_global(kk)
            cX = cute.local_tile(self.idX, self.tiler_mn, (self.rg, kg))
            if self.k.cfg.stage == "smem" or self.from_smem:
                if not self._staged:
                    self._stage_all()
                pred = _predicate(self.thr_async.partition_S(cX), self.M, self.N)
                gT = self.thr_async.partition_S(
                    cute.local_tile(self.mX, self.tiler_mn, (self.rg, kg))
                )
                dT = self.thr_async.partition_D(
                    cute.local_tile(self.smem_x, self.tiler_mn, (0, kg - self.cta_base))
                )
                fx = cute.make_fragment_like(gT)
                cute.autovec_copy(dT, fx)
            else:
                pred = _predicate(self.thr_x.partition_S(cX), self.M, self.N)
                gX = self.thr_x.partition_S(
                    cute.local_tile(self.mX, self.tiler_mn, (self.rg, kg))
                )
                fx = cute.make_fragment_like(gX)
                fx.fill(self.mX.element_type(self.pad))
                cute.copy(self.atom_x, gX, fx, pred=pred)
            self.frags[kk] = (pred, fx)
        return self.frags[kk]

    def x(self, kk, e):
        """The fp32 value of element ``e`` of block ``kk``; an absent element is the pad (the register path fills its
        fragments before the predicated load; the cp.async path's zero fill is replaced by a select on the predicate)."""
        pred, fx = self.block(kk)
        v = fx[e].to(Float32)
        if self.k.cfg.stage == "smem" and not self.from_smem:
            return _select(pred[e // self.k.vec, 0, 0], v, Float32(self.pad))
        return v


class _DrawPeers(_CutePeers):
    """The hardware peers of ``lse.row`` plus a recorder: during the sum pass it keeps the butterfly partner of every
    lane step and the sibling lists of the warp and CTA exchanges (the tree's nodes the draw's prefixes are made of), and
    it reduces the draw's two integers (first crossing: min; last positive: max) over the same levels."""

    def __init__(
        self,
        xbuf,
        group,
        warp_id,
        lane_id,
        cbuf=None,
        cta_rank=None,
        ibuf=None,
        icbuf=None,
        W=1,
        C=1,
    ):
        """The row kernel's peers with the integer-reduce buffers and the recording state."""
        super().__init__(xbuf, group, warp_id, lane_id, cbuf, cta_rank)
        self.ibuf, self.icbuf, self.W, self.C = ibuf, icbuf, W, C
        self.recording = False
        self.lane_partners: dict = {}
        self.warp_sibs = None
        self.cta_sibs = None

    def shuffle_xor(self, key, value, mask):
        """The engine's butterfly step; during the sum pass the partner's value is recorded per mask."""
        p = super().shuffle_xor(key, value, mask)
        if self.recording and value is not None:
            self.lane_partners[mask] = p
        return p

    def exchange(self, step, state):
        """The engine's warp / CTA exchange; during the sum pass the sibling values are recorded."""
        out = super().exchange(step, state)
        if self.recording:
            keys = [k for k, v in state.items() if v is not None]
            assert len(keys) == 1, keys
            if step.level == "cta":
                self.cta_sibs = out[keys[0]]
            else:
                self.warp_sibs = out[keys[0]]
        return out

    def int_reduce(self, first, last):
        """The exact integer reduction of the two addresses over lanes, warps and CTAs: ``first`` by
        minimum, ``last`` by maximum."""
        for mask in (1, 2, 4, 8, 16):
            first = _imin(first, cute.arch.shuffle_sync_bfly(first, offset=mask))
            last = _imax(last, cute.arch.shuffle_sync_bfly(last, offset=mask))
        if self.W > 1:
            cute.arch.barrier()
            self.ibuf[self.group, 0, self.warp_id, self.lane_id] = first
            self.ibuf[self.group, 1, self.warp_id, self.lane_id] = last
            cute.arch.barrier()
            for sid in range(self.W):
                first = _imin(first, self.ibuf[self.group, 0, sid, self.lane_id])
                last = _imax(last, self.ibuf[self.group, 1, sid, self.lane_id])
        if self.C > 1:
            self.icbuf[self.group, 0, self.lane_id] = first
            self.icbuf[self.group, 1, self.lane_id] = last
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            self.n_cluster_exchanges += 1
            bases = [
                _map_shared_rank(self.icbuf.iterator, Int32(sid))
                for sid in range(self.C)
            ]
            off0 = cute.crd2idx((self.group, 0, self.lane_id), self.icbuf.layout) * 4
            off1 = cute.crd2idx((self.group, 1, self.lane_id), self.icbuf.layout) * 4
            for b in bases:
                first = _imin(first, _ld_shared_cluster_i32(b + off0))
                last = _imax(last, _ld_shared_cluster_i32(b + off1))
        return first, last


class DrawRowKernel:
    """``sample.icdf.v1`` for one ``Kp`` bucket (``N`` runtime), one input kind (logits / probabilities), one
    temperature form, with or without an index gather and the drawn token's log-probability."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        Kp: int,
        cfg: LSEConfig,
        from_logits: bool,
        temp_rows: bool,
        has_indices: bool,
        return_logprob: bool,
    ):
        """The kernel object of one Kp bucket and specialisation: the plans, the units, the staging
        decision and the smem budget."""
        N = Kp * BLOCK
        cfg.validate(N)
        assert not has_indices or cfg.rows_per_cta == 1, (
            "an index gather needs one row per CTA"
        )
        assert not return_logprob or from_logits
        self.dtype, self.N, self.Kp, self.cfg = dtype, N, Kp, cfg
        self.from_logits, self.temp_rows = from_logits, temp_rows
        self.has_indices, self.return_logprob = has_indices, return_logprob
        pl = cfg.placement(N)
        self.plan_max = canon.plan(max_reduction(N, None), pl)
        self.plan_sum = canon.plan(sum_reduction(N, None), pl)
        assert self.plan_sum.units["grid"] == 1
        self.W, self.C = self.plan_sum.units["warp"], self.plan_sum.units["cta"]
        assert cfg.cluster == self.C and self.plan_sum.units["lane"] == LANES
        self.rows = cfg.rows_per_cta
        self.kpw = cfg.blocks_per_warp(N)
        self.vec = cfg.vec_elems
        self.num_threads = cfg.num_threads
        self.n_entries = max(
            (
                len(self.plan_sum.keys_at(i))
                for i, s in enumerate(self.plan_sum.steps)
                if s.level == "warp"
            ),
            default=0,
        )
        self.n_entries_cta = max(
            (
                len(self.plan_sum.keys_at(i))
                for i, s in enumerate(self.plan_sum.steps)
                if s.level == "cta"
            ),
            default=0,
        )
        self.elem_bytes = dtype.width // 8
        if cfg.stage == "smem":
            assert cfg.smem_stage_ok(N, self.elem_bytes)
        tile = self.rows * cfg.blocks_per_cta(N) * BLOCK * self.elem_bytes
        # the share lives in shared memory: cp.async for the staged config, the register path for a long
        # register-resident share
        self.stage_tile = cfg.stage == "smem" or (
            cfg.stage == "regs"
            and cfg.elems_per_thread(N) > STAGE_ABOVE_ELEMS
            and tile <= SMEM_BUDGET
        )
        self.tile_bytes = tile if self.stage_tile else 0

    def _tv_layout(self, vec):
        """The thread-value layout of the tiled copy for ``vec`` elements per thread."""
        rows = self.rows
        return cute.make_layout(
            ((LANES, rows), (vec, V // vec, 1)),
            stride=((V * rows, 1), (rows, vec * rows, V * rows)),
        )

    def _smem_bytes(self) -> int:
        """The dynamic shared memory: the exchange and integer buffers, and the staged tile when used."""
        b = self.rows * self.n_entries * self.W * LANES * 4 if self.W > 1 else 0
        b += (
            self.rows * 2 * self.W * LANES * 4 if self.W > 1 else 0
        )  # the integer reduce
        if self.C > 1:
            b += self.rows * self.n_entries_cta * LANES * 4 + self.rows * 2 * LANES * 4
        if self.stage_tile:
            b += self.tile_bytes + 128
        return b + 128

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        mIdx: cute.Tensor,
        mTok: cute.Tensor,
        mValid: cute.Tensor,
        mLogp: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        draw: Int32,
        M: Int64,
        Msrc: Int64,
        N: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        k = self.kernel(
            mX,
            mSeed,
            mOff,
            mIdx,
            mTok,
            mValid,
            mLogp,
            mTemp,
            c_const,
            draw,
            M,
            Msrc,
            N,
            enable_pdl,
            self._tv_layout(self.vec),
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
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        mIdx: cute.Tensor,
        mTok: cute.Tensor,
        mValid: cute.Tensor,
        mLogp: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        draw: Int32,
        M: Int64,
        Msrc: Int64,
        N: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        W, C, rows = self.W, self.C, self.rows
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        if cutlass.const_expr(C > 1):
            c_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            rg = bidx // C
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
        # the source row (an index gather reads row indices[row]; one row per CTA then)
        if cutlass.const_expr(self.has_indices):
            src = mIdx[row_c]
            src_ok = (src >= 0) & (src < Msrc)
            src_c = _select(src_ok, src, Int64(0))
            tile_row = src_c.to(Int32)
        else:
            src_c = row_c
            tile_row = rg

        # ---- shared memory ------------------------------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        xbuf = cbuf = smem_x = ibuf = icbuf = None
        if cutlass.const_expr(self.stage_tile):
            smem_x = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(
                    (rows, self.cfg.blocks_per_cta(self.N) * BLOCK), order=(1, 0)
                ),
                byte_alignment=16,
            )
        if cutlass.const_expr(W > 1):
            xbuf = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows, self.n_entries, W, LANES)),
                byte_alignment=16,
            )
            ibuf = smem.allocate_tensor(
                Int32, cute.make_layout((rows, 2, W, LANES)), byte_alignment=16
            )
        if cutlass.const_expr(C > 1):
            cbuf = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows, self.n_entries_cta, LANES)),
                byte_alignment=16,
            )
            icbuf = smem.allocate_tensor(
                Int32, cute.make_layout((rows, 2, LANES)), byte_alignment=16
            )

        in_bits = self.vec * mX.element_type.width
        thr_x, atom_x = _tiled_copy(mX, in_bits, tv_layout, tiler_mn, vtid)
        thr_async = atom_async = None
        if cutlass.const_expr(self.cfg.stage == "smem"):
            atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=in_bits,
            )
            thr_async = cute.make_tiled_copy(atom_async, tv_layout, tiler_mn).get_slice(
                vtid
            )
        idX = cute.make_identity_tensor(mX.shape)

        if cutlass.const_expr(self.temp_rows):
            c = scale(fp, mTemp[row_c])
        else:
            c = c_const

        pad = NEG_INF if self.from_logits else 0.0
        cta_base = c_rank * self.cfg.blocks_per_cta(self.N)
        loader = _DrawLoader(
            self,
            self.plan_sum,
            coords,
            mX,
            idX,
            tiler_mn,
            tile_row,
            Msrc,
            N,
            thr_x,
            atom_x,
            smem_x if self.cfg.stage == "smem" else None,
            thr_async,
            atom_async,
            cta_base,
            pad=pad,
        )
        if cutlass.const_expr(self.stage_tile and self.cfg.stage != "smem"):
            loader.stage_from_regs(smem_x, cta_base)
            cute.arch.barrier()
        peers = _DrawPeers(
            xbuf, row_in_cta, w, lane_idx, cbuf, c_rank, ibuf, icbuf, W, C
        )

        # ===== the weights' tree: (max pass,) sum pass with the peers recording ======================================
        nm = Float32(0.0)
        m = Float32(0.0)
        if cutlass.const_expr(self.from_logits):
            state, start = canon.init_state(self.plan_max, fp, loader.max_leaf)
            m = canon.run(self.plan_max, state, start, peers, fp)
            nm = fp.neg(m)
            loader.nm, loader.c = nm, c
            leaf = loader.sum_leaf
        else:
            leaf = loader.max_leaf  # the given weights themselves
        peers.recording = True
        state, start = canon.init_state(self.plan_sum, fp, leaf)
        totals = _block_totals(self.plan_sum, state, self.kpw)
        S = canon.run(self.plan_sum, state, start, peers, fp)
        peers.recording = False

        # ===== the uniform and the threshold ============================================================================
        key = request_key(mSeed[row_c], mOff[row_c])
        k0 = u64_lo32(key)
        k1 = u64_lo32(u64_shr(key, Int32(32)))
        w0, _, _, _ = philox_block(draw, Int32(0), k0, k1)
        u = uniform32(w0)
        thr = fp.mul(u, S)

        # ===== the prefix nodes and the crossing pass (plain Python at trace time: static loops) ========================
        first, last = _crossing_pass(
            self, loader, leaf, peers, totals, thr, t, w, c_rank
        )
        first, last = peers.int_reduce(first, last)

        # ===== the row's answer ==========================================================================================
        a = _select(first == Int32(INT_MAX), last, first)
        s_ok = (Float32(0.0) < S) & (Float32(float("inf")) > S)
        valid = s_ok & (a >= Int32(0))
        if cutlass.const_expr(self.has_indices):
            valid = (
                valid & src_ok
            )  # an index outside the source rows: an invalid output row, not row 0's draw
        slot = a >> 3
        lane_a = slot >> (self.Kp.bit_length() - 1)
        block_a = slot & Int32(self.Kp - 1)
        token = ((block_a << 5) + lane_a) * 8 + (a & 7)
        token = _select(valid, token, Int32(-1))
        if t == 0:
            if w == 0:
                if c_rank == 0:
                    if row_ok:
                        mTok[row] = token
                        mValid[row] = _select(valid, Int32(1), Int32(0))
        if cutlass.const_expr(self.return_logprob):
            tok_c = _select(valid, token, Int32(0))
            xt = mX[src_c, tok_c].to(Float32)
            lp = logp(fp, shifted(fp, xt, nm, c), fp.neg(lg2(fp, S)))
            lp = _select(valid, lp, Float32(NAN))
            if t == 0:
                if w == 0:
                    if c_rank == 0:
                        if row_ok:
                            mLogp[row] = lp
        peers.finish()
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _block_totals(plan, state, kpw):
    """The slot-fold totals of this thread's blocks, in block order, from the state ``canon.init_state`` returns."""
    ri = plan.reg_index("block")
    if ri is None:
        return [state[tuple(None for _ in range(plan.n_reg))]]
    return [
        state[tuple(kk if i == ri else None for i in range(plan.n_reg))]
        for kk in range(kpw)
    ]


def _crossing_pass(k, loader, leaf, peers, totals, thr, lane, w, c_rank):
    """Per block of this thread: ``outer`` (the add chain, ascending bit order from ``+0``, of the left-sibling nodes of
    the slot's tree address: the thread's block tree for the register bits, the pairwise nodes over the exchanged warp /
    CTA siblings for those bits, the butterfly partners for the lane bits), then the slot fold; the first present element
    with a positive weight and ``add(outer, fold) >= thr`` (min tree address) and the last present element with a positive weight (max)."""
    fp = CuteF32
    W, C, kpw, Kp = k.W, k.C, k.kpw, k.Kp
    plan = k.plan_sum
    ri = plan.reg_index("block")
    lb = kpw.bit_length() - 1
    lnodes = _pairwise_levels(fp, totals)
    wnodes = _pairwise_levels(fp, peers.warp_sibs) if W > 1 else None
    cnodes = _pairwise_levels(fp, peers.cta_sibs) if C > 1 else None
    zero = Float32(0.0)
    kp_log = Kp.bit_length() - 1
    first = Int32(INT_MAX)
    last = Int32(-1)
    for kk in range(kpw):
        acc = zero
        for b in range(lb):
            if (kk >> b) & 1:
                acc = fp.add(acc, lnodes[b][(kk >> b) - 1])
        if W > 1:
            for kbit in range(W.bit_length() - 1):
                g = (w >> kbit) - 1
                node = (
                    _pick(wnodes[kbit], g) if len(wnodes[kbit]) > 1 else wnodes[kbit][0]
                )
                acc = fp.add(acc, _select(((w >> kbit) & 1) == 1, node, zero))
        if C > 1:
            for kbit in range(C.bit_length() - 1):
                g = (c_rank >> kbit) - 1
                node = (
                    _pick(cnodes[kbit], g) if len(cnodes[kbit]) > 1 else cnodes[kbit][0]
                )
                acc = fp.add(acc, _select(((c_rank >> kbit) & 1) == 1, node, zero))
        for kbit in range(5):
            acc = fp.add(
                acc,
                _select(
                    ((lane >> kbit) & 1) == 1, peers.lane_partners[1 << kbit], zero
                ),
            )
        outer = acc
        kg = loader.k_global(kk)
        pred = loader.block(kk)[0]
        base = ((lane << kp_log) + kg) << 3
        running = zero
        for e in range(V):
            key_e = (
                tuple(kk if i == ri else e for i in range(plan.n_reg))
                if ri is not None
                else (e,)
            )
            we = leaf(key_e)
            running = fp.add(running, we)
            f = fp.add(outer, running)
            present = pred[e // k.vec, 0, 0]
            a = base + e
            pos = present & (we > zero)
            first = _imin(first, _select(pos & (f >= thr), a, Int32(INT_MAX)))
            last = _imax(last, _select(pos, a, Int32(-1)))
    return first, last


def kernel_name(
    dtype_str: str,
    Kp: int,
    cfg: LSEConfig,
    from_logits: bool,
    temp_rows: bool,
    has_indices: bool,
    return_logprob: bool,
    enable_pdl: bool,
) -> str:
    """The cache object's name: one per (dtype, Kp bucket, config, input kind, temperature form, gather,
    log-probability output, pdl)."""
    return jit.sanitize_symbol_name(
        f"draw_{'logits' if from_logits else 'probs'}_{dtype_str}_Kp{Kp}_{cfg.describe()}_{'trow' if temp_rows else 'tconst'}"
        f"{'_idx' if has_indices else ''}{'_lp' if return_logprob else ''}_pdl{int(enable_pdl)}"
    )


def get_compiled_kernel(
    dtype_str: str,
    N: int,
    cfg: LSEConfig,
    from_logits: bool,
    temp_rows: bool,
    has_indices: bool,
    return_logprob: bool,
    enable_pdl: bool,
):
    """The compiled kernel serving vocabulary ``N``: one per (dtype, Kp bucket, config, input kind, temperature form,
    index gather, log-probability output, pdl); ``N``, ``M`` and the row stride are runtime arguments."""
    from .schedule import padded_blocks

    return _compile_bucket(
        dtype_str,
        padded_blocks(N),
        cfg,
        from_logits,
        temp_rows,
        has_indices,
        return_logprob,
        enable_pdl,
    )


@functools.cache
def _compile_bucket(
    dtype_str, Kp, cfg, from_logits, temp_rows, has_indices, return_logprob, enable_pdl
):
    """Compile, or load from the cache, the kernel of one Kp bucket and specialisation."""
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = kernel_name(
        dtype_str,
        Kp,
        cfg,
        from_logits,
        temp_rows,
        has_indices,
        return_logprob,
        enable_pdl,
    )

    def compile_fn():
        obj = DrawRowKernel(
            dtype, Kp, cfg, from_logits, temp_rows, has_indices, return_logprob
        )
        sym_m, sym_n = cute.sym_int(64), cute.sym_int(32)
        x = cute.runtime.make_fake_tensor(
            dtype,
            (sym_m, sym_n),
            (cute.sym_int64(divisibility=cfg.vec_elems), 1),
            assumed_align=max(cfg.vec_elems * (dtype.width // 8), 1),
        )

        def vector(dt, align):
            # every per-row vector has its own size symbol: the FFI entry then checks ndim / dtype only, so the unused
            # ones may be small dummies
            return cute.runtime.make_fake_tensor(
                dt, (cute.sym_int(32),), (1,), assumed_align=align
            )

        args = (
            x,
            vector(Int64, 8),
            vector(Int64, 8),
            vector(Int64, 8),
            vector(Int32, 4),
            vector(Int32, 4),
            vector(Float32, 4),
            vector(Float32, 4),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            obj,
            *args,
            Float32(1.0),
            Int32(0),
            Int64(1),
            Int64(1),
            Int32(1),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(DRAW_MODULE, name, compile_fn)
