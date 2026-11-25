# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The truncated draw: ``support.row.v1``'s thresholds, then ``sample.icdf.v1`` over the kept weights, in one launch (the
kernel behind ``flashinfer.sampling.top_k_sampling_from_probs`` / ``top_p_sampling_from_probs`` /
``min_p_sampling_from_probs`` / ``top_k_top_p_sampling_from_probs`` / ``top_k_top_p_sampling_from_logits`` under
``batch_invariant=True``).

No new numeric class: the draw is the draw class's reading of the canonical sum tree over the *masked* weights
``w'_j = keep_j ? w_j : +0`` (its total is the kept mass, its threshold ``mul(u, mass)``, its prefix nodes the masked
tree's -- so the draw is as placement-invariant as the trees), and the kept set is the support class's integer procedure
over the weights' bit patterns. The kernel runs the support kernel's passes (the trees, the radix selections) with the draw
kernel's recording peers, then the masked tree with the recording on, the uniform from the request stream (the row's
``(seed, offset)`` key at the draw's block, word 0), and the crossing pass over the masked leaf; a row is invalid when its
kept mass is not a positive finite number (nothing kept -- ``k <= 0``, or no positive weight -- or a non-finite row). The
drawn token's log-probability, on request (logits, ``return_logprob``), is the full distribution's: ``lse.row``'s ``logp``
at the token, ``mul(add(z, neg(lg2(S))), LN2)`` with ``S`` the row's sum tree -- the same bits as ``log_probs(logits,
token)``, what a trainer re-scores; never renormalised over the kept set. The one renormalised over the kept set,
``mul(add(z, neg(lg2(mass))), LN2)`` (the probability the token was drawn with), is a named extra (``return_kept_logprob``).

Built on the two kernels' plumbing (``support_kernel``: the staged words, the radix selection with its histograms and early
finish, the integer reducer, the masked tree; ``draw_kernel``: the loader, the recording peers, the block totals and the
crossing pass); its own cache module (``DRAW_SUPPORT_MODULE``), so an edit here never recompiles the other kernels (an edit
to either of them recompiles this one too).

The tail on the list (``draw_support_tail``): after the top-k threshold the kept candidates are compacted once WITH their
positions, and when they fit the list (64 entries; the usual case) the kept-set sum, the top-p rank, the final mass and the
crossing are computed on the list inside every CTA -- the same trees (the identity leaves are exact) and the same integer
procedures, bit for bit, with no further sweep over the row; the dense passes above remain the fallback for larger kept sets.
"""

from __future__ import annotations

from typing import Optional

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64

from .. import canon, jit
from .cute_ext import CuteF32
from .draw_kernel import (
    INT_MAX,
    _block_totals,
    _crossing_pass,
    _DrawLoader,
    _DrawPeers,
    _imax,
)
from .draw_support_tail import (
    alloc_tail,
    gather_lists,
    list_tail,
    reset_box,
    tail_smem_bytes,
)
from .kernel import NAN, NEG_INF, _select, _tiled_copy
from .schedule import BLOCK, LANES, SMEM_BUDGET, V, LSEConfig
from .spec import lg2, logp, max_reduction, scale, shifted, sum_reduction
from .stream_kernel import philox_block, request_key, u64_lo32, u64_shr, uniform32
from .support_kernel import (
    CAP,
    INF_BITS,
    WPT,
    _alloc_hist,
    _compact_sweep,
    _list_filter,
    _cvt_rzi_s64_f32,
    _bitcast_i32_to_f32,
    _f32_bits,
    _neg_or_nan,
    _IntReducer,
    _kept_shortcut,
    _kind,
    _masked_sum,
    _Radix,
    _list_totals,
    _radix_select,
    _select_small,
    _smem_base,
    _smem_layout,
    _weight_bits,
    chunking,
    count_list_default,
    n_passes_for,
)
from .support_spec import FIX_SCALE, Filters

DRAW_SUPPORT_MODULE = "batch_invariant_sampling_draw_support"


# PDL is the only runtime convenience argument of the truncated samplers' kernels (2026-11-19; the `griddepcontrol` instructions
# are always emitted and inert without the launch attribute). Every other convenience dimension is compile-time, each for a
# measured reason (the review of #9695): the INDEX GATHER as a runtime flag cost 50-90 registers on the 256-thread cluster picks
# (the row base from a loaded index instead of the blockIdx-derived row: one CTA per SM fewer, 0.59-0.92x); the TEMPERATURE FORM
# and the KEPT-SET LOG-PROBABILITY as runtime flags put the chain and the per-row scale into every logits kernel -- 122 -> 140-150
# registers at the T256 C2 picks of 32769-65536 x 80-144 rows (0.6x), the `return_logprob` kernels 1.7-2.2x on B200, the plain
# logits entry 2-8 % everywhere. The probability kernels have neither the temperature nor the kept chain: 2 variants per key
# (the gather); the logits kernels 8. `return_logprob` adds the row-sum pass to the k-only / kept-shortcut forms and was never a
# flag.


class DrawSupportKernel:
    """The truncated draw for one ``Kp`` bucket (``N`` runtime): the input kind (logits / probabilities), the filter set
    (which of top-k / top-p / min-p are on, their order), the temperature form, with or without an index gather, the drawn
    token's log-probability under the full distribution and / or under the kept set, and the selection's placement
    (``early_finish``, the support kernel's)."""

    def __init__(
        self,
        dtype,
        Kp: int,
        cfg: LSEConfig,
        from_logits: bool,
        filters: Filters,
        temp_rows: bool,
        has_indices: bool,
        return_logprob: bool,
        early_finish: str = "mass",
        return_kept_logprob: bool = False,
        score: bool = False,
        count_list: Optional[bool] = None,
    ):
        N = Kp * BLOCK
        cfg.validate(N)
        assert cfg.rows_per_cta == 1, "the truncated draw holds one row per CTA group"
        assert not (return_logprob or return_kept_logprob) or from_logits
        assert early_finish in ("", "mass", "all")
        # the score form (``log_probs`` under the sampler's filters): the exact form's selection and kept mass, then the
        # given token's log-probabilities instead of a draw -- logits only, the kept-set value always, no stream, no gather
        assert not score or (from_logits and not has_indices), (
            "the score form takes logits (the kept-set log-probability is its output) and gathers no source rows"
        )
        self.score = score
        self.has_indices, self.temp_rows = has_indices, temp_rows
        self.return_kept_logprob = (
            return_kept_logprob or score
        )  # the score form's output is the kept-set value
        self.early_finish, self.study = early_finish, ""
        self.dtype, self.N, self.Kp, self.cfg = dtype, N, Kp, cfg
        self.from_logits, self.filters, self.return_logprob = (
            from_logits,
            filters,
            return_logprob,
        )
        self.use_k, self.use_p, self.use_m = (
            filters.top_k is not None,
            filters.top_p is not None,
            filters.min_p is not None,
        )
        assert self.use_k or self.use_p or self.use_m
        self.chunk_bits, self.n_chunks = chunking(N)
        pl = cfg.placement(N)
        self.plan_max = canon.plan(max_reduction(N, None), pl)
        self.plan_sum = canon.plan(sum_reduction(N, None), pl)
        self.W, self.C = self.plan_sum.units["warp"], self.plan_sum.units["cta"]
        assert cfg.cluster == self.C and self.plan_sum.units["grid"] == 1
        self.rows = 1
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
        self.kind = _kind(dtype)
        self.n_passes = n_passes_for(self.kind, from_logits)
        # the count selection's list mode: the rule's choice, or forced either way (a placement the suite pins)
        self.count_list = (
            count_list_default(
                self.n_passes, cfg.blocks_per_cta(N) * BLOCK * self.elem_bytes
            )
            if count_list is None
            else bool(count_list)
        )
        _, tile, _, _, _, _ = _smem_layout(self)
        assert tile <= SMEM_BUDGET, (
            "the CTA's share of the row must fit the staging budget (use a cluster)"
        )
        if cfg.stage == "smem":
            assert cfg.smem_stage_ok(N, self.elem_bytes)

    def _tv_layout(self, vec):
        """The thread-value layout of a tiled copy over one row with ``vec`` elements per thread."""
        return cute.make_layout(
            ((LANES, 1), (vec, V // vec, 1)), stride=((V, 1), (1, vec, V))
        )

    def _smem_bytes(self) -> int:
        """The dynamic shared memory: the exchange buffers (the peers' floats, their two integers, the reducer's), the
        staged tile, the histograms, the candidate list (with the kept positions) and the list tail's scratch."""
        b = 0
        if self.W > 1:
            b += self.n_entries * self.W * LANES * 4 + 2 * self.W * LANES * 4
            b += self.W * LANES * 4
        if self.C > 1:
            b += self.n_entries_cta * LANES * 4 + 2 * LANES * 4 + LANES * 4
        _, tile, hist, mslot, mred, cand = _smem_layout(self)
        if not (self.use_k or self.use_p):
            cand += (
                (CAP + 1) * 4 + 2 * 4 + 32
            )  # a min-p-only set: the list tail's list and box
        return (
            b
            + tile
            + hist
            + mslot
            + mred
            + cand
            + (CAP + 1) * 4
            + tail_smem_bytes()
            + 256
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        mIdx: cute.Tensor,
        mIds: cute.Tensor,
        mK: cute.Tensor,
        mP: cute.Tensor,
        mMinP: cute.Tensor,
        mTemp: cute.Tensor,
        mTok: cute.Tensor,
        mValid: cute.Tensor,
        mLogp: cute.Tensor,
        mLogpK: cute.Tensor,
        mMass: cute.Tensor,
        c_const: Float32,
        draw: Int32,
        M: Int64,
        Msrc: Int64,
        N: Int32,
        enable_pdl: Int32,
        stream,
    ):
        k = self.kernel(
            mX,
            mSeed,
            mOff,
            mIdx,
            mIds,
            mK,
            mP,
            mMinP,
            mTemp,
            mTok,
            mValid,
            mLogp,
            mLogpK,
            mMass,
            c_const,
            draw,
            M,
            Msrc,
            N,
            self._tv_layout(self.vec),
            (1, BLOCK),
        )
        if cutlass.const_expr(self.C > 1):
            k.launch(
                grid=[M * self.C, 1, 1],
                block=[self.num_threads, 1, 1],
                cluster=[self.C, 1, 1],
                smem=self._smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )
        else:
            k.launch(
                grid=[M, 1, 1],
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
        mIds: cute.Tensor,
        mK: cute.Tensor,
        mP: cute.Tensor,
        mMinP: cute.Tensor,
        mTemp: cute.Tensor,
        mTok: cute.Tensor,
        mValid: cute.Tensor,
        mLogp: cute.Tensor,
        mLogpK: cute.Tensor,
        mMass: cute.Tensor,
        c_const: Float32,
        draw: Int32,
        M: Int64,
        Msrc: Int64,
        N: Int32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()  # no effect unless the launch carried the PDL attribute (a runtime launch argument)
        W, C = self.W, self.C
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        if cutlass.const_expr(C > 1):
            c_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            rg = bidx // C
        else:
            c_rank = Int32(0)
            rg = bidx
        w = warp_idx % W
        t = lane_idx
        tid = warp_idx * LANES + lane_idx
        coords = {"lane": t, "warp": w, "cta": c_rank, "grid": 0}
        row = rg.to(Int64)
        row_ok = row < M
        row_c = _select(row_ok, row, M - 1)
        # the source row (an index gather reads row indices[row]): a COMPILE-TIME variant -- see the module header
        src_ok = row_ok  # the gather branch overrides it; a value before the staged regions below (the DSL needs one)
        if cutlass.const_expr(self.has_indices):
            src = mIdx[row_c]
            src_ok = (src >= 0) & (src < Msrc)
            src_c = _select(src_ok, src, Int64(0))
            tile_row = src_c.to(Int32)
        else:
            src_c = row_c
            tile_row = rg

        # ---- shared memory: the staged share, the exchange buffers, the histograms ------------------------------
        smem = cutlass.utils.SmemAllocator()
        xbuf = cbuf = ibuf = icbuf = rbuf = rcbuf = None
        n_share = self.cfg.blocks_per_cta(self.N) * BLOCK
        smem_x = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout((1, n_share), order=(1, 0)),
            byte_alignment=16,
        )
        if cutlass.const_expr(W > 1):
            xbuf = smem.allocate_tensor(
                Float32,
                cute.make_layout((1, self.n_entries, W, LANES)),
                byte_alignment=16,
            )
            ibuf = smem.allocate_tensor(
                Int32, cute.make_layout((1, 2, W, LANES)), byte_alignment=16
            )
            rbuf = smem.allocate_tensor(
                Int32, cute.make_layout((W, LANES)), byte_alignment=16
            )
        if cutlass.const_expr(C > 1):
            cbuf = smem.allocate_tensor(
                Float32,
                cute.make_layout((1, self.n_entries_cta, LANES)),
                byte_alignment=16,
            )
            icbuf = smem.allocate_tensor(
                Int32, cute.make_layout((1, 2, LANES)), byte_alignment=16
            )
            rcbuf = smem.allocate_tensor(
                Int32, cute.make_layout((LANES,)), byte_alignment=16
            )
        hist, mslot, mred, gcnt, gmass, cand, cbox, lst, lpos = _alloc_hist(self, smem)
        if cutlass.const_expr(cand is None):  # a min-p-only set
            cand = smem.allocate_tensor(
                Int32, cute.make_layout((CAP + 1,)), byte_alignment=16
            )
            cbox = smem.allocate_tensor(
                Int32, cute.make_layout((2,)), byte_alignment=16
            )
        cpos = smem.allocate_tensor(
            Int32, cute.make_layout((CAP + 1,)), byte_alignment=16
        )  # the kept candidates' positions
        tout = alloc_tail(smem)

        in_bits = self.vec * mX.element_type.width
        thr_x, atom_x = _tiled_copy(mX, in_bits, tv_layout, tiler_mn, t)
        thr_async = atom_async = None
        if cutlass.const_expr(self.cfg.stage == "smem"):
            atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=in_bits,
            )
            thr_async = cute.make_tiled_copy(atom_async, tv_layout, tiler_mn).get_slice(
                t
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
        # the tile is staged here, before anything reads it: the radix sweeps read the words directly, and with no row-sum
        # tree before them (the top_k_first order, top-k alone) no leaf read would have triggered the cp.async staging
        if cutlass.const_expr(self.cfg.stage != "smem"):
            loader.stage_from_regs(smem_x, cta_base)
        else:
            loader._stage_all()
        cute.arch.barrier()
        peers = _DrawPeers(xbuf, Int32(0), w, lane_idx, cbuf, c_rank, ibuf, icbuf, W, C)
        red = _IntReducer(W, C, w, lane_idx, rbuf, rcbuf)

        # ===== the weights: (max,) sum ==============================================================================
        nm = Float32(0.0)
        wmax = Float32(1.0)
        if cutlass.const_expr(self.from_logits):
            state, start = canon.init_state(self.plan_max, fp, loader.max_leaf)
            m = canon.run(self.plan_max, state, start, peers, fp)
            nm = fp.neg(m)
            loader.nm, loader.c = nm, c
            leaf = loader.sum_leaf
        else:
            leaf = loader.max_leaf
            if cutlass.const_expr(self.use_m):
                state, start = canon.init_state(self.plan_max, fp, loader.max_leaf)
                wmax = canon.run(self.plan_max, state, start, peers, fp)
        words = cute.make_tensor(
            cute.recast_ptr(smem_x.iterator, dtype=Int32),
            cute.make_layout((n_share * self.elem_bytes // 4,)),
        )
        rx = _Radix(
            words,
            Int32(n_share * self.elem_bytes // 4),
            cta_base * BLOCK,
            N,
            nm,
            c,
            hist,
            mslot,
            mred,
            self.kind,
            self.from_logits,
            gcnt,
            gmass,
            cand,
            lst,
            cbox,
            lpos,
        )
        # the row sum is read by a top-p selection over the whole row (the joint order, or no top-k) -- with the top_k_first
        # shortcut the top-p base is the top-k kept mass (the masked tree below) -- and by the drawn token's
        # full-distribution log-probability
        S = Float32(1.0)
        if cutlass.const_expr(
            (self.use_p and not _kept_shortcut(self)) or self.return_logprob
        ):
            state, start = canon.init_state(self.plan_sum, fp, leaf)
            S = canon.run(self.plan_sum, state, start, peers, fp)

        # ===== the uniform: the row's key at the draw's block, word 0 (the score form gathers the given token: no key) ==
        u = Float32(0.0)
        if cutlass.const_expr(not self.score):
            key = request_key(mSeed[row_c], mOff[row_c])
            k0 = u64_lo32(key)
            k1 = u64_lo32(u64_shr(key, Int32(32)))
            w0, _, _, _ = philox_block(draw, Int32(0), k0, k1)
            u = uniform32(w0)

        # ===== the list-tail flow (docs 21.3): the top-k threshold over the row, the kept compaction WITH positions, and the
        # tail -- the kept-set sum, the top-p rank, the final mass and the crossing -- on the list inside every CTA when the
        # kept set fits it; the dense sweeps as the fallback. Same bits: the same integer procedures and the same trees.
        kp_log = self.Kp.bit_length() - 1
        thr_bits = Int32(0)
        n_list = Int32(-1)  # the count tail's list count when the kept set is on it
        kk = Int32(0)
        n_all = Int32(
            CAP + 1
        )  # the tail's list count: the top-k kept set (filtered from the count tail's list)
        neg_k = Boolean(
            False
        )  # the top-k pass admits no negative weight / NaN: its candidate count then falls short of N
        if cutlass.const_expr(self.use_k):
            kk = mK[row_c]
            # k <= 0: nothing kept (INF_BITS; the dense tail over an all-masked row, whose flagged leaves still report a
            # negative weight or a NaN as the contract's NaN mass -- the count pass that flags the list paths is skipped
            # here); k >= N: everything kept (threshold 0, the list beyond its capacity -- the dense path); otherwise the
            # selection
            thr_bits = _select(kk <= Int32(0), Int32(INF_BITS), Int32(0))
            if (kk > Int32(0)) & (kk < N):
                thr_bits, n_list, tot_c0 = _topk_threshold(
                    self, rx, tid, t, kk, peers, red
                )
                neg_k = tot_c0 != N
                # the kept set leaves the count tail's list NOW: the list aliases the mass slots that a top-p selection
                # over the row clears (the joint order; a shortcut whose kept set exceeds the list) -- the tail's floor
                # applies any later threshold (the final kept set is a subset of this one)
                if n_list >= Int32(0):
                    n_all = _list_filter(
                        rx.lst,
                        rx.lpos,
                        n_list,
                        thr_bits,
                        rx.cand,
                        cpos,
                        rx.cbox,
                        tid,
                        t,
                        self.num_threads,
                    )
                else:
                    if cutlass.const_expr(_kept_shortcut(self)):
                        n_all = _kept_compact(
                            self, rx, tid, t, w, c_rank, thr_bits, peers, cpos
                        )
        fits = Boolean(False)
        if cutlass.const_expr(self.use_p):
            base = Int32(0)
            Sp = S
            if cutlass.const_expr(_kept_shortcut(self)):
                base = thr_bits
                fits = n_all <= Int32(CAP)
                if fits:
                    Sp, _ta, _tb = list_tail(
                        cand,
                        cpos,
                        n_all,
                        base,
                        Float32(0.0),
                        Boolean(False),
                        w,
                        t,
                        tout,
                        kp_log,
                    )
                else:
                    Sp = _masked_sum(self, leaf, peers, thr_bits)
                loader.frags.clear()  # fragments first loaded inside the branch must not be reused after it
            scale_fix = fp.div(Float32(FIX_SCALE), Sp)
            pv = mP[row_c]
            target = _cvt_rzi_s64_f32(fp.mul(pv, Float32(FIX_SCALE)))
            sel_p = Int32(0)
            tot_c = Int32(0)
            tot_m = Int64(0)
            min_pos = Int32(INT_MAX)
            if cutlass.const_expr(_kept_shortcut(self)):
                if fits:
                    tot_m, min_pos = _list_totals(cand, n_all, t, scale_fix)
                    tot_c = n_all
                    sel_p = _select_small(
                        cand, n_all, t, w, Int32(0), target, scale_fix, True
                    )
                else:
                    sel_p, tot_c, tot_m, min_pos, _nl = _radix_select(
                        self,
                        rx,
                        tid,
                        t,
                        True,
                        Int32(0),
                        target,
                        scale_fix,
                        base,
                        peers,
                        red,
                    )
            else:
                sel_p, tot_c, tot_m, min_pos, _nl = _radix_select(
                    self,
                    rx,
                    tid,
                    t,
                    True,
                    Int32(0),
                    target,
                    scale_fix,
                    base,
                    peers,
                    red,
                )
            use_radix = (tot_m >= target) & (tot_c > Int32(0)) & (pv < Float32(1.0))
            fallback = _select(min_pos == Int32(INT_MAX), Int32(INF_BITS), min_pos)
            thr_bits = _imax(thr_bits, _select(use_radix, sel_p, fallback))
            thr_bits = _select(
                scale_fix < Float32(float("inf")), thr_bits, Int32(INF_BITS)
            )
        if cutlass.const_expr(self.use_m):
            thr_bits = _imax(thr_bits, _f32_bits(fp.mul(wmax, mMinP[row_c])))
        if cutlass.const_expr(not _kept_shortcut(self)):
            if cutlass.const_expr(self.use_k):
                # the tail's list holds the top-k kept set (filtered above; the final threshold is at or above the top-k one
                # and the tail's floor applies it); without a list the row is compacted at the final threshold when the
                # top-k bound says the kept set fits the list (k <= CAP), the dense sweeps otherwise
                if (n_list < Int32(0)) & ((kk > Int32(0)) & (kk < N)):
                    if kk <= Int32(CAP):
                        n_all = _kept_compact(
                            self, rx, tid, t, w, c_rank, thr_bits, peers, cpos
                        )
                    else:
                        n_all = Int32(CAP + 1)
            # without a top-k bound (top-p alone, min-p alone) the kept set's size is unknown before a sweep and at the
            # production thresholds exceeds the list (hundreds of tokens at p = 0.9 / min-p = 0.05 on the operator's probe
            # rows): the row takes the dense sweeps directly -- a compaction that overflows is a wasted pass (2026-11-17
            # A/B against the tree before the rework: min-p alone 1.25-2.9x and top-p alone 1.0-2.5x slower with it)
            fits = n_all <= Int32(CAP)
        mass = Float32(0.0)
        first_tok = Int32(-1)
        last_tok = Int32(-1)
        if fits:
            mass, first_tok, last_tok = list_tail(
                cand,
                cpos,
                n_all,
                thr_bits,
                u,
                Boolean(not self.score),
                w,
                t,
                tout,
                kp_log,
            )
        else:
            mass, first_tok, last_tok = _dense_tail(
                self, loader, leaf, peers, thr_bits, u, t, w, c_rank
            )
        loader.frags.clear()
        # the input domain (a negative weight or a NaN in the row): NaN mass, which fails every validity test below -- the
        # list tail's rows know it from the top-k pass (its candidate count), the dense tail's from its own masked tree
        mass = _select(neg_k, Float32(NAN), mass)
        if cutlass.const_expr(self.score):
            token, valid, mass, lp, lpk = _score_token(
                self, mX, mIds, row_c, src_c, N, S, mass, thr_bits, nm, c, fp
            )
        else:
            tok = _select(first_tok >= Int32(0), first_tok, last_tok)
            s_ok = (mass > Float32(0.0)) & (mass < Float32(float("inf")))
            valid = (
                s_ok & (tok >= Int32(0)) & src_ok
            )  # an index outside the source rows: an invalid output row
            token = _select(valid, tok, Int32(-1))
            lp = Float32(NAN)
            lpk = Float32(NAN)
            if cutlass.const_expr(self.return_logprob or self.return_kept_logprob):
                tok_c = _select(valid, token, Int32(0))
                xt = mX[src_c, tok_c].to(Float32)
                zt = shifted(fp, xt, nm, c)
                if cutlass.const_expr(self.return_logprob):
                    lp = _select(valid, logp(fp, zt, fp.neg(lg2(fp, S))), Float32(NAN))
                if cutlass.const_expr(self.return_kept_logprob):
                    lpk = _select(
                        valid, logp(fp, zt, fp.neg(lg2(fp, mass))), Float32(NAN)
                    )
        if t == 0:
            if w == 0:
                if c_rank == 0:
                    if row_ok:
                        if cutlass.const_expr(
                            not self.score
                        ):  # the score form draws nothing
                            mTok[row] = token
                        mValid[row] = _select(valid, Int32(1), Int32(0))
                        mMass[row] = mass
                        if cutlass.const_expr(self.return_logprob):
                            mLogp[row] = lp
                        if cutlass.const_expr(self.return_kept_logprob):
                            mLogpK[row] = lpk
        peers.finish()
        cute.arch.griddepcontrol_launch_dependents()


def _topk_threshold(k, rx, tid, lane, kk, peers, red):
    """The top-k threshold pattern of a row with ``0 < k < N``, the count tail's list count (-1 when the kept set is not on
    its list) and pass 0's cluster-wide candidate count -- ``N`` unless the row holds a negative weight or a NaN
    (``_radix_sweep``'s value test admits neither): three scalars, so the kernel's row-uniform branch around the selection
    carries no tuple."""
    r = _radix_select(
        k, rx, tid, lane, False, kk, Int64(0), Float32(0.0), Int32(0), peers, red
    )
    return r[0], r[4], r[1]


def _kept_compact(k, rx, tid, lane, warp, c_rank, base_bits, peers, cpos):
    """The positive weights with pattern ``>= base_bits`` compacted into the candidate list WITH their positions (every CTA its
    share, then the cluster's lists appended through ``mapa``); returns the cluster-wide count (CTA-uniform). The list holds at
    most ``CAP`` entries: the caller takes the list tail when the count fits, the dense sweeps otherwise."""
    n = k.num_threads
    n_iters = -(-(k.cfg.blocks_per_cta(k.N) * BLOCK * k.elem_bytes // 4) // (n * WPT))
    reset_box(rx.cbox, tid)
    cute.arch.barrier()
    _compact_sweep(
        rx.words,
        rx.n_words,
        rx.col0,
        rx.N,
        tid,
        lane,
        rx.nm,
        rx.c,
        Int32(0),
        base_bits,
        rx.cand,
        _smem_base(rx.cbox),
        31,
        rx.kind,
        rx.from_logits,
        True,
        n,
        n_iters,
        1 if rx.kind == 0 else 2,
        Int32(INT_MAX),
        CAP,
        cpos,
        True,
    )
    cute.arch.barrier()
    n_all = gather_lists(rx.cand, cpos, rx.cbox, tid, c_rank, k.C, n)
    if k.C > 1:
        peers.n_cluster_exchanges += 1
    return n_all


def _addr_token(k, a, ok):
    """The token of tree address ``a`` (``-1`` unless ``ok``)."""
    slot = a >> 3
    lane_a = slot >> (k.Kp.bit_length() - 1)
    block_a = slot & Int32(k.Kp - 1)
    return _select(ok, ((block_a << 5) + lane_a) * 8 + (a & 7), Int32(-1))


def _flagged(leaf):
    """``leaf`` with a side effect: ``flags`` collects ``_neg_or_nan(we)`` for every value it returns -- one predicate compare
    per element, no register, no extra pass: the input-domain test of a probability row (a negative weight) and of any row (a
    NaN) inside a tree the kernel runs anyway. Returns ``(the wrapped leaf, flags)``; ``_poison`` combines them."""
    flags = []

    def f(key):
        we = leaf(key)
        flags.append(_neg_or_nan(we))
        return we

    return f, flags


def _any(flags):
    """The OR of the flags as a PAIRWISE tree. A chain (``neg = neg | flag`` leaf after leaf) is a serial critical path of
    70-odd dependent predicate ops that the scheduler covers by hoisting the leaves' loads and conversions ahead of it: the
    top-k kernel at bf16 152064 went from 64 to 117 registers on that form alone (2026-11-18, attributed by ablation), and the
    Int32 chain before it spilled the T512 kernels. The tree is seven deep and ptxas still fuses each first-level OR into its
    compare (``FSETP.LTU.OR``)."""
    while len(flags) > 1:
        nxt = [flags[i] | flags[i + 1] for i in range(0, len(flags) - 1, 2)]
        if len(flags) % 2:
            nxt.append(flags[-1])
        flags = nxt
    return flags[0]


def _poison(state, flags):
    """One partial of a tree's per-thread state (what ``canon.init_state`` returns) becomes NaN where any of the thread's flags
    is set, so the tree's result is NaN on every thread of the row: the row-wide test rides the tree's own exchanges, no extra
    reduction; clear flags leave the partial, and every bit of the tree, untouched."""
    if flags:
        k0 = next(iter(state))
        state[k0] = _select(_any(flags), Float32(NAN), state[k0])


def _dense_tail(k, loader, leaf, peers, thr_bits, u, t, w, c_rank):
    """The list tail's fallback over the row: the kept mass as the masked tree with the peers recording, and (the draw forms)
    the crossing over the masked leaf -- returns ``(mass, first token, last token)`` with ``-1`` tokens where none. The kept
    test is one compare on the value (``we >= thr``: the pattern test's set for the class's non-negative weights, no magnitude
    mask), and the input-domain test rides the tree: its leaves flag a negative weight or a NaN (``_flagged``) and the flag
    poisons one partial (``_poison``), so such a row's mass is NaN on every thread -- no token, no validity; no extra pass or
    exchange (2026-11-18, the review's second round; docs 21.9)."""
    fp = CuteF32
    thr = _bitcast_i32_to_f32(thr_bits)
    flagged, neg = _flagged(leaf)

    def masked_leaf(key):  # the crossing's leaf: the kept value, no side effect
        we = leaf(key)
        return _select(we >= thr, we, Float32(0.0))

    def tree_leaf(key):  # the tree's leaf: the same value, the flag on the way
        we = flagged(key)
        return _select(we >= thr, we, Float32(0.0))

    peers.recording = True
    state, start = canon.init_state(k.plan_sum, fp, tree_leaf)
    _poison(state, neg)
    totals = _block_totals(k.plan_sum, state, k.kpw)
    mass = canon.run(k.plan_sum, state, start, peers, fp)
    peers.recording = False
    first = Int32(INT_MAX)
    last = Int32(-1)
    if not k.score:
        thr_u = fp.mul(u, mass)
        first, last = _crossing_pass(
            k, loader, masked_leaf, peers, totals, thr_u, t, w, c_rank
        )
        first, last = peers.int_reduce(first, last)
    return (
        mass,
        _addr_token(k, first, first != Int32(INT_MAX)),
        _addr_token(k, last, last >= Int32(0)),
    )


def _score_token(k, mX, mIds, row_c, src_c, N, S, mass, thr_bits, nm, c, fp):
    """The score form (``log_probs`` under the sampler's filters): the given token's log-probability under the kept set,
    ``mul(add(z_t, neg(lg2(mass))), LN2)`` -- the drawn token's ``kept_logprob`` chain at a given token -- and under the full
    distribution (over ``S``) when requested. A token outside the kept set has probability zero there: ``-inf``; an id outside
    ``[0, N)``, or a row whose kept mass is not a positive finite number: NaN (the row class's ``logp`` for a bad id, the
    draw's invalid row). Returns the exact form's tuple ``(token, valid, mass, logprob, kept_logprob)`` with the token ``-1``
    (nothing is drawn) and ``valid`` the membership."""
    tid64 = mIds[row_c]
    in_range = (tid64 >= Int64(0)) & (tid64 < N.to(Int64))
    tok_c = _select(in_range, tid64, Int64(0)).to(Int32)
    xt = mX[src_c, tok_c].to(Float32)
    zt = shifted(fp, xt, nm, c)
    s_ok = (Float32(0.0) < mass) & (Float32(float("inf")) > mass)
    kept = in_range & s_ok & (_weight_bits(fp.ex2(zt)) >= thr_bits)
    lpk = _select(
        kept,
        logp(fp, zt, fp.neg(lg2(fp, mass))),
        _select(in_range & s_ok, Float32(NEG_INF), Float32(NAN)),
    )
    lp = Float32(NAN)
    if k.return_logprob:
        lp = _select(in_range, logp(fp, zt, fp.neg(lg2(fp, S))), Float32(NAN))
    return Int32(-1), kept, mass, lp, lpk


def kernel_name(
    dtype_str: str,
    Kp: int,
    cfg: LSEConfig,
    from_logits: bool,
    filters: Filters,
    temp_rows: bool,
    has_indices: bool,
    return_logprob: bool,
    enable_pdl: bool,
    early_finish: str = "mass",
    return_kept_logprob: bool = False,
    score: bool = False,
    count_list: Optional[bool] = None,
) -> str:
    """The cache object's name: one per (dtype, Kp bucket, config, input kind, filter set, temperature form, the index
    gather, the two log-probability outputs, the selection's placements, the score form); PDL is a runtime launch argument
    (``enable_pdl`` is accepted and ignored)."""
    return jit.sanitize_symbol_name(
        f"drawsup{'s' if score else ''}_{'logits' if from_logits else 'probs'}_{filters.name}_{dtype_str}_Kp{Kp}_{cfg.describe()}"
        f"_{'trow' if temp_rows else 'tconst'}{'_idx' if has_indices else ''}{'_lp' if return_logprob else ''}"
        f"{'_lpk' if return_kept_logprob else ''}"
        f"{f'_ef{early_finish}' if early_finish else ''}"
        f"{'' if count_list is None else ('_cl1' if count_list else '_cl0')}"
    )


def get_compiled_kernel(
    dtype_str: str,
    N: int,
    cfg: LSEConfig,
    from_logits: bool,
    filters: Filters,
    temp_rows: bool,
    has_indices: bool,
    return_logprob: bool,
    enable_pdl: bool,
    early_finish: str = "mass",
    return_kept_logprob: bool = False,
    score: bool = False,
    count_list: Optional[bool] = None,
):
    """The compiled kernel serving vocabulary ``N``: one per (dtype, Kp bucket, config, input kind, filter set, temperature
    form, the index gather, the two log-probability outputs, the placements, the score form; the probability kernels have
    neither the temperature nor the kept chain and key both False); ``N``, ``M``, the row strides and PDL are runtime
    arguments (``enable_pdl`` is accepted and ignored here)."""
    return _compile_bucket(
        *_key(
            dtype_str,
            N,
            cfg,
            from_logits,
            filters,
            temp_rows,
            has_indices,
            return_logprob,
            enable_pdl,
            early_finish,
            return_kept_logprob,
            score,
            count_list,
        )
    )


def _key(
    dtype_str,
    N,
    cfg,
    from_logits,
    filters,
    temp_rows,
    has_indices,
    return_logprob,
    enable_pdl,
    early_finish="mass",
    return_kept_logprob=False,
    score=False,
    count_list=None,
):
    """The compile key of the kernel serving ``N``: PDL is a runtime launch argument and leaves it; the probability kernels
    have neither the temperature nor the kept chain, so those two dimensions are keyed False for them."""
    from .schedule import padded_blocks

    return (
        dtype_str,
        padded_blocks(N),
        cfg,
        from_logits,
        filters,
        bool(temp_rows and from_logits),
        bool(has_indices),
        return_logprob,
        False,
        early_finish,
        bool((return_kept_logprob or score) and from_logits),
        score,
        count_list,
    )


def precompile(
    dtype_str: str,
    N: int,
    cfg: LSEConfig,
    from_logits: bool,
    filters: Filters,
    temp_rows: bool,
    has_indices: bool,
    return_logprob: bool,
    enable_pdl: bool,
    early_finish: str = "mass",
    return_kept_logprob: bool = False,
    score: bool = False,
    count_list: Optional[bool] = None,
) -> None:
    """Compile the kernel serving ``N`` into the on-disk cache without loading it, outside the op family's module lock
    (:func:`jit.precompile`): what a precompile pool's worker calls. ``get_compiled_kernel`` holds the lock for the whole build
    (concurrent processes missing the same kernel compile it once), so workers compiling different kernels of one family
    through it would run serially."""
    jit.precompile(
        DRAW_SUPPORT_MODULE,
        *_spec(
            *_key(
                dtype_str,
                N,
                cfg,
                from_logits,
                filters,
                temp_rows,
                has_indices,
                return_logprob,
                enable_pdl,
                early_finish,
                return_kept_logprob,
                score,
                count_list,
            )
        ),
    )


def _spec(
    dtype_str,
    Kp,
    cfg,
    from_logits,
    filters,
    temp_rows,
    has_indices,
    return_logprob,
    enable_pdl,
    early_finish,
    return_kept_logprob=False,
    score=False,
    count_list=None,
):
    """The cache object's name and the compile function of one Kp bucket and specialisation."""
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = kernel_name(
        dtype_str,
        Kp,
        cfg,
        from_logits,
        filters,
        temp_rows,
        has_indices,
        return_logprob,
        enable_pdl,
        early_finish,
        return_kept_logprob,
        score,
        count_list,
    )

    def compile_fn():
        obj = DrawSupportKernel(
            dtype,
            Kp,
            cfg,
            from_logits,
            filters,
            temp_rows,
            has_indices,
            return_logprob,
            early_finish,
            return_kept_logprob,
            score,
            count_list,
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
            vector(Int64, 8),  # seed
            vector(Int64, 8),  # offset
            vector(Int64, 8),  # indices
            vector(Int64, 8),  # token ids (the score form)
            vector(Int32, 4),  # top_k
            vector(Float32, 4),  # top_p
            vector(Float32, 4),  # min_p
            vector(Float32, 4),  # temperature
            vector(Int32, 4),  # tokens
            vector(Int32, 4),  # valid
            vector(Float32, 4),  # logprob (the full distribution's)
            vector(Float32, 4),  # kept logprob (renormalised over the kept set)
            vector(Float32, 4),  # mass
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
            Int32(0),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return name, compile_fn


@functools.cache
def _compile_bucket(*key):
    """Compile, or load from the cache, the kernel of one compile key (the serving path: the module lock is held for the
    whole build)."""
    return jit.build_and_load(DRAW_SUPPORT_MODULE, *_spec(*key))
