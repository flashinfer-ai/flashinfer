# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The ``lse.row`` CuTe-DSL kernel: one CTA group per row, two canonical passes (max, sum) driven by the canon engine,
then the requested outputs.

The reductions are not written here: the kernel builds the two :class:`canon.Plan` objects from the config's placement,
loads leaves through ``canon.init_state`` and runs ``canon.run`` with a hardware :class:`canon.Peers` (shuffles, smem
exchange, distributed-shared-memory exchange for the cluster members). The chains around them (``spec.shifted / weight
/ prob / lse_nat / logp``) are the spec's own functions on the ``CuteF32`` backend. Kernel-specific is the plumbing
only: tiled copies, predicates, the ``-inf`` fill of absent elements, the staging of the fragments between the passes.

The vocabulary ``N`` is a *runtime* argument: a kernel is compiled per ``Kp`` bucket (``padded_blocks(N)``) and serves
every ``N`` of the bucket. Elements beyond ``N`` read as ``-inf`` (the fragments are filled before the predicated
loads), the identity of both trees (``spec`` digits' ``identity_note``), so the bucket kernel computes the spec of
every ``N`` it serves. A temperature is either one host-computed ``c`` (scalar / none) or a per-row tensor, from which
the kernel forms ``c = div(LOG2E, T_row)`` with the same ``div.rn``.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import arith, llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import canon, jit
from .cute_ext import CuteF32
from .schedule import BLOCK, LANES, V, LSEConfig
from .spec import (
    Mode,
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

NEG_INF = float("-inf")
NAN = float("nan")
LSE_MODULE = (
    "batch_invariant_sampling_lse"  # the on-disk cache module of every lse.row kernel
)


@cute.jit
def _predicate(tXcX: cute.Tensor, M: Int64, N: Int32) -> cute.Tensor:
    """Per-vector predicate: (row, col) < (M, N)."""
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
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k], (M, N)
            )
    return tXpX


def _select(cond, a, b):
    """``cond ? a : b`` on two values of one numeric type (``selp``; no arithmetic)."""
    ty = type(a)
    return ty(arith.select(cond.ir_value(), ty(a).ir_value(), ty(b).ir_value()))


@dsl_user_op
def _map_shared_rank(
    smem_ptr: cute.Pointer, peer_rank: Int32, *, loc=None, ip=None
) -> Int32:
    """``mapa.shared::cluster``: the address of ``smem_ptr`` in the peer CTA's shared memory (a move of bits)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), peer_rank.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_shared_cluster_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """``ld.shared::cluster.f32``: one fp32 from a peer CTA's shared memory (bits move; no arithmetic). Marked as having
    side effects so it is never hoisted above the cluster barrier that publishes the value."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _tiled_copy(mT, bits, tv_layout, tiler_mn, vtid):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mT.element_type, num_bits_per_copy=bits
    )
    return cute.make_tiled_copy(atom, tv_layout, tiler_mn).get_slice(vtid), atom


class _RowLoader:
    """``canon.init_state`` leaf loader for both passes. Serves the fp32 logit of an element out of per-block fragments
    (loaded the first time a key touches a block; cleared between blocks unless the config keeps registers). Absent
    elements (beyond ``N``, or rows beyond ``M``) read ``-inf``: the fragment is filled with it before the predicated
    load, which skips the predicated-off vectors."""

    MAX_INFLIGHT = 8  # cp.async per thread kept pending (two commit groups); bounds issue burstiness

    def __init__(
        self,
        k,
        plan,
        coords,
        mX,
        idX,
        tiler_mn,
        rg,
        M,
        N,
        thr_x,
        atom_x,
        smem_x=None,
        thr_async=None,
        atom_async=None,
        cta_base=None,
    ):
        self.k, self.plan, self.coords = k, plan, coords
        self.mX, self.idX, self.tiler_mn, self.rg, self.M, self.N = (
            mX,
            idX,
            tiler_mn,
            rg,
            M,
            N,
        )
        self.thr_x, self.atom_x = thr_x, atom_x
        self.smem_x, self.thr_async, self.atom_async, self.cta_base = (
            smem_x,
            thr_async,
            atom_async,
            cta_base,
        )
        self._staged = False
        self.ri_elem = plan.reg_index("elem")
        self.ri_block = plan.reg_index("block")
        self.frags = {}
        self.nm = self.c = None

    def key_for_block(self, kk):
        key = [0] * self.plan.n_reg
        if self.ri_block is not None:  # Kp == W: the block digit has no register factor
            key[self.ri_block] = kk
        return tuple(key)

    def k_global(self, kk):
        return self.plan.digit_value("block", self.coords, self.key_for_block(kk))

    def _stage_all(self):
        """cp.async every block of this warp's share into the CTA's tile (predicated-off vectors are zero-filled by the
        hardware; they read as ``-inf`` through :meth:`x`), at most MAX_INFLIGHT copies per thread pending."""
        n = self.k.kpw
        per_op = V // self.k.vec
        ops_per_group = max(1, (self.MAX_INFLIGHT // 2) // per_op)
        pending_ops, groups = 0, 0
        for kk in range(n):
            kg = self.k_global(kk)
            cX = cute.local_tile(self.idX, self.tiler_mn, (self.rg, kg))
            pred = _predicate(self.thr_async.partition_S(cX), self.M, self.N)
            gT = self.thr_async.partition_S(
                cute.local_tile(self.mX, self.tiler_mn, (self.rg, kg))
            )
            dT = self.thr_async.partition_D(
                cute.local_tile(self.smem_x, self.tiler_mn, (0, kg - self.cta_base))
            )
            cute.copy(self.atom_async, gT, dT, pred=pred)
            pending_ops += 1
            if pending_ops == ops_per_group:
                cute.arch.cp_async_commit_group()
                groups += 1
                pending_ops = 0
                if groups >= 2:
                    cute.arch.cp_async_wait_group(1)
        if pending_ops:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self._staged = True

    def block(self, kk):
        if kk not in self.frags:
            if not self.k.cfg.keep_regs:
                self.frags.clear()
            kg = self.k_global(kk)
            cX = cute.local_tile(self.idX, self.tiler_mn, (self.rg, kg))
            if self.k.cfg.stage == "smem":
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
                fx.fill(self.mX.element_type(NEG_INF))
                cute.copy(self.atom_x, gX, fx, pred=pred)
            self.frags[kk] = (pred, fx)
        return self.frags[kk]

    def x(self, kk, e):
        """The fp32 logit of element ``e`` of block ``kk`` (exact widening of a 16-bit input); an absent element is ``-inf``
        (the register path fills its fragments before the predicated load; the staged path selects on the predicate)."""
        pred, fx = self.block(kk)
        v = fx[e].to(Float32)
        if self.k.cfg.stage == "smem":
            return _select(pred[e // self.k.vec, 0, 0], v, Float32(NEG_INF))
        return v

    def max_leaf(self, key):
        return self.x(
            key[self.ri_block] if self.ri_block is not None else 0, key[self.ri_elem]
        )

    def sum_leaf(self, key):
        fp = CuteF32
        x = self.max_leaf(key)
        return fp.ex2(shifted(fp, x, self.nm, self.c))


class _CutePeers(canon.Peers):
    """The hardware peers of one thread: lane shuffles, the cross-warp exchange through this CTA's shared memory and,
    for the cluster members, the cross-CTA exchange through distributed shared memory (every CTA writes its per-lane
    partials to its own ``cbuf``, a cluster barrier (release / acquire) publishes them, every thread reads the
    siblings' ``cbuf`` through ``mapa`` and combines them in factor order)."""

    def __init__(self, xbuf, group, warp_id, lane_id, cbuf=None, cta_rank=None):
        self.xbuf, self.group, self.warp_id, self.lane_id = (
            xbuf,
            group,
            warp_id,
            lane_id,
        )
        self.cbuf, self.cta_rank = cbuf, cta_rank
        self.n_exchanges = 0
        self.n_cluster_exchanges = 0

    def shuffle_xor(self, key, value, mask):
        if value is None:
            return None
        return cute.arch.shuffle_sync_bfly(value, offset=mask)

    def exchange(self, step, state):
        if step.level == "cta":
            return self._exchange_cluster(step, state)
        keys = [k for k, v in state.items() if v is not None]
        if self.n_exchanges:
            cute.arch.barrier()  # previous readers must finish before we overwrite
        for e, k in enumerate(keys):
            self.xbuf[self.group, e, self.warp_id, self.lane_id] = state[k]
        cute.arch.barrier()
        self.n_exchanges += 1
        if step.warp_stride == 1 and step.radix == step.warp_total:
            sib_ids = list(range(step.radix))
        else:
            my = (self.warp_id // step.warp_stride) % step.radix
            sib_ids = [
                self.warp_id + (s - my) * step.warp_stride for s in range(step.radix)
            ]
        out = {k: [None] * step.radix for k in state}
        for e, k in enumerate(keys):
            out[k] = [self.xbuf[self.group, e, sid, self.lane_id] for sid in sib_ids]
        return out

    def _exchange_cluster(self, step, state):
        keys = [k for k, v in state.items() if v is not None]
        if self.n_cluster_exchanges:
            cute.arch.cluster_arrive()  # the peers' readers of the previous exchange must finish before we overwrite
            cute.arch.cluster_wait()
        for e, k in enumerate(keys):
            self.cbuf[self.group, e, self.lane_id] = state[k]
        cute.arch.cluster_arrive()  # release: our partials are visible to the cluster ...
        cute.arch.cluster_wait()  # ... acquire: and the peers' partials to us
        self.n_cluster_exchanges += 1
        if step.warp_stride == 1 and step.radix == step.warp_total:
            sib_ids = list(range(step.radix))
        else:
            my = (self.cta_rank // step.warp_stride) % step.radix
            sib_ids = [
                self.cta_rank + (s - my) * step.warp_stride for s in range(step.radix)
            ]
        bases = [_map_shared_rank(self.cbuf.iterator, Int32(sid)) for sid in sib_ids]
        out = {k: [None] * step.radix for k in state}
        for e, k in enumerate(keys):
            off = cute.crd2idx((self.group, e, self.lane_id), self.cbuf.layout) * 4
            out[k] = [_ld_shared_cluster_f32(b + off) for b in bases]
        return out

    def finish(self):
        """No CTA leaves while a peer may still read its ``cbuf``."""
        if self.n_cluster_exchanges:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


class LSERowKernel:
    """``lse.row`` for one ``Kp`` bucket (``N`` is a runtime argument), one :class:`Mode` and one temperature form."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        Kp: int,
        cfg: LSEConfig,
        mode: Mode,
        temp_rows: bool,
        k_slots: int = 1,
    ):
        N = (
            Kp * BLOCK
        )  # a representative N of this bucket: everything below depends on N only via Kp
        cfg.validate(N)
        self.dtype, self.N, self.Kp, self.cfg, self.mode, self.temp_rows = (
            dtype,
            N,
            Kp,
            cfg,
            mode,
            temp_rows,
        )
        # token ids per row the log-probability output is specialised for: 1 (one id, every thread forms it) or a
        # multiple of 32 (the lanes of warp 0 take the ids in turn; `K` at run time predicates the tail)
        assert k_slots == 1 or (k_slots > 0 and k_slots % LANES == 0), k_slots
        self.k_slots = k_slots
        self.out_dtype = dtype if (mode.probs and not mode.probs_f32) else Float32
        pl = cfg.placement(N)
        self.plan_max = canon.plan(max_reduction(N, None), pl)
        self.plan_sum = canon.plan(sum_reduction(N, None), pl)
        assert self.plan_max.units["grid"] == 1
        self.W = self.plan_max.units["warp"]
        self.C = self.plan_max.units["cta"]
        assert cfg.cluster == self.C and self.plan_max.units["lane"] == LANES
        self.rows = cfg.rows_per_cta
        self.kpw = cfg.blocks_per_warp(N)
        self.vec = cfg.vec_elems
        self.num_threads = cfg.num_threads
        warp_steps = [i for i, s in enumerate(self.plan_max.steps) if s.level == "warp"]
        self.n_entries = max(
            (len(self.plan_max.keys_at(i)) for i in warp_steps), default=0
        )
        cta_steps = [i for i, s in enumerate(self.plan_max.steps) if s.level == "cta"]
        self.n_entries_cta = max(
            (len(self.plan_max.keys_at(i)) for i in cta_steps), default=0
        )
        self.elem_bytes = dtype.width // 8
        if cfg.stage == "smem":
            assert cfg.smem_stage_ok(N, self.elem_bytes), (
                "smem staging needs 16-byte vectors, aligned rows and a tile under budget"
            )
        # the output's vector: the same element ownership, at most 16 bytes per copy
        self.vec_out = min(self.vec, max(1, 128 // self.out_dtype.width))

    def _tv_layout(self, vec: int):
        rows = self.rows
        shape = ((LANES, rows), (vec, V // vec, 1))
        stride = ((V * rows, 1), (rows, vec * rows, V * rows))
        return cute.make_layout(shape, stride=stride)

    def _smem_bytes(self) -> int:
        b = self.rows * self.n_entries * self.W * LANES * 4 if self.W > 1 else 0
        if self.C > 1:
            b += self.rows * self.n_entries_cta * LANES * 4
        if self.cfg.stage == "smem":
            b += self.cfg.smem_tile_bytes(self.N, self.elem_bytes) + 128
        return b

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mLse: cute.Tensor,
        mTok: cute.Tensor,
        mLogp: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        M: Int64,
        N: Int32,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        k = self.kernel(
            mX,
            mY,
            mLse,
            mTok,
            mLogp,
            mTemp,
            c_const,
            M,
            N,
            K,
            enable_pdl,
            self._tv_layout(self.vec),
            self._tv_layout(self.vec_out),
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
        mY: cute.Tensor,
        mLse: cute.Tensor,
        mTok: cute.Tensor,
        mLogp: cute.Tensor,
        mTemp: cute.Tensor,
        c_const: Float32,
        M: Int64,
        N: Int32,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tv_layout_out: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        W, rows = self.W, self.rows
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        # ---- unit coordinates: (row group = cluster, CTA in cluster, row in CTA, warp in row, lane) -----------------
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
        row_c = _select(
            row_ok, row, M - 1
        )  # an in-bounds row index for the per-row scalar loads of tail rows

        # ---- smem: the staged tile, the cross-warp exchange, the cross-CTA exchange ----------------------------------
        smem = cutlass.utils.SmemAllocator()
        xbuf = cbuf = smem_x = None
        if cutlass.const_expr(self.cfg.stage == "smem"):
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
        if cutlass.const_expr(self.C > 1):
            cbuf = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows, self.n_entries_cta, LANES)),
                byte_alignment=16,
            )

        # ---- tiled copies ----------------------------------------------------------------------------------------
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

        # ---- the scale ---------------------------------------------------------------------------------------------
        if cutlass.const_expr(self.temp_rows):
            c = scale(fp, mTemp[row_c])
        else:
            c = c_const

        # ===== pass 1: the row maximum (driven by the plan) =============================================================
        loader = _RowLoader(
            self,
            self.plan_max,
            coords,
            mX,
            idX,
            tiler_mn,
            rg,
            M,
            N,
            thr_x,
            atom_x,
            smem_x,
            thr_async,
            atom_async,
            c_rank * self.cfg.blocks_per_cta(self.N),
        )
        peers = _CutePeers(xbuf, row_in_cta, w, lane_idx, cbuf, c_rank)
        state, start = canon.init_state(self.plan_max, fp, loader.max_leaf)
        m = canon.run(self.plan_max, state, start, peers, fp)
        nm = fp.neg(m)

        # ===== pass 2: the sum of exponentials (the same plan shape) ======================================================
        loader.nm, loader.c = nm, c
        state, start = canon.init_state(self.plan_sum, fp, loader.sum_leaf)
        s = canon.run(self.plan_sum, state, start, peers, fp)

        # ===== the per-row scalars ===========================================================================================
        lg2s = None
        if cutlass.const_expr(self.mode.lse or self.mode.logp or self.mode.logprobs):
            lg2s = lg2(fp, s)
        if cutlass.const_expr(self.mode.lse):
            lse_v = lse_nat(fp, m, c, lg2s)
            if t == 0:
                if w == 0:
                    if c_rank == 0:
                        if row_ok:
                            mLse[row] = lse_v
        if cutlass.const_expr(self.mode.logp):
            nlg2s = fp.neg(lg2s)
            if cutlass.const_expr(self.k_slots == 1):
                # one id per row: every thread forms it, thread 0 of the row's first CTA stores it
                tok = mTok[row_c, 0]
                tok_ok = (tok >= 0) & (tok < N.to(Int64))
                tok_c = _select(tok_ok, tok, Int64(0))
                xt = mX[row_c, tok_c].to(Float32)
                lp = logp(fp, shifted(fp, xt, nm, c), nlg2s)
                lp = _select(tok_ok, lp, Float32(NAN))
                if t == 0:
                    if w == 0:
                        if c_rank == 0:
                            if row_ok:
                                mLogp[row, 0] = lp
            else:
                # K ids per row: lane `t` of every warp forms ids t, t + 32, ... (the same chain per id; the gathers of
                # a warp's lanes are independent loads), warp 0 of the row's first CTA stores the ids below `K`
                for it in cutlass.range_constexpr(self.k_slots // LANES):
                    kk = Int32(it * LANES) + t
                    k_ok = kk < K
                    kc = _select(k_ok, kk, Int32(0))
                    tok = mTok[row_c, kc]
                    tok_ok = (tok >= 0) & (tok < N.to(Int64))
                    tok_c = _select(tok_ok, tok, Int64(0))
                    xt = mX[row_c, tok_c].to(Float32)
                    lp = logp(fp, shifted(fp, xt, nm, c), nlg2s)
                    lp = _select(tok_ok, lp, Float32(NAN))
                    if w == 0:
                        if c_rank == 0:
                            if row_ok:
                                if k_ok:
                                    mLogp[row, kc] = lp

        # ===== pass 3: the row outputs ======================================================================================
        if cutlass.const_expr(self.mode.row_output):
            thr_y, atom_y = _tiled_copy(
                mY, self.vec_out * mY.element_type.width, tv_layout_out, tiler_mn, vtid
            )
            inv_s = inv_sum(fp, s) if cutlass.const_expr(self.mode.probs) else None
            nlg = fp.neg(lg2s) if cutlass.const_expr(self.mode.logprobs) else None
            for kk in cutlass.range_constexpr(self.kpw):
                kg = loader.k_global(kk)
                cY = cute.local_tile(idX, tiler_mn, (rg, kg))
                pred_y = _predicate(thr_y.partition_S(cY), M, N)
                gY = thr_y.partition_S(cute.local_tile(mY, tiler_mn, (rg, kg)))
                fy = cute.make_fragment_like(gY)
                _row_outputs(
                    self, loader, kk, fy, nm, c, inv_s, nlg, mY.element_type, fp
                )
                cute.copy(atom_y, fy, gY, pred=pred_y)

        peers.finish()
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _row_outputs(k, loader, kk, fy, nm, c, inv_s, nlg, out_dtype, fp):
    """The output elements of block ``kk`` (plain Python at trace time): ``z`` recomputed identically from the kept /
    re-read fragment, then the probability (converted to the output dtype) or the log-probability."""
    for e in range(V):
        z = shifted(fp, loader.x(kk, e), nm, c)
        if k.mode.probs:
            fy[e] = prob(fp, fp.ex2(z), inv_s).to(out_dtype)
        else:
            fy[e] = logp(fp, z, nlg)


def kernel_name(
    mode: Mode,
    dtype_str: str,
    Kp: int,
    cfg: LSEConfig,
    temp_rows: bool,
    enable_pdl: bool,
    k_slots: int = 1,
) -> str:
    return jit.sanitize_symbol_name(
        f"lse_{mode.name}_{dtype_str}_Kp{Kp}_{cfg.describe()}_{'trow' if temp_rows else 'tconst'}_pdl{int(enable_pdl)}"
        + (f"_k{k_slots}" if k_slots != 1 else "")
    )


def k_slots_for(K: int) -> int:
    """The compile-time id slots per row serving ``K`` ids: 1, else the multiple of 32 at or above ``K``."""
    return 1 if K <= 1 else LANES * (-(-int(K) // LANES))


def get_compiled_kernel(
    dtype_str: str,
    N: int,
    cfg: LSEConfig,
    mode: Mode,
    temp_rows: bool,
    enable_pdl: bool,
    k_slots: int = 1,
):
    """The compiled kernel serving vocabulary ``N``: one per (dtype, Kp bucket, config, mode, temperature form, pdl,
    id slots); ``N``, ``M``, ``K`` and the row stride are runtime arguments."""
    from .schedule import padded_blocks

    return _compile_bucket(
        dtype_str, padded_blocks(N), cfg, mode, temp_rows, enable_pdl, k_slots
    )


@functools.cache
def _compile_bucket(
    dtype_str: str,
    Kp: int,
    cfg: LSEConfig,
    mode: Mode,
    temp_rows: bool,
    enable_pdl: bool,
    k_slots: int = 1,
):
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = kernel_name(mode, dtype_str, Kp, cfg, temp_rows, enable_pdl, k_slots)

    def compile_fn():
        obj = LSERowKernel(dtype, Kp, cfg, mode, temp_rows, k_slots)
        sym_m, sym_n = cute.sym_int(64), cute.sym_int(32)

        def rows(dt, vec, shape):
            return cute.runtime.make_fake_tensor(
                dt,
                shape,
                (cute.sym_int64(divisibility=vec), 1),
                assumed_align=max(vec * (dt.width // 8), 1),
            )

        def vector(dt, align):
            # every per-row vector has its own size symbol: the FFI entry then checks ndim / dtype only, so the
            # unused ones may be small dummies
            return cute.runtime.make_fake_tensor(
                dt, (cute.sym_int(32),), (1,), assumed_align=align
            )

        # the row output shares the input's shape symbols when it is written; otherwise it is a dummy of any shape
        y_shape = (
            (sym_m, sym_n) if mode.row_output else (cute.sym_int(64), cute.sym_int(32))
        )

        def ids(dt, align):
            # the token ids and their log-probabilities: (rows, K) with their own size symbols (dummies when unused)
            return cute.runtime.make_fake_tensor(
                dt,
                (cute.sym_int(64), cute.sym_int(32)),
                (cute.sym_int64(divisibility=1), 1),
                assumed_align=align,
            )

        args = (
            rows(dtype, cfg.vec_elems, (sym_m, sym_n)),
            rows(obj.out_dtype, obj.vec_out, y_shape),
            vector(Float32, 4),
            ids(Int64, 8),
            ids(Float32, 4),
            vector(Float32, 4),
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            obj,
            *args,
            Float32(1.0),
            Int64(1),
            Int32(1),
            Int32(1),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(LSE_MODULE, name, compile_fn)
