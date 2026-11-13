# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Routing kernel: one warp per token.

Lane ``l`` owns experts ``[l P, (l+1) P)`` with ``P = Ep / 32`` (the placement ``expert: [(P, reg), (32, lane)]`` of the
softmax sum; experts beyond ``E`` are absent and padded with ``+0``, which is exact for the all-positive summands). The
row maximum is a warp max (exact in every order; the sign of a zero maximum never reaches the bits), the softmax sum
runs the engine's plan with lane shuffles, the top-k selection is k rounds of a warp argmax under the strict order
``(score desc, index asc)`` (equal to the rank definition the CPU uses), and the weights are ``spec.slot_weights``
through ``CuteF32``. The logit methods select on the logits themselves (the weights the logits or their sigmoid); the
group-limited methods run the group stage before the selection (two argmax rounds per group, the kept groups chosen on
warp-uniform scores, the other groups' experts masked out). All arithmetic through ``CuteF32``; comparisons are exact. A
NaN logit is the spec's row rule (:mod:`.spec`): the warp max skips it (``max.f32``), its softmax term is ``+0`` and its
selection score NaN (a select, no arithmetic), so it is never selected. A slot no selectable expert reaches (fewer than
``k`` experts with a non-NaN score) takes the lowest expert index the row has not taken, with a NaN weight: the row's
output is NaN and every id stays in ``[0, E)``.
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ...batch_invariant import canon, jit
from ...batch_invariant.fp_interface.cute import get_cutlass_dtype
from ...batch_invariant.quantization.cute_ext import CuteF32
from . import spec as S
from . import pdl

LANES = 32
WARPS = 4


def _select_f32(cond, a, b):
    """``a`` where ``cond`` else ``b`` (``arith.select``, one ``selp.f32``: no arithmetic)."""
    from cutlass._mlir.dialects import arith

    return Float32(
        arith.select(cond.ir_value(), Float32(a).ir_value(), Float32(b).ir_value())
    )


class _LanePeers(canon.Peers):
    def shuffle_xor(self, key, value, mask):
        if value is None:
            return None
        return cute.arch.shuffle_sync_bfly(value, offset=mask)

    def exchange(
        self, step, state
    ):  # pragma: no cover - the placement has no warp-level factor
        raise NotImplementedError


class RouteKernel:
    """One warp per token: the scores over the expert bucket (``P`` experts per lane), the strict top-k selection and the
    method's weights, written in slot order."""

    def __init__(self, Ep: int, k: int, method: S.RoutingMethod):
        assert Ep % LANES == 0 and Ep // LANES in (1, 2, 4, 8, 16, 32) and 1 <= k <= 16
        self.Ep, self.P, self.k, self.method = Ep, Ep // LANES, k, method
        red = S.softmax_reduction(Ep, lambda fp, e: None)
        self.plan = canon.plan(
            red, canon.Placement.make({"expert": [(self.P, "reg"), (LANES, "lane")]})
        )

    @cute.jit
    def __call__(
        self,
        mL: cute.Tensor,
        mBias: cute.Tensor,
        mScale: cute.Tensor,
        mExp: cute.Tensor,
        mW: cute.Tensor,
        T: Int32,
        E: Int32,
        stream,
    ):
        self.kernel(mL, mBias, mScale, mExp, mW, T, E).launch(
            grid=[cute.ceil_div(T, WARPS), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mL: cute.Tensor,
        mBias: cute.Tensor,
        mScale: cute.Tensor,
        mExp: cute.Tensor,
        mW: cute.Tensor,
        T: Int32,
        E: Int32,
    ):
        k = self.k
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()  # programmatic dependent launch: the logits may come from the previous kernel
        t_raw = bidx * WARPS + tidx // LANES
        valid = t_raw < T
        t = cutlass.min(
            t_raw, T - Int32(1)
        )  # out-of-range warps recompute a valid token and store nothing
        out_e = cute.make_rmem_tensor((k,), Int32)
        out_w = cute.make_rmem_tensor((k,), Float32)
        route_token(
            self.P,
            k,
            self.method,
            self.plan,
            mL,
            mBias,
            mScale,
            t,
            lane,
            E,
            out_e,
            out_w,
        )
        for r in cutlass.range_constexpr(k):
            if (lane == Int32(r)) & valid:
                mExp[t, r] = out_e[r]
                mW[t, r] = out_w[r]
        pdl.trigger_one_wave()


@cute.jit
def route_token(
    P: cutlass.Constexpr,
    k: cutlass.Constexpr,
    meth: cutlass.Constexpr,
    plan: cutlass.Constexpr,
    mL: cute.Tensor,
    mBias: cute.Tensor,
    mScale: cute.Tensor,
    t: Int32,
    lane: Int32,
    E: Int32,
    out_e: cute.Tensor,
    out_w: cute.Tensor,
):
    """One token's routing by one warp (the class's routing term): the warp-uniform results land in ``out_e[r]`` /
    ``out_w[r]`` for the ``k`` slots. Shared by the route kernel and the fused route + plan kernel: the same traced code,
    the same bits. A NaN logit is skipped by the row maximum, its softmax term is ``+0`` and its selection score NaN
    (``spec``'s row rule), so it is never selected (no comparison holds); when fewer than ``k`` experts are selectable,
    each remaining slot takes the lowest expert index the row has not taken with a NaN weight (the row's output is NaN;
    every id stays in ``[0, E)``)."""
    neg_inf = CuteF32.const(float("-inf"))
    nan = CuteF32.const(float("nan"))
    # the lane's experts: logits (absent -> -inf), selection scores q, weights
    logits = []
    for i in cutlass.range_constexpr(P):
        e = lane * P + i
        v = neg_inf
        if e < E:
            v = mL[t, e].to(Float32)
        logits.append(v)
    isnan = [l != l for l in logits]
    if cutlass.const_expr(meth.score == "sigmoid"):
        wgt = [S.sigmoid_term(CuteF32, l) for l in logits]
        q = []
        for i in cutlass.range_constexpr(P):
            e = lane * P + i
            qi = wgt[i]
            if cutlass.const_expr(meth.bias):
                b = CuteF32.const(0.0)
                if e < E:
                    b = mBias[e]
                qi = CuteF32.add(wgt[i], b)
            qi = _select_f32(
                isnan[i], nan, qi
            )  # the declared exp of a NaN is the clamp's value: the score is set NaN
            if e >= E:
                qi = neg_inf
            q.append(qi)
    elif cutlass.const_expr(meth.score == "logit"):
        # the plain top-k of the logits (no normalisation): the weights are the logits themselves or their sigmoid
        if cutlass.const_expr(meth.weight_kind == "sigmoid"):
            wgt = [S.sigmoid_term(CuteF32, l) for l in logits]
        else:
            wgt = list(logits)
        q = logits
    else:
        m = logits[0]
        for i in cutlass.range_constexpr(1, P):
            m = CuteF32.fmax(m, logits[i])
        for sh in cutlass.range_constexpr(5):
            m = CuteF32.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << sh))
        z = []
        for i in cutlass.range_constexpr(P):
            e = lane * P + i
            zi = CuteF32.const(0.0)  # absent: +0 (identity for the all-positive sum)
            if e < E:
                zi = S.exp_term(CuteF32, CuteF32.add(logits[i], CuteF32.neg(m)))
            zi = _select_f32(
                isnan[i], CuteF32.const(0.0), zi
            )  # a NaN logit: +0, nothing in the sum
            z.append(zi)
        state, start = canon.init_state(
            plan, CuteF32, lambda key: z[key[0] if key else 0]
        )  # P = 1: no register factor
        Ssum = canon.run(plan, state, start, _LanePeers(), CuteF32)
        wgt = [CuteF32.div(zi, Ssum) for zi in z]
        q = logits
    if cutlass.const_expr(meth.n_group > 1):
        # the group-limited stage (spec.group_scores / select_topk_grouped): per group two rounds of warp argmax under
        # (q desc, index asc) over the group's experts give its two largest scores, add of them is the group score (NaN when
        # a round finds nothing: a NaN score is never picked, an absent expert belongs to no group); the topk_group best
        # groups are kept in the same strict order over the warp-uniform group scores (the lowest untaken groups fill when
        # fewer are selectable); an expert outside a kept group leaves the selection with a NaN score
        n_group, topk_group = meth.n_group, meth.topk_group
        G = E // Int32(n_group)
        grp = [Int32(lane * P + i) // G for i in range(P)]
        gsc = []
        for j in cutlass.range_constexpr(n_group):
            bv1 = neg_inf
            bi1 = Int32(0x7FFFFFFF)
            for i in cutlass.range_constexpr(P):
                e = Int32(lane * P + i)
                inside = grp[i] == Int32(j)
                better = (q[i] > bv1) | ((q[i] == bv1) & (e < bi1))
                if inside & better:
                    bv1 = q[i]
                    bi1 = e
            for sh in cutlass.range_constexpr(5):
                ov = cute.arch.shuffle_sync_bfly(bv1, offset=1 << sh)
                oi = cute.arch.shuffle_sync_bfly(bi1, offset=1 << sh)
                take = (ov > bv1) | ((ov == bv1) & (oi < bi1))
                if take:
                    bv1 = ov
                    bi1 = oi
            bv2 = neg_inf
            bi2 = Int32(0x7FFFFFFF)
            for i in cutlass.range_constexpr(P):
                e = Int32(lane * P + i)
                inside = (grp[i] == Int32(j)) & (e != bi1)
                better = (q[i] > bv2) | ((q[i] == bv2) & (e < bi2))
                if inside & better:
                    bv2 = q[i]
                    bi2 = e
            for sh in cutlass.range_constexpr(5):
                ov = cute.arch.shuffle_sync_bfly(bv2, offset=1 << sh)
                oi = cute.arch.shuffle_sync_bfly(bi2, offset=1 << sh)
                take = (ov > bv2) | ((ov == bv2) & (oi < bi2))
                if take:
                    bv2 = ov
                    bi2 = oi
            g = S.group_score_term(CuteF32, bv1, bv2)
            if bi2 >= E:
                g = nan
            gsc.append(g)
        gtaken = Int32(0)
        for _r in cutlass.range_constexpr(topk_group):
            bgv = neg_inf
            bgj = Int32(0x7FFFFFFF)
            for j in cutlass.range_constexpr(n_group):
                free = ((gtaken >> j) & Int32(1)) == Int32(0)
                better = (gsc[j] > bgv) | ((gsc[j] == bgv) & (Int32(j) < bgj))
                if free & better:
                    bgv = gsc[j]
                    bgj = Int32(j)
            if bgj >= Int32(n_group):
                cand = Int32(0x7FFFFFFF)
                for j in cutlass.range_constexpr(n_group):
                    free = ((gtaken >> j) & Int32(1)) == Int32(0)
                    if free & (Int32(j) < cand):
                        cand = Int32(j)
                bgj = cand
            gtaken = gtaken | (Int32(1) << bgj)
        q2 = []
        for i in cutlass.range_constexpr(P):
            qi = q[i]
            drop = (grp[i] >= Int32(n_group)) | (
                ((gtaken >> cutlass.min(grp[i], Int32(31))) & Int32(1)) == Int32(0)
            )
            if drop:
                qi = nan
            q2.append(qi)
        q = q2
    # k rounds of warp argmax under (q desc, index asc)
    taken = Int32(0)
    sel_e = []
    sel_w = []
    for _r in cutlass.range_constexpr(k):
        bv = neg_inf
        bi = Int32(0x7FFFFFFF)
        for i in cutlass.range_constexpr(P):
            e = Int32(lane * P + i)
            free = ((taken >> i) & Int32(1)) == Int32(0)
            better = (q[i] > bv) | ((q[i] == bv) & (e < bi))
            if free & better:
                bv = q[i]
                bi = e
        for sh in cutlass.range_constexpr(5):
            ov = cute.arch.shuffle_sync_bfly(bv, offset=1 << sh)
            oi = cute.arch.shuffle_sync_bfly(bi, offset=1 << sh)
            take = (ov > bv) | ((ov == bv) & (oi < bi))
            if take:
                bv = ov
                bi = oi
        # no selectable expert left (fewer than k experts with a non-NaN score: the round reached an absent expert of the
        # bucket or the sentinel): the slot takes the lowest expert index the row has not taken, with a NaN weight, so the
        # row's output is NaN and no id outside [0, E) leaves the kernel (a per-row rule: batch-invariant); a row with k
        # selectable experts never takes the branch (bi is warp-uniform after the reduction)
        no_pick = bi >= E
        if no_pick:
            cand = Int32(0x7FFFFFFF)
            for i in cutlass.range_constexpr(P):
                e = Int32(lane * P + i)
                free = ((taken >> i) & Int32(1)) == Int32(0)
                if free & (e < E) & (e < cand):
                    cand = e
            for sh in cutlass.range_constexpr(5):
                oc = cute.arch.shuffle_sync_bfly(cand, offset=1 << sh)
                cand = cutlass.min(cand, oc)
            bi = cand
        owner = bi // Int32(P)
        slot = bi % Int32(P)
        wv = CuteF32.const(0.0)
        for i in cutlass.range_constexpr(P):
            if slot == Int32(i):
                wv = wgt[i]
        wv = cute.arch.shuffle_sync(wv, offset=owner)
        if no_pick:
            wv = CuteF32.const(float("nan"))
        if lane == owner:
            taken = taken | (Int32(1) << slot)
        sel_e.append(bi)
        sel_w.append(wv)
    scale = mScale[0] if cutlass.const_expr(meth.scaled) else None
    ws = S.slot_weights(CuteF32, sel_w, list(range(k)), meth, scale)
    for r in cutlass.range_constexpr(k):
        out_e[r] = sel_e[r]
        out_w[r] = ws[r]


def route_kernel_name(Ep: int, k: int, method_name: str, in_fmt: str) -> str:
    return jit.sanitize_symbol_name(f"route_Ep{Ep}_k{k}_{method_name}_{in_fmt}")


@functools.cache
def get_compiled_route(
    Ep: int, k: int, method: S.RoutingMethod, in_fmt: str = "f32"
) -> Any:
    """``kernel(logits [T, E], bias [E] f32, scale [1] f32, experts [T, k] i32, weights [T, k] f32, T, E)``; ``Ep`` = the
    expert bucket (``max(32, pow2ceil(E))``), ``T`` and ``E`` runtime."""
    in_dt = get_cutlass_dtype(in_fmt)
    name = route_kernel_name(Ep, k, method.name, in_fmt)

    def compile_fn():
        sym_t, sym_e = cute.sym_int(32), cute.sym_int(32)
        l_fake = cute.runtime.make_fake_tensor(
            in_dt, (sym_t, sym_e), (cute.sym_int64(), 1), assumed_align=4
        )
        b_fake = cute.runtime.make_fake_tensor(Float32, (sym_e,), (1,), assumed_align=4)
        s_fake = cute.runtime.make_fake_tensor(Float32, (1,), (1,), assumed_align=4)
        e_fake = cute.runtime.make_fake_tensor(
            Int32, (sym_t, k), (k, 1), assumed_align=4
        )
        w_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_t, k), (k, 1), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            RouteKernel(Ep, k, method),
            l_fake,
            b_fake,
            s_fake,
            e_fake,
            w_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load("batch_invariant_moe", name, compile_fn)
