# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing and the permutation plan in one launch for small token counts.

One CTA of 32 warps: phase A routes the tokens with the route kernel's own per-token code (:func:`.route.route_token`: one warp
per token, the tokens of a warp ``t = warp, warp + 32, ...``), writing the slots to global memory and to shared memory; phase B
builds the histogram, the group tables and the positions from the shared copy with the two-launch plan's ranking (per-warp
``match.sync`` ranks over 32-slot chunks, an exclusive prefix over the warps per expert: ``PlanBlocksKernel`` /
``PlanFinishKernel``'s integer arithmetic) and writes ``counts``, ``dest``, ``src``. The routing arithmetic is the route kernel's
(the same traced code); the plan's outputs are integers defined by the routing alone: a placement of ``route`` + ``permute_plan``
(the same bits), one launch instead of two. Limits: ``T k <= ROUTE_PLAN_MAX_SLOTS``, ``E <= MAX_E``; the operators take the fused
form up to ``ROUTE_PLAN_MAX_TOKENS`` tokens (one CTA's warps route the tokens serially beyond a few per warp, where the two-launch
form's parallel plan wins).

``pad_align > 1`` adds the padded packings' second position table (the block-scaled backends: the A rows dense, the scale
factors and row scales at the positions of the groups padded to ``pad_align``-row atoms): ``dest_pad[i]`` the padded position
of expanded row ``i`` and ``pos_pad[p]`` the padded position of dense row ``p``, two more writes in the position phase from a
second pair of group tables (the integers of the two-launch dual plan).
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32, Uint32

from ...batch_invariant import canon, jit
from ...batch_invariant.fp_interface.cute import get_cutlass_dtype
from . import spec as S
from .blocks.scheduler import MAX_E, build_group_tables
from .permute import LANES, MODULE
from .route import route_token

THREADS = 1024  # 32 warps route the tokens
WARPS_RP = THREADS // LANES
RANK_WARPS = 8  # warps that rank the slots (the two-launch plan's per-block algorithm)
IDS_PER_THREAD = 8
ROUTE_PLAN_MAX_SLOTS = (
    RANK_WARPS * IDS_PER_THREAD * LANES
)  # 2048 expanded rows per launch (the kernel's bound)
ROUTE_PLAN_MAX_TOKENS = (
    16  # the dispatch rule: the fused form pays up to this token count
)


class RoutePlanKernel:
    """``pad_align``: 1 for the dense plan alone; > 1 also fills ``dest_pad`` / ``pos_pad`` (the positions with the groups
    padded to ``pad_align`` rows)."""

    def __init__(self, Ep: int, k: int, method: S.RoutingMethod, pad_align: int = 1):
        assert Ep % LANES == 0 and Ep // LANES in (1, 2, 4, 8, 16, 32) and 1 <= k <= 16
        assert pad_align >= 1, pad_align
        self.Ep, self.P, self.k, self.method = Ep, Ep // LANES, k, method
        self.pad_align = pad_align
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
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        T: Int32,
        E: Int32,
        stream,
    ):
        self.kernel(
            mL, mBias, mScale, mExp, mW, mCounts, mDest, mSrc, mDestPad, mPosPad, T, E
        ).launch(grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        mL: cute.Tensor,
        mBias: cute.Tensor,
        mScale: cute.Tensor,
        mExp: cute.Tensor,
        mW: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        T: Int32,
        E: Int32,
    ):
        P, k, meth = self.P, self.k, self.method
        padded = self.pad_align > 1
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // LANES
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        smem = cutlass.utils.SmemAllocator()
        s_exp = smem.allocate_tensor(
            Int32, cute.make_layout((ROUTE_PLAN_MAX_SLOTS,)), byte_alignment=16
        )
        hist = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E,)), byte_alignment=16
        )
        s_off = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_pre = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_off_pad = s_pre_pad = None
        if cutlass.const_expr(padded):  # the padded packing's group tables
            s_off_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
            s_pre_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
        wcnt = smem.allocate_tensor(
            Int16, cute.make_layout((RANK_WARPS * MAX_E,)), byte_alignment=16
        )
        for j in cutlass.range(tidx, RANK_WARPS * MAX_E, THREADS):
            wcnt[j] = Int16(0)
        # phase A: the tokens of this warp, the route kernel's per-token code
        out_e = cute.make_rmem_tensor((k,), Int32)
        out_w = cute.make_rmem_tensor((k,), Float32)
        for t in cutlass.range(warp, T, WARPS_RP):
            route_token(
                P, k, meth, self.plan, mL, mBias, mScale, t, lane, E, out_e, out_w
            )
            for r in cutlass.range_constexpr(k):
                if lane == Int32(r):
                    mExp[t, r] = out_e[r]
                    mW[t, r] = out_w[r]
                    s_exp[t * Int32(k) + Int32(r)] = out_e[r]
        cute.arch.barrier()
        n = T * Int32(k)
        # phase B1: per-warp ranks of the slots in slot order (rank warp w owns slots [w 256, (w + 1) 256) in 8 chunks of 32)
        es = cute.make_rmem_tensor((IDS_PER_THREAD,), Int32)
        rs = cute.make_rmem_tensor((IDS_PER_THREAD,), Int32)
        if warp < Int32(RANK_WARPS):
            base_w = warp * Int32(IDS_PER_THREAD * LANES)
            for c in cutlass.range_constexpr(IDS_PER_THREAD):
                i = base_w + Int32(c * LANES) + lane
                valid = i < n
                e = Int32(-1)
                if valid:
                    e = s_exp[i]
                m = cute.arch.match_sync(Uint32(0xFFFFFFFF), e)
                intra = Int32(cute.arch.popc(m & cute.arch.lanemask_lt()))
                r = Int32(0)
                if valid:
                    r = wcnt[warp * MAX_E + e].to(Int32) + intra
                cute.arch.sync_warp()
                if valid & (intra == Int32(0)):
                    wcnt[warp * MAX_E + e] = (
                        wcnt[warp * MAX_E + e].to(Int32) + Int32(cute.arch.popc(m))
                    ).to(Int16)
                cute.arch.sync_warp()
                es[c] = e
                rs[c] = r
        cute.arch.barrier()
        # phase B2: per expert, the exclusive prefix over the rank warps (in place) and the total = the histogram
        for e in cutlass.range(tidx, E, THREADS):
            run = Int32(0)
            for w in cutlass.range_constexpr(RANK_WARPS):
                v = wcnt[w * MAX_E + e].to(Int32)
                wcnt[w * MAX_E + e] = run.to(Int16)
                run = run + v
            hist[e] = run
            mCounts[e] = run
        cute.arch.barrier()
        if warp == 0:
            build_group_tables(hist, E, 1, s_off, s_pre, 1)
            if cutlass.const_expr(padded):
                build_group_tables(hist, E, 1, s_off_pad, s_pre_pad, self.pad_align)
        cute.arch.barrier()
        # phase B3: the positions (and, padded, the second table: the same rank from the padded group offsets)
        if warp < Int32(RANK_WARPS):
            base_w = warp * Int32(IDS_PER_THREAD * LANES)
            for c in cutlass.range_constexpr(IDS_PER_THREAD):
                i = base_w + Int32(c * LANES) + lane
                if i < n:
                    e = es[c]
                    rank = wcnt[warp * MAX_E + e].to(Int32) + rs[c]
                    p = s_off[e] + rank
                    mDest[i] = p
                    mSrc[p] = i
                    if cutlass.const_expr(padded):
                        pp = s_off_pad[e] + rank
                        mDestPad[i] = pp
                        mPosPad[p] = pp
        cute.arch.griddepcontrol_launch_dependents()


def route_plan_kernel_name(
    Ep: int, k: int, method_name: str, in_fmt: str, pad_align: int = 1
) -> str:
    return jit.sanitize_symbol_name(
        f"moe_route_plan_Ep{Ep}_k{k}_{method_name}_{in_fmt}"
        + (f"_p{pad_align}" if pad_align > 1 else "")
    )


@functools.cache
def get_compiled_route_plan(
    Ep: int, k: int, method: S.RoutingMethod, in_fmt: str = "f32", pad_align: int = 1
) -> Any:
    """``kernel(logits [T, E], bias [E] f32, scale [1] f32, experts [T, k] i32, weights [T, k] f32, counts [E] i32,
    dest [T k] i32, src [T k] i32, dest_pad [T k] i32, pos_pad [T k] i32, T, E)``; ``Ep`` the expert bucket, ``T`` and
    ``E`` runtime; ``dest_pad`` / ``pos_pad`` are written for ``pad_align > 1`` only (any int32 tensors otherwise)."""
    in_dt = get_cutlass_dtype(in_fmt)

    def compile_fn():
        sym_t, sym_e, sym_n = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
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
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        src_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        dp_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        pp_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            RoutePlanKernel(Ep, k, method, pad_align),
            l_fake,
            b_fake,
            s_fake,
            e_fake,
            w_fake,
            c_fake,
            d_fake,
            src_fake,
            dp_fake,
            pp_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE,
        route_plan_kernel_name(Ep, k, method.name, in_fmt, pad_align),
        compile_fn,
    )
