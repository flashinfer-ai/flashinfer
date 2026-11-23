# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, SIM102, B905
"""FROST MTP verify GDN block for Qwen3.8-27B (TP1): speculative conv update + SiLU
(CONV, else on an already convolved input), BA projection, gated delta rule over
each request's tokens, gated RMSNorm and static FP8 output quantization in one
launch.

The speculative conv takes ``num_accepted_tokens``: history
``conv[slot, :, a-1 .. a+1]`` (a = accepted), new state
``[old[a], old[a+1], x_0 .. x_{T-1}]`` written in place to slot
``state_indices[n, 0]``. Q/K channels are shared by the three V heads of a K head:
every CTA reads the old history and the last of the 3 x CTAS readers (per-request
ticket, as in the decode kernel) writes the shifted Q/K state.

State semantics (the per-token state layout): a request starts from the state
stored for its last accepted token (``state_indices[n, accepted - 1]``), writes
the state after every token t to ``state_indices[n, t]`` (slots <= 0 are
skipped), and a request without a source state outputs zeros.

JOURNAL (not exposed by the FlashInfer API) keeps one state per request instead
of one per token: the request's base slot ``state_indices[n, 0]`` holds the state
at the start of the previous verify, and slot ``state_indices[n, 1]`` that
verify's per-token update terms (decay, the per-row innovation, the normalized k;
region A or B, alternating). The next verify replays the ``accepted`` tokens with
the forward's own arithmetic (bit-identical), stores the result as the new base
and journals its tokens. ``jmode[slot, head]`` says which region is live (0: none,
the per-token layout above). A journal materialize step restores the per-token
layout for readers of that layout.

Grid: 48 V heads x CTAS x requests, 256 threads; the CTAS CTAs of a head form a
cluster and split its 128 state rows. Every token's q/k/v/z slice is staged in
shared memory at entry, each half-warp keeps its state rows in registers across
the request's tokens, and the per-token RMSNorm partial sums are exchanged once,
after the last token. Rounding follows the unfused op chain: BA in BF16, BF16
core output, BF16 normed value, then FP32-reciprocal FP8 quantization.
"""

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.cute_dsl.fp4_common import (
    bfloat2_to_float2_scaled,
    cvt_f32x2_to_bfloat2,
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_v4_u32,
)

from .gdn import (
    as_bits,
    as_float,
    cluster_sync,
    peer_float,
    quant_fp8,
    reader_done,
    stock_conv_silu,
    warp_sum,
)


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def push_peer_f32(
    saddr: cutlass.Int32,
    value: cutlass.Float32,
    mbar: cutlass.Int32,
    rank: cutlass.Int32,
    *,
    loc=None,
    ip=None,
):
    """Store one FP32 at this CTA's shared address ``saddr`` mapped into cluster peer ``rank``, crediting 4 bytes to
    the peer's mbarrier at ``mbar`` (also a local address mapped the same way)."""
    llvm.inline_asm(
        None,
        [
            saddr.ir_value(loc=loc, ip=ip),
            value.bitcast(cutlass.Int32).ir_value(loc=loc, ip=ip),
            mbar.ir_value(loc=loc, ip=ip),
            rank.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 ra, rb; mapa.shared::cluster.u32 ra, $0, $3; mapa.shared::cluster.u32 rb, $2, $3; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $1, [rb]; }",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.wait.acquire.aligned;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_v4(
    addr: cutlass.Int32,
    a: cutlass.Uint32,
    b: cutlass.Uint32,
    c: cutlass.Uint32,
    d: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [x.ir_value(loc=loc, ip=ip) for x in (addr, a, b, c, d)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# The state update h = fma(innovation, k, h * decay) in explicitly rounded steps, so that neither LLVM nor ptxas
# picks a different contraction in the forward (where h * decay also feeds the dot product) and in the journal
# replay: both layouts then hold bit-identical states.
@dsl_user_op
def mul_rn(
    a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_rn(
    a: cutlass.Float32, b: cutlass.Float32, c: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                a.ir_value(loc=loc, ip=ip),
                b.ir_value(loc=loc, ip=ip),
                c.ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_async_smem(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_store(
    gaddr: cutlass.Int64,
    saddr: cutlass.Int32,
    nbytes: cutlass.Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [x.ir_value(loc=loc, ip=ip) for x in (gaddr, saddr, nbytes)],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_commit(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def bulk_wait_read(pending: int):
    """Wait until at most ``pending`` committed bulk groups still read shared memory."""

    @dsl_user_op
    def wait(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            f"cp.async.bulk.wait_group.read {pending};",
            "~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    wait()


@dsl_user_op
def bulk_wait_all(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.wait_group 0;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def conv_stage_unit(
    win: cute.Tensor, wts: cute.Tensor, stage: cute.Tensor, u: cutlass.Int32
):
    """One (8-channel group, token) unit of the speculative conv + SiLU, staged as BF16 pairs."""
    for i in cutlass.range_constexpr(4):
        res = cute.make_rmem_tensor((2,), cutlass.Float32)
        for hlf in cutlass.range_constexpr(2):
            ch = 2 * i + hlf
            total = cutlass.Float32(0)
            for e in cutlass.range_constexpr(4):
                x0, x1 = bfloat2_to_float2_scaled(win[e, i], cutlass.Float32(1))
                xv = x1 if hlf == 1 else x0
                # Tap e of channel ch is element ch * 4 + e of the group's 64-byte weight run.
                w0, w1 = bfloat2_to_float2_scaled(
                    wts[ch // 2, (ch % 2) * 2 + e // 2], cutlass.Float32(1)
                )
                wv = w1 if e % 2 == 1 else w0
                total = total + (xv * wv).to(cutlass.BFloat16).to(cutlass.Float32)
            res[hlf] = stock_conv_silu(total)
        stage[u * 4 + i] = cvt_f32x2_to_bfloat2(res[0], res[1])


THREADS = 256
JENTRY = 260  # floats per (head, token) journal entry: 128 innovations, 128 k, the decay, padding to 16 B
# Head h's journal lives inside head h's own 128 x 128 block of slot 1 (regions A, B of TMAX entries each), so the
# materialize step, which may store a state into slot 1, only ever overwrites the journal it has already read.


@cute.kernel
def kernel(
    qkv: cute.Tensor,
    z: cute.Tensor,
    hidden: cute.Tensor,
    baweight: cute.Tensor,
    al: cute.Tensor,
    dt: cute.Tensor,
    state: cute.Tensor,
    indices: cute.Tensor,
    cu: cute.Tensor,
    accepted: cute.Tensor,
    w: cute.Tensor,
    scale: cute.Tensor,
    qout: cute.Tensor,
    out: cute.Tensor,
    cs: cute.Tensor,
    cw: cute.Tensor,
    counts: cute.Tensor,
    jmode: cute.Tensor,
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    IS: cutlass.Constexpr,
    TMAX: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    CTAS: cutlass.Constexpr,
    CONV: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    BULK: cutlass.Constexpr,
    RELEASE: cutlass.Constexpr,
    JOURNAL: cutlass.Constexpr,
    PDL: cutlass.Constexpr,
    XPUSH: cutlass.Constexpr,
    LEAN: cutlass.Constexpr,
):
    NBUF: cutlass.Constexpr = BULK
    JREG: cutlass.Constexpr = TMAX * JENTRY  # floats per journal region (per head)
    EARLY_JR: cutlass.Constexpr = JOURNAL and PDL
    VR: cutlass.Constexpr = 128 // CTAS  # state rows per CTA
    RH: cutlass.Constexpr = VR // 16  # state rows per half-warp
    SL: cutlass.Constexpr = 32 + VR // 4  # 16-byte staging slots per token: q, k, v, z
    tid, _, _ = cute.arch.thread_idx()
    bid, n, _ = cute.arch.block_idx()
    rank = bid % CTAS
    head = bid // CTAS
    kh = head // 3
    if cutlass.const_expr(RELEASE):
        # Let the out projection (a PDL consumer) launch now; its wait still covers this grid.
        cute.arch.griddepcontrol_launch_dependents()
    lane = tid % 16
    hw = tid // 16
    kcol = lane * 8
    vrow0 = hw * RH
    warp = tid // 32
    smem = cutlass.memory.SmemAllocator()
    red = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(8 * 2 * TMAX), byte_alignment=16
    )
    ba = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(2 * TMAX), byte_alignment=16
    )
    part = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(16 * TMAX), byte_alignment=16
    )
    tot = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(TMAX), byte_alignment=16
    )
    tsc = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(4 * TMAX), byte_alignment=16
    )
    tot2 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(TMAX), byte_alignment=16
    )
    stage = smem.allocate_tensor(
        cutlass.Uint32, cute.make_layout(TMAX * SL * 4), byte_alignment=16
    )
    ticket = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), byte_alignment=16)
    if cutlass.const_expr(XPUSH):
        recv = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(TMAX), byte_alignment=16
        )
        xbar = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout(1), byte_alignment=8
        )
    JW: cutlass.Constexpr = JREG // 4  # 16-byte words of this head's journal region
    if cutlass.const_expr(JOURNAL):
        jbuf = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(JREG), byte_alignment=16
        )
    if cutlass.const_expr(NBUF > 0):
        # Per-token state staging, a ring of NBUF buffers: this CTA's VR rows are one contiguous run of the
        # slot, written by one bulk copy per token instead of every thread's global stores.
        hbuf = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(NBUF * VR * 128), byte_alignment=128
        )

    neg_a = -cute.math.exp(al[head].to(cutlass.Float32))
    bias = dt[head].to(cutlass.Float32)
    # The gated norm's weight and the static FP8 scale at entry, off the output tail: every output pair of a lane
    # (pair lane + 16 k) is the same state row since RH divides 16.
    wnorm = w[rank * VR + hw * RH + tid % 16 % RH].to(cutlass.Float32)
    qscale_out = scale[0]
    bos = cu[n]
    ntok = cu[n + 1] - bos
    acc = accepted[n]
    src = cutlass.Int64(0)
    if acc > 0:
        if acc <= TMAX:
            src = indices[n * IS + acc - 1].to(cutlass.Int64)
    jm = cutlass.Int32(0)
    wbase = cutlass.Int32(0)  # store the new base state to s0
    s0 = cutlass.Int64(0)
    s1 = cutlass.Int64(0)
    if cutlass.const_expr(JOURNAL):
        s0 = indices[n * IS].to(cutlass.Int64)
        s1 = indices[n * IS + 1].to(cutlass.Int64)
        if src > 0:
            if s0 > 0:
                if src != s0:
                    wbase = cutlass.Int32(1)
                jm = jmode[s0 * 48 + head]
                if jm != 0:
                    src = s0
                    wbase = cutlass.Int32(1)

    if src > 0:
        if cutlass.const_expr(CTAS > 1 and XPUSH):
            # The peer pushes its norm sums into this CTA's barrier; initialized and published at entry, waited
            # on (long satisfied) before the first push.
            if tid == 0:
                cute.arch.mbarrier_init(xbar.iterator, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
        # The source state rows: RH rows of 128 per half-warp, eight K columns per lane.
        h = cute.make_rmem_tensor((RH, 8), cutlass.Float32)
        jv = []
        if cutlass.const_expr(JOURNAL):
            # The head's live journal region, loaded alongside the state (a stale region when none is live).
            jr = (
                s1 * SS
                + head * 16384
                + (cutlass.max(jm, 1) - 1).to(cutlass.Int64) * JREG
            )
            for s in cutlass.range_constexpr((JW + THREADS - 1) // THREADS):
                jv.append(
                    ld_global_v4_u32(
                        get_ptr_as_int64(
                            state, jr + cutlass.min(s * THREADS + tid, JW - 1) * 4
                        )
                    )
                )
        for i in cutlass.range_constexpr(RH):
            off = src * SS + (head * 128 + rank * VR + vrow0 + i) * 128 + kcol
            p0 = ld_global_v4_u32(get_ptr_as_int64(state, off))
            p1 = ld_global_v4_u32(get_ptr_as_int64(state, off + 4))
            for j in cutlass.range_constexpr(4):
                h[i, j] = as_float(p0[j])
                h[i, j + 4] = as_float(p1[j])
        # BA projection inputs for every token (one dot pair per token over the 5120 inputs, three K
        # chunks). With the fused conv they are issued first so their latency overlaps it; without it,
        # after staging, which needs fewer live registers.
        bws = []
        aws = []
        hxs = []
        if cutlass.const_expr(CONV and not LEAN):
            for c in cutlass.range_constexpr(3):
                idx = cutlass.min(c * 2048 + tid * 8, 5120 - 8)
                bws.append(
                    ld_global_v4_u32(get_ptr_as_int64(baweight, head * 5120 + idx))
                )
                aws.append(
                    ld_global_v4_u32(
                        get_ptr_as_int64(baweight, (head + 48) * 5120 + idx)
                    )
                )
        if cutlass.const_expr(EARLY_JR):
            # Under PDL the replay and the base store run before the dependency wait, under the QKVZ GEMM.
            for s in cutlass.range_constexpr((JW + THREADS - 1) // THREADS):
                if s * THREADS + tid < JW:
                    for e in cutlass.range_constexpr(4):
                        jbuf[(s * THREADS + tid) * 4 + e] = jv[s][e]
            cute.arch.barrier()
        if cutlass.const_expr(EARLY_JR):
            if jm != 0:
                # Replay the previous verify's accepted tokens: the forward's decay multiply, then its update.
                for t in cutlass.range_constexpr(TMAX):
                    if t < acc:
                        je = t * JENTRY
                        jdecay = as_float(jbuf[je + 256])
                        for i in cutlass.range_constexpr(RH):
                            jinn = as_float(jbuf[je + rank * VR + vrow0 + i])
                            for j in cutlass.range_constexpr(8):
                                h[i, j] = fma_rn(
                                    jinn,
                                    as_float(jbuf[je + 128 + kcol + j]),
                                    mul_rn(h[i, j], jdecay),
                                )
            if wbase != 0:
                if s0 > 0:
                    # The new base: one state store per request instead of one per token.
                    for i in cutlass.range_constexpr(RH):
                        sa = (
                            (hbuf.iterator + (vrow0 + i) * 128 + kcol)
                            .toint()
                            .to(cutlass.Int32)
                        )
                        st_shared_v4(
                            sa,
                            as_bits(h[i, 0]),
                            as_bits(h[i, 1]),
                            as_bits(h[i, 2]),
                            as_bits(h[i, 3]),
                        )
                        st_shared_v4(
                            sa + 16,
                            as_bits(h[i, 4]),
                            as_bits(h[i, 5]),
                            as_bits(h[i, 6]),
                            as_bits(h[i, 7]),
                        )
                    fence_async_smem()
                    cute.arch.barrier()
                    if tid == 0:
                        bulk_store(
                            get_ptr_as_int64(
                                state, s0 * SS + (head * 128 + rank * VR) * 128
                            ),
                            hbuf.iterator.toint().to(cutlass.Int32),
                            cutlass.Int32(VR * 128 * 4),
                        )
                        bulk_commit()
        if cutlass.const_expr(CONV):
            # One (8-channel group, token) unit per thread and round (CVR rounds): conv groups for q, k (this K
            # head) and v (this CTA's rows), then z groups. A unit's window cv_win holds
            # seq[t .. t+3] of seq = [old[a-1], old[a], old[a+1], x_0, ..., x_{T-1}]. The history entries and the
            # conv weights come from earlier steps, so they load before the PDL wait.
            cv_NG: cutlass.Constexpr = 32 + VR // 8
            CVR: cutlass.Constexpr = (TMAX * SL + THREADS - 1) // THREADS
            cv_slot = indices[n * IS].to(cutlass.Int64)
            cv_first = (acc - 1).to(cutlass.Int64)
            cv_ts = []
            cv_js = []
            cv_cs = []
            cv_wins = []
            cv_wtss = []
            for cv_r in cutlass.range_constexpr(0 if LEAN else CVR):
                cv_u = tid + cv_r * THREADS
                cv_t = cv_u // SL
                cv_j = cv_u % SL
                cv_c = kh * 128 + cv_j * 8
                if cv_j >= 16:
                    cv_c = 2048 + kh * 128 + (cv_j - 16) * 8
                if cv_j >= 32:
                    cv_c = 4096 + head * 128 + rank * VR + (cv_j - 32) * 8
                cv_win = cute.make_rmem_tensor((4, 4), cutlass.Uint32)
                cv_wts = cute.make_rmem_tensor((4, 4), cutlass.Uint32)
                cv_ts.append(cv_t)
                cv_js.append(cv_j)
                cv_cs.append(cv_c)
                cv_wins.append(cv_win)
                cv_wtss.append(cv_wts)
                if cv_slot > 0:
                    if cv_t < ntok:
                        if cv_j < cv_NG:
                            for cv_e in cutlass.range_constexpr(4):
                                cv_si = cv_t + cv_e
                                if cv_si < 3:
                                    cv_vals = ld_global_v4_u32(
                                        get_ptr_as_int64(
                                            cs,
                                            cv_slot * CSS
                                            + (cv_first + cv_si.to(cutlass.Int64)) * CST
                                            + cv_c,
                                        )
                                    )
                                    for cv_i in cutlass.range_constexpr(4):
                                        cv_win[cv_e, cv_i] = cv_vals[cv_i]
                            for cv_k4 in cutlass.range_constexpr(4):
                                cv_vals = ld_global_v4_u32(
                                    get_ptr_as_int64(cw, cv_c * 4 + cv_k4 * 8)
                                )
                                for cv_i in cutlass.range_constexpr(4):
                                    cv_wts[cv_k4, cv_i] = cv_vals[cv_i]
        if cutlass.const_expr(PDL):
            # Everything above (state, journal, BA weights, conv history and weights) was written before this
            # step's graph; the QKVZ projection (qkv, z) and the normed hidden it consumed are read only past
            # this point.
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(CONV and not LEAN):
            for c in cutlass.range_constexpr(3):
                idx = cutlass.min(c * 2048 + tid * 8, 5120 - 8)
                for t in cutlass.range_constexpr(TMAX):
                    tt = cutlass.min(cutlass.Int32(t), ntok - 1)
                    hxs.append(
                        ld_global_v4_u32(
                            get_ptr_as_int64(hidden, (bos + tt) * 5120 + idx)
                        )
                    )
        if cutlass.const_expr(CONV and LEAN):
            # Two CTAs per SM cap a thread at 128 registers: one round at a time after the wait, so only that
            # round's window and weights are live next to the state rows.
            for lc_r in cutlass.range(CVR, unroll=1):
                lc_u = tid + lc_r * THREADS
                lc_t = lc_u // SL
                lc_j = lc_u % SL
                lc_c = kh * 128 + lc_j * 8
                if lc_j >= 16:
                    lc_c = 2048 + kh * 128 + (lc_j - 16) * 8
                if lc_j >= 32:
                    lc_c = 4096 + head * 128 + rank * VR + (lc_j - 32) * 8
                lc_win = cute.make_rmem_tensor((4, 4), cutlass.Uint32)
                lc_wts = cute.make_rmem_tensor((4, 4), cutlass.Uint32)
                if cv_slot > 0:
                    if lc_t < ntok:
                        if lc_j < cv_NG:
                            for lc_e in cutlass.range_constexpr(4):
                                lc_si = lc_t + lc_e
                                if lc_si < 3:
                                    lc_vals = ld_global_v4_u32(
                                        get_ptr_as_int64(
                                            cs,
                                            cv_slot * CSS
                                            + (cv_first + lc_si.to(cutlass.Int64)) * CST
                                            + lc_c,
                                        )
                                    )
                                    for lc_i in cutlass.range_constexpr(4):
                                        lc_win[lc_e, lc_i] = lc_vals[lc_i]
                                if lc_si >= 3:
                                    lc_vals = ld_global_v4_u32(
                                        get_ptr_as_int64(
                                            qkv,
                                            (bos + lc_si - 3).to(cutlass.Int64) * XS
                                            + lc_c,
                                        )
                                    )
                                    for lc_i in cutlass.range_constexpr(4):
                                        lc_win[lc_e, lc_i] = lc_vals[lc_i]
                            for lc_k4 in cutlass.range_constexpr(4):
                                lc_vals = ld_global_v4_u32(
                                    get_ptr_as_int64(cw, lc_c * 4 + lc_k4 * 8)
                                )
                                for lc_i in cutlass.range_constexpr(4):
                                    lc_wts[lc_k4, lc_i] = lc_vals[lc_i]
                            conv_stage_unit(lc_win, lc_wts, stage, lc_u)
                        else:
                            lc_row = (bos + lc_t).to(cutlass.Int64)
                            lc_vals = ld_global_v4_u32(
                                get_ptr_as_int64(
                                    z,
                                    lc_row * ZS
                                    + head * 128
                                    + rank * VR
                                    + (lc_j - cv_NG) * 8,
                                )
                            )
                            for lc_i in cutlass.range_constexpr(4):
                                stage[lc_u * 4 + lc_i] = lc_vals[lc_i]
        if cutlass.const_expr(CONV):
            # The new tokens' window entries (the QKVZ projection's rows), then the conv and the staging.
            for cv_r in cutlass.range_constexpr(0 if LEAN else CVR):
                cv_u = tid + cv_r * THREADS
                cv_t = cv_ts[cv_r]
                cv_j = cv_js[cv_r]
                cv_c = cv_cs[cv_r]
                cv_win = cv_wins[cv_r]
                cv_wts = cv_wtss[cv_r]
                if cv_slot > 0:
                    if cv_t < ntok:
                        if cv_j < cv_NG:
                            for cv_e in cutlass.range_constexpr(4):
                                cv_si = cv_t + cv_e
                                if cv_si >= 3:
                                    cv_vals = ld_global_v4_u32(
                                        get_ptr_as_int64(
                                            qkv,
                                            (bos + cv_si - 3).to(cutlass.Int64) * XS
                                            + cv_c,
                                        )
                                    )
                                    for cv_i in cutlass.range_constexpr(4):
                                        cv_win[cv_e, cv_i] = cv_vals[cv_i]
                            conv_stage_unit(cv_win, cv_wts, stage, cv_u)
                        else:
                            cv_row = (bos + cv_t).to(cutlass.Int64)
                            cv_vals = ld_global_v4_u32(
                                get_ptr_as_int64(
                                    z,
                                    cv_row * ZS
                                    + head * 128
                                    + rank * VR
                                    + (cv_j - cv_NG) * 8,
                                )
                            )
                            for cv_i in cutlass.range_constexpr(4):
                                stage[cv_u * 4 + cv_i] = cv_vals[cv_i]
            # Every old-history read of this CTA is complete past this barrier.
            cute.arch.barrier()
        else:
            # Stage every token's q, k (this K head), v and z (this CTA's rows).
            for s in cutlass.range_constexpr((TMAX * SL + THREADS - 1) // THREADS):
                slot = s * THREADS + tid
                if slot < TMAX * SL:
                    t = slot // SL
                    j = slot % SL
                    if t < ntok:
                        row = (bos + t).to(cutlass.Int64)
                        ptr = get_ptr_as_int64(qkv, row * XS + kh * 128 + j * 8)
                        if j >= 16:
                            ptr = get_ptr_as_int64(
                                qkv, row * XS + 2048 + kh * 128 + (j - 16) * 8
                            )
                        if j >= 32:
                            ptr = get_ptr_as_int64(
                                qkv,
                                row * XS + 4096 + head * 128 + rank * VR + (j - 32) * 8,
                            )
                        if j >= 32 + VR // 8:
                            ptr = get_ptr_as_int64(
                                z,
                                row * ZS
                                + head * 128
                                + rank * VR
                                + (j - 32 - VR // 8) * 8,
                            )
                        vals = ld_global_v4_u32(ptr)
                        for i in cutlass.range_constexpr(4):
                            stage[slot * 4 + i] = vals[i]
        if cutlass.const_expr(not CONV):
            for c in cutlass.range_constexpr(3):
                idx = cutlass.min(c * 2048 + tid * 8, 5120 - 8)
                bws.append(
                    ld_global_v4_u32(get_ptr_as_int64(baweight, head * 5120 + idx))
                )
                aws.append(
                    ld_global_v4_u32(
                        get_ptr_as_int64(baweight, (head + 48) * 5120 + idx)
                    )
                )
                for t in cutlass.range_constexpr(TMAX):
                    tt = cutlass.min(cutlass.Int32(t), ntok - 1)
                    hxs.append(
                        ld_global_v4_u32(
                            get_ptr_as_int64(hidden, (bos + tt) * 5120 + idx)
                        )
                    )
        # The three K chunks accumulate in registers, so there is a single reduction round.
        bdots = cute.make_rmem_tensor((TMAX,), cutlass.Float32)
        adots = cute.make_rmem_tensor((TMAX,), cutlass.Float32)
        for t in cutlass.range_constexpr(TMAX):
            bdots[t] = cutlass.Float32(0)
            adots[t] = cutlass.Float32(0)
        if cutlass.const_expr(LEAN):
            # Same accumulation order as below, one K chunk's weights and hidden rows live at a time.
            for lb_c in cutlass.range(3, unroll=1):
                if lb_c * 2048 + tid * 8 < 5120:
                    lb_idx = lb_c * 2048 + tid * 8
                    lb_bw = ld_global_v4_u32(
                        get_ptr_as_int64(baweight, head * 5120 + lb_idx)
                    )
                    lb_aw = ld_global_v4_u32(
                        get_ptr_as_int64(baweight, (head + 48) * 5120 + lb_idx)
                    )
                    for t in cutlass.range_constexpr(TMAX):
                        lb_tt = cutlass.min(cutlass.Int32(t), ntok - 1)
                        lb_hx = ld_global_v4_u32(
                            get_ptr_as_int64(hidden, (bos + lb_tt) * 5120 + lb_idx)
                        )
                        for j in cutlass.range_constexpr(4):
                            lb_h0, lb_h1 = bfloat2_to_float2_scaled(
                                lb_hx[j], cutlass.Float32(1)
                            )
                            lb_b0, lb_b1 = bfloat2_to_float2_scaled(
                                lb_bw[j], cutlass.Float32(1)
                            )
                            lb_a0, lb_a1 = bfloat2_to_float2_scaled(
                                lb_aw[j], cutlass.Float32(1)
                            )
                            bdots[t] = bdots[t] + lb_h0 * lb_b0
                            bdots[t] = bdots[t] + lb_h1 * lb_b1
                            adots[t] = adots[t] + lb_h0 * lb_a0
                            adots[t] = adots[t] + lb_h1 * lb_a1
        for c in cutlass.range_constexpr(0 if LEAN else 3):
            if c * 2048 + tid * 8 < 5120:
                for t in cutlass.range_constexpr(TMAX):
                    hx = hxs[c * TMAX + t]
                    for j in cutlass.range_constexpr(4):
                        h0, h1 = bfloat2_to_float2_scaled(hx[j], cutlass.Float32(1))
                        b0, b1 = bfloat2_to_float2_scaled(bws[c][j], cutlass.Float32(1))
                        a0, a1 = bfloat2_to_float2_scaled(aws[c][j], cutlass.Float32(1))
                        bdots[t] = bdots[t] + h0 * b0
                        bdots[t] = bdots[t] + h1 * b1
                        adots[t] = adots[t] + h0 * a0
                        adots[t] = adots[t] + h1 * a1
        for t in cutlass.range_constexpr(TMAX):
            bd = bdots[t]
            ad = adots[t]
            for step in cutlass.range_constexpr(5):
                bd = bd + cute.arch.shuffle_sync_bfly(bd, offset=1 << step)
                ad = ad + cute.arch.shuffle_sync_bfly(ad, offset=1 << step)
            if tid % 32 == 0:
                red[(warp * TMAX + t) * 2] = bd
                red[(warp * TMAX + t) * 2 + 1] = ad
        if cutlass.const_expr(CONV):
            # The last warp has two K chunks to the others' three: it takes the per-request
            # reader ticket while they finish.
            if cv_slot > 0:
                if tid == THREADS - 1:
                    ticket[0] = reader_done(
                        get_ptr_as_int64(counts, (n * 16 + kh) * 32),
                        cutlass.Int32(3 * CTAS - 1),
                    )
        if cutlass.const_expr(JOURNAL and not EARLY_JR):
            # The journal loads were issued with the state's; the barrier below publishes them for the replay.
            for s in cutlass.range_constexpr((JW + THREADS - 1) // THREADS):
                if s * THREADS + tid < JW:
                    for e in cutlass.range_constexpr(4):
                        jbuf[(s * THREADS + tid) * 4 + e] = jv[s][e]
        # Per-token q/k RMS scales, off the recurrence: half-warp t reduces token t exactly as the token loop's
        # half-warps did (same lanes, same butterfly), so the scaled k (and the journal) is unchanged.
        if cutlass.const_expr(not CONV):
            cute.arch.barrier()  # the staging stores (the fused conv path has its own barrier after them)
        pt = cutlass.min(hw, TMAX - 1)
        pqs = cutlass.Float32(0)
        pks = cutlass.Float32(0)
        for j in cutlass.range_constexpr(4):
            pq0, pq1 = bfloat2_to_float2_scaled(
                stage[(pt * SL + lane) * 4 + j], cutlass.Float32(1)
            )
            pk0, pk1 = bfloat2_to_float2_scaled(
                stage[(pt * SL + 16 + lane) * 4 + j], cutlass.Float32(1)
            )
            pqs = pqs + pq0 * pq0
            pqs = pqs + pq1 * pq1
            pks = pks + pk0 * pk0
            pks = pks + pk1 * pk1
        pqscale = cutlass.Float32(0.08838834764831845) / cute.math.sqrt(
            warp_sum(pqs) + cutlass.Float32(1e-6)
        )
        pkscale = cutlass.Float32(1) / cute.math.sqrt(
            warp_sum(pks) + cutlass.Float32(1e-6)
        )
        if hw < TMAX:
            if lane == 0:
                tsc[pt * 4] = pqscale
                tsc[pt * 4 + 1] = pkscale
        cute.arch.barrier()
        if tid < 2 * TMAX:
            total = cutlass.Float32(0)
            for wi in cutlass.range_constexpr(8):
                total = total + red[wi * 2 * TMAX + tid]
            ba[tid] = total.to(cutlass.BFloat16).to(cutlass.Float32)
            # Per-token gate scalars, off the recurrence: thread 2t has token t's b (beta), 2t + 1 its a (decay).
            gv = ba[tid]
            if tid % 2 == 0:
                tsc[(tid // 2) * 4 + 3] = cutlass.Float32(1) / (
                    cutlass.Float32(1) + cute.math.exp(-gv)
                )
            else:
                gx = gv + bias
                soft = gx
                if gx <= cutlass.Float32(20):
                    soft = cute.math.log(cutlass.Float32(1) + cute.math.exp(gx))
                tsc[(tid // 2) * 4 + 2] = cute.math.exp(neg_a * soft)
        if cutlass.const_expr(CONV and LEAN):
            # The lean conv kept no windows: a unit's newest entry is its token's QKVZ row, re-read here. The
            # token-0 units (all in round 0) also move two old entries down; they read them before a barrier
            # that orders those reads before every unit's history store.
            if cv_slot > 0:
                ls_last = ticket[0] == 3 * CTAS - 1
                ls_t0 = tid // SL
                ls_j0 = tid % SL
                ls_c0 = kh * 128 + ls_j0 * 8
                if ls_j0 >= 16:
                    ls_c0 = 2048 + kh * 128 + (ls_j0 - 16) * 8
                if ls_j0 >= 32:
                    ls_c0 = 4096 + head * 128 + rank * VR + (ls_j0 - 32) * 8
                ls_own0 = ls_j0 >= 32
                if ls_last:
                    ls_own0 = ls_j0 >= 0
                ls_move = cutlass.Boolean(False)
                if ls_t0 == 0:
                    if ntok > 0:
                        if ls_j0 < cv_NG:
                            if ls_own0:
                                ls_move = cutlass.Boolean(True)
                ls_base0 = cv_slot * CSS + ls_c0
                ls_old = cute.make_rmem_tensor((2, 4), cutlass.Uint32)
                if ls_move:
                    for ls_e in cutlass.range_constexpr(2):
                        ls_v = ld_global_v4_u32(
                            get_ptr_as_int64(cs, ls_base0 + (cv_first + 1 + ls_e) * CST)
                        )
                        for ls_i in cutlass.range_constexpr(4):
                            ls_old[ls_e, ls_i] = ls_v[ls_i]
                cute.arch.barrier()
                for ls_r in cutlass.range(CVR, unroll=1):
                    ls_u = tid + ls_r * THREADS
                    ls_t = ls_u // SL
                    ls_j = ls_u % SL
                    ls_c = kh * 128 + ls_j * 8
                    if ls_j >= 16:
                        ls_c = 2048 + kh * 128 + (ls_j - 16) * 8
                    if ls_j >= 32:
                        ls_c = 4096 + head * 128 + rank * VR + (ls_j - 32) * 8
                    ls_own = ls_j >= 32
                    if ls_last:
                        ls_own = ls_j >= 0
                    if ls_t < ntok:
                        if ls_j < cv_NG:
                            if ls_own:
                                ls_x = ld_global_v4_u32(
                                    get_ptr_as_int64(
                                        qkv, (bos + ls_t).to(cutlass.Int64) * XS + ls_c
                                    )
                                )
                                st_global_v4_u32(
                                    get_ptr_as_int64(
                                        cs,
                                        cv_slot * CSS
                                        + ls_c
                                        + (2 + ls_t).to(cutlass.Int64) * CST,
                                    ),
                                    ls_x[0],
                                    ls_x[1],
                                    ls_x[2],
                                    ls_x[3],
                                )
                if ls_move:
                    for ls_e in cutlass.range_constexpr(2):
                        st_global_v4_u32(
                            get_ptr_as_int64(cs, ls_base0 + ls_e * CST),
                            ls_old[ls_e, 0],
                            ls_old[ls_e, 1],
                            ls_old[ls_e, 2],
                            ls_old[ls_e, 3],
                        )
        if cutlass.const_expr(CONV and not LEAN):
            # V channels belong to this CTA; Q/K state is shifted by the last reader only.
            if cv_slot > 0:
                cv_last = ticket[0] == 3 * CTAS - 1
                for cv_r in cutlass.range_constexpr(CVR):
                    cv_t = cv_ts[cv_r]
                    cv_j = cv_js[cv_r]
                    cv_c = cv_cs[cv_r]
                    cv_win = cv_wins[cv_r]
                    cv_owner = cv_j >= 32
                    if cv_last:
                        cv_owner = cv_j >= 0
                    if cv_t < ntok:
                        if cv_j < cv_NG:
                            if cv_owner:
                                cv_base = cv_slot * CSS + cv_c
                                st_global_v4_u32(
                                    get_ptr_as_int64(
                                        cs, cv_base + (2 + cv_t).to(cutlass.Int64) * CST
                                    ),
                                    cv_win[3, 0],
                                    cv_win[3, 1],
                                    cv_win[3, 2],
                                    cv_win[3, 3],
                                )
                                # Token-0 units are all in round 0 (SL <= THREADS).
                                if cutlass.const_expr(cv_r == 0):
                                    if cv_t == 0:
                                        st_global_v4_u32(
                                            get_ptr_as_int64(cs, cv_base),
                                            cv_win[1, 0],
                                            cv_win[1, 1],
                                            cv_win[1, 2],
                                            cv_win[1, 3],
                                        )
                                        st_global_v4_u32(
                                            get_ptr_as_int64(cs, cv_base + CST),
                                            cv_win[2, 0],
                                            cv_win[2, 1],
                                            cv_win[2, 2],
                                            cv_win[2, 3],
                                        )
        if cutlass.const_expr(JOURNAL and not EARLY_JR):
            # Replay and the base store come after staging and the BA projection, so the state and journal
            # loads overlap them.
            if jm != 0:
                # Replay the previous verify's accepted tokens: the forward's decay multiply, then its update.
                for t in cutlass.range_constexpr(TMAX):
                    if t < acc:
                        je = t * JENTRY
                        jdecay = as_float(jbuf[je + 256])
                        for i in cutlass.range_constexpr(RH):
                            jinn = as_float(jbuf[je + rank * VR + vrow0 + i])
                            for j in cutlass.range_constexpr(8):
                                h[i, j] = fma_rn(
                                    jinn,
                                    as_float(jbuf[je + 128 + kcol + j]),
                                    mul_rn(h[i, j], jdecay),
                                )
            if wbase != 0:
                if s0 > 0:
                    # The new base: one state store per request instead of one per token.
                    for i in cutlass.range_constexpr(RH):
                        sa = (
                            (hbuf.iterator + (vrow0 + i) * 128 + kcol)
                            .toint()
                            .to(cutlass.Int32)
                        )
                        st_shared_v4(
                            sa,
                            as_bits(h[i, 0]),
                            as_bits(h[i, 1]),
                            as_bits(h[i, 2]),
                            as_bits(h[i, 3]),
                        )
                        st_shared_v4(
                            sa + 16,
                            as_bits(h[i, 4]),
                            as_bits(h[i, 5]),
                            as_bits(h[i, 6]),
                            as_bits(h[i, 7]),
                        )
                    fence_async_smem()
                    cute.arch.barrier()
                    if tid == 0:
                        bulk_store(
                            get_ptr_as_int64(
                                state, s0 * SS + (head * 128 + rank * VR) * 128
                            ),
                            hbuf.iterator.toint().to(cutlass.Int32),
                            cutlass.Int32(VR * 128 * 4),
                        )
                        bulk_commit()
        cute.arch.barrier()
        # Core outputs, the (token, row) pairs of a half-warp spread over its 16 lanes: pair p = t * RH + i
        # lives in lane p % 16, slot p // 16 (every lane holds every row's reduced value).
        OK: cutlass.Constexpr = (TMAX * RH + 15) // 16
        outk = cute.make_rmem_tensor((OK,), cutlass.Float32)
        for k in cutlass.range_constexpr(OK):
            outk[k] = cutlass.Float32(0)
        for t in cutlass.range_constexpr(TMAX):
            if t < ntok:
                q = cute.make_rmem_tensor((8,), cutlass.Float32)
                k = cute.make_rmem_tensor((8,), cutlass.Float32)
                v = cute.make_rmem_tensor((RH,), cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    q[2 * j], q[2 * j + 1] = bfloat2_to_float2_scaled(
                        stage[(t * SL + lane) * 4 + j], cutlass.Float32(1)
                    )
                    k[2 * j], k[2 * j + 1] = bfloat2_to_float2_scaled(
                        stage[(t * SL + 16 + lane) * 4 + j], cutlass.Float32(1)
                    )
                for i in cutlass.range_constexpr(RH // 2):
                    word = (t * SL + 32 + vrow0 // 8) * 4 + (vrow0 % 8) // 2 + i
                    v[2 * i], v[2 * i + 1] = bfloat2_to_float2_scaled(
                        stage[word], cutlass.Float32(1)
                    )
                qscale = tsc[t * 4]
                kscale = tsc[t * 4 + 1]
                for j in cutlass.range_constexpr(8):
                    q[j] = q[j] * qscale
                    k[j] = k[j] * kscale
                decay = tsc[t * 4 + 2]
                beta = tsc[t * 4 + 3]
                dots = cute.make_rmem_tensor((RH,), cutlass.Float32)
                for i in cutlass.range_constexpr(RH):
                    dots[i] = cutlass.Float32(0)
                    for j in cutlass.range_constexpr(8):
                        h[i, j] = mul_rn(h[i, j], decay)
                        dots[i] = dots[i] + h[i, j] * k[j]
                for step in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(RH):
                        dots[i] = dots[i] + cute.arch.shuffle_sync_bfly(
                            dots[i], offset=1 << step
                        )
                innov = cute.make_rmem_tensor((RH,), cutlass.Float32)
                for i in cutlass.range_constexpr(RH):
                    innovation = (v[i] - dots[i]) * beta
                    innov[i] = innovation
                    dots[i] = cutlass.Float32(0)
                    for j in cutlass.range_constexpr(8):
                        h[i, j] = fma_rn(innovation, k[j], h[i, j])
                        dots[i] = dots[i] + h[i, j] * q[j]
                for step in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(RH):
                        dots[i] = dots[i] + cute.arch.shuffle_sync_bfly(
                            dots[i], offset=1 << step
                        )
                ss = cutlass.Float32(0)
                for i in cutlass.range_constexpr(RH):
                    o = dots[i].to(cutlass.BFloat16).to(cutlass.Float32)
                    ss = ss + o * o
                    if lane == (t * RH + i) % 16:
                        outk[(t * RH + i) // 16] = o
                if lane == 0:
                    part[t * 16 + hw] = ss
                dst = indices[n * IS + t].to(cutlass.Int64)
                if cutlass.const_expr(JOURNAL):
                    if s1 > 0:
                        # Write the region the replay did not read (B after A, A after B or none).
                        jw = s1 * SS + head * 16384 + t * JENTRY
                        if jm == 1:
                            jw = jw + JREG
                        if lane == 0:
                            for i in cutlass.range_constexpr(RH // 4):
                                st_global_v4_u32(
                                    get_ptr_as_int64(
                                        state, jw + rank * VR + vrow0 + i * 4
                                    ),
                                    as_bits(innov[i * 4]),
                                    as_bits(innov[i * 4 + 1]),
                                    as_bits(innov[i * 4 + 2]),
                                    as_bits(innov[i * 4 + 3]),
                                )
                        if rank == 0:
                            if hw == 0:
                                st_global_v4_u32(
                                    get_ptr_as_int64(state, jw + 128 + kcol),
                                    as_bits(k[0]),
                                    as_bits(k[1]),
                                    as_bits(k[2]),
                                    as_bits(k[3]),
                                )
                                st_global_v4_u32(
                                    get_ptr_as_int64(state, jw + 128 + kcol + 4),
                                    as_bits(k[4]),
                                    as_bits(k[5]),
                                    as_bits(k[6]),
                                    as_bits(k[7]),
                                )
                            if tid == 0:
                                state[jw + 256] = decay
                elif cutlass.const_expr(NBUF > 0):
                    # Every token commits one group (empty without a slot), so the ring wait is exact.
                    if cutlass.const_expr(t >= NBUF):
                        if tid == 0:
                            bulk_wait_read(NBUF - 1)
                        cute.arch.barrier()
                    if dst > 0:
                        for i in cutlass.range_constexpr(RH):
                            sa = (
                                (
                                    hbuf.iterator
                                    + ((t % NBUF) * VR + vrow0 + i) * 128
                                    + kcol
                                )
                                .toint()
                                .to(cutlass.Int32)
                            )
                            st_shared_v4(
                                sa,
                                as_bits(h[i, 0]),
                                as_bits(h[i, 1]),
                                as_bits(h[i, 2]),
                                as_bits(h[i, 3]),
                            )
                            st_shared_v4(
                                sa + 16,
                                as_bits(h[i, 4]),
                                as_bits(h[i, 5]),
                                as_bits(h[i, 6]),
                                as_bits(h[i, 7]),
                            )
                        fence_async_smem()
                        cute.arch.barrier()
                        if tid == 0:
                            bulk_store(
                                get_ptr_as_int64(
                                    state, dst * SS + (head * 128 + rank * VR) * 128
                                ),
                                (hbuf.iterator + (t % NBUF) * VR * 128)
                                .toint()
                                .to(cutlass.Int32),
                                cutlass.Int32(VR * 128 * 4),
                            )
                    if tid == 0:
                        bulk_commit()
                else:
                    if dst > 0:
                        for i in cutlass.range_constexpr(RH):
                            off = (
                                dst * SS
                                + (head * 128 + rank * VR + vrow0 + i) * 128
                                + kcol
                            )
                            st_global_v4_u32(
                                get_ptr_as_int64(state, off),
                                as_bits(h[i, 0]),
                                as_bits(h[i, 1]),
                                as_bits(h[i, 2]),
                                as_bits(h[i, 3]),
                            )
                            st_global_v4_u32(
                                get_ptr_as_int64(state, off + 4),
                                as_bits(h[i, 4]),
                                as_bits(h[i, 5]),
                                as_bits(h[i, 6]),
                                as_bits(h[i, 7]),
                            )
        cute.arch.barrier()
        if cutlass.const_expr(CTAS > 1 and XPUSH):
            # One-way exchange: each token's partial sum goes straight into the peer's shared memory (st.async
            # crediting its barrier), so no cluster round trip and no peer read; own + peer as before.
            cute.arch.cluster_wait()
            if tid < TMAX:
                total = cutlass.Float32(0)
                for p in cutlass.range_constexpr(16):
                    total = total + part[tid * 16 + p]
                tot[tid] = total
                push_peer_f32(
                    (recv.iterator + tid).toint().to(cutlass.Int32),
                    total,
                    xbar.iterator.toint().to(cutlass.Int32),
                    1 - rank,
                )
            if tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(xbar.iterator, TMAX * 4)
            cute.arch.mbarrier_wait(xbar.iterator, 0)
            if tid < TMAX:
                tot2[tid] = tot[tid] + recv[tid]
            cute.arch.barrier()
        else:
            if tid < TMAX:
                total = cutlass.Float32(0)
                for p in cutlass.range_constexpr(16):
                    total = total + part[tid * 16 + p]
                tot[tid] = total
            cute.arch.barrier()
            if cutlass.const_expr(CTAS > 1):
                cluster_sync()
            # The head's per-token sums (own + peer), read once per token into shared memory, then release the
            # peer: the output math overlaps its remaining reads of this CTA's sums.
            if tid < TMAX:
                sumv = tot[tid]
                if cutlass.const_expr(CTAS > 1):
                    sumv = sumv + peer_float(
                        (tot.iterator + tid).toint().to(cutlass.Int32), 1 - rank
                    )
                tot2[tid] = sumv
            cute.arch.barrier()
            if cutlass.const_expr(CTAS > 1):
                cluster_arrive()
        for k in cutlass.range_constexpr(OK):
            pr = lane + 16 * k
            if pr < TMAX * RH:
                t = pr // RH
                if t < ntok:
                    vrow = vrow0 + pr % RH
                    wv = wnorm
                    inv = cute.math.rsqrt(
                        tot2[t] / cutlass.Float32(128) + cutlass.Float32(EPS)
                    )
                    g0, g1 = bfloat2_to_float2_scaled(
                        stage[
                            (t * SL + 32 + VR // 8 + vrow // 8) * 4 + (vrow % 8) // 2
                        ],
                        cutlass.Float32(1),
                    )
                    gate = g0
                    if vrow % 2 == 1:
                        gate = g1
                    rounded = (
                        ((outk[k] * inv) * wv)
                        * (gate / (cutlass.Float32(1) + cute.math.exp(-gate)))
                    ).to(cutlass.BFloat16)
                    oi = (bos + t) * 6144 + head * 128 + rank * VR + vrow
                    qout[oi] = quant_fp8(rounded.to(cutlass.Float32), qscale_out)
                    out[oi] = rounded
        # This CTA's shared memory must outlive the peer's reads of it, and the state copies' reads of it.
        if cutlass.const_expr(CTAS > 1 and not XPUSH):
            cluster_wait()
        if cutlass.const_expr(NBUF > 0):
            if tid == 0:
                bulk_wait_all()
        if cutlass.const_expr(JOURNAL):
            if s0 > 0:
                if s1 > 0:
                    # Rank 0 flips the mode once both ranks have read it: the cluster wait above, or (XPUSH)
                    # the peer's pushed sums, sent long after its read at entry.
                    if rank == 0:
                        if tid == 0:
                            jnext = cutlass.Int32(1)
                            if jm == 1:
                                jnext = cutlass.Int32(2)
                            jmode[s0 * 48 + head] = jnext
    else:
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
        if lane < RH:
            for t in cutlass.range_constexpr(TMAX):
                if t < ntok:
                    oi = (bos + t) * 6144 + head * 128 + rank * VR + vrow0 + lane
                    qout[oi] = cutlass.Uint8(0)
                    out[oi] = cutlass.BFloat16(0)


@cute.jit
def launch(
    qkv,
    z,
    hidden,
    baweight,
    al,
    dt,
    state,
    indices,
    cu,
    accepted,
    w,
    scale,
    qout,
    out,
    cs,
    cw,
    counts,
    jmode,
    R: cutlass.Int32,
    XS: cutlass.Constexpr,
    ZS: cutlass.Constexpr,
    SS: cutlass.Constexpr,
    IS: cutlass.Constexpr,
    TMAX: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
    CTAS: cutlass.Constexpr,
    CONV: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CST: cutlass.Constexpr,
    MINB: cutlass.Constexpr,
    BULK: cutlass.Constexpr,
    RELEASE: cutlass.Constexpr,
    JOURNAL: cutlass.Constexpr,
    PDL: cutlass.Constexpr,
    XPUSH: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    kernel(
        qkv,
        z,
        hidden,
        baweight,
        al,
        dt,
        state,
        indices,
        cu,
        accepted,
        w,
        scale,
        qout,
        out,
        cs,
        cw,
        counts,
        jmode,
        XS,
        ZS,
        SS,
        IS,
        TMAX,
        EPS,
        CTAS,
        CONV,
        CSS,
        CST,
        BULK,
        RELEASE,
        JOURNAL,
        PDL,
        XPUSH,
        CONV and MINB > 1,
    ).launch(
        grid=(48 * CTAS, R, 1),
        block=(THREADS, 1, 1),
        cluster=(CTAS, 1, 1),
        stream=stream,
        use_pdl=PDL,
        min_blocks_per_mp=MINB,
        max_number_threads=[THREADS, 1, 1],
    )


def _flat(t: torch.Tensor):
    """A 1-D view covering a (possibly strided) tensor's whole footprint."""
    n = 1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride()))
    return from_dlpack(
        t.detach().as_strided((n,), (1,)), assumed_align=16 if n > 1 else 4
    )


class FrostGdnVerify:
    """Compiled per (max tokens per request, input / cache strides), shared by
    every GDN layer. ``qkv`` is the QKV part of the projected QKVZ rows (BF16,
    row-strided; pre-conv when ``fuses_conv``, else after a separate speculative conv),
    ``z`` its Z part viewed [rows, 48, 128], ``hidden`` the BF16 normed layer input;
    ``indices`` [requests, max tokens], ``cu`` the per-request token offsets and
    ``accepted`` the per-request accepted-token counts; ``cs`` the conv
    state viewed [slots, 10240, state_len] and ``cw`` the conv weight [10240, 4].
    Returns the FP8 [rows, 48, 128] output."""

    def __init__(
        self,
        device: torch.device,
        eps: float,
        ctas: int | None = None,
        conv: bool = True,
        minb: int | None = None,
        bulk: bool | None = None,
        release: bool = True,
        journal: bool = True,
        pdl: bool = True,
        push: bool = True,
    ):
        assert ctas in (None, 1, 2)
        self.device, self.eps, self.fixed_ctas = device, float(eps), ctas
        self.conv = conv
        self.fixed_minb = minb
        self.fixed_bulk = bulk
        self.release = release
        self.journal = journal
        self.pdl = pdl
        self.push = push  # one-way norm-sum exchange between a head's two CTAs
        self.no_journal = torch.zeros(1, dtype=torch.int32, device=device)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.compiled: dict[tuple, Any] = {}
        self.buffers: dict[int, dict[str, torch.Tensor]] = {}

    def minb(self, requests: int) -> int:
        """Resident CTAs per SM the launch asks ptxas for (caps registers at 65536 / (MINB x 256)): two whenever
        the grid exceeds one wave."""
        if self.fixed_minb is not None:
            return self.fixed_minb
        return 2 if 48 * self.ctas(requests) * requests > self.sms else 1

    def bulk(self, requests: int, tmax: int = 4) -> int:
        """Ring buffers for the bulk state copies (0: per-thread global stores). A token's rows take 32 KB at
        two CTAs per head, 64 KB at one: every token buffered when that fits in 128 KB, else three (one CTA per
        SM, 192 KB) or one (two CTAs per SM)."""
        if self.journal:
            return 1  # the base store only
        if self.fixed_bulk is not None and not self.fixed_bulk:
            return 0
        per = 128 // self.ctas(requests) * 128 * 4
        if per * tmax <= 128 * 1024:
            return tmax
        return min(tmax, 3) if self.minb(requests) == 1 else 1

    def fuses_conv(self, cs: torch.Tensor, indices: torch.Tensor) -> bool:
        """At most two (8-channel group, token) units per thread, channel-innermost conv layout, state = T + 2,
        and the grid within one wave or at most four requests (beyond that a separate conv plus the verify is
        faster)."""
        requests, tmax = indices.shape
        ctas = self.ctas(requests)
        return (
            self.conv
            and tmax * (32 + 128 // ctas // 4) <= 2 * THREADS
            and (48 * ctas * requests <= self.sms or requests <= 4)
            and cs.stride(1) == 1
            and cs.shape[2] == tmax + 2
            and cs.stride(0) % 8 == 0
            and cs.stride(2) % 8 == 0
            and cs.data_ptr() % 16 == 0
        )

    def ctas(self, requests: int) -> int:
        """CTAs per V head: two at one request, one otherwise (each the faster choice)."""
        if self.fixed_ctas is not None:
            return self.fixed_ctas
        return 2 if requests == 1 else 1

    def outputs(self, rows: int):
        if rows not in self.buffers:
            self.buffers[rows] = dict(
                qout=torch.zeros(
                    rows, 48, 128, dtype=torch.float8_e4m3fn, device=self.device
                ),
                out=torch.zeros(
                    rows, 48, 128, dtype=torch.bfloat16, device=self.device
                ),
                # Per-request conv-reader tickets, 32-int stride per K head; self-resetting.
                counts=torch.zeros(
                    rows * 16 * 32, dtype=torch.int32, device=self.device
                ),
            )
        return self.buffers[rows]

    def key(self, qkv, z, state, indices, cs):
        return (
            self.ctas(indices.shape[0]),
            qkv.stride(0),
            z.stride(0),
            state.stride(0),
            indices.stride(0),
            indices.shape[1],
            self.fuses_conv(cs, indices),
            cs.stride(0),
            cs.stride(2),
            self.minb(indices.shape[0]),
            self.bulk(*indices.shape),
            self.release,
            self.journal,
            self.pdl,
            self.push,
        )

    def _args(
        self,
        rows,
        qkv,
        z,
        hidden,
        baw,
        al,
        dt,
        state,
        indices,
        cu,
        accepted,
        w,
        scale,
        cs,
        cw,
        jmode=None,
    ):
        assert hidden.is_contiguous() and baw.is_contiguous() and indices.stride(1) == 1
        assert cu.is_contiguous() and accepted.is_contiguous() and cw.is_contiguous()
        if self.journal:
            assert (
                jmode is not None
                and jmode.shape == (state.shape[0], 48)
                and jmode.is_contiguous()
            )
            assert indices.shape[1] >= 2 and 2 * indices.shape[1] * JENTRY <= 128 * 128
        b = self.outputs(rows)
        return [
            _flat(t)
            for t in (
                qkv,
                z,
                hidden,
                baw,
                al,
                dt,
                state,
                indices,
                cu,
                accepted,
                w,
                scale,
                b["qout"].view(torch.uint8),
                b["out"],
                cs,
                cw,
                b["counts"],
                jmode if self.journal else self.no_journal,
            )
        ]

    def compile(
        self,
        rows,
        qkv,
        z,
        hidden,
        baw,
        al,
        dt,
        state,
        indices,
        cu,
        accepted,
        w,
        scale,
        cs,
        cw,
        jmode=None,
    ):
        key = self.key(qkv, z, state, indices, cs)
        if key in self.compiled:
            return
        assert state.dtype == torch.float32 and state.shape[1:] == (48, 128, 128)
        assert state.stride()[1:] == (16384, 128, 1)
        assert cw.shape == (10240, 4) and cs.dtype == cw.dtype == torch.bfloat16
        args = self._args(
            rows,
            qkv,
            z,
            hidden,
            baw,
            al,
            dt,
            state,
            indices,
            cu,
            accepted,
            w,
            scale,
            cs,
            cw,
            jmode,
        )
        self.compiled[key] = cute.compile(
            launch,
            *args,
            cutlass.Int32(indices.shape[0]),
            qkv.stride(0),
            z.stride(0),
            state.stride(0),
            indices.stride(0),
            indices.shape[1],
            self.eps,
            self.ctas(indices.shape[0]),
            self.fuses_conv(cs, indices),
            cs.stride(0),
            cs.stride(2),
            self.minb(indices.shape[0]),
            self.bulk(*indices.shape),
            self.release,
            self.journal,
            self.pdl,
            self.push,
            make_fake_stream(),
        )

    def __call__(
        self,
        rows,
        requests,
        qkv,
        z,
        hidden,
        baw,
        al,
        dt,
        state,
        indices,
        cu,
        accepted,
        w,
        scale,
        cs,
        cw,
        jmode=None,
    ):
        key = self.key(qkv, z, state, indices, cs)
        args = self._args(
            rows,
            qkv,
            z,
            hidden,
            baw,
            al,
            dt,
            state,
            indices,
            cu,
            accepted,
            w,
            scale,
            cs,
            cw,
            jmode,
        )
        self.compiled[key](
            *args,
            cutlass.Int32(requests),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return self.outputs(rows)["qout"]
