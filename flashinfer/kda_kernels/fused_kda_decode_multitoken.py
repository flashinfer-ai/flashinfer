"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Fused KDA multi-token decode (conv + SiLU, gated delta-rule, gated RMSNorm).

Architecture -- one coherent design, one GPU kernel body, one launch:

  * The D x D recurrent state of one (sequence n, head h) pair is held in
    registers across the whole packed token loop, so the checkpoint pool is
    read once and only the mandatory per-token checkpoints are written back.
  * One CTA owns one sequence/head pair at ordinary grids. Low grids split the
    head's value rows over a four- or eight-CTA cluster. Thread (rg, kq) owns
    rows `rg*R ..` and k positions `jj*(4*KQ) + kq*4 + m`; state traffic stays
    fully coalesced and reductions over k use log2(KQ)-step butterflies.
  * Value rows are independent in the delta rule. Split ranks exchange only
    their per-token sums of squares for the gated RMSNorm denominator.
  * (SPLIT, R, KQ) are selected from live SM count and sequence-head geometry.
  * Each split rank computes the front-end into a small shared staging buffer.
    All ranks rendezvous after loading accepted convolution history, before
    partitioned rolling-window stores, so source/destination aliasing is safe.
"""

import functools
import math
from pathlib import Path

import cutlass
import cutlass.utils
from cutlass import cute
import cuda.bindings.driver as cuda
import torch
import tvm_ffi  # noqa: F401 -- TVM FFI supplies the active PyTorch stream

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from ..norm import utils as norm_utils

_D = 128
_W = 4
_L2_EPS = 1.0e-6
_LOG2E = 1.4426950408889634  # log2(e); exp(x) == exp2(x*log2e)
_CUTE_DSL_MODULE = "fused_kda_decode_multitoken"
_SOURCE_FILES = (
    str(Path(__file__).resolve()),
    str(Path(norm_utils.__file__).resolve()),
)


@cute.jit
def _fexp(x):
    # fast exp(x) via ex2.approx.ftz
    return cute.math.exp2(x * _LOG2E, fastmath=True)


@cute.jit
def _fsigmoid(x):
    # fast sigmoid = 1/(1+exp(-x)) using ex2.approx.ftz + rcp.approx
    return cute.arch.rcp_approx(1.0 + cute.math.exp2((-x) * _LOG2E, fastmath=True))


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _required_smem_bytes(T, R, KQ, NCH, SPLIT=1):
    """Return the statically allocated shared memory for one packed CTA."""
    threads = (((_D // SPLIT) // NCH) // R) * KQ
    warps = threads // 32
    offset = 0
    for size, alignment in (
        (T * 3 * _D * 2, 16),
        (T * _D * 4, 16),
        (T * _D * 4, 16),
        (T * 4 * 4, 16),
        (T * warps * 4, 16),
        (T * 4, 16),
        (T * 4, 16),
        (8, 8),
        (SPLIT * T * 4, 16),
    ):
        offset = _align_up(offset, alignment) + size
    return offset


def _svec(base_addr, elem_off, dtype=None):
    """4 consecutive SMEM elements as an aligned tile (one wide LDS)."""
    dtype = cutlass.Float32 if dtype is None else dtype
    w = dtype.width // 8
    return cute.make_tensor(
        cute.make_ptr(
            dtype,
            base_addr + elem_off * w,
            cutlass.AddressSpace.smem,
            assumed_align=4 * w,
        ),
        cute.make_layout(4),
    )


@cute.kernel
def _kda_kernel(
    gX: cute.Tensor,
    gW: cute.Tensor,
    gConv: cute.Tensor,
    gGate: cute.Tensor,
    gBeta: cute.Tensor,
    gAlog: cute.Tensor,
    gDtb: cute.Tensor,
    gIdx: cute.Tensor,
    gQueryStart: cute.Tensor,
    gAccepted: cute.Tensor,
    gState: cute.Tensor,
    gOG: cute.Tensor,
    gNW: cute.Tensor,
    gOut: cute.Tensor,
    T: cutlass.Constexpr,
    H: cutlass.Constexpr,
    lb: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    R: cutlass.Constexpr,
    KQ: cutlass.Constexpr,
    SPLIT: cutlass.Constexpr,
    NCH: cutlass.Constexpr,
):
    RB = _D // SPLIT
    RC = RB // NCH
    JJ = _D // (KQ * 4)
    NT = (RC // R) * KQ
    NWARP = NT // 32
    LOGKQ = int(math.log2(KQ))

    tid, _, _ = cute.arch.thread_idx()
    bx, n, _ = cute.arch.block_idx()
    h = bx // SPLIT
    rank = bx % SPLIT
    row0 = rank * RB

    dim = H * _D
    scale = 1.0 / math.sqrt(_D)

    smem = cutlass.utils.SmemAllocator()
    sQKV = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout(T * 3 * _D), byte_alignment=16
    )
    sDec = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(T * _D), byte_alignment=16
    )
    sOut = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(T * RB), byte_alignment=16
    )
    sRed = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(T * 4), byte_alignment=16
    )
    sRed2 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(T * NWARP), byte_alignment=16
    )
    sBeta = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(T), byte_alignment=16
    )
    sTot = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T), byte_alignment=16)
    # per-token partial sums-of-squares exchanged with the cluster peers
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=8)
    mail_ptr = smem.allocate_array(cutlass.Float32, SPLIT * T, byte_alignment=16)
    sMail = cute.make_tensor(mail_ptr, cute.make_layout(SPLIT * T))

    rg = tid // KQ  # row group: owns rows rg*R .. rg*R+R-1
    kq = tid % KQ  # lane group inside the row
    warp = tid // 32
    lane = tid % 32

    bos = gQueryStart[n]
    eos = gQueryStart[n + 1]
    seq_len = eos - bos
    accepted = cute.min(
        cute.max(gAccepted[n] - cutlass.Int32(1), cutlass.Int32(0)),
        cutlass.Int32(T - 1),
    )
    conv_slot = gIdx[n, 0]
    state_slot = gIdx[n, accepted]
    livef = (
        cute.min(
            cute.max(conv_slot.to(cutlass.Float32), cutlass.Float32(0.0)),
            cutlass.Float32(1.0),
        )
        * cute.min(
            cute.max(state_slot.to(cutlass.Float32), cutlass.Float32(0.0)),
            cutlass.Float32(1.0),
        )
        * cute.min(
            cute.max(seq_len.to(cutlass.Float32), cutlass.Float32(0.0)),
            cutlass.Float32(1.0),
        )
    )
    src = cute.max(state_slot, cutlass.Int32(0))
    conv_src = cute.max(conv_slot, cutlass.Int32(0))
    live = livef > 0.5
    slots = [gIdx[n, t] for t in range(T)]
    p_state = gState.iterator.toint()
    state_page = gState.stride[0]

    if cutlass.const_expr(SPLIT > 1):
        # Arm the cluster mailbox up front: at kernel entry the peers are still
        # in lockstep, so this rendezvous is nearly free, whereas the same
        # rendezvous placed at the reduction would sit on the critical path.
        if tid == 0:
            cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    # When the CTA's rows fit in a single register chunk there is no chunk loop
    # to overlap global latency with, so the two loads that are only consumed
    # at the very end (output gate, norm weight) and the checkpoint read itself
    # are issued up front, before the front end.
    NEP = (T * RB + NT - 1) // NT
    if cutlass.const_expr(NCH == 1):
        ogv = []
        nwv = []
        for blk in cutlass.range_constexpr(NEP):
            pidx = cute.min(blk * NT + tid, cutlass.Int32(T * RB - 1))
            pvv = row0 + pidx % RB
            og = cutlass.Float32(0.0)
            if seq_len > 0:
                token = cute.min(pidx // RB, seq_len - 1)
                og = _fsigmoid(gOG[bos + token, h * _D + pvv].to(cutlass.Float32))
            ogv.append(og)
            nwv.append(gNW[pvv])

    src_base = p_state + src.to(cutlass.Int64) * (state_page * 4)
    if cutlass.const_expr(NCH == 1):
        sfrg0 = [
            cute.make_rmem_tensor(cute.make_layout((4, JJ)), cutlass.Float32)
            for _ in range(R)
        ]
        lane_off0 = [
            ((h * _D + row0 + rg * R + r) * _D + kq * 4).to(cutlass.Int64) * 4
            for r in range(R)
        ]
        for r in cutlass.range_constexpr(R):
            cute.autovec_copy(
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        src_base + lane_off0[r],
                        cutlass.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4, JJ), stride=(1, KQ * 4)),
                ),
                sfrg0[r],
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
            )

    # ========== stage 1: causal conv + SiLU, decay, beta, checkpoints ========
    # 4*D "channel tasks": 3*D convolution channels then D decay channels.
    for cb in cutlass.range_constexpr((4 * _D + NT - 1) // NT):
        c = cb * NT + tid
        ci = cutlass.Int32(0)
        cd = cutlass.Int32(0)
        chan = cutlass.Int32(0)
        hh0 = cutlass.Float32(0.0)
        hh1 = cutlass.Float32(0.0)
        hh2 = cutlass.Float32(0.0)
        tp0 = cutlass.Float32(0.0)
        tp1 = cutlass.Float32(0.0)
        tp2 = cutlass.Float32(0.0)
        tp3 = cutlass.Float32(0.0)
        if c < 3 * _D:
            ci = c // _D
            cd = c % _D
            chan = ci * dim + h * _D + cd
            hh0 = gConv[conv_src, chan, accepted].to(cutlass.Float32)
            hh1 = gConv[conv_src, chan, accepted + 1].to(cutlass.Float32)
            hh2 = gConv[conv_src, chan, accepted + 2].to(cutlass.Float32)

            tp0 = gW[ci, 0, h * _D + cd]
            tp1 = gW[ci, 1, h * _D + cd]
            tp2 = gW[ci, 2, h * _D + cd]
            tp3 = gW[ci, 3, h * _D + cd]

        if cutlass.const_expr(SPLIT > 1):
            # All cluster ranks read the shared convolution source. No rank may
            # overwrite the rolling window until every peer has captured its
            # accepted history in registers.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        if c < 3 * _D:
            # the conv history bytes are partitioned over the cluster ranks
            mine = ((c * SPLIT) // (3 * _D)) == rank

            if live:
                if mine:
                    gConv[conv_src, chan, 0] = hh1.to(cutlass.BFloat16)
                    gConv[conv_src, chan, 1] = hh2.to(cutlass.BFloat16)

            for t in cutlass.range_constexpr(T):
                if t < seq_len:
                    xv = gX[bos + t, chan].to(cutlass.Float32)
                    mixed = tp0 * hh0 + tp1 * hh1 + tp2 * hh2 + tp3 * xv
                    mixed = mixed * _fsigmoid(mixed)
                    sQKV[t * (3 * _D) + c] = mixed.to(cutlass.BFloat16)
                    hh0 = hh1
                    hh1 = hh2
                    hh2 = xv

                    if live:
                        if mine:
                            gConv[conv_src, chan, t + 2] = xv.to(cutlass.BFloat16)
        elif c < 4 * _D:
            dd = c - 3 * _D
            a_h = _fexp(gAlog[h])
            dtb = gDtb[h * _D + dd]
            for t in cutlass.range_constexpr(T):
                if t < seq_len:
                    gi = gGate[bos + t, h * _D + dd].to(cutlass.Float32) + dtb
                    sDec[t * _D + dd] = _fexp(lb * _fsigmoid(a_h * gi))

    if tid < T:
        if tid < seq_len:
            sBeta[tid] = _fsigmoid(gBeta[bos + tid, h].to(cutlass.Float32))

    cute.arch.barrier()

    # ---- L2 norms of q, k and their cross term: one warp per token, each
    # ---- lane owning 4 channels, so the whole 128-wide reduction is a single
    # ---- butterfly and the token loop below reads 3 ready-made scalars
    # ---- instead of re-reducing 12 partials per token.
    a_qkv = sQKV.iterator.toint()
    for nb in cutlass.range_constexpr((T * 32 + NT - 1) // NT):
        idx = nb * NT + tid
        if idx < T * 32:
            t = idx // 32
            l = idx % 32
            if t < seq_len:
                nq4 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
                nk4 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
                cute.autovec_copy(
                    _svec(a_qkv, t * (3 * _D) + l * 4, cutlass.BFloat16), nq4
                )
                cute.autovec_copy(
                    _svec(a_qkv, t * (3 * _D) + _D + l * 4, cutlass.BFloat16),
                    nk4,
                )
                sqq = cutlass.Float32(0.0)
                skk = cutlass.Float32(0.0)
                skq = cutlass.Float32(0.0)
                for m in cutlass.range_constexpr(4):
                    qv = nq4[m].to(cutlass.Float32)
                    kv = nk4[m].to(cutlass.Float32)
                    sqq = sqq + qv * qv
                    skk = skk + kv * kv
                    skq = skq + kv * qv
                for off in cutlass.range_constexpr(5):
                    sqq = sqq + cute.arch.shuffle_sync_bfly(sqq, 1 << off)
                    skk = skk + cute.arch.shuffle_sync_bfly(skk, 1 << off)
                    skq = skq + cute.arch.shuffle_sync_bfly(skq, 1 << off)
                if l == 0:
                    sRed[t * 4] = (
                        cute.rsqrt(sqq + _L2_EPS, approx=True, ftz=True) * scale
                    )
                    sRed[t * 4 + 1] = cute.rsqrt(skk + _L2_EPS, approx=True, ftz=True)
                    sRed[t * 4 + 2] = skq
    cute.arch.barrier()

    # ========== stage 2: register-resident gated delta-rule recurrence =======
    # The RB rows this CTA owns are walked in NCH register chunks of RC rows.
    # NCH trades register residency for occupancy: the chunk state is what sits
    # in registers, so raising NCH shrinks the per-thread tile and lets more
    # CTAs (more memory-level parallelism) sit on an SM at once.  Rows are
    # independent under the delta rule, so the chunk order is immaterial.
    a_dec = sDec.iterator.toint()
    rtot = [cutlass.Float32(0.0) for _ in range(T)]

    for ch in cutlass.range_constexpr(NCH):
        if cutlass.const_expr(NCH == 1):
            sfrg = sfrg0
            lane_off = lane_off0
        else:
            sfrg = [
                cute.make_rmem_tensor(cute.make_layout((4, JJ)), cutlass.Float32)
                for _ in range(R)
            ]
            lane_off = [
                ((h * _D + row0 + ch * RC + rg * R + r) * _D + kq * 4).to(cutlass.Int64)
                * 4
                for r in range(R)
            ]
            for r in cutlass.range_constexpr(R):
                gtile = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        src_base + lane_off[r],
                        cutlass.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4, JJ), stride=(1, KQ * 4)),
                )
                cute.autovec_copy(
                    gtile,
                    sfrg[r],
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                )

        for t in cutlass.range_constexpr(T):
            if t < seq_len:
                qb = t * (3 * _D)
                db = t * _D
                rn_q = sRed[t * 4]
                rn_k = sRed[t * 4 + 1]
                bkq = sRed[t * 4 + 2]
                bt = sBeta[t]

                # One pass over the state: decay it and contract it with BOTH the
                # raw key and the raw query.  Because the update is s + d*K, the
                # recurrent readout is <s,Q> + d*<K,Q>, so a single reduction per
                # token suffices.
                acc = [cutlass.Float32(0.0) for _ in range(R)]
                aq = [cutlass.Float32(0.0) for _ in range(R)]
                kreg = [
                    cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
                    for _ in range(JJ)
                ]
                for jj in cutlass.range_constexpr(JJ):
                    k0 = jj * (KQ * 4) + kq * 4
                    d4 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
                    q4 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
                    cute.autovec_copy(_svec(a_dec, db + k0), d4)
                    cute.autovec_copy(
                        _svec(a_qkv, qb + _D + k0, cutlass.BFloat16), kreg[jj]
                    )
                    cute.autovec_copy(_svec(a_qkv, qb + k0, cutlass.BFloat16), q4)
                    for m in cutlass.range_constexpr(4):
                        dc = d4[m]
                        kv = kreg[jj][m].to(cutlass.Float32)
                        qv = q4[m].to(cutlass.Float32)
                        for r in cutlass.range_constexpr(R):
                            s = sfrg[r][m, jj] * dc
                            sfrg[r][m, jj] = s
                            acc[r] = acc[r] + s * kv
                            aq[r] = aq[r] + s * qv

                d2 = []
                rec = []
                for r in cutlass.range_constexpr(R):
                    a = acc[r]
                    b = aq[r]
                    for off in cutlass.range_constexpr(LOGKQ):
                        a = a + cute.arch.shuffle_sync_bfly(a, 1 << off)
                        b = b + cute.arch.shuffle_sync_bfly(b, 1 << off)
                    dd2 = (
                        (
                            sQKV[qb + 2 * _D + row0 + ch * RC + rg * R + r].to(
                                cutlass.Float32
                            )
                            - a * rn_k
                        )
                        * bt
                        * rn_k
                    )
                    d2.append(dd2)
                    rec.append((b + dd2 * bkq) * rn_q)

                for jj in cutlass.range_constexpr(JJ):
                    for m in cutlass.range_constexpr(4):
                        kv = kreg[jj][m].to(cutlass.Float32)
                        for r in cutlass.range_constexpr(R):
                            sfrg[r][m, jj] = sfrg[r][m, jj] + d2[r] * kv

                rsum = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(R):
                    # Match the composed vLLM path and the legacy fused kernel:
                    # recurrent output is materialized as BF16 before RMSNorm.
                    x = rec[r].to(cutlass.BFloat16).to(cutlass.Float32) * livef
                    rsum = rsum + x * x
                    if kq == 0:
                        sOut[t * RB + ch * RC + rg * R + r] = x
                # rsum is uniform inside each KQ-lane group, so only the groups of
                # the warp still have to be combined to get this warp's row sum.
                for off in cutlass.range_constexpr(LOGKQ, 5):
                    rsum = rsum + cute.arch.shuffle_sync_bfly(rsum, 1 << off)
                rtot[t] = rtot[t] + rsum

                if live:
                    if slots[t] > 0:
                        dst_base = p_state + slots[t].to(cutlass.Int64) * (
                            state_page * 4
                        )
                        for r in cutlass.range_constexpr(R):
                            gwtile = cute.make_tensor(
                                cute.make_ptr(
                                    cutlass.Float32,
                                    dst_base + lane_off[r],
                                    cutlass.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((4, JJ), stride=(1, KQ * 4)),
                            )
                            cute.autovec_copy(
                                sfrg[r],
                                gwtile,
                                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                            )

    if lane == 0:
        for t in cutlass.range_constexpr(T):
            sRed2[t * NWARP + warp] = rtot[t]

    cute.arch.barrier()

    # ---- collapse the per-warp partials to one per token, then (when the
    # ---- head is spread over a cluster) add in the peer ranks' partials
    # ---- straight out of distributed shared memory.
    if cutlass.const_expr(SPLIT > 1):
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, SPLIT * T * 4)
        cute.arch.barrier()
        if tid < T:
            tot = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(NWARP):
                tot = tot + sRed2[tid * NWARP + w]
            slot = mail_ptr + (rank * T + tid)
            for p in cutlass.range_constexpr(SPLIT):
                norm_utils.store_shared_remote(tot, slot, mbar, cutlass.Int32(p))
        cute.arch.fence_acq_rel_cta()
        if tid == 0:
            cute.arch.mbarrier_wait(mbar, 0)
        cute.arch.barrier()
        if tid < T:
            tot = cutlass.Float32(0.0)
            for p in cutlass.range_constexpr(SPLIT):
                tot = tot + sMail[p * T + tid]
            sTot[tid] = cute.rsqrt(tot * (1.0 / _D) + eps, approx=True, ftz=True)
        cute.arch.barrier()
    else:
        if tid < T:
            tot = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(NWARP):
                tot = tot + sRed2[tid * NWARP + w]
            sTot[tid] = cute.rsqrt(tot * (1.0 / _D) + eps, approx=True, ftz=True)
        cute.arch.barrier()

    # ========== stage 3: gated RMSNorm epilogue =============================
    # one thread per (token, row) so the bf16 output store is fully coalesced
    if cutlass.const_expr(NCH == 1):
        for blk in cutlass.range_constexpr(NEP):
            idx = blk * NT + tid
            if idx < T * RB:
                token = idx // RB
                if token < seq_len:
                    gOut[bos + token, h * _D + row0 + idx % RB] = (
                        sOut[idx] * sTot[token] * nwv[blk] * ogv[blk]
                    ).to(cutlass.BFloat16)
    else:
        for blk in cutlass.range_constexpr(NEP):
            idx = blk * NT + tid
            if idx < T * RB:
                ete = idx // RB
                evv = row0 + idx % RB
                if ete < seq_len:
                    eog = _fsigmoid(gOG[bos + ete, h * _D + evv].to(cutlass.Float32))
                    gOut[bos + ete, h * _D + evv] = (
                        sOut[idx] * sTot[ete] * gNW[evv] * eog
                    ).to(cutlass.BFloat16)


@cute.jit
def _kda_launch(
    m_x,
    m_w,
    m_conv,
    m_gate,
    m_beta,
    m_alog,
    m_dtb,
    m_idx,
    m_query_start,
    m_accepted,
    m_state,
    m_og,
    m_nw,
    m_out,
    x_row_stride: cutlass.Int64,
    beta_row_stride: cutlass.Int64,
    output_gate_row_stride: cutlass.Int64,
    stream: cuda.CUstream,
    T: cutlass.Constexpr,
    H: cutlass.Constexpr,
    lb: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    R: cutlass.Constexpr,
    KQ: cutlass.Constexpr,
    bpm: cutlass.Constexpr,
    SPLIT: cutlass.Constexpr,
    NCH: cutlass.Constexpr,
):
    p_x = m_x.iterator.toint()
    p_w = m_w.iterator.toint()
    p_gate = m_gate.iterator.toint()
    p_beta = m_beta.iterator.toint()
    p_alog = m_alog.iterator.toint()
    p_dtb = m_dtb.iterator.toint()
    p_idx = m_idx.iterator.toint()
    p_og = m_og.iterator.toint()
    p_nw = m_nw.iterator.toint()
    p_out = m_out.iterator.toint()
    dim = H * _D
    BF = cutlass.BFloat16
    F32 = cutlass.Float32
    I32 = cutlass.Int32
    GM = cutlass.AddressSpace.gmem
    N = m_idx.shape[0]
    num_rows = m_x.shape[0]

    def _t(dtype, addr, shape, stride):
        return cute.make_tensor(
            cute.make_ptr(dtype, addr, GM, assumed_align=16),
            cute.make_layout(shape, stride=stride),
        )

    gX = _t(BF, p_x, (num_rows, 3 * dim), (x_row_stride, 1))
    gW = _t(F32, p_w, (3, _W, dim), (_W * dim, dim, 1))
    gConv = m_conv
    gGate = _t(BF, p_gate, (num_rows, dim), (dim, 1))
    gBeta = _t(BF, p_beta, (num_rows, H), (beta_row_stride, 1))
    gAlog = _t(F32, p_alog, (H,), (1,))
    gDtb = _t(F32, p_dtb, (dim,), (1,))
    gIdx = _t(I32, p_idx, (N, T), (T, 1))
    gQueryStart = m_query_start
    gAccepted = m_accepted
    gOG = _t(BF, p_og, (num_rows, dim), (output_gate_row_stride, 1))
    gNW = _t(F32, p_nw, (_D,), (1,))
    gOut = _t(BF, p_out, (num_rows, dim), (dim, 1))

    NTHR = (((_D // SPLIT) // NCH) // R) * KQ
    kernel = _kda_kernel(
        gX,
        gW,
        gConv,
        gGate,
        gBeta,
        gAlog,
        gDtb,
        gIdx,
        gQueryStart,
        gAccepted,
        m_state,
        gOG,
        gNW,
        gOut,
        T,
        H,
        lb,
        eps,
        R,
        KQ,
        SPLIT,
        NCH,
    )
    if cutlass.const_expr(SPLIT > 1):
        kernel.launch(
            grid=(H * SPLIT, N, 1),
            block=(NTHR, 1, 1),
            cluster=(SPLIT, 1, 1),
            stream=stream,
        )
    else:
        kernel.launch(
            grid=(H, N, 1),
            block=(NTHR, 1, 1),
            min_blocks_per_mp=bpm,
            stream=stream,
        )


# (rows per thread, k-lane groups, min CTAs per SM, row chunks per CTA).
# NCH walks the CTA's rows in register chunks: raising it shrinks the
# per-thread tile and buys occupancy (memory-level parallelism on the
# checkpoint stream) at the price of re-reading the staged decay/key/query
# once per chunk and token, so the chunk count is bounded by the token loop.
_TILE_TALL = (2, 16, 1, 1)
_TILE_T1 = (2, 16, 4, 4)
_TILE_WIDE_SHORT = (1, 16, 4, 4)
_TILE_WIDE_LONG = (2, 16, 3, 2)
# Split tiles must keep NTHR == 4*D. The pre-store cluster rendezvous is safe
# because every convolution-history channel is loaded in that single pass.
_TILE_SPLIT = {
    4: (1, 16, 1, 1),
    8: (1, 32, 1, 1),
}


def _pick_split(sequence_heads, sm_count):
    if sequence_heads * 8 <= sm_count:
        return 8
    if sequence_heads * 4 <= sm_count:
        return 4
    return 1


def _compact_fake(shape, dtype):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        assumed_align=16,
        stride_order=tuple(reversed(range(len(shape)))),
    )


@functools.cache
def _get_compiled_kernel(
    T,
    H,
    lower_bound,
    norm_eps,
    R,
    KQ,
    bpm,
    NCH,
    SPLIT,
):
    dim = H * _D
    sequences = cute.sym_int()
    boundaries = cute.sym_int()
    rows = cute.sym_int()
    slots = cute.sym_int()
    fake_inputs = (
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (rows, 3 * dim),
            (cute.sym_int64(), 1),
            assumed_align=16,
        ),
        _compact_fake((3, _W, dim), cutlass.Float32),
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (slots, 3 * dim, T + _W - 2),
            (cute.sym_int64(divisibility=16), 1, 3 * dim),
            assumed_align=16,
        ),
        _compact_fake((1, rows, H, _D), cutlass.BFloat16),
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (1, rows, H),
            (cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=16,
        ),
        _compact_fake((H,), cutlass.Float32),
        _compact_fake((dim,), cutlass.Float32),
        _compact_fake((sequences, T), cutlass.Int32),
        _compact_fake((boundaries,), cutlass.Int32),
        _compact_fake((sequences,), cutlass.Int32),
        cute.runtime.make_fake_tensor(
            cutlass.Float32,
            (slots, H, _D, _D),
            (cute.sym_int64(divisibility=16), _D * _D, _D, 1),
            assumed_align=16,
        ),
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (rows, H, _D),
            (cute.sym_int64(), _D, 1),
            assumed_align=16,
        ),
        _compact_fake((_D,), cutlass.Float32),
        _compact_fake((1, rows, H, _D), cutlass.BFloat16),
    )
    kernel_name = (
        f"d128_w4_h{H}_t{T}"
        f"_lb{str(lower_bound).replace('.', '_').replace('-', 'm')}"
        f"_eps{str(norm_eps).replace('.', '_').replace('-', 'm')}"
        f"_r{R}_kq{KQ}_bpm{bpm}_nch{NCH}_split{SPLIT}"
    )
    return build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        kernel_name,
        lambda: cute.compile(
            _kda_launch,
            *fake_inputs,
            1,
            1,
            1,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            T,
            H,
            lower_bound,
            norm_eps,
            R,
            KQ,
            bpm,
            SPLIT,
            NCH,
            options="--enable-tvm-ffi --generate-line-info",
        ),
        extra_key_files=_SOURCE_FILES,
    )


def _run_fused_kda_decode_multitoken(
    x,
    weight,
    conv_state,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state_indices,
    query_start_loc,
    num_accepted_tokens,
    state,
    output_gate,
    norm_weight,
    lower_bound,
    norm_eps,
    out,
):
    key = (
        state_indices.shape[1],
        A_log.shape[0],
        float(lower_bound),
        float(norm_eps),
    )
    T, H, lower_bound, norm_eps = key
    N = state_indices.shape[0]
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    SPLIT = _pick_split(N * H, sm_count)
    if T == 1:
        SPLIT = 1
        R, KQ, bpm, NCH = _TILE_T1
    elif SPLIT > 1:
        R, KQ, bpm, NCH = _TILE_SPLIT[SPLIT]
    elif 2 * sm_count > N * H:
        R, KQ, bpm, NCH = _TILE_TALL
    elif T <= 3:
        R, KQ, bpm, NCH = _TILE_WIDE_SHORT
    else:
        R, KQ, bpm, NCH = _TILE_WIDE_LONG
    properties = torch.cuda.get_device_properties(x.device)
    required_smem = _required_smem_bytes(T, R, KQ, NCH, SPLIT)
    if required_smem > properties.shared_memory_per_block_optin:
        raise ValueError(
            f"T={T} requires {required_smem} bytes of shared memory, but "
            f"{x.device} supports {properties.shared_memory_per_block_optin} bytes per block"
        )
    entry = _get_compiled_kernel(
        T,
        H,
        lower_bound,
        norm_eps,
        R,
        KQ,
        bpm,
        NCH,
        SPLIT,
    )

    entry(
        x,
        weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        state,
        output_gate,
        norm_weight,
        out,
        x.stride(0),
        raw_beta.stride(1),
        output_gate.stride(0),
    )


def _check_cuda_tensor(name, tensor, dtype):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


@torch.no_grad()
def run_fused_kda_decode_multitoken(
    x,
    weight,
    conv_state,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state_indices,
    query_start_loc,
    num_accepted_tokens,
    state,
    output_gate,
    norm_weight,
    lower_bound=-5.0,
    norm_eps=1e-5,
    output=None,
):
    """Run packed fused KDA and write one recurrent checkpoint per token.

    ``query_start_loc`` supplies each sequence's packed token range and
    ``num_accepted_tokens`` selects the source checkpoint and convolution
    history offset from the preceding verification window. Zero-length or
    non-positive rows do not mutate the cache. Destination stores are guarded
    per token as a final safety net.
    """
    _check_cuda_tensor("x", x, torch.bfloat16)
    _check_cuda_tensor("weight", weight, torch.float32)
    _check_cuda_tensor("conv_state", conv_state, torch.bfloat16)
    _check_cuda_tensor("raw_gate", raw_gate, torch.bfloat16)
    _check_cuda_tensor("raw_beta", raw_beta, torch.bfloat16)
    _check_cuda_tensor("A_log", A_log, torch.float32)
    _check_cuda_tensor("dt_bias", dt_bias, torch.float32)
    _check_cuda_tensor("state_indices", state_indices, torch.int32)
    _check_cuda_tensor("query_start_loc", query_start_loc, torch.int32)
    _check_cuda_tensor("num_accepted_tokens", num_accepted_tokens, torch.int32)
    _check_cuda_tensor("state", state, torch.float32)
    _check_cuda_tensor("output_gate", output_gate, torch.bfloat16)
    _check_cuda_tensor("norm_weight", norm_weight, torch.float32)

    major, _ = torch.cuda.get_device_capability(x.device)
    if major != 10:
        raise NotImplementedError("packed fused KDA T>1 currently requires SM10x")

    if state_indices.ndim != 2 or not state_indices.is_contiguous():
        raise ValueError("state_indices must be contiguous with shape [N, T]")
    num_sequences, num_tokens = state_indices.shape
    if num_sequences <= 0 or num_tokens <= 0:
        raise ValueError("state_indices must contain at least one sequence and token")
    if x.ndim != 2 or x.shape[0] > num_sequences * num_tokens:
        raise ValueError("x must have at most N * T packed rows")
    if x.shape[1] % (3 * _D) or x.stride(1) != 1:
        raise ValueError("x must have a contiguous channel dimension of 3 * H * 128")
    num_heads = x.shape[1] // (3 * _D)
    hidden_size = num_heads * _D
    if num_heads not in (12, 24, 32, 48, 96):
        raise ValueError("H must be one of 12, 24, 32, 48, or 96")
    if weight.shape != (3, _W, hidden_size) or not weight.is_contiguous():
        raise ValueError("weight must be contiguous with shape [3, 4, H * 128]")
    if (
        conv_state.ndim != 3
        or conv_state.shape[1:] != (3 * hidden_size, num_tokens + _W - 2)
        or conv_state.stride(1) != 1
        or conv_state.stride(2) != 3 * hidden_size
    ):
        raise ValueError("conv_state must use the extended paged SD cache layout")
    num_rows = x.shape[0]
    if raw_gate.shape != (1, num_rows, num_heads, _D):
        raise ValueError("raw_gate must have shape [1, num_rows, H, 128]")
    if not raw_gate.is_contiguous():
        raise ValueError("raw_gate must be contiguous")
    if raw_beta.shape != (1, num_rows, num_heads) or raw_beta.stride(2) != 1:
        raise ValueError("raw_beta must have shape [1, num_rows, H]")
    if (
        query_start_loc.shape != (num_sequences + 1,)
        or not query_start_loc.is_contiguous()
    ):
        raise ValueError("query_start_loc must be contiguous with shape [N + 1]")
    if (
        num_accepted_tokens.shape != (num_sequences,)
        or not num_accepted_tokens.is_contiguous()
    ):
        raise ValueError("num_accepted_tokens must be contiguous with shape [N]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must be contiguous with shape [H]")
    if dt_bias.shape != (hidden_size,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must be contiguous with shape [H * 128]")
    if (
        state.ndim != 4
        or state.shape[1:] != (num_heads, _D, _D)
        or state.stride()[1:] != (_D * _D, _D, 1)
    ):
        raise ValueError("state must have shape [slots, H, 128, 128]")
    if conv_state.shape[0] != state.shape[0]:
        raise ValueError("conv_state and state must have matching slot counts")
    if output_gate.ndim == 4:
        if output_gate.shape[0] != 1:
            raise ValueError("4D output_gate must have leading dimension one")
        output_gate = output_gate[0]
    if output_gate.shape != (num_rows, num_heads, _D):
        raise ValueError("output_gate must have shape [num_rows, H, 128]")
    if output_gate.stride(2) != 1 or output_gate.stride(1) != _D:
        raise ValueError("output_gate must be contiguous within each token")
    if norm_weight.shape != (_D,) or not norm_weight.is_contiguous():
        raise ValueError("norm_weight must be contiguous with shape [128]")
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("lower_bound must be a finite negative float")
    if not math.isfinite(norm_eps) or norm_eps < 0:
        raise ValueError("norm_eps must be finite and non-negative")

    tensors = (
        weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        state,
        output_gate,
        norm_weight,
    )
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")
    expected_output_shape = (1, num_rows, num_heads, _D)
    if output is None:
        output = torch.empty(
            expected_output_shape, dtype=torch.bfloat16, device=x.device
        )
    else:
        _check_cuda_tensor("output", output, torch.bfloat16)
        if output.shape != expected_output_shape or not output.is_contiguous():
            raise ValueError(
                "output must be contiguous with shape [1, num_rows, H, 128]"
            )

    _run_fused_kda_decode_multitoken(
        x,
        weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        state,
        output_gate,
        norm_weight,
        float(lower_bound),
        float(norm_eps),
        output,
    )
    return output
