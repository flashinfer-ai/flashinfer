# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120/SM121 routed-expert MoE with NVFP4 activations and NVFP4 expert weights.

Weights are read in place from the canonical NVFP4 storage shared with the
W4A16 kernels in ``moe_nvfp4_w4a16``: packed E2M1 nibbles (``[gate; up]`` rows
for SwiGLU), 128x4-swizzled E4M3 block scales and a per-expert FP32 global
scale. Nothing is repacked.

All launches run on the caller's stream without host synchronization:

1. route: histogram, scan and scatter of (token, expert) pairs into expert-major
   order plus the ragged M-tile work list of both grouped GEMMs; the other CTAs
   quantize ``hidden_states`` to NVFP4 once per token and zero the output;
2. GEMM1: grouped NVFP4 x NVFP4 tensor-core GEMM (K = H); for SwiGLU the
   activation and the NVFP4 quantization of the intermediate are fused into its
   epilogue;
3. ReLU2 only: activation and NVFP4 quantization of the intermediate;
4. GEMM2: grouped NVFP4 x NVFP4 GEMM (K = I, N = H) whose router-weighted BF16
   partials are reduced into the output.

Activations use 16-element E4M3 block scales, ``s = e4m3(min(amax * g / 6,
448))``, with a per-layer global scale ``g``. A slot whose expert id is outside
``[0, E)`` (padding or a non-local expert) is dropped by the router and adds
nothing to the output.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims

FP4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32
I64 = cutlass.Int64
U8 = cutlass.Uint8
U32 = cutlass.Uint32

BN = 128
NTHR = 256
NTR = 256


_PTX_F4X8 = (
    "{\n.reg .b8 b0,b1,b2,b3;\n"
    "cvt.rn.satfinite.e2m1x2.f32 b0, {$r1}, {$r0};\n"
    "cvt.rn.satfinite.e2m1x2.f32 b1, {$r3}, {$r2};\n"
    "cvt.rn.satfinite.e2m1x2.f32 b2, {$r5}, {$r4};\n"
    "cvt.rn.satfinite.e2m1x2.f32 b3, {$r7}, {$r6};\n"
    "mov.b32 {$w0}, {b0,b1,b2,b3};\n}\n"
)


_PTX_RED_V4 = (
    "{\n.reg .b64 pol;\n"
    "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
    "red.global.add.noftz.L2::cache_hint.v4.bf16x2 [{$r0}], "
    "{ {$r1},{$r2},{$r3},{$r4} }, pol;\n}\n"
)


def _red_v4(addr, a, b, c, d):
    """One 16B vector reduction == four packed bf16x2 adds (sm_90+ PTX 8.1)."""
    prims.inline_ptx_hl(_PTX_RED_V4, read_only_args=[addr, a, b, c, d])


def _sw(ptr, word_off, align=16):
    return cute.make_ptr(
        U32, ptr.toint() + word_off * 4, cute.AddressSpace.smem, assumed_align=align
    )


def _gp(t, dtype, byte_off, align=16):
    return cute.make_ptr(
        dtype,
        t.iterator.toint() + byte_off,
        cute.AddressSpace.gmem,
        assumed_align=align,
    )


def _pack_fp4x8(v, off, r):
    """Pack 8 scaled fp32 values into one uint32 of E2M1 nibbles (low nibble = even index)."""
    return prims.inline_ptx_hl(
        _PTX_F4X8,
        write_only_types=[I32],
        read_only_args=[v[off + i] * r for i in range(8)],
    )


def _e4m3_byte(x):
    p = prims.inline_ptx_hl(
        "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};\n",
        write_only_types=[cutlass.Int16],
        read_only_args=[x, x],
    )
    return I32(p) & 0xFF


def _sf_mma_layout(rows, TK, stages):
    """SMEM layout of a `rows` x TK scale-factor tile in the canonical 128x4 swizzle.

    Packed to exactly `rows` rows so a 32- or 64-row M tile does not reserve the
    128-row footprint; that shared memory buys extra mainloop stages instead.
    """
    nk = TK // 64
    unit = rows * 4
    return cute.make_layout(
        ((32, rows // 32), (16, 4, nk), stages),
        stride=((4 * (rows // 32), 4), (0, 1, unit), nk * unit),
    )


def _build(T, H, I, N13, E, TOPK, swiglu):
    P = T * TOPK
    # Rows per expert average P/E (about 20 for a 1024-token prefill with 512
    # experts and top-10), so a 32-row M tile keeps one tile per expert (identical weight traffic) while halving
    # the padded tensor-core work relative to a 64-row tile.
    BM = 32 if P <= 24 * E else 64
    PA = P + BM

    # Widest K tile that still leaves room for two pipeline stages: a bigger K tile
    # means fewer, larger per-row weight requests and fewer mainloop barriers.
    def _per_stage(TK):
        # A and B values plus the B block scales; the activation block scales are
        # staged once for the whole K instead of once per pipeline stage.
        nk = TK // 64
        return (BM + BN) * TK // 2 + nk * BN * 4

    def _sfa_bytes(K):
        return (K // 64) * BM * 4

    def _epi_pad(TK, ST, K, panel):
        # The epilogue panel is only written after a barrier that retires every
        # mainloop MMA, so it overlays the scale staging; only a shortfall is paid.
        return max(0, panel - (_sfa_bytes(K) + (TK // 64) * BN * 4 * ST))

    def _total(TK, ST, K, panel, extra):
        return (
            _per_stage(TK) * ST
            + _sfa_bytes(K)
            + _epi_pad(TK, ST, K, panel)
            + extra
            + 1024
        )

    def _pick(K, panel, extra, budget):
        for tk in (512, 256, 128, 64):
            if K % tk == 0 and _total(tk, 2, K, panel, extra) <= budget:
                st = 2
                while (
                    st + 1 <= min(8, K // tk + 1)
                    and _total(tk, st + 1, K, panel, extra) <= budget
                ):
                    st += 1
                return tk, st
        return 64, 2

    PANEL1 = BM * 128 if swiglu else 0
    PANEL2 = BM * 128
    # GEMM1 keeps the widest K tile it can: long contiguous weight bursts matter
    # more there than occupancy.  GEMM2's K is short enough that its whole work item
    # already fits in one stage set, so the block has no intra-block load/compute
    # overlap at all; sizing it under half the 102400 B carveout lets two CTAs share
    # an SM and cover each other's prologue, MMA and epilogue phases.
    TK1, ST1 = _pick(H, PANEL1, 128, 98000)
    TK2, ST2 = _pick(I, PANEL2, 256, 49152)
    if _total(TK2, ST2, I, PANEL2, 256) > 49152:
        TK2, ST2 = _pick(I, PANEL2, 256, 98000)
    EPI1 = _epi_pad(TK1, ST1, H, PANEL1)
    EPI2 = _epi_pad(TK2, ST2, I, PANEL2)
    NT1 = (I // 64) if swiglu else ((N13 + BN - 1) // BN)
    NT2 = H // BN
    WMAX = min(E, P) + (P + BM - 1) // BM
    # WMAX is a worst-case bound on sum_e ceil(rows_e / BM); for the routing the
    # model actually produces it is ~40% too large, and with ~96KB of smem per CTA
    # every surplus block still occupies an SM for a launch plus an L2 read.  Launch
    # the expected count and let blocks grid-stride over any overflow.
    GY = min(WMAX, min(E, P) + max(1, (P + 8 * BM - 1) // (8 * BM)))
    ECH = (E + NTR - 1) // NTR
    NTR_LOG = 8
    N13_PAD = (N13 + 127) // 128 * 128
    mma_op_cls = cute.nvgpu.warp.MmaMXF4NVF4Op

    warp_mnk = (2, 4, 1) if BM == 32 else (4, 2, 1)

    def make_mma(TK):
        perm = sm120_utils.get_permutation_mnk((BM, BN, TK), 16, False)
        return cute.make_tiled_mma(
            mma_op_cls(FP4, F32, E4M3), cute.make_layout(warp_mnk), permutation_mnk=perm
        )

    def ab_smem_layout(TK, stages, rows):
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, FP4, TK), FP4
        )
        return cute.tile_to_shape(atom, (rows, TK, stages), order=(0, 1, 2))

    def coop_frag(p):
        if cute.rank(p) == 3:
            return cute.make_fragment_like(p)
        s = p.layout.shape
        d = p.layout.stride
        nl = cute.make_layout((s[0], s[1][0], s[1][1]), stride=(d[0], d[1][0], d[1][1]))
        return cute.make_fragment_like(cute.make_tensor(p.iterator, nl))

    # ---------------------------- routing + activation quantization + zero-fill
    # One launch: block 0 builds the expert-major routing map and the ragged
    # m-tile work list; every other block quantizes hidden_states once per token
    # (the NVFP4 encoding is expert independent) and zero-fills the bf16 output.
    NQ = T * (H // 64)
    NZ = T * (H // 8)

    @cute.kernel
    def k_prep(
        mIds: cute.Tensor,
        mW: cute.Tensor,
        mPt: cute.Tensor,
        mPw: cute.Tensor,
        mWk: cute.Tensor,
        mMeta: cute.Tensor,
        mX: cute.Tensor,
        mGs: cute.Tensor,
        mAqW: cute.Tensor,
        mSfB: cute.Tensor,
        mOut: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        scnt = smem.allocate_tensor(I32, cute.make_layout(E), byte_alignment=16)
        soff = smem.allocate_tensor(I32, cute.make_layout(E), byte_alignment=16)
        sr1 = smem.allocate_tensor(I32, cute.make_layout(NTR), byte_alignment=16)
        sr2 = smem.allocate_tensor(I32, cute.make_layout(NTR), byte_alignment=16)
        if bid == I32(0):
            for e in cutlass.range(tid, E, NTR, unroll=1):
                scnt[e] = I32(0)
            cute.arch.sync_threads()
            for i in cutlass.range(tid, P, NTR, unroll=1):
                e = mIds[i]
                if e >= I32(0):
                    if e < I32(E):
                        cute.arch.atomic_add(scnt.iterator + e, I32(1), scope="cta")
            cute.arch.sync_threads()
            lsum = I32(0)
            ltil = I32(0)
            for j in cutlass.range_constexpr(ECH):
                ee = tid * ECH + j
                if ee < I32(E):
                    c = scnt[ee]
                    lsum = lsum + c
                    ltil = ltil + (c + (BM - 1)) // BM
            sr1[tid] = lsum
            sr2[tid] = ltil
            cute.arch.sync_threads()
            for d in cutlass.range_constexpr(NTR_LOG):
                dd = 1 << d
                a1 = I32(0)
                a2 = I32(0)
                if tid >= I32(dd):
                    a1 = sr1[tid - dd]
                    a2 = sr2[tid - dd]
                cute.arch.sync_threads()
                sr1[tid] = sr1[tid] + a1
                sr2[tid] = sr2[tid] + a2
                cute.arch.sync_threads()
            if tid == NTR - 1:
                mMeta[0] = sr2[tid]
            acc = sr1[tid] - lsum
            tac = sr2[tid] - ltil
            for j in cutlass.range_constexpr(ECH):
                ee = tid * ECH + j
                if ee < I32(E):
                    c = scnt[ee]
                    scnt[ee] = I32(0)
                    soff[ee] = acc
                    nt = (c + (BM - 1)) // BM
                    for t in cutlass.range(0, nt, 1, unroll=1):
                        rr = c - t * BM
                        if rr > I32(BM):
                            rr = I32(BM)
                        mWk[0, tac + t] = ee
                        mWk[1, tac + t] = acc + t * BM
                        mWk[2, tac + t] = rr
                    acc = acc + c
                    tac = tac + nt
            cute.arch.sync_threads()
            for i in cutlass.range(tid, P, NTR, unroll=1):
                e = mIds[i]
                if e >= I32(0):
                    if e < I32(E):
                        slot = cute.arch.atomic_add(
                            scnt.iterator + e, I32(1), scope="cta"
                        )
                        pos = soff[e] + slot
                        mPt[pos] = I32(i // TOPK)
                        mPw[pos] = mW[i]
        else:
            idx = (bid - I32(1)) * NTHR + tid
            if idx < I32(NZ):
                rz = cute.make_rmem_tensor(cute.make_layout(8), BF16)
                for j in cutlass.range_constexpr(8):
                    rz[j] = BF16(0.0)
                cute.autovec_copy(
                    rz,
                    cute.make_tensor(mOut.iterator + I64(idx) * 8, cute.make_layout(8)),
                )
            else:
                q = idx - I32(NZ)
                if q < I32(NQ):
                    nch = H // 64
                    tk_ = q // nch
                    c = q % nch
                    g = mGs[0]
                    rx = cute.make_rmem_tensor(cute.make_layout(64), BF16)
                    cute.autovec_copy(
                        cute.make_tensor(
                            mX.iterator + (I64(tk_) * H + I64(c) * 64),
                            cute.make_layout(64),
                        ),
                        rx,
                    )
                    v = cute.make_rmem_tensor(cute.make_layout(64), F32)
                    for j in cutlass.range_constexpr(64):
                        v[j] = rx[j].to(F32)
                    sreg = cute.make_rmem_tensor(cute.make_layout(4), E4M3)
                    sbyt = cute.recast_tensor(sreg, U8)
                    for b in cutlass.range_constexpr(4):
                        am = F32(0.0)
                        for j in cutlass.range_constexpr(16):
                            x = v[b * 16 + j]
                            am = cute.arch.fmax(am, cute.arch.fmax(x, -x))
                        sc = -cute.arch.fmax(-(am * (g * F32(1.0 / 6.0))), F32(-448.0))
                        sbyt[b] = U8(_e4m3_byte(sc))
                    rw = cute.make_rmem_tensor(cute.make_layout(8), I32)
                    for b in cutlass.range_constexpr(4):
                        ratio = g / cute.arch.fmax(sreg[b].to(F32), F32(1.0e-20))
                        for w in cutlass.range_constexpr(2):
                            rw[b * 2 + w] = _pack_fp4x8(v, b * 16 + w * 8, ratio)
                    cute.autovec_copy(
                        rw,
                        cute.make_tensor(
                            mAqW.iterator + (I64(tk_) * (H // 8) + I64(c) * 8),
                            cute.make_layout(8),
                        ),
                    )
                    cute.autovec_copy(
                        sbyt,
                        cute.make_tensor(
                            mSfB.iterator + (I64(tk_) * (H // 16) + I64(c) * 4),
                            cute.make_layout(4),
                        ),
                    )

    @cute.kernel
    def k_act(mH: cute.Tensor, mGs: cute.Tensor, mAqW: cute.Tensor, mSfB: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        idx = bid * NTHR + tid
        nch = I // 32
        if idx < I32(P * nch):
            p = idx // nch
            c = idx % nch
            g = mGs[0]
            base = I64(p) * N13 + I64(c) * 32
            v = cute.make_rmem_tensor(cute.make_layout(32), F32)
            cute.autovec_copy(
                cute.make_tensor(mH.iterator + base, cute.make_layout(32)), v
            )
            if cutlass.const_expr(swiglu):
                ru = cute.make_rmem_tensor(cute.make_layout(32), F32)
                cute.autovec_copy(
                    cute.make_tensor(mH.iterator + (base + I), cute.make_layout(32)), ru
                )
                for j in cutlass.range_constexpr(32):
                    x = v[j]
                    v[j] = ((x / (cute.exp(-x) + F32(1.0))) * ru[j]).to(BF16).to(F32)
            else:
                for j in cutlass.range_constexpr(32):
                    x = cute.arch.fmax(v[j], F32(0.0))
                    v[j] = (x * x).to(BF16).to(F32)
            wbase = I64(p) * (I // 8) + I64(c) * 4
            sbase = I64(p) * (I // 16) + I64(c) * 2
            sreg = cute.make_rmem_tensor(cute.make_layout(2), E4M3)
            sbyt = cute.recast_tensor(sreg, U8)
            for b in cutlass.range_constexpr(2):
                am = F32(0.0)
                for j in cutlass.range_constexpr(16):
                    x = v[b * 16 + j]
                    am = cute.arch.fmax(am, cute.arch.fmax(x, -x))
                sc = -cute.arch.fmax(-(am * (g * F32(1.0 / 6.0))), F32(-448.0))
                sbb = _e4m3_byte(sc)
                sbyt[b] = U8(sbb)
                mSfB[sbase + b] = U8(sbb)
            for b in cutlass.range_constexpr(2):
                ratio = g / cute.arch.fmax(sreg[b].to(F32), F32(1.0e-20))
                for w in cutlass.range_constexpr(2):
                    mAqW[wbase + b * 2 + w] = _pack_fp4x8(v, b * 16 + w * 8, ratio)

    # --------------------------------------------------------------- grouped GEMM
    def gemm_kernel(TK, ST, K, NPAD, NCOL, is2, fuse=False, gath=False, epad=0):
        sfa_words_total = (K // 64) * BM
        sfb_words_total = (TK // 64) * BN * ST
        nk = TK // 64
        NKT = K // 64
        KB = TK // 64
        tpr = max(4, TK // 32)
        rows_pass = NTHR // tpr
        cp_per = TK // tpr
        rows_passA = min(BM, rows_pass)
        tprA = NTHR // rows_passA
        cp_perA = TK // tprA
        npassA = BM // rows_passA
        alA = min(16, max(4, cp_perA // 2))
        k_tiles = K // TK
        nsfa = BM * (K // 64)
        nsfb = BN * nk
        reps_a = (nsfa + NTHR - 1) // NTHR
        reps_b = max(1, nsfb // NTHR)
        NPB = NPAD // 128
        NVAL = BM * BN // NTHR  # accumulator values per thread
        HALFV = NVAL * 64 // BN  # values belonging to one 64-wide N half
        NHALF = BN // 64  # number of 64-wide N halves
        NCH = (BM * 8) // NTHR  # 16B output chunks per thread per half
        CPR = 64 // 8  # 16B chunks per row in a half
        nfuse = 64 * nk
        reps_f = max(1, nfuse // NTHR)
        sfb_flat = (NCOL % 128 == 0) and not fuse
        nb16 = nk * 32

        @cute.kernel
        def kern(
            mA: cute.Tensor,
            mSFA: cute.Tensor,
            mB: cute.Tensor,
            mSFB: cute.Tensor,
            mGsW: cute.Tensor,
            mGsA: cute.Tensor,
            mWk: cute.Tensor,
            mMeta: cute.Tensor,
            mOut: cute.Tensor,
            mPw: cute.Tensor,
            mPt: cute.Tensor,
            mQw: cute.Tensor,
            mQs: cute.Tensor,
            mG2: cute.Tensor,
        ):
            tid, _, _ = cute.arch.thread_idx()
            ntile, widx0, _ = cute.arch.block_idx()

            tiled_mma = make_mma(TK)
            a_lay = ab_smem_layout(TK, ST, BM)
            b_lay = ab_smem_layout(TK, ST, BN)
            # Expert weights and their block scales are pure streaming traffic: no CTA
            # ever revisits a byte, so bypass L1 (.cg) and leave the whole L1 working
            # set to the gathered activations.  cache=global only accepts 128b cp.async
            # on SM12x, hence the width gate.
            wop = (
                cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL)
                if cp_per * 4 >= 128
                else cpasync.CopyG2SOp()
            )
            cp_atom = cute.make_copy_atom(wop, FP4, num_bits_per_copy=cp_per * 4)
            cp_atomA = cute.make_copy_atom(
                cpasync.CopyG2SOp(), FP4, num_bits_per_copy=cp_perA * 4
            )
            cp4 = cute.make_copy_atom(cpasync.CopyG2SOp(), U32, num_bits_per_copy=32)
            cp16 = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                U32,
                num_bits_per_copy=128,
            )
            cp8 = cute.make_copy_atom(cpasync.CopyG2SOp(), U32, num_bits_per_copy=64)
            tiled_cp = cute.make_tiled_copy_tv(
                cp_atom,
                cute.make_ordered_layout((rows_pass, tpr), order=(1, 0)),
                cute.make_layout((1, cp_per)),
            )
            tiled_cpA = cute.make_tiled_copy_tv(
                cp_atomA,
                cute.make_ordered_layout((rows_passA, tprA), order=(1, 0)),
                cute.make_layout((1, cp_perA)),
            )
            ldm_atom = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), FP4
            )
            sf_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), E4M3)
            sm_cp_A = cute.make_tiled_copy_A(ldm_atom, tiled_mma)
            sm_cp_B = cute.make_tiled_copy_B(ldm_atom, tiled_mma)
            sm_cp_SFA = cute.make_tiled_copy(
                sf_cp_atom,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            sm_cp_SFB = cute.make_tiled_copy(
                sf_cp_atom,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            smem = utils.SmemAllocator()
            sA = smem.allocate_tensor(
                FP4, a_lay.outer, byte_alignment=1024, swizzle=a_lay.inner
            )
            sB = smem.allocate_tensor(
                FP4, b_lay.outer, byte_alignment=1024, swizzle=b_lay.inner
            )
            if cutlass.const_expr(gath):
                stok = smem.allocate_tensor(
                    I32, cute.make_layout(BM), byte_alignment=16
                )
            if cutlass.const_expr(is2):
                srw = smem.allocate_tensor(F32, cute.make_layout(BM), byte_alignment=16)
                stk = smem.allocate_tensor(I32, cute.make_layout(BM), byte_alignment=16)
            sfa_w = smem.allocate_tensor(
                U32, cute.make_layout(sfa_words_total), byte_alignment=1024
            )
            sfb_w = smem.allocate_tensor(
                U32, cute.make_layout(sfb_words_total), byte_alignment=16
            )
            if cutlass.const_expr(epad > 0):
                smem.allocate_tensor(
                    U32, cute.make_layout(epad // 4), byte_alignment=16
                )
            if cutlass.const_expr(fuse):
                sEp = cute.make_tensor(
                    cute.recast_ptr(sfa_w.iterator, dtype=BF16),
                    cute.make_layout((BM, 64), stride=(64, 1)),
                )
            if cutlass.const_expr(is2):
                sEp2 = cute.make_tensor(
                    cute.recast_ptr(sfa_w.iterator, dtype=BF16),
                    cute.make_layout((BM, 64), stride=(64, 1)),
                )
            # The activation scales of one work item are staged for the whole K at
            # once: a k-tile window of this layout is bit-identical to the per-stage
            # layout, so the mainloop simply indexes k-tile kt instead of kt % ST.
            sSFA = cute.make_tensor(
                cute.recast_ptr(sfa_w.iterator, dtype=E4M3),
                _sf_mma_layout(BM, TK, k_tiles),
            )
            sSFB = cute.make_tensor(
                cute.recast_ptr(sfb_w.iterator, dtype=E4M3), _sf_mma_layout(BN, TK, ST)
            )

            thr_mma = tiled_mma.get_slice(tid)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            vv = thr_mma.thr_layout_vmnk.get_flat_coord(tid)
            fa = cute.make_tensor(
                sSFA.iterator,
                sm120_utils.thrfrg_SFA(sSFA[None, None, 0].layout, thr_mma),
            )
            tCrSFA = coop_frag(fa[(vv[0], (vv[1], vv[3])), (None, None)])
            fb = cute.make_tensor(
                sSFB.iterator,
                sm120_utils.thrfrg_SFB(sSFB[None, None, 0].layout, thr_mma),
            )
            tCrSFB = coop_frag(fb[(vv[0], (vv[2], vv[3])), (None, None)])

            ta = sm_cp_A.get_slice(tid)
            tb = sm_cp_B.get_slice(tid)
            tsa = sm_cp_SFA.get_slice(tid)
            tsb = sm_cp_SFB.get_slice(tid)
            tCsA_v = ta.partition_S(sA)
            tCrA_v = ta.retile(tCrA)
            tCsB_v = tb.partition_S(sB)
            tCrB_v = tb.retile(tCrB)
            tCsSFA_v = tsa.partition_S(sSFA)
            tCrSFA_v = tsa.retile(tCrSFA)
            tCsSFB_v = tsb.partition_S(sSFB)
            tCrSFB_v = tsb.retile(tCrSFB)
            thr_cp = tiled_cp.get_slice(tid)
            thr_cpA = tiled_cpA.get_slice(tid)
            srcl = cute.make_layout(((cp_perA, 1),), stride=((1, 0),))

            for widx in cutlass.range(widx0, mMeta[0], GY, unroll=1):
                # Only a reused CTA needs to fence the preceding epilogue before its
                # next prologue overwrites the shared A/B/scale stages.
                if widx != widx0:
                    cute.arch.sync_threads()
                e = mWk[0, widx]
                r0 = mWk[1, widx]
                rows = mWk[2, widx]
                if cutlass.const_expr(fuse):
                    n0 = I32(ntile) * 64
                else:
                    n0 = I32(ntile) * BN
                    if n0 > I32(NCOL - BN):
                        n0 = I32(NCOL - BN)

                if cutlass.const_expr(gath):
                    if tid < I32(BM):
                        tkv = I32(0)
                        if tid < rows:
                            tkv = mPt[r0 + tid]
                        stok[tid] = tkv
                    cute.arch.sync_threads()
                gA_u8 = cute.make_tensor(
                    _gp(mA, U8, I64(r0) * (K // 2)),
                    cute.make_layout((BM, K // 2), stride=(K // 2, 1)),
                )
                gA = cute.local_tile(
                    cute.recast_tensor(gA_u8, FP4), (BM, TK), (0, None)
                )
                abase = mA.iterator.toint() + I64((tid % tprA) * (cp_perA // 2))
                arow = tid // tprA
                if cutlass.const_expr(fuse):
                    gB_u8 = cute.make_tensor(
                        _gp(mB, U8, (I64(e) * NCOL + I64(n0)) * (K // 2)),
                        cute.make_layout(
                            ((64, 2), K // 2), stride=((K // 2, I * (K // 2)), 1)
                        ),
                    )
                else:
                    gB_u8 = cute.make_tensor(
                        _gp(mB, U8, (I64(e) * NCOL + I64(n0)) * (K // 2)),
                        cute.make_layout((BN, K // 2), stride=(K // 2, 1)),
                    )
                gB = cute.local_tile(
                    cute.recast_tensor(gB_u8, FP4), (BN, TK), (0, None)
                )
                tAg = thr_cpA.partition_S(gA)
                tAs = thr_cpA.partition_D(sA)
                tBg = thr_cp.partition_S(gB)
                tBs = thr_cp.partition_D(sB)

                sfa_row0 = I64(0)
                if cutlass.const_expr(not gath):
                    sfa_row0 = I64(r0) * NKT
                sfb_exp = I64(e) * (NPB * NKT * 128)

                # Activation block scales for the whole K in one coalesced pass.  The
                # k-groups of a row are adjacent words, so walking them inside a thread
                # group gives every warp a contiguous run instead of BM distinct
                # rows of one 32 B sector each.
                for it in cutlass.range_constexpr(reps_a):
                    sidx = tid + it * NTHR
                    m = sidx // NKT
                    g = sidx % NKT
                    if cutlass.const_expr(gath):
                        mr0 = I64(stok[m % BM])
                    else:
                        mr0 = I64(m)
                    if I32(m) < rows:
                        cute.copy(
                            cp4,
                            cute.make_tensor(
                                _gp(mSFA, U32, (sfa_row0 + mr0 * NKT + I64(g)) * 4, 4),
                                cute.make_layout(1),
                            ),
                            cute.make_tensor(
                                sfa_w.iterator
                                + ((m % 32) * (BM // 32) + (m // 32) + g * BM),
                                cute.make_layout(1),
                            ),
                        )
                cute.arch.cp_async_commit_group()

                for s in cutlass.range_constexpr(ST - 1):
                    if cutlass.const_expr(s < k_tiles):
                        if cutlass.const_expr(gath):
                            for ip in cutlass.range_constexpr(npassA):
                                mrw = arow + I32(ip * rows_passA)
                                if mrw < rows:
                                    cute.copy(
                                        cp_atomA,
                                        cute.make_tensor(
                                            cute.recast_ptr(
                                                cute.make_ptr(
                                                    U8,
                                                    abase
                                                    + I64(stok[mrw]) * (K // 2)
                                                    + I64(s * (TK // 2)),
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=alA,
                                                ),
                                                dtype=FP4,
                                            ),
                                            srcl,
                                        ),
                                        tAs[None, ip, 0, s],
                                    )
                        else:
                            if I32(tid // tprA) < rows:
                                cute.copy(
                                    cp_atomA,
                                    tAg[None, None, None, s],
                                    tAs[None, None, None, s],
                                )
                        cute.copy(
                            cp_atom, tBg[None, None, None, s], tBs[None, None, None, s]
                        )
                        if cutlass.const_expr(sfb_flat):
                            if tid < I32(nb16):
                                wof = tid * 4
                                cgf = wof // 128
                                off = wof % 128
                                sbf = (
                                    sfb_exp
                                    + I64(n0 // 128) * (NKT * 128)
                                    + I64(s * nk + cgf) * 128
                                    + I64(off)
                                )
                                cute.copy(
                                    cp16,
                                    cute.make_tensor(
                                        _gp(mSFB, U32, sbf * 4, 16), cute.make_layout(4)
                                    ),
                                    cute.make_tensor(
                                        _sw(
                                            sfb_w.iterator,
                                            cgf * 128 + off + s * (nk * 128),
                                            16,
                                        ),
                                        cute.make_layout(4),
                                    ),
                                )
                        elif cutlass.const_expr(fuse):
                            for it in cutlass.range_constexpr(reps_f):
                                fidx = (tid % nfuse) + it * NTHR
                                hh = fidx // (32 * nk)
                                cg = (fidx // 32) % nk
                                cc = fidx % 32
                                # gate rows start at n0 and up rows at I + n0; each
                                # half has its own 64-row slot in its 128-row block
                                nrow = I32(n0) + hh * I
                                nblk = I64(nrow) // 128
                                hb8 = ((nrow % 128) // 64) * 2
                                sb = (
                                    sfb_exp
                                    + nblk * (NKT * 128)
                                    + I64(s * nk + cg) * 128
                                    + I64(cc * 4 + hb8)
                                )
                                cute.copy(
                                    cp8,
                                    cute.make_tensor(
                                        _gp(mSFB, U32, sb * 4, 8), cute.make_layout(2)
                                    ),
                                    cute.make_tensor(
                                        _sw(
                                            sfb_w.iterator,
                                            cg * 128 + cc * 4 + hh * 2 + s * (nk * 128),
                                            8,
                                        ),
                                        cute.make_layout(2),
                                    ),
                                )
                        else:
                            for it in cutlass.range_constexpr(reps_b):
                                sidx = (tid % nsfb) + it * NTHR
                                cg = sidx // BN
                                m = sidx % BN
                                dst = (
                                    (m % 32) * 4 + (m // 32) + cg * 128 + s * (nk * 128)
                                )
                                nrow = n0 + I32(m)
                                sb = (
                                    sfb_exp
                                    + I64(nrow // 128) * (NKT * 128)
                                    + I64(s * nk + cg) * 128
                                    + I64((nrow % 32) * 4 + ((nrow // 32) % 4))
                                )
                                cute.copy(
                                    cp4,
                                    cute.make_tensor(
                                        _gp(mSFB, U32, sb * 4, 4), cute.make_layout(1)
                                    ),
                                    cute.make_tensor(
                                        sfb_w.iterator + dst, cute.make_layout(1)
                                    ),
                                )
                    cute.arch.cp_async_commit_group()

                idt = thr_mma.partition_C(cute.make_identity_tensor((BM, BN)))
                acc = cute.make_rmem_tensor(idt.shape, F32)
                acc.fill(0.0)

                # If the prologue already covers every K tile, nothing more is copied,
                # so the per-tile barrier pair collapses into a single wait.
                if cutlass.const_expr(k_tiles <= ST - 1):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                # A depth-two unroll exposes independent K iterations at a moderate
                # code-size and register cost.
                for kt in cutlass.range(0, k_tiles, 1, unroll=2):
                    if cutlass.const_expr(k_tiles > ST - 1):
                        cute.arch.cp_async_wait_group(ST - 2)
                        cute.arch.sync_threads()
                    nxt = kt + (ST - 1)
                    if nxt < I32(k_tiles):
                        nst = nxt % ST
                        if cutlass.const_expr(gath):
                            for ip in cutlass.range_constexpr(npassA):
                                mrw = arow + I32(ip * rows_passA)
                                if mrw < rows:
                                    cute.copy(
                                        cp_atomA,
                                        cute.make_tensor(
                                            cute.recast_ptr(
                                                cute.make_ptr(
                                                    U8,
                                                    abase
                                                    + I64(stok[mrw]) * (K // 2)
                                                    + I64(nxt) * (TK // 2),
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=alA,
                                                ),
                                                dtype=FP4,
                                            ),
                                            srcl,
                                        ),
                                        tAs[None, ip, 0, nst],
                                    )
                        else:
                            if I32(tid // tprA) < rows:
                                cute.copy(
                                    cp_atomA,
                                    tAg[None, None, None, nxt],
                                    tAs[None, None, None, nst],
                                )
                        cute.copy(
                            cp_atom,
                            tBg[None, None, None, nxt],
                            tBs[None, None, None, nst],
                        )
                        if cutlass.const_expr(sfb_flat):
                            if tid < I32(nb16):
                                wof = tid * 4
                                cgf = wof // 128
                                off = wof % 128
                                sbf = (
                                    sfb_exp
                                    + I64(n0 // 128) * (NKT * 128)
                                    + I64(nxt * nk + cgf) * 128
                                    + I64(off)
                                )
                                cute.copy(
                                    cp16,
                                    cute.make_tensor(
                                        _gp(mSFB, U32, sbf * 4, 16), cute.make_layout(4)
                                    ),
                                    cute.make_tensor(
                                        _sw(
                                            sfb_w.iterator,
                                            cgf * 128 + off + nst * (nk * 128),
                                            16,
                                        ),
                                        cute.make_layout(4),
                                    ),
                                )
                        elif cutlass.const_expr(fuse):
                            for it in cutlass.range_constexpr(reps_f):
                                fidx = (tid % nfuse) + it * NTHR
                                hh = fidx // (32 * nk)
                                cg = (fidx // 32) % nk
                                cc = fidx % 32
                                # gate rows start at n0 and up rows at I + n0; each
                                # half has its own 64-row slot in its 128-row block
                                nrow = I32(n0) + hh * I
                                nblk = I64(nrow) // 128
                                hb8 = ((nrow % 128) // 64) * 2
                                sb = (
                                    sfb_exp
                                    + nblk * (NKT * 128)
                                    + I64(nxt * nk + cg) * 128
                                    + I64(cc * 4 + hb8)
                                )
                                cute.copy(
                                    cp8,
                                    cute.make_tensor(
                                        _gp(mSFB, U32, sb * 4, 8), cute.make_layout(2)
                                    ),
                                    cute.make_tensor(
                                        _sw(
                                            sfb_w.iterator,
                                            cg * 128
                                            + cc * 4
                                            + hh * 2
                                            + nst * (nk * 128),
                                            8,
                                        ),
                                        cute.make_layout(2),
                                    ),
                                )
                        else:
                            for it in cutlass.range_constexpr(reps_b):
                                sidx = (tid % nsfb) + it * NTHR
                                cg = sidx // BN
                                m = sidx % BN
                                dst = (
                                    (m % 32) * 4
                                    + (m // 32)
                                    + cg * 128
                                    + nst * (nk * 128)
                                )
                                nrow = n0 + I32(m)
                                sb = (
                                    sfb_exp
                                    + I64(nrow // 128) * (NKT * 128)
                                    + I64(nxt * nk + cg) * 128
                                    + I64((nrow % 32) * 4 + ((nrow // 32) % 4))
                                )
                                cute.copy(
                                    cp4,
                                    cute.make_tensor(
                                        _gp(mSFB, U32, sb * 4, 4), cute.make_layout(1)
                                    ),
                                    cute.make_tensor(
                                        sfb_w.iterator + dst, cute.make_layout(1)
                                    ),
                                )
                    cute.arch.cp_async_commit_group()
                    st = kt % ST
                    for kb in cutlass.range_constexpr(KB):
                        cute.copy(
                            sm_cp_A, tCsA_v[None, None, kb, st], tCrA_v[None, None, kb]
                        )
                        cute.copy(
                            sm_cp_B, tCsB_v[None, None, kb, st], tCrB_v[None, None, kb]
                        )
                        cute.copy(
                            sm_cp_SFA,
                            cute.filter_zeros(tCsSFA_v[None, None, None, kt])[
                                None, None, kb
                            ],
                            cute.filter_zeros(tCrSFA_v)[None, None, kb],
                        )
                        cute.copy(
                            sm_cp_SFB,
                            cute.filter_zeros(tCsSFB_v[None, None, None, st])[
                                None, None, kb
                            ],
                            cute.filter_zeros(tCrSFB_v)[None, None, kb],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc,
                            [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                            [tCrB[None, None, kb], tCrSFB[None, None, kb]],
                            acc,
                        )

                scale = mGsW[e] / mGsA[0]
                if cutlass.const_expr(fuse):
                    MN = BN // 32
                    VM = (BM * BN // NTHR) // MN
                    cute.arch.sync_threads()
                    for i in cutlass.range_constexpr(2 * VM):
                        gv = acc[i] * scale
                        uv = acc[i + 2 * VM] * scale
                        av = (gv / (cute.exp(-gv) + F32(1.0))) * uv
                        sEp[idt[i][0], idt[i][1]] = av.to(BF16)
                    cute.arch.sync_threads()
                    mq = tid // 4
                    bq = tid % 4
                    if mq < rows:
                        g2 = mG2[0]
                        rv = cute.make_rmem_tensor(cute.make_layout(16), BF16)
                        cute.autovec_copy(
                            cute.make_tensor(
                                sEp.iterator + (mq * 64 + bq * 16), cute.make_layout(16)
                            ),
                            rv,
                        )
                        vq = cute.make_rmem_tensor(cute.make_layout(16), F32)
                        am = F32(0.0)
                        for j in cutlass.range_constexpr(16):
                            vq[j] = rv[j].to(F32)
                            am = cute.arch.fmax(am, cute.arch.fmax(vq[j], -vq[j]))
                        sc = -cute.arch.fmax(-(am * (g2 * F32(1.0 / 6.0))), F32(-448.0))
                        sbb = _e4m3_byte(sc)
                        sreg = cute.make_rmem_tensor(cute.make_layout(1), E4M3)
                        cute.recast_tensor(sreg, U8)[0] = U8(sbb)
                        mQs[I64(r0 + mq) * (I // 16) + I64(n0 // 16 + bq)] = U8(sbb)
                        ratio = g2 / cute.arch.fmax(sreg[0].to(F32), F32(1.0e-20))
                        wb = I64(r0 + mq) * (I // 8) + I64(n0 // 8 + bq * 2)
                        for w in cutlass.range_constexpr(2):
                            mQw[wb + w] = _pack_fp4x8(vq, w * 8, ratio)
                elif cutlass.const_expr(is2):
                    obase = mOut.iterator.toint()
                    if tid < I32(BM):
                        rwv = F32(0.0)
                        tkv = I32(0)
                        if tid < rows:
                            rwv = mPw[r0 + tid] * scale
                            tkv = mPt[r0 + tid]
                        srw[tid] = rwv
                        stk[tid] = tkv
                    for hf in cutlass.range_constexpr(NHALF):
                        cute.arch.sync_threads()
                        for i in cutlass.range_constexpr(hf * HALFV, (hf + 1) * HALFV):
                            m = idt[i][0]
                            sEp2[m, idt[i][1] - hf * 64] = (acc[i] * srw[m]).to(BF16)
                        cute.arch.sync_threads()
                        for ch in cutlass.range_constexpr(NCH):
                            ii = tid + ch * NTHR
                            m = ii // CPR
                            j = ii % CPR
                            if m < rows:
                                rv = cute.make_rmem_tensor(cute.make_layout(4), I32)
                                cute.autovec_copy(
                                    cute.make_tensor(
                                        _sw(sEp2.iterator, m * 32 + j * 4, 16),
                                        cute.make_layout(4),
                                    ),
                                    rv,
                                )
                                adr = (
                                    obase
                                    + (I64(stk[m]) * NCOL + I64(n0 + hf * 64 + j * 8))
                                    * 2
                                )
                                _red_v4(adr, rv[0], rv[1], rv[2], rv[3])
                else:
                    gC = cute.make_tensor(
                        _gp(mOut, F32, (I64(r0) * NCOL + I64(n0)) * 4),
                        cute.make_layout((BM, BN), stride=(NCOL, 1)),
                    )
                    tCgC = thr_mma.partition_C(gC)
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        m = idt[i][0]
                        if m < rows:
                            tCgC[i] = acc[i] * scale

        return kern

    k_gemm1 = gemm_kernel(
        TK1, ST1, H, N13_PAD, N13, False, fuse=swiglu, gath=True, epad=EPI1
    )
    k_gemm2 = gemm_kernel(TK2, ST2, I, H, H, True, epad=EPI2)

    @cute.jit
    def prog(
        mIds,
        mW,
        mX,
        mW13,
        mW13sf,
        mW13gs,
        mW2,
        mW2sf,
        mW2gs,
        mA13,
        mA2,
        mPt,
        mPw,
        mWk,
        mMeta,
        mAq1W,
        mAq1B,
        mSf1W,
        mSf1B,
        mHb,
        mAq2W,
        mAq2B,
        mSf2W,
        mSf2B,
        mOut,
        stream: cuda.CUstream,
    ):
        npz = (NZ + NQ + NTHR - 1) // NTHR
        k_prep(mIds, mW, mPt, mPw, mWk, mMeta, mX, mA13, mAq1W, mSf1B, mOut).launch(
            grid=[1 + npz, 1, 1], block=[NTHR, 1, 1], stream=stream
        )
        k_gemm1(
            mAq1B,
            mSf1W,
            mW13,
            mW13sf,
            mW13gs,
            mA13,
            mWk,
            mMeta,
            mHb,
            mPw,
            mPt,
            mAq2W,
            mSf2B,
            mA2,
        ).launch(
            grid=[NT1, GY, 1],
            block=[NTHR, 1, 1],
            max_number_threads=[NTHR, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )
        if cutlass.const_expr(not swiglu):
            na = (P * (I // 32) + NTHR - 1) // NTHR
            k_act(mHb, mA2, mAq2W, mSf2B).launch(
                grid=[na, 1, 1], block=[NTHR, 1, 1], stream=stream
            )
        k_gemm2(
            mAq2B,
            mSf2W,
            mW2,
            mW2sf,
            mW2gs,
            mA2,
            mWk,
            mMeta,
            mOut,
            mPw,
            mPt,
            mAq2W,
            mSf2B,
            mA2,
        ).launch(grid=[NT2, GY, 1], block=[NTHR, 1, 1], stream=stream)

    return prog, dict(P=P, PA=PA, WMAX=WMAX)


# Plans and compiled kernels only: scratch belongs to the caller (see
# ``allocate_workspace``) so layers and concurrent streams never share it.
_COMPILED: dict = {}


def _check_shape(H: int, I: int, N13: int) -> None:
    if H % 128 or I % 64:
        raise ValueError(
            f"SM12x W4A4 MoE needs hidden_size % 128 == 0 and "
            f"intermediate_size % 64 == 0, got {H} and {I}."
        )
    if N13 not in (I, 2 * I):
        raise ValueError(f"w13 must have I or 2*I rows, got {N13} for I={I}.")


def _entry(T, H, I, N13, E, TOPK, dev):
    _check_shape(H, I, N13)
    key = (dev, T, H, I, N13, E, TOPK)
    entry = _COMPILED.get(key)
    if entry is None:
        swiglu = N13 == 2 * I
        prog, meta = _build(T, H, I, N13, E, TOPK, swiglu)
        P, PA, WMAX = meta["P"], meta["PA"], meta["WMAX"]
        shapes = (
            ((P,), torch.int32),  # token of each expert-major pair
            ((P,), torch.float32),  # router weight of each pair
            ((3, WMAX), torch.int32),  # M-tile work list: expert, first row, rows
            ((4,), torch.int32),  # work-list length
            ((T * (H // 2),), torch.uint8),  # NVFP4 hidden states
            ((T * (H // 16),), torch.uint8),  # their block scales
            ((1 if swiglu else P * N13,), torch.float32),  # ReLU2 GEMM1 output
            ((PA * (I // 2),), torch.uint8),  # NVFP4 intermediate
            ((PA * (I // 16),), torch.uint8),  # its block scales
        )
        entry = [prog, shapes, None]
        _COMPILED[key] = entry
    return entry


def allocate_workspace(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    is_gated: bool,
    device: torch.device,
) -> list[torch.Tensor]:
    """Scratch for one ``run_moe_w4a4`` token count.

    Every byte is rewritten from the inputs on each call, but a workspace must
    not be used by two launches that can run concurrently.
    """
    N13 = intermediate_size * (2 if is_gated else 1)
    _, shapes, _ = _entry(
        num_tokens,
        hidden_size,
        intermediate_size,
        N13,
        num_experts,
        top_k,
        torch.device(device),
    )
    return [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in shapes]


@torch.no_grad()
def run_moe_w4a4(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_sf: torch.Tensor,
    w13_gs: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    w2_gs: torch.Tensor,
    a13_gs: torch.Tensor,
    a2_gs: torch.Tensor,
    out: torch.Tensor,
    workspace: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Routed-expert W4A4 MoE forward into ``out``.

    Args:
        hidden_states: BF16 ``[T, H]``; quantized to NVFP4 inside the op.
        topk_ids: int32 ``[T, TOPK]`` expert ids; slots with an id outside
            ``[0, E)`` contribute nothing.
        topk_weights: FP32 ``[T, TOPK]`` router weights.
        w13: uint8 ``[E, N13, H/2]``; ``N13 == 2*I`` is SwiGLU with gate rows
            first, ``N13 == I`` is ReLU2.
        w13_sf: E4M3 ``[E, pad128(N13), pad4(H/16)]`` 128x4-swizzled scales.
        w13_gs: FP32 ``[E]`` global scales.
        w2, w2_sf, w2_gs: the same for the ``[E, H, I/2]`` down projection.
        a13_gs: FP32 one-element activation global scale of the GEMM1 input
            (the quantization multiplier, i.e. ``1 / input_scale``).
        a2_gs: the same for the GEMM2 input.
        out: BF16 ``[T, H]`` output; overwritten.
        workspace: from ``allocate_workspace`` for this token count; a fresh
            one is allocated when omitted.

    Requires ``H % 128 == 0`` and ``I % 64 == 0``. Kernels are specialized and
    cached per token count and geometry; the first call for a new shape
    compiles and must not run under CUDA graph capture.
    """
    T, H = hidden_states.shape
    E, N13, _ = w13.shape
    I = w2.shape[2] * 2
    TOPK = topk_ids.shape[1]
    dev = hidden_states.device
    pad = lambda n, m: (n + m - 1) // m * m  # noqa: E731
    if tuple(w13_sf.shape) != (E, pad(N13, 128), pad(H // 16, 4)) or tuple(
        w2_sf.shape
    ) != (E, pad(H, 128), pad(I // 16, 4)):
        raise ValueError("w13_sf / w2_sf must be [E, pad128(rows), pad4(K/16)].")
    entry = _entry(T, H, I, N13, E, TOPK, dev)
    prog, shapes, compiled = entry
    if compiled is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SM12x W4A4 MoE must be compiled for this shape before graph capture."
        )
    if workspace is None:
        workspace = allocate_workspace(T, H, I, E, TOPK, N13 == 2 * I, dev)
    elif [tuple(t.shape) for t in workspace] != [s for s, _ in shapes]:
        raise ValueError("workspace was allocated for a different shape.")
    pt, pw, wk, mt, aq1, sf1, hb, aq2, sf2 = workspace

    args = (
        topk_ids.reshape(-1),
        topk_weights.reshape(-1),
        hidden_states,
        w13.reshape(-1),
        w13_sf.view(torch.uint8).reshape(-1).view(torch.int32),
        w13_gs,
        w2.reshape(-1),
        w2_sf.view(torch.uint8).reshape(-1).view(torch.int32),
        w2_gs,
        a13_gs.reshape(1),
        a2_gs.reshape(1),
        pt,
        pw,
        wk,
        mt,
        aq1.view(torch.int32),
        aq1,
        sf1.view(torch.int32),
        sf1,
        hb,
        aq2.view(torch.int32),
        aq2,
        sf2.view(torch.int32),
        sf2,
        out,
    )
    if compiled is None:
        fake = [from_dlpack(t, assumed_align=16, enable_tvm_ffi=True) for t in args]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(prog, *fake, stream, options="--enable-tvm-ffi")
        entry[2] = compiled
    compiled(*args)
    return out
