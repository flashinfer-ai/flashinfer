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

---------------------------------------------------------------------------
KDA output-only (frozen-state) decode on tcgen05 / UMMA / TMEM (SM10x): the
``backend="umma"`` arm of ``flashinfer.kda_kernels.kda_decode_wy_output_only``
(``kda_wy_output_only`` and the vLLM ``kda_recoverssm_verify`` drop-in).

The ReplaySSM UMMA kernel (KDA ucache flush) with an empty history: the state is
read, never written, and there is no ring. Per (sequence b, value head hv), with
T <= 16 tokens, per-channel cumulative log-decay cum_t and k̂ / q̂ the L2-normalized
key / (scaled) query of the query head i_h = hv / (HV / H):

    khat_t = k̂_t e^{cum_t},   ktil_s = k̂_s e^{-cum_s},   qhat_t = q̂_t e^{cum_t}
    u_t    = beta_t (v_t - S0 khat_t - sum_{s<t} (khat_t . ktil_s) u_s)
    y_t    = S0 qhat_t + sum_{s<=t} (qhat_t . ktil_s) u_s

u_t is also the RecoverSSM correction (emit mode).

CTA = (value head, sequence), grid (HV, B), 192 threads:
  warp 0   : mbarrier inits, TMA of the bf16 state (two 16 KB SW128 K halves).
  warp 1   : TMEM allocation at entry (32 columns), the UMMA issuer.
  warps 2-5: consumers: row-layout L2 norm, channel-layout gate (X_r / X_c / Ktil
             tiles), the mma.sync Gram [X_c] . Ktil^T, the per-V-row forward
             substitution epilogue (TMEM lane = V row).
UMMA: readout R = S0 . X_r^T, M128 N{16, 32} K128 (kind::f16, BF16 A / B, fp32 TMEM);
X_r = [khat; qhat] (2T rows, zero rows after), so R holds S0 khat_t in column t and
S0 qhat_t in column T + t.

The kernel takes ``KdaDecodeWyOutputOnlyKernel``'s argument list (dense
[B, T, ...] tensors, or the vLLM drop-in's packed tokens with runtime token
strides, ragged lengths, null slots and slot-indexed caches).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.gdn_kernels import _umma_helpers as U

K_DIM = 128
V_DIM = 128
XC_PITCH = 272  # bytes per row of the row-major X_c / Ktil tiles (136 bf16)
LOG2_E = 1.4426950408889634
_EPS = 1e-6  # inside the L2-norm rsqrt, as the WY kernel
_TMEM_COLS = 32

GATE_PRECOMPUTED = 0
GATE_LOWER_BOUND = 1
GATE_SOFTPLUS = 2


def _bf16_idesc(m: int, n: int):
    """kind::f16 instruction descriptor with BF16 A and B (bits 7 and 10), fp32 D."""
    return U.f16_idesc(m, n) | Uint32((1 << 7) | (1 << 10))


def _sel(cond, a, b, ty):
    """Branch-free typed select."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


@dsl_user_op
def _ldmatrix_x4(addr: Int32, *, loc=None, ip=None):
    """ldmatrix .x4 (m16n8k16 A fragment of a row-major tile)."""
    st = llvm.StructType.get_literal([_T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ldmatrix_x2(addr: Int32, *, loc=None, ip=None):
    """ldmatrix .x2 (m16n8k16 B fragment of an [n][k] row-major tile)."""
    st = llvm.StructType.get_literal([_T.i32()] * 2)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {$0,$1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _ex2(x: Float32, *, loc=None, ip=None) -> Float32:
    """ex2.approx.ftz.f32 (the WY kernel's _exp2_approx_f32)."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrt.approx.ftz.f32."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _bar_sync_ua(barrier_id: int, number_of_threads: int, *, loc=None, ip=None) -> None:
    """Named-barrier sync without ``.aligned`` (multi-site barriers; synccheck)."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_u16_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Predicated ld.global.u16 (zero-extended); 0 where pred == 0."""
    r = llvm.inline_asm(
        _T.i32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tmov.u32 $0, 0;\n\t"
        "setp.ne.s32 p, $1, 0;\n\t@p ld.global.u16 h, [$2];\n\t"
        "@p cvt.u32.u16 $0, h;\n\t}",
        "=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def _ldg_v4_or0(pred: Int32, addr: Int64, *, loc=None, ip=None):
    """Predicated ld.global.v4.b32: four words, 0 where pred == 0."""
    st = llvm.StructType.get_literal([_T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
        "mov.b32 $3, 0;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _stg_u16(addr: Int64, w: Uint32, *, loc=None, ip=None) -> None:
    """st.global.u16 of the low 16 bits of ``w``."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u16 [$0], h;\n\t}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_f32(addr: Int64, x: Float32, *, loc=None, ip=None) -> None:
    """st.global.f32."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(x).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _x_sw128_off(row, c, half_bytes):
    """Byte offset of element (row, k = c) in a K-major SW128 operand tile: two
    64-element K halves ``half_bytes`` apart, 128 B rows, 16 B chunks XOR-swizzled
    with (row % 8)."""
    return (
        (c // 64) * half_bytes
        + row * 128
        + ((((c % 64) // 8) ^ (row % 8)) * 16)
        + (c % 8) * 2
    )


class KdaOutputOnlyUmma:
    """UMMA output-only KDA decode; the argument list of ``KdaDecodeWyOutputOnlyKernel``."""

    def __init__(
        self,
        T: int,
        gate_mode: int = GATE_PRECOMPUTED,
        has_dt_bias: bool = False,
        beta_is_logit: bool = False,
        emit_corrections: bool = False,
        vllm_dropin: bool = False,
        null_min: int = 1,
    ):
        assert 1 <= T <= 16, T
        self.T = T
        self.gate_mode = int(gate_mode)
        self.has_dt_bias = bool(has_dt_bias)
        self.beta_is_logit = bool(beta_is_logit)
        self.dropin = bool(vllm_dropin)
        self.emit = bool(emit_corrections) or self.dropin
        self.null_min = int(null_min)
        self.NX = 16 if T <= 8 else 32  # X_r / X_c rows (2T live)
        self.KT = 8 if T <= 8 else 16  # Ktil rows (T live)
        self.num_threads = 192
        # SMEM: 32 KB state + X_r + X_c + Ktil + Gram + tables (~44 KB at T <= 8,
        # ~58 KB above): 4 / 3 CTAs per SM
        self.min_blocks = 4 if T <= 8 else 3

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gG: cute.Tensor,
        gBeta: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gOut: cute.Tensor,
        gCorr: cute.Tensor,
        gKg: cute.Tensor,
        gQsl: cute.Tensor,
        scale: Float32,
        lower_bound: Float32,
        s_q_tok: Int32,
        s_k_tok: Int32,
        s_v_tok: Int32,
        s_g_tok: Int32,
        s_b_tok: Int32,
        s_o_tok: Int32,
        s_c_blk: Int32,
        s_c_head: Int32,
        s_c_pos: Int32,
        s_kg_blk: Int32,
        s_kg_head: Int32,
        s_kg_pos: Int32,
        s_h0_blk: Int64,
        HV: Int32,
        V_DIM_: Int32,
        H: Int32,
        stream,
    ):
        B_val = gH0idx.layout.shape[0]
        # pool rebuilt with static inner strides and the runtime (possibly padded) block
        # stride (the WY kernel's scheme); TMA box = (V rows, 64 K) per half
        blk = cute.assume(s_h0_blk, divby=8)
        state = cute.make_tensor(
            gH0.iterator,
            cute.make_layout(
                (gH0.layout.shape[0], HV, V_DIM, K_DIM),
                stride=(blk, V_DIM * K_DIM, K_DIM, 1),
            ),
        )
        elems = 64
        smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (1, 1, V_DIM, (elems, 1), 1),
                stride=(0, 0, elems, (1, V_DIM * elems), V_DIM * elems),
            ),
        )
        tma_ld = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(state, (None, None, None, elems)),
            smem_layout,
            cta_tiler=(1, 1, V_DIM, elems),
        )
        self.kernel(
            gQ,
            gK,
            gV,
            gG,
            gBeta,
            gAlog,
            gDtbias,
            gH0idx,
            gOut,
            gCorr,
            gKg,
            gQsl,
            scale,
            lower_bound,
            s_q_tok,
            s_k_tok,
            s_v_tok,
            s_g_tok,
            s_b_tok,
            s_o_tok,
            s_c_blk,
            s_c_head,
            s_c_pos,
            s_kg_blk,
            s_kg_head,
            s_kg_pos,
            HV,
            H,
            tma_ld,
        ).launch(
            grid=(HV, B_val, 1),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gG: cute.Tensor,
        gBeta: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0idx: cute.Tensor,
        gOut: cute.Tensor,
        gCorr: cute.Tensor,
        gKg: cute.Tensor,
        gQsl: cute.Tensor,
        scale: Float32,
        lower_bound: Float32,
        s_q_tok: Int32,
        s_k_tok: Int32,
        s_v_tok: Int32,
        s_g_tok: Int32,
        s_b_tok: Int32,
        s_o_tok: Int32,
        s_c_blk: Int32,
        s_c_head: Int32,
        s_c_pos: Int32,
        s_kg_blk: Int32,
        s_kg_head: Int32,
        s_kg_pos: Int32,
        HV: Int32,
        H: Int32,
        tma_ld: cpasync.TmaInfo,
    ):
        T = self.T
        T2 = 2 * T
        NX = self.NX
        KT = self.KT
        XH = NX * 128  # bytes per K half of the X_r tile
        NP = (T2 + 7) // 8  # row-layout passes (8 X rows per pass)
        PARTIAL = T2 % 8 != 0
        DROPIN = self.dropin
        EMIT = self.emit
        GM = self.gate_mode
        TT = 16  # coefficient-table row pitch

        tid, _, _ = cute.arch.thread_idx()
        hv, brow, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        i_h = hv // (HV // H)  # query head (GQA)

        # ------------------------------------------------------------------ SMEM
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            BFloat16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(NX * K_DIM), byte_alignment=1024
        )
        s_xc = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(NX * XC_PITCH // 2), byte_alignment=128
        )
        s_ktil = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(KT * XC_PITCH // 2), byte_alignment=128
        )
        s_gram = smem.allocate_tensor(
            Float32, cute.make_layout(NX * KT), byte_alignment=16
        )
        s_coef = smem.allocate_tensor(
            Float32, cute.make_layout(2 * TT * TT + TT), byte_alignment=16
        )
        st_full = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2)]
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        mma_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        _CO_A = 0
        _CO_BQ = TT * TT
        _CO_BETA = 2 * TT * TT

        if warp == 1:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()
            U.fence_before_sync()
            cute.arch.barrier_arrive(barrier_id=2, number_of_threads=160)

        # ---------------------------------------------------------- row metadata
        raw_idx = Int32(gH0idx[brow])
        bos = Int32(0)
        seq_len = Int32(T)
        if cutlass.const_expr(DROPIN):
            bos = Int32(gQsl[brow])
            seq_len = Int32(gQsl[brow + 1]) - bos
        is_null = raw_idx < Int32(self.null_min)
        sidx = _sel(raw_idx < 0, 0, raw_idx, Int32)  # in-range state load
        if cutlass.const_expr(not DROPIN):
            is_null = raw_idx < 0  # unused outside the drop-in

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(st_full[0], 1)
                cute.arch.mbarrier_init(st_full[1], 1)
                cute.arch.mbarrier_init(x_ready, 128)
                cute.arch.mbarrier_init(mma_done, 1)
                cute.arch.mbarrier_init(tmem_done, 128)
                cute.arch.mbarrier_init_fence()
            cute.arch.fence_acq_rel_cta()
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=64)
            cute.arch.barrier_arrive(barrier_id=4, number_of_threads=160)

        # token row of token 0 (dense [B, T, heads, D] or packed [tok, heads, D])
        tok0 = bos if DROPIN else brow * T

        ct = (tid - 64) & 127  # consumer index: channel / (row, lane) / TMEM lane
        row = ((warp & 3) << 5) | lane  # V row of this consumer (= TMEM lane)
        r8 = ct >> 4
        l16 = ct & 15
        q_base = gQ.iterator.toint()
        k_base = gK.iterator.toint()
        if cutlass.const_expr(DROPIN):
            sq_t = s_q_tok
            sk_t = s_k_tok
            sv_t = s_v_tok
            sg_t = s_g_tok
            sb_t = s_b_tok
            so_t = s_o_tok
        else:
            sq_t = H * K_DIM
            sk_t = H * K_DIM
            sv_t = HV * V_DIM
            sg_t = HV * K_DIM
            sb_t = HV
            so_t = HV * V_DIM
        xw = []
        for p in cutlass.range_constexpr(NP):
            xr = r8 + 8 * p
            is_q = xr >= T
            t_k = _sel(is_q, 0, xr, Int32)
            t_q = _sel(is_q, xr - T, 0, Int32)
            if cutlass.const_expr(PARTIAL and p == NP - 1):
                t_q = _sel(xr >= T2, 0, t_q, Int32)
            tt_ = _sel(is_q, t_q, t_k, Int32)
            live = Int32(tt_ < seq_len)
            if cutlass.const_expr(PARTIAL and p == NP - 1):
                live = Int32((tt_ < seq_len) & (xr < T2))
            ak = k_base + (
                Int64((tok0 + t_k)) * Int64(sk_t) + Int64(i_h * K_DIM + l16 * 8)
            ) * Int64(2)
            aq = q_base + (
                Int64((tok0 + t_q)) * Int64(sq_t) + Int64(i_h * K_DIM + l16 * 8)
            ) * Int64(2)
            xw.append(_ldg_v4_or0(live, _sel(is_q, aq, ak, Int64)))
        g_base = gG.iterator.toint()
        graw = []
        for t in cutlass.range_constexpr(T):
            graw.append(
                _ldg_u16_if(
                    Int32(t < seq_len),
                    g_base
                    + (Int64((tok0 + t)) * Int64(sg_t) + Int64(hv * K_DIM + ct))
                    * Int64(2),
                )
            )
        v_base = gV.iterator.toint()
        vv = []
        for t in cutlass.range_constexpr(T):
            vv.append(
                _ldg_u16_if(
                    Int32(t < seq_len),
                    v_base
                    + (Int64((tok0 + t)) * Int64(sv_t) + Int64(hv * V_DIM + row))
                    * Int64(2),
                )
            )
        tb = _sel(ct < T, ct, 0, Int32)
        b_base = gBeta.iterator.toint()
        braw = _ldg_u16_if(
            Int32((ct < T) & (tb < seq_len)),
            b_base + (Int64((tok0 + tb)) * Int64(sb_t) + Int64(hv)) * Int64(2),
        )
        eA = Float32(0.0)
        dtb = Float32(0.0)
        if cutlass.const_expr(GM != GATE_PRECOMPUTED):
            eA = _ex2(Float32(gAlog[i_h]) * Float32(LOG2_E))
            if cutlass.const_expr(self.has_dt_bias):
                dtb = Float32(gDtbias[i_h * K_DIM + ct])

        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        xc_addr = s_xc.iterator.toint()
        ktil_addr = s_ktil.iterator.toint()

        if warp == 0:
            # ============================================================ producer
            cpasync.prefetch_descriptor(tma_ld.atom)
            for gk in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(st_full[gk], 16384)
                U.simple_tma_copy(
                    tma_ld.atom,
                    tma_ld.tma_tensor[sidx, hv, None, (None, gk)],
                    s_state[None, (None, gk)],
                    st_full[gk],
                )
        elif warp == 1:
            # ============================================================ UMMA issuer
            tbase = tmem_addr[0]
            sd = U.sdesc_sw128(V_DIM * 128)
            x_desc = sd + Uint64(x_addr >> 4)
            stage_desc = [
                sd + Uint64(st_addr >> 4),
                sd + Uint64((st_addr + 16384) >> 4),
            ]
            idesc = _bf16_idesc(128, NX)
            cute.arch.barrier(barrier_id=3, number_of_threads=64)
            cute.arch.mbarrier_wait(x_ready, 0)
            U.fence_after_sync()
            for gk in cutlass.range_constexpr(2):
                cute.arch.mbarrier_wait(st_full[gk], 0)
                U.fence_after_sync()
                for ka in cutlass.range_constexpr(4):
                    U.umma_ss(
                        tbase,
                        stage_desc[gk] + Uint64((ka * 32) >> 4),
                        x_desc + Uint64((gk * XH + ka * 32) >> 4),
                        idesc,
                        (gk > 0) or (ka > 0),
                    )
            U.umma_commit(mma_done)
            cute.arch.mbarrier_wait(tmem_done, 0)
            U.fence_after_sync()
            U.tmem_dealloc(tbase, _TMEM_COLS)
        else:
            # ============================================================ consumers
            cw = warp - 2
            kg_base = gKg.iterator.toint()
            kg_live = Int32(1)
            if cutlass.const_expr(DROPIN):
                kg_live = Int32(raw_idx >= Int32(self.null_min))
            # -------- P1: L2 norm (row layout), raw k̂ / q̂ into X_c, kg cache k part
            for p in cutlass.range_constexpr(NP):
                xr = r8 + 8 * p
                ss = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    lo = U.bf16x2_lo_f32(xw[p][i])
                    hi = U.bf16x2_hi_f32(xw[p][i])
                    ss = ss + lo * lo
                    ss = ss + hi * hi
                for off in (8, 4, 2, 1):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, off)
                inv = _rsqrt_approx(ss + Float32(_EPS))
                inv = _sel(xr >= T, inv * scale, inv, Float32)
                nw = []
                for i in cutlass.range_constexpr(4):
                    nw.append(
                        U.pack_bf16x2(
                            U.bf16x2_lo_f32(xw[p][i]) * inv,
                            U.bf16x2_hi_f32(xw[p][i]) * inv,
                        )
                    )
                xcw = nw
                if cutlass.const_expr(PARTIAL and p == NP - 1):
                    xcw = [_sel(xr < T2, w_, Uint32(0), Uint32) for w_ in nw]
                    if xr >= T2:
                        zd = Uint32(0)
                        U.sts_v4(
                            x_addr
                            + (l16 // 8) * XH
                            + xr * 128
                            + (((l16 % 8) ^ (xr % 8)) * 16),
                            zd,
                            zd,
                            zd,
                            zd,
                        )
                U.sts_v4(
                    xc_addr + xr * XC_PITCH + l16 * 16, xcw[0], xcw[1], xcw[2], xcw[3]
                )
                if cutlass.const_expr(EMIT):
                    if xr < T:
                        # kg cache key half: L2-normed k (dense emit) or raw k (drop-in)
                        if cutlass.const_expr(DROPIN):
                            if (kg_live != 0) & (xr < seq_len):
                                kga = kg_base + (
                                    Int64(raw_idx) * Int64(s_kg_blk)
                                    + Int64(hv * s_kg_head + xr * s_kg_pos + l16 * 8)
                                ) * Int64(2)
                                U.stg_v4(kga, xw[p][0], xw[p][1], xw[p][2], xw[p][3])
                        else:
                            kga = kg_base + (
                                (Int64((tok0 + xr)) * Int64(HV) + Int64(hv))
                                * Int64(2 * K_DIM)
                                + Int64(l16 * 8)
                            ) * Int64(2)
                            U.stg_v4(kga, nw[0], nw[1], nw[2], nw[3])
            z = Uint32(0)
            for p in cutlass.range_constexpr(NP, NX // 8):
                zr = r8 + 8 * p
                U.sts_v4(xc_addr + zr * XC_PITCH + l16 * 16, z, z, z, z)
                U.sts_v4(
                    x_addr + (l16 // 8) * XH + zr * 128 + (((l16 % 8) ^ (zr % 8)) * 16),
                    z,
                    z,
                    z,
                    z,
                )
            if cutlass.const_expr(T < KT):
                if ct < (KT - T) * 16:
                    U.sts_v4(ktil_addr + (T + r8) * XC_PITCH + l16 * 16, z, z, z, z)
            _bar_sync_ua(1, 128)

            # -------- P2: gate + operand tiles (channel layout, c = ct)
            c = ct
            cum = Float32(0.0)
            for t in cutlass.range_constexpr(T):
                gx = U.u16_as_f32(graw[t], True)
                if cutlass.const_expr(EMIT):
                    if cutlass.const_expr(DROPIN):
                        if (kg_live != 0) & (Int32(t) < seq_len):
                            _stg_u16(
                                kg_base
                                + (
                                    Int64(raw_idx) * Int64(s_kg_blk)
                                    + Int64(hv * s_kg_head + t * s_kg_pos + K_DIM + c)
                                )
                                * Int64(2),
                                graw[t],
                            )
                    else:
                        _stg_u16(
                            kg_base
                            + (
                                (Int64((tok0 + t)) * Int64(HV) + Int64(hv))
                                * Int64(2 * K_DIM)
                                + Int64(K_DIM + c)
                            )
                            * Int64(2),
                            graw[t],
                        )
                glog = gx
                if cutlass.const_expr(GM == GATE_LOWER_BOUND):
                    xg = gx
                    if cutlass.const_expr(self.has_dt_bias):
                        xg = gx + dtb
                    sig = Float32(1.0) / (
                        Float32(1.0) + _ex2((Float32(0.0) - eA) * xg * Float32(LOG2_E))
                    )
                    glog = lower_bound * sig
                elif cutlass.const_expr(GM == GATE_SOFTPLUS):
                    xg = gx
                    if cutlass.const_expr(self.has_dt_bias):
                        xg = gx + dtb
                    sp = cute.math.log(Float32(1.0) + _ex2(xg * Float32(LOG2_E)))
                    glog = (Float32(0.0) - eA) * sp
                if cutlass.const_expr(DROPIN):
                    glog = _sel(Int32(t) < seq_len, glog, 0.0, Float32)
                cum = cum + glog
                ep = _ex2(cum * Float32(LOG2_E))
                en = _ex2((Float32(0.0) - cum) * Float32(LOG2_E))
                kh = U.u16_as_f32(U.lds_u16(xc_addr + t * XC_PITCH + c * 2), True)
                qh = U.u16_as_f32(U.lds_u16(xc_addr + (T + t) * XC_PITCH + c * 2), True)
                kw = U.f32_to_u16(kh * ep, True)
                qw = U.f32_to_u16(qh * ep, True)
                U.sts_u16(xc_addr + t * XC_PITCH + c * 2, kw)
                U.sts_u16(xc_addr + (T + t) * XC_PITCH + c * 2, qw)
                U.sts_u16(ktil_addr + t * XC_PITCH + c * 2, U.f32_to_u16(kh * en, True))
                U.sts_u16(x_addr + _x_sw128_off(t, c, XH), kw)
                U.sts_u16(x_addr + _x_sw128_off(T + t, c, XH), qw)
            if ct < T:
                bv = U.u16_as_f32(braw, True)
                if cutlass.const_expr(self.beta_is_logit):
                    bv = Float32(1.0) / (
                        Float32(1.0) + _ex2((Float32(0.0) - bv) * Float32(LOG2_E))
                    )
                s_coef[_CO_BETA + ct] = bv
            cute.arch.barrier(barrier_id=4, number_of_threads=160)  # warp 0's inits
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(x_ready)
            _bar_sync_ua(1, 128)

            # -------- P4: mma.sync Gram [X_c] . Ktil^T (fp32, s_gram [NX][KT])
            grp = lane >> 2
            c4 = lane & 3
            NTILE = (NX // 16) * (KT // 8)
            if cw < NTILE:
                mt = cw >> 1 if KT == 16 else cw
                nt = cw & 1 if KT == 16 else 0
                a_ln = xc_addr + (mt * 16 + lane % 16) * XC_PITCH + (lane // 16) * 16
                b_ln = (
                    ktil_addr + (nt * 8 + lane % 8) * XC_PITCH + ((lane // 8) % 2) * 16
                )
                acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for ks in cutlass.range_constexpr(8):
                    fa = _ldmatrix_x4(a_ln + ks * 32)
                    fb = _ldmatrix_x2(b_ln + ks * 32)
                    acc = U.mma_bf16_16816(
                        fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], acc
                    )
                r0 = mt * 16 + grp
                c0 = nt * 8 + 2 * c4
                s_gram[r0 * KT + c0] = acc[0]
                s_gram[r0 * KT + c0 + 1] = acc[1]
                s_gram[(r0 + 8) * KT + c0] = acc[2]
                s_gram[(r0 + 8) * KT + c0 + 1] = acc[3]
            _bar_sync_ua(1, 128)

            # -------- P5: intra-step coefficient tables (fp32, masked by select)
            NT_ = T * T
            for i in cutlass.range_constexpr((2 * NT_ + 127) // 128):
                e = ct + 128 * i
                if e < 2 * NT_:
                    tab = e // NT_
                    tt = (e // T) % T
                    ss_ = e % T
                    gv = s_gram[(tab * T + tt) * KT + ss_]
                    bt = s_coef[_CO_BETA + tt]
                    av = _sel(ss_ < tt, bt * gv, 0.0, Float32)
                    bq = _sel(ss_ <= tt, gv, 0.0, Float32)
                    if tab == 0:
                        s_coef[_CO_A + tt * TT + ss_] = av
                    else:
                        s_coef[_CO_BQ + tt * TT + ss_] = bq
            cute.arch.barrier(barrier_id=2, number_of_threads=160)  # TMEM address
            U.fence_after_sync()
            tbase = tmem_addr[0]
            _bar_sync_ua(1, 128)  # coefficient tables

            # -------- epilogue: per V row forward substitution
            cute.arch.mbarrier_wait(mma_done, 0)
            U.fence_after_sync()
            nld = 1 << (T - 1).bit_length()
            hk = U.tmem_ld(ct, tbase, nld)
            hq = U.tmem_ld(ct, tbase + T, nld)
            U.tmem_wait_ld()
            U.fence_before_sync()
            cute.arch.mbarrier_arrive(tmem_done)
            o_base = gOut.iterator.toint()
            c_base = gCorr.iterator.toint()
            uu: list = []
            for t in cutlass.range_constexpr(T):
                bt = s_coef[_CO_BETA + t]
                acc_u = bt * (U.u16_as_f32(vv[t], True) - hk[t])
                for s_ in cutlass.range_constexpr(t):
                    acc_u = acc_u - s_coef[_CO_A + t * TT + s_] * uu[s_]
                uu.append(acc_u)
            for t in cutlass.range_constexpr(T):
                y = hq[t]
                for s_ in cutlass.range_constexpr(t + 1):
                    y = y + s_coef[_CO_BQ + t * TT + s_] * uu[s_]
                if cutlass.const_expr(DROPIN):
                    y = _sel(is_null, 0.0, y, Float32)
                    if Int32(t) < seq_len:
                        _stg_u16(
                            o_base
                            + (
                                Int64((tok0 + t)) * Int64(so_t)
                                + Int64(hv * V_DIM + row)
                            )
                            * Int64(2),
                            U.f32_to_u16(y, True),
                        )
                        if kg_live != 0:
                            _stg_f32(
                                c_base
                                + (
                                    Int64(raw_idx) * Int64(s_c_blk)
                                    + Int64(hv * s_c_head + t * s_c_pos + row)
                                )
                                * Int64(4),
                                uu[t],
                            )
                else:
                    oa = (
                        Int64((tok0 + t)) * Int64(so_t) + Int64(hv * V_DIM + row)
                    ) * Int64(2)
                    _stg_u16(o_base + oa, U.f32_to_u16(y, True))
                    if cutlass.const_expr(EMIT):
                        _stg_u16(c_base + oa, U.f32_to_u16(uu[t], True))
