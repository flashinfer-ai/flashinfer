"""CuTeDSL KDA (Kimi Delta Attention) MTP decode kernel — WY-parallel,
OUTPUT-ONLY (frozen state).

Computes the decode output for 1..16 tokens per sequence from a read-only
initial recurrent state and never writes state back. This is the KDA analog of
``gdn_kernels.gdn_decode_bf16_wy_output_only`` (the GDN WY output-only kernel):
the serial per-token delta-rule recurrence is replaced by a chunk-parallel
(WY-representation) formulation evaluated with tensor-core MMAs.

This module contains the full frozen-decode surface, not just the WY kernel:

- the WY-parallel tensor-core kernel described below (the default for large
  problems), including its packed varlen / RecoverSSM-verify variant
  (``vllm_dropin``: ragged ``query_start_loc`` lengths, null slots,
  slot-indexed fp32 correction + kg caches);
- a grouped register-recurrent OUTPUT-ONLY kernel
  (``_kda_oo_grouped_kernel``) — an output-only fork of
  ``recurrent_kda._grouped_kda_kernel`` (#4001) with the per-token
  state-checkpoint writes and spec-decode bookkeeping removed — used as the
  fallback when both the batch (``B * HV``) and the window (``T``) are too
  small to fill the GPU (the WY kernel's fixed pipeline latency dominates
  there);
- the batch-size dispatcher and host wrappers that select between them.

Being a fork, fixes to the upstream grouped kernel do not propagate here
automatically. This docstring is canonical for the math; the host wrappers'
docstrings are canonical for tensor shapes, dtypes, and cache layouts.

Math (per (batch, value-head); state S is [V, K], per-channel decay on K):

  recurrence:  S_t = S_{t-1} ⊙ a_t  (a_t[k] = exp(g_t[k]), broadcast over V)
               u_t = S_t k_t
               S_t += beta_t (v_t - u_t) k_t^T
               o_t = S_t q_t

  WY form:     cum_t[k] = sum_{s<=t} g_s[k]        (per-channel log cumsum)
               khat_t = k_t * exp(cum_t)           qhat_t = q_t * exp(cum_t)
               ktil_t = k_t * exp(-cum_t)
               M      = strict_lower(khat @ ktil^T)      # decay folds in
               Tmat   = (I + diag(beta) M)^{-1} diag(beta)
               QT     = tril(qhat @ ktil^T) @ Tmat       # incl. diagonal
               A_full = qhat - QT @ khat
               O      = A_full @ S0^T + QT @ V

  verify caches (emit mode; the RecoverSSM commit-kernel inputs):
               U      = Tmat @ (V - khat @ S0^T)  # row t = beta_t (v_t - u_t)
               kg_t   = (k_t | raw g_t)           # RAW key | raw gate (vLLM
                                                  # caches the unnormalized k)

In emit mode the kernel additionally writes these per-token caches
(slot-indexed; fp32 U, bf16 kg) for a downstream commit/recovery kernel: U
falls out of one extra state GEMM sharing the TMA-resident state tile plus a
single T x T MMA, and kg is captured in the gate stage before khat overwrites
the normalized key in SMEM.

The only structural differences from the GDN kernel are (1) the per-channel
gate cumsum + khat/ktil/qhat elementwise scaling stage (which replaces GDN's
per-token scalar gamma path and its post-GEMM exp(cum_r - cum_c) decay
factors), and (2) one extra [T, K] SMEM tile because khat and ktil must
coexist. The T x T triangular inverse, the TMA-loaded state GEMM (half-K
double-issue into a SW128-swizzled tile), the QT@V epilogue and the
SMEM-staged coalesced output flush are inherited unchanged.

Gate modes (compile-time):
  GATE_PRECOMPUTED (0): g is the log-space gate, used as-is.
  GATE_LOWER_BOUND (1): g_log = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
                        (the Kimi K3 contract; lower_bound < 0, e.g. -5).
  GATE_SOFTPLUS    (2): g_log = -exp(A_log) * softplus(g + dt_bias)
                        (the original Kimi-Linear gate).

Numerical-range note: ktil rows carry exp(-cum) which grows with total decay.
For the Kimi K3 lower-bound gate, |cum| <= |lower_bound| * T = 80, and
exp(80) * |k| (k L2-normalized) is representable in bf16/f32 with adequate
relative precision (each causal product khat_s[c]*ktil_r[c] is individually
bounded by |k_s[c] k_r[c]|, so there is no cancellation). The unbounded
softplus gate can overflow for pathologically strong decay (roughly
sum_t |g_log_t[k]| > ~85); the grouped recurrent kernel below remains the
fallback there.

Requires SM90+ (TMA + mbarrier), validated on SM100a (B200); K == V == 128; bf16 I/O and state.
"""

import math
import os as _os
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
from cutlass import const_expr
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.arch import sync_threads
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T

# Reuse the validated PTX / layout helpers from the GDN WY kernel — the tile
# geometry (T=16, K=V=128, K_PADDED=136, BF_PAD=24, SW128 sH) is identical, so
# these are shared verbatim rather than re-transcribed.
from ..gdn_kernels.gdn_decode_bf16_wy_output_only import (
    _afull_4mma,
    _cp_async_bf16x8,
    _cp_async_commit_group,
    _cp_async_wait_group_0,
    _dot_sq_bf16x2,
    _exp2_approx_f32,
    _fused_ab_4mma_serial_brow,
    _fused_ab_1mma,
    _h_gemm_4v,
    _ldmatrix_x4,
    _lds_v4_b32,
    _make_sH_sw128_layout_half,
    _mul_bf16x2_f32,
    _qtv_4mma,
    _rsqrt_approx_f32,
    _smat_off,
    _st_global_v4_b32,
    _sts_bf16x2_f32,
    _sw128_xor,
)


def _sts_f32x2(smem_addr_i32, lo_f32, hi_f32):
    """st.shared.v2.f32: store an adjacent f32 pair (address 8 B aligned).

    Used to stage fp32 correction fragments into the h_buf-aliased W tile
    without a bf16 round-trip (the fp32 sibling of _sts_bf16x2_f32).
    """
    r = llvm.inline_asm(
        mlir_T.i32(),
        [smem_addr_i32.ir_value(), lo_f32.ir_value(), hi_f32.ir_value()],
        "{ st.shared.v2.f32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(r)


# Fixed dims (must match the imported helpers' module constants)
T = 16
K_DIM = 128
V_DIM_C = 128
EPS = 1e-6
io = cutlass.BFloat16
f32 = cutlass.Float32
WARP = 32
THREADS = 128

K_HALF = K_DIM // 2  # 64 — half-K streaming for the state (sH) TMA
K_PADDED = K_DIM + 8  # 136 — padded row stride for sKhat / sKtil / sQV
V_PADDED = V_DIM_C + 8  # 136

TK_PAD = T * K_PADDED
TT = T * T
BF_PAD = 24

LOG2_E = 1.4426950408889634

# Gate modes (compile-time constexpr)
GATE_PRECOMPUTED = 0
GATE_LOWER_BOUND = 1
GATE_SOFTPLUS = 2


class KdaDecodeWyOutputOnlyKernel:
    """CuTeDSL KDA MTP decode — WY-parallel, output-only (frozen state)."""

    def __init__(
        self,
        min_blocks_per_mp=2,
        t_input=16,
        gate_mode=GATE_PRECOMPUTED,
        has_dt_bias=False,
        beta_is_logit=False,
        n_valid=16,
        emit_corrections=False,
        vllm_dropin=False,
        null_min=1,
    ):
        """Bind the compile-time specialization knobs (see class docstring)."""
        self._min_blocks_per_mp = min_blocks_per_mp
        # T-aware compute-specialization bucket (4 / 8 / 16).
        self._t_input = int(t_input)
        self._gate_mode = int(gate_mode)
        self._has_dt_bias = bool(has_dt_bias)
        self._beta_is_logit = bool(beta_is_logit)
        # Emit-corrections (vLLM RecoverSSM verify contract): also write the
        # per-token corrections U_t = beta_t * (v_t - u_t) and a kg cache
        # (L2-normalized k | raw gate) consumed by the commit kernel.
        self._emit = bool(emit_corrections)
        # Native short-T: number of token rows actually present in the
        # q/k/v/g/beta gmem tensors (their batch stride uses this count).
        # When n_valid < T the kernel loads only those rows and zeros the
        # SMEM tails itself — no host staging copies. The output tensor also
        # has n_valid rows; stores are gated accordingly.
        self._n_valid = int(n_valid)
        # vLLM RecoverSSM drop-in contract (_kda_recoverssm_verify_kernel):
        # varlen-packed [1, total_tokens, ...] inputs addressed via
        # query_start_loc + runtime token strides, runtime per-sequence
        # length masks, null-slot semantics (state_idx <= 0 -> zero outputs,
        # no cache writes), slot-indexed fp32 correction cache
        # [blocks, H, spec_len, V], and a raw-k | raw-g kg cache
        # [blocks, H, spec_len, 2K]. Implies emit machinery and beta logits.
        self._dropin = bool(vllm_dropin)
        # Smallest valid state slot in dropin mode: slots below it are null
        # (zero outputs, untouched caches). 1 = vLLM (slot 0 reserved),
        # 0 = flashinfer recurrent_kda (only negative slots are padding).
        self._null_min = int(null_min)
        if self._dropin:
            self._emit = True
            # beta_is_logit is NOT forced here: the vLLM contract uses beta
            # logits (wrapper default), but the unified recurrent_kda frozen
            # mode also routes pre-sigmoided betas through this path.
            self._n_valid = 16  # runtime seq_len masks replace native-T

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
        scale: cutlass.Float32,
        lower_bound: cutlass.Float32,
        s_q_tok: cutlass.Int32,
        s_k_tok: cutlass.Int32,
        s_v_tok: cutlass.Int32,
        s_g_tok: cutlass.Int32,
        s_b_tok: cutlass.Int32,
        s_o_tok: cutlass.Int32,
        s_c_blk: cutlass.Int32,
        s_c_head: cutlass.Int32,
        s_c_pos: cutlass.Int32,
        s_kg_blk: cutlass.Int32,
        s_kg_head: cutlass.Int32,
        s_kg_pos: cutlass.Int32,
        s_h0_blk: cutlass.Int64,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Build the state TMA atom and launch one CTA per (batch, head)."""
        op = MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)
        # Batch size (grid extent: one CTA column per sequence) — NOT the
        # state-pool size; see the pool comment below.
        B_val = gH0idx.layout.shape[0]
        # State TMA: rebuild the pool layout with STATIC inner strides
        # (blocks are dense [HV, V, K]) and a RUNTIME block stride — real
        # serving pools (e.g. vLLM's KDA cache) pad between blocks. The TMA
        # tile modes must be static, so the incoming (possibly fully
        # dynamic) descriptor layout is not used directly. Reorder to
        # (V, K, HV, pool) so the per-CTA tile is (V_DIM_C, K_HALF); the
        # trailing (HV, pool) modes survive tma_partition as outer coords.
        _h0_blk = cute.assume(s_h0_blk, divby=8)  # 16 B alignment (bf16)
        # gH0.layout.shape[0] is the state-pool SLOT COUNT (the TMA
        # descriptor's outer extent) — not B_val: slot indices >= B are legal,
        # so substituting the batch size would shrink the TMA box and corrupt
        # loads for high slots. Both extents are runtime-dynamic (mode-0
        # shape-dynamic descriptors); no constexpr is at stake.
        gH0_dense = cute.make_tensor(
            gH0.iterator,
            cute.make_layout(
                (gH0.layout.shape[0], HV, V_DIM_C, K_DIM),
                stride=(_h0_blk, V_DIM_C * K_DIM, K_DIM, 1),
            ),
        )
        gH0_vkhp = cute.make_tensor(
            gH0_dense.iterator,
            cute.select(gH0_dense.layout, mode=[2, 3, 1, 0]),
        )
        sH_tma_layout = _make_sH_sw128_layout_half()
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gH0_vkhp,
            sH_tma_layout,
            (V_DIM_C, K_HALF),
        )
        self.kernel(
            gQ,
            gK,
            gV,
            gG,
            gBeta,
            gAlog,
            gDtbias,
            gH0,
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
            tiled_mma,
            HV,
            V_DIM,
            H,
            tma_atom_h,
            tma_tensor_h,
        ).launch(
            grid=(1, HV, B_val),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp,
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
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gOut: cute.Tensor,
        gCorr: cute.Tensor,
        gKg: cute.Tensor,
        gQsl: cute.Tensor,
        scale: cutlass.Float32,
        lower_bound: cutlass.Float32,
        s_q_tok: cutlass.Int32,
        s_k_tok: cutlass.Int32,
        s_v_tok: cutlass.Int32,
        s_g_tok: cutlass.Int32,
        s_b_tok: cutlass.Int32,
        s_o_tok: cutlass.Int32,
        s_c_blk: cutlass.Int32,
        s_c_head: cutlass.Int32,
        s_c_pos: cutlass.Int32,
        s_kg_blk: cutlass.Int32,
        s_kg_head: cutlass.Int32,
        s_kg_pos: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
    ):
        """Device kernel: WY-parallel output-only KDA decode (see module
        docstring for the math and the section comments for the pipeline)."""
        # Strides — contiguous staged layout: q/k [B,T,H,K], v [B,T,HV,V],
        # g [B,T,HV,K], beta [B,T,HV], out [B,T,HV,V].
        _nv = self._n_valid
        sq_h = K_DIM
        sq_t = H * K_DIM
        sq_b = _nv * sq_t
        sk_h = K_DIM
        sk_t = H * K_DIM
        sk_b = _nv * sk_t
        sv_hv = V_DIM
        sv_t = HV * V_DIM
        sv_b = _nv * sv_t
        sg_hv = K_DIM
        sg_t = HV * K_DIM
        sg_b = _nv * sg_t
        sb_hv = cutlass.Int32(1)
        sb_t = HV
        sb_b = _nv * HV
        so_hv = V_DIM
        so_t = HV * V_DIM
        so_b = _nv * so_t
        skg_hv = 2 * K_DIM
        skg_t = HV * 2 * K_DIM
        skg_b = _nv * skg_t

        tidx, _, _ = cute.arch.thread_idx()
        _pid_vt, pid_hv, pid_b = cute.arch.block_idx()
        lane_id = tidx & 31
        warp_id = tidx // WARP

        # GQA head mapping (query head for q/k; gate/beta/v are per value head)
        i_h = pid_hv // (HV // H)
        cache_idx = gH0idx.iterator[pid_b]

        # (vllm-dropin) varlen metadata: per-sequence token range and runtime
        # length; null-slot semantics (state_idx <= NULL_BLOCK_ID = 0 -> zero
        # outputs, no cache writes). The state slot is clamped for the TMA so
        # the discarded null-path load stays in bounds (slot 0 exists).
        _bos = cutlass.Int32(0)
        _seq_len = cutlass.Int32(_nv)
        _state_idx = cache_idx
        _is_null = False
        if const_expr(self._dropin):
            _bos = gQsl.iterator[pid_b].to(cutlass.Int32)
            _seq_len = gQsl.iterator[pid_b + 1].to(cutlass.Int32) - _bos
            _state_idx = cache_idx
            _is_null = _state_idx < cutlass.Int32(self._null_min)
            if cache_idx < 0:
                cache_idx = cutlass.Int32(0)

        # Hoist warp-3 beta LDGs to kernel start (maximum HBM latency hiding).
        _beta_bf16 = f32(0.0)
        if const_expr(self._dropin):
            if warp_id == 3 and lane_id < _seq_len:
                _beta_bf16 = gBeta.iterator[(_bos + lane_id) * s_b_tok + pid_hv].to(f32)
        else:
            if warp_id == 3 and lane_id < _nv:
                _beta_bf16 = gBeta.iterator[
                    pid_b * sb_b + lane_id * sb_t + pid_hv * sb_hv
                ].to(f32)
        # Per-head gate constants for the in-kernel gate modes: A_log scalar
        # (broadcast) and one dt_bias element per K channel (thread == channel).
        _exp_A = f32(0.0)
        _dtb = f32(0.0)
        if const_expr(self._gate_mode != GATE_PRECOMPUTED):
            _exp_A = _exp2_approx_f32(gAlog.iterator[i_h].to(f32) * f32(LOG2_E))
            if const_expr(self._has_dt_bias):
                _dtb = gDtbias.iterator[i_h * K_DIM + tidx].to(f32)

        smem = utils.SmemAllocator()

        # The fp32 W (corrections) stage lives inside h_buf (see the emit
        # epilogue): out-stage region [0, 8704) is dead after the output
        # flush and the D tile sits at [8704, 13056). Reusing it reclaims
        # ~8.7 KB/CTA of SMEM (one occupancy step at large batch); this
        # placeholder stays only to avoid struct churn.
        _WSTAGE_N = 8

        @cute.struct
        class SS:
            """Per-CTA SMEM plan (buffers documented inline)."""

            h_load_mbar: cute.struct.MemRange[Int64, 1]
            # khat_buf: K loads here; becomes khat (k * exp(cum)) in place.
            khat_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # ktil_buf: ktil (k * exp(-cum)); later overwritten by A_full.
            ktil_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # qv_buf: Q loads here; becomes qhat in place; later aliased by V
            # (qhat's last read is the A_full epilogue; a sync precedes the V
            # cp.async fill — same discipline as the GDN kernel's qv aliasing).
            qv_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # g_buf: raw/log gate tile [T, K] bf16 (row-contiguous, no pad —
            # consumed row-wise by the per-channel cumsum, conflict-free).
            g_buf: cute.struct.Align[cute.struct.MemRange[io, T * K_DIM], 128]
            # h_buf: state half-tile (V=128 x K=64 bf16, SW128) = 16 KiB;
            # reused across the two TMA halves; aliased by the output stage.
            h_buf: cute.struct.Align[cute.struct.MemRange[io, V_DIM_C * K_HALF], 128]
            tmat_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            beta: cute.struct.Align[cute.struct.MemRange[f32, WARP], 128]
            mat_fp32: cute.struct.Align[cute.struct.MemRange[f32, TT], 128]
            scratch_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            scratch2_bf: cute.struct.Align[cute.struct.MemRange[io, T * BF_PAD], 128]
            # (vllm-dropin) fp32 staging tile for the corrections flush
            # (vLLM's correction cache is fp32). Collapsed to 8 elems when
            # the drop-in mode is off so other modes keep their occupancy.
            wstage_f32: cute.struct.Align[cute.struct.MemRange[f32, _WSTAGE_N], 128]

        st = smem.allocate(SS)
        sK = st.khat_buf.get_tensor(
            cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1))
        )
        sKtil = st.ktil_buf.get_tensor(
            cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1))
        )
        sQ = st.qv_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        sV = st.qv_buf.get_tensor(cute.make_layout((T, V_PADDED), stride=(V_PADDED, 1)))
        sG = st.g_buf.get_tensor(cute.make_layout((T, K_DIM), stride=(K_DIM, 1)))
        sH_layout = _make_sH_sw128_layout_half()
        sH = st.h_buf.get_tensor(sH_layout.outer, swizzle=sH_layout.inner)
        sTmat = st.tmat_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sBeta = st.beta.get_tensor(cute.make_layout((WARP,)))
        sMat = st.mat_fp32.get_tensor(cute.make_layout((T, T), stride=(T, 1)))
        sNegL = st.scratch_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))
        sPowk = st.scratch2_bf.get_tensor(cute.make_layout((T, T), stride=(BF_PAD, 1)))

        # mbarrier init for the state TMA load (single arriver + TX bytes).
        mbar_h_ptr = st.h_load_mbar.data_ptr()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_h_ptr, 1)
        cute.arch.mbarrier_init_fence()
        sync_threads()

        # Partition the state TMA tensor for this CTA's (cache_idx, pid_hv).
        gH_tiled = cute.flat_divide(tma_tensor_h, (V_DIM_C, K_HALF))
        gH_slice0 = gH_tiled[None, None, None, 0, pid_hv, cache_idx]
        gH_slice1 = gH_tiled[None, None, None, 1, pid_hv, cache_idx]
        gH_grp0 = cute.group_modes(gH_slice0, 0, 3)
        gH_grp1 = cute.group_modes(gH_slice1, 0, 3)
        sH_grp = cute.group_modes(sH, 0, 2)
        tHsH0, tHgH0 = cpasync.tma_partition(
            tma_atom_h, 0, cute.make_layout(1), sH_grp, gH_grp0
        )
        tHsH1, tHgH1 = cpasync.tma_partition(
            tma_atom_h, 0, cute.make_layout(1), sH_grp, gH_grp1
        )

        thr_mma = tiled_mma.get_slice(lane_id)
        tCsC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((T, 8)))
        acc = cute.make_fragment_like(tCsC)
        _ldm_row = (lane_id % 8) + ((lane_id // 8) % 2) * Int32(8)

        EPT_TT = TT // THREADS

        # ============================================================
        # cp.async stage 1: K + Q + G (8 bf16 / instr, .ca)
        # ============================================================
        k_base = pid_b * sk_b + i_h * sk_h
        q_base = pid_b * sq_b + i_h * sq_h
        g_base = pid_b * sg_b + pid_hv * sg_hv
        _sq_tok = cutlass.Int32(sq_t)
        _sk_tok = cutlass.Int32(sk_t)
        _sg_tok = cutlass.Int32(sg_t)
        if const_expr(self._dropin):
            # varlen token addressing: rows live at (bos + t) * token_stride
            k_base = _bos * s_k_tok + i_h * sk_h
            q_base = _bos * s_q_tok + i_h * sq_h
            g_base = _bos * s_g_tok + pid_hv * sg_hv
            _sq_tok = s_q_tok
            _sk_tok = s_k_tok
            _sg_tok = s_g_tok
        _gK_base = gK.iterator.toint()
        _gQ_base = gQ.iterator.toint()
        _gG_base = gG.iterator.toint()
        _sK_i32 = cute.recast_tensor(sK, cutlass.Int32)
        _sQ_i32 = cute.recast_tensor(sQ, cutlass.Int32)
        _kpad_i32 = K_PADDED // 2
        _sK_base_async = sK.iterator.toint()
        _sQ_base_async = sQ.iterator.toint()
        _sG_base_async = sG.iterator.toint()
        for i in cutlass.range_constexpr(T * K_DIM // (THREADS * 8)):
            _kq_group = tidx + i * THREADS
            _kq_row = _kq_group // Int32(K_DIM // 8)
            _kq_col_bf16_async = (_kq_group % Int32(K_DIM // 8)) * Int32(8)
            _smem_byte_off = _kq_row * Int32(K_PADDED * 2) + _kq_col_bf16_async * Int32(
                2
            )
            _g_smem_byte_off = _kq_row * Int32(K_DIM * 2) + _kq_col_bf16_async * Int32(
                2
            )
            # (native short-T) the gmem tensors hold only n_valid rows; skip
            # the cp.async for rows >= n_valid (OOB). The sK/sQ SMEM tails are
            # zeroed after the wait below; sG tail rows are skipped by the
            # constexpr gate-stage loop, so they may stay garbage.
            # (vllm-dropin) rows are gated by the runtime per-sequence length
            # and addressed with runtime token strides.
            if const_expr(self._dropin):
                if _kq_row < _seq_len:
                    _cp_async_bf16x8(
                        _gK_base,
                        k_base + _kq_row * _sk_tok + _kq_col_bf16_async,
                        _sK_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gQ_base,
                        q_base + _kq_row * _sq_tok + _kq_col_bf16_async,
                        _sQ_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gG_base,
                        g_base + _kq_row * _sg_tok + _kq_col_bf16_async,
                        _sG_base_async + _g_smem_byte_off,
                    )
            elif const_expr(self._n_valid < T):
                if _kq_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gK_base,
                        k_base + _kq_row * sk_t + _kq_col_bf16_async,
                        _sK_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gQ_base,
                        q_base + _kq_row * sq_t + _kq_col_bf16_async,
                        _sQ_base_async + _smem_byte_off,
                    )
                    _cp_async_bf16x8(
                        _gG_base,
                        g_base + _kq_row * sg_t + _kq_col_bf16_async,
                        _sG_base_async + _g_smem_byte_off,
                    )
            else:
                _cp_async_bf16x8(
                    _gK_base,
                    k_base + _kq_row * sk_t + _kq_col_bf16_async,
                    _sK_base_async + _smem_byte_off,
                )
                _cp_async_bf16x8(
                    _gQ_base,
                    q_base + _kq_row * sq_t + _kq_col_bf16_async,
                    _sQ_base_async + _smem_byte_off,
                )
                _cp_async_bf16x8(
                    _gG_base,
                    g_base + _kq_row * sg_t + _kq_col_bf16_async,
                    _sG_base_async + _g_smem_byte_off,
                )
        _cp_async_commit_group()  # group 0 = K + Q + G

        # ============================================================
        # State TMA: issue FIRST half (K = 0..63). Hidden behind Phase 1/2.
        # ============================================================
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,  # 16384 B (half tile)
                )
            cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)

        # ============================================================
        # warp 3: beta (per-token scalar). No gamma cumsum here — KDA's decay
        # is per-channel and handled in the post-L2norm cumsum stage below.
        # ============================================================
        if warp_id == 3:
            beta_val = f32(0.0)
            if lane_id < T:
                beta_val = _beta_bf16
                if const_expr(self._beta_is_logit):
                    beta_val = f32(1.0) / (
                        f32(1.0) + _exp2_approx_f32((f32(0.0) - beta_val) * f32(LOG2_E))
                    )
            if lane_id < T:
                sBeta.iterator[lane_id] = beta_val

        # ============================================================
        # Wait for K+Q+G, then L2-norm K (warps 0,1) and Q (warps 2,3).
        # Identical to the GDN kernel (t-aware warp skip at T<=8).
        # ============================================================
        _cp_async_wait_group_0()
        # (native short-T) zero the sK/sQ tail rows [n_valid:T) that were not
        # loaded — the MMAs read the full tile, and the zero rows carry the
        # t<=8 dead-elision proofs. tidx covers K_DIM=128 cols exactly; the
        # pad cols are never read.
        # (vllm-dropin) the tail is [seq_len:T) at runtime.
        if const_expr(self._dropin):
            for _zr in cutlass.range_constexpr(T):
                if cutlass.Int32(_zr) >= _seq_len:
                    sK.iterator[_zr * K_PADDED + tidx] = io(0.0)
                    sQ.iterator[_zr * K_PADDED + tidx] = io(0.0)
        elif const_expr(self._n_valid < T):
            for _zr in cutlass.range_constexpr(self._n_valid, T):
                sK.iterator[_zr * K_PADDED + tidx] = io(0.0)
                sQ.iterator[_zr * K_PADDED + tidx] = io(0.0)
        sync_threads()

        # (vllm-dropin) kg cache: RAW k and RAW gate per valid token, stored
        # BEFORE the L2 norm mutates sK (the vLLM verify kernel caches the
        # unnormalized key). Slot-indexed layout [blocks, H, spec_len, 2K];
        # rows >= seq_len and null slots are left untouched, matching the
        # Triton kernel's token_valid masking. The extra sync publishes the
        # reads before the L2-norm warps overwrite sK in place.
        if const_expr(self._dropin):
            # skip null slots (avoid DSL `not` on runtime bool)
            if _state_idx >= cutlass.Int32(self._null_min):
                _kg_cta = _state_idx * s_kg_blk + pid_hv * s_kg_head
                for _kt in cutlass.range_constexpr(T):
                    if cutlass.Int32(_kt) < _seq_len:
                        _kg_off = _kg_cta + _kt * s_kg_pos + tidx
                        gKg.iterator[_kg_off] = sK.iterator[_kt * K_PADDED + tidx]
                        gKg.iterator[_kg_off + K_DIM] = sG.iterator[_kt * K_DIM + tidx]
            sync_threads()

        if const_expr(self._t_input <= 8):
            if warp_id == Int32(0):
                norm_row = lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sK_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                for c in cutlass.range_constexpr(16):
                    _sK_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sK_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
            if warp_id == Int32(2):
                norm_row = lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sQ_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                inv_norm = inv_norm * scale
                for c in cutlass.range_constexpr(16):
                    _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        else:
            if warp_id < 2:
                norm_row = warp_id * 8 + lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sK_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                for c in cutlass.range_constexpr(16):
                    _sK_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sK_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
            if warp_id >= 2:
                norm_row = (warp_id - 2) * 8 + lane_id // 4
                norm_quarter = lane_id % 4
                _norm_off_i32 = norm_row * (K_PADDED // 2) + norm_quarter
                partial = f32(0.0)
                for c in cutlass.range_constexpr(16):
                    packed = _sQ_i32.iterator[_norm_off_i32 + 4 * c]
                    partial = _dot_sq_bf16x2(packed, partial)
                for d in [1, 2]:
                    other = cute.arch.shuffle_sync(
                        partial, Int32(lane_id ^ d), Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    partial = partial + other
                inv_norm = _rsqrt_approx_f32(partial + f32(EPS))
                inv_norm = inv_norm * scale
                for c in cutlass.range_constexpr(16):
                    _sQ_i32.iterator[_norm_off_i32 + 4 * c] = _mul_bf16x2_f32(
                        _sQ_i32.iterator[_norm_off_i32 + 4 * c], inv_norm
                    )
        sync_threads()

        # ============================================================
        # KDA GATE STAGE (replaces GDN's scalar gamma path).
        # Thread tidx owns K-channel c = tidx (THREADS == K_DIM == 128).
        # Serial cumsum over the T token rows; forms in place:
        #   sK[t,c]    <- k[t,c] * exp(cum_t[c])     (khat)
        #   sKtil[t,c] <- k[t,c] * exp(-cum_t[c])    (ktil)
        #   sQ[t,c]    <- q[t,c] * exp(cum_t[c])     (qhat)
        # Row-wise SMEM access: 128 consecutive threads hit 128 consecutive
        # bf16 elements per step — conflict-free. Zero-padded tail rows
        # (t >= t_input) have k=q=0, so khat/ktil/qhat stay zero regardless
        # of the (arbitrary) tail gate values; cum for valid rows only
        # accumulates over valid rows, so the tail cannot contaminate them.
        # ============================================================
        _cum = f32(0.0)
        # (native short-T) tail rows [n_valid:T) have no gate data (sG not
        # loaded); sK/sQ tails are already zero, so khat/qhat stay zero. Only
        # sKtil must be explicitly zeroed (it was never written).
        if const_expr(self._n_valid < T):
            for _zt in cutlass.range_constexpr(self._n_valid, T):
                sKtil.iterator[_zt * K_PADDED + tidx] = io(0.0)
        for t in cutlass.range_constexpr(self._n_valid):
            _graw = sG.iterator[t * K_DIM + tidx].to(f32)
            _glog = f32(0.0)
            if const_expr(self._gate_mode == GATE_LOWER_BOUND):
                _x = _graw
                if const_expr(self._has_dt_bias):
                    _x = _graw + _dtb
                _sig = f32(1.0) / (
                    f32(1.0) + _exp2_approx_f32((f32(0.0) - _exp_A) * _x * f32(LOG2_E))
                )
                _glog = lower_bound * _sig
            elif const_expr(self._gate_mode == GATE_SOFTPLUS):
                _x = _graw
                if const_expr(self._has_dt_bias):
                    _x = _graw + _dtb
                # softplus(x) = log(1 + exp(x))
                _sp = cute.math.log(f32(1.0) + _exp2_approx_f32(_x * f32(LOG2_E)))
                _glog = (f32(0.0) - _exp_A) * _sp
            else:
                _glog = _graw
            if const_expr(self._dropin):
                # rows past this sequence's length carry garbage gates; force
                # a zero log-gate so the cumulative sums stay finite (their
                # khat/ktil/qhat rows are zero anyway: k = q = 0 there).
                if cutlass.Int32(t) >= _seq_len:
                    _glog = f32(0.0)
            _cum = _cum + _glog
            _ep = _exp2_approx_f32(_cum * f32(LOG2_E))
            _en = _exp2_approx_f32((f32(0.0) - _cum) * f32(LOG2_E))
            _koff = t * K_PADDED + tidx
            _kval = sK.iterator[_koff].to(f32)
            _qval = sQ.iterator[_koff].to(f32)
            if const_expr(self._emit and not self._dropin):
                # kg cache row: sK still holds the L2-normalized k here; sG
                # holds the raw gate. Coalesced 2-byte stores across channels.
                _kg_off = pid_b * skg_b + t * skg_t + pid_hv * skg_hv + tidx
                gKg.iterator[_kg_off] = sK.iterator[_koff]
                gKg.iterator[_kg_off + K_DIM] = sG.iterator[t * K_DIM + tidx]
            sK.iterator[_koff] = (_kval * _ep).to(io)
            sKtil.iterator[_koff] = (_kval * _en).to(io)
            sQ.iterator[_koff] = (_qval * _ep).to(io)
        sync_threads()

        # ============================================================
        # KKT (warps 0-1): khat @ ktil^T   ||   QKT (warps 2-3): qhat @ ktil^T
        # The per-channel decay is already folded into the operands, so no
        # post-GEMM decay factors are needed (unlike GDN).
        # ============================================================
        acc.fill(f32(0.0))
        _sK_int = sK.iterator.toint()
        _sKtil_int = sKtil.iterator.toint()
        _sQ_int = sQ.iterator.toint()
        _rs_kpad = Int32(K_PADDED * 2)
        _lane_mod16 = lane_id & Int32(15)
        _lane_hi = (lane_id >> Int32(4)) * Int32(16)
        _lane_mod8 = lane_id % Int32(8)
        _lane_b_col = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
        for kk_group in cutlass.range_constexpr(K_DIM // 16 // 4):
            k_group_off = kk_group * 4 * 16 * Int32(2)
            if warp_id < 2:
                col_off = warp_id * 8
                _a_base = _sK_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sKtil_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
                )
                acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                    _fused_ab_4mma_serial_brow(
                        _a_base,
                        _b_direct,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                )
            if warp_id >= 2:
                col_off = (warp_id - Int32(2)) * Int32(8)
                _a_base = _sQ_int + _lane_mod16 * _rs_kpad + _lane_hi + k_group_off
                _b_direct = (
                    _sKtil_int
                    + (col_off + _lane_mod8) * _rs_kpad
                    + k_group_off
                    + _lane_b_col
                )
                acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                    _fused_ab_4mma_serial_brow(
                        _a_base,
                        _b_direct,
                        acc.iterator[0],
                        acc.iterator[1],
                        acc.iterator[2],
                        acc.iterator[3],
                    )
                )
        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2
        if warp_id < 2:
            col_off = warp_id * 8
            sMat.iterator[_smat_off(_r0, col_off + _c0)] = acc.iterator[0]
            sMat.iterator[_smat_off(_r0, col_off + _c0 + 1)] = acc.iterator[1]
            sMat.iterator[_smat_off(_r0 + 8, col_off + _c0)] = acc.iterator[2]
            sMat.iterator[_smat_off(_r0 + 8, col_off + _c0 + 1)] = acc.iterator[3]
        if warp_id >= 2:
            col_off = (warp_id - Int32(2)) * Int32(8)
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0] = acc.iterator[0].to(io)
            sNegL.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[1].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[2].to(io)
            sNegL.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = acc.iterator[3].to(
                io
            )
        sync_threads()

        # ============================================================
        # PHASE 2 masking: causal-mask QKT (keep diag) and form
        # -beta_r * strict_lower(KKT) — no decay factors (already folded).
        # ============================================================
        for idx in cutlass.range_constexpr(EPT_TT):
            flat = tidx + idx * THREADS
            r = flat // T
            c = flat % T
            qkt = sNegL.iterator[r * BF_PAD + c].to(f32)
            sNegL.iterator[r * BF_PAD + c] = qkt.to(io) if r >= c else io(0.0)
            kkt_val = sMat.iterator[_smat_off(r, c)]
            negL_val = (f32(0.0) - sBeta.iterator[r] * kkt_val) if r > c else f32(0.0)
            sTmat.iterator[r * BF_PAD + c] = negL_val.to(io)
        sync_threads()

        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2

        # ============================================================
        # Block inverse (register-resident forward substitution) — identical
        # to the GDN kernel, incl. the T-aware skip structure.
        #   Tmat = (I - M)^{-1} diag(beta), M = -beta_r * strict_lower(KKT)
        # ============================================================
        if const_expr(self._t_input <= 8):
            if warp_id == Int32(0):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t00 = [None] * 8
                    _x_t00[0] = (
                        sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                    )
                    if const_expr(self._t_input <= 4):
                        for _r in cutlass.range_constexpr(1, 4):
                            _accum = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                            for _k in cutlass.range_constexpr(_r):
                                _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                                _accum = _accum + _m_rk * _x_t00[_k]
                            _x_t00[_r] = _accum
                        for _r in cutlass.range_constexpr(4, 8):
                            _x_t00[_r] = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                    else:
                        for _r in cutlass.range_constexpr(1, 8):
                            _accum = (
                                sBeta.iterator[Int32(_r)]
                                if _col == Int32(_r)
                                else f32(0.0)
                            )
                            for _k in cutlass.range_constexpr(_r):
                                _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                                _accum = _accum + _m_rk * _x_t00[_k]
                            _x_t00[_r] = _accum
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        _v = (
                            sBeta.iterator[Int32(8 + _r)]
                            if _col == Int32(_r)
                            else f32(0.0)
                        )
                        sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _v
            sync_threads()
            sync_threads()
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = f32(0.0)
            sync_threads()
        else:
            if warp_id == Int32(0):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t00 = [None] * 8
                    _x_t00[0] = (
                        sBeta.iterator[Int32(0)] if _col == Int32(0) else f32(0.0)
                    )
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(_r)] if _col == Int32(_r) else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[Int32(_r * BF_PAD + _k)].to(f32)
                            _accum = _accum + _m_rk * _x_t00[_k]
                        _x_t00[_r] = _accum
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(_r, _col)] = _x_t00[_r]
            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t11 = [None] * 8
                    _x_t11[0] = (
                        sBeta.iterator[Int32(8)] if _col == Int32(0) else f32(0.0)
                    )
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = (
                            sBeta.iterator[Int32(8 + _r)]
                            if _col == Int32(_r)
                            else f32(0.0)
                        )
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[
                                Int32((8 + _r) * BF_PAD + 8 + _k)
                            ].to(f32)
                            _accum = _accum + _m_rk * _x_t11[_k]
                        _x_t11[_r] = _accum
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, 8 + _col)] = _x_t11[_r]
            sync_threads()

            if warp_id == Int32(0):
                for _p in cutlass.range_constexpr(2):
                    _i = Int32(_p * 4) + (lane_id >> Int32(3))
                    _j = lane_id & Int32(7)
                    _y_ij = f32(0.0)
                    for _k in cutlass.range_constexpr(8):
                        _m_ik = sTmat.iterator[
                            (Int32(8) + _i) * Int32(BF_PAD) + Int32(_k)
                        ].to(f32)
                        _t_kj = sMat.iterator[_smat_off(_k, _j)]
                        _y_ij = _y_ij + _m_ik * _t_kj
                    sMat.iterator[_smat_off(_i, 8 + _j)] = _y_ij
            sync_threads()

            if warp_id == Int32(1):
                if lane_id < Int32(8):
                    _col = lane_id
                    _x_t10 = [None] * 8
                    _x_t10[0] = sMat.iterator[_smat_off(0, 8 + _col)]
                    for _r in cutlass.range_constexpr(1, 8):
                        _accum = sMat.iterator[_smat_off(_r, 8 + _col)]
                        for _k in cutlass.range_constexpr(_r):
                            _m_rk = sTmat.iterator[
                                Int32((8 + _r) * BF_PAD + 8 + _k)
                            ].to(f32)
                            _accum = _accum + _m_rk * _x_t10[_k]
                        _x_t10[_r] = _accum
                    for _r in cutlass.range_constexpr(8):
                        sMat.iterator[_smat_off(8 + _r, _col)] = _x_t10[_r]
            sync_threads()

        # Stage final Tmat (bf16) to sTmat, zero top-right.
        _flat0_v11 = tidx
        _flat1_v11 = tidx + Int32(THREADS)
        _r0_v11 = _flat0_v11 // Int32(T)
        _c0_v11 = _flat0_v11 % Int32(T)
        _r1_v11 = _flat1_v11 // Int32(T)
        _c1_v11 = _flat1_v11 % Int32(T)
        _v0_v11 = sMat.iterator[_smat_off(_r0_v11, _c0_v11)]
        _v1_v11 = sMat.iterator[_smat_off(_r1_v11, _c1_v11)]
        _out0_v11 = (
            io(0.0) if (_r0_v11 < Int32(8) and _c0_v11 >= Int32(8)) else _v0_v11.to(io)
        )
        _out1_v11 = (
            io(0.0) if (_r1_v11 < Int32(8) and _c1_v11 >= Int32(8)) else _v1_v11.to(io)
        )
        sTmat.iterator[_r0_v11 * BF_PAD + _c0_v11] = _out0_v11
        sTmat.iterator[_r1_v11 * BF_PAD + _c1_v11] = _out1_v11
        sync_threads()

        # ============================================================
        # QT = masked-QKT @ Tmat → sPowk
        # ============================================================
        if warp_id < 2:
            acc.fill(f32(0.0))
            col_off = warp_id * 8
            _qt_a_addr = (
                sNegL.iterator.toint() + _lane_mod16 * Int32(BF_PAD * 2) + _lane_hi
            )
            _qt_b_addr = (
                sTmat.iterator.toint()
                + _ldm_row * Int32(BF_PAD * 2)
                + col_off * Int32(2)
            )
            acc.iterator[0], acc.iterator[1], acc.iterator[2], acc.iterator[3] = (
                _fused_ab_1mma(
                    _qt_a_addr,
                    _qt_b_addr,
                    acc.iterator[0],
                    acc.iterator[1],
                    acc.iterator[2],
                    acc.iterator[3],
                )
            )
            _r0 = lane_id // 4
            _c0 = (lane_id & 3) * 2
            sPowk.iterator[_r0 * BF_PAD + col_off + _c0] = acc.iterator[0].to(io)
            sPowk.iterator[_r0 * BF_PAD + col_off + _c0 + 1] = acc.iterator[1].to(io)
            sPowk.iterator[(_r0 + 8) * BF_PAD + col_off + _c0] = acc.iterator[2].to(io)
            sPowk.iterator[(_r0 + 8) * BF_PAD + col_off + _c0 + 1] = acc.iterator[3].to(
                io
            )
        sync_threads()

        # ============================================================
        # A_full = qhat - QT @ khat → written to sKtil (ktil is dead after
        # the KKT/QKT MMAs above). No eK re-scaling stage is needed — khat
        # was already formed by the gate stage.
        # ============================================================
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sPowk, lane_id)
        _af_b_base = _sK_int + _ldm_row * Int32(K_PADDED * 2) + warp_id * Int32(16)
        _afr = _afull_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _af_b_base)

        _r0 = lane_id // 4
        _c0 = (lane_id & 3) * 2
        BK_GROUPS = K_DIM // 32  # = 4
        for bk_idx in cutlass.range_constexpr(BK_GROUPS):
            k_col = bk_idx * 32 + warp_id * 8
            sKtil.iterator[_r0 * K_PADDED + k_col + _c0] = (
                sQ.iterator[_r0 * K_PADDED + k_col + _c0].to(f32) - _afr[bk_idx * 4]
            ).to(io)
            sKtil.iterator[_r0 * K_PADDED + k_col + _c0 + 1] = (
                sQ.iterator[_r0 * K_PADDED + k_col + _c0 + 1].to(f32)
                - _afr[bk_idx * 4 + 1]
            ).to(io)
            sKtil.iterator[(_r0 + 8) * K_PADDED + k_col + _c0] = (
                sQ.iterator[(_r0 + 8) * K_PADDED + k_col + _c0].to(f32)
                - _afr[bk_idx * 4 + 2]
            ).to(io)
            sKtil.iterator[(_r0 + 8) * K_PADDED + k_col + _c0 + 1] = (
                sQ.iterator[(_r0 + 8) * K_PADDED + k_col + _c0 + 1].to(f32)
                - _afr[bk_idx * 4 + 3]
            ).to(io)

        # CRITICAL barrier — sQ (qhat) aliases sV. All threads must finish
        # reading sQ before any cp.async fills sV (same bytes).
        sync_threads()

        # ============================================================
        # Load V tile via cp.async — overlaps with the state GEMM below.
        # Same t-aware iteration count / zero-propagation proof as GDN:
        # sPowk[0..t-1, 8..15] == 0 at t<=8, so sV rows [8:16) garbage * 0 = 0.
        # ============================================================
        _gV_base = gV.iterator.toint()
        _sV_base_async = sV.iterator.toint()
        _v_base_bf16 = pid_b * sv_b + pid_hv * sv_hv
        _sv_tok = cutlass.Int32(sv_t)
        if const_expr(self._dropin):
            _v_base_bf16 = _bos * s_v_tok + pid_hv * sv_hv
            _sv_tok = s_v_tok
        _v_iters = 1 if self._t_input <= 8 else (T * V_DIM_C // (THREADS * 8))
        for i in cutlass.range_constexpr(_v_iters):
            _v_group = tidx + i * THREADS
            _v_row = _v_group // Int32(V_DIM_C // 8)
            _v_col_bf16_async = (_v_group % Int32(V_DIM_C // 8)) * Int32(8)
            _smem_byte_off_v = _v_row * Int32(V_PADDED * 2) + _v_col_bf16_async * Int32(
                2
            )
            # (native short-T) v holds only n_valid rows; skip OOB rows. The
            # working-set tail is zeroed after the V wait below.
            # (vllm-dropin) rows gated by runtime seq_len, runtime stride.
            if const_expr(self._dropin):
                if _v_row < _seq_len:
                    _cp_async_bf16x8(
                        _gV_base,
                        _v_base_bf16 + _v_row * _sv_tok + _v_col_bf16_async,
                        _sV_base_async + _smem_byte_off_v,
                    )
            elif const_expr(self._n_valid < T):
                if _v_row < Int32(self._n_valid):
                    _cp_async_bf16x8(
                        _gV_base,
                        _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                        _sV_base_async + _smem_byte_off_v,
                    )
            else:
                _cp_async_bf16x8(
                    _gV_base,
                    _v_base_bf16 + _v_row * sv_t + _v_col_bf16_async,
                    _sV_base_async + _smem_byte_off_v,
                )
        _cp_async_commit_group()

        # Wait for the first state half.
        cute.arch.mbarrier_wait(mbar_h_ptr, 0)
        cute.arch.fence_view_async_shared()
        sync_threads()

        # ============================================================
        # State GEMM: WH[16, 128] = A_full[16, 128] @ S0^T (half-K double
        # issue). A operand now reads from sKtil (A_full), stride K_PADDED.
        # ============================================================
        wh_acc_0 = cute.make_fragment_like(tCsC)
        wh_acc_0.fill(f32(0.0))
        wh_acc_1 = cute.make_fragment_like(tCsC)
        wh_acc_1.fill(f32(0.0))
        wh_acc_2 = cute.make_fragment_like(tCsC)
        wh_acc_2.fill(f32(0.0))
        wh_acc_3 = cute.make_fragment_like(tCsC)
        wh_acc_3.fill(f32(0.0))
        # EMIT: accumulators for U_hat = khat @ S0^T (shares the resident
        # state tile with the A_full GEMM; ~free extra MMAs on loaded data).
        u_acc_0 = cute.make_fragment_like(tCsC)
        u_acc_1 = cute.make_fragment_like(tCsC)
        u_acc_2 = cute.make_fragment_like(tCsC)
        u_acc_3 = cute.make_fragment_like(tCsC)
        if const_expr(self._emit):
            u_acc_0.fill(f32(0.0))
            u_acc_1.fill(f32(0.0))
            u_acc_2.fill(f32(0.0))
            u_acc_3.fill(f32(0.0))

        _sA_base_vl = sKtil.iterator.toint()  # A operand (A_full)
        _sH_base_vl = sH.iterator.toint()  # B operand (state, SW128, half-K)
        _rs_a = Int32(K_PADDED * 2)
        _rs_b = Int32(K_HALF * 2)

        _b_lane_row = lane_id % Int32(8)
        _b_col_inner = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
        _vg_base_row = warp_id * Int32(32)

        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32(ka_local * 16 * 2)
            col_byte_off_b = Int32(ka_local * 16 * 2)
            _a_addr = _sA_base_vl + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
            _b0_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(0) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b1_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(8) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b2_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(16) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b3_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(24) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b0 = _sw128_xor(_b0_l)
            _b1 = _sw128_xor(_b1_l)
            _b2 = _sw128_xor(_b2_l)
            _b3 = _sw128_xor(_b3_l)

            _r = _h_gemm_4v(
                _a_addr,
                _b0,
                _b1,
                _b2,
                _b3,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
                wh_acc_0.iterator[2],
                wh_acc_0.iterator[3],
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
                wh_acc_1.iterator[2],
                wh_acc_1.iterator[3],
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
                wh_acc_2.iterator[2],
                wh_acc_2.iterator[3],
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
                wh_acc_3.iterator[2],
                wh_acc_3.iterator[3],
            )
            wh_acc_0.iterator[0] = _r[0]
            wh_acc_0.iterator[1] = _r[1]
            wh_acc_0.iterator[2] = _r[2]
            wh_acc_0.iterator[3] = _r[3]
            wh_acc_1.iterator[0] = _r[4]
            wh_acc_1.iterator[1] = _r[5]
            wh_acc_1.iterator[2] = _r[6]
            wh_acc_1.iterator[3] = _r[7]
            wh_acc_2.iterator[0] = _r[8]
            wh_acc_2.iterator[1] = _r[9]
            wh_acc_2.iterator[2] = _r[10]
            wh_acc_2.iterator[3] = _r[11]
            wh_acc_3.iterator[0] = _r[12]
            wh_acc_3.iterator[1] = _r[13]
            wh_acc_3.iterator[2] = _r[14]
            wh_acc_3.iterator[3] = _r[15]
            if const_expr(self._emit):
                _a_addr_u = _sK_int + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
                _ru = _h_gemm_4v(
                    _a_addr_u,
                    _b0,
                    _b1,
                    _b2,
                    _b3,
                    u_acc_0.iterator[0],
                    u_acc_0.iterator[1],
                    u_acc_0.iterator[2],
                    u_acc_0.iterator[3],
                    u_acc_1.iterator[0],
                    u_acc_1.iterator[1],
                    u_acc_1.iterator[2],
                    u_acc_1.iterator[3],
                    u_acc_2.iterator[0],
                    u_acc_2.iterator[1],
                    u_acc_2.iterator[2],
                    u_acc_2.iterator[3],
                    u_acc_3.iterator[0],
                    u_acc_3.iterator[1],
                    u_acc_3.iterator[2],
                    u_acc_3.iterator[3],
                )
                u_acc_0.iterator[0] = _ru[0]
                u_acc_0.iterator[1] = _ru[1]
                u_acc_0.iterator[2] = _ru[2]
                u_acc_0.iterator[3] = _ru[3]
                u_acc_1.iterator[0] = _ru[4]
                u_acc_1.iterator[1] = _ru[5]
                u_acc_1.iterator[2] = _ru[6]
                u_acc_1.iterator[3] = _ru[7]
                u_acc_2.iterator[0] = _ru[8]
                u_acc_2.iterator[1] = _ru[9]
                u_acc_2.iterator[2] = _ru[10]
                u_acc_2.iterator[3] = _ru[11]
                u_acc_3.iterator[0] = _ru[12]
                u_acc_3.iterator[1] = _ru[13]
                u_acc_3.iterator[2] = _ru[14]
                u_acc_3.iterator[3] = _ru[15]

        # Issue second state half (K = 64..127) after all warps finished
        # reading sH.
        sync_threads()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_h_ptr,
                    V_DIM_C * K_HALF * 2,
                )
            cute.copy(tma_atom_h, tHgH1, tHsH1, tma_bar_ptr=mbar_h_ptr)

        cute.arch.mbarrier_wait(mbar_h_ptr, 1)
        cute.arch.fence_view_async_shared()
        sync_threads()

        for ka_local in cutlass.range_constexpr(4):
            col_byte_off_a = Int32((4 + ka_local) * 16 * 2)
            col_byte_off_b = Int32(ka_local * 16 * 2)
            _a_addr = _sA_base_vl + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
            _b0_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(0) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b1_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(8) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b2_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(16) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b3_l = (
                _sH_base_vl
                + (_vg_base_row + Int32(24) + _b_lane_row) * _rs_b
                + _b_col_inner
                + col_byte_off_b
            )
            _b0 = _sw128_xor(_b0_l)
            _b1 = _sw128_xor(_b1_l)
            _b2 = _sw128_xor(_b2_l)
            _b3 = _sw128_xor(_b3_l)

            _r = _h_gemm_4v(
                _a_addr,
                _b0,
                _b1,
                _b2,
                _b3,
                wh_acc_0.iterator[0],
                wh_acc_0.iterator[1],
                wh_acc_0.iterator[2],
                wh_acc_0.iterator[3],
                wh_acc_1.iterator[0],
                wh_acc_1.iterator[1],
                wh_acc_1.iterator[2],
                wh_acc_1.iterator[3],
                wh_acc_2.iterator[0],
                wh_acc_2.iterator[1],
                wh_acc_2.iterator[2],
                wh_acc_2.iterator[3],
                wh_acc_3.iterator[0],
                wh_acc_3.iterator[1],
                wh_acc_3.iterator[2],
                wh_acc_3.iterator[3],
            )
            wh_acc_0.iterator[0] = _r[0]
            wh_acc_0.iterator[1] = _r[1]
            wh_acc_0.iterator[2] = _r[2]
            wh_acc_0.iterator[3] = _r[3]
            wh_acc_1.iterator[0] = _r[4]
            wh_acc_1.iterator[1] = _r[5]
            wh_acc_1.iterator[2] = _r[6]
            wh_acc_1.iterator[3] = _r[7]
            wh_acc_2.iterator[0] = _r[8]
            wh_acc_2.iterator[1] = _r[9]
            wh_acc_2.iterator[2] = _r[10]
            wh_acc_2.iterator[3] = _r[11]
            wh_acc_3.iterator[0] = _r[12]
            wh_acc_3.iterator[1] = _r[13]
            wh_acc_3.iterator[2] = _r[14]
            wh_acc_3.iterator[3] = _r[15]
            if const_expr(self._emit):
                _a_addr_u = _sK_int + _lane_mod16 * _rs_a + _lane_hi + col_byte_off_a
                _ru = _h_gemm_4v(
                    _a_addr_u,
                    _b0,
                    _b1,
                    _b2,
                    _b3,
                    u_acc_0.iterator[0],
                    u_acc_0.iterator[1],
                    u_acc_0.iterator[2],
                    u_acc_0.iterator[3],
                    u_acc_1.iterator[0],
                    u_acc_1.iterator[1],
                    u_acc_1.iterator[2],
                    u_acc_1.iterator[3],
                    u_acc_2.iterator[0],
                    u_acc_2.iterator[1],
                    u_acc_2.iterator[2],
                    u_acc_2.iterator[3],
                    u_acc_3.iterator[0],
                    u_acc_3.iterator[1],
                    u_acc_3.iterator[2],
                    u_acc_3.iterator[3],
                )
                u_acc_0.iterator[0] = _ru[0]
                u_acc_0.iterator[1] = _ru[1]
                u_acc_0.iterator[2] = _ru[2]
                u_acc_0.iterator[3] = _ru[3]
                u_acc_1.iterator[0] = _ru[4]
                u_acc_1.iterator[1] = _ru[5]
                u_acc_1.iterator[2] = _ru[6]
                u_acc_1.iterator[3] = _ru[7]
                u_acc_2.iterator[0] = _ru[8]
                u_acc_2.iterator[1] = _ru[9]
                u_acc_2.iterator[2] = _ru[10]
                u_acc_2.iterator[3] = _ru[11]
                u_acc_3.iterator[0] = _ru[12]
                u_acc_3.iterator[1] = _ru[13]
                u_acc_3.iterator[2] = _ru[14]
                u_acc_3.iterator[3] = _ru[15]

        # ============================================================
        # OUTPUT: out = WH + QT@V, SMEM-staged coalesced flush.
        # ============================================================
        _qt_a0, _qt_a1, _qt_a2, _qt_a3 = _ldmatrix_x4(sPowk, lane_id)
        _sV_base = sV.iterator.toint()
        _gOut_base = gOut.iterator.toint()
        _out_base = pid_b * so_b + pid_hv * so_hv
        _so_tok = cutlass.Int32(so_t)
        if const_expr(self._dropin):
            _out_base = _bos * s_o_tok + pid_hv * so_hv
            _so_tok = s_o_tok
        _v_off_base = Int32(0)
        _sOutStage_base = sH.iterator.toint()
        # EMIT: D = (V - U_hat) staging tile in the upper part of h_buf at
        # byte 8704 (8704 + 4352 <= 16384). The fp32 W tile later reuses
        # bytes [0, 8704) — the out-stage region, dead after the output
        # flush (ordered by the pre-W-MMA sync) — so W stores and D reads
        # never overlap and need no extra barrier.
        _sUStage_base = _sOutStage_base + Int32(8704)

        _qtv_base = _sV_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
        _cp_async_wait_group_0()
        # (native short-T) zero the sV working-set tail rows that were not
        # loaded. Working set = 8 rows at t<=8 (rows [8:16) garbage is
        # proof-safe: sPowk[*, 8:16) == 0), 16 rows at t=16.
        if const_expr(self._dropin):
            _v_ws_d = 8 if self._t_input <= 8 else T
            for _zr in cutlass.range_constexpr(_v_ws_d):
                if cutlass.Int32(_zr) >= _seq_len:
                    sV.iterator[_zr * V_PADDED + tidx] = io(0.0)
        elif const_expr(self._n_valid < T):
            _v_ws = 8 if self._t_input <= 8 else T
            for _zr in cutlass.range_constexpr(self._n_valid, _v_ws):
                sV.iterator[_zr * V_PADDED + tidx] = io(0.0)
        sync_threads()
        _qtvr = _qtv_4mma(_qt_a0, _qt_a1, _qt_a2, _qt_a3, _qtv_base)

        for h_iter in cutlass.range_constexpr(4):
            h = warp_id * 4 + h_iter
            acc.iterator[0] = _qtvr[h_iter * 4]
            acc.iterator[1] = _qtvr[h_iter * 4 + 1]
            acc.iterator[2] = _qtvr[h_iter * 4 + 2]
            acc.iterator[3] = _qtvr[h_iter * 4 + 3]
            if h_iter == 0:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_0.iterator[j]
            if h_iter == 1:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_1.iterator[j]
            if h_iter == 2:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_2.iterator[j]
            if h_iter == 3:
                for j in cutlass.range_constexpr(4):
                    acc.iterator[j] = acc.iterator[j] + wh_acc_3.iterator[j]
            _out_r0 = lane_id // 4
            _out_c0 = (lane_id & 3) * 2
            _stg_col = h * 8 + _out_c0
            _sts_bf16x2_f32(
                _sOutStage_base + (_out_r0 * V_PADDED + _stg_col) * 2,
                acc.iterator[0],
                acc.iterator[1],
            )
            if const_expr(self._t_input > 8):
                _sts_bf16x2_f32(
                    _sOutStage_base + ((_out_r0 + 8) * V_PADDED + _stg_col) * 2,
                    acc.iterator[2],
                    acc.iterator[3],
                )
            if const_expr(self._emit):
                # Stage D = V - U_hat directly (sV still resident): the W MMA
                # below consumes it. Fragment coords match the output tile.
                _ua = [u_acc_0, u_acc_1, u_acc_2, u_acc_3][h_iter]
                _d0 = (
                    sV.iterator[_out_r0 * V_PADDED + _stg_col].to(f32) - _ua.iterator[0]
                )
                _d1 = (
                    sV.iterator[_out_r0 * V_PADDED + _stg_col + 1].to(f32)
                    - _ua.iterator[1]
                )
                _sts_bf16x2_f32(
                    _sUStage_base + (_out_r0 * V_PADDED + _stg_col) * 2, _d0, _d1
                )
                if const_expr(self._t_input > 8):
                    _d2 = (
                        sV.iterator[(_out_r0 + 8) * V_PADDED + _stg_col].to(f32)
                        - _ua.iterator[2]
                    )
                    _d3 = (
                        sV.iterator[(_out_r0 + 8) * V_PADDED + _stg_col + 1].to(f32)
                        - _ua.iterator[3]
                    )
                    _sts_bf16x2_f32(
                        _sUStage_base + ((_out_r0 + 8) * V_PADDED + _stg_col) * 2,
                        _d2,
                        _d3,
                    )

        sync_threads()
        for _fl_pass in cutlass.range_constexpr(2 if self._t_input > 8 else 1):
            _fl_chunk = _fl_pass * 128 + tidx
            _fl_row = _fl_chunk // 16
            _fl_pos = _fl_chunk & 15
            _fl_lds = _sOutStage_base + _fl_row * Int32(V_PADDED * 2) + _fl_pos * 16
            _fl_off = _out_base + _fl_row * _so_tok + _v_off_base + _fl_pos * 8
            _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
            if const_expr(self._dropin):
                # runtime per-sequence gating; null slots emit zeros for the
                # valid rows (matching the Triton kernel's null-block path)
                # and leave invalid rows untouched.
                if _is_null:
                    _v0 = Int32(0)
                    _v1 = Int32(0)
                    _v2 = Int32(0)
                    _v3 = Int32(0)
                if _fl_row < _seq_len:
                    _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)
            else:
                # Store exactly n_valid rows (the output tensor has n_valid
                # rows).
                _flush_rows = 16 if self._t_input > 8 else 8
                if const_expr(self._n_valid >= _flush_rows):
                    _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)
                else:
                    if _fl_row < Int32(self._n_valid):
                        _st_global_v4_b32(_gOut_base, _fl_off, _v0, _v1, _v2, _v3)

        if const_expr(self._emit):
            # ============================================================
            # EMIT: W = Tmat @ (V - U_hat) — the per-token corrections
            # U_t = beta_t (v_t - u_t) for all T tokens in one T x T MMA.
            # Rows >= t_input of the D tile may hold garbage at t<=8; they
            # multiply Tmat's zeroed top-right block, and rows >= n_valid
            # are never stored.
            # ============================================================
            sync_threads()  # output flush done; offset-0 stage reusable
            _w_a0, _w_a1, _w_a2, _w_a3 = _ldmatrix_x4(sTmat, lane_id)
            _wd_base = (
                _sUStage_base + _ldm_row * Int32(V_PADDED * 2) + warp_id * Int32(64)
            )
            _wr = _qtv_4mma(_w_a0, _w_a1, _w_a2, _w_a3, _wd_base)
            if const_expr(self._dropin):
                # fp32 corrections (vLLM contract): stage W fragments to the
                # dedicated f32 tile (no bf16 rounding), then flush 16 B
                # chunks into the slot-indexed [blocks, H, spec_len, V]
                # cache. Null slots and rows >= seq_len are never written.
                for h_iter in cutlass.range_constexpr(4):
                    h = warp_id * 4 + h_iter
                    _out_r0 = lane_id // 4
                    _out_c0 = (lane_id & 3) * 2
                    _stg_col = h * 8 + _out_c0
                    _sts_f32x2(
                        _sOutStage_base + (_out_r0 * V_PADDED + _stg_col) * 4,
                        _wr[h_iter * 4],
                        _wr[h_iter * 4 + 1],
                    )
                    if const_expr(self._t_input > 8):
                        _sts_f32x2(
                            _sOutStage_base + ((_out_r0 + 8) * V_PADDED + _stg_col) * 4,
                            _wr[h_iter * 4 + 2],
                            _wr[h_iter * 4 + 3],
                        )
                sync_threads()
                _gCorr_base = gCorr.iterator.toint()
                _sW_base = _sOutStage_base
                # slot-indexed base in f32 elements; the 16 B store helper
                # offsets in 2-byte units, so f32 element index * 2.
                _corr_cta = _state_idx * s_c_blk + pid_hv * s_c_head
                # 32 x 16 B chunks per row of 128 f32.
                for _fl_pass in cutlass.range_constexpr(4 if self._t_input > 8 else 2):
                    _fl_chunk = _fl_pass * 128 + tidx
                    _fl_row = _fl_chunk // 32
                    _fl_pos = _fl_chunk % 32
                    _fl_lds = _sW_base + _fl_row * Int32(V_PADDED * 4) + _fl_pos * 16
                    _fl_off = (_corr_cta + _fl_row * s_c_pos + _fl_pos * 4) * 2
                    _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
                    if _state_idx >= cutlass.Int32(self._null_min):
                        if _fl_row < _seq_len:
                            _st_global_v4_b32(_gCorr_base, _fl_off, _v0, _v1, _v2, _v3)
            else:
                for h_iter in cutlass.range_constexpr(4):
                    h = warp_id * 4 + h_iter
                    _out_r0 = lane_id // 4
                    _out_c0 = (lane_id & 3) * 2
                    _stg_col = h * 8 + _out_c0
                    _sts_bf16x2_f32(
                        _sOutStage_base + (_out_r0 * V_PADDED + _stg_col) * 2,
                        _wr[h_iter * 4],
                        _wr[h_iter * 4 + 1],
                    )
                    if const_expr(self._t_input > 8):
                        _sts_bf16x2_f32(
                            _sOutStage_base + ((_out_r0 + 8) * V_PADDED + _stg_col) * 2,
                            _wr[h_iter * 4 + 2],
                            _wr[h_iter * 4 + 3],
                        )
                sync_threads()
                _gCorr_base = gCorr.iterator.toint()
                _corr_base = pid_b * so_b + pid_hv * so_hv
                for _fl_pass in cutlass.range_constexpr(2 if self._t_input > 8 else 1):
                    _fl_chunk = _fl_pass * 128 + tidx
                    _fl_row = _fl_chunk // 16
                    _fl_pos = _fl_chunk & 15
                    _fl_lds = (
                        _sOutStage_base + _fl_row * Int32(V_PADDED * 2) + _fl_pos * 16
                    )
                    _fl_off = _corr_base + _fl_row * so_t + _fl_pos * 8
                    _v0, _v1, _v2, _v3 = _lds_v4_b32(_fl_lds)
                    _flush_rows_w = 16 if self._t_input > 8 else 8
                    if const_expr(self._n_valid >= _flush_rows_w):
                        _st_global_v4_b32(_gCorr_base, _fl_off, _v0, _v1, _v2, _v3)
                    else:
                        if _fl_row < Int32(self._n_valid):
                            _st_global_v4_b32(_gCorr_base, _fl_off, _v0, _v1, _v2, _v3)


# ============================================================================
# Small-problem (small B * HV AND small T) grouped OUTPUT-ONLY kernel.
#
# At small B * HV * T (few sequences AND short windows; at T >= 5 the WY
# kernel wins at every batch size) the WY kernel's per-CTA pipeline latency
# (~4 us: barrier
# chain + TMA + MMA phases) exceeds the serial recurrence cost and its grid
# (B * HV CTAs) cannot fill the GPU. This path is an output-only fork of
# ``recurrent_kda._grouped_kda_kernel``: token data is staged to SMEM once
# (phase A), then a barrier-free serial recurrence over the T tokens runs in
# registers (phase B). Removed relative to the baseline: the per-token bf16
# state-checkpoint writes (T x D x D x HV bytes of store traffic per
# sequence), the cu_seqlens / ssm-slot / accepted-token bookkeeping, and the
# orphan-suffix zeroing. grid = (HV, B, VSPLIT), block = D * KS / VSPLIT.
# ============================================================================

_GLOG2E = 1.4426950408889634


@cute.kernel
def _kda_oo_grouped_kernel(
    q: cute.Tensor,  # [B*T, H, D] bf16 (static inner strides)
    k: cute.Tensor,  # [B*T, H, D] bf16
    v: cute.Tensor,  # [B*T, HV, D] bf16
    g: cute.Tensor,  # [B*T, HV, D] bf16
    beta: cute.Tensor,  # [B*T, HV] bf16
    a_log: cute.Tensor,  # [H] f32
    dt_bias: cute.Tensor,  # [H*D] f32
    src: cute.Tensor,  # [pool, HV, D, D] bf16 (read-only state)
    src_idx: cute.Tensor,  # [B] i32
    out: cute.Tensor,  # [B*T, HV, D] bf16
    corr: cute.Tensor,  # [B*T, HV, D] bf16 (EMIT: per-token correction U)
    kg: cute.Tensor,  # [B*T, HV, 2*D] bf16 (EMIT: k_norm | raw_g)
    scale: cutlass.Float32,
    lower_bound: cutlass.Float32,
    D: cutlass.Constexpr[int],
    T_TOK: cutlass.Constexpr[int],
    KS: cutlass.Constexpr[int],
    RATIO: cutlass.Constexpr[int],
    GATE_MODE: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    BETA_LOGIT: cutlass.Constexpr[int],
    VSPLIT: cutlass.Constexpr[int],
    EMIT: cutlass.Constexpr[int],
):
    """Grouped register-recurrent output-only kernel (small-batch backend):
    phase A stages all tokens to SMEM once, phase B runs the barrier-free
    serial recurrence in registers; no state writeback."""
    tid, _, _ = cute.arch.thread_idx()
    hv, n, vz = cute.arch.block_idx()
    h = hv // RATIO
    KC: cutlass.Constexpr = D // KS  # state elems per thread
    G: cutlass.Constexpr = KC // 8  # 16B granules per thread
    CPB: cutlass.Constexpr = D // VSPLIT  # columns per block
    NT: cutlass.Constexpr = (D * KS) // VSPLIT  # threads per block
    EPT: cutlass.Constexpr = max(D // NT, 1)  # preprocess elems per thread
    SW: cutlass.Constexpr = min(D, NT) // 32  # warp partials in L2 reduce

    v_idx = vz * CPB + tid // KS  # global state column owned
    part = tid % KS  # slice of that column

    smem = utils.SmemAllocator()
    s_eg = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T_TOK * D), 16)
    s_kr = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T_TOK * D), 16)
    s_qr = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T_TOK * D), 16)
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(T_TOK * 16), 16)

    eg_v = cute.make_tensor(
        s_eg.iterator, cute.make_layout((8, G, KS, T_TOK), stride=(1, KS * 8, 8, D))
    )[None, None, part, None]
    kr_v = cute.make_tensor(
        s_kr.iterator, cute.make_layout((8, G, KS, T_TOK), stride=(1, KS * 8, 8, D))
    )[None, None, part, None]
    qr_v = cute.make_tensor(
        s_qr.iterator, cute.make_layout((8, G, KS, T_TOK), stride=(1, KS * 8, 8, D))
    )[None, None, part, None]

    # ---- read-only initial state ------------------------------------------
    s = cute.make_rmem_tensor((8, G), cutlass.Float32)
    slot0 = src_idx[n]
    if slot0 < 0:
        slot0 = cutlass.Int32(0)
    row0 = src[slot0, hv, v_idx, None]
    grow0 = cute.make_tensor(
        row0.iterator, cute.make_layout((8, G, KS), stride=(1, KS * 8, 8))
    )
    gv0 = grow0[None, None, part]
    sb0 = cute.make_rmem_tensor((8, G), cutlass.BFloat16)
    cute.autovec_copy(
        gv0, sb0, l1c_evict_priority=cute.nvgpu.common.CacheEvictionPriority.NO_ALLOCATE
    )
    s.store(sb0.load().to(cutlass.Float32))

    # loop-invariant gate constants
    d = tid % D
    av = cutlass.Float32(1.0)
    if cutlass.const_expr(GATE_MODE != GATE_PRECOMPUTED):
        av = cute.exp2(a_log[h] * _GLOG2E, fastmath=True)
    dtbs = []
    for e in cutlass.range_constexpr(EPT):
        if cutlass.const_expr(GATE_MODE != GATE_PRECOMPUTED and HAS_DT_BIAS != 0):
            dtbs.append(dt_bias[h * D + d + e * NT])
        else:
            dtbs.append(cutlass.Float32(0.0))

    # ---- phase A: preprocess ALL tokens, one barrier -----------------------
    ves = []
    bbs = []
    for t in cutlass.range_constexpr(T_TOK):
        pidx = n * T_TOK + t
        ves.append(v[pidx, hv, v_idx].to(cutlass.Float32))
        bbv = beta[pidx, hv].to(cutlass.Float32)
        if cutlass.const_expr(BETA_LOGIT):
            bbv = 1.0 / (1.0 + cute.exp2(-bbv * _GLOG2E, fastmath=True))
        bbs.append(bbv)

        sqp = cutlass.Float32(0.0)
        skp = cutlass.Float32(0.0)
        for e in cutlass.range_constexpr(EPT):
            de = d + e * NT
            qe = q[pidx, h, de].to(cutlass.Float32)
            ke = k[pidx, h, de].to(cutlass.Float32)
            ge = g[pidx, hv, de].to(cutlass.Float32)
            sqp += qe * qe
            skp += ke * ke

            gate = ge
            if cutlass.const_expr(GATE_MODE != GATE_PRECOMPUTED):
                x = ge
                if cutlass.const_expr(HAS_DT_BIAS):
                    x = x + dtbs[e]
                if cutlass.const_expr(GATE_MODE == GATE_SOFTPLUS):
                    sp = cute.log1p(cute.exp2(x * _GLOG2E, fastmath=True))
                    if x > 20.0:
                        sp = x
                    gate = -av * sp
                else:
                    sig = 1.0 / (1.0 + cute.exp2(-(av * x) * _GLOG2E, fastmath=True))
                    gate = lower_bound * sig

            s_eg[t * D + de] = cute.exp2(gate * _GLOG2E, fastmath=True)
            s_kr[t * D + de] = ke
            s_qr[t * D + de] = qe

        sq = cute.arch.warp_reduction_sum(sqp)
        sk = cute.arch.warp_reduction_sum(skp)
        lane = tid % 32
        wid = tid // 32
        if (lane == 0) & (wid < SW):
            s_red[t * 16 + wid] = sq
            s_red[t * 16 + 8 + wid] = sk
    cute.arch.sync_threads()

    pf = cute.make_rmem_tensor((8,), cutlass.Float32)

    # ---- phase B: barrier-free sequential recurrence, no state writes ------
    for t in cutlass.range_constexpr(T_TOK):
        ve = ves[t]
        bb = bbs[t]

        sqt = cutlass.Float32(0.0)
        skt = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(SW):
            sqt += s_red[t * 16 + w]
            skt += s_red[t * 16 + 8 + w]
        rk = cute.rsqrt(skt + 1e-6)
        rq = cute.rsqrt(sqt + 1e-6) * scale

        # pass 1: decay state + raw prediction
        svec = s[None, 0].load() * eg_v[None, 0, t].load()
        s[None, 0].store(svec)
        pvec = kr_v[None, 0, t].load() * svec
        for gi in cutlass.range_constexpr(1, G, 1):
            svec = s[None, gi].load() * eg_v[None, gi, t].load()
            s[None, gi].store(svec)
            pvec = pvec + kr_v[None, gi, t].load() * svec
        pf.store(pvec)
        pred = ((pf[0] + pf[1]) + (pf[2] + pf[3])) + ((pf[4] + pf[5]) + (pf[6] + pf[7]))
        if cutlass.const_expr(KS > 1):
            for off_i in cutlass.range_constexpr((KS - 1).bit_length()):
                pred += cute.arch.shuffle_sync_bfly(pred, 1 << off_i)
        deltak = rk * bb * (ve - rk * pred)
        if cutlass.const_expr(EMIT):
            # correction U_t = beta * (v - pred_normalized) = deltak / rk;
            # same (token, v-column) ownership as the output store.
            # NOTE: derived from the deltak SSA value rather than re-computing
            # bb * (ve - rk * pred) — the recomputed expression traced to a
            # stale operand (~1e-1 error) while deltak itself is exact.
            if part == 0:
                corr[n * T_TOK + t, hv, v_idx] = (
                    deltak * (cutlass.Float32(1.0) / rk)
                ).to(cutlass.BFloat16)
            # kg cache: normalized k | raw gate. One VSPLIT block writes it
            # (identical values across vz); threads cover D via EPT strides.
            if vz == 0:
                for e in cutlass.range_constexpr(EPT):
                    de = (tid % D) + e * NT
                    kg[n * T_TOK + t, hv, de] = (s_kr[t * D + de] * rk).to(
                        cutlass.BFloat16
                    )
                    kg[n * T_TOK + t, hv, D + de] = g[n * T_TOK + t, hv, de]

        # pass 2: rank-1 update + raw output projection
        svec = s[None, 0].load() + kr_v[None, 0, t].load() * deltak
        s[None, 0].store(svec)
        ovec = qr_v[None, 0, t].load() * svec
        for gi in cutlass.range_constexpr(1, G, 1):
            svec = s[None, gi].load() + kr_v[None, gi, t].load() * deltak
            s[None, gi].store(svec)
            ovec = ovec + qr_v[None, gi, t].load() * svec
        pf.store(ovec)
        o = ((pf[0] + pf[1]) + (pf[2] + pf[3])) + ((pf[4] + pf[5]) + (pf[6] + pf[7]))
        if cutlass.const_expr(KS > 1):
            for off_i in cutlass.range_constexpr((KS - 1).bit_length()):
                o += cute.arch.shuffle_sync_bfly(o, 1 << off_i)
        if part == 0:
            out[n * T_TOK + t, hv, v_idx] = (rq * o).to(cutlass.BFloat16)
        # No state checkpoint write — output-only.


@cute.jit
def _kda_oo_grouped_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    src: cute.Tensor,
    src_idx: cute.Tensor,
    out: cute.Tensor,
    corr: cute.Tensor,
    kg: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    scale: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    T_TOK: cutlass.Constexpr[int],
    KS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    GATE_MODE: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    BETA_LOGIT: cutlass.Constexpr[int],
    VSPLIT: cutlass.Constexpr[int],
    EMIT: cutlass.Constexpr[int],
):
    """Launch helper: rebuild flat constexpr-stride token views (cheap
    per-element address math) and launch the grouped kernel."""
    D: cutlass.Constexpr = 128
    B = q.shape[0]
    qt = B * T_TOK
    # Flat token views with constexpr inner strides — only the token count is
    # dynamic (matches the baseline _grouped_kda_host addressing, which is
    # required for cheap per-element address math in phase A).
    q3 = cute.make_tensor(
        q.iterator, cute.make_layout((qt, H, D), stride=(H * D, D, 1))
    )
    k3 = cute.make_tensor(
        k.iterator, cute.make_layout((qt, H, D), stride=(H * D, D, 1))
    )
    v3 = cute.make_tensor(
        v.iterator, cute.make_layout((qt, HV, D), stride=(HV * D, D, 1))
    )
    g3 = cute.make_tensor(
        g.iterator, cute.make_layout((qt, HV, D), stride=(HV * D, D, 1))
    )
    b2 = cute.make_tensor(beta.iterator, cute.make_layout((qt, HV), stride=(HV, 1)))
    o3 = cute.make_tensor(
        out.iterator, cute.make_layout((qt, HV, D), stride=(HV * D, D, 1))
    )
    # Dummy 1-elem tensors are passed when EMIT=0; the views are never read.
    c3 = cute.make_tensor(
        corr.iterator, cute.make_layout((qt, HV, D), stride=(HV * D, D, 1))
    )
    kg3 = cute.make_tensor(
        kg.iterator, cute.make_layout((qt, HV, 2 * D), stride=(HV * 2 * D, 2 * D, 1))
    )
    pool = src.shape[0]
    sr4 = cute.make_tensor(
        src.iterator,
        cute.make_layout((pool, HV, D, D), stride=(HV * D * D, D * D, D, 1)),
    )
    _kda_oo_grouped_kernel(
        q3,
        k3,
        v3,
        g3,
        b2,
        a_log,
        dt_bias,
        sr4,
        src_idx,
        o3,
        c3,
        kg3,
        scale,
        lower_bound,
        D,
        T_TOK,
        KS,
        HV // H,
        GATE_MODE,
        HAS_DT_BIAS,
        BETA_LOGIT,
        VSPLIT,
        EMIT,
    ).launch(
        grid=(HV, B, VSPLIT),
        block=(D * KS // VSPLIT, 1, 1),
        stream=stream,
        preferred_smem_carveout=25,
    )


# ============================================================================
# Public entry point — output-only KDA MTP/verify decode.
# ============================================================================

_CACHE: dict = {}
# WY-vs-recurrent dispatch threshold on B * HV at T<=2 (a quarter of it at
# T in (2, 4]). Measured crossover on B200; override with
# FLASHINFER_KDA_OO_REC_MAX_BH for tuning.
_REC_DISPATCH_MAX_BH = int(_os.environ.get("FLASHINFER_KDA_OO_REC_MAX_BH", "192"))
# Cached zero placeholders for the unused A_log/dt_bias kernel args in the
# precomputed-gate mode (avoids two per-call allocations).
_DUMMY: dict = {}


def _dummy_f32(device, n):
    """Cached fp32 zeros placeholder for unused kernel args."""
    key = (str(device), n)
    t = _DUMMY.get(key)
    if t is None:
        with torch.inference_mode(False):
            t = torch.zeros(n, dtype=torch.float32, device=device)
        _DUMMY[key] = t
    return t


def _dummy_i32(device):
    """Cached int32 zeros placeholder for unused kernel args."""
    key = (str(device), "i32")
    t = _DUMMY.get(key)
    if t is None:
        with torch.inference_mode(False):
            t = torch.zeros(8, dtype=torch.int32, device=device)
        _DUMMY[key] = t
    return t


def _dummy_bf16(device):
    """Cached bf16 zeros placeholder for unused kernel args."""
    key = (str(device), "bf16")
    t = _DUMMY.get(key)
    if t is None:
        with torch.inference_mode(False):
            t = torch.zeros(8, dtype=torch.bfloat16, device=device)
        _DUMMY[key] = t
    return t


def _compile_options(device: torch.device) -> tuple:
    """Explicit CuTe compile target for SM12x; default target elsewhere."""
    major, minor = torch.cuda.get_device_capability(device)
    return (cute.GPUArch(f"sm_{major}{minor}a"),) if major == 12 else ()


def kda_wy_output_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    beta_is_logit: bool = False,
    output: Optional[torch.Tensor] = None,
    backend: str = "auto",
    emit_corrections: bool = False,
    corrections_out: Optional[torch.Tensor] = None,
    kg_cache_out: Optional[torch.Tensor] = None,
):
    """Output-only (frozen-state) KDA decode over 1..16 tokens per sequence.

    Computes the KDA (Kimi Delta Attention) outputs for ``T`` speculative /
    MTP tokens from a read-only initial state, without writing state back —
    the verify-path counterpart of :func:`flashinfer.recurrent_kda`. Uses a
    chunk-parallel WY formulation on tensor cores instead of the serial
    recurrence.

    Args:
        q: ``[B, T, H, 128]`` bf16 query (1 <= T <= 16).
        k: ``[B, T, H, 128]`` bf16 key.
        v: ``[B, T, HV, 128]`` bf16 value. GQA when ``HV != H``.
        g: ``[B, T, HV, 128]`` bf16 per-channel gate. Log-space when
            ``use_gate_in_kernel=False``; raw otherwise.
        beta: ``[B, T, HV]`` bf16 delta-rule learning rate (pre-sigmoided
            unless ``beta_is_logit=True``).
        initial_state_source: ``[pool, HV, 128, 128]`` bf16 read-only state
            pool, layout ``[V, K]`` per head. Never written.
        initial_state_indices: ``[B]`` int32 slot per sequence (default
            ``arange(B)``).
        A_log: ``[H]`` float32 log-decay (required if ``use_gate_in_kernel``).
        dt_bias: ``[H*128]`` float32 per-channel decay bias (optional).
        scale: query scale; defaults to ``K**-0.5``.
        use_gate_in_kernel: compute the gate in-kernel from raw ``g``.
        lower_bound: if set (negative), the Kimi K3
            ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` gate;
            otherwise the softplus gate. Only with ``use_gate_in_kernel``.
        beta_is_logit: apply sigmoid to beta inside the kernel.
        output: optional preallocated ``[B, T, HV, 128]`` bf16 output.

    Returns:
        ``[B, T, HV, 128]`` bf16 output.

    Notes:
        Q/K L2 normalization is always applied in-kernel (eps 1e-6), matching
        ``recurrent_kda(use_qk_l2norm_in_kernel=True)``.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [B, T, H, K]; got {tuple(q.shape)}")
    B, T_in, H, K_dim = q.shape
    if v.ndim != 4 or v.shape[:2] != (B, T_in):
        raise ValueError(f"v must be [B={B}, T={T_in}, HV, V]; got {tuple(v.shape)}")
    HV = v.shape[2]
    V_dim = v.shape[3]
    device = q.device
    # Cross-tensor agreement: wrong shapes here would flow into hand-computed
    # cp.async offsets (out-of-bounds reads), so fail loudly instead.
    if k.shape != q.shape:
        raise ValueError(f"k must match q {tuple(q.shape)}; got {tuple(k.shape)}")
    if g.shape != (B, T_in, HV, K_dim):
        raise ValueError(
            f"g must be [B={B}, T={T_in}, HV={HV}, K={K_dim}]; got {tuple(g.shape)}"
        )
    if beta.shape != (B, T_in, HV):
        raise ValueError(
            f"beta must be [B={B}, T={T_in}, HV={HV}]; got {tuple(beta.shape)}"
        )
    for name, t_ in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if t_.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bf16; got {t_.dtype}")
        if t_.device != device:
            raise ValueError(f"{name} must be on {device}; got {t_.device}")
    if initial_state_source.dtype != torch.bfloat16:
        raise ValueError("initial_state_source must be bf16 (pool, HV, V, K)")
    if initial_state_source.ndim != 4 or initial_state_source.shape[1:] != (
        HV,
        V_dim,
        K_dim,
    ):
        raise ValueError(
            f"initial_state_source must be [pool, HV={HV}, V={V_dim}, "
            f"K={K_dim}]; got {tuple(initial_state_source.shape)}"
        )
    if initial_state_source.device != device:
        raise ValueError("initial_state_source must be on the inputs' device")
    if initial_state_indices is not None and (
        initial_state_indices.ndim != 1 or initial_state_indices.shape[0] != B
    ):
        raise ValueError(
            f"initial_state_indices must be [B={B}]; "
            f"got {tuple(initial_state_indices.shape)}"
        )
    assert K_dim == K_DIM and V_dim == V_DIM_C, (
        f"this kernel requires K==V=={K_DIM}; got K={K_dim}, V={V_dim}"
    )
    assert 1 <= T_in <= T, f"T must be in [1, {T}]; got {T_in}"
    assert H >= 1 and HV % H == 0, f"HV ({HV}) must be a multiple of H ({H})"

    if scale is None:
        scale = 1.0 / math.sqrt(K_dim)
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)
    else:
        initial_state_indices = initial_state_indices.contiguous()
        if initial_state_indices.dtype != torch.int32:
            initial_state_indices = initial_state_indices.to(torch.int32)

    if use_gate_in_kernel:
        assert A_log is not None, "A_log required with use_gate_in_kernel"
        gate_mode = GATE_LOWER_BOUND if lower_bound is not None else GATE_SOFTPLUS
        if lower_bound is not None:
            assert lower_bound < 0, "lower_bound must be negative"
    else:
        gate_mode = GATE_PRECOMPUTED
    has_dt_bias = use_gate_in_kernel and dt_bias is not None
    lb = float(lower_bound) if lower_bound is not None else 0.0

    if A_log is None:
        A_log = _dummy_f32(device, H)
    else:
        A_log = A_log.to(torch.float32).contiguous()
    if dt_bias is None:
        dt_bias = _dummy_f32(device, H * K_dim)
    else:
        dt_bias = dt_bias.to(torch.float32).contiguous().view(-1)
        assert dt_bias.numel() == H * K_dim

    h0 = initial_state_source.contiguous()

    # Native short-T path (all T): the kernel reads the real [B, T, ...]
    # tensors (batch stride = T * token stride), loads only T rows and zeros
    # its SMEM tails — no host staging copies.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    if output is not None:
        assert output.shape == (B, T_in, HV, V_dim) and output.is_contiguous()
        out_t = output
    else:
        out_t = torch.empty(B, T_in, HV, V_dim, dtype=torch.bfloat16, device=device)

    # emit-corrections (vLLM RecoverSSM verify contract): corrections
    # U_t = beta_t * (v_t - u_t) with shape [B, T, HV, V], and a kg cache
    # [B, T, HV, 2K] holding (L2-normalized k | raw gate) per token.
    if emit_corrections:
        corr_t = (
            corrections_out
            if corrections_out is not None
            else torch.empty(B, T_in, HV, V_dim, dtype=torch.bfloat16, device=device)
        )
        kg_t = (
            kg_cache_out
            if kg_cache_out is not None
            else torch.empty(
                B, T_in, HV, 2 * K_dim, dtype=torch.bfloat16, device=device
            )
        )
        assert corr_t.shape == (B, T_in, HV, V_dim) and corr_t.is_contiguous()
        assert kg_t.shape == (B, T_in, HV, 2 * K_dim) and kg_t.is_contiguous()
    else:
        corr_t = _dummy_bf16(device)
        kg_t = _dummy_bf16(device)

    cc = torch.cuda.get_device_capability(device)
    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    mk = from_dlpack

    def mk_dyn(t):
        """Compact descriptor with a dynamic leading (batch) dim."""
        return mk(t, 16).mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(t.dim())), divisibility=1
        )

    # Batch-size dispatch: at small B * HV * T the WY kernel's fixed pipeline
    # latency (~4.8 us) dominates and its grid (B * HV CTAs) cannot fill the
    # GPU; the grouped register recurrence (sans state writeback) is faster
    # there. Thresholds from a full T x B crossover sweep on B200 (HV=12,
    # precomputed gate): rec wins up to B*HV ~192 at T=1, ~48 at T in [2,4],
    # ~24 at T in [5,7]; the WY kernel wins everywhere at T>=8. With the
    # in-kernel gate the recurrence pays per-element transform cost, shifting
    # every crossover toward WY — thresholds are halved there (measured at
    # T=8 where WY already wins at every batch; interior points estimated).
    _bh = B * HV
    _scale = 1 if gate_mode == GATE_PRECOMPUTED else 2
    if T_in == 1:
        _rec_max = _REC_DISPATCH_MAX_BH // _scale
    elif T_in <= 4:
        _rec_max = _REC_DISPATCH_MAX_BH // (4 * _scale)
    elif T_in <= 7:
        _rec_max = _REC_DISPATCH_MAX_BH // (8 * _scale)
    else:
        _rec_max = 0
    use_rec = backend == "recurrent" or (backend == "auto" and _bh <= _rec_max)
    if use_rec:
        rec_args = [
            mk_dyn(q),
            mk_dyn(k),
            mk_dyn(v),
            mk_dyn(g),
            mk_dyn(beta),
            mk_dyn(h0),
            mk_dyn(initial_state_indices),
            mk_dyn(out_t),
            mk_dyn(corr_t),
            mk_dyn(kg_t),
            mk(A_log, 16),
            mk(dt_bias, 16),
            float(scale),
            lb,
            stream,
        ]
        # Batch-sensitive schedule (measured on B200, T=8 emit sweep — the
        # same "value slice varies with batch" finding as the vLLM verify
        # kernel): small batches want more K-slicing per column, large
        # batches want wide 64-column blocks.
        if B * HV < 96:
            ks, vsplit = (4, 8)
        else:
            ks, vsplit = (1, 2)
        if T_in == 1:
            ks, vsplit = (4, 4)
        _ks_env = _os.environ.get("FLASHINFER_KDA_OO_REC_KS")
        _vs_env = _os.environ.get("FLASHINFER_KDA_OO_REC_VSPLIT")
        if _ks_env:
            ks = int(_ks_env)
        if _vs_env:
            vsplit = int(_vs_env)
        rec_key = (
            "rec",
            str(device),
            cc,
            T_in,
            ks,
            vsplit,
            gate_mode,
            has_dt_bias,
            bool(beta_is_logit),
            bool(emit_corrections),
            HV,
            H,
        )
        if rec_key not in _CACHE:
            options = _compile_options(device)
            compile_fn = cute.compile[options] if options else cute.compile
            _CACHE[rec_key] = compile_fn(
                _kda_oo_grouped_launch,
                *rec_args,
                T_in,
                ks,
                H,
                HV,
                gate_mode,
                1 if has_dt_bias else 0,
                1 if beta_is_logit else 0,
                vsplit,
                1 if emit_corrections else 0,
            )
        _CACHE[rec_key](*rec_args)
        if emit_corrections:
            return out_t, corr_t, kg_t
        return out_t

    _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    _total_ctas = HV * B
    _needed = math.ceil(_total_ctas / _num_sms)
    mbp = max(1, min(_needed + 1, 8))
    _mbp_env = _os.environ.get("FLASHINFER_KDA_OO_MBP")
    if _mbp_env:
        mbp = int(_mbp_env)
    t_disc = 4 if T_in <= 4 else (8 if T_in <= 8 else 16)

    cache_key = (
        str(device),
        cc,
        mbp,
        t_disc,
        T_in,
        gate_mode,
        has_dt_bias,
        bool(beta_is_logit),
        bool(emit_corrections),
        HV,
        H,
        V_dim,
    )
    args = [
        mk_dyn(q),
        mk_dyn(k),
        mk_dyn(v),
        mk_dyn(g),
        mk_dyn(beta),
        mk(A_log, 16),
        mk(dt_bias, 16),
        mk_dyn(h0),
        mk_dyn(initial_state_indices),
        mk_dyn(out_t),
        mk_dyn(corr_t),
        mk_dyn(kg_t),
        mk_dyn(_dummy_i32(device)),
        float(scale),
        lb,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        HV * V_dim * K_dim,  # compact pool block stride (h0 is contiguous)
        HV,
        V_dim,
        H,
        stream,
    ]

    if cache_key not in _CACHE:
        kernel = KdaDecodeWyOutputOnlyKernel(
            min_blocks_per_mp=mbp,
            t_input=t_disc,
            gate_mode=gate_mode,
            has_dt_bias=has_dt_bias,
            beta_is_logit=beta_is_logit,
            n_valid=T_in,
            emit_corrections=emit_corrections,
        )
        options = _compile_options(device)
        _CACHE[cache_key] = (
            cute.compile[options](kernel, *args)
            if options
            else cute.compile(kernel, *args)
        )
    _CACHE[cache_key](*args)
    if emit_corrections:
        return out_t, corr_t, kg_t
    return out_t


__all__ = [
    "kda_wy_output_only",
    "KdaDecodeWyOutputOnlyKernel",
    "GATE_PRECOMPUTED",
    "GATE_LOWER_BOUND",
    "GATE_SOFTPLUS",
    "K_DIM",
    "V_DIM_C",
]


# ============================================================================
# vLLM RecoverSSM drop-in verify — exact signature/contract match for
# vllm/models/kimi_k3/nvidia/ops/recoverssm.py::kda_recoverssm_verify.
# ============================================================================

NULL_BLOCK_ID = 0  # matches vllm.v1.attention.backends.utils.NULL_BLOCK_ID


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor = None,
    *,
    use_gate_in_kernel: bool = True,
    beta_is_logit: bool = True,
    scale: Optional[float] = None,
    null_min: int = 1,
) -> torch.Tensor:
    """Drop-in replacement for vLLM's ``kda_recoverssm_verify`` (Kimi K3).

    Verifies a KDA speculative window without modifying its checkpoint,
    writing the same three artifacts as the vLLM Triton kernel: the
    attention outputs, the fp32 per-token correction cache
    ``[num_blocks, H, spec_query_len, V]`` (``sigmoid(raw_beta) *
    (v - S_decayed @ k_norm)``), and the raw-key/raw-gate cache
    ``[num_blocks, H, spec_query_len, 2K]`` — slot-indexed, with rows past
    each sequence's ``query_len`` and null slots (``state_indices <=
    NULL_BLOCK_ID``) left untouched (null slots write zero outputs).

    Implementation: the WY-parallel tensor-core kernel (per-channel decay
    folded into scaled operand tiles; T x T triangular inverse; TMA-loaded
    state GEMMs). Requires SM90+ (validated on SM100a / B200),
    ``K = V = 128``, ``spec_query_len <= 16``, and a bf16 checkpoint pool.
    The checkpoint rows must be dense, but its block stride may include padding.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if key_dim != K_DIM or value_dim != V_DIM_C:
        raise ValueError("this kernel requires K == V == 128")
    if spec_query_len < 1 or spec_query_len > T:
        raise ValueError("spec_query_len must be in [1, 16]")
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(t.stride()[2:] != (key_dim, 1) for t in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (num_heads, value_dim, key_dim):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    if checkpoint_state.dtype != torch.bfloat16:
        raise ValueError(
            "this drop-in requires a bf16 checkpoint pool (fp32 pools are "
            "not supported yet)"
        )
    if checkpoint_state.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise ValueError("KDA RecoverSSM checkpoint rows must be contiguous")
    if checkpoint_state.stride(0) % 8 != 0:
        raise ValueError("KDA RecoverSSM checkpoint block stride must be 16B-aligned")
    if correction_cache.shape != (num_blocks, num_heads, spec_query_len, value_dim):
        raise ValueError("KDA RecoverSSM correction buffer shape is incompatible")
    if kg_cache.shape != (num_blocks, num_heads, spec_query_len, 2 * key_dim):
        raise ValueError("KDA RecoverSSM key/gate buffer shape is incompatible")
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype or k.dtype != torch.bfloat16:
        raise ValueError("KDA RecoverSSM activations and kg buffer must be bf16")
    if correction_cache.stride(3) != 1 or kg_cache.stride(3) != 1:
        raise ValueError("KDA RecoverSSM cache innermost dims must be contiguous")
    if any(s % 4 != 0 for s in correction_cache.stride()[:3]):
        raise ValueError("KDA RecoverSSM correction strides must be 16B-aligned")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    if out.dtype != v.dtype:
        raise ValueError("KDA RecoverSSM output must match the activation dtype")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, raw_g, raw_beta)):
        raise ValueError("KDA RecoverSSM activations must be bf16")
    if any(
        t.device != q.device
        for t in (
            k,
            v,
            raw_g,
            raw_beta,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM tensors must share one device")
    if total_tokens == 0:
        return out

    device = q.device
    query_start_loc = query_start_loc.to(torch.int32)
    state_indices = state_indices.to(torch.int32).contiguous()
    A_log_f = A_log.to(torch.float32).contiguous()
    dt_bias_f = dt_bias.to(torch.float32).contiguous().view(-1)

    HV = H = num_heads
    t_disc = 4 if spec_query_len <= 4 else (8 if spec_query_len <= 8 else 16)
    _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    mbp = max(1, min(math.ceil(HV * batch / _num_sms) + 1, 8))
    # Gate modes mirror recurrent_kda: precomputed log-space gate, the
    # Kimi K3 lower-bound sigmoid gate, or the Kimi-Linear softplus gate.
    if use_gate_in_kernel:
        gate_mode = GATE_LOWER_BOUND if lower_bound is not None else GATE_SOFTPLUS
    else:
        gate_mode = GATE_PRECOMPUTED
    lb = float(lower_bound) if lower_bound is not None else 0.0
    q_scale = float(scale) if scale is not None else float(key_dim**-0.5)

    cc = torch.cuda.get_device_capability(device)
    cache_key = (
        "dropin",
        str(device),
        cc,
        mbp,
        t_disc,
        gate_mode,
        bool(beta_is_logit),
        int(null_min),
        HV,
        H,
    )
    mk = from_dlpack

    def mk_any(t):
        """Arbitrary-stride view: only the base pointer is consumed (all
        addressing uses the runtime stride args), leading dim dynamic."""
        return mk(t, 16).mark_layout_dynamic(leading_dim=t.dim() - 1)

    def mk_dyn(t):
        """Compact descriptor with a dynamic leading dim."""
        return mk(t, 16).mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(t.dim())), divisibility=1
        )

    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    args = [
        mk_any(q),
        mk_any(k),
        mk_any(v),
        mk_any(raw_g),
        mk_any(raw_beta),
        mk(A_log_f, 16),
        mk(dt_bias_f, 16),
        mk_any(checkpoint_state),
        mk_dyn(state_indices),
        mk_any(out),
        mk_any(correction_cache),
        mk_any(kg_cache),
        mk_dyn(query_start_loc),
        q_scale,
        lb,
        int(q.stride(1)),
        int(k.stride(1)),
        int(v.stride(1)),
        int(raw_g.stride(1)),
        int(raw_beta.stride(1)),
        int(out.stride(1)),
        int(correction_cache.stride(0)),
        int(correction_cache.stride(1)),
        int(correction_cache.stride(2)),
        int(kg_cache.stride(0)),
        int(kg_cache.stride(1)),
        int(kg_cache.stride(2)),
        int(checkpoint_state.stride(0)),  # padded block stride supported
        HV,
        value_dim,
        H,
        stream,
    ]
    if cache_key not in _CACHE:
        kernel = KdaDecodeWyOutputOnlyKernel(
            min_blocks_per_mp=mbp,
            t_input=t_disc,
            gate_mode=gate_mode,
            has_dt_bias=(gate_mode != GATE_PRECOMPUTED),
            beta_is_logit=beta_is_logit,
            n_valid=16,
            emit_corrections=True,
            vllm_dropin=True,
            null_min=null_min,
        )
        options = _compile_options(device)
        _CACHE[cache_key] = (
            cute.compile[options](kernel, *args)
            if options
            else cute.compile(kernel, *args)
        )
    _CACHE[cache_key](*args)
    return out
