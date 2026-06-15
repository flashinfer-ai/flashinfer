from enum import IntEnum

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp, warpgroup, cpasync
from ...utils import get_device_sm_count, _get_cache_buf
from .alpha import AlphaProcessor
from .collective_store_tma import CollectiveStoreTma
from .custom_compile_cache import KeyedCompileMixin, cached_compile
from .collective_inverse_hmma import CollectiveInverse
from .helpers import SM90, round_down, select_tensor_10
from .schedule import WorkDesc


# ─── Named-barrier IDs used by the compute kernel ────────────────────────────
# Must not conflict with each other or with pipeline barrier storage.


class NamedBarrier(IntEnum):
    MATH_WG0 = 4  # OrderedMathBarriers: StreamkBarrier0
    MATH_WG1 = 5  # OrderedMathBarriers: StreamkBarrier1
    KK_SYNC = 13  # sync all 128 WG0 threads before collective_inverse


class WarpGroupRole(IntEnum):
    LDST = 0
    MATH_STATE0 = 1
    MATH_STATE1 = 2
    MATH_AUX = 3


class LoadStoreWarpRole(IntEnum):
    STORE_O = 0
    LOAD_QKV = 1
    LOAD_BETA = 2
    LOAD_ALPHA = 3


class MathWarpGroupRole(IntEnum):
    STATE0 = 0
    STATE1 = 1
    AUX = 2
    KK = 0
    QK = 1


# ─── Warp-specialized delta-rule kernel ───────────────────────────────────────
# Grid: (num_seqs * num_sab_heads, 1, 1)
# Block: 512 threads → WG0=LD/ST, WG1/WG2=state math, WG3=aux math.
#
# needs_alpha / needs_beta / needs_init_state are class attributes set in __init__.
# The JIT compiler specialises per instance, so they are compile-time booleans
# inside the kernel without any parameter-passing trickery.


class _FullyFusedDeltaRuleSm90(KeyedCompileMixin):
    @staticmethod
    def get_register_requirements(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
        num_state_mma_warp_groups: int,
        threads_per_warp_group: int,
    ) -> tuple[int, int, int]:
        reg_alloc_granularity = 8
        load_registers = 40 - 2 * reg_alloc_granularity
        aux_registers = 128 - load_registers
        total_registers = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // threads_per_warp_group
        )
        state_mma_registers = round_down(
            (total_registers - load_registers - aux_registers)
            // num_state_mma_warp_groups,
            reg_alloc_granularity,
        )
        return (
            min(248, load_registers),
            min(248, state_mma_registers),
            min(248, aux_registers),
        )

    @staticmethod
    def can_implement(
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_size: int,
        element_size: int,
    ) -> bool:
        ratio = (
            num_q_heads // num_v_heads
            if num_q_heads > num_v_heads
            else num_v_heads // num_q_heads
        )
        is_gva_enabled = num_v_heads > num_q_heads

        is_gqa_like = (
            (num_k_heads == num_v_heads)
            and (num_q_heads == ratio * num_k_heads)
            and (num_q_heads == ratio * num_v_heads)
        )
        is_gva_like = (
            (num_q_heads == num_k_heads)
            and (num_v_heads == ratio * num_q_heads)
            and (num_v_heads == ratio * num_k_heads)
        )

        alignment = 16 // element_size
        return (
            ((not is_gva_enabled and is_gqa_like) or (is_gva_enabled and is_gva_like))
            and (head_size <= 128)
            and ((head_size % alignment) == 0)
        )

    def __init__(
        self,
        needs_alpha: bool,
        needs_beta: bool,
        needs_init_state: bool,
        needs_checkpointing: bool,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.needs_alpha = needs_alpha
        self.needs_beta = needs_beta
        self.needs_init_state = needs_init_state
        self.needs_checkpointing = needs_checkpointing
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.inverse_dtype = cutlass.Float16
        self.BLK_Q = 64
        self.BLK_KV = 64
        self.D = 128
        self.q_stage = 2
        self.k_stage = 3
        self.v_stage = 2
        self.o_stage = 2
        self.qk_stage = 2
        self.kk_stage = 2
        self.alpha_beta_stage = 5

    def get_next_work(
        self,
        cu_seqlens: cute.Tensor,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ) -> WorkDesc:
        bx, _, _ = cute.arch.block_idx()
        seq_idx = bx // num_sab_heads
        o_head_idx = bx % num_sab_heads
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        tok_start = cutlass.Int32(cu_seqlens[seq_idx])
        tok_end = cutlass.Int32(cu_seqlens[seq_idx + 1])

        return WorkDesc(
            seq_idx=seq_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_start,
            seq_len=tok_end - tok_start,
            tile_idx=cutlass.Int32(0),
        )

    # ─── Ordered 2-WG math barriers ───────────────────────────────────────────
    # Translates flat::OrderedNamedBarriers<UseReservedNB, NB0, NB1>.
    # wg_idx: MathWarpGroupRole.KK or MathWarpGroupRole.QK.

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        """Pre-arrive at WG0's barrier so WG0 is unblocked on the first wait."""
        if wg_idx == MathWarpGroupRole.QK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32):
        """Arrive+wait on this WG's own ordered barrier."""
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)
        else:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256)

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        """Arrive at the other WG's barrier to unblock it."""
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    # ─── kk_store_and_inv ─────────────────────────────────────────────────────

    @cute.jit
    def _kk_store_and_inv(
        self,
        tKKrKK: cute.Tensor,  # fp32 KK accumulator (from 128-thread kk_tiled_mma)
        kk_tiled_mma,
        kk_thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV) 8×8-tiled smem
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element for MMA operand
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) - used when needs_beta
        beta_pipe_idx: cutlass.Int32,
        tKKcMkk: cute.Tensor,  # coordinate mapping for KK fragment
    ):
        """Store tKKrKK → sKK_inv, Inverse, optionally reload+beta."""
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.inverse_dtype
        )
        tiled_store = cute.make_tiled_copy_C(stsm_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(kk_thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_cv = thr_store.retile(tKKrKK)
        tKKrKK_inv = cute.make_fragment_like(tKKrKK_cv, self.inverse_dtype)
        for i in cutlass.range_constexpr(cute.size(tKKrKK_cv)):
            tKKrKK_inv[i] = self.inverse_dtype(tKKrKK_cv[i])
        cute.copy(tiled_store, tKKrKK_inv, tKKsKK)

        cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
        CollectiveInverse().run(sKK_inv, NamedBarrier.KK_SYNC)

        if cutlass.const_expr(self.needs_beta or self.dtype != self.inverse_dtype):
            cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
            ldsm_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.inverse_dtype,
            )
            tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
            thr_load = tiled_load.get_slice(kk_thread_idx)
            tKKrKK_cpy = cute.make_fragment_like(tKKrKK_inv)
            tKKrKK_cvt = cute.make_fragment_like(tKKrKK_inv, self.dtype)
            cute.copy(tiled_load, thr_load.partition_S(sKK_inv), tKKrKK_cpy)
            tKKcMkk_cv = thr_load.retile(tKKcMkk)

            for i in cutlass.range_constexpr(cute.size(tKKrKK_cpy)):
                if cutlass.const_expr(self.needs_beta):
                    _, t = tKKcMkk_cv[i]
                    tKKrKK_cvt[i] = self.dtype(
                        cutlass.Float32(tKKrKK_cpy[i])
                        * cutlass.Float32(sBeta[t, beta_pipe_idx])
                    )
                else:
                    tKKrKK_cvt[i] = self.dtype(tKKrKK_cpy[i])

            tKKsKK2 = thr_store.partition_D(sKK_opd)
            cute.copy(tiled_store, tKKrKK_cvt, tKKsKK2)

    # ─── qk_and_kk_epi ───────────────────────────────────────────────────────

    @cute.jit
    def qk_and_kk_epi(
        self,
        tQKrQK: cute.Tensor,
        tKKrKK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        alpha_stage: cutlass.Int32,
        beta_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_beta):
            beta_row = sBeta[None, beta_stage]
            for i in cutlass.range_constexpr(cute.size(tKKcMkk)):
                s, _ = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cutlass.Float32(beta_row[s])

        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tKKcMkk)):
                # s, t = tKKcMkk[i]
                s, t = tQKcMqk[i]
                alpha = cute.math.exp2(
                    cutlass.Float32(alpha_cumlog[s]) - cutlass.Float32(alpha_cumlog[t]),
                    fastmath=True,
                )
                tQKrQK[i] = tQKrQK[i] * alpha * scale
                tKKrKK[i] = tKKrKK[i] * alpha
        else:
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                tQKrQK[i] = tQKrQK[i] * scale

        for i in cutlass.range_constexpr(cute.size(tKKcMkk)):
            # s, t = tKKcMkk[i]
            s, t = tQKcMqk[i]
            pred = s >= t
            tQKrQK[i] = tQKrQK[i] if pred else cutlass.Float32(0.0)
            tKKrKK[i] = tKKrKK[i] if pred else cutlass.Float32(0.0)
            if cutlass.const_expr(is_final_block):
                pred = s < B or t < B
                tQKrQK[i] = tQKrQK[i] if pred else cutlass.Float32(0.0)
                tKKrKK[i] = tKKrKK[i] if pred else cutlass.Float32(0.0)

    # ─── qk_store ─────────────────────────────────────────────────────────────

    @cute.jit
    def qk_store(
        self,
        tQKrQK: cute.Tensor,
        sQK: cute.Tensor,
        qk_tiled_mma,
        qk_thread_idx: cutlass.Int32,
    ):
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        qk_tiled_copy = cute.make_tiled_copy_C(stsm_atom, qk_tiled_mma)
        qk_thr_copy = qk_tiled_copy.get_slice(qk_thread_idx)
        tQKsQK = qk_thr_copy.partition_D(sQK)
        tQKrQK_cvt = cute.make_fragment_like(tQKrQK, self.dtype)
        tQKrQK_cvt_cv = qk_thr_copy.retile(tQKrQK_cvt)
        for i in cutlass.range_constexpr(cute.size(tQKrQK)):
            tQKrQK_cvt[i] = self.dtype(tQKrQK[i])
        cute.copy(qk_tiled_copy, tQKrQK_cvt_cv, tQKsQK)

    # ─── o1_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def o1_epi(
        self,
        tOrO: cute.Tensor,
        tOcO: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cpscale = sAlpha[None, AlphaProcessor.CUMPROD_SCALE, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                _, tok_q = tOcO[i]
                tOrO[i] = cutlass.Float32(alpha_cpscale[tok_q]) * tOrO[i]
        else:
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                tOrO[i] = scale * tOrO[i]

    # ─── sk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def sk_epi(
        self,
        tSKrSK: cute.Tensor,
        tSKcSK: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cp = sAlpha[None, AlphaProcessor.CUMPROD, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tSKrSK)):
                _, tok_kv = tSKcSK[i]
                tSKrSK[i] = tSKrSK[i] * cutlass.Float32(alpha_cp[tok_kv])

    # ─── sk_load_v ────────────────────────────────────────────────────────────

    @cute.jit
    def sk_load_v(
        self,
        tSKrSK: cute.Tensor,
        sV_DS: cute.Tensor,
        sk_tiled_copy_C,
        sk_thr_copy_C,
        v_stage: cutlass.Int32,
    ) -> cute.Tensor:
        tSKrV = cute.make_fragment_like(tSKrSK, self.dtype)
        tSKrV_cv = sk_thr_copy_C.retile(tSKrV)
        tSKsV = sk_thr_copy_C.partition_S(sV_DS)
        cute.copy(sk_tiled_copy_C, tSKsV[None, None, None, v_stage], tSKrV_cv)
        return tSKrV

    # ─── kv_decay_v ───────────────────────────────────────────────────────────

    @cute.jit
    def kv_decay_v(
        self,
        tKVrV: cute.Tensor,
        tKVcV: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            block_log = cutlass.Float32(alpha_cumlog[B - cutlass.Int32(1)])
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                coeff = cute.math.exp2(
                    block_log - cutlass.Float32(alpha_cumlog[tok]), fastmath=True
                )
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        coeff = cutlass.Float32(0.0)
                tKVrV[i] = self.dtype(cutlass.Float32(tKVrV[i]) * coeff)
        else:
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        tKVrV[i] = self.dtype(0.0)

    # ─── o_store ──────────────────────────────────────────────────────────────

    @cute.jit
    def o_store(
        self,
        tOrO: cute.Tensor,
        tOsO: cute.Tensor,
        o_tiled_copy_r2s,
        o_thr_copy_r2s,
    ):
        tOrO_f16 = cute.make_fragment_like(tOrO, self.dtype)
        for i in cutlass.range_constexpr(cute.size(tOrO)):
            tOrO_f16[i] = self.dtype(tOrO[i])
        tOrO_cv = o_thr_copy_r2s.retile(tOrO_f16)
        cute.arch.fence_view_async_shared()
        cute.copy(o_tiled_copy_r2s, tOrO_cv, tOsO)
        cute.arch.fence_view_async_shared()

    # ─── TMA load helpers ────────────────────────────────────────────────────

    @cute.jit
    def load_qkv_tma(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        blk: cutlass.Int32,
        tok_start: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
    ):
        blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)

        sK = sK_DS[None, None, k_producer_state.index]
        mK = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_k[None, None, k_head_idx],
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        k_pipeline.producer_acquire(k_producer_state)
        cute.copy(
            tma_atom_k,
            tKgK,
            tKsK,
            tma_bar_ptr=k_pipeline.producer_get_barrier(k_producer_state),
        )
        k_pipeline.producer_commit(k_producer_state)
        k_producer_state.advance()

        sQ = sQ_SD[None, None, q_producer_state.index]
        mQ = cute.domain_offset(
            (blk_tok, cutlass.Int32(0)),
            tma_tensor_q[None, None, q_head_idx],
        )
        gQ = cute.zipped_divide(mQ, (self.BLK_Q, self.D))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        q_pipeline.producer_acquire(q_producer_state)
        cute.copy(
            tma_atom_q,
            tQgQ,
            tQsQ,
            tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state),
        )
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        sV = sV_DS[None, None, v_producer_state.index]
        mV = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_v[None, None, v_head_idx],
        )
        gV = cute.zipped_divide(mV, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        v_pipeline.producer_acquire(v_producer_state)
        cute.copy(
            tma_atom_v,
            tVgV,
            tVsV,
            tma_bar_ptr=v_pipeline.producer_get_barrier(v_producer_state),
        )
        v_pipeline.producer_commit(v_producer_state)
        v_producer_state.advance()
        return q_producer_state, k_producer_state, v_producer_state

    # ─── load_alpha ───────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_alpha (scalar load).
    # Caller must sync before calling AlphaProcessor on the loaded data.

    @cute.jit
    def load_alpha(
        self,
        sAlpha: cute.Tensor,
        g_alpha: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        alpha_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sAlpha_k = sAlpha[None, None, alpha_stage]
        num_iters = self.BLK_Q // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = g_alpha[
                    tok * num_sab_heads + sab_head_idx
                ]
            else:
                sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = cutlass.Float32(1.0)

    # ─── load_beta ────────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_beta.

    @cute.jit
    def load_beta(
        self,
        sBeta: cute.Tensor,
        g_beta: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sBeta_k = sBeta[None, beta_stage]
        num_iters = self.BLK_KV // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                sBeta_k[row] = g_beta[tok * num_sab_heads + sab_head_idx]
            else:
                sBeta_k[row] = cutlass.Float32(0.0)

    # ─── kv_load / kv_store ───────────────────────────────────────────────────

    @cute.jit
    def kv_load(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_tiled_mma,
        thread_idx: cutlass.Int32,
    ):
        copy_atom_kv = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gKV.element_type
        )
        tiled_copy_kv = cute.make_tiled_copy_C(copy_atom_kv, kv_tiled_mma)
        thr_copy_kv = tiled_copy_kv.get_slice(thread_idx)
        tKVgKV = thr_copy_kv.partition_S(select_tensor_10(gKV))
        cute.copy(tiled_copy_kv, tKVgKV, tKVrKV)

    @cute.jit
    def kv_store(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_tiled_mma,
        thread_idx: cutlass.Int32,
    ):
        copy_atom_kv = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gKV.element_type
        )
        tiled_copy_kv = cute.make_tiled_copy_C(copy_atom_kv, kv_tiled_mma)
        thr_copy_kv = tiled_copy_kv.get_slice(thread_idx)
        tKVgKV = thr_copy_kv.partition_D(select_tensor_10(gKV))
        cute.copy(tiled_copy_kv, tKVrKV, tKVgKV)

    @cute.jit
    def maybe_store_checkpoint(
        self,
        tKVrKV: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        kv_tiled_mma,
        thread_idx: cutlass.Int32,
        seq_idx: cutlass.Int32,
        o_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_checkpointing):
            if (
                block_end <= seq_len
                and block_end % checkpoint_every_n_tokens == cutlass.Int32(0)
            ):
                checkpoint_idx = (
                    cutlass.Int32(checkpoint_cu_starts[seq_idx])
                    + block_end // checkpoint_every_n_tokens
                    - cutlass.Int32(1)
                )
                checkpoint_layout = cute.make_ordered_layout(
                    (self.D, self.D, num_sab_heads, total_checkpoints),
                    order=(0, 1, 2, 3),
                )
                mCheckpoint = cute.make_tensor(
                    g_state_checkpoints.iterator, checkpoint_layout
                )
                gCheckpointKV = mCheckpoint[None, None, o_head_idx, checkpoint_idx]
                self.kv_store(tKVrKV, gCheckpointKV, kv_tiled_mma, thread_idx)

    # ─── compute_loop_body ───────────────────────────────────────────────────
    # Translates the C++ compute_loop_body lambda captured inside compute().
    # Called by Math WGs (tidx >= 128) for one block iteration.

    @cute.jit
    def compute_loop_body(
        self,
        # Smem tensors (staged; caller indexes the active stage)
        sQ_SD: cute.Tensor,  # (BlkQ, D, StagesQ)  – row-major atom, swizzled
        sK_SD: cute.Tensor,  # (BlkKV, D, StagesK) – same atom
        sK_DS: cute.Tensor,  # (D, BlkKV, StagesK) – K transposed
        sV_DS: cute.Tensor,  # (D, BlkKV, StagesV) – V transposed
        sQK: cute.Tensor,  # (BlkQ, BlkKV, StagesQK)
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV, StagesKK)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element
        sO: cute.Tensor,  # O output smem (staged)
        sAlpha: cute.Tensor,  # (BlkQ, AlphaProcessor.NUM_CHANNELS, StagesAlpha) or zero-shaped
        kv_tiled_mma,
        # Mainloop pipelines and active read states
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        o_pipeline,
        o_producer_state,
        qk_pipeline,
        qk_consumer_state,
        kk_pipeline,
        kk_consumer_state,
        alpha_pipeline,
        alpha_consumer_state,
        # Compile-time flags
        is_first_block: bool,
        is_final_block: bool,
        # Valid token count for masking on final block
        B: cutlass.Int32,
        # Running KV state (D×D fp32, in registers across all blocks)
        tKVrKV: cute.Tensor,
        # Scale factor
        scale: cutlass.Float32,
        # WG role: MathWarpGroupRole.STATE0 or MathWarpGroupRole.STATE1.
        wg_idx: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx - cutlass.Int32(128)  # relative to compute threads
        # ── TiledMMAs ─────────────────────────────────────────────────────────
        blk_q = cute.size(sQ_SD, mode=[0])
        blk_kv = cute.size(sK_SD, mode=[0])
        d = cute.size(sQ_SD, mode=[1])
        mma_atom_o1 = cute.make_mma_atom(
            warpgroup.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                (64, blk_q, 16),
                warpgroup.OperandSource.RMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        mma_atom_o2 = cute.make_mma_atom(
            warpgroup.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                (64, blk_q, 16),
                warpgroup.OperandSource.RMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        mma_atom_sk = cute.make_mma_atom(
            warpgroup.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                (64, blk_kv, 16),
                warpgroup.OperandSource.RMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        mma_atom_newv = cute.make_mma_atom(
            warpgroup.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                (64, blk_kv, 16),
                warpgroup.OperandSource.RMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )

        # O1/O2/SK/NewV: two state warpgroups cooperate as in the C++ Hopper GMMA path.
        o1_tiled_mma = cute.make_tiled_mma(mma_atom_o1, cute.make_layout((2, 1, 1)))
        o2_tiled_mma = cute.make_tiled_mma(mma_atom_o2, cute.make_layout((2, 1, 1)))
        sk_tiled_mma = cute.make_tiled_mma(mma_atom_sk, cute.make_layout((2, 1, 1)))
        newv_tiled_mma = cute.make_tiled_mma(mma_atom_newv, cute.make_layout((2, 1, 1)))

        # ── Thread slices ─────────────────────────────────────────────────────
        sk_thr_mma = sk_tiled_mma.get_slice(thread_idx)
        newv_thr_mma = newv_tiled_mma.get_slice(thread_idx)
        o1_thr_mma = o1_tiled_mma.get_slice(thread_idx)
        o2_thr_mma = o2_tiled_mma.get_slice(thread_idx)
        kv_thr_mma = kv_tiled_mma.get_slice(thread_idx)

        # ── Copy atoms ────────────────────────────────────────────────────────
        ldsm_t4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )

        # ── SK copies ─────────────────────────────────────────────────────────
        # SK C: V loaded from sV_DS (col-major D×BlkKV) with LDSM_T.
        sk_tiled_copy_C = cute.make_tiled_copy_C(ldsm_t4, sk_tiled_mma)
        sk_thr_copy_C = sk_tiled_copy_C.get_slice(thread_idx)
        tSKsK = sk_thr_mma.partition_B(sK_SD)
        tSKrK = sk_thr_mma.make_fragment_B(tSKsK)

        # ── NewV copies ───────────────────────────────────────────────────────
        tNewVsB = newv_thr_mma.partition_B(sKK_opd)
        tNewVrB = newv_thr_mma.make_fragment_B(tNewVsB)

        # ── KV copies ─────────────────────────────────────────────────────────
        tKVsK = kv_thr_mma.partition_B(sK_DS)
        tKVrK = kv_thr_mma.make_fragment_B(tKVsK)

        # ── O1/O2 copies ──────────────────────────────────────────────────────
        tOsQ = o1_thr_mma.partition_B(sQ_SD)
        tOrQ = o1_thr_mma.make_fragment_B(tOsQ)
        tOsQK = o2_thr_mma.partition_B(sQK)
        tOrQK = o2_thr_mma.make_fragment_B(tOsQK)

        # ── O store (R→S STSM) ────────────────────────────────────────────────
        o_stsm = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        o_tiled_copy_r2s = cute.make_tiled_copy_C(o_stsm, o1_tiled_mma)
        o_thr_copy_r2s = o_tiled_copy_r2s.get_slice(thread_idx)
        tOsO = o_thr_copy_r2s.partition_D(sO)

        # ── Coordinate tensors for masking / alpha/beta indexing ──────────────
        cO = cute.make_identity_tensor((d, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((d, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((d, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)

        # ── O1: KV_state @ Q (both state WGs, skip on first block) ───────────
        q_pipeline.consumer_wait(q_consumer_state)
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_wait(alpha_consumer_state)
            cute.arch.fence_view_async_shared()
        tOrO = o1_thr_mma.make_fragment_C(o1_thr_mma.partition_shape_C((d, blk_q)))
        if cutlass.const_expr(not is_first_block):
            tOrKV = SM90.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
            SM90.warpgroup_fence_operand(tOrKV)
            SM90.warpgroup_fence_operand(tOrO)
            cute.nvgpu.warpgroup.fence()
            self._math_order_wait(wg_idx)
            SM90.wgmma_gemm_zero_acc(
                o1_tiled_mma,
                tOrO,
                tOrKV,
                tOrQ[None, None, None, q_consumer_state.index],
            )
            cute.nvgpu.warpgroup.commit_group()
            self._math_order_notify(wg_idx)
            cute.nvgpu.warpgroup.wait_group(0)
            self.o1_epi(tOrO, tOcO, sAlpha, alpha_consumer_state.index, scale)
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        # ── SK: KV_state @ K^T (result negated below via V - SK) ─────────────
        k_pipeline.consumer_wait(k_consumer_state)
        tSKrSK = sk_thr_mma.make_fragment_C(sk_thr_mma.partition_shape_C((d, blk_kv)))
        if cutlass.const_expr(not is_first_block):
            tSKrS = SM90.make_acc_into_op(tKVrKV, sk_tiled_mma, self.dtype)
            SM90.warpgroup_fence_operand(tSKrSK)
            SM90.warpgroup_fence_operand(tSKrS)
            cute.nvgpu.warpgroup.fence()
            self._math_order_wait(wg_idx)
            SM90.wgmma_gemm_zero_acc(
                sk_tiled_mma,
                tSKrSK,
                tSKrS,
                tSKrK[None, None, None, k_consumer_state.index],
            )
            cute.nvgpu.warpgroup.commit_group()
            self._math_order_notify(wg_idx)
            cute.nvgpu.warpgroup.wait_group(0)

        # ── Load V from smem ──────────────────────────────────────────────────
        v_pipeline.consumer_wait(v_consumer_state)
        tSKrV = self.sk_load_v(
            tSKrSK,
            sV_DS,
            sk_tiled_copy_C,
            sk_thr_copy_C,
            v_consumer_state.index,
        )

        # sk_epi + V - SK  (SK=0 on first block, so V - SK = V)
        if cutlass.const_expr(not is_first_block):
            self.sk_epi(tSKrSK, tSKcSK, sAlpha, alpha_consumer_state.index)
            for i in cutlass.range_constexpr(cute.size(tSKrV)):
                tSKrV[i] = tSKrV[i] - self.dtype(tSKrSK[i])

        # ── NewV = (V - SK) @ T^T  (ordered: WG0 first) ──────────────────────
        tNewVrA = SM90.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = newv_thr_mma.make_fragment_C(
            newv_thr_mma.partition_shape_C((d, blk_kv))
        )
        kk_pipeline.consumer_wait(kk_consumer_state)
        SM90.warpgroup_fence_operand(tNewVrA)
        SM90.warpgroup_fence_operand(tNewVrC)
        cute.nvgpu.warpgroup.fence()
        self._math_order_wait(wg_idx)
        SM90.wgmma_gemm_zero_acc(
            newv_tiled_mma,
            tNewVrC,
            tNewVrA,
            tNewVrB[None, None, None, kk_consumer_state.index],
        )
        cute.nvgpu.warpgroup.commit_group()
        self._math_order_notify(wg_idx)
        cute.nvgpu.warpgroup.wait_group(0)
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()
        kk_pipeline.consumer_release(kk_consumer_state)
        kk_consumer_state.advance()

        # ── O2 = O1 + NewV @ QK  (ordered: WG0 first) ────────────────────────
        tOrV_or_tKVrV = SM90.make_acc_into_op(tNewVrC, kv_tiled_mma, self.dtype)
        qk_pipeline.consumer_wait(qk_consumer_state)
        SM90.warpgroup_fence_operand(tOrV_or_tKVrV)
        SM90.warpgroup_fence_operand(tOrO)
        cute.nvgpu.warpgroup.fence()
        self._math_order_wait(wg_idx)
        SM90.wgmma_gemm(
            o2_tiled_mma,
            tOrO,
            tOrV_or_tKVrV,
            tOrQK[None, None, None, qk_consumer_state.index],
            not is_first_block,
        )
        cute.nvgpu.warpgroup.commit_group()
        self._math_order_notify(wg_idx)
        cute.nvgpu.warpgroup.wait_group(0)
        qk_pipeline.consumer_release(qk_consumer_state)
        qk_consumer_state.advance()

        # ── O store to smem ───────────────────────────────────────────────────
        o_pipeline.producer_acquire(o_producer_state)
        self.o_store(
            tOrO,
            tOsO[None, None, None, o_producer_state.index],
            o_tiled_copy_r2s,
            o_thr_copy_r2s,
        )
        o_pipeline.producer_commit(o_producer_state)
        o_producer_state.advance()

        # ── KV state update ───────────────────────────────────────────────────
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[
                    B - cutlass.Int32(1),
                    AlphaProcessor.CUMPROD,
                    alpha_consumer_state.index,
                ]
            )

        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        self.kv_decay_v(
            tOrV_or_tKVrV, tKVcV, sAlpha, alpha_consumer_state.index, is_final_block, B
        )

        # KV += NewV @ K
        SM90.warpgroup_fence_operand(tOrV_or_tKVrV)
        SM90.warpgroup_fence_operand(tKVrKV)
        cute.nvgpu.warpgroup.fence()
        self._math_order_wait(wg_idx)
        SM90.wgmma_gemm(
            kv_tiled_mma,
            tKVrKV,
            tOrV_or_tKVrV,
            tKVrK[None, None, None, k_consumer_state.index],
            True,
        )
        cute.nvgpu.warpgroup.commit_group()
        self._math_order_notify(wg_idx)
        cute.nvgpu.warpgroup.wait_group(0)
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_release(alpha_consumer_state)
            alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            qk_consumer_state,
            kk_consumer_state,
            alpha_consumer_state,
        )

    # ─── Warp role entry points ──────────────────────────────────────────────
    # The current DSL bridge still uses CTA-wide sync epochs, but each role owns
    # its own loop, matching the C++ warp-specialized dispatch shape.

    @cute.jit
    def run_load_qkv_role(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
    ):
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
            ) = self.load_qkv_tma(
                sQ_SD,
                sK_DS,
                sV_DS,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                tma_atom_v,
                tma_tensor_v,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                blk,
                tok_start,
                q_head_idx,
                k_head_idx,
                v_head_idx,
            )

    @cute.jit
    def run_load_alpha_role(
        self,
        sAlpha: cute.Tensor,
        g_alpha: cute.Tensor,
        alpha_pipeline,
        scale: cutlass.Float32,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            blk_tok = tok_start + blk * cutlass.Int32(self.BLK_Q)
            if cutlass.const_expr(self.needs_alpha):
                alpha_pipeline.producer_acquire(alpha_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_alpha(
                    sAlpha,
                    g_alpha,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    alpha_producer_state.index,
                )
                AlphaProcessor().run(
                    sAlpha[None, None, alpha_producer_state.index], scale
                )
                cute.arch.fence_view_async_shared()
                alpha_pipeline.producer_commit(alpha_producer_state)
                alpha_producer_state.advance()

    @cute.jit
    def run_load_beta_role(
        self,
        sBeta: cute.Tensor,
        g_beta: cute.Tensor,
        beta_pipeline,
        num_blocks: cutlass.Int32,
        tok_start: cutlass.Int32,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
            if cutlass.const_expr(self.needs_beta):
                beta_pipeline.producer_acquire(beta_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_beta(
                    sBeta,
                    g_beta,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    beta_producer_state.index,
                )
                cute.arch.fence_view_async_shared()
                beta_pipeline.producer_commit(beta_producer_state)
                beta_producer_state.advance()

    @cute.jit
    def run_state_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        o_pipeline,
        qk_pipeline,
        kk_pipeline,
        alpha_pipeline,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        wg_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        self._math_order_init(wg_idx)
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        qk_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.qk_stage
        )
        kk_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kk_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (64, self.D, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.MN,
                )
            ),
            cute.make_layout((2, 1, 1)),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = kv_thr_mma.make_fragment_C(
            kv_thr_mma.partition_shape_C((self.D, self.D))
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_seqs), order=(0, 1, 2, 3)
        )
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        mState = cute.make_tensor(g_state.iterator, state_layout)
        gStateKV = mState[None, None, o_head_idx, work_desc.seq_idx]
        if cutlass.const_expr(self.needs_init_state):
            mInitState = cute.make_tensor(g_init_state.iterator, state_layout)
            gInitKV = mInitState[None, None, o_head_idx, work_desc.seq_idx]
            self.kv_load(tKVrKV, gInitKV, kv_tiled_mma, math_tidx)

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        if cutlass.const_expr(self.needs_init_state):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                qk_consumer_state,
                kk_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                qk_pipeline,
                qk_consumer_state,
                kk_pipeline,
                kk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                True,
                first_B,
                tKVrKV,
                scale,
                wg_idx,
            )
        else:
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                qk_consumer_state,
                kk_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                qk_pipeline,
                qk_consumer_state,
                kk_pipeline,
                kk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                True,
                True,
                first_B,
                tKVrKV,
                scale,
                wg_idx,
            )
        self.maybe_store_checkpoint(
            tKVrKV,
            g_state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
            kv_tiled_mma,
            math_tidx,
            work_desc.seq_idx,
            o_head_idx,
            num_sab_heads,
            total_checkpoints,
            cutlass.Int32(self.BLK_KV),
            work_desc.seq_len,
        )

        for blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                qk_consumer_state,
                kk_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                qk_pipeline,
                qk_consumer_state,
                kk_pipeline,
                kk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                False,
                cutlass.Int32(self.BLK_KV),
                tKVrKV,
                scale,
                wg_idx,
            )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_tiled_mma,
                math_tidx,
                work_desc.seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
            )

        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                qk_consumer_state,
                kk_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                qk_pipeline,
                qk_consumer_state,
                kk_pipeline,
                kk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_tiled_mma,
                math_tidx,
                work_desc.seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
            )
        self.kv_store(tKVrKV, gStateKV, kv_tiled_mma, math_tidx)

    @cute.jit
    def run_aux_loop_body(
        self,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        qk_pipeline,
        qk_producer_state,
        kk_pipeline,
        kk_producer_state,
        alpha_pipeline,
        alpha_consumer_state,
        beta_pipeline,
        beta_consumer_state,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        blk: cutlass.Int32,
        is_final_block: bool,
        qk_tiled_mma,
        kk_tiled_mma,
        tQKrQ: cute.Tensor,
        tQKrK: cute.Tensor,
        tKKrA: cute.Tensor,
        tKKrB: cute.Tensor,
        tQKcMqk: cute.Tensor,
        tKKcMkk: cute.Tensor,
        aux_tidx: cutlass.Int32,
    ):
        B = cutlass.Int32(self.BLK_KV)
        if cutlass.const_expr(is_final_block):
            B = work_desc.seq_len - blk * cutlass.Int32(self.BLK_KV)

        k_pipeline.consumer_wait(k_consumer_state)

        tKKrKK = kk_tiled_mma.get_slice(aux_tidx).make_fragment_C(
            kk_tiled_mma.get_slice(aux_tidx).partition_shape_C(
                (self.BLK_KV, self.BLK_KV)
            )
        )
        cute.nvgpu.warpgroup.fence()
        SM90.wgmma_gemm_zero_acc(
            kk_tiled_mma,
            tKKrKK,
            tKKrA[None, None, None, k_consumer_state.index],
            tKKrB[None, None, None, k_consumer_state.index],
        )
        cute.nvgpu.warpgroup.commit_group()

        q_pipeline.consumer_wait(q_consumer_state)
        tQKrQK = qk_tiled_mma.get_slice(aux_tidx).make_fragment_C(
            qk_tiled_mma.get_slice(aux_tidx).partition_shape_C(
                (self.BLK_Q, self.BLK_KV)
            )
        )
        cute.nvgpu.warpgroup.fence()
        SM90.wgmma_gemm_zero_acc(
            qk_tiled_mma,
            tQKrQK,
            tQKrQ[None, None, None, q_consumer_state.index],
            tQKrK[None, None, None, k_consumer_state.index],
        )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_wait(alpha_consumer_state)
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_wait(beta_consumer_state)
        cute.arch.fence_view_async_shared()

        self.qk_and_kk_epi(
            tQKrQK,
            tKKrKK,
            tQKcMqk,
            tKKcMkk,
            sAlpha,
            sBeta,
            alpha_consumer_state.index,
            beta_consumer_state.index,
            is_final_block,
            B,
            scale,
        )

        kk_pipeline.producer_acquire(kk_producer_state)
        self._kk_store_and_inv(
            tKKrKK,
            kk_tiled_mma,
            aux_tidx,
            sKK_inv[None, None, kk_producer_state.index],
            sKK_opd[None, None, kk_producer_state.index],
            sBeta,
            beta_consumer_state.index,
            tKKcMkk,
        )
        cute.arch.fence_view_async_shared()
        kk_pipeline.producer_commit(kk_producer_state)
        kk_producer_state.advance()

        qk_pipeline.producer_acquire(qk_producer_state)
        self.qk_store(
            tQKrQK,
            sQK[None, None, qk_producer_state.index],
            qk_tiled_mma,
            aux_tidx,
        )
        cute.arch.fence_view_async_shared()
        qk_pipeline.producer_commit(qk_producer_state)
        qk_producer_state.advance()

        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_release(alpha_consumer_state)
            alpha_consumer_state.advance()
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_release(beta_consumer_state)
            beta_consumer_state.advance()

        return (
            q_consumer_state,
            k_consumer_state,
            qk_producer_state,
            kk_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        )

    @cute.jit
    def run_aux_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        qk_pipeline,
        kk_pipeline,
        alpha_pipeline,
        beta_pipeline,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        math_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
    ):
        aux_tidx = math_tidx % cutlass.Int32(128)

        qk_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (self.BLK_Q, self.BLK_KV, 16),
                    warpgroup.OperandSource.SMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                )
            ),
            cute.make_layout((1, 1, 1)),
        )
        kk_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (self.BLK_KV, self.BLK_KV, 16),
                    warpgroup.OperandSource.SMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                )
            ),
            cute.make_layout((1, 1, 1)),
        )

        qk_thr_mma = qk_tiled_mma.get_slice(aux_tidx)
        kk_thr_mma = kk_tiled_mma.get_slice(aux_tidx)

        tQKsQ = qk_thr_mma.partition_A(sQ_SD)
        tQKsK = qk_thr_mma.partition_B(sK_SD)
        tQKrQ = qk_thr_mma.make_fragment_A(tQKsQ)
        tQKrK = qk_thr_mma.make_fragment_B(tQKsK)

        tKKsA = kk_thr_mma.partition_A(sK_SD)
        tKKsB = kk_thr_mma.partition_B(sK_SD)
        tKKrA = kk_thr_mma.make_fragment_A(tKKsA)
        tKKrB = kk_thr_mma.make_fragment_B(tKKsB)

        cMqk = cute.make_identity_tensor((self.BLK_Q, self.BLK_KV))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        tKKcMkk = kk_thr_mma.partition_C(cMqk)

        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        qk_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.qk_stage
        )
        kk_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kk_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        for blk in cutlass.range(num_blocks - cutlass.Int32(1), unroll=1):
            (
                q_consumer_state,
                k_consumer_state,
                qk_producer_state,
                kk_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.run_aux_loop_body(
                sQK,
                sKK_inv,
                sKK_opd,
                sAlpha,
                sBeta,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                qk_pipeline,
                qk_producer_state,
                kk_pipeline,
                kk_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                work_desc,
                scale,
                blk,
                False,
                qk_tiled_mma,
                kk_tiled_mma,
                tQKrQ,
                tQKrK,
                tKKrA,
                tKKrB,
                tQKcMqk,
                tKKcMkk,
                aux_tidx,
            )

        last_blk = num_blocks - cutlass.Int32(1)
        (
            q_consumer_state,
            k_consumer_state,
            qk_producer_state,
            kk_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        ) = self.run_aux_loop_body(
            sQK,
            sKK_inv,
            sKK_opd,
            sAlpha,
            sBeta,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            qk_pipeline,
            qk_producer_state,
            kk_pipeline,
            kk_producer_state,
            alpha_pipeline,
            alpha_consumer_state,
            beta_pipeline,
            beta_consumer_state,
            work_desc,
            scale,
            last_blk,
            True,
            qk_tiled_mma,
            kk_tiled_mma,
            tQKrQ,
            tQKrK,
            tKKrA,
            tKKrB,
            tQKcMqk,
            tKKcMkk,
            aux_tidx,
        )

    # ─── Kernel entry point ───────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_o: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        g_tensormaps: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
        grid_x: int,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        q_smem_layout = cute.slice_(q_storage_layout, (None, None, 0))
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        k_smem_layout = cute.slice_(k_storage_layout_ds, (None, None, 0))
        v_smem_layout = cute.slice_(v_storage_layout_ds, (None, None, 0))
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        o_smem_layout = cute.slice_(o_storage_layout, (None, None, 0))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op, g_q, q_smem_layout, (self.BLK_Q, self.D)
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op, g_k, k_smem_layout, (self.D, self.BLK_KV)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op, g_v, v_smem_layout, (self.D, self.BLK_KV)
        )

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op, g_o, o_smem_layout, (self.D, self.BLK_Q)
        )

        dtype_bytes = self.dtype.width // 8
        self.tma_load_q_bytes = cute.size(q_smem_layout) * dtype_bytes
        self.tma_load_k_bytes = cute.size(k_smem_layout) * dtype_bytes
        self.tma_load_v_bytes = cute.size(v_smem_layout) * dtype_bytes

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV, self.qk_stage), order=(0, 1, 2)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.kk_stage), order=(0, 1, 2)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        beta_storage_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            qk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.qk_stage * 2]
            kk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kk_stage * 2]
            alpha_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.alpha_beta_stage * 2
            ]
            beta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.alpha_beta_stage * 2
            ]

            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)],
                128,
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)],
                128,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)],
                128,
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)],
                16,
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[
                    self.inverse_dtype, cute.cosize(kk_storage_layout)
                ],
                16,
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)],
                128,
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(alpha_storage_layout)
                ],
                16,
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_storage_layout)],
                16,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            g_alpha,
            g_beta,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            g_state,
            g_init_state,
            g_state_checkpoints,
            checkpoint_cu_starts,
            g_tensormaps,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            total_checkpoints,
            checkpoint_every_n_tokens,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(512, 1, 1),
            max_number_threads=(512, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        g_tensormaps: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_STATE_MMA_WARP_GROUPS = 2
        NUM_AUX_MMA_WARP_GROUPS = 1
        THREADS_PER_WARP_GROUP = 128
        WARPS_PER_WARP_GROUP = 4
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_STATE_MMA_WARP_GROUPS + NUM_AUX_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        (
            load_registers,
            state_mma_registers,
            aux_mma_registers,
        ) = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_STATE_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)
        )
        ldst_warp_role = cute.arch.make_warp_uniform(
            warp_idx % cutlass.Int32(WARPS_PER_WARP_GROUP)
        )

        if warp_idx == LoadStoreWarpRole.LOAD_QKV:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)

        work_desc = self.get_next_work(
            cu_seqlens,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
        )
        tok_end = work_desc.tok_offset + work_desc.seq_len
        num_blocks = (
            work_desc.seq_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        # math_tidx / wg_idx: valid for Math WG threads; LdSt WG gets negative values (unused)
        math_tidx = tidx - cutlass.Int32(THREADS_PER_WARP_GROUP)
        wg_idx = math_tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)

        # ── Smem allocation ───────────────────────────────────────────────────
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV, self.qk_stage), order=(0, 1, 2)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.kk_stage), order=(0, 1, 2)
        )
        sKK_inv = storage.smem_kk.get_tensor(kk_layout)
        kk_opd_ptr = cute.recast_ptr(storage.smem_kk.data_ptr(), dtype=self.dtype)
        sKK_opd = cute.make_tensor(kk_opd_ptr, kk_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        beta_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))
        sBeta = storage.smem_beta.get_tensor(beta_layout)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # PipelineTmaAsync release is signalled by one lane per consumer warp.
        qk_load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (NUM_STATE_MMA_WARP_GROUPS + NUM_AUX_MMA_WARP_GROUPS)
            * WARPS_PER_WARP_GROUP,
        )
        v_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            NUM_STATE_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP,
        )
        vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        alpha_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (NUM_AUX_MMA_WARP_GROUPS + NUM_STATE_MMA_WARP_GROUPS)
            * THREADS_PER_WARP_GROUP,
        )
        beta_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_AUX_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_STATE_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=load_producer_group,
            consumer_group=qk_load_consumer_group,
            tx_count=self.tma_load_q_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=load_producer_group,
            consumer_group=qk_load_consumer_group,
            tx_count=self.tma_load_k_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=load_producer_group,
            consumer_group=v_consumer_group,
            tx_count=self.tma_load_v_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
            num_stages=self.o_stage,
            producer_group=o_producer_group,
            consumer_group=o_consumer_group,
        )
        qk_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.qk_mbar_ptr.data_ptr(),
            num_stages=self.qk_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_AUX_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_STATE_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
        )
        kk_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.kk_mbar_ptr.data_ptr(),
            num_stages=self.kk_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_AUX_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_STATE_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=alpha_consumer_group,
        )
        beta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.beta_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=beta_consumer_group,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if (
            work_desc.seq_len != cutlass.Int32(0)
            and warp_group_idx == WarpGroupRole.LDST
        ):
            cute.arch.setmaxregister_decrease(load_registers)
            if ldst_warp_role == LoadStoreWarpRole.LOAD_QKV:
                self.run_load_qkv_role(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    tma_atom_q,
                    tma_tensor_q,
                    tma_atom_k,
                    tma_tensor_k,
                    tma_atom_v,
                    tma_tensor_v,
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                )
            elif ldst_warp_role == LoadStoreWarpRole.STORE_O:
                CollectiveStoreTma(self.BLK_Q, self.D).run(
                    sO,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    o_pipeline,
                    num_blocks,
                    work_desc,
                    num_seqs,
                    self.o_stage,
                    num_q_heads,
                    num_v_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                self.run_load_beta_role(
                    sBeta,
                    g_beta,
                    beta_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                self.run_load_alpha_role(
                    sAlpha,
                    g_alpha,
                    alpha_pipeline,
                    scale,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
        elif work_desc.seq_len != cutlass.Int32(0):
            if warp_group_idx == WarpGroupRole.MATH_AUX:
                cute.arch.setmaxregister_decrease(aux_mma_registers)
                self.run_aux_math_role(
                    sQ_SD,
                    sK_SD,
                    sQK,
                    sKK_inv,
                    sKK_opd,
                    sAlpha,
                    sBeta,
                    q_pipeline,
                    k_pipeline,
                    qk_pipeline,
                    kk_pipeline,
                    alpha_pipeline,
                    beta_pipeline,
                    work_desc,
                    scale,
                    math_tidx,
                    num_blocks,
                )
            else:
                cute.arch.setmaxregister_increase(state_mma_registers)
                self.run_state_math_role(
                    sQ_SD,
                    sK_SD,
                    sK_DS,
                    sV_DS,
                    sQK,
                    sKK_inv,
                    sKK_opd,
                    sO,
                    sAlpha,
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    o_pipeline,
                    qk_pipeline,
                    kk_pipeline,
                    alpha_pipeline,
                    g_state,
                    g_init_state,
                    g_state_checkpoints,
                    checkpoint_cu_starts,
                    work_desc,
                    scale,
                    wg_idx,
                    math_tidx,
                    num_blocks,
                    num_q_heads,
                    num_v_heads,
                    num_sab_heads,
                    num_seqs,
                    total_checkpoints,
                    checkpoint_every_n_tokens,
                )


# ─── Public API ──────────────────────────────────────────────────────────────


def delta_rule_prefill_dsl_sm90(
    o: torch.Tensor,  # (total_seqlen, num_o_heads, D) fp16/bf16, output
    state: torch.Tensor,  # (num_seqs, num_sab_heads, D, D) fp32, output
    q: torch.Tensor,  # (total_seqlen, num_q_heads, D)
    k: torch.Tensor,  # (total_seqlen, num_k_heads, D)
    v: torch.Tensor,  # (total_seqlen, num_v_heads, D)
    init_state: torch.Tensor | None,  # (num_seqs, num_sab_heads, D, D) fp32, optional
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,  # (num_seqs+1,) int64
    scale: float,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
):
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_driver

    D = q.shape[-1]

    num_seqs = cu_seqlens.shape[0] - 1
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)

    if not _FullyFusedDeltaRuleSm90.can_implement(
        num_q_heads, num_k_heads, num_v_heads, D, q.element_size()
    ):
        raise RuntimeError("can_implement failed")
    if D != 128:
        raise RuntimeError(f"DSL kernel only supports D=128, got {D}")

    needs_alpha = alpha is not None
    needs_beta = beta is not None
    needs_init_state = init_state is not None
    needs_checkpointing = checkpoint_every_n_tokens > 0
    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }.get(q.dtype)
    if kernel_dtype is None:
        raise RuntimeError(f"DSL kernel only supports fp16/bf16 inputs, got {q.dtype}")

    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError(
            f"q/k/v/o dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}, {o.dtype}"
        )
    if alpha is not None and alpha.dtype != torch.float32:
        raise RuntimeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    if beta is not None and beta.dtype != torch.float32:
        raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
    if init_state is not None and init_state.dtype != torch.float32:
        raise RuntimeError(
            f"init_state must have dtype torch.float32, got {init_state.dtype}"
        )
    if state.dtype != torch.float32:
        raise RuntimeError(f"state must have dtype torch.float32, got {state.dtype}")
    if cu_seqlens.dtype != torch.int64:
        raise RuntimeError(
            f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
        )

    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("o", o),
        ("state", state),
        ("cu_seqlens", cu_seqlens),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("alpha", alpha), ("beta", beta), ("init_state", init_state)):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    total_seqlen = q.shape[0]
    num_o_heads = o.shape[1]
    q_tma = q.as_strided(
        (total_seqlen, D, num_q_heads),
        (num_q_heads * D, 1, D),
    )
    k_tma = k.as_strided(
        (D, total_seqlen, num_k_heads),
        (1, num_k_heads * D, D),
    )
    v_tma = v.as_strided(
        (D, total_seqlen, num_v_heads),
        (1, num_v_heads * D, D),
    )
    o_tma = o.as_strided(
        (D, total_seqlen, num_o_heads),
        (1, num_o_heads * D, D),
    )
    total_checkpoints = state_checkpoints.shape[0] if needs_checkpointing else 1

    workspace_size = get_device_sm_count(q.device) * 128
    tensormaps_t = _get_cache_buf("gdn_cp_prefill_tensormaps", workspace_size, q.device)

    stream_val = torch.cuda.current_stream().cuda_stream
    stream = cuda_driver.CUstream(stream_val)

    enable_tvm_ffi = True
    if enable_tvm_ffi:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )

    # Keep head counts and varlen extents runtime values across cached compiles.
    q_cute = from_dlpack(q_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    k_cute = from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    v_cute = from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    o_cute = from_dlpack(o_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    alpha_cute = (
        from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_alpha
        else None
    )
    beta_cute = (
        from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_beta
        else None
    )
    state_cute = from_dlpack(state.reshape(-1), assumed_align=16).mark_layout_dynamic()
    init_state_cute = (
        from_dlpack(init_state.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_init_state
        else None
    )
    state_checkpoints_cute = (
        from_dlpack(
            state_checkpoints.reshape(-1), assumed_align=16
        ).mark_layout_dynamic()
        if needs_checkpointing
        else None
    )
    checkpoint_cu_cute = (
        from_dlpack(checkpoint_cu_starts, assumed_align=8).mark_layout_dynamic()
        if needs_checkpointing
        else None
    )
    tensormaps_cute = from_dlpack(tensormaps_t, assumed_align=128).mark_layout_dynamic()
    cu_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()

    delta_rule_kernel = _FullyFusedDeltaRuleSm90(
        needs_alpha, needs_beta, needs_init_state, needs_checkpointing, kernel_dtype
    )

    kernel_args = (
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        alpha_cute,
        beta_cute,
        state_cute,
        init_state_cute,
        state_checkpoints_cute,
        checkpoint_cu_cute,
        tensormaps_cute,
        cu_cute,
        cutlass.Float32(scale),
        cutlass.Int32(num_q_heads),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(num_seqs),
        cutlass.Int32(total_checkpoints),
        cutlass.Int32(checkpoint_every_n_tokens),
        num_seqs * num_sab_heads,
        stream,
    )
    compiled_delta_rule_kernel = cached_compile(
        delta_rule_kernel,
        *kernel_args,
        compile_options=(cute.GPUArch("sm_90a"),),
    )
    compiled_delta_rule_kernel(*kernel_args)
