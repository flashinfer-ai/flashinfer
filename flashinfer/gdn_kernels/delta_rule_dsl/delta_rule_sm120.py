import functools
from enum import IntEnum

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp, warpgroup, cpasync
from ...utils import get_device_sm_count, _get_cache_buf
from .alpha import AlphaProcessor
from .collective_store_tma import CollectiveStoreTma
from .custom_compile_cache import (
    KeyedCompileMixin,
    cached_compile,
    get_cached_compile,
    sm12x_compile_options,
)
from .collective_inverse_hmma import CollectiveInverse
from .helpers import SM80, round_down, state_dtype_to_cutlass
from .schedule import WorkDesc
from .varlen_helper import (
    check_checkpoint_key_range,
    check_slot_args,
    checkpoint_row_align,
    is_integer_dtype,
)


@functools.cache
def _sm120_compile_options(device):
    return (cute.EnableTVMFFI(True),) + sm12x_compile_options(device)


# ─── Named-barrier IDs used by the compute kernel ────────────────────────────
# Must not conflict with each other or with pipeline barrier storage.


class NamedBarrier(IntEnum):
    MATH_WG0 = 4  # OrderedMathBarriers: StreamkBarrier0
    MATH_WG1 = 5  # OrderedMathBarriers: StreamkBarrier1
    KK_SYNC = 13  # sync all 128 WG0 threads before collective_inverse
    QK_SYNC = 14  # sync all 128 WG1 threads before the QK store into sQK


class WarpGroupRole(IntEnum):
    LDST = 0
    MATH_KK = 1
    MATH_QK = 2


class LoadStoreWarpRole(IntEnum):
    LOAD_QKV = 0
    STORE_O = 1
    LOAD_BETA = 2
    LOAD_ALPHA = 3


class MathWarpGroupRole(IntEnum):
    KK = 0
    QK = 1


# ─── Warp-specialized delta-rule kernel ───────────────────────────────────────
# Grid: (num_seqs * num_sab_heads, 1, 1)
# Block: 384 threads → WG0=[0,127], WG1=[128,255], WG2=[256,383]
#
# needs_alpha / needs_beta / needs_init_state are class attributes set in __init__.
# The JIT compiler specialises per instance, so they are compile-time booleans
# inside the kernel without any parameter-passing trickery.


class _FullyFusedDeltaRuleSm120(KeyedCompileMixin):
    @staticmethod
    def get_register_requirements(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
        num_mma_warp_groups: int,
        threads_per_warp_group: int,
    ) -> tuple[int, int]:
        reg_alloc_granularity = 8
        load_registers = 40 - 2 * reg_alloc_granularity
        total_registers = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // threads_per_warp_group
        )
        mma_registers = round_down(
            (total_registers - load_registers) // num_mma_warp_groups,
            reg_alloc_granularity,
        )
        return min(248, load_registers), min(248, mma_registers)

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
        initial_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        checkpoint_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        use_state_indices: bool = False,
        cu_seqlens_dtype: torch.dtype = torch.int64,
        state_indices_dtype: torch.dtype | None = None,
        checkpoint_cu_starts_dtype: torch.dtype | None = None,
        state_inner_strides: tuple[int, ...] | None = None,
        init_state_inner_strides: tuple[int, ...] | None = None,
        use_output_state_indices: bool = False,
        use_checkpoint_state_indices: bool = False,
        use_prefix_lens: bool = False,
        output_state_indices_dtype: torch.dtype | None = None,
        checkpoint_state_indices_dtype: torch.dtype | None = None,
        prefix_lens_dtype: torch.dtype | None = None,
        checkpoint_row_align: int | None = None,
    ):
        self.needs_alpha = needs_alpha
        self.needs_beta = needs_beta
        self.needs_init_state = needs_init_state
        self.needs_checkpointing = needs_checkpointing
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.checkpoint_state_dtype = checkpoint_state_dtype
        self.use_state_indices = use_state_indices
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.state_indices_dtype = state_indices_dtype
        self.checkpoint_cu_starts_dtype = checkpoint_cu_starts_dtype
        self.state_inner_strides = state_inner_strides
        self.init_state_inner_strides = init_state_inner_strides
        self._set_slot_flags(
            use_output_state_indices,
            use_checkpoint_state_indices,
            use_prefix_lens,
            output_state_indices_dtype,
            checkpoint_state_indices_dtype,
            prefix_lens_dtype,
            checkpoint_row_align,
        )
        self.inverse_dtype = cutlass.Float16
        self.BLK_Q = 64
        self.BLK_KV = 64
        self.D = 128
        self.q_stage = 1
        self.k_stage = 2
        self.v_stage = 1
        self.o_stage = 1
        self.alpha_beta_stage = 2
        self.manual_cache_key(
            "needs_alpha",
            "needs_beta",
            "needs_init_state",
            "needs_checkpointing",
            "dtype",
            "acc_dtype",
            "initial_state_dtype",
            "state_dtype",
            "checkpoint_state_dtype",
            "use_state_indices",
            "cu_seqlens_dtype",
            "state_indices_dtype",
            "checkpoint_cu_starts_dtype",
            "state_inner_strides",
            "init_state_inner_strides",
            *self.SLOT_KEY_ATTRS,
            "inverse_dtype",
            "BLK_Q",
            "BLK_KV",
            "D",
            "q_stage",
            "k_stage",
            "v_stage",
            "o_stage",
            "alpha_beta_stage",
        )

    # Compile-key attributes of the slot-addressing flags.
    SLOT_KEY_ATTRS = (
        "use_output_state_indices",
        "use_checkpoint_state_indices",
        "use_prefix_lens",
        "output_state_indices_dtype",
        "checkpoint_state_indices_dtype",
        "prefix_lens_dtype",
        "checkpoint_row_align",
    )

    def _set_slot_flags(
        self,
        use_output_state_indices: bool = False,
        use_checkpoint_state_indices: bool = False,
        use_prefix_lens: bool = False,
        output_state_indices_dtype: torch.dtype | None = None,
        checkpoint_state_indices_dtype: torch.dtype | None = None,
        prefix_lens_dtype: torch.dtype | None = None,
        checkpoint_row_align: int | None = None,
    ):
        """Slot addressing (paged recurrent state).

        Each flag gates its own code path, so calls without these arguments
        compile to the same kernel. ``use_output_state_indices``: the final
        state goes to its own row (negative: skipped).
        ``use_checkpoint_state_indices``: checkpoints go to pool rows
        (negative: skipped). ``use_prefix_lens``: absolute checkpoint
        positions; an unaligned prefix is processed from its 64-aligned
        position (virtual left padding). With any of them a negative read row
        starts from the zero state. ``checkpoint_row_align``: alignment in
        elements of the checkpoint pool rows (see ``checkpoint_row_align``).
        """
        self.use_output_state_indices = use_output_state_indices
        self.use_checkpoint_state_indices = use_checkpoint_state_indices
        self.use_prefix_lens = use_prefix_lens
        self.output_state_indices_dtype = output_state_indices_dtype
        self.checkpoint_state_indices_dtype = checkpoint_state_indices_dtype
        self.prefix_lens_dtype = prefix_lens_dtype
        self.checkpoint_row_align = checkpoint_row_align
        self.slot_mode = (
            use_output_state_indices or use_checkpoint_state_indices or use_prefix_lens
        )

    @property
    def chunked_slot_checkpoints(self) -> bool:
        """The slot checkpoint paths run the blocks between two checkpoint
        positions as one loop (see ``run_math_role``)."""
        return bool(
            self.needs_checkpointing
            and (self.use_prefix_lens or self.use_checkpoint_state_indices)
        )

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
        tok_start = cu_seqlens[seq_idx]
        seq_len = cutlass.Int32(cu_seqlens[seq_idx + 1] - tok_start)

        return WorkDesc(
            seq_idx=seq_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_start,
            seq_len=seq_len,
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
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
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
        tKKrKK_inv = cute.make_fragment_like(tKKrKK, self.inverse_dtype)
        tKKrKK_cv = thr_store.retile(tKKrKK_inv)
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            tKKrKK_inv[i] = self.inverse_dtype(tKKrKK[i])
        cute.copy(tiled_store, tKKrKK_cv, tKKsKK)

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
            tKKrKK_cv2 = thr_load.retile(tKKrKK_cpy)
            cute.copy(tiled_load, thr_load.partition_S(sKK_inv), tKKrKK_cv2)
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
            tKKrKK_cv3 = thr_store.retile(tKKrKK_cvt)
            cute.copy(tiled_store, tKKrKK_cv3, tKKsKK2)

    # ─── kk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def kk_epi(
        self,
        tKKrKK: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        alpha_stage: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, t = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cute.math.exp2(
                    cutlass.Float32(alpha_cumlog[s]) - cutlass.Float32(alpha_cumlog[t]),
                    fastmath=True,
                )
        if cutlass.const_expr(self.needs_beta):
            beta_row = sBeta[None, beta_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, _ = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cutlass.Float32(beta_row[s])

    # ─── qk_or_kk_mask ────────────────────────────────────────────────────────

    @cute.jit
    def qk_or_kk_mask(
        self,
        frag: cute.Tensor,
        coord_tensor: cute.Tensor,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        for i in cutlass.range_constexpr(cute.size(frag)):
            s, t = coord_tensor[i]
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and (s < B and t < B)
            if not pred:
                frag[i] = cutlass.Float32(0.0)

    # ─── qk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def qk_epi(
        self,
        tQKrQK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                s, t = tQKcMqk[i]
                tQKrQK[i] = (
                    tQKrQK[i]
                    * cute.math.exp2(
                        cutlass.Float32(alpha_cumlog[s])
                        - cutlass.Float32(alpha_cumlog[t]),
                        fastmath=True,
                    )
                    * scale
                )
        else:
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                tQKrQK[i] = tQKrQK[i] * scale

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
        tok_start,
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
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        alpha_stage: cutlass.Int32,
        tok_begin=None,
    ):
        # Rows outside [tok_begin, tok_end) get gate 1: padding rows before the
        # sequence (tok_begin given) leave the state unchanged.
        lane_id = cute.arch.lane_idx()
        sAlpha_k = sAlpha[None, None, alpha_stage]
        num_iters = self.BLK_Q // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if cutlass.const_expr(tok_begin is None):
                if tok < tok_end:
                    sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = g_alpha[
                        tok * num_sab_heads + sab_head_idx
                    ]
                else:
                    sAlpha_k[row, AlphaProcessor.CUMSUM_LOG] = cutlass.Float32(1.0)
            else:
                if tok >= tok_begin and tok < tok_end:
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
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        beta_stage: cutlass.Int32,
        tok_begin=None,
    ):
        # Rows outside [tok_begin, tok_end) get beta 0: with gate 1 they are
        # identity rows that neither read nor write the state.
        lane_id = cute.arch.lane_idx()
        sBeta_k = sBeta[None, beta_stage]
        num_iters = self.BLK_KV // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if cutlass.const_expr(tok_begin is None):
                if tok < tok_end:
                    sBeta_k[row] = g_beta[tok * num_sab_heads + sab_head_idx]
                else:
                    sBeta_k[row] = cutlass.Float32(0.0)
            else:
                if tok >= tok_begin and tok < tok_end:
                    sBeta_k[row] = g_beta[tok * num_sab_heads + sab_head_idx]
                else:
                    sBeta_k[row] = cutlass.Float32(0.0)

    # ─── kv_load / kv_store ───────────────────────────────────────────────────

    @cute.jit
    def kv_load(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            tKVrKV[i] = gKV[k_idx, v_idx].to(self.acc_dtype)

    @cute.jit
    def kv_store(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            gKV[k_idx, v_idx] = tKVrKV[i].to(gKV.element_type)

    @cute.jit
    def kv_store_pairs(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        """``kv_store`` into a tile with the static layout (D, D):(1, D)
        whose base is aligned to two elements. Fragment elements 2j and
        2j + 1 of every thread are (v, k) and (v, k + 1) with k even (see
        ``check_kv_pairs``): each pair is one vector store of the same two
        values, half the store instructions of ``kv_store``."""
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        dtype = gKV.element_type
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=2 * dtype.width
        )
        pair = cute.make_rmem_tensor(2, dtype)
        for j in cutlass.range_constexpr(cute.size(tKVrKV) // 2):
            v_idx, k_idx = tKVcKV[2 * j]
            pair[0] = tKVrKV[2 * j].to(dtype)
            pair[1] = tKVrKV[2 * j + 1].to(dtype)
            dst = cute.make_tensor(
                cute.domain_offset((k_idx, v_idx), gKV).iterator.align(
                    2 * dtype.width // 8
                ),
                cute.make_layout(2),
            )
            cute.copy(store_atom, pair, dst)

    def check_kv_pairs(self, kv_tiled_mma):
        """Trace-time check of the fragment order ``kv_store_pairs`` relies
        on, for every thread of the state MMA."""
        c_kv = cute.make_identity_tensor((self.D, self.D))
        for t in range(cute.size(kv_tiled_mma.thr_layout_vmnk)):
            coords = kv_tiled_mma.get_slice(t).partition_C(c_kv)
            for j in range(cute.size(coords) // 2):
                v0, k0 = coords[2 * j]
                v1, k1 = coords[2 * j + 1]
                assert v1 == v0 and k1 == k0 + 1 and k0 % 2 == 0, (t, j)

    @property
    def paired_row_checkpoints(self) -> bool:
        """Pool-row checkpoints of the slot paths are written as pairs (rows
        aligned to two elements or more, see ``checkpoint_row_align``)."""
        return bool(
            self.needs_checkpointing
            and self.use_checkpoint_state_indices
            and self.checkpoint_row_align is not None
            and self.checkpoint_row_align > 1
        )

    @cute.jit
    def store_slot_checkpoint_run(
        self,
        tKVrKV: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        kv_thr_mma,
        seq_idx: cutlass.Int32,
        o_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        ckpt_key,
        row_pref,
    ):
        """``maybe_store_checkpoint`` of the slot paths, with pool-row
        checkpoints written as pairs when ``paired_row_checkpoints``."""
        if cutlass.const_expr(self.paired_row_checkpoints):
            if row_pref >= cutlass.Int32(0):
                row_stride = cute.assume(
                    g_state_checkpoints.stride[0], divby=self.checkpoint_row_align
                )
                pool_layout = cute.make_layout(
                    (self.D, self.D, num_sab_heads, g_state_checkpoints.shape[0]),
                    stride=(1, self.D, self.D * self.D, row_stride),
                )
                mPool = cute.make_tensor(g_state_checkpoints.iterator, pool_layout)
                self.kv_store_pairs(
                    tKVrKV, mPool[None, None, o_head_idx, row_pref], kv_thr_mma
                )
        else:
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                block_end,
                seq_len,
                ckpt_key=ckpt_key,
                row_pref=row_pref,
            )

    @cute.jit
    def maybe_store_checkpoint(
        self,
        tKVrKV: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        kv_thr_mma,
        seq_idx: cutlass.Int32,
        o_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        ckpt_key=None,
        row_pref=None,
    ):
        if cutlass.const_expr(
            self.needs_checkpointing
            and not (self.use_prefix_lens or self.use_checkpoint_state_indices)
        ):
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
                self.kv_store(tKVrKV, gCheckpointKV, kv_thr_mma)
        elif cutlass.const_expr(self.needs_checkpointing):
            self._store_slot_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                block_end,
                seq_len,
                ckpt_key,
                row_pref,
            )

    @cute.jit
    def checkpoint_entry_key(
        self,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        seq_idx: cutlass.Int32,
        ckpt_origin,
    ):
        """Per-sequence key of ``checkpoint_entry``, computed once before the
        block loop (None on the packed path without slot arguments).

        ``key = (checkpoint_cu_starts[seq] - 1) * N + origin % N`` with
        ``origin`` the absolute position of the work item's first row (0
        without ``prefix_lens``). ``key + block_end`` is then congruent to the
        absolute block end modulo N, and its quotient by N is the entry of the
        checkpoint due there. One live register replaces the origin and the
        sequence's first entry in the register-bound math warp groups.
        """
        key = None
        if cutlass.const_expr(
            self.needs_checkpointing
            and (self.use_prefix_lens or self.use_checkpoint_state_indices)
        ):
            key = (
                cutlass.Int32(checkpoint_cu_starts[seq_idx]) - cutlass.Int32(1)
            ) * checkpoint_every_n_tokens
            if cutlass.const_expr(self.use_prefix_lens):
                key = key + ckpt_origin % checkpoint_every_n_tokens
        return key

    @cute.jit
    def checkpoint_entry(
        self,
        checkpoint_every_n_tokens: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        ckpt_key,
    ):
        """Checkpoint entry due at the end of a block, or -1 if none is due.

        ``block_end`` and ``seq_len`` count from the work item's first row;
        ``ckpt_key`` comes from ``checkpoint_entry_key``. A checkpoint is due
        when the absolute block end is a multiple of N; the j-th one of the
        sequence in this call is entry ``checkpoint_cu_starts[seq] + j``.
        """
        pos = ckpt_key + block_end
        entry = cutlass.Int32(-1)
        if block_end <= seq_len and pos % checkpoint_every_n_tokens == cutlass.Int32(0):
            # pos >= 0 here: key >= -N and block_end > 0.
            entry = pos // checkpoint_every_n_tokens
        return entry

    @cute.jit
    def prefetch_checkpoint_row(
        self,
        checkpoint_every_n_tokens: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        ckpt_key,
        g_checkpoint_rows: cute.Tensor,
    ):
        """Pool row of the checkpoint due at ``block_end`` (-1: none due or
        skipped), or None without ``checkpoint_state_indices``.

        Called before the block's math so that the load does not stall the
        checkpoint store after it.
        """
        row = None
        if cutlass.const_expr(
            self.needs_checkpointing and self.use_checkpoint_state_indices
        ):
            row = cutlass.Int32(-1)
            entry = self.checkpoint_entry(
                checkpoint_every_n_tokens, block_end, seq_len, ckpt_key
            )
            if entry >= cutlass.Int32(0):
                row = cutlass.Int32(g_checkpoint_rows[entry])
        return row

    @cute.jit
    def _store_slot_checkpoint(
        self,
        tKVrKV: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        kv_thr_mma,
        seq_idx: cutlass.Int32,
        o_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        ckpt_key,
        row_pref,
        v_row_offset=None,
    ):
        """Checkpoint at absolute positions (``prefix_lens``) and/or into pool
        rows (``checkpoint_state_indices``; ``row_pref`` from
        ``prefetch_checkpoint_row``, skipped when negative).
        ``v_row_offset`` selects the V band of a V-split block (sm_80)."""
        if cutlass.const_expr(self.use_checkpoint_state_indices):
            if row_pref >= cutlass.Int32(0):
                # Pool rows are dense [H, D, D] blocks (checked on the host)
                # addressed by the pool's row stride, which may be padded.
                row_stride = g_state_checkpoints.stride[0]
                if cutlass.const_expr(self.checkpoint_row_align > 1):
                    # Without it the row offset has no known alignment and
                    # every checkpoint is written with scalar stores.
                    row_stride = cute.assume(
                        row_stride, divby=self.checkpoint_row_align
                    )
                pool_layout = cute.make_layout(
                    (self.D, self.D, num_sab_heads, g_state_checkpoints.shape[0]),
                    stride=(1, self.D, self.D * self.D, row_stride),
                )
                mPool = cute.make_tensor(g_state_checkpoints.iterator, pool_layout)
                gCheckpointKV = mPool[None, None, o_head_idx, row_pref]
                if cutlass.const_expr(v_row_offset is not None):
                    gCheckpointKV = cute.domain_offset(
                        (cutlass.Int32(0), v_row_offset), gCheckpointKV
                    )
                self.kv_store(tKVrKV, gCheckpointKV, kv_thr_mma)
        else:
            entry = self.checkpoint_entry(
                checkpoint_every_n_tokens, block_end, seq_len, ckpt_key
            )
            if entry >= cutlass.Int32(0):
                checkpoint_layout = cute.make_ordered_layout(
                    (
                        self.D,
                        self.D,
                        cutlass.Int64(num_sab_heads),
                        total_checkpoints,
                    ),
                    order=(0, 1, 2, 3),
                )
                mCheckpoint = cute.make_tensor(
                    g_state_checkpoints.iterator, checkpoint_layout
                )
                gCheckpointKV = mCheckpoint[None, None, o_head_idx, entry]
                if cutlass.const_expr(v_row_offset is not None):
                    gCheckpointKV = cute.domain_offset(
                        (cutlass.Int32(0), v_row_offset), gCheckpointKV
                    )
                self.kv_store(tKVrKV, gCheckpointKV, kv_thr_mma)

    # ─── compute_loop_body ───────────────────────────────────────────────────
    # Translates the C++ compute_loop_body lambda captured inside compute().
    # Called by Math WGs (tidx >= 128) for one block iteration.

    @cute.jit
    def _gemm_upper_triangular_b(self, tiled_mma, t_a, t_b, t_acc, b_shape):
        """Compute C += A @ B.T; B[n,k] == 0 for n < k, so B.T is upper triangular."""
        atom = cute.make_mma_atom(tiled_mma.op)
        cB = cute.make_identity_tensor(cute.product_each(b_shape))
        # M groups share B coordinates; inspect one M group to avoid duplicate work.
        b_thread_layout = cute.slice_(tiled_mma.thr_layout_vmnk, (None, 0, None, None))
        b_coords = [
            tiled_mma.get_slice(b_thread_layout(tid)).partition_B(cB)
            for tid in range(cute.size(b_thread_layout))
        ]
        # qk_or_kk_mask zeros n < k for QK/KK in partition_B's (N,K) coordinates.
        # CollectiveInverse updates only diagonal/lower blocks; beta scaling and
        # dtype conversion also preserve these zeros.
        for k in cutlass.range_constexpr(cute.size(t_b, mode=[2])):
            for n in cutlass.range_constexpr(cute.size(t_b, mode=[1])):
                if cutlass.const_expr(
                    any(
                        thread_coords[i, n, k][0] >= thread_coords[i, n, k][1]
                        for thread_coords in b_coords
                        for i in range(cute.size(t_b, mode=[0]))
                    )
                ):
                    b = t_b[None, n, k]
                    b = cute.make_tensor(
                        b.iterator, cute.append(b.layout, cute.make_layout(1))
                    )
                    c = t_acc[None, None, n]
                    c = cute.make_tensor(
                        c.iterator, cute.append(c.layout, cute.make_layout(1))
                    )
                    cute.gemm(atom, c, t_a[None, None, k], b, c)

    @cute.jit
    def compute_loop_body(
        self,
        # Smem tensors (staged; caller indexes the active stage)
        sQ_SD: cute.Tensor,  # (BlkQ, D, StagesQ)  – row-major atom, swizzled
        sK_SD: cute.Tensor,  # (BlkKV, D, StagesK) – same atom
        sK_DS: cute.Tensor,  # (D, BlkKV, StagesK) – K transposed
        sV_DS: cute.Tensor,  # (D, BlkKV, StagesV) – V transposed
        sQK: cute.Tensor,  # (BlkQ, BlkKV)
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element
        sO: cute.Tensor,  # O output smem (staged)
        sAlpha: cute.Tensor,  # (BlkQ, AlphaProcessor.NUM_CHANNELS, StagesAlpha) or zero-shaped
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) or zero-shaped
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
        alpha_pipeline,
        alpha_consumer_state,
        beta_pipeline,
        beta_consumer_state,
        # Compile-time flags
        is_first_block: bool,
        is_final_block: bool,
        # Valid token count for masking on final block
        B: cutlass.Int32,
        # Running KV state (D×D fp32, in registers across all blocks)
        tKVrKV: cute.Tensor,
        # Scale factor
        scale: cutlass.Float32,
        # WG role: MathWarpGroupRole.KK or MathWarpGroupRole.QK.
        wg_idx: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx - cutlass.Int32(128)  # relative to compute threads
        kk_thread_idx = thread_idx % cutlass.Int32(128)
        qk_thread_idx = thread_idx % cutlass.Int32(128)

        # ── TiledMMAs ─────────────────────────────────────────────────────────
        blk_q = cute.size(sQ_SD, mode=[0])
        blk_kv = cute.size(sK_SD, mode=[0])
        d = cute.size(sQ_SD, mode=[1])
        tile_shape_qk = (blk_q, blk_kv, d)
        tile_shape_kk = tile_shape_qk
        tile_shape_o1 = (d, blk_q, d)
        tile_shape_o2 = (d, blk_q, blk_kv)
        tile_shape_sk = (d, blk_kv, d)
        tile_shape_newv = (d, blk_kv, blk_kv)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        v_stage = v_consumer_state.index
        o_stage = cutlass.Int32(0)
        alpha_stage = alpha_consumer_state.index
        beta_stage = beta_consumer_state.index

        mma_atom_4w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))
        mma_atom_8w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))

        # QK/KK: 4 warps × 16M = 64M  (1 warpgroup, 128 threads)
        qk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_qk
        )
        kk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_kk
        )

        # O1/O2/SK/NewV: 8 warps × 16M = 128M (both warpgroups, 256 threads)
        o1_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o1
        )
        o2_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o2
        )
        sk_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_sk
        )
        newv_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_newv
        )

        # ── Thread slices ─────────────────────────────────────────────────────
        qk_thr_mma = qk_tiled_mma.get_slice(qk_thread_idx)
        kk_thr_mma = kk_tiled_mma.get_slice(kk_thread_idx)
        sk_thr_mma = sk_tiled_mma.get_slice(thread_idx)
        newv_thr_mma = newv_tiled_mma.get_slice(thread_idx)
        o1_thr_mma = o1_tiled_mma.get_slice(thread_idx)
        o2_thr_mma = o2_tiled_mma.get_slice(thread_idx)
        kv_thr_mma = kv_tiled_mma.get_slice(thread_idx)

        # ── Copy atoms ────────────────────────────────────────────────────────
        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        ldsm_t4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )

        # ── Active smem slices (extract 2D from staged tensors) ───────────────
        sQ_k = sQ_SD[None, None, q_stage]  # (BlkQ, D)
        sK_SD_k = sK_SD[None, None, k_stage]  # (BlkKV, D)
        sK_DS_k = sK_DS[None, None, k_stage]  # (D, BlkKV)

        # ── QK copies ─────────────────────────────────────────────────────────
        qk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, qk_tiled_mma)
        qk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, qk_tiled_mma)
        qk_thr_copy_A = qk_tiled_copy_A.get_slice(qk_thread_idx)
        qk_thr_copy_B = qk_tiled_copy_B.get_slice(qk_thread_idx)

        tQKrQ = qk_thr_mma.make_fragment_A(qk_thr_mma.partition_A(sQ_k))
        tQKrQ_cv = qk_thr_copy_A.retile(tQKrQ)
        tQKsQ = qk_thr_copy_A.partition_S(sQ_SD)
        tQKrK = qk_thr_mma.make_fragment_B(qk_thr_mma.partition_B(sK_SD_k))
        tQKrK_cv = qk_thr_copy_B.retile(tQKrK)
        tQKsK = qk_thr_copy_B.partition_S(sK_SD)

        # ── KK copies (same atom as QK) ───────────────────────────────────────
        kk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, kk_tiled_mma)
        kk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, kk_tiled_mma)
        kk_thr_copy_A = kk_tiled_copy_A.get_slice(kk_thread_idx)
        kk_thr_copy_B = kk_tiled_copy_B.get_slice(kk_thread_idx)

        tKKrA = kk_thr_mma.make_fragment_A(kk_thr_mma.partition_A(sK_SD_k))
        tKKrA_cv = kk_thr_copy_A.retile(tKKrA)
        tKKsA = kk_thr_copy_A.partition_S(sK_SD)
        tKKrB = kk_thr_mma.make_fragment_B(kk_thr_mma.partition_B(sK_SD_k))
        tKKrB_cv = kk_thr_copy_B.retile(tKKrB)
        tKKsB = kk_thr_copy_B.partition_S(sK_SD)

        # ── SK copies ─────────────────────────────────────────────────────────
        # SK B: K loaded from sK_SD (row-major BlkKV×D) with LDSM_N — matches C++ SK B-operand
        # SK C: V loaded from sV_DS (col-major D×BlkKV) with LDSM_T
        sk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, sk_tiled_mma)
        sk_tiled_copy_C = cute.make_tiled_copy_C(ldsm_t4, sk_tiled_mma)
        sk_thr_copy_B = sk_tiled_copy_B.get_slice(thread_idx)
        sk_thr_copy_C = sk_tiled_copy_C.get_slice(thread_idx)

        # Work around DSL make_fragment_B not accepting partition_shape_B output directly.
        tSKrK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_B(cute.slice_(tile_shape_sk, (0, None, None))),
            self.dtype,
        )
        tSKrK_cv = sk_thr_copy_B.retile(tSKrK)
        tSKsK = sk_thr_copy_B.partition_S(sK_SD)

        # ── NewV copies ───────────────────────────────────────────────────────
        newv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, newv_tiled_mma)
        newv_thr_copy_B = newv_tiled_copy_B.get_slice(thread_idx)
        tNewVrB = newv_thr_mma.make_fragment_B(newv_thr_mma.partition_B(sKK_opd))
        tNewVrB_cv = newv_thr_copy_B.retile(tNewVrB)
        tNewVsB = newv_thr_copy_B.partition_S(sKK_opd)

        # ── KV copies ─────────────────────────────────────────────────────────
        kv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_t4, kv_tiled_mma)
        kv_thr_copy_B = kv_tiled_copy_B.get_slice(thread_idx)
        tKVrK = kv_thr_mma.make_fragment_B(kv_thr_mma.partition_B(sK_DS_k))
        tKVrK_cv = kv_thr_copy_B.retile(tKVrK)
        tKVsK = kv_thr_copy_B.partition_S(sK_DS)

        # ── O1/O2 copies ──────────────────────────────────────────────────────
        o1_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o1_tiled_mma)
        o2_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o2_tiled_mma)
        o1_thr_copy_B = o1_tiled_copy_B.get_slice(thread_idx)
        o2_thr_copy_B = o2_tiled_copy_B.get_slice(thread_idx)

        # Direct partition_B(sQ_k) preserves the swizzled Q layout here and produces
        # a non-C++ B fragment shape; derive the fragment from TileShapeO1 instead.
        tOrQ = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_B(cute.slice_(tile_shape_o1, (0, None, None))),
            self.dtype,
        )
        tOrQ_cv = o1_thr_copy_B.retile(tOrQ)
        tOsQ = o1_thr_copy_B.partition_S(sQ_SD)
        tOrQK = o2_thr_mma.make_fragment_B(o2_thr_mma.partition_B(sQK))
        tOrQK_cv = o2_thr_copy_B.retile(tOrQK)
        tOsQK = o2_thr_copy_B.partition_S(sQK)

        # ── O store (R→S STSM) ────────────────────────────────────────────────
        o_stsm = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        o_tiled_copy_r2s = cute.make_tiled_copy_C(o_stsm, o1_tiled_mma)
        o_thr_copy_r2s = o_tiled_copy_r2s.get_slice(thread_idx)
        tOsO = o_thr_copy_r2s.partition_D(sO)

        # ── Coordinate tensors for masking / alpha/beta indexing ──────────────
        cMqk = cute.make_identity_tensor((blk_q, blk_kv))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        cMkk = cMqk  # same shape (BlkKV == BlkQ == 64)
        tKKcMkk = kk_thr_mma.partition_C(cMkk)
        cO = cute.make_identity_tensor((d, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((d, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((d, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)

        # ── KK GEMM (WG0 only) ────────────────────────────────────────────────
        k_pipeline.consumer_wait(k_consumer_state)
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_wait(alpha_consumer_state)
            cute.arch.fence_view_async_shared()
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_wait(beta_consumer_state)
            cute.arch.fence_view_async_shared()
        # Match the C++ reject-non-role-first shape; ptxas keeps BRA.U around
        # the role body instead of predicating the HMMA/LDSM/STSM sequence.
        if wg_idx != MathWarpGroupRole.KK:
            cute.arch.sync_warp()
        else:
            cute.copy(kk_tiled_copy_A, tKKsA[None, None, None, k_stage], tKKrA_cv)
            cute.copy(kk_tiled_copy_B, tKKsB[None, None, None, k_stage], tKKrB_cv)
            tKKrKK = cute.make_rmem_tensor(
                kk_thr_mma.partition_shape_C((blk_kv, blk_kv)), self.acc_dtype
            )
            tKKrKK.fill(self.acc_dtype(0.0))
            cute.gemm(kk_tiled_mma, tKKrKK, tKKrA, tKKrB, tKKrKK)
            self.kk_epi(tKKrKK, tKKcMkk, sAlpha, sBeta, alpha_stage, beta_stage)
            self.qk_or_kk_mask(tKKrKK, tKKcMkk, is_final_block, B)
            self._kk_store_and_inv(
                tKKrKK,
                kk_tiled_mma,
                kk_thread_idx,
                sKK_inv,
                sKK_opd,
                sBeta,
                beta_stage,
                tKKcMkk,
            )
        if cutlass.const_expr(self.needs_beta):
            # Reads of the beta stage complete before its refill (see the k
            # release below).
            cute.arch.fence_view_async_shared()
            beta_pipeline.consumer_release(beta_consumer_state)
            beta_consumer_state.advance()

        # ── QK GEMM (WG1 only) ────────────────────────────────────────────────
        q_pipeline.consumer_wait(q_consumer_state)
        if wg_idx != MathWarpGroupRole.QK:
            cute.arch.sync_warp()
        else:
            cute.copy(qk_tiled_copy_A, tQKsQ[None, None, None, q_stage], tQKrQ_cv)
            cute.copy(qk_tiled_copy_B, tQKsK[None, None, None, k_stage], tQKrK_cv)
            tQKrQK = cute.make_rmem_tensor(
                qk_thr_mma.partition_shape_C((blk_q, blk_kv)), self.acc_dtype
            )
            tQKrQK.fill(self.acc_dtype(0.0))
            cute.gemm(qk_tiled_mma, tQKrQK, tQKrQ, tQKrK, tQKrQK)
            self.qk_epi(tQKrQK, tQKcMqk, sAlpha, alpha_stage, scale)
            self.qk_or_kk_mask(tQKrQK, tQKcMqk, is_final_block, B)
            # Every QK warp reads all of sQK in the previous block's O2, and
            # nothing else orders those reads before the other QK warps'
            # stores of this block.
            cute.arch.barrier(barrier_id=NamedBarrier.QK_SYNC, number_of_threads=128)
            self.qk_store(tQKrQK, sQK, qk_tiled_mma, qk_thread_idx)

        # ── O1: KV_state @ Q (both WGs, skip on first block) ─────────────────
        tOrO = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_C((d, blk_q)), self.acc_dtype
        )
        tOrO.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            cute.copy(o1_tiled_copy_B, tOsQ[None, None, None, q_stage], tOrQ_cv)
            tOrKV = SM80.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
            cute.gemm(o1_tiled_mma, tOrO, tOrKV, tOrQ, tOrO)
            self.o1_epi(tOrO, tOcO, sAlpha, alpha_stage, scale)
        # Reads of the q stage complete before its refill (see the k release).
        cute.arch.fence_view_async_shared()
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        # ── SK: KV_state @ K^T (result negated below via V - SK) ─────────────
        tSKrSK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype
        )
        tSKrSK.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            tSKrS = SM80.make_acc_into_op(tKVrKV, sk_tiled_mma, self.dtype)
            cute.copy(sk_tiled_copy_B, tSKsK[None, None, None, k_stage], tSKrK_cv)
            cute.gemm(sk_tiled_mma, tSKrSK, tSKrS, tSKrK, tSKrSK)

        # ── Load V from smem ──────────────────────────────────────────────────
        v_pipeline.consumer_wait(v_consumer_state)
        tSKrV = self.sk_load_v(tSKrSK, sV_DS, sk_tiled_copy_C, sk_thr_copy_C, v_stage)

        # sk_epi + V - SK  (SK=0 on first block, so V - SK = V)
        if cutlass.const_expr(not is_first_block):
            self.sk_epi(tSKrSK, tSKcSK, sAlpha, alpha_stage)
            for i in cutlass.range_constexpr(cute.size(tSKrV)):
                tSKrV[i] = tSKrV[i] - self.dtype(tSKrSK[i])

        # ── NewV = (V - SK) @ T^T  (ordered: WG0 first) ──────────────────────
        tNewVrA = SM80.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = cute.make_rmem_tensor(
            newv_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype
        )
        self._math_order_wait(wg_idx)
        cute.copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv)
        tNewVrC.fill(self.acc_dtype(0.0))
        self._gemm_upper_triangular_b(
            newv_tiled_mma, tNewVrA, tNewVrB, tNewVrC, sKK_opd.shape
        )
        self._math_order_notify(wg_idx)
        # Reads of the v stage complete before its refill (see the k release).
        cute.arch.fence_view_async_shared()
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()

        # ── O2 = O1 + NewV @ QK  (ordered: WG0 first) ────────────────────────
        tOrNewV = SM80.make_acc_into_op(tNewVrC, o2_tiled_mma, self.dtype)
        self._math_order_wait(wg_idx)
        cute.copy(o2_tiled_copy_B, tOsQK, tOrQK_cv)
        self._gemm_upper_triangular_b(o2_tiled_mma, tOrNewV, tOrQK, tOrO, sQK.shape)
        self._math_order_notify(wg_idx)

        # ── O store to smem ───────────────────────────────────────────────────
        o_pipeline.producer_acquire(o_producer_state)
        self.o_store(
            tOrO,
            tOsO[None, None, None, o_stage],
            o_tiled_copy_r2s,
            o_thr_copy_r2s,
        )
        o_pipeline.producer_commit(o_producer_state)
        o_producer_state.advance()

        # ── KV state update ───────────────────────────────────────────────────
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[B - cutlass.Int32(1), AlphaProcessor.CUMPROD, alpha_stage]
            )

        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        self.kv_decay_v(tOrNewV, tKVcV, sAlpha, alpha_stage, is_final_block, B)

        # KV += NewV @ K
        cute.copy(kv_tiled_copy_B, tKVsK[None, None, None, k_stage], tKVrK_cv)
        cute.gemm(kv_tiled_mma, tKVrKV, tOrNewV, tKVrK, tKVrKV)
        # Fence before every stage release. A release is one mbarrier arrive
        # per warp (lane 0), and it does not wait for this warp's ldmatrix
        # reads of the stage above. Once all math warps have arrived, TMA
        # (async proxy) refills the stage. When the memory pipe is backed up,
        # e.g. by the other warp group's checkpoint stores, the refill can
        # land before a queued read, and the KV update then uses the K of
        # block + 2. The fence completes this thread's shared-memory reads
        # before the release. It also covers the alpha reads released below.
        cute.arch.fence_view_async_shared()
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
            alpha_consumer_state,
            beta_consumer_state,
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
        tok_start,
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
        tok_start,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        tok_begin=None,
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
                    tok_begin,
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
        tok_start,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        tok_begin=None,
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
                    tok_begin,
                )
                cute.arch.fence_view_async_shared()
                beta_pipeline.producer_commit(beta_producer_state)
                beta_producer_state.advance()

    @cute.jit
    def run_math_role(
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
        sBeta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        o_pipeline,
        alpha_pipeline,
        beta_pipeline,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
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
        g_output_state_indices: cute.Tensor = None,
        g_checkpoint_rows: cute.Tensor = None,
        ckpt_origin=None,
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
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=(self.D, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        if cutlass.const_expr(self.paired_row_checkpoints):
            self.check_kv_pairs(kv_tiled_mma)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.D, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        packed_state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_seqs), order=(0, 1, 2, 3)
        )
        state_ref_shape = (g_state.shape[0], g_state.shape[1], self.D, self.D)
        state_ref_layout = cute.make_layout(state_ref_shape, stride=g_state.stride)
        indexed_state_layout = cute.select(state_ref_layout, mode=[3, 2, 1, 0])
        state_idx = work_desc.seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices[work_desc.seq_idx])
            state_layout = indexed_state_layout
        else:
            state_layout = packed_state_layout
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        mState = cute.make_tensor(g_state.iterator, state_layout)
        write_idx = state_idx
        if cutlass.const_expr(self.use_output_state_indices):
            write_idx = cutlass.Int32(g_output_state_indices[work_desc.seq_idx])
        gStateKV = mState[None, None, o_head_idx, write_idx]
        if cutlass.const_expr(self.needs_init_state):
            init_state_ref_layout = cute.make_layout(
                state_ref_shape, stride=g_init_state.stride
            )
            indexed_init_state_layout = cute.select(
                init_state_ref_layout, mode=[3, 2, 1, 0]
            )
            if cutlass.const_expr(self.use_state_indices):
                init_state_layout = indexed_init_state_layout
            else:
                init_state_layout = packed_state_layout
            mInitState = cute.make_tensor(g_init_state.iterator, init_state_layout)
            gInitKV = mInitState[None, None, o_head_idx, state_idx]
            if cutlass.const_expr(self.slot_mode):
                # A negative read row starts from the zero state. Each thread
                # reads its state elements here, before it writes any
                # checkpoint or final state, so read and write rows may alias.
                if state_idx >= cutlass.Int32(0):
                    self.kv_load(tKVrKV, gInitKV, kv_thr_mma)
            else:
                self.kv_load(tKVrKV, gInitKV, kv_thr_mma)

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        ckpt_key = self.checkpoint_entry_key(
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
            work_desc.seq_idx,
            ckpt_origin,
        )
        first_row = self.prefetch_checkpoint_row(
            checkpoint_every_n_tokens,
            cutlass.Int32(self.BLK_KV),
            work_desc.seq_len,
            ckpt_key,
            g_checkpoint_rows,
        )
        if cutlass.const_expr(self.needs_init_state):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
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
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
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
                alpha_consumer_state,
                beta_consumer_state,
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
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
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
            kv_thr_mma,
            work_desc.seq_idx,
            o_head_idx,
            num_sab_heads,
            total_checkpoints,
            cutlass.Int32(self.BLK_KV),
            work_desc.seq_len,
            ckpt_key=ckpt_key,
            row_pref=first_row,
        )

        if cutlass.const_expr(self.chunked_slot_checkpoints):
            # The blocks up to the next checkpoint position run as one inner
            # loop without checkpoint code, and the checkpoint due at the end
            # of the run (if any) is stored after it. The per-block form below
            # carries the predicated pool-row stores and the due test through
            # every block, which slows the math warps of long sequences.
            # Same blocks, same due test, same stores: a run ends at the first
            # block end at or after the next position p with
            # (ckpt_key + p) % N == 0, and no earlier block end of the run is
            # such a position.
            mid_end = num_blocks - cutlass.Int32(1)
            first_end = cutlass.Int32(self.BLK_KV)
            gap = (ckpt_key + first_end) % checkpoint_every_n_tokens
            if gap < cutlass.Int32(0):
                gap = gap + checkpoint_every_n_tokens
            next_pos = first_end + checkpoint_every_n_tokens - gap
            blk = cutlass.Int32(1)
            while blk < mid_end:
                stop = (next_pos + cutlass.Int32(self.BLK_KV - 1)) // cutlass.Int32(
                    self.BLK_KV
                )
                if stop > mid_end:
                    stop = mid_end
                stop_end = stop * cutlass.Int32(self.BLK_KV)
                run_row = self.prefetch_checkpoint_row(
                    checkpoint_every_n_tokens,
                    stop_end,
                    work_desc.seq_len,
                    ckpt_key,
                    g_checkpoint_rows,
                )
                for _ in cutlass.range(blk, stop, cutlass.Int32(1), unroll=1):
                    (
                        q_consumer_state,
                        k_consumer_state,
                        v_consumer_state,
                        o_producer_state,
                        alpha_consumer_state,
                        beta_consumer_state,
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
                        sBeta,
                        kv_tiled_mma,
                        q_pipeline,
                        q_consumer_state,
                        k_pipeline,
                        k_consumer_state,
                        v_pipeline,
                        v_consumer_state,
                        o_pipeline,
                        o_producer_state,
                        alpha_pipeline,
                        alpha_consumer_state,
                        beta_pipeline,
                        beta_consumer_state,
                        False,
                        False,
                        cutlass.Int32(self.BLK_KV),
                        tKVrKV,
                        scale,
                        wg_idx,
                    )
                self.store_slot_checkpoint_run(
                    tKVrKV,
                    g_state_checkpoints,
                    checkpoint_cu_starts,
                    checkpoint_every_n_tokens,
                    kv_thr_mma,
                    work_desc.seq_idx,
                    o_head_idx,
                    num_sab_heads,
                    total_checkpoints,
                    stop_end,
                    work_desc.seq_len,
                    ckpt_key,
                    run_row,
                )
                while next_pos <= stop_end:
                    next_pos = next_pos + checkpoint_every_n_tokens
                blk = stop
        else:
            for blk in cutlass.range(
                cutlass.Int32(1),
                num_blocks - cutlass.Int32(1),
                cutlass.Int32(1),
                unroll=1,
            ):
                loop_row = self.prefetch_checkpoint_row(
                    checkpoint_every_n_tokens,
                    (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    ckpt_key,
                    g_checkpoint_rows,
                )
                (
                    q_consumer_state,
                    k_consumer_state,
                    v_consumer_state,
                    o_producer_state,
                    alpha_consumer_state,
                    beta_consumer_state,
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
                    sBeta,
                    kv_tiled_mma,
                    q_pipeline,
                    q_consumer_state,
                    k_pipeline,
                    k_consumer_state,
                    v_pipeline,
                    v_consumer_state,
                    o_pipeline,
                    o_producer_state,
                    alpha_pipeline,
                    alpha_consumer_state,
                    beta_pipeline,
                    beta_consumer_state,
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
                    kv_thr_mma,
                    work_desc.seq_idx,
                    o_head_idx,
                    num_sab_heads,
                    total_checkpoints,
                    (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    ckpt_key=ckpt_key,
                    row_pref=loop_row,
                )

        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            last_row = self.prefetch_checkpoint_row(
                checkpoint_every_n_tokens,
                (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
                ckpt_key,
                g_checkpoint_rows,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
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
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
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
                kv_thr_mma,
                work_desc.seq_idx,
                o_head_idx,
                num_sab_heads,
                total_checkpoints,
                (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
                ckpt_key=ckpt_key,
                row_pref=last_row,
            )
        if cutlass.const_expr(self.slot_mode):
            if write_idx >= cutlass.Int32(0):
                self.kv_store(tKVrKV, gStateKV, kv_thr_mma)
        else:
            self.kv_store(tKVrKV, gStateKV, kv_thr_mma)

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
        g_state_indices: cute.Tensor,
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
        g_output_state_indices: cute.Tensor,
        g_checkpoint_rows: cute.Tensor,
        g_prefix_lens: cute.Tensor,
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
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
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

        self.shared_storage: type = SharedStorage

        # The O tensor itself is needed only for the plain stores of the first
        # tile of a left-padded sequence.
        g_o_plain = None
        if cutlass.const_expr(self.use_prefix_lens):
            g_o_plain = g_o

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
            g_state_indices,
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
            g_o_plain,
            g_output_state_indices,
            g_checkpoint_rows,
            g_prefix_lens,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(384, 1, 1),
            max_number_threads=(384, 1, 1),
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
        g_state_indices: cute.Tensor,
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
        g_o: cute.Tensor,
        g_output_state_indices: cute.Tensor,
        g_checkpoint_rows: cute.Tensor,
        g_prefix_lens: cute.Tensor,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_MMA_WARP_GROUPS = 2
        THREADS_PER_WARP_GROUP = 128
        WARPS_PER_WARP_GROUP = 4
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        load_registers, mma_registers = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_MMA_WARP_GROUPS,
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
        pad = None
        ckpt_origin = None
        load_tok_begin = None
        if cutlass.const_expr(self.use_prefix_lens):
            # Padding rows lie before the sequence: alpha/beta loads mask them.
            load_tok_begin = work_desc.tok_offset
            work_desc, pad, ckpt_origin = work_desc.left_padded(
                g_prefix_lens, self.BLK_KV
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
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
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
        load_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        vector_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_q_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_k_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_v_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
            num_stages=self.o_stage,
            producer_group=o_producer_group,
            consumer_group=o_consumer_group,
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=vector_consumer_group,
        )
        beta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.beta_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=vector_consumer_group,
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
                if cutlass.const_expr(self.use_prefix_lens):
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
                        g_o,
                        pad,
                    )
                else:
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
                    load_tok_begin,
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
                    load_tok_begin,
                )
        elif work_desc.seq_len != cutlass.Int32(0):
            cute.arch.setmaxregister_increase(mma_registers)

            self.run_math_role(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                q_pipeline,
                k_pipeline,
                v_pipeline,
                o_pipeline,
                alpha_pipeline,
                beta_pipeline,
                g_state,
                g_init_state,
                g_state_indices,
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
                g_output_state_indices,
                g_checkpoint_rows,
                ckpt_origin,
            )


# ─── Public API ──────────────────────────────────────────────────────────────


@functools.cache
def _get_prefill_kernel(
    needs_alpha,
    needs_beta,
    needs_init_state,
    needs_checkpointing,
    kernel_dtype,
    initial_state_dtype,
    state_dtype,
    checkpoint_state_dtype,
    use_state_indices,
    cu_seqlens_dtype,
    state_indices_dtype,
    checkpoint_cu_starts_dtype,
    state_inner_strides,
    init_state_inner_strides,
    output_state_indices_dtype=None,
    checkpoint_state_indices_dtype=None,
    prefix_lens_dtype=None,
    checkpoint_row_align=None,
):
    return _FullyFusedDeltaRuleSm120(
        needs_alpha,
        needs_beta,
        needs_init_state,
        needs_checkpointing,
        kernel_dtype,
        initial_state_dtype=state_dtype_to_cutlass(initial_state_dtype),
        state_dtype=state_dtype_to_cutlass(state_dtype),
        checkpoint_state_dtype=state_dtype_to_cutlass(checkpoint_state_dtype),
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=cu_seqlens_dtype,
        state_indices_dtype=state_indices_dtype,
        checkpoint_cu_starts_dtype=checkpoint_cu_starts_dtype,
        state_inner_strides=state_inner_strides,
        init_state_inner_strides=init_state_inner_strides,
        use_output_state_indices=output_state_indices_dtype is not None,
        use_checkpoint_state_indices=checkpoint_state_indices_dtype is not None,
        use_prefix_lens=prefix_lens_dtype is not None,
        output_state_indices_dtype=output_state_indices_dtype,
        checkpoint_state_indices_dtype=checkpoint_state_indices_dtype,
        prefix_lens_dtype=prefix_lens_dtype,
        checkpoint_row_align=checkpoint_row_align,
    )


def delta_rule_prefill_dsl(
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
    state_indices: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
    prefix_lens: torch.Tensor | None = None,
):
    """GDN prefill (CuTe DSL).

    ``output_state_indices`` ([num_seqs]; needs ``state_indices``) writes
    sequence i's final state to row ``output_state_indices[i]`` of ``state``
    instead of row ``state_indices[i]``; ``checkpoint_state_indices`` maps
    checkpoint entry e to row ``checkpoint_state_indices[e]`` of the pool
    ``state_checkpoints`` [C, H, D, D]; ``prefix_lens`` ([num_seqs]) makes
    checkpoints absolute: one is due after local token n when
    ``(prefix_lens[i] + n) % checkpoint_every_n_tokens == 0``. Negative rows
    skip the write; with any of them a negative ``state_indices`` row starts
    from the zero state. ``prefix_lens`` needs ``beta``.
    """
    import cuda.bindings.driver as cuda_driver

    device = q.device
    D = q.shape[-1]

    num_seqs = cu_seqlens.shape[0] - 1
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)

    if not _FullyFusedDeltaRuleSm120.can_implement(
        num_q_heads, num_k_heads, num_v_heads, D, q.element_size()
    ):
        raise RuntimeError("can_implement failed")
    if D != 128:
        raise RuntimeError(f"DSL kernel only supports D=128, got {D}")

    needs_alpha = alpha is not None
    needs_beta = beta is not None
    needs_init_state = init_state is not None
    needs_checkpointing = checkpoint_every_n_tokens > 0
    use_state_indices = state_indices is not None
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
    if init_state is not None:
        state_dtype_to_cutlass(init_state.dtype)
    state_dtype_to_cutlass(state.dtype)
    if state_checkpoints is not None:
        state_dtype_to_cutlass(state_checkpoints.dtype)
    if not is_integer_dtype(cu_seqlens.dtype):
        raise RuntimeError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    expected_state_tail = (num_sab_heads, D, D)
    for name, tensor in (("state", state), ("init_state", init_state)):
        if tensor is None:
            continue
        if (
            not use_state_indices and tensor.shape != (num_seqs, *expected_state_tail)
        ) or (use_state_indices and tuple(tensor.shape[1:]) != expected_state_tail):
            raise RuntimeError(
                f"{name} must have shape "
                f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_tail}, "
                f"got {tuple(tensor.shape)}"
            )
    if use_state_indices and (
        not is_integer_dtype(state_indices.dtype) or state_indices.shape != (num_seqs,)
    ):
        raise RuntimeError(
            f"state_indices must have shape {(num_seqs,)} and an integer dtype"
        )

    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("o", o),
        ("cu_seqlens", cu_seqlens),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (
        ("alpha", alpha),
        ("beta", beta),
        ("state_indices", state_indices),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("state", state), ("init_state", init_state)):
        if tensor is None:
            continue
        if not use_state_indices and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    use_output_state_indices = output_state_indices is not None
    use_checkpoint_state_indices = checkpoint_state_indices is not None
    use_prefix_lens = prefix_lens is not None
    check_slot_args(
        num_seqs,
        num_sab_heads,
        D,
        device,
        use_state_indices,
        needs_checkpointing,
        beta,
        state_checkpoints,
        output_state_indices,
        checkpoint_state_indices,
        prefix_lens,
    )
    row_align = (
        checkpoint_row_align(state_checkpoints)
        if use_checkpoint_state_indices
        else None
    )
    if use_checkpoint_state_indices or use_prefix_lens:
        check_checkpoint_key_range(
            checkpoint_state_indices.numel()
            if use_checkpoint_state_indices
            else state_checkpoints.shape[0],
            checkpoint_every_n_tokens,
            q.shape[0],
        )

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

    workspace_size = get_device_sm_count(device) * 128
    tensormaps_t = _get_cache_buf("gdn_prefill_tensormaps", workspace_size, device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

    delta_rule_kernel = _get_prefill_kernel(
        needs_alpha,
        needs_beta,
        needs_init_state,
        needs_checkpointing,
        kernel_dtype,
        initial_state_dtype=(init_state.dtype if needs_init_state else torch.float32),
        state_dtype=state.dtype,
        checkpoint_state_dtype=(
            state_checkpoints.dtype if needs_checkpointing else torch.float32
        ),
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=cu_seqlens.dtype,
        state_indices_dtype=state_indices.dtype if use_state_indices else None,
        checkpoint_cu_starts_dtype=(
            checkpoint_cu_starts.dtype if needs_checkpointing else None
        ),
        state_inner_strides=(tuple(state.stride()[1:]) if use_state_indices else None),
        init_state_inner_strides=(
            tuple(init_state.stride()[1:])
            if use_state_indices and needs_init_state
            else None
        ),
        output_state_indices_dtype=(
            output_state_indices.dtype if use_output_state_indices else None
        ),
        checkpoint_state_indices_dtype=(
            checkpoint_state_indices.dtype if use_checkpoint_state_indices else None
        ),
        prefix_lens_dtype=prefix_lens.dtype if use_prefix_lens else None,
        checkpoint_row_align=row_align,
    )
    # A checkpoint pool is addressed by its row stride (which may be padded),
    # so it is passed whole, never flattened through a copy.
    checkpoints_align = 16
    if not needs_checkpointing:
        checkpoints_arg = None
    elif use_checkpoint_state_indices:
        checkpoints_arg = state_checkpoints
        checkpoints_align = row_align * state_checkpoints.element_size()
    else:
        checkpoints_arg = state_checkpoints.reshape(-1)

    compile_options = _sm120_compile_options(device)
    compiled_delta_rule_kernel = get_cached_compile(delta_rule_kernel, compile_options)
    if compiled_delta_rule_kernel is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        kernel_args = (
            from_dlpack(q_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(o_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            (
                from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic()
                if needs_alpha
                else None
            ),
            (
                from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic()
                if needs_beta
                else None
            ),
            from_dlpack(state, assumed_align=16).mark_layout_dynamic(),
            (
                from_dlpack(init_state, assumed_align=16).mark_layout_dynamic()
                if needs_init_state
                else None
            ),
            (
                from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
                if use_state_indices
                else None
            ),
            (
                from_dlpack(
                    checkpoints_arg, assumed_align=checkpoints_align
                ).mark_layout_dynamic()
                if needs_checkpointing
                else None
            ),
            (
                from_dlpack(checkpoint_cu_starts, assumed_align=8).mark_layout_dynamic()
                if needs_checkpointing
                else None
            ),
            from_dlpack(tensormaps_t, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Float32(scale),
            cutlass.Int32(num_q_heads),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(num_seqs),
            cutlass.Int32(total_checkpoints),
            cutlass.Int32(checkpoint_every_n_tokens),
            *(
                (
                    from_dlpack(tensor, assumed_align=4).mark_layout_dynamic()
                    if tensor is not None
                    else None
                )
                for tensor in (
                    output_state_indices,
                    checkpoint_state_indices,
                    prefix_lens,
                )
            ),
            num_seqs * num_sab_heads,
            stream,
        )
        compiled_delta_rule_kernel = cached_compile(
            delta_rule_kernel,
            *kernel_args,
            compile_options=compile_options,
        )
    compiled_delta_rule_kernel(
        q_tma,
        k_tma,
        v_tma,
        o_tma,
        alpha.reshape(-1) if needs_alpha else None,
        beta.reshape(-1) if needs_beta else None,
        state,
        init_state if needs_init_state else None,
        state_indices if use_state_indices else None,
        checkpoints_arg,
        checkpoint_cu_starts if needs_checkpointing else None,
        tensormaps_t,
        cu_seqlens,
        scale,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        num_sab_heads,
        num_seqs,
        total_checkpoints,
        checkpoint_every_n_tokens,
        output_state_indices,
        checkpoint_state_indices,
        prefix_lens,
        num_seqs * num_sab_heads,
        stream,
    )
