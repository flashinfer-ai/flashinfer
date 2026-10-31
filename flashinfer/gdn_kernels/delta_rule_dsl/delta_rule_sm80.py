import functools
from enum import IntEnum

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp, warpgroup, cpasync
from .alpha import AlphaProcessor
from .collective_store_sm80 import CollectiveStoreSm80
from .pipeline_sm80 import PipelineCpAsyncSm80
from .custom_compile_cache import (
    cached_compile,
    get_cached_compile,
    sm8x_compile_options,
)
from .collective_inverse_hmma import CollectiveInverse
from .delta_rule_sm120 import _FullyFusedDeltaRuleSm120
from .helpers import SM80, state_dtype_to_cutlass
from .schedule import WorkDesc
from .varlen_helper import (
    check_checkpoint_key_range,
    check_slot_args,
    checkpoint_row_align,
    is_integer_dtype,
)


@functools.cache
def _sm80_compile_options(device):
    """The compile options this kernel is built with: the device's SM8x arch."""
    return (cute.EnableTVMFFI(True),) + sm8x_compile_options(device)


# One warp group runs the KK and QK bodies in sequence; it also sets `_BLK_V`.
NUM_MMA_WARP_GROUPS = 1
THREADS_PER_WARP_GROUP = 128
# Every thread issues loads: a thread can only wait on cp.async it issued.
LOAD_THREADS = NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
NUM_MMA_WARPS = LOAD_THREADS // 32
# Rows of C one m16n8k16 atom produces.
_MMA_ATOM_M = 16
# V rows one block owns. It must be the warps' M span: an accumulator is only
# a valid next A operand at one M repeat per warp.
_BLK_V = NUM_MMA_WARPS * _MMA_ATOM_M
# bf16 elements in one 16 B cp.async access.
_LOAD_VEC_ELEMS = 8
_LOAD_ALIGN_BYTES = 16


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _check_state_dtype_supported(**tensors) -> None:
    """Reject an FP8 state, which the SM8x kernels fail to compile."""
    for name, tensor in tensors.items():
        if tensor is not None and tensor.dtype in _FP8_DTYPES:
            raise NotImplementedError(
                f"{name} is {tensor.dtype}; the SM8x GDN prefill kernels do not "
                "support an FP8 state"
            )


def _check_load_alignment(**tensors: torch.Tensor) -> None:
    """Reject a base or stride that breaks the 16 B alignment `_aligned` assumes."""
    for name, t in tensors.items():
        if t is None:
            continue
        if t.data_ptr() % _LOAD_ALIGN_BYTES != 0:
            raise ValueError(
                f"{name} must be {_LOAD_ALIGN_BYTES}-byte aligned for the sm_80 "
                f"delta-rule kernel, got address {t.data_ptr():#x}"
            )
        bad = [
            (dim, s)
            for dim, s in enumerate(t.stride())
            if s != 1 and s % _LOAD_VEC_ELEMS != 0
        ]
        if bad:
            raise ValueError(
                f"{name} strides {t.stride()} are not vectorizable for the sm_80 "
                f"delta-rule kernel: dim(s) {[d for d, _ in bad]} must be a "
                f"multiple of {_LOAD_VEC_ELEMS} elements or contiguous"
            )


# ─── Named-barrier IDs used by the compute kernel ────────────────────────────
# Must not conflict with each other or with pipeline barrier storage.


class NamedBarrier(IntEnum):
    MATH_WG0 = 4  # OrderedMathBarriers: StreamkBarrier0
    MATH_WG1 = 5  # OrderedMathBarriers: StreamkBarrier1
    KK_SYNC = 13  # sync WG0's threads before collective_inverse


# No load/store role: the DSL emits no mbarrier for sm_80, so the whole block loads.
class MathWarpGroupRole(IntEnum):
    KK = 0
    QK = 1


# ─── Warp-specialized delta-rule kernel ───────────────────────────────────────
# Grid: (num_seqs * num_sab_heads * D // D_v, 1, 1)
# Block: LOAD_THREADS threads, NUM_MMA_WARP_GROUPS group(s) of 128
#
# needs_alpha / needs_beta / needs_init_state are class attributes set in __init__.
# The JIT compiler specialises per instance, so they are compile-time booleans
# inside the kernel without any parameter-passing trickery.


class _FullyFusedDeltaRuleSm80(_FullyFusedDeltaRuleSm120):
    # Epilogue, masking, decay and alpha/beta loads are inherited from SM120.
    # sm_80 has no setmaxnreg, so `get_register_requirements` is never called.

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
        blk_v: int | None = None,
        v32: bool = False,
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
        # One stage: loads never run ahead of the math (prefetch is zero), so a
        # second buffer would give no overlap.
        self.k_stage = 1
        self.v_stage = 1
        self.o_stage = 1
        self.alpha_beta_stage = 2

        # State V rows one block owns. Q, K, beta and the chunk inverse do not
        # depend on V, so each block of a split recomputes them.
        self.D_v = self.D if blk_v is None else int(blk_v)
        if self.D % self.D_v:
            raise ValueError(f"blk_v {self.D_v} must divide head_size {self.D}")
        # v32: a band of half the warps' M span. The V GEMMs split N across
        # warps and hand accumulators on through shared memory.
        self.v32 = bool(v32)
        if self.v32:
            if self.D_v * 2 != NUM_MMA_WARPS * _MMA_ATOM_M:
                raise ValueError(
                    f"v32 wants blk_v {NUM_MMA_WARPS * _MMA_ATOM_M // 2}, "
                    f"half the warps' M span, and got {self.D_v}"
                )
            if NUM_MMA_WARPS != 4:
                raise ValueError(
                    f"v32 splits 4 warps as 2 in M and 2 in N; "
                    f"NUM_MMA_WARPS is {NUM_MMA_WARPS}"
                )
        elif self.D_v != NUM_MMA_WARPS * _MMA_ATOM_M:
            raise ValueError(
                f"blk_v {self.D_v} must equal the warps' M span, "
                f"{NUM_MMA_WARPS} * {_MMA_ATOM_M} = {NUM_MMA_WARPS * _MMA_ATOM_M}: "
                f"accumulators are reinterpreted as operands, and that is only "
                f"valid at one M repeat per warp -- unless v32 is set, which "
                f"routes them through shared memory instead"
            )
        self.num_v_splits = self.D // self.D_v

        self.o_writer = CollectiveStoreSm80(self.BLK_Q, self.D_v, LOAD_THREADS)
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
            "v32",
            "BLK_Q",
            "BLK_KV",
            "D",
            "D_v",
            "q_stage",
            "k_stage",
            "v_stage",
            "o_stage",
            "alpha_beta_stage",
        )

    def get_next_work(
        self,
        cu_seqlens: cute.Tensor,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ) -> WorkDesc:
        """Which (sequence, head, value slice) this block owns."""
        bx, _, _ = cute.arch.block_idx()
        # Value slice varies fastest: blocks sharing Q/K are adjacent, for L2 reuse.
        v_slice_idx = cutlass.Int32(0)
        if cutlass.const_expr(self.num_v_splits > 1):
            v_slice_idx = bx % cutlass.Int32(self.num_v_splits)
            bx = bx // cutlass.Int32(self.num_v_splits)
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
            v_slice_idx=v_slice_idx,
        )

    # ─── Ordered 2-WG math barriers ───────────────────────────────────────────
    # Translates flat::OrderedNamedBarriers<UseReservedNB, NB0, NB1>.
    # wg_idx: MathWarpGroupRole.KK or MathWarpGroupRole.QK.

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        """Pre-arrive at WG0's barrier; a no-op with one warp group."""
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS == 1):
            return
        if wg_idx == MathWarpGroupRole.QK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=LOAD_THREADS
            )

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32):
        """Arrive+wait on this WG's barrier; with one WG, a block barrier for sQK."""
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS == 1):
            cute.arch.barrier()
            return
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=LOAD_THREADS
            )
        else:
            cute.arch.barrier(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=LOAD_THREADS
            )

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        """Arrive at the other WG's barrier to unblock it."""
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS == 1):
            return
        if wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=LOAD_THREADS
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=LOAD_THREADS
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
        # No stmatrix on sm_80; plain R2S on the C layout writes the same addresses.
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.inverse_dtype)
        tiled_store = cute.make_tiled_copy_C(r2s_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(kk_thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_inv = cute.make_fragment_like(tKKrKK, self.inverse_dtype)
        tKKrKK_cv = thr_store.retile(tKKrKK_inv)
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            tKKrKK_inv[i] = self.inverse_dtype(tKKrKK[i])
        cute.copy(tiled_store, tKKrKK_cv, tKKsKK)

        cute.arch.barrier(
            barrier_id=NamedBarrier.KK_SYNC, number_of_threads=THREADS_PER_WARP_GROUP
        )
        CollectiveInverse(has_stmatrix=False).run(sKK_inv, NamedBarrier.KK_SYNC)

        if cutlass.const_expr(self.needs_beta or self.dtype != self.inverse_dtype):
            cute.arch.barrier(
                barrier_id=NamedBarrier.KK_SYNC,
                number_of_threads=THREADS_PER_WARP_GROUP,
            )
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

    # ─── qk_store ─────────────────────────────────────────────────────────────

    @cute.jit
    def qk_store(
        self,
        tQKrQK: cute.Tensor,
        sQK: cute.Tensor,
        qk_tiled_mma,
        qk_thread_idx: cutlass.Int32,
    ):
        """Convert the QK accumulator to Element and publish it to shared."""
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.dtype)
        qk_tiled_copy = cute.make_tiled_copy_C(r2s_atom, qk_tiled_mma)
        qk_thr_copy = qk_tiled_copy.get_slice(qk_thread_idx)
        tQKsQK = qk_thr_copy.partition_D(sQK)
        tQKrQK_cvt = cute.make_fragment_like(tQKrQK, self.dtype)
        tQKrQK_cvt_cv = qk_thr_copy.retile(tQKrQK_cvt)
        for i in cutlass.range_constexpr(cute.size(tQKrQK)):
            tQKrQK_cvt[i] = self.dtype(tQKrQK[i])
        cute.copy(qk_tiled_copy, tQKrQK_cvt_cv, tQKsQK)

    # ─── o_store ──────────────────────────────────────────────────────────────

    @cute.jit
    def o_store(
        self,
        tOrO: cute.Tensor,
        tOcO: cute.Tensor,
        sO: cute.Tensor,
    ):
        """Write O to sO by its (d, token) coordinates: sm_80 has no stmatrix.trans."""
        for i in cutlass.range_constexpr(cute.size(tOrO)):
            d_i, tok = tOcO[i]
            sO[d_i, tok] = self.dtype(tOrO[i])

    # ─── cp.async load helpers ───────────────────────────────────────────────

    @cute.jit
    def _rep_slice(self, t: cute.Tensor, rep, d_is_mode1: cutlass.Constexpr):
        """Pick one repeat of a staged tile, whichever axis `d` sits on."""
        if cutlass.const_expr(d_is_mode1):
            return t[None, rep, 0]
        return t[None, 0, rep]

    @cute.jit
    def _aligned(self, t: cute.Tensor):
        """Restate 16 B alignment after slicing; `_check_load_alignment` checks it."""
        return cute.make_tensor(t.iterator.align(_LOAD_ALIGN_BYTES), t.layout)

    def load_tv_shape_dn(self, d: int):
        """Thread-value shape for a (d, token) tile; a split V is narrower than K."""
        lanes_along_d = d // self.elems_per_lane
        return (
            ((lanes_along_d, LOAD_THREADS // lanes_along_d), (1, lanes_along_d)),
            (self.elems_per_lane, 1),
        )

    @cute.jit
    def _copy_tile(
        self,
        gTile: cute.Tensor,
        sTile: cute.Tensor,
        rows: int,
        tid: cutlass.Int32,
        row_limit: cutlass.Int32,
        d_is_mode1: cutlass.Constexpr,
        d: cutlass.Constexpr = None,
        row_first=None,
    ):
        """Issue one tile with cp.async, zero-filling rows past ``row_limit``
        (and, when ``row_first`` is given, rows before it)."""
        # The tile's own d: a split V is narrower than Q and K.
        tile_d = cutlass.const_expr(self.D if d is None else d)
        if cutlass.const_expr(d_is_mode1):
            tv_shape, val_shape = self.q_load_tv_shape, self.q_load_val_shape
        else:
            tv_shape, val_shape = self.load_tv_shape_dn(tile_d)
        tv_layout = cute.make_layout(tv_shape[0], stride=tv_shape[1])
        val_layout = cute.make_layout(val_shape)
        load_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128
        )
        tiled_copy = cute.make_tiled_copy_tv(load_atom, tv_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tid)
        tSrc = thr_copy.partition_S(gTile)
        tDst = thr_copy.partition_D(sTile)
        # partition_S/D give (atom value, rest_m, rest_n); only the row axis repeats.
        lanes_along_d = cutlass.const_expr(tile_d // self.elems_per_lane)
        rows_at_a_time = cutlass.Int32(LOAD_THREADS) // cutlass.Int32(lanes_along_d)
        lane_row = tid // cutlass.Int32(lanes_along_d)
        rep_mode = 1 if cutlass.const_expr(d_is_mode1) else 2
        for rep in cutlass.range_constexpr(cute.size(tSrc, mode=[rep_mode])):
            row = cutlass.Int32(rep) * rows_at_a_time + lane_row
            # Zero, not skip: the math reads the whole tile and masking needs zeros.
            if cutlass.const_expr(row_first is None):
                if row < row_limit:
                    cute.copy(
                        load_atom,
                        self._aligned(self._rep_slice(tSrc, rep, d_is_mode1)),
                        self._rep_slice(tDst, rep, d_is_mode1),
                    )
                else:
                    self._rep_slice(tDst, rep, d_is_mode1).fill(self.dtype(0.0))
            else:
                # Rows before the sequence (virtual left padding) are never
                # read: they may lie before the tensor.
                if row < row_limit and row >= row_first:
                    cute.copy(
                        load_atom,
                        self._aligned(self._rep_slice(tSrc, rep, d_is_mode1)),
                        self._rep_slice(tDst, rep, d_is_mode1),
                    )
                else:
                    self._rep_slice(tDst, rep, d_is_mode1).fill(self.dtype(0.0))

    @cute.jit
    def load_qkv_cpasync(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        blk: cutlass.Int32,
        tok_start,
        tok_end: cutlass.Int32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        v_row_offset: cutlass.Int32,
        tid: cutlass.Int32,
        tok_begin=None,
    ):
        """Issue this block's share of the Q, K and V tiles on `cp.async`.

        With ``tok_begin`` (virtual left padding) rows before it are zeroed.
        """
        blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
        # Rows of this tile the sequence owns.
        rows_live = tok_end - blk_tok
        row_first = None
        if cutlass.const_expr(tok_begin is not None):
            row_first = tok_begin - blk_tok

        # K first, then Q, then V -- the order the math side waits in.
        sK = sK_DS[None, None, k_producer_state.index]
        mK = cute.domain_offset(
            (cutlass.Int32(0), blk_tok), gK_full[None, None, k_head_idx]
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        k_pipeline.producer_acquire(k_producer_state)
        self._copy_tile(gK, sK, self.BLK_KV, tid, rows_live, False, row_first=row_first)
        cute.arch.cp_async_commit_group()
        k_pipeline.producer_commit(k_producer_state)
        k_producer_state.advance()

        sQ = sQ_SD[None, None, q_producer_state.index]
        mQ = cute.domain_offset(
            (blk_tok, cutlass.Int32(0)), gQ_full[None, None, q_head_idx]
        )
        gQ = cute.zipped_divide(mQ, (self.BLK_Q, self.D))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        q_pipeline.producer_acquire(q_producer_state)
        self._copy_tile(gQ, sQ, self.BLK_Q, tid, rows_live, True, row_first=row_first)
        cute.arch.cp_async_commit_group()
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        sV = sV_DS[None, None, v_producer_state.index]
        # A block reads only its D_v rows of V.
        mV = cute.domain_offset(
            (v_row_offset, blk_tok), gV_full[None, None, v_head_idx]
        )
        gV = cute.zipped_divide(mV, (self.D_v, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        v_pipeline.producer_acquire(v_producer_state)
        self._copy_tile(
            gV, sV, self.BLK_KV, tid, rows_live, False, d=self.D_v, row_first=row_first
        )
        cute.arch.cp_async_commit_group()
        v_pipeline.producer_commit(v_producer_state)
        v_producer_state.advance()
        return q_producer_state, k_producer_state, v_producer_state

    @cute.jit
    def issue_block_loads(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        alpha_pipeline,
        alpha_producer_state,
        beta_pipeline,
        beta_producer_state,
        blk: cutlass.Int32,
        t_block_start: cutlass.Int32,
        tok_start,
        tok_end: cutlass.Int32,
        scale: cutlass.Float32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        v_row_offset: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        tid: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tok_begin=None,
    ):
        """Fetch one block's Q/K/V (cp.async) and alpha/beta (plain loads).

        ``tok_begin``: the sequence's first row when the block may start
        before it (virtual left padding); earlier rows are identity rows.
        """
        (
            q_producer_state,
            k_producer_state,
            v_producer_state,
        ) = self.load_qkv_cpasync(
            sQ_SD,
            sK_DS,
            sV_DS,
            gQ_full,
            gK_full,
            gV_full,
            q_pipeline,
            q_producer_state,
            k_pipeline,
            k_producer_state,
            v_pipeline,
            v_producer_state,
            blk,
            tok_start,
            tok_end,
            q_head_idx,
            k_head_idx,
            v_head_idx,
            v_row_offset,
            tid,
            tok_begin=tok_begin,
        )

        # One warp each for alpha's in-place scan (a second would race it) and
        # beta. Neither branch has a barrier, so restricting them is safe.
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.producer_acquire(alpha_producer_state)
            if warp_idx == cutlass.Int32(0):
                blk_tok = tok_start + blk * cutlass.Int32(self.BLK_Q)
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
            alpha_pipeline.producer_commit(alpha_producer_state)
            alpha_producer_state.advance()

        # The context-parallel subclass overrides issue_beta_stage without a
        # tok_begin parameter, so pass it only when it is set.
        if cutlass.const_expr(tok_begin is None):
            beta_producer_state = self.issue_beta_stage(
                sBeta,
                g_beta,
                beta_pipeline,
                beta_producer_state,
                blk,
                t_block_start,
                tok_start,
                tok_end,
                sab_head_idx,
                num_sab_heads,
                tid,
                warp_idx,
            )
        else:
            beta_producer_state = self.issue_beta_stage(
                sBeta,
                g_beta,
                beta_pipeline,
                beta_producer_state,
                blk,
                t_block_start,
                tok_start,
                tok_end,
                sab_head_idx,
                num_sab_heads,
                tid,
                warp_idx,
                tok_begin=tok_begin,
            )

        return (
            q_producer_state,
            k_producer_state,
            v_producer_state,
            alpha_producer_state,
            beta_producer_state,
        )

    @cute.jit
    def issue_beta_stage(
        self,
        sBeta: cute.Tensor,
        g_beta: cute.Tensor,
        beta_pipeline,
        beta_producer_state,
        blk: cutlass.Int32,
        t_block_start: cutlass.Int32,
        tok_start,
        tok_end: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        tid: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tok_begin=None,
    ):
        """Load beta into stage five; `CPDeltaRulePrefillSm80` overrides it with T."""
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.producer_acquire(beta_producer_state)
            if warp_idx == cutlass.Int32(1):
                blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
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
            beta_pipeline.producer_commit(beta_producer_state)
            beta_producer_state.advance()
        return beta_producer_state

    def _k_major_atom_kind(self, elems: int):
        """The widest K-major swizzle atom a row of `elems` can carry."""
        # `tile_to_shape` does not narrow an atom wider than the tile.
        bytes_per_row = elems * self.dtype.width // 8
        if bytes_per_row >= 128:
            return warpgroup.SmemLayoutAtomKind.K_SW128
        if bytes_per_row >= 64:
            return warpgroup.SmemLayoutAtomKind.K_SW64
        if bytes_per_row >= 32:
            return warpgroup.SmemLayoutAtomKind.K_SW32
        return warpgroup.SmemLayoutAtomKind.K_INTER

    # ─── v32 handoff ──────────────────────────────────────────────────────────

    @cute.jit
    def _handoff_put(self, acc, producer_mma, thread_idx, sBuf):
        """Store an accumulator, as Element, to the buffer its consumer reads A from."""
        r2s = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.dtype)
        tiled = cute.make_tiled_copy_C(r2s, producer_mma)
        thr = tiled.get_slice(thread_idx)
        frag = cute.make_fragment_like(acc, self.dtype)
        for i in cutlass.range_constexpr(cute.size(acc)):
            frag[i] = self.dtype(acc[i])
        cute.copy(tiled, thr.retile(frag), thr.partition_D(sBuf))

    @cute.jit
    def _handoff_get_a(self, consumer_mma, thread_idx, sBuf, ldsm_atom):
        """Read an A operand for the consuming MMA out of a handoff buffer."""
        thr_mma = consumer_mma.get_slice(thread_idx)
        frag = thr_mma.make_fragment_A(thr_mma.partition_A(sBuf))
        tiled = cute.make_tiled_copy_A(ldsm_atom, consumer_mma)
        thr = tiled.get_slice(thread_idx)
        cute.copy(tiled, thr.partition_S(sBuf), thr.retile(frag))
        return frag

    # ─── v32 handoff layouts ──────────────────────────────────────────────────

    def _handoff_layouts(self, atom):
        """(state, block) layouts of the v32 handoff buffers: (D_v, K), K contiguous."""
        state = cute.coalesce(
            cute.tile_to_shape(atom, (self.D_v, self.D), order=(0, 1)),
            target_profile=(1, 1),
        )
        blk = cute.coalesce(
            cute.tile_to_shape(atom, (self.D_v, self.BLK_KV), order=(0, 1)),
            target_profile=(1, 1),
        )
        return state, blk

    # ─── kv_load / kv_store ───────────────────────────────────────────────────

    @cute.jit
    def kv_load(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        """Read a (k, v) state tile into the accumulator by coordinate."""
        c_kv = cute.make_identity_tensor((self.D_v, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            tKVrKV[i] = gKV[k_idx, v_idx].to(self.acc_dtype)

    @cute.jit
    def kv_load_if(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
        pred,
    ):
        """``kv_load`` of each element under ``pred`` (else it is kept)."""
        c_kv = cute.make_identity_tensor((self.D_v, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            if pred:
                v_idx, k_idx = tKVcKV[i]
                tKVrKV[i] = gKV[k_idx, v_idx].to(self.acc_dtype)

    @cute.jit
    def kv_store(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D_v, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            gKV[k_idx, v_idx] = tKVrKV[i].to(gKV.element_type)

    @property
    def slot_ckpt(self) -> bool:
        """Checkpoints at absolute positions or into pool rows (``prefix_lens``
        or ``checkpoint_state_indices``), scheduled in shared memory."""
        return self.use_prefix_lens or self.use_checkpoint_state_indices

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
        v_row_offset: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        s_ckpt: cute.Tensor = None,
    ):
        """Checkpoint when the block ends on a period multiple; the tail is not one.

        With ``prefix_lens`` or ``checkpoint_state_indices`` (``slot_ckpt``)
        ``stage_checkpoint`` has already decided for this block: its slot in
        ``s_ckpt`` holds the pool row (``checkpoint_state_indices``) or the
        packed entry of the checkpoint due at ``block_end``, or -1. No key,
        entry or row stays in a register across the block loop (the sm_80
        math loop already uses all 255 registers; one more live value made
        ptxas spill in the loop), and the store path does no division.
        """
        if cutlass.const_expr(self.needs_checkpointing and not self.slot_ckpt):
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
                    (self.D, self.D, cutlass.Int64(num_sab_heads), total_checkpoints),
                    order=(0, 1, 2, 3),
                )
                mCheckpoint = cute.make_tensor(
                    g_state_checkpoints.iterator, checkpoint_layout
                )
                gCheckpointKV = cute.domain_offset(
                    (cutlass.Int32(0), v_row_offset),
                    mCheckpoint[None, None, o_head_idx, checkpoint_idx],
                )
                self.kv_store(tKVrKV, gCheckpointKV, kv_thr_mma)
        elif cutlass.const_expr(self.needs_checkpointing):
            target = s_ckpt[self._ckpt_slot(block_end)]
            if target >= cutlass.Int32(0):
                if cutlass.const_expr(self.use_checkpoint_state_indices):
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
                        None,
                        target,
                        v_row_offset=v_row_offset,
                    )
                else:
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
                    gCheckpointKV = cute.domain_offset(
                        (cutlass.Int32(0), v_row_offset),
                        mCheckpoint[None, None, o_head_idx, target],
                    )
                    self.kv_store(tKVrKV, gCheckpointKV, kv_thr_mma)

    def _ckpt_slot(self, block_end: cutlass.Int32):
        """``s_ckpt`` slot of the block ending at ``block_end``: 2 + block parity."""
        return cutlass.Int32(2) + (
            block_end // cutlass.Int32(self.BLK_KV)
        ) % cutlass.Int32(2)

    @cute.jit
    def init_checkpoint_schedule(
        self,
        s_ckpt: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        seq_idx: cutlass.Int32,
        ckpt_origin,
        tidx: cutlass.Int32,
    ):
        """Thread 0 writes the work item's first due block end (``s_ckpt[0]``)
        and its entry (``s_ckpt[1]``) before the barrier that starts the math.

        A checkpoint is due where the absolute position is a multiple of N
        (``checkpoint_every_n_tokens``, a multiple of the 64-token block):
        at block end N without ``prefix_lens``, at N - origin % N with it
        (``origin``: absolute position of the first, 64-aligned row). The
        first due checkpoint is entry ``checkpoint_cu_starts[seq]``.
        """
        if cutlass.const_expr(self.needs_checkpointing and self.slot_ckpt):
            if tidx == cutlass.Int32(0):
                due = checkpoint_every_n_tokens
                if cutlass.const_expr(self.use_prefix_lens):
                    due = (
                        checkpoint_every_n_tokens
                        - ckpt_origin % checkpoint_every_n_tokens
                    )
                s_ckpt[0] = due
                s_ckpt[1] = cutlass.Int32(checkpoint_cu_starts[seq_idx])

    @cute.jit
    def stage_checkpoint(
        self,
        g_checkpoint_rows,
        s_ckpt: cute.Tensor,
        checkpoint_every_n_tokens: cutlass.Int32,
        block_end: cutlass.Int32,
        seq_len: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        """Thread 0 decides whether a checkpoint is due at ``block_end`` and
        writes its pool row (``checkpoint_state_indices``) or packed entry to
        the block's ``s_ckpt`` slot, or -1, for ``maybe_store_checkpoint``.

        Called after the block's tile loads are issued: the row load overlaps
        theirs, and the barrier after the first tile wait publishes the slot.
        Loading the row at the store put one global-load latency on every due
        block (every block with a 64-token period). The due test compares
        with the next due block end, so no thread divides by N in the loop.
        Two slots by block parity: the previous block's store may still read
        the other one.
        """
        if cutlass.const_expr(self.needs_checkpointing and self.slot_ckpt):
            if tidx == cutlass.Int32(0):
                target = cutlass.Int32(-1)
                if block_end == s_ckpt[0] and block_end <= seq_len:
                    target = s_ckpt[1]
                    s_ckpt[0] = block_end + checkpoint_every_n_tokens
                    s_ckpt[1] = target + cutlass.Int32(1)
                    if cutlass.const_expr(self.use_checkpoint_state_indices):
                        target = cutlass.Int32(g_checkpoint_rows[target])
                s_ckpt[self._ckpt_slot(block_end)] = target

    # ─── compute_loop_body ───────────────────────────────────────────────────
    # Translates the C++ compute_loop_body lambda captured inside compute().
    # Called by every math thread for one block iteration.

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
        # v32 only: (state, u, new V, new V decayed). Last and defaulted: the
        # chunk-parallel kernels call this method positionally.
        sHandoff=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # No load/store warp group, so math threads start at 0, not at 128.
        thread_idx = tidx
        kk_thread_idx = thread_idx % cutlass.Int32(128)
        qk_thread_idx = thread_idx % cutlass.Int32(128)

        # ── TiledMMAs ─────────────────────────────────────────────────────────
        blk_q = cute.size(sQ_SD, mode=[0])
        blk_kv = cute.size(sK_SD, mode=[0])
        d = cute.size(sQ_SD, mode=[1])
        # V rows this block owns: the M mode of every V-dependent GEMM.
        d_v = cute.size(sV_DS, mode=[0])
        tile_shape_qk = (blk_q, blk_kv, d)
        tile_shape_kk = tile_shape_qk
        tile_shape_o1 = (d_v, blk_q, d)
        tile_shape_o2 = (d_v, blk_q, blk_kv)
        tile_shape_sk = (d_v, blk_kv, d)
        tile_shape_newv = (d_v, blk_kv, blk_kv)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        v_stage = v_consumer_state.index
        o_stage = cutlass.Int32(0)
        alpha_stage = alpha_consumer_state.index
        beta_stage = beta_consumer_state.index

        mma_atom = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))

        # QK/KK: up to 4 warps × 16M = 64M, one warp group's worth
        qk_tiled_mma = cute.make_tiled_mma(
            mma_atom,
            cute.make_layout((min(4, NUM_MMA_WARPS), 1, 1)),
            permutation_mnk=tile_shape_qk,
        )
        kk_tiled_mma = cute.make_tiled_mma(
            mma_atom,
            cute.make_layout((min(4, NUM_MMA_WARPS), 1, 1)),
            permutation_mnk=tile_shape_kk,
        )

        # O1/O2/SK/NewV: M-only warps, so each accumulator is the next A operand.
        # v32 splits warps 2x2 in M and N and hands accumulators through smem.
        if cutlass.const_expr(self.v32):
            v_warps = cute.make_layout((2, 2, 1))
        else:
            v_warps = cute.make_layout((NUM_MMA_WARPS, 1, 1))
        o1_tiled_mma = cute.make_tiled_mma(
            mma_atom, v_warps, permutation_mnk=tile_shape_o1
        )
        o2_tiled_mma = cute.make_tiled_mma(
            mma_atom, v_warps, permutation_mnk=tile_shape_o2
        )
        sk_tiled_mma = cute.make_tiled_mma(
            mma_atom, v_warps, permutation_mnk=tile_shape_sk
        )
        newv_tiled_mma = cute.make_tiled_mma(
            mma_atom, v_warps, permutation_mnk=tile_shape_newv
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

        # ── Coordinate tensors for masking / alpha/beta indexing ──────────────
        cMqk = cute.make_identity_tensor((blk_q, blk_kv))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        cMkk = cMqk  # same shape (BlkKV == BlkQ == 64)
        tKKcMkk = kk_thr_mma.partition_C(cMkk)
        # No per-stage release: the barrier CollectiveStoreSm80 takes after
        # writing O orders buffer reuse before the next issue_block_loads.
        cO = cute.make_identity_tensor((d_v, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((d_v, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((d_v, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)
        # v32 decays new V in its producer's C layout, before publishing it.
        tNewVcNewV = newv_thr_mma.partition_C(cV)

        # ── KK GEMM (WG0 only) ────────────────────────────────────────────────
        # Prefetch is zero, so this wait drains and publishes all five inputs.
        k_pipeline.consumer_wait(k_consumer_state)
        # Match the C++ reject-non-role-first shape; ptxas keeps BRA.U around
        # the role body instead of predicating the HMMA/LDSM/STSM sequence.
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS > 1) and wg_idx != (
            MathWarpGroupRole.KK
        ):
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
            beta_consumer_state.advance()
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS == 1):
            # One warp group runs both bodies in turn; every warp writes both buffers.
            cute.arch.barrier()

        # ── QK GEMM (WG1 only) ────────────────────────────────────────────────
        q_pipeline.consumer_wait(q_consumer_state)
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS > 1) and wg_idx != (
            MathWarpGroupRole.QK
        ):
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
            self.qk_store(tQKrQK, sQK, qk_tiled_mma, qk_thread_idx)
        if cutlass.const_expr(NUM_MMA_WARP_GROUPS == 1):
            cute.arch.barrier()

        # ── The state handoff, read by both O1 and S@K ───────────────────────
        # No barrier first: the previous block's handoff barriers fence its reads.
        if cutlass.const_expr(self.v32 and not is_first_block):
            self._handoff_put(tKVrKV, kv_tiled_mma, thread_idx, sHandoff[0])
            cute.arch.barrier()

        # ── O1: KV_state @ Q (both WGs, skip on first block) ─────────────────
        tOrO = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_C((d_v, blk_q)), self.acc_dtype
        )
        tOrO.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            cute.copy(o1_tiled_copy_B, tOsQ[None, None, None, q_stage], tOrQ_cv)
            if cutlass.const_expr(self.v32):
                tOrKV = self._handoff_get_a(
                    o1_tiled_mma, thread_idx, sHandoff[0], ldsm_n4
                )
            else:
                tOrKV = SM80.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
            cute.gemm(o1_tiled_mma, tOrO, tOrKV, tOrQ, tOrO)
            self.o1_epi(tOrO, tOcO, sAlpha, alpha_stage, scale)
        q_consumer_state.advance()

        # ── SK: KV_state @ K^T (result negated below via V - SK) ─────────────
        tSKrSK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_C((d_v, blk_kv)), self.acc_dtype
        )
        tSKrSK.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            if cutlass.const_expr(self.v32):
                tSKrS = self._handoff_get_a(
                    sk_tiled_mma, thread_idx, sHandoff[0], ldsm_n4
                )
            else:
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
        if cutlass.const_expr(self.v32):
            self._handoff_put(tSKrV, sk_tiled_mma, thread_idx, sHandoff[1])
            cute.arch.barrier()
            tNewVrA = self._handoff_get_a(
                newv_tiled_mma, thread_idx, sHandoff[1], ldsm_n4
            )
        else:
            tNewVrA = SM80.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = cute.make_rmem_tensor(
            newv_thr_mma.partition_shape_C((d_v, blk_kv)), self.acc_dtype
        )
        self._math_order_wait(wg_idx)
        cute.copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv)
        tNewVrC.fill(self.acc_dtype(0.0))
        cute.gemm(newv_tiled_mma, tNewVrC, tNewVrA, tNewVrB, tNewVrC)
        self._math_order_notify(wg_idx)
        v_consumer_state.advance()

        # ── O2 = O1 + NewV @ QK  (ordered: WG0 first) ────────────────────────
        # v32 publishes new V twice, one barrier: raw for O2, decayed for the state.
        if cutlass.const_expr(self.v32):
            tNewVrE = cute.make_fragment_like(tNewVrC, self.dtype)
            for i in cutlass.range_constexpr(cute.size(tNewVrC)):
                tNewVrE[i] = self.dtype(tNewVrC[i])
            self._handoff_put(tNewVrE, newv_tiled_mma, thread_idx, sHandoff[2])
            tNewVrD = cute.make_fragment_like(tNewVrE)
            for i in cutlass.range_constexpr(cute.size(tNewVrE)):
                tNewVrD[i] = tNewVrE[i]
            self.kv_decay_v(tNewVrD, tNewVcNewV, sAlpha, alpha_stage, is_final_block, B)
            self._handoff_put(tNewVrD, newv_tiled_mma, thread_idx, sHandoff[3])
            cute.arch.barrier()
            tOrNewV = self._handoff_get_a(
                o2_tiled_mma, thread_idx, sHandoff[2], ldsm_n4
            )
        else:
            tOrNewV = SM80.make_acc_into_op(tNewVrC, o2_tiled_mma, self.dtype)
        self._math_order_wait(wg_idx)
        cute.copy(o2_tiled_copy_B, tOsQK, tOrQK_cv)
        cute.gemm(o2_tiled_mma, tOrO, tOrNewV, tOrQK, tOrO)
        self._math_order_notify(wg_idx)

        # ── O store to smem ───────────────────────────────────────────────────
        # No pipeline: the caller's store brackets the copy-out with barriers.
        self.o_store(tOrO, tOcO, sO[None, None, o_stage])

        # ── KV state update ───────────────────────────────────────────────────
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[B - cutlass.Int32(1), AlphaProcessor.CUMPROD, alpha_stage]
            )

        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        if cutlass.const_expr(self.v32):
            tKVrNewV = self._handoff_get_a(
                kv_tiled_mma, thread_idx, sHandoff[3], ldsm_n4
            )
        else:
            self.kv_decay_v(tOrNewV, tKVcV, sAlpha, alpha_stage, is_final_block, B)
            tKVrNewV = tOrNewV

        # KV += NewV @ K
        cute.copy(kv_tiled_copy_B, tKVsK[None, None, None, k_stage], tKVrK_cv)
        cute.gemm(kv_tiled_mma, tKVrKV, tKVrNewV, tKVrK, tKVrKV)
        k_consumer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            alpha_consumer_state,
            beta_consumer_state,
        )

    # ─── Block entry point ───────────────────────────────────────────────────
    # One path for the whole block: the DSL emits no mbarrier for sm_80 to split roles.

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
        sHandoff,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        gO_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
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
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tok_end: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
        g_output_state_indices: cute.Tensor = None,
        g_checkpoint_rows: cute.Tensor = None,
        s_ckpt: cute.Tensor = None,
        tok_begin=None,
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
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_warps = cute.make_layout((NUM_MMA_WARPS, 1, 1))
        if cutlass.const_expr(self.v32):
            kv_warps = cute.make_layout((2, 2, 1))
        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            kv_warps,
            permutation_mnk=(self.D_v, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.D_v, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        packed_state_layout = cute.make_ordered_layout(
            (self.D, self.D, cutlass.Int64(num_sab_heads), num_seqs), order=(0, 1, 2, 3)
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
        # (k, v) of this block's V band; bands are disjoint across blocks.
        v_row_offset = work_desc.v_row_offset(self.D_v)
        # With output_state_indices the final row is read after the block loop
        # (as are the checkpoint rows), so it holds no register during it.
        gStateKV = None
        if cutlass.const_expr(not self.use_output_state_indices):
            gStateKV = cute.domain_offset(
                (cutlass.Int32(0), v_row_offset),
                mState[None, None, o_head_idx, state_idx],
            )
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
            gInitKV = cute.domain_offset(
                (cutlass.Int32(0), v_row_offset),
                mInitState[None, None, o_head_idx, state_idx],
            )
            if cutlass.const_expr(self.slot_mode):
                # A negative read row starts from the zero state. Each thread
                # reads its state elements here, before it writes any
                # checkpoint or final state, so read and write rows may alias.
                self.kv_load_if(
                    tKVrKV, gInitKV, kv_thr_mma, state_idx >= cutlass.Int32(0)
                )
            else:
                self.kv_load(tKVrKV, gInitKV, kv_thr_mma)

        if cutlass.const_expr(
            self.needs_init_state and (self.slot_mode or self.needs_checkpointing)
        ):
            # One loop over every block: one code copy of the masked block body
            # (first and last block) and one of the full one. The schedule
            # below instantiates the masked body twice and the loads, O store
            # and checkpoint store three times; a work item of a few blocks
            # runs mostly that straight-line code, and the larger slot and
            # checkpoint kernels paid for it in instruction fetch. Later blocks
            # take tok_begin too: no row of theirs lies before it, so nothing
            # changes.
            for blk in cutlass.range(
                cutlass.Int32(0), num_blocks, cutlass.Int32(1), unroll=1
            ):
                blk_end = (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV)
                (
                    q_producer_state,
                    k_producer_state,
                    v_producer_state,
                    alpha_producer_state,
                    beta_producer_state,
                ) = self.issue_block_loads(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    sAlpha,
                    sBeta,
                    gQ_full,
                    gK_full,
                    gV_full,
                    g_alpha,
                    g_beta,
                    q_pipeline,
                    q_producer_state,
                    k_pipeline,
                    k_producer_state,
                    v_pipeline,
                    v_producer_state,
                    alpha_pipeline,
                    alpha_producer_state,
                    beta_pipeline,
                    beta_producer_state,
                    blk,
                    cutlass.Int32(0),
                    work_desc.tok_offset,
                    tok_end,
                    scale,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                    work_desc.v_row_offset(self.D_v),
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                    tidx,
                    warp_idx,
                    tok_begin=tok_begin,
                )
                self.stage_checkpoint(
                    g_checkpoint_rows,
                    s_ckpt,
                    checkpoint_every_n_tokens,
                    blk_end,
                    work_desc.seq_len,
                    tidx,
                )
                if blk == cutlass.Int32(0) or blk_end >= work_desc.seq_len:
                    # First or last block: masked to its live rows.
                    blk_B = work_desc.seq_len - blk * cutlass.Int32(self.BLK_KV)
                    if blk_B > cutlass.Int32(self.BLK_KV):
                        blk_B = cutlass.Int32(self.BLK_KV)
                    (
                        q_consumer_state,
                        k_consumer_state,
                        v_consumer_state,
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
                        alpha_pipeline,
                        alpha_consumer_state,
                        beta_pipeline,
                        beta_consumer_state,
                        False,
                        True,
                        blk_B,
                        tKVrKV,
                        scale,
                        wg_idx,
                        sHandoff=sHandoff,
                    )
                else:
                    (
                        q_consumer_state,
                        k_consumer_state,
                        v_consumer_state,
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
                        sHandoff=sHandoff,
                    )
                self.o_writer.run(
                    sO[None, None, 0],
                    gO_full,
                    work_desc,
                    blk,
                    num_q_heads,
                    num_v_heads,
                    tidx,
                    tok_begin=tok_begin,
                )
                self.maybe_store_checkpoint(
                    tKVrKV,
                    g_state_checkpoints,
                    checkpoint_cu_starts,
                    checkpoint_every_n_tokens,
                    kv_thr_mma,
                    work_desc.seq_idx,
                    o_head_idx,
                    v_row_offset,
                    num_sab_heads,
                    total_checkpoints,
                    blk_end,
                    work_desc.seq_len,
                    s_ckpt=s_ckpt,
                )
        else:
            first_B = work_desc.seq_len
            if first_B > cutlass.Int32(self.BLK_KV):
                first_B = cutlass.Int32(self.BLK_KV)
            if cutlass.const_expr(self.needs_init_state):
                (
                    q_producer_state,
                    k_producer_state,
                    v_producer_state,
                    alpha_producer_state,
                    beta_producer_state,
                ) = self.issue_block_loads(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    sAlpha,
                    sBeta,
                    gQ_full,
                    gK_full,
                    gV_full,
                    g_alpha,
                    g_beta,
                    q_pipeline,
                    q_producer_state,
                    k_pipeline,
                    k_producer_state,
                    v_pipeline,
                    v_producer_state,
                    alpha_pipeline,
                    alpha_producer_state,
                    beta_pipeline,
                    beta_producer_state,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    work_desc.tok_offset,
                    tok_end,
                    scale,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                    work_desc.v_row_offset(self.D_v),
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                    tidx,
                    warp_idx,
                    tok_begin=tok_begin,
                )
                self.stage_checkpoint(
                    g_checkpoint_rows,
                    s_ckpt,
                    checkpoint_every_n_tokens,
                    cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    tidx,
                )
                (
                    q_consumer_state,
                    k_consumer_state,
                    v_consumer_state,
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
                    sHandoff=sHandoff,
                )
                self.o_writer.run(
                    sO[None, None, 0],
                    gO_full,
                    work_desc,
                    cutlass.Int32(0),
                    num_q_heads,
                    num_v_heads,
                    tidx,
                    tok_begin=tok_begin,
                )
            else:
                (
                    q_producer_state,
                    k_producer_state,
                    v_producer_state,
                    alpha_producer_state,
                    beta_producer_state,
                ) = self.issue_block_loads(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    sAlpha,
                    sBeta,
                    gQ_full,
                    gK_full,
                    gV_full,
                    g_alpha,
                    g_beta,
                    q_pipeline,
                    q_producer_state,
                    k_pipeline,
                    k_producer_state,
                    v_pipeline,
                    v_producer_state,
                    alpha_pipeline,
                    alpha_producer_state,
                    beta_pipeline,
                    beta_producer_state,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    work_desc.tok_offset,
                    tok_end,
                    scale,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                    work_desc.v_row_offset(self.D_v),
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                    tidx,
                    warp_idx,
                    tok_begin=tok_begin,
                )
                self.stage_checkpoint(
                    g_checkpoint_rows,
                    s_ckpt,
                    checkpoint_every_n_tokens,
                    cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    tidx,
                )
                (
                    q_consumer_state,
                    k_consumer_state,
                    v_consumer_state,
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
                    sHandoff=sHandoff,
                )
                self.o_writer.run(
                    sO[None, None, 0],
                    gO_full,
                    work_desc,
                    cutlass.Int32(0),
                    num_q_heads,
                    num_v_heads,
                    tidx,
                    tok_begin=tok_begin,
                )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                work_desc.seq_idx,
                o_head_idx,
                v_row_offset,
                num_sab_heads,
                total_checkpoints,
                cutlass.Int32(self.BLK_KV),
                work_desc.seq_len,
                s_ckpt=s_ckpt,
            )

            # Virtual left padding (fewer than BLK_KV rows) lies in block 0 only:
            # later blocks take no tok_begin, so it is not live across the loop.
            for blk in cutlass.range(
                cutlass.Int32(1),
                num_blocks - cutlass.Int32(1),
                cutlass.Int32(1),
                unroll=1,
            ):
                (
                    q_producer_state,
                    k_producer_state,
                    v_producer_state,
                    alpha_producer_state,
                    beta_producer_state,
                ) = self.issue_block_loads(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    sAlpha,
                    sBeta,
                    gQ_full,
                    gK_full,
                    gV_full,
                    g_alpha,
                    g_beta,
                    q_pipeline,
                    q_producer_state,
                    k_pipeline,
                    k_producer_state,
                    v_pipeline,
                    v_producer_state,
                    alpha_pipeline,
                    alpha_producer_state,
                    beta_pipeline,
                    beta_producer_state,
                    blk,
                    cutlass.Int32(0),
                    work_desc.tok_offset,
                    tok_end,
                    scale,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                    work_desc.v_row_offset(self.D_v),
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                    tidx,
                    warp_idx,
                )
                self.stage_checkpoint(
                    g_checkpoint_rows,
                    s_ckpt,
                    checkpoint_every_n_tokens,
                    (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    tidx,
                )
                (
                    q_consumer_state,
                    k_consumer_state,
                    v_consumer_state,
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
                    sHandoff=sHandoff,
                )
                self.o_writer.run(
                    sO[None, None, 0],
                    gO_full,
                    work_desc,
                    blk,
                    num_q_heads,
                    num_v_heads,
                    tidx,
                )
                self.maybe_store_checkpoint(
                    tKVrKV,
                    g_state_checkpoints,
                    checkpoint_cu_starts,
                    checkpoint_every_n_tokens,
                    kv_thr_mma,
                    work_desc.seq_idx,
                    o_head_idx,
                    v_row_offset,
                    num_sab_heads,
                    total_checkpoints,
                    (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    s_ckpt=s_ckpt,
                )

            if num_blocks != cutlass.Int32(1):
                last_blk = num_blocks - cutlass.Int32(1)
                last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
                (
                    q_producer_state,
                    k_producer_state,
                    v_producer_state,
                    alpha_producer_state,
                    beta_producer_state,
                ) = self.issue_block_loads(
                    sQ_SD,
                    sK_DS,
                    sV_DS,
                    sAlpha,
                    sBeta,
                    gQ_full,
                    gK_full,
                    gV_full,
                    g_alpha,
                    g_beta,
                    q_pipeline,
                    q_producer_state,
                    k_pipeline,
                    k_producer_state,
                    v_pipeline,
                    v_producer_state,
                    alpha_pipeline,
                    alpha_producer_state,
                    beta_pipeline,
                    beta_producer_state,
                    last_blk,
                    cutlass.Int32(0),
                    work_desc.tok_offset,
                    tok_end,
                    scale,
                    work_desc.q_head_idx(),
                    work_desc.k_head_idx(num_q_heads, num_v_heads),
                    work_desc.v_head_idx(),
                    work_desc.v_row_offset(self.D_v),
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                    tidx,
                    warp_idx,
                )
                self.stage_checkpoint(
                    g_checkpoint_rows,
                    s_ckpt,
                    checkpoint_every_n_tokens,
                    (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    tidx,
                )
                (
                    q_consumer_state,
                    k_consumer_state,
                    v_consumer_state,
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
                    sHandoff=sHandoff,
                )
                self.o_writer.run(
                    sO[None, None, 0],
                    gO_full,
                    work_desc,
                    last_blk,
                    num_q_heads,
                    num_v_heads,
                    tidx,
                )
                self.maybe_store_checkpoint(
                    tKVrKV,
                    g_state_checkpoints,
                    checkpoint_cu_starts,
                    checkpoint_every_n_tokens,
                    kv_thr_mma,
                    work_desc.seq_idx,
                    o_head_idx,
                    v_row_offset,
                    num_sab_heads,
                    total_checkpoints,
                    (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                    work_desc.seq_len,
                    s_ckpt=s_ckpt,
                )
        if cutlass.const_expr(self.use_output_state_indices):
            write_idx = cutlass.Int32(g_output_state_indices[work_desc.seq_idx])
            if write_idx >= cutlass.Int32(0):
                gFinalKV = cute.domain_offset(
                    (cutlass.Int32(0), v_row_offset),
                    mState[None, None, o_head_idx, write_idx],
                )
                self.kv_store(tKVrKV, gFinalKV, kv_thr_mma)
        elif cutlass.const_expr(self.slot_mode):
            if state_idx >= cutlass.Int32(0):
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
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_smem_layout_atom = warpgroup.make_smem_layout_atom(
            self._k_major_atom_kind(self.D_v), self.dtype
        )
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                v_smem_layout_atom,
                (self.BLK_KV, self.D_v, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D_v, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )

        # 128-bit cp.async lanes along contiguous d. Only shapes are host state:
        # a tiled copy rejects an atom or layout built outside the trace.
        elems_per_lane = 128 // self.dtype.width
        self.elems_per_lane = elems_per_lane
        # d is mode 1 of Q (token, d) and mode 0 of K and V (d, token).
        lanes_along_d = self.D // elems_per_lane
        rows_at_a_time = LOAD_THREADS // lanes_along_d
        self.lanes_along_d = lanes_along_d
        self.load_rows_at_a_time = rows_at_a_time
        # Q: lanes split d within a row, thread rows walk tokens.
        self.q_load_tv_shape = ((rows_at_a_time, lanes_along_d), (lanes_along_d, 1))
        self.q_load_val_shape = (1, elems_per_lane)
        # K and V use `load_tv_shape_dn` per width: a split V is narrower than K.

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

        # Plain ints: struct member lengths must be static; a composed cosize is not.
        handoff_state_elems = self.D_v * self.D
        handoff_blk_elems = self.D_v * self.BLK_KV

        @cute.struct
        class SharedStorage:
            # No mbarrier storage: the sm_80 pipelines count cp.async groups.
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

        # v32 appends its handoff buffers: a zero-length MemRange can't be allocated.
        # `type`, not the class: v32 and the subclasses assign other structs.
        storage_cls: type = SharedStorage
        if cutlass.const_expr(self.v32):
            # `_annotations` is not in the stubs; ruff rejects a constant getattr.
            annotations = dict(SharedStorage._annotations)  # type: ignore[attr-defined]
            for name, elems in (
                ("smem_hstate", handoff_state_elems),
                ("smem_hu", handoff_blk_elems),
                ("smem_hnewv", handoff_blk_elems),
                ("smem_hnewv_decayed", handoff_blk_elems),
            ):
                annotations[name] = cute.struct.Align[
                    cute.struct.MemRange[self.dtype, elems], 128
                ]
            storage_cls = cute.struct(
                type("SharedStorage", (), {"__annotations__": annotations})
            )

        if cutlass.const_expr(self.needs_checkpointing and self.slot_ckpt):
            # The checkpoint schedule: next due block end, its entry, and one
            # slot per block parity (see stage_checkpoint).
            annotations = dict(storage_cls._annotations)  # type: ignore[attr-defined]
            annotations["smem_ckpt"] = cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4], 16
            ]
            storage_cls = cute.struct(
                type("SharedStorage", (), {"__annotations__": annotations})
            )

        self.shared_storage = storage_cls

        self.kernel(
            g_alpha,
            g_beta,
            g_q,
            g_k,
            g_v,
            g_o,
            g_state,
            g_init_state,
            g_state_indices,
            g_state_checkpoints,
            checkpoint_cu_starts,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            total_checkpoints,
            checkpoint_every_n_tokens,
            g_output_state_indices,
            g_checkpoint_rows,
            g_prefix_lens,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(LOAD_THREADS, 1, 1),
            max_number_threads=(LOAD_THREADS, 1, 1),
            stream=stream,
            # Two would cap ptxas at 128 registers a thread and force spills.
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        gO_full: cute.Tensor,
        g_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
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
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        work_desc = self.get_next_work(
            cu_seqlens,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
        )
        tok_begin = None
        ckpt_origin = None
        if cutlass.const_expr(self.use_prefix_lens):
            tok_begin = work_desc.tok_offset
            work_desc, _, ckpt_origin = work_desc.left_padded(
                g_prefix_lens, self.BLK_KV
            )
        tok_end = work_desc.tok_offset + work_desc.seq_len
        num_blocks = (
            work_desc.seq_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        # Every thread is a math thread, so its math index is its block index.
        math_tidx = tidx
        wg_idx = tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)

        # ── Smem allocation ───────────────────────────────────────────────────
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        s_ckpt = None
        if cutlass.const_expr(self.needs_checkpointing and self.slot_ckpt):
            s_ckpt = storage.smem_ckpt.get_tensor(cute.make_layout(4))
            self.init_checkpoint_schedule(
                s_ckpt,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                work_desc.seq_idx,
                ckpt_origin,
                tidx,
            )

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

        v_smem_layout_atom = warpgroup.make_smem_layout_atom(
            self._k_major_atom_kind(self.D_v), self.dtype
        )
        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                v_smem_layout_atom,
                (self.BLK_KV, self.D_v, self.v_stage),
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
            (self.D_v, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        beta_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))
        sBeta = storage.smem_beta.get_tensor(beta_layout)

        # v32 handoff buffers; without v32 there is no storage for them.
        sHandoff = None
        if cutlass.const_expr(self.v32):
            hstate_layout, hblk_layout = self._handoff_layouts(qkv_smem_layout_atom)
            sHandoff = (
                storage.smem_hstate.get_tensor(
                    hstate_layout.outer, swizzle=hstate_layout.inner
                ),
                storage.smem_hu.get_tensor(
                    hblk_layout.outer, swizzle=hblk_layout.inner
                ),
                storage.smem_hnewv.get_tensor(
                    hblk_layout.outer, swizzle=hblk_layout.inner
                ),
                storage.smem_hnewv_decayed.get_tensor(
                    hblk_layout.outer, swizzle=hblk_layout.inner
                ),
            )

        q_pipeline = PipelineCpAsyncSm80(self.q_stage)
        k_pipeline = PipelineCpAsyncSm80(self.k_stage)
        v_pipeline = PipelineCpAsyncSm80(self.v_stage)
        alpha_pipeline = PipelineCpAsyncSm80(self.alpha_beta_stage)
        beta_pipeline = PipelineCpAsyncSm80(self.alpha_beta_stage)
        # No mbarriers to fence; one barrier lines the block up before the first load.
        cute.arch.sync_threads()

        if work_desc.seq_len != cutlass.Int32(0):
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
                sHandoff,
                gQ_full,
                gK_full,
                gV_full,
                gO_full,
                g_alpha,
                g_beta,
                q_pipeline,
                k_pipeline,
                v_pipeline,
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
                tidx,
                warp_idx,
                tok_end,
                num_blocks,
                num_q_heads,
                num_v_heads,
                num_sab_heads,
                num_seqs,
                total_checkpoints,
                checkpoint_every_n_tokens,
                g_output_state_indices,
                g_checkpoint_rows,
                s_ckpt,
                tok_begin,
            )


# ─── Public API ──────────────────────────────────────────────────────────────


#: Benchmark override offering v32 without the dispatcher; `_v32_applies` gates it.
_V32_PROTOTYPE = False


def _v32_applies(
    *,
    enabled: bool = False,
    kernel_dtype,
    initial_state_dtype=None,
    num_seqs,
    total_seqlen,
    num_q_heads,
    num_v_heads,
    needs_init_state,
    needs_checkpointing,
    use_state_indices,
) -> bool:
    """Whether v32 is offered and the call matches its one measured shape exactly."""
    if not (enabled or _V32_PROTOTYPE):
        return False
    return (
        kernel_dtype is cutlass.BFloat16
        and (initial_state_dtype is None or initial_state_dtype is cutlass.BFloat16)
        and num_seqs == 1
        and total_seqlen == 8192
        and num_q_heads == 4
        and num_v_heads == 16
        and needs_init_state
        and not needs_checkpointing
        and not use_state_indices
    )


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
    blk_v,
    v32=False,
    output_state_indices_dtype=None,
    checkpoint_state_indices_dtype=None,
    prefix_lens_dtype=None,
    checkpoint_row_align=None,
):
    """One cached kernel object per compile key; the arguments are its fields."""
    return _FullyFusedDeltaRuleSm80(
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
        blk_v=blk_v,
        v32=v32,
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
    # Private: the dispatcher's offer of v32, from host data this function lacks;
    # `_v32_applies` still gates it.
    _v32: bool = False,
    output_state_indices: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
    prefix_lens: torch.Tensor | None = None,
):
    """GDN prefill (CuTe DSL, sm_80).

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

    if not _FullyFusedDeltaRuleSm80.can_implement(
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
        # Passed as reshape(-1), which copies a strided tensor and loses the
        # writes. A pool addressed by checkpoint_state_indices is passed whole.
        (
            "state_checkpoints",
            state_checkpoints if checkpoint_state_indices is None else None,
        ),
        # The compile cache key has no strides; a baked unit stride would be reused.
        ("checkpoint_cu_starts", checkpoint_cu_starts),
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
    q_view = q.as_strided(
        (total_seqlen, D, num_q_heads),
        (num_q_heads * D, 1, D),
    )
    k_view = k.as_strided(
        (D, total_seqlen, num_k_heads),
        (1, num_k_heads * D, D),
    )
    v_view = v.as_strided(
        (D, total_seqlen, num_v_heads),
        (1, num_v_heads * D, D),
    )
    o_view = o.as_strided(
        (D, total_seqlen, num_o_heads),
        (1, num_o_heads * D, D),
    )
    _check_load_alignment(q=q_view, k=k_view, v=v_view, o=o_view)
    _check_state_dtype_supported(
        state=state,
        initial_state=init_state if needs_init_state else None,
        state_checkpoints=state_checkpoints if needs_checkpointing else None,
    )
    total_checkpoints = state_checkpoints.shape[0] if needs_checkpointing else 1

    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

    use_v32 = _v32_applies(
        enabled=_v32,
        kernel_dtype=kernel_dtype,
        initial_state_dtype=(
            state_dtype_to_cutlass(init_state.dtype) if needs_init_state else None
        ),
        num_seqs=num_seqs,
        total_seqlen=total_seqlen,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        needs_init_state=needs_init_state,
        needs_checkpointing=needs_checkpointing,
        use_state_indices=use_state_indices,
    )

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
        blk_v=_BLK_V // (2 if use_v32 else 1),
        v32=use_v32,
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
    # One block per (sequence, head, value slice).
    grid_x = num_seqs * num_sab_heads * delta_rule_kernel.num_v_splits

    compile_options = _sm80_compile_options(device)
    compiled_delta_rule_kernel = get_cached_compile(delta_rule_kernel, compile_options)
    if compiled_delta_rule_kernel is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        kernel_args = (
            from_dlpack(q_view, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(k_view, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_view, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(o_view, assumed_align=16).mark_layout_dynamic(leading_dim=0),
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
            grid_x,
            stream,
        )
        compiled_delta_rule_kernel = cached_compile(
            delta_rule_kernel,
            *kernel_args,
            compile_options=compile_options,
        )
    compiled_delta_rule_kernel(
        q_view,
        k_view,
        v_view,
        o_view,
        alpha.reshape(-1) if needs_alpha else None,
        beta.reshape(-1) if needs_beta else None,
        state,
        init_state if needs_init_state else None,
        state_indices if use_state_indices else None,
        checkpoints_arg,
        checkpoint_cu_starts if needs_checkpointing else None,
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
        grid_x,
        stream,
    )
