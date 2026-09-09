# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""W4A16 adaptation of the current swapped MegaMoE epilogue.

FC1 changes: internal gate16/up16 accumulators receive FP32 expert
alphas and SwiGLU, then store BF16 directly. Prepared weights share W4A4's
gate16/up16 ordering and decode directly into operand-A TMEM.
The current FC2 process pipeline, BF16 return router, and phase-aware
completion tracker are reused. Its non-overlap subtiles are statically unrolled.
No activation quantization, scale-factor output, or epilogue SMEM is used.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int64
from cutlass.cute.nvgpu import tcgen05

from src.iket_compat import iket
from common.moe_utils import fmin, fmax
from src.flag_batch import GpuReleaseFlagBatchTracker
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from moe_nvfp4_swapab.moe_persistent_scheduler import (
    MoESchedConsumer,
    MoESchedExtension,
)
from moe_nvfp4_swapab.epilogue_refactor import (
    NvFp4OptinalEpiArgs,
    Region,
    SwapABFc1Epilogue,
    SwapABFc2Epilogue,
    SwapABSwigluFp4Epilogue,
    TmemTranspose16x32,
)


class W4A16Epilogue(SwapABSwigluFp4Epilogue):
    """Current scheduler/return contract with BF16 FC1 and two acc stages."""

    def __init__(
        self,
        *,
        mma_tiler_mnk,
        cluster_shape_mn,
        use_2cta_instrs,
        fc1_output_dtype,
        combine_format,
        non_ubulk_fc2_store=True,
        in_kernel_fc2_reduce=False,
        token_back_by_dispatch=False,
        acc_dtype=cutlass.Float32,
        allow_overlap_acc=False,
        static_expert_shape=None,
        gate_up_clamp=None,
        epi_flag_batch=(1, 1),
    ):
        if (
            mma_tiler_mnk
            not in ((128, 64, 256), (128, 128, 256), (256, 64, 256), (256, 128, 256))
            or (
                cluster_shape_mn != (2, 1)
                and not (cluster_shape_mn == (1, 1) and mma_tiler_mnk == (128, 64, 256))
            )
            or use_2cta_instrs != (mma_tiler_mnk[0] == 256)
        ):
            raise ValueError("W4A16 requires a supported M/N/K tile and cluster shape.")
        if (
            fc1_output_dtype is not cutlass.BFloat16
            or combine_format.act_dtype is not cutlass.BFloat16
            or combine_format.is_quantized
            or acc_dtype is not cutlass.Float32
            or not non_ubulk_fc2_store
            or in_kernel_fc2_reduce
            or allow_overlap_acc
        ):
            raise ValueError(
                "W4A16 requires FP32 accumulators and direct BF16 handoffs"
            )
        # The current base validates its own quantized FC1 helper. Initialize
        # its geometry with that required dtype, then replace only resources
        # belonging to our BF16 helper. The FC2 helper receives real BF16 wire.
        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
            sf_vec_size=16,
            fc1_output_dtype=cutlass.Float4E2M1FN,
            combine_format=combine_format,
            non_ubulk_fc2_store=non_ubulk_fc2_store,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            acc_dtype=acc_dtype,
            allow_overlap_acc=False,
            static_expert_shape=static_expert_shape,
            gate_up_clamp=gate_up_clamp,
            epi_flag_batch=epi_flag_batch,
        )
        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_sf_dtype = None
        self.sf_vec_size = None
        self.acc_sf_cols = 0
        self.epi_smem_bytes = 0

    # This is the current upstream run loop; its only functional substitution
    # is W4A16Fc1Epilogue in place of SwapABFc1Epilogue. Keep its stage
    # selection, release tracker, barriers, and tail flush in sync with it.
    @cute.jit
    def run(
        self,
        epi_smem_storage,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        # ── Sched ────────────────────────────────────────────────────────
        sched_consumer: MoESchedConsumer,
        sched_ext: MoESchedExtension,
        # ── tensors ──────────────────────────────────
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,  # Domain of fake (m, n, l)
        fc1_output_sf: cute.Tensor,  # Domain of fake (m, n, l)
        fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
        fc1_done_counter: cute.Tensor,  # 1D tensor
        tidx: cutlass.Int32,
        optional_epi_args: NvFp4OptinalEpiArgs = None,  # Epilogue optinal runtime arguments.
        token_comm_args=None,  # Only valid when enable token communication
    ):
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = NvFp4OptinalEpiArgs(
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                topk_scores=None,
            )
        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(
                self.tmem_acc_layout_py_obj[0],
                stride=self.tmem_acc_layout_py_obj[1],
            ),
        )

        fc1_epi = W4A16Fc1Epilogue(
            self,
            tidx,
            epi_smem_storage,
            sched_ext,
            tma_atom_fc1_output,
            fc1_output,
            fc1_output_sf,
            fc1_done_counter,
            optional_epi_args,
        )
        fc2_epi = W4A16Fc2Epilogue(
            self, tidx, epi_smem_storage, fc2_output, token_comm_args, optional_epi_args
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages
        )
        wait_only_named_barrier = pipeline.NamedBarrier(
            barrier_id=self._EpilogueSyncWaitBarId,
            num_threads=32 * self._EpilogueWarpCnt,
        )
        is_odd_turn = cutlass.Int32(1)
        work_tile_info = sched_consumer.consume_work()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx % (self._EpilogueWarpCnt * 32),
        )

        while work_tile_info.is_valid_tile:
            if cutlass.const_expr(self.overlapping_accum):
                tmem_stage_idx = acc_consumer_state.phase
            else:
                tmem_stage_idx = acc_consumer_state.index
            tmem_acc_current = tmem_acc[None, None, tmem_stage_idx]
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                # The __call__ args should only take the while loop args, leave all loop irrevalent args to the init.
                fc1_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            else:
                # The __call__ args should only take the while loop args, leave all loop irrevalent args to the init.
                fc2_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            iket.range_pop()

            prev_work_tile_info = work_tile_info
            cur_was_linear1 = prev_work_tile_info.phase == cutlass.Int32(
                BlockPhase.Linear1
            )

            acc_consumer_state.advance()
            if cutlass.const_expr(self.overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn

            work_tile_info = sched_consumer.consume_work()

            # Drain pending FC1 stores before publishing the fc1-done counter.
            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            # _fence_rel_gpu()
            wait_only_named_barrier.arrive_and_wait()

            # Publish completion for the work tile snapshotted above.
            if cur_was_linear1:
                flag_tracker = fc1_epi.signal_fc1_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
            else:
                flag_tracker = fc2_epi.signal_fc2_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
        # Tail flush
        flag_tracker.fire()


class W4A16Fc2Epilogue(SwapABFc2Epilogue):
    """Static BF16 subtiles with inherited routing, stores and completion."""

    @cute.jit
    def __call__(
        self,
        work_tile_info,
        tmem_acc_tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
    ):
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        fc2_output_router = self._make_output_router(work_tile_info)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor,
            (self._EpilogueFc2HiddenTileSize, self._EpilogueTokenTileSize),
        )[None, None, 0, None]
        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        iket.range_push("fc2_epi")
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        # W4A16 rejects overlapping accumulators. Static subtiles make the
        # inherited STG router indices constants (four issues per subtile),
        # allowing its prefetched pointer/valid arrays to stay in registers.
        for i in cutlass.range_constexpr(self.subtile_cnt):
            subtile_idx = cutlass.Int32(i)
            if subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                if cutlass.const_expr(i == self.subtile_cnt - 1):
                    self._run_last_subtile(
                        subtile_idx=subtile_idx,
                        tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                            None, None, subtile_idx
                        ],
                        fc2_output_router=fc2_output_router,
                        alpha_val=alpha_val,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                    )
                else:
                    self.run_subtile(
                        subtile_idx=subtile_idx,
                        tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                            None, None, subtile_idx
                        ],
                        preload_acc=None,
                        fc2_output_router=fc2_output_router,
                        alpha_val=alpha_val,
                        release_after_ldtm=False,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                    )
            elif cutlass.const_expr(i == self.subtile_cnt - 1):
                # Includes zero tokens and N128 tiles with only the first
                # subtile valid. This is the existing valid guard's else arm.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_last_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor,
        fc2_output_router,
        alpha_val,
        acc_pipeline,
        acc_consumer_state,
    ):
        # Unlike raw LDTM, post-reorder's return ends all accumulator-TMEM
        # scratch use. Its result is RMEM; only register packing and STG remain.
        process_pipeline = self.process_pipeline
        loaded = process_pipeline.tmem_acc_load(
            tmem_subtile_tensor=tmem_subtile_tensor, epi=self
        )
        casted = process_pipeline.f2fp(*loaded, alpha_val=alpha_val)
        pre_store = process_pipeline.post_f2fp_reorder(
            casted=casted, tmem_subtile_view=tmem_subtile_tensor
        )
        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)
        process_pipeline.store_function(
            epi=self,
            subtile=pre_store,
            subtile_idx=subtile_idx,
            fc2_output_router=fc2_output_router,
        )


class W4A16Fc1Epilogue(SwapABFc1Epilogue):
    """BF16 FC1 body; inherit current immutable wrapper and done signaling."""

    @cute.jit
    def _swiglu_act(self, t_swiglu, t_up, t_gate, prob=None):
        # Match the local W4A16 kernel's up * (gate * sigmoid(gate))
        # association before the BF16 FC1 handoff.
        for i in cutlass.range_constexpr(0, cute.size(t_swiglu), 2):
            gate = (t_gate[i], t_gate[i + 1])
            gate_log2e = cute.arch.mul_packed_f32x2(
                gate, (-1.4426950408889634, -1.4426950408889634)
            )
            denominator = cute.arch.add_packed_f32x2(
                (
                    cute.math.exp2(gate_log2e[0], fastmath=True),
                    cute.math.exp2(gate_log2e[1], fastmath=True),
                ),
                (1.0, 1.0),
            )
            sigmoid = (
                cute.arch.rcp_approx(denominator[0]),
                cute.arch.rcp_approx(denominator[1]),
            )
            silu = cute.arch.mul_packed_f32x2(gate, sigmoid)
            t_swiglu[i], t_swiglu[i + 1] = cute.arch.mul_packed_f32x2(
                (t_up[i], t_up[i + 1]), silu
            )

    def __init__(
        self,
        base,
        tidx,
        epi_smem_storage,
        sched_ext,
        tma_atom_fc1_output,
        fc1_output,
        fc1_output_sf,
        fc1_done_counter,
        optional_epi_args,
    ):
        # Bypass the quantized parent's SMEM construction. These are the same
        # loop-invariant fields consumed by its inherited completion method.
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        self.sched_ext = sched_ext
        self.fc1_output = fc1_output
        self.fc1_done_counter = fc1_done_counter
        self.optional_epi_args = optional_epi_args
        self._freeze()

    @cute.jit
    def __call__(
        self,
        work_tile_info,
        tmem_acc_tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
    ):
        real_fc1_output, _ = self.sched_ext.get_gmem_tensor(
            "c", self.fc1_output, work_tile_info
        )
        weight_alpha = self.optional_epi_args.fc1_alpha[work_tile_info.expert_idx]
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("fc1_epi")
        # Keep prior subtiles in the established loop and specialize only the
        # final one. N64 has no prior subtile; N128 has one.
        if cutlass.const_expr(self.subtile_cnt > 1):
            for subtile_idx in cutlass.range(self.subtile_cnt - 1, unroll=1):
                if subtile_idx * 64 < work_tile_info.valid_tokens_in_cta_tile:
                    self._run_fc1_bf16_subtile(
                        tmem_acc_tensor,
                        cutlass.Int32(0),
                        subtile_idx,
                        real_fc1_output,
                        work_tile_info,
                        self.warp_idx,
                        self.tidx,
                        weight_alpha,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        release_after_scratch=False,
                    )
        last_subtile_idx = cutlass.Int32(self.subtile_cnt - 1)
        if last_subtile_idx * 64 < work_tile_info.valid_tokens_in_cta_tile:
            self._run_fc1_bf16_subtile(
                tmem_acc_tensor,
                cutlass.Int32(0),
                last_subtile_idx,
                real_fc1_output,
                work_tile_info,
                self.warp_idx,
                self.tidx,
                weight_alpha,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                release_after_scratch=True,
            )
        else:
            # Includes zero tokens and N128 tiles with only the first
            # subtile valid. All earlier stores retain their original order.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_fc1_bf16_subtile(
        self,
        tmem_acc_tensor,
        stage_offset,
        subtile_idx,
        real_fc1_output,
        work_tile_info,
        warp_idx,
        tidx,
        weight_alpha,
        acc_pipeline,
        acc_consumer_state,
        release_after_scratch: cutlass.Constexpr[bool],
    ):
        lane = tidx % 32
        # Prepared gate16/up16 rows keep both operands within one warp.
        # TCGEN05 restricts each warp to its own 32 TMEM datapaths; both
        # activation operands must therefore stay inside that warp's band.
        gate_feature = warp_idx * 32
        up_feature = gate_feature + 16
        load_atom = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=256
        )
        for half in cutlass.range_constexpr(2):
            token_col = stage_offset + subtile_idx * 64 + half * 32
            gate_ptr = tmem_acc_tensor.iterator + cute.assume(
                (gate_feature << 16) + token_col, divby=16
            )
            up_ptr = tmem_acc_tensor.iterator + cute.assume(
                (up_feature << 16) + token_col, divby=16
            )
            gate = cute.make_rmem_tensor((16,), cutlass.Float32)
            up = cute.make_rmem_tensor((16,), cutlass.Float32)
            cute.copy(
                load_atom,
                cute.make_tensor(gate_ptr, TmemTranspose16x32._tmem_layout(16, 32)),
                TmemTranspose16x32._rmem_copy_view(gate, 16),
            )
            cute.copy(
                load_atom,
                cute.make_tensor(up_ptr, TmemTranspose16x32._tmem_layout(16, 32)),
                TmemTranspose16x32._rmem_copy_view(up, 16),
            )
            # Complete this warp's loads before its in-place transpose
            # overwrites the same32-feature band.
            cute.arch.fence_view_async_tmem_load()
            gate.store(gate.load() * weight_alpha)
            up.store(up.load() * weight_alpha)
            # Preserve the prior W4A16 sequence: post-alpha gate upper clamp,
            # symmetric up clamp, then up * (gate * sigmoid(gate)).
            if cutlass.const_expr(self.gate_up_clamp is not None):
                for i in cutlass.range_constexpr(cute.size(up)):
                    gate[i] = fmin(gate[i], self.gate_up_clamp)
                    up[i] = fmin(up[i], self.gate_up_clamp)
                    up[i] = fmax(up[i], -self.gate_up_clamp)
            activated = cute.make_rmem_tensor((16,), cutlass.Float32)
            self._swiglu_act(activated, up, gate)
            scratch_ptr = tmem_acc_tensor.iterator + cute.assume(
                ((warp_idx * 32) << 16) + token_col, divby=16
            )
            transpose = TmemTranspose16x32(
                scratch_ptr,
                Region.Top,
                reg_tensor=activated,
            )
            transpose.r1_perm()
            transpose.r1_store()
            transpose.r2_load()
            transpose.r2_store()
            transpose.r3_load_top()
            transpose.r3_load_bot()
            transpose.r3_perm()
            transpose.r3_store()
            transpose.r4_load_top()
            transpose.r4_load_bot()
            if cutlass.const_expr(release_after_scratch and half == 1):
                # Both final scratch reads now feed RMEM. The following
                # permutation, BF16 conversion and STG do not use TMEM.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
            transpose.r4_perm()
            # CuTe's dynamic branch may carry tensors, but not the plain
            # Python transpose helper. Resolve its output before branching.
            transposed_output = transpose.output

            token_in_tile = subtile_idx * 64 + half * 32 + lane
            output_column = work_tile_info.tile_m_idx * 64 + warp_idx * 16
            if (
                token_in_tile < work_tile_info.valid_tokens_in_cta_tile
                and output_column < real_fc1_output.shape[1]
            ):
                token_row = work_tile_info.tile_n_idx * self.cta_tile_n + token_in_tile
                output = cute.make_rmem_tensor((16,), cutlass.BFloat16)
                output.store(transposed_output.load().to(cutlass.BFloat16))
                row = cute.local_tile(
                    real_fc1_output,
                    (1, 16, 1),
                    (token_row, output_column // 16, 0),
                )
                aligned = cute.make_ptr(
                    cutlass.BFloat16,
                    row.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                cute.copy(
                    store_atom, output, cute.make_tensor(aligned, cute.make_layout(16))
                )
