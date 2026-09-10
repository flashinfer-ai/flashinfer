# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""W4A16 epilogue for the swapped MegaMoE pipeline.

FC1 changes: internal gate16/up16 accumulators receive FP32 expert
alphas and SwiGLU, then store BF16 directly. Prepared weights share W4A4's
gate16/up16 ordering and decode directly into operand-A TMEM.
FC2 owns the BF16 return router and store path, with statically unrolled
non-overlap subtiles and the common phase-aware completion tracker.
No activation quantization, scale-factor output, or epilogue SMEM is used.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int64
from cutlass.cute.nvgpu import tcgen05

from .fc1_fc2_fuse_sched import BlockPhase
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import iket
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import fmin, fmax
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import GpuReleaseFlagBatchTracker
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    MoESchedConsumer,
    MoESchedExtension,
)
from .tmem_epilogue import (
    W4A16EpiArgs,
    Region,
    TmemTranspose16x32,
    EpilogueContext,
    Fc2OutputRouter,
    make_bf16_fc2_stg_process_pipeline,
)


class W4A16Epilogue:
    """Current scheduler/return contract with BF16 FC1 and two acc stages."""

    _EpilogueSyncWaitBarId = 1
    _EpilogueAsyncBarIdBase = 4
    _EpilogueFc1GateUpInterleave = 16
    _EpilogueTokenTileSize = 64
    _EpilogueFc1IntermediateGateUpTileSize = 128
    _EpilogueFc1IntermediateDownTileSize = 64
    _EpilogueFc2HiddenTileSize = 128
    _EpilogueWarpCnt = 4
    _TmemColsTotal = 512

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
        self.fc2_use_bulk = False
        self.reduce_topk_in_kernel = False
        self.token_back_by_dispatch = token_back_by_dispatch
        self.combine_format = combine_format
        self.fc1_output_dtype = fc1_output_dtype
        self.acc_dtype = acc_dtype
        self.fc1_output_sf_dtype = None
        self.sf_vec_size = None
        self.gate_up_clamp = gate_up_clamp
        fc1_batch, fc2_batch = (1, 1) if epi_flag_batch is None else epi_flag_batch
        self.fc1_epi_flag_batch = max(1, min(32, int(fc1_batch)))
        self.fc2_epi_flag_batch = max(1, min(32, int(fc2_batch)))
        self.cluster_tile_intermediate_downproj = (
            self._EpilogueFc1IntermediateDownTileSize * cluster_shape_mn[0]
        )
        self.cta_tile_m = self._EpilogueFc2HiddenTileSize
        self.cta_tile_n = mma_tiler_mnk[1]
        self.cta_tile_k = mma_tiler_mnk[2]
        self.static_expert_shape = static_expert_shape
        self.acc_tmem_cols = self.cta_tile_n
        self.acc_sf_cols = 0
        self.fc2_hidden_needs_predicate = not (
            static_expert_shape is not None
            and static_expert_shape[2] % (self.cta_tile_m * cluster_shape_mn[0]) == 0
        )
        self.intermediate_downproj = (
            static_expert_shape[1] // 2 if static_expert_shape is not None else None
        )
        self.subtile_cnt = self.cta_tile_n // self._EpilogueTokenTileSize
        self.overlapping_accum = False
        self.num_acc_stage = 2
        self.num_acc_pipeline_stages = 2
        self.overlapped_tmem_cols = 0
        self.epi_smem_bytes = 0
        self.tmem_acc_layout_py_obj = (
            (self.cta_tile_m, self.cta_tile_n, self.num_acc_stage),
            (1 << 16, 1, self.cta_tile_n),
        )

    # Preserve the existing W4A16 stage selection, completion ordering,
    # release tracker, barriers and tail flush.
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
        optional_epi_args: W4A16EpiArgs = None,  # Epilogue optinal runtime arguments.
        token_comm_args=None,  # Only valid when enable token communication
    ):
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = W4A16EpiArgs(
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
        flag_tracker.fire()


class W4A16Fc2Epilogue(EpilogueContext):
    """Static BF16 subtiles with owned routing, stores and completion."""

    def __init__(
        self,
        base,
        tidx,
        epi_smem_storage,
        fc2_output,
        token_comm_args,
        optional_epi_args,
    ):
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        self.fc2_output = fc2_output
        self.token_comm_args = token_comm_args
        self.optional_epi_args = optional_epi_args
        self.smem_tensor = None
        self.process_pipeline = make_bf16_fc2_stg_process_pipeline(
            cta_token_tile_size=base.cta_tile_n,
            cta_hidden_tile_size=base.cta_tile_m,
        )
        self._freeze()

    @cute.jit
    def signal_fc2_done(self, work_tile_info, next_work_tile_info, flag_tracker):
        publish: cutlass.Constexpr = self.token_back_by_dispatch
        if cutlass.const_expr(publish):
            flag_addr = (
                self.token_comm_args.fc2_done_counter.iterator
                + work_tile_info.expert_idx
            ).toint()
        else:
            flag_addr = Int64(0)
        no_fire: cutlass.Constexpr = not publish
        return flag_tracker.accumulate(
            next_work_tile_info.phase, self.fc2_epi_flag_batch, flag_addr, no_fire
        )

    @cute.jit
    def _make_output_router(
        self,
        work_tile_info,
    ) -> "Fc2OutputRouter":
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n)
        )
        hidden_base_this_cta_tile = work_tile_info.tile_m_idx * cutlass.Int32(
            self.cta_tile_m
        )
        valid_hidden_this_cta_tile = (
            cutlass.Int32(self.fc2_output.shape[2]) - hidden_base_this_cta_tile
        )
        if valid_hidden_this_cta_tile < 0:
            valid_hidden_this_cta_tile = 0
        if valid_hidden_this_cta_tile > self._EpilogueFc2HiddenTileSize:
            valid_hidden_this_cta_tile = self._EpilogueFc2HiddenTileSize

        metadata_u32 = None
        peer_rank_ptr_mapper = None
        data_token_base = task_tile_data_row_start
        if cutlass.const_expr(
            self.token_comm_args is not None and not self.token_back_by_dispatch
        ):
            metadata_u32 = cute.domain_offset(
                (task_tile_data_row_start, 0),
                cute.recast_tensor(
                    self.token_comm_args.token_src_metadata,
                    cutlass.Uint32,
                ),
            )
            peer_rank_ptr_mapper = self.token_comm_args.peer_rank_ptr_mapper
            data_token_base = None

        base_outputs = self.fc2_output
        token_bases = data_token_base
        output_mappings = self.process_pipeline.store_out_mapping

        return Fc2OutputRouter(
            metadata=metadata_u32,
            token_bases=token_bases,
            base_outputs=base_outputs,
            hidden_base_this_cta_tile=hidden_base_this_cta_tile,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden_this_cta_tile=valid_hidden_this_cta_tile,
            output_mappings=output_mappings,
            epi_tid=self.tidx,
        ).prefetch()

    @cute.jit
    def run_subtile(
        self,
        subtile_idx: cutlass.Int32,
        # (hidden_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        preload_acc,
        fc2_output_router: "Fc2OutputRouter",
        alpha_val: Optional[cutlass.Float32],
        release_after_ldtm,
        acc_pipeline,
        acc_consumer_state,
    ):
        process_pipeline = self.process_pipeline
        if cutlass.const_expr(preload_acc is None):
            loaded = process_pipeline.tmem_acc_load(
                tmem_subtile_tensor=tmem_subtile_tensor,
                epi=self,
            )
            if release_after_ldtm:
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
        else:
            loaded = preload_acc

        casted = process_pipeline.f2fp(
            *loaded,
            alpha_val=alpha_val,
        )
        # reorder returns a bare RMEM fragment in the store's expected pre-store
        # distribution; reorder + store are paired 1:1 inside the pipeline.
        pre_store = process_pipeline.post_f2fp_reorder(
            casted=casted,
            tmem_subtile_view=tmem_subtile_tensor,
        )
        process_pipeline.store_function(
            epi=self,
            subtile=pre_store,
            subtile_idx=subtile_idx,
            fc2_output_router=fc2_output_router,
        )

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
        # STG router indices constants (four issues per subtile),
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


class W4A16Fc1Epilogue(EpilogueContext):
    """BF16 FC1 body and exact phase-aware done signaling."""

    @cute.jit
    def signal_fc1_done(self, work_tile_info, next_work_tile_info, flag_tracker):
        # Only in-bound intermediate_downproj tiles signal; OOB -> null slot.
        if cutlass.const_expr(
            self.static_expert_shape is None
            or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
        ):
            in_bound = (
                work_tile_info.tile_m_idx * self._EpilogueFc1IntermediateDownTileSize
                < self.fc1_output.shape[1]
            )
        else:
            in_bound = True
        slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
        flag_addr = Int64(0)
        if in_bound:
            flag_addr = (self.fc1_done_counter.iterator + slot).toint()
        return flag_tracker.accumulate(
            next_work_tile_info.phase,
            self.fc1_epi_flag_batch,
            flag_addr,
        )

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
        # Loop-invariant fields consumed by the BF16 body and completion method.
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
