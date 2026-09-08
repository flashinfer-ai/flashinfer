# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Online NVFP4 weight decoding in the BF16 fused MegaMoE schedule.

The scheduler, BF16 activation TMA path and communication protocol follow the
vendored BF16 kernel. Four warps decode packed weight tiles directly into the
BF16 B-operand pipeline; both GEMMs and dispatch/combine remain one kernel.
"""

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
from cutlass.base_dsl.dsl import extract_mlir_attributes

from moe_bf16_glu.megamoe_kernel_bf16 import Sm100MegaMoEBf16Kernel
from moe_bf16_glu.custom_ext_bf16 import GluBf16Fc12SchedExtension
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase, MoEFusedFc12SchedulerParams
from moe_nvfp4_swapab.moe_utils import spin_wait
from src.token_comm import TokenCommArgs as ExtractedTokenCommArgs
from src.iket_compat import iket
from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_utils import (
    e2m1x16_e4m3_to_bf16x16,
)

from .epilogue import W4A16Epilogue


class _ScaledTokenCommArgs:
    """Keep local FC2 weight scaling separate from the communication ABI."""

    def __init__(self, comm, fc2_alpha):
        self.comm = comm
        self.fc2_alpha = fc2_alpha

    def __getattr__(self, name):
        return getattr(self.comm, name)

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.comm) + extract_mlir_values(self.fc2_alpha)

    def __extract_mlir_attributes__(self):
        return extract_mlir_attributes(self.comm) + extract_mlir_attributes(
            self.fc2_alpha
        )

    def __new_from_mlir_values__(self, values):
        n = len(extract_mlir_values(self.comm))
        return _ScaledTokenCommArgs(
            new_from_mlir_values(self.comm, values[:n]),
            new_from_mlir_values(self.fc2_alpha, values[n:]),
        )


class Sm100W4A16MegaMoEKernel(Sm100MegaMoEBf16Kernel):
    """BF16 dispatch and compute with online-decoded NVFP4 expert weights."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.token_back_standalone:
            raise ValueError(
                "W4A16 currently supports epi_warps or reuse_dispatch_warps."
            )
        if self.apply_topk_in_fc1 or self.fc2_in_kernel_topk_reduce:
            raise ValueError("W4A16 applies routing weights after the BF16 FC2 output.")
        self.weight_warp_ids = (6, 12, 13, 14)
        self.num_weight_threads = 128
        self.threads_per_cta = 480
        self.token_comm.num_other_warps += 3
        self.token_comm.num_other_threads += 96
        self.token_comm.num_total_threads += 96
        self.token_comm.kernel_tail_threads += 96

    def _setup_attributes(self):
        super()._setup_attributes()
        self.threads_per_cta = 480
        self.epilogue = W4A16Epilogue(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            glu_clamp=self.gate_up_clamp,
            apply_topk_in_fc1=False,
            use_stg_fc1=self.use_stg_fc1,
        )

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,  # (T, hidden) BF16
        topk_idx: cute.Tensor,  # (T, num_topk) Int64
        topk_weights: cute.Tensor,  # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_c: Optional[cute.Tensor],  # fc1 c output
        # Combine destination (peer write target via the epilogue Fc2OutputDest).
        combine_output: cute.Tensor,  # (T, num_topk, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Tensor,  # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,  # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the BF16 MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``topk_weights`` / ``combine_output`` MUST point
            into memory reachable via
            ``peer_rank_ptr_mapper.ptr_map_to_rank(...)`` (NVSHMEM symmetric
            heap).  Single-rank degenerate runs are allowed.
          * ``topk_idx`` is read on the local rank only.
          * ``fc1_weight`` / ``fc2_weight`` are local-only.

        ``combine_output`` is the MoE-domain ``(max_tokens_per_rank, num_topk,
        hidden)`` BF16 storage; the epilogue maps each pool row back to the
        source rank's ``[src_token, src_topk, :]`` slot via ``token_comm_args``
        (form A; host reduces the topk axis).
        """
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        hidden = self.hidden

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + BF16 view
        # (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(local_workspace, "l1_token_buffer")
        l1_token_buffer_bf16 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        l1_topk_weights_buffer = self._view_local(
            local_workspace,
            "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(local_workspace, "l1_arrival_count")
        # token_src_metadata storage = (pool_token_capacity, TokenSrcMetadata.nbytes) Uint8;
        # dispatch_pull writes one packed Int64 per pool token row (see TokenSrcMetadata).
        token_src_metadata = self._view_local(
            local_workspace,
            "token_src_metadata",
        )
        expert_send_count = self._view_local(local_workspace, "expert_send_count")
        grid_sync_counter = self._view_local(local_workspace, "grid_sync_counter")
        nvlink_barrier_counter = self._view_local(
            local_workspace,
            "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_done_counter = self._view_local(local_workspace, "fc1_done_counter")

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace,
                "load_balance_counter",
            )

        token_back_schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace,
                "token_back_schedule_counter",
            ).iterator

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace,
                "fc2_output_workspace",
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (pool_token_capacity * hidden * 2,),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            fc2_done_counter = self._view_local(local_workspace, "fc2_done_counter")
            combine_output_u8 = cute.recast_tensor(combine_output, cutlass.Uint8)
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_u8 = combine_output
            fc2_output_target = combine_output

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace,
            "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(shared_workspace, "expert_recv_count")
        expert_recv_count_sum = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace,
            "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum`` buffer --
        # low32 bits hold per-expert total token count after _dispatch_barrier;
        # zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )
        local_zero_prefix = self._make_typed_view(
            local_workspace,
            0,
            cutlass.Int32,
            (self.local_zero_i32_count,),
            (1,),
            16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace,
            0,
            cutlass.Int32,
            (self.shared_zero_i32_count,),
            (1,),
            16,
        )

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            # BF16 tokens carry no scale-factor sideband: the SF slots ride the
            # TokenCommArgs None-skipping serialization, and dispatch's SF
            # loops are compiled out by sf_uint32_per_token=0.
            input_sf_buffer=None,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=None,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_u8,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=peer_rank_ptr_mapper_host.rank_idx,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=0,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            sm_count=sm_count,
        )

        token_comm_args = _ScaledTokenCommArgs(token_comm_args, fc2_alpha)
        self._launch_fc12(
            activation=l1_token_buffer_bf16,
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc1_alpha=fc1_alpha,
            fc1_output=fc1_output,
            fc1_c=fc1_c,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )

    @cute.jit
    def _launch_fc12(
        self,
        activation,
        fc1_weight,
        fc1_weight_sf,
        fc1_alpha,
        fc1_output,
        fc2_weight,
        fc2_weight_sf,
        fc2_output,
        topk_scores,
        fc1_done_counter,
        offs,
        max_active_clusters,
        stream,
        load_balance_counter,
        expert_token_sizes,
        token_comm_args,
        fc1_c=None,
    ):
        experts, intermediate_gateup, hidden = self.static_expert_shape
        intermediate = intermediate_gateup // 2
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (activation.shape[0], hidden, 1), stride=(activation.stride[0], 1, 0)
            ),
        )
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (fc1_output.shape[0], intermediate, 1),
                stride=(fc1_output.stride[0], 1, 0),
            ),
        )
        fc2_output_gemm = cute.make_tensor(
            fc2_output.iterator,
            cute.make_layout(
                (fc2_output.shape[0], hidden, 1), stride=(fc2_output.stride[0], 1, 0)
            ),
        )
        self.a_dtype = cutlass.BFloat16
        self.b_dtype = cutlass.BFloat16
        self.fc1_output_dtype = cutlass.BFloat16
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        self.fc1_output_layout = utils.LayoutEnum.from_tensor(fc1_output_gemm)
        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_activation, tma_tensor_fc1_activation = (
            cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                activation_gemm,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_atom_fc2_activation, tma_tensor_fc2_activation = (
            cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                fc1_output_gemm,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_atom_fc1_output, tma_tensor_fc1_output = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            fc1_output_gemm,
            self.epilogue.smem_layout_one_stage,
            self.epilogue.epi_tile,
        )
        counter_ptr = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            counter_ptr = load_balance_counter.iterator
        sched_params = MoEFusedFc12SchedulerParams(
            scenario=self.scenario,
            expert_shape=(experts, intermediate_gateup, hidden),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            load_balance_mode=self.load_balance_mode,
            load_balance_counter_ptr=counter_ptr,
            override_num_stages=self.num_sched_stages,
            is_swap_ab=False,
            expert_token_prefix_sum=None,
            expert_token_sizes=expert_token_sizes,
        )
        self.kernel(
            tiled_mma,
            tma_atom_fc1_activation,
            tma_tensor_fc1_activation,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            tma_atom_fc2_activation,
            tma_tensor_fc2_activation,
            activation_gemm,
            fc1_output_gemm,
            fc2_output_gemm,
            fc1_weight,
            fc1_weight_sf,
            fc1_alpha,
            fc2_weight,
            fc2_weight_sf,
            topk_scores,
            fc1_done_counter,
            offs,
            sched_params,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.d_smem_layout_staged,
            token_comm_args=token_comm_args,
        ).launch(
            grid=sched_params.get_grid_shape(max_active_clusters),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.jit
    def _decode_weight_tile(
        self, weight, scale, sB, identity, expert, tile_n, k_tile, stage, weight_thread
    ):
        # B's MMA partition describes both the CTA slice and its SMEM layout.
        # Each lane decodes one complete per-16 block, sharing its scale load.
        atom_n = cute.size(sB.shape[0][0])
        atom_k = cute.size(sB.shape[0][1])
        rows = atom_n * sB.shape[1]
        cols = atom_k * sB.shape[2]
        blocks_per_row = cols // 16
        for block in cutlass.range(
            weight_thread, rows * blocks_per_row, self.num_weight_threads, unroll=1
        ):
            row = block // blocks_per_row
            col = block % blocks_per_row * 16
            coord = ((row % atom_n, col % atom_k), row // atom_n, col // atom_k)
            logical = identity[coord]
            n = tile_n * self.cta_tile_shape_mnk[1] + logical[0]
            k = k_tile * self.mma_tiler[2] + logical[1]
            lo = cutlass.Uint32(0)
            hi = cutlass.Uint32(0)
            sf = cutlass.Uint32(0)
            if n < weight.shape[1] and k < weight.shape[2] * 2:
                offset = (cutlass.Int64(expert) * weight.shape[1] + n) * weight.shape[
                    2
                ] + k // 2
                packed = cute.make_ptr(
                    cutlass.Uint32,
                    weight.iterator.toint() + offset,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                lo = packed[0]
                hi = packed[1]
                sf_offset = (cutlass.Int64(expert) * scale.shape[1] + n) * scale.shape[
                    2
                ] + k // 16
                sf_ptr = cute.make_ptr(
                    cutlass.Uint8,
                    scale.iterator.toint() + sf_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=1,
                )
                sf = cutlass.Uint32(sf_ptr[0])
            decoded = e2m1x16_e4m3_to_bf16x16(lo, hi, sf)
            for element in cutlass.range_constexpr(16):
                kc = col + element
                sB[
                    ((row % atom_n, kc % atom_k), row // atom_n, kc // atom_k, stage)
                ] = decoded[element]

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_activation_1: cute.CopyAtom,
        tma_tensor_fc1_activation_1: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
        tma_atom_fc2_activation: cute.CopyAtom,
        tma_tensor_fc2_activation: cute.Tensor,
        # GEMM-domain tensors (fc1)
        activation_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2)
        fc2_output_gemm: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        # topk + cross-phase sync workspace
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        # Scheduling
        offs: Optional[cute.Tensor],
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        d_smem_layout_staged: Optional[Union[cute.Layout, cute.ComposedLayout]],
        c_smem_layout_staged: Optional[Union[cute.Layout, cute.ComposedLayout]] = None,
        tma_atom_c: Optional[cute.CopyAtom] = None,
        tma_tensor_c: Optional[cute.Tensor] = None,
        token_comm_args=None,
    ):
        """Device kernel for the fused fc1+fc2 GLU BF16 grouped GEMM.

        Lean (``force_static_sched=True``) path: 7-warp specialization with
        no empty / drain_aux warps and no expert-wise TMA desc rewriting.

        Epilogue is fully owned by ``self.epilogue.run(...)`` -- the four epi
        warps make a single call that drives the entire 2-phase task-tile
        loop (acc consumer state, subtile dispatch, TMA commit/drain, and
        the piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``).
        """
        cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        cute.slice_(b_smem_layout_staged, (None, None, None, 0))

        # fc2 waits for all fc1 intermediate N-tiles in the same token block.
        # Each N-tile is processed by atom_thr_size CTAs (both CTA0 and CTA1 increment
        # the counter), so the threshold must account for both CTAs' contributions.
        ext_fc2_spin_threshold = (
            (self.intermediate_gateup + self.cta_tile_shape_mnk[1] - 1)
            // self.cta_tile_shape_mnk[1]
            * self.epilogue._atom_thr_size
        )

        ext = GluBf16Fc12SchedExtension(
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
            cluster_m=self.epilogue._atom_thr_size,
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # SharedStorage.
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(
            sched_params, ext, num_drain_warps=0
        )

        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            sched_storage: SchedStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # MegaMoE-only dispatch-warp SMEM (pull_buffer, mbarriers, etc.).
        # Kept out of ``SharedStorage`` so the lean path never allocates it.
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None

        # ── Pipelines: separate producer/consumer groups for A and B. ──

        a_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        a_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mcast_ctas_a
        )
        a_producer, a_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=a_pipeline_consumer_group,
            tx_count=self.num_tma_load_a_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        b_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_weight_threads * self.atom_thr_size
        )
        b_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        b_producer, b_consumer = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            len(self.epilogue_warp_id) * 32 * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Sched
        num_sched_consumer_threads = 32 * len(
            (
                self.tma_a_warp_id,
                *self.weight_warp_ids,
                self.mma_warp_id,
                *self.epilogue_warp_id,
            )
        )
        scheduler = SchedCls.create(
            sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            sched_storage=storage.sched_storage,
            num_consumer_threads=num_sched_consumer_threads,
            ext=ext,
        )
        sched_consumer = scheduler.make_consumer()

        # Issue the first scheduler claim before cluster init wait so the
        # atomic/offsets latency overlaps with pipeline setup.
        # Under MegaMoE + static load-balance, ``internal_init`` walks
        # per-expert sizes from ``expert_recv_count_sum`` -- those are not
        # valid until the dispatch barrier completes, so we defer init to the
        # sched warp (after ``token_comm_hook_sched_warp_pre_init_wait``).
        early_internal_init = (self.load_balance_mode == "atomic_counter") or (
            not self.enable_token_comm
        )

        if cutlass.const_expr(early_internal_init):
            scheduler.internal_init(
                warp_idx=warp_idx,
                sched_warp_id=self.sched_warp_id,
            )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ── SMEM tensors A / B (shared by fc1 / fc2) ──
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # acc_fake layout: (MMA, MMA_M, MMA_N, STAGE).  The two acc stages
        # tile TMEM back-to-back (2 x 256 cols = the full 512-col budget).
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before TMEM alloc.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        mma_tiler_k = self.mma_tiler[2]
        # ``self.hidden`` / ``self.intermediate_downproj``
        # both resolve to ``hidden`` / ``intermediate_downproj``.  Under
        # ``static_expert_shape`` they are codegen-time Python ints
        # (rewritten on ``fc1_weight`` / ``fc2_weight`` at ``__call__``
        # entry); otherwise they are runtime Int32 from tensor metadata.
        # The arithmetic below folds to an immediate in the static path.
        k_tile_cnt_fc1 = (self.hidden + mma_tiler_k - 1) // mma_tiler_k
        k_tile_cnt_fc2 = (self.intermediate_downproj + mma_tiler_k - 1) // mma_tiler_k
        # fc2 spin threshold: each fc1 N-tile (intermediate direction) is incremented
        # by atom_thr_size CTAs.  In non-swap-AB, self.intermediate_gateup is the N
        # dimension (intermediate_gateup), so divide by cta_tile_shape_mnk[1] (N tile),
        # not cta_tile_shape_mnk[0] (M/token tile).  Matches ext_fc2_spin_threshold.
        fc2_spin_threshold = (
            (self.intermediate_gateup + self.cta_tile_shape_mnk[1] - 1)
            // self.cta_tile_shape_mnk[1]
        ) * self.epilogue._atom_thr_size

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp (warp 7) — lean path
        # ════════════════════════════════════════════════════════════════════
        if warp_idx == self.sched_warp_id:
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            if cutlass.const_expr(not early_internal_init):
                scheduler.internal_init(
                    warp_idx=warp_idx,
                    sched_warp_id=self.sched_warp_id,
                )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                ext.prefetch_for_expert(scheduler.current_work.expert_idx)
                scheduler.publish_work()
                scheduler.gen_next_work()
            # Sentinel publish (current_work is already invalid here).
            scheduler.publish_work()
            scheduler.produce_tail()

        # ════════════════════════════════════════════════════════════════════
        # TMA load warps (warps 5 / 6)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A loads activations into the A pipeline (and waits for fc1
        # workspace readiness in the fc2 phase).
        # TMA-B loads weights into the B pipeline.

        # ── TMA-A warp (warp 5) ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            _iket_active = tidx == cutlass.Int32(160)
            a_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )

            # non-swap-AB FC1: activation (A) is partitioned per-CTA (like original B).
            # b_cta_layout=(2,) and mcast_mode=1 → each CTA loads its own token range.
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
            cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )
                if is_phase_linear1:
                    # ── fc1 phase A-side (non-swap-AB: A=activations, per-CTA partitioned) ──
                    # Activations are split per CTA (tokens 0-127 for CTA 0, 128-255 for CTA 1).
                    # Use b_cta_layout + m-coord so each CTA loads its own token range.
                    # mcast_mode=1 → same M-coord = only self → no actual multicast. ✓
                    if _iket_active:
                        iket.range_push("tma_token_fc1")
                    # MegaMoE: spin until the dispatch warps have pulled this
                    # task tile's token activations into the L1 token buffer.
                    # No-op on the lean path (activations resident at launch).
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args,
                        work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc1_activation",
                        tma_tensor_fc1_activation_1,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]

                    a_producer.reset()
                    peek_a_empty_status = a_producer.try_acquire()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(peek_a_empty_status)
                        peek_a_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_a_empty_status = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_activation_1,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                else:
                    # ── fc2 phase A-side: load fc1_output (M=tokens) + wait for fc1 done ──
                    #
                    # Non-swap-AB fc2: A=fc1_output (M=tokens). tile_m_idx is the
                    # CTA-level token block. Counter wait moved here from TMA-B.
                    if _iket_active:
                        iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile_info.cumulative_token_block_count
                        + work_tile_info.tile_m_idx
                        // cutlass.Int32(self.epilogue._atom_thr_size)
                    )
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    # Always spin (no peek shortcut) to guarantee counter=4 in this warp,
                    # then use acquire semantics + cross-proxy fence to ensure fc1_output
                    # writes (from generic proxy) are visible to the TMA async proxy load.
                    if _iket_active:
                        iket.range_push("tma_token_fc2_a_wait")
                    spin_wait(
                        counter_ptr,
                        lambda v: v >= fc2_spin_threshold,
                        fail_sleep_cycles=20,
                    )
                    if _iket_active:
                        iket.range_pop()
                    cute.arch.load(
                        counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu"
                    )
                    cute.arch.fence_proxy("async")
                    cute.arch.fence_proxy("async.global")

                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc2_activation",
                        tma_tensor_fc2_activation,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_activation,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )

                    # fc2 A-side = fc1_output (M=tokens). tAgA is cluster-level
                    # indexed, so divide tile_m_idx by cluster_m to get the
                    # cluster block index — same formula as fc1 A-side.
                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]

                    a_producer.reset()
                    peek_a_empty_status = a_producer.try_acquire()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(peek_a_empty_status)
                        peek_a_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_a_empty_status = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_activation,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )

                if _iket_active:
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            a_producer.tail()

        # Weight producer: four warps decode the selected expert online.
        if warp_idx == 6 or (warp_idx >= 12 and warp_idx <= 14):
            weight_thread = cute.arch.lane_idx()
            if warp_idx >= 12:
                weight_thread += (warp_idx - 11) * 32
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            identity = thr_mma.partition_B(
                cute.make_identity_tensor((self.mma_tiler[1], self.mma_tiler[2]))
            )
            work_tile_info = sched_consumer.consume_work()
            while work_tile_info.is_valid_tile:
                b_producer.reset()
                if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                    for k_tile in cutlass.range(k_tile_cnt_fc1, unroll=1):
                        handle = b_producer.acquire_and_advance()
                        self._decode_weight_tile(
                            fc1_weight,
                            fc1_weight_sf,
                            sB,
                            identity,
                            work_tile_info.expert_idx,
                            work_tile_info.tile_n_idx,
                            k_tile,
                            handle.index,
                            weight_thread,
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        handle.commit()
                else:
                    for k_tile in cutlass.range(k_tile_cnt_fc2, unroll=1):
                        handle = b_producer.acquire_and_advance()
                        self._decode_weight_tile(
                            fc2_weight,
                            fc2_weight_sf,
                            sB,
                            identity,
                            work_tile_info.expert_idx,
                            work_tile_info.tile_n_idx,
                            k_tile,
                            handle.index,
                            weight_thread,
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        handle.commit()
                work_tile_info = sched_consumer.consume_work()
            b_producer.tail()

        # ════════════════════════════════════════════════════════════════════
        # MMA warp (warp 4)
        # ════════════════════════════════════════════════════════════════════
        #
        # Both phases share tiled_mma and TMEM; only K-tile count differs.
        if warp_idx == self.mma_warp_id:
            _iket_active = tidx == cutlass.Int32(128)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_base = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            # K-tile counts ``k_tile_cnt_fc1`` / ``k_tile_cnt_fc2`` come
            # from the enclosing scope (computed once before the TMA warps).

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )
                # Prebind k_tile_cnt due to DSL AST.
                k_tile_cnt = cutlass.Int32(0)
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                    if _iket_active:
                        iket.range_push("mma_fc1")
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    if _iket_active:
                        iket.range_push("mma_fc2")

                acc_stage_index = acc_producer_state.index

                if is_leader_cta:
                    tCtAcc = acc_base[(None, None, None, acc_stage_index)]

                    if _iket_active:
                        iket.range_push("mma_ab_wait")
                    a_consumer.reset()
                    b_consumer.reset()
                    peek_a_full_status = cutlass.Boolean(1)
                    peek_b_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_a_full_status = a_consumer.try_wait()
                        peek_b_full_status = b_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)
                    if _iket_active:
                        iket.range_pop()

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle_a = a_consumer.wait_and_advance(peek_a_full_status)
                        handle_b = b_consumer.wait_and_advance(peek_b_full_status)
                        peek_a_full_status = cutlass.Boolean(1)
                        peek_b_full_status = cutlass.Boolean(1)
                        if handle_a.count + 1 < k_tile_cnt:
                            peek_a_full_status = a_consumer.try_wait()
                            peek_b_full_status = b_consumer.try_wait()

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, None, handle_a.index)],
                            tCrB[(None, None, None, handle_b.index)],
                            tCtAcc,
                        )
                        handle_a.release()
                        handle_b.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                if _iket_active:
                    iket.range_pop()

                work_tile_info = sched_consumer.consume_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ── sD SMEM (fc1 TMA-output staging; fc2 doesn't use it) ──
        # Direct STG consumes the BF16 registers directly, so keeping this
        # allocation would waste exactly one A+B stage for the baseline shape.
        if cutlass.const_expr(self.use_stg_fc1):
            sD = None
        else:
            sD = smem.allocate_tensor(
                element_type=self.fc1_output_dtype,
                layout=d_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=d_smem_layout_staged.inner,
            )

        # ── sC SMEM (raw gate+up Float32, ping-pong; only when generate_c=True) ──
        if cutlass.const_expr(self.generate_c):
            sC = smem.allocate_tensor(
                element_type=self.epilogue._c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps (warps 0-3)
        # ════════════════════════════════════════════════════════════════════
        #
        # Fully delegated to ``self.epilogue.run(...)`` -- the epilogue owns
        # the entire 2-phase task-tile loop.
        if warp_idx < self.mma_warp_id:
            epi_warp_idx = warp_idx

            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_tensor = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            # Build common kwargs for the epilogue.
            if cutlass.const_expr(self.generate_c):
                _smem_c_raw_arg = sC
                _tma_atom_c_arg = tma_atom_c
                _gmem_c_arg = tma_tensor_c
            else:
                _smem_c_raw_arg = None
                _tma_atom_c_arg = None
                _gmem_c_arg = None
            if cutlass.const_expr(self.use_stg_fc1):
                _gmem_fc1_output_arg = fc1_output_gemm
            else:
                _gmem_fc1_output_arg = tma_tensor_fc1_output
            _run_kwargs = dict(
                tmem_acc_tensor=acc_tensor,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                smem_fc1_output_buffer=sD,
                tma_atom_fc1_output=tma_atom_fc1_output,
                gmem_fc1_output=_gmem_fc1_output_arg,
                gmem_topk_scores=topk_scores,
                alpha=fc1_alpha,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                smem_c_buffer=_smem_c_raw_arg,
                tma_atom_c=_tma_atom_c_arg,
                gmem_c=_gmem_c_arg,
                warp_idx=epi_warp_idx,
                tidx=tidx,
            )

            # MegaMoE: pass token_comm_args only when it is a real bundle (not
            # None).  Passing Python None explicitly to @cute.jit methods
            # triggers a CuteDSL codegen issue; const_expr dispatch avoids any
            # None-as-JIT-argument path.
            if cutlass.const_expr(token_comm_args is not None):
                self.epilogue.run(**_run_kwargs, token_comm_args=token_comm_args)
            else:
                self.epilogue.run(**_run_kwargs)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.fence_acq_rel_sys()

        # ════════════════════════════════════════════════════════════════════
        # Dispatch warps hook (warps 8-11; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` → warps 8-11 don't exist (threads_per_cta
        # = 256), so the guard is const_expr-eliminated in the lean path.
        if cutlass.const_expr(self.enable_token_comm):
            if (
                warp_idx >= self.dispatch_warp_id[0]
                and warp_idx <= self.dispatch_warp_id[-1]
            ):
                lane_idx_for_dispatch = cute.arch.lane_idx()
                if cutlass.const_expr(self.token_back_standalone):
                    if warp_idx < self.token_back_warp_id[0]:
                        self.token_comm_hook_dispatch_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                    else:
                        self.token_comm_hook_token_back_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                else:
                    self.token_comm_hook_dispatch_warp_body(
                        token_comm_args,
                        token_comm_storage,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx_for_dispatch,
                        tidx=tidx,
                    )

            # ════════════════════════════════════════════════════════════════════
            # Kernel tail hook (MegaMoE-only; lean base = no-op)
            # ════════════════════════════════════════════════════════════════════
            lane_idx = cute.arch.lane_idx()
            self.token_comm_hook_kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
