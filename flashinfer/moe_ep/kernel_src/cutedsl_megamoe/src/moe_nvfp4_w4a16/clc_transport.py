# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-grid transport launches surrounding elastic W4A16 CLC compute."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cutlass_dsl import Int32, Int64

from src.token_comm import TokenInPullTokenBackPush


class _FixedGridTokenComm(TokenInPullTokenBackPush):
    @cute.jit
    def _cta_linear_id(self):
        # Transport has a fixed grid; its physical x-fastest IDs are dense
        # independently of the compute scheduler's swap-AB coordinate layout.
        x, y, z = cute.arch.block_idx()
        grid_x, grid_y, _ = cute.arch.grid_dim()
        return Int32(x) + Int32(grid_x) * (Int32(y) + Int32(grid_y) * Int32(z))


class ClcTransport:
    """Reuse vendor transport with a separate 128-thread participant group.

    Dispatch, compute and finish must use the same stream and argument storage.
    Only finish performs the existing final shared/local counter reset.
    """

    def __init__(self, template: TokenInPullTokenBackPush):
        self.comm = _FixedGridTokenComm(
            world_size=template.world_size,
            num_topk=template.num_topk,
            num_experts_per_rank=template.num_experts_per_rank,
            num_total_experts=template.num_total_experts,
            hidden=template.hidden,
            fc1_token_dtype=template.fc1_token_dtype,
            sf_uint32_per_token=template.sf_uint32_per_token,
            token_padding_block=template.token_padding_block,
            sf_padding_block=template.sf_padding_block,
            cluster_tile_tokens=template.cluster_tile_tokens,
            cluster_shape_mn=template.cluster_shape_mn,
            dispatch_warp_start=0,
            num_other_warps=0,
            combine_format=template.combine_format,
            token_back_by_dispatch=template.token_back_by_dispatch,
            fc2_publishes_per_token_cluster_tile=(
                template.fc2_publishes_per_token_cluster_tile
            ),
            token_back_reduce_topk=template.token_back_reduce_topk,
            token_back_standalone=False,
            flag_batch=template._flag_batch,
            is_swap_ab=template.is_swap_ab,
            sf_atom_swizzled=template.sf_atom_swizzled,
            token_back_schedule_mode=template.token_back_schedule_mode,
        )

    @cute.jit
    def dispatch(
        self,
        comm_args,
        *,
        grid: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        stream,
    ):
        assert grid[0] * grid[1] * grid[2] == comm_args.sm_count
        self._dispatch_kernel(comm_args).launch(
            grid=grid,
            block=(128, 1, 1),
            cluster=cluster,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def finish(
        self,
        comm_args,
        *,
        grid: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        stream,
    ):
        assert grid[0] * grid[1] * grid[2] == comm_args.sm_count
        self._finish_kernel(comm_args).launch(
            grid=grid,
            block=(128, 1, 1),
            cluster=cluster,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def _dispatch_kernel(self, args):
        comm = self.comm
        smem = utils.SmemAllocator()
        storage = smem.allocate(comm.extra_smem_storage_class())
        cta = comm._cta_linear_id()
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32

        comm.dispatch_prep(
            storage,
            args.topk_idx,
            args.expert_send_count,
            args.src_token_topk_idx,
            args.peer_rank_ptr_mapper,
            cta,
            warp,
            lane,
            local_rank=args.local_rank,
            num_tokens=args.input_token_buffer.shape[0],
            num_sms=args.sm_count,
        )
        comm.dispatch_barrier(
            args.expert_send_count,
            args.expert_recv_count,
            args.expert_recv_count_sum,
            args.nvlink_barrier_signal,
            args.grid_sync_counter,
            args.peer_rank_ptr_mapper,
            cta,
            warp,
            lane,
            local_rank=args.local_rank,
            num_sms=args.sm_count,
            nvlink_barrier_counter=args.nvlink_barrier_counter,
        )
        comm.dispatch_pull(
            storage,
            args.input_token_buffer,
            args.input_sf_buffer,
            args.input_topk_weights_buffer,
            args.src_token_topk_idx,
            args.expert_recv_count,
            args.expert_recv_count_sum,
            args.fc1_input_token_buffer,
            args.fc1_input_sf_buffer,
            args.fc1_input_topk_weights_buffer,
            args.fc1_ready_counter,
            args.token_src_metadata,
            args.peer_rank_ptr_mapper,
            cta,
            warp,
            lane,
            num_sms=args.sm_count,
        )
        # Kernel completion on the same stream replaces the fused NB9
        # dispatch-to-scheduler handshake. Counters remain live for compute.

    @cute.kernel
    def _finish_kernel(self, args):
        comm = self.comm
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        cta = comm._cta_linear_id()

        if cutlass.const_expr(comm.enable_token_back):
            smem = utils.SmemAllocator()
            storage = smem.allocate(comm.extra_smem_storage_class())
            mbar = storage.pull_mbar.data_ptr()
            if lane == Int32(0):
                cute.arch.mbarrier_init(mbar + warp, 1)
            cute.arch.sync_warp()

            # Reconstruct the existing return walk from retained global
            # counts; dispatch's shared memory and phase do not cross launches.
            count_per_lane: cutlass.Constexpr = (comm.num_experts_per_rank + 31) // 32
            stored_counts = []
            for _ in cutlass.range_constexpr(count_per_lane):
                stored_counts.append(Int32(0))
            for i in cutlass.range_constexpr(count_per_lane):
                expert = Int32(i * 32) + lane
                if expert < Int32(comm.num_experts_per_rank):
                    packed = args.expert_recv_count_sum[expert]
                    stored_counts[i] = Int32(Int64(packed) & Int64(0xFFFFFFFF))
            cute.arch.sync_warp()

            comm.token_back_by_push(
                storage.pull_buffer.data_ptr(),
                mbar,
                args.fc2_output_workspace,
                args.fc2_done_counter,
                args.token_src_metadata,
                args.combine_output,
                args.combine_sf,
                args.fc2_output_sf,
                args.token_back_schedule_counter,
                args.peer_rank_ptr_mapper,
                Int32(0),
                stored_counts,
                cta,
                warp,
                lane,
                local_rank=args.local_rank,
                num_sms=args.sm_count,
                chunk_bytes=comm.hidden_bytes,
            )

        # epi_warps already wrote the output from FC2. Both return modes
        # retain the original final drain, shared publish and local reset.
        comm.kernel_tail(args, warp_idx=warp, lane_idx=lane, tidx=tid)
