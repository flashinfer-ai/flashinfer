# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""W4A16 tail phases for overlapping deterministic reduction with cleanup."""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import TokenInPullTokenBackPush


class W4A16TokenComm(TokenInPullTokenBackPush):
    """Reuse transport primitives and own the split tail used by W4A16.

    The drain completes all FC2 returns before retired warp groups reduce.
    Cleanup touches counter prefixes only, so it can overlap that reduction.
    """

    @cute.jit
    def stage_inputs(
        self,
        hidden: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
    ):
        """Stage live inputs from dispatch warps before routing preparation."""
        cta_linear_id = self._cta_linear_id()
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
        dispatch_thread = local_warp_idx * self.warp_threads + lane_idx
        chunks_per_token: cutlass.Constexpr[int] = self.hidden_bytes // 16
        live_chunks = hidden.shape[0] * chunks_per_token
        capacity = token_comm_args.input_token_buffer.shape[0]
        vector_source: cutlass.Constexpr[bool] = (
            cute.is_static(hidden.stride)
            and hidden.stride == (self.hidden_bytes // 2, 1)
            and hidden.iterator.alignment >= 16
        )
        chunk = cta_linear_id * self.num_dispatch_threads + dispatch_thread
        chunk_stride = token_comm_args.sm_count * self.num_dispatch_threads
        if chunk < live_chunks:
            if cutlass.const_expr(vector_source):
                # Keep flat 16-byte copies for proven contiguous sources.
                source = cute.make_tensor(
                    hidden.iterator, cute.make_layout((8, live_chunks), stride=(1, 8))
                )
            target = cute.make_tensor(
                token_comm_args.input_token_buffer.iterator,
                cute.make_layout((8, capacity * chunks_per_token), stride=(1, 8)),
            )
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            while chunk < live_chunks:
                values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                if cutlass.const_expr(vector_source):
                    cute.copy(atom, source[None, chunk], values)
                else:
                    row = chunk // chunks_per_token
                    source_chunks = cute.zipped_divide(hidden[row, None], (8,))
                    cute.autovec_copy(
                        source_chunks[(None,), (chunk % chunks_per_token,)], values
                    )
                cute.copy(atom, values, target[None, chunk])
                chunk += chunk_stride
            # Each writer publishes to the async proxy before the existing
            # dispatch grid/NVLink publication permits remote TMA pulls.
            cute.arch.fence_proxy("async.global")

        # Match dispatch_prep's ownership exactly: each lane later reads its
        # own staged IDs, so preparation needs no additional grid rendezvous.
        tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
        active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
        base_token_for_warp = (
            cta_linear_id * self.num_dispatch_warps + local_warp_idx
        ) * tokens_per_warp
        grid_token_stride = (
            token_comm_args.sm_count * self.num_dispatch_warps * tokens_per_warp
        )
        t = base_token_for_warp
        while t < capacity:
            token_global = t + lane_idx // self.num_topk
            if lane_idx < active_lanes and token_global < capacity:
                topk_slot = lane_idx % self.num_topk
                if token_global < hidden.shape[0]:
                    token_comm_args.topk_idx[token_global, topk_slot] = cutlass.Int64(
                        ids[token_global, topk_slot]
                    )
                    token_comm_args.input_topk_weights_buffer[
                        token_global, topk_slot
                    ] = weights[token_global, topk_slot]
                else:
                    token_comm_args.topk_idx[token_global, topk_slot] = cutlass.Int64(
                        -1
                    )
            t += grid_token_stride

    @cute.jit
    def kernel_tail_drain(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
    ):
        """Drain cross-rank returns after the caller's CTA rendezvous."""
        if (warp_idx >= self.dispatch_warp_start) and (
            warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
        ):
            cta_linear_id = self._cta_linear_id()
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            # Normal launches retain input publication and FC2 drain. Kernel
            # replay needs four calls so the cross-device signal self-cancels:
            # input publication, this padding call, drain and cleanup publication.
            # MEGA_USE_NCU selects the replay protocol at compile time.
            if cutlass.const_expr(os.environ.get("MEGA_USE_NCU", "0") == "1"):
                self.nvlink_barrier(
                    token_comm_args.nvlink_barrier_signal,
                    token_comm_args.nvlink_barrier_counter,
                    token_comm_args.grid_sync_counter,
                    token_comm_args.peer_rank_ptr_mapper,
                    cta_linear_id,
                    local_warp_idx,
                    lane_idx,
                    num_sms=token_comm_args.sm_count,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=True,
                )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )

    @cute.jit
    def kernel_tail_cleanup(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
    ):
        if (warp_idx >= self.dispatch_warp_start) and (
            warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
        ):
            cta_linear_id = self._cta_linear_id()
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            if cutlass.const_expr(os.environ.get("MEGA_USE_NCU", "0") == "1"):
                # Replay can restore the bank selector without restoring peer
                # counters, so preserve the active-bank tail reset in this mode.
                self.tail_reset_counters(
                    token_comm_args,
                    token_comm_args.shared_zero_prefix,
                    cta_linear_id=cta_linear_id,
                    local_warp_idx=local_warp_idx,
                    lane_idx=lane_idx,
                )
                # Publish every rank's clear before replay can reuse the
                # restored active bank and write another rank's counters.
                self.nvlink_barrier(
                    token_comm_args.nvlink_barrier_signal,
                    token_comm_args.nvlink_barrier_counter,
                    token_comm_args.grid_sync_counter,
                    token_comm_args.peer_rank_ptr_mapper,
                    cta_linear_id,
                    local_warp_idx,
                    lane_idx,
                    num_sms=token_comm_args.sm_count,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=True,
                )
            # Input publication and drain use four grid phases, leaving the
            # grid counter zero before this rank-local reset.
            self.tail_reset_counters(
                token_comm_args,
                token_comm_args.local_zero_prefix,
                cta_linear_id=cta_linear_id,
                local_warp_idx=local_warp_idx,
                lane_idx=lane_idx,
            )
