# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""BF16 token pulling with pool writes overlapping the next input transfer."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    GpuReleaseFlagBatchTracker,
    TokenInPullTokenBackPush,
    TokenSrcMetadata,
    fns_b32,
    iket as _iket,
    ldg_f32_raw,
    tma_load_1d_raw,
    tma_store_1d,
)


class _Bf16TokenComm(TokenInPullTokenBackPush):
    """Keep universal metadata, storage and return protocols; specialize BF16 pull."""

    @cute.jit
    def dispatch_pull(
        self,
        token_comm_storage,
        input_token_buffer,
        input_sf_buffer,
        input_topk_weights_buffer,
        src_token_topk_idx,
        expert_recv_count,
        expert_recv_count_sum,
        fc1_input_token_buffer,
        fc1_input_sf_buffer,
        fc1_input_topk_weights_buffer,
        fc1_ready_counter,
        token_src_metadata,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
    ):
        assert self.fc1_token_dtype is cutlass.BFloat16
        assert self.sf_uint32_per_token == 0
        assert input_sf_buffer is None and fc1_input_sf_buffer is None

        # Each warp reuses one inherited BF16 pull buffer.
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
        cute.arch.sync_warp()

        phase_bit = Int32(0)

        current_expert_idx = Int32(-1)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_task_tile_offset = Int32(0)

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=Int32(0),
            phase=Int32(0),
            tid=lane_idx,
        )

        stored_rank_count_lane = Int32(0)

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
        cute.arch.sync_warp()

        lane_received_tokens = Int32(0)
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            lane_received_tokens = (
                lane_received_tokens + stored_num_tokens_per_expert[i]
            )
        total_received_tokens = Int32(
            cute.arch.warp_redux_sync(lane_received_tokens, "add")
        )

        num_global_warps: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps
        token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

        _iket_pull_emit = (
            (sm_idx == Int32(0)) and (warp_idx == Int32(0)) and (lane_idx == Int32(0))
        )

        while token_idx < total_received_tokens:
            if _iket_pull_emit:
                _iket.range_push("Pull.ChooseToken")
            old_expert_idx = current_expert_idx
            # The outer total bound guarantees a containing expert.
            while token_idx >= expert_end_idx:
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count
                # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                prev_task_tile_count = (
                    prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                ) // Int32(self.cluster_tile_tokens)
                expert_task_tile_offset = expert_task_tile_offset + prev_task_tile_count
                current_expert_idx = current_expert_idx + Int32(1)
                expert_start_idx = expert_end_idx
                valid_value = Int32(0)
                for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                    if current_expert_idx == Int32(i * self.warp_threads) + lane_idx:
                        valid_value = stored_num_tokens_per_expert[i]
                total_for_expert = cute.arch.shuffle_sync(
                    valid_value, current_expert_idx % Int32(self.warp_threads)
                )
                expert_end_idx = expert_end_idx + total_for_expert

            if old_expert_idx != current_expert_idx:
                if lane_idx < Int32(self.world_size):
                    stored_rank_count_lane = Int32(
                        expert_recv_count[lane_idx, current_expert_idx]
                    )
                else:
                    stored_rank_count_lane = Int32(0)

            token_idx_in_expert = token_idx - expert_start_idx
            slot_idx = token_idx_in_expert
            offset = Int32(0)
            remaining_lane = stored_rank_count_lane

            current_rank_in_expert_idx = Int32(0)
            token_idx_in_rank = Int32(0)

            decided = Int32(0)
            # Stop rank peeling as soon as the containing round is found.
            while decided == Int32(0):
                active = remaining_lane > Int32(0)
                mask = cute.arch.vote_ballot_sync(active)
                num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                v_for_min = Int32(0x7FFFFFFF)
                if active:
                    v_for_min = remaining_lane
                length = Int32(cute.arch.warp_redux_sync(v_for_min, "min"))

                # A valid remaining slot always has an active source rank.
                num_round_tokens = length * num_active_ranks
                if slot_idx < num_round_tokens:
                    slot_idx_in_round = slot_idx % num_active_ranks
                    current_rank_in_expert_idx = fns_b32(
                        Int32(mask),
                        Int32(0),
                        slot_idx_in_round + Int32(1),
                    )
                    token_idx_in_rank = offset + (slot_idx // num_active_ranks)
                    decided = Int32(1)
                else:
                    slot_idx = slot_idx - num_round_tokens
                    offset = offset + length
                    if remaining_lane > length:
                        remaining_lane = remaining_lane - length
                    else:
                        remaining_lane = Int32(0)

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.ChooseToken
                _iket.range_push("Pull.TMA_NVLink_Roundtrip")

            src_token_topk = Uint32(
                src_token_topk_idx[
                    current_expert_idx,
                    current_rank_in_expert_idx,
                    token_idx_in_rank,
                ]
            )
            src_token = Int32(src_token_topk // Uint32(self.num_topk))
            src_topk = Int32(src_token_topk % Uint32(self.num_topk))

            cur_peer_offset = peer_rank_ptr_mapper.map(
                Int64(0), current_rank_in_expert_idx, Int64(0)
            )
            inp_tok_local_base = input_token_buffer.iterator.toint()
            inp_w_local_base = input_topk_weights_buffer.iterator.toint()

            # Mapping and tracker bookkeeping can overlap the preceding store's
            # SMEM read. Complete that read only before overwriting its buffer.
            if cutlass.const_expr(self._flag_batch > 1):
                cute.arch.cp_async_bulk_wait_group(0, read=True)

            with cute.arch.elect_one():
                pull_buffer_warp_ptr = pull_buffer_ptr + (
                    warp_idx * Int32(self.hidden_bytes)
                )
                tma_src_addr = (
                    inp_tok_local_base
                    + cur_peer_offset
                    + Int64(src_token * Int32(self.hidden_bytes))
                )
                tma_load_1d_raw(
                    pull_buffer_warp_ptr,
                    tma_src_addr,
                    pull_mbar_ptr + warp_idx,
                    Int32(self.hidden_bytes),
                )

            if _iket_pull_emit:
                _iket.range_push("Pull.SF_LDG_STG")

            pool_token_idx = (
                expert_pool_block_offset * Int32(self.token_padding_block)
                + token_idx_in_expert
            )
            weight_addr = (
                inp_w_local_base
                + cur_peer_offset
                + Int64((src_token * Int32(self.num_topk) + src_topk) * Int32(4))
            )
            weight = ldg_f32_raw(weight_addr)

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                _iket.range_push("Pull.Weight_LDG")  # (= ST phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                )
                cute.arch.mbarrier_wait(
                    pull_mbar_ptr + warp_idx,
                    phase_bit,
                )

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                _iket.range_push("Pull.TMA_Store")

            # The previous pool write overlaps this token's pull and mapping.
            # Drain it before issuing another store. F1 has already drained
            # every write to publish that token immediately.
            if cutlass.const_expr(self._flag_batch > 1):
                cute.arch.cp_async_bulk_wait_group(0)
            with cute.arch.elect_one():
                pull_buffer_warp_ptr = pull_buffer_ptr + (
                    warp_idx * Int32(self.hidden_bytes)
                )
                tma_store_1d(
                    fc1_input_token_buffer.iterator
                    # T=128k) × self.hidden_bytes overflows int32 (max 2.1 G).
                    # 64-bit address math is required for large token pools.
                    + (Int64(pool_token_idx) * Int64(self.hidden_bytes)),
                    pull_buffer_warp_ptr,
                    Int32(self.hidden_bytes),
                )

                TokenSrcMetadata(
                    src_rank=current_rank_in_expert_idx,
                    src_token=src_token,
                    src_topk=src_topk,
                ).store(
                    token_src_metadata.iterator
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )

                cute.arch.cp_async_bulk_commit_group()
                fc1_input_topk_weights_buffer[pool_token_idx] = weight

            # Full batches require completed writes before publication. The
            # next pull waits for partial-batch source reads before buffer reuse.
            if cutlass.const_expr(self._flag_batch == 1):
                cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_warp()
            else:
                if flag_tracker.cumulated_flags + Int32(1) == Int32(self._flag_batch):
                    cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.sync_warp()

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.TMA_Store
                _iket.range_push("Pull.Arrival_Atomic")

            task_tile_idx = expert_task_tile_offset + (
                token_idx_in_expert // Int32(self.cluster_tile_tokens)
            )

            task_tile_addr = (fc1_ready_counter.iterator + task_tile_idx).toint()
            flag_tracker = flag_tracker.accumulate(
                Int32(0),
                self._flag_batch,
                task_tile_addr,
            )

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.Arrival_Atomic

            phase_bit = phase_bit ^ Int32(1)

            token_idx = token_idx + Int32(num_global_warps)

        # F1 already drains and publishes every token inside the loop.
        if cutlass.const_expr(self._flag_batch > 1):
            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_warp()
        flag_tracker.fire()
        cute.arch.sync_warp()

        return phase_bit, stored_num_tokens_per_expert
