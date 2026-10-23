# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overwrite-only count exchange and a single drain barrier for MXFP8 MegaMoE.

Kept outside the vendored helper so other backends retain their reset protocol.
"""

import os
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.grid_sync import (
    software_grid_sync,
)
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.ptx_helpers import (
    fns_b32,
    ldg_b32_raw,
    ldg_f32_raw,
    tma_load_1d_raw,
    tma_store_1d,
    stg_b64_raw,
    red_add_release_sys_s32_raw,
)

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.flag_batch import (
    GpuReleaseFlagBatchTracker,
)
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.sf_swizzle import (
    sf_atom_int32_offset,
)
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.src.src.token_comm import (
    TokenSrcMetadata,
    _iket,
)

from .wait_stats import Slot, SLOTS, globaltimer_lo


@dsl_user_op
def _startup_phase_assert(actual, expected, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [actual.ir_value(), expected.ir_value()],
        "{ .reg .pred p; setp.ne.s32 p, $0, $1; @p trap; }",
        "r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def overwrite_count_token_comm_class(base: Any) -> type:
    class OverwriteCountTokenComm(base):  # type: ignore[misc, valid-type]
        overwrite_counts = True
        cta_local_counts = False

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
            input_fc1_activation_per_token_scale_buffer=None,
            dispatched_fc1_activation_per_token_scale_buffer=None,
        ):
            # The shared dispatcher also serves per-token-scaled backends.
            # This override carries MXFP8 block-32 scales only.
            if cutlass.const_expr(
                input_fc1_activation_per_token_scale_buffer is not None
                or dispatched_fc1_activation_per_token_scale_buffer is not None
            ):
                raise ValueError("MXFP8 dispatch requires block-32 activation scales")
            # MemRange does not support dynamic indexing here; use raw pointers.
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
            # SF rows use their own padding; token and SF pool offsets can diverge.
            expert_sf_pool_block_offset = Int32(0)

            # ── Release-flag batching ────────────────────────────────────────
            # Delay fc1-ready counter publication with the same rotating-lane
            # tracker used by the epilogue.  Each token's TMA store to the FC1 pool
            # is drained CTA-locally by ``cp_async_bulk_wait_group(0)`` before its
            # release target is accumulated; the eventual red.release.gpu add
            # publishes the corresponding pool data to GPU scope.
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

            num_global_warps: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps
            token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

            _iket_pull_emit = (
                (sm_idx == Int32(0))
                and (warp_idx == Int32(0))
                and (lane_idx == Int32(0))
            )

            while current_expert_idx < Int32(self.num_experts_per_rank):
                if _iket_pull_emit:
                    _iket.range_push("Pull.ChooseToken")
                old_expert_idx = current_expert_idx
                while (token_idx >= expert_end_idx) and (
                    current_expert_idx < Int32(self.num_experts_per_rank)
                ):
                    prev_valid_count = expert_end_idx - expert_start_idx
                    prev_block_count = (
                        prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                    ) // Int32(self.token_padding_block)
                    expert_pool_block_offset = (
                        expert_pool_block_offset + prev_block_count
                    )
                    # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                    prev_task_tile_count = (
                        prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                    expert_task_tile_offset = (
                        expert_task_tile_offset + prev_task_tile_count
                    )
                    # Mirror cumul for the SF axis granularity (self.sf_padding_block).
                    prev_sf_block_count = (
                        prev_valid_count + Int32(self.sf_padding_block) - Int32(1)
                    ) // Int32(self.sf_padding_block)
                    expert_sf_pool_block_offset = (
                        expert_sf_pool_block_offset + prev_sf_block_count
                    )
                    current_expert_idx = current_expert_idx + Int32(1)
                    if current_expert_idx < Int32(self.num_experts_per_rank):
                        expert_start_idx = expert_end_idx
                        valid_value = Int32(0)
                        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                            if (
                                current_expert_idx
                                == Int32(i * self.warp_threads) + lane_idx
                            ):
                                valid_value = stored_num_tokens_per_expert[i]
                        total_for_expert = cute.arch.shuffle_sync(
                            valid_value, current_expert_idx % Int32(self.warp_threads)
                        )
                        expert_end_idx = expert_end_idx + total_for_expert

                if current_expert_idx < Int32(self.num_experts_per_rank):
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

                    # Round-robin min-peeling. Every undecided round retires at
                    # least one rank (the minimum), so the loop ends within
                    # world_size + 1 rounds; near-uniform routing decides in 1-2.
                    # A dynamic loop instead of world_size + 1 predicated unrolled
                    # rounds avoids issuing work after the pull range is decided.
                    decided = Int32(0)
                    while decided == Int32(0):
                        active = remaining_lane > Int32(0)
                        mask = cute.arch.vote_ballot_sync(active)
                        num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                        v_for_min = Int32(0x7FFFFFFF)
                        if active:
                            v_for_min = remaining_lane
                        length = Int32(cute.arch.warp_redux_sync(v_for_min, "min"))

                        if num_active_ranks > Int32(0):
                            num_round_tokens = length * num_active_ranks
                            if slot_idx < num_round_tokens:
                                slot_idx_in_round = slot_idx % num_active_ranks
                                current_rank_in_expert_idx = fns_b32(
                                    Int32(mask),
                                    Int32(0),
                                    slot_idx_in_round + Int32(1),
                                )
                                token_idx_in_rank = offset + (
                                    slot_idx // num_active_ranks
                                )
                                decided = Int32(1)
                            else:
                                slot_idx = slot_idx - num_round_tokens
                                offset = offset + length
                                if remaining_lane > length:
                                    remaining_lane = remaining_lane - length
                                else:
                                    remaining_lane = Int32(0)
                        else:
                            decided = Int32(1)

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
                    # SF-free callers pass input_sf_buffer=None (their SF loops are
                    # compiled out by sf_uint32_per_token=0, so this base address
                    # is never consumed).
                    if cutlass.const_expr(input_sf_buffer is not None):
                        inp_sf_local_base = input_sf_buffer.iterator.toint()
                    inp_w_local_base = input_topk_weights_buffer.iterator.toint()

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
                    cute.arch.sync_warp()

                    # Exclusive pull ranges: SF_Load
                    # and Weight_LDG cover load issue only; their latency surfaces
                    # in SF_Weight_STG, the first use. TMA_Wait is the row load's
                    # remaining mbarrier wait; the outer TMA_NVLink_Roundtrip spans
                    # issue to completion.
                    if _iket_pull_emit:
                        _iket.range_push("Pull.SF_Load")

                    sf_token_in_pool_axis = (
                        expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                        + token_idx_in_expert
                    )
                    pool_token_idx = (
                        expert_pool_block_offset * Int32(self.token_padding_block)
                        + token_idx_in_expert
                    )
                    sf_passes: cutlass.Constexpr[int] = (
                        self.sf_uint32_per_token + 31
                    ) // 32

                    sf_vals = []
                    for _ in cutlass.range_constexpr(0, sf_passes, 1):
                        sf_vals.append(Int32(0))

                    for i in cutlass.range_constexpr(0, sf_passes, 1):
                        j = Int32(i * self.warp_threads) + lane_idx
                        if j < Int32(self.sf_uint32_per_token):
                            sf_addr = (
                                inp_sf_local_base
                                + cur_peer_offset
                                + Int64(
                                    (src_token * Int32(self.sf_uint32_per_token) + j)
                                    * Int32(4)
                                )
                            )
                            sf_vals[i] = ldg_b32_raw(sf_addr)

                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.SF_Load
                        _iket.range_push("Pull.Weight_LDG")
                    weight = Float32(0.0)
                    if lane_idx == Int32(0):
                        weight_addr = (
                            inp_w_local_base
                            + cur_peer_offset
                            + Int64(
                                (src_token * Int32(self.num_topk) + src_topk) * Int32(4)
                            )
                        )
                        weight = ldg_f32_raw(weight_addr)

                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.Weight_LDG
                        _iket.range_push("Pull.SF_Weight_STG")

                    for i in cutlass.range_constexpr(0, sf_passes, 1):
                        j = Int32(i * self.warp_threads) + lane_idx
                        if j < Int32(self.sf_uint32_per_token):
                            if cutlass.const_expr(self.sf_atom_swizzled):
                                sf_int32_pos = sf_atom_int32_offset(
                                    sf_token_in_pool_axis,
                                    j,
                                    num_k_atoms=self.sf_uint32_per_token,
                                )
                            else:
                                sf_int32_pos = (
                                    sf_token_in_pool_axis
                                    * Int32(self.sf_uint32_per_token)
                                    + j
                                )
                            fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                    cute.arch.sync_warp()

                    if lane_idx == Int32(0):
                        fc1_input_topk_weights_buffer[pool_token_idx] = weight

                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.SF_Weight_STG
                        _iket.range_push("Pull.TMA_Wait")
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                        )
                        cute.arch.mbarrier_wait(
                            pull_mbar_ptr + warp_idx,
                            phase_bit,
                        )

                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.TMA_Wait
                        _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                        _iket.range_push("Pull.TMA_Store")

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

                    if _iket_pull_emit:
                        _iket.range_push("Pull.Metadata_STG")
                    with cute.arch.elect_one():
                        TokenSrcMetadata(
                            src_rank=current_rank_in_expert_idx,
                            src_token=src_token,
                            src_topk=src_topk,
                        ).store(
                            token_src_metadata.iterator
                            + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                        )
                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.Metadata_STG

                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)

                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.TMA_Store
                        _iket.range_push("Pull.Arrival_Atomic")

                    # Accumulate this token's release target into the rotating-lane
                    # batch tracker.  task_tile_idx is warp-uniform (token_idx /
                    # expert offsets are warp-wide), so every lane runs the same
                    # state-machine transition while only one lane records the
                    # current address.
                    task_tile_idx = expert_task_tile_offset + (
                        token_idx_in_expert // Int32(self.cluster_tile_tokens)
                    )

                    task_tile_addr = (
                        fc1_ready_counter.iterator + task_tile_idx
                    ).toint()
                    flag_tracker = flag_tracker.accumulate(
                        Int32(0),
                        self._flag_batch,
                        task_tile_addr,
                    )
                    cute.arch.sync_warp()

                    if _iket_pull_emit:
                        _iket.range_pop()  # Pull.Arrival_Atomic
                    # Pull progress on every dispatch warp: an instant per token,
                    # followed by one instant per set bit of its local expert.
                    if lane_idx == Int32(0):
                        _iket.range_push("Pull.TokenDone")
                        _iket.range_pop()
                        for bit in cutlass.range_constexpr(
                            max(1, (self.num_experts_per_rank - 1).bit_length())
                        ):
                            if (current_expert_idx >> Int32(bit)) & Int32(1) != Int32(
                                0
                            ):
                                _iket.range_push(f"expert_b{bit}")
                                _iket.range_pop()

                    phase_bit = phase_bit ^ Int32(1)

                    token_idx = token_idx + Int32(num_global_warps)

            # Tail flush: publish any leftover (< self._flag_batch) accumulated release.
            flag_tracker.fire()
            cute.arch.sync_warp()

            return phase_bit, stored_num_tokens_per_expert

        @cute.jit
        def dispatch_barrier(
            self,
            expert_send_count,
            expert_recv_count,
            expert_recv_count_sum,
            nvlink_barrier_signal,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            *,
            local_rank,
            num_sms,
            nvlink_barrier_counter,
        ):
            # software_grid_sync expects a dispatch-group-relative thread id.
            tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

            # Snapshot before the grid rendezvous: CTA 0 cannot advance the
            # phase until every CTA has read it. Tail/NCU phase cycling is intact.
            if cutlass.const_expr(self.cta_local_counts):
                startup_status = cute.arch.load(
                    nvlink_barrier_counter.iterator, Int32, sem="relaxed", scope="gpu"
                ) & Int32(3)
                if cutlass.const_expr(hasattr(self, "_stamp")):
                    if warp_idx == Int32(0) and lane_idx == Int32(0):
                        phase_ptr = (
                            grid_sync_counter.iterator
                            + Int32(self.stats_word_offsets["grid_sync_counter"])
                            + sm_idx * Int32(SLOTS)
                            + Int32(Slot.STARTUP_PHASE_RAW)
                        )
                        cute.make_tensor(phase_ptr, cute.make_layout(1))[0] = (
                            startup_status
                        )

            if cutlass.const_expr(hasattr(self, "_stamp")):
                grid_start = globaltimer_lo()
            software_grid_sync(
                grid_sync_counter,
                sm_idx,
                num_sms,
                tid_in_group,
                num_threads=self.num_dispatch_threads,
            )

            if cutlass.const_expr(self.cta_local_counts and hasattr(self, "_stamp")):
                if warp_idx == Int32(0) and lane_idx == Int32(0):
                    # The first grid sync orders CTA 0's debug snapshot. Unlike
                    # the live barrier counter, it cannot advance during acquire.
                    phase_ptr = (
                        grid_sync_counter.iterator
                        + Int32(self.stats_word_offsets["grid_sync_counter"])
                        + Int32(Slot.STARTUP_PHASE_RAW)
                    )
                    expected_phase = cute.arch.load(
                        phase_ptr, Int32, sem="relaxed", scope="gpu"
                    )
                    _startup_phase_assert(startup_status, expected_phase)

            if cutlass.const_expr(hasattr(self, "_stamp")):
                send_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_GRID_SYNC_NS,
                    send_start - grid_start,
                )
            if sm_idx == 0:
                for offset in cutlass.range_constexpr(
                    0,
                    self.num_total_experts,
                    self.experts_per_dispatch_pass,
                ):
                    expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                    if expert_id < Int32(self.num_total_experts):
                        dst_rank = expert_id // Int32(self.num_experts_per_rank)
                        dst_local_expert = expert_id % Int32(self.num_experts_per_rank)
                        status_u64 = cute.arch.load(
                            expert_send_count.iterator + expert_id,
                            Int64,
                            sem="relaxed",
                            scope="gpu",
                        )
                        token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                        erc_local_base = expert_recv_count.iterator.toint()
                        erc_elem_off = (
                            Int32(local_rank) * Int32(self.num_experts_per_rank)
                            + dst_local_expert
                        ) * Int32(8)
                        erc_peer_addr = peer_rank_ptr_mapper.map(
                            erc_local_base,
                            dst_rank,
                            Int64(erc_elem_off),
                        )
                        stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            if cutlass.const_expr(hasattr(self, "_stamp")):
                rendezvous_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_COUNT_SEND_NS,
                    rendezvous_start - send_start,
                )
            if cutlass.const_expr(self.cta_local_counts):
                self.startup_acquire(
                    nvlink_barrier_signal,
                    nvlink_barrier_counter,
                    peer_rank_ptr_mapper,
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    startup_status,
                    grid_sync_counter,
                )
            else:
                self.nvlink_barrier(
                    nvlink_barrier_signal,
                    nvlink_barrier_counter,
                    grid_sync_counter,
                    peer_rank_ptr_mapper,
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    num_sms=num_sms,
                    prologue_grid_sync=False,
                    epilogue_grid_sync=False,
                )
            if cutlass.const_expr(hasattr(self, "_stamp")):
                sum_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_RENDEZVOUS_NS,
                    sum_start - rendezvous_start,
                )

            # Each CTA acquired peer count stores through startup_acquire.
            # Private sums reuse prep's now-dead scratch. The dispatch barrier
            # orders shared stores before pull; the dispatch-to-scheduler named
            # barrier carries the same acquire and sums to the scheduler warp.
            # The global packed diagnostic copy has exactly one writer per word.
            if cutlass.const_expr(self.cta_local_counts):
                global_count_sum = self.global_count_sum
                for offset in cutlass.range_constexpr(
                    0, self.num_experts_per_rank, self.num_dispatch_threads
                ):
                    expert = Int32(offset) + tid_in_group
                    if expert < Int32(self.num_experts_per_rank):
                        total = Int64(0)
                        for rank in cutlass.range_constexpr(self.world_size):
                            total += expert_recv_count[rank, expert]
                        expert_recv_count_sum[expert] = Int32(total)
                        if sm_idx == Int32(0):
                            global_count_sum[expert] = (
                                Int64(self.world_size * num_sms) << Int64(32)
                            ) | total
                cute.arch.barrier(
                    barrier_id=self.dispatch_intra_cta_bar_id,
                    number_of_threads=self.num_dispatch_threads,
                )
            else:
                if sm_idx == Int32(0):
                    cute.arch.barrier(
                        barrier_id=self.dispatch_intra_cta_bar_id,
                        number_of_threads=self.num_dispatch_threads,
                    )
                    for offset in cutlass.range_constexpr(
                        0, self.num_experts_per_rank, self.num_dispatch_threads
                    ):
                        expert = Int32(offset) + tid_in_group
                        if expert < Int32(self.num_experts_per_rank):
                            total = Int64(0)
                            for rank in cutlass.range_constexpr(self.world_size):
                                total += expert_recv_count[rank, expert]
                            expert_recv_count_sum[expert] = (
                                Int64(self.world_size * num_sms) << Int64(32)
                            ) | total
            if cutlass.const_expr(hasattr(self, "_stamp")):
                grid_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_COUNT_SUM_NS,
                    grid_start - sum_start,
                )
            if cutlass.const_expr(not self.cta_local_counts):
                software_grid_sync(
                    grid_sync_counter,
                    sm_idx,
                    num_sms,
                    tid_in_group,
                    num_threads=self.num_dispatch_threads,
                )

                if cutlass.const_expr(hasattr(self, "_stamp")):
                    self._stamp(
                        grid_sync_counter.iterator,
                        "grid_sync_counter",
                        sm_idx,
                        warp_idx,
                        lane_idx,
                        Slot.DISPATCH_GRID_SYNC_NS,
                        globaltimer_lo() - grid_start,
                    )

        @cute.jit
        def startup_acquire(
            self,
            signal,
            counter,
            mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            status,
            grid_sync_counter,
        ):
            if cutlass.const_expr(hasattr(self, "_stamp")):
                self._nvlink_probed = True
                self._nvlink_start = globaltimer_lo()
                self._stamp(
                    grid_sync_counter.iterator,
                    "grid_sync_counter",
                    sm_idx,
                    warp_idx,
                    lane_idx,
                    Slot.DISPATCH_SYNC_SEND_NS,
                    self._nvlink_start - self._barrier_start,
                )
            phase = status & Int32(1)
            delta = Int32(1)
            target = Int32(self.world_size)
            if (status >> Int32(1)) != Int32(0):
                delta = Int32(-1)
                target = Int32(0)
            if cutlass.const_expr(hasattr(self, "_stamp")):
                send_at = globaltimer_lo()
            if warp_idx == Int32(0):
                if sm_idx == Int32(0):
                    if lane_idx < Int32(self.world_size):
                        addr = mapper.map(
                            signal.iterator.toint(), lane_idx, Int64(phase * Int32(4))
                        )
                        red_add_release_sys_s32_raw(addr, delta)
                    cute.arch.sync_warp()
                    if lane_idx == Int32(0):
                        cute.arch.atomic_add(
                            counter.iterator, Int32(1), sem="relaxed", scope="gpu"
                        )
                if lane_idx == Int32(0):
                    # Memory-model chain: peer count stores -> peer system fence
                    # -> peer release-add -> this CTA's system acquire -> CTA
                    # barrier -> count reads. Every CTA executes its own acquire;
                    # no CTA relies on another CTA's polling load for visibility.
                    while (
                        cute.arch.load(
                            signal.iterator + phase, Int32, sem="acquire", scope="sys"
                        )
                        != target
                    ):
                        pass
                    if cutlass.const_expr(hasattr(self, "_stamp")):
                        if sm_idx == Int32(0):
                            self._record_startup_signal(
                                grid_sync_counter, send_at, globaltimer_lo() - send_at
                            )
            cute.arch.barrier(
                barrier_id=self.dispatch_intra_cta_bar_id,
                number_of_threads=self.num_dispatch_threads,
            )

            if cutlass.const_expr(hasattr(self, "_stamp")):
                self._signal_done = globaltimer_lo()

        @cute.jit
        def kernel_tail(
            self,
            token_comm_args,
            *,
            warp_idx,
            lane_idx,
            tidx,
            tail_work: cutlass.Constexpr = None,
        ):
            nb = pipeline.NamedBarrier(
                barrier_id=self.kernel_tail_named_barrier_id,
                num_threads=self.kernel_tail_threads,
            )
            # Mxfp8DispatchPhase fences every producer's peer stores before this
            # rendezvous. All count/slot readers and local counter users are done.
            nb.arrive_and_wait()
            if (warp_idx >= self.dispatch_warp_start) and (
                warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
            ):
                cta = self._cta_linear_id()
                warp = Int32(warp_idx) - Int32(self.dispatch_warp_start)
                # Normal launches have two NVLink barriers (startup and drain).
                # Keep the phase counter: the two signal slots self-cancel across
                # TWO launches, not one. NCU restores local counters but cannot
                # restore peer signal memory, so kernel replay still needs four
                # calls per launch. Add two profiling-only barriers in that case.
                for _ in cutlass.range_constexpr(
                    3 if os.environ.get("MEGA_USE_NCU", "0") == "1" else 1
                ):
                    self.nvlink_barrier(
                        token_comm_args.nvlink_barrier_signal,
                        token_comm_args.nvlink_barrier_counter,
                        token_comm_args.grid_sync_counter,
                        token_comm_args.peer_rank_ptr_mapper,
                        cta,
                        warp,
                        lane_idx,
                        num_sms=token_comm_args.sm_count,
                        prologue_grid_sync=True,
                        epilogue_grid_sync=True,
                    )

            if cutlass.const_expr(tail_work is not None):
                # Publish the drain acquire before any warp reads peer route terms.
                nb.arrive_and_wait()

            if (warp_idx >= self.dispatch_warp_start) and (
                warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
            ):
                # Rank-local reset overlaps other warps' reduction. Stream order
                # protects these counters on relaunch; persistent phase counters
                # live outside this prefix and are never reset here.
                self.tail_reset_counters(
                    token_comm_args,
                    token_comm_args.local_zero_prefix,
                    cta_linear_id=self._cta_linear_id(),
                    local_warp_idx=Int32(warp_idx) - Int32(self.dispatch_warp_start),
                    lane_idx=lane_idx,
                )

            if cutlass.const_expr(tail_work is not None):
                tail_work(
                    token_comm_args,
                    self._cta_linear_id(),
                    tidx,
                    self.kernel_tail_threads,
                )
            # No final rendezvous is needed: a peer may overwrite counts and slot
            # metadata now (their readers finished before the drain), but cannot
            # overwrite route_terms until its next startup barrier, which waits
            # for this rank to finish reducing and enter its next launch.

    return OverwriteCountTokenComm
