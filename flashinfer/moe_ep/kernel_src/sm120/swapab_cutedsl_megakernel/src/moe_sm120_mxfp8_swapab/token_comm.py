"""SM120 token communication specialization for NVSHMEM SYSMEM heaps.

The common protocol stays in :mod:`src.token_comm`.  This subclass carries the
SM120/SYSMEM adaptations: peer-store barriers, local receive-count reduction,
tile-ordered dispatch publication, dispatch/compute overlap, and lane-strided
token-back stores.
"""

import os
from typing import Literal

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint8, Uint32

try:
    from cutlass.cute import iket as _iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    try:
        from cutlass.cute.experimental import iket as _iket  # type: ignore
    except ImportError:
        from src.iket_compat import iket as _iket

from .moe_utils import spin_wait
from .sm120_ptx_helpers import (
    lds_b32_raw,
    red_add_relaxed_sys_v2_bf16x2_raw,
)
from src.flag_batch import GpuReleaseFlagBatchTracker
from src.grid_sync import software_grid_sync
from src.ptx_helpers import (
    fns_b32,
    ldg_b32_raw,
    ldg_f32_raw,
    read_clock64,
    red_add_release_gpu_s32,
    stg_b32_raw,
    stg_b64_raw,
    tma_load_1d_raw,
    tma_store_1d,
)
from src.sf_swizzle import sf_atom_int32_offset
from src.token_comm import CombineFormat, TokenInPullTokenBackPush, TokenSrcMetadata


class Sm120SysmemTokenInPullTokenBackPush(TokenInPullTokenBackPush):
    """Specialize common token communication for the SM120 SYSMEM path."""

    def __init__(
        self,
        *,
        dispatch_pull_mode: Literal[
            "token_strided", "tile_cooperative"
        ] = "token_strided",
        dispatch_warps_per_tile: int = 4,
        dispatch_compute_overlap: bool = True,
        streaming_fc12: bool = False,
        **kwargs,
    ) -> None:
        local_rank = kwargs.pop("local_rank")
        fc2_output_dtype = kwargs.pop("fc2_output_dtype", None)
        kwargs["combine_format"] = CombineFormat(
            act_dtype=(
                fc2_output_dtype
                if fc2_output_dtype is not None
                else cutlass.BFloat16
            ),
            scale_dtype=None,
            scale_block=None,
        )
        kwargs["token_back_by_dispatch"] = fc2_output_dtype is not None
        super().__init__(**kwargs)
        self.local_rank = local_rank

        if dispatch_pull_mode not in ("token_strided", "tile_cooperative"):
            raise ValueError(
                "dispatch_pull_mode must be 'token_strided' or "
                f"'tile_cooperative'; got {dispatch_pull_mode!r}."
            )
        if dispatch_warps_per_tile not in (4, 8, 16, 32):
            raise ValueError(
                "dispatch_warps_per_tile must be one of 4/8/16/32, got "
                f"{dispatch_warps_per_tile}."
            )
        if dispatch_warps_per_tile > self.cluster_tile_tokens:
            raise ValueError(
                "dispatch_warps_per_tile must not exceed cluster_tile_tokens; "
                f"got {dispatch_warps_per_tile} > {self.cluster_tile_tokens}."
            )
        self.dispatch_pull_mode = dispatch_pull_mode
        self.dispatch_warps_per_tile = dispatch_warps_per_tile
        self.dispatch_compute_overlap = dispatch_compute_overlap
        self.streaming_fc12 = streaming_fc12
        overlap_env = os.environ.get("MEGA_DISPATCH_COMPUTE_OVERLAP")
        if overlap_env is not None:
            overlap_env = overlap_env.strip().lower()
            if overlap_env not in ("0", "1", "false", "true", "off", "on"):
                raise ValueError(
                    "MEGA_DISPATCH_COMPUTE_OVERLAP must be a boolean value, "
                    f"got {overlap_env!r}."
                )
            self.dispatch_compute_overlap = overlap_env in ("1", "true", "on")

    def extra_smem_storage_class(self) -> type:
        hidden_bytes = self.hidden_bytes
        num_total_experts = self.num_total_experts

        if self.token_back_standalone:
            @cute.struct
            class TokenCommStorage:
                pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
                smem_expert_count: cute.struct.MemRange[
                    Int32, num_total_experts
                ]
                pull_buffer: cute.struct.Align[
                    cute.struct.MemRange[
                        Uint8, self.num_dispatch_warps * hidden_bytes
                    ],
                    128,
                ]
                tb_pull_mbar: cute.struct.MemRange[Int64, self.num_token_back_warps]
                tb_pull_buffer: cute.struct.Align[
                    cute.struct.MemRange[
                        Uint8, self.num_token_back_warps * self.tb_chunk_bytes
                    ],
                    128,
                ]

            return TokenCommStorage

        @cute.struct
        class TokenCommStorage:
            pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
            smem_expert_count: cute.struct.MemRange[
                Int32, num_total_experts
            ]
            pull_buffer: cute.struct.Align[
                cute.struct.MemRange[
                    Uint8, self.num_dispatch_warps * hidden_bytes
                ],
                128,
            ]

        return TokenCommStorage

    @cute.jit
    def fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        if cutlass.const_expr(self.is_swap_ab):
            counter_slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
            peek_threshold = work_tile_info.valid_tokens_in_tile
        else:
            counter_slot = (
                work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx // cutlass.Int32(self.cluster_shape_mn[0])
            )
            peek_threshold = work_tile_info.valid_tokens_in_cluster

        counter_ptr = token_comm_args.fc1_ready_counter.iterator + counter_slot
        if not work_tile_info.peek_ready:
            _iket.range_push("tma_token_fc1_wait")
            spin_wait(
                counter_ptr,
                lambda v: v >= peek_threshold,
                fail_sleep_cycles=1000,
            )
            _iket.range_pop()
        cute.arch.fence_acq_rel_sys()
        cute.arch.fence_proxy("async.global")

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
        num_sms,
        nvlink_barrier_counter=None,
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                           num_threads=self.num_dispatch_threads)

        if sm_idx == 0:
            for offset in cutlass.range_constexpr(
                0, self.num_total_experts, self.experts_per_dispatch_pass,
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
                        Int32(self.local_rank) * Int32(self.num_experts_per_rank) + dst_local_expert
                    ) * Int32(8)
                    erc_peer_addr = peer_rank_ptr_mapper.map(
                        erc_local_base, dst_rank, Int64(erc_elem_off),
                    )
                    stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
            cute.arch.fence_acq_rel_sys()
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        self.nvlink_barrier(
            nvlink_barrier_signal,
            nvlink_barrier_counter,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            slot=0,
            num_sms=num_sms,
            prologue_grid_sync=False,
            epilogue_grid_sync=True,
        )

        if sm_idx == 0:
            for offset in cutlass.range_constexpr(
                0, self.num_experts_per_rank, self.experts_per_dispatch_pass,
            ):
                local_expert = Int32(offset + warp_idx * self.warp_threads + lane_idx)
                if local_expert < Int32(self.num_experts_per_rank):
                    total_count = Int32(0)
                    for rank in cutlass.range_constexpr(0, self.world_size, 1):
                        packed = expert_recv_count[rank, local_expert]
                        total_count = total_count + Int32(packed & Int64(0xFFFFFFFF))
                    publishers = Int64(self.world_size) * Int64(num_sms)
                    expert_recv_count_sum[local_expert] = (
                        (publishers << Int64(32))
                        | (Int64(total_count) & Int64(0xFFFFFFFF))
                    )
            cute.arch.fence_acq_rel_gpu()

        software_grid_sync(
            grid_sync_counter,
            sm_idx,
            num_sms,
            tid_in_group,
            num_threads=self.num_dispatch_threads,
        )

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
        if cutlass.const_expr(self.dispatch_pull_mode == "tile_cooperative"):
            return self._dispatch_pull_tile_cooperative(
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
                num_sms=num_sms,
            )
        return self._dispatch_pull_token_strided(
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
            num_sms=num_sms,
        )

    @cute.jit
    def _dispatch_pull_token_strided(
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
        # MemRange does not support dynamic indexing here; use raw pointers.
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        mbar_ptr_warp = pull_mbar_ptr + warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(mbar_ptr_warp, 1)
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
                    for i in cutlass.range_constexpr(
                        0, NUM_EXPERTS_PER_LANE, 1
                    ):
                        if current_expert_idx == Int32(i * self.warp_threads) + lane_idx:
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

                decided = Int32(0)
                for _round in cutlass.range_constexpr(0, self.world_size + 1, 1):
                    if decided == Int32(0):
                        active = remaining_lane > Int32(0)
                        mask = cute.arch.vote_ballot_sync(active)
                        num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                        v_for_min = Int32(0x7FFFFFFF)
                        if active:
                            v_for_min = remaining_lane
                        length = Int32(
                            cute.arch.warp_redux_sync(v_for_min, "min")
                        )

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
                inp_sf_local_base = input_sf_buffer.iterator.toint()
                inp_w_local_base = input_topk_weights_buffer.iterator.toint()

                sf_token_in_pool_axis = (
                    expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                    + token_idx_in_expert
                )
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )
                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr_warp, Int32(self.hidden_bytes)
                    )
                    tma_src_addr = (
                        inp_tok_local_base
                        + cur_peer_offset
                        + Int64(src_token * Int32(self.hidden_bytes))
                    )
                    tma_load_1d_raw(
                        pull_buffer_warp_ptr,
                        tma_src_addr,
                        mbar_ptr_warp,
                        Int32(self.hidden_bytes),
                    )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_push("Pull.SF_LDG_STG")

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
                    _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                    _iket.range_push("Pull.Weight_LDG")   # (= ST phase)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_int32_pos = sf_atom_int32_offset(
                            sf_token_in_pool_axis,
                            j,
                            num_k_atoms=self.sf_uint32_per_token,
                        )
                        fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                cute.arch.sync_warp()

                if lane_idx == Int32(0):
                    fc1_input_topk_weights_buffer[pool_token_idx] = weight

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                    _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                    _iket.range_push("Pull.TMA_Store")

                with cute.arch.elect_one():
                    cute.arch.mbarrier_wait(
                        mbar_ptr_warp,
                        phase_bit,
                    )

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
                    _iket.range_pop()  # Pull.TMA_Store
                    _iket.range_push("Pull.Arrival_Atomic")

                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.fence_proxy("async.global")

                # The delayed release may be fired by a rotating lane other
                # than the elect-one lane that issued/waited the TMA store.
                # Rendezvous before publishing fc1_ready_counter so every
                # lane observes token, scale, weight, and metadata completion.
                cute.arch.sync_warp()
                cute.arch.fence_acq_rel_sys()
                cute.arch.fence_proxy("async.global")

                # Accumulate this token's release target into the rotating-lane
                # batch tracker.  task_tile_idx is warp-uniform (token_idx /
                # expert offsets are warp-wide), so every lane runs the same
                # state-machine transition while only one lane records the
                # current address.
                task_tile_idx = expert_task_tile_offset + (
                    token_idx_in_expert // Int32(self.cluster_tile_tokens)
                )

                task_tile_addr = (fc1_ready_counter.iterator + task_tile_idx).toint()
                flag_tracker = flag_tracker.accumulate(
                    Int32(0), self._flag_batch, task_tile_addr,
                )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Arrival_Atomic

                phase_bit = phase_bit ^ Int32(1)

                token_idx = token_idx + Int32(num_global_warps)

        # Tail flush: publish any leftover (< self._flag_batch) accumulated release.
        flag_tracker.fire()
        cute.arch.sync_warp()

        return phase_bit, stored_num_tokens_per_expert

    @cute.jit
    def _dispatch_pull_tile_cooperative(
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
        """Pull complete FC1 token tiles in scheduler-consumption order.

        A compile-time number of consecutive dispatch warps cooperates on one
        token tile.  Every warp drains all of its TMA stores, then contributes
        its completed-token count with one release reduction.  Compared with
        token-strided pull this preserves the expert-major/tile-major scheduler
        frontier and replaces per-token ready-counter traffic with one update
        per participating warp and tile.
        """
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        mbar_ptr_warp = pull_mbar_ptr + warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(mbar_ptr_warp, 1)
        cute.arch.sync_warp()

        phase_bit = Int32(0)
        current_expert_idx = Int32(-1)
        current_expert_token_count = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_sf_pool_block_offset = Int32(0)
        expert_task_tile_start = Int32(0)
        expert_task_tile_end = Int32(0)
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

        WARPS_PER_TILE: cutlass.Constexpr[int] = self.dispatch_warps_per_tile
        num_global_warps: cutlass.Constexpr[int] = (
            num_sms * self.num_dispatch_warps
        )
        num_tile_groups: cutlass.Constexpr[int] = num_global_warps // WARPS_PER_TILE
        global_warp_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx
        task_tile_idx = global_warp_idx // Int32(WARPS_PER_TILE)
        warp_in_tile = global_warp_idx % Int32(WARPS_PER_TILE)
        # Ignore a possible incomplete tail group.  A partial group would
        # leave some token positions without a producer and, worse, alias the
        # full groups after adding ``num_tile_groups`` to task_tile_idx.
        if global_warp_idx >= Int32(num_tile_groups * WARPS_PER_TILE):
            task_tile_idx = Int32(0x7FFFFFFF)

        inp_tok_local_base = input_token_buffer.iterator.toint()
        inp_sf_local_base = input_sf_buffer.iterator.toint()
        inp_w_local_base = input_topk_weights_buffer.iterator.toint()
        sf_passes: cutlass.Constexpr[int] = (
            self.sf_uint32_per_token + 31
        ) // 32

        while current_expert_idx < Int32(self.num_experts_per_rank):
            old_expert_idx = current_expert_idx
            while (task_tile_idx >= expert_task_tile_end) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = current_expert_token_count
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = (
                    expert_pool_block_offset + prev_block_count
                )
                prev_sf_block_count = (
                    prev_valid_count + Int32(self.sf_padding_block) - Int32(1)
                ) // Int32(self.sf_padding_block)
                expert_sf_pool_block_offset = (
                    expert_sf_pool_block_offset + prev_sf_block_count
                )
                expert_task_tile_start = expert_task_tile_end
                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(
                        0, NUM_EXPERTS_PER_LANE, 1
                    ):
                        if current_expert_idx == Int32(i * self.warp_threads) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    current_expert_token_count = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    current_task_tile_count = (
                        current_expert_token_count
                        + Int32(self.cluster_tile_tokens)
                        - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                    expert_task_tile_end = (
                        expert_task_tile_end + current_task_tile_count
                    )

            if current_expert_idx < Int32(self.num_experts_per_rank):
                if old_expert_idx != current_expert_idx:
                    if lane_idx < Int32(self.world_size):
                        stored_rank_count_lane = Int32(
                            expert_recv_count[lane_idx, current_expert_idx]
                        )
                    else:
                        stored_rank_count_lane = Int32(0)

                tile_idx_in_expert = task_tile_idx - expert_task_tile_start
                first_token_in_expert = (
                    tile_idx_in_expert * Int32(self.cluster_tile_tokens)
                )
                valid_tokens_in_tile = cutlass.min(
                    current_expert_token_count - first_token_in_expert,
                    Int32(self.cluster_tile_tokens),
                )

                token_in_tile = warp_in_tile
                published_tokens = Int32(0)
                while token_in_tile < valid_tokens_in_tile:
                    token_idx_in_expert = first_token_in_expert + token_in_tile
                    slot_idx = token_idx_in_expert
                    offset = Int32(0)
                    remaining_lane = stored_rank_count_lane
                    current_rank_in_expert_idx = Int32(0)
                    token_idx_in_rank = Int32(0)

                    decided = Int32(0)
                    for _round in cutlass.range_constexpr(
                        0, self.world_size + 1, 1
                    ):
                        if decided == Int32(0):
                            active = remaining_lane > Int32(0)
                            mask = cute.arch.vote_ballot_sync(active)
                            num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                            v_for_min = Int32(0x7FFFFFFF)
                            if active:
                                v_for_min = remaining_lane
                            length = Int32(
                                cute.arch.warp_redux_sync(v_for_min, "min")
                            )

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

                    sf_token_in_pool_axis = (
                        expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                        + token_idx_in_expert
                    )
                    pool_token_idx = (
                        expert_pool_block_offset * Int32(self.token_padding_block)
                        + token_idx_in_expert
                    )

                    with cute.arch.elect_one():
                        pull_buffer_warp_ptr = pull_buffer_ptr + (
                            warp_idx * Int32(self.hidden_bytes)
                        )
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr_warp, Int32(self.hidden_bytes)
                        )
                        tma_src_addr = (
                            inp_tok_local_base
                            + cur_peer_offset
                            + Int64(src_token * Int32(self.hidden_bytes))
                        )
                        tma_load_1d_raw(
                            pull_buffer_warp_ptr,
                            tma_src_addr,
                            mbar_ptr_warp,
                            Int32(self.hidden_bytes),
                        )
                    cute.arch.sync_warp()

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
                                    (
                                        src_token * Int32(self.sf_uint32_per_token)
                                        + j
                                    )
                                    * Int32(4)
                                )
                            )
                            sf_vals[i] = ldg_b32_raw(sf_addr)

                    weight = Float32(0.0)
                    if lane_idx == Int32(0):
                        weight_addr = (
                            inp_w_local_base
                            + cur_peer_offset
                            + Int64(
                                (
                                    src_token * Int32(self.num_topk) + src_topk
                                )
                                * Int32(4)
                            )
                        )
                        weight = ldg_f32_raw(weight_addr)

                    for i in cutlass.range_constexpr(0, sf_passes, 1):
                        j = Int32(i * self.warp_threads) + lane_idx
                        if j < Int32(self.sf_uint32_per_token):
                            sf_int32_pos = sf_atom_int32_offset(
                                sf_token_in_pool_axis,
                                j,
                                num_k_atoms=self.sf_uint32_per_token,
                            )
                            fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                    cute.arch.sync_warp()

                    if lane_idx == Int32(0):
                        fc1_input_topk_weights_buffer[pool_token_idx] = weight

                    with cute.arch.elect_one():
                        cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                        pull_buffer_warp_ptr = pull_buffer_ptr + (
                            warp_idx * Int32(self.hidden_bytes)
                        )
                        tma_store_1d(
                            fc1_input_token_buffer.iterator
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
                        cute.arch.cp_async_bulk_wait_group(0)

                    phase_bit = phase_bit ^ Int32(1)
                    published_tokens = published_tokens + Int32(1)
                    token_in_tile = token_in_tile + Int32(WARPS_PER_TILE)

                # One release update per participating warp and tile.  The
                # consumer threshold is the number of valid tokens, so the
                # tile becomes visible only after all cooperating warps have
                # drained and published their portions.
                cute.arch.sync_warp()
                cute.arch.fence_acq_rel_sys()
                cute.arch.fence_proxy("async.global")
                if published_tokens > Int32(0):
                    task_tile_addr = (
                        fc1_ready_counter.iterator + task_tile_idx
                    ).toint()
                    with cute.arch.elect_one():
                        counter_ptr = cute.make_ptr(
                            Int32,
                            task_tile_addr,
                            AddressSpace.gmem,
                            assumed_align=4,
                        )
                        red_add_release_gpu_s32(counter_ptr, published_tokens)

                task_tile_idx = task_tile_idx + Int32(num_tile_groups)

        return phase_bit, stored_num_tokens_per_expert

    @cute.jit
    def token_back_by_push(
        self,
        pull_buffer_ptr,
        pull_mbar_ptr,
        fc2_output_workspace,
        fc2_done_counter,
        token_src_metadata,
        combine_output,
        token_back_schedule_counter,
        peer_rank_ptr_mapper,
        phase_bit,
        stored_num_tokens_per_expert,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        chunk_bytes: cutlass.Constexpr[int],
    ):
        _iket_emit = (sm_idx == Int32(0)) and (warp_idx == Int32(0))
        avg_token_back_window = Int32(2500)

        # Chunk the fc2 token in ``chunk_bytes`` pieces; the last piece carries
        # the remainder so any chunk_bytes works for any fc2_token_bytes.
        fc2_token_bytes: cutlass.Constexpr[int] = self.fc2_token_bytes
        num_chunks: cutlass.Constexpr[int] = (
            fc2_token_bytes + chunk_bytes - 1
        ) // chunk_bytes
        last_chunk_bytes: cutlass.Constexpr[int] = (
            fc2_token_bytes - (num_chunks - 1) * chunk_bytes
        )
        redg_bytes_per_lane: cutlass.Constexpr[int] = 8
        redg_bytes_per_warp: cutlass.Constexpr[int] = (
            self.warp_threads * redg_bytes_per_lane
        )
        redg_iters_per_full_chunk: cutlass.Constexpr[int] = (
            chunk_bytes + redg_bytes_per_warp - 1
        ) // redg_bytes_per_warp
        copy_bytes_per_lane: cutlass.Constexpr[int] = 4
        copy_bytes_per_warp: cutlass.Constexpr[int] = (
            self.warp_threads * copy_bytes_per_lane
        )
        copy_iters_per_full_chunk: cutlass.Constexpr[int] = (
            chunk_bytes + copy_bytes_per_warp - 1
        ) // copy_bytes_per_warp

        num_experts_per_lane: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        num_global_warps: cutlass.Constexpr[int] = (
            num_sms * self.num_dispatch_warps
        )
        schedule_mode = self.token_back_schedule_mode
        atomic_batch = self.token_back_atomic_batch

        # static: stride by the global warp count.  atomic_counter: consume one
        # slot of the current batch, refilling via one grid-scoped
        # atomicAdd(atomic_batch) when exhausted so fast warps keep stealing
        # work.  cuTeDSL forbids closures over enclosing locals -> pass all in.
        def update_token_idx(
            token_idx, batch_remaining, lane_idx, schedule_counter,
            schedule_mode, atomic_batch, num_global_warps,
        ):
            if cutlass.const_expr(schedule_mode == "atomic_counter"):
                batch_remaining = batch_remaining - Int32(1)
                if batch_remaining == Int32(0):
                    base = Int32(0)
                    if lane_idx == Int32(0):
                        base = cute.arch.atomic_add(
                            schedule_counter, Int32(atomic_batch),
                            sem="relaxed", scope="gpu",
                        )
                    token_idx = cute.arch.shuffle_sync(base, Int32(0))
                    batch_remaining = Int32(atomic_batch)
                else:
                    token_idx = token_idx + Int32(1)
            else:
                token_idx = token_idx + Int32(num_global_warps)
            return token_idx, batch_remaining

        if cutlass.const_expr(schedule_mode == "atomic_counter"):
            # Prime the first batch: batch_remaining=1 makes update_token_idx
            # decrement to 0 and pull the initial atomic batch.
            token_idx = Int32(0)
            batch_remaining = Int32(1)
            token_idx, batch_remaining = update_token_idx(
                token_idx, batch_remaining, lane_idx,
                token_back_schedule_counter,
                schedule_mode, atomic_batch, num_global_warps,
            )
        else:
            token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx
            batch_remaining = Int32(0)

        current_expert_idx = Int32(-1)
        confirmed_expert_idx = Int32(-1)
        confirmed_token_tile_slot = Int32(-1)
        cur_expert_expected = Int32(0)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_token_tile_offset = Int32(0)

        while current_expert_idx < Int32(self.num_experts_per_rank):
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
                prev_token_tile_count = (
                    prev_valid_count
                    + Int32(self.cluster_tile_tokens)
                    - Int32(1)
                ) // Int32(self.cluster_tile_tokens)
                expert_token_tile_offset = (
                    expert_token_tile_offset + prev_token_tile_count
                )

                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(
                        0, num_experts_per_lane, 1
                    ):
                        if current_expert_idx == Int32(
                            i * self.warp_threads
                        ) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

                    cluster_tile_cnt = (
                        total_for_expert
                        + Int32(self.cluster_tile_tokens)
                        - Int32(1)
                    ) // Int32(self.cluster_tile_tokens)
                    # Stash the threshold; the wait is deferred to the expert we
                    # actually land on, so stepped-over experts are never waited.
                    cur_expert_expected = cluster_tile_cnt * Int32(
                        self.fc2_publishes_per_token_cluster_tile
                    )

            if current_expert_idx < Int32(self.num_experts_per_rank):
                remain_experts = Int32(self.num_experts_per_rank) - current_expert_idx
                token_idx_in_expert = token_idx - expert_start_idx
                if cutlass.const_expr(self.streaming_fc12):
                    token_tile_slot = (
                        expert_token_tile_offset
                        + token_idx_in_expert // Int32(self.cluster_tile_tokens)
                    )
                    if token_tile_slot != confirmed_token_tile_slot:
                        if _iket_emit:
                            _iket.range_push("token_back_wait_fc2_tile")
                        spin_wait(
                            fc2_done_counter.iterator + token_tile_slot,
                            lambda v: v
                            >= Int32(self.fc2_publishes_per_token_cluster_tile),
                            fail_sleep_cycles=500,
                        )
                        if _iket_emit:
                            _iket.range_pop()
                            _iket.mark("token_back_tile_ready", token_tile_slot)
                        confirmed_token_tile_slot = token_tile_slot
                else:
                    # Grouped schedule publishes one aggregate counter per
                    # expert, so token-back waits once before entering it.
                    if current_expert_idx > confirmed_expert_idx:
                        if _iket_emit:
                            _iket.range_push("token_back_wait_fc2_expert")
                        spin_wait(
                            fc2_done_counter.iterator + current_expert_idx,
                            lambda v: v >= cur_expert_expected,
                            fail_sleep_cycles=500,
                        )
                        if _iket_emit:
                            _iket.range_pop()
                        confirmed_expert_idx = current_expert_idx

                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )

                md = TokenSrcMetadata.load(
                    token_src_metadata.iterator
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                src_rank = md.src_rank
                src_token = md.src_token
                src_topk = md.src_topk
                is_remote_token_back = src_rank != Int32(self.local_rank)

                local_token_addr = (
                    fc2_output_workspace.iterator.toint()
                    + Int64(pool_token_idx) * Int64(fc2_token_bytes)
                )
                peer_combine_base = peer_rank_ptr_mapper.map(
                    combine_output.iterator.toint(),
                    src_rank,
                    Int64(0),
                )
                if cutlass.const_expr(self.token_back_reduce_topk):
                    peer_token_offset = Int64(src_token) * Int64(fc2_token_bytes)
                else:
                    peer_token_offset = (
                        Int64(src_token * Int32(self.num_topk) + src_topk)
                        * Int64(fc2_token_bytes)
                    )
                peer_token_addr = peer_combine_base + peer_token_offset

                smem_ptr_warp = pull_buffer_ptr + warp_idx * Int32(chunk_bytes)
                mbar_ptr_warp = pull_mbar_ptr + warp_idx

                if _iket_emit:
                    _iket.range_push("token_back_copy")
                cute.arch.sync_warp()

                for chunk in cutlass.range(num_chunks, unroll=1):
                    t0 = read_clock64()
                    chunk_off = Int64(chunk * chunk_bytes)
                    peer_chunk_addr = peer_token_addr + chunk_off

                    this_bytes = Int32(chunk_bytes)
                    if cutlass.const_expr(last_chunk_bytes != chunk_bytes):
                        if chunk == Int32(num_chunks - 1):
                            this_bytes = Int32(last_chunk_bytes)

                    if cutlass.const_expr(self.token_back_reduce_topk):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr_warp, this_bytes,
                            )
                            tma_load_1d_raw(
                                smem_ptr_warp,
                                local_token_addr + chunk_off,
                                mbar_ptr_warp,
                                this_bytes,
                            )
                            cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                        cute.arch.sync_warp()
                        for redg_iter in cutlass.range_constexpr(
                            0, redg_iters_per_full_chunk, 1
                        ):
                            byte_off = Int32(
                                redg_iter * redg_bytes_per_warp
                            ) + lane_idx * Int32(redg_bytes_per_lane)
                            if byte_off + Int32(redg_bytes_per_lane) <= this_bytes:
                                v0 = lds_b32_raw(smem_ptr_warp + byte_off)
                                v1 = lds_b32_raw(
                                    smem_ptr_warp
                                    + byte_off
                                    + Int32(4)
                                )
                                red_add_relaxed_sys_v2_bf16x2_raw(
                                    peer_chunk_addr + Int64(byte_off),
                                    v0,
                                    v1,
                                )
                    else:
                        for copy_iter in cutlass.range_constexpr(
                            0, copy_iters_per_full_chunk, 1
                        ):
                            byte_off = Int32(
                                copy_iter * copy_bytes_per_warp
                            ) + lane_idx * Int32(copy_bytes_per_lane)
                            if byte_off + Int32(copy_bytes_per_lane) <= this_bytes:
                                v = ldg_b32_raw(
                                    local_token_addr
                                    + chunk_off
                                    + Int64(byte_off)
                                )
                                stg_b32_raw(peer_chunk_addr + Int64(byte_off), v)
                        cute.arch.fence_acq_rel_sys()
                    phase_bit = phase_bit ^ Int32(1)
                    t1 = read_clock64()
                    current_window = Int32(t1 - t0)
                    if is_remote_token_back and remain_experts > Int32(4):
                        avg_token_back_window = self._adaptive_pace(
                            avg_token_back_window, current_window, lo=1000, hi=5000,
                        )

                if _iket_emit:
                    _iket.range_pop()

                token_idx, batch_remaining = update_token_idx(
                    token_idx, batch_remaining, lane_idx,
                    token_back_schedule_counter,
                    schedule_mode, atomic_batch, num_global_warps,
                )
        # if lane_idx == 0:
        #     cute.printf("<{}>", avg_token_back_window)

        cute.arch.fence_acq_rel_sys()

    @cute.jit
    def nvlink_barrier(
        self,
        nvlink_barrier_signal,
        nvlink_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        slot: cutlass.Constexpr[int],
        num_sms,
        prologue_grid_sync: cutlass.Constexpr[bool],
        epilogue_grid_sync: cutlass.Constexpr[bool],
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        if prologue_grid_sync:
            software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                               num_threads=self.num_dispatch_threads)

        if sm_idx == 0:
            if warp_idx == 0:
                signal_phase = Int32(slot)
                target = Int32(1)
                if cutlass.const_expr(nvlink_barrier_counter is not None):
                    status = nvlink_barrier_counter[0] & Int32(3)
                    signal_phase = status & Int32(1)
                    signal_sign = status >> Int32(1)
                    if signal_sign != Int32(0):
                        target = Int32(0)

                nbs_local_base = nvlink_barrier_signal.iterator.toint()
                if lane_idx < Int32(self.world_size):
                    signal_slot = (
                        signal_phase * Int32(self.world_size)
                        + Int32(self.local_rank)
                    )
                    lane_peer_addr = peer_rank_ptr_mapper.map(
                        nbs_local_base, lane_idx,
                        Int64(signal_slot * Int32(4)),
                    )
                    stg_b32_raw(lane_peer_addr, target)
                cute.arch.fence_acq_rel_sys()
                cute.arch.sync_warp()

                if lane_idx == 0:
                    if cutlass.const_expr(nvlink_barrier_counter is not None):
                        cute.arch.atomic_add(
                            nvlink_barrier_counter.iterator,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    ready = Int32(0)
                    while ready == Int32(0):
                        all_ready = Int32(1)
                        for rank in cutlass.range_constexpr(0, self.world_size, 1):
                            local_signal_ptr = (
                                nvlink_barrier_signal.iterator
                                + signal_phase * Int32(self.world_size)
                                + Int32(rank)
                            )
                            if cute.arch.load(
                                local_signal_ptr,
                                Int32,
                                sem="acquire",
                                scope="sys",
                            ) != target:
                                all_ready = Int32(0)
                        if all_ready != Int32(0):
                            ready = Int32(1)

        if epilogue_grid_sync:
            software_grid_sync(grid_sync_counter, sm_idx, num_sms, tid_in_group,
                               num_threads=self.num_dispatch_threads)

    @cute.jit
    def dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx)
            + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
            * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        iket_active = (cta_linear_id == Int32(0)) and (local_warp_idx == Int32(0))
        if iket_active:
            _iket.range_push("Dispatch_Prep")

        self.dispatch_prep(
            token_comm_storage,
            token_comm_args.topk_idx,
            token_comm_args.expert_send_count,
            token_comm_args.src_token_topk_idx,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            local_rank=self.local_rank,
            num_tokens=token_comm_args.input_token_buffer.shape[0],
            num_sms=token_comm_args.sm_count,
        )

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Barrier")

        self.dispatch_barrier(
            token_comm_args.expert_send_count,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.nvlink_barrier_signal,
            token_comm_args.grid_sync_counter,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            nvlink_barrier_counter=token_comm_args.nvlink_barrier_counter,
        )

        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        if cutlass.const_expr(self.dispatch_compute_overlap):
            # Expert sizes are globally visible.  Publish scheduler work now;
            # each FC1 tile independently waits on its dispatch-ready counter.
            nb_dispatch_to_sched.arrive()

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Pull")

        phase_bit, stored_num_tokens_per_expert = self.dispatch_pull(
            token_comm_storage,
            token_comm_args.input_token_buffer,
            token_comm_args.input_sf_buffer,
            token_comm_args.input_topk_weights_buffer,
            token_comm_args.src_token_topk_idx,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.fc1_input_token_buffer,
            token_comm_args.fc1_input_sf_buffer,
            token_comm_args.fc1_input_topk_weights_buffer,
            token_comm_args.fc1_ready_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
        )

        if cutlass.const_expr(not self.dispatch_compute_overlap):
            # A/B reference path: preserve the same pull algorithm and all
            # other kernel constants, but delay compute until pull completes.
            nb_dispatch_to_sched.arrive()

        if iket_active:
            _iket.range_pop()

        if cutlass.const_expr(self.enable_token_back and not self.token_back_standalone):
            if iket_active:
                _iket.range_push("Token_Back_By_Push")

            self.token_back_by_push(
                token_comm_storage.pull_buffer.data_ptr(),
                token_comm_storage.pull_mbar.data_ptr(),
                token_comm_args.fc2_output_workspace,
                token_comm_args.fc2_done_counter,
                token_comm_args.token_src_metadata,
                token_comm_args.combine_output,
                token_comm_args.token_back_schedule_counter,
                token_comm_args.peer_rank_ptr_mapper,
                phase_bit,
                stored_num_tokens_per_expert,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
                chunk_bytes=self.hidden_bytes,
            )

            if iket_active:
                _iket.range_pop()

    @cute.jit
    def token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx)
            + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
            * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.token_back_warp_start)

        # Handshake: dispatch_barrier done => expert_recv_count_sum populated.
        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive_and_wait()

        tb_pull_mbar_ptr = token_comm_storage.tb_pull_mbar.data_ptr()
        tb_pull_buffer_ptr = token_comm_storage.tb_pull_buffer.data_ptr()
        tb_mbar_ptr_warp = tb_pull_mbar_ptr + local_warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(tb_mbar_ptr_warp, 1)
        cute.arch.sync_warp()

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = token_comm_args.expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
        cute.arch.sync_warp()

        iket_active = (cta_linear_id == Int32(0)) and (local_warp_idx == Int32(0))
        if iket_active:
            _iket.range_push("Token_Back_By_Push_Standalone")

        self.token_back_by_push(
            tb_pull_buffer_ptr,
            tb_pull_mbar_ptr,
            token_comm_args.fc2_output_workspace,
            token_comm_args.fc2_done_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.combine_output,
            token_comm_args.token_back_schedule_counter,
            token_comm_args.peer_rank_ptr_mapper,
            Int32(0),
            stored_num_tokens_per_expert,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            chunk_bytes=self.tb_chunk_bytes,
        )

        if iket_active:
            _iket.range_pop()

    @cute.jit
    def tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        """Reset the original SM120 workspace counters in place."""
        thread_linear = (
            (cta_linear_id * Int32(self.num_dispatch_warps) + local_warp_idx)
            * Int32(self.warp_threads)
            + lane_idx
        )
        stride = Int32(token_comm_args.sm_count * self.num_dispatch_threads)

        recv_total: cutlass.Constexpr[int] = (
            self.world_size * self.num_experts_per_rank
        )
        i = thread_linear
        while i < Int32(recv_total):
            rank_idx = i // Int32(self.num_experts_per_rank)
            expert_idx = i % Int32(self.num_experts_per_rank)
            token_comm_args.expert_recv_count[rank_idx, expert_idx] = Int64(0)
            i = i + stride

        i = thread_linear
        while i < Int32(self.num_experts_per_rank):
            token_comm_args.expert_recv_count_sum[i] = Int64(0)
            i = i + stride

        if cutlass.const_expr(self.enable_token_back):
            i = thread_linear
            while i < Int32(self.num_experts_per_rank):
                token_comm_args.fc2_done_counter[i] = Int32(0)
                i = i + stride

        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            if thread_linear == Int32(0):
                token_comm_args.token_back_schedule_counter.store(Int32(0))

    @cute.jit
    def kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Preserve the SM120 three-barrier cleanup protocol."""
        nb_kernel_tail = pipeline.NamedBarrier(
            barrier_id=self.kernel_tail_named_barrier_id,
            num_threads=self.kernel_tail_threads,
        )
        nb_kernel_tail.arrive_and_wait()

        if (warp_idx >= self.dispatch_warp_start) and (
            warp_idx < self.dispatch_warp_start + self.num_dispatch_warps
        ):
            bidx, bidy, bidz = cute.arch.block_idx()
            cta_linear_id = (
                Int32(bidx)
                + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
                + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
                * Int32(bidz)
            )
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=1,
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
                slot=1,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            self.tail_reset_shared_counters(
                token_comm_args,
                cta_linear_id=cta_linear_id,
                local_warp_idx=local_warp_idx,
                lane_idx=lane_idx,
            )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=0,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
