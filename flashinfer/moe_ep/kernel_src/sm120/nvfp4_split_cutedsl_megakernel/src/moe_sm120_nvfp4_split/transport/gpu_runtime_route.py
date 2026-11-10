"""GPU-only route/control-plan builders for dynamic MoE routing.

The builders deliberately use fixed-capacity outputs.  No tensor shape or
host scalar depends on the current ``topk_idx``; every replay may therefore
install a different routing table without a CPU synchronization.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64
from cuda.bindings import driver as cuda

from ..moe_utils import spin_wait

class GpuDispatchRouteBuilder:
    """Build sparse data lists and source-side owner-plan route chunks."""

    def __init__(
        self,
        world_size: int,
        num_tokens: int,
        num_topk: int,
        num_experts_per_rank: int,
        owner_route_chunk_elements: int = 255,
        owner_route_capacity: int = 0,
        owner_count_record_elements: int = 256,
        owner_plan_slots: int = 2,
    ) -> None:
        if world_size > 31:
            raise ValueError("GPU dispatch owner mask supports at most 31 ranks")
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts_per_rank
        self.owner_route_chunk_elements = owner_route_chunk_elements
        self.owner_route_block_elements = owner_route_chunk_elements + 1
        self.max_routes_per_owner = num_tokens * num_topk
        self.owner_route_capacity = owner_route_capacity
        self.owner_count_record_elements = owner_count_record_elements
        self.owner_plan_slots = owner_plan_slots
        if owner_route_capacity and owner_route_capacity < (
            self.max_routes_per_owner
            + (self.max_routes_per_owner + owner_route_chunk_elements - 1)
            // owner_route_chunk_elements
            + num_experts_per_rank
        ):
            raise ValueError("owner route payload capacity is too small")

    @cute.kernel
    def kernel(
        self,
        topk_idx: cute.Tensor,
        owner_expert_counts: cute.Tensor,
        owner_token_counts: cute.Tensor,
        epoch: Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        token = block * Int32(256) + tid
        if token < Int32(self.num_tokens):
            plan_slot = epoch % Int32(self.owner_plan_slots)
            for slot in cutlass.range_constexpr(0, self.num_topk, 1):
                expert = Int32(
                    topk_idx[token * Int32(self.num_topk) + Int32(slot)]
                )
                if expert >= Int32(0):
                    owner = expert // Int32(self.num_experts_per_rank)
                    if owner < Int32(self.world_size):
                        local_expert = expert - (
                            owner * Int32(self.num_experts_per_rank)
                        )
                        cute.arch.atomic_add(
                            owner_expert_counts.iterator
                            + plan_slot
                            * Int32(
                                self.world_size
                                * self.owner_count_record_elements
                            )
                            + owner * Int32(self.owner_count_record_elements)
                            + local_expert,
                            Int64(1),
                            sem="relaxed",
                            scope="gpu",
                        )
            # Assign each unique (token, owner) payload to the first local
            # expert that consumes it.  Prefixing these groups later makes
            # dispatch data follow the same expert-major order as the owner
            # route plan without replicating an activation shared by several
            # experts on one owner.
            for owner_const in cutlass.range_constexpr(
                0, self.world_size, 1
            ):
                min_local_expert = Int32(self.num_experts_per_rank)
                for slot in cutlass.range_constexpr(0, self.num_topk, 1):
                    expert = Int32(
                        topk_idx[
                            token * Int32(self.num_topk) + Int32(slot)
                        ]
                    )
                    owner_begin = Int32(
                        owner_const * self.num_experts_per_rank
                    )
                    owner_end = owner_begin + Int32(
                        self.num_experts_per_rank
                    )
                    if expert >= owner_begin and expert < owner_end:
                        local_expert = expert - owner_begin
                        if local_expert < min_local_expert:
                            min_local_expert = local_expert
                if min_local_expert < Int32(self.num_experts_per_rank):
                    cute.arch.atomic_add(
                        owner_token_counts.iterator
                        + Int32(owner_const * self.num_experts_per_rank)
                        + min_local_expert,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )

    @cute.kernel
    def prefix_kernel(
        self,
        owner_expert_counts: cute.Tensor,
        owner_expert_bases: cute.Tensor,
        owner_route_output_bases: cute.Tensor,
        owner_expert_cursors: cute.Tensor,
        owner_token_counts: cute.Tensor,
        owner_token_bases: cute.Tensor,
        owner_token_cursors: cute.Tensor,
        send_counts: cute.Tensor,
        epoch: Int32,
    ):
        owner, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if tid == Int32(0):
            plan_slot = epoch % Int32(self.owner_plan_slots)
            route_base = Int32(0)
            output_base = Int32(0)
            token_base = Int32(0)
            for expert in cutlass.range_constexpr(
                0, self.num_experts_per_rank, 1
            ):
                plan_index = (
                    owner * Int32(self.num_experts_per_rank)
                    + Int32(expert)
                )
                count_index = (
                    plan_slot
                    * Int32(
                        self.world_size * self.owner_count_record_elements
                    )
                    + owner * Int32(self.owner_count_record_elements)
                    + Int32(expert)
                )
                owner_expert_bases[plan_index] = route_base
                owner_route_output_bases[plan_index] = output_base
                owner_expert_cursors[plan_index] = Int32(0)
                count = Int32(owner_expert_counts[count_index])
                output_base = output_base + count
                padded = (
                    count
                    + Int32(self.owner_route_chunk_elements)
                    - Int32(1)
                ) // Int32(self.owner_route_chunk_elements)
                route_base = route_base + padded * Int32(
                    self.owner_route_block_elements
                )
                token_group_index = (
                    owner * Int32(self.num_experts_per_rank)
                    + Int32(expert)
                )
                owner_token_bases[token_group_index] = token_base
                owner_token_cursors[token_group_index] = Int32(0)
                token_base = token_base + owner_token_counts[
                    token_group_index
                ]
            send_counts[owner] = token_base

    @cute.kernel
    def fill_send_tokens_kernel(
        self,
        topk_idx: cute.Tensor,
        owner_token_bases: cute.Tensor,
        owner_token_cursors: cute.Tensor,
        send_token_indices: cute.Tensor,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        token = block * Int32(256) + tid
        if token < Int32(self.num_tokens):
            for owner_const in cutlass.range_constexpr(
                0, self.world_size, 1
            ):
                min_local_expert = Int32(self.num_experts_per_rank)
                for slot in cutlass.range_constexpr(0, self.num_topk, 1):
                    expert = Int32(
                        topk_idx[
                            token * Int32(self.num_topk) + Int32(slot)
                        ]
                    )
                    owner_begin = Int32(
                        owner_const * self.num_experts_per_rank
                    )
                    owner_end = owner_begin + Int32(
                        self.num_experts_per_rank
                    )
                    if expert >= owner_begin and expert < owner_end:
                        local_expert = expert - owner_begin
                        if local_expert < min_local_expert:
                            min_local_expert = local_expert
                if min_local_expert < Int32(self.num_experts_per_rank):
                    group_index = Int32(
                        owner_const * self.num_experts_per_rank
                    ) + min_local_expert
                    position = cute.arch.atomic_add(
                        owner_token_cursors.iterator + group_index,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    send_token_indices[
                        Int32(owner_const * self.num_tokens)
                        + owner_token_bases[group_index]
                        + position
                    ] = token

    @cute.kernel
    def fill_owner_routes_kernel(
        self,
        topk_idx: cute.Tensor,
        owner_expert_bases: cute.Tensor,
        owner_route_output_bases: cute.Tensor,
        owner_expert_cursors: cute.Tensor,
        owner_routes: cute.Tensor,
        combine_recv_output_rows: cute.Tensor,
        epoch: Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        route = block * Int32(256) + tid
        total_routes = Int32(self.num_tokens * self.num_topk)
        if route < total_routes:
            plan_slot = epoch % Int32(self.owner_plan_slots)
            expert = Int32(topk_idx[route])
            if expert >= Int32(0):
                owner = expert // Int32(self.num_experts_per_rank)
                if owner < Int32(self.world_size):
                    local_expert = expert - (
                        owner * Int32(self.num_experts_per_rank)
                    )
                    plan_index = (
                        owner * Int32(self.num_experts_per_rank)
                        + local_expert
                    )
                    position = cute.arch.atomic_add(
                        owner_expert_cursors.iterator + plan_index,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    route_offset = (
                        owner_expert_bases[plan_index]
                        + (
                            position
                            // Int32(self.owner_route_chunk_elements)
                        )
                        * Int32(self.owner_route_block_elements)
                        + position % Int32(self.owner_route_chunk_elements)
                    )
                    owner_routes[
                        plan_slot
                        * Int32(
                            self.world_size * self.owner_route_capacity
                        )
                        + owner * Int32(self.owner_route_capacity)
                        + route_offset
                    ] = Int64(route)
                    combine_recv_output_rows[
                        owner * Int32(self.max_routes_per_owner)
                        + owner_route_output_bases[plan_index]
                        + position
                    ] = route

    @cute.jit
    def launcher(
        self,
        topk_idx: cute.Tensor,
        send_token_indices: cute.Tensor,
        send_counts: cute.Tensor,
        owner_expert_counts: cute.Tensor,
        owner_expert_bases: cute.Tensor,
        owner_route_output_bases: cute.Tensor,
        owner_expert_cursors: cute.Tensor,
        owner_token_counts: cute.Tensor,
        owner_token_bases: cute.Tensor,
        owner_token_cursors: cute.Tensor,
        owner_routes: cute.Tensor,
        combine_recv_output_rows: cute.Tensor,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            topk_idx,
            owner_expert_counts,
            owner_token_counts,
            epoch,
        ).launch(
            grid=[(self.num_tokens + 255) // 256, 1, 1],
            block=[256, 1, 1],
            stream=stream,
        )
        self.prefix_kernel(
            owner_expert_counts,
            owner_expert_bases,
            owner_route_output_bases,
            owner_expert_cursors,
            owner_token_counts,
            owner_token_bases,
            owner_token_cursors,
            send_counts,
            epoch,
        ).launch(
            grid=[self.world_size, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )
        self.fill_send_tokens_kernel(
            topk_idx,
            owner_token_bases,
            owner_token_cursors,
            send_token_indices,
        ).launch(
            grid=[(self.num_tokens + 255) // 256, 1, 1],
            block=[256, 1, 1],
            stream=stream,
        )
        self.fill_owner_routes_kernel(
            topk_idx,
            owner_expert_bases,
            owner_route_output_bases,
            owner_expert_cursors,
            owner_routes,
            combine_recv_output_rows,
            epoch,
        ).launch(
            grid=[
                (self.num_tokens * self.num_topk + 255) // 256,
                1,
                1,
            ],
            block=[256, 1, 1],
            stream=stream,
        )



class GpuChunkedOwnerRoutePlanBuilder:
    """Install count-first extents, then source/expert metadata chunks.

    The count launcher is the only dependency of the K1/K2 graph: it waits for
    one compact record from every source and publishes stable expert extents.
    The metadata launcher runs on a communication stream, waits independently
    for each source/expert route chunk, copies it into K1's existing metadata
    layout, builds the corresponding combine rows, and finally publishes a
    local replay-epoch readiness word consumed by K1.
    """

    def __init__(
        self,
        world_size: int,
        num_tokens: int,
        num_topk: int,
        num_experts_per_rank: int,
        token_padding_block: int,
        fc2_token_tile_tokens: int,
        fc2_publishes_per_token_cluster_tile: int,
        producer_sm_count: int,
        route_capacity: int,
        *,
        route_block_elements: int = 256,
        plan_slots: int = 2,
    ) -> None:
        self.world_size = world_size
        self.num_tokens = num_tokens
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts_per_rank
        self.max_routes_per_source = num_tokens * num_topk
        self.token_padding_block = token_padding_block
        self.fc2_token_tile_tokens = fc2_token_tile_tokens
        self.fc2_publishes_per_token_cluster_tile = (
            fc2_publishes_per_token_cluster_tile
        )
        self.producer_sm_count = producer_sm_count
        self.route_capacity = route_capacity
        self.route_block_elements = route_block_elements
        self.route_payload_elements = route_block_elements - 1
        self.count_record_elements = route_block_elements
        self.commit_elements = 1
        self.plan_slots = plan_slots

    @cute.kernel
    def install_counts_kernel(
        self,
        count_records: cute.Tensor,
        count_signals: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        expert_pool_bases: cute.Tensor,
        combine_source_expert_bases: cute.Tensor,
        combine_send_counts: cute.Tensor,
        source_expert_route_bases: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        plan_slot = epoch % Int32(self.plan_slots)
        count_slot_base = plan_slot * Int32(
            self.world_size * self.count_record_elements
        )
        if tid < Int32(self.world_size):
            source = tid
            spin_wait(
                count_signals.iterator
                + (
                    plan_slot * Int32(self.world_size)
                    + source
                )
                * Int32(self.commit_elements),
                lambda value: value >= Int64(epoch),
                fail_sleep_cycles=500,
            )
            route_base = Int32(0)
            for expert in cutlass.range_constexpr(
                0, self.num_experts_per_rank, 1
            ):
                count = Int32(
                    count_records[
                        count_slot_base
                        + source * Int32(self.count_record_elements)
                        + Int32(expert)
                    ]
                )
                expert_recv_count[
                    source * Int32(self.num_experts_per_rank)
                    + Int32(expert)
                ] = Int64(count)
                source_expert_route_bases[
                    source * Int32(self.num_experts_per_rank)
                    + Int32(expert)
                ] = route_base
                chunks = (
                    count
                    + Int32(self.route_payload_elements)
                    - Int32(1)
                ) // Int32(self.route_payload_elements)
                route_base = route_base + chunks * Int32(
                    self.route_block_elements
                )
        cute.arch.fence_acq_rel_sys()
        cute.arch.barrier(barrier_id=0, number_of_threads=128)
        if tid == Int32(0):
            pool_base = Int32(0)
            for expert in cutlass.range_constexpr(
                0, self.num_experts_per_rank, 1
            ):
                expert_pool_bases[Int32(expert)] = pool_base
                total = Int32(0)
                for source in cutlass.range_constexpr(
                    0, self.world_size, 1
                ):
                    total = total + Int32(
                        expert_recv_count[
                            Int32(
                                source * self.num_experts_per_rank
                                + expert
                            )
                        ]
                    )
                publishers = Int64(
                    self.world_size * self.producer_sm_count
                )
                expert_recv_count_sum[Int32(expert)] = (
                    (publishers << Int64(32))
                    | (Int64(total) & Int64(0xFFFFFFFF))
                )
                padded = (
                    total + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                pool_base = pool_base + padded * Int32(
                    self.token_padding_block
                )
            for source in cutlass.range_constexpr(0, self.world_size, 1):
                source_base = Int32(0)
                for expert in cutlass.range_constexpr(
                    0, self.num_experts_per_rank, 1
                ):
                    combine_source_expert_bases[
                        Int32(source * self.num_experts_per_rank + expert)
                    ] = source_base
                    source_base = source_base + Int32(
                        expert_recv_count[
                            Int32(
                                source * self.num_experts_per_rank
                                + expert
                            )
                        ]
                    )
                combine_send_counts[Int32(source)] = source_base
        cute.arch.fence_acq_rel_sys()

    @cute.kernel
    def install_metadata_kernel(
        self,
        route_records: cute.Tensor,
        route_signals: cute.Tensor,
        source_expert_route_bases: cute.Tensor,
        src_token_topk_idx: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        expert_pool_bases: cute.Tensor,
        combine_source_expert_bases: cute.Tensor,
        combine_send_pool_rows: cute.Tensor,
        combine_send_output_rows: cute.Tensor,
        combine_send_ready_slots: cute.Tensor,
        combine_send_ready_values: cute.Tensor,
        combine_pool_to_stage_rows: cute.Tensor,
        owner_plan_ready: cute.Tensor,
        epoch: Int32,
    ):
        source, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        plan_slot = epoch % Int32(self.plan_slots)
        source_record_base = (
            plan_slot
            * Int32(self.world_size * self.route_capacity)
            + source * Int32(self.route_capacity)
        )
        for expert in cutlass.range_constexpr(
            0, self.num_experts_per_rank, 1
        ):
            source_signal_base = (
                (
                    plan_slot
                    * Int32(
                        self.world_size * self.num_experts_per_rank
                    )
                    + source * Int32(self.num_experts_per_rank)
                    + Int32(expert)
                )
                * Int32(self.commit_elements)
            )
            if tid == Int32(0):
                spin_wait(
                    route_signals.iterator + source_signal_base,
                    lambda value: value >= Int64(epoch),
                    fail_sleep_cycles=500,
                )
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            count_index = (
                source * Int32(self.num_experts_per_rank)
                + Int32(expert)
            )
            count = Int32(expert_recv_count[count_index])
            route_base = source_expert_route_bases[count_index]
            chunks = (
                count + Int32(self.route_payload_elements) - Int32(1)
            ) // Int32(self.route_payload_elements)
            position = tid
            while position < count:
                route_offset = (
                    route_base
                    + (
                        position // Int32(self.route_payload_elements)
                    )
                    * Int32(self.route_block_elements)
                    + position % Int32(self.route_payload_elements)
                )
                route_in_source = Int32(
                    route_records[source_record_base + route_offset]
                )
                metadata_index = (
                    (
                        Int32(expert) * Int32(self.world_size)
                        + source
                    )
                    * Int32(self.max_routes_per_source)
                    + position
                )
                src_token_topk_idx[metadata_index] = route_in_source

                position_in_expert = Int32(0)
                for other_source in cutlass.range_constexpr(
                    0, self.world_size, 1
                ):
                    other_count = Int32(
                        expert_recv_count[
                            Int32(
                                other_source * self.num_experts_per_rank
                                + expert
                            )
                        ]
                    )
                    position_in_expert = position_in_expert + cutlass.min(
                        position,
                        other_count,
                    )
                    if (
                        Int32(other_source) < source
                        and other_count > position
                    ):
                        position_in_expert = position_in_expert + Int32(1)

                pool_row = (
                    expert_pool_bases[Int32(expert)]
                    + position_in_expert
                )
                send_position = (
                    combine_source_expert_bases[count_index] + position
                )
                send_index = (
                    source * Int32(self.max_routes_per_source)
                    + send_position
                )
                combine_send_pool_rows[send_index] = Int64(pool_row)
                combine_pool_to_stage_rows[pool_row] = send_index
                combine_send_output_rows[send_index] = route_in_source
                packed_sum = expert_recv_count_sum[Int32(expert)]
                total_for_expert = Int32(
                    Int64(packed_sum) & Int64(0xFFFFFFFF)
                )
                ready_tiles = (
                    total_for_expert
                    + Int32(self.fc2_token_tile_tokens)
                    - Int32(1)
                ) // Int32(self.fc2_token_tile_tokens)
                combine_send_ready_values[send_index] = (
                    ready_tiles
                    * Int32(
                        self.fc2_publishes_per_token_cluster_tile
                    )
                )
                position = position + Int32(128)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            position = tid
            while position < count:
                send_position = (
                    combine_source_expert_bases[count_index] + position
                )
                send_index = (
                    source * Int32(self.max_routes_per_source)
                    + send_position
                )
                combine_send_ready_slots[send_index] = (
                    epoch * Int32(self.num_experts_per_rank)
                    + Int32(expert)
                )
                position = position + Int32(128)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            if tid == Int32(0):
                owner_plan_ready[count_index] = epoch
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)

    @cute.jit
    def count_launcher(
        self,
        count_records: cute.Tensor,
        count_signals: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        expert_pool_bases: cute.Tensor,
        combine_source_expert_bases: cute.Tensor,
        combine_send_counts: cute.Tensor,
        source_expert_route_bases: cute.Tensor,
        own_rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.install_counts_kernel(
            count_records,
            count_signals,
            expert_recv_count,
            expert_recv_count_sum,
            expert_pool_bases,
            combine_source_expert_bases,
            combine_send_counts,
            source_expert_route_bases,
            own_rank,
            epoch,
        ).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.jit
    def metadata_launcher(
        self,
        route_records: cute.Tensor,
        route_signals: cute.Tensor,
        source_expert_route_bases: cute.Tensor,
        src_token_topk_idx: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        expert_pool_bases: cute.Tensor,
        combine_source_expert_bases: cute.Tensor,
        combine_send_pool_rows: cute.Tensor,
        combine_send_output_rows: cute.Tensor,
        combine_send_ready_slots: cute.Tensor,
        combine_send_ready_values: cute.Tensor,
        combine_pool_to_stage_rows: cute.Tensor,
        owner_plan_ready: cute.Tensor,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.install_metadata_kernel(
            route_records,
            route_signals,
            source_expert_route_bases,
            src_token_topk_idx,
            expert_recv_count,
            expert_recv_count_sum,
            expert_pool_bases,
            combine_source_expert_bases,
            combine_send_pool_rows,
            combine_send_output_rows,
            combine_send_ready_slots,
            combine_send_ready_values,
            combine_pool_to_stage_rows,
            owner_plan_ready,
            epoch,
        ).launch(
            grid=[self.world_size, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )
