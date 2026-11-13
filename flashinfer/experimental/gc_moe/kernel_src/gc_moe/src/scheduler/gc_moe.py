"""Two-CTA GC task handoff and the four task phase identifiers.

The 32-byte task lives in each CTA's SMEM. The scheduler warp in the leader
CTA publishes both copies with transaction-counted DSM stores; the two
epilogue warpgroups per CTA retire a slot through the leader empty barrier.
"""

from __future__ import annotations

from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64

from ..common.math import warp_inclusive_sum
from ..ptx.ld_st import (
    ld_volatile_u32,
    ld_volatile_u64,
    ld_acquire_sys_u64,
    atom_add_u32,
    st_async_cluster_u32x4,
)
from ..ptx.utils import (
    elect_one_sync,
    warp_first_set_bit,
)


class GCBlockPhase(IntEnum):
    None_ = 0
    Linear1 = 1
    Linear2 = 2
    SharedLinear1 = 3
    SharedLinear2 = 4


TASK_WORDS = 8
TASK_BYTES = 32
SCHEDULE_STAGES = 2
TASK_EMPTY_CONSUMERS = 512


@cute.jit
def publish_task32(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                   stage: Int32, peer_cta: Int32,
                   block_phase: Uint32, local_expert: Uint32,
                   m_block: Uint32, n_cluster: Uint32,
                   pool_block: Uint32, valid_m: Uint32,
                   shape_n: Uint32, shape_k: Uint32) -> None:
    cute.arch.mbarrier_arrive_and_expect_tx(full_ptr + stage, TASK_BYTES, peer_cta)
    base = stage * TASK_WORDS
    st_async_cluster_u32x4(task_ptr, full_ptr + stage, peer_cta, base,
                  block_phase, local_expert, m_block, n_cluster)
    st_async_cluster_u32x4(task_ptr, full_ptr + stage, peer_cta, base + 4,
                  pool_block, valid_m, shape_n, shape_k)


@cute.jit
def release_task_info(empty_ptr: cute.Pointer, stage: Int32) -> None:
    cute.arch.fence_acq_rel_cta()
    cute.arch.mbarrier_arrive(empty_ptr + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def claim_task(task_count_ptr: cute.Pointer, lane: Int32) -> Uint32:
    task_idx = Uint32(0)
    if elect_one_sync():
        task_idx = atom_add_u32(task_count_ptr, Uint32(1))
    return cute.arch.shuffle_sync(task_idx, 0)


@cute.jit
def wait_task_slot(empty_ptr: cute.Pointer, iteration: Uint32) -> None:
    """Wait for the next task slot before entering its claim loop."""
    stage = Int32(iteration % 2)
    parity = Int32((iteration // 2) % 2)
    cute.arch.mbarrier_wait(empty_ptr + stage, parity ^ 1)


@cute.jit
def publish_iteration(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                      iteration: Uint32,
                      lane: Int32, block_phase: Uint32,
                      local_expert: Uint32,
                      m_block: Uint32, n_cluster: Uint32,
                      pool_block: Uint32, valid_m: Uint32,
                      shape_n: Uint32, shape_k: Uint32) -> None:
    stage = Int32(iteration % 2)
    if lane < 2:
        publish_task32(task_ptr, full_ptr, stage, lane,
                       block_phase, local_expert, m_block, n_cluster,
                       pool_block, valid_m, shape_n, shape_k)
    cute.arch.sync_warp()


@cute.jit
def publish_shared_phase(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                         empty_ptr: cute.Pointer, counter_ptr: cute.Pointer,
                         iteration: Uint32, lane: Int32,
                         cluster_idx: Uint32,
                         phase: Uint32, num_tokens: Uint32,
                         block_m: cutlass.Constexpr,
                         n_clusters: cutlass.Constexpr,
                         num_compute_sms: cutlass.Constexpr,
                         shape_n: cutlass.Constexpr,
                         shape_k: cutlass.Constexpr) -> Uint32:
    """Reference shared_mainloop: static one task/cluster, else global claims."""
    m_blocks = (num_tokens + Uint32(block_m - 1)) // Uint32(block_m)
    total = m_blocks * Uint32(n_clusters)
    if total <= Uint32(num_compute_sms // 2):
        wait_task_slot(empty_ptr, iteration)
        if cluster_idx < Uint32(total):
            m_block = cluster_idx // Uint32(n_clusters)
            n_cluster = cluster_idx % Uint32(n_clusters)
            valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
            publish_iteration(task_ptr, full_ptr, iteration,
                              lane, phase, Uint32(0), m_block, n_cluster, m_block,
                              valid_m, Uint32(shape_n), Uint32(shape_k))
            iteration += Uint32(1)
            # Match the reference's wait before the static path's failed claim.
            wait_task_slot(empty_ptr, iteration)
    else:
        scheduling = True
        while scheduling:
            wait_task_slot(empty_ptr, iteration)
            task_idx = claim_task(counter_ptr, lane)
            if task_idx >= Uint32(total):
                scheduling = False
            else:
                m_block = task_idx // Uint32(n_clusters)
                n_cluster = task_idx % Uint32(n_clusters)
                valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
                publish_iteration(task_ptr, full_ptr, iteration,
                                  lane, phase, Uint32(0), m_block, n_cluster, m_block,
                                  valid_m, Uint32(shape_n), Uint32(shape_k))
                iteration += Uint32(1)
    return iteration


@cute.jit
def locate_routed_pool_block(counts: cute.Tensor,
                             pool_block: Uint32, lane: Uint32,
                             local_experts: cutlass.Constexpr,
                             block_m: cutlass.Constexpr):
    """Reference GCMoEScheduler::create_task from per-lane count registers."""
    expert = Uint32(0)
    m_block = Uint32(0)
    valid_m = Uint32(0)
    block_offset = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        expert_idx = Uint32(wave * 32) + lane
        received = counts[wave]
        blocks = (received + Uint32(block_m - 1)) // Uint32(block_m)
        inclusive = warp_inclusive_sum(blocks, lane)
        lane_offset = block_offset + inclusive - blocks
        is_owner = expert_idx < Uint32(local_experts) and \
            pool_block >= lane_offset and pool_block < lane_offset + blocks
        owner_mask = Uint32(cute.arch.vote_ballot_sync(is_owner))
        if owner_mask != Uint32(0):
            owner_lane = warp_first_set_bit(owner_mask)
            local_m = pool_block - lane_offset
            local_valid = min(Uint32(block_m),
                              received - local_m * Uint32(block_m))
            expert = cute.arch.shuffle_sync(expert_idx, owner_lane)
            m_block = cute.arch.shuffle_sync(local_m, owner_lane)
            valid_m = cute.arch.shuffle_sync(local_valid, owner_lane)
        block_offset += cute.arch.shuffle_sync(inclusive, 31)
    return expert, m_block, valid_m


@cute.jit
def cached_num_tokens(counts: cute.Tensor, expert: Int32,
                      lane: Uint32,
                      local_experts: cutlass.Constexpr) -> Uint32:
    """GCMoEScheduler::get_num_tokens from expert wave*32+lane registers."""
    selected = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        if Uint32(expert) == Uint32(wave * 32) + lane:
            selected = counts[wave]
    return cute.arch.shuffle_sync(selected, Uint32(expert) % Uint32(32))


@cute.jit
def advance_expert(counts: cute.Tensor, token_idx: Uint32,
                   current_expert: Int32, expert_start: Uint32,
                   expert_end: Uint32, pool_block_offset: Uint32,
                   lane: Uint32, local_experts: cutlass.Constexpr,
                   block_m: cutlass.Constexpr):
    """Advance the unpadded token interval and its padded pool prefix."""
    while token_idx >= expert_end and current_expert < Int32(local_experts):
        current_expert += Int32(1)
        if current_expert < Int32(local_experts):
            pool_block_offset += (expert_end - expert_start +
                                  Uint32(block_m - 1)) // Uint32(block_m)
            expert_start = expert_end
            expert_end += cached_num_tokens(
                counts, current_expert, lane, local_experts)
    return current_expert, expert_start, expert_end, pool_block_offset


@cute.jit
def schedule_tasks(task: cute.Pointer, full: cute.Pointer, empty: cute.Pointer,
                   recv_ptr: cute.Pointer, l1_count_ptr: cute.Pointer,
                   l2_count_ptr: cute.Pointer, shared_l1_count_ptr: cute.Pointer,
                   shared_l2_count_ptr: cute.Pointer, expert_counts: cute.Tensor,
                   iteration: Uint32, lane: Int32, block: Int32, num_tokens: Uint32,
                   block_m: cutlass.Constexpr,
                   compute_sms: cutlass.Constexpr,
                   dispatch_sms: cutlass.Constexpr,
                   hidden: cutlass.Constexpr,
                   intermediate: cutlass.Constexpr,
                   l1_clusters: cutlass.Constexpr,
                   l2_clusters: cutlass.Constexpr,
                   local_experts: cutlass.Constexpr,
                   num_ranks: cutlass.Constexpr,
                   shared_experts: cutlass.Constexpr,
) -> None:
    """Publish shared and interleaved routed tasks, then the end sentinel."""
    shared_m_blocks = (num_tokens + Uint32(block_m - 1)) // \
        Uint32(block_m)
    shared_l2_early = (shared_m_blocks * Uint32(l2_clusters) <=
                       Uint32(compute_sms))
    if cutlass.const_expr(shared_experts > 0):
        shared_l1_clusters = l1_clusters * shared_experts
        iteration = publish_shared_phase(
            task, full, empty, shared_l1_count_ptr, iteration,
            Int32(lane), Uint32(block // 2),
            Uint32(GCBlockPhase.SharedLinear1),
            num_tokens, block_m, shared_l1_clusters,
            compute_sms, 2 * intermediate * shared_experts,
            hidden)
        if shared_l2_early:
            iteration = publish_shared_phase(
                task, full, empty, shared_l2_count_ptr, iteration,
                Int32(lane), Uint32(block // 2),
                Uint32(GCBlockPhase.SharedLinear2),
                num_tokens, block_m, l2_clusters,
                compute_sms, hidden,
                intermediate * shared_experts)
    # Each lane owns experts wave*32+lane in registers, as in
    # GCMoEScheduler. The count cache survives routed L1/L2 task
    # claims, without an E-dependent shared prefix allocation.
    for wave in cutlass.range_constexpr(
            (local_experts + 31) // 32):
        local_expert = Uint32(wave * 32 + lane)
        count = Uint64(0)
        if local_expert < Uint32(local_experts):
            while Uint32(count >> 32) != Uint32(
                    dispatch_sms * num_ranks):
                count = ld_volatile_u64(recv_ptr + local_expert * 2)
            count = ld_acquire_sys_u64(recv_ptr + local_expert * 2)
        expert_counts[wave] = Uint32(count)
    cute.arch.sync_warp()
    lane_m_blocks = Uint32(0)
    for wave in cutlass.range_constexpr(
            (local_experts + 31) // 32):
        received = expert_counts[wave]
        own_blocks = (received + Uint32(block_m - 1)) // \
            Uint32(block_m)
        lane_m_blocks += own_blocks
    routed_m_blocks = cute.arch.warp_redux_sync(
        lane_m_blocks, 'add')
    # Match GCMoEScheduler::get_gc_num_l1_warmup_waves and
    # get_next_task for one locality domain: enough L1 work is
    # issued before the first L2 wave, then one L1 follows each
    # successful L2 claim until the L1 queue is exhausted.
    num_clusters = compute_sms // 2
    first_l2_m_blocks = (num_clusters + l2_clusters - 1) // l2_clusters
    first_l2_warmup = (first_l2_m_blocks * l1_clusters +
                       num_clusters - 1) // num_clusters
    interleave_diff = max(l1_clusters - l2_clusters, 0)
    l1_warmup = Uint32(0)
    if routed_m_blocks > 0:
        interleave_warmup = (l1_clusters +
                             (routed_m_blocks - 1) * interleave_diff +
                             num_clusters - 1) // num_clusters + 1
        total_l1_waves = (routed_m_blocks * l1_clusters +
                          num_clusters - 1) // num_clusters
        l1_warmup = min(total_l1_waves,
                        max(Uint32(first_l2_warmup), interleave_warmup))
    l1_done = Uint32(0xffffffff)
    scheduling = True
    while scheduling:
        wait_task_slot(empty, iteration)
        fetching = True
        while fetching:
            if l1_warmup != l1_done and l1_warmup > 0:
                l1_warmup -= Uint32(1)
                task_idx = claim_task(l1_count_ptr, Int32(lane))
                if task_idx >= routed_m_blocks * l1_clusters:
                    l1_warmup = l1_done
                else:
                    pool_block = task_idx // l1_clusters
                    n_cluster = task_idx % l1_clusters
                    local_expert, m_block, valid_m = \
                        locate_routed_pool_block(
                            expert_counts, pool_block, Uint32(lane),
                            local_experts, block_m)
                    publish_iteration(task, full, iteration, Int32(lane),
                                      Uint32(GCBlockPhase.Linear1), local_expert,
                                      m_block, n_cluster, pool_block, valid_m,
                                      Uint32(2 * intermediate), Uint32(hidden))
                    iteration = Uint32(iteration + Uint32(1))
                    fetching = False
            else:
                task_idx = claim_task(l2_count_ptr, Int32(lane))
                if task_idx >= routed_m_blocks * l2_clusters:
                    scheduling = False
                    fetching = False
                else:
                    if l1_warmup != l1_done:
                        l1_warmup = Uint32(1)
                    pool_block = task_idx // l2_clusters
                    n_cluster = task_idx % l2_clusters
                    local_expert, m_block, valid_m = \
                        locate_routed_pool_block(
                            expert_counts, pool_block, Uint32(lane),
                            local_experts, block_m)
                    required_l1 = (pool_block + 1) * l1_clusters
                    while ld_volatile_u32(l1_count_ptr) < required_l1:
                        pass
                    publish_iteration(task, full, iteration, Int32(lane),
                                      Uint32(GCBlockPhase.Linear2), local_expert,
                                      m_block, n_cluster, pool_block, valid_m,
                                      Uint32(hidden), Uint32(intermediate))
                    iteration = Uint32(iteration + Uint32(1))
                    fetching = False
    if cutlass.const_expr(shared_experts > 0):
        if not shared_l2_early:
            iteration = publish_shared_phase(
                task, full, empty, shared_l2_count_ptr, iteration,
                Int32(lane), Uint32(block // 2),
                Uint32(GCBlockPhase.SharedLinear2),
                num_tokens, block_m, l2_clusters,
                compute_sms, hidden,
                intermediate * shared_experts)
    wait_task_slot(empty, iteration)
    publish_iteration(task, full, iteration, Int32(lane),
                      Uint32(GCBlockPhase.None_), Uint32(0),
                      Uint32(0), Uint32(0),
                      Uint32(0), Uint32(0), Uint32(0), Uint32(0))
