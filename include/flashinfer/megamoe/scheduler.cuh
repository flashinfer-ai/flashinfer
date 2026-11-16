#pragma once

#include <cutlass/arch/barrier.h>

#include <flashinfer/megamoe/detail/common.cuh>
#include <flashinfer/megamoe/detail/ptx.cuh>
#include <flashinfer/megamoe/layout.cuh>

namespace flashinfer::megamoe::sched {

// Computation phase for the current block
enum class BlockPhase { None = 0, Linear1 = 1, Linear2 = 2 };

template <
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K, uint32_t kNumExpertsPerRank,
    uint32_t kNumExpertsPerWave, uint32_t kNumSMs, uint32_t kNumRanks, uint32_t kClusterSize = 2,
    uint32_t kPoolBlockM = BLOCK_M,
    uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
    uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N, uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
    uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K, uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
  FI_MEGAMOE_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
  FI_MEGAMOE_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
  FI_MEGAMOE_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
  FI_MEGAMOE_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
  FI_MEGAMOE_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");
  FI_MEGAMOE_STATIC_ASSERT(kPoolBlockM > 0, "Pool block M must be positive");

  // For 2-CTA clusters, neighbour SMs share the same m_block_idx with adjacent
  // n_block_idx; the asserts below guarantee that pairing is always possible.
  // SM90 / single-CTA paths set kClusterSize = 1 and do not need this.
  FI_MEGAMOE_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "Invalid cluster size");
  FI_MEGAMOE_STATIC_ASSERT(kClusterSize == 1 or kNumSMs % 2 == 0,
                           "Number of SMs must be even for 2-CTA cluster");
  FI_MEGAMOE_STATIC_ASSERT(kClusterSize == 1 or kNumL1BlockNs % 2 == 0,
                           "L1 N block count must be even for 2-CTA cluster");
  FI_MEGAMOE_STATIC_ASSERT(kClusterSize == 1 or kNumL2BlockNs % 2 == 0,
                           "L2 N block count must be even for 2-CTA cluster");

  // Arrival counts
  const layout::Workspace& workspace;

  // Scheduler state
  BlockPhase next_phase = BlockPhase::Linear1;

  // Current expert and block indices
  uint32_t current_local_expert_idx = 0;
  uint32_t current_num_tokens = 0;
  uint32_t current_pool_block_offset = 0;
  uint32_t block_idx = 0;
  uint32_t m_block_idx = 0;
  uint32_t n_block_idx = 0;

  // Pre-cached per-expert token counts (filled during `for_each_block` init)
  // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
  uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

  CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace)
      : workspace(workspace) {
    block_idx = blockIdx.x;
  }

  CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
    return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
  }

  CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
    uint32_t valid_value;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ? stored_num_tokens_per_expert[i]
                                                                 : valid_value;
    }
    return ptx::exchange(valid_value, expert_idx % 32);
  }

  // Get pool block offset for a given expert index from a per-lane token count array
  CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
    uint32_t num_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (i * 32 + ptx::get_lane_idx() < expert_idx)
        num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], kPoolBlockM);
    }
    return __reduce_add_sync(0xffffffff, num_blocks);
  }

  CUTLASS_DEVICE void advance_expert_idx() {
    current_pool_block_offset += get_current_num_pool_blocks();
    current_local_expert_idx += 1;
    current_num_tokens = get_num_tokens(current_local_expert_idx);
  }

  CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
    current_local_expert_idx = expert_idx;
    current_num_tokens = get_num_tokens(expert_idx);
    current_pool_block_offset = get_pool_block_offset(expert_idx);
  }

  CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
    return current_pool_block_offset;
  }

  CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
    return math::ceil_div(current_num_tokens, BLOCK_M);
  }

  CUTLASS_DEVICE uint32_t get_current_num_pool_blocks() const {
    return math::ceil_div(current_num_tokens, kPoolBlockM);
  }

  CUTLASS_DEVICE uint32_t get_current_pool_token_offset() const {
    return current_pool_block_offset * kPoolBlockM;
  }

  CUTLASS_DEVICE uint32_t get_current_block_pool_token_idx() const {
    return get_current_pool_token_offset() + m_block_idx * BLOCK_M;
  }

  template <bool kDoUMMAAligned = false>
  CUTLASS_DEVICE uint32_t get_valid_m() const {
    const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
    return kDoUMMAAligned ? math::align(m, 16u) : m;
  }

  CUTLASS_DEVICE bool fetch_next_l1_block() {
    const auto wave_end_expert_idx = get_wave_expert_end_idx();
    while (current_local_expert_idx < wave_end_expert_idx) {
      const auto num_m_blocks = get_current_num_m_blocks();
      m_block_idx = block_idx / kNumL1BlockNs;
      if (m_block_idx < num_m_blocks) return true;

      // Current expert is fully assigned, move to the next
      block_idx -= num_m_blocks * kNumL1BlockNs;
      advance_expert_idx();
    }
    return false;
  }

  CUTLASS_DEVICE bool fetch_next_l2_block() {
    const auto wave_end_expert_idx = get_wave_expert_end_idx();
    while (current_local_expert_idx < wave_end_expert_idx) {
      const auto num_m_blocks = get_current_num_m_blocks();
      if (block_idx < num_m_blocks * kNumL2BlockNs) {
        m_block_idx = block_idx / kNumL2BlockNs;
        return true;
      }

      // Current expert is fully assigned, move to the next
      block_idx -= num_m_blocks * kNumL2BlockNs;
      advance_expert_idx();
    }
    return false;
  }

  // Core state machine: assigns the next block
  CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
    while (true) {
      if (current_local_expert_idx >= kNumExpertsPerRank) break;

      if (next_phase == BlockPhase::Linear1) {
        if (fetch_next_l1_block()) {
          // Found a new L1 block
          n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
          // Jump to next block
          block_idx += kNumSMs;
          return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
        } else {
          // L1 for the current wave is complete, transition to L2
          next_phase = BlockPhase::Linear2;
          set_expert_idx(
              math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
        }
      } else {
        if (fetch_next_l2_block()) {
          // Found a new L2 block
          n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
          // Jump to next block
          block_idx += kNumSMs;
          return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
        } else {
          // Move to L1 of the next wave
          next_phase = BlockPhase::Linear1;
        }
      }
    }

    // All waves and experts are fully processed
    return {BlockPhase::None, 0, 0, 0};
  }

  CUTLASS_DEVICE void fetch_expert_recv_count() {
// NOTES: each lane caches experts at indices (i * 32 + lane_idx)
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const auto expert_idx = i * 32 + ptx::get_lane_idx();
      uint64_t value = 0;
      if (expert_idx < kNumExpertsPerRank) {
        do {
          value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
        } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
      }
      stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();
  }

  template <typename Func>
  CUTLASS_DEVICE void for_each_block(Func&& func) {
    // Wait for all expert counters to be finalized
    fetch_expert_recv_count();

    // Initialize current expert with 0
    set_expert_idx(0);

    // Iterate over all blocks
    // TODO: add swizzle within expert waves for better L2 cache utilization
    while (true) {
      CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx,
                    n_block_idx);
      if (block_phase == BlockPhase::None) break;

      func(block_phase, current_local_expert_idx,
           block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs, m_block_idx,
           n_block_idx);
    }
  }

  template <typename Func>
  CUTLASS_DEVICE void for_each_linear1_block(Func&& func) {
    // Split-kernel mode: K1 owns only dispatch + Linear1. Unlike
    // for_each_block(), do not burn scheduler iterations on Linear2 blocks.
    fetch_expert_recv_count();
    set_expert_idx(0);
    while (current_local_expert_idx < kNumExpertsPerRank) {
      if (fetch_next_l1_block()) {
        n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
        block_idx += kNumSMs;
        func(current_local_expert_idx, kNumL1BlockKs, m_block_idx, n_block_idx);
      } else if (current_local_expert_idx >= kNumExpertsPerRank) {
        break;
      }
    }
  }

  template <typename Func>
  CUTLASS_DEVICE void for_each_linear2_block(Func&& func) {
    // Split-kernel mode: K2 starts after K1 globally completes, so all L2
    // ready masks are already final. Schedule Linear2 blocks directly.
    fetch_expert_recv_count();
    set_expert_idx(0);
    while (current_local_expert_idx < kNumExpertsPerRank) {
      if (fetch_next_l2_block()) {
        n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
        block_idx += kNumSMs;
        func(current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
      } else if (current_local_expert_idx >= kNumExpertsPerRank) {
        break;
      }
    }
  }
};

// Return the minimum all-L1 warm-up needed before the D40 scheduler starts
// interleaving L2 claims.  This prevents an L2 readiness wait from forming a
// cycle with L1 task publication.
CUTLASS_HOST_DEVICE constexpr int get_num_l1_warmup_waves(const int& num_total_m_blocks,
                                                          const int& num_workers,
                                                          const int& num_l1_n_blocks,
                                                          const int& num_l2_n_blocks) {
  if (num_total_m_blocks <= 0) return 0;
  const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_workers, num_l2_n_blocks);
  const int num_l1_warmup_waves_for_first_l2_wave =
      math::constexpr_ceil_div(num_first_l2_wave_m_blocks * num_l1_n_blocks, num_workers);

  const int num_interleave_task_diff_per_m_block =
      num_l1_n_blocks > num_l2_n_blocks ? num_l1_n_blocks - num_l2_n_blocks : 0;
  const int num_warmup_waves_for_interleave_schedule =
      math::constexpr_ceil_div(
          num_l1_n_blocks + (num_total_m_blocks - 1) * num_interleave_task_diff_per_m_block,
          num_workers) +
      1;

  return cute::max(num_l1_warmup_waves_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Dynamic task payload shared by the dev-m producer and all consumers in one
// CTA.  The layout is part of the validated two-stage mailbox protocol.
struct alignas(16) TaskInfo {
  BlockPhase block_phase;
  uint32_t local_expert_idx;
  uint32_t m_block_idx;
  uint32_t n_block_idx;
  uint32_t pool_block_idx;
  uint32_t valid_m;
  uint32_t shape_n;
  uint32_t shape_k;

  CUTLASS_HOST_DEVICE
  TaskInfo() : TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

  CUTLASS_HOST_DEVICE
  TaskInfo(const BlockPhase& block_phase, const uint32_t& local_expert_idx,
           const uint32_t& m_block_idx, const uint32_t& n_block_idx, const uint32_t& pool_block_idx,
           const uint32_t& valid_m, const uint32_t& shape_n, const uint32_t& shape_k)
      : block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_block_idx(n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m),
        shape_n(shape_n),
        shape_k(shape_k) {}

  CUTLASS_HOST_DEVICE bool is_valid() const { return block_phase != BlockPhase::None; }
};

FI_MEGAMOE_STATIC_ASSERT(sizeof(TaskInfo) == 32, "Invalid task payload layout");

// SM90 one-CTA dynamic scheduler from Aichen dev-m.  The weight-loader warp
// is the sole task producer and publishes work through a two-stage shared
// mailbox.  The static scheduler above remains intact for bigM split and the
// D40 KF static-RS arm.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t L1_SHAPE_N,
          uint32_t L1_SHAPE_K, uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank, uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N>
struct InterleavedMegaMoEScheduler {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  using task_info_t = TaskInfo;

  static constexpr uint32_t kNumScheduleStages = 2;
  static constexpr uint32_t kNumL1WavesDone = 0xffffffffu;

  FI_MEGAMOE_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid L1 shape");
  FI_MEGAMOE_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid L2 shape");
  FI_MEGAMOE_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 K shape");
  FI_MEGAMOE_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid L2 K shape");
  FI_MEGAMOE_STATIC_ASSERT(kNumL1BlockNs <= 64, "L1 readiness mask is too small");
  FI_MEGAMOE_STATIC_ASSERT(kNumL2BlockNs >= kNumL1BlockNs,
                           "Alternating scheduler requires at least as many L2 tasks as L1 tasks");

  const layout::Workspace& workspace;
  Barrier* task_info_full_barriers;
  Barrier* task_info_empty_barriers;
  task_info_t* task_infos;

  uint32_t sched_stage_idx = 0;
  uint32_t sched_phase = 0;
  uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
  uint32_t num_total_m_blocks = 0;
  uint32_t num_l1_warmup_waves = 0;

  CUTLASS_DEVICE
  InterleavedMegaMoEScheduler(const layout::Workspace& workspace, Barrier* task_info_full_barriers,
                              Barrier* task_info_empty_barriers, task_info_t* task_infos)
      : workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

  CUTLASS_DEVICE void advance_schedule_pipeline() {
    sched_stage_idx ^= 1u;
    sched_phase ^= sched_stage_idx == 0;
  }

  CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
    uint32_t valid_value = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ? stored_num_tokens_per_expert[i]
                                                                 : valid_value;
    }
    return ptx::exchange(valid_value, expert_idx % 32);
  }

  CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
    uint32_t num_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (i * 32 + ptx::get_lane_idx() < expert_idx)
        num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
    }
    return __reduce_add_sync(0xffffffff, num_blocks);
  }

  // Push dispatch (`push_done_ptr != nullptr`): wait for every rank's DONE
  // signal (lane 0 relaxed spin, then one acquire.sys load; bar.warp.sync
  // extends the ordering to the other lanes), then read the low words of the
  // recv-count sums of the given parity slot (the per-row remote tickets).
  CUTLASS_DEVICE void fetch_expert_recv_count(const int* push_done_ptr = nullptr,
                                              const int& push_done_target = 0,
                                              const uint32_t& parity = 0) {
    if (push_done_ptr != nullptr) {
      if (ptx::get_lane_idx() == 0) {
        auto* spin_dbg = workspace.get_spin_debug_ptr();
        FI_MEGAMOE_SPIN_WHILE_DEBUG(ptx::ld_volatile(push_done_ptr) - push_done_target < 0, 90010,
                                    spin_dbg, ptx::ld_volatile(push_done_ptr), push_done_target);
        FI_MEGAMOE_SPIN_WHILE_DEBUG(ptx::ld_acq_sys(push_done_ptr) - push_done_target < 0, 90020,
                                    spin_dbg, ptx::ld_acq_sys(push_done_ptr), push_done_target);
      }
      __syncwarp();
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const auto expert_idx = i * 32 + ptx::get_lane_idx();
      uint64_t value = 0;
      if (expert_idx < kNumExpertsPerRank) {
        if (push_done_ptr != nullptr) {
          value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx, parity));
        } else {
          do {
            value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
          } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
        }
      }
      stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();

    num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank);
    const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1BlockNs;
    const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs);
    const uint32_t min_l1_warmup_waves =
        get_num_l1_warmup_waves(num_total_m_blocks, kNumSMs, kNumL1BlockNs, kNumL2BlockNs);
    num_l1_warmup_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
  }

  CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase, const uint32_t& task_idx,
                                         const uint32_t& num_n_blocks, const uint32_t& shape_n,
                                         const uint32_t& shape_k) const {
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t pool_block_idx = task_idx / num_n_blocks;
    const uint32_t n_block_idx = task_idx % num_n_blocks;

    task_info_t result(block_phase, 0, 0, n_block_idx, pool_block_idx, 0, shape_n, shape_k);
    uint32_t block_offset = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * 32 + lane_idx;
      const uint32_t num_tokens = stored_num_tokens_per_expert[i];
      const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
      const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
      const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
      const bool is_owner = expert_idx < kNumExpertsPerRank &&
                            pool_block_idx >= lane_pool_block_offset &&
                            pool_block_idx < lane_pool_block_offset + num_m_blocks;
      const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

      if (owner_mask) {
        const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
        const uint32_t owner_m_block_idx = pool_block_idx - lane_pool_block_offset;
        const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
        result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
        result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
        result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
      }
      block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
    }
    return result;
  }

  static CUTLASS_DEVICE uint32_t get_next_task_idx(uint32_t* task_count_ptr) {
    uint32_t result = 0;
    if (cute::elect_one_sync()) result = ptx::atomic_add(task_count_ptr, 1u);
    return ptx::exchange(result, 0);
  }

  CUTLASS_DEVICE task_info_t claim_next_task() {
    while (true) {
      if (num_l1_warmup_waves != kNumL1WavesDone && num_l1_warmup_waves > 0) {
        --num_l1_warmup_waves;
        const uint32_t task_idx = get_next_task_idx(workspace.get_l1_task_count_ptr());
        if (task_idx >= num_total_m_blocks * kNumL1BlockNs) {
          num_l1_warmup_waves = kNumL1WavesDone;
          continue;
        }
        return create_task(BlockPhase::Linear1, task_idx, kNumL1BlockNs, L1_SHAPE_N, L1_SHAPE_K);
      }

      const uint32_t task_idx = get_next_task_idx(workspace.get_l2_task_count_ptr());
      if (task_idx >= num_total_m_blocks * kNumL2BlockNs) break;

      if (num_l1_warmup_waves != kNumL1WavesDone) num_l1_warmup_waves = 1;

      auto task_info =
          create_task(BlockPhase::Linear2, task_idx, kNumL2BlockNs, L2_SHAPE_N, L2_SHAPE_K);
      const uint32_t num_required_l1_tasks = (task_info.pool_block_idx + 1) * kNumL1BlockNs;
      while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < num_required_l1_tasks) {
      }
      return task_info;
    }
    return task_info_t();
  }

  CUTLASS_DEVICE void wait_task_slot_empty() const {
    task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1u);
  }

  CUTLASS_DEVICE void publish_task(const task_info_t& task_info) {
    if (cute::elect_one_sync()) {
      task_infos[sched_stage_idx] = task_info;
      __threadfence_block();
      task_info_full_barriers[sched_stage_idx].arrive();
    }
    __syncwarp();
    advance_schedule_pipeline();
  }

  CUTLASS_DEVICE bool get_published_task(task_info_t& task_info) {
    task_info_full_barriers[sched_stage_idx].wait(sched_phase);
    asm volatile("" ::: "memory");
    task_info = task_infos[sched_stage_idx];
    advance_schedule_pipeline();
    return task_info.is_valid();
  }

  CUTLASS_DEVICE void release_task_info(const uint32_t& lane_idx) const {
    if (lane_idx == 0) task_info_empty_barriers[sched_stage_idx ^ 1u].arrive();
  }
};

}  // namespace flashinfer::megamoe::sched
