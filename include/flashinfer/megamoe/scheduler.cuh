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

  template <bool AcquirePublished = false>
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
        if constexpr (AcquirePublished)
          value = ptx::ld_acq_sys(workspace.get_expert_recv_count_sum_ptr(expert_idx));
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

// Return the minimum all-L1 warm-up needed before the scheduler starts
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

// Dynamic task payload shared by the weight loader and all consumers in one
// CTA.  The layout is part of the validated two-stage mailbox protocol.
struct alignas(16) TaskInfo {
  BlockPhase block_phase;
  uint32_t local_expert_idx;
  uint32_t m_block_idx;
  uint32_t n_block_idx;
  uint32_t pool_block_idx;
  uint32_t valid_m;
  uint32_t shape_n;
  // The block holds rows pulled from other nodes: claimed before the other
  // blocks, and its FC2 completion is counted for the early inter-node return.
  uint32_t remote_dependent;

  CUTLASS_HOST_DEVICE
  TaskInfo() : TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

  CUTLASS_HOST_DEVICE
  TaskInfo(const BlockPhase& block_phase, const uint32_t& local_expert_idx,
           const uint32_t& m_block_idx, const uint32_t& n_block_idx, const uint32_t& pool_block_idx,
           const uint32_t& valid_m, const uint32_t& shape_n, const uint32_t& remote_dependent)
      : block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_block_idx(n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m),
        shape_n(shape_n),
        remote_dependent(remote_dependent) {}

  CUTLASS_HOST_DEVICE bool is_valid() const { return block_phase != BlockPhase::None; }
};

FI_MEGAMOE_STATIC_ASSERT(sizeof(TaskInfo) == 32, "Invalid task payload layout");

// The weight-loader warp is the sole task producer in each CTA and publishes
// work through a two-stage shared mailbox. Static plans use MegaMoEScheduler.
template <
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K, uint32_t kNumExpertsPerRank, uint32_t kNumSMs,
    uint32_t kNumRanks, uint32_t kNumLocalRanks = kNumRanks,
    uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
    uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N, uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N>
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
  // Whole local M blocks can run while the remote input is in flight. The
  // bootstrap counter is closed atomically as soon as remote counts are ready;
  // its low bits are the exact FC1 N-task prefix already claimed. Main claims
  // prioritize every remote-dependent block, then finish the local FC1 suffix.
  // Local FC2 stays last to overlap the remote return. Splitting only at full
  // M blocks preserves the total issued M rows for every expert.
  uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
  uint32_t stored_num_local_pulled_per_expert[kNumExpertsPerLane] = {};
  uint32_t stored_num_remote_tokens_per_expert[kNumExpertsPerLane] = {};
  uint32_t num_local_m_blocks = 0;
  uint32_t num_remote_m_blocks = 0;
  uint32_t num_remote_dependent_m_blocks = 0;
  uint32_t num_bootstrap_l1_tasks = 0;
  uint32_t num_total_m_blocks = 0;
  uint32_t num_l1_warmup_waves = 0;
  bool remote_fetched = false;
  static constexpr bool kSingleNode = kNumLocalRanks == kNumRanks;
  static constexpr uint32_t kBootstrapClosed = 0x80000000u;

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

  CUTLASS_DEVICE uint32_t expert_value(const uint32_t (&values)[kNumExpertsPerLane],
                                       const uint32_t& expert_idx) const {
    uint32_t valid_value = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i)
      valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ? values[i] : valid_value;
    return ptx::exchange(valid_value, expert_idx % 32);
  }

  // Rows from this node's ranks (every row with one node).
  CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
    return expert_value(stored_num_tokens_per_expert, expert_idx);
  }
  // Of them, the rows pulled into the expert's local segment.
  CUTLASS_DEVICE uint32_t get_num_local_pulled(const uint32_t& expert_idx) const {
    return expert_value(stored_num_local_pulled_per_expert, expert_idx);
  }
  // Rows from other nodes (valid after `fetch_remote_rows`).
  CUTLASS_DEVICE uint32_t get_num_remote_tokens(const uint32_t& expert_idx) const {
    return expert_value(stored_num_remote_tokens_per_expert, expert_idx);
  }

  // Rows of lane entry `i` in one segment, and the segment's blocks.
  CUTLASS_DEVICE uint32_t segment_rows(const uint32_t& i, const bool& remote) const {
    return remote ? stored_num_tokens_per_expert[i] - stored_num_local_pulled_per_expert[i] +
                        stored_num_remote_tokens_per_expert[i]
                  : stored_num_local_pulled_per_expert[i];
  }
  CUTLASS_DEVICE uint32_t sum_segment_blocks(const bool& remote) const {
    uint32_t num_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i)
      num_blocks += math::ceil_div(segment_rows(i, remote), BLOCK_M);
    return __reduce_add_sync(0xffffffff, num_blocks);
  }
  // First pool block of the expert's segment.
  CUTLASS_DEVICE uint32_t get_segment_block_offset(const uint32_t& expert_idx,
                                                   const bool& remote) const {
    uint32_t num_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (i * 32 + ptx::get_lane_idx() < expert_idx)
        num_blocks += math::ceil_div(segment_rows(i, remote), BLOCK_M);
    }
    return (remote ? num_local_m_blocks : 0u) + __reduce_add_sync(0xffffffff, num_blocks);
  }

  // Push dispatch (`push_done_ptr != nullptr`): wait for every rank's DONE
  // signal (lane 0 relaxed spin, then one acquire.sys load; bar.warp.sync
  // extends the ordering to the other lanes), then read the low words of the
  // recv-count sums of the given parity slot (the per-row remote tickets).
  template <bool AcquirePublished = false>
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
          } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumLocalRanks);
          if constexpr (AcquirePublished)
            value = ptx::ld_acq_sys(workspace.get_expert_recv_count_sum_ptr(expert_idx));
        }
      }
      const uint32_t num_tokens = static_cast<uint32_t>(value);
      const uint32_t leftover = num_tokens % BLOCK_M;
      stored_num_tokens_per_expert[i] = num_tokens;
      stored_num_local_pulled_per_expert[i] = kSingleNode ? num_tokens : num_tokens - leftover;
      stored_num_remote_tokens_per_expert[i] = 0;
    }
    __syncwarp();

    num_local_m_blocks = sum_segment_blocks(false);
    num_remote_m_blocks = 0;
    num_remote_dependent_m_blocks = 0;
    num_total_m_blocks = num_local_m_blocks;
    num_bootstrap_l1_tasks = 0;
    remote_fetched = kSingleNode;
    update_l1_warmup();
  }

  CUTLASS_DEVICE uint32_t num_main_l1_tasks() const {
    return num_total_m_blocks * kNumL1BlockNs - num_bootstrap_l1_tasks;
  }

  CUTLASS_DEVICE void update_l1_warmup() {
    const uint32_t waves = math::ceil_div(num_main_l1_tasks(), kNumSMs);
    num_l1_warmup_waves = cute::min(static_cast<uint32_t>(get_num_l1_warmup_waves(
                                        num_total_m_blocks, kNumSMs, kNumL1BlockNs, kNumL2BlockNs)),
                                    waves);
  }

  // Counts of the sources on other nodes, which their mirror on this node
  // publishes with a sys-scope release after its list copies: spin on the
  // remote sum of every local expert until all of those sources are in (the
  // low word is the expert's remote row count), then lay out the remote
  // segments. One acquire per expert also orders the per-source counts and the
  // lists that the pull reads afterwards.
  CUTLASS_DEVICE void fetch_remote_rows() {
    if (remote_fetched) return;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * 32 + ptx::get_lane_idx();
      uint64_t value = 0;
      if (expert_idx < kNumExpertsPerRank) {
        do {
          value = ptx::ld_acq_sys(workspace.get_expert_recv_count_remote_sum_ptr(expert_idx));
        } while (static_cast<uint32_t>(value >> 32) != kNumSMs * (kNumRanks - kNumLocalRanks));
      }
      stored_num_remote_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();
    // Close and reserve are atomic operations on the same word. A successful
    // reserve racing the close is included in H and MUST still be published.
    // A later reserve observes the closed bit and cannot duplicate a main task.
    uint32_t state = 0;
    if (ptx::get_lane_idx() == 0)
      state = atomicOr(workspace.get_bootstrap_l1_count_ptr(), kBootstrapClosed);
    num_bootstrap_l1_tasks = ptx::exchange(state, 0) & ~kBootstrapClosed;
    num_remote_m_blocks = sum_segment_blocks(true);
    uint32_t dependent_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i)
      dependent_blocks += math::ceil_div(stored_num_remote_tokens_per_expert[i], BLOCK_M);
    num_remote_dependent_m_blocks = __reduce_add_sync(0xffffffff, dependent_blocks);
    num_total_m_blocks = num_local_m_blocks + num_remote_m_blocks;
    update_l1_warmup();
    remote_fetched = true;
  }

  enum class TaskGroup : uint32_t { Local, Remote, Tail };

  // Main FC1/FC2 use the same three-group order. Only actual remote rows
  // constrain the return: remote-segment local tails follow every dependent
  // block rather than delaying other experts' remote work.
  CUTLASS_DEVICE task_info_t create_group_task(const BlockPhase& block_phase,
                                               const uint32_t& task_idx,
                                               const uint32_t& num_n_blocks,
                                               const uint32_t& shape_n,
                                               const TaskGroup& group) const {
    const uint32_t lane_idx = ptx::get_lane_idx();
    const bool remote_segment = group != TaskGroup::Local;
    const uint32_t group_block_idx = task_idx / num_n_blocks;
    const uint32_t n_block_idx = task_idx % num_n_blocks;
    task_info_t result(block_phase, 0, 0, n_block_idx, 0, 0, shape_n,
                       group == TaskGroup::Remote ? 1u : 0u);
    uint32_t group_offset = 0, pool_offset = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * 32 + lane_idx;
      const uint32_t rows = segment_rows(i, remote_segment);
      const uint32_t segment_blocks = math::ceil_div(rows, BLOCK_M);
      const uint32_t dependent_blocks =
          math::ceil_div(stored_num_remote_tokens_per_expert[i], BLOCK_M);
      const uint32_t group_blocks = group == TaskGroup::Remote ? dependent_blocks
                                    : group == TaskGroup::Tail ? segment_blocks - dependent_blocks
                                                               : segment_blocks;
      const uint32_t group_inclusive = math::warp_inclusive_sum(group_blocks, lane_idx);
      const uint32_t lane_group_offset = group_offset + group_inclusive - group_blocks;
      const uint32_t pool_inclusive = math::warp_inclusive_sum(segment_blocks, lane_idx);
      const uint32_t lane_pool_offset = pool_offset + pool_inclusive - segment_blocks;
      const bool is_owner = expert_idx < kNumExpertsPerRank and
                            group_block_idx >= lane_group_offset and
                            group_block_idx < lane_group_offset + group_blocks;
      const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);
      if (owner_mask) {
        const uint32_t owner_lane = static_cast<uint32_t>(__ffs(owner_mask) - 1);
        const uint32_t m_block = group_block_idx - lane_group_offset +
                                 (group == TaskGroup::Tail ? dependent_blocks : 0u);
        result.local_expert_idx = ptx::exchange(expert_idx, owner_lane);
        result.m_block_idx = ptx::exchange(m_block, owner_lane);
        result.valid_m = ptx::exchange(cute::min(rows - m_block * BLOCK_M, BLOCK_M), owner_lane);
        result.pool_block_idx = (remote_segment ? num_local_m_blocks : 0u) +
                                ptx::exchange(lane_pool_offset + m_block, owner_lane);
      }
      group_offset += ptx::exchange(group_inclusive, 31);
      pool_offset += ptx::exchange(pool_inclusive, 31);
    }
    return result;
  }

  CUTLASS_DEVICE task_info_t create_main_task(const BlockPhase& block_phase,
                                              const uint32_t& task_idx,
                                              const uint32_t& num_n_blocks,
                                              const uint32_t& shape_n) const {
    const uint32_t dependent_tasks = num_remote_dependent_m_blocks * num_n_blocks;
    const uint32_t remote_segment_tasks = num_remote_m_blocks * num_n_blocks;
    if (task_idx < dependent_tasks)
      return create_group_task(block_phase, task_idx, num_n_blocks, shape_n, TaskGroup::Remote);
    if (task_idx < remote_segment_tasks)
      return create_group_task(block_phase, task_idx - dependent_tasks, num_n_blocks, shape_n,
                               TaskGroup::Tail);
    const uint32_t skip = block_phase == BlockPhase::Linear1 ? num_bootstrap_l1_tasks : 0u;
    return create_group_task(block_phase, skip + task_idx - remote_segment_tasks, num_n_blocks,
                             shape_n, TaskGroup::Local);
  }

  static CUTLASS_DEVICE uint32_t get_next_task_idx(uint32_t* task_count_ptr) {
    uint32_t result = 0;
    if (cute::elect_one_sync()) result = ptx::atomic_add(task_count_ptr, 1u);
    return ptx::exchange(result, 0);
  }

  CUTLASS_DEVICE task_info_t claim_next_task() {
    // Do not consume the main FC1 counter until every producer knows the same
    // stable bootstrap prefix. Dispatch closes it immediately after acquiring
    // the remote layout; a producer with no local work left may also close it.
    while (not remote_fetched) {
      uint32_t state = 0;
      if (ptx::get_lane_idx() == 0)
        state = ptx::ld_volatile(workspace.get_bootstrap_l1_count_ptr());
      state = ptx::exchange(state, 0);
      if ((state & kBootstrapClosed) or state >= num_local_m_blocks * kNumL1BlockNs) {
        fetch_remote_rows();
        break;
      }
      uint32_t observed = 0;
      if (ptx::get_lane_idx() == 0)
        observed = atomicCAS(workspace.get_bootstrap_l1_count_ptr(), state, state + 1u);
      if (ptx::exchange(observed, 0) == state)
        return create_group_task(BlockPhase::Linear1, state, kNumL1BlockNs, L1_SHAPE_N,
                                 TaskGroup::Local);
    }

    while (true) {
      if (num_l1_warmup_waves != kNumL1WavesDone && num_l1_warmup_waves > 0) {
        --num_l1_warmup_waves;
        const uint32_t task_idx = get_next_task_idx(workspace.get_l1_task_count_ptr());
        if (task_idx >= num_main_l1_tasks()) {
          num_l1_warmup_waves = kNumL1WavesDone;
          continue;
        }
        return create_main_task(BlockPhase::Linear1, task_idx, kNumL1BlockNs, L1_SHAPE_N);
      }

      const uint32_t task_idx = get_next_task_idx(workspace.get_l2_task_count_ptr());
      if (task_idx >= num_total_m_blocks * kNumL2BlockNs) break;
      if (num_l1_warmup_waves != kNumL1WavesDone) num_l1_warmup_waves = 1;

      // Dependent and remote-tail groups share the first main FC1 ordinal
      // range. Whole-local FC2 uses the stable bootstrap-adjusted suffix.
      const uint32_t remote_tasks = num_remote_m_blocks * kNumL2BlockNs;
      const uint32_t block_l1_end =
          ((task_idx < remote_tasks ? task_idx : task_idx - remote_tasks) / kNumL2BlockNs + 1u) *
          kNumL1BlockNs;
      const uint32_t required_l1 =
          task_idx < remote_tasks
              ? block_l1_end
              : num_remote_m_blocks * kNumL1BlockNs + (block_l1_end > num_bootstrap_l1_tasks
                                                           ? block_l1_end - num_bootstrap_l1_tasks
                                                           : 0u);
      // Claim progress prevents a scheduler dependency cycle. The per-pool
      // acquire readiness mask separately waits for actual FC1 completion,
      // including bootstrap tasks whose publication raced the close.
      while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < required_l1) {
      }
      return create_main_task(BlockPhase::Linear2, task_idx, kNumL2BlockNs, L2_SHAPE_N);
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
