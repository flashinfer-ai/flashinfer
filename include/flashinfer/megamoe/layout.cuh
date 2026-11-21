#pragma once

#include <cute/numeric/math.hpp>
#include <flashinfer/megamoe/detail/common.cuh>

// Bounded spin (trap after ~10 s at 2 GHz) used by the counter-based
// synchronisation paths (push dispatch DONE flags).
//
// Trap only: a device `printf` (vprintf ABI call) anywhere in the persistent
// kernel costs ~15 % on the math warps (measured: the ABI call changes the
// register allocation of the whole kernel), so no message is printed.
//
// `FI_MEGAMOE_SPIN_WHILE_DEBUG` can publish the spin's call-site tag and the
// observed/expected counter words to `Workspace::get_spin_debug_ptr()` right
// before trapping, so that a bare `cudaErrorLaunchFailure` can be attributed to
// a specific spin after the fact. That store is compiled in ONLY when
// `FI_MEGAMOE_SPIN_DEBUG_TAGS` is defined (host env `FI_MEGAMOE_NVFP4_SPIN_DEBUG=1`, which also
// renames the JIT kernel): on this kernel even a never-executed branch costs
// instruction-cache footprint and ptxas scheduling -- measured +160 SASS
// instructions per kernel when it was compiled in unconditionally -- so the
// default build must generate byte-identical code to the no-debug variant.
//
// `dbg` may be nullptr (the pull path never sets it); the store is predicated.
#ifndef FI_MEGAMOE_SPIN_WHILE_DEBUG
#if defined(FI_MEGAMOE_SPIN_DEBUG_TAGS)
#define FI_MEGAMOE_SPIN_WHILE_DEBUG(cond, tag, dbg, observed, expected)                     \
  for (long long __spin_t0 = clock64(); (cond);) {                                          \
    if (clock64() - __spin_t0 > 20000000000ll) {                                            \
      uint32_t* __spin_dbg = (dbg);                                                         \
      if (__spin_dbg != nullptr) {                                                          \
        __spin_dbg[0] = static_cast<uint32_t>(tag);                                         \
        __spin_dbg[1] = static_cast<uint32_t>(observed);                                    \
        __spin_dbg[2] = static_cast<uint32_t>(expected);                                    \
        __spin_dbg[3] =                                                                     \
            (static_cast<uint32_t>(blockIdx.x) << 16) | static_cast<uint32_t>(threadIdx.x); \
        __threadfence_system();                                                             \
      }                                                                                     \
      asm volatile("trap;");                                                                \
    }                                                                                       \
  }
#else
#define FI_MEGAMOE_SPIN_WHILE_DEBUG(cond, tag, dbg, observed, expected) \
  for (long long __spin_t0 = clock64(); (cond);) {                      \
    if (clock64() - __spin_t0 > 20000000000ll) asm volatile("trap;");   \
  }
#endif
#endif

#ifndef FI_MEGAMOE_SPIN_WHILE
#define FI_MEGAMOE_SPIN_WHILE(cond, tag) FI_MEGAMOE_SPIN_WHILE_DEBUG(cond, tag, nullptr, 0u, 0u)
#endif

namespace flashinfer::megamoe::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;
// Two shared mailbox stages, each with full/empty barriers and one TaskInfo.
static constexpr int kSM90InterleavedSchedulerSMEMBytes = 96;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M
// alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank,
                                                        T num_topk, T num_experts_per_rank) {
  const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
  const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
  return math::constexpr_align(num_max_recv_tokens * num_max_experts_per_token +
                                   num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
                               static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
  return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
  uint32_t rank_idx;
  uint32_t token_idx;
  uint32_t topk_idx;
};

struct Workspace {
  void* base;
  uint32_t num_ranks, num_experts;
  uint32_t num_experts_per_rank;
  uint32_t num_max_tokens_per_rank;
  uint32_t num_max_recv_tokens_per_expert;

  // Pool capacity: all local experts share a contiguous token pool
  uint32_t num_max_pool_tokens;
  uint32_t num_max_pool_blocks;

  // Grid/NVLink state plus the two dynamic task counters. Expert
  // counters begin at the next 128-byte cache line so the hot task counters
  // do not share an L2 sector with communication state.
  static constexpr uint64_t kNumBarrierSignalBytes = 128;

  CUTLASS_HOST_DEVICE
  Workspace(void* base, const uint32_t& num_ranks, const uint32_t& num_experts,
            const uint32_t& num_max_tokens_per_rank, const uint32_t& num_topk)
      : base(base),
        num_ranks(num_ranks),
        num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
    num_experts_per_rank = num_experts / num_ranks;
    num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
    num_max_pool_tokens =
        get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
  }

  CUTLASS_HOST_DEVICE
  uint64_t get_num_bytes() const {
    uint64_t num_bytes = 0;

    // Barrier and in-kernel scheduling counters
    num_bytes += kNumBarrierSignalBytes;

    // Expert send/recv count
    num_bytes += num_experts * sizeof(uint64_t) * 2;

    // Expert recv count sum (three sets: launch parity for the rotated push
    // pool, then the sources on other nodes), each set padded to whole 128 B
    // lines (remote sys-scope tickets land here)
    num_bytes += math::align<uint64_t>(num_experts_per_rank * sizeof(uint64_t), 128) * 3;

    // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
    num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

    // L2 block arrival mask
    num_bytes += num_max_pool_blocks * sizeof(uint64_t);

    // Dispatch pulling source token-topk
    num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

    // Combine push source indices
    num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

    // Align to TMA descriptor requirements
    num_bytes = math::align<uint64_t>(num_bytes, 16);
    return num_bytes;
  }

  CUTLASS_HOST_DEVICE
  void* get_end_ptr() const { return math::advance_ptr(base, get_num_bytes()); }

  // Grid sync counters: `kNumBarrierSignalBytes` layout
  // [ 0..15]: 4 x `uint32_t` grid sync counters
  // [16..20]: `uint32_t` NVLink barrier counter
  // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
  static constexpr uint32_t kNumMaxGridSyncCounters = 4;

  template <uint32_t kIndex = 0>
  CUTLASS_DEVICE uint32_t* get_grid_sync_count_ptr() const {
    FI_MEGAMOE_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
    return static_cast<uint32_t*>(base) + kIndex;
  }

  CUTLASS_DEVICE
  uint32_t* get_nvl_barrier_counter_ptr() const {
    return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
  }

  CUTLASS_DEVICE
  int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
    // NOTES: the signal is signed, as we may minus
    return math::advance_ptr<int>(
        base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
  }

  CUTLASS_DEVICE
  uint32_t* get_l1_task_count_ptr() const { return math::advance_ptr<uint32_t>(base, 28u); }

  CUTLASS_DEVICE
  uint32_t* get_l2_task_count_ptr() const { return math::advance_ptr<uint32_t>(base, 32u); }

  // Inter-node early return of the MXFP4 fused kernel:
  // [36..39] FC2 tasks of the blocks that hold rows pulled from other nodes
  //          completed on this rank (gpu-scope release per task; reset in the cleanup)
  CUTLASS_DEVICE
  uint32_t* get_remote_l2_done_ptr() const { return math::advance_ptr<uint32_t>(base, 36u); }

  // [40..43] adaptive local FC1 bootstrap cursor; high bit closes the queue,
  // low bits are the stable claimed N-task prefix. Reset after math completes.
  CUTLASS_DEVICE
  uint32_t* get_bootstrap_l1_count_ptr() const { return math::advance_ptr<uint32_t>(base, 40u); }

  // Counter-based words (second 64 B sector of the signal line, away from the
  // task counters updated by GPU-scope atomics throughout the kernel):
  // [64..67] int push DONE count (monotonic; remote red.release.sys, +1 per rank per launch)
  // [68..71] push epoch (launches completed; SM0 bumps it in the cleanup; DONE target =
  //          ranks * (epoch + 1); also the pool parity slot)
  // [72..75] push CTA arrival count (last CTA signals DONE and resets it)
  // [76..91] spin-timeout debug word (u32[4] = {tag, observed, expected,
  //          blockIdx|threadIdx}; written once right before a spin traps)
  CUTLASS_DEVICE
  int* get_push_done_count_ptr() const { return math::advance_ptr<int>(base, 64u); }

  CUTLASS_DEVICE
  uint32_t* get_push_epoch_ptr() const { return math::advance_ptr<uint32_t>(base, 68u); }

  CUTLASS_DEVICE
  uint32_t* get_push_cta_arrival_ptr() const { return math::advance_ptr<uint32_t>(base, 72u); }

  // Attribution for a `FI_MEGAMOE_SPIN_WHILE_DEBUG` timeout: a kernel that traps out of a
  // counter spin leaves its call-site tag and the observed/expected counter values
  // here, so the host can tell a protocol stall from any other launch failure.
  CUTLASS_DEVICE
  uint32_t* get_spin_debug_ptr() const { return math::advance_ptr<uint32_t>(base, 76u); }

  // [92..95] ranks of this node whose remote-dependent FC2 tasks are all done (one
  //          sys-scope release per rank per launch; reset in the cleanup)
  CUTLASS_DEVICE
  int* get_remote_rows_ready_ptr() const { return math::advance_ptr<int>(base, 92u); }

  CUTLASS_DEVICE
  uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
    return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
  }

  CUTLASS_DEVICE
  uint64_t* get_expert_recv_count_ptr(const uint32_t& rank_idx = 0,
                                      const uint32_t& expert_idx = 0) const {
    return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
  }

  CUTLASS_DEVICE
  uint32_t get_recv_count_sum_set_stride() const {
    // uint64 words per parity set (whole 128 B lines)
    return static_cast<uint32_t>(
        math::align<uint64_t>(num_experts_per_rank * sizeof(uint64_t), 128) / sizeof(uint64_t));
  }

  CUTLASS_DEVICE
  uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
    return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
  }

  // Second recv-count-sum set (push dispatch with the rotated pool: launch parity 1)
  CUTLASS_DEVICE
  uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx,
                                          const uint32_t& parity) const {
    return get_expert_send_count_ptr(num_experts * 2) +
           (parity & 1u) * get_recv_count_sum_set_stride() + expert_idx;
  }

  // Third set: the counts of the sources on other nodes, published by their
  // mirror on this node (the two-wave schedule lays the local wave out from
  // the first set and the remote wave from this one).
  CUTLASS_DEVICE
  uint64_t* get_expert_recv_count_remote_sum_ptr(const uint32_t& expert_idx = 0) const {
    return get_expert_recv_count_sum_ptr(get_recv_count_sum_set_stride() * 2) + expert_idx;
  }

  CUTLASS_DEVICE
  uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
    const auto base = get_expert_recv_count_sum_ptr(get_recv_count_sum_set_stride() * 3);
    return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
  }

  CUTLASS_DEVICE
  uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
    // Pad L1 entry count to even so that the `l2_arrival_mask` is 8-byte aligned
    const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
    return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
  }

  // For dispatch pulling
  CUTLASS_DEVICE
  uint32_t* get_src_token_topk_idx_ptr(const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0,
                                       const uint32_t& token_idx = 0) const {
    const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
    return reinterpret_cast<uint32_t*>(base) +
           expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
           rank_idx * num_max_recv_tokens_per_expert + token_idx;
  }

  // For combine usages
  CUTLASS_DEVICE
  TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
    const auto base =
        reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
    return base + pool_token_idx;
  }
};

struct Data {
  uint32_t num_bytes;
  bool require_tma_alignment;
  void* base;

  CUTLASS_HOST_DEVICE
  constexpr explicit Data(const uint32_t& num_bytes, const bool& require_tma_alignment = true,
                          void* base = nullptr)
      : num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
    FI_MEGAMOE_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
  }

  template <typename dtype_t = uint32_t>
  CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
    return static_cast<dtype_t>(num_bytes);
  }

  template <typename dtype_t = void>
  CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
    return static_cast<dtype_t*>(base);
  }

  CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) { base = ptr; }
};

struct Buffer {
  Data data_layout;
  uint32_t num_ranks;
  uint32_t num_max_tokens_per_rank;

  void* base;

  CUTLASS_HOST_DEVICE
  Buffer(const Data& data_layout, const uint32_t& num_ranks,
         const uint32_t& max_num_tokens_per_rank, void* base = nullptr)
      : data_layout(data_layout),
        num_ranks(num_ranks),
        num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

  CUTLASS_HOST_DEVICE
  uint64_t get_num_bytes_per_rank() const {
    return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
  }

  CUTLASS_HOST_DEVICE
  uint64_t get_num_bytes() const { return get_num_bytes_per_rank() * num_ranks; }

  template <typename dtype_t = void>
  CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
    return static_cast<dtype_t*>(base);
  }

  CUTLASS_HOST_DEVICE
  void* get_end_ptr() const { return math::advance_ptr(base, get_num_bytes()); }

  CUTLASS_HOST_DEVICE
  Buffer get_rank_buffer(const uint32_t& rank_idx) const {
    return {data_layout, 1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)};
  }

  CUTLASS_HOST_DEVICE
  Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
    FI_MEGAMOE_DEVICE_ASSERT(num_ranks == 1 or global);
    return Data(data_layout.num_bytes, data_layout.require_tma_alignment,
                math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx));
  }
};

// Regions a rank adds for every node it cannot reach over NVLink. Rank S and
// its mirror on the other node exchange tokens through these: S packs the rows
// that have an expert on that node into `outbound`, puts them into the mirror's
// `inbox`, the mirror's owners pull from the inbox as from a local source and
// store their FC2 rows into the mirror's `outbox` (the `combine` region of S as
// the mirror's node sees it); the mirror reduces the outbox into
// `return_outbound` and puts it into S's `return_inbox`. The inbox repeats the
// byte layout of the four input buffers, so a puller addresses it with one
// offset per source rank; the outbox repeats the combine buffer's layout.
// With no remote node every size is zero and the single-node ABI is unchanged.
struct RemoteRegions {
  uint32_t num_remote_nodes, num_experts_per_node, capacity, topk;
  uint64_t input_block_bytes, combine_block_bytes, row_bf16_bytes;
  void* base;

  // Header of an inbox / outbound block: the number of packed tokens, then per
  // expert of the receiving node its count, then per expert a fixed-stride list
  // of `capacity` entries `slot * topk + k` (the index pass writes it in place,
  // so nothing is compacted before the put).
  static constexpr uint32_t kHeaderControlBytes = 128;

  CUTLASS_HOST_DEVICE
  RemoteRegions(void* base, uint32_t num_remote_nodes, uint32_t num_experts_per_node,
                uint32_t capacity, uint32_t topk, uint32_t hidden, uint64_t input_block_bytes,
                uint64_t combine_block_bytes)
      : num_remote_nodes(num_remote_nodes),
        num_experts_per_node(num_experts_per_node),
        capacity(capacity),
        topk(topk),
        input_block_bytes(input_block_bytes),
        combine_block_bytes(combine_block_bytes),
        row_bf16_bytes(static_cast<uint64_t>(hidden) * sizeof(uint16_t)),
        base(base) {}

  CUTLASS_HOST_DEVICE uint64_t header_bytes() const {
    return math::align<uint64_t>(
        kHeaderControlBytes + num_experts_per_node * sizeof(uint32_t) +
            static_cast<uint64_t>(num_experts_per_node) * capacity * sizeof(uint32_t),
        128);
  }
  CUTLASS_HOST_DEVICE uint64_t inbox_bytes() const { return header_bytes() + input_block_bytes; }
  CUTLASS_HOST_DEVICE uint64_t return_bytes() const {
    return math::align<uint64_t>(capacity * row_bf16_bytes, 128);
  }
  CUTLASS_HOST_DEVICE uint64_t slot_map_bytes() const {
    return math::align<uint64_t>(capacity * sizeof(int32_t) + 128, 128);
  }
  // Per remote node: inbox, outbound, outbox, return inbox, return outbound,
  // slot map with control words.
  CUTLASS_HOST_DEVICE uint64_t bytes_per_node() const {
    return inbox_bytes() * 2 + math::align<uint64_t>(combine_block_bytes, 128) +
           return_bytes() * 2 + slot_map_bytes();
  }
  CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const {
    return math::align<uint64_t>(bytes_per_node() * num_remote_nodes, 16);
  }
  CUTLASS_HOST_DEVICE void* get_end_ptr() const { return math::advance_ptr(base, get_num_bytes()); }

  CUTLASS_HOST_DEVICE void* node_base(const uint32_t& node) const {
    return math::advance_ptr(base, bytes_per_node() * node);
  }
  // Inbox: the input block the mirror filled, then the header, so that the
  // scales, routing and header leave in one put after the rows.
  CUTLASS_HOST_DEVICE void* inbox_input_block(const uint32_t& node) const {
    return node_base(node);
  }
  CUTLASS_HOST_DEVICE void* inbox_header(const uint32_t& node) const {
    return math::advance_ptr(node_base(node), input_block_bytes);
  }
  // Outbound: what this rank sends to that node, same shape as the inbox.
  CUTLASS_HOST_DEVICE void* outbound_input_block(const uint32_t& node) const {
    return math::advance_ptr(node_base(node), inbox_bytes());
  }
  CUTLASS_HOST_DEVICE void* outbound_header(const uint32_t& node) const {
    return math::advance_ptr(outbound_input_block(node), input_block_bytes);
  }
  // Outbox: FC2 rows of the mirror's packed tokens, combine-buffer layout.
  CUTLASS_HOST_DEVICE void* outbox(const uint32_t& node) const {
    return math::advance_ptr(node_base(node), inbox_bytes() * 2);
  }
  CUTLASS_HOST_DEVICE void* return_inbox(const uint32_t& node) const {
    return math::advance_ptr(outbox(node), math::align<uint64_t>(combine_block_bytes, 128));
  }
  CUTLASS_HOST_DEVICE void* return_outbound(const uint32_t& node) const {
    return math::advance_ptr(return_inbox(node), return_bytes());
  }
  // `packed_slot_of_token[capacity]` (-1 = no expert on that node), then the
  // packed-token counter in the trailing 128 bytes.
  CUTLASS_HOST_DEVICE int32_t* packed_slot_map(const uint32_t& node) const {
    return static_cast<int32_t*>(math::advance_ptr(return_outbound(node), return_bytes()));
  }
  // Control words in the aligned trailing 128 bytes after the slot map: byte
  // [0] packed tokens, [8] inbox flag, [16] return flag (both carriers use these
  // monotonic window-local flags), [24] launch epoch (node 0's block only).
  CUTLASS_HOST_DEVICE uint32_t* packed_token_count(const uint32_t& node) const {
    return static_cast<uint32_t*>(math::advance_ptr(
        packed_slot_map(node), math::align<uint64_t>(capacity * sizeof(int32_t), 128)));
  }
  CUTLASS_HOST_DEVICE uint64_t* inbox_flag(const uint32_t& node) const {
    return reinterpret_cast<uint64_t*>(packed_token_count(node) + 2);
  }
  CUTLASS_HOST_DEVICE uint64_t* return_flag(const uint32_t& node) const {
    return reinterpret_cast<uint64_t*>(packed_token_count(node) + 4);
  }
  CUTLASS_HOST_DEVICE uint32_t* launch_epoch() const { return packed_token_count(0) + 6; }
  // Header: control word [0] packed tokens, then per expert of the receiving
  // node its count, then per expert its list of `capacity` entries.
  CUTLASS_HOST_DEVICE uint32_t* header_control(void* header) const {
    return static_cast<uint32_t*>(header);
  }
  CUTLASS_HOST_DEVICE uint32_t* header_expert_counts(void* header) const {
    return static_cast<uint32_t*>(math::advance_ptr(header, kHeaderControlBytes));
  }
  CUTLASS_HOST_DEVICE uint32_t* header_list(void* header, const uint32_t& expert) const {
    return header_expert_counts(header) + num_experts_per_node +
           static_cast<uint64_t>(expert) * capacity;
  }
  // The list of `expert` of `node` that this rank's index pass fills, in the
  // outbound header.
  CUTLASS_HOST_DEVICE uint32_t* outbound_list(const uint32_t& node, const uint32_t& expert) const {
    return header_list(outbound_header(node), expert);
  }
};

// One allocation ABI shared by NVFP4 fused/split and MXFP4 fused. Activation
// scales reserve per-64 capacity for L2; BN256 kernels use its dense per-128
// prefix. The combine region is indexed by the original token and top-k slot.
struct MegaMoEBuffers {
  Workspace workspace;
  Buffer input_token_buffer, input_sf_buffer, input_topk_idx_buffer, input_topk_weights_buffer;
  Buffer l1_token_buffer, l1_sf_buffer, l1_topk_weights_buffer;
  Buffer l2_token_buffer, l2_sf_buffer, combine_token_buffer;
  // Inter-node inbox / outbox regions; empty with one node.
  RemoteRegions remote;

  CUTLASS_HOST_DEVICE
  MegaMoEBuffers(void* base, uint32_t ranks, uint32_t experts, uint32_t capacity, uint32_t topk,
                 uint32_t hidden, uint32_t intermediate, uint32_t pool_tokens,
                 uint32_t sf_pool_tokens, uint32_t num_remote_nodes = 0)
      : workspace(base, ranks, experts, capacity, topk),
        input_token_buffer(Data(hidden), 1, capacity, workspace.get_end_ptr()),
        input_sf_buffer(Data(hidden / 32, false), 1, capacity, input_token_buffer.get_end_ptr()),
        input_topk_idx_buffer(Data(topk * sizeof(int64_t), false), 1, capacity,
                              input_sf_buffer.get_end_ptr()),
        input_topk_weights_buffer(Data(topk * sizeof(float), false), 1, capacity,
                                  input_topk_idx_buffer.get_end_ptr()),
        l1_token_buffer(Data(hidden), 1, pool_tokens, input_topk_weights_buffer.get_end_ptr()),
        l1_sf_buffer(Data(hidden / 32, false), 1, sf_pool_tokens, l1_token_buffer.get_end_ptr()),
        l1_topk_weights_buffer(Data(sizeof(float), false), 1, pool_tokens,
                               l1_sf_buffer.get_end_ptr()),
        l2_token_buffer(Data(intermediate), 1, pool_tokens, l1_topk_weights_buffer.get_end_ptr()),
        l2_sf_buffer(Data(intermediate / 16, false), 1, sf_pool_tokens,
                     l2_token_buffer.get_end_ptr()),
        combine_token_buffer(Data(hidden * sizeof(uint16_t)), topk, capacity,
                             l2_sf_buffer.get_end_ptr()),
        remote(combine_token_buffer.get_end_ptr(), num_remote_nodes,
               ranks / (num_remote_nodes + 1) * (experts / ranks), capacity, topk, hidden,
               reinterpret_cast<uint64_t>(input_topk_weights_buffer.get_end_ptr()) -
                   reinterpret_cast<uint64_t>(input_token_buffer.get_base_ptr()),
               combine_token_buffer.get_num_bytes()) {}
};

}  // namespace flashinfer::megamoe::layout
