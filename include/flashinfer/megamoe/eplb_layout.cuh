#pragma once

// Host/device shared layout of the replica plan of the SM90 MXFP4 MegaMoE
// kernel: the symmetric workspace that holds the plan tables and the
// description of the symmetric weight arena the in-kernel weight copy reads.
// The kernels live in `mega_moe_eplb.cuh`.

#include <flashinfer/megamoe/detail/common.cuh>
#include <flashinfer/megamoe/sym_buffer.cuh>

namespace flashinfer::megamoe::eplb {

// Symmetric workspace of the replica plan. All offsets are in 32-bit words so
// the host and the kernels slice the same buffer the same way.
//
// `R` ranks, `E` logical experts, `H = E / R` resident experts per rank and `S`
// redundant expert slots per rank. Rank `r` holds logical experts
// `[r * H, (r + 1) * H)` in its local slots `[0, H)` (its resident experts) and
// replicas of other ranks' experts in local slots `[H, H + S)`. The physical
// slot id of local slot `j` on rank `r` is `r * (H + S) + j`; the MegaMoE kernel
// runs on `R * (H + S)` physical experts and never sees logical ids.
struct Workspace {
  // The in-kernel weight copy tracks the redundant slots in 32-bit masks.
  static constexpr uint32_t kMaxRedundantSlots = 32;
  // The in-kernel planner keeps one lane per rank.
  static constexpr uint32_t kMaxRanks = 32;
  // CTA size of the planner, remap and weight-copy launches.
  static constexpr uint32_t kNumPlanThreads = 256;

  uint32_t num_ranks, num_experts, num_redundant_slots;

  CUTLASS_HOST_DEVICE constexpr Workspace(const uint32_t& num_ranks, const uint32_t& num_experts,
                                          const uint32_t& num_redundant_slots)
      : num_ranks(num_ranks), num_experts(num_experts), num_redundant_slots(num_redundant_slots) {}

  // Logical expert held by redundant slot `s` of rank `r`, -1 when empty. Must
  // be identical on every rank: [R][S] int32
  CUTLASS_HOST_DEVICE constexpr uint32_t slot_expert_offset() const { return 0; }
  // Copies of expert `e` as int2 {rank, cumulative blocks} in assignment order,
  // resident rank first, terminated by rank -1 unless all `R` entries are used.
  // The remap sends token block `b` to the first copy whose bound exceeds `b`;
  // past the last bound it stays on the last copy: [E][R] int2
  CUTLASS_HOST_DEVICE constexpr uint32_t copies_offset() const {
    return math::constexpr_align(slot_expert_offset() + num_ranks * num_redundant_slots, 2u);
  }
  // Ordinal base of this rank's tokens per expert: the tokens the ranks below
  // this one route to the expert, so block splits agree across ranks: [E] int32
  CUTLASS_HOST_DEVICE constexpr uint32_t expert_base_offset() const {
    return copies_offset() + 2 * num_experts * num_ranks;
  }
  // Running ordinal counters of the remap; zero between launches: [E] uint32
  CUTLASS_HOST_DEVICE constexpr uint32_t local_counter_offset() const {
    return expert_base_offset() + num_experts;
  }
  // This rank's redundant slots whose expert changed, as int2 {slot, expert},
  // and their count. Consumed by the weight-copy kernel or by the in-kernel
  // weight copy, which clears the count once the list is applied: [S] int2
  CUTLASS_HOST_DEVICE constexpr uint32_t copy_list_offset() const {
    return math::constexpr_align(local_counter_offset() + num_experts, 2u);
  }
  CUTLASS_HOST_DEVICE constexpr uint32_t copy_count_offset() const {
    return copy_list_offset() + 2 * num_redundant_slots;
  }
  // Readiness counters of the in-kernel weight copy, one pair per redundant
  // slot (group 0: the L1 planes, group 1: the L2 planes). Zero for every listed
  // slot before the launch; each CTA of the MegaMoE kernel adds one per group
  // once its share of the slot is written, so a slot is ready when the counter
  // equals the grid size: [S][2] uint32
  CUTLASS_HOST_DEVICE constexpr uint32_t copy_done_offset() const {
    return math::constexpr_align(copy_count_offset() + 1, 2u);
  }
  // Completion ticket of the remap: its last CTA zeroes the ordinal counters
  // and this word, so no fill launch is needed between two remaps.
  CUTLASS_HOST_DEVICE constexpr uint32_t remap_done_offset() const {
    return copy_done_offset() + 2 * num_redundant_slots;
  }
  // The in-kernel planner (`eplb_plan_remap_kernel`) exchanges the per-rank
  // expert histograms through the words below. Generation flags written by the
  // peers, double-buffered by generation parity: [2][kMaxRanks]
  CUTLASS_HOST_DEVICE constexpr uint32_t flags_offset() const { return remap_done_offset() + 1; }
  // Expert histograms pushed by every rank, double-buffered: [2][R][E]
  CUTLASS_HOST_DEVICE constexpr uint32_t hist_offset() const {
    return flags_offset() + 2 * kMaxRanks;
  }
  // This rank's histogram, accumulated by the CTAs of one launch with device
  // atomics; the planning CTA reads and clears it: [E]
  CUTLASS_HOST_DEVICE constexpr uint32_t local_hist_offset() const {
    return hist_offset() + 2 * num_ranks * num_experts;
  }
  // Arrival ticket of one launch, cleared by the planning CTA.
  CUTLASS_HOST_DEVICE constexpr uint32_t plan_ticket_offset() const {
    return local_hist_offset() + num_experts;
  }
  // Generation whose tables are published, released by the planning CTA and
  // acquired by the remapping CTAs of the same launch.
  CUTLASS_HOST_DEVICE constexpr uint32_t plan_ready_offset() const {
    return plan_ticket_offset() + 1;
  }
  // Generation of the last plan; every launch plans generation + 1 and the
  // planning CTA advances the word, so the launch carries no host counter and
  // can be captured in a CUDA graph.
  CUTLASS_HOST_DEVICE constexpr uint32_t plan_generation_offset() const {
    return plan_ready_offset() + 1;
  }
  CUTLASS_HOST_DEVICE constexpr uint32_t num_words() const {
    return math::constexpr_align(plan_generation_offset() + 1, 4u);
  }
};

// Shared memory of the in-kernel planner in bytes: the regions of `PlanSmem`
// in `mega_moe_eplb.cuh`, in order. `num_warps` private histograms are
// kept while the tokens are counted, so hot experts do not serialise one
// shared atomic stream.
CUTLASS_HOST_DEVICE constexpr uint32_t plan_smem_bytes(const uint32_t& num_ranks,
                                                       const uint32_t& num_experts,
                                                       const uint32_t& num_redundant_slots,
                                                       const uint32_t& num_warps) {
  const auto r = num_ranks, e = num_experts, s = num_redundant_slots;
  uint32_t bytes = 0;
  bytes += r * e * sizeof(int32_t);                         // hist
  bytes += num_warps * e * sizeof(int32_t);                 // per-warp histograms
  bytes += e * r * sizeof(int32_t);                         // q
  bytes += 2 * r * s * sizeof(int32_t);                     // replica list: index, snapshot
  bytes += e * sizeof(int32_t);                             // b
  bytes += math::constexpr_align(r, 4u) * sizeof(int32_t);  // resident loads
  bytes += 2 * r * s * sizeof(int32_t);                     // previous and new slot tables
  bytes += math::constexpr_align(r * e, 4u);                // holds
  return bytes;
}

// One weight plane of the symmetric weight arena: `bytes_per_slot` contiguous
// bytes per local slot, starting `offset` bytes into the arena; copied in
// `vec_bytes` units (16 when the plane allows it, else 4).
struct CopyPlane {
  uint64_t offset;
  uint32_t bytes_per_slot;
  uint32_t vec_bytes;
};

struct CopyPlanes {
  static constexpr uint32_t kMaxPlanes = 8;
  CopyPlane planes[kMaxPlanes];
  uint32_t num_planes;
};

// Everything the MegaMoE kernel needs to copy the changed redundant slots
// itself (`kInKernelWeightCopy`): the symmetric weight arena, its planes
// ordered L1 group first (`num_l1_planes` of them) then L2 group, and the copy
// list, its count and the readiness counters inside the workspace. The slot
// geometry is `num_resident_experts` resident slots followed by
// `num_redundant_slots` redundant slots; the kernel's `kNumExpertsPerRank` is
// their sum.
template <uint32_t kNumRanks>
struct InKernelCopy {
  layout::SymBuffer<kNumRanks> arena;
  CopyPlanes planes;
  const int2* copy_list;
  // Cleared by the kernel once consumed, so a later launch without a new plan
  // copies nothing.
  int32_t* copy_count;
  uint32_t* copy_done;
  uint32_t num_resident_experts;
  uint32_t num_redundant_slots;
  uint32_t num_l1_planes;
};

}  // namespace flashinfer::megamoe::eplb
