#pragma once

// Replica plan kernels of MegaMoE: plain CUDA tied to no architecture. The
// in-kernel weight copy helpers at the end are built into the SM90 MXFP4 fused
// kernel.
//
// Every rank owns `H = E / R` resident experts in local slots [0, H) and `S`
// redundant expert slots in [H, H + S); the MegaMoE kernel only ever sees
// physical slot ids `rank * (H + S) + local_slot`. A replica plan is the set of
// workspace tables of `eplb_layout.cuh`: which expert each redundant slot
// holds and how each expert's token blocks split across its copies. The
// kernels here turn a logical routing table into a physical one under that
// plan:
//   * remap: the logical-to-physical remap of the top-k ids, rewriting
//            `topk_idx` from logical experts to physical slots;
//   * copy:  fetch the weights of experts newly assigned to this rank's
//            redundant slots from their resident ranks over NVLink, either as
//            a separate kernel or inside the MegaMoE kernel (the in-kernel
//            weight copy, `kInKernelWeightCopy`);
//   * plan:  exchange the per-rank expert histograms over the symmetric
//            workspace and run the same deterministic block-granular planner
//            on every rank, then remap, in one launch (`eplb_plan_remap_kernel`).
// Who writes the tables is up to the caller: a framework may install its own
// plan through the workspace views instead of the in-kernel planner.

#include <flashinfer/megamoe/detail/common.cuh>
#include <flashinfer/megamoe/detail/ptx.cuh>
#include <flashinfer/megamoe/eplb_layout.cuh>

namespace flashinfer::megamoe::eplb {

// Every wait on another CTA or on a peer rank traps after this many SM cycles
// (about 30 s), so a lost rank shows as a launch failure instead of a hang.
static constexpr int64_t kSpinTimeoutCycles = 30ll * 2000000000ll;

// Lane-cooperative argmin over a 64-bit key packing (primary, secondary);
// `kNoneKey` when no lane holds a candidate.
static constexpr uint64_t kNoneKey = ~0ull;

CUTLASS_DEVICE uint64_t warp_min_key(uint64_t key) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const auto other = __shfl_xor_sync(0xffffffffu, key, offset);
    key = other < key ? other : key;
  }
  return key;
}

CUTLASS_DEVICE uint64_t make_key(const uint32_t& primary, const uint32_t& secondary) {
  return (static_cast<uint64_t>(primary) << 32) | secondary;
}

// Shared memory of the planner; `plan_smem_bytes` in the layout header sizes
// it with the same regions in the same order.
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumRedundantSlots,
          uint32_t kNumThreads>
struct PlanSmem {
  static constexpr uint32_t R = kNumRanks, E = kNumExperts, S = kNumRedundantSlots;
  static constexpr uint32_t kNumWarps = kNumThreads / 32;
  int32_t* hist;       // [R][E] histograms of every rank; row `rank` is counted locally
  int32_t* warp_hist;  // [kNumWarps][E] private histograms while counting
  int32_t* q;          // [E][R] blocks of expert e on rank r
  int32_t* rep_idx;    // [R * S] replica list for pruning: e * R + r
  int32_t* rep_v;      // [R * S] replica list for pruning: allocation snapshot
  int32_t* b;          // [E] blocks per expert
  int32_t* load;       // [R] resident loads (the greedy keeps them in registers)
  int32_t* prev_slot;  // [R][S] redundant slot table before the plan
  int32_t* new_slot;   // [R][S] redundant slot table after the plan
  uint8_t* holds;      // [R][E] rank r holds expert e (resident rank, active or cached copy)

  CUTLASS_DEVICE explicit PlanSmem(uint8_t* smem) {
    hist = reinterpret_cast<int32_t*>(smem);
    warp_hist = hist + R * E;
    q = warp_hist + kNumWarps * E;
    rep_idx = q + E * R;
    rep_v = rep_idx + R * S;
    b = rep_v + R * S;
    load = b + E;
    prev_slot = load + math::constexpr_align(R, 4u);
    new_slot = prev_slot + R * S;
    holds = reinterpret_cast<uint8_t*>(new_slot + R * S);
  }
};

// Workspace pointers of one launch (`parity` selects the flag and histogram
// double buffer of the generation).
struct PlanGlobals {
  uint32_t* flags;
  int32_t* hist;
  int32_t* slot_expert;
  int2* copies;
  int32_t* expert_base;
  uint32_t* local_counter;
  int2* copy_list;
  int32_t* copy_count;
  uint32_t* copy_done;
  uint32_t* local_hist;
  uint32_t* plan_ticket;
  uint32_t* plan_ready;
  uint32_t* plan_generation;

  CUTLASS_DEVICE PlanGlobals(uint32_t* ws_base, const Workspace& layout, const uint32_t& parity) {
    flags = ws_base + layout.flags_offset() + parity * Workspace::kMaxRanks;
    hist = reinterpret_cast<int32_t*>(ws_base + layout.hist_offset()) +
           parity * layout.num_ranks * layout.num_experts;
    slot_expert = reinterpret_cast<int32_t*>(ws_base + layout.slot_expert_offset());
    copies = reinterpret_cast<int2*>(ws_base + layout.copies_offset());
    expert_base = reinterpret_cast<int32_t*>(ws_base + layout.expert_base_offset());
    local_counter = ws_base + layout.local_counter_offset();
    copy_list = reinterpret_cast<int2*>(ws_base + layout.copy_list_offset());
    copy_count = reinterpret_cast<int32_t*>(ws_base + layout.copy_count_offset());
    copy_done = ws_base + layout.copy_done_offset();
    local_hist = ws_base + layout.local_hist_offset();
    plan_ticket = ws_base + layout.plan_ticket_offset();
    plan_ready = ws_base + layout.plan_ready_offset();
    plan_generation = ws_base + layout.plan_generation_offset();
  }
};

// Histogram of `topk_idx[begin + thread_idx :: stride]` (entries below `end`)
// into `hist_row[E]`: each warp counts into its own copy, the copies are
// summed afterwards. Ends with the block synchronised.
template <uint32_t kNumExperts, uint32_t kNumThreads>
CUTLASS_DEVICE void eplb_count_tokens(int32_t* hist_row, int32_t* warp_hist,
                                      const int64_t* __restrict__ topk_idx, const uint32_t& begin,
                                      const uint32_t& end, const uint32_t& stride,
                                      const uint32_t& thread_idx) {
  constexpr uint32_t E = kNumExperts, kNumWarps = kNumThreads / 32;
  const uint32_t warp_idx = thread_idx >> 5;
  for (uint32_t i = thread_idx; i < kNumWarps * E; i += kNumThreads) warp_hist[i] = 0;
  __syncthreads();
  auto* mine = warp_hist + warp_idx * E;
  // Four loads in flight per thread; the shared atomics follow.
  uint32_t i = begin + thread_idx;
  for (; i + 3 * stride < end; i += 4 * stride) {
    const auto e0 = __ldg(topk_idx + i);
    const auto e1 = __ldg(topk_idx + i + stride);
    const auto e2 = __ldg(topk_idx + i + 2 * stride);
    const auto e3 = __ldg(topk_idx + i + 3 * stride);
    if (e0 >= 0) atomicAdd_block(mine + static_cast<uint32_t>(e0), 1);
    if (e1 >= 0) atomicAdd_block(mine + static_cast<uint32_t>(e1), 1);
    if (e2 >= 0) atomicAdd_block(mine + static_cast<uint32_t>(e2), 1);
    if (e3 >= 0) atomicAdd_block(mine + static_cast<uint32_t>(e3), 1);
  }
  for (; i < end; i += stride) {
    const auto e = __ldg(topk_idx + i);
    if (e >= 0) atomicAdd_block(mine + static_cast<uint32_t>(e), 1);
  }
  __syncthreads();
  for (uint32_t e = thread_idx; e < E; e += kNumThreads) {
    int32_t sum = 0;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) sum += warp_hist[w * E + e];
    hist_row[e] = sum;
  }
  __syncthreads();
}

// The planner proper: exchange the histograms, wait for every peer, run the
// deterministic greedy (warp 0) and publish the tables. Expects this rank's
// histogram in `sm.hist[rank * E ...]` and the block synchronised; ends with
// the block synchronised and every table written (not yet fenced).
//
// The plan is MegaFlux's load-only block-granular greedy (arXiv 2610.00671,
// App. A.1): every expert's token blocks start on its resident rank; ranks
// above the mean load shift blocks of their largest resident experts to the
// least loaded rank that already holds the expert (an active or cached copy)
// or, failing that, to a fresh redundant slot; a replica whose allocation fits
// elsewhere under the maximum load is folded back; single blocks move between
// the extremes while they share an expert. The test module holds a serial
// reference of the same decisions; only the bookkeeping differs here: the
// greedy keeps the per-rank loads, replica counts and visit marks in one lane
// per rank, picks the receiver with a single tiered argmin (existing copies
// rank below fresh ones), and the replica collection and the slot assignment
// scan experts 32 at a time with ballots.
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumRedundantSlots,
          uint32_t kNumThreads>
CUTLASS_DEVICE void eplb_plan_from_histogram(
    const layout::SymBuffer<kNumRanks>& ws,
    const PlanSmem<kNumRanks, kNumExperts, kNumRedundantSlots, kNumThreads>& sm,
    const PlanGlobals& g, const uint32_t& block_m, const uint32_t& open_gate_blocks,
    const uint32_t& generation) {
  constexpr uint32_t R = kNumRanks, E = kNumExperts, S = kNumRedundantSlots;
  constexpr uint32_t H = E / R, kNumWarps = kNumThreads / 32;
  constexpr uint32_t kExpertsPerLane = math::constexpr_ceil_div(H, 32u);
  FI_MEGAMOE_STATIC_ASSERT(kExpertsPerLane <= 32,
                           "Too many resident experts per rank for the tried mask");
  FI_MEGAMOE_STATIC_ASSERT(S <= Workspace::kMaxRedundantSlots,
                           "Redundant slots are tracked in a 32-bit mask");
  const uint32_t thread_idx = threadIdx.x, lane = thread_idx & 31, warp_idx = thread_idx >> 5;
  const uint32_t rank = ws.rank_idx;
  const uint32_t full = 0xffffffffu;
  auto q_at = [&](const uint32_t& e, const uint32_t& r) -> int32_t& { return sm.q[e * R + r]; };

  // Push the histogram to every rank, then publish this generation.
  for (uint32_t i = thread_idx; i < R * E; i += kNumThreads) {
    const uint32_t dst_rank = i / E, e = i % E;
    *ws.map(g.hist + rank * E + e, dst_rank) = sm.hist[rank * E + e];
  }
  __syncthreads();
  // One lane per peer: a system fence then the flag, issued together (a
  // serial loop of release stores over NVLink cost about 1 us each).
  if (thread_idx < R) {
    __threadfence_system();
    ptx::st_release_sys(ws.map(g.flags + rank, thread_idx), generation);
  }
  if (thread_idx < R) {
    const auto start_clock = clock64();
    while (ptx::ld_acq_sys(g.flags + thread_idx) < generation) {
      if (clock64() - start_clock >= kSpinTimeoutCycles) {
        printf("MegaMoE replica planner timeout: rank=%u waiting for rank=%u generation=%u\n", rank,
               thread_idx, generation);
        FI_MEGAMOE_TRAP();
      }
    }
  }
  __syncthreads();

  // Setup: all histograms, the previous slot table, holders, blocks.
  for (uint32_t i = thread_idx; i < R * E; i += kNumThreads)
    sm.hist[i] =
        static_cast<int32_t>(ptx::ld_volatile(reinterpret_cast<const uint32_t*>(g.hist + i)));
  for (uint32_t i = thread_idx; i < R * S; i += kNumThreads) sm.prev_slot[i] = g.slot_expert[i];
  for (uint32_t i = thread_idx; i < R * E; i += kNumThreads) sm.holds[i] = 0;
  __syncthreads();
  for (uint32_t i = thread_idx; i < R * S; i += kNumThreads) {
    const auto e = sm.prev_slot[i];
    if (e >= 0) sm.holds[(i / S) * E + e] = 1;
  }
  for (uint32_t e = thread_idx; e < E; e += kNumThreads) {
    int32_t total = 0;
#pragma unroll 4
    for (uint32_t r = 0; r < R; ++r) total += sm.hist[r * E + e];
    sm.b[e] = (total + block_m - 1) / block_m;
    for (uint32_t r = 0; r < R; ++r) q_at(e, r) = r == e / H ? sm.b[e] : 0;
    sm.holds[(e / H) * E + e] = 1;
  }
  __syncthreads();
  // Every expert starts on its resident rank, so a rank's load is the block
  // sum of its resident range; one warp per rank.
  for (uint32_t r = warp_idx; r < R; r += kNumWarps) {
    int32_t sum = 0;
    for (uint32_t e = r * H + lane; e < (r + 1) * H; e += 32) sum += sm.b[e];
    sum = __reduce_add_sync(full, sum);
    if (lane == 0) sm.load[r] = sum;
  }
  __syncthreads();

  // The sequential planner runs on warp 0; lane r keeps rank r's state (the
  // other warps hold unused copies, so the state survives the all-warp replica
  // collection between the greedy and the pruning).
  const bool is_rank = lane < R;
  int32_t my_load = is_rank ? sm.load[lane] : 0;
  int32_t my_nrep = 0;
  auto load_of = [&](const uint32_t& r) { return __shfl_sync(full, my_load, r); };
  // `from` and `to` differ; each updates its own column and load.
  auto move = [&](const uint32_t& e, const uint32_t& from, const uint32_t& to, const int32_t& v) {
    if (lane == to) {
      const int32_t before = q_at(e, to);
      if (before == 0 and to != e / H) my_nrep += 1;
      q_at(e, to) = before + v;
      my_load += v;
    }
    if (lane == from) {
      q_at(e, from) -= v;
      my_load -= v;
    }
    __syncwarp();
  };
  if (warp_idx == 0) {
    const int32_t total_blocks = __reduce_add_sync(full, my_load);
    const int32_t tau = (total_blocks + R - 1) / R;
    const int32_t gate = open_gate_blocks > 1 ? static_cast<int32_t>(open_gate_blocks) : 1;

    // Greedy rounds: shift blocks from overloaded resident ranks towards tau.
    for (uint32_t round = 0; round < S + 2; ++round) {
      bool moved = false, visited = false;
      for (uint32_t k = 0; k < R; ++k) {
        uint64_t key = kNoneKey;
        if (is_rank and not visited and my_load > tau)
          key = make_key(static_cast<uint32_t>(0x7fffffff - (my_load - tau)), lane);
        key = warp_min_key(key);
        if (key == kNoneKey) break;
        const uint32_t d = static_cast<uint32_t>(key);
        if (lane == d) visited = true;
        // Bit j: resident expert d * H + lane + 32 * j already tried in this visit.
        uint32_t tried = 0;
        int32_t load_d = load_of(d);
        while (load_d > tau) {
          uint64_t best = kNoneKey;
#pragma unroll
          for (uint32_t j = 0; j < kExpertsPerLane; ++j) {
            const uint32_t e = d * H + lane + 32 * j;
            if (lane + 32 * j < H and not((tried >> j) & 1)) {
              const int32_t qe = q_at(e, d);
              if (qe > 0) {
                const auto cand = make_key(static_cast<uint32_t>(0x7fffffff - qe), e);
                best = cand < best ? cand : best;
              }
            }
          }
          best = warp_min_key(best);
          if (best == kNoneKey) break;
          const uint32_t e = static_cast<uint32_t>(best), local = e - d * H;
          if (lane == (local & 31)) tried |= 1u << (local >> 5);
          const int32_t avail = q_at(e, d);
          // Receivers: ranks already holding the expert (active or cached
          // copy) first, then fresh slots; least loaded first within a tier.
          // One tiered argmin picks both.
          uint64_t pick = kNoneKey;
          int32_t v = 0;
          if (is_rank and lane != d) {
            const int32_t room = tau - my_load;
            const int32_t qe = q_at(e, lane);
            const bool held = qe > 0 or sm.holds[lane * E + e] != 0;
            v = avail < room ? avail : room;
            if (held) {
              if (v > 0 and (qe > 0 or my_nrep < static_cast<int32_t>(S)))
                pick = make_key(static_cast<uint32_t>(my_load), lane);
            } else if (my_load < tau and my_nrep < static_cast<int32_t>(S) and v >= gate) {
              pick = make_key(0x80000000u | static_cast<uint32_t>(my_load), lane);
            }
          }
          const uint64_t chosen = warp_min_key(pick);
          if (chosen == kNoneKey) continue;
          const uint32_t to = static_cast<uint32_t>(chosen);
          const int32_t move_v = __shfl_sync(full, v, to);
          move(e, d, to, move_v);
          moved = true;
          load_d = load_of(d);
        }
      }
      if (not moved) break;
    }
  }
  __syncthreads();

  // Replica list for the pruning: each rank hosts at most S replicas; one warp
  // per rank collects its own in expert order, the whole column read up front
  // and 32 experts per ballot.
  constexpr uint32_t kNumReplicaSlots = R * S;
  constexpr uint32_t kNumReplicasPerLane = math::constexpr_ceil_div(kNumReplicaSlots, 32u);
  constexpr uint32_t kNumExpertChunks = math::constexpr_ceil_div(E, 32u);
  for (uint32_t r = warp_idx; r < R; r += kNumWarps) {
    int32_t column[kNumExpertChunks];
#pragma unroll
    for (uint32_t j = 0; j < kNumExpertChunks; ++j) {
      const uint32_t e = 32 * j + lane;
      column[j] = e < E ? q_at(e, r) : 0;
    }
    uint32_t n = 0;
#pragma unroll
    for (uint32_t j = 0; j < kNumExpertChunks; ++j) {
      const uint32_t e = 32 * j + lane;
      const bool is_replica = e < E and e / H != r and column[j] > 0;
      uint32_t mask = __ballot_sync(full, is_replica);
      while (mask != 0 and n < S) {
        const uint32_t bit = __ffs(mask) - 1;
        const int32_t v = __shfl_sync(full, column[j], bit);
        if (lane == 0) {
          sm.rep_idx[r * S + n] = static_cast<int32_t>((32 * j + bit) * R + r);
          sm.rep_v[r * S + n] = v;
        }
        ++n;
        mask &= mask - 1;
      }
    }
    if (lane == 0) {
      for (; n < S; ++n) sm.rep_v[r * S + n] = 0;
    }
  }
  __syncthreads();

  if (warp_idx == 0) {
    // Pruning: a replica whose allocation fits elsewhere under the current
    // maximum load is folded back, resident rank first, then the least loaded
    // copy.
    const int32_t max_load = __reduce_max_sync(full, is_rank ? my_load : 0);
    while (true) {
      uint64_t best = kNoneKey;
#pragma unroll
      for (uint32_t j = 0; j < kNumReplicasPerLane; ++j) {
        const uint32_t k = lane + 32 * j;
        if (k < kNumReplicaSlots and sm.rep_v[k] > 0) {
          // Order by (snapshot allocation, expert, rank); both fit in 32 bits.
          const auto cand =
              make_key(static_cast<uint32_t>(sm.rep_v[k]), static_cast<uint32_t>(sm.rep_idx[k]));
          best = cand < best ? cand : best;
        }
      }
      best = warp_min_key(best);
      if (best == kNoneKey) break;
      const uint32_t i = static_cast<uint32_t>(best), e = i / R, r = i % R;
      const int32_t v = static_cast<int32_t>(best >> 32);
#pragma unroll
      for (uint32_t j = 0; j < kNumReplicasPerLane; ++j) {
        const uint32_t k = lane + 32 * j;
        if (k < kNumReplicaSlots and sm.rep_idx[k] == static_cast<int32_t>(i) and sm.rep_v[k] == v)
          sm.rep_v[k] = 0;
      }
      __syncwarp();
      if (q_at(e, r) != v) continue;
      const uint32_t resident_rank = e / H;
      uint32_t target = ~0u;
      if (load_of(resident_rank) + v <= max_load) {
        target = resident_rank;
      } else {
        uint64_t pick = kNoneKey;
        if (is_rank and lane != r and lane != resident_rank and q_at(e, lane) > 0 and
            my_load + v <= max_load)
          pick = make_key(static_cast<uint32_t>(my_load), lane);
        pick = warp_min_key(pick);
        if (pick != kNoneKey) target = static_cast<uint32_t>(pick);
      }
      if (target != ~0u) move(e, r, target, v);
    }

    // Single-block moves between the extremes while they share an expert.
    for (uint32_t iter = 0; iter < E * R; ++iter) {
      uint64_t hi_key =
          is_rank ? make_key(static_cast<uint32_t>(0x7fffffff - my_load), lane) : kNoneKey;
      uint64_t lo_key = is_rank ? make_key(static_cast<uint32_t>(my_load), lane) : kNoneKey;
      hi_key = warp_min_key(hi_key);
      lo_key = warp_min_key(lo_key);
      const uint32_t hi = static_cast<uint32_t>(hi_key), lo = static_cast<uint32_t>(lo_key);
      if (load_of(hi) - load_of(lo) <= 1) break;
      uint32_t shared = ~0u;
      for (uint32_t e = lane; e < E; e += 32) {
        if (q_at(e, hi) > 0 and q_at(e, lo) > 0) {
          shared = e;
          break;
        }
      }
      shared = __reduce_min_sync(full, shared);
      if (shared == ~0u) break;
      move(shared, hi, lo, 1);
    }
  }
  __syncthreads();

  // Redundant slot assignment, one warp per rank: keep cached slots that are
  // still used, then fill free slots with new experts in expert order.
  for (uint32_t r = warp_idx; r < R; r += kNumWarps) {
    int32_t kept = -1;  // lane s < S: expert of redundant slot s
    if (lane < S) {
      const auto e = sm.prev_slot[r * S + lane];
      kept = (e >= 0 and q_at(static_cast<uint32_t>(e), r) > 0) ? e : -1;
    }
    uint32_t free_mask = __ballot_sync(full, lane < S and kept < 0);
    for (uint32_t base = 0; base < E and free_mask != 0; base += 32) {
      const uint32_t e = base + lane;
      bool needed = e < E and e / H != r and q_at(e, r) > 0;
#pragma unroll 1
      for (uint32_t s = 0; s < S; ++s)
        needed &= __shfl_sync(full, kept, s) != static_cast<int32_t>(e);
      uint32_t mask = __ballot_sync(full, needed);
      while (mask != 0 and free_mask != 0) {
        const uint32_t es = base + (__ffs(mask) - 1), s = __ffs(free_mask) - 1;
        if (lane == s) kept = static_cast<int32_t>(es);
        free_mask &= free_mask - 1;
        mask &= mask - 1;
      }
    }
    if (lane < S) sm.new_slot[r * S + lane] = kept;
  }
  __syncthreads();

  // Publish: copies, ordinal bases, counters, slot table and copy list.
  for (uint32_t e = thread_idx; e < E; e += kNumThreads) {
    const uint32_t resident_rank = e / H;
    uint32_t k = 0;
    int32_t cum = q_at(e, resident_rank);
    g.copies[e * R + k++] = make_int2(static_cast<int32_t>(resident_rank), cum);
    for (uint32_t r = 0; r < R; ++r) {
      if (r != resident_rank and q_at(e, r) > 0) {
        cum += q_at(e, r);
        g.copies[e * R + k++] = make_int2(static_cast<int32_t>(r), cum);
      }
    }
    if (k < R) g.copies[e * R + k] = make_int2(-1, cum);
    int32_t base = 0;
    for (uint32_t r = 0; r < rank; ++r) base += sm.hist[r * E + e];
    g.expert_base[e] = base;
    g.local_counter[e] = 0;
  }
  if (thread_idx == 0) {
    int32_t count = 0;
    for (uint32_t s = 0; s < S; ++s) {
      const auto e = sm.new_slot[rank * S + s];
      if (e >= 0 and e != sm.prev_slot[rank * S + s]) {
        g.copy_list[count++] = make_int2(static_cast<int32_t>(s), e);
        // In-kernel weight copy: the MegaMoE kernel counts its CTAs into these.
        g.copy_done[s * 2] = 0;
        g.copy_done[s * 2 + 1] = 0;
      }
    }
    *g.copy_count = count;
  }
  for (uint32_t i = thread_idx; i < R * S; i += kNumThreads) g.slot_expert[i] = sm.new_slot[i];
  __syncthreads();
}

// Rewrite this CTA's grid-strided share of `topk_idx` from logical experts to
// physical slots under the published tables. Every entry of an expert needs a
// distinct ordinal within the rank; the CTA counts its share first and
// reserves one contiguous range per expert with a single device atomic, then
// hands out the ordinals from shared memory, so a hot expert sees one atomic
// per CTA instead of one per token. Any assignment of ordinals is a valid
// split; only the counts per (rank, expert) matter. Afterwards the counters
// are left at zero for the next remap: the last CTA to finish (ticket order
// after a fence, so every CTA's atomics are visible) clears them and the
// ticket.
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumRedundantSlots,
          uint32_t kNumThreads>
CUTLASS_DEVICE void eplb_remap_share(uint32_t* ws_base, const Workspace& layout, int64_t* topk_idx,
                                     const uint32_t& num_tokens, const uint32_t& num_topk,
                                     const uint32_t& block_m) {
  constexpr uint32_t R = kNumRanks, E = kNumExperts, S = kNumRedundantSlots;
  constexpr uint32_t H = E / R;
  const auto* slot_expert = reinterpret_cast<const int32_t*>(ws_base + layout.slot_expert_offset());
  const auto* copies = reinterpret_cast<const int2*>(ws_base + layout.copies_offset());
  const auto* expert_base = reinterpret_cast<const int32_t*>(ws_base + layout.expert_base_offset());
  auto* local_counter = ws_base + layout.local_counter_offset();
  auto* remap_done = ws_base + layout.remap_done_offset();

  // Per-expert count of this CTA's share, then the first ordinal of its range;
  // the count doubles as the cursor afterwards.
  __shared__ int32_t cta_cursor[E];
  __shared__ int32_t cta_base[E];
  const uint32_t total = num_tokens * num_topk;
  const uint32_t begin = blockIdx.x * kNumThreads + threadIdx.x, stride = gridDim.x * kNumThreads;
  for (uint32_t e = threadIdx.x; e < E; e += kNumThreads) cta_cursor[e] = 0;
  __syncthreads();
  for (uint32_t i = begin; i < total; i += stride) {
    const auto e64 = topk_idx[i];
    if (e64 >= 0) atomicAdd_block(cta_cursor + static_cast<uint32_t>(e64), 1);
  }
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < E; e += kNumThreads) {
    const auto count = cta_cursor[e];
    if (count > 0) {
      cta_base[e] = expert_base[e] + static_cast<int32_t>(atomicAdd(local_counter + e,
                                                                    static_cast<uint32_t>(count)));
      cta_cursor[e] = 0;
    }
  }
  __syncthreads();
  for (uint32_t i = begin; i < total; i += stride) {
    const auto e64 = topk_idx[i];
    if (e64 < 0) continue;
    const uint32_t e = static_cast<uint32_t>(e64);
    const uint32_t ordinal =
        static_cast<uint32_t>(cta_base[e] + atomicAdd_block(cta_cursor + e, 1));
    const int32_t block = static_cast<int32_t>(ordinal / block_m);
    int32_t dest = e / H;
#pragma unroll 1
    for (uint32_t k = 0; k < R; ++k) {
      const auto copy = copies[e * R + k];
      if (copy.x < 0) break;
      dest = copy.x;
      if (block < copy.y) break;
    }
    uint32_t local_slot = e - dest * H;
    if (static_cast<uint32_t>(dest) != e / H) {
      local_slot = ~0u;
      for (uint32_t s = 0; s < S; ++s)
        local_slot = slot_expert[dest * S + s] == static_cast<int32_t>(e) ? H + s : local_slot;
      // A plan whose copies table sends blocks to a rank holding no slot of the
      // expert (one written by the framework) fails here, not by silent misrouting.
      FI_MEGAMOE_TRAP_ONLY_DEVICE_ASSERT(local_slot != ~0u);
    }
    topk_idx[i] = static_cast<int64_t>(dest) * (H + S) + local_slot;
  }

  __shared__ uint32_t last_cta;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_cta = atomicAdd(remap_done, 1u) == gridDim.x - 1 ? 1u : 0u;
  __syncthreads();
  if (last_cta) {
    __threadfence();
    for (uint32_t e = threadIdx.x; e < E; e += kNumThreads) local_counter[e] = 0;
    if (threadIdx.x == 0) *remap_done = 0;
  }
}

// ---------------------------------------------------------------------------
// Plan and remap in one launch.
//
// Every CTA counts its grid-strided share of the routing into a block
// histogram and adds it to this rank's histogram in the workspace with device
// atomics. The last CTA to arrive (fence, then a ticket) reads that histogram,
// clears it, exchanges it with the peers and runs the planner; when the tables
// are written it releases `plan_ready = generation`. The other CTAs spin on
// that word (each is already resident, so nothing waits on an unscheduled CTA)
// and then every CTA, the planner included, remaps its share of the routing
// exactly as `eplb_remap_kernel` does. The generation is a workspace word that
// the planning CTA advances once every CTA has read it, so the launch carries
// no host counter and can be captured in a CUDA graph. The weight copy, when
// not fused into the MegaMoE kernel, follows this launch in stream order.
//
// The grid is `min(num_sms, ceil(num_tokens * num_topk / kNumThreads))`.
// ---------------------------------------------------------------------------
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumRedundantSlots,
          uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
    eplb_plan_remap_kernel(const __grid_constant__ layout::SymBuffer<kNumRanks> ws,
                           int64_t* topk_idx, const uint32_t num_tokens, const uint32_t num_topk,
                           const uint32_t block_m, const uint32_t open_gate_blocks) {
  FI_MEGAMOE_STATIC_ASSERT(kNumRanks <= Workspace::kMaxRanks, "Too many ranks");
  FI_MEGAMOE_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Experts must divide across ranks");
  constexpr uint32_t R = kNumRanks, E = kNumExperts, S = kNumRedundantSlots;
  const Workspace layout(R, E, S);
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t rank = ws.rank_idx;

  extern __shared__ __align__(16) uint8_t smem_buffer[];
  __shared__ uint32_t is_planner;
  const PlanSmem<R, E, S, kNumThreads> sm(smem_buffer);
  auto* ws_base = ws.template get_base_ptr<uint32_t*>();
  // Read before the arrival ticket: the planning CTA advances the word only
  // after every CTA has arrived.
  const uint32_t generation = ptx::ld_volatile(ws_base + layout.plan_generation_offset()) + 1;
  const PlanGlobals g(ws_base, layout, generation & 1);

  // This CTA's share of the histogram, into the rank's accumulator.
  auto* hist_row = sm.hist + rank * E;
  eplb_count_tokens<E, kNumThreads>(hist_row, sm.warp_hist, topk_idx, blockIdx.x * kNumThreads,
                                    num_tokens * num_topk, gridDim.x * kNumThreads, thread_idx);
  for (uint32_t e = thread_idx; e < E; e += kNumThreads) {
    const auto count = hist_row[e];
    if (count != 0) ptx::red_add(g.local_hist + e, static_cast<uint32_t>(count));
  }
  __threadfence();
  __syncthreads();
  if (thread_idx == 0) is_planner = atomicAdd(g.plan_ticket, 1u) == gridDim.x - 1 ? 1u : 0u;
  __syncthreads();

  if (is_planner) {
    // Every CTA's share has landed: take the histogram and reset the
    // accumulator and the ticket for the next launch.
    __threadfence();
    for (uint32_t e = thread_idx; e < E; e += kNumThreads) {
      hist_row[e] = static_cast<int32_t>(ptx::ld_volatile(g.local_hist + e));
      g.local_hist[e] = 0;
    }
    if (thread_idx == 0) *g.plan_ticket = 0;
    __syncthreads();
    eplb_plan_from_histogram<R, E, S, kNumThreads>(ws, sm, g, block_m, open_gate_blocks,
                                                   generation);
    __threadfence();
    __syncthreads();
    if (thread_idx == 0) {
      ptx::st_release_gpu(g.plan_ready, generation);
      *g.plan_generation = generation;
    }
  } else {
    if (thread_idx == 0) {
      const auto start_clock = clock64();
      while (ptx::ld_acq(g.plan_ready) < generation) {
        if (clock64() - start_clock >= kSpinTimeoutCycles) {
          printf(
              "MegaMoE replica planner timeout: rank=%u cta=%u waiting for the planning CTA, "
              "generation=%u\n",
              rank, blockIdx.x, generation);
          FI_MEGAMOE_TRAP();
        }
      }
    }
    __syncthreads();
  }

  eplb_remap_share<R, E, S, kNumThreads>(ws_base, layout, topk_idx, num_tokens, num_topk, block_m);
}

// Remap only, under the published tables. Grid of at most one CTA per SM;
// `topk_idx` is rewritten in place.
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumRedundantSlots,
          uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
    eplb_remap_kernel(uint32_t* __restrict__ ws_base, int64_t* topk_idx, const uint32_t num_tokens,
                      const uint32_t num_topk, const uint32_t block_m) {
  FI_MEGAMOE_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Experts must divide across ranks");
  const Workspace layout(kNumRanks, kNumExperts, kNumRedundantSlots);
  eplb_remap_share<kNumRanks, kNumExperts, kNumRedundantSlots, kNumThreads>(
      ws_base, layout, topk_idx, num_tokens, num_topk, block_m);
}

// Planes `[plane_begin, plane_end)` of local slot `dst_slot`, filled from the
// resident slot of `expert` on its rank and strided over `num_ctas * kNumThreads`
// threads. 16-byte planes are the weights; the 4-byte planes are the global
// scales (a few words), handled by the same loop in 4-byte units. The source is
// never written after the weights are installed, so no remote synchronisation
// is needed.
template <uint32_t kNumRanks, uint32_t kNumThreads>
CUTLASS_DEVICE void eplb_copy_slot_planes(const layout::SymBuffer<kNumRanks>& arena,
                                          const CopyPlanes& planes, const uint32_t& plane_begin,
                                          const uint32_t& plane_end, const uint32_t& dst_slot,
                                          const uint32_t& expert,
                                          const uint32_t& num_resident_experts,
                                          const uint32_t& cta_idx, const uint32_t& num_ctas,
                                          const uint32_t& thread_idx) {
  const uint32_t resident_rank = expert / num_resident_experts;
  const uint32_t src_slot = expert - resident_rank * num_resident_experts;
  auto* base = arena.template get_base_ptr<uint8_t*>();
  const uint32_t stride = num_ctas * kNumThreads;
  const uint32_t start = cta_idx * kNumThreads + thread_idx;
  for (uint32_t p = plane_begin; p < plane_end; ++p) {
    const auto& plane = planes.planes[p];
    const uint8_t* src =
        arena.map(base + plane.offset + static_cast<uint64_t>(src_slot) * plane.bytes_per_slot,
                  resident_rank);
    uint8_t* dst = base + plane.offset + static_cast<uint64_t>(dst_slot) * plane.bytes_per_slot;
    if (plane.vec_bytes == 16) {
      const auto* s16 = reinterpret_cast<const uint4*>(src);
      auto* d16 = reinterpret_cast<uint4*>(dst);
      const uint32_t n = plane.bytes_per_slot / 16;
      // Four loads in flight per thread: NVLink latency, not issue rate, bounds
      // the copier.
      uint32_t i = start;
      for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 v0 = __ldg(s16 + i);
        const uint4 v1 = __ldg(s16 + i + stride);
        const uint4 v2 = __ldg(s16 + i + 2 * stride);
        const uint4 v3 = __ldg(s16 + i + 3 * stride);
        d16[i] = v0;
        d16[i + stride] = v1;
        d16[i + 2 * stride] = v2;
        d16[i + 3 * stride] = v3;
      }
      for (; i < n; i += stride) d16[i] = __ldg(s16 + i);
    } else {
      const uint32_t n = plane.bytes_per_slot / 4;
      for (uint32_t i = start; i < n; i += stride)
        reinterpret_cast<uint32_t*>(dst)[i] = __ldg(reinterpret_cast<const uint32_t*>(src) + i);
    }
  }
}

// Separate weight copy: CTA (x, y) copies a `gridDim.y`-strided share of every
// plane of listed slot `x` from the resident rank.
template <uint32_t kNumRanks, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
    eplb_copy_weights_kernel(const __grid_constant__ layout::SymBuffer<kNumRanks> arena,
                             const __grid_constant__ CopyPlanes planes,
                             const int2* __restrict__ copy_list,
                             const int32_t* __restrict__ copy_count,
                             const uint32_t num_resident_experts) {
  if (static_cast<int32_t>(blockIdx.x) >= *copy_count) return;
  const auto entry = copy_list[blockIdx.x];
  eplb_copy_slot_planes<kNumRanks, kNumThreads>(
      arena, planes, 0u, planes.num_planes, num_resident_experts + static_cast<uint32_t>(entry.x),
      static_cast<uint32_t>(entry.y), num_resident_experts, blockIdx.y, gridDim.y, threadIdx.x);
}

// ---------------------------------------------------------------------------
// In-kernel weight copy: the MegaMoE kernel moves the changed redundant slots
// itself.
//
// Every CTA copies a strided share of each listed slot, one plane group at a
// time (L1 planes, then L2 planes), and adds one to the slot's group counter
// with release semantics once its share is written. A slot's group is ready
// when the counter equals the grid size; the weight loader waits for it with
// an acquire load before the first bulk copy of that slot. Slots that are not
// in the copy list, and resident slots, are never waited on.
// ---------------------------------------------------------------------------

// Bit `s` set when redundant slot `s` is in this launch's copy list.
template <uint32_t kNumRanks>
CUTLASS_DEVICE uint32_t in_kernel_copy_pending_mask(const InKernelCopy<kNumRanks>& fc) {
  uint32_t mask = 0;
  const auto count = static_cast<uint32_t>(ptx::ld_volatile(fc.copy_count));
  for (uint32_t i = 0; i < count and i < fc.num_redundant_slots; ++i)
    mask |= 1u << static_cast<uint32_t>(fc.copy_list[i].x);
  return mask;
}

// This CTA's share of every listed slot, L1 group then L2 group, each group
// published with one release per slot. `sync` is a barrier over the
// `kNumThreads` copier threads of this CTA; `thread_idx` indexes them.
template <uint32_t kNumRanks, uint32_t kNumThreads, typename SyncFn>
CUTLASS_DEVICE void in_kernel_copy_share(const InKernelCopy<kNumRanks>& fc, const uint32_t& cta_idx,
                                         const uint32_t& num_ctas, const uint32_t& thread_idx,
                                         SyncFn&& sync) {
  const auto count =
      min(static_cast<uint32_t>(ptx::ld_volatile(fc.copy_count)), fc.num_redundant_slots);
  if (count == 0) return;
#pragma unroll 1
  for (uint32_t group = 0; group < 2; ++group) {
    const uint32_t plane_begin = group == 0 ? 0 : fc.num_l1_planes;
    const uint32_t plane_end = group == 0 ? fc.num_l1_planes : fc.planes.num_planes;
#pragma unroll 1
    for (uint32_t i = 0; i < count; ++i) {
      const auto entry = fc.copy_list[i];
      eplb_copy_slot_planes<kNumRanks, kNumThreads>(
          fc.arena, fc.planes, plane_begin, plane_end,
          fc.num_resident_experts + static_cast<uint32_t>(entry.x), static_cast<uint32_t>(entry.y),
          fc.num_resident_experts, cta_idx, num_ctas, thread_idx);
    }
    // Make every copier thread's stores visible at GPU scope before one thread
    // publishes the group; the release is cumulative over the barrier.
    __threadfence();
    sync();
    if (thread_idx == 0) {
      for (uint32_t i = 0; i < count; ++i)
        ptx::red_add_rel(fc.copy_done + static_cast<uint32_t>(fc.copy_list[i].x) * 2 + group, 1u);
    }
  }
}

// Wait until `num_ctas` CTAs have published the group counter `counter`
// (`copy_done + slot * 2 + group`), then order the following async-proxy
// (bulk copy) reads after it. Kept free of the argument struct: it runs in the
// weight-loader warp inside its register budget. Traps on timeout.
CUTLASS_DEVICE void in_kernel_copy_wait_slot(const uint32_t* counter, const uint32_t& num_ctas) {
  const auto start_clock = clock64();
  while (ptx::ld_acq(counter) < num_ctas)
    FI_MEGAMOE_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSpinTimeoutCycles);
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

}  // namespace flashinfer::megamoe::eplb
