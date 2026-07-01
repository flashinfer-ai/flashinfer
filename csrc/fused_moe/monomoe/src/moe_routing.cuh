
#pragma once
#ifndef MOE_GATING_CU
#define MOE_GATING_CU

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cuda_bf16.h>

#include <cfloat>
#include <cstdint>

#include "moe_internal.h"

namespace monomoe {

/**
 * @brief Lowest-index expert tie-break for top-k selection.
 *
 * Given each lane's local-argmax (`my_score`, `my_expert`) and the
 * warp-wide max score, return the SMALLEST `my_expert` among the lanes
 * whose `my_score` equals `warp_max`.  This matches the reference
 * convention (and CUTLASS): when several experts tie on score, the one
 * with the lower index wins.  Selecting by lowest *lane* would be wrong
 * because lane L owns experts {L, L+32, ...}, so the lowest tied lane can
 * hold a higher expert index than a higher tied lane.
 */
__device__ static inline uint32_t warp_lowest_tied_expert(float my_score, uint32_t my_expert,
                                                          float warp_max) {
  uint32_t cand = (my_score == warp_max) ? my_expert : 0xFFFFFFFFU;
  for (int off = 16; off >= 1; off /= 2) {
    cand = min(cand, __shfl_xor_sync(0xFFFFFFFFU, cand, off, 32));
  }
  return cand;
}

/**
 * @brief Warp-cooperative softmax over logits distributed across threads.
 *
 * Each thread holds exactly @p Count values (compile-time).  The static
 * count is required so the loops over the per-thread `logits[]` array
 * are fully unrollable — when the count is a runtime argument, ptxas
 * can't prove the index pattern is static and `logits[]` ends up on
 * the stack frame as local memory, producing the uncoalesced LDL
 * traffic flagged by the NCU "Local Memory" rule.
 */
template <uint32_t Count>
__device__ static __forceinline__ void warp_softmax_inplace(float* logits) {
  float local_max = -FLT_MAX;
#pragma unroll
  for (uint32_t i = 0; i < Count; i++) local_max = fmaxf(local_max, logits[i]);
  float global_max = warp_reduce_max_float(local_max);

  float local_sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < Count; i++) {
    logits[i] = __expf(logits[i] - global_max);
    local_sum += logits[i];
  }
  for (int off = 16; off >= 1; off /= 2)
    local_sum += __shfl_xor_sync(0xFFFFFFFFU, local_sum, off, 32);

  float inv_sum = 1.0f / local_sum;
#pragma unroll
  for (uint32_t i = 0; i < Count; i++) logits[i] *= inv_sum;
}

/**
 * @brief Top-K expert selection for BS <= 8, writing results into shmem.
 *
 * Computes all K selections via warp reduction and stores them in
 * shmem->topk_ids_flat / shmem->topk_weights_flat.
 *
 * Must be called by calc warps only.
 *
 * Optimization: only the row max (over all 256 experts) is needed to do
 * top-k selection — softmax / sigmoid are both monotonic, so the top-k
 * IDs picked by raw `(x - max)` are identical to those picked by
 * softmax(x) or sigmoid(x).  We therefore defer all `__expf` calls to
 * the post-selection step, where only `top_k` (= 8) values are
 * exponentiated instead of all `NUM_EXPERTS` (= 256).  This cuts
 * ~248 expf per warp × 8 warps = ~2000 expf calls per kernel
 * invocation, which dominates the routing phase wall clock.
 *
 * For `softmax + renormalize=True` (the Qwen3.5 case) the math
 * simplifies further: the softmax denominator `sum_all_exp` cancels
 * with the renormalize denominator, so we don't need it at all.
 *   weight_k = (exp(x_k - max) / sum_all) / (sum_topk / sum_all)
 *            = exp(x_k - max) / sum_topk_exp
 *
 * For `softmax + renormalize=False` we still need `sum_all_exp` to
 * normalize, which requires all 256 expf calls — that case falls
 * back to the original path.
 */
template <typename Dims>
__device__ void topK_BS8(uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
                         const __nv_bfloat16* __restrict__ router_logits, uint32_t num_tokens,
                         MoE_SHM<Dims>* shmem) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");
  static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX,
                "Batch size or number of experts too high for uint32 indices.");

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  uint32_t warp_idx = get_calc_warp<Dims>();
  uint32_t tid = get_thread<Dims>();

  if (warp_idx >= num_tokens) {
    if (warp_idx < 8) {
      // padding slots — nothing to do for topk path
    }
    return;
  }

  // Per-thread expert slice.  Each thread owns experts at indices
  // {tid, tid + 32, tid + 64, ...} stepping by warp size, so for
  // NUM_EXPERTS == k * 32 every thread holds exactly k entries.
  // The `% 32 == 0` invariant lets us replace the previous runtime-
  // bounded fill loop (which left `scores[]`/`expert_id[]` on the
  // stack frame as local memory and produced uncoalesced LDL
  // traffic flagged by the NCU "Local Memory" rule) with a
  // statically-bounded `#pragma unroll` loop.  ptxas can then
  // promote both arrays to registers.
  static_assert(Dims::NUM_EXPERTS % 32u == 0u,
                "topK_BS8 requires NUM_EXPERTS to be a multiple of 32 so "
                "every thread owns exactly NUM_EXPERTS/32 experts; this "
                "lets the per-thread scores[]/expert_id[] arrays stay "
                "in registers (avoiding local-memory spills).");
  constexpr uint32_t MAX_PER_THREAD = Dims::NUM_EXPERTS / 32u;
  float scores[MAX_PER_THREAD];
  uint32_t expert_id[MAX_PER_THREAD];

#pragma unroll
  for (uint32_t i = 0; i < MAX_PER_THREAD; ++i) {
    const uint32_t idx = i * 32u + tid;
    scores[i] = (float)router_logits[warp_idx * Dims::NUM_EXPERTS + idx];
    expert_id[i] = idx;
  }

  // ── Slow path: softmax + renormalize=False ─────────────────────────────
  // Needs the full softmax denominator (sum over all 256 exp values), so
  // we cannot skip the bulk expf calls.  Falls back to the original
  // warp-cooperative softmax over all experts.
  if (scoring_func == ScoringFunc::SOFTMAX && !renormalize) {
    warp_softmax_inplace<MAX_PER_THREAD>(scores);
    for (uint32_t k = 0; k < top_k; k++) {
      float max_val = -FLT_MAX;
      uint32_t max_expert = 0;
#pragma unroll
      for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
        if (scores[i] > max_val) {
          max_val = scores[i];
          max_expert = expert_id[i];
        }
      }
      float warp_max = warp_reduce_max_float(max_val);
      // Tie-break to the lowest expert index (not the lowest lane).
      uint32_t winning_expert = warp_lowest_tied_expert(max_val, max_expert, warp_max);
      if (tid == 0) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)winning_expert;
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = warp_max;
      }
#pragma unroll
      for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
        if (expert_id[i] == winning_expert) scores[i] = -FLT_MAX;
      }
    }
    return;
  }

  // ── Fast path: softmax+renormalize=True OR sigmoid (any renorm) ────────
  //
  // Selection is done on raw logits for both softmax and sigmoid.
  // Both functions are monotonically increasing, so the top-k IDs are
  // identical under either the raw-logit or post-activation ordering.
  //
  // For softmax numerical safety we defer the row_max subtraction to the
  // post-selection step: the k=0 winner IS the row maximum, so we
  // subtract topk_scores[0] when computing expf — only K values instead
  // of all NUM_EXPERTS.  This eliminates a full warp reduction + N
  // subtractions from the critical path.

  // ── Top-k selection over the (now-shifted-or-raw) logits ──────────────
  // Track each selected slot's `score_for_choice` (used for ordering) and
  // the expert id.  For both paths, `score_for_choice` is the value used
  // to pick winners; we'll convert to the actual softmax / sigmoid
  // weight in the next step.
  float topk_scores[MoE_SHM<Dims>::MAX_TOPK];
  uint32_t topk_experts[MoE_SHM<Dims>::MAX_TOPK];
  for (uint32_t k = 0; k < top_k; k++) {
    float max_val = -FLT_MAX;
    uint32_t max_expert = 0;
#pragma unroll
    for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
      if (scores[i] > max_val) {
        max_val = scores[i];
        max_expert = expert_id[i];
      }
    }
    float warp_max = warp_reduce_max_float(max_val);
    // Tie-break to the lowest expert index (not the lowest lane).
    uint32_t winning_expert = warp_lowest_tied_expert(max_val, max_expert, warp_max);
    topk_scores[k] = warp_max;
    topk_experts[k] = winning_expert;
#pragma unroll
    for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
      if (expert_id[i] == winning_expert) scores[i] = -FLT_MAX;
    }
  }

  // ── Convert raw selected logits → activation values + (re)normalize ───
  // Only thread 0 does the final write to SHM.  All threads on the warp
  // hold identical `topk_scores` / `topk_experts` arrays (filled via
  // `__shfl_sync` above), so picking thread 0 is arbitrary.
  if (tid == 0) {
    if (scoring_func == ScoringFunc::SOFTMAX) {
      // Softmax + renormalize=True (the renormalize=False case returned
      // earlier).  weight_k = exp(x_k - max) / sum_topk_exp.
      // topk_scores[0] is the row maximum (k=0 finds the global max).
      float row_max = topk_scores[0];
      float exp_vals[MoE_SHM<Dims>::MAX_TOPK];
      float sum_exp = 0.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        exp_vals[k] = __expf(topk_scores[k] - row_max);
        sum_exp += exp_vals[k];
      }
      float inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)topk_experts[k];
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = exp_vals[k] * inv;
      }
    } else {
      // Sigmoid path: weight_k = 1 / (1 + exp(-x_k)).
      // If renormalize, divide by sum of selected sigmoid values (matches
      // the original two-step "compute sigmoid → renormalize topk" math).
      float sig_vals[MoE_SHM<Dims>::MAX_TOPK];
      float sum_sig = 0.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        sig_vals[k] = 1.0f / (1.0f + __expf(-topk_scores[k]));
        sum_sig += sig_vals[k];
      }
      float inv = renormalize ? ((sum_sig > 0.0f) ? (1.0f / sum_sig) : 1.0f) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)topk_experts[k];
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = sig_vals[k] * inv;
      }
    }
  }
}

/**
 * @brief Prepares the BS8 tiny path for top-K single-pass.
 *
 * Reads topk_ids_flat (filled by topK_BS8) and builds:
 *  - experts[0..expert_count-1].id  — ordered list of unique expert ids
 *    active in this batch (first_token/last_token are unused in BS8)
 *  - expert_count                   — number of unique experts
 *
 * Token-to-expert assignment is NOT sorted here. The per-expert loop in
 * moe_kernel_topk_BS8 scans topk_ids_flat directly for each token.
 */
template <typename Dims>
__device__ void prepare_moe_topk_BS8(uint32_t batch_size, uint32_t top_k,
                                     MoE_SHM<Dims>* __restrict__ shm,
                                     MoEGemmSpec<Dims>* __restrict__ spec) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");
  static_assert(use_tma_v<Dims>,
                "BS8 prepare path is TMA-only after the 3-phase rewrite. "
                "All instantiated BS8 variants set USE_TMA=true and the "
                "kernel asserts use_tma in moe_kernel_topk_BS8 — the "
                "non-TMA branch is unreachable.");
  // `spec` is currently unused by this helper; suppress the
  // unused-parameter warning (kept on the signature for symmetry with
  // the other phase helpers and possible future use).
  (void)spec;

  // Only warp 0 (threads 0–31) participates; the other calc threads exit
  // early and block on the caller's `__syncthreads()`.
  if (threadIdx.x >= 32) return;

  // ────────────────────────────────────────────────────────────────────────
  //  3-phase prepare for the BS8 TMA+WGMMA path
  // ────────────────────────────────────────────────────────────────────────
  // Replaces the previous 6 sub-passes (1a/1b/1c + 2a/2b/2c + 3) with three
  // phases.  Two structural simplifications drive the speedup:
  //
  //   1. The 256-bit active-expert bitset (old 1a + 1b) is gone.  The
  //      `expert_routed_count[]` we have to write anyway already encodes
  //      the same information: `expert_routed_count[eid] > 0` IS the
  //      active-expert mask.  Eliminates 8×reg-OR builds, 8×5 SHFL.B32
  //      butterfly OR-reduce, and the lane-0..7 popcount/scan pair.
  //
  //   2. Each lane caches its (up to 2) routed eids in registers across
  //      Phases A and C — Phase C's two `load_eid` SHM reads (≤ 64 SHM
  //      loads warp-wide, on the post-syncthreads critical path) become
  //      register reads.
  //
  // Phase A — Tally + cache eids   (folds 1a + 2a + 2b)
  //   * Each lane loads its 1–2 pair eids (eid0, eid1) and keeps them
  //     in registers.
  //   * Cooperative zero of `expert_routed_count[256]` (each lane
  //     writes BLK = 8 contiguous u8 entries).
  //   * Tally via __match_any_sync — peers sharing an eid form a
  //     warp peer group; the lowest-id lane writes the popcount.
  //
  // Phase B — Fused prefix sum + active-expert enumeration  (folds 1c + 2c)
  //   * Each lane sweeps its 8 contiguous expert_routed_count[] entries
  //     in one 8-iter loop, computing two per-lane locals:
  //         lane_total   = Σ counts            (drives expert_slot_start)
  //         lane_actives = Σ (count > 0)       (drives experts[].id)
  //     plus the two exclusive-prefix arrays count_prefix[8],
  //     active_prefix[8] in registers.
  //   * Single dual-value warp scan: 5-step butterfly carries both
  //     totals through the same shuffle traffic.
  //   * Each lane writes back 8 entries:
  //         expert_slot_start[eid]  = slot_offset + count_prefix[i]
  //     and for every active eid (count > 0) appends to experts[]:
  //         experts[active_offset + active_prefix[i]].id = eid
  //   * Lane 0 publishes `expert_count` for downstream consumers.
  //
  // Phase C — Slot assignment      (Pass 3, register-fed)
  //   * Identical math: __match_any_sync intra-chunk rank + cross-chunk
  //     ballot-shfl carry → sorted_slot[pair].  The two `load_eid` calls
  //     are gone; eid0/eid1 are already in registers from Phase A.
  //
  // Ordering invariants preserved end-to-end:
  //   * `experts[].id`              monotonically increasing in eid
  //                                 (lanes process in ascending tid; within
  //                                  a lane, the 8 entries are in ascending
  //                                  eid order).
  //   * `sorted_slot[pair]`         strictly ascending within each expert
  //                                 when pairs are enumerated in lex order
  //                                 — byte-identical to the v3/v1 semantics.

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  constexpr uint32_t MAX_PAIRS = Dims::BS * MAX_TOPK;  // ≤ 64 for BS=8
  constexpr uint32_t BLK = Dims::NUM_EXPERTS / 32;     // 8 for E=256
  static_assert(Dims::NUM_EXPERTS % 32 == 0,
                "NUM_EXPERTS must be a multiple of 32 for warp blocking of "
                "expert_routed_count[] / expert_slot_start[].");
  static_assert(MAX_PAIRS <= 64, "Phase A caches up to 2 pair-eids per lane (BS·top_k ≤ 64).");

  const uint32_t tid = threadIdx.x;
  const uint32_t n_pairs = batch_size * top_k;
  auto* tma_shm = &shm->tiny_wgmma_tma;

  // Sentinel-fill the routing-time inverse map `expert_tok_krank[E*BS]`
  // to 0xFF ("token does not route to this expert") before the Phase C
  // scatter overwrites the routed (eid, tok) cells.  Total bytes =
  // NUM_EXPERTS * BS.  The 32 lanes sweep the array as a strided grid of
  // 16-byte STS.128 stores: at step `v` lane `tid` writes vector
  // `v*32 + tid` (so the warp covers 512 B contiguously per step, then
  // strides on).  The STS.128 base is 16-aligned by the `alignas(16)` on
  // the array.  This runs on warp 0 alongside the rest of the prepare
  // phase; the epilogues read the table only after the caller's Phase-2
  // trailing `__syncthreads()`.
  {
    constexpr uint32_t KRANK_BYTES = Dims::NUM_EXPERTS * Dims::BS;
    static_assert(KRANK_BYTES % (32u * 16u) == 0u,
                  "expert_tok_krank fill assumes NUM_EXPERTS*BS is a multiple "
                  "of 32 lanes x 16 B (STS.128 per lane).");
    constexpr uint32_t FILL_VECS_PER_LANE = KRANK_BYTES / (32u * 16u);
    const uint4 sentinel = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    uint4* base = reinterpret_cast<uint4*>(&tma_shm->expert_tok_krank[0]);
#pragma unroll
    for (uint32_t v = 0; v < FILL_VECS_PER_LANE; ++v) {
      base[v * 32u + tid] = sentinel;
    }
  }
  __syncwarp();

  // ───────────────── Phase A — Tally + cache eids ─────────────────────────

  // Cache eid0 / eid1 in registers — reused in Phase A tally AND Phase C
  // ranking.  Out-of-range pairs hold the 0xFFFF sentinel.
  const uint32_t p0 = tid;        // chunk-0 pair index (lane → pair)
  const uint32_t p1 = tid + 32u;  // chunk-1 pair index
  auto load_pair_eid = [&](uint32_t pair) -> uint16_t {
    if (pair >= n_pairs) return (uint16_t)0xFFFF;
    const uint32_t tok = pair / top_k;
    const uint32_t k = pair % top_k;
    return shm->topk_ids_flat[tok * MAX_TOPK + k];
  };
  const uint16_t eid0 = load_pair_eid(p0);
  const uint16_t eid1 = load_pair_eid(p1);

  // Cooperative zero of expert_routed_count[NUM_EXPERTS].  Each lane owns
  // BLK = 8 contiguous u8 entries; issue as one STS.64 per lane (32 lanes
  // × 8 B = 256 B → 2 SHM transactions) instead of eight STS.U8 (8
  // transactions).  `expert_routed_count` is declared `alignas(16)` so
  // the per-lane base `tid * BLK` is always 8-byte aligned for BLK=8.
  static_assert(BLK == 8u,
                "Vectorized zero assumes BLK = 8 (one uint64 per lane). "
                "Update the cast width if NUM_EXPERTS / 32 ever changes.");
  *reinterpret_cast<uint64_t*>(&tma_shm->expert_routed_count[tid * BLK]) = 0ull;
  __syncwarp();

  // Tally via __match_any_sync.  For each pair-slot, peers sharing an eid
  // form a warp peer group; the lowest-id lane writes the popcount-sized
  // increment.  Sentinel lanes form their own peer group, but the
  // `eid < NUM_EXPERTS` guard suppresses the write.  __syncwarp() between
  // slots serializes slot-1's RMW against slot-0's stores.
  {
    const uint32_t key = static_cast<uint32_t>(eid0);
    const uint32_t match = __match_any_sync(FULL_MASK, key);
    const uint32_t count = __popc(match);
    const uint32_t lowest = __ffs(match) - 1u;
    if (eid0 < Dims::NUM_EXPERTS && tid == lowest) {
      tma_shm->expert_routed_count[eid0] += static_cast<uint8_t>(count);
    }
    __syncwarp();
  }
  if constexpr (MAX_PAIRS > 32u) {
    const uint32_t key = static_cast<uint32_t>(eid1);
    const uint32_t match = __match_any_sync(FULL_MASK, key);
    const uint32_t count = __popc(match);
    const uint32_t lowest = __ffs(match) - 1u;
    if (eid1 < Dims::NUM_EXPERTS && tid == lowest) {
      tma_shm->expert_routed_count[eid1] += static_cast<uint8_t>(count);
    }
    __syncwarp();
  }

  // ────────── Phase B — Fused prefix sum + active-expert enum ─────────────

  // Per-lane sweep over the 8 owned entries.  Builds local counts +
  // exclusive prefixes for both totals (count and active flag) in one
  // pass.  `local_counts[i]` is cached so the eid emit pass below
  // doesn't re-read SHM.
  //
  // Bulk-load all 8 u8 counts as a single uint64 (8 B, lane-aligned)
  // instead of 8 separate LDS.U8 — one SHM transaction per lane.
  uint32_t local_counts[BLK];
  uint32_t count_prefix[BLK];   // exclusive prefix of counts within block
  uint32_t active_prefix[BLK];  // exclusive prefix of (count > 0) within block
  uint32_t lane_total = 0;
  uint32_t lane_actives = 0;
  const uint64_t packed_counts =
      *reinterpret_cast<const uint64_t*>(&tma_shm->expert_routed_count[tid * BLK]);
#pragma unroll
  for (uint32_t i = 0; i < BLK; ++i) {
    const uint32_t v = static_cast<uint32_t>((packed_counts >> (i * 8u)) & 0xFFu);
    local_counts[i] = v;
    count_prefix[i] = lane_total;
    active_prefix[i] = lane_actives;
    lane_total += v;
    lane_actives += (v > 0u) ? 1u : 0u;
  }

  // Dual-value warp inclusive scan: both totals share the same 5-step
  // butterfly traffic.  Sequential dependency is on the `if (tid >= off)`
  // accumulator only — the two add chains are independent.
  uint32_t scan_total = lane_total;
  uint32_t scan_active = lane_actives;
#pragma unroll
  for (int off = 1; off <= 16; off *= 2) {
    const uint32_t t_total = __shfl_up_sync(FULL_MASK, scan_total, off, 32);
    const uint32_t t_active = __shfl_up_sync(FULL_MASK, scan_active, off, 32);
    if (static_cast<int>(tid) >= off) {
      scan_total += t_total;
      scan_active += t_active;
    }
  }
  const uint32_t lane_slot_offset = scan_total - lane_total;       // exclusive
  const uint32_t lane_active_offset = scan_active - lane_actives;  // exclusive
  // Total active expert count comes from lane 31's inclusive scan.
  const uint32_t expert_count = __shfl_sync(FULL_MASK, scan_active, 31);

  // Combined writeback: every lane writes its 8 expert_slot_start entries
  // in ascending eid order, and emits an experts[].id entry for each
  // active eid.  Since lanes process in ascending tid and within a lane
  // entries are in ascending eid, the global experts[] enumeration is
  // monotonically increasing in eid — same invariant as the old bitset
  // path.
  //
  // The 8 expert_slot_start writes are packed into one STS.128 per lane:
  // `expert_slot_start` is u16, BLK = 8, so each lane's slice is exactly
  // 16 B and lane base `tid * BLK * 2 = tid * 16` is naturally 16-B
  // aligned.  Values are bounded by `n_pairs <= 64`, so casting through
  // u16 is safe.  Sparse `experts[].id` writes stay scalar — their
  // targets are non-contiguous across lanes.
  uint4 packed_starts;
  {
    uint32_t lo[8];
#pragma unroll
    for (uint32_t i = 0; i < BLK; ++i) {
      lo[i] = lane_slot_offset + count_prefix[i];
    }
    // Pack 8 u16 lanes into a uint4 (4 × u32 = 8 × u16).
    packed_starts.x = lo[0] | (lo[1] << 16);
    packed_starts.y = lo[2] | (lo[3] << 16);
    packed_starts.z = lo[4] | (lo[5] << 16);
    packed_starts.w = lo[6] | (lo[7] << 16);
  }
  *reinterpret_cast<uint4*>(&tma_shm->expert_slot_start[tid * BLK]) = packed_starts;
#pragma unroll
  for (uint32_t i = 0; i < BLK; ++i) {
    const uint32_t eid = tid * BLK + i;
    const uint32_t v = local_counts[i];
    if (v > 0u) {
      const uint32_t out = lane_active_offset + active_prefix[i];
      shm->experts[out].id = eid;
    }
  }
  __syncwarp();

  // Lane 0 publishes expert_count (used by downstream consumers in the
  // up- and down-projection helpers, which iterate `shm->experts[e].id`
  // directly for e ∈ [0, expert_count)).
  if (tid == 0) {
    shm->expert_count = expert_count;
  }

  // ───────────────── Phase C — Slot assignment ────────────────────────────
  // Identical math to the v3 Pass 3 — see the long comment block in git
  // history for the evolution from v1 (single-threaded write_head) to v2
  // (warp-parallel inner serial scan) to v3 (__match_any_sync).  The
  // only change here is that eid0 / eid1 are register-resident from
  // Phase A, eliminating the two `load_eid` SHM reads per lane (≤ 64
  // SHM loads warp-wide on the critical path).

  const uint32_t lane_mask = (1u << tid) - 1u;

  // Chunk-0 intra-chunk rank.
  const uint32_t match0 = __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid0));
  const uint32_t rank0 = __popc(match0 & lane_mask);

  // Chunk-1 intra-chunk rank.
  const uint32_t match1 = __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid1));
  const uint32_t rank1_intra = __popc(match1 & lane_mask);

  // Chunk-1 cross-chunk carry: for each lane's eid1, count how many
  // chunk-0 pairs share that eid.  Rotate eid1 through the warp via 32
  // shfl+ballot+popc rounds.  Skipped when n_pairs ≤ 32 (no chunk-1
  // pairs to rank).
  uint32_t rank1_carry = 0;
  if (n_pairs > 32) {
#pragma unroll
    for (int src = 0; src < 32; ++src) {
      const uint32_t q = __shfl_sync(FULL_MASK, static_cast<uint32_t>(eid1), src);
      const uint32_t b = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid0) == q);
      if (static_cast<int>(tid) == src) rank1_carry = __popc(b);
    }
  }

  if (p0 < n_pairs && eid0 != 0xFFFF) {
    tma_shm->sorted_slot[p0] = static_cast<uint8_t>(tma_shm->expert_slot_start[eid0] + rank0);
    // Scatter the (eid, tok) → k rank into the routing-time inverse map.
    // `p0 = tok*top_k + k`, so k = p0 % top_k and tok = p0 / top_k.
    // Top-k selection guarantees distinct experts per token, so
    // (eid0, tok) is unique — no write conflict with the p1 store below
    // or with any other lane.
    const uint32_t tok0 = p0 / top_k;
    const uint32_t k0 = p0 - tok0 * top_k;
    tma_shm->expert_tok_krank[eid0 * Dims::BS + tok0] = static_cast<uint8_t>(k0);
  }
  if (p1 < n_pairs && eid1 != 0xFFFF) {
    tma_shm->sorted_slot[p1] =
        static_cast<uint8_t>(tma_shm->expert_slot_start[eid1] + rank1_intra + rank1_carry);
    const uint32_t tok1 = p1 / top_k;
    const uint32_t k1 = p1 - tok1 * top_k;
    tma_shm->expert_tok_krank[eid1 * Dims::BS + tok1] = static_cast<uint8_t>(k1);
  }
}

}  // namespace monomoe

#endif
