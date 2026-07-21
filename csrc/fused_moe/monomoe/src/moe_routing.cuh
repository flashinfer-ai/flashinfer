
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

#define FULL_MASK 0xFFFFFFFFU

namespace monomoe {

/**
 * @brief Tie-break-by-lowest-expert-index argmax winner.
 *
 * Among all lanes whose `my_max == warp_max`, returns the SMALLEST
 * `my_expert` (warp-uniform).  Matches the production vLLM `topk_softmax`
 * tie-break ("lower expert index wins") — a lowest-lane tie-break would
 * pick a different expert because experts map to lanes as `expert % 32`.
 */
__device__ static inline uint32_t warp_min_expert_with_max(float my_max, float warp_max,
                                                           uint32_t my_expert) {
  uint32_t cand = (my_max == warp_max) ? my_expert : 0xFFFFFFFFu;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    uint32_t other = __shfl_xor_sync(FULL_MASK, cand, off);
    cand = other < cand ? other : cand;
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
    local_sum += __shfl_xor_sync(FULL_MASK, local_sum, off, 32);

  float inv_sum = 1.0f / local_sum;
#pragma unroll
  for (uint32_t i = 0; i < Count; i++) logits[i] *= inv_sum;
}

/**
 * @brief Top-K expert selection for one token (one warp per token),
 * writing shmem->topk_ids_flat / topk_weights_flat.
 *
 * Calc warps only.
 *
 * Fast path: softmax and sigmoid are monotone, so top-k selection runs on
 * the raw logits and only the k winners are exponentiated.  For
 * softmax + renormalize the global denominator cancels against the
 * renormalizer (weight_k = exp(x_k - max) / sum_topk_exp), so it is never
 * computed.  softmax + renormalize=False needs the full denominator and
 * falls back to a whole-row softmax.
 *
 * @param expert_bias  Optional per-expert selection bias [NUM_EXPERTS]
 *   (GLM-style noaux_tc routing).  When non-null, winners are ranked by
 *   `sigmoid(logit) + bias[e]` while the routing WEIGHT stays the unbiased
 *   sigmoid (recovered post-selection as `metric - bias`).  Sigmoid scoring
 *   only.  Null => raw-logit ranking.
 * @param routed_scaling_factor  Scalar folded into the shared weight
 *   normalizer (exact — the routed output is linear in the weights).
 */
template <typename Dims>
__device__ static __forceinline__ void topK_one_token(
    uint32_t warp_idx, uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
    const __nv_bfloat16* __restrict__ router_logits, MoE_SHM<Dims>* shmem,
    const float* __restrict__ expert_bias, float routed_scaling_factor) {
  static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX,
                "Batch size or number of experts too high for uint32 indices.");

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  uint32_t tid = get_thread<Dims>();

  // Per-thread expert slice: thread t owns experts {t, t+32, t+64, ...}.
  // NUM_EXPERTS % 32 == 0 keeps the fill loop statically bounded so the
  // scores[] / expert_id[] arrays stay in registers instead of spilling
  // to local memory.
  static_assert(Dims::NUM_EXPERTS % 32u == 0u,
                "topK requires NUM_EXPERTS to be a multiple of 32 so "
                "every thread owns exactly NUM_EXPERTS/32 experts (keeps the "
                "per-thread arrays register-resident).");
  constexpr uint32_t MAX_PER_THREAD = Dims::NUM_EXPERTS / 32u;
  float scores[MAX_PER_THREAD];
  uint32_t expert_id[MAX_PER_THREAD];

  // Selection metric: raw logit (unbiased — monotone in the activation),
  // or sigmoid(logit) + bias when expert_bias is set (the bias breaks
  // monotonicity, so the sigmoid must be materialized for selection).
  const bool use_bias = (expert_bias != nullptr);
#pragma unroll
  for (uint32_t i = 0; i < MAX_PER_THREAD; ++i) {
    const uint32_t idx = i * 32u + tid;
    const float logit = (float)router_logits[warp_idx * Dims::NUM_EXPERTS + idx];
    const float metric = use_bias ? (1.0f / (1.0f + __expf(-logit)) + expert_bias[idx]) : logit;
    // Invalid/padded rows can carry NaN/Inf router scores under CUDA graph
    // replay. Treat them as very low scores, but keep them distinct from the
    // already-selected sentinel (-inf) so top-k still returns unique experts.
    scores[i] = (isnan(metric) || isinf(metric)) ? -FLT_MAX : metric;
    expert_id[i] = idx;
  }

  // ── Slow path: softmax + renormalize=False ─────────────────────────────
  if (scoring_func == ScoringFunc::SOFTMAX && !renormalize) {
    warp_softmax_inplace<MAX_PER_THREAD>(scores);
    constexpr float EXCLUDED_SCORE = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
      float max_val = EXCLUDED_SCORE;
      uint32_t max_expert = 0xFFFFFFFFu;
#pragma unroll
      for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
        if (scores[i] > max_val || (scores[i] == max_val && expert_id[i] < max_expert)) {
          max_val = scores[i];
          max_expert = expert_id[i];
        }
      }
      float warp_max = warp_reduce_max_float(max_val);
      uint32_t winning_expert = warp_min_expert_with_max(max_val, warp_max, max_expert);
      float winning_weight = warp_max * routed_scaling_factor;
      if (tid == 0u) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)winning_expert;
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = winning_weight;
      }
#pragma unroll
      for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
        if (expert_id[i] == winning_expert) scores[i] = EXCLUDED_SCORE;
      }
    }
    return;
  }

  // ── Fast path: softmax+renormalize=True OR sigmoid (any renorm) ────────
  // Select the top-k on the metric; activations are applied post-selection
  // to only k values.  The k=0 winner IS the row maximum, so the softmax
  // numerical-safety shift uses topk_scores[0] — no extra warp reduction.
  float topk_scores[MoE_SHM<Dims>::MAX_TOPK];
  uint32_t topk_experts[MoE_SHM<Dims>::MAX_TOPK];
  constexpr float EXCLUDED_SCORE = -INFINITY;
  for (uint32_t k = 0; k < top_k; k++) {
    float max_val = EXCLUDED_SCORE;
    uint32_t max_expert = 0xFFFFFFFFu;
#pragma unroll
    for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
      if (scores[i] > max_val || (scores[i] == max_val && expert_id[i] < max_expert)) {
        max_val = scores[i];
        max_expert = expert_id[i];
      }
    }
    float warp_max = warp_reduce_max_float(max_val);
    uint32_t winning_expert = warp_min_expert_with_max(max_val, warp_max, max_expert);
    topk_scores[k] = warp_max;
    topk_experts[k] = winning_expert;
#pragma unroll
    for (uint32_t i = 0; i < MAX_PER_THREAD; i++) {
      if (expert_id[i] == winning_expert) scores[i] = EXCLUDED_SCORE;
    }
  }

  // Convert selected scores → weights + (re)normalize.  All lanes hold
  // identical topk arrays; thread 0 writes.
  if (tid == 0) {
    if (scoring_func == ScoringFunc::SOFTMAX) {
      // softmax + renormalize=True: weight_k = exp(x_k - max) / sum_topk.
      // (Biased selection is sigmoid-only, so no bias recovery here.)
      float row_max = topk_scores[0];
      float exp_vals[MoE_SHM<Dims>::MAX_TOPK];
      float sum_exp = 0.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        exp_vals[k] = __expf(topk_scores[k] - row_max);
        sum_exp += exp_vals[k];
      }
      float inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
      inv *= routed_scaling_factor;
      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)topk_experts[k];
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = exp_vals[k] * inv;
      }
    } else {
      // Sigmoid: weight_k = sigmoid(x_k), optionally renormalized over the
      // selected set.  Under biased selection the stored metric is
      // sigmoid + bias, so the unbiased weight is metric - bias — no
      // second sigmoid.
      float sig_vals[MoE_SHM<Dims>::MAX_TOPK];
      float sum_sig = 0.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        sig_vals[k] = use_bias ? (topk_scores[k] - expert_bias[topk_experts[k]])
                               : (1.0f / (1.0f + __expf(-topk_scores[k])));
        sum_sig += sig_vals[k];
      }
      float inv = renormalize ? ((sum_sig > 0.0f) ? (1.0f / sum_sig) : 1.0f) : 1.0f;
      inv *= routed_scaling_factor;
      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)topk_experts[k];
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = sig_vals[k] * inv;
      }
    }
  }
}

/**
 * @brief Top-K driver: maps calc warps onto tokens and runs
 * topK_one_token per (warp, token).
 *
 * BS<=8: each calc warp routes exactly one token (`tok == warp_idx`);
 * padding warps (`warp_idx >= num_tokens`) do nothing.
 *
 */
template <typename Dims>
__device__ void topK(uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
                     const __nv_bfloat16* __restrict__ router_logits, uint32_t num_tokens,
                     MoE_SHM<Dims>* shmem, const float* __restrict__ expert_bias = nullptr,
                     float routed_scaling_factor = 1.0f) {
  static_assert(Dims::BS <= 8, "topK supports BS<=8");

  const uint32_t warp_idx = get_calc_warp<Dims>();
  if (warp_idx < num_tokens) {
    topK_one_token<Dims>(warp_idx, top_k, scoring_func, renormalize, router_logits, shmem,
                         expert_bias, routed_scaling_factor);
  }
}

/**
 * @brief Builds the routing tables from topk_ids_flat (warp 0 only).
 *
 * Outputs:
 *   experts[0..expert_count-1].id — active experts, ascending id
 *   expert_count
 *   expert_slot_start[]           — first temp_fp8 row per expert
 *   expert_routed_count[]         — routed pairs per expert
 *   sorted_slot[pair]             — temp_fp8 destination row per routed
 *                                   (tok, k) pair (expert-sorted layout)
 *   down_rank[eid][tok]           — intra-expert rank (0xFF = unrouted)
 *
 * Three phases:
 *   A — vectorized zero of expert_routed_count + 0xFF seed of down_rank,
 *       then tally via __match_any_sync (each lane's 1–2 pair eids stay
 *       cached in registers through Phase C).
 *   B — fused dual warp scan (count prefix + active-expert prefix) →
 *       expert_slot_start (packed u16 vector stores), experts[] (ascending
 *       eid by construction: lanes ascend, in-lane entries ascend), and
 *       expert_count.
 *   C — intra-expert rank per pair via __match_any_sync + cross-chunk
 *       carry → sorted_slot and down_rank.
 */
template <typename Dims>
__device__ void prepare_moe_topk(uint32_t batch_size, uint32_t top_k,
                                 MoE_SHM<Dims>* __restrict__ shm) {
  static_assert(Dims::BS <= 8, "prepare_moe_topk supports BS<=8");
  static_assert(use_tma<Dims>::value, "BS8 prepare path is TMA-only.");

  // Only warp 0 participates; other threads exit and block on the
  // caller's __syncthreads().
  if (threadIdx.x >= 32) return;

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  constexpr uint32_t MAX_PAIRS = Dims::BS * MAX_TOPK;  // 64 (BS8)
  constexpr uint32_t BLK = Dims::NUM_EXPERTS / 32;
  static_assert(Dims::NUM_EXPERTS % 32 == 0,
                "NUM_EXPERTS must be a multiple of 32 for warp blocking of "
                "expert_routed_count[] / expert_slot_start[].");
  static_assert(MAX_PAIRS <= 128, "Phase A caches up to 4 pair-eids per lane (BS * top_k <= 128).");

  const uint32_t tid = threadIdx.x;
  const uint32_t n_pairs = batch_size * top_k;
  auto* tma_shm = &shm->tiny_wgmma_tma;

  // ───────────────── Phase A — tally + cache eids ─────────────────────────

  // Cache eid0 / eid1 in registers — reused in the Phase-A tally AND the
  // Phase-C ranking.  Out-of-range pairs hold the 0xFFFF sentinel.
  const uint32_t p0 = tid;        // chunk-0 pair index (lane → pair)
  const uint32_t p1 = tid + 32u;  // chunk-1 pair index
  // chunk-2 / chunk-3 pair indices — only meaningful when MAX_PAIRS > 64
  // (BS=16: 4 pairs/lane).  On BS=8 they are unused and DCE'd, so no new
  // SASS is emitted for the BS8 instantiation.
  [[maybe_unused]] const uint32_t p2 = tid + 64u;  // chunk-2 pair index
  [[maybe_unused]] const uint32_t p3 = tid + 96u;  // chunk-3 pair index
  auto load_pair_eid = [&](uint32_t pair) -> uint16_t {
    if (pair >= n_pairs) return (uint16_t)0xFFFF;
    const uint32_t tok = pair / top_k;
    const uint32_t k = pair % top_k;
    return shm->topk_ids_flat[tok * MAX_TOPK + k];
  };
  const uint16_t eid0 = load_pair_eid(p0);
  const uint16_t eid1 = load_pair_eid(p1);
  // chunk-2 / chunk-3 eid caches: the SHM loads are gated under
  // `MAX_PAIRS > 64u` so BS=8 issues no load and the variables fold to a
  // dead sentinel that ptxas eliminates.
  [[maybe_unused]] uint16_t eid2 = (uint16_t)0xFFFF;
  [[maybe_unused]] uint16_t eid3 = (uint16_t)0xFFFF;
  if constexpr (MAX_PAIRS > 64u) {
    eid2 = load_pair_eid(p2);
    eid3 = load_pair_eid(p3);
  }

  // Vectorized zero of expert_routed_count: each lane owns BLK contiguous
  // u8 entries; the store width must equal the lane slice (the base is
  // only BLK-byte aligned).
  static_assert(BLK == 2u || BLK == 4u || BLK == 8u || BLK == 16u,
                "Vectorized zero supports BLK in {2,4,8,16} (NUM_EXPERTS in "
                "{64,128,256,512}); add a store width for other counts.");
  {
    auto* zbase = &tma_shm->expert_routed_count[tid * BLK];
    if constexpr (BLK == 2u) {
      *reinterpret_cast<uint16_t*>(zbase) = 0u;
    } else if constexpr (BLK == 4u) {
      *reinterpret_cast<uint32_t*>(zbase) = 0u;
    } else if constexpr (BLK == 8u) {
      *reinterpret_cast<uint64_t*>(zbase) = 0ull;
    } else {  // BLK == 16: two u64 (the base is only 8-byte aligned)
      reinterpret_cast<uint64_t*>(zbase)[0] = 0ull;
      reinterpret_cast<uint64_t*>(zbase)[1] = 0ull;
    }
  }

  // Seed down_rank to the 0xFF "unrouted" sentinel (uint4 stores; the
  // array is alignas(16) and NUM_EXPERTS*BS is a multiple of 16).  Phase C
  // overwrites only the routed cells.
  {
    constexpr uint32_t DOWN_RANK_BYTES = Dims::NUM_EXPERTS * Dims::BS;
    static_assert(DOWN_RANK_BYTES % 16u == 0u,
                  "down_rank uint4 seed requires NUM_EXPERTS*BS % 16 == 0");
    auto* rbase = reinterpret_cast<uint4*>(&tma_shm->down_rank[0][0]);
    const uint4 fill = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
#pragma unroll
    for (uint32_t i = tid; i < DOWN_RANK_BYTES / 16u; i += 32u) {
      rbase[i] = fill;
    }
  }
  __syncwarp();

  // Tally via __match_any_sync: peers sharing an eid form a peer group;
  // the lowest lane writes the popcount.  Sentinel lanes form their own
  // group but the eid bound suppresses the write.  __syncwarp between the
  // two chunks serializes chunk-1's RMW against chunk-0's stores.
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
  if constexpr (MAX_PAIRS > 64u) {
    {
      const uint32_t key = static_cast<uint32_t>(eid2);
      const uint32_t match = __match_any_sync(FULL_MASK, key);
      const uint32_t count = __popc(match);
      const uint32_t lowest = __ffs(match) - 1u;
      if (eid2 < Dims::NUM_EXPERTS && tid == lowest) {
        tma_shm->expert_routed_count[eid2] += static_cast<uint8_t>(count);
      }
      __syncwarp();
    }
    {
      const uint32_t key = static_cast<uint32_t>(eid3);
      const uint32_t match = __match_any_sync(FULL_MASK, key);
      const uint32_t count = __popc(match);
      const uint32_t lowest = __ffs(match) - 1u;
      if (eid3 < Dims::NUM_EXPERTS && tid == lowest) {
        tma_shm->expert_routed_count[eid3] += static_cast<uint8_t>(count);
      }
      __syncwarp();
    }
  }

  // ────────── Phase B — fused prefix sum + active-expert enum ─────────────

  // Per-lane sweep over the owned counts: bulk-load them as one vector
  // (width == BLK bytes — the lane base is only BLK-byte aligned), build
  // both exclusive prefixes (count and active flag) in one pass.
  uint32_t local_counts[BLK];
  uint32_t count_prefix[BLK];
  uint32_t active_prefix[BLK];
  uint32_t lane_total = 0;
  uint32_t lane_actives = 0;
  uint64_t packed_lo = 0, packed_hi = 0;  // hi used only for BLK==16
  {
    const auto* cbase = &tma_shm->expert_routed_count[tid * BLK];
    if constexpr (BLK == 2u) {
      packed_lo = *reinterpret_cast<const uint16_t*>(cbase);
    } else if constexpr (BLK == 4u) {
      packed_lo = *reinterpret_cast<const uint32_t*>(cbase);
    } else if constexpr (BLK == 8u) {
      packed_lo = *reinterpret_cast<const uint64_t*>(cbase);
    } else {  // BLK == 16
      packed_lo = reinterpret_cast<const uint64_t*>(cbase)[0];
      packed_hi = reinterpret_cast<const uint64_t*>(cbase)[1];
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < BLK; ++i) {
    const uint64_t src = (i < 8u) ? packed_lo : packed_hi;
    const uint32_t sh = (i & 7u) * 8u;
    const uint32_t v = static_cast<uint32_t>((src >> sh) & 0xFFu);
    local_counts[i] = v;
    count_prefix[i] = lane_total;
    active_prefix[i] = lane_actives;
    lane_total += v;
    lane_actives += (v > 0u) ? 1u : 0u;
  }

  // Dual-value warp inclusive scan: both totals share the same 5-step
  // butterfly traffic.
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
  const uint32_t lane_slot_offset = scan_total - lane_total;
  const uint32_t lane_active_offset = scan_active - lane_actives;
  const uint32_t expert_count = __shfl_sync(FULL_MASK, scan_active, 31);

  // Writeback: expert_slot_start as packed u16 vector stores (values are
  // bounded by n_pairs <= 64, so the u16 cast is safe; the alignas(16)
  // base makes every width naturally aligned), plus a scalar experts[].id
  // emit per active eid.  Lanes ascend and in-lane entries ascend, so the
  // experts[] enumeration is ascending in eid.
  {
    static_assert(BLK == 2u || BLK == 4u || BLK == 8u || BLK == 16u,
                  "expert_slot_start packed store supports BLK in {2,4,8,16}.");
    uint32_t w[8];
#pragma unroll
    for (uint32_t j = 0; j < BLK / 2u; ++j) {
      const uint32_t a = lane_slot_offset + count_prefix[2u * j];
      const uint32_t b = lane_slot_offset + count_prefix[2u * j + 1u];
      w[j] = a | (b << 16);
    }
    auto* dst = reinterpret_cast<uint32_t*>(&tma_shm->expert_slot_start[tid * BLK]);
    if constexpr (BLK == 2u) {
      dst[0] = w[0];
    } else if constexpr (BLK == 4u) {
      *reinterpret_cast<uint2*>(dst) = make_uint2(w[0], w[1]);
    } else if constexpr (BLK == 8u) {
      *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {  // BLK == 16: two uint4
      *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
      *(reinterpret_cast<uint4*>(dst) + 1) = make_uint4(w[4], w[5], w[6], w[7]);
    }
  }
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

  if (tid == 0) {
    shm->expert_count = expert_count;
  }

  // ───────────────── Phase C — slot assignment ────────────────────────────
  // eid0 / eid1 are register-resident from Phase A.

  const uint32_t lane_mask = (1u << tid) - 1u;

  // Intra-chunk ranks.
  const uint32_t match0 = __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid0));
  const uint32_t rank0 = __popc(match0 & lane_mask);

  const uint32_t match1 = __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid1));
  const uint32_t rank1_intra = __popc(match1 & lane_mask);

  // Chunk-1 cross-chunk carry: for each lane's eid1, count the chunk-0
  // pairs sharing that eid (32 shfl+ballot+popc rounds).
  uint32_t rank1_carry = 0;
  if (n_pairs > 32) {
#pragma unroll
    for (int src = 0; src < 32; ++src) {
      const uint32_t q = __shfl_sync(FULL_MASK, static_cast<uint32_t>(eid1), src);
      const uint32_t b = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid0) == q);
      if (static_cast<int>(tid) == src) rank1_carry = __popc(b);
    }
  }

  // Chunk-2 / chunk-3 ranking (BS=16 only, MAX_PAIRS > 64).  A pair's
  // global rank within its expert's slab is its intra-chunk rank plus the
  // count of same-eid pairs in all LOWER-indexed chunks (chunk order is
  // pair index order: tid, tid+32, tid+64, tid+96).
  [[maybe_unused]] uint32_t rank2_intra = 0;
  [[maybe_unused]] uint32_t rank2_carry = 0;
  [[maybe_unused]] uint32_t rank3_intra = 0;
  [[maybe_unused]] uint32_t rank3_carry = 0;
  if constexpr (MAX_PAIRS > 64u) {
    const uint32_t match2 = __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid2));
    rank2_intra = __popc(match2 & lane_mask);
    const uint32_t match3 = __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid3));
    rank3_intra = __popc(match3 & lane_mask);

    // Chunk-2 carry: same-eid pairs in chunks 0 and 1.
    if (n_pairs > 64) {
#pragma unroll
      for (int src = 0; src < 32; ++src) {
        const uint32_t q = __shfl_sync(FULL_MASK, static_cast<uint32_t>(eid2), src);
        const uint32_t b0 = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid0) == q);
        const uint32_t b1 = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid1) == q);
        if (static_cast<int>(tid) == src) rank2_carry = __popc(b0) + __popc(b1);
      }
    }
    // Chunk-3 carry: same-eid pairs in chunks 0, 1 and 2.
    if (n_pairs > 96) {
#pragma unroll
      for (int src = 0; src < 32; ++src) {
        const uint32_t q = __shfl_sync(FULL_MASK, static_cast<uint32_t>(eid3), src);
        const uint32_t b0 = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid0) == q);
        const uint32_t b1 = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid1) == q);
        const uint32_t b2 = __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid2) == q);
        if (static_cast<int>(tid) == src) rank3_carry = __popc(b0) + __popc(b1) + __popc(b2);
      }
    }
  }

  // Record sorted_slot and down_rank together — `rank` is exactly the
  // value the down-proj would otherwise recompute per expert.  Each
  // (eid, tok) pair is unique (a token's top-k experts are distinct), so
  // the writes are race-free.
  if (p0 < n_pairs && eid0 != 0xFFFF) {
    const uint8_t slot0 = static_cast<uint8_t>(tma_shm->expert_slot_start[eid0] + rank0);
    tma_shm->sorted_slot[p0] = slot0;
    tma_shm->down_rank[eid0][p0 / top_k] = static_cast<uint8_t>(rank0);
  }
  if (p1 < n_pairs && eid1 != 0xFFFF) {
    const uint8_t slot1 =
        static_cast<uint8_t>(tma_shm->expert_slot_start[eid1] + rank1_intra + rank1_carry);
    tma_shm->sorted_slot[p1] = slot1;
    tma_shm->down_rank[eid1][p1 / top_k] = static_cast<uint8_t>(rank1_intra + rank1_carry);
  }
  if constexpr (MAX_PAIRS > 64u) {
    if (p2 < n_pairs && eid2 != 0xFFFF) {
      const uint32_t rank2 = rank2_intra + rank2_carry;
      const uint8_t slot2 = static_cast<uint8_t>(tma_shm->expert_slot_start[eid2] + rank2);
      tma_shm->sorted_slot[p2] = slot2;
      tma_shm->down_rank[eid2][p2 / top_k] = static_cast<uint8_t>(rank2);
    }
    if (p3 < n_pairs && eid3 != 0xFFFF) {
      const uint32_t rank3 = rank3_intra + rank3_carry;
      const uint8_t slot3 = static_cast<uint8_t>(tma_shm->expert_slot_start[eid3] + rank3);
      tma_shm->sorted_slot[p3] = slot3;
      tma_shm->down_rank[eid3][p3 / top_k] = static_cast<uint8_t>(rank3);
    }
  }
}

}  // namespace monomoe

#endif
