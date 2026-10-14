
#pragma once
#ifndef MOE_UP_PROJECTION_CU
#define MOE_UP_PROJECTION_CU

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cuda.h>
#include <cuda_fp8.h>

#include "moe_interface.h"
#include "moe_internal.h"
#include "moe_tma.h"
#include "ptx_utils.h"

namespace monomoe {

/**
 * @brief Async (cp.async) per-expert up-projection block-scale prefetch.
 *
 * Issues the expert's block-scale tile as non-blocking 4-byte cp.async
 * copies so the cold-miss DRAM latency overlaps the previous expert's
 * K-loop instead of sitting exposed at the cross-expert boundary.  The
 * caller must cp_async_commit_group() after issuing and drain with
 * cp_async_wait_group<0>() before the values are read.
 *
 * `base_row_up` is the first weight row (in the lower, gate half of the
 * 2*N rows) owned by this block.  Prefetch-warp 0 only; the 32-strided
 * loop covers tiles larger than one warp (TILE = 2 * UP_SCALE_COLS, e.g.
 * 48 for K=3072 — a plain `thread < TILE` guard would silently drop the
 * upper up-half scales).
 */
template <typename Dims>
__device__ inline void moe_prefetch_up_scale_for_row_async(
    const S_element* __restrict__ expert_scales_up, std::uint32_t id, unsigned base_row_up,
    S_element* __restrict__ dest) {
  constexpr uint32_t COLS = Dims::UP_SCALE_COLS;
  constexpr uint32_t TILE = 2 * COLS;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  if (warp == 0) {
    for (uint32_t i = thread; i < TILE; i += 32u) {
      uint32_t rb_local = i / COLS;  // 0 → gate half, 1 → up half
      uint32_t kb = i % COLS;
      uint32_t row = base_row_up + rb_local * Dims::N;
      uint32_t rb_global = row / Dims::BLOCK_SCALE_ROW;
      const S_element* src = &expert_scales_up[id * Dims::UP_SCALE_ROWS * Dims::UP_SCALE_COLS +
                                               rb_global * Dims::UP_SCALE_COLS + kb];
      cp_async_cg_4(&dest[i], src);
    }
  }
}

/**
 * @brief Up-projection for BS <= 8, single-atom interleaved layout
 * (UP_COL_HALVES == 1).
 *
 * TMA + WGMMA pipeline; see docs/design_docs/monomoe_kernel.md "Phase 3 — up-projection".  The
 * weight tensor must be pre-interleaved in Python
 * (interleave_for_tma_wgmma_up_v2) so one 128×128 TMA fetches a full WGMMA
 * A-tile in the gate/up pair layout.  The activation operand comes from
 * `fp8_act_full`, produced once per launch by Phase 2.
 *
 * Entry contract (established by moe_kernel_topk_impl):
 *   * bar_w[*] initialized and release-fenced; slot 0 NOT pre-armed (this
 *     helper primes the pipeline for expert_start; for later experts the
 *     previous expert's cross-expert stitch does).
 *   * fp8_act_full / act_scale populated and published by the Phase-2
 *     trailing __syncthreads().
 */
template <typename Dims>
__device__ inline void moe_up_projection_allexperts_wgmma_tma(
    const A_element* __restrict__ activations_in, const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up, std::uint32_t top_k, std::uint32_t batch_size,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& up_weights_desc, CUtensorMap const& activations_desc,
    std::uint32_t up_block_idx = 0xffffffffu, std::uint32_t expert_start = 0,
    std::uint32_t expert_stride = 1) {
  static_assert(Dims::BS <= 8, "allexperts up-proj supports BS<=8.");
  using CoreDims = MoECoreDims<Dims>;

  // All GM reads go through the TMA descriptors; the raw pointers stay on
  // the signature for parity with the kernel interface.
  (void)activations_in;
  (void)expert_weights_up;

  constexpr uint32_t W_UP_M = CoreDims::W_UP_TILE_WGMMA;     // 128
  constexpr uint32_t K_STEP_WGMMA = CoreDims::K_STEP_WGMMA;  // 128
  constexpr uint32_t K_STEP = CoreDims::K_STEP_UP;
  constexpr uint32_t K_SUBSTEPS = CoreDims::K_SUBSTEPS_UP;
  constexpr uint32_t K_TILES = CoreDims::K_TILES_UP;
  constexpr uint32_t WGMMAS_PER_SUBSTEP = CoreDims::WGMMAS_PER_STEP;  // 4
  constexpr uint32_t UP_SCALE_COLS = Dims::UP_SCALE_COLS;
  // Weight-TMA lookahead depth + derived arm distance (see MoECoreDims).
  constexpr uint32_t SLOTS = CoreDims::UP_W_SLOTS;
  constexpr uint32_t ARM_DISTANCE = CoreDims::UP_ARM_DISTANCE;

  // A operand: 128×128 Major::K B128-swizzled weight tile.  The TMA
  // hardware applies the core-matrix XOR swizzle at write time; the
  // matching CUTLASS strides are LBO=16 (K-core-matrix within the 1024-B
  // atom), SBO=1024 (next M-atom), swizzle=1.
  constexpr uint64_t A_LBO = 16ULL;
  constexpr uint64_t A_SBO = 1024ULL;
  constexpr uint32_t A_SWIZZLE = 1u;
  // B operand: fp8 activations from fp8_act_full, SWIZZLE_NONE.  LBO steps
  // between successive 16-B K-chunks; the padded 9-token-row layout makes
  // that 9 * 16 = 144 B (the 9th row is skipped — see fp8_act_full).
  constexpr uint64_t B_LBO =
      static_cast<uint64_t>(MoE_SHM<Dims>::TinyDataWGMMA_TMA::FP8_ACT_T_TILE_PADDED) *
      static_cast<uint64_t>(MoE_SHM<Dims>::TinyDataWGMMA_TMA::FP8_ACT_K_CHUNK);
  // B_SBO = stride between adjacent WGMMA N core matrices (8 tokens each).
  // BS<=8: the m64n8k32 issue has a single N core matrix, so the hardware
  // never consumes SBO; keep it equal to B_LBO so the emitted descriptor
  // immediate stays byte-identical to the historical BS8 stream (the value
  // is inert).
  constexpr uint64_t B_SBO = B_LBO;

  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;  // 0..11
  const unsigned lane = thread_in_block & 31;
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;

  auto* shm = &shmem->tiny_wgmma_tma;

  const unsigned effective_bid = (up_block_idx == 0xffffffffu) ? blockIdx.x : up_block_idx;
  // Each block owns 128 M rows = 2 WG stripes × 64 rows; WG0's gate rows
  // start at base_row_up, WG1's at base_row_up + 32.
  const unsigned base_row_up = effective_bid * (W_UP_M / 2);
  const std::uint32_t expert_count = shmem->expert_count;

  // Per-thread fp32 accumulators for WGMMA m64n8k32 (BS<=8).
  float chunk_d0 = 0.f, chunk_d1 = 0.f, chunk_d2 = 0.f, chunk_d3 = 0.f;
  float final_d0 = 0.f, final_d1 = 0.f, final_d2 = 0.f, final_d3 = 0.f;

  // A non-multiple K_TILES would phase-shift the slot index across the
  // expert boundary and break the mbarrier parity chain (authoritative
  // check in MoECoreDims; mirrored here at the consuming kernel).
  static_assert(K_TILES % SLOTS == 0,
                "SLOTS-deep up-proj pipeline requires K_TILES_UP to be a "
                "multiple of UP_W_SLOTS so the cross-expert stitch lands "
                "the next expert's iters [0,A) on slots [0,A).");

  constexpr uint32_t UP_W_TX_BYTES_PER_SUBSTEP = 16384u;  // 128×128 fp8 atom
  constexpr uint32_t UP_W_TX_BYTES_TOTAL = UP_W_TX_BYTES_PER_SUBSTEP * K_SUBSTEPS;

  // Issue the K_SUBSTEPS weight TMAs for one slot of (expert, outer-K
  // iter).  Caller pre-arms bar_w[slot] with UP_W_TX_BYTES_TOTAL.
  auto tma_slot = [&](uint32_t slot, uint32_t eid, uint32_t k_iter) {
#pragma unroll
    for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
      tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/eid, /*N=*/Dims::N,
                             /*base_row_up=*/base_row_up,
                             /*k_start=*/k_iter * K_STEP + kk * K_STEP_WGMMA,
                             /*dest_slot=*/&shm->w_wgmma[slot][kk * W_UP_M][0],
                             /*bar=*/&shm->bar_w[slot]);
    }
  };

  // ── Pre-loop: prime slots [0, ARM_DISTANCE) for expert_start ──────────
  // At iter s the launcher arms slot (s+A) % S for logical iter s+A; the
  // pre-loop primes iters 0..A-1.
  if (is_tma_launcher_thread<Dims>() && expert_start < expert_count) {
    const uint32_t first_id = shmem->experts[expert_start].id;
#pragma unroll
    for (uint32_t a = 0; a < ARM_DISTANCE; ++a) {
      mbarrier_arrive_expect_tx(&shm->bar_w[a], UP_W_TX_BYTES_TOTAL);
      tma_slot(a, first_id, a);
    }
  }

  // Deferred-writeback state: expert e's SiLU + fp8 quant + GM writeback
  // runs on prefetch warps during expert e+1's K-loop; the LAST expert
  // drains inline after the loop.  Uniform across threads.
  bool has_pending_writeback = false;

  // Scale ping-pong: expert e consumes up_scale[cur_scale_slot] and
  // prefetches e+1 into the other slot during its K-loop.
  uint32_t cur_scale_slot = 0;
  {
    if (is_prefetch_warp<Dims>() && expert_start < expert_count) {
      moe_prefetch_up_scale_for_row_async<Dims>(expert_scales_up, shmem->experts[expert_start].id,
                                                base_row_up, shm->up_scale[0]);
      cp_async_commit_group();
    }
  }

  // Per-slot mbarrier parity, hoisted OUT of the expert loop so it stays
  // continuously synced with the bar_w[slot] hardware phase across expert
  // transitions (same reasoning as the down-proj's hoisted parity_w/parity_a).
  // A per-expert reset to 0 is only correct when each slot completes an
  // EVEN number of phases per expert (K_TILES / SLOTS even); for an odd
  // quotient (e.g. K_TILES=12, SLOTS=4 → 3) the barrier ends the expert at
  // phase 1 and a blind reset makes the next expert's first wait pass on
  // the stale phase, corrupting the arm/wait pairing.
  uint32_t parity_w[SLOTS];
#pragma unroll
  for (uint32_t i = 0; i < SLOTS; ++i) parity_w[i] = 0u;

  // ── Expert loop ───────────────────────────────────────────────────────
  for (uint32_t e = expert_start; e < expert_count; e += expert_stride) {
    const uint32_t id = shmem->experts[e].id;
    const bool has_next_e = (e + expert_stride < expert_count);
    const uint32_t next_id = has_next_e ? shmem->experts[e + expert_stride].id : 0u;

    final_d0 = final_d1 = final_d2 = final_d3 = 0.f;

    // Drain this expert's prefetched scale (issued a full K-loop ago);
    // prefetch the next expert's into the other ping-pong slot.
    {
      if (is_prefetch_warp<Dims>()) {
        cp_async_wait_group<0>();
        if (has_next_e) {
          moe_prefetch_up_scale_for_row_async<Dims>(expert_scales_up, next_id, base_row_up,
                                                    shm->up_scale[cur_scale_slot ^ 1u]);
          cp_async_commit_group();
        }
      }
    }

    // ── Per-expert routing cache populate ──
    // One calc thread per token: vector-load the token's 8 topk ids (16 B)
    // and all 8 weights up front, then select the match in registers.
    // This avoids the dependent SHM chain (id scan → weight load at a
    // rank-dependent address) that otherwise stalls all 384 threads at the
    // publish sync below.  The same sync publishes up_scale and the cache.
    {
      constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
      if (thread_in_block < batch_size && MAX_TOPK == 8u) {
        const std::uint32_t tok = thread_in_block;
        // topk_ids_flat is 8-byte aligned; each token slab is 16 B in, so
        // two uint2 loads are safely aligned.
        const uint16_t* idb = &shmem->topk_ids_flat[tok * MAX_TOPK];
        uint16_t ids[8];
        *reinterpret_cast<uint2*>(&ids[0]) = *reinterpret_cast<const uint2*>(idb);
        *reinterpret_cast<uint2*>(&ids[4]) = *reinterpret_cast<const uint2*>(idb + 4);

        const S_element* wb = &shmem->topk_weights_flat[tok * MAX_TOPK];
        float w[8];
#pragma unroll
        for (std::uint32_t k = 0; k < 8u; ++k) w[k] = wb[k];

        // Branchless select — topk ids are distinct, so at most one match.
        const uint16_t target = (uint16_t)id;
        uint8_t k_found = 0xFFu;
        float rw_found = 0.f;
#pragma unroll
        for (std::uint32_t k = 0; k < 8u; ++k) {
          const bool m = (k < top_k) && (ids[k] == target);
          if (m) {
            k_found = static_cast<uint8_t>(k);
            rw_found = w[k];
          }
        }
        shm->up_rank_for_tok[tok] = k_found;
        shm->up_rw_for_tok[tok] = rw_found;
      }
    }

    // Publish up_scale (prefetch warp) + the routing cache (calc warps)
    // to the whole block.  Outside the K-loop, so the loop still has at
    // most one __syncthreads per iteration.
    __syncthreads();

    // ── Main K-loop ───────────────────────────────────────────────────
    // Per iteration: calc warps wait bar_w[s % S] then run K_SUBSTEPS ×
    // (4 chained WGMMAs + scale-apply); the launcher concurrently arms +
    // TMAs the slot A iters ahead (intra-expert, or the next expert's
    // first iters via the cross-expert stitch).
    for (uint32_t s = 0; s < K_TILES; ++s) {
      const uint32_t cur_slot = s & (SLOTS - 1u);
      const uint32_t arm_slot = (s + ARM_DISTANCE) & (SLOTS - 1u);

      if (is_calc) {
        while (!mbarrier_try_wait_parity(&shm->bar_w[cur_slot], parity_w[cur_slot])) {
        }
        parity_w[cur_slot] ^= 1;

        // Substep kk occupies SHM rows [kk*128, +128); the WG row offset
        // (0 / 64) picks the per-WG half within each atom.
        const void* a_slot_base = (const void*)&shm->w_wgmma[cur_slot][0][0];
        const uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;
        constexpr uint32_t K_SUBSTEP_W_BYTES = 16384u;
        constexpr uint32_t A_K_STRIDE = 2u * static_cast<uint32_t>(A_LBO);

#pragma unroll
        for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
          // One fence per dependent-WGMMA group (paired with the
          // commit/wait below).
          wgmma_fence();

          const void* a_kk_base =
              (const void*)((const char*)a_slot_base + kk * K_SUBSTEP_W_BYTES + wg_offset_bytes);
          const uint32_t kblk = s * K_SUBSTEPS + kk;

#pragma unroll
          for (uint32_t j = 0; j < WGMMAS_PER_SUBSTEP; ++j) {
            const void* a_ptr = (const void*)((const char*)a_kk_base + j * A_K_STRIDE);
            const void* b_ptr = (const void*)&shm->fp8_act_full[kblk][j * 2][0][0];
            uint64_t desc_a = make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
            uint64_t desc_b = make_wgmma_desc(b_ptr, B_LBO, B_SBO, 0);
            wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d0, chunk_d1, chunk_d2, chunk_d3);
          }

          wgmma_commit_group();
          wgmma_wait_group<0>();

          // Scale-apply at the K=128 boundary (the block-wise FP8 scale
          // granularity).  Pair layout: d0/d1 are gate rows, d2/d3 up
          // rows within the same warp, each with its own weight scale.
          const uint32_t tok_02 = (lane % 4) * 2;
          const uint32_t tok_13 = tok_02 + 1;
          const float as_02 = shmem->act_scale[kblk][tok_02];
          const float as_13 = shmem->act_scale[kblk][tok_13];
          {
            const S_element* ws_buf = shm->up_scale[cur_scale_slot];
            const float ws_gate = ws_buf[kblk + 0u];
            const float ws_up = ws_buf[kblk + UP_SCALE_COLS];
            final_d0 += chunk_d0 * ws_gate * as_02;
            final_d1 += chunk_d1 * ws_gate * as_13;
            final_d2 += chunk_d2 * ws_up * as_02;
            final_d3 += chunk_d3 * ws_up * as_13;
          }
          chunk_d0 = chunk_d1 = chunk_d2 = chunk_d3 = 0.f;
        }
      }

      // Launcher (runs in parallel with the WGMMAs above): arm + TMA the
      // slot A iters ahead — intra-expert while s+A < K_TILES, else the
      // cross-expert stitch into the next expert's iters [0, A).  On the
      // last expert's last A iters the barriers are left idle (Phase 4
      // re-initializes them).
      if (is_tma_launcher_thread<Dims>()) {
        const uint32_t target_iter = s + ARM_DISTANCE;
        if (target_iter < K_TILES) {
          mbarrier_arrive_expect_tx(&shm->bar_w[arm_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
          tma_slot(arm_slot, id, target_iter);
        } else if (has_next_e) {
          mbarrier_arrive_expect_tx(&shm->bar_w[arm_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
          tma_slot(arm_slot, next_id, target_iter - K_TILES);
        }
      }

      // ── Deferred SiLU + fp8 quant writeback for the PREVIOUS expert ──
      //
      // One PF warp per token (32 lanes cover a token's 64 output features
      // via 2 post_silu reads; warp_reduce_max needs the whole warp), so a
      // wave is PF = 4 tokens.  Waves are spaced evenly over the first
      // K_TILES-1 iterations — the last iteration stays free for the
      // cross-expert stitch, and spacing lets DRAM drain between the
      // temp_fp8 store bursts.  Numerically schedule-invariant:
      // post_silu_scratch holds the previous expert's values for the whole
      // current K-loop and the routing tables are immutable.
      static_assert(K_TILES >= 2, "Deferred up-proj writeback requires K_TILES >= 2.");
      constexpr uint32_t PF = CoreDims::PREFETCH_WARP_COUNT;  // 4
      constexpr uint32_t WAVES = (Dims::BS + PF - 1u) / PF;
      constexpr uint32_t DEFER_ITERS = K_TILES - 1u;
      static_assert(WAVES <= DEFER_ITERS,
                    "Not enough K-loop iterations to give each up-proj "
                    "epilogue wave its own K-step (need ceil(BS/4) <= "
                    "K_TILES-1).");
      if (has_pending_writeback && is_prefetch_warp<Dims>() && s < DEFER_ITERS) {
        const unsigned pf_warp = warp - CoreDims::CALC_WARP_COUNT;  // 0..3
        const uint32_t col_in_half = lane;                          // 0..31
#pragma unroll
        for (uint32_t w = 0; w < WAVES; ++w) {
          // s, w, pf_warp are lane-uniform, so the warp stays in lockstep
          // for the reduce; distinct s per wave ⇒ each token once.
          if (s != w * DEFER_ITERS / WAVES) continue;
          const uint32_t tok = w * PF + pf_warp;

          // Read the PREVIOUS expert's rank snapshot (the current expert's
          // K-loop top already overwrote up_rank_for_tok).  Also filters
          // tokens that don't route to the previous expert.
          bool store_local = false;
          std::uint32_t dest_row_local = 0;
          if (tok < batch_size) {
            const uint8_t k = shm->up_rank_for_tok_prev[tok];
            if (k != 0xFFu) {
              store_local = true;
              dest_row_local = shm->sorted_slot[tok * top_k + k];
            }
          }

          // rw * silu(gate) * up was already baked in by the calc-warp
          // epilogue; rows [0..31] = WG0, [64..95] = WG1.
          const float val1_l = shm->partial_result.post_silu_scratch[col_in_half][tok];
          const float val2_l = shm->partial_result.post_silu_scratch[col_in_half + 64][tok];

          const std::uint32_t out_col_1_l = base_row_up + col_in_half;
          const std::uint32_t out_col_2_l = base_row_up + 32 + col_in_half;
          const bool write1_l = store_local && (out_col_1_l < Dims::N);
          const bool write2_l = store_local && (out_col_2_l < Dims::N);
          float v1 = write1_l ? val1_l : 0.f;
          float v2 = write2_l ? val2_l : 0.f;

          float local_max_l = fmaxf(fabsf(v1), fabsf(v2));
          float block_max_l = warp_reduce_max_float(local_max_l);
          // Eps-clamp tiny maxima: a block_max slightly above FLT_MIN
          // still overflows inv_scale (448/block_max > FLT_MAX for
          // block_max < ~1.32e-36), NaN-ing the whole block after the
          // fp8 cast. 1e-10 matches vLLM's group-quant eps.
          block_max_l = fmaxf(block_max_l, 1e-10f);
          constexpr float FP8_MAX = 448.0f;
          constexpr float FP8_MAX_INV = 1.0f / 448.0f;
          const float block_scale_l = block_max_l * FP8_MAX_INV;
          const float inv_scale_l = FP8_MAX / block_max_l;
          const AQ_element q1_l = (AQ_element)(v1 * inv_scale_l);
          const AQ_element q2_l = (AQ_element)(v2 * inv_scale_l);

          if (store_local && tok < batch_size) {
            // (warp-uniform branch: tok / store_local are per-warp values)
            if (write1_l) {
              spec->temp_fp8[dest_row_local * Dims::N + out_col_1_l] = q1_l;
            }
            if (write2_l) {
              spec->temp_fp8[dest_row_local * Dims::N + out_col_2_l] = q2_l;
            }
            // Sentinel handoff: payload first, then the scale as a
            // release-published readiness flag (syncwarp + fence inside).
            moe_publish_act_scale<Dims>(spec, shmem->scale_parity, dest_row_local, effective_bid,
                                        block_scale_l, lane);
          }
        }
      }

      // Inter-iteration sync: the launcher re-arms a slot SLOTS iters
      // after its previous arm; without this rendezvous the launcher (a
      // single never-waiting thread) could race through all arms before
      // the calc warps consume the matching waits, double-arming a
      // still-pending mbarrier.
      __syncthreads();
    }  // end K-loop

    // ── Calc-warp epilogue: per-lane combine → post_silu_scratch ──
    //
    // Pair layout after the K-loop:
    //   final_d0 = gate(r) tok_even   final_d1 = gate(r) tok_odd
    //   final_d2 = up(r)   tok_even   final_d3 = up(r)   tok_odd
    // with r = warp_in_wg*8 + lane/4 within the WG's 32-row half.
    // Calc warps only combine and store; reduce-max / quantize / GM
    // stores are deferred to the PF body above during the next expert.
    {
      if (is_calc) {
        const std::uint32_t tok_even = (lane % 4) * 2;
        const std::uint32_t tok_odd = tok_even + 1;

        // up_rw_for_tok is 0.0f for unrouted tokens, so the rw value
        // doubles as the route predicate.
        float rw_even = 0.f, rw_odd = 0.f;
        bool store_even = false, store_odd = false;

        if (tok_even < batch_size) {
          rw_even = shm->up_rw_for_tok[tok_even];
          store_even = (rw_even != 0.f);
        }
        if (tok_odd < batch_size) {
          rw_odd = shm->up_rw_for_tok[tok_odd];
          store_odd = (rw_odd != 0.f);
        }

        // silu(gate) * up * rw.  __fdividef (approximate reciprocal) is
        // bit-adequate here — the result is fp8-quantized downstream —
        // and shortens the SFU dependency chain vs the IEEE divide.
        float val0 = __fdividef(rw_even * final_d2 * final_d0, 1.0f + __expf(-final_d0));
        float val1 = __fdividef(rw_odd * final_d3 * final_d1, 1.0f + __expf(-final_d1));

        if (!store_even) val0 = 0.f;
        if (!store_odd) val1 = 0.f;

        const uint32_t row_in_tile = (is_wg1 ? 64u : 0u) + warp_in_wg * 8 + lane / 4;
        shm->partial_result.post_silu_scratch[row_in_tile][tok_even] = val0;
        shm->partial_result.post_silu_scratch[row_in_tile][tok_odd] = val1;

        // Snapshot this expert's ranks for the next expert's PF body (the
        // next K-loop top overwrites up_rank_for_tok).  Published by the
        // inter-expert __syncthreads below.
        if (warp == 0 && lane < Dims::BS) {
          shm->up_rank_for_tok_prev[lane] = shm->up_rank_for_tok[lane];
        }
      }

      has_pending_writeback = true;
    }

    // Publishes post_silu_scratch + the rank snapshot to PF warps, and
    // keeps the launcher from racing ahead of the calc warps' stitch
    // consumption.  The next expert's first weight TMAs were already
    // issued by the in-loop stitch — the mbarrier chain carries across
    // the expert boundary with no extra work here.
    __syncthreads();

    cur_scale_slot ^= 1u;
  }  // end expert loop

  // ── Post-loop drain: the LAST expert has no successor to defer to ─────
  // Same math as the PF body, run inline on calc warps (one token per
  // warp) now that the K-loop is done.
  if (has_pending_writeback && is_calc) {
    // BS<=8: one pass, tok = warp (0..7).
    constexpr uint32_t DRAIN_PASSES = (Dims::BS + 7u) / 8u;  // == 1 for BS<=8
    const uint32_t col_in_half = lane;                       // 0..31
#pragma unroll
    for (uint32_t t_off = 0; t_off < DRAIN_PASSES; ++t_off) {
      const uint32_t tok = warp + t_off * CoreDims::CALC_WARP_COUNT;
      bool store_local = false;
      std::uint32_t dest_row_local = 0;
      if (tok < batch_size) {
        const uint8_t k = shm->up_rank_for_tok_prev[tok];
        if (k != 0xFFu) {
          store_local = true;
          dest_row_local = shm->sorted_slot[tok * top_k + k];
        }
      }

      const float val1_l = shm->partial_result.post_silu_scratch[col_in_half][tok];
      const float val2_l = shm->partial_result.post_silu_scratch[col_in_half + 64][tok];

      const std::uint32_t out_col_1_l = base_row_up + col_in_half;
      const std::uint32_t out_col_2_l = base_row_up + 32 + col_in_half;
      const bool write1_l = store_local && (out_col_1_l < Dims::N);
      const bool write2_l = store_local && (out_col_2_l < Dims::N);
      float v1 = write1_l ? val1_l : 0.f;
      float v2 = write2_l ? val2_l : 0.f;

      float local_max_l = fmaxf(fabsf(v1), fabsf(v2));
      float block_max_l = warp_reduce_max_float(local_max_l);
      // Eps-clamp tiny maxima (overflow-safe inv_scale); see the
      // pair-layout epilogue above for the full rationale.
      block_max_l = fmaxf(block_max_l, 1e-10f);
      constexpr float FP8_MAX = 448.0f;
      constexpr float FP8_MAX_INV = 1.0f / 448.0f;
      const float block_scale_l = block_max_l * FP8_MAX_INV;
      const float inv_scale_l = FP8_MAX / block_max_l;
      const AQ_element q1_l = (AQ_element)(v1 * inv_scale_l);
      const AQ_element q2_l = (AQ_element)(v2 * inv_scale_l);

      if (store_local && tok < batch_size) {
        // (warp-uniform branch: tok is per-warp here)
        if (write1_l) {
          spec->temp_fp8[dest_row_local * Dims::N + out_col_1_l] = q1_l;
        }
        if (write2_l) {
          spec->temp_fp8[dest_row_local * Dims::N + out_col_2_l] = q2_l;
        }
        // Sentinel handoff publish (see moe_publish_act_scale).
        moe_publish_act_scale<Dims>(spec, shmem->scale_parity, dest_row_local, effective_bid,
                                    block_scale_l, lane);
      }
    }
  }
}

}  // namespace monomoe

#endif
