
#pragma once
#ifndef MOE_DOWN_PROJECTION_CU
#define MOE_DOWN_PROJECTION_CU

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
 * @brief Per-expert activation-scale loader (down-projection).
 *
 * Loads ALL T_TILE × (N / DOWN_ACT_BLOCK_SIZE) per-token activation scales
 * for the current expert into `a_down_scale[block][tok]`, once per expert
 * (the K-loop's scale-apply then indexes SHM directly — no per-K-step
 * reload).
 *
 * Rank-indexing invariant: `a_down_scale[block][rank]` must match the token
 * whose fp8 payload the bulk activation TMA fetches from temp_fp8 row
 * `expert_slot_start[id] + rank` — the same row the up-proj epilogue wrote
 * via sorted_slot, alongside temp_act_scale at that row.  Reading scales
 * from that row guarantees payload/scale alignment.
 *
 * GM temp_act_scale is [row][block]; SHM a_down_scale is [block][tok]
 * (bank-conflict-free reads in the WGMMA scale-apply) — this loop
 * transposes on the fly.
 *
 * Runs on prefetch warp `RunPw` only; the caller must __syncthreads()
 * before the K-loop reads a_down_scale.
 */
template <typename Dims, std::size_t ScaleHalves, std::size_t ScaleTok, unsigned RunPw = 0u>
__device__ inline void moe_load_down_wgmma_act_scale_per_expert(
    const MoEGemmSpec<Dims>* __restrict__ spec, const MoE_SHM<Dims>* __restrict__ shmem,
    std::uint32_t id, S_element (&dest_scale)[ScaleHalves][ScaleTok]) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(ScaleTok == CoreDims::T_TILE,
                "down-proj activation scale tile must have T_TILE=8 tokens");
  static_assert(ScaleHalves == Dims::N / CoreDims::DOWN_ACT_BLOCK_SIZE,
                "down-proj activation scale tile must have one block per "
                "up-block across the full reduction dim");

  constexpr unsigned SCALE_COLS = Dims::N / CoreDims::DOWN_ACT_BLOCK_SIZE;

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();

  if (pw != RunPw) return;

  const auto* tma_shm = &shmem->tiny_wgmma_tma;
  const uint32_t routed_count = static_cast<uint32_t>(tma_shm->expert_routed_count[id]);
  const uint32_t expert_start = static_cast<uint32_t>(tma_shm->expert_slot_start[id]);

  // Sentinel handoff (replaces the site-#2 barrier): the up-proj epilogue
  // release-publishes each scale AFTER its fp8 payload segment, so
  // polling a cell until it turns non-zero is the readiness wait for
  // exactly the payload this expert consumes.  Device-scope atomic reads
  // (ld.acquire.gpu on SM90) bypass the non-coherent L1.
  uint32_t* scale_bits = reinterpret_cast<uint32_t*>(
      moe_act_scale_buf<Dims>(const_cast<MoEGemmSpec<Dims>*>(spec), shmem->scale_parity));

  constexpr unsigned TOTAL = (unsigned)(ScaleTok * ScaleHalves);
  for (unsigned i = thread; i < TOTAL; i += 32u) {
    const unsigned slot_row = i / SCALE_COLS;
    const unsigned half = i % SCALE_COLS;

    if (slot_row < routed_count) {
      const uint32_t source_row = expert_start + slot_row;
      uint32_t* cell = scale_bits + source_row * SCALE_COLS + half;
      uint32_t bits = atomicAdd(cell, 0u);
      while (moe_scale_is_sentinel(bits)) {
        bits = atomicAdd(cell, 0u);
      }
      dest_scale[half][slot_row] = __uint_as_float(bits);
    } else {
      // Unused rank — never consumed (the rank-filtered accumulate skips
      // it); zeroed for cleanliness.
      dest_scale[half][slot_row] = 0.f;
    }
  }

  // Acquire for the payload behind the observed flags, then join the
  // warp: this loader runs on the TMA launcher's warp, so after the
  // syncwarp the launcher's activation-TMA issue (later in program
  // order) is gated on EVERY lane's polls, not just its own.
  __threadfence();
  __syncwarp();
}

/**
 * @brief Single-thread readiness wait for one expert's published scales.
 *
 * Sentinel-handoff companion for the inter-expert LOOKAHEAD activation
 * TMA: the launcher issues the NEXT expert's K=0 activation tile while
 * the current expert is still computing, i.e. BEFORE the next expert's
 * loader poll runs — so the launcher must confirm the next expert's
 * payload is published first.  Sweep-polls all (row, half) cells of the
 * expert (independent reads pipeline within a sweep; re-sweeps until no
 * sentinel remains), then acquires.
 *
 * Runs on ONE thread (the TMA launcher).  Weight TMAs need no such gate
 * (weights are static); issue them before this wait to keep DRAM busy.
 */
template <typename Dims>
__device__ inline void moe_wait_expert_scales_published(const MoEGemmSpec<Dims>* __restrict__ spec,
                                                        const MoE_SHM<Dims>* __restrict__ shmem,
                                                        std::uint32_t expert_start,
                                                        std::uint32_t routed_count) {
  constexpr uint32_t SCALE_COLS = Dims::N / MoECoreDims<Dims>::DOWN_ACT_BLOCK_SIZE;
  uint32_t* base = reinterpret_cast<uint32_t*>(
      moe_act_scale_buf<Dims>(const_cast<MoEGemmSpec<Dims>*>(spec), shmem->scale_parity));
  bool pending = true;
  while (pending) {
    pending = false;
    for (uint32_t r = 0; r < routed_count; ++r) {
      for (uint32_t h = 0; h < SCALE_COLS; ++h) {
        uint32_t* cell = base + (expert_start + r) * SCALE_COLS + h;
        if (moe_scale_is_sentinel(atomicAdd(cell, 0u))) {
          pending = true;
        }
      }
    }
  }
  __threadfence();
}

/**
 * @brief Per-expert weight-scale loader (down-projection).
 *
 * Loads the expert's fp32 weight scales for one 128-row M-atom into
 * `dest[wg][col_block]` — one scale per (warpgroup, col-block along N).
 * WG0 covers weight rows [base_col, +64), WG1 rows [base_col+64, +64);
 * for a 128-aligned base_col both fall in the same 128-row scale block,
 * but each is computed independently for generality.
 *
 * Runs on prefetch warp 1 (warp 9) so it issues in parallel with the
 * activation-scale loader on warp 8.  Payload is tiny (≤ 32 B for the
 * E256_N512_K2048 shape) — plain synchronous SHM stores.
 */
template <typename Dims, std::size_t DestRows, std::size_t DestCols>
__device__ inline void moe_load_down_wgmma_weight_scale_tile(
    const S_element* __restrict__ expert_scales_down, std::uint32_t id, std::uint32_t base_col,
    S_element (&dest)[DestRows][DestCols]) {
  static_assert(DestRows == 2, "down-proj weight-scale tile must have 2 row-blocks (WG0, WG1)");

  constexpr uint32_t COLS = DestCols;

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();

  if (pw == 1 && thread < DestRows * COLS) {
    const uint32_t wg = thread / COLS;
    const uint32_t cb = thread % COLS;
    const uint32_t weight_row = base_col + (wg == 0 ? 0u : 64u);
    const uint32_t rb = weight_row / 128;
    dest[wg][cb] = expert_scales_down[id * Dims::DOWN_SCALE_ROWS * Dims::DOWN_SCALE_COLS +
                                      rb * Dims::DOWN_SCALE_COLS + cb];
  }
}

/**
 * @brief TMA + WGMMA down-projection for BS <= 8.
 *
 * See docs/design_docs/monomoe_kernel.md "Phase 4 — down-projection".  Both the fp8 weight tile and
 * the fp8 intermediate-activation tile are loaded via bulk TMA (SWZ128),
 * issued by the TMA launcher thread with completion on bar_w / bar_a
 * (2-slot double buffers, re-initialized in this function's prologue).
 * Each block accumulates its DOWN_COL_TILE output cols across its expert
 * group's experts and atomicAdds the result into
 * `spec->down_partial_out[BS][HIDDEN_STATES]`.
 */
template <typename Dims>
__device__ inline void moe_down_projection_allexperts_wgmma_tma(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t top_k, std::uint32_t batch_size,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& down_weights_desc, CUtensorMap const& down_activations_desc) {
  static_assert(Dims::BS <= 8, "moe_down_projection_allexperts_wgmma_tma supports BS<=8.");
  using CoreDims = MoECoreDims<Dims>;

  // All weight GM reads go through the TMA descriptor; top_k's rank scan
  // was replaced by the routing-recorded down_rank table.  Both stay on
  // the signature for parity with the up-proj helpers.
  (void)expert_weights_down;
  (void)top_k;

  constexpr std::uint32_t K_TILE_W = CoreDims::K_TILE_WGMMA;  // 32
  constexpr std::uint32_t K_STEP_DOWN = CoreDims::K_STEP_DOWN;
  constexpr std::uint32_t K_STEP_WGMMA = CoreDims::K_STEP_WGMMA;  // 128
  constexpr std::uint32_t K_SUBSTEPS_DOWN = CoreDims::K_SUBSTEPS_DOWN;
  constexpr std::uint32_t WGMMAS_PER_SUBSTEP = K_STEP_WGMMA / K_TILE_W;  // 4
  constexpr std::uint32_t K_TILES_DOWN = Dims::N / K_STEP_DOWN;
  constexpr std::uint32_t DOWN_COL_TILE = CoreDims::DOWN_COL_TILE;
  constexpr std::uint32_t DOWN_GRID = CoreDims::DOWN_GRID;
  constexpr std::uint32_t DOWN_GROUPS = CoreDims::DOWN_GROUPS;
  // The two WGs cover 128 output cols per WGMMA pass, so a block runs
  // DOWN_COL_HALVES sequential passes per K-step to cover its
  // DOWN_COL_TILE cols.
  constexpr std::uint32_t DOWN_COL_HALVES = DOWN_COL_TILE / 128u;
  static_assert(DOWN_COL_TILE % 128u == 0,
                "DOWN_COL_TILE must be a multiple of 128 for the TMA+WGMMA "
                "down-projection (the weight tile is laid out as 128-row "
                "SWZ128 atoms on the M axis).");
  static_assert(DOWN_COL_HALVES <= 4u,
                "DOWN_COL_TILE > 512 not supported by the WGMMA down-proj "
                "M-axis structure.");
  // Per-K-step TMA transfer sizes: one 128×128 fp8 weight atom is 16 KB,
  // issued DOWN_COL_HALVES × K_SUBSTEPS_DOWN times per step; one
  // activation atom is 1024 B (8 tok × 128 K), issued K_SUBSTEPS_DOWN
  // times.  Each bar is armed once with the total.
  constexpr std::uint32_t DOWN_W_TX_BYTES_PER_HALF = 16384u;
  constexpr std::uint32_t DOWN_W_TX_BYTES_TOTAL =
      DOWN_W_TX_BYTES_PER_HALF * DOWN_COL_HALVES * K_SUBSTEPS_DOWN;
  // Activation atom bytes scale with T_TILE (8 tok × 128 K = 1024 B on
  // the BS8 path).
  constexpr std::uint32_t DOWN_A_TX_BYTES_TOTAL =
      CoreDims::T_TILE * MoE_SHM<Dims>::TinyDataWGMMA_TMA::FP8_ACT_NUM_CHUNKS *
      MoE_SHM<Dims>::TinyDataWGMMA_TMA::FP8_ACT_K_CHUNK * K_SUBSTEPS_DOWN;
  constexpr std::uint32_t W_DOWN_SCALE_COLS = MoE_SHM<Dims>::TinyDataWGMMA_TMA::W_DOWN_SCALE_COLS;

  static_assert(Dims::N % K_STEP_DOWN == 0, "Dims::N must be a multiple of K_STEP_DOWN");
  static_assert(K_STEP_DOWN % K_STEP_WGMMA == 0,
                "K_STEP_DOWN must be a multiple of K_STEP_WGMMA (=128, the "
                "SWZ128 atom K-width)");

  // A operand (weights): SWZ128 canonical Major::K layout produced by the
  // TMA from the RAW row-major [E, K, N] tensor (no pre-interleave).
  constexpr std::uint64_t A_LBO = 16ULL;
  constexpr std::uint64_t A_SBO = 1024ULL;
  constexpr std::uint32_t A_SWIZZLE = 1u;
  // B operand (activations): a single 8-token × 128-K SWZ128 atom.
  // BS8 (m64n8k32) has a single atom so SBO is unused; the historical
  // 128 B value is preserved to keep BS8 SASS byte-identical.
  constexpr std::uint64_t B_LBO = 16ULL;
  constexpr std::uint64_t B_SBO = 128ULL;
  constexpr std::uint32_t B_SWIZZLE = 1u;

  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;
  const unsigned lane = thread_in_block & 31;
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;
  const unsigned my_wg = is_wg1 ? 1u : 0u;

  auto* shm = &shmem->tiny_wgmma_tma;

  // Grid-to-(expert-group, output-col-tile) mapping.
  const std::uint32_t down_group = blockIdx.x / DOWN_GRID;
  const std::uint32_t down_block_idx = blockIdx.x % DOWN_GRID;
  const std::uint32_t base_col = down_block_idx * DOWN_COL_TILE;

  // Per-thread accumulators: per M-half, the 4-WGMMA K-chain splits each
  // 128-K substep into a "lo" chunk (K[0..63]) and "hi" chunk (K[64..127])
  // so the two 64-K activation-scale halves can be applied separately.
  float chunk_d_lo[DOWN_COL_HALVES][4] = {{0.f}};
  float chunk_d_hi[DOWN_COL_HALVES][4] = {{0.f}};
  float final_d[DOWN_COL_HALVES][4] = {{0.f}};

  // ── Prologue: zero out_accum + re-init the Phase-4 mbarriers ──────────
  // The site-#2 barrier guarantees the barriers are idle at entry.  The
  // two targets are disjoint SHM, so one trailing sync publishes both
  // (issued at the top of the first expert iteration below).
  for (unsigned idx = thread_in_block; idx < Dims::BS * DOWN_COL_TILE; idx += blockDim.x) {
    const unsigned tok = idx / DOWN_COL_TILE;
    const unsigned col = idx % DOWN_COL_TILE;
    shm->out_accum[tok][col] = 0.f;
  }
  constexpr std::uint32_t DOWN_PIPE_DEPTH = CoreDims::DOWN_PIPE_DEPTH;
  if (is_tma_launcher_thread<Dims>()) {
#pragma unroll
    for (std::uint32_t i = 0; i < DOWN_PIPE_DEPTH; ++i) {
      mbarrier_init(&shm->bar_w[i], 1u);
      mbarrier_init(&shm->bar_a[i], 1u);
    }
    fence_mbarrier_init_release_cluster();
  }
  __syncthreads();

  const std::uint32_t expert_count = shmem->expert_count;

  // Parity state is hoisted OUT of the expert loop: for K_TILES_DOWN ==
  // DOWN_PIPE_DEPTH each slot is visited once per expert and ends at
  // phase 1, so a blind per-expert reset to 0 would deadlock the next
  // expert's wait.  Keeping it continuous works for any K_TILES_DOWN.
  std::uint32_t parity_w[DOWN_PIPE_DEPTH] = {};
  std::uint32_t parity_a[DOWN_PIPE_DEPTH] = {};

  // ── Per-expert loop (start = down_group, stride = DOWN_GROUPS) ────────
  for (std::uint32_t e = down_group; e < expert_count; e += DOWN_GROUPS) {
    const std::uint32_t id = shmem->experts[e].id;

    const std::uint32_t routed_count = static_cast<std::uint32_t>(shm->expert_routed_count[id]);
    const std::uint32_t expert_start = static_cast<std::uint32_t>(shm->expert_slot_start[id]);

#pragma unroll
    for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
#pragma unroll
      for (std::uint32_t r = 0; r < 4u; ++r) {
        final_d[h][r] = 0.f;
      }
    }

    // Load weight scales (warp 9) + per-expert activation scales (warp 8)
    // in parallel; both are fixed across the expert's K-steps.
    if (is_prefetch_warp<Dims>()) {
#pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        moe_load_down_wgmma_weight_scale_tile<Dims>(expert_scales_down, id, base_col + h * 128u,
                                                    shm->w_down_scale[h]);
      }
      moe_load_down_wgmma_act_scale_per_expert<Dims, Dims::N / CoreDims::DOWN_ACT_BLOCK_SIZE,
                                               CoreDims::T_TILE,
                                               /*RunPw=*/0u>(spec, shmem, id, shm->a_down_scale);
    }

    // ── Priming: K-step 0 weight + activation TMAs ────────────────────
    // First expert only — for later experts the previous expert's
    // last-K-step launcher already issued the inter-expert lookahead into
    // slot 0 (see the launcher branch in the K-loop).
    //
    // The weight descriptor's row box is pinned to 128 host-side (the TMA
    // boxDim cap is 256 and DOWN_COL_TILE=384 exceeds it), so one 128-row
    // TMA is issued per (kk-substep, h-half) atom, landing at SHM row
    // (kk*DOWN_COL_HALVES + h)*128.  All atoms retire into one bar_w arm.
    const bool need_first_expert_prime = (e == down_group);
    if (need_first_expert_prime && is_tma_launcher_thread<Dims>()) {
      // 2-deep: prime K-step 0 only (the in-loop launcher covers s+1).
      // 4-deep: prime K-steps 0 AND 1 into ring slots 0,1 — the in-loop
      // launcher then arms slot (s+2)&3 each iter, so it fills slots 2,3
      // at s=0,1 and cross-stitches the NEXT expert into slots 0,1 at the
      // last two iters.
      constexpr std::uint32_t PRIME_STEPS = (DOWN_PIPE_DEPTH == 4u) ? 2u : 1u;
#pragma unroll
      for (std::uint32_t ps = 0; ps < PRIME_STEPS; ++ps) {
        const std::uint32_t prime_k = ps * K_STEP_DOWN;
        mbarrier_arrive_expect_tx(&shm->bar_w[ps],
                                  /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
#pragma unroll
        for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
#pragma unroll
          for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
            W_element* dest_base = &shm->w_down_wgmma[ps][(kk * DOWN_COL_HALVES + h) * 128u][0];
            tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/id,
                                     /*K=*/Dims::HIDDEN_STATES,
                                     /*base_col=*/base_col + h * 128u,
                                     /*k_start=*/prime_k + kk * K_STEP_WGMMA,
                                     /*dest_smem_ptr=*/(void*)dest_base,
                                     /*bar_smem_ptr=*/&shm->bar_w[ps]);
          }
        }

        // Activation tile — only when tokens route to this expert.  With
        // routed_count == 0 nothing is armed; the WGMMA computes on
        // garbage that the rank-filtered accumulate never reads (fp8 e4m3
        // has no NaN encoding, so garbage can't fault).
        if (routed_count > 0u) {
          mbarrier_arrive_expect_tx(&shm->bar_a[ps],
                                    /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
#pragma unroll
          for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
            tma_load_down_wgmma_activation_bulk(
                down_activations_desc,
                /*k_start=*/prime_k + kk * K_STEP_WGMMA,
                /*expert_slot_start=*/expert_start,
                /*dest_smem_ptr=*/&shm->a_down_wgmma[ps][kk][0][0][0],
                /*bar_smem_ptr=*/&shm->bar_a[ps]);
          }
        }
      }
    }

    // Publish the prologue zero-fill + mbarrier inits and this expert's
    // scale tiles before the WGMMA consumers wait on the barriers.
    __syncthreads();

    // ── Main K-loop ─────────────────────────────────────────────────────
    // Per step: calc warps wait bar_w/bar_a and run the WGMMA chain; the
    // launcher concurrently prefetches step s+1 (or the next expert's
    // step 0 at the last step); prefetch warps run the PREVIOUS expert's
    // deferred accumulate, sliced across the first K_TILES_DOWN - 1 steps.
    const bool has_prev_expert = (e != down_group);
    const std::uint32_t prev_id = has_prev_expert ? shmem->experts[e - DOWN_GROUPS].id : 0u;
    for (std::uint32_t s = 0; s < K_TILES_DOWN; ++s) {
      const std::uint32_t read_slot = s & (DOWN_PIPE_DEPTH - 1u);

      if (is_calc) {
        while (!mbarrier_try_wait_parity(&shm->bar_w[read_slot], parity_w[read_slot])) {
        }
        parity_w[read_slot] ^= 1;

        if (routed_count > 0u) {
          while (!mbarrier_try_wait_parity(&shm->bar_a[read_slot], parity_a[read_slot])) {
          }
          parity_a[read_slot] ^= 1;
        }

        const void* a_slot_base = (const void*)&shm->w_down_wgmma[read_slot][0][0];
        const std::uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;
        constexpr std::uint32_t W_ATOM_BYTES = DOWN_W_TX_BYTES_PER_HALF;

        constexpr std::uint32_t A_K_STRIDE = 2u * A_LBO;
        constexpr std::uint32_t B_K_STRIDE = 2u * B_LBO;
        const void* b_slot_base = (const void*)&shm->a_down_wgmma[read_slot][0][0][0][0];
        // Bytes per K-substep activation atom: T_TILE rows × 128 B/row
        // (1024 B at BS<=8, 2048 B at BS=16).
        constexpr std::uint32_t B_SUBSTEP_BYTES = CoreDims::T_TILE * 128u;

#pragma unroll
        for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
          // One fence per commit/wait pair; hoisting it out of the kk loop
          // would merge the substeps into one WGMMA supergroup without the
          // intermediate scale-apply ordering — incorrect.
          wgmma_fence();

          const void* b_kk_base = (const void*)((const char*)b_slot_base + kk * B_SUBSTEP_BYTES);

#pragma unroll
          for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
            const std::uint32_t atom_off_bytes = (kk * DOWN_COL_HALVES + h) * W_ATOM_BYTES;
            const void* a_base =
                (const void*)((const char*)a_slot_base + atom_off_bytes + wg_offset_bytes);

            // 2 chained WGMMAs into the lo chunk (K[0..63])...
#pragma unroll
            for (std::uint32_t j = 0; j < 2; ++j) {
              const void* a_ptr = (const void*)((const char*)a_base + j * A_K_STRIDE);
              const void* b_ptr = (const void*)((const char*)b_kk_base + j * B_K_STRIDE);
              std::uint64_t desc_a = make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
              std::uint64_t desc_b = make_wgmma_desc(b_ptr, B_LBO, B_SBO, B_SWIZZLE);
              wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d_lo[h][0], chunk_d_lo[h][1],
                                           chunk_d_lo[h][2], chunk_d_lo[h][3]);
            }

            // ...and 2 into the hi chunk (K[64..127]).
#pragma unroll
            for (std::uint32_t j = 2; j < WGMMAS_PER_SUBSTEP; ++j) {
              const void* a_ptr = (const void*)((const char*)a_base + j * A_K_STRIDE);
              const void* b_ptr = (const void*)((const char*)b_kk_base + j * B_K_STRIDE);
              std::uint64_t desc_a = make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
              std::uint64_t desc_b = make_wgmma_desc(b_ptr, B_LBO, B_SBO, B_SWIZZLE);
              wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d_hi[h][0], chunk_d_hi[h][1],
                                           chunk_d_hi[h][2], chunk_d_hi[h][3]);
            }
          }

          wgmma_commit_group();
          wgmma_wait_group<0>();

          // ── Scale-apply at the K=128 boundary (per substep, per half) ─
          // The lo/hi chunks map to the activation block covering their
          // global-K start; for DOWN_ACT_BLOCK_SIZE=64 hi = lo + 1, for
          // 128 lo and hi share a block.
          const std::uint32_t tok_02 = (lane % 4) * 2;
          const std::uint32_t tok_13 = (lane % 4) * 2 + 1;

          constexpr std::uint32_t ACT_BLK = CoreDims::DOWN_ACT_BLOCK_SIZE;
          const std::uint32_t global_k = (s * K_SUBSTEPS_DOWN + kk) * 128u;
          const std::uint32_t block_lo = global_k / ACT_BLK;
          const std::uint32_t block_hi = (global_k + 64u) / ACT_BLK;
          const float as_lo_02 = shm->a_down_scale[block_lo][tok_02];
          const float as_hi_02 = shm->a_down_scale[block_hi][tok_02];
          const float as_lo_13 = shm->a_down_scale[block_lo][tok_13];
          const float as_hi_13 = shm->a_down_scale[block_hi][tok_13];

          // Global 128-K block index along Dims::N for (s, kk).
          const std::uint32_t ws_col = (W_DOWN_SCALE_COLS > 1) ? (s * K_SUBSTEPS_DOWN + kk) : 0u;

#pragma unroll
          for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
            const float ws = shm->w_down_scale[h][my_wg][ws_col];
            final_d[h][0] += chunk_d_lo[h][0] * as_lo_02 * ws + chunk_d_hi[h][0] * as_hi_02 * ws;
            final_d[h][1] += chunk_d_lo[h][1] * as_lo_13 * ws + chunk_d_hi[h][1] * as_hi_13 * ws;
            final_d[h][2] += chunk_d_lo[h][2] * as_lo_02 * ws + chunk_d_hi[h][2] * as_hi_02 * ws;
            final_d[h][3] += chunk_d_lo[h][3] * as_lo_13 * ws + chunk_d_hi[h][3] * as_hi_13 * ws;
            chunk_d_lo[h][0] = chunk_d_lo[h][1] = chunk_d_lo[h][2] = chunk_d_lo[h][3] = 0.f;
            chunk_d_hi[h][0] = chunk_d_hi[h][1] = chunk_d_hi[h][2] = chunk_d_hi[h][3] = 0.f;
          }
        }  // end kk substep loop
      }

      // Launcher (in parallel with the WGMMAs above):
      //   (a) s+1 < K_TILES_DOWN — intra-expert prefetch of the next
      //       K-step into the other slot;
      //   (b) last step + next expert exists — INTER-EXPERT LOOKAHEAD:
      //       prefetch the next expert's K=0 tiles into the slot about to
      //       be freed, keeping DRAM busy through the writeback/accumulate
      //       phase.  With K_TILES_DOWN even that slot is index 0 — the
      //       slot the next expert's s=0 wait reads, with parity already
      //       aligned via the hoisted parity state.
      static_assert(K_TILES_DOWN % 2u == 0u,
                    "Inter-expert lookahead requires K_TILES_DOWN to be even "
                    "so the slot freed at the last step is the slot the next "
                    "expert's s=0 wait reads.  If this fires for a new shape, "
                    "either disable the lookahead branch or introduce a "
                    "per-expert slot offset.");
      if (is_tma_launcher_thread<Dims>()) {
        if constexpr (DOWN_PIPE_DEPTH == 4u) {
          // ── 4-deep rolling ring, arm distance 2 ─────────────────────
          // At iter s arm ring slot (s+2)&3 for the tile its consumer
          // reads 2 iters later, giving each weight TMA two compute
          // windows to land.  Two source cases:
          //   (A) intra-expert: s+2 < K_TILES_DOWN → CURRENT expert's
          //       tile s+2 at k=(s+2)*K_STEP_DOWN.
          //   (B) cross-expert: s+2 >= K_TILES_DOWN → NEXT expert's
          //       tile s+2-K_TILES_DOWN (0..1) into slot (s+2)&3, which
          //       equals the tile index (K_TILES_DOWN % 4 == 0 —
          //       asserted in MoECoreDims), i.e. exactly the slot the
          //       next expert's early waits read.
          // (Arm distance 3 — stitching a THIRD next-expert tile across
          // the boundary — was measured 2026-07-15 after the boundary-
          // shrink change and was flat on BS16 and ~5% slower on the
          // BS8 tunable path; the shrunken boundary no longer needs the
          // extra byte coverage, and the deeper stitch adds launcher
          // overhead.  Keep distance 2.)
          const std::uint32_t la_tile = s + 2u;
          const std::uint32_t la_slot = la_tile & 3u;
          if (la_tile < K_TILES_DOWN) {
            // (A) intra-expert tile s+2 of the CURRENT expert.
            const std::uint32_t nk = la_tile * K_STEP_DOWN;
            mbarrier_arrive_expect_tx(&shm->bar_w[la_slot],
                                      /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
#pragma unroll
            for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
#pragma unroll
              for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
                W_element* dest_base =
                    &shm->w_down_wgmma[la_slot][(kk * DOWN_COL_HALVES + h) * 128u][0];
                tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/id,
                                         /*K=*/Dims::HIDDEN_STATES,
                                         /*base_col=*/base_col + h * 128u,
                                         /*k_start=*/nk + kk * K_STEP_WGMMA,
                                         /*dest_smem_ptr=*/(void*)dest_base,
                                         /*bar_smem_ptr=*/&shm->bar_w[la_slot]);
              }
            }
            if (routed_count > 0u) {
              mbarrier_arrive_expect_tx(&shm->bar_a[la_slot],
                                        /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
#pragma unroll
              for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
                tma_load_down_wgmma_activation_bulk(down_activations_desc,
                                                    /*k_start=*/nk + kk * K_STEP_WGMMA,
                                                    /*expert_slot_start=*/expert_start,
                                                    /*dest_smem_ptr=*/
                                                    &shm->a_down_wgmma[la_slot][kk][0][0][0],
                                                    /*bar_smem_ptr=*/&shm->bar_a[la_slot]);
              }
            }
          } else if (e + DOWN_GROUPS < expert_count) {
            // (B) cross-expert stitch: NEXT expert's tile 0 or 1.
            const std::uint32_t next_e = e + DOWN_GROUPS;
            const std::uint32_t next_id = shmem->experts[next_e].id;
            const std::uint32_t next_routed_count =
                static_cast<std::uint32_t>(shm->expert_routed_count[next_id]);
            const std::uint32_t next_expert_start =
                static_cast<std::uint32_t>(shm->expert_slot_start[next_id]);
            const std::uint32_t next_tile = la_tile - K_TILES_DOWN;  // 0..1
            const std::uint32_t nk = next_tile * K_STEP_DOWN;
            mbarrier_arrive_expect_tx(&shm->bar_w[la_slot],
                                      /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
#pragma unroll
            for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
#pragma unroll
              for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
                W_element* dest_base =
                    &shm->w_down_wgmma[la_slot][(kk * DOWN_COL_HALVES + h) * 128u][0];
                tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/next_id,
                                         /*K=*/Dims::HIDDEN_STATES,
                                         /*base_col=*/base_col + h * 128u,
                                         /*k_start=*/nk + kk * K_STEP_WGMMA,
                                         /*dest_smem_ptr=*/(void*)dest_base,
                                         /*bar_smem_ptr=*/&shm->bar_w[la_slot]);
              }
            }
            // Sentinel-gate + issue the next expert's activation tiles
            // here.  After the first successful poll (case B) the case-C
            // poll returns immediately — the sentinel is monotone within
            // a launch.
            if (next_routed_count > 0u) {
              moe_wait_expert_scales_published<Dims>(spec, shmem, next_expert_start,
                                                     next_routed_count);
              mbarrier_arrive_expect_tx(&shm->bar_a[la_slot],
                                        /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
#pragma unroll
              for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
                tma_load_down_wgmma_activation_bulk(down_activations_desc,
                                                    /*k_start=*/nk + kk * K_STEP_WGMMA,
                                                    /*expert_slot_start=*/next_expert_start,
                                                    /*dest_smem_ptr=*/
                                                    &shm->a_down_wgmma[la_slot][kk][0][0][0],
                                                    /*bar_smem_ptr=*/&shm->bar_a[la_slot]);
              }
            }
          }
        } else if (s + 1 < K_TILES_DOWN) {
          const std::uint32_t next_slot = (s + 1) & 1;
          const std::uint32_t next_k_start = (s + 1) * K_STEP_DOWN;

          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
#pragma unroll
          for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
#pragma unroll
            for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
              W_element* dest_base =
                  &shm->w_down_wgmma[next_slot][(kk * DOWN_COL_HALVES + h) * 128u][0];
              tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/id,
                                       /*K=*/Dims::HIDDEN_STATES,
                                       /*base_col=*/base_col + h * 128u,
                                       /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                                       /*dest_smem_ptr=*/(void*)dest_base,
                                       /*bar_smem_ptr=*/&shm->bar_w[next_slot]);
            }
          }

          if (routed_count > 0u) {
            mbarrier_arrive_expect_tx(&shm->bar_a[next_slot],
                                      /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
#pragma unroll
            for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
              tma_load_down_wgmma_activation_bulk(down_activations_desc,
                                                  /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                                                  /*expert_slot_start=*/expert_start,
                                                  /*dest_smem_ptr=*/
                                                  &shm->a_down_wgmma[next_slot][kk][0][0][0],
                                                  /*bar_smem_ptr=*/&shm->bar_a[next_slot]);
            }
          }
        } else if (s + 1 >= K_TILES_DOWN && e + DOWN_GROUPS < expert_count) {
          const std::uint32_t lookahead_slot = (s + 1) & 1;  // == 0
          const std::uint32_t next_e = e + DOWN_GROUPS;
          const std::uint32_t next_id = shmem->experts[next_e].id;

          mbarrier_arrive_expect_tx(&shm->bar_w[lookahead_slot],
                                    /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
#pragma unroll
          for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
#pragma unroll
            for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
              W_element* dest_base =
                  &shm->w_down_wgmma[lookahead_slot][(kk * DOWN_COL_HALVES + h) * 128u][0];
              tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/next_id,
                                       /*K=*/Dims::HIDDEN_STATES,
                                       /*base_col=*/base_col + h * 128u,
                                       /*k_start=*/kk * K_STEP_WGMMA,
                                       /*dest_smem_ptr=*/(void*)dest_base,
                                       /*bar_smem_ptr=*/&shm->bar_w[lookahead_slot]);
            }
          }

          // Sentinel-gate + issue the next expert's activation tile here.
          // The next expert's fp8 payload may still be in flight from its
          // up-group — confirm publication before the activation TMA reads
          // it.  (Weight TMAs above are static data and issue without the
          // gate.)
          const std::uint32_t next_routed_count =
              static_cast<std::uint32_t>(shm->expert_routed_count[next_id]);
          const std::uint32_t next_expert_start =
              static_cast<std::uint32_t>(shm->expert_slot_start[next_id]);
          if (next_routed_count > 0u) {
            moe_wait_expert_scales_published<Dims>(spec, shmem, next_expert_start,
                                                   next_routed_count);
            mbarrier_arrive_expect_tx(&shm->bar_a[lookahead_slot],
                                      /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
#pragma unroll
            for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
              tma_load_down_wgmma_activation_bulk(down_activations_desc,
                                                  /*k_start=*/kk * K_STEP_WGMMA,
                                                  /*expert_slot_start=*/next_expert_start,
                                                  /*dest_smem_ptr=*/
                                                  &shm->a_down_wgmma[lookahead_slot][kk][0][0][0],
                                                  /*bar_smem_ptr=*/&shm->bar_a[lookahead_slot]);
            }
          }
        }
      }

      // ── Deferred accumulate of the PREVIOUS expert (prefetch warps) ──
      //
      // out_accum[tok][col] += down_out[col][rank] for the previous
      // expert, sliced into K_TILES_DOWN - 1 disjoint (tok, col) ranges —
      // one per K-step — so on many-K-step shapes the accumulate hides
      // behind each step's WGMMA window instead of spilling past a single
      // one.  The LAST step carries no accumulate: its trailing
      // __syncthreads() then cleanly separates the final down_out READ
      // from this expert's epilogue WRITE of the same buffer (and keeps
      // PF SHM traffic off the inter-expert lookahead step).
      //
      // Safety: down_out is stable across the whole K-loop (written only
      // by the previous expert's epilogue, published by the expert-top
      // sync); down_rank is routing-recorded and immutable in Phase 4;
      // slices are disjoint so += never double-adds.
      if (is_prefetch_warp<Dims>()) {
        constexpr unsigned NUM_ACC_STEPS = (K_TILES_DOWN > 1u) ? (K_TILES_DOWN - 1u) : 1u;
        if (has_prev_expert && s < NUM_ACC_STEPS) {
          const unsigned acc_slice = s;
          const unsigned thread_in_pf =
              thread_in_block - CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
          constexpr unsigned PF_THREADS =
              CoreDims::PREFETCH_WARP_COUNT * CoreDims::THREADS_PER_WARP;
          const uint8_t* prev_rank = shm->down_rank[prev_id];
          // Slice s covers [floor(s*total/N), floor((s+1)*total/N)):
          // adjacent, disjoint, covers [0, total) for any runtime total.
          const unsigned total = batch_size * DOWN_COL_TILE;
          const unsigned start = (acc_slice * total) / NUM_ACC_STEPS;
          const unsigned end = ((acc_slice + 1u) * total) / NUM_ACC_STEPS;
          for (unsigned tok_col = start + thread_in_pf; tok_col < end; tok_col += PF_THREADS) {
            const unsigned tok = tok_col / DOWN_COL_TILE;
            const unsigned col = tok_col % DOWN_COL_TILE;
            const uint8_t rank_u8 = prev_rank[tok];
            if (rank_u8 != 0xFFu) {
              shm->out_accum[tok][col] += shm->partial_result.down_out[col][rank_u8];
            }
          }
        }
      }

      // Align all warps before the next step's WGMMA reads the new slot.
      __syncthreads();
    }  // end K-loop

    // ── End-of-expert: final_d → down_out[DOWN_COL_TILE][8] ─────────────
    // Half h covers output cols [base_col + h*128, +128); within a half,
    // WG1 owns the upper 64 cols.
    if (is_calc) {
      const std::uint32_t wg_row_offset = is_wg1 ? 64u : 0u;
      const std::uint32_t col_base = (lane % 4) * 2;
#pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        const std::uint32_t row_base = h * 128u + wg_row_offset + warp_in_wg * 16 + lane / 4;
        shm->partial_result.down_out[row_base + 0][col_base + 0] = final_d[h][0];
        shm->partial_result.down_out[row_base + 0][col_base + 1] = final_d[h][1];
        shm->partial_result.down_out[row_base + 8][col_base + 0] = final_d[h][2];
        shm->partial_result.down_out[row_base + 8][col_base + 1] = final_d[h][3];
      }
    }

    // Publishes this expert's down_out to the next expert's deferred
    // accumulate (staged path); also orders this expert's out_accum RMWs
    // before the next expert's (different threads can target the same
    // (tok, col) across experts).
    __syncthreads();

  }  // end expert loop

  // ── Final accumulate for the LAST expert in this block's group ────────
  // The deferred accumulate covers all but the last visited expert; drain
  // it here with all warps (the K-loop is done, PF warps are free).  The
  // expert loop's trailing sync already published down_out.  Blocks whose
  // group had zero experts skip (down_out was never written).
  if (expert_count > down_group) {
    const std::uint32_t e_last =
        down_group + ((expert_count - 1u - down_group) / DOWN_GROUPS) * DOWN_GROUPS;
    const std::uint32_t last_id = shmem->experts[e_last].id;
    const uint8_t* last_rank = shm->down_rank[last_id];
    for (unsigned tok_col = thread_in_block; tok_col < batch_size * DOWN_COL_TILE;
         tok_col += blockDim.x) {
      const unsigned tok = tok_col / DOWN_COL_TILE;
      const unsigned col = tok_col % DOWN_COL_TILE;
      const uint8_t rank_u8 = last_rank[tok];
      if (rank_u8 != 0xFFu) {
        shm->out_accum[tok][col] += shm->partial_result.down_out[col][rank_u8];
      }
    }
  }
  __syncthreads();

  // ── atomicAdd out_accum into the global single-buffer accumulator ─────
  // Phase 5 reads each cell once (no cross-group reduction); the
  // col-stripe barrier gates it on all contributing blocks finishing.
  float* gm_partial = spec->down_partial_out;
  for (unsigned idx = thread_in_block; idx < batch_size * DOWN_COL_TILE; idx += blockDim.x) {
    const unsigned tok = idx / DOWN_COL_TILE;
    const unsigned col = idx % DOWN_COL_TILE;
    atomicAdd(gm_partial + tok * Dims::HIDDEN_STATES + base_col + col, shm->out_accum[tok][col]);
  }
}

}  // namespace monomoe

#endif
