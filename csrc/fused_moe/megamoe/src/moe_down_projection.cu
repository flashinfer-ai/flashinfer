
#pragma once
#ifndef MOE_DOWN_PROJECTION_CU
  #define MOE_DOWN_PROJECTION_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cuda.h>
  #include <cuda/pipeline>
  #include <cuda_fp8.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "ptx_utils.h"
  #include "moe_debug.h"
  #include "moe_tma.h"

///////////////////////////////////////////////////////////////////////////////
//
// This file holds two down-projection paths sharing the same file:
//
//   * moe_down_projection_topk             (BS > 8, scalar / mma.sync,
//                                            cp.async-based)
//   * moe_down_projection_BS8_allexperts_wgmma_tma
//                                          (BS <= 8, WGMMA + TMA + SWZ128)
//
// Design notes for the BS8 TMA+WGMMA path:
//
// * Each block owns DOWN_COL_TILE = 256 output cols of one expert.  WG0
//   covers M-rows [0..63] and WG1 covers [64..127] of each 128-row
//   M-half via m64n8k32 WGMMAs.
//
// * Weight + activation tiles are loaded via cp.async.bulk.tensor.2d
//   (TMA) into a double-buffered SHM slot and signalled through
//   mbarriers.  The K-loop runs K_TILES_DOWN outer iterations of
//   K_STEP_DOWN K-bytes each (= 4 inner 128-K substeps with separate
//   scale-apply at every K=128 boundary for block-wise FP8).
//
// * Expert transitions overlap via inter-expert lookahead (the last
//   K-step of expert e issues TMAs for expert e+DOWN_GROUPS into the
//   slot that's about to be freed).
//
// * Weight + activation scales are tiny (≤ 256 B per expert) and
//   loaded once per expert from prefetch warps 8 / 9 in parallel —
//   no per-K-step scale fetch.
//
// * Phase-4 epilogue atomicAdds the per-block fp32 partial sum into
//   a single-buffer `spec->down_partial_out[BS][HIDDEN_STATES]`;
//   Phase 5 reads each cell once and casts to bf16.  No cross-group
//   reduction.
//
///////////////////////////////////////////////////////////////////////////////

namespace moe_monokernel {

/**
 * @brief Initiate the copy of expert weights and scales from Global to Shared
 * Memory
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * expert weights and corresponding weights. The copy operations will be queued
 * in the given @a pipe, which the caller must use to wait for their completion.
 *
 * While the expert is selected by @a id, the tile to copy is implicitly
 * selected by the @c blockIdx. The result is stored in the tile @a w_index
 * within @a shm.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @param expert_weights_down Pointer weights array of shape [NUM_EXPERTS,
 * HIDDEN_STATES, N] in expert, row-major order. Individual elements are in
 * __nv_fp8_e4m3 format. Stored in Global Memory.
 * @param expert_scales_down Pointer scales array of shape [NUM_EXPERTS,
 * HIDDEN_STATES] in row-major order. Individual elements are in __nv_fp8_e4m3
 * format. Stored in Global Memory.
 * @param id Expert index within @a expert_weights_down and @a
 * expert_scales_down.
 * @param shm Shared Memory struct to store the result to.
 * @param w_index Index of tile to use within @a shm.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims>
__device__ inline void moe_request_down_expert(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t id,
    typename MoE_SHM<Dims>::U::Gemm2Data* shm, std::uint32_t w_index,
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  const unsigned base_row = blockIdx.x * MoECoreDims<Dims>::W_DOWN_TILE;

  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned d_thread = threadIdx.x % (2 * CoreDims::THREADS_PER_WARP);
  const unsigned d_warp = get_prefetch_warp<Dims>() / 2;

  // request W tile
  {
    const unsigned chunk_size = 16;
    static_assert(Dims::N <= 2 * CoreDims::THREADS_PER_WARP * chunk_size);

    // Compiler on H200 barfs out on plain FP8 transfers as it fails to
    // propagate alignment guarantees: just case to a 32-bit value; we know all
    // data to be 16-byte aligned.
    const OpaqueElement* weights =
        (const OpaqueElement*)(expert_weights_down +
                               id * Dims::N * Dims::HIDDEN_STATES +
                               base_row * Dims::N);
    for (unsigned row = d_warp, i = 0;
         i < CoreDims::W_DOWN_TILE / (CoreDims::PREFETCH_WARP_COUNT / 2);
         row += CoreDims::PREFETCH_WARP_COUNT / 2, i++) {
      unsigned col = d_thread * chunk_size;
      // "clever" condition to allow for compile-time optimization (becomes
      // no-op on H200)
      if (Dims::N == 2 * CoreDims::THREADS_PER_WARP * chunk_size ||
          col < Dims::N) {
        copy128(shm->w[w_index][row][col],
                weights[(row * Dims::N + col) / sizeof(OpaqueElement)], pipe);
      }
    }
  }

  // request Scale tile — block-wise 2D scale
  if (d_warp == 0) {
    constexpr uint32_t SCALE_TILE_SIZE =
        MoE_SHM<Dims>::U::Gemm2Data::DOWN_SCALE_TILE_SIZE;
    constexpr uint32_t COL_BLOCKS = (Dims::N + 127) / 128;
    if (d_thread < SCALE_TILE_SIZE) {
      uint32_t rb = d_thread / COL_BLOCKS;
      uint32_t cb = d_thread % COL_BLOCKS;
      uint32_t global_rb = (base_row / 128) + rb;
      shm->scale[w_index][d_thread] =
          expert_scales_down[id * Dims::DOWN_SCALE_ROWS *
                                 Dims::DOWN_SCALE_COLS +
                             global_rb * Dims::DOWN_SCALE_COLS + cb];
    }
  }
}

/**
 * @brief Per-expert activation-scale loader (down-projection, TMA path).
 *
 * Loads ALL `T_TILE × (Dims::N / 64)` per-token activation scales for
 * the current expert into `a_down_scale[col_block][tok]`.  Called ONCE
 * per expert at the top of the per-expert loop, NOT per K-step.  This
 * is the hoisted-out-of-K-loop variant: the K-loop's WGMMA scale-apply
 * indexes `a_down_scale[s * K_SUBSTEPS_DOWN * 2 + 2 * kk + half][tok]`
 * directly, removing the per-K-step cp.async + pipe drain that the
 * older variant required.
 *
 * Rank-indexing invariant (R11.3, R12.9):
 *   `a_down_scale[col_block][rank]` must match the scale for the token
 *   whose fp8 payload sits at `a_down_wgmma[slot][?][rank][kc][ki]`.
 *   That fp8 payload is loaded by the bulk-per-expert TMA from GM rows
 *   `[expert_slot_start[id], expert_slot_start[id] + routed_count)`
 *   of `spec->temp_fp8`.  Those same rows were written by the Phase-3
 *   up-proj epilogue at `dest_row = sorted_slot[pair] =
 *   expert_slot_start[id] + rank`, alongside
 *   `temp_act_scale[dest_row][...]` at the same row, so reading scales
 *   from row `expert_slot_start[id] + rank` guarantees byte-level
 *   alignment between the fp8 payload and its per-token scale.
 *
 * Thread mapping:
 *   - Gated to prefetch warp `RUN_PW` (one of 0..3 = warps 8..11).
 *   - Within that warp, lanes 0..(T_TILE * (N/64) - 1) each load one
 *     scale.  For Qwen3.5 (T_TILE=8, N/64=8): 64 lanes — but we only
 *     have 32 in a warp, so each lane does TWO loads via a stride-32
 *     loop.  Tiny work either way (≤256 B per expert).
 *
 * @tparam Dims         MoE dims.
 * @tparam ScaleTok     Must be T_TILE (8).
 * @tparam ScaleHalves  Must be Dims::N / 64.
 * @tparam RunPw        Prefetch-warp gate (0..3).  Use 0 for warp 8,
 *                      1 for warp 9, etc.
 * @param spec          Global scratchpad (reads `temp_act_scale`).
 * @param shmem         SHM struct (reads `expert_routed_count`,
 *                      `expert_slot_start`).
 * @param id            Current expert id.
 * @param dest_scale    SHM `a_down_scale[N/64][T_TILE]`.
 *
 * @note Prefetch-warp only.  Caller MUST ensure a `__syncthreads()`
 *       before the K-loop reads `a_down_scale`.
 */
template <typename Dims, std::size_t ScaleHalves, std::size_t ScaleTok,
          unsigned RunPw = 0u>
__device__ inline void moe_load_down_wgmma_act_scale_per_expert(
    const MoEGemmSpec<Dims>* __restrict__ spec,
    const MoE_SHM<Dims>* __restrict__ shmem, std::uint32_t id,
    S_element (&dest_scale)[ScaleHalves][ScaleTok]) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(ScaleTok == CoreDims::T_TILE,
                "down-proj activation scale tile must have T_TILE=8 tokens");
  static_assert(ScaleHalves == Dims::N / 64u,
                "down-proj activation scale tile must have N/64 halves "
                "per token (one per per-64-col up-block, full reduction "
                "dim)");

  constexpr unsigned ACT_BLOCK = 64;                    // per-64-col up-block
  constexpr unsigned SCALE_COLS = Dims::N / ACT_BLOCK;  // = ScaleHalves

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3

  if (pw != RunPw) return;

  // Cache routed_count / expert_start once per thread (cheap broadcast).
  const auto* tma_shm = &shmem->u.tiny_wgmma_tma;
  const uint32_t routed_count =
      static_cast<uint32_t>(tma_shm->expert_routed_count[id]);
  const uint32_t expert_start =
      static_cast<uint32_t>(tma_shm->expert_slot_start[id]);

  // Strided loop over the (slot_row, half) plane.  Total entries =
  // T_TILE * SCALE_COLS = 8 * 8 = 64 for Qwen3.5; one warp's 32 lanes
  // do 2 loads each (stride 32).  GM `temp_act_scale` is `[row][half]`
  // (row-major, written by the up-proj epilogue) but SHM
  // `a_down_scale` is `[half][row]` (chosen for bank-conflict-free
  // reads in the down-proj WGMMA scale-apply, see comment on the SHM
  // declaration in `MoE_SHM::U::TinyDataWGMMA_TMA`).  This loop
  // transposes on the fly.
  constexpr unsigned TOTAL = (unsigned)(ScaleTok * ScaleHalves);
  for (unsigned i = thread; i < TOTAL; i += 32u) {
    const unsigned slot_row = i / SCALE_COLS;  // 0..T_TILE-1
    const unsigned half = i % SCALE_COLS;      // 0..ScaleHalves-1

    if (slot_row < routed_count) {
      const uint32_t source_row = expert_start + slot_row;
      dest_scale[half][slot_row] =
          spec->temp_act_scale[source_row * SCALE_COLS + half];
    } else {
      // Unused rank — see preamble in the BS8 down-proj kernel for
      // why the value doesn't matter.  Set to 0 for cleanliness.
      dest_scale[half][slot_row] = 0.f;
    }
  }
}

/**
 * @brief v1 streaming-pipeline WGMMA down-projection weight-scale tile loader.
 *
 * Loads one K-step's worth of the down-projection's fp32 weight scales
 * into a SHM slot — one scale per (warpgroup, col-block) pair.  There
 * are 2 warpgroups (WG0 covers weight rows `[base_col..base_col+63]`,
 * WG1 covers rows `[base_col+64..base_col+127]`) and `W_DOWN_SCALE_COLS`
 * col-blocks along the K = Dims::N dimension.  (For block-wise quant,
 * `W_DOWN_SCALE_COLS == Dims::DOWN_SCALE_COLS == N/128`; for per-channel,
 * it is a 1-wide placeholder and these bytes are never consumed.)
 *
 * The down-proj weight matrix is `[NUM_EXPERTS, HIDDEN_STATES, N]`, so
 * the 128 output cols that this down-block owns map to 128 weight rows
 * `[base_col, base_col + 128)`.  Because `base_col` is a multiple of
 * 128, WG0's 64 weight rows and WG1's 64 weight rows always fall within
 * the SAME 128-row scale-block (so `row_block_wg0 == row_block_wg1`),
 * but we still compute and store both independently for generality.
 *
 * Scale tensor layout (block-wise):
 *   `expert_scales_down[NUM_EXPERTS][DOWN_SCALE_ROWS][DOWN_SCALE_COLS]` fp32,
 *   where `DOWN_SCALE_ROWS = HIDDEN_STATES / 128`
 *   and   `DOWN_SCALE_COLS = N             / 128`.
 *
 * SHM layout written:
 *   `dest[wg][col_block]` for `wg in [0, 2)`, `col_block in [0,
 * W_DOWN_SCALE_COLS)`.
 *
 * Thread distribution: the first prefetch warp loads all
 * `2 * W_DOWN_SCALE_COLS` scalars synchronously.  For Qwen3.5-35B
 * (W_DOWN_SCALE_COLS=4) this is just 8 fp32 values — well under a warp.
 * No async copy is used; the synchronous SHM stores complete as soon
 * as the warp retires them.
 *
 * @tparam Dims             MoE dims.
 * @tparam DestRows         Must be 2 (WG0, WG1).
 * @tparam DestCols         Must be `W_DOWN_SCALE_COLS` (= N/128 for
 * block-wise).
 * @param  expert_scales_down  GM pointer to the full scale tensor.
 * @param  id                  Expert index.
 * @param  base_col            First output col of this block's M stripe
 *                             (multiple of 128).
 * @param  dest                SHM slot `[2 row-blocks][W_DOWN_SCALE_COLS]`
 *                             fp32.
 *
 * @note Prefetch-warp only.  Unlike the fp8 tile loaders, this function
 *       uses synchronous SHM stores (no `pipe`) because the payload is
 *       tiny (≤ 32 bytes for Qwen3.5-35B) and fits comfortably in a
 *       single warp-sized burst.
 */
template <typename Dims, std::size_t DestRows, std::size_t DestCols>
__device__ inline void moe_load_down_wgmma_weight_scale_tile(
    const S_element* __restrict__ expert_scales_down, std::uint32_t id,
    std::uint32_t base_col, S_element (&dest)[DestRows][DestCols]) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(
      DestRows == 2,
      "down-proj weight-scale tile must have 2 row-blocks (WG0, WG1)");

  constexpr uint32_t COLS = DestCols;  // W_DOWN_SCALE_COLS = N/128 (block-wise)

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3

  // WG0 covers weight rows [base_col    .. base_col + 63]  → row-block
  // (base_col      ) / 128 WG1 covers weight rows [base_col+64 .. base_col +
  // 127] → row-block (base_col + 64 ) / 128
  //
  // For the canonical layout (`base_col` multiple of 128) these two
  // indices are equal.  Computing them independently keeps the loader
  // correct if a future config picks a non-multiple-of-128 base_col.
  //
  // Run on prefetch warp 1 (= block warp 9).  The matching activation
  // scale loader (`moe_load_down_wgmma_act_scale_per_expert`) runs on
  // prefetch warp 0 (= block warp 8) — both at the top of the
  // per-expert loop — so the two scale loads issue in parallel on
  // independent warps instead of serializing on warp 8 alone.
  if (pw == 1 && thread < DestRows * COLS) {
    const uint32_t wg = thread / COLS;  // 0..1
    const uint32_t cb = thread % COLS;  // col-block index in [0, COLS)
    const uint32_t weight_row = base_col + (wg == 0 ? 0u : 64u);
    const uint32_t rb = weight_row / 128;
    dest[wg][cb] =
        expert_scales_down[id * Dims::DOWN_SCALE_ROWS * Dims::DOWN_SCALE_COLS +
                           rb * Dims::DOWN_SCALE_COLS + cb];
  }
}

/**
 * @brief Initiate the copy of up-projection output ('temp tokens') from Global
 * to Shared Memory
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * token activations. The copy operations will be queued in the given @a pipe,
 * which the caller must use to wait for their completion.
 *
 * The number of tokens copied is limited by the capacity of @a dest as well as
 * the remaining range of tokens specified by @a expert and the starting offset
 * @a a_row.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @param source Pointer activation array of shape [BS, N] in row-major order.
 *               Individual elements are in BF16 format.
 *               Stored in Global Memory.
 * @param expert Specified the range of rows relevant for the current expert.
 * @param a_row Offset of the first row within the range of @a expert to copy.
 * @param dest Shared Memory tile to store the copy.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_request_temp_token(
    const A_element* __restrict__ source, const ExpertRef& expert,
    unsigned a_row, A_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // async transfers are 16 bytes / thread
  const unsigned chunk_size = 16 / sizeof(*source);

  const A_element* t = &source[expert.first_token * Dims::N];
  unsigned int a_rows = expert.last_token - expert.first_token;
  for (unsigned i = warp; i < Rows; i += CoreDims::PREFETCH_WARP_COUNT) {
    if (i + a_row < a_rows) {
      for (unsigned col = thread * chunk_size; col < Cols;
           col += CoreDims::THREADS_PER_WARP * chunk_size) {
        copy128(dest[i][col], t[(i + a_row) * Dims::N + col], pipe);
      }
    }
  }
}

/**
 * @brief Top-K single-pass down-projection reduction with weighted
 * accumulation.
 *
 *  - @p row0 / @p row1 are sorted positions; the original token index is
 *    looked up via @c shmem->path.bs64.token_indexes_topk[sorted_pos].
 *  - The output is accumulated with @c += (not @c =) because each original
 *    token may receive contributions from multiple experts.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_down_reduction_topk(
    const float (&partial_result)[Rows][Cols], bool store_row0, bool store_row1,
    unsigned sorted_row0, unsigned sorted_row1,
    const MoE_SHM<Dims>* __restrict__ shmem, R_element* __restrict__ result) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;

  // Map sorted positions to original token indexes
  unsigned orig_row0 =
      store_row0 ? shmem->path.bs64.token_indexes_topk[sorted_row0] : 0;
  unsigned orig_row1 =
      store_row1 ? shmem->path.bs64.token_indexes_topk[sorted_row1] : 0;

  for (unsigned w_row = warp * CoreDims::W_DOWN_MMA_TILE;
       w_row < CoreDims::W_DOWN_TILE;
       w_row += CoreDims::W_DOWN_MMA_TILE * CoreDims::TOTAL_WARP_COUNT) {
    float d0 = partial_result[w_row / 2][thread + 0];
    float d1 = partial_result[w_row / 2][thread + 32];
    float d2 = partial_result[w_row / 2][thread + 64];
    float d3 = partial_result[w_row / 2][thread + 96];

    for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
      d0 += partial_result[w_row / 2 + i][thread + 0];
      d1 += partial_result[w_row / 2 + i][thread + 32];
      d2 += partial_result[w_row / 2 + i][thread + 64];
      d3 += partial_result[w_row / 2 + i][thread + 96];
    }

    // Accumulate into original token positions (+=) weighted by routing_weight
    // path.bs64.token_weights[sorted_pos] = routing_weight (populated in
    // prepare_moe_topk_BSx_Ey — we read it directly to avoid a separate
    // copy-back pass).
    if (store_row0) {
      float rw0 = shmem->path.bs64.token_weights[sorted_row0];
      unsigned col0 =
          orig_row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row;
      result[col0 + 0] = (R_element)((float)result[col0 + 0] + rw0 * d0);
      if (CoreDims::W_DOWN_TILE % 16 == 0 || w_row + 8 < CoreDims::W_DOWN_TILE)
        result[col0 + 8] = (R_element)((float)result[col0 + 8] + rw0 * d2);
    }
    if (store_row1) {
      float rw1 = shmem->path.bs64.token_weights[sorted_row1];
      unsigned col1 =
          orig_row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row;
      result[col1 + 0] = (R_element)((float)result[col1 + 0] + rw1 * d1);
      if (CoreDims::W_DOWN_TILE % 16 == 0 || w_row + 8 < CoreDims::W_DOWN_TILE)
        result[col1 + 8] = (R_element)((float)result[col1 + 8] + rw1 * d3);
    }
  }
}

/**
 * @brief Down-projection MMA with fp8 activations (shared by BS8 and BS64).
 *
 * Uses mma_fp8_fp8 (m16n8k32) — both weights and activations are fp8.
 *
 * D-output mapping (from PTX spec):
 *   d0 → D[t/4,     (t%4)*2]     row=t/4,     col=(t%4)*2
 *   d1 → D[t/4,     (t%4)*2 + 1] row=t/4,     col=(t%4)*2 + 1
 *   d2 → D[t/4 + 8, (t%4)*2]     row=t/4 + 8, col=(t%4)*2
 *   d3 → D[t/4 + 8, (t%4)*2 + 1] row=t/4 + 8, col=(t%4)*2 + 1
 *
 * D-columns are tokens. d0/d2 use token=(t%4)*2, d1/d3 use token=(t%4)*2+1.
 * Each token has its own per-block activation scale.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols,
          std::size_t OutRows, std::size_t OutCols, std::size_t ScaleRows,
          std::size_t ScaleCols>
__device__ inline void moe_down_mult_fp8(
    const W_element __restrict__ (&weights)[Rows][Cols],
    const float* __restrict__ scale, const AQ_element* __restrict__ act_row,
    const float (&act_block_scales_all)[ScaleRows][ScaleCols],
    std::uint32_t tok_02, std::uint32_t tok_13, bool store_row0,
    bool store_row1, float (&partial_result)[OutRows][OutCols]) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_calc_warp<Dims>();

  for (unsigned w_row = 0; w_row < CoreDims::W_DOWN_TILE;
       w_row += CoreDims::W_DOWN_MMA_TILE) {
    constexpr uint32_t COL_BLOCKS = (Dims::N + 127) / 128;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (unsigned base_col = warp * CoreDims::K_TILE, i = 0;
         i < Dims::N / CoreDims::BLOCK_STRIDE;
         base_col += CoreDims::BLOCK_STRIDE, i++) {
      float md0 = 0.f, md1 = 0.f, md2 = 0.f, md3 = 0.f;
      unsigned far_row =
          (Rows % 16 == 8 && w_row + 8 == Rows) ? w_row : w_row + 8;

      __nv_fp8x4_e4m3 w0 =
          *(__nv_fp8x4_e4m3*)&weights[w_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 0];
      __nv_fp8x4_e4m3 w1 =
          *(__nv_fp8x4_e4m3*)&weights[far_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 0];
      __nv_fp8x4_e4m3 w2 =
          *(__nv_fp8x4_e4m3*)&weights[w_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 16];
      __nv_fp8x4_e4m3 w3 =
          *(__nv_fp8x4_e4m3*)&weights[far_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 16];

      __nv_fp8x4_e4m3 b0 =
          *(__nv_fp8x4_e4m3*)&act_row[base_col + 4 * (thread % 4) + 0];
      __nv_fp8x4_e4m3 b1 =
          *(__nv_fp8x4_e4m3*)&act_row[base_col + 4 * (thread % 4) + 16];

      mma_fp8_fp8(md0, md1, md2, md3, w0, w1, w2, w3, b0, b1, 0.f, 0.f, 0.f,
                  0.f);

      unsigned n_block = base_col / 128;
      float ws = scale[0 * COL_BLOCKS + n_block];
      // d0/d2 use tok_02's scale; d1/d3 use tok_13's scale
      float as_02 = store_row0 ? act_block_scales_all[tok_02][n_block] : 0.f;
      float as_13 = store_row1 ? act_block_scales_all[tok_13][n_block] : 0.f;

      d0 += md0 * ws * as_02;
      d1 += md1 * ws * as_13;
      d2 += md2 * ws * as_02;
      d3 += md3 * ws * as_13;
    }

    if (store_row0) {
      partial_result[w_row / 2 + warp][thread + 0] = d0;
      partial_result[w_row / 2 + warp][thread + 64] = d2;
    }
    if (store_row1) {
      partial_result[w_row / 2 + warp][thread + 32] = d1;
      partial_result[w_row / 2 + warp][thread + 96] = d3;
    }
  }
}

/**
 * @brief Top-K single-pass down-projection for BS > 8.
 *
 * Uses the same fp8 MMA approach as BS8:
 *   1. Fetch bf16 SiLU output from global memory → t_bf16 staging buffer
 *   2. Quantize bf16 → fp8 with per-token block-wise scales → t_fp8
 *   3. MMA: fp8 weights × fp8 activations via mma_fp8_fp8 (m16n8k32)
 *   4. Reduce with correct D-mapping token assignment
 *
 * Uses token_indexes_topk (sorted positions → original tokens).
 * Calls moe_down_reduction_topk which accumulates with +=.
 * The output buffer must be zeroed before calling this function.
 */
template <typename Dims>
__device__ inline void moe_down_projection_topk(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ result, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS > 8,
                "Tiny is handled by its own kernel. Do not use "
                "moe_down_projection_topk for BS<=8");
  using CoreDims = MoECoreDims<Dims>;
  using MoE_SHM_t = MoE_SHM<Dims>;

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  typename MoE_SHM_t::U::Gemm2Data* shm = &shmem->u.gemm2;
  std::uint32_t expert_count = shmem->expert_count;

  const ExpertRef& first_expert = shmem->experts[0];

  assert(expert_count > 0);

  // Quantization constants
  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;
  constexpr std::uint32_t FLOATS_PER_LOAD = 4;
  constexpr std::uint32_t ACT_DOWN_BLOCK = 128;
  constexpr std::uint32_t NUM_DOWN_BLOCKS = Dims::N / ACT_DOWN_BLOCK;
  static_assert(Dims::N % ACT_DOWN_BLOCK == 0,
                "N must be divisible by activation block size");

  // Prime: prefetch first expert's weights + first token tile (bf16)
  if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_DOWN
    pipe.producer_acquire();
    moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down,
                                  first_expert.id, shm, 0, pipe);
    moe_request_temp_token<Dims>(spec->temp_bf16, first_expert, 0,
                                 shm->t_bf16[0], pipe);
    pipe.producer_commit();
  #endif
  }

  std::uint32_t t_index = 1;  // ping-pong for t_bf16 fetch
  std::uint32_t w_index = 1;  // ping-pong for weights
  std::uint32_t q_index = 0;  // ping-pong for t_fp8 quantized buffer

  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const ExpertRef& expert = shmem->experts[e];
    unsigned int a_rows = expert.last_token - expert.first_token;
    w_index ^= 1;

    for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::T_TILE) {
      t_index ^= 1;

      // Wait for bf16 temp tokens + weights to arrive
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        printf("[DBG64 DOWN_BF16 e=0] t_bf16[%u][0][0..7]:", t_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->t_bf16[t_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_BF16 e=0] t_bf16[%u][0][508..511]:", t_index);
        for (int i = 508; i < 512; i++)
          printf(" %.4f", (float)shm->t_bf16[t_index][0][i]);
        printf("\n");
      }
      // Also print for expert 1 to check later-expert data
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 1 && a_row == 0) {
        printf(
            "[DBG64 DOWN_BF16 e=1] expert_id=%u first=%u last=%u a_rows=%u\n",
            shmem->experts[1].id, shmem->experts[1].first_token,
            shmem->experts[1].last_token,
            shmem->experts[1].last_token - shmem->experts[1].first_token);
        printf("[DBG64 DOWN_BF16 e=1] t_bf16[%u][0][0..7]:", t_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->t_bf16[t_index][0][i]);
        printf("\n");
        // Also print what's in global memory at that address
        unsigned ft = shmem->experts[1].first_token;
        printf("[DBG64 DOWN_BF16 e=1] spec->temp_bf16[ft=%u][0..7]:", ft);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)spec->temp_bf16[ft * Dims::N + i]);
        printf("\n");
      }
  #endif

      if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_DOWN
        // Prefetch next tile while calc warps quantize + MMA
        pipe.producer_acquire();
        if (e + 1 < expert_count && a_row == 0) {
          moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down,
                                        shmem->experts[e + 1].id, shm,
                                        w_index ^ 1, pipe);
        }
        if (a_row + CoreDims::T_TILE < a_rows) {
          moe_request_temp_token<Dims>(spec->temp_bf16, expert,
                                       a_row + CoreDims::T_TILE,
                                       shm->t_bf16[t_index ^ 1], pipe);
        } else if (e + 1 < expert_count) {
          moe_request_temp_token<Dims>(spec->temp_bf16, shmem->experts[e + 1],
                                       0, shm->t_bf16[t_index ^ 1], pipe);
        }
        pipe.producer_commit();
  #endif
      } else {
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
        // ── Calc warps: quantize bf16 → fp8 ──────────────────────────────
        // Each calc warp handles one or more tokens.
        // Per-block (1, 128) quantization: each 128-element block gets its
        // own scale. With N=512, that's 4 blocks per row.
        const std::uint32_t cw = get_calc_warp<Dims>();
        unsigned valid_tokens = min((unsigned)CoreDims::T_TILE, a_rows - a_row);
        for (std::uint32_t tok = cw; tok < valid_tokens;
             tok += CoreDims::CALC_WARP_COUNT) {
          float regs[NUM_DOWN_BLOCKS * 4];

    #pragma unroll
          for (std::uint32_t blk = 0; blk < NUM_DOWN_BLOCKS; ++blk) {
            std::uint32_t blk_start = blk * ACT_DOWN_BLOCK;
            std::uint32_t col = blk_start + thread * FLOATS_PER_LOAD;

            // Read bf16 from staging buffer and convert to fp32 in registers.
            // 4 bf16 values per thread per block iteration, read as 2×bf162.
            __nv_bfloat162 bf_01 = *reinterpret_cast<const __nv_bfloat162*>(
                &shm->t_bf16[t_index][tok][col + 0]);
            __nv_bfloat162 bf_23 = *reinterpret_cast<const __nv_bfloat162*>(
                &shm->t_bf16[t_index][tok][col + 2]);
            float2 f01 = __bfloat1622float2(bf_01);
            float2 f23 = __bfloat1622float2(bf_23);
            regs[blk * 4 + 0] = f01.x;
            regs[blk * 4 + 1] = f01.y;
            regs[blk * 4 + 2] = f23.x;
            regs[blk * 4 + 3] = f23.y;

            float local_max = fmaxf(fmaxf(fabsf(f01.x), fabsf(f01.y)),
                                    fmaxf(fabsf(f23.x), fabsf(f23.y)));
            float blk_max = warp_reduce_max_float(local_max);
            if (blk_max < __FLT_MIN__) blk_max = 1.f;

            float blk_scale = blk_max * FP8_MAX_INV;
            float blk_inv_scale = FP8_MAX / blk_max;

            // Quantize → write fp8 to SHM
            __nv_fp8x4_e4m3 q{float4{regs[blk * 4 + 0] * blk_inv_scale,
                                     regs[blk * 4 + 1] * blk_inv_scale,
                                     regs[blk * 4 + 2] * blk_inv_scale,
                                     regs[blk * 4 + 3] * blk_inv_scale}};
            *reinterpret_cast<__nv_fp8x4_e4m3*>(
                &shm->t_fp8[q_index][tok][col]) = q;

            if (thread == 0) shm->t_scale[q_index][tok][blk] = blk_scale;
          }
        }
  #endif
      }

      __syncthreads();

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        printf("[DBG64 DOWN_QUANT e=0] t_fp8[%u][0][0..7]:", q_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->t_fp8[q_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] t_scale[%u][0][0..3]:", q_index);
        for (int i = 0; i < 4; i++)
          printf(" %.6f", shm->t_scale[q_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] w[%u] row0[0..7]:", w_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->w[w_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] w[%u] row1[0..7]:", w_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->w[w_index][1][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] w_scale[%u][0..3]:", w_index);
        for (int i = 0; i < 4; i++) printf(" %.6f", shm->scale[w_index][i]);
        printf("\n");
      }
  #endif

      // ── MMA: fp8 weights × fp8 activations ────────────────────────────
      if (!is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
        static_assert(CoreDims::W_DOWN_TILE % 8 == 0);

        // D-output mapping (m16n8k32):
        //   d0/d2 → token=(t%4)*2,  d1/d3 → token=(t%4)*2+1
        const std::uint32_t tok_02 = (thread % 4) * 2;
        const std::uint32_t tok_13 = (thread % 4) * 2 + 1;
        bool s0 = tok_02 < min((unsigned)CoreDims::T_TILE, a_rows - a_row);
        bool s1 = tok_13 < min((unsigned)CoreDims::T_TILE, a_rows - a_row);

        moe_down_mult_fp8<Dims>(shm->w[w_index], shm->scale[w_index],
                                shm->t_fp8[q_index][thread / 4],
                                shm->t_scale[q_index], tok_02, tok_13, s0, s1,
                                shm->partial_result);
  #endif
      }

      __syncthreads();

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        printf("[DBG64 DOWN_MMA e=0] partial_result[0][0..7]:");
        for (int i = 0; i < 8; i++) printf(" %.4f", shm->partial_result[0][i]);
        printf("\n");
        printf("[DBG64 DOWN_MMA e=0] partial_result[1][0..7]:");
        for (int i = 0; i < 8; i++) printf(" %.4f", shm->partial_result[1][i]);
        printf("\n");
      }
  #endif

      // ── Reduce: accumulate into original token positions ───────────────
      // D-mapping: d0/d2 → sorted_row0 = (t%4)*2, d1/d3 → sorted_row1
      unsigned sorted_row0 = expert.first_token + a_row + (thread % 4) * 2;
      unsigned sorted_row1 = sorted_row0 + 1;
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
      moe_down_reduction_topk<Dims>(shm->partial_result,
                                    sorted_row0 < expert.last_token,
                                    sorted_row1 < expert.last_token,
                                    sorted_row0, sorted_row1, shmem, result);
  #endif

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        unsigned orig_tok = shmem->path.bs64.token_indexes_topk[sorted_row0];
        float rw = shmem->path.bs64.token_weights[sorted_row0];
        printf(
            "[DBG64 DOWN_REDUCE e=0] sorted_row0=%u orig_tok=%u rw=%.4f "
            "store=%d\n",
            sorted_row0, orig_tok, rw, (int)(sorted_row0 < expert.last_token));
        printf("[DBG64 DOWN_REDUCE e=0] result[tok=%u][0..7]:", orig_tok);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)result[orig_tok * Dims::HIDDEN_STATES + i]);
        printf("\n");
      }
  #endif

      q_index ^= 1;
    }

    __syncthreads();
  }

  #ifdef DEBUG_MOE_PRINT
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("[DBG64 DOWN_FINAL] result[tok=0][0..7] after all experts:");
    for (int i = 0; i < 8; i++)
      printf(" %.4f", (float)result[0 * Dims::HIDDEN_STATES + i]);
    printf("\n");
  }
  #endif
}

}  // namespace moe_monokernel

namespace moe_monokernel {

///////////////////////////////////////////////////////////////////////////////
//
// moe_down_projection_BS8_allexperts_wgmma_tma
//
// TMA + WGMMA down-projection for BS <= 8.  Both the fp8 expert-weight
// tile and the fp8 intermediate-activation tile are loaded via
// `cp.async.bulk.tensor.2d` with `CU_TENSOR_MAP_SWIZZLE_128B`, issued by
// a single TMA launcher thread (warp 8, lane 0).  Completion of each
// tile is signalled via SHM mbarriers (`bar_w[2]`, `bar_a[2]` reused
// from Phase 3); consumer warps wait via `mbarrier.try_wait.parity`.
//
// Descriptors are built host-side in the torch binding wrapper and
// passed to the top-level kernel as `__grid_constant__ CUtensorMap
// const` parameters.  At this device-side helper boundary they appear
// as `CUtensorMap const&` — the `__grid_constant__` qualifier only
// applies at the kernel-function boundary.
//
template <typename Dims>
__device__ inline void moe_down_projection_BS8_allexperts_wgmma_tma(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t top_k,
    std::uint32_t batch_size, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem, CUtensorMap const& down_weights_desc,
    CUtensorMap const& down_activations_desc) {
  static_assert(Dims::BS <= 8,
                "moe_down_projection_BS8_allexperts_wgmma_tma is BS<=8 only");
  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // `expert_weights_down` is retained on the parameter list for signature
  // parity with the `cp.async` reference but is not dereferenced on the
  // TMA path — all weight GM reads go through `down_weights_desc`.
  (void)expert_weights_down;

  // ── Compile-time constants ────────────────────────────────────────────
  constexpr std::uint32_t K_TILE_W = CoreDims::K_TILE_WGMMA;    // 32
  constexpr std::uint32_t K_STEP_DOWN = CoreDims::K_STEP_DOWN;  // 128 or 256
  // K_STEP_WGMMA is the inner SWZ128 atom K-width (always 128).  Each
  // outer K-step packs K_SUBSTEPS_DOWN = K_STEP_DOWN / K_STEP_WGMMA
  // 128-K sub-blocks; the WGMMA loop chains 4 m64n8k32 instructions
  // per sub-block (lo+hi pair around the 128-K block-wise scale
  // boundary).
  constexpr std::uint32_t K_STEP_WGMMA = CoreDims::K_STEP_WGMMA;         // 128
  constexpr std::uint32_t K_SUBSTEPS_DOWN = CoreDims::K_SUBSTEPS_DOWN;   // 1/2
  constexpr std::uint32_t WGMMAS_PER_SUBSTEP = K_STEP_WGMMA / K_TILE_W;  // 4
  constexpr std::uint32_t K_TILES_DOWN = Dims::N / K_STEP_DOWN;          // N/K
  constexpr std::uint32_t DOWN_COL_TILE =
      CoreDims::DOWN_COL_TILE;                                  // 128 or 256
  constexpr std::uint32_t DOWN_GRID = CoreDims::DOWN_GRID;      // 16 or 8
  constexpr std::uint32_t DOWN_GROUPS = CoreDims::DOWN_GROUPS;  // 8 or 16
  // Phase-2a layout alignment: one block owns `DOWN_COL_TILE` output
  // cols.  The two WGs (64 output cols each per WGMMA pass) together
  // cover 128 cols per pass, so we need `HALVES = DOWN_COL_TILE / 128`
  // sequential passes per K-step:
  //   * Pre Phase 2a (DOWN_COL_TILE=128): HALVES=1, pre-alignment 128-col
  //     behaviour.  Weight tile in SHM is 128×128.
  //   * Post Phase 2a (DOWN_COL_TILE=256, BS8 TMA+WGMMA only): HALVES=2,
  //     weight tile in SHM is 256×128.  Pass 0 covers output rows
  //     [base_col+0..base_col+127], pass 1 covers [+128..+255].
  constexpr std::uint32_t DOWN_COL_HALVES = DOWN_COL_TILE / 128u;  // 1 or 2
  static_assert(DOWN_COL_TILE % 128u == 0,
                "DOWN_COL_TILE must be a multiple of 128 for the TMA+WGMMA "
                "down-projection (the weight tile is laid out as 128-row "
                "SWZ128 atoms on the M axis).");
  static_assert(DOWN_COL_HALVES <= 2u,
                "DOWN_COL_TILE > 256 not supported by the Phase-2a "
                "two-halves WGMMA structure.");
  // Per-K-step weight TMA transfer size.  Each 128×128 fp8 atom is
  // 16384 B; an outer K-step issues
  //   `DOWN_COL_HALVES * K_SUBSTEPS_DOWN`
  // such atoms total — `DOWN_COL_HALVES` along the M axis × the
  // `K_SUBSTEPS_DOWN` 128-K sub-blocks that make up one outer step —
  // stacking them along the M axis as contiguous 128-row SWZ128 atoms.
  constexpr std::uint32_t DOWN_W_TX_BYTES_PER_HALF = 16384u;
  constexpr std::uint32_t DOWN_W_TX_BYTES_TOTAL =
      DOWN_W_TX_BYTES_PER_HALF * DOWN_COL_HALVES *
      K_SUBSTEPS_DOWN;  // 16384 * halves * substeps
  // Per-K-step activation TMA transfer size.  Each 1024-B SWZ128 atom
  // covers one 128-K sub-block; an outer K-step issues
  // `K_SUBSTEPS_DOWN` such atoms back-to-back along the K axis.
  constexpr std::uint32_t DOWN_A_TX_BYTES_TOTAL = 1024u * K_SUBSTEPS_DOWN;
  constexpr std::uint32_t W_DOWN_SCALE_COLS =
      MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::W_DOWN_SCALE_COLS;

  static_assert(Dims::N % K_STEP_DOWN == 0,
                "Dims::N must be a multiple of K_STEP_DOWN");
  static_assert(K_STEP_DOWN % K_STEP_WGMMA == 0,
                "K_STEP_DOWN must be a multiple of K_STEP_WGMMA (=128, the "
                "SWZ128 atom K-width)");

  // A descriptor strides for the 128×128 fp8 weight tile under
  // SWIZZLE_128B.  The TMA hardware applies the 8-row × 128-byte
  // core-matrix XOR swizzle at write time, so each 1024-B atom holds
  // one 8-row M-block.  CUTLASS Major::K B128 layout:
  //   LBO = 16 B   (one K-core-matrix within the 1024-B atom)
  //   SBO = 1024 B (next M-block atom)
  //   swizzle_mode = 1
  // The Python pre-interleave is NOT applied — the raw `[E, K, N]`
  // row-major fp8 weight tensor is fed to the TMA, and the swizzle
  // hardware produces the canonical layout in SHM.
  constexpr std::uint64_t A_LBO = 16ULL;
  constexpr std::uint64_t A_SBO = 1024ULL;
  constexpr std::uint32_t A_SWIZZLE = 1u;
  // B descriptor strides for the 8-token × 128-K fp8 activation tile
  // under SWIZZLE_128B (token-major Major::K B128 canonical layout).
  // The TMA hardware applies the 8-row × 128-byte XOR at write time:
  //   logical byte(tok, kc, ki) = tok * 128 + kc * 16 + ki
  //   SHM byte = logical byte  XOR  swizzle(tok bits)
  // CUTLASS strides for this layout:
  //   LBO = 16 B   (stride between 16-B K-chunks along K inside a row)
  //   SBO = 128 B  (stride between 8-row atoms; unused here, N=8 → 1 atom)
  //   swizzle = 1  (SWZ128)
  constexpr std::uint64_t B_LBO = 16ULL;
  constexpr std::uint64_t B_SBO = 128ULL;
  constexpr std::uint32_t B_SWIZZLE = 1u;

  // ── Warp / lane identity ──────────────────────────────────────────────
  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;  // 0..11
  const unsigned lane = thread_in_block & 31;  // 0..31
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;  // 0..3 within each WG
  const unsigned my_wg = is_wg1 ? 1u : 0u;

  // TMA path uses the `tiny_wgmma_tma` union variant (byte-identical to
  // `tiny_wgmma` plus 32 B of mbarriers + the reorg tables at the tail).
  auto* shm = &shmem->u.tiny_wgmma_tma;

  // ── Grid-to-(expert-group, output-col-tile) mapping ───────────────────
  const std::uint32_t down_group = blockIdx.x / DOWN_GRID;
  const std::uint32_t down_block_idx = blockIdx.x % DOWN_GRID;
  const std::uint32_t base_col = down_block_idx * DOWN_COL_TILE;

  // ── Per-thread fp32 accumulators ──────────────────────────────────────
  // One m64n8k32 WGMMA holds 4 fp32 accumulators per thread (d0/d1/d2/d3).
  // The down-projection's inner K chain stacks 4 WGMMAs per K-step into a
  // "lo" chunk (K[0..63], 2 WGMMAs) and "hi" chunk (K[64..127], 2 WGMMAs).
  // Phase-2a generalizes this to `DOWN_COL_HALVES` parallel halves along
  // the M axis, so we hold `4 * HALVES` final accumulators per thread:
  //   * chunk_d_lo[h][0..3] / chunk_d_hi[h][0..3] — per-half per-K-chunk
  //                                                  scratch, reset each
  //                                                  K-step.
  //   * final_d[h][0..3]                           — per-half per-expert
  //                                                  K-loop accumulator
  //                                                  (scaled by ws·as).
  float chunk_d_lo[DOWN_COL_HALVES][4] = {{0.f}};
  float chunk_d_hi[DOWN_COL_HALVES][4] = {{0.f}};
  float final_d[DOWN_COL_HALVES][4] = {{0.f}};

  // ── Zero out_accum + Phase-4 mbarrier (re-)initialization ────────────
  //
  // Two independent SHM publishes happen here, merged behind a single
  // block-wide sync:
  //
  //   (1) Zero per-block SHM `out_accum[BS][DOWN_COL_TILE]`.  `out_accum`
  //       is not read until the per-expert accumulate loop below, which
  //       follows several additional `__syncthreads()` (priming drain +
  //       per-K-iter syncs).  A single trailing sync here is sufficient.
  //
  //   (2) Re-initialize the 4 TMA mbarriers (R4.1, R4.9).  The Phase-3→4
  //       `grid.sync()` guarantees the barriers are idle at entry; we
  //       reset them to `arrival_count = 1` on the launcher thread and
  //       publish with `fence.mbarrier_init.release.cluster`.  They are
  //       not armed or waited on until inside the expert loop, strictly
  //       after this sync.
  //
  // Both targets (`out_accum`, `bar_w/bar_a`) are disjoint SHM regions
  // so the two operations race-freely run in parallel; only one sync is
  // needed to publish both.
  //
  // The `out_accum` zero-fill is unconditional — its cost is negligible
  // (1 KB per block with 384 threads) and keeping it well-defined
  // simplifies reasoning under either profile flag.  The mbarrier inits
  // are also unconditional: they cost only 4 SHM stores + 1 fence and
  // leave the barriers in a known-idle state.  The K-loop
  // waits/arms are themselves gated on SKIP_PREFETCH, so uninit'd
  // barriers are never a concern.
  for (unsigned idx = thread_in_block; idx < Dims::BS * DOWN_COL_TILE;
       idx += blockDim.x) {
    const unsigned tok = idx / DOWN_COL_TILE;
    const unsigned col = idx % DOWN_COL_TILE;
    shm->out_accum[tok][col] = 0.f;
  }
  if (is_tma_launcher_thread<Dims>()) {
    mbarrier_init(&shm->bar_w[0], 1u);
    mbarrier_init(&shm->bar_w[1], 1u);
    mbarrier_init(&shm->bar_a[0], 1u);
    mbarrier_init(&shm->bar_a[1], 1u);
    fence_mbarrier_init_release_cluster();
  }
  __syncthreads();

  MONO_PHASE_TIMESTAMP(t_down_after_prologue);

  const std::uint32_t expert_count = shmem->expert_count;

  // Per-slot mbarrier parity state.  Hoisted OUT of the expert loop so
  // it stays continuously synced with the bar_{w,a}[slot] hardware
  // phase across expert transitions.  Resetting to 0 every expert is
  // only correct when each slot is visited an even number of times per
  // expert (so the bar returns to phase 0); for K_STEP_DOWN=256 with
  // K_TILES_DOWN=2 each slot is visited exactly once per expert and
  // ends at phase 1, so a blind reset would deadlock the next
  // expert's `try_wait_parity(0)` call.  Keeping this state continuous
  // works for ANY K_TILES_DOWN, and adds zero cost.
  std::uint32_t parity_w[2] = {0u, 0u};
  std::uint32_t parity_a[2] = {0u, 0u};

  // ── Per-expert loop (expert_start = down_group, stride = DOWN_GROUPS) ─
  for (std::uint32_t e = down_group; e < expert_count; e += DOWN_GROUPS) {
    const std::uint32_t id = shmem->experts[e].id;

    // Per-expert (expert, token) reorganization state (R11).  These
    // values are populated by `prepare_moe_topk_BS8` in Phase 1/2 and
    // drive the bulk activation TMA's outer coordinate and row count.
    const std::uint32_t routed_count =
        static_cast<std::uint32_t>(shm->expert_routed_count[id]);
    const std::uint32_t expert_start =
        static_cast<std::uint32_t>(shm->expert_slot_start[id]);

    // Reset per-expert `final_d` accumulator.
  #pragma unroll
    for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
  #pragma unroll
      for (std::uint32_t r = 0; r < 4u; ++r) {
        final_d[h][r] = 0.f;
      }
    }

    // ── Load weight scales + per-expert activation scales (parallel) ───
    // Both scale tensors are fixed across K-steps for this expert, so
    // we load them once per expert at the top of the per-expert loop.
    //
    // Weight scales: warp 9 (RunPw=1, see helper for rationale).
    //   Loaded via plain SHM stores from the GM pointer; each `h` call
    //   covers a 128-row M-block × full K (= W_DOWN_SCALE_COLS=N/128
    //   col-blocks).
    //
    // Activation scales: warp 8 (RunPw=0).  Loaded as a per-expert
    //   tile of `T_TILE × (N/64)` fp32 (= 256 B for Qwen3.5).  Hoisted
    //   out of the K-loop so the WGMMA scale-apply reads the full set
    //   from SHM with `a_down_scale[tok][s * K_SUBSTEPS_DOWN * 2 + 2 *
    //   kk + half]` indexing — no per-K-step reload.
    //
    // Both loaders run inside `if (is_prefetch_warp<Dims>())`; their
    // `RunPw` template arg routes the actual stores onto the correct
    // single warp, so warps 8 and 9 issue in parallel on independent
    // SM sub-schedulers.
    if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_DOWN
    #pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        moe_load_down_wgmma_weight_scale_tile<Dims>(
            expert_scales_down, id, base_col + h * 128u, shm->w_down_scale[h]);
      }
      moe_load_down_wgmma_act_scale_per_expert<Dims, Dims::N / 64u,
                                               CoreDims::T_TILE,
                                               /*RunPw=*/0u>(spec, shmem, id,
                                                             shm->a_down_scale);
  #endif
    }

    // ── Priming: prefetch slot 0 (w + a + a_scale for K-step 0) ────────
    //
    // First-iteration priming only.  For e > down_group, the previous
    // expert's last-K-iteration in-loop launcher already issued the
    // INTER-EXPERT LOOKAHEAD TMAs into slot 0 (see the launcher's
    // `else if (e + DOWN_GROUPS < expert_count)` branch below), so we
    // skip the weight + activation TMA arm/issue here — only the
    // weight scales (cheap cp.async, no bar) and the activation
    // scales (also cp.async, separate `pipe`) need fresh loads per
    // expert.
    //
    // The fp8 weight and activation tiles land via
    // `cp.async.bulk.tensor.2d` issued by the launcher thread, with
    // completion signalled on `bar_w[0]` / `bar_a[0]`.  The activation
    // scale tile loads via cp.async through `pipe`.
    //
    // Compiled out under MONO_PROFILE_SKIP_PREFETCH_DOWN; the matching
    // compute-side `mbarrier_try_wait_parity` on bar_{w,a}[0] inside
    // the K-loop below is also compiled out so there is no
    // spin-forever deadlock.
    const bool need_first_expert_prime = (e == down_group);
    if (need_first_expert_prime && is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_DOWN
      // Weight tile for K-step 0: arm bar_w[0] with the TOTAL tx_bytes
      // and issue ONE TMA per 128-K substep covering the full
      // DOWN_COL_TILE M-rows.  The descriptor's boxDim row dimension
      // is set to DOWN_COL_TILE host-side (see
      // `create_down_weight_tma_desc(... row_box=DOWN_COL_TILE)`), so
      // each TMA delivers `DOWN_COL_TILE * 128` bytes at once
      // (16 KB for DOWN_COL_TILE=128, 32 KB for DOWN_COL_TILE=256).
      // The two 128-row sub-atoms inside a 256-row delivery still
      // observe the SWZ128 byte layout, so the consumer-side WGMMA A
      // descriptor (which addresses one 128-row sub-atom per WGMMA
      // call) is unchanged.
      //
      // Per-K-substep destination row offset into `w_down_wgmma[slot]`:
      //   kk-th substep starts at row `kk * DOWN_COL_TILE`.
      // All atoms retire into the same bar_w[0], so a single
      // `mbarrier.try_wait.parity` on the compute side drains them all.
      mbarrier_arrive_expect_tx(&shm->bar_w[0],
                                /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
    #pragma unroll
      for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
        W_element* dest_base = &shm->w_down_wgmma[0][kk * DOWN_COL_TILE][0];
        tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/id,
                                 /*K=*/Dims::HIDDEN_STATES,
                                 /*base_col=*/base_col,
                                 /*k_start=*/kk * K_STEP_WGMMA,
                                 /*dest_smem_ptr=*/(void*)dest_base,
                                 /*bar_smem_ptr=*/&shm->bar_w[0]);
      }

      // Activation tile for K-step 0: only issue when routed_count > 0.
      // One TMA per 128-K sub-block (each delivers a 1024-B SWZ128
      // atom into `a_down_wgmma[slot][kk]`).  All atoms retire into
      // `bar_a[0]`; the launcher arms it once with the total tx_bytes.
      if (routed_count > 0u) {
        mbarrier_arrive_expect_tx(&shm->bar_a[0],
                                  /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
    #pragma unroll
        for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
          tma_load_down_wgmma_activation_bulk(
              down_activations_desc,
              /*k_start=*/kk * K_STEP_WGMMA,
              /*expert_slot_start=*/expert_start,
              /*dest_smem_ptr=*/&shm->a_down_wgmma[0][kk][0][0][0],
              /*bar_smem_ptr=*/&shm->bar_a[0]);
        }
      }
  #endif
    }
    // Activation-slot tail (rows [routed_count, T_TILE)) is left
    // uninitialized.  WGMMAs for ranks ≥ routed_count produce
    // garbage in calc-thread accumulators that map to `down_out`
    // columns the rank-filtered accumulate never reads:
    //   if (rank_u8 != 0xFFu) out_accum[tok][col] += down_out[col][rank_u8];
    // `rank_for_tok[tok]` only emits ranks in [0, routed_count) (or
    // sentinel 0xFF), so the garbage columns are written but never
    // read.  fp8 e4m3 has no NaN encoding, so the WGMMA can't trip
    // an exception on garbage input.

    // Activation scales for this expert were already loaded above
    // (top-of-expert per-expert loader on warp 8).  No per-K-step
    // scale load is needed: the WGMMA scale-apply indexes the
    // already-resident `a_down_scale[tok][global_half]` directly.

    // Publish the prologue's `out_accum` zero-fill + mbarrier inits and
    // the top-of-expert weight + activation scale tiles across the
    // block before the WGMMA consumers wait on bar_w[0] / bar_a[0].
    __syncthreads();

    // ── Main K-loop ─────────────────────────────────────────────────────
    //
    // Both WEIGHT and ACTIVATION tiles are loaded via TMA (mbarrier
    // wait).  Activation scales for the WHOLE expert are pre-loaded
    // at the top of the per-expert loop (see
    // `moe_load_down_wgmma_act_scale_per_expert` above), so the
    // K-loop's WGMMA scale-apply just indexes the already-resident
    // `a_down_scale[tok][global_half]` directly — no per-K-step
    // scale fetch.
    //
    // PIPELINED ACCUMULATE: at iter s=0 of every expert AFTER the
    // first one, prefetch warps run the PREVIOUS expert's
    // `out_accum += down_out[col][rank]` accumulate concurrently with
    // calc warps' WGMMA + the launcher's intra-expert prefetch.  This
    // hides the per-expert epilogue accumulate behind useful WGMMA
    // work (~1 µs/expert at BS=8) and uses the prefetch warps that
    // were otherwise idle (TMA issue is serial on lane 0; the rest of
    // warps 9..11 had nothing to do).  Hazards:
    //   * `down_out` is single-buffered.  It's written by expert e's
    //     end-of-K-loop epilogue and read by expert e+1's iter-0
    //     deferred accumulate; the writeback `__syncthreads()` and
    //     the priming-block `__syncthreads()` (both at the start of
    //     each expert) make the data visible to iter 0.  The next
    //     write to `down_out` (at end of expert e+1's K-loop) is
    //     ordered after the iter-0 sync drain, so the previous
    //     expert's accumulate is guaranteed to have completed.
    //   * `rank_for_tok` follows the same lifetime as `down_out`
    //     (computed at end of expert e, read at iter 0 of expert e+1).
    //   * The LAST expert's accumulate is performed AFTER the expert
    //     loop (one final pass before the GM writeback).
    const bool has_prev_expert = (e != down_group);
    for (std::uint32_t s = 0; s < K_TILES_DOWN; ++s) {
      const std::uint32_t read_slot = s & 1;

      // ── COMPUTE half: WGMMA + scale-apply || TMA prefetch step s+1 ──
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_DOWN
        // Wait for this step's weight tile to be fully in SHM.  The
        // launcher pre-armed bar_w[read_slot] with tx=16384 before
        // issuing the 128x128 weight TMA (priming for s=0, previous
        // compute-half for s>0).
        //
        // Tied to SKIP_PREFETCH so the wait and the launcher's arm are
        // compiled in/out together — skipping only one of them would
        // either spin forever (skip arm) or fire without a consumer
        // (skip wait).
        while (!mbarrier_try_wait_parity(&shm->bar_w[read_slot],
                                         parity_w[read_slot])) {
        }
        parity_w[read_slot] ^= 1;

        // Wait for this step's activation tile if any tokens route to
        // this expert.  When routed_count == 0 the launcher did not
        // arm bar_a[read_slot] and did not issue a TMA; the WGMMA
        // reads garbage from the un-initialized SHM slot, but every
        // thread's accumulator maps to a rank that the
        // rank-filtered accumulate never reads (see safety argument
        // at the priming block).  No wait needed.
        if (routed_count > 0u) {
          while (!mbarrier_try_wait_parity(&shm->bar_a[read_slot],
                                           parity_a[read_slot])) {
          }
          parity_a[read_slot] ^= 1;
        }
  #endif

  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
        // A descriptor bases per WG per (kk-substep, h-half).  The
        // weight tile is laid out as `K_SUBSTEPS_DOWN * DOWN_COL_HALVES`
        // contiguous 128-row M-slabs (substep 0 occupies rows
        // [0..DOWN_COL_TILE-1], substep 1 occupies rows
        // [DOWN_COL_TILE..2*DOWN_COL_TILE-1], …).  Within each slab,
        // half 0 spans the first `128 * DOWN_COL_HALVES / 2` … see the
        // priming-block comment for the row-offset formula.  WG0 owns
        // rows [0..63] of each (kk, h) atom and WG1 owns rows [64..127].
        const void* a_slot_base =
            (const void*)&shm->w_down_wgmma[read_slot][0][0];
        const std::uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;
        // Bytes between consecutive 128×128 atoms (K_STEP_WGMMA-K-wide,
        // 128-rows-tall) in `w_down_wgmma`; identical to a single TMA
        // payload size.
        constexpr std::uint32_t W_ATOM_BYTES = DOWN_W_TX_BYTES_PER_HALF;

        // Per-WGMMA K-advancement:
        //   A: 2 * A_LBO = 32 B inside the 1024-B A atom.
        //   B: 2 * B_LBO = 32 B inside the 1024-B B atom (next 2
        //     K-chunks of each token row).
        constexpr std::uint32_t A_K_STRIDE = 2u * A_LBO;
        constexpr std::uint32_t B_K_STRIDE = 2u * B_LBO;
        const void* b_slot_base =
            (const void*)&shm->a_down_wgmma[read_slot][0][0][0][0];
        // Bytes between consecutive 8 tok × 128 K SWZ128 activation
        // atoms (one per K-substep) in `a_down_wgmma[slot]`.
        constexpr std::uint32_t B_SUBSTEP_BYTES = 1024u;

          // Per-(kk, h) WGMMA passes.  Each pass runs the same
          // 4-chained-m64n8k32 structure as the legacy K_STEP_DOWN=128
          // kernel and accumulates into `chunk_d_lo[h]` / `chunk_d_hi[h]`,
          // then applies the (ws, as) scales at the K=128 boundary and
          // folds into `final_d[h]`.  Per-expert outer accumulator
          // semantics are unchanged — the substep merely adds another
          // inner level that itself runs 4 chained WGMMAs + scale apply.
    #pragma unroll
        for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
          // A new `wgmma.fence` is required at the start of every
          // group of dependent WGMMAs (one fence ↔ one
          // commit-group/wait-group pair below).  Folding the fence
          // outside the kk loop relies on a single commit+wait pair
          // covering all substeps, which would force the compiler to
          // emit a single "supergroup" of WGMMAs without
          // intermediate scale-apply ordering — incorrect.
          wgmma_fence();

          // Per-substep activation base: kk-th 1024-B SWZ128 atom.
          const void* b_kk_base =
              (const void*)((const char*)b_slot_base + kk * B_SUBSTEP_BYTES);

    #pragma unroll
          for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
            // Weight-tile atom row offset for (kk, h):
            //   row_off = (kk * DOWN_COL_HALVES + h) * 128.
            // SHM byte offset = row_off * 128 (since each row is the
            // full 128-K SWZ128 line) = (kk * HALVES + h) * 16384.
            const std::uint32_t atom_off_bytes =
                (kk * DOWN_COL_HALVES + h) * W_ATOM_BYTES;
            const void* a_base =
                (const void*)((const char*)a_slot_base + atom_off_bytes +
                              wg_offset_bytes);

              // 4 chained WGMMAs into chunk_d_lo[h]  (K[0..63], j = 0, 1).
    #pragma unroll
            for (std::uint32_t j = 0; j < 2; ++j) {
              const void* a_ptr =
                  (const void*)((const char*)a_base + j * A_K_STRIDE);
              const void* b_ptr =
                  (const void*)((const char*)b_kk_base + j * B_K_STRIDE);
              std::uint64_t desc_a =
                  make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
              std::uint64_t desc_b =
                  make_wgmma_desc(b_ptr, B_LBO, B_SBO, B_SWIZZLE);
              wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d_lo[h][0],
                                           chunk_d_lo[h][1], chunk_d_lo[h][2],
                                           chunk_d_lo[h][3]);
            }

              // 4 chained WGMMAs into chunk_d_hi[h]  (K[64..127], j = 2, 3).
    #pragma unroll
            for (std::uint32_t j = 2; j < WGMMAS_PER_SUBSTEP; ++j) {
              const void* a_ptr =
                  (const void*)((const char*)a_base + j * A_K_STRIDE);
              const void* b_ptr =
                  (const void*)((const char*)b_kk_base + j * B_K_STRIDE);
              std::uint64_t desc_a =
                  make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
              std::uint64_t desc_b =
                  make_wgmma_desc(b_ptr, B_LBO, B_SBO, B_SWIZZLE);
              wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d_hi[h][0],
                                           chunk_d_hi[h][1], chunk_d_hi[h][2],
                                           chunk_d_hi[h][3]);
            }
          }

          wgmma_commit_group();
          wgmma_wait_group<0>();

          // ── Scale-apply at the K=128 boundary (per-substep, per-half) ─
          // Each (kk, h) uses its own weight-scale entry
          // `w_down_scale[h][my_wg][ws_col]` (ws_col = global 128-K
          // index along Dims::N) and its own per-token activation
          // scale halves indexed at the (s, kk)-derived global 64-K
          // half offset in the per-expert `a_down_scale[tok][...]`.
          const std::uint32_t tok_02 = (lane % 4) * 2;
          const std::uint32_t tok_13 = (lane % 4) * 2 + 1;

          // Activation scales are now hoisted to a per-expert
          // single-buffer `a_down_scale[global_half][tok]`.  The global
          // 64-K half index for outer step `s`, substep `kk`, half
          // `half ∈ {0, 1}` is `s * K_SUBSTEPS_DOWN * 2 + kk * 2 +
          // half` (= `s * (K_STEP_DOWN/64) + 2*kk + half`).
          //
          // Layout note: the legacy `[tok][global_half]` layout caused
          // a 2-way bank conflict on these LDS sites because the row
          // stride along `tok` was 32 B = 8 banks and lane groups
          // {0,4} mapped to the same bank.  The transpose to
          // `[global_half][tok]` (matching `MoE_SHM::act_scale`) puts
          // every (tok_02 ∈ {0,2,4,6}) lookup on a distinct bank
          // within one 32-B row → conflict-free.
          const std::uint32_t base_half = s * K_SUBSTEPS_DOWN * 2u + 2u * kk;
          const float as_lo_02 = shm->a_down_scale[base_half + 0u][tok_02];
          const float as_hi_02 = shm->a_down_scale[base_half + 1u][tok_02];
          const float as_lo_13 = shm->a_down_scale[base_half + 0u][tok_13];
          const float as_hi_13 = shm->a_down_scale[base_half + 1u][tok_13];

          // Global 128-K block index along Dims::N for this (s, kk):
          //   global_kblock = (s * K_STEP_DOWN + kk * K_STEP_WGMMA) / 128
          //                 = s * K_SUBSTEPS_DOWN + kk
          const std::uint32_t ws_col =
              (W_DOWN_SCALE_COLS > 1) ? (s * K_SUBSTEPS_DOWN + kk) : 0u;

    #pragma unroll
          for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
            const float ws = shm->w_down_scale[h][my_wg][ws_col];
            final_d[h][0] += chunk_d_lo[h][0] * as_lo_02 * ws +
                             chunk_d_hi[h][0] * as_hi_02 * ws;
            final_d[h][1] += chunk_d_lo[h][1] * as_lo_13 * ws +
                             chunk_d_hi[h][1] * as_hi_13 * ws;
            final_d[h][2] += chunk_d_lo[h][2] * as_lo_02 * ws +
                             chunk_d_hi[h][2] * as_hi_02 * ws;
            final_d[h][3] += chunk_d_lo[h][3] * as_lo_13 * ws +
                             chunk_d_hi[h][3] * as_hi_13 * ws;
            chunk_d_lo[h][0] = chunk_d_lo[h][1] = chunk_d_lo[h][2] =
                chunk_d_lo[h][3] = 0.f;
            chunk_d_hi[h][0] = chunk_d_hi[h][1] = chunk_d_hi[h][2] =
                chunk_d_hi[h][3] = 0.f;
          }
        }  // end kk substep loop
  #endif
      }

      // Launcher runs IN PARALLEL with the WGMMA above.  Two cases:
      //
      //   (a) `s + 1 < K_TILES_DOWN` — intra-expert prefetch: arms the
      //       OTHER slot for the current expert's next K-step.
      //   (b) `s + 1 == K_TILES_DOWN` AND there is a next expert in
      //       this block's group — INTER-EXPERT LOOKAHEAD: arms the
      //       slot that's about to be freed (slot 0 for K_TILES_DOWN
      //       even, see static_assert below) and issues the next
      //       expert's K=0 weight + activation TMAs into it.  This
      //       keeps DRAM busy during the current expert's writeback /
      //       accumulate phase, eliminating the GPU-quiet "blue gap"
      //       at expert transitions in the NCU timeline.
      //
      // The next-expert priming block at the top of the expert loop
      // detects `e > down_group` and skips its TMA arm/issue when the
      // lookahead has already armed bar_{w,a}[0] — only the (cheap)
      // weight + activation scale cp.async loads still run.
      //
      // Compiled out under MONO_PROFILE_SKIP_PREFETCH_DOWN; the matching
      // compute-side waits on bar_{w,a}[next_slot] in the next
      // iteration are also compiled out so there is no spin-forever
      // deadlock.
      static_assert(K_TILES_DOWN % 2u == 0u,
                    "Inter-expert lookahead requires K_TILES_DOWN to be "
                    "even so the freshly-freed slot at s=K_TILES_DOWN-1 "
                    "(== (s+1)&1 == 0) is the same slot the next expert's "
                    "s=0 wait reads.  All currently instantiated Dims "
                    "variants satisfy this (Qwen3.5 N=512 with K_STEP_DOWN "
                    "in {128, 256} → K_TILES_DOWN in {4, 2}).  If this "
                    "fires for a future shape, either disable the "
                    "lookahead branch or introduce a per-expert slot "
                    "offset.");
      if (is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_DOWN
        if (s + 1 < K_TILES_DOWN) {
          const std::uint32_t next_slot = (s + 1) & 1;
          const std::uint32_t next_k_start = (s + 1) * K_STEP_DOWN;

          // Next weight tile — ONE TMA per 128-K substep (covers the
          // full DOWN_COL_TILE M-rows), same structure as the priming
          // block.  Single bar_w arm with the TOTAL tx_bytes drains
          // every atom in one wait on the compute side.
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
    #pragma unroll
          for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
            W_element* dest_base =
                &shm->w_down_wgmma[next_slot][kk * DOWN_COL_TILE][0];
            tma_load_down_wgmma_tile(
                down_weights_desc, /*expert_id=*/id,
                /*K=*/Dims::HIDDEN_STATES,
                /*base_col=*/base_col,
                /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                /*dest_smem_ptr=*/(void*)dest_base,
                /*bar_smem_ptr=*/&shm->bar_w[next_slot]);
          }

          // Next activation tile — only if any tokens route to this
          // expert.  K_SUBSTEPS_DOWN back-to-back 1024-B atoms.
          if (routed_count > 0u) {
            mbarrier_arrive_expect_tx(&shm->bar_a[next_slot],
                                      /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
    #pragma unroll
            for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
              tma_load_down_wgmma_activation_bulk(
                  down_activations_desc,
                  /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                  /*expert_slot_start=*/expert_start,
                  /*dest_smem_ptr=*/
                  &shm->a_down_wgmma[next_slot][kk][0][0][0],
                  /*bar_smem_ptr=*/&shm->bar_a[next_slot]);
            }
          }
        } else if (e + DOWN_GROUPS < expert_count) {
          // ── INTER-EXPERT LOOKAHEAD ────────────────────────────────
          //
          // Prefetch the NEXT expert's K=0 weight + activation tiles
          // into the slot that's about to be freed.  With
          // K_TILES_DOWN even, that slot is index 0 — same slot the
          // next expert's `read_slot = s & 1` lookup picks at s=0,
          // so the next expert's K-loop starts on data already in
          // SHM with the parity bookkeeping already aligned via the
          // hoisted `parity_w[]` / `parity_a[]` state.
          const std::uint32_t lookahead_slot = (s + 1) & 1;  // == 0
          const std::uint32_t next_e = e + DOWN_GROUPS;
          const std::uint32_t next_id = shmem->experts[next_e].id;
          const std::uint32_t next_routed_count =
              static_cast<std::uint32_t>(shm->expert_routed_count[next_id]);
          const std::uint32_t next_expert_start =
              static_cast<std::uint32_t>(shm->expert_slot_start[next_id]);

          mbarrier_arrive_expect_tx(&shm->bar_w[lookahead_slot],
                                    /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
    #pragma unroll
          for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
            W_element* dest_base =
                &shm->w_down_wgmma[lookahead_slot][kk * DOWN_COL_TILE][0];
            tma_load_down_wgmma_tile(
                down_weights_desc, /*expert_id=*/next_id,
                /*K=*/Dims::HIDDEN_STATES,
                /*base_col=*/base_col,
                /*k_start=*/kk * K_STEP_WGMMA,
                /*dest_smem_ptr=*/(void*)dest_base,
                /*bar_smem_ptr=*/&shm->bar_w[lookahead_slot]);
          }

          if (next_routed_count > 0u) {
            mbarrier_arrive_expect_tx(&shm->bar_a[lookahead_slot],
                                      /*tx_bytes=*/DOWN_A_TX_BYTES_TOTAL);
    #pragma unroll
            for (std::uint32_t kk = 0; kk < K_SUBSTEPS_DOWN; ++kk) {
              tma_load_down_wgmma_activation_bulk(
                  down_activations_desc,
                  /*k_start=*/kk * K_STEP_WGMMA,
                  /*expert_slot_start=*/next_expert_start,
                  /*dest_smem_ptr=*/
                  &shm->a_down_wgmma[lookahead_slot][kk][0][0][0],
                  /*bar_smem_ptr=*/&shm->bar_a[lookahead_slot]);
            }
          }
        }
  #endif
      }
      // Activation-slot tail is left uninitialized; see the matching
      // comment in the per-expert priming block above.

      // Activation scales are loaded once per expert (top of expert
      // loop) — no per-K-step scale reload.  Prefetch warps still
      // run the PREVIOUS expert's deferred accumulate at iter 0 of
      // every expert AFTER the first concurrently with calc warps'
      // iter-0 WGMMA (see the K-loop block comment above for the
      // lifetime/hazard argument).
      if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
        // Deferred accumulate of the PREVIOUS expert.  Runs on
        // prefetch warps only (warps 8..11, 128 threads), iterating
        // over the (tok, col) plane in stride-128 chunks.  Calc warps
        // run their iter-0 WGMMAs in parallel.
        //
        // Reads:
        //   * `shm->partial_result.down_out` — written by the
        //     previous expert's end-of-K-loop writeback, published by
        //     the expert-loop top __syncthreads.
        //   * `shm->rank_for_tok` — computed at the end of the
        //     previous expert (after its writeback, before the
        //     trailing __syncthreads), so it observes the same
        //     publication path.
        // Writes:
        //   * `shm->out_accum[tok][col]` — `+=` only, no contention
        //     with calc warps (they don't touch out_accum during the
        //     K-loop).
        if (has_prev_expert && s == 0u) {
          const unsigned thread_in_pf =
              thread_in_block -
              CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
          constexpr unsigned PF_THREADS =
              CoreDims::PREFETCH_WARP_COUNT * CoreDims::THREADS_PER_WARP;
          for (unsigned tok_col = thread_in_pf;
               tok_col < batch_size * DOWN_COL_TILE; tok_col += PF_THREADS) {
            const unsigned tok = tok_col / DOWN_COL_TILE;
            const unsigned col = tok_col % DOWN_COL_TILE;
            const uint8_t rank_u8 = shm->rank_for_tok[tok];
            if (rank_u8 != 0xFFu) {
              shm->out_accum[tok][col] +=
                  shm->partial_result.down_out[col][rank_u8];
            }
          }
        }
  #endif
      }

      // Align all warps before the next iteration's WGMMA reads the
      // new slot.  Activation scales were hoisted out of the K-loop
      // (see top-of-expert per-expert loader), so no pipe drain is
      // needed here — only the cross-warp `__syncthreads()`.
      __syncthreads();
    }  // end K-loop

    MONO_PHASE_TIMESTAMP_IF(t_down_after_expert0_kloop, e == down_group);

    // ── End-of-expert: write final_d → partial_result.down_out[DCT][8] ─
    //
    // Each half `h` contributes to output cols
    // `[base_col + h*128, base_col + h*128 + 127]`, which map to
    // `down_out` rows `[h*128 + wg_row_offset + warp_in_wg*16 + lane/4]`
    // (row_base) / `[row_base + 8]` for the d0..d3 halves.
    //
    // WG1 adds +64 to the row offset within its 128-row half because
    // WG1 owns output cols [h*128+64 .. h*128+127] within that half.
    if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
      const std::uint32_t wg_row_offset = is_wg1 ? 64u : 0u;
      const std::uint32_t col_base = (lane % 4) * 2;
    #pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        const std::uint32_t row_base =
            h * 128u + wg_row_offset + warp_in_wg * 16 + lane / 4;
        shm->partial_result.down_out[row_base + 0][col_base + 0] =
            final_d[h][0];
        shm->partial_result.down_out[row_base + 0][col_base + 1] =
            final_d[h][1];
        shm->partial_result.down_out[row_base + 8][col_base + 0] =
            final_d[h][2];
        shm->partial_result.down_out[row_base + 8][col_base + 1] =
            final_d[h][3];
      }
  #endif
    }

    // ── Compute rank_for_tok for the current expert ──────────────────
    //
    // Pre-compute rank for each token once per expert.  Sentinel 0xFF
    // means this token does not route through the current expert (no
    // contribution).
    //
    // The accumulate that consumes both `down_out` and `rank_for_tok`
    // is DEFERRED to the next expert's iter-0 prefetch-warp body (see
    // the K-loop block comment above).  For the LAST expert in a
    // group, an unconditional accumulate runs after the expert loop.
    //
    // 8 threads (one per token) compute the rank in parallel; the
    // remaining 376 threads idle.  Runs together with the writeback
    // above behind the same trailing __syncthreads().
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
    if (thread_in_block < batch_size) {
      const unsigned tok = thread_in_block;
      uint8_t rank_val = 0xFFu;
      for (std::uint32_t k = 0; k < top_k; ++k) {
        if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)id) {
          const std::uint32_t pair = tok * top_k + k;
          rank_val = static_cast<uint8_t>(
              static_cast<std::uint32_t>(shm->sorted_slot[pair]) -
              expert_start);
          break;
        }
      }
      shm->rank_for_tok[tok] = rank_val;
    }
  #endif
    __syncthreads();

    MONO_PHASE_TIMESTAMP_IF(t_down_after_expert0_accum, e == down_group);
  }  // end expert loop

  // ── Final accumulate for the LAST expert in this block's group ────────
  //
  // The pipelined deferred accumulate inside the K-loop covers experts
  // [down_group, down_group + DOWN_GROUPS, …, expert_count - DOWN_GROUPS).
  // The LAST expert visited (`e_last = down_group + ((expert_count -
  // 1 - down_group) / DOWN_GROUPS) * DOWN_GROUPS`) has nothing to
  // defer to, so we run one final accumulate here.  All warps
  // participate (now that the K-loop is done, the prefetch warps are
  // free to help).
  //
  // The expert-loop's trailing __syncthreads() already published this
  // block's `down_out` and `rank_for_tok`.  No additional sync
  // required before reading them.
  //
  // For blocks whose group has zero experts (expert_count <=
  // down_group), `e_started` stays false and the accumulate is
  // skipped — `down_out` was never written for this block.
  #ifndef MONO_PROFILE_SKIP_CALC_DOWN
  if (expert_count > down_group) {
    for (unsigned tok_col = thread_in_block;
         tok_col < batch_size * DOWN_COL_TILE; tok_col += blockDim.x) {
      const unsigned tok = tok_col / DOWN_COL_TILE;
      const unsigned col = tok_col % DOWN_COL_TILE;
      const uint8_t rank_u8 = shm->rank_for_tok[tok];
      if (rank_u8 != 0xFFu) {
        shm->out_accum[tok][col] += shm->partial_result.down_out[col][rank_u8];
      }
    }
  }
  __syncthreads();
  #endif

  MONO_PHASE_TIMESTAMP(t_down_after_all_experts);

  // ── After all experts in this group: atomicAdd out_accum → GM partial ─
  //
  // EXPERIMENT: instead of writing to a per-group slice
  // (`down_partial_out[down_group][tok][col]`) and reducing across
  // groups in Phase 5, every contributing block atomicAdds its
  // `out_accum` slice into the SAME single-buffer `partial[BS][HIDDEN]`
  // region (the first `BS * HIDDEN_STATES` floats of the original
  // `down_partial_out` allocation, treated as a single buffer).
  //
  // Phase 5 then reads each cell ONCE and casts to bf16 — no 16-way
  // sum.  The colstripe barrier still gates Phase 5 since all 16
  // contributing blocks must finish their atomicAdds first.
  //
  // The buffer is zero-initialized at kernel entry (see
  // `moe_kernel_topk` prologue), and the up-proj-completion barrier
  // (#2) publishes that zero across all blocks before any Phase 4
  // atomicAdd fires.
  float* gm_partial = spec->down_partial_out;  // single-buffer view
  for (unsigned idx = thread_in_block; idx < batch_size * DOWN_COL_TILE;
       idx += blockDim.x) {
    const unsigned tok = idx / DOWN_COL_TILE;
    const unsigned col = idx % DOWN_COL_TILE;
    atomicAdd(gm_partial + tok * Dims::HIDDEN_STATES + base_col + col,
              shm->out_accum[tok][col]);
  }
}

}  // namespace moe_monokernel

#endif
