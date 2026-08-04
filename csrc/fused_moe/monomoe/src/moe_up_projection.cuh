
#pragma once
#ifndef MOE_UP_PROJECTION_CU
#define MOE_UP_PROJECTION_CU

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cuda.h>
#include <cuda_fp8.h>

#include "moe_down_projection.cuh"
#include "moe_interface.h"
#include "moe_internal.h"
#include "moe_tma.h"
#include "ptx_utils.h"

// Phase 3 up-projection — `moe_up_projection_BS8_..._tma`
// (docs/design_docs/monomoe_kernel.md §1/§5/§6).  It
// streams fp8 weight tiles via a single TMA launcher (warp 8 lane 0) into
// double-buffered SHM (TinyDataWGMMA_TMA layout, see moe_internal.h) and the
// calc warps consume them via chained `wgmma.mma_async`.  Per-function and
// per-line specifics are documented inline below.

namespace monomoe {

/**
 * @brief Load this block's slice of block-wise up-projection scales into SHM.
 *
 * `base_row_up` is the first weight row (in the lower half of the 2*N rows)
 * owned by this block, i.e. `base_row_up ∈ [0, N)` with 8-row granularity.
 * This lets callers decouple the scale fetch from `blockIdx.x` — needed
 * by the two-expert-group BS8 design where both groups reuse the same
 * row-tile layout but with different blockIdx ranges.
 */
template <typename Dims>
__device__ inline void moe_request_up_scale_for_row(const S_element* __restrict__ expert_scales_up,
                                                    std::uint32_t id, unsigned base_row_up,
                                                    S_element* __restrict__ dest) {
  constexpr uint32_t COLS = Dims::UP_SCALE_COLS;  // e.g. 16 for K=2048
  constexpr uint32_t TILE = 2 * COLS;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // Only the first prefetch warp loads the scales — 32 scalars total for
  // Qwen3.5 (2×16). Synchronous shared-memory writes are fine; we don't
  // need async copy for 128 bytes.
  if (warp == 0 && thread < TILE) {
    uint32_t rb_local = thread / COLS;  // 0 → low half, 1 → upper half
    uint32_t kb = thread % COLS;
    uint32_t row = base_row_up + rb_local * Dims::N;
    uint32_t rb_global = row / Dims::BLOCK_SCALE_ROW;
    dest[thread] = expert_scales_up[id * Dims::UP_SCALE_ROWS * Dims::UP_SCALE_COLS +
                                    rb_global * Dims::UP_SCALE_COLS + kb];
  }
}

///////////////////////////////////////////////////////////////////////////////
//
// moe_up_projection_BS8_allexperts_wgmma_tma
//
// TMA+WGMMA up-projection for BS<=8. Only variant of the BS8 up-proj
// kernel: the cp.async reference path has been removed. Replaces the
// prefetch-warp `cp.async` loaders for the fp8 expert-weight tile with
// `cp.async.bulk.tensor.2d` issued by a single TMA launcher thread
// (warp 8, lane 0). Completion is signalled via SHM mbarrier `bar_w[2]`;
// consumer warps wait with `mbarrier.try_wait.parity` instead of
// `cuda::pipeline_consumer_wait_prior`.  The activation operand is
// sourced from `fp8_act_full` (populated by Phase 1 + Phase 2 of
// `moe_kernel_topk_BS8`); no per-K-step bf16-input TMA or `bar_a` arm
// fires from this helper.
//
// Descriptors are built host-side in the torch binding wrapper and passed
// to the top-level kernel as `__grid_constant__ CUtensorMap const`
// parameters. At the device-side inlined helper boundary (this function),
// they appear as `CUtensorMap const&` — the `__grid_constant__` qualifier
// only applies at the kernel-function boundary.
//

// ── Per-(token, lane) SiLU + fp8 quantization + writeback body ─────────
//
// Factored out of the up-proj epilogue so the same body can run
// (a) on calc warps inline for the LAST expert in a block's per-expert
// loop (no future iter-0 to defer to) and (b) on prefetch warps as
// "deferred work for the previous expert" at iter-0 of the next
// expert, overlapped with calc warps' WGMMA.  The two paths differ
// only in the (warp → tok) mapping, not in the per-(tok, lane) work.
//
// Caller contract:
//   * `wgmma_out` MUST have been written by the producing expert's
//     end-of-K-loop final_d store and published to the calling warps.
//     - Calc-warp inline path (last expert): the `__syncthreads()`
//       AFTER the wgmma_out write covers this.
//     - Deferred prefetch path (non-last experts): the same sync —
//       which sits at the bottom of expert e's loop body — also gates
//       iter-0 of expert e+1 where this helper is invoked.
//   * `lane` MUST be in [0, 32); `tok` MUST be in [0, BS).  Out-of-
//     range tokens must be filtered by the caller.
//   * All 32 lanes of the warp MUST call this in lockstep with the
//     same `tok` so the warp-reduce in the body sees all 64 cols.
//   * `id` is the producing expert's id (the one whose final_d sits
//     in `wgmma_out`), NOT necessarily the current iter's expert.
//
// Hazard: writes only `spec->temp_fp8` / `spec->temp_act_scale` (GM)
// and reads `wgmma_out` + `shmem->topk_ids_flat` + `shm->sorted_slot`
// (SHM).  No writes to SHM that calc warps touch in the K-loop, so
// the prefetch-warp variant runs concurrently with calc warps' iter-0
// WGMMA without contention.
template <typename Dims>
__device__ __forceinline__ void up_silu_quant_writeback_one_token(
    MoE_SHM<Dims>* __restrict__ shmem, MoEGemmSpec<Dims>* __restrict__ spec,
    typename MoE_SHM<Dims>::TinyDataWGMMA_TMA* __restrict__ shm, std::uint32_t id,
    std::uint32_t tok, std::uint32_t col_in_half, std::uint32_t lane, std::uint32_t base_row_up,
    std::uint32_t effective_bid, std::uint32_t top_k, std::uint32_t batch_size) {
  constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // Uniform-across-warp: find this token's top-K match for the
  // current expert.  All 32 lanes of the warp agree on these.
  bool store = false;
  float rw = 0.f;
  std::uint32_t dest_row = 0;
  if (tok < batch_size) {
    // Single lookup into the routing-time inverse map
    // `expert_tok_krank[id][tok]` (built once by `prepare_moe_topk_BS8`,
    // see moe_internal.h) — replaces the per-expert 8-iter scan over
    // `topk_ids_flat` that both the inline and deferred paths used to
    // run.  A valid `k` yields the routing weight and the fp8 writeback
    // row; sentinel 0xFF means this token does not route to expert `id`.
    const uint8_t k = shm->expert_tok_krank[id * Dims::BS + tok];
    if (k != 0xFFu) {
      store = true;
      rw = shmem->topk_weights_flat[tok * MAX_TOPK + k];
      const std::uint32_t pair = tok * top_k + k;
      dest_row = shm->sorted_slot[pair];
    }
  }

  // Compute val1 / val2 on every lane of the warp (needed so that the
  // subsequent warp-reduce sees all 64 cols of this up-block).
  const float gate1 = shm->partial_result.wgmma_out[col_in_half][tok];
  const float up1 = shm->partial_result.wgmma_out[col_in_half + 32][tok];
  const float gate2 = shm->partial_result.wgmma_out[col_in_half + 64][tok];
  const float up2 = shm->partial_result.wgmma_out[col_in_half + 96][tok];

  float val1 = rw * up1 * gate1 / (1.0f + __expf(-gate1));
  float val2 = rw * up2 * gate2 / (1.0f + __expf(-gate2));

  const std::uint32_t out_col_1 = base_row_up + col_in_half;
  const std::uint32_t out_col_2 = base_row_up + 32 + col_in_half;
  const bool write1 = store && (out_col_1 < Dims::N);
  const bool write2 = store && (out_col_2 < Dims::N);

  if (!write1) val1 = 0.f;
  if (!write2) val2 = 0.f;

  // Warp-reduce max(|val1|, |val2|) across the 32 lanes → max over all
  // 64 output cols of this up-block for this token.
  float local_max = fmaxf(fabsf(val1), fabsf(val2));
  float block_max = warp_reduce_max_float(local_max);
  if (block_max < __FLT_MIN__) block_max = 1.0f;

  constexpr float FP8_MAX = 448.0f;
  constexpr float FP8_MAX_INV = 1.0f / 448.0f;
  const float block_scale = block_max * FP8_MAX_INV;
  const float inv_scale = FP8_MAX / block_max;

  const AQ_element q1 = (AQ_element)(val1 * inv_scale);
  const AQ_element q2 = (AQ_element)(val2 * inv_scale);

  if (store && tok < batch_size) {
    if (write1) {
      spec->temp_fp8[dest_row * Dims::N + out_col_1] = q1;
    }
    if (write2) {
      spec->temp_fp8[dest_row * Dims::N + out_col_2] = q2;
    }
    if (lane == 0) {
      constexpr std::uint32_t SCALE_COLS =
          MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;  // = Dims::N / 64
      spec->temp_act_scale[dest_row * SCALE_COLS + effective_bid] = block_scale;
    }
  }
}

template <typename Dims>
__device__ inline void moe_up_projection_BS8_allexperts_wgmma_tma(
    const A_element* __restrict__ activations_in, const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up, std::uint32_t top_k, std::uint32_t batch_size,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& up_weights_desc, CUtensorMap const& activations_desc,
    std::uint32_t up_block_idx = 0xffffffffu, std::uint32_t expert_start = 0,
    std::uint32_t expert_stride = 1) {
  static_assert(Dims::BS <= 8);
  using CoreDims = MoECoreDims<Dims>;

  // `activations_in` / `expert_weights_up` are retained on the parameter
  // list for signature parity with the `cp.async` reference but are not
  // dereferenced on the TMA path — all GM reads go through the two TMA
  // descriptors.
  (void)activations_in;
  (void)expert_weights_up;

  // Caller contract:
  //   * The Phase-1 routing-window TMA (in `moe.cuh`) has already
  //     fetched the full BF16 input tile into `bf16_in_full`.
  //   * Phase 2 has run `routing_phase_quantize` and a block-wide
  //     `__syncthreads()`, so `fp8_act_full[k_block]` and
  //     `act_scale[token][k_block]` are visible to every calc warp
  //     for `k_block in [0, K_BLOCKS_TOTAL)`.
  //   * `bar_w[0..1]` have been initialized and release-fenced.
  //
  // Stage A pipeline:
  //   * Pre-loop: helper arms bar_w[0] + TMAs w[0] of expert_start at k=0.
  //   * K-loop: iter s waits bar_w[s%2], runs WGMMA +
  //             scale-apply (B operand from `fp8_act_full`), and the
  //             launcher arms the NEXT slot's bar_w + weight TMAs
  //             (intra-expert s+1 or next expert's k=0 stitch).
  //   * No bar_a, no bf16 TMAs, no QUANT half, no QUANT/COMPUTE
  //     __syncthreads() — the activation operand is already in
  //     `fp8_act_full` for the entire K range.

  // ── Compile-time constants (v1) ─────────────────────────────────────
  // Byte-for-byte mirror of the `cp.async` reference variant's constants
  // so that SHM layouts, WGMMA descriptors, and K-step sizing remain
  // identical across the two paths (design P1/P2).
  constexpr uint32_t W_UP_M = CoreDims::W_UP_TILE_WGMMA;              // 128
  constexpr uint32_t K_STEP_WGMMA = CoreDims::K_STEP_WGMMA;           // 128
  constexpr uint32_t K_STEP = CoreDims::K_STEP_UP;                    // 128 / 256
  constexpr uint32_t K_SUBSTEPS = CoreDims::K_SUBSTEPS_UP;            // 1 / 2
  constexpr uint32_t K_TILES = CoreDims::K_TILES_UP;                  // K/K_STEP
  constexpr uint32_t WGMMAS_PER_SUBSTEP = CoreDims::WGMMAS_PER_STEP;  // 4
  constexpr uint32_t UP_SCALE_COLS = Dims::UP_SCALE_COLS;             // 16

  // Descriptor strides for 128×128 Major::K B128-swizzled A operand.
  //
  // The TMA hardware applies the 8-row × 128-byte core-matrix XOR
  // swizzle at write time, so each 1024-B atom holds one 8-row M-block.
  // CUTLASS Major::K B128 layout:
  //   LBO = 16 B   (one K-core-matrix within the 1024-B atom)
  //   SBO = 1024 B (next M-block atom)
  //   swizzle_mode = 1
  // The Python pre-interleave repacks gate/up row stripes so that a
  // single 128x128 TMA fetches the full WGMMA A-tile; it does NOT
  // apply the canonical core-matrix byte permutation (TMA does that).
  constexpr uint64_t A_LBO = 16ULL;
  constexpr uint64_t A_SBO = 1024ULL;
  constexpr uint32_t A_SWIZZLE = 1u;
  // B operand (K-major, N=8): 1 N-block, LBO between K-core-matrices.
  // Always SWIZZLE_NONE — the activation tile is 8-token × 128-K bf16
  // and small enough that bank-conflict cost is bounded.
  //
  // `B_LBO` = bytes between successive 8-row × 16-byte WGMMA core
  // matrices along K = the byte stride between successive kc atoms in
  // `fp8_act_full`'s `[FP8_NUM_CHUNKS][T_TILE_PADDED][FP8_K_CHUNK]`
  // per-kblk layout (see comment on `fp8_act_full` in `moe_internal.h`
  // for the kc-padding design).  The pad widens each kc atom from 128
  // B (T_TILE=8) to 144 B (T_TILE_PADDED=9), and the 9th token row of
  // every kc atom is unused — the WGMMA core matrix is still rows
  // [0..7] × bytes [0..15] (= 128 B contiguous) at the head of each
  // atom, and `B_LBO = 144` steps over the unused 9th row to land on
  // the next kc atom's core matrix.
  constexpr uint64_t B_LBO =
      static_cast<uint64_t>(MoE_SHM<Dims>::TinyDataWGMMA_TMA::FP8_ACT_T_TILE_PADDED) *
      static_cast<uint64_t>(MoE_SHM<Dims>::TinyDataWGMMA_TMA::FP8_ACT_K_CHUNK);  // 9 × 16 =
                                                                                 // 144 B
  constexpr uint64_t B_SBO = B_LBO;  // unused (only 1 N-block for N=8)

  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;  // 0..11
  const unsigned lane = thread_in_block & 31;
  const bool is_wg0 = (warp < 4);
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;  // 0..3 within each WG
  // Gate/up split within each WG: warps 0,1 (in-WG) → gate rows [0..31];
  // warps 2,3 (in-WG) → up rows [32..63] (within the WG's 64-row stripe).
  const bool is_gate_half = (warp_in_wg < 2);
  (void)thread_in_block;
  (void)is_wg0;

  // TMA path uses the `tiny_wgmma_tma` union variant (byte-identical to
  // `tiny_wgmma` plus 32 B of mbarriers at the tail).
  auto* shm = &shmem->tiny_wgmma_tma;

  const unsigned effective_bid = (up_block_idx == 0xffffffffu) ? blockIdx.x : up_block_idx;
  // Each block owns 128 M rows = 2 WG stripes × 64 rows.  WG0's gate
  // rows start at base_row_up; WG1's gate rows start at base_row_up + 32.
  const unsigned base_row_up = effective_bid * (W_UP_M / 2);
  const std::uint32_t expert_count = shmem->expert_count;

  // Per-thread fp32 accumulators for WGMMA m64n8k32.
  // WG0 and WG1 threads each hold their own 4-register accumulator for
  // their respective M stripe.
  float chunk_d0 = 0.f, chunk_d1 = 0.f, chunk_d2 = 0.f, chunk_d3 = 0.f;
  float final_d0 = 0.f, final_d1 = 0.f, final_d2 = 0.f, final_d3 = 0.f;

  // ── Phase-3 preamble ──────────────────────────────────────────────────
  //
  // Entry contract (post-fusion):
  //   * `bar_w[0..1]` are initialized (arrival_count=1) and
  //     release-fenced by the kernel prologue in `moe.cuh`.
  //   * Phase 1 + Phase 2 have populated `bf16_in_full` and then
  //     `fp8_act_full` + `act_scale[token][k_block]` for every
  //     `k_block ∈ [0, K_BLOCKS_TOTAL)`; the Phase-2 trailing
  //     `__syncthreads()` published those writes to all warps.
  //   * `bar_a[0..1]` are NOT initialized by `moe.cuh`'s prologue —
  //     they are reused by the down-projection in Phase 4 and
  //     re-initialized in the down-proj prologue.
  //
  // This helper never re-initializes barriers and never issues
  // bf16-input TMAs; the K-loop reads FP8 activations directly from
  // `fp8_act_full`.

  // Stage A requires an even K_TILES so the end-of-K-loop launcher arm
  // lands on `next_slot = K_TILES % 2 = 0`.  That slot-0 stitch is what
  // the next expert's iter-0 COMPUTE waits on; an odd K_TILES would
  // land the stitch on slot 1, breaking the cross-expert mbarrier
  // chain.  At K_STEP_UP=128 (default) HIDDEN_STATES=2048 → K_TILES=16;
  // at K_STEP_UP=256 → K_TILES=8; both satisfy the invariant.
  static_assert(K_TILES % 2 == 0,
                "Stage-A pipeline requires K_TILES to be even so the "
                "end-of-loop stitch arms the same slot that the next "
                "expert's iter-0 COMPUTE waits on.");

  // ── Pre-loop: arm bar_w[0] + TMA w[0] of expert_start at k=0 ──────────
  //
  // bar_w[0] is not pre-armed by the caller; this helper fires the
  // first expert's weight TMA here.  iter-0 COMPUTE waits on
  // bar_w[0] before issuing WGMMAs.
  // For subsequent experts inside the same helper invocation, the
  // previous expert's K-loop stitch (at s=K_TILES-1 COMPUTE) arms
  // bar_w[0] + TMAs w[0] of the next expert.  No pre-loop work there.
  constexpr uint32_t UP_W_TX_BYTES_PER_SUBSTEP = 16384u;  // 128×128 fp8 atom
  constexpr uint32_t UP_W_TX_BYTES_TOTAL = UP_W_TX_BYTES_PER_SUBSTEP * K_SUBSTEPS;  // 16 KB / 32 KB
  // The bf16-input TMA + `bar_a` arm have been removed from this
  // helper.  Phase-1 + Phase-2 in `moe.cuh` populate
  // `fp8_act_full` once per kernel invocation; the K-loop reads it
  // directly.  The legacy `UP_A_TX_BYTES_*` constants live in the
  // kernel prologue for now (until a later cleanup removes the legacy
  // hoisted Step A entirely).
  if (is_tma_launcher_thread<Dims>() && expert_start < expert_count) {
    const uint32_t first_id = shmem->experts[expert_start].id;
    mbarrier_arrive_expect_tx(&shm->bar_w[0],
                              /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
#pragma unroll
    for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
      tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/first_id,
                             /*N=*/Dims::N,
                             /*base_row_up=*/base_row_up,
                             /*k_start=*/kk * K_STEP_WGMMA,
                             /*dest_slot=*/&shm->w_wgmma[0][kk * W_UP_M][0],
                             /*bar=*/&shm->bar_w[0]);
    }
#ifdef MONO_PROFILE_BARW_4DEEP
    // 2-deep lookahead variant: also pre-arm bar_w[1] with
    // expert_start's iter-1 weight tile, so iter-1's calc-warp
    // wait doesn't have to wait for the launcher to issue the TMA
    // from cold.  This stitches the lookahead 1 iter earlier than
    // the default pipeline: the launcher starts at iter 0 arming
    // bar_w[2] for iter 2, then the wraparound is 4-deep instead
    // of 2-deep.
    static_assert(K_TILES >= 4,
                  "BARW_4DEEP requires K_TILES >= 4 to fit a 2-deep lookahead "
                  "without wrap-around collisions on bar_w[4].");
    mbarrier_arrive_expect_tx(&shm->bar_w[1],
                              /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
#pragma unroll
    for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
      tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/first_id,
                             /*N=*/Dims::N,
                             /*base_row_up=*/base_row_up,
                             /*k_start=*/K_STEP + kk * K_STEP_WGMMA,
                             /*dest_slot=*/&shm->w_wgmma[1][kk * W_UP_M][0],
                             /*bar=*/&shm->bar_w[1]);
    }
#endif  // MONO_PROFILE_BARW_4DEEP
  }

  // ── Deferred-writeback bookkeeping ────────────────────────────────────
  //
  // Phase-3's per-expert SiLU+fp8 quant writeback runs in two modes:
  //
  //   1. INLINE (default, when the macro below is undefined): runs
  //      on calc warps at the bottom of each expert iteration with
  //      two surrounding __syncthreads, exactly as before.  The
  //      timestamps `t_up_e0_iter0_after_*`, `t_up_e0_iter1_after_*`,
  //      `t_up_e0_after_expert0_*` capture this path's cost.
  //
  //   2. DEFERRED (when MONO_PROFILE_DEFER_UP_EPILOGUE is defined):
  //      the writeback is moved to iter `s == 0` of the next expert,
  //      handled by prefetch warps (8..11), so calc warps' iter-0
  //      WGMMAs run concurrently with the previous expert's
  //      SiLU+quant.  The LAST expert in the per-block range has no
  //      next iter-0 to defer to and runs its writeback inline on
  //      calc warps after the expert loop ends.
  //
  // The macro lets us A/B-test the two paths against the same
  // phase-timing instrumentation so we can attribute every µs in
  // the cross-expert window to a specific stage.
  //
  // `prev_id_for_writeback` and `has_pending_writeback` are uniform
  // across threads (the loop runs in lockstep) and only matter under
  // the deferred path.
  uint32_t prev_id_for_writeback = 0;
  bool has_pending_writeback = false;
  (void)prev_id_for_writeback;
  (void)has_pending_writeback;

  // ── Phase-3 expert loop ───────────────────────────────────────────────
  for (uint32_t e = expert_start; e < expert_count; e += expert_stride) {
    const uint32_t id = shmem->experts[e].id;
    const bool has_next_e = (e + expert_stride < expert_count);
    const uint32_t next_id = has_next_e ? shmem->experts[e + expert_stride].id : 0u;

    // Per-expert parity state.  bar_w[0] is always pre-armed at the
    // start of each expert (by the helper pre-loop above for
    // expert_start, or by the prior expert's stitch for subsequent
    // experts), so register 0 correctly expects physical 1 on
    // the first try_wait.parity.  bar_w[1] is first armed inside
    // this expert's iter 0 COMPUTE, so register 0 expects physical 1
    // on iter 1's first wait.
#ifdef MONO_PROFILE_BARW_4DEEP
    uint32_t parity_w[4] = {0, 0, 0, 0};
#else
    uint32_t parity_w[2] = {0, 0};
#endif

    // Reset per-expert accumulators.
    final_d0 = final_d1 = final_d2 = final_d3 = 0.f;

    // Load this expert's block-wise weight scales.  Scales cover 2
    // row-blocks (gate + up) × UP_SCALE_COLS col-blocks.  Both WGs share
    // the same scales (see moe_internal.h comment on UP_SCALE_TILE_SIZE).
    // Synchronous SHM write (32 elements) by prefetch warp 0; the
    // iter-0 QUANT→COMPUTE sync below publishes it to calc warps.
    if (is_prefetch_warp<Dims>()) {
      moe_request_up_scale_for_row<Dims>(expert_scales_up, id, base_row_up, shm->up_scale[0]);
    }

    // Publish the per-expert `up_scale` write from the prefetch
    // warp (above) to the calc warps that consume it inside the
    // K-loop scale-apply.  The legacy QUANT/COMPUTE sync used to
    // serve this purpose; it has been removed.  This sync sits OUTSIDE
    // the K-loop, so the "at most one __syncthreads() per outer K-step
    // iteration" invariant is preserved.
    __syncthreads();

    // ── Main K-loop (FP8-direct: COMPUTE-only, mbarrier-only sync) ─────
    //
    // Pipeline per iteration:
    //   COMPUTE half:
    //     calc:      wait bar_w[s%2]; K_SUBSTEPS × (4× WGMMA + scale-apply).
    //                B operand reads `fp8_act_full[s * K_SUBSTEPS + kk]`
    //                — single-buffer FP8 produced once per kernel
    //                invocation by Phase 2's `routing_phase_quantize`
    //                and published by the Phase-2 trailing
    //                `__syncthreads()` in `moe.cuh`.
    //     launcher:  arm + TMA the NEXT slot's K_SUBSTEPS weight atoms
    //                (UP_W_TX_BYTES_TOTAL bytes total).  No bar_a arm
    //                and no bf16-input TMA.
    //                target = (s+1, current expert) for intra-expert
    //                         steps, or (0, next expert) on the last
    //                         step when a next expert is scheduled.
    //                When no next step and no next expert, skip — the
    //                trailing barriers are left idle; Phase 4 reinits.
    //   No QUANT half, no QUANT/COMPUTE __syncthreads().  The next
    //   iter's COMPUTE wait on bar_w[next_slot] re-establishes acquire
    //   ordering for the weight TMA's async writes.
    for (uint32_t s = 0; s < K_TILES; ++s) {
#ifdef MONO_PROFILE_BARW_4DEEP
      // 2-deep lookahead: launcher arms `bar_w[(s+2) & 3]` and
      // calc waits on `bar_w[s & 3]`.  Wraparound is 4 iters; the
      // launcher's arm at iter `s+2` lands 2 iters before the
      // matching consumer wait, giving DRAM extra time to drain
      // the cross-expert stitch and the iter-1 weight TMA.
      const uint32_t cur_slot = s & 3u;
      const uint32_t next_slot = (s + 2u) & 3u;
      const bool has_next_s = (s + 2u < K_TILES);
#else
      const uint32_t cur_slot = s & 1;
      const uint32_t next_slot = (s + 1) & 1;
      const bool has_next_s = (s + 1 < K_TILES);
#endif

      // ───── COMPUTE half ─────────────────────────────────────────────
      if (is_calc) {
        // Wait on weight tile arrival.  Compiled in/out together with
        // the launcher's arm; under SKIP_PREFETCH the launcher elides
        // the arm so skipping the wait avoids a spin-forever deadlock.
        while (!mbarrier_try_wait_parity(&shm->bar_w[cur_slot], parity_w[cur_slot])) {
        }
        parity_w[cur_slot] ^= 1;

        // Phase-timing: after-wait on iter 0 / iter 1 of expert 0 and
        // expert 1 (calc warp 0 lane 0 = threadIdx.x == 0).

        // WGMMA descriptor bases per WG.  In the M-stacked SHM layout,
        // substep `kk` occupies SHM rows `[kk*128 .. kk*128 + 128)`,
        // so the WG row offset (0 for WG0, 64 for WG1) is added on top
        // of `kk * 128` to pick the per-WG 64-row half within each
        // 128-row substep atom.
        // WG0 substep 0: rows [0..63]    → &w_wgmma[slot][0][0]
        // WG1 substep 0: rows [64..127]  → &w_wgmma[slot][64][0]
        // WG0 substep 1: rows [128..191] → &w_wgmma[slot][128][0]
        // WG1 substep 1: rows [192..255] → &w_wgmma[slot][192][0]
        const void* a_slot_base = (const void*)&shm->w_wgmma[cur_slot][0][0];
        const uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;
        // Bytes between consecutive 128-row substep atoms in
        // `w_wgmma[slot]`: 128 rows × 128 K-bytes = 16 KB.
        constexpr uint32_t K_SUBSTEP_W_BYTES = 16384u;

        // Per-substep activation base: B operand reads
        // `fp8_act_full[s * K_SUBSTEPS + kk]` (single buffer covering
        // all K substeps; produced once per kernel invocation by
        // Phase 2's `routing_phase_quantize`).

        // Chain 4 WGMMAs per K-substep, each consuming K=32 (= 2
        // consecutive K-chunks of 16 from the fp8 activation tile).
        // Scales are applied at every K=128 boundary (matching the
        // block-wise FP8 scale granularity).
        constexpr uint32_t A_K_STRIDE = 2u * static_cast<uint32_t>(A_LBO);
#pragma unroll
        for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
          // A new `wgmma.fence` is required at the start of every group
          // of dependent WGMMAs (one fence ↔ one commit-group/wait-group
          // pair below).
          wgmma_fence();

          // Per-substep weight base: kk-th 128-row substep atom + this
          // WG's 64-row half within the atom.
          const void* a_kk_base =
              (const void*)((const char*)a_slot_base + kk * K_SUBSTEP_W_BYTES + wg_offset_bytes);

          // Single-buffer activation atom for this (s, kk):
          //   fp8_act_full[s * K_SUBSTEPS + kk][...]
          // Replaces the legacy `fp8_act[cur_slot][kk]` indexing.
          // `cur_slot` is unused for the activation
          // operand and remains in scope only for the weight tile.
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

          // ── Scale-apply at the K=128 boundary (per-substep) ──────────
          //
          // Scale indices for outer step `s`, substep `kk`:
          //   * activation: `act_scale[tok][s * K_SUBSTEPS + kk]`
          //   * weight:     `up_scale[0][s * K_SUBSTEPS + kk + ws_off]`
          // Indexing matches the legacy form; the values
          // are now produced by `routing_phase_quantize` instead of
          // by the per-K-step `moe_streaming_quantize_k128` call.
          const uint32_t ws_off = is_gate_half ? 0u : UP_SCALE_COLS;
          const float ws = shm->up_scale[0][kblk + ws_off];
          const uint32_t tok_02 = (lane % 4) * 2;
          const uint32_t tok_13 = tok_02 + 1;
          // SHM `act_scale` is laid out as `[blk][tok]` (see comment on
          // its declaration in `MoE_SHM`); the index swap from the
          // legacy `[tok][blk]` form is cosmetic at the source level
          // but eliminates the 4-way bank conflict NCU flagged on
          // these LDS sites.
          const float as_02 = shmem->act_scale[kblk][tok_02];
          const float as_13 = shmem->act_scale[kblk][tok_13];
          final_d0 += chunk_d0 * ws * as_02;
          final_d1 += chunk_d1 * ws * as_13;
          final_d2 += chunk_d2 * ws * as_02;
          final_d3 += chunk_d3 * ws * as_13;
          chunk_d0 = chunk_d1 = chunk_d2 = chunk_d3 = 0.f;
        }
      }

      // Phase-timing: after-compute on iter 0 / iter 1 of expert 0
      // and expert 1.  Outside the calc-only block so the macro's
      // `is_calc` gating doesn't suppress threadIdx.x == 0 — but
      // threadIdx.x == 0 is itself in calc, so the capture lands
      // at the same point either way.

      // Launcher runs IN PARALLEL with the WGMMA above.  Only the
      // weight TMA + bar_w arm remain; the bf16-input TMA + bar_a
      // arm have been removed.  The activation operand is
      // sourced from `fp8_act_full`, which is produced once per
      // kernel invocation by Phase 2 — there is nothing to fetch
      // per K-step on the activation side.
      //
      // For K_STEP > K_STEP_WGMMA the launcher issues K_SUBSTEPS_UP
      // back-to-back weight TMAs per slot (one per 128-K substep,
      // stacked along the K axis in SHM).  bar_w is armed once with
      // the TOTAL tx_bytes so a single `mbarrier.try_wait.parity` on
      // the calc side drains all atoms.
      if (is_tma_launcher_thread<Dims>()) {
#ifdef MONO_PROFILE_BARW_4DEEP
        // 2-deep lookahead: at iter `s` arm `bar_w[(s+2)&3]` for
        // the iter-(s+2) weight tile.  Three cases by source:
        //   (A) Intra-expert: s+2 < K_TILES → fetch CURRENT expert's
        //                     iter-(s+2) tile.
        //   (B) Cross-expert iter-0: s+2 == K_TILES (i.e. s ==
        //                     K_TILES-2) AND has_next_e → fetch
        //                     NEXT expert's iter-0 tile.
        //   (C) Cross-expert iter-1: s+2 == K_TILES+1 (i.e. s ==
        //                     K_TILES-1) AND has_next_e → fetch
        //                     NEXT expert's iter-1 tile.
        //   Else: idle.
        //
        // Cases (B) and (C) together replace the single-deep
        // "stitch" from the original pipeline; they pre-load both
        // iter-0 AND iter-1 of the next expert during the current
        // expert's last two K-iters.  The matching pre-loop in the
        // helper does the same for the first expert.
        if (has_next_s) {
          // Case (A): intra-expert fetch of (s+2)-th tile.
          const uint32_t next_k_start = (s + 2u) * K_STEP;
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
#pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/id,
                                   /*N=*/Dims::N,
                                   /*base_row_up=*/base_row_up,
                                   /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                                   /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                                   /*bar=*/&shm->bar_w[next_slot]);
          }
        } else if (has_next_e) {
          // Cases (B)/(C): cross-expert stitch.
          //   At s == K_TILES-2: fetch next expert's iter-0.
          //   At s == K_TILES-1: fetch next expert's iter-1.
          const uint32_t next_e_iter = (s == K_TILES - 2u) ? 0u : 1u;
          const uint32_t next_e_k_start = next_e_iter * K_STEP;
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
#pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/next_id,
                                   /*N=*/Dims::N,
                                   /*base_row_up=*/base_row_up,
                                   /*k_start=*/next_e_k_start + kk * K_STEP_WGMMA,
                                   /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                                   /*bar=*/&shm->bar_w[next_slot]);
          }
        }
        // Else: last expert, last two iters — leave barriers idle.
#else  // !MONO_PROFILE_BARW_4DEEP
        if (has_next_s) {
          // Intra-expert: fetch (s+1) tile of the CURRENT expert.
          const uint32_t next_k_start = (s + 1) * K_STEP;
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
#pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/id,
                                   /*N=*/Dims::N,
                                   /*base_row_up=*/base_row_up,
                                   /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                                   /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                                   /*bar=*/&shm->bar_w[next_slot]);
          }
        } else if (has_next_e) {
          // End-of-expert stitch: fetch iter-0 weight tile of the
          // NEXT expert.  For K_TILES even, `next_slot == 0` —
          // matches the next expert's iter-0 cur_slot.
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
#pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/next_id,
                                   /*N=*/Dims::N,
                                   /*base_row_up=*/base_row_up,
                                   /*k_start=*/kk * K_STEP_WGMMA,
                                   /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                                   /*bar=*/&shm->bar_w[next_slot]);
          }
        }
        // Else: last expert's last iteration — leave barriers idle.
#endif  // MONO_PROFILE_BARW_4DEEP
      }

#ifdef MONO_PROFILE_DEFER_UP_EPILOGUE
      // ── Deferred SiLU + fp8 quant writeback for the PREVIOUS expert ──
      //
      // Runs on prefetch warps (8..11) of every expert AFTER the first.
      // Scheduling goal (two constraints together):
      //   (1) FULLY USE THE 4 PREFETCH WARPS whenever the epilogue runs —
      //       one warp per token (32 lanes cover a token's 64 output cols
      //       via 4 gate/up pairs, and `warp_reduce_max_float` for the
      //       fp8 block-scale needs the whole warp), so a "wave" is
      //       PF = PREFETCH_WARP_COUNT = 4 tokens processed in lockstep.
      //   (2) DISTRIBUTE THE WAVES ACROSS K-STEPS — the BS tokens split
      //       into `WAVES = ceil(BS / PF)` waves, each fired at ONE
      //       K-loop iteration, spaced evenly over the first
      //       `DEFER_ITERS = K_TILES - 1` iterations (the LAST iteration,
      //       s == K_TILES-1, is left free — it carries the cross-expert
      //       weight-TMA stitch on the launcher thread).
      //
      // Wave w fires at iteration `s_w = w * DEFER_ITERS / WAVES`; within
      // it prefetch warp 8+i handles token `w*PF + i`.  Examples:
      //   * BS=8,  K_TILES=8  → WAVES=2, waves at s=0,3 (4 warps each).
      //   * BS=16, K_TILES=8  → WAVES=4, waves at s=0,1,3,5 (4 warps each,
      //                         four iterations — the requested layout).
      // vs. the old burst (all waves back-to-back at s=0,1): same 4-warp
      // width, but spacing the waves lets DRAM drain the `temp_fp8` store
      // burst between waves and overlaps each with more calc-warp WGMMA.
      //
      // `WAVES <= DEFER_ITERS` guarantees every wave lands on its own
      // iteration (distinct `s_w`), so the helper is called at most once
      // per warp per iteration — no double-processing.
      //
      // Numerically identical to any schedule: `wgmma_out` holds the
      // PREVIOUS expert's data for the entire current K-loop (the current
      // expert overwrites it only AFTER the loop), and `expert_tok_krank`
      // / `sorted_slot` / `topk_weights_flat` are routing-time-immutable,
      // so which iteration a token is processed at does not change the
      // result.
      static_assert(K_TILES >= 2, "Deferred up-proj writeback requires K_TILES >= 2.");
      constexpr uint32_t PF = CoreDims::PREFETCH_WARP_COUNT;  // 4
      constexpr uint32_t WAVES = (Dims::BS + PF - 1u) / PF;   // ceil(BS/PF)
      constexpr uint32_t DEFER_ITERS = K_TILES - 1u;
      static_assert(WAVES <= DEFER_ITERS,
                    "Not enough K-loop iterations to give each up-proj epilogue "
                    "wave its own K-step (need ceil(BS/4) <= K_TILES-1).");
      if (has_pending_writeback && is_prefetch_warp<Dims>() && s < DEFER_ITERS) {
        const unsigned pf_warp = warp - CoreDims::CALC_WARP_COUNT;  // 0..3
        const uint32_t col_in_half = lane;                          // 0..31
#pragma unroll
        for (uint32_t w = 0; w < WAVES; ++w) {
          // Uniform across the block (s) and the warp (w, pf_warp are
          // lane-independent), so all 32 lanes stay in lockstep for the
          // warp-reduce inside the helper.
          if (s == w * DEFER_ITERS / WAVES) {
            const uint32_t tok = w * PF + pf_warp;
            if (tok < Dims::BS) {
              up_silu_quant_writeback_one_token<Dims>(shmem, spec, shm, prev_id_for_writeback, tok,
                                                      col_in_half, lane, base_row_up, effective_bid,
                                                      top_k, batch_size);
            }
          }
        }
      }
#endif  // MONO_PROFILE_DEFER_UP_EPILOGUE

      // ── Inter-iteration sync ──
      //
      // The launcher arms `bar_w[next_slot]` at iter s.  Two
      // iterations later (iter s+2), the launcher arms the SAME
      // `bar_w[next_slot]` again (because the slot index repeats every
      // 2 iters in the ping-pong).  For the second arm not to
      // double-arm an mbarrier whose current phase is still pending,
      // the calc-warp consume of that slot at iter s+1 must complete
      // BEFORE iter s+2's launcher arm runs.
      //
      // The legacy QUANT/COMPUTE __syncthreads() served this role.
      // Without it, the launcher (a single warp-8-lane-0 thread that
      // never waits) can race ahead through all K_TILES launcher
      // arms before any calc warp consumes its bar_w wait.  This
      // single end-of-iter sync re-establishes ordering: every iter,
      // all warps (including launcher and calc) rendezvous at the
      // sync, so the launcher cannot arm the next-slot mbarrier until
      // the calc warps' wait on the same slot has completed.
      //
      // At most one __syncthreads() per outer K-step iteration;
      // this sync publishes the launcher's `mbarrier_arrive_expect_tx`
      // (a producer-side state mutation on bar_w) to the calc warps
      // that will issue `try_wait_parity` against it next iter.
      __syncthreads();
    }  // end K-loop

    // ── End-of-expert: write final_d to partial_result.wgmma_out[128][8] ──
    // Canonical WGMMA D-matrix layout per thread (m64n8k32):
    //   d[0]: row = warp_in_wg*16 + lane/4 + 0,  col = (lane%4)*2 + 0
    //   d[1]: row = warp_in_wg*16 + lane/4 + 0,  col = (lane%4)*2 + 1
    //   d[2]: row = warp_in_wg*16 + lane/4 + 8,  col = (lane%4)*2 + 0
    //   d[3]: row = warp_in_wg*16 + lane/4 + 8,  col = (lane%4)*2 + 1
    // For WG1, rows shift by +64 in the full 128-row output tile.
    //
    // Up-projection epilogue: (a) the wgmma_out SHM store, (b) the
    // inter-warp sync that publishes wgmma_out, and (c) the
    // SiLU+fp8-quant writeback helper.
    if (is_calc) {
      const uint32_t wg_row_offset = is_wg1 ? 64u : 0u;
      const uint32_t row_base = wg_row_offset + warp_in_wg * 16 + lane / 4;
      const uint32_t col_base = (lane % 4) * 2;
      shm->partial_result.wgmma_out[row_base + 0][col_base + 0] = final_d0;
      shm->partial_result.wgmma_out[row_base + 0][col_base + 1] = final_d1;
      shm->partial_result.wgmma_out[row_base + 8][col_base + 0] = final_d2;
      shm->partial_result.wgmma_out[row_base + 8][col_base + 1] = final_d3;
    }

    // (The per-expert K-loop-tail scan that populated `up_rank_for_tok`
    // is gone: the up-proj epilogue now reads the routing-time
    // `expert_tok_krank[id][tok]` table built once in
    // `prepare_moe_topk_BS8`.  See `up_silu_quant_writeback_one_token`.)

    __syncthreads();

    // Phase-timing: after the wgmma_out store + publish sync.  The
    // SiLU+fp8-quant helper that follows reads `wgmma_out`, so the
    // sync must complete first.  Δ to `t_up_after_expert0_kloop`
    // measures the cost of the wgmma_out store + the publish sync.

#ifdef MONO_PROFILE_DEFER_UP_EPILOGUE
    // Deferred path: mark this expert's SiLU+quant writeback as
    // pending; the prefetch-warp body inside the next expert's
    // K-loop iters 0/1 will perform it.
    prev_id_for_writeback = id;
    has_pending_writeback = true;
#else
    // Inline path (default): SiLU + fp8 quant + GM writeback runs
    // here on calc warps with the same (warp → tok) mapping as
    // before.  See `up_silu_quant_writeback_one_token` for the
    // per-(tok, lane) body.
    if (is_calc) {
      const uint32_t tok = warp;          // 0..7, one per calc warp
      const uint32_t col_in_half = lane;  // 0..31
      up_silu_quant_writeback_one_token<Dims>(shmem, spec, shm, id, tok, col_in_half, lane,
                                              base_row_up, effective_bid, top_k, batch_size);
    }
#endif  // MONO_PROFILE_DEFER_UP_EPILOGUE

    // ── Tail of expert loop ──
    //
    // The next expert's iter-0 TMAs (weight + bf16) were already issued
    // during this expert's s=K_TILES-1 COMPUTE half via the
    // cross-expert stitch (see the launcher branch in the K-loop).
    // No tail prefetch is needed here; the mbarrier chain carries
    // across the expert boundary without any __syncthreads().
    //
    // A single trailing __syncthreads() publishes the SiLU writeback's
    // spec->temp_fp8 / spec->temp_act_scale writes to every thread in
    // the block before the next iteration's prefetch warp overwrites
    // `shm->up_scale[0]`.  This sync also aligns the launcher thread
    // with calc warps so the launcher cannot race ahead and issue the
    // stitch-triggered mbarrier arm for expert e+2 before expert e+1's
    // iter-0 QUANT wait has consumed the stitch arrival for expert e+1.
    __syncthreads();

  }  // end expert loop

#ifdef MONO_PROFILE_DEFER_UP_EPILOGUE
  // ── Post-loop drain for the LAST expert (deferred path only) ──
  //
  // The deferred path inside the K-loop processes expert e's
  // wgmma_out at iter 0/1 of expert e+1.  The LAST expert visited
  // (expert_count - expert_stride for stride==1) has nothing to
  // defer to, so we run its writeback inline now, on calc warps,
  // with the original (warp → tok) mapping (warps 0..7, 1 token
  // each).  All warps are free at this point — the K-loop is done.
  if (has_pending_writeback && is_calc) {
    const uint32_t tok = warp;          // 0..7
    const uint32_t col_in_half = lane;  // 0..31
    up_silu_quant_writeback_one_token<Dims>(shmem, spec, shm, prev_id_for_writeback, tok,
                                            col_in_half, lane, base_row_up, effective_bid, top_k,
                                            batch_size);
  }
#endif  // MONO_PROFILE_DEFER_UP_EPILOGUE
}

}  // namespace monomoe

#endif
