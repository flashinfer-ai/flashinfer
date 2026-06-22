
#pragma once
#ifndef MOE_INTERNAL_H
#define MOE_INTERNAL_H

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include "moe_interface.h"

// Full 32-lane warp mask for the `*_sync` warp intrinsics.  Previously
// supplied by the vLLM tree's `cuda_utils.h`; defined here so the megamoe
// sources are self-contained under FlashInfer's JIT build.
#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

// ── Default pipeline variants ───────────────────────────────────────────────
// Two scheduling optimizations are enabled by default.  Both are pure
// pipeline / data-movement reschedules — they do not change the kernel's
// numerical result (verified bit-identical against the non-variant path),
// only how work overlaps:
//
//   MONO_PROFILE_BARW_4DEEP       : up-proj weight TMA pipeline is 2-deep
//                                   (4 mbarrier slots, lookahead `(s+2)&3`)
//                                   instead of single-deep, hiding more of
//                                   the cross-/intra-expert weight TMA
//                                   latency behind compute.
//   MONO_PROFILE_DEFER_UP_EPILOGUE: the up-proj SiLU+fp8-quant writeback is
//                                   deferred onto the prefetch warps during
//                                   the next expert's first K-loop iters,
//                                   overlapping it with calc-warp WGMMA.
//
// Define MONO_PROFILE_NO_BARW_4DEEP / MONO_PROFILE_NO_DEFER_UP_EPILOGUE (e.g.
// via an nvcc -D flag) to fall back to the simpler single-deep / inline-
// epilogue paths, which are kept compiled-out below for A/B comparison.
#if !defined(MONO_PROFILE_BARW_4DEEP) && !defined(MONO_PROFILE_NO_BARW_4DEEP)
#define MONO_PROFILE_BARW_4DEEP
#endif
#if !defined(MONO_PROFILE_DEFER_UP_EPILOGUE) && !defined(MONO_PROFILE_NO_DEFER_UP_EPILOGUE)
#define MONO_PROFILE_DEFER_UP_EPILOGUE
#endif


namespace moe_monokernel {

using T_element = float;              //< Type of fp32 accumulators (partial results, out_accum)
using OpaqueElement = std::uint32_t;  //< Auxiliary 32-bit type used to generate
                                      // better assembly code in loads

/**
 * @brief Offsets into the @c token_indexes field
 *
 * This is an offset array. To find all the tokens that belong to expert @c id :
 * <tt>
 * for (int i = first_token; i < last_token; i++) {
 *    int token_index = token_indexes[i];
 * }
 */
struct ExpertRef {
  std::uint16_t first_token;
  std::uint16_t last_token;
  std::uint32_t id;
};

// ── WGMMA / TMA opt-in detection (forward declarations) ────────────────────
// These SFINAE helpers let `MoEGemmSpec<Dims>` pick the variant-dependent
// `DOWN_COL_TILE` below (128 normally, 256 for the BS8 TMA+WGMMA variant
// after Phase 2a layout alignment).  Full definitions (with detailed
// comments) live further down the file; the forward declarations here only
// need to expose `::value` so compile-time expressions can use them.
template <typename Dims>
struct use_wgmma {
  template <typename D>
  static constexpr auto test(int) -> decltype(D::KernelConfig::USE_WGMMA, bool()) {
    return D::KernelConfig::USE_WGMMA;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

template <typename Dims>
struct use_tma {
  template <typename D>
  static constexpr auto test(int) -> decltype(D::KernelConfig::USE_TMA, bool()) {
    return D::KernelConfig::USE_TMA;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

// ── Down-proj K-step size opt-in detection ──────────────────────────────
// `Dims::KernelConfig::K_STEP_DOWN` is optional; default to 128 (= the
// up-proj K_STEP_WGMMA, which also matches the SWZ128 atom width).  Set
// to 256 in a Dims variant to halve the number of outer K-iterations of
// the down-proj K-loop, doubling the work per iteration so the launcher
// + barrier-wait overhead is amortized over more compute.
//
// Must be a multiple of 128 (the SWZ128 atom K-width) and a divisor of
// `Dims::N`.
//
// Costs of K_STEP_DOWN > 128:
//   * `w_down_wgmma` slot grows to `DOWN_COL_TILE * K_STEP_DOWN` bytes.
//   * `a_down_wgmma` slot grows to `T_TILE * K_STEP_DOWN` bytes.
//   * `a_down_scale` is fixed at `T_TILE * (Dims::N / 64)` floats per
//     expert (full reduction-dim scale set, hoisted out of the K-loop).
//   * One `bar_w` arm covers
//     `DOWN_W_TX_BYTES_PER_HALF * DOWN_COL_HALVES * K_SUBSTEPS_DOWN`
//     bytes (single mbarrier wait still drains all atoms).
//   * The WGMMA loop runs `K_SUBSTEPS_DOWN` inner 128-K sub-blocks per
//     outer K-step; scales are still applied at every 128-K boundary.
template <typename Dims>
struct down_k_step {
 private:
  template <typename D>
  static constexpr auto test(int) -> decltype((std::uint32_t)D::KernelConfig::K_STEP_DOWN) {
    return (std::uint32_t)D::KernelConfig::K_STEP_DOWN;
  }
  template <typename>
  static constexpr std::uint32_t test(...) {
    return 128u;
  }

 public:
  static constexpr std::uint32_t value = test<Dims>(0);
};

// ── Up-proj K-step size opt-in detection ────────────────────────────────
// `Dims::KernelConfig::K_STEP_UP` is optional; default to 128 (= the
// up-proj's existing K_STEP_WGMMA, which matches the SWZ128 atom
// K-width and the per-K-step block-wise FP8 scale boundary).  Set to
// 256 in a Dims variant to halve the number of outer K-iterations of
// the up-proj K-loop, doubling per-iter work so the launcher + bar
// arms + sync overhead is amortized over more compute.
//
// Must be a multiple of 128 (the SWZ128 atom K-width) and a divisor of
// `Dims::HIDDEN_STATES` (the up-proj reduction dim).
//
// Costs of K_STEP_UP > 128:
//   * `w_wgmma` slot grows to `W_UP_TILE_WGMMA * K_STEP_UP` bytes.
//   * One `bar_w` arm covers `K_STEP_UP / K_STEP_WGMMA` ×
//     the per-128-K payload; a single mbarrier wait drains all
//     substeps.
//   * The WGMMA loop runs `K_SUBSTEPS_UP = K_STEP_UP / 128`
//     inner 128-K sub-blocks per outer K-step; scales are applied at
//     every 128-K boundary (matching the block-wise FP8 quantization
//     granularity).
template <typename Dims>
struct up_k_step {
 private:
  template <typename D>
  static constexpr auto test(int) -> decltype((std::uint32_t)D::KernelConfig::K_STEP_UP) {
    return (std::uint32_t)D::KernelConfig::K_STEP_UP;
  }
  template <typename>
  static constexpr std::uint32_t test(...) {
    return 128u;
  }

 public:
  static constexpr std::uint32_t value = test<Dims>(0);
};

/**
 * @brief Scratchpad memory for use within the monokernel.
 *
 * Place in global memory.
 *
 */
template <typename Dims>
struct MoEGemmSpec {
  static constexpr uint32_t SPEC_MAX_TOPK = 8;
  // Virtual batch size: each token may be routed to up to SPEC_MAX_TOPK
  // experts, so the sorted temp buffer must hold BS * SPEC_MAX_TOPK rows. BS <=
  // 8 now also uses BS * SPEC_MAX_TOPK rows because the split-phase design
  // writes one row per (token, expert) pair into spec->temp_bf16.
  static constexpr uint32_t TEMP_ROWS = Dims::BS * SPEC_MAX_TOPK + 8;
  static constexpr uint32_t TEMP_ROWS_TMA = Dims::BS * SPEC_MAX_TOPK;

  AQ_element activations[Dims::BS][Dims::HIDDEN_STATES];  //< Quantized activations

  // Kept for struct layout stability — removing it would shift
  // TEMP_FP8_OFFSET and break the down-activation TMA descriptor.
  A_element temp_bf16[TEMP_ROWS * Dims::N];

  // ── WGMMA-path-only up-proj → down-proj scratchpad ───────────────────
  // Used when `use_wgmma<Dims>::value == true`.  The up-projection
  // epilogue fuses per-64-col fp8 quantization into the SiLU writeback
  // and stores the fp8 activations plus per-(virtual-row, up-block)
  // fp32 scales into these buffers; the WGMMA down-projection consumes
  // them directly (no bf16→fp8 re-quantization pass).
  //
  // Layouts:
  //   temp_fp8        [TEMP_ROWS][N]            fp8
  //   temp_act_scale  [TEMP_ROWS][N / 64]       fp32
  //                   one scale per (virtual_row, up_block_idx)
  //
  // For Qwen3.5-35B (BS=8, top_k=8, N=512): TEMP_ROWS = 72
  //   temp_fp8       = 72 ×  512 × 1 B = 36.0 KB
  //   temp_act_scale = 72 ×    8 × 4 B =  2.25 KB
  //
  // These are separate from temp_bf16 (kept for the scalar path) so
  // non-WGMMA builds are byte-identical.
  static constexpr uint32_t DOWN_ACT_BLOCK_SIZE = 64;
  static_assert(Dims::N % DOWN_ACT_BLOCK_SIZE == 0,
                "Dims::N must be a multiple of 64 for the WGMMA down-proj "
                "per-64-col activation quantization scheme");
  static_assert(Dims::HIDDEN_STATES % 128 == 0,
                "Dims::HIDDEN_STATES must be a multiple of 128 for the "
                "WGMMA down-proj 128-cols-per-block grid layout");
  static constexpr uint32_t TEMP_ACT_SCALE_COLS = Dims::N / DOWN_ACT_BLOCK_SIZE;
  AQ_element temp_fp8[TEMP_ROWS * Dims::N];
  float temp_act_scale[TEMP_ROWS * TEMP_ACT_SCALE_COLS];

  // Byte offset of `temp_fp8` inside `MoEGemmSpec<Dims>`, exposed as a
  // compile-time constant so the host-side TMA wrapper can compute the
  // device pointer to `spec->temp_fp8` from the scratchpad base without
  // reading the struct layout at runtime (spec R9.2).  Consumed by
  // `create_down_activation_tma_desc` when building the down-projection
  // activation descriptor.
  static constexpr size_t TEMP_FP8_OFFSET = offsetof(MoEGemmSpec<Dims>, temp_fp8);

  // Down-projection block / group layout:
  //   For the BS8 TMA+WGMMA variant (`use_tma<Dims>::value == true` and
  //   Dims::BS <= 8), `DOWN_COL_TILE = 256`; otherwise 128.  Grids:
  //
  //   BS8 TMA+WGMMA: DOWN_COL_TILE=256, DOWN_GRID=8, DOWN_GROUPS=16
  //
  //   The `DOWN_GROUPS == UP_GROUPS` alignment in the BS8 TMA path is
  //   the prerequisite for the Phase-2b Expert_Barrier at site #2.
  //   `DOWN_GROUPS` is mirrored in MoECoreDims (defined later in this
  //   file); the two MUST match — MoECoreDims contains a
  //   static_assert that cross-checks.
  //
  // The down-projection writes its result via fp32 atomicAdd into a
  // single-buffer `down_partial_out[BS][HIDDEN_STATES]` (no per-group
  // dimension); Phase 5 reads each cell once and casts to bf16.
  static constexpr uint32_t DOWN_COL_TILE = (use_tma<Dims>::value && Dims::BS <= 8) ? 256u : 128u;
  static constexpr uint32_t DOWN_GRID = Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;
  // Per-(BS, HIDDEN_STATES) GM accumulator buffer for the WGMMA
  // down-projection.  Each contributing block atomicAdds its
  // `out_accum[tok][col]` slice into this single buffer at the end of
  // Phase 4; Phase 5 reads each cell ONCE and casts to bf16 (no
  // cross-group reduction).
  //
  // Shape: [BS][HIDDEN_STATES] fp32.
  // For Qwen3.5-35B (BS=8, HIDDEN_STATES=2048):
  //   8 × 2048 × 4 B = 64 KB.
  //
  // Stored fp32 (not bf16) so the 16-way atomicAdd preserves
  // precision; the bf16 cast happens once in Phase 5.
  //
  // The buffer is zero-initialized at the top of moe_kernel_topk_BS8;
  // the up-proj-completion barrier (#2) publishes that zero across
  // all blocks before any Phase-4 atomicAdd fires.  No per-group
  // dimension exists — the colstripe barrier (#3) ensures all 16
  // contributing blocks finish their atomicAdds before Phase 5
  // reads.
  float down_partial_out[Dims::BS * Dims::HIDDEN_STATES];

  // Per-token block-wise activation quantization scales.
  // Block size = 128 along K dimension → K/128 scales per token.
  // act_scale[tok][blk] = max(|x_tok[blk*128..(blk+1)*128-1]|) / 448
  static constexpr uint32_t ACT_BLOCK_SIZE = 128;
  static constexpr uint32_t ACT_SCALE_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK_SIZE - 1) / ACT_BLOCK_SIZE;
  float act_scale[Dims::BS][ACT_SCALE_BLOCKS];

  // ── Software barrier counters ────────────────────
  //
  // Placed at the TAIL of `MoEGemmSpec<Dims>`, AFTER `act_scale`, so
  // `TEMP_FP8_OFFSET = offsetof(MoEGemmSpec<Dims>, temp_fp8)` stays
  // byte-identical to its pre-migration value.  The host-side TMA
  // descriptor factory in `moe_wrapper.cu` derives the device pointer
  // to `spec->temp_fp8` from that compile-time constant, 
  // so inserting any new field BEFORE
  // `temp_fp8` would silently break the down-activation TMA path.
  //
  // Lifetime / initialization:
  //   * Host zero-initializes the whole scratchpad (including these
  //     counters) once per process via `cudaMemsetAsync` on the first
  //     launch.
  //   * Subsequent launches inherit the counter state from the
  //     previous kernel's exit: the ping-pong discipline is
  //     self-maintaining — each barrier call's seed `atomicExch`
  //     overwrites the prior-call `0x80000000` on the same slot in its
  //     atomic step, and any arrivals that landed on the slot between
  //     calls are folded back in by the seed-thread's follow-up
  //     `atomicAdd(c, prior)`.  No host re-zero is required
  //     across kernel invocations.
  //
  // Call-site mapping:
  //   * `grid_barrier.slot[2]` — the Phase-1 full-grid ping-pong pair.
  //     Used by:
  //       - Sites #2, #3 (BS8) — Phase 3→4 and Phase 4→5. These are
  //         Grid_Barrier in Phase 1 and get downgraded to
  //         Expert_Barrier / ColStripe_Barrier (below) in Phase 2b.
  //
  //   * `partial_barrier.expert_slot[NUM_EXPERTS][2]` — Phase-2b
  //     Expert_Barrier counter region, one Counter_Pair per expert
  //     group id (== up_group).  Used at site #2 only, BS8 only.
  //     Arrival count = UP_GRID (8 blocks per expert group after
  //     Phase-2a layout alignment, `DOWN_COL_TILE = 256`).
  //
  //   * `partial_barrier.colstripe_slot[DOWN_GRID][2]` — Phase-2b
  //     ColStripe_Barrier counter region, one Counter_Pair per
  //     output col stripe id (== `blockIdx.x % DOWN_GRID`).  Used at
  //     site #3 only, BS8 only.  Arrival count = DOWN_GROUPS
  //     (16 blocks per col stripe after Phase-2a alignment).
  //
  // Uses the local `DOWN_GRID` (declared above on this struct) rather
  // than `MoECoreDims<Dims>::DOWN_GRID` because `MoECoreDims` is
  // defined LATER in this file than `MoEGemmSpec`, making the
  // qualified name an incomplete-type forward reference here.  The
  // two must match, and `MoECoreDims` carries a `static_assert` that
  // cross-checks (see the DOWN_GROUPS cross-check further down).
  //
  // Sizing (Design "Sizing"): for NUM_EXPERTS=256, DOWN_GRID=16
  // (Phase-1) or 8 (post Phase-2a), the barrier counters total
  // 2 × 4 + 256 × 2 × 4 + DOWN_GRID × 2 × 4 B ≤ 2120 B — negligible
  // vs. the MB-scale scratchpad.
  struct {
    uint32_t slot[2];
  } grid_barrier;
  struct {
    uint32_t expert_slot[Dims::NUM_EXPERTS][2];
    uint32_t colstripe_slot[DOWN_GRID][2];
  } partial_barrier;
};

// Maximum supported dimensions for shared memory and scratchpad allocation
// sizes
#if USE_SMALL_SETUP
// SHM limits batch size to ~2k
using Dims_Max = MoEDimensions<1024, 256, 1024, 256>;
#else
using Dims_Max = MoEDimensions<1024, 1024, 5120, 256>;
#endif

// ── Block-wise quantization detection (forward declaration) ──────────────
// These helpers are used in the MoE_SHM layout below and defined with full
// SFINAE semantics further down. Here we just need the compile-time bool,
// so we duplicate the minimal detection inline.
template <typename Dims>
struct shm_is_block_wise {
  template <typename D>
  static constexpr auto test(int) -> decltype(D::QUANT_GRAN, bool()) {
    return D::QUANT_GRAN == QuantGranularity::BLOCK_WISE;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

// Number of column-blocks in the up-projection scale tensor. For block-wise
// this is ceil(K / BLOCK_SCALE_COL); for per-channel we return 1 (unused
// placeholder so the SHM field is harmlessly tiny).
template <typename Dims, bool IsBlockWise = shm_is_block_wise<Dims>::value>
struct shm_up_scale_cols {
  static constexpr uint32_t value = 1;
};
template <typename Dims>
struct shm_up_scale_cols<Dims, true> {
  static constexpr uint32_t value = Dims::UP_SCALE_COLS;
};

// Number of column-blocks in the down-projection scale tensor. For
// block-wise this is ceil(N / BLOCK_SCALE_COL); for per-channel we return 1
// (unused placeholder so the SHM field is harmlessly tiny).  Mirrors
// shm_up_scale_cols.
template <typename Dims, bool IsBlockWise = shm_is_block_wise<Dims>::value>
struct shm_down_scale_cols {
  static constexpr uint32_t value = 1;
};
template <typename Dims>
struct shm_down_scale_cols<Dims, true> {
  static constexpr uint32_t value = Dims::DOWN_SCALE_COLS;
};

// ── WGMMA opt-in detection ───────────────────────────────────────────────
// `Dims::KernelConfig::USE_WGMMA` is optional; default to false for all
// existing Dims variants so the current mma.sync path stays in use.
// Only the new Dims_BS8_..._WGMMA variant sets USE_WGMMA=true.
//
// NOTE: The primary definition lives near the top of this file (before
// `MoEGemmSpec<Dims>`) so `MoEGemmSpec` can use `use_wgmma<Dims>::value`
// to select the variant-dependent `DOWN_COL_TILE` for Phase 2a.  The
// block comment below documents the same detection scheme for readers
// who land on the later usage sites first.

// ── TMA opt-in detection ────────────────────────────────────────────────
// `Dims::KernelConfig::USE_TMA` is optional; default to false for all
// existing Dims variants so the current cp.async WGMMA path stays in use.
// Only the new Dims_BS8_..._WGMMA_TMA variant sets USE_TMA=true.
//
// NOTE: The primary definition lives near the top of this file (before
// `MoEGemmSpec<Dims>`) — see the comment on `use_wgmma` above.

/**
 * @brief contains various constants used within the MoE monokernel.
 */
template <typename Dims>
struct MoECoreDims {
  using MoEDims = Dims;

  // GPU configuration.
  static constexpr std::uint32_t THREADS_PER_WARP = 32;
  static constexpr std::uint32_t TOTAL_WARP_COUNT =
      Dims::KernelConfig::BLOCK_SIZE / THREADS_PER_WARP;
  static constexpr std::uint32_t CALC_WARP_COUNT = 8;
  static constexpr std::uint32_t PREFETCH_WARP_COUNT = TOTAL_WARP_COUNT - CALC_WARP_COUNT;

  // MMA 1 matrix tile dimensions.
  static constexpr std::uint32_t A_TILE = 8;
  static constexpr std::uint32_t W_UP_TILE = 16;
  static constexpr std::uint32_t K_TILE = 32;

  // ── WGMMA-only tile dimensions (v1 dual-WG K=128 streaming) ──────────
  // Used by the WGMMA up-proj path (Phase 2-3 when USE_WGMMA=true).
  //
  // Layout of the 128-row weight tile per block:
  //   rows [0  .. 31]  : WG0 gate rows [base    .. base+31]
  //   rows [32 .. 63]  : WG0 up   rows [base+N  .. base+N+31]
  //   rows [64 .. 95]  : WG1 gate rows [base+32 .. base+63]
  //   rows [96 .. 127] : WG1 up   rows [base+N+32 .. base+N+63]
  //
  // Per K-step each WG issues 4 chained wgmma.mma_async.m64n8k32, which
  // together consume K=128. The weight tile is single-buffered; bf16 input
  // and fp8 activation tiles are double-buffered. The streaming pipeline
  // alternates between even half-stages (WGMMA + bf16 prefetch) and odd
  // half-stages (quantize + weight prefetch).
  //
  //   W_UP_TILE_WGMMA  = 128  — M dim of the block's weight tile
  //                             (64 rows per WG × 2 WGs)
  //   W_UP_COLS_WGMMA  = 64   — output columns per block per K-step
  //                             (W_UP_TILE_WGMMA / 2)
  //   K_TILE_WGMMA     = 32   — K width of one m64n8k32 instruction
  //                             (hardware-fixed for fp8)
  //   K_STEP_WGMMA     = 128  — K consumed per outer K-step (=4 × K_TILE_WGMMA)
  //   K_TILES_WGMMA    = K/K_STEP_WGMMA  — outer K iterations per expert
  //   WGMMAS_PER_STEP  = 4    — WGMMAs chained per WG per K-step
  //   UP_GRID_WGMMA    = 2*N / W_UP_TILE_WGMMA  — blocks per expert
  static constexpr std::uint32_t W_UP_TILE_WGMMA = 128;
  static constexpr std::uint32_t W_UP_COLS_WGMMA = W_UP_TILE_WGMMA / 2;
  static constexpr std::uint32_t K_TILE_WGMMA = 32;
  static constexpr std::uint32_t K_STEP_WGMMA = 128;
  static constexpr std::uint32_t WGMMAS_PER_STEP = K_STEP_WGMMA / K_TILE_WGMMA;  // 4
  static constexpr std::uint32_t K_TILES_WGMMA = Dims::HIDDEN_STATES / K_STEP_WGMMA;
  static constexpr std::uint32_t UP_GRID_WGMMA = 2 * Dims::N / W_UP_TILE_WGMMA;

  static_assert(K_STEP_WGMMA % K_TILE_WGMMA == 0,
                "K_STEP_WGMMA must be a multiple of K_TILE_WGMMA");
  static_assert(!use_wgmma<Dims>::value || Dims::HIDDEN_STATES % K_STEP_WGMMA == 0,
                "HIDDEN_STATES must be a multiple of 128 for the WGMMA path "
                "(one K-step consumes K=128)");
  static_assert(!use_wgmma<Dims>::value || (2 * Dims::N) % W_UP_TILE_WGMMA == 0,
                "2*N must be a multiple of 128 for the WGMMA path "
                "(one block produces 128 output rows per K-step)");

  // ── Down-proj outer K-step (Dims::KernelConfig::K_STEP_DOWN tunable) ─
  // K_STEP_DOWN is the K-width consumed per OUTER K-step of the down-
  // projection K-loop; it's a multiple of K_STEP_WGMMA = 128 (the
  // SWZ128 atom K-width).  Defaults to 128 (= K_STEP_WGMMA) for full
  // backward compatibility; setting `KernelConfig::K_STEP_DOWN = 256`
  // doubles per-iter compute / data movement, halving the iter count.
  //
  // Each outer K-step runs `K_SUBSTEPS_DOWN = K_STEP_DOWN / K_STEP_WGMMA`
  // inner 128-K sub-blocks; scales are applied at every 128-K boundary
  // (matching the block-wise quantization granularity).
  static constexpr std::uint32_t K_STEP_DOWN = down_k_step<Dims>::value;
  static constexpr std::uint32_t K_SUBSTEPS_DOWN = K_STEP_DOWN / K_STEP_WGMMA;
  static_assert(K_STEP_DOWN >= K_STEP_WGMMA && K_STEP_DOWN % K_STEP_WGMMA == 0,
                "K_STEP_DOWN must be a positive multiple of K_STEP_WGMMA "
                "(=128, the SWZ128 atom K-width).");
  static_assert(!use_wgmma<Dims>::value || Dims::N % K_STEP_DOWN == 0,
                "Dims::N must be a multiple of K_STEP_DOWN for the WGMMA "
                "down-projection (one outer K-step consumes K_STEP_DOWN "
                "K-elements).");

  // ── Up-proj outer K-step (Dims::KernelConfig::K_STEP_UP tunable) ────
  // K_STEP_UP is the K-width consumed per OUTER K-step of the up-
  // projection K-loop; multiple of K_STEP_WGMMA = 128 (the SWZ128 atom
  // K-width and the per-128-K block-wise FP8 scale boundary).
  // Defaults to 128 (= K_STEP_WGMMA) for full backward compatibility;
  // setting `KernelConfig::K_STEP_UP = 256` halves the K-loop iter
  // count and doubles per-iter QUANT/COMPUTE work.
  //
  // Each outer K-step runs `K_SUBSTEPS_UP = K_STEP_UP / K_STEP_WGMMA`
  // inner 128-K sub-blocks; the QUANT half quantizes one 128-K bf16
  // input chunk per substep, and the COMPUTE half runs 4 chained
  // m64n8k32 WGMMAs + scale-apply per substep.
  static constexpr std::uint32_t K_STEP_UP = up_k_step<Dims>::value;
  static constexpr std::uint32_t K_SUBSTEPS_UP = K_STEP_UP / K_STEP_WGMMA;
  static_assert(K_STEP_UP >= K_STEP_WGMMA && K_STEP_UP % K_STEP_WGMMA == 0,
                "K_STEP_UP must be a positive multiple of K_STEP_WGMMA "
                "(=128, the SWZ128 atom K-width).");
  static_assert(!use_wgmma<Dims>::value || Dims::HIDDEN_STATES % K_STEP_UP == 0,
                "Dims::HIDDEN_STATES must be a multiple of K_STEP_UP for "
                "the WGMMA up-projection (one outer K-step consumes "
                "K_STEP_UP K-elements of the reduction dim).");
  // Up-proj outer K-loop iteration count (replaces K_TILES_WGMMA in
  // call sites that should follow the K_STEP_UP setting).
  static constexpr std::uint32_t K_TILES_UP = Dims::HIDDEN_STATES / K_STEP_UP;

  // Effective M (row-tile) size of one block's up-proj work — 64 for the
  // WGMMA path, 16 for the scalar path.  Used to compute UP_GRID = 2*N/M.
  static constexpr std::uint32_t W_UP_TILE_EFFECTIVE =
      use_wgmma<Dims>::value ? W_UP_TILE_WGMMA : W_UP_TILE;

  // ── WGMMA down-projection grid layout ────────────────────────────────
  // Each down-block owns DOWN_COL_TILE output cols within
  // Dims::HIDDEN_STATES, so DOWN_GRID = HIDDEN_STATES / DOWN_COL_TILE
  // blocks cover one expert's full output.  The remaining grid blocks
  // process DIFFERENT expert groups in parallel: DOWN_GROUPS =
  // GRID_SIZE / DOWN_GRID expert groups each accumulate the
  // contribution of their assigned experts via fp32 atomicAdd into
  // the single-buffer `spec->down_partial_out[BS][HIDDEN_STATES]`.
  // Phase 5 reads each cell once and casts to bf16 (no cross-group
  // reduction).
  //
  // BS8 TMA+WGMMA (Phase 2a layout alignment): DOWN_COL_TILE = 256.
  //   DOWN_GRID   = 2048 / 256 = 8 blocks per expert
  //   DOWN_GROUPS = 128  / 8   = 16 expert groups (== UP_GROUPS)
  // This alignment makes the 8 blocks `[g*8, g*8+7]` form both
  // `up_group = g` and `down_group = g` for the same expert set, so
  // the producer-set of site #2 (Phase 3 → Phase 4) becomes identical
  // to its consumer-set, enabling the Expert_Barrier in Phase 2b.
  //
  // The variant-dependent value MUST match
  // `MoEGemmSpec<Dims>::DOWN_COL_TILE`; the static_assert further down
  // cross-checks their derived DOWN_GROUPS.
  static constexpr std::uint32_t DOWN_COL_TILE =
      (use_tma<Dims>::value && Dims::BS <= 8) ? 256u : 128u;
  static constexpr std::uint32_t DOWN_GRID = Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr std::uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;

  static_assert(!use_wgmma<Dims>::value || Dims::HIDDEN_STATES % DOWN_COL_TILE == 0,
                "HIDDEN_STATES must be a multiple of DOWN_COL_TILE for the "
                "WGMMA down-projection (one down-block owns DOWN_COL_TILE "
                "output cols)");
  static_assert(!use_wgmma<Dims>::value || Dims::KernelConfig::GRID_SIZE % DOWN_GRID == 0,
                "GRID_SIZE must be a multiple of DOWN_GRID for the WGMMA "
                "down-projection (expert groups partition the grid)");
  static_assert(!use_wgmma<Dims>::value || DOWN_GROUPS <= Dims::NUM_EXPERTS,
                "DOWN_GROUPS cannot exceed NUM_EXPERTS (each expert group "
                "must process at least one expert)");

  // Cross-check that MoEGemmSpec's mirror of DOWN_COL_TILE / DOWN_GROUPS
  // (computed locally there to avoid a forward reference) matches this
  // one.  If they diverge, the kernel's per-block col-stripe ownership
  // would disagree with the GM accumulator buffer's size and the
  // colstripe barrier's arrival count, silently corrupting Phase 5.
  static_assert(MoEGemmSpec<Dims>::DOWN_COL_TILE == DOWN_COL_TILE,
                "MoEGemmSpec::DOWN_COL_TILE must match "
                "MoECoreDims::DOWN_COL_TILE — check the variant-dependent "
                "DOWN_COL_TILE definition in both places.");
  static_assert(MoEGemmSpec<Dims>::DOWN_GROUPS == DOWN_GROUPS,
                "MoEGemmSpec::DOWN_GROUPS must match MoECoreDims::DOWN_GROUPS "
                "— check the DOWN_COL_TILE definition in both places.");

  // GEMM 2 matrix tile dimensions.
  static constexpr std::uint32_t W_DOWN_MMA_TILE = 16;
  static constexpr std::uint32_t W_DOWN_TILE = Dims::HIDDEN_STATES / Dims::KernelConfig::GRID_SIZE;
  static constexpr std::uint32_t T_TILE = 8;

  static constexpr unsigned BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE;

  static constexpr unsigned PADDING =
      32;  // this works *slightly* better than 16 due to reduced L2 transfers

  // Row padding (in bytes) for the down-projection fp8 tiles (both the
  // weight tile w[].down and the activation tile a.down). The MMA inner
  // loop reads each row with `byte_offset = row * stride + 4 * (t % 4)`
  // plus a row-stride of `t / 4`. For the reads to hit all 32 banks, we
  // need `(stride_bytes / 4) % 32 >= 4`, i.e. the row stride in dwords
  // must leave at least 4 unique banks per row step so the `t % 4`
  // contribution (0..3) doesn't collide across rows.
  //
  // With N=512 (stride 128 dwords, which is 0 mod 32 → 8-way conflict),
  // adding 16 bytes (4 dwords) gives 132 dwords → 4 mod 32. Combined
  // with t%4 this covers all 32 banks uniformly. PADDING=32 (the global
  // constant above) gives 136 dwords → 8 mod 32, only 4 unique banks
  // per 4 rows → 2-way conflict. So we use DOWN_ROW_PADDING=16 here.
  static constexpr unsigned DOWN_ROW_PADDING = 16;
  static constexpr unsigned K_DIM_PADDED_A = Dims::HIDDEN_STATES;
  static constexpr unsigned K_DIM_PADDED_W = Dims::HIDDEN_STATES;
  static constexpr unsigned K_DIM_HALF_PADDED_A = Dims::HIDDEN_STATES / 2;
};

// 1 tile per warp
// 20 warps x 2 params x 1k = 20k pre-fetch
template <typename Dims>
struct MoE_SHM {
  using CoreDims = MoECoreDims<Dims>;
  union U {
    // ── TinyDataWGMMA_TMA: SHM layout for the TMA+WGMMA up-proj path ──
    //
    // Used when `use_wgmma<Dims>::value && use_tma<Dims>::value` are both
    // true — the TMA-based activation & weight loading path for the BS8
    // WGMMA up- and down-projections.
    //
    // Routing-window pipeline (Phase 1 → Phase 2 → Phase 3):
    //   * `bf16_in_full` — routing-window BF16 buffer.  Populated in
    //     Phase 1 by the TMA launcher via a single 16-issue
    //     `cp.async.bulk.tensor.2d` loop covering the full per-block
    //     `[BS][HIDDEN_STATES]` BF16 input tile.  Completion is
    //     signalled on `bar_rwin`; consumed by Phase 2's
    //     `routing_phase_quantize` and dead by the Phase-2 trailing
    //     `__syncthreads()`.
    //   * `fp8_act_full` — routing-phase-quantized FP8 buffer covering
    //     all `K_BLOCKS_TOTAL` 128-K substeps.  Written by Phase 2's
    //     `routing_phase_quantize` (warps 1..11), read by the Phase-3
    //     up-projection K-loop with no double-buffer slot alternation.
    //   * `a_down_wgmma` — down-projection activation buffer (Phase 4).
    //     Populated by the down-proj's per-expert TMA bulk activation
    //     load and consumed by the down-proj WGMMA K-loop.  Aliases
    //     SHM bytes that are dead by the time Phase 4 starts (the
    //     Phase-3 → Phase-4 grid sync serializes the reuse).
    //
    // mbarriers (all `alignas(16)` to satisfy the SM90 mbarrier PTX
    // alignment requirement):
    //   * bar_w[2]  — weight-tile mbarriers (one per double-buffer
    //                 slot).  Armed by the TMA launcher with the
    //                 per-arm weight tx_bytes before the
    //                 `cp.async.bulk.tensor.2d` stripe sequence.
    //                 Consumed by the WGMMA warps via
    //                 `mbarrier.try_wait.parity` in both the up- and
    //                 down-projection K-loops.
    //   * bar_a[2]  — down-projection activation-tile mbarriers (one
    //                 per double-buffer slot).  Armed and consumed
    //                 EXCLUSIVELY by `moe_down_projection_BS8_..._tma`
    //                 (Phase 4); re-initialized in the down-proj
    //                 prologue.  No up-proj consumer remains —
    //                 Phase-3 sources its activation operand from
    //                 `fp8_act_full`.
    //   * bar_rwin  — routing-window mbarrier (`arrival_count = 1`).
    //                 Armed by the TMA launcher in Phase 1 with
    //                 `tx_bytes = BS * K_BLOCKS_TOTAL * K_STEP_WGMMA *
    //                 sizeof(A_element)` (= 32 KB for Qwen3.5).
    //                 Consumed by warps 1..11 at the start of Phase 2
    //                 before reading `bf16_in_full`.
    struct TinyDataWGMMA_TMA {
      // ── K-substep constants ────────────────────────────────────────
      // `UP_K_SUBSTEPS` is the number of 128-K SWZ128 atoms per outer
      // up-proj K-step (= K_STEP_UP / K_STEP_WGMMA).
      static constexpr uint32_t UP_K_SUBSTEPS = CoreDims::K_SUBSTEPS_UP;
      // Total number of 128-K SWZ128 atoms per token along the K axis
      // (= Dims::HIDDEN_STATES / K_STEP_WGMMA).  Equal to 16 for
      // Qwen3.5 (HIDDEN_STATES = 2048).  Used by the new Phase-1
      // routing-window TMA load (covers the full BF16 input tile in
      // K_BLOCKS_TOTAL bulk loads) and by the new single-buffer
      // `fp8_act_full` that covers all K substeps for Phase-3
      // direct FP8 reads.
      static constexpr uint32_t K_BLOCKS_TOTAL = Dims::HIDDEN_STATES / CoreDims::K_STEP_WGMMA;

      // ── BS-dependent sizing clamp for the BS8-path-only fields ─────
      // The new `bf16_in_full` and `fp8_act_full` fields below are
      // consumed only by the BS8 TMA+WGMMA path.
      // The `tiny_wgmma_tma` variant of `union U` is, however, present
      // in `MoE_SHM<Dims>` for every Dims, and `sizeof(MoE_SHM<Dims>)`
      // takes the max across all union members.
      //
      // Tile-major shape: the BF16 input SHM buffer is shaped as
      // `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` rather than the natural-
      // looking `[BS][HIDDEN_STATES]` because the activation TMA
      // descriptor (`create_activations_tma_desc`) is configured with
      // `boxDim = (128, 8)` (innermost = K, outer = tokens) and
      // SWIZZLE_NONE.  Each `cp.async.bulk.tensor.2d` issued by
      // `moe_load_full_bf16_input` writes a COMPACT 8 × 128 BF16 box
      // (= 2 KB) to SHM with the box's outer-row stride equal to the
      // INNER box dim (256 B), NOT to the destination's logical row
      // stride.  With a `[BS][HIDDEN_STATES]` SHM layout (row stride
      // = HIDDEN_STATES * 2 = 4096 B for Qwen3.5), consecutive
      // K-substep TMA writes would overlap by ~1.75 KB and corrupt
      // each other (the TMA byte layout would not match the
      // consumer's row-strided indexing).
      //
      // The tile-major `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` layout
      // gives every K-substep its own self-contained 2 KB slab whose
      // bytes are exactly the bytes the TMA writes for that
      // `(coord0 = k_start, coord1 = 0)` issue.  Consumers
      // (`routing_phase_quantize`) read `bf16_in_full[kblk][token]`
      // as a natural `[K_STEP_WGMMA]` row — no row-stride
      // reinterpretation needed.
      //
      // Total size is identical to the prior `[BS][HIDDEN_STATES]`
      // shape: 16 × 8 × 128 × 2 = 32 KB for Qwen3.5, so the union
      // with `w_wgmma` / `w_down_wgmma` and the per-Dims SHM-budget
      // static_assert (≤ 228 KB) are unaffected.
      static constexpr uint32_t BF16_IN_FULL_K_BLOCKS = (Dims::BS <= 8) ? K_BLOCKS_TOTAL : 1;
      static constexpr uint32_t BF16_IN_FULL_BS = (Dims::BS <= 8) ? Dims::BS : 1;
      static constexpr uint32_t BF16_IN_FULL_K = (Dims::BS <= 8) ? CoreDims::K_STEP_WGMMA : 1;
      static constexpr uint32_t FP8_ACT_FULL_K_BLOCKS = (Dims::BS <= 8) ? K_BLOCKS_TOTAL : 1;

      static constexpr uint32_t FP8_ACT_K_CHUNK = 16;
      // Number of 16-K fp8 chunks per ONE 128-K SWZ128 atom (the unit
      // shared by the up-proj's per-substep layout and the down-proj's
      // per-substep layout).  Always 128 / 16 = 8.
      static constexpr uint32_t FP8_ACT_NUM_CHUNKS =
          CoreDims::K_STEP_WGMMA / FP8_ACT_K_CHUNK;  // 128 / 16 = 8

      // Down-proj activation tile holds one outer K-step's worth of
      // fp8 activations: `K_SUBSTEPS_DOWN` SWZ128 atoms stacked along
      // the K axis (each atom is 8 tok × 128 K-bytes = 1 KB).  For
      // K_STEP_DOWN=128 this collapses to the legacy single-atom 1 KB
      // slot; for K_STEP_DOWN=256 it grows to 2 KB.
      static constexpr uint32_t DOWN_ACT_K_SUBSTEPS = CoreDims::K_SUBSTEPS_DOWN;
      static constexpr uint32_t DOWN_FP8_ACT_NUM_CHUNKS =
          CoreDims::K_STEP_DOWN / FP8_ACT_K_CHUNK;  // 128 or 256 / 16

      // 1024-byte alignment required by SWIZZLE_128B on the down-proj
      // activation TMA: the XOR pattern uses low bits of the SHM
      // address and only behaves consistently within 1024-B-aligned
      // regions.
      //
      // `a_down_wgmma` is the down-projection activation buffer
      // (Phase 4).  It uses the (sub-step, token, kc, ki) view that
      // matches the CUTLASS Major::K B128 layout after the TMA's
      // SWZ128 XOR.  The sub-step dimension is collapsed into the
      // leading axis as a sequence of `DOWN_ACT_K_SUBSTEPS` 1024-B
      // atoms — each atom is exactly one K_STEP_WGMMA=128 K-substep
      // of the outer K-step.
      alignas(1024)
          AQ_element a_down_wgmma[2][DOWN_ACT_K_SUBSTEPS][CoreDims::T_TILE][FP8_ACT_NUM_CHUNKS]
                                 [FP8_ACT_K_CHUNK];  // 2 KB (down, K=128)
                                                     // 4 KB (down, K=256)

      // ── NEW: single-buffer fp8_act covering all K substeps ──────────
      // Single-buffer FP8 activation buffer for the BS8 TMA+WGMMA
      // post-fusion Phase-3 reads.  Indexed by `k_block ∈
      // [0, K_BLOCKS_TOTAL)`; the up-proj K-loop reads
      // `fp8_act_full[s * UP_K_SUBSTEPS + kk][...]` with no slot
      // alternation.  Layout per `k_block`:
      //   `[FP8_ACT_NUM_CHUNKS][T_TILE_PADDED][FP8_ACT_K_CHUNK]`
      // where `T_TILE_PADDED = T_TILE + 1 = 9`.  The 9th token-row in
      // each kc atom is unused padding — its purpose is to break the
      // 128-byte kc stride that caused an 8-way bank conflict on the
      // routing-quantize STS (lanes with the same `t%4` were targeting
      // the same bank across kc steps because `kc*128 B = 0 mod 32
      // banks`).  With the pad, the kc stride becomes
      // `T_TILE_PADDED * FP8_ACT_K_CHUNK = 9 * 16 = 144 B = 36 banks
      // mod 32 = 4`, so kc=0..7 write to disjoint banks within each
      // `t%4` lane group → conflict-free.
      //
      // The matching WGMMA B descriptor for the up-proj K-loop is
      // updated from `B_LBO = 128` to `B_LBO = T_TILE_PADDED *
      // FP8_ACT_K_CHUNK = 144` to step across the new kc stride.  The
      // 8-row × 16-byte WGMMA core matrix at the head of each kc atom
      // is still 128 B contiguous (rows 0..7) and the unused 9th row
      // is skipped by the LBO step.
      //
      // Sizing (Qwen3.5, K_BLOCKS_TOTAL=16):
      //   * Old: 16 × 8 × 8 × 16 = 16 KB.
      //   * New: 16 × 8 × 9 × 16 = 18 KB (+2 KB).
      //
      static constexpr uint32_t FP8_ACT_T_TILE_PADDED = CoreDims::T_TILE + 1u;
      alignas(1024) AQ_element fp8_act_full[FP8_ACT_FULL_K_BLOCKS][FP8_ACT_NUM_CHUNKS]
                                           [FP8_ACT_T_TILE_PADDED][FP8_ACT_K_CHUNK];

      static constexpr uint32_t W_WGMMA_M = 128;  // M dim of weight tile (up-proj)
      // Down-proj tile M dim tracks DOWN_COL_TILE (Phase 2a): 128 for the
      // non-TMA path, 256 for the BS8 TMA+WGMMA variant after the
      // Phase-2a layout alignment (DOWN_COL_TILE = 256).  Must stay a
      // multiple of 128 so the SWIZZLE_128B core-matrix atoms still tile
      // the outer M axis cleanly.
      static constexpr uint32_t W_DOWN_WGMMA_M = CoreDims::DOWN_COL_TILE;
      // K dim of the up-proj weight tile is held at K_STEP_WGMMA (= 128)
      // so the row stride matches the SWIZZLE_128B core-matrix width.
      // For K_STEP_UP > 128 we stack `K_SUBSTEPS_UP` 128-K SWZ128 atoms
      // along the M axis (substep 0 → rows [0..127], substep 1 → rows
      // [128..255], …) — same trick the down-proj uses for K_STEP_DOWN.
      // Each atom remains a self-contained 1024-B-aligned 128×128 region
      // so the TMA swizzle and the WGMMA A-descriptor (which addresses
      // one 128-row sub-atom per call) both work without a row-stride
      // change.
      static constexpr uint32_t W_WGMMA_K = CoreDims::K_STEP_WGMMA;
      static constexpr uint32_t W_WGMMA_M_TOTAL = W_WGMMA_M * CoreDims::K_SUBSTEPS_UP;
      // Down-proj outer K-step width (tunable via Dims::KernelConfig::
      // K_STEP_DOWN, default = 128).  Each outer K-step packs
      // `K_SUBSTEPS_DOWN` 128-K sub-blocks into the same SHM slot,
      // stacked along the M axis as additional 128-row atoms (so the
      // existing 128×128 SWZ128 atom layout is reused unchanged).
      static constexpr uint32_t W_DOWN_WGMMA_K = CoreDims::K_STEP_DOWN;
      static constexpr uint32_t W_DOWN_WGMMA_M_TOTAL = W_DOWN_WGMMA_M * CoreDims::K_SUBSTEPS_DOWN;
      union {
        // 1024-byte alignment required by SWIZZLE_128B: the XOR
        // pattern uses low bits of the SHM address and only behaves
        // consistently within 1024-byte-aligned regions. Both
        // `w_wgmma` and `w_down_wgmma` alias the same SHM bytes, so
        // the alignas applies to both views.
        //
        // ── NEW: full BF16 input tile, unioned with w_wgmma + w_down_wgmma.
        //
        // `bf16_in_full[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]`
        //   = 16 × 8 × 128 × 2 = 32 KB for Qwen3.5.
        // Lifetime: written by the Phase-1 routing-window TMA, read
        // by the Phase-2 routing_phase_quantize, dead by the Phase-2
        // trailing __syncthreads().  Aliases bytes with w_wgmma /
        // w_down_wgmma whose lifetimes start strictly later (Phase 3
        // up-proj weight TMA / Phase 4 down-proj weight TMA), so the
        // union is byte-disjoint at any instant in time.
        //
        // Tile-major `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` layout (vs.
        // the apparently-natural `[BS][HIDDEN_STATES]`): the
        // activation TMA descriptor uses `boxDim = (128, 8)` and
        // SWIZZLE_NONE, so each
        // `cp.async.bulk.tensor.2d` writes a COMPACT 8 × 128 BF16
        // box to SHM whose outer (token) stride equals the inner
        // box dim (256 B), NOT the destination's logical row stride.
        // With a `[BS][HIDDEN_STATES]` layout (row stride 4 KB on
        // Qwen3.5), consecutive K-substep TMA writes would overlap
        // by ~1.75 KB and corrupt each other.  The tile-major
        // layout gives every K-substep its own self-contained 2 KB
        // slot at offset `kblk * (BS * K_STEP_WGMMA * 2)`, matching
        // the TMA's natural compact write order, and lets
        // `routing_phase_quantize` read `bf16_in_full[kblk][token]`
        // as a natural `[K_STEP_WGMMA]` row.
        //
        // Aligned to 1024 B to inherit the SWZ128 alignment of the
        // unioned weight buffers (the bf16 activation TMA descriptor
        // itself uses SWIZZLE_NONE so 16 B alignment would suffice).
        //
        //
        // Sizing (per slot):
        //   w_wgmma[2][W_WGMMA_M_TOTAL][K_STEP_WGMMA=128]
        //     K_STEP_UP=128: W_WGMMA_M_TOTAL=128 → 16 KB
        //     K_STEP_UP=256: W_WGMMA_M_TOTAL=256 → 32 KB
        //                    (2 substeps × 128 M rows × 128 K bytes
        //                     stacked along M)
        //   w_down_wgmma[2][W_DOWN_WGMMA_M_TOTAL][K_STEP_WGMMA=128]
        //     DOWN_COL_TILE=128, K_STEP_DOWN=128: 16 KB
        //     DOWN_COL_TILE=256, K_STEP_DOWN=128: 32 KB
        //     DOWN_COL_TILE=256, K_STEP_DOWN=256: 64 KB (2 substeps ×
        //                                                256 M rows ×
        //                                                128 K bytes
        //                                                stacked along M)
        //
        // The union picks max(BF16_IN_FULL_BYTES, W_UP_BYTES,
        // W_DOWN_BYTES); the smaller view's tail bytes are unused
        // during its phase (Phase 1/2 doesn't touch the weight
        // views, Phase 3 doesn't touch the bf16 input view, and
        // Phase 4 doesn't touch the up-proj weight view — separated
        // by the Phase-2 trailing __syncthreads() and the Phase 3→4
        // grid sync).
        alignas(1024) A_element bf16_in_full[BF16_IN_FULL_K_BLOCKS][BF16_IN_FULL_BS]
                                            [BF16_IN_FULL_K];  // 32 KB (Phase 1/2)
#ifdef MONO_PROFILE_BARW_4DEEP
        // Up-proj weight slots (BARW_4DEEP variant): 4 slots for the
        // 2-iter lookahead pipeline.  At iter `s` the launcher
        // arms+TMAs slot `(s+2) & 3`; calc waits on `bar_w[s & 3]`.
        // Slots wrap modulo 4 every 4 iters, and the calc-side
        // wait at iter `s+1` provides the publish acquire on slot
        // `(s+1) & 3` before iter `s+3`'s arm targets the same slot
        // (3-iter wraparound > 2-iter calc/launcher gap).
        //
        // SHM cost: same as the 2-slot variant, because the union
        // is dominated by `w_down_wgmma` at 128 KB.
        alignas(1024)
            W_element w_wgmma[4][W_WGMMA_M_TOTAL][W_WGMMA_K];  // 4 slots × 32 KB = 128 KB total
#else
        alignas(1024) W_element
            w_wgmma[2][W_WGMMA_M_TOTAL][W_WGMMA_K];  // 128 wide × M stacked atoms (up-proj)
#endif
        alignas(1024) W_element
            w_down_wgmma[2][W_DOWN_WGMMA_M_TOTAL][CoreDims::K_STEP_WGMMA];  // 128 wide × M
                                                                            // stacked atoms
                                                                            // (down-proj)
      };

      static constexpr uint32_t DOWN_ACT_HALVES_PER_EXPERT =
          Dims::N / 64u;  // 8 for N=512 (one fp32 scale per per-64-K
                          // up-block per token, full reduction dim)
      // Per-token activation scales for the WHOLE expert, loaded once
      // at the top of the per-expert loop (NOT per K-step).  The K-loop
      // indexes this as `a_down_scale[s * K_SUBSTEPS_DOWN * 2 + 2 * kk +
      // half][tok]` to pick the half covering the current 64-K
      // sub-block.  Hoisting to per-expert removes the per-K-step
      // cp.async + pipe drain that was the only consumer of the
      // `cuda::pipeline` in the down-proj path.
      //
      // Layout note (bank-conflict avoidance, 2026-05): the inner index
      // is `tok` for the same reason as `MoE_SHM::act_scale` above —
      // the WGMMA scale-apply broadcasts each `(global_half, tok)` pair
      // across the 8 four-lane groups in a warp, so a `[half][tok]`
      // layout puts every read on a distinct bank within one 32-B row,
      // eliminating the 2-way bank conflict the legacy `[tok][half]`
      // layout produced (32-B row stride mod 32 banks lined `tok=0,4`
      // up on the same bank).
      S_element a_down_scale[DOWN_ACT_HALVES_PER_EXPERT][CoreDims::T_TILE];

      static constexpr uint32_t W_DOWN_SCALE_COLS = shm_down_scale_cols<Dims>::value;
      S_element w_down_scale[2][2][W_DOWN_SCALE_COLS];

      static constexpr uint32_t DOWN_SCALE_TILE_SIZE =
          ((CoreDims::W_DOWN_TILE + 127) / 128) * ((Dims::N + 127) / 128);
      S_element scale[2][DOWN_SCALE_TILE_SIZE + CoreDims::PADDING];

      static constexpr uint32_t UP_SCALE_TILE_SIZE = 2 * shm_up_scale_cols<Dims>::value;
      S_element up_scale[2][UP_SCALE_TILE_SIZE];

      union {
        T_element up[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];
        T_element down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2]
                      [CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
        // wgmma_out: bank-conflict-free row stride.
        //
        // Reader access pattern (in `up_silu_quant_writeback_one_token`):
        // every lane in a warp reads `wgmma_out[col_in_half + offset][tok]`
        // with `col_in_half = lane` (0..31) and `tok = warp` (uniform).
        // Address = base + lane*(row_stride_bytes) + tok*4.  With the
        // natural row stride of 8 floats = 32 bytes = 8 banks, lane k's
        // bank is `(lane*8 + tok) mod 32`, which collapses 32 lanes into
        // only 4 distinct banks → 8-way bank conflict on every LDS (4
        // LDS per (lane, tok) for gate1/up1/gate2/up2).
        //
        // Padding the row stride to 9 floats = 36 bytes makes the bank
        // index `(lane*9 + tok) mod 32`, which is bijective over 32
        // lanes (gcd(9, 32) = 1) → conflict-free reads.  The 9th column
        // is unused.
        //
        // SHM cost: 128 × 1 × 4 = 512 extra bytes for `wgmma_out`, but
        // the union is dominated by `down_out[256][8] = 8 KB`, so the
        // total union footprint is unchanged.
        T_element wgmma_out[128][CoreDims::T_TILE + 1];
        T_element down_out[CoreDims::DOWN_COL_TILE][CoreDims::T_TILE];
      } partial_result;

      static constexpr uint32_t OUT_ACCUM_ROW_PAD = 1;
      static constexpr uint32_t OUT_ACCUM_COLS = CoreDims::W_DOWN_TILE > CoreDims::DOWN_COL_TILE
                                                     ? CoreDims::W_DOWN_TILE
                                                     : CoreDims::DOWN_COL_TILE;
      T_element out_accum[Dims::BS][OUT_ACCUM_COLS + OUT_ACCUM_ROW_PAD];

      // ── TMA-only extensions ─────────────────────────────────────────
      //
      // Weight-tile mbarriers (one per double-buffer slot).  The launcher
      // arms `bar_w[slot]` with `mbarrier.arrive.expect_tx tx_bytes=16384`
      // before the 4 sub-tile TMAs that populate `w_wgmma[slot]`; WGMMA
      // consumers poll via `mbarrier.try_wait.parity` (R3.1, R3.3, R3.5).
      //
      // ── Sized for the deepest lookahead the up-proj K-loop uses ──
      //
      // The default 1-deep lookahead (steady-state pipeline) only uses
      // slots [0, 1].  The 2-deep lookahead variant (gated on
      // `MONO_PROFILE_BARW_4DEEP`) uses all four slots to stagger the
      // cross-expert stitch one iter earlier, giving DRAM more time to
      // drain between the stitch and the next expert's iter-1 weight
      // TMA.  The down-proj path (which has its own pipeline structure)
      // re-initializes only slots [0, 1] in its prologue and leaves
      // slots [2, 3] untouched — they aren't waited on by the down-proj
      // K-loop, so leaving them un-init is safe.
      //
      // Note on SHM impact: `w_wgmma` is unioned with `w_down_wgmma` at
      // 128 KB (DOWN_COL_TILE=256, K_STEP_DOWN=256), which dominates
      // the union regardless of `w_wgmma`'s slot count.  Widening
      // `w_wgmma[2]` to `w_wgmma[4]` (also gated on the same macro,
      // see below) does not increase the union size.
      //
      // Down-projection activation-tile mbarriers (one per double-
      // buffer slot).  Used EXCLUSIVELY by the down-projection
      // (Phase 4): the down-proj launcher arms `bar_a[slot]` before
      // each per-expert bulk activation TMA into `a_down_wgmma[slot]`,
      // and the down-proj WGMMA K-loop waits via
      // `mbarrier.try_wait.parity`.  Re-initialized in the down-proj
      // prologue (`moe_down_projection.cu`) so the up-side init in
      // `moe.cu`'s prologue is no longer required.  No up-projection
      // consumer remains — Phase 3 sources its activation operand
      // from `fp8_act_full`.
      //
      // `alignas(16)` satisfies R11.4 and the 16-byte alignment that the
      // SM90 `mbarrier.*.shared::cta.b64` instructions require.
#ifdef MONO_PROFILE_BARW_4DEEP
      alignas(16) uint64_t bar_w[4];  // 32 B (2-deep lookahead)
#else
      alignas(16) uint64_t bar_w[2];  // 16 B (1-deep lookahead)
#endif
      alignas(16) uint64_t bar_a[2];  // 16 B

      // ── routing-window mbarrier ─────────────────
      // Single mbarrier (`arrival_count = 1`) used to hand off the
      // Phase-1 routing-window TMA load (full BF16 input tile, 32 KB)
      // from the TMA launcher thread to warps 1..11 at the start of
      // Phase 2.  Armed once by the launcher with
      //   `tx_bytes = BS * K_BLOCKS_TOTAL * K_STEP_WGMMA *
      //              sizeof(A_element)`
      // (= 8 × 16 × 128 × 2 = 32 KB for Qwen3.5).  Initialized in the
      // kernel prologue alongside `bar_w[0..1]`, before the
      // `fence_mbarrier_init_release_cluster()` and any
      // `mbarrier.arrive.expect_tx`.  Re-init not needed; the wait
      // drains it.
      //
      // `alignas(16)` matches `bar_w` / `bar_a` and the SM90
      // `mbarrier.*.shared::cta.b64` instruction alignment.
      alignas(16) uint64_t bar_rwin;

      // ── Phase 3 → Phase 4 (expert, token) reorganization tables ─────
      //
      // Added by the `tma-wgmma-down-projection` spec for the Phase-4
      // TMA activation-load path (R11).  Only populated when
      // `use_tma<Dims>::value` is true — on the cp.async reference path
      // these fields are allocated inside the `tiny_wgmma_tma` union
      // variant but never read or written, so non-TMA SHM layouts stay
      // byte-identical (R13.1, R13.2).  These fields live in the
      // `tiny_wgmma_tma` variant only (not in `TinyDataWGMMA`) so the
      // non-TMA WGMMA variant's SHM layout is also unchanged.
      //
      //   expert_slot_start[id]   = first row in spec->temp_fp8 reserved
      //                             for expert `id` under the expert-
      //                             sorted layout produced by the
      //                             Phase-3 epilogue.  Inactive experts
      //                             have the same value as the next
      //                             active expert (zero-width slice).
      //   expert_routed_count[id] = number of routed (tok, k_in_topk)
      //                             pairs that select expert `id`;
      //                             range [0, batch_size * top_k].
      //   sorted_slot[pair]       = destination row in spec->temp_fp8
      //                             for the up-proj SiLU+fp8 writeback,
      //                             where `pair = tok * top_k + k_in_topk`.
      //                             Value = expert_slot_start[eid] +
      //                             intra-expert rank of (tok, k_in_topk).
      //
      // Sizing for BS=8, top_k=8, NUM_EXPERTS=256:
      //   expert_slot_start   : uint16 × 256 = 512 B  (max value < 64)
      //   expert_routed_count : uint8  × 256 = 256 B  (max value ≤ 64)
      //   sorted_slot         : uint8  ×  64 =  64 B  (max value < 64)
      //   Total ≤ 832 B, comfortably inside the 228 KB SHM budget
      //   (R14.1, R14.2, R14.3).
      //
      // Access pattern (R11.1, R11.2, R11.7):
      //   * Phase 3 epilogue reads `sorted_slot[pair]` once per routed
      //     pair to pick the fp8 writeback row.
      //   * Phase 4 launcher reads `expert_slot_start[id]` and
      //     `expert_routed_count[id]` once per expert to parameterize
      //     the bulk activation TMA.
      //   * Phase 4 epilogue walks the (tok, k_in_topk) grid
      //     in `topk_ids_flat`, filters by the current expert id, then
      //     derives the intra-expert rank as
      //     `sorted_slot[pair] - expert_slot_start[id]` to index the
      //     SHM slot — no dedicated inverse table needed.
      static constexpr uint32_t MAX_TOPK = 8;
      static constexpr uint32_t MAX_PAIRS = Dims::BS * MAX_TOPK;
      // `alignas(16)` is required (NOT cosmetic): Phase B of
      // `prepare_moe_topk_BS8` (in `moe_routing.cu`) emits a packed
      // 16-byte STS.128 store per warp lane via
      //   `*reinterpret_cast<uint4*>(&expert_slot_start[tid * BLK]) =
      //    packed_starts;`
      // with `BLK = 8` u16 elements (= 16 B) per lane.  The compiler
      // emits a 128-bit vector shared store that requires the
      // destination address to be 16-byte aligned; `tid * BLK * 2 =
      // tid * 16` is 16-aligned only if the array base itself is
      // 16-aligned.  Without this `alignas(16)` the routing-window
      // mbarrier `bar_rwin` (8 B + alignas(16) → 8 B of trailing
      // padding consumed by the next field) shifts
      // `expert_slot_start[]` to an 8-B-aligned-but-not-16-B-aligned
      // offset, which causes a `cudaErrorMisalignedAddress` at the
      // STS.128 issue.  The writer comment in `moe_routing.cu`
      // already documents this expectation; this `alignas(16)` makes
      // the contract explicit on the declaration side so future SHM
      // layout changes cannot silently break it.
      alignas(16) uint16_t expert_slot_start[Dims::NUM_EXPERTS];
      // `alignas(16)` is required (NOT cosmetic): `prepare_moe_topk_BS8`
      // (in `moe_routing.cu`) zero-inits this array with a per-lane
      // STS.64 (`*reinterpret_cast<uint64_t*>(&expert_routed_count[tid *
      // BLK]) = 0ull`, BLK = 8) and later reads it back with the matching
      // 64-bit `reinterpret_cast` load.  Both require the array base to be
      // ≥8-byte aligned (per-lane base `tid * BLK` is 8-aligned only when
      // the base itself is).  It currently lands 16-aligned by virtue of
      // the preceding `alignas(16) uint16_t expert_slot_start[256]` (= 512
      // B, a multiple of 16), but that is incidental — making the
      // alignment explicit here means a future SHM-layout change to the
      // preceding field cannot silently shift this to an unaligned offset
      // and trigger `cudaErrorMisalignedAddress` at the STS.64 / LD.64.
      alignas(16) uint8_t expert_routed_count[Dims::NUM_EXPERTS];
      uint8_t sorted_slot[MAX_PAIRS];
      // Per-expert per-token cached rank used by the down-proj
      // accumulate loop (Phase 4 epilogue).  Rebuilt at the top of each
      // expert iteration by 8 threads (one per token) so the inner
      // (tok, col) loop can do a single SHM lookup instead of an
      // 8-iter inner scan over `topk_ids_flat`.  Sentinel 0xFF means
      // "this token does not route to the current expert; skip the
      // contribution".  Sized to `Dims::BS = 8` bytes — negligible
      // SHM overhead.
      uint8_t rank_for_tok[Dims::BS];
#ifdef MONO_PROFILE_DEFER_UP_EPILOGUE
      // Per-expert per-token cached top-k INDEX for the up-proj
      // SiLU+fp8 quant writeback under DEFER.  For each token `tok`,
      // holds the smallest `k ∈ [0, top_k)` such that
      // `topk_ids_flat[tok*MAX_TOPK + k] == id`, or sentinel `0xFF`
      // if no such `k` exists (token does not route through this
      // expert).
      //
      // Computed once per expert by 8 calc threads at the K-loop
      // tail (no extra sync — published by the existing wgmma_out
      // publish `__syncthreads()`).  Read by the deferred SiLU body
      // on prefetch warps inside the next expert's K-loop iters
      // 0/1, replacing the 8-iter scan.
      //
      // ── Why DEFER-only ──
      // The inline path also does the topk scan but every (lane, tok)
      // pair is broadcasting the same SHM bytes — bank-conflict-free,
      // cheap.  The deferred path runs the scan CONCURRENTLY with
      // calc-warp WGMMA which contends for SHM banks; that's where
      // the scan stretches from ~0.05 µs to ~0.46 µs.  The cache
      // replaces the contended scan with a single `[tok]` byte read
      // from a hot SHM cacheline.
      //
      // SHM cost: 8 bytes per block (Dims::BS = 8).
      uint8_t up_rank_for_tok[Dims::BS];
#endif
    } tiny_wgmma_tma;

    // ── Aliasing safety static_asserts ─────────────────
    // The new `bf16_in_full` field of `TinyDataWGMMA_TMA` is unioned
    // with `w_wgmma` and `w_down_wgmma` so the BS8 TMA+WGMMA Phase-1/2
    // BF16 input tile shares bytes with the up- and down-projection
    // weight tiles (lifetimes are strictly disjoint.
    // Verify at compile time that the three views start at the same 
    // offset within `TinyDataWGMMA_TMA`
    // so a future layout change that accidentally moves `bf16_in_full`
    // out of the anonymous union (and therefore breaks aliasing) is
    // caught at the build step rather than producing silent SHM
    // corruption at runtime.
    //
    // Placed immediately after the `TinyDataWGMMA_TMA` struct
    // definition (the struct is now complete here; the asserts cannot
    // live inside the struct because `offsetof` requires a complete
    // type).  Still inside `union U` of the enclosing
    // `MoE_SHM<Dims>` so `Dims` is in scope.
    //
    static_assert(offsetof(typename U::TinyDataWGMMA_TMA, bf16_in_full) ==
                      offsetof(typename U::TinyDataWGMMA_TMA, w_wgmma),
                  "bf16_in_full must alias w_wgmma exactly.");
    static_assert(offsetof(typename U::TinyDataWGMMA_TMA, bf16_in_full) ==
                      offsetof(typename U::TinyDataWGMMA_TMA, w_down_wgmma),
                  "bf16_in_full must alias w_down_wgmma exactly.");
  } u;

  static_assert(Dims::NUM_EXPERTS <= 65535,
                "Number of experts too high, cannot store as uint16 anymore.");

  // act_scale[blk][tok] = max(|x_tok[blk*128..(blk+1)*128-1]|)/448
  //
  // Per-token block-wise activation quantization scales for up-projection.
  //
  // Layout note (bank-conflict avoidance, 2026-05): the inner index is
  // `tok` so the row stride along `blk` is `Dims::BS * sizeof(float) = 32 B
  // = 8 banks` (for BS=8).  The up-proj WGMMA scale-apply broadcasts a
  // single `(blk, tok)` pair across the 8 four-lane groups in a warp, so
  // every lane-group reads the same `blk` row but a different `tok`
  // word.  With this layout each `tok ∈ {0,2,4,6}` (resp. {1,3,5,7})
  // lands on a distinct bank within the same row → conflict-free.  The
  // legacy `[Dims::BS][ACT_SCALE_BLOCKS]` layout had a 64-B row stride
  // (= 16 banks); for BS=8 the four toks read from the same warp lane
  // mapped to the same bank, producing a 4-way conflict per LDS that
  // NCU flagged as the dominant excessive-wavefront source.  
  static constexpr uint32_t ACT_BLOCK_SIZE = 128;
  static constexpr uint32_t ACT_SCALE_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK_SIZE - 1) / ACT_BLOCK_SIZE;
  S_element act_scale[ACT_SCALE_BLOCKS][Dims::BS];

  // Unique experts active in this batch, with their sorted token ranges.
  // Filled by prepare_moe_topk_BS8.
  ExpertRef experts[Dims::NUM_EXPERTS];
  std::uint32_t expert_count;

  // Flat routing results: [token * MAX_TOPK + k] = expert id / routing weight
  // for the k-th selection of that token. Written by topK_BS8.
  // MAX_TOPK = 8 covers top_k up to 8.
  static constexpr uint32_t MAX_TOPK = 8;
  alignas(uint64_t) uint16_t topk_ids_flat[(Dims::BS < 8 ? 8 : Dims::BS) * MAX_TOPK];
  S_element topk_weights_flat[(Dims::BS < 8 ? 8 : Dims::BS) * MAX_TOPK];
};

/**
 * @brief Returns the amount of shared memory necessary to run @c moe_kernel
 * with template parameter @p Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_shmem_size() {
  static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS,
                "Dimension larger than the maximum supported dimension.");
  // Per-block dynamic SHM budget on Hopper (H100) is 228 KB; we target
  // 224 KB to leave margin for driver overhead.
  static_assert(sizeof(MoE_SHM<Dims>) <= 224 * 1024,
                "MoE_SHM layout exceeds the 224 KB per-block SHM budget.");
  return sizeof(MoE_SHM<Dims>);
}

constexpr size_t get_moe_max_shmem_size() { return sizeof(MoE_SHM<Dims_Max>); }

constexpr size_t get_moe_max_scratchpad_size() { return sizeof(MoEGemmSpec<Dims_Max>); }

template <typename Dims>
inline __device__ bool is_calc_warp() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x < CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ bool is_prefetch_warp() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x >= CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_thread() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x % CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_any_warp() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_calc_warp() {
  using CoreDims = MoECoreDims<Dims>;
  assert(is_calc_warp<Dims>());
  return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_prefetch_warp() {
  using CoreDims = MoECoreDims<Dims>;
  assert(is_prefetch_warp<Dims>());
  return threadIdx.x / CoreDims::THREADS_PER_WARP - CoreDims::CALC_WARP_COUNT;
}

/**
 * @brief Identifies the single TMA launcher thread (warp 8, lane 0).
 *
 * Returns `true` for exactly one thread in the block. Only meaningful inside
 * Phase 3 (the WGMMA up-projection K-loop) of the TMA kernel variant; that
 * thread is the unique issuer of every `cp.async.bulk.tensor.2d` and
 * `mbarrier.arrive.expect_tx` for the block during Phase 3.
 */
template <typename Dims>
inline __device__ bool is_tma_launcher_thread() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x == 8u * CoreDims::THREADS_PER_WARP;
}

/**
 * @brief Synchronizes the first 256 threads of the calling CUDA block
 *
 * This is a collective operation that needs to be called by all of the first
 * 256 threads in each CUDA block.
 *
 */
template <typename Dims>
__device__ __forceinline__ void sync_calc_threads() {
  // First 256 threads
  using CoreDims = MoECoreDims<Dims>;
  static_assert(CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP == 256,
                "Adapt the thread number if sync_calc_threads");
  __asm volatile("bar.sync  15, 256;\n");
}

/**
 * @brief Computes the maximum value within a warp
 *
 * This is a collective operation. Each thread in a warp needs to call it.
 * The resulting maximum value is returned on all threads.
 *
 */
__device__ static inline float warp_reduce_max_float(float value) {
  for (int i = 16; i >= 1; i /= 2) {
    value = fmaxf(__shfl_xor_sync(0xffffffff, value, i, 32), value);
  }
  return value;
}

/**
 * @brief Reinterprets the bit-pattern of @p x to type @p To
 */
template <typename To, typename From>
__device__ static __forceinline__ To type_pun(From x) {
  static_assert(sizeof(To) == sizeof(From), "Types of different size");
  To y;
  // This memcpy is optimized out by NVCC
  memcpy(&y, &x, sizeof(From));
  return y;
}

}  // namespace moe_monokernel

#endif
