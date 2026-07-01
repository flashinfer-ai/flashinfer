
#pragma once
#ifndef MOE_INTERNAL_H
#define MOE_INTERNAL_H

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include "moe_interface.h"

// Full 32-lane warp mask for the `*_sync` warp intrinsics.  Previously
// supplied by the vLLM tree's `cuda_utils.h`; defined here so the monomoe
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

namespace monomoe {

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

// ── KernelConfig accessors ───────────────────────────────────────────────
// `KernelConfig` always carries USE_WGMMA / USE_TMA / K_STEP_DOWN / K_STEP_UP:
// the generic `MoEDimensions` provides defaults (false / false / 128 / 128)
// and concrete shape variants override them, so these are plain reads.
//
// K_STEP_{UP,DOWN} must be a multiple of 128 (the SWZ128 atom K-width) and a
// divisor of the reduction dim (HIDDEN_STATES / N respectively).  Bumping a
// K-step to 256 halves the outer K-iteration count (amortizing launch / bar /
// sync overhead over more compute) at the cost of a larger per-slot SHM tile.
template <typename Dims>
inline constexpr bool use_wgmma_v = Dims::KernelConfig::USE_WGMMA;
template <typename Dims>
inline constexpr bool use_tma_v = Dims::KernelConfig::USE_TMA;
template <typename Dims>
inline constexpr std::uint32_t down_k_step_v = Dims::KernelConfig::K_STEP_DOWN;
template <typename Dims>
inline constexpr std::uint32_t up_k_step_v = Dims::KernelConfig::K_STEP_UP;

/**
 * @brief Scratchpad memory for use within the MonoMoe kernel.
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
  // Used when `use_wgmma_v<Dims> == true`.  The up-projection
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

  // Byte offset of `temp_fp8`, so the host can address it from the
  // scratchpad base without reading the layout at runtime (docs/design_docs/monomoe_kernel.md §4).
  static constexpr size_t TEMP_FP8_OFFSET = offsetof(MoEGemmSpec<Dims>, temp_fp8);

  // Down-projection group layout (docs/design_docs/monomoe_kernel.md §6): DOWN_COL_TILE=256,
  // DOWN_GRID=8, DOWN_GROUPS=16 for the BS8 TMA path.  `DOWN_GROUPS ==
  // UP_GROUPS` is the prerequisite for the Site-#2 Expert_Barrier; the value
  // is mirrored in MoECoreDims with a cross-checking static_assert.
  static constexpr uint32_t DOWN_COL_TILE = (use_tma_v<Dims> && Dims::BS <= 8) ? 256u : 128u;
  static constexpr uint32_t DOWN_GRID = Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;
  // Single-buffer fp32 accumulator for the down-projection (docs/design_docs/monomoe_kernel.md §1).
  // [BS][HIDDEN_STATES] = 64 KB for Qwen3.5.  fp32 (not bf16) so the 16-way
  // Phase-4 atomicAdd keeps precision; Phase 5 casts each cell once.  Zeroed
  // at kernel entry, fenced by the Site-#2/#3 barriers.
  float down_partial_out[Dims::BS * Dims::HIDDEN_STATES];

  // Per-token block-wise activation quantization scales.
  // Block size = 128 along K dimension → K/128 scales per token.
  // act_scale[tok][blk] = max(|x_tok[blk*128..(blk+1)*128-1]|) / 448
  static constexpr uint32_t ACT_BLOCK_SIZE = 128;
  static constexpr uint32_t ACT_SCALE_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK_SIZE - 1) / ACT_BLOCK_SIZE;
  float act_scale[Dims::BS][ACT_SCALE_BLOCKS];

  // Software barrier counters (docs/design_docs/monomoe_kernel.md §2), at the TAIL of the struct so
  // TEMP_FP8_OFFSET is unaffected (docs/design_docs/monomoe_kernel.md §4).  Counter_Pairs are
  // ping-pong slots: `grid_barrier.slot[2]` (all blocks),
  // `expert_slot[id][2]` (one per expert group, arrival UP_GRID),
  // `colstripe_slot[id][2]` (one per col stripe, arrival DOWN_GROUPS).
  // ~2 KB total — negligible.  Uses the local `DOWN_GRID` (not
  // `MoECoreDims<Dims>::DOWN_GRID`, which is an incomplete forward ref here);
  // the two match via a MoECoreDims static_assert.
  struct {
    uint32_t slot[2];
  } grid_barrier;
  struct {
    uint32_t expert_slot[Dims::NUM_EXPERTS][2];
    uint32_t colstripe_slot[DOWN_GRID][2];
  } partial_barrier;
};

// Upper bound on the supported dimensions, used by the `static_assert`
// bounds-checks in `get_moe_shmem_size<Dims>()`.
using Dims_Max = MoEDimensions<1024, 1024, 5120, 256>;

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
// `MoEGemmSpec<Dims>`) so `MoEGemmSpec` can use `use_wgmma_v<Dims>`
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
 * @brief contains various constants used within the MonoMoe kernel.
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
  static_assert(!use_wgmma_v<Dims> || Dims::HIDDEN_STATES % K_STEP_WGMMA == 0,
                "HIDDEN_STATES must be a multiple of 128 for the WGMMA path "
                "(one K-step consumes K=128)");
  static_assert(!use_wgmma_v<Dims> || (2 * Dims::N) % W_UP_TILE_WGMMA == 0,
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
  static constexpr std::uint32_t K_STEP_DOWN = down_k_step_v<Dims>;
  static constexpr std::uint32_t K_SUBSTEPS_DOWN = K_STEP_DOWN / K_STEP_WGMMA;
  static_assert(K_STEP_DOWN >= K_STEP_WGMMA && K_STEP_DOWN % K_STEP_WGMMA == 0,
                "K_STEP_DOWN must be a positive multiple of K_STEP_WGMMA "
                "(=128, the SWZ128 atom K-width).");
  static_assert(!use_wgmma_v<Dims> || Dims::N % K_STEP_DOWN == 0,
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
  static constexpr std::uint32_t K_STEP_UP = up_k_step_v<Dims>;
  static constexpr std::uint32_t K_SUBSTEPS_UP = K_STEP_UP / K_STEP_WGMMA;
  static_assert(K_STEP_UP >= K_STEP_WGMMA && K_STEP_UP % K_STEP_WGMMA == 0,
                "K_STEP_UP must be a positive multiple of K_STEP_WGMMA "
                "(=128, the SWZ128 atom K-width).");
  static_assert(!use_wgmma_v<Dims> || Dims::HIDDEN_STATES % K_STEP_UP == 0,
                "Dims::HIDDEN_STATES must be a multiple of K_STEP_UP for "
                "the WGMMA up-projection (one outer K-step consumes "
                "K_STEP_UP K-elements of the reduction dim).");
  // Up-proj outer K-loop iteration count (replaces K_TILES_WGMMA in
  // call sites that should follow the K_STEP_UP setting).
  static constexpr std::uint32_t K_TILES_UP = Dims::HIDDEN_STATES / K_STEP_UP;

  // Effective M (row-tile) size of one block's up-proj work — 64 for the
  // WGMMA path, 16 for the scalar path.  Used to compute UP_GRID = 2*N/M.
  static constexpr std::uint32_t W_UP_TILE_EFFECTIVE =
      use_wgmma_v<Dims> ? W_UP_TILE_WGMMA : W_UP_TILE;

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
  static constexpr std::uint32_t DOWN_COL_TILE = (use_tma_v<Dims> && Dims::BS <= 8) ? 256u : 128u;
  static constexpr std::uint32_t DOWN_GRID = Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr std::uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;

  static_assert(!use_wgmma_v<Dims> || Dims::HIDDEN_STATES % DOWN_COL_TILE == 0,
                "HIDDEN_STATES must be a multiple of DOWN_COL_TILE for the "
                "WGMMA down-projection (one down-block owns DOWN_COL_TILE "
                "output cols)");
  static_assert(!use_wgmma_v<Dims> || Dims::KernelConfig::GRID_SIZE % DOWN_GRID == 0,
                "GRID_SIZE must be a multiple of DOWN_GRID for the WGMMA "
                "down-projection (expert groups partition the grid)");
  static_assert(!use_wgmma_v<Dims> || DOWN_GROUPS <= Dims::NUM_EXPERTS,
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
  // `tiny_wgmma_tma` holds the SHM for the TMA+WGMMA path.  Its inner
  // anonymous unions overlap buffers with disjoint lifetimes (notably the
  // Phase-1 `bf16_in_full` over the Phase-3/4 weight tiles), which is where
  // the real SHM byte-sharing happens.

  // ── TinyDataWGMMA_TMA: SHM layout for the TMA+WGMMA up-proj path ──
  //
  // Used when `use_wgmma_v<Dims> && use_tma_v<Dims>` are both
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

    // BF16 input buffer in tile-major `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]`
    // (not `[BS][HIDDEN_STATES]`) so each `boxDim=(128,8)` TMA write lands
    // in its own 2 KB slab without overlapping the next K-substep; full
    // rationale in docs/design_docs/monomoe_kernel.md §5.  Total 32 KB,
    // unchanged from the natural shape, so the union and SHM budget are
    // unaffected.
    static constexpr uint32_t BF16_IN_FULL_K_BLOCKS = K_BLOCKS_TOTAL;
    static constexpr uint32_t BF16_IN_FULL_BS = Dims::BS;
    static constexpr uint32_t BF16_IN_FULL_K = CoreDims::K_STEP_WGMMA;
    static constexpr uint32_t FP8_ACT_FULL_K_BLOCKS = K_BLOCKS_TOTAL;

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
      // Tile-major `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` (not
      // `[BS][HIDDEN_STATES]`) so each `boxDim=(128,8)` TMA write gets its
      // own 2 KB slot in the TMA's compact write order, and
      // `routing_phase_quantize` reads `bf16_in_full[kblk][token]` as a
      // natural row — rationale in docs/design_docs/monomoe_kernel.md §5.
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
      alignas(
          1024) W_element w_wgmma[4][W_WGMMA_M_TOTAL][W_WGMMA_K];  // 4 slots × 32 KB = 128 KB total
#else
      alignas(1024)
          W_element w_wgmma[2][W_WGMMA_M_TOTAL][W_WGMMA_K];  // 128 wide × M stacked atoms (up-proj)
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
    // consumers poll via `mbarrier.try_wait.parity`.
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
    // prologue (`moe_down_projection.cuh`) so the up-side init in
    // `moe.cuh`'s prologue is no longer required.  No up-projection
    // consumer remains — Phase 3 sources its activation operand
    // from `fp8_act_full`.
    //
    // `alignas(16)` satisfies the 16-byte alignment that the
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
    // TMA activation-load path.  Only populated when
    // `use_tma_v<Dims>` is true — on the cp.async reference path
    // these fields are allocated inside the `tiny_wgmma_tma` union
    // variant but never read or written, so non-TMA SHM layouts stay
    // byte-identical.  These fields live in the
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
    //   Total ≤ 832 B, comfortably inside the 228 KB SHM budget.
    //
    // Access pattern:
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
    // `prepare_moe_topk_BS8` (in `moe_routing.cuh`) emits a packed
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
    // STS.128 issue.  The writer comment in `moe_routing.cuh`
    // already documents this expectation; this `alignas(16)` makes
    // the contract explicit on the declaration side so future SHM
    // layout changes cannot silently break it.
    alignas(16) uint16_t expert_slot_start[Dims::NUM_EXPERTS];
    // `alignas(16)` is required (NOT cosmetic): `prepare_moe_topk_BS8`
    // (in `moe_routing.cuh`) zero-inits this array with a per-lane
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
    // ── Routing-time inverse map: (expert id, token) → top-k rank ────────
    //
    // `expert_tok_krank[eid * Dims::BS + tok]` holds the top-k rank
    // `k ∈ [0, top_k)` at which token `tok` routes to expert `eid`, or
    // sentinel `0xFF` when the token does not route through that expert.
    //
    // Built ONCE, at routing time, by `prepare_moe_topk_BS8` (Phase C in
    // `moe_routing.cuh`), which already enumerates every routed
    // `(tok, k, eid)` pair to fill `sorted_slot`.  It scatters `k` into
    // this table on the same pass — no extra scan.  This is the single
    // source of truth both projection epilogues consume instead of each
    // re-deriving `k` per expert:
    //
    //   * Up-proj  (`up_silu_quant_writeback_one_token`): reads
    //     `k = expert_tok_krank[id*BS + tok]`; a valid `k` yields the
    //     routing weight `topk_weights_flat[tok*MAX_TOPK + k]` and the
    //     fp8 writeback row `sorted_slot[tok*top_k + k]`.
    //   * Down-proj (Phase-4 epilogue): reads the same `k` and derives
    //     the intra-expert slot `sorted_slot[tok*top_k + k] -
    //     expert_slot_start[id]` into the per-expert `rank_for_tok`
    //     staging array.
    //
    // Replaces the previous per-expert 8-iter scans over
    // `topk_ids_flat` (one at the up-proj K-loop tail, one at the
    // down-proj expert-loop tail) that ran O(BS·top_k) work for EVERY
    // expert this block processes — pure duplication of routing-time
    // information.  Because the table is immutable after routing, it
    // also removes the single-buffer lifetime hazard the old
    // per-expert `up_rank_for_tok` / scan-fed `rank_for_tok` carried
    // across the deferred-epilogue expert boundary.
    //
    // Uniqueness: top-k selection picks DISTINCT experts per token (the
    // selection loop masks each chosen expert to -FLT_MAX), so a given
    // (eid, tok) pair occurs at most once — the scatter has no
    // write conflict and needs no tie-break.
    //
    // SHM cost: NUM_EXPERTS * BS bytes (2 KB for E=256, BS=8).
    //
    // `alignas(16)`: `prepare_moe_topk_BS8` sentinel-fills this table with
    // a per-lane vectorized `uint4` (STS.128) store, which requires a
    // 16-byte-aligned base.  It currently lands 16-aligned after the
    // 64-B `sorted_slot`, but making it explicit means a future layout
    // change to a preceding field cannot silently misalign the fill.
    alignas(16) uint8_t expert_tok_krank[Dims::NUM_EXPERTS * Dims::BS];
    // Per-expert per-token cached intra-expert SLOT rank used by the
    // down-proj accumulate loop (Phase 4 epilogue).  Rebuilt at the top
    // of each expert iteration by 8 threads (one per token) — now a
    // single `expert_tok_krank` lookup + `sorted_slot` read instead of
    // an 8-iter scan.  Kept as a per-expert `[BS]` staging array (rather
    // than reading the routing table directly) because the inner
    // (tok, col) accumulate loop reads it many times per token; a hot
    // `[BS]`-sized row keeps that a broadcast.  Sentinel 0xFF means
    // "this token does not route to the current expert; skip the
    // contribution".
    uint8_t rank_for_tok[Dims::BS];
  } tiny_wgmma_tma;

  // ── Aliasing safety static_asserts ─────────────────
  // `bf16_in_full` is unioned with `w_wgmma` and `w_down_wgmma` so the BS8
  // TMA+WGMMA Phase-1/2 BF16 input tile shares bytes with the up- and
  // down-projection weight tiles (lifetimes are strictly disjoint).  Verify at
  // compile time that the three views start at the same offset within
  // `TinyDataWGMMA_TMA`, so a future layout change that accidentally moves
  // `bf16_in_full` out of the inner anonymous union (and therefore breaks
  // aliasing) is caught at the build step rather than producing silent SHM
  // corruption at runtime.  Placed after the struct definition because
  // `offsetof` requires a complete type.
  static_assert(offsetof(TinyDataWGMMA_TMA, bf16_in_full) == offsetof(TinyDataWGMMA_TMA, w_wgmma),
                "bf16_in_full must alias w_wgmma exactly.");
  static_assert(offsetof(TinyDataWGMMA_TMA, bf16_in_full) ==
                    offsetof(TinyDataWGMMA_TMA, w_down_wgmma),
                "bf16_in_full must alias w_down_wgmma exactly.");

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

}  // namespace monomoe

#endif
