
#pragma once
#ifndef MOE_INTERNAL_H
#define MOE_INTERNAL_H

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include "moe_interface.h"

namespace monomoe {

using T_element = float;  //< fp32 accumulators (partial results, out_accum)

/**
 * @brief One active expert.  Filled by prepare_moe_topk in ascending
 * expert-id order; only the first `expert_count` entries are valid.
 */
struct ExpertRef {
  std::uint32_t id;
};

// ── KernelConfig opt-in detection ───────────────────────────────────────────
// The tunable knobs are optional members of `Dims::KernelConfig`; each
// `xxx<Dims>::value` helper reads the member when present and falls back to
// the historical default otherwise.  See docs/design_docs/monomoe_kernel.md "Tunable configs".

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

// Gate/up pair layout for the up-projection epilogue.  Set by all shipped
// BS8 shapes; enables the register-resident combine + deferred writeback.
template <typename Dims>
struct use_pair_layout {
  template <typename D>
  static constexpr auto test(int) -> decltype(D::KernelConfig::USE_PAIR_LAYOUT, bool()) {
    return D::KernelConfig::USE_PAIR_LAYOUT;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

// K_STEP_DOWN: K-width per outer down-proj K-step.  Multiple of 128 (the
// SWZ128 atom width); must divide Dims::N.  Larger steps amortize launcher
// and barrier-wait overhead over more compute at the cost of larger
// `w_down_wgmma` / `a_down_wgmma` slots.
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

// DOWN_PIPE_DEPTH: down-proj weight/activation TMA ring depth (tunable via
// the config table's DPD column).  Optional on the Dims tag; defaults to 2
// (the classic double buffer — SHM layout and SASS byte-identical for every
// Dims that doesn't declare it).  4
// selects the rolling ring: at iter s the launcher arms slot (s+2)&3, so a
// weight tile has two compute windows to land and the fetch stream runs
// through the per-expert epilogue/sync bubble.  Must be a power of two so
// `s & (DOWN_PIPE_DEPTH - 1)` folds to a mask.
template <typename Dims>
struct down_pipe_depth {
 private:
  template <typename D>
  static constexpr auto test(int) -> decltype((std::uint32_t)D::KernelConfig::DOWN_PIPE_DEPTH) {
    return (std::uint32_t)D::KernelConfig::DOWN_PIPE_DEPTH;
  }
  template <typename>
  static constexpr std::uint32_t test(...) {
    return 2u;
  }

 public:
  static constexpr std::uint32_t value = test<Dims>(0);
};

// K_STEP_UP: K-width per outer up-proj K-step.  Multiple of 128; must divide
// Dims::HIDDEN_STATES.  Scales are still applied at every 128-K boundary
// (the block-wise FP8 quantization granularity).
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

// DOWN_COL_TILE: output cols owned per down-block.  Multiple of 128; must
// divide HIDDEN_STATES.
template <typename Dims>
struct down_col_tile {
 private:
  template <typename D>
  static constexpr auto test(int) -> decltype((std::uint32_t)D::KernelConfig::DOWN_COL_TILE) {
    return (std::uint32_t)D::KernelConfig::DOWN_COL_TILE;
  }
  template <typename>
  static constexpr std::uint32_t test(...) {
    return (use_tma<Dims>::value && Dims::BS <= 8) ? 256u : 128u;
  }

 public:
  static constexpr std::uint32_t value = test<Dims>(0);
};

// UP_W_SLOTS: number of physical bar_w / w_wgmma slots in the up-proj
// weight-TMA lookahead pipeline.  The prefetch distance is derived:
// UP_ARM_DISTANCE = max(1, SLOTS - 2), which keeps a slot's re-arm strictly
// after its previous consumer wait (SLOTS >= ARM + 2).  Default is the
// historical per-shape depth: 4 for single-atom shapes (UP_COL_HALVES == 1),
// 2 for two-atom shapes.
template <typename Dims>
struct up_w_slots {
 private:
  template <typename D>
  static constexpr auto test(int) -> decltype((std::uint32_t)D::KernelConfig::UP_W_SLOTS) {
    return (std::uint32_t)D::KernelConfig::UP_W_SLOTS;
  }
  template <typename D>
  static constexpr std::uint32_t test(...) {
    return (((2u * D::N * down_col_tile<D>::value) / (128u * D::HIDDEN_STATES) >= 2u)) ? 2u : 4u;
  }

 public:
  static constexpr std::uint32_t value = test<Dims>(0);
};

// UP_COL_HALVES: stacked 128-row WGMMA M-atoms per up-block (1 or 2).
//
// When ABSENT, derived from the coupling identity that pins
// UP_GROUPS == DOWN_GROUPS (so the site-#2 expert barrier's producer set
// equals its consumer set):
//   UP_COL_HALVES = (2*N * DOWN_COL_TILE) / (128 * HIDDEN_STATES), min 1.
//
// When PRESENT, taken verbatim and the up/down grids are DECOUPLED
// (UP_GROUPS need not equal DOWN_GROUPS); the sentinel Phase-3→4 handoff
// covers this case natively (consumers poll the published scales they
// read, so producer set != consumer set needs no special protocol).
template <typename Dims>
struct up_col_halves {
 private:
  template <typename D>
  static constexpr auto test(int) -> decltype((std::uint32_t)D::KernelConfig::UP_COL_HALVES) {
    return (std::uint32_t)D::KernelConfig::UP_COL_HALVES;
  }
  template <typename D>
  static constexpr std::uint32_t test(...) {
    return ((2u * D::N * down_col_tile<D>::value) / (128u * D::HIDDEN_STATES) > 0u)
               ? (2u * D::N * down_col_tile<D>::value) / (128u * D::HIDDEN_STATES)
               : 1u;
  }

 public:
  static constexpr std::uint32_t value = test<Dims>(0);
};

/**
 * @brief Global-memory scratchpad for the monokernel.
 *
 * LAYOUT INVARIANT: the host-side TMA wrapper computes the device pointer to
 * `temp_fp8` as `scratchpad_base + TEMP_FP8_OFFSET`, and the Python debug
 * readers in test_monokernel_accuracy.py assume the head field order
 * (activations, temp_fp8, temp_act_scale, down_partial_out).  Never insert
 * fields before `down_partial_out`; new fields go at the tail.
 */
template <typename Dims>
struct MoEGemmSpec {
  static constexpr uint32_t SPEC_MAX_TOPK = 8;
  // Virtual batch size: up to SPEC_MAX_TOPK routed rows per token, plus 8
  // rows of guard padding.
  static constexpr uint32_t TEMP_ROWS = Dims::BS * SPEC_MAX_TOPK + 8;
  // The down-activation TMA descriptor's outer extent excludes the guard
  // padding — the up-proj epilogue only writes rows [0, BS * SPEC_MAX_TOPK).
  static constexpr uint32_t TEMP_ROWS_TMA = Dims::BS * SPEC_MAX_TOPK;

  AQ_element activations[Dims::BS][Dims::HIDDEN_STATES];

  // Phase 3 → Phase 4 handoff: fp8 SiLU output in expert-sorted rows (each
  // expert's routed tokens occupy a contiguous slab — see sorted_slot), with
  // one fp32 scale per (row, up-block) where an up-block covers
  // DOWN_ACT_BLOCK_SIZE = UP_COL_HALVES * 64 intermediate features.
  static constexpr uint32_t UP_COL_HALVES_LOCAL = up_col_halves<Dims>::value;
  static constexpr uint32_t DOWN_ACT_BLOCK_SIZE = (UP_COL_HALVES_LOCAL * 128u) / 2u;
  static_assert(Dims::N % DOWN_ACT_BLOCK_SIZE == 0,
                "Dims::N must be a multiple of DOWN_ACT_BLOCK_SIZE for the "
                "WGMMA down-proj per-block activation quantization scheme");
  static_assert(Dims::HIDDEN_STATES % 128 == 0,
                "Dims::HIDDEN_STATES must be a multiple of 128 for the "
                "WGMMA down-proj 128-cols-per-block grid layout");
  static constexpr uint32_t TEMP_ACT_SCALE_COLS = Dims::N / DOWN_ACT_BLOCK_SIZE;
  AQ_element temp_fp8[TEMP_ROWS * Dims::N];
  float temp_act_scale[TEMP_ROWS * TEMP_ACT_SCALE_COLS];

  // Byte offset of `temp_fp8`, exposed so the host TMA wrapper can address
  // it without reading the struct layout at runtime.
  static constexpr size_t TEMP_FP8_OFFSET = offsetof(MoEGemmSpec<Dims>, temp_fp8);

  // Mirrors of the MoECoreDims values (MoECoreDims is defined later in this
  // file; static_asserts there cross-check the two).
  static constexpr uint32_t DOWN_COL_TILE = down_col_tile<Dims>::value;
  static constexpr uint32_t DOWN_GRID = Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;

  // Phase-4 accumulator: every contributing block atomicAdds its fp32
  // partial sum here; Phase 5 reads each cell once and casts to bf16.
  // Zeroed at kernel entry; the site-#2 barrier publishes the zero before
  // any Phase-4 atomicAdd.  fp32 so the multi-block atomicAdd preserves
  // precision.
  float down_partial_out[Dims::BS * Dims::HIDDEN_STATES];

  // ── Sentinel-based Phase 3 → 4 handoff state (tail fields) ────────────
  //
  // The site-#2 barrier is replaced by data-path readiness (see the
  // site-#2 note in moe.cu and the up/down projection publish/poll
  // sites): a `temp_act_scale` cell doubles as the readiness
  // flag for its fp8 payload segment, with bit pattern 0x00000000 (+0.0f)
  // as the "not yet published" sentinel.  Published scales are clamped to
  // >= FLT_MIN (positive normal) at the publish site, so 0.0 is never a
  // valid value — even under flush-to-zero.  Using 0.0 (not NaN) means
  // the host-side `torch.zeros` scratchpad allocation establishes the
  // sentinel invariant for EVERY scratchpad (one per layer) with no
  // per-scratchpad host init.
  //
  // Reset discipline (double buffer + async reset): consumers must never
  // observe a LEFTOVER scale from the previous launch, so the scale
  // buffer ping-pongs per launch.  Launch parity comes from
  // `launch_flip[blockIdx.x]` — a per-block PRIVATE persistent counter
  // each block increments exactly once per launch (no cross-block
  // synchronization needed; all blocks agree because all count the same
  // launches).  Each launch writes/polls the parity-selected buffer and
  // zero-refills the OTHER buffer for the next launch, off the critical
  // path.
  //
  // `temp_act_scale_alt` lives at the TAIL (not next to `temp_act_scale`)
  // to preserve the head-field layout invariant documented above.
  float temp_act_scale_alt[TEMP_ROWS * TEMP_ACT_SCALE_COLS];
  uint32_t launch_flip[Dims::KernelConfig::GRID_SIZE];

  // ── Phase 4 → 5 readiness flags (replaces the site-#3 barrier) ────────
  //
  // `down_partial_out` is atomicAdd-accumulated, so its readiness cannot
  // be encoded in the data (a partial sum looks like a complete one).
  // Instead each contributing block bumps its col-stripe's arrival
  // counter (fence + atomicAdd) after its Phase-4 adds; the stripe's
  // single Phase-5 writer polls until the count reaches DOWN_GROUPS.
  // Only the 8 Phase-5 writers ever wait — the other blocks publish and
  // exit (the old colstripe_barrier made all 128 blocks spin).
  //
  // Same double-buffer parity reset as the scale sentinel: launch parity
  // selects the active row; the prologue zero-refills the OTHER row for
  // the next launch.  torch.zeros allocation covers the first launch.
  uint32_t down_ready[2][DOWN_GRID];

  static constexpr size_t TEMP_ACT_SCALE_OFFSET = offsetof(MoEGemmSpec<Dims>, temp_act_scale);
  static constexpr size_t TEMP_ACT_SCALE_ALT_OFFSET =
      offsetof(MoEGemmSpec<Dims>, temp_act_scale_alt);
  static constexpr size_t TEMP_ACT_SCALE_BYTES = sizeof(float) * TEMP_ROWS * TEMP_ACT_SCALE_COLS;
};

// ── Sentinel handoff helpers (Phase 3 → 4) ────────────────────────────────

// "Not yet published" test.  The sentinel is exactly +0.0f (bit pattern
// 0x00000000); published scales are clamped to >= FLT_MIN so no valid
// publication can produce it.  (-0.0f cannot occur either: the buffer is
// only ever zero-filled or published-to.)
__device__ __forceinline__ bool moe_scale_is_sentinel(uint32_t bits) { return bits == 0u; }

// Parity-selected scale buffer for the CURRENT launch.
template <typename Dims>
__device__ __forceinline__ float* moe_act_scale_buf(MoEGemmSpec<Dims>* __restrict__ spec,
                                                    uint32_t parity) {
  return parity ? spec->temp_act_scale_alt : spec->temp_act_scale;
}

/**
 * @brief Publish one (row, up-block) activation scale as payload + flag.
 *
 * Caller contract: every lane of the calling warp has already issued its
 * fp8 payload stores for this (row, up-block) segment, and the call is
 * warp-uniform.  `__syncwarp()` joins the lanes, the fence releases the
 * payload at device scope, and the `atomicExch` makes the flag store
 * morally strong so a consumer's device-scope poll that observes a
 * non-sentinel value is guaranteed to also observe the payload.
 *
 * The published value is clamped to >= FLT_MIN so a subnormal scale can
 * never flush to 0.0 (the sentinel) and hang a consumer; a scale that
 * tiny dequantizes to ~0 either way, so the clamp is numerically inert.
 */
template <typename Dims>
__device__ __forceinline__ void moe_publish_act_scale(MoEGemmSpec<Dims>* __restrict__ spec,
                                                      uint32_t parity, uint32_t row,
                                                      uint32_t up_block, float scale,
                                                      unsigned lane) {
  __syncwarp();
  if (lane == 0) {
    __threadfence();
    constexpr uint32_t SCALE_COLS = MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;
    atomicExch(&moe_act_scale_buf<Dims>(spec, parity)[row * SCALE_COLS + up_block],
               fmaxf(scale, __FLT_MIN__));
  }
}

// Maximum supported dimensions.  Sizes only the max-SHM / max-scratchpad
// bookkeeping (`get_moe_max_*`); never launched.
using Dims_Max = MoEDimensions<1024, 1024, 6144, 512>;

// Block-wise quantization detection for the SHM scale-tile sizing below.
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

/**
 * @brief Derived compile-time constants of the monokernel.
 */
template <typename Dims>
struct MoECoreDims {
  using MoEDims = Dims;

  static constexpr std::uint32_t THREADS_PER_WARP = 32;
  static constexpr std::uint32_t TOTAL_WARP_COUNT =
      Dims::KernelConfig::BLOCK_SIZE / THREADS_PER_WARP;
  static constexpr std::uint32_t CALC_WARP_COUNT = 8;
  static constexpr std::uint32_t PREFETCH_WARP_COUNT = TOTAL_WARP_COUNT - CALC_WARP_COUNT;

  // Legacy scalar-path row tile; only feeds W_UP_TILE_EFFECTIVE for
  // non-WGMMA Dims (Dims_Max bookkeeping).
  static constexpr std::uint32_t W_UP_TILE = 16;

  // ── WGMMA tile geometry ───────────────────────────────────────────────
  // One 128-row weight tile per block per K-step, split across two
  // warpgroups (WG0 rows [0..63], WG1 rows [64..127]).  Each WG chains
  // WGMMAS_PER_STEP m64n8k32 WGMMAs per 128-K substep.
  static constexpr std::uint32_t W_UP_TILE_WGMMA = 128;
  static constexpr std::uint32_t K_TILE_WGMMA = 32;   // K per fp8 WGMMA
  static constexpr std::uint32_t K_STEP_WGMMA = 128;  // SWZ128 atom K-width
  static constexpr std::uint32_t WGMMAS_PER_STEP = K_STEP_WGMMA / K_TILE_WGMMA;  // 4

  static_assert(K_STEP_WGMMA % K_TILE_WGMMA == 0,
                "K_STEP_WGMMA must be a multiple of K_TILE_WGMMA");
  static_assert(!use_wgmma<Dims>::value || Dims::HIDDEN_STATES % K_STEP_WGMMA == 0,
                "HIDDEN_STATES must be a multiple of 128 for the WGMMA path "
                "(one K-step consumes K=128)");
  static_assert(!use_wgmma<Dims>::value || (2 * Dims::N) % W_UP_TILE_WGMMA == 0,
                "2*N must be a multiple of 128 for the WGMMA path "
                "(one block produces 128 output rows per K-step)");

  // ── Down-proj outer K-step (K_STEP_DOWN tunable) ──────────────────────
  static constexpr std::uint32_t K_STEP_DOWN = down_k_step<Dims>::value;
  static constexpr std::uint32_t K_SUBSTEPS_DOWN = K_STEP_DOWN / K_STEP_WGMMA;
  static_assert(K_STEP_DOWN >= K_STEP_WGMMA && K_STEP_DOWN % K_STEP_WGMMA == 0,
                "K_STEP_DOWN must be a positive multiple of K_STEP_WGMMA "
                "(=128, the SWZ128 atom K-width).");
  static_assert(!use_wgmma<Dims>::value || Dims::N % K_STEP_DOWN == 0,
                "Dims::N must be a multiple of K_STEP_DOWN for the WGMMA "
                "down-projection.");

  // ── Down-proj TMA pipeline depth (DOWN_PIPE_DEPTH tunable) ────────────
  // 2 = classic double buffer (default); 4 = rolling ring with 2 K-steps
  // of lookahead (arm slot (s+2)&3 at iter s).  See the down_pipe_depth
  // detector for the opt-in mechanics.
  static constexpr std::uint32_t DOWN_PIPE_DEPTH = down_pipe_depth<Dims>::value;
  static_assert(DOWN_PIPE_DEPTH == 2u || DOWN_PIPE_DEPTH == 4u,
                "DOWN_PIPE_DEPTH must be 2 (double buffer) or 4 (rolling "
                "ring); other depths have no launcher implementation.");
  static_assert(DOWN_PIPE_DEPTH == 2u || (Dims::N / K_STEP_DOWN) % DOWN_PIPE_DEPTH == 0u,
                "DOWN_PIPE_DEPTH == 4 requires K_TILES_DOWN (= N / "
                "K_STEP_DOWN) to be a multiple of 4 so the cross-expert "
                "stitch at iters K_TILES-2 / K_TILES-1 lands the next "
                "expert's tiles 0,1 on ring slots 0,1.");

  // ── Up-proj outer K-step (K_STEP_UP tunable) ──────────────────────────
  static constexpr std::uint32_t K_STEP_UP = up_k_step<Dims>::value;
  static constexpr std::uint32_t K_SUBSTEPS_UP = K_STEP_UP / K_STEP_WGMMA;
  static_assert(K_STEP_UP >= K_STEP_WGMMA && K_STEP_UP % K_STEP_WGMMA == 0,
                "K_STEP_UP must be a positive multiple of K_STEP_WGMMA "
                "(=128, the SWZ128 atom K-width).");
  static_assert(!use_wgmma<Dims>::value || Dims::HIDDEN_STATES % K_STEP_UP == 0,
                "Dims::HIDDEN_STATES must be a multiple of K_STEP_UP for "
                "the WGMMA up-projection.");
  static constexpr std::uint32_t K_TILES_UP = Dims::HIDDEN_STATES / K_STEP_UP;

  // ── Up-proj weight-TMA lookahead (UP_W_SLOTS tunable) ─────────────────
  // The launcher arms slot (s + UP_ARM_DISTANCE) % UP_W_SLOTS for logical
  // outer-K iter s + A; the consumer waits slot s % UP_W_SLOTS.  See the
  // `up_w_slots` detector for the arm-distance derivation.
  static constexpr std::uint32_t UP_W_SLOTS = up_w_slots<Dims>::value;
  static constexpr std::uint32_t UP_ARM_DISTANCE = (UP_W_SLOTS > 2u) ? (UP_W_SLOTS - 2u) : 1u;
  static_assert(!use_wgmma<Dims>::value || UP_W_SLOTS >= 2u,
                "UP_W_SLOTS must be >= 2 (the up-proj weight pipeline needs "
                "at least a ping-pong double buffer).");
  static_assert(!use_wgmma<Dims>::value || (UP_W_SLOTS & (UP_W_SLOTS - 1u)) == 0u,
                "UP_W_SLOTS must be a power of two so the slot index can use "
                "a cheap `s & (UP_W_SLOTS-1)` mask.");
  static_assert(!use_wgmma<Dims>::value || K_TILES_UP % UP_W_SLOTS == 0,
                "K_TILES_UP must be a multiple of UP_W_SLOTS so the "
                "cross-expert weight-TMA stitch lands the next expert's "
                "iters [0, UP_ARM_DISTANCE) on slots [0, UP_ARM_DISTANCE), "
                "keeping per-slot arm/wait parity balanced across experts.");
  static_assert(!use_wgmma<Dims>::value || K_TILES_UP >= UP_W_SLOTS,
                "K_TILES_UP must be >= UP_W_SLOTS so the pre-loop can prime "
                "UP_ARM_DISTANCE slots without wrapping past the K-loop.");

  // ── Up-proj col-halves (M-axis 128-row atoms per block) ───────────────
  // 1 = interleaved single-TMA layout; 2 = raw two-TMA layout (also the
  // decoupled up/down-grid shapes).  See the `up_col_halves` detector.
  static constexpr std::uint32_t UP_COL_HALVES = up_col_halves<Dims>::value;
  static_assert(!use_wgmma<Dims>::value || UP_COL_HALVES <= 2u,
                "UP_COL_HALVES > 2 not supported by the up-proj M-axis "
                "WGMMA loop (post_silu_scratch is sized for 2 atoms).");

  // Activation-quantization block size along N: one up-block owns
  // UP_COL_HALVES * 64 gate features (each atom packs 64 gate + 64 up rows).
  static constexpr std::uint32_t DOWN_ACT_BLOCK_SIZE = UP_COL_HALVES * 64u;
  static constexpr std::uint32_t DOWN_ACT_HALVES = Dims::N / DOWN_ACT_BLOCK_SIZE;

  // Effective M (row-tile) size of one block's up-proj work; drives
  // UP_GRID = 2*N / W_UP_TILE_EFFECTIVE.
  static constexpr std::uint32_t W_UP_TILE_EFFECTIVE =
      use_wgmma<Dims>::value ? (UP_COL_HALVES * W_UP_TILE_WGMMA) : W_UP_TILE;

  // ── Down-projection grid layout ───────────────────────────────────────
  // DOWN_GRID blocks cover one expert's HIDDEN_STATES output cols;
  // DOWN_GROUPS expert groups run in parallel, each atomicAdding into the
  // single-buffer `spec->down_partial_out`.
  static constexpr std::uint32_t DOWN_COL_TILE = down_col_tile<Dims>::value;
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

  // MoEGemmSpec recomputes these locally (it is defined earlier in this
  // file); a divergence would silently corrupt Phase 5.
  static_assert(MoEGemmSpec<Dims>::DOWN_COL_TILE == DOWN_COL_TILE,
                "MoEGemmSpec::DOWN_COL_TILE must match "
                "MoECoreDims::DOWN_COL_TILE.");
  static_assert(MoEGemmSpec<Dims>::DOWN_GROUPS == DOWN_GROUPS,
                "MoEGemmSpec::DOWN_GROUPS must match "
                "MoECoreDims::DOWN_GROUPS.");

  // Per-block token-tile width consumed by the FP8 activation staging
  // and the down-proj WGMMA B operand.  8 for BS<=8 (one 8-token SWZ128
  // atom).
  static constexpr std::uint32_t T_TILE = 8u;
};

/**
 * @brief Dynamic shared-memory layout.
 *
 * The dominant space is a union whose members have strictly disjoint
 * lifetimes (separated by the Phase-2 trailing __syncthreads() and the
 * Phase 3 → 4 barrier).  See docs/design_docs/monomoe_kernel.md "Shared memory".
 */
template <typename Dims>
struct MoE_SHM {
  using CoreDims = MoECoreDims<Dims>;
  // SHM layout for the BS8 TMA+WGMMA path (the only BS8 variant).
  struct TinyDataWGMMA_TMA {
    // 128-K SWZ128 atoms per token along the full K axis.
    static constexpr uint32_t K_BLOCKS_TOTAL = Dims::HIDDEN_STATES / CoreDims::K_STEP_WGMMA;

    // Live extents for the BS8 WGMMA+TMA tag (the only variant).  BS8 SASS
    // bit-identity is preserved (these evaluate identically to the former
    // `(Dims::BS <= 16) ? live : 1` since BS = 8).
    static constexpr uint32_t BF16_IN_FULL_K_BLOCKS = K_BLOCKS_TOTAL;
    static constexpr uint32_t BF16_IN_FULL_BS = Dims::BS;
    static constexpr uint32_t BF16_IN_FULL_K = CoreDims::K_STEP_WGMMA;
    static constexpr uint32_t FP8_ACT_FULL_K_BLOCKS = K_BLOCKS_TOTAL;

    static constexpr uint32_t FP8_ACT_K_CHUNK = 16;
    // 16-K fp8 chunks per 128-K SWZ128 atom.
    static constexpr uint32_t FP8_ACT_NUM_CHUNKS = CoreDims::K_STEP_WGMMA / FP8_ACT_K_CHUNK;  // 8

    static constexpr uint32_t DOWN_ACT_K_SUBSTEPS = CoreDims::K_SUBSTEPS_DOWN;

    // Down-proj activation ring (DOWN_PIPE_DEPTH slots; 2 = the classic
    // double buffer): one outer K-step's worth of fp8 activations =
    // K_SUBSTEPS_DOWN SWZ128 atoms (8 tok × 128 K-bytes each) per slot.
    // 1024-B alignment required by SWIZZLE_128B (the XOR pattern is only
    // consistent within 1024-B-aligned regions).
    alignas(1024) AQ_element a_down_wgmma[CoreDims::DOWN_PIPE_DEPTH][DOWN_ACT_K_SUBSTEPS]
                                         [CoreDims::T_TILE][FP8_ACT_NUM_CHUNKS][FP8_ACT_K_CHUNK];

    // Single-buffer FP8 activations covering the full K range, produced
    // once per launch by Phase-2 routing_phase_quantize and read directly
    // by the Phase-3 K-loop (no slot alternation).
    //
    // The token dim is padded 8 → 9 rows per 16-B chunk: the natural
    // 128-B chunk stride put every same-`t%4` lane on the same bank
    // (8-way STS conflict); 9 rows make the stride 144 B = 4 banks mod
    // 32, conflict-free.  The 9th row is never written; the up-proj
    // WGMMA B descriptor steps over it with LBO = 144.
    static constexpr uint32_t FP8_ACT_T_TILE_PADDED = CoreDims::T_TILE + 1u;
    alignas(1024) AQ_element fp8_act_full[FP8_ACT_FULL_K_BLOCKS][FP8_ACT_NUM_CHUNKS]
                                         [FP8_ACT_T_TILE_PADDED][FP8_ACT_K_CHUNK];

    static constexpr uint32_t W_WGMMA_M = 128;
    static constexpr uint32_t W_DOWN_WGMMA_M = CoreDims::DOWN_COL_TILE;
    static constexpr uint32_t W_WGMMA_K = CoreDims::K_STEP_WGMMA;
    // The up weight tile stacks the K substeps AND the UP_COL_HALVES
    // M-atoms along the M axis (substep kk, half h → rows
    // [(kk*UCH + h)*128, +128)), so each atom remains a self-contained
    // 1024-B-aligned 128×128 SWZ128 region.
    static constexpr uint32_t W_WGMMA_M_TOTAL =
        W_WGMMA_M * CoreDims::K_SUBSTEPS_UP * CoreDims::UP_COL_HALVES;
    // The down weight tile stacks its K substeps along M the same way.
    static constexpr uint32_t W_DOWN_WGMMA_K = CoreDims::K_STEP_DOWN;
    static constexpr uint32_t W_DOWN_WGMMA_M_TOTAL = W_DOWN_WGMMA_M * CoreDims::K_SUBSTEPS_DOWN;
    static constexpr uint32_t UP_W_SLOTS = CoreDims::UP_W_SLOTS;

    // The three views below alias: bf16_in_full is dead by the Phase-2
    // trailing sync (before any weight TMA fires), and w_wgmma's last
    // consumer wait precedes the Phase-4 reuse as w_down_wgmma (the
    // site-#2 barrier serializes the transition).
    union {
      // Routing-window BF16 input tile (Phase 1/2).  Tile-major
      // [K_BLOCKS_TOTAL][BS][K_STEP_WGMMA] — NOT [BS][HIDDEN] — because
      // each activation TMA writes a compact 8×128 box whose row stride
      // is the box's own inner dim (256 B), not the destination's
      // logical row stride; each K-substep therefore needs its own
      // self-contained 2 KB slab.
      alignas(1024) A_element bf16_in_full[BF16_IN_FULL_K_BLOCKS][BF16_IN_FULL_BS][BF16_IN_FULL_K];
      // Up-proj weight slots (Phase 3), UP_W_SLOTS deep.
      alignas(1024) W_element w_wgmma[UP_W_SLOTS][W_WGMMA_M_TOTAL][W_WGMMA_K];
      // Down-proj weight ring (Phase 4), DOWN_PIPE_DEPTH slots (2 = the
      // classic double buffer).  Slot size scales with K_STEP_DOWN, so a
      // 4-deep ring at KDN=128 occupies the same bytes as the 2-deep at
      // KDN=256.
      alignas(1024) W_element
          w_down_wgmma[CoreDims::DOWN_PIPE_DEPTH][W_DOWN_WGMMA_M_TOTAL][CoreDims::K_STEP_WGMMA];
    };

    // Per-expert down-proj activation scales, loaded once per expert
    // (not per K-step).  [block][tok] layout: the WGMMA scale-apply
    // broadcasts each (block, tok) pair across the 8 four-lane groups,
    // so the transposed layout puts every read on a distinct bank.
    static constexpr uint32_t DOWN_ACT_HALVES_PER_EXPERT = CoreDims::DOWN_ACT_HALVES;
    S_element a_down_scale[DOWN_ACT_HALVES_PER_EXPERT][CoreDims::T_TILE];

    // Down-proj weight scales: [M-atom half][WG0/WG1][col-block].
    static constexpr uint32_t W_DOWN_SCALE_COLS = shm_down_scale_cols<Dims>::value;
    static constexpr uint32_t W_DOWN_SCALE_HALVES = CoreDims::DOWN_COL_TILE / 128u;
    S_element w_down_scale[W_DOWN_SCALE_HALVES][2][W_DOWN_SCALE_COLS];

    // Up-proj weight scales, ping-pong: expert e consumes slot
    // cur_scale_slot while e+1 prefetches into the other slot.
    static constexpr uint32_t UP_SCALE_TILE_SIZE = 2 * shm_up_scale_cols<Dims>::value;
    S_element up_scale[2][UP_SCALE_TILE_SIZE];

    // down_out is intentionally unpadded: its bank conflicts (a 4-way STS on
    // the epilogue writeback, a 16-way LDS on the deferred accumulate) sit on
    // prefetch/epilogue slots hidden behind WGMMA and TMA waits, so padding
    // the stride gave no measured speedup (and +1 regressed the STS.64 pair
    // merge).
    static constexpr uint32_t DOWN_OUT_ROWS = CoreDims::DOWN_COL_TILE;
    union {
      // Down-proj per-expert output scratch (Phase-4 epilogue →
      // deferred accumulate).
      T_element down_out[DOWN_OUT_ROWS][CoreDims::T_TILE];
      // Up-proj post-SiLU fp32 scratch: calc warps write the per-lane
      // silu(gate)*up*rw combine at the K-loop tail of expert e; PF
      // warps drain it during expert e+1's K-loop (deferred epilogue).
      // Atom h uses rows [h*128, +128); within an atom only rows
      // [0..31] (WG0) and [64..95] (WG1) carry values.  The +1 column
      // pad makes the [col][tok] read pattern bank-conflict-free
      // (row stride 9 floats, gcd(9, 32) = 1).
      //
      // Aliasing down_out is safe: down_out belongs to Phase 4, which
      // is barrier-separated from every post_silu_scratch access.
      T_element post_silu_scratch[CoreDims::UP_COL_HALVES * 128][CoreDims::T_TILE + 1];
    } partial_result;

    static constexpr uint32_t OUT_ACCUM_ROW_PAD = 1;
    T_element out_accum[Dims::BS][CoreDims::DOWN_COL_TILE + OUT_ACCUM_ROW_PAD];

    // ── mbarriers (16-B alignment required by SM90 mbarrier PTX) ─────
    // bar_w: up-proj weight pipeline, one per lookahead slot; the
    // down-proj reuses slots 0..DOWN_PIPE_DEPTH-1 as its ring (sized
    // for whichever phase needs more).  bar_a: down-proj activation
    // ring, armed/consumed exclusively by Phase 4 (re-initialized
    // there).  bar_rwin: Phase-1 routing-window load (arrival_count =
    // 1), waited on by warps 1..11 at the start of Phase 2.
    static constexpr uint32_t BAR_W_COUNT =
        (UP_W_SLOTS > CoreDims::DOWN_PIPE_DEPTH) ? UP_W_SLOTS : CoreDims::DOWN_PIPE_DEPTH;
    alignas(16) uint64_t bar_w[BAR_W_COUNT];
    alignas(16) uint64_t bar_a[CoreDims::DOWN_PIPE_DEPTH];
    alignas(16) uint64_t bar_rwin;

    // ── Phase 3 → Phase 4 (expert, token) reorganization tables ──────
    //
    //   expert_slot_start[id]   = first row in spec->temp_fp8 reserved
    //                             for expert id (expert-sorted layout).
    //   expert_routed_count[id] = routed (tok, k) pairs selecting id.
    //   sorted_slot[pair]       = destination temp_fp8 row for the
    //                             up-proj writeback,
    //                             pair = tok * top_k + k.
    //
    // alignas(16) on expert_slot_start is REQUIRED, not cosmetic:
    // routing Phase B writes it with packed 16-B vector stores whose
    // per-lane address is only 16-B aligned if the base is.
    static constexpr uint32_t MAX_TOPK = 8;
    static constexpr uint32_t MAX_PAIRS = Dims::BS * MAX_TOPK;
    alignas(16) uint16_t expert_slot_start[Dims::NUM_EXPERTS];
    uint8_t expert_routed_count[Dims::NUM_EXPERTS];
    uint8_t sorted_slot[MAX_PAIRS];
    // Per-(expert, token) intra-expert rank, recorded once in routing
    // Phase C and read-only afterwards (so any expert's row may be read
    // at any point of Phase 4).  0xFF = token does not route to that
    // expert.  alignas(16) enables the vectorized uint4 0xFF seed.
    alignas(16) uint8_t down_rank[Dims::NUM_EXPERTS][Dims::BS];
    // Per-expert routing cache for the up-proj epilogue: for each token,
    // the top-k index k with topk_ids_flat[tok*MAX_TOPK + k] == id (0xFF
    // if none) and the matching routing weight (0.0f if none — the rw
    // value doubles as the routed predicate).  Populated at each
    // expert's K-loop top; published by the same sync as up_scale.
    static constexpr uint32_t UP_RANK_FOR_TOK_LEN = (use_pair_layout<Dims>::value ? Dims::BS : 0u);
    uint8_t up_rank_for_tok[UP_RANK_FOR_TOK_LEN];
    static constexpr uint32_t UP_RW_FOR_TOK_LEN = UP_RANK_FOR_TOK_LEN;
    S_element up_rw_for_tok[UP_RW_FOR_TOK_LEN];
    // Snapshot of the PREVIOUS expert's ranks, taken at the calc-warp
    // epilogue of expert e (before e+1's K-loop top overwrites
    // up_rank_for_tok).  The deferred PF writeback of expert e reads
    // this during e+1's K-loop.
    static constexpr uint32_t UP_RANK_FOR_TOK_PREV_LEN =
        (use_pair_layout<Dims>::value ? Dims::BS : 0u);
    uint8_t up_rank_for_tok_prev[UP_RANK_FOR_TOK_PREV_LEN];
  } tiny_wgmma_tma;

  // The bf16 input view must alias the weight views exactly — a layout
  // change that moves bf16_in_full out of the anonymous union would
  // silently corrupt SHM at runtime; catch it at compile time.
  static_assert(offsetof(TinyDataWGMMA_TMA, bf16_in_full) == offsetof(TinyDataWGMMA_TMA, w_wgmma),
                "bf16_in_full must alias w_wgmma exactly.");
  static_assert(offsetof(TinyDataWGMMA_TMA, bf16_in_full) ==
                    offsetof(TinyDataWGMMA_TMA, w_down_wgmma),
                "bf16_in_full must alias w_down_wgmma exactly.");

  static_assert(Dims::NUM_EXPERTS <= 65535,
                "Number of experts too high, cannot store as uint16 anymore.");

  // ── Common fields ────────────────────────────────────────────────────────

  // Per-token per-128-K-block activation quantization scales for the
  // up-projection: act_scale[blk][tok] = max(|x_tok[blk*128 .. +128)|)/448.
  // [blk][tok] (not [tok][blk]) so the up-proj WGMMA scale-apply broadcast
  // reads land on distinct banks.
  static constexpr uint32_t ACT_BLOCK_SIZE = 128;
  static constexpr uint32_t ACT_SCALE_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK_SIZE - 1) / ACT_BLOCK_SIZE;
  S_element act_scale[ACT_SCALE_BLOCKS][Dims::BS];

  // Launch parity for the sentinel Phase-3→4 handoff (selects which
  // temp_act_scale buffer this launch publishes/polls).  Written by
  // thread 0 in the kernel prologue from spec->launch_flip[blockIdx.x];
  // published to all warps by the prologue __syncthreads.
  uint32_t scale_parity;

  // Unique experts active in this batch, ascending id; filled by
  // prepare_moe_topk.
  ExpertRef experts[Dims::NUM_EXPERTS];
  std::uint32_t expert_count;

  // Flat routing results: [tok * MAX_TOPK + k] = expert id / routing weight
  // of the token's k-th selection.  Written by topK.  The uint64
  // alignment lets consumers vector-load one token's 8 ids.
  static constexpr uint32_t MAX_TOPK = 8;
  alignas(uint64_t) uint16_t topk_ids_flat[(Dims::BS < 8 ? 8 : Dims::BS) * MAX_TOPK];
  S_element topk_weights_flat[(Dims::BS < 8 ? 8 : Dims::BS) * MAX_TOPK];
};

/**
 * @brief Dynamic shared memory required by moe_kernel_topk<Dims>.
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_shmem_size() {
  static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS,
                "Dimension larger than the maximum supported dimension.");
  // H100/H200 per-block opt-in budget is 228 KB; target 224 KB to leave
  // margin for driver overhead.
  static_assert(sizeof(MoE_SHM<Dims>) <= 224 * 1024,
                "MoE_SHM layout exceeds the 224 KB per-block SHM budget.");
  return sizeof(MoE_SHM<Dims>);
}

constexpr size_t get_moe_max_shmem_size() { return sizeof(MoE_SHM<Dims_Max>); }

constexpr size_t get_moe_max_scratchpad_size() { return sizeof(MoEGemmSpec<Dims_Max>); }

// ── Warp identity helpers ───────────────────────────────────────────────────

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
 * @brief Synchronizes the first 256 threads (the calc warps) of the block.
 *
 * Collective: must be called by all of the first 256 threads.
 */
template <typename Dims>
__device__ __forceinline__ void sync_calc_threads() {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP == 256,
                "Adapt the thread number in sync_calc_threads");
  __asm volatile("bar.sync  15, 256;\n");
}

/**
 * @brief Warp-wide max reduction; the result is returned on all lanes.
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
