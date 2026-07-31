
#pragma once
#ifndef MOE_SCALE_INPUTS_CU
#define MOE_SCALE_INPUTS_CU

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "moe_grid_barrier.h"
#include "moe_interface.h"
#include "moe_internal.h"
#include "ptx_utils.h"

namespace monomoe {

/**
 * @brief Sets NaNs (positive or negative, any payload) to 0.0 (bit-pattern all
 * 0).
 *
 * Values other than NaN remain unchanged.
 */
__device__ static __forceinline__ __nv_bfloat162 mask_NaNs_to_zero(__nv_bfloat162 xs) {
  return type_pun<__nv_bfloat162>(type_pun<uint32_t>(xs) & __heq2_mask(xs, xs));
}

// Internal linkage
namespace {

/**
 * @brief Struct for keeping a chunk of 8 bfloat16 in registers.
 */
struct BF16x8 {
  float4 raw;  // Storage for 8 BFloat16 values. Never accessed as fp32.

  /**
   * @brief Loads a BF16x8 from memory.
   *
   * The address @p a must be BF16x8 aligned.
   */
  __device__ static BF16x8 load(const A_element* a) {
    assert(reinterpret_cast<uintptr_t>(a) % 16 == 0);
    BF16x8 val{*reinterpret_cast<const float4*>(a)};
    return val;
  }

  /**
   * @brief Stores a BF16x8 to memory.
   *
   * The address @p a must be BF16x8 aligned.
   */
  __device__ void store_to(A_element* a) {
    assert(reinterpret_cast<uintptr_t>(a) % 16 == 0);
    *reinterpret_cast<float4*>(a) = raw;
  }

  /**
   * @brief Returns bfloat16 0 and 1 as pair.
   */
  __device__ __nv_bfloat162 first_pair() const { return type_pun<__nv_bfloat162>(raw.x); }
  /**
   * @brief Returns bfloat16 2 and 3 as pair.
   */
  __device__ __nv_bfloat162 second_pair() const { return type_pun<__nv_bfloat162>(raw.y); }
  /**
   * @brief Returns bfloat16 5 and 5 as pair.
   */
  __device__ __nv_bfloat162 third_pair() const { return type_pun<__nv_bfloat162>(raw.z); }
  /**
   * @brief Returns bfloat16 6 and 7 as pair.
   */
  __device__ __nv_bfloat162 fourth_pair() const { return type_pun<__nv_bfloat162>(raw.w); }

  /**
   * @brief Converts 8 Bfloat16 values to FP8 E4M3 in accordance with vLLM's MoE
   * activation quantization.
   *
   * Scales, clamps (incl. NaN replacement), rounds and converts each BFloat16
   * to FP8 E4M3. Scaling is done with float accuracy. Clamping and saturation
   * are implemented via __NVSATFINITE semantics of the FP8 conversion.
   *
   * @param scale Scaling factor to use.
   * @returns Eight FP8 E4M3 packed into a uint64_t
   */
  __device__ uint64_t to_fp8x8(float scale) const {
    // We do not need to actually clamp. Clamping is handled implicitly by the
    // satfinite semantics of the float->e4m3 conversion. We only need to
    // swallow NaNs. Here, we set them to 0.
    __nv_bfloat162 bf0 = mask_NaNs_to_zero(first_pair());
    __nv_bfloat162 bf1 = mask_NaNs_to_zero(second_pair());
    __nv_bfloat162 bf2 = mask_NaNs_to_zero(third_pair());
    __nv_bfloat162 bf3 = mask_NaNs_to_zero(fourth_pair());

    float2 f0 = __bfloat1622float2(bf0);
    float2 f1 = __bfloat1622float2(bf1);
    float2 f2 = __bfloat1622float2(bf2);
    float2 f3 = __bfloat1622float2(bf3);

    __nv_fp8x4_e4m3 converted0{float4{f0.x * scale, f0.y * scale, f1.x * scale, f1.y * scale}};
    __nv_fp8x4_e4m3 converted1{float4{f2.x * scale, f2.y * scale, f3.x * scale, f3.y * scale}};

    return type_pun<uint32_t>(converted0) | ((uint64_t)type_pun<uint32_t>(converted1) << 32);
  }
};

}  // namespace

/**
 * @brief Per-K-tile bf16 → fp8 quantization (single-row BF16 view).
 *
 * Called by exactly one warp per `(token, k_block)` pair.  Each call
 * converts 128 bf16 K-values for ONE token into 128 fp8 K-values with
 * a per-token-per-128-K-block scale.
 *
 * Input layout:
 *   bf16_row[0..127]      — contiguous bf16 activations for the
 *                            token slot `tok`, for the current 128-K
 *                            substep.
 *
 * Output layout (canonical WGMMA K-major):
 *   fp8_act[kc][tok][ki]  — where kc = 0..7, tok = 0..7, ki = 0..15,
 *                            and kc*16 + ki = global K within the tile.
 *                            The helper writes ONLY the slice
 *                            [kc][tok][ki] for the caller-supplied
 *                            `tok`; other tokens at the same k_block
 *                            are written by other warps.
 *
 * Scale output:
 *   act_scale_for_step    — one fp32 scale value for this (tok, k_block)
 *                            pair.  Caller supplies a pointer to the
 *                            `&shmem->act_scale[k_block][tok]` slot.
 *
 * Thread distribution: 32 threads per warp, each owning 4 K-values
 * (4 × 32 = 128 = one full 128-K substep).  Warp-reduce finds the
 * block max.  If `tok >= batch_size`, the warp zero-fills its fp8 slot
 * and writes `act_scale = 1.0f` (neutral — the scale-apply will
 * multiply by zero via the `as_0X = (tok < batch_size) ? ... : 0.f`
 * guard in the up-proj kernel, so this scale value is never consumed).
 *
 * The canonical fp8 output layout means thread t writes its 4 quantized
 * values at
 *   fp8_act[(col+i)/16][tok][(col+i)%16]
 * where col = t * 4 and i in 0..3.  With 4 consecutive K-values per
 * thread and a 16-wide chunk, all 4 values fall in the same kc.
 *
 * Refactor note:
 * The BF16 view was narrowed from a `[T_TILE=8][128]` 2 KB atom to a
 * `[128]` single-row view.  This unblocks calling the helper from the
 * routing-phase fusion path, where the BF16 source is the full
 * tile-major `bf16_in_full[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` SHM
 * buffer populated by `moe_load_full_bf16_input` — the caller passes
 * `bf16_in_full[k_block][token]` directly as a natural `[128]` row
 * (no row-stride reinterpretation needed).  All FP8 writes and the
 * scale-output convention are unchanged — the refactored helper
 * still accepts `tok` and writes the same `fp8_act[kc][tok][ki]`
 * bytes as before.  Existing callers that previously passed
 * `slot[kk]` (a `[8][128]` tile) now pass `slot[kk][tok]` (a `[128]`
 * row).
 *
 * @tparam Dims              MoE dims.
 * @tparam BF16InCols        Must be K_STEP_WGMMA (128).
 * @tparam Fp8NumChunks      Must be K_STEP_WGMMA / 16 (8).
 * @tparam Fp8Tok            Must be T_TILE (8).
 * @tparam Fp8KInner         Must be 16.
 * @param  bf16_row          SHM-resident bf16 row for token `tok`,
 *                            for the current 128-K substep.
 * @param  fp8_act           SHM output fp8 tile
 *                            [Fp8NumChunks][T_TILE][Fp8KInner].
 * @param  tok               Token slot this call is writing.
 * @param  batch_size        Number of real tokens (remaining zero-filled).
 * @param  act_scale_for_step fp32 destination for this (tok, k_block)
 *                            scale.
 */
template <typename Dims, std::size_t BF16InCols, std::size_t Fp8NumChunks, std::size_t Fp8Tok,
          std::size_t Fp8KInner>
__device__ __forceinline__ void moe_streaming_quantize_k128(
    const A_element (&bf16_row)[BF16InCols], AQ_element (&fp8_act)[Fp8NumChunks][Fp8Tok][Fp8KInner],
    std::uint32_t tok, std::uint32_t batch_size, float* __restrict__ act_scale_for_step) {
  static_assert(Dims::BS <= 8, "Streaming quantize is for BS<=8");
  static_assert(BF16InCols == 128, "bf16_row must have 128 K cols");
  static_assert(Fp8NumChunks == 8, "fp8_act must have 8 K-chunks of 16");
  static_assert(Fp8Tok == 8 || Fp8Tok == 9,
                "fp8_act must have 8 (legacy) or 9 (kc-padded for "
                "bank-conflict avoidance) token rows; only the first "
                "8 are written, the 9th — when present — is unused "
                "padding that breaks the 128-byte kc stride");
  static_assert(Fp8KInner == 16, "fp8_act inner dim must be 16");

  const std::uint32_t thread = get_thread<Dims>();  // 0..31
  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  if (tok >= batch_size) {
    // Zero-fill the unused token slot so WGMMA's B operand sees no
    // stray fp8 NaN bit patterns.  32 threads × 4 bytes = 128 B =
    // one full token row across all 8 K-chunks.
    const uint32_t col = thread * 4;  // 0, 4, 8, ..., 124
    const uint32_t kc = col / 16;
    const uint32_t ki = col % 16;
    fp8_act[kc][tok][ki + 0] = (AQ_element)0;
    fp8_act[kc][tok][ki + 1] = (AQ_element)0;
    fp8_act[kc][tok][ki + 2] = (AQ_element)0;
    fp8_act[kc][tok][ki + 3] = (AQ_element)0;
    if (thread == 0) *act_scale_for_step = 1.0f;
    return;
  }

  // Real token path: load 4 bf16, warp-reduce max, quantize.
  const uint32_t col = thread * 4;
  __nv_bfloat162 bf_01 = *reinterpret_cast<const __nv_bfloat162*>(&bf16_row[col + 0]);
  __nv_bfloat162 bf_23 = *reinterpret_cast<const __nv_bfloat162*>(&bf16_row[col + 2]);
  bf_01 = mask_NaNs_to_zero(bf_01);
  bf_23 = mask_NaNs_to_zero(bf_23);
  float2 f01 = __bfloat1622float2(bf_01);
  float2 f23 = __bfloat1622float2(bf_23);
  float r0 = f01.x, r1 = f01.y, r2 = f23.x, r3 = f23.y;

  float local_max = fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3)));
  float blk_max = warp_reduce_max_float(local_max);
  if (blk_max < __FLT_MIN__) blk_max = 1.f;

  const float blk_act_scale = blk_max * FP8_MAX_INV;
  const float blk_inv_scale = FP8_MAX / blk_max;

  AQ_element q0 = (AQ_element)(r0 * blk_inv_scale);
  AQ_element q1 = (AQ_element)(r1 * blk_inv_scale);
  AQ_element q2 = (AQ_element)(r2 * blk_inv_scale);
  AQ_element q3 = (AQ_element)(r3 * blk_inv_scale);

  const uint32_t kc = col / 16;
  const uint32_t ki = col % 16;
  fp8_act[kc][tok][ki + 0] = q0;
  fp8_act[kc][tok][ki + 1] = q1;
  fp8_act[kc][tok][ki + 2] = q2;
  fp8_act[kc][tok][ki + 3] = q3;

  if (thread == 0) *act_scale_for_step = blk_act_scale;
}

/**
 * @brief Phase-2 routing-window BF16 → FP8 quantization (BS8 TMA+WGMMA only).
 *
 * Called from `moe_kernel_topk_BS8`'s Phase-2 dispatch on warps 1..11
 * (warp 0 runs `prepare_moe_topk_BS8` instead).  Distributes the
 * `BS * K_BLOCKS_TOTAL = 128` (token, k_block) quantization tasks
 * across the 11 participating warps using a stride-11 partition over
 * the linear pair index `i = token * K_BLOCKS_TOTAL + k_block`.  Each
 * warp-owned pair invokes `moe_streaming_quantize_k128<Dims>` exactly
 * once with a single-row BF16 view onto the prefetched
 * `bf16_in_full[k_block][token]`.
 *
 * The BF16 source uses the tile-major
 * `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` SHM layout populated by the
 * Phase-1 routing-window TMA (`moe_load_full_bf16_input`).  Each
 * K-substep slot is a self-contained 2 KB region whose bytes match
 * the activation TMA's compact `boxDim = (128, 8)` write layout —
 * see the doc comment on `TinyDataWGMMA_TMA::bf16_in_full` in
 * `moe_internal.h` for the rationale.
 *
 * Caller contract (NOT enforced inside the helper):
 *   1. The caller MUST gate warp 0 OUT of this call.  Warp 0 runs
 *      `prepare_moe_topk_BS8` concurrently and is also gated out of
 *      the `bar_rwin` wait described below.
 *   2. The caller MUST wait on the routing-window mbarrier
 *      (`tiny_wgmma_tma.bar_rwin`) on every warp 1..11 thread before
 *      invoking this helper.  Reading `bf16_in_full` before the wait
 *      succeeds is a data race against the in-flight Phase-1 TMA load.
 *   3. The caller MUST emit a block-wide `__syncthreads()` AFTER this
 *      helper returns to publish the FP8 atom + scale writes to all
 *      warps before the Phase-3 up-projection K-loop reads them.
 *   4. Ragged batches (`token >= batch_size`) are handled inside this
 *      helper via the existing zero-fill + scale = 1.0f path of
 *      `moe_streaming_quantize_k128`.
 *
 * Scope: this helper is only valid when
 * `Dims::BS <= 8` AND the call site is on the BS8 TMA+WGMMA path.
 * The static_assert below makes miss-instantiation a build error.
 *
 * Work distribution: warp `w ∈ [1, 12)` owns pair indices
 *   { i ∈ [0, BS * K_BLOCKS_TOTAL) : (i % 11) == (w - 1) }.
 * With `BS = 8`, `K_BLOCKS_TOTAL = 16`, `PAIRS_TOTAL = 128`, and 11
 * warps, every pair is owned by exactly one warp.  Warp 0 owns no
 * pair.  Imbalance is at most one extra pair per warp (~9%), which
 * is acceptable given the alternative is wasting warps.
 *
 * @tparam Dims          MoE dims.
 * @tparam KBlocks       Outer extent of `bf16_in_full`
 *                       (= K_BLOCKS_TOTAL).
 * @tparam Bs            Middle extent of `bf16_in_full` (= Dims::BS).
 * @tparam KStep         Inner extent of `bf16_in_full` (= K_STEP_WGMMA
 *                       = 128).
 * @tparam Fp8KBlocks    Outer extent of `fp8_act_full` (= K_BLOCKS_TOTAL).
 * @tparam Fp8NumChunks  Must be 8 (= K_STEP_WGMMA / FP8_ACT_K_CHUNK).
 * @tparam Fp8Tok        Must be 8 (= T_TILE).
 * @tparam Fp8KInner     Must be 16 (= FP8_ACT_K_CHUNK).
 * @tparam ScaleBs       Outer extent of `act_scale` (= Dims::BS).
 * @tparam ScaleKBlocks  Inner extent of `act_scale` (= ACT_SCALE_BLOCKS
 *                       = HIDDEN_STATES / 128).
 *
 * @param bf16_in_full   SHM-resident BF16 input tile populated by the
 *                       Phase-1 routing-window TMA load, shaped
 *                       `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]`.
 * @param fp8_act_full   SHM-resident FP8 output, single-buffer indexed
 *                       by `k_block`.
 * @param act_scale      Per-token per-128-K-block FP8 scales.  Lives
 *                       in the `MoE_SHM` common region (next to the
 *                       routing scratchpad).
 * @param batch_size     Real token count (≤ Dims::BS); ragged tokens
 *                       get the helper's zero-fill + scale = 1.0f path.
 */
template <typename Dims, std::size_t KBlocks, std::size_t Bs, std::size_t KStep,
          std::size_t Fp8KBlocks, std::size_t Fp8NumChunks, std::size_t Fp8Tok,
          std::size_t Fp8KInner, std::size_t ScaleKBlocks, std::size_t ScaleBs>
__device__ inline void routing_phase_quantize(
    const A_element (&bf16_in_full)[KBlocks][Bs][KStep],
    AQ_element (&fp8_act_full)[Fp8KBlocks][Fp8NumChunks][Fp8Tok][Fp8KInner],
    float (&act_scale)[ScaleKBlocks][ScaleBs], std::uint32_t batch_size) {
  using CoreDims = MoECoreDims<Dims>;

  // Scope guard — BS8 TMA+WGMMA only.
  static_assert(Dims::BS <= 8, "routing_phase_quantize is BS8-only");
  // 128-K SWZ128 atoms per token along K (= HIDDEN_STATES / 128).  16
  // for Qwen3.5.  This is the same constant declared as
  // TinyDataWGMMA_TMA::K_BLOCKS_TOTAL; we recompute it locally so the
  // helper does not need to friend that struct.
  constexpr std::uint32_t K_BLOCKS_TOTAL = Dims::HIDDEN_STATES / CoreDims::K_STEP_WGMMA;
  static_assert(KBlocks == K_BLOCKS_TOTAL,
                "bf16_in_full outer extent must be K_BLOCKS_TOTAL for BS8");
  static_assert(Bs == Dims::BS, "bf16_in_full middle extent must be Dims::BS for BS8");
  static_assert(KStep == CoreDims::K_STEP_WGMMA,
                "bf16_in_full inner extent must be K_STEP_WGMMA (128)");
  static_assert(Fp8NumChunks == 8, "fp8_act_full middle dim must be 8");
  static_assert(Fp8Tok == CoreDims::T_TILE || Fp8Tok == CoreDims::T_TILE + 1,
                "fp8_act_full token dim must be T_TILE (legacy) or "
                "T_TILE+1 (padded layout that breaks the 128-byte kc "
                "stride to avoid bank conflicts on the routing-quantize "
                "STS — see comment on `MoE_SHM::TinyDataWGMMA_TMA::"
                "fp8_act_full` for the design)");
  static_assert(Fp8KInner == 16, "fp8_act_full inner dim must be 16");
  static_assert(ScaleBs == Dims::BS, "act_scale inner extent must be Dims::BS");
  static_assert(Fp8KBlocks == K_BLOCKS_TOTAL,
                "fp8_act_full outer extent must be K_BLOCKS_TOTAL for BS8");
  static_assert(ScaleKBlocks == K_BLOCKS_TOTAL, "act_scale outer extent must be K_BLOCKS_TOTAL");

  constexpr std::uint32_t PAIRS_TOTAL = Dims::BS * K_BLOCKS_TOTAL;
  // Warps 1..11 participate (warp 0 runs prepare_moe_topk_BS8).
  constexpr std::uint32_t NUM_QUANT_WARPS = 11u;

  const std::uint32_t warp = get_any_warp<Dims>();  // ∈ [1, 12)
  // Defense-in-depth: warp 0 must be gated out by the caller.  Failing
  // this assert at runtime would mean warp 0 also wrote some pair —
  // since `w_idx = warp - 1u` would underflow, we'd partition the work
  // wrong.  This is checked in the caller's `if (warp == 0) { ... }
  // else { routing_phase_quantize(...) }` dispatch.
  assert(warp >= 1u && warp < 12u);
  const std::uint32_t w_idx = warp - 1u;  // ∈ [0, 11)

// Stride-NUM_QUANT_WARPS partition over the linear pair index.
// `#pragma unroll 1` keeps the loop body small — each iteration is
// already a moe_streaming_quantize_k128 call which is __forceinline.
#pragma unroll 1
  for (std::uint32_t i = w_idx; i < PAIRS_TOTAL; i += NUM_QUANT_WARPS) {
    const std::uint32_t token = i / K_BLOCKS_TOTAL;
    const std::uint32_t kblk = i % K_BLOCKS_TOTAL;

    // Tile-major BF16 source: `bf16_in_full[kblk][token]` is the
    // natural `[K_STEP_WGMMA = 128]` row written by the Phase-1 TMA's
    // compact 2 KB box for this K-substep.  No row-stride
    // reinterpretation is needed — the array reference type already
    // matches the helper's `[128]` row signature.
    const auto& bf_row = bf16_in_full[kblk][token];

    // FP8 view for this k_block: fp8_act_full[kblk] — same
    // [kc][tok][ki] shape the helper writes into.  The helper's `tok`
    // argument selects which token slot to write FP8 bytes into; other
    // tokens at the same k_block are written by other warps owning
    // other pairs.
    auto& fp8_atom = fp8_act_full[kblk];

    moe_streaming_quantize_k128<Dims>(bf_row, fp8_atom, /*tok=*/token, batch_size,
                                      &act_scale[kblk][token]);
  }
}

}  // namespace monomoe

#endif
