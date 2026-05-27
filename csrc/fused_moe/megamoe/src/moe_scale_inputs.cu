
#pragma once
#ifndef MOE_SCALE_INPUTS_CU
  #define MOE_SCALE_INPUTS_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cstdint>
  #include <cstring>

  #include <cuda/pipeline>
  #include <cuda_bf16.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "moe_grid_barrier.h"
  #include "ptx_utils.h"
  #include "moe_debug.h"

namespace moe_monokernel {

/**
 * @brief Sets NaNs (positive or negative, any payload) to 0.0 (bit-pattern all
 * 0).
 *
 * Values other than NaN remain unchanged.
 */
__device__ static __forceinline__ __nv_bfloat162
mask_NaNs_to_zero(__nv_bfloat162 xs) {
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
  __device__ __nv_bfloat162 first_pair() const {
    return type_pun<__nv_bfloat162>(raw.x);
  }
  /**
   * @brief Returns bfloat16 2 and 3 as pair.
   */
  __device__ __nv_bfloat162 second_pair() const {
    return type_pun<__nv_bfloat162>(raw.y);
  }
  /**
   * @brief Returns bfloat16 5 and 5 as pair.
   */
  __device__ __nv_bfloat162 third_pair() const {
    return type_pun<__nv_bfloat162>(raw.z);
  }
  /**
   * @brief Returns bfloat16 6 and 7 as pair.
   */
  __device__ __nv_bfloat162 fourth_pair() const {
    return type_pun<__nv_bfloat162>(raw.w);
  }

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

    __nv_fp8x4_e4m3 converted0{
        float4{f0.x * scale, f0.y * scale, f1.x * scale, f1.y * scale}};
    __nv_fp8x4_e4m3 converted1{
        float4{f2.x * scale, f2.y * scale, f3.x * scale, f3.y * scale}};

    return type_pun<uint32_t>(converted0) |
           ((uint64_t)type_pun<uint32_t>(converted1) << 32);
  }
};

}  // namespace

/**
 * @brief Fetches all activations for a single token from global to shared
 * memory.
 *
 * Fetches all @c Dims::HIDDEN_STATES activations for a single token
 * asynchronously from global to shared memory. Before accessing the shared
 * memory values, wait for the transfer via @c pipe .
 *
 * @param source Pointer to the first activation of the token (in global
 * memory).
 * @param dest Pointer to the first activation of the token (in shared memory).
 * @param pipe CUDA pipeline to execute the transfer in
 */
template <typename Dims>
__device__ void moe_fetch_activation_async(
    const A_element* __restrict__ source, A_element* __restrict dest,
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;

  const std::uint32_t thread = get_thread<Dims>();
  const std::uint32_t warp = get_any_warp<Dims>();
  const std::uint32_t thread_chunk_size = 16 / sizeof(*source);
  const std::uint32_t chunk_size =
      CoreDims::THREADS_PER_WARP * thread_chunk_size;

  pipe.producer_acquire();
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    copy128(dest[k], source[k], pipe);
  }
  pipe.producer_commit();
}

/**
 * @brief Quantizes activation values for a single token (BS8 path).
 *
 * Reads bf16 activations from shared memory, computes per-block (1, 128)
 * activation scales: act_scale[blk] = max(|x[blk*128..(blk+1)*128-1]|) / 448,
 * writes fp8 quantized activations to @p activation_out with a 32-byte swizzle
 * (rotate_col_32) so that MMA loads can use the same rotation pattern as
 * weights, eliminating shared-memory bank conflicts.
 *
 * @param [in]  activation_in  bf16 activations in shared memory (16-byte
 * aligned)
 * @param [out] activation_out fp8 quantized activations (8-byte aligned)
 * @param [in]  row            Row index within the tile (used for swizzle)
 * @param [out] act_scales_out Array of per-block scales (K/128 elements)
 */
template <typename Dims>
__device__ void moe_scale_activation_BS8(
    const A_element* __restrict__ activation_in,
    AQ_element* __restrict__ activation_out, std::uint32_t row,
    float* __restrict__ act_scales_out) {
  static_assert(Dims::BS <= 8, "This function is only for use with BS up to 8");
  assert((uintptr_t)activation_in != (uintptr_t)activation_out);
  static_assert(Dims::HIDDEN_STATES * sizeof(A_element) % 16 == 0);
  static_assert(Dims::HIDDEN_STATES % 8 == 0);
  assert((uintptr_t)activation_in % 16 == 0);
  assert((uintptr_t)activation_out % 8 == 0);

  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t ACT_BLOCK = 128;
  constexpr uint32_t NUM_ACT_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK - 1) / ACT_BLOCK;
  static_assert(Dims::HIDDEN_STATES % ACT_BLOCK == 0,
                "HIDDEN_STATES must be divisible by activation block size");

  // Per-thread chunk chosen so that one warp iteration covers exactly one
  // 128-element quant block with all 32 lanes active (32 × 4 = 128).
  // This also keeps the converted fp32 values in 4 registers between the
  // block-max reduction and the quantize+write step, eliminating the
  // redundant bf16 reload that the prior 8-element-per-thread version did.
  constexpr uint32_t FLOATS_PER_LOAD = 4;  // 4 bf16 = 8 bytes per thread
  static_assert(CoreDims::THREADS_PER_WARP * FLOATS_PER_LOAD == ACT_BLOCK,
                "Warp iteration must equal one 128-element quant block");

  const std::uint32_t thread = get_thread<Dims>();

  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  // Process each 128-element block separately, single pass per block.
  #pragma unroll
  for (uint32_t blk = 0; blk < NUM_ACT_BLOCKS; ++blk) {
    uint32_t blk_start = blk * ACT_BLOCK;
    uint32_t col = blk_start + thread * FLOATS_PER_LOAD;

    // Load 4 bf16 as 2× bf162 → convert to 4 floats in registers
    __nv_bfloat162 bf_01 =
        *reinterpret_cast<const __nv_bfloat162*>(&activation_in[col + 0]);
    __nv_bfloat162 bf_23 =
        *reinterpret_cast<const __nv_bfloat162*>(&activation_in[col + 2]);
    // Swallow NaNs to 0 (same semantics as BF16x8::to_fp8x8)
    bf_01 = mask_NaNs_to_zero(bf_01);
    bf_23 = mask_NaNs_to_zero(bf_23);
    float2 f01 = __bfloat1622float2(bf_01);
    float2 f23 = __bfloat1622float2(bf_23);
    float r0 = f01.x, r1 = f01.y, r2 = f23.x, r3 = f23.y;

    float local_max =
        fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3)));
    float blk_max = warp_reduce_max_float(local_max);
    if (blk_max < __FLT_MIN__) blk_max = 1.f;

    float blk_act_scale = blk_max * FP8_MAX_INV;  // = max/448
    float blk_inv_scale = FP8_MAX / blk_max;      // = 448/max

    // Quantize the 4 floats we already have in registers → fp8x4 (4 bytes)
    // and write with rotate_col_32 swizzle so MMA loads hit distinct banks.
    __nv_fp8x4_e4m3 q{float4{r0 * blk_inv_scale, r1 * blk_inv_scale,
                             r2 * blk_inv_scale, r3 * blk_inv_scale}};
    uint32_t packed = type_pun<uint32_t>(q);
    uint32_t swz_col = rotate_col_32(col, row);
    *reinterpret_cast<uint32_t*>(&activation_out[swz_col]) = packed;

    // Store per-block scale
    if (thread == 0) act_scales_out[blk] = blk_act_scale;
  }
}

/**
 * @brief WGMMA variant of @ref moe_scale_activation_BS8.
 *
 * Writes the quantized fp8 activations in **canonical WGMMA K-major**
 * layout for the B operand of `wgmma.mma_async.m64nNk32.e4m3`:
 *
 *   out[k_chunk16][tok_0_7][k_inner_0_15]
 *
 * where k_chunk16 ∈ [0, K/16) — one core matrix per k_chunk.  Each
 * core matrix is 8 tokens × 16 contiguous K-bytes, matching the
 * hardware's 8×16-byte core-matrix tile (PTX §9.7.16.5.1.2).
 *
 * This layout is **N-outer, K-inner**: within a core matrix, the
 * 16 contiguous bytes are 16 consecutive K-values for ONE token.
 * That's the only layout WGMMA accepts for fp8 operands — they must
 * always be K-major.
 *
 * The function only writes its owned token slot (`row`); other tokens
 * at the same k_chunk are written by other calc warps.
 *
 * @param activation_in  bf16 input activations for this token (K elements)
 * @param dest           Shape [K/16][8][16] fp8 K-major buffer
 * @param row            Token slot (0..BS-1) that this call populates
 * @param act_scales_out Per-token per-block scales (K/128 elements)
 */
template <typename Dims, std::size_t DestRows, std::size_t DestCols,
          std::size_t DestN>
__device__ void moe_scale_activation_BS8_wgmma(
    const A_element* __restrict__ activation_in,
    AQ_element (&dest)[DestRows][DestCols][DestN], std::uint32_t row,
    float* __restrict__ act_scales_out) {
  static_assert(Dims::BS <= 8, "This function is only for use with BS up to 8");
  // Canonical WGMMA K-major B-operand layout:
  //   dest[k_chunk16][tok_0_7][k_inner_0_15]
  static_assert(DestRows == Dims::HIDDEN_STATES / 16,
                "dest outer dim must be K/16 (number of K-chunks of 16)");
  static_assert(DestCols == 8, "dest middle dim must be 8 (N rows per tile)");
  static_assert(DestN == 16,
                "dest inner dim must be 16 (K bytes per core matrix)");
  static_assert(Dims::HIDDEN_STATES % 128 == 0,
                "HIDDEN_STATES must be a multiple of 128 (activation block)");

  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t ACT_BLOCK = 128;
  constexpr uint32_t NUM_ACT_BLOCKS = Dims::HIDDEN_STATES / ACT_BLOCK;
  constexpr uint32_t FLOATS_PER_LOAD = 4;
  static_assert(CoreDims::THREADS_PER_WARP * FLOATS_PER_LOAD == ACT_BLOCK,
                "Warp iteration must equal one 128-element quant block");

  const std::uint32_t thread = get_thread<Dims>();
  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  // Process each 128-element block. Each thread covers 4 K-values.
  // With 8 K-chunks (of 16) per 128-block, thread*4 is always aligned
  // such that its 4 K-values fall within a single 16-wide K-chunk.
  #pragma unroll
  for (uint32_t blk = 0; blk < NUM_ACT_BLOCKS; ++blk) {
    uint32_t blk_start = blk * ACT_BLOCK;
    uint32_t col = blk_start + thread * FLOATS_PER_LOAD;

    __nv_bfloat162 bf_01 =
        *reinterpret_cast<const __nv_bfloat162*>(&activation_in[col + 0]);
    __nv_bfloat162 bf_23 =
        *reinterpret_cast<const __nv_bfloat162*>(&activation_in[col + 2]);
    bf_01 = mask_NaNs_to_zero(bf_01);
    bf_23 = mask_NaNs_to_zero(bf_23);
    float2 f01 = __bfloat1622float2(bf_01);
    float2 f23 = __bfloat1622float2(bf_23);
    float r0 = f01.x, r1 = f01.y, r2 = f23.x, r3 = f23.y;

    float local_max =
        fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3)));
    float blk_max = warp_reduce_max_float(local_max);
    if (blk_max < __FLT_MIN__) blk_max = 1.f;

    float blk_act_scale = blk_max * FP8_MAX_INV;
    float blk_inv_scale = FP8_MAX / blk_max;

    AQ_element q0 = (AQ_element)(r0 * blk_inv_scale);
    AQ_element q1 = (AQ_element)(r1 * blk_inv_scale);
    AQ_element q2 = (AQ_element)(r2 * blk_inv_scale);
    AQ_element q3 = (AQ_element)(r3 * blk_inv_scale);

    // K-major canonical write:
    //   dest[(col+i)/16][row][(col+i)%16] = q_i
    // Each thread owns 4 consecutive K-values (col..col+3). Since
    // col is always aligned to 4 and chunk width is 16, the 4 values
    // always fall within the same k_chunk (col/16 == (col+3)/16 when
    // col%16 ∈ {0, 4, 8, 12}).
    uint32_t kc = col / 16;
    uint32_t ki = col % 16;
    dest[kc][row][ki + 0] = q0;
    dest[kc][row][ki + 1] = q1;
    dest[kc][row][ki + 2] = q2;
    dest[kc][row][ki + 3] = q3;

    if (thread == 0) act_scales_out[blk] = blk_act_scale;
  }
}

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
 * Refactor note (topk-bs8-tma-prefetch-quant-fusion task 3.1):
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
template <typename Dims, std::size_t BF16InCols, std::size_t Fp8NumChunks,
          std::size_t Fp8Tok, std::size_t Fp8KInner>
__device__ __forceinline__ void moe_streaming_quantize_k128(
    const A_element (&bf16_row)[BF16InCols],
    AQ_element (&fp8_act)[Fp8NumChunks][Fp8Tok][Fp8KInner], std::uint32_t tok,
    std::uint32_t batch_size, float* __restrict__ act_scale_for_step) {
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
  __nv_bfloat162 bf_01 =
      *reinterpret_cast<const __nv_bfloat162*>(&bf16_row[col + 0]);
  __nv_bfloat162 bf_23 =
      *reinterpret_cast<const __nv_bfloat162*>(&bf16_row[col + 2]);
  bf_01 = mask_NaNs_to_zero(bf_01);
  bf_23 = mask_NaNs_to_zero(bf_23);
  float2 f01 = __bfloat1622float2(bf_01);
  float2 f23 = __bfloat1622float2(bf_23);
  float r0 = f01.x, r1 = f01.y, r2 = f23.x, r3 = f23.y;

  float local_max =
      fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3)));
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
 *      (`u.tiny_wgmma_tma.bar_rwin`) on every warp 1..11 thread before
 *      invoking this helper.  Reading `bf16_in_full` before the wait
 *      succeeds is a data race against the in-flight Phase-1 TMA load.
 *   3. The caller MUST emit a block-wide `__syncthreads()` AFTER this
 *      helper returns to publish the FP8 atom + scale writes to all
 *      warps before the Phase-3 up-projection K-loop reads them.
 *   4. Ragged batches (`token >= batch_size`) are handled inside this
 *      helper via the existing zero-fill + scale = 1.0f path of
 *      `moe_streaming_quantize_k128` (Req 2.7).
 *
 * Scope (Req 7.1, 7.2, 7.3, 7.6): this helper is only valid when
 * `Dims::BS <= 8` AND the call site is on the BS8 TMA+WGMMA path.
 * The static_assert below makes miss-instantiation a build error.
 *
 * Work distribution (Req 2.4): warp `w ∈ [1, 12)` owns pair indices
 *   { i ∈ [0, BS * K_BLOCKS_TOTAL) : (i % 11) == (w - 1) }.
 * With `BS = 8`, `K_BLOCKS_TOTAL = 16`, `PAIRS_TOTAL = 128`, and 11
 * warps, every pair is owned by exactly one warp.  Warp 0 owns no
 * pair.  Imbalance is at most one extra pair per warp (~9%), which
 * is acceptable given the alternative is wasting warps.
 *
 * @tparam Dims          MoE dims.
 * @tparam KBlocks       Outer extent of `bf16_in_full`
 *                       (= K_BLOCKS_TOTAL for BS8; the field is
 *                       BS-clamped to 1 for BS64, see
 *                       TinyDataWGMMA_TMA::BF16_IN_FULL_K_BLOCKS).
 * @tparam Bs            Middle extent of `bf16_in_full` (= Dims::BS
 *                       for BS8, BS-clamped to 1 for BS64).
 * @tparam KStep         Inner extent of `bf16_in_full` (= K_STEP_WGMMA
 *                       = 128 for BS8, BS-clamped to 1 for BS64).
 * @tparam Fp8KBlocks    Outer extent of `fp8_act_full` (= K_BLOCKS_TOTAL
 *                       for BS8, BS-clamped to 1 for BS64).
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
 *
 * Implements R2.3, R2.4, R2.6, R2.7, R2.8.
 * Design: "New helper: routing_phase_quantize",
 *         "Phase 2 — Prepare (concurrent across 12 warps)".
 */
template <typename Dims, std::size_t KBlocks, std::size_t Bs, std::size_t KStep,
          std::size_t Fp8KBlocks, std::size_t Fp8NumChunks, std::size_t Fp8Tok,
          std::size_t Fp8KInner, std::size_t ScaleKBlocks, std::size_t ScaleBs>
__device__ inline void routing_phase_quantize(
    const A_element (&bf16_in_full)[KBlocks][Bs][KStep],
    AQ_element (&fp8_act_full)[Fp8KBlocks][Fp8NumChunks][Fp8Tok][Fp8KInner],
    float (&act_scale)[ScaleKBlocks][ScaleBs], std::uint32_t batch_size) {
  using CoreDims = MoECoreDims<Dims>;

  // Scope guard — BS8 TMA+WGMMA only (Req 7.1, 7.2, 7.3, 7.6).
  static_assert(Dims::BS <= 8,
                "routing_phase_quantize is BS8-only (Req 7.1, 7.2, 7.3, 7.6)");
  // 128-K SWZ128 atoms per token along K (= HIDDEN_STATES / 128).  16
  // for Qwen3.5.  This is the same constant declared as
  // TinyDataWGMMA_TMA::K_BLOCKS_TOTAL; we recompute it locally so the
  // helper does not need to friend that struct.
  constexpr std::uint32_t K_BLOCKS_TOTAL =
      Dims::HIDDEN_STATES / CoreDims::K_STEP_WGMMA;
  static_assert(KBlocks == K_BLOCKS_TOTAL,
                "bf16_in_full outer extent must be K_BLOCKS_TOTAL for BS8");
  static_assert(Bs == Dims::BS,
                "bf16_in_full middle extent must be Dims::BS for BS8");
  static_assert(KStep == CoreDims::K_STEP_WGMMA,
                "bf16_in_full inner extent must be K_STEP_WGMMA (128)");
  static_assert(Fp8NumChunks == 8, "fp8_act_full middle dim must be 8");
  static_assert(Fp8Tok == CoreDims::T_TILE || Fp8Tok == CoreDims::T_TILE + 1,
                "fp8_act_full token dim must be T_TILE (legacy) or "
                "T_TILE+1 (padded layout that breaks the 128-byte kc "
                "stride to avoid bank conflicts on the routing-quantize "
                "STS — see comment on `MoE_SHM::U::TinyDataWGMMA_TMA::"
                "fp8_act_full` for the design)");
  static_assert(Fp8KInner == 16, "fp8_act_full inner dim must be 16");
  static_assert(ScaleBs == Dims::BS, "act_scale inner extent must be Dims::BS");
  static_assert(Fp8KBlocks == K_BLOCKS_TOTAL,
                "fp8_act_full outer extent must be K_BLOCKS_TOTAL for BS8");
  static_assert(ScaleKBlocks == K_BLOCKS_TOTAL,
                "act_scale outer extent must be K_BLOCKS_TOTAL");

  constexpr std::uint32_t PAIRS_TOTAL = Dims::BS * K_BLOCKS_TOTAL;
  // Warps 1..11 participate (warp 0 runs prepare_moe_topk_BS8).
  constexpr std::uint32_t NUM_QUANT_WARPS = 11u;

  const std::uint32_t warp = get_any_warp<Dims>();  // ∈ [1, 12)
  // Defense-in-depth: warp 0 must be gated out by the caller.  Failing
  // this assert at runtime would mean warp 0 also wrote some pair —
  // since `w_idx = warp - 1u` would underflow, we'd partition the work
  // wrong.  This is checked in the caller's `if (warp == 0) { ... }
  // else { routing_phase_quantize(...) }` dispatch (Req 2.1).
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

    moe_streaming_quantize_k128<Dims>(bf_row, fp8_atom, /*tok=*/token,
                                      batch_size, &act_scale[kblk][token]);
  }
}

namespace detail {

/**
 * @brief Scales activation values for a single token (BS > 8 path).
 *
 * Quantizes one token's activations from global memory to fp8, computing
 * per-block (1, 128) activation scales:
 *   act_scale[blk] = max(|x[blk*128..(blk+1)*128-1]|) / 448
 * and writing them to @p act_scales_out.
 * No routing weight folding — act_scales are stored separately.
 */
template <typename Dims>
__device__ static void moe_scale_activation_BSx_chunk(
    const A_element* __restrict__ activation_in, A_element* __restrict__ temp,
    AQ_element* __restrict__ activation_out,
    float* __restrict__ act_scales_out) {
  assert((uintptr_t)activation_in != (uintptr_t)temp);
  assert((uintptr_t)activation_out != (uintptr_t)temp);
  assert((uintptr_t)activation_in != (uintptr_t)activation_out);
  using CoreDims = MoECoreDims<Dims>;

  constexpr uint32_t ACT_BLOCK = 128;
  constexpr uint32_t NUM_ACT_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK - 1) / ACT_BLOCK;

  const std::uint32_t thread = get_thread<Dims>();
  const std::uint32_t thread_chunk_size =
      sizeof(BF16x8) / sizeof(*activation_in);
  const std::uint32_t chunk_size =
      CoreDims::THREADS_PER_WARP * thread_chunk_size;

  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  // First pass: copy to temp (needed by caller) and compute per-block max
  // We process the entire row but track max per 128-element block.
  float block_max[NUM_ACT_BLOCKS];
  for (uint32_t b = 0; b < NUM_ACT_BLOCKS; ++b) block_max[b] = 0.f;

  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    BF16x8 chunk_val = BF16x8::load(activation_in + k);
    chunk_val.store_to(&temp[k]);

    __nv_bfloat162 a0 = __habs2(chunk_val.first_pair());
    __nv_bfloat162 a1 = __habs2(chunk_val.second_pair());
    __nv_bfloat162 a2 = __habs2(chunk_val.third_pair());
    __nv_bfloat162 a3 = __habs2(chunk_val.fourth_pair());
    __nv_bfloat162 mx = __hmax2(__hmax2(a0, a1), __hmax2(a2, a3));
    float local_max = (float)__hmax(mx.x, mx.y);

    uint32_t blk = k / ACT_BLOCK;
    block_max[blk] = fmaxf(block_max[blk], local_max);
  }

  // Warp-reduce each block's max
  for (uint32_t b = 0; b < NUM_ACT_BLOCKS; ++b) {
    block_max[b] = warp_reduce_max_float(block_max[b]);
    if (block_max[b] < __FLT_MIN__) block_max[b] = 1.f;
  }

  // Second pass: quantize each block with its own scale
  uint64_t* activation_out8 = reinterpret_cast<uint64_t*>(activation_out);
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    uint32_t blk = k / ACT_BLOCK;
    float inv_scale = FP8_MAX / block_max[blk];
    BF16x8 chunk_val = BF16x8::load(activation_in + k);
    activation_out8[k / 8] = chunk_val.to_fp8x8(inv_scale);
  }

  // Store per-block scales
  if (thread == 0) {
    for (uint32_t b = 0; b < NUM_ACT_BLOCKS; ++b) {
      act_scales_out[b] = block_max[b] * FP8_MAX_INV;
    }
  }
}

}  // namespace detail

/**
 * @brief Quantizes activations for all tokens (BS > 8 path).
 *
 * Writes fp8 quantized activations to spec->activations[i] and
 * act_scale per token to shmem->act_scale[i].
 * This function is collective across all CUDA blocks.
 */
template <typename Dims>
__device__ void moe_scale_activation_BSx(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    uint32_t* __restrict__ grid_counters, uint32_t& grid_phase) {
  static_assert(Dims::BS > 8,
                "BS=8 is handled by its own kernel. Do not use "
                "moe_scale_inputs for BS<=8");
  static_assert(Dims::HIDDEN_STATES * sizeof(A_element) % 16 == 0,
                "Next token activation will not be properly aligned.");
  static_assert(
      Dims::HIDDEN_STATES % 8 == 0,
      "Next quantized token activation will not be properly aligned.");

  assert((uintptr_t)activations_in % 16 == 0);
  assert((uintptr_t)spec->activations % 8 == 0);

  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t NUM_ACT_BLOCKS = MoEGemmSpec<Dims>::ACT_SCALE_BLOCKS;

  if (is_calc_warp<Dims>()) {
    const std::uint32_t global_warp_count =
        gridDim.x * CoreDims::CALC_WARP_COUNT;
    const std::uint32_t warp = get_calc_warp<Dims>();
    const std::uint32_t global_warp =
        blockIdx.x * CoreDims::CALC_WARP_COUNT + warp;

    for (std::uint32_t i = global_warp; i < token_count;
         i += global_warp_count) {
      detail::moe_scale_activation_BSx_chunk<Dims>(
          activations_in + i * Dims::HIDDEN_STATES, shmem->u.rescale.a[warp],
          spec->activations[i], spec->act_scale[i]);
    }
  }

  // spec->act_scale is written by different blocks — make visible to all
  // via the software grid barrier (formerly cooperative_groups::
  // this_grid().sync()).  See Requirement 3.1/3.3 and Design Site #5.
  moe_monokernel::grid_barrier<Dims::KernelConfig::GRID_SIZE>(grid_counters,
                                                              grid_phase);

  // copy per-block act_scale into shmem for fast per-token access.
  //
  // GM `spec->act_scale` is `[BS][BLK]` (cheap coalesced GM stores
  // produced by the per-block quantization above) but SHM
  // `shmem->act_scale` is `[BLK][BS]` (chosen for bank-conflict-free
  // WGMMA scale-apply reads, see comment on the SHM declaration in
  // `MoE_SHM`).  This loop transposes on the fly.
  for (uint32_t i = threadIdx.x; i < token_count * NUM_ACT_BLOCKS;
       i += blockDim.x) {
    uint32_t tok = i / NUM_ACT_BLOCKS;
    uint32_t blk = i % NUM_ACT_BLOCKS;
    shmem->act_scale[blk][tok] = spec->act_scale[tok][blk];
  }

  __syncthreads();

  #ifdef DEBUG_MOE_PRINT
  // Print activation quantization results for first 2 tokens
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (uint32_t tok = 0; tok < min(token_count, (uint32_t)2); ++tok) {
      printf("[DBG64 ACT_QUANT tok=%u] act_scale (%u blocks):", tok,
             NUM_ACT_BLOCKS);
      for (uint32_t b = 0; b < NUM_ACT_BLOCKS; ++b)
        printf(" %.6f", shmem->act_scale[b][tok]);
      printf("\n");
      printf("[DBG64 ACT_QUANT tok=%u] fp8[0..7]:", tok);
      for (int i = 0; i < 8; i++)
        printf(" %.4f", (float)spec->activations[tok][i]);
      printf("\n");
    }
  }
  #endif
}

}  // namespace moe_monokernel

#endif
