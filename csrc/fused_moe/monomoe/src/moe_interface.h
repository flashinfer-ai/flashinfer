#ifndef MOE_INTERFACE_H
#define MOE_INTERFACE_H

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace monomoe {

// Weight quantization granularity
enum class QuantGranularity : uint32_t {
  PER_CHANNEL = 0,  // one scale per row (original)
  BLOCK_WISE = 1,   // one scale per (block_row, block_col) tile
};

template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts>
struct MoEDimensions {
  static constexpr uint32_t HIDDEN_STATES = k;
  static constexpr uint32_t K = k;
  static constexpr uint32_t N = n;
  static constexpr uint32_t BS = m;
  static constexpr uint32_t M = m;
  static constexpr uint32_t NUM_EXPERTS = num_experts;

  // Default: per-channel quantization (backward compatible)
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::PER_CHANNEL;
  static constexpr uint32_t BLOCK_SCALE_ROW = 0;
  static constexpr uint32_t BLOCK_SCALE_COL = 0;

  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = (2 * N) / 16;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
    // Defaults for the generic (non-WGMMA) path.  Concrete shape variants
    // (e.g. Dims_BS8_..._WGMMA_TMA) override these.  K_STEP_* default to 128
    // (the SWZ128 atom K-width).
    static constexpr bool USE_WGMMA = false;
    static constexpr bool USE_TMA = false;
    static constexpr std::uint32_t K_STEP_DOWN = 128;
    static constexpr std::uint32_t K_STEP_UP = 128;
  };
};

// The single BS8 shape variant: TMA + WGMMA, SWIZZLE_128B on both weight
// sides.  Phase 3 (up-proj) uses the Hopper wgmma.mma_async fp8 path with a
// v1 dual-warpgroup K=128 streaming pipeline.  Group geometry (UP_GRID,
// UP_GROUPS, DOWN_GRID, DOWN_GROUPS) and the TMA caller contracts live in
// docs/design_docs/monomoe_kernel.md §5/§6; per-tensor specifics are in the moe_tma.h factory docs.
struct Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA {
  static constexpr uint32_t HIDDEN_STATES = 2048;
  static constexpr uint32_t K = 2048;
  static constexpr uint32_t N = 512;
  static constexpr uint32_t BS = 8;
  static constexpr uint32_t M = 8;
  static constexpr uint32_t NUM_EXPERTS = 256;
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::BLOCK_WISE;
  static constexpr uint32_t BLOCK_SCALE_ROW = 128;
  static constexpr uint32_t BLOCK_SCALE_COL = 128;
  static constexpr uint32_t UP_SCALE_ROWS = (2 * N + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;  // 8
  static constexpr uint32_t UP_SCALE_COLS = (K + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;      // 16
  static constexpr uint32_t DOWN_SCALE_ROWS = (K + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;    // 16
  static constexpr uint32_t DOWN_SCALE_COLS = (N + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;    // 4
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = 128;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
    static constexpr bool USE_WGMMA = true;
    // Enables the TMA-based weight + activation load path in Phase 3 of
    // the BS8 WGMMA up-projection kernel.
    static constexpr bool USE_TMA = true;
    // Down-projection outer K-step width (must be a multiple of 128).
    // Default 128 matches the SWZ128 atom K-width and reproduces the
    // legacy single-substep behaviour.  Set to 256 to halve the number
    // of outer K-iterations (Dims::N / K_STEP_DOWN), at the cost of
    // doubling the per-slot weight tile in SHM (see the K-step note in
    // moe_internal.h).
    static constexpr std::uint32_t K_STEP_DOWN = 256;
    // Up-projection outer K-step width (must be a multiple of 128).
    // Default 128 matches the SWZ128 atom K-width and reproduces the
    // legacy single-substep behaviour.  Set to 256 to halve the number
    // of outer K-iterations (Dims::HIDDEN_STATES / K_STEP_UP), at the
    // cost of doubling the per-slot bf16/fp8 activation tile and the
    // per-slot weight tile in SHM (see the K-step note in moe_internal.h).
    static constexpr std::uint32_t K_STEP_UP = 256;
  };
};

// Scoring function enum for routing
enum class ScoringFunc : uint32_t {
  SIGMOID = 0,
  SOFTMAX = 1,
};

using W_element = __nv_fp8_e4m3;   // expert weights
using A_element = __nv_bfloat16;   // activations as they go into the GEMM
using AQ_element = __nv_fp8_e4m3;  // activations after quantization
using S_element = float;           // scaling factors
using R_element = __nv_bfloat16;   // MoE output

/**
 * @brief W8A8 MoE kernel with configurable top-K routing, scoring function,
 *        and renormalization.
 *
 * Designed for Qwen3.5-30B-A3B FP8 (softmax scoring, top_k=8, 256 experts).
 * Also supports block-wise (128×128) FP8 quantization for Qwen3.5-35B.
 *
 * @param [in] activations_in Input activations. Shape: [M, K]
 * @param [in] token_count Number of active tokens
 * @param [in] router_logits Router logits. Shape: [M, E]
 * @param [in] expert_weights_up Up-projection weights. Shape: [E, 2*N, K]
 * @param [in] expert_scales_up Up-projection scales.
 *             Per-channel: Shape [E, 2*N]
 *             Block-wise:  Shape [E, ceil(2*N/128), ceil(K/128)]
 * @param [in] expert_weights_down Down-projection weights. Shape: [E, K, N]
 * @param [in] expert_scales_down Down-projection scales.
 *             Per-channel: Shape [E, K]
 *             Block-wise:  Shape [E, ceil(K/128), ceil(N/128)]
 * @param [out] activations_out Output buffer. Shape: [M, K]
 * @param [out] scratchpad Global memory for temporary data
 * @param [in] scratchpad_size Size of the scratchpad
 * @param [in] shmem_size Size of the shared memory
 * @param [in] top_k Number of experts to select per token
 * @param [in] scoring_func Scoring function (SIGMOID or SOFTMAX)
 * @param [in] renormalize Whether to renormalize top-K weights to sum to 1
 */
template <typename Dims>
__global__ extern void moe_kernel_topk(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits, const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up, const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down, R_element* __restrict activations_out,
    void* __restrict__ scratchpad, size_t scratchpad_size, size_t shmem_size, std::uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize, __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc);

}  // namespace monomoe

#endif
