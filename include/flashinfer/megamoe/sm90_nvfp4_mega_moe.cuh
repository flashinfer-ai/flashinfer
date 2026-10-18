#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cstdint>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <flashinfer/megamoe/barrier.cuh>
#include <flashinfer/megamoe/detail/common.cuh>
#include <flashinfer/megamoe/detail/mma_sm90.cuh>
#include <flashinfer/megamoe/detail/ptx.cuh>
#include <flashinfer/megamoe/detail/tma_copy.cuh>
#include <flashinfer/megamoe/layout.cuh>
#include <flashinfer/megamoe/scheduler.cuh>
#include <flashinfer/megamoe/sm90_nvfp4_mega_moe_mode2_dequant.cuh>
#include <flashinfer/megamoe/sym_buffer.cuh>
#include <type_traits>

namespace flashinfer::megamoe {

// ============================================================================
// SM90 (Hopper) NVFP4 MegaMoE kernels
// ----------------------------------------------------------------------------
// Split pipeline:
//   * L1 uses cluster=2 to pair adjacent 64-column output halves.
//   * L2 uses independent cluster=1 CTAs.
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * GEMM TMA-load warps (1 for A+SFA, 1 for B+SFB) feed the pipeline stages.
//   * Math warpgroups (1 or 2, totalling kNumEpilogueThreads) consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout.
//         The deployed split kernel pairs two 64-column N halves in one CTA,
//         combines their row-wise maxima, quantizes the full 128-column output
//         group with one E4M3 scale, and TMA-stores it to the local L1 buffer.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the L2 math warps run the COMBINE step (top-k
//     reduction in BF16) — ported verbatim from the SM100 kernel.
//
// Public BN256 requests use sm90_nvfp4_mega_moe_small_m.cuh. The general host
// runtime instantiates only the split kernels below for BN128 deployment
// weights.
// ============================================================================

template <uint32_t kNumMaxTokensPerRank, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kNumExpertsPerWave,
          uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens, uint32_t kNumStages,
          uint32_t kNumSMs, uint32_t kNumRanks, float kActivationClamp, bool kFastMath,
          bool kL2ArrivalCounterRequested = false, bool kDispatchDequantRequested = false>
CUTLASS_GLOBAL __launch_bounds__(512, 1) void sm90_nvfp4_mega_moe_split_l1_impl(
    int* cumulative_local_expert_recv_stats, const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
    const float* __restrict__ l1_global_scales) {
  constexpr uint32_t BLOCK_M = 128;
  constexpr uint32_t BLOCK_N = 128;
  constexpr uint32_t BLOCK_K = 128;
  constexpr uint32_t kClusterSize = 2;
  constexpr uint32_t kNumDispatchThreads = 128;
  constexpr uint32_t kNumNonEpilogueThreads = 128;
  constexpr uint32_t kNumEpilogueThreads = 256;
  constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
  constexpr uint32_t L1_SHAPE_K = kHidden;
  constexpr uint32_t L2_SHAPE_N = kHidden;
  constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
  constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
  constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
  constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
  constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
  constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <flashinfer/megamoe/sm90_nvfp4_mega_moe_split_l1_body.inl>
}

template <uint32_t kNumMaxTokensPerRank, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kNumExpertsPerWave,
          uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens, uint32_t kNumStages,
          uint32_t kNumSMs, uint32_t kNumRanks, float kActivationClamp, bool kFastMath>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void sm90_nvfp4_mega_moe_split_l2_impl(
    void* y, int* cumulative_local_expert_recv_stats, const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
    const float* __restrict__ l2_global_scales) {
  constexpr uint32_t BLOCK_M = 128;
  constexpr uint32_t BLOCK_N = 128;
  constexpr uint32_t BLOCK_K = 128;
  constexpr uint32_t kNumNonEpilogueThreads = 128;
  constexpr uint32_t kNumEpilogueThreads = 256;
  constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
  constexpr uint32_t L1_SHAPE_K = kHidden;
  constexpr uint32_t L2_SHAPE_N = kHidden;
  constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
  constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
  constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <flashinfer/megamoe/sm90_nvfp4_mega_moe_split_l2_body.inl>
}

}  // namespace flashinfer::megamoe

#pragma clang diagnostic pop
