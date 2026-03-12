#ifndef FLASHINFER_MAMBA_INVOKE_SELECTIVE_STATE_UPDATE_MTP_CUH_
#define FLASHINFER_MAMBA_INVOKE_SELECTIVE_STATE_UPDATE_MTP_CUH_

#include <cuda_runtime_api.h>

#include <iostream>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "create_tensor_map.cuh"
#include "kernel_selective_state_update_mtp_simple.cuh"
#ifdef FLASHINFER_MAMBA_ENABLE_SM100
#include "kernel_selective_state_update_mtp_vertical.cuh"
#endif

namespace flashinfer::mamba::mtp {

using namespace conversion;

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void invokeSelectiveStateUpdateMTP(SelectiveStateMTPParams& params, SSUAlgorithm algorithm,
                                   cudaStream_t stream) {
  constexpr bool scaleState = !std::is_same_v<state_scale_t, void>;
  // Stochastic rounding is only implemented for fp16 state
  if constexpr (PHILOX_ROUNDS > 0) {
    static_assert(std::is_same_v<state_t, half>,
                  "Stochastic rounding (PHILOX_ROUNDS > 0) only supports fp16 state");
  }
  FLASHINFER_CHECK(algorithm == SSUAlgorithm::kAuto || algorithm == SSUAlgorithm::kSimple ||
                       algorithm == SSUAlgorithm::kVertical,
                   "MTP selective_state_update only supports 'auto', 'simple', or 'vertical' "
                   "algorithm, got ",
                   static_cast<int32_t>(algorithm));
  // ── Auto algorithm selection ──────────────────────────────────────────────
  if (algorithm == SSUAlgorithm::kAuto) {
#ifdef FLASHINFER_MAMBA_ENABLE_SM100
    // Vertical kernel doesn't support scaleState
    if (scaleState)
      algorithm = SSUAlgorithm::kSimple;
    else
      algorithm = (params.batch >= 32) ? SSUAlgorithm::kVertical : SSUAlgorithm::kSimple;
#else
    algorithm = SSUAlgorithm::kSimple;
#endif
  }

  // Common alignment checks for all kernels
  check_ptr_alignment_input_vars<input_t>(params);

  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;

  FLASHINFER_CHECK_ALIGNMENT(params.state, sizeof(load_state_t));
  FLASHINFER_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
                   "state head stride must be aligned to ", sizeof(load_state_t), " bytes");

  // Output pointer alignment (both kernels do vectorized stores)
  constexpr auto outputLoadSize = getVectorLoadSizeForFullUtilization<input_t, DIM>();
  using load_output_t = PackedAligned<input_t, outputLoadSize>;
  FLASHINFER_CHECK_ALIGNMENT(params.output, sizeof(load_output_t));

  // Intermediate states pointer alignment (vectorized stores in both kernels)
  if (params.intermediate_states) {
    FLASHINFER_CHECK_ALIGNMENT(params.intermediate_states, sizeof(load_state_t));
  }

  // ── Vertical MTP kernel (SM100+ only) ────────────────────────────────────
#ifdef FLASHINFER_MAMBA_ENABLE_SM100
  if (algorithm == SSUAlgorithm::kVertical) {
    FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                     ") must be divisible by ngroups (", params.ngroups,
                     ") for vertical algorithm");
    // Vertical kernel processes DIM in passes of rowsPerWarpPerPass=4 rows per warp,
    // so DIM must be divisible by (NUM_COMPUTE_WARPS_PER_GROUP * 4) = 16.
    constexpr int kVerticalDimAlignment = NUM_COMPUTE_WARPS_PER_GROUP * 4;
    static_assert(DIM % kVerticalDimAlignment == 0,
                  "Vertical kernel requires DIM divisible by 16 (NUM_COMPUTE_WARPS_PER_GROUP * 4)");
    FLASHINFER_CHECK(!scaleState, "vertical algorithm does not support scaled (quantized) state");

    constexpr int NUM_IN_STAGES = 1;

    dispatchRatio(
        params, std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64>{}, [&]<int HEADS_PER_GROUP>() {
          using sram_t =
              SharedStorageVertical<input_t, state_t, NTOKENS_MTP, DIM, DSTATE, NUM_IN_STAGES>;
          constexpr size_t smem_size = sizeof(sram_t);

          auto func = selective_state_update_kernel_vertical_mtp<
              input_t, weight_t, matrixA_t, state_t, stateIndex_t, NTOKENS_MTP, DIM, DSTATE,
              HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_IN_STAGES>;

          int const total_heads = params.nheads;
          int const num_chunks = (total_heads + NUM_COMPUTE_GROUPS - 1) / NUM_COMPUTE_GROUPS;
          dim3 grid(params.batch, num_chunks);
          dim3 block(warpSize, NUM_WARPS);

          // state descriptor: tile covers full DIM×DSTATE (used for TMA load only)
          auto state_tensor = tma::buildNdDescriptor(
              typeid(state_t),
              /*shapes*/ {DSTATE, DIM, params.nheads, params.state_cache_size},
              /*strides*/ {1, DSTATE, DSTATE * DIM, params.state_stride_batch},
              /*tiles*/ {DSTATE, DIM, 1, 1}, params.state);

          // B/C descriptor
          auto B_tensor = tma::buildNdDescriptor(
              typeid(input_t),
              {(uint64_t)DSTATE, (uint64_t)params.ngroups, (uint64_t)params.ntokens_mtp,
               (uint64_t)params.batch},
              {1, (uint64_t)DSTATE, (uint64_t)params.B_stride_mtp, (uint64_t)params.B_stride_batch},
              {DSTATE, 1, NTOKENS_MTP, 1}, params.B);

          auto C_tensor = tma::buildNdDescriptor(
              typeid(input_t),
              {(uint64_t)DSTATE, (uint64_t)params.ngroups, (uint64_t)params.ntokens_mtp,
               (uint64_t)params.batch},
              {1, (uint64_t)DSTATE, (uint64_t)params.C_stride_mtp, (uint64_t)params.C_stride_batch},
              {DSTATE, 1, NTOKENS_MTP, 1}, params.C);

          // x descriptor: tile covers full DIM, all tokens
          auto x_tensor = tma::buildNdDescriptor(
              typeid(input_t),
              /*shapes*/
              {(uint64_t)DIM, (uint64_t)params.nheads, (uint64_t)params.ntokens_mtp,
               (uint64_t)params.batch},
              /*strides*/
              {1, (uint64_t)DIM, (uint64_t)params.x_stride_mtp, (uint64_t)params.x_stride_batch},
              /*tiles*/ {DIM, 1, NTOKENS_MTP, 1}, params.x);

          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          func<<<grid, block, smem_size, stream>>>(params, state_tensor, B_tensor, C_tensor,
                                                   x_tensor);
        });
    return;
  }
#else
  FLASHINFER_CHECK(algorithm != SSUAlgorithm::kVertical,
                   "vertical MTP algorithm requires SM100+ (Blackwell); "
                   "recompile with FLASHINFER_MAMBA_ENABLE_SM100");
#endif

  // ── Simple MTP kernel ────────────────────────────────────────────────────
  constexpr int numWarps = 4;
  constexpr int stateRowsPerWarpPerStage = 4;
  constexpr int stateRowsPerBlockPerStage = stateRowsPerWarpPerStage * numWarps;
  int const total_tiles = params.batch * params.nheads;
  int const num_sms = GetCudaMultiProcessorCount();

  dim3 block(warpSize, numWarps);
  if (total_tiles < num_sms * 2) {
    // Small tile per CTA (stateRowsPerBlockPerStage * DSTATE): split dim across grid.z for GPU
    // occupancy
    int const dim_tiles = (DIM + stateRowsPerBlockPerStage - 1) / stateRowsPerBlockPerStage;
    dim3 grid(params.batch, params.nheads, dim_tiles);
    auto func = selective_state_update_kernel_simple_mtp<
        input_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t, NTOKENS_MTP, DIM,
        DSTATE, stateRowsPerBlockPerStage, PHILOX_ROUNDS, numWarps>;
    using sram_t =
        SharedStorageSimple<input_t, state_t, state_scale_t, NTOKENS_MTP, stateRowsPerBlockPerStage,
                            DSTATE, stateRowsPerBlockPerStage>;
    constexpr size_t smem_size = sizeof(sram_t);
    FLASHINFER_CUDA_CHECK(
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    func<<<grid, block, smem_size, stream>>>(params);
  } else {
    // Full tile per CTA (DIM * DSTATE): enough blocks for occupancy, no dim splitting needed
    dim3 grid(params.batch, params.nheads);
    auto func = selective_state_update_kernel_simple_mtp<input_t, weight_t, matrixA_t, state_t,
                                                         stateIndex_t, state_scale_t, NTOKENS_MTP,
                                                         DIM, DSTATE, DIM, PHILOX_ROUNDS, numWarps>;
    using sram_t = SharedStorageSimple<input_t, state_t, state_scale_t, NTOKENS_MTP, DIM, DSTATE,
                                       stateRowsPerBlockPerStage>;
    constexpr size_t smem_size = sizeof(sram_t);
    FLASHINFER_CUDA_CHECK(
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    func<<<grid, block, smem_size, stream>>>(params);
  }
}

}  // namespace flashinfer::mamba::mtp

#endif  // FLASHINFER_MAMBA_INVOKE_SELECTIVE_STATE_UPDATE_MTP_CUH_
