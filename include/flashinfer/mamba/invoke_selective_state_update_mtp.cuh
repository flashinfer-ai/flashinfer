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
#include "kernel_selective_state_update_mtp_horizontal.cuh"
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
                       algorithm == SSUAlgorithm::kVertical ||
                       algorithm == SSUAlgorithm::kHorizontal,
                   "MTP selective_state_update only supports 'auto', 'simple', 'vertical', "
                   "or 'horizontal' algorithm, got ",
                   static_cast<int32_t>(algorithm));
  // ── Auto algorithm selection ──────────────────────────────────────────────
  if (algorithm == SSUAlgorithm::kAuto) {
#ifdef FLASHINFER_MAMBA_ENABLE_SM100
    // Horizontal/vertical kernels don't support scaleState or varlen
    if (scaleState || params.cu_seqlens)
      algorithm = SSUAlgorithm::kSimple;
    else
      algorithm = (params.batch >= 32) ? SSUAlgorithm::kHorizontal : SSUAlgorithm::kSimple;
#else
    algorithm = SSUAlgorithm::kSimple;
#endif
  }

  // Common alignment checks for all kernels
  check_ptr_alignment_input_vars<input_t>(params);

  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;

  FLASHINFER_CHECK_ALIGNMENT(params.state, alignof(load_state_t));
  FLASHINFER_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
                   "state head stride must be aligned to ", sizeof(load_state_t), " bytes");

  // Output pointer alignment (both kernels do vectorized stores)
  constexpr auto outputLoadSize = getVectorLoadSizeForFullUtilization<input_t, DIM>();
  using load_output_t = PackedAligned<input_t, outputLoadSize>;
  FLASHINFER_CHECK_ALIGNMENT(params.output, alignof(load_output_t));

  // Intermediate states pointer alignment (vectorized stores in both kernels)
  if (params.intermediate_states) {
    FLASHINFER_CHECK_ALIGNMENT(params.intermediate_states, alignof(load_state_t));
  }

  // ── Vertical MTP kernel (SM100+ only) ────────────────────────────────────
#ifdef FLASHINFER_MAMBA_ENABLE_SM100
  if (algorithm == SSUAlgorithm::kVertical) {
    FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                     ") must be divisible by ngroups (", params.ngroups,
                     ") for vertical algorithm");
    constexpr int kVerticalDimAlignment = warpSize;  // epilogue: elemsPerThread = DIM / warpSize
    FLASHINFER_CHECK(DIM % kVerticalDimAlignment == 0,
                     "Vertical kernel requires DIM divisible by 32 (warpSize), got DIM=", DIM);
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

          auto state_tensor = tma::buildNdDescriptor(
              typeid(state_t),
              /*shapes*/ {DSTATE, DIM, params.nheads, params.state_cache_size},
              /*strides*/ {1, DSTATE, DSTATE * DIM, params.state_stride_batch},
              /*tiles*/ {DSTATE, DIM, 1, 1}, params.state);

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

  // ── Horizontal MTP kernel (SM100+ only, HEADS_PER_CTA heads/CTA, pipelined) ──
  if (algorithm == SSUAlgorithm::kHorizontal) {
    FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                     ") must be divisible by ngroups (", params.ngroups,
                     ") for horizontal algorithm");
    constexpr int kHorizontalDimAlignment =
        horiz::NUM_COMPUTE_WARPS_PER_GROUP * horiz::ROWS_PER_WARP;
    FLASHINFER_CHECK(DIM % kHorizontalDimAlignment == 0,
                     "Horizontal kernel requires DIM divisible by ", kHorizontalDimAlignment,
                     " (NUM_COMPUTE_WARPS_PER_GROUP * ROWS_PER_WARP), got DIM=", DIM);
    FLASHINFER_CHECK(!scaleState, "horizontal algorithm does not support scaled (quantized) state");

    constexpr int NUM_IN_STAGES = 2;
    // TMA_STATE_ROWS: rows of DIM per TMA transaction. Must be a multiple of ROWS_PER_PASS.
    // Larger values = fewer barrier syncs but more smem per pipeline stage.
    constexpr int TMA_STATE_ROWS = 2 * horiz::ROWS_PER_PASS;

    dispatchRatio(
        params, std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64>{}, [&]<int HEADS_PER_GROUP>() {
          constexpr int HEADS_PER_CTA = 1;
          static_assert(HEADS_PER_GROUP % HEADS_PER_CTA == 0);

          using sram_t = GroupStorageHorizontal<input_t, state_t, NTOKENS_MTP, DIM, DSTATE,
                                                NUM_IN_STAGES, TMA_STATE_ROWS, HEADS_PER_CTA>;
          constexpr size_t smem_size = sizeof(sram_t);

          auto func = selective_state_update_kernel_horizontal_mtp<
              input_t, weight_t, matrixA_t, state_t, stateIndex_t, NTOKENS_MTP, DIM, DSTATE,
              HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_IN_STAGES, TMA_STATE_ROWS, HEADS_PER_CTA>;

          FLASHINFER_CHECK(params.nheads % HEADS_PER_CTA == 0, "nheads (", params.nheads,
                           ") must be divisible by HEADS_PER_CTA (", HEADS_PER_CTA,
                           ") for horizontal algorithm");

          dim3 grid(params.batch, params.nheads / HEADS_PER_CTA);
          dim3 block(warpSize, horiz::NUM_WARPS);

          // TMA state descriptor: tile by TMA_STATE_ROWS
          auto state_tensor = tma::buildNdDescriptor(
              typeid(state_t),
              /*shapes*/ {DSTATE, DIM, params.nheads, params.state_cache_size},
              /*strides*/ {1, DSTATE, DSTATE * DIM, params.state_stride_batch},
              /*tiles*/ {DSTATE, TMA_STATE_ROWS, 1, 1}, params.state);

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
  FLASHINFER_CHECK(algorithm != SSUAlgorithm::kVertical && algorithm != SSUAlgorithm::kHorizontal,
                   "vertical/horizontal MTP algorithm requires SM100+ (Blackwell); "
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
