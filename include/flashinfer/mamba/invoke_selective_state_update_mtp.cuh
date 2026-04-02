#ifndef FLASHINFER_MAMBA_INVOKE_SELECTIVE_STATE_UPDATE_MTP_CUH_
#define FLASHINFER_MAMBA_INVOKE_SELECTIVE_STATE_UPDATE_MTP_CUH_

#include <cuda_runtime_api.h>

#include <algorithm>
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
                       algorithm == SSUAlgorithm::kHorizontal ||
                       algorithm == SSUAlgorithm::kAsyncHorizontal,
                   "MTP selective_state_update only supports 'auto', 'simple', 'vertical', "
                   "'horizontal', or 'async_horizontal' algorithm, got ",
                   static_cast<int32_t>(algorithm));
  // kAsyncHorizontal is now merged into kSimple
  if (algorithm == SSUAlgorithm::kAsyncHorizontal) {
    algorithm = SSUAlgorithm::kSimple;
  }
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
    FLASHINFER_CHECK(params.cu_seqlens == nullptr,
                     "vertical algorithm does not support varlen (cu_seqlens)");

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
    FLASHINFER_CHECK(params.cu_seqlens == nullptr,
                     "horizontal algorithm does not support varlen (cu_seqlens)");

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

          // TMA state descriptor: single wide tile of DSTATE_PAD columns.
          // DSTATE_PAD is DSTATE rounded up to 128 bytes (32 banks), eliminating
          // bank conflicts. OOB padding is handled in registers, not smem.
          constexpr int DSTATE_PAD = sram_t::DSTATE_PAD;
          auto state_tensor = tma::buildNdDescriptor(
              typeid(state_t),
              /*shapes*/ {DSTATE, DIM, params.nheads, params.state_cache_size},
              /*strides*/ {1, DSTATE, DSTATE * DIM, params.state_stride_batch},
              /*tiles*/ {DSTATE_PAD, TMA_STATE_ROWS, 1, 1}, params.state);

          // B/C: tile by DSTATE_PAD to match padded smem layout.
          auto B_tensor = tma::buildNdDescriptor(
              typeid(input_t),
              {(uint64_t)DSTATE, (uint64_t)params.ngroups, (uint64_t)params.ntokens_mtp,
               (uint64_t)params.batch},
              {1, (uint64_t)DSTATE, (uint64_t)params.B_stride_mtp, (uint64_t)params.B_stride_batch},
              {DSTATE_PAD, 1, NTOKENS_MTP, 1}, params.B);

          auto C_tensor = tma::buildNdDescriptor(
              typeid(input_t),
              {(uint64_t)DSTATE, (uint64_t)params.ngroups, (uint64_t)params.ntokens_mtp,
               (uint64_t)params.batch},
              {1, (uint64_t)DSTATE, (uint64_t)params.C_stride_mtp, (uint64_t)params.C_stride_batch},
              {DSTATE_PAD, 1, NTOKENS_MTP, 1}, params.C);

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

  // ── Simple MTP kernel (SM80+, cp.async, no TMA) ─────────────────────────
  {
    constexpr int NUM_WARPS = 4;
    constexpr int kRowsPerPass = NUM_WARPS * simple_horiz::ROWS_PER_WARP;

    FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                     ") must be divisible by ngroups (", params.ngroups, ") for simple algorithm");
    // Determine CTAS_PER_HEAD: split DIM across grid.z for more parallelism at small batch
    int const total_tiles = params.batch * params.nheads;
    int const num_sms = GetCudaMultiProcessorCount();

    // Pick CTAS_PER_HEAD to saturate the GPU: ratio = target_ctas / total_tiles,
    // clamped to [1, max_ctas]. DIM_PER_CTA must be >= ROWS_PER_PASS.
    // With 128 threads and 48 regs/thread, registers limit to 10 blocks/SM.
    constexpr int kBlocksPerSM = 10;
    constexpr int kMaxCtas = DIM / kRowsPerPass;
    int const target_ctas = num_sms * kBlocksPerSM;
    int const ctas_per_head = std::clamp(target_ctas / max(total_tiles, 1), 1, kMaxCtas);

    auto launch = [&]<int CTAS_PER_HEAD>() {
      constexpr int DIM_PER_CTA = DIM / CTAS_PER_HEAD;
      static_assert(DIM % CTAS_PER_HEAD == 0);
      static_assert(DIM_PER_CTA % kRowsPerPass == 0);

      dispatchRatio(
          params, std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64>{}, [&]<int HEADS_PER_GROUP>() {
            constexpr int DSTATE_PAD = padDstate<input_t>(DSTATE);
            constexpr int kRowsPerPassLocal = NUM_WARPS * simple_horiz::ROWS_PER_WARP;
            constexpr int kNumPasses = DIM_PER_CTA / kRowsPerPassLocal;
            constexpr int kStateStages = (kNumPasses == 1) ? 1 : 2;
            using sram_t = SimpleStorage<input_t, state_t, NTOKENS_MTP, DIM_PER_CTA, DSTATE_PAD,
                                         kRowsPerPassLocal, kStateStages>;
            constexpr size_t smem_size = sizeof(sram_t);

            auto func = selective_state_update_kernel_simple_mtp<
                input_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t, NTOKENS_MTP,
                DIM, DSTATE, HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, CTAS_PER_HEAD>;

            dim3 grid(params.batch, params.nheads, CTAS_PER_HEAD);
            dim3 block(warpSize, NUM_WARPS);

            FLASHINFER_CUDA_CHECK(
                cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            func<<<grid, block, smem_size, stream>>>(params);
          });
    };

    // Dispatch to the largest instantiated CTAS_PER_HEAD <= ctas_per_head.
    // Use if constexpr to avoid compiling invalid template instantiations.
    if constexpr (DIM / 4 >= kRowsPerPass) {
      if (ctas_per_head >= 4) {
        launch.template operator()<4>();
        return;
      }
    }
    if constexpr (DIM / 2 >= kRowsPerPass) {
      if (ctas_per_head >= 2) {
        launch.template operator()<2>();
        return;
      }
    }
    launch.template operator()<1>();
  }
}

}  // namespace flashinfer::mamba::mtp

#endif  // FLASHINFER_MAMBA_INVOKE_SELECTIVE_STATE_UPDATE_MTP_CUH_
