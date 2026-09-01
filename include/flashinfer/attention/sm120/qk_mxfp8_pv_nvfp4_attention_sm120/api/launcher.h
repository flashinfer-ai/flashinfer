/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include "../common/params.h"
#include "../common/static_switch.h"
#include "../compute/epilogue/lse_writer.cuh"
#include "../kernel/attention_kernel.h"
#include "../kernel/scheduler.h"
#include "../kernel/traits.h"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "flashinfer/utils.cuh"

namespace qk_mxfp8_pv_nvfp4_attention {

inline constexpr int kStagesNonCausal = 2;
inline constexpr int kStagesCausal = 1;
inline constexpr int kPersistentGridMultiplierNonCausal = 2;
inline constexpr int kPersistentGridMultiplierCausal = 4;
inline constexpr int kSingleTileSchedulerMBlocks = 32;

/**
 * Kernel Launcher
 *
 *
 */
template <typename Kernel_traits, bool Is_causal, bool ReturnLSE, typename Scheduler>
void run_flash_fwd_with_scheduler(Flash_fwd_params& params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using ElementSF = typename Kernel_traits::ElementSF;
  using ElementPV = typename Kernel_traits::ElementPV;
  using ElementSFPV = typename Kernel_traits::ElementSFPV;  // V SF: UE4M3 (PV path)
  using ElementOut = typename Kernel_traits::ElementOut;
  using ElementDS = typename Kernel_traits::ElementDS;
  using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

  using CollectiveMainloop =
      qk_mxfp8_pv_nvfp4_attention::CollectiveMainloopFwd<Kernel_traits, Is_causal>;
  using CollectiveEpilogue = qk_mxfp8_pv_nvfp4_attention::CollectiveEpilogueFwd<Kernel_traits>;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {// Q tensor
       static_cast<Element const*>(params.q_ptr),
       {params.seqlen_q, params.d, params.h, params.b},  // shape_Q
       {params.unpadded_seqlen_q, params.d, params.h, params.b},
       {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q

       // K tensor
       static_cast<Element const*>(params.k_ptr),
       {params.seqlen_k, params.d, params.h_k, params.b},                         // shape_K
       {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
       {params.unpadded_seqlen_k, params.d, params.h_k, params.b},  // shape_K (unpadded)

       // V tensor (transposed, FP4 packed)
       static_cast<ElementPV const*>(params.v_ptr),
       {params.d, params.seqlen_k, params.h_k, params.b},                         // shape_Vt
       {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_Vt

       // Scale factors
       static_cast<ElementSF const*>(params.sfq_ptr),
       {params.seqlen_q, params.d, params.h, params.b},  // shape_SFQ
       static_cast<ElementSF const*>(params.sfk_ptr),
       {params.seqlen_k, params.d, params.h_k, params.b},  // shape_SFK
       static_cast<ElementSFPV const*>(params.sfv_ptr),    // V SF: UE4M3 (PV path)
       {params.d, params.seqlen_k, params.h_k, params.b},  // shape_SFVt

       // Delta_s correction
       cutlass::FastDivmod(params.h_h_k_ratio),
       // Softmax scale
       params.scale_softmax_log2});

  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          // O tensor
          static_cast<ElementOut*>(params.o_ptr),
          {params.seqlen_q, params.d, params.h, params.b},                           // shape_O
          {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},  // stride_O

          // LSE (LogSumExp) tensor
          static_cast<float*>(params.softmax_lse_ptr),
          {_1{}, params.seqlen_q, params.h * params.seqlen_q},  // stride_LSE
      });

  int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
  num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});

  typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  void* kernel = (void*)qk_mxfp8_pv_nvfp4_attention::attention_kernel_ws<Kernel_traits, Is_causal,
                                                                         ReturnLSE, Scheduler>;

  int smem_size = sizeof(typename Kernel_traits::SharedStorage);
  if (smem_size >= 48 * 1024) {
    static bool const smem_attr_set = [&]() {
      FLASHINFER_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      return true;
    }();
    (void)smem_attr_set;
  }

  static constexpr int ctaSize = Kernel_traits::kNWarps * 32;

  params.m_block_divmod = cutlass::FastDivmod(num_blocks_m);
  params.total_blocks = num_blocks_m * params.h * params.b;

  int device_id = 0;
  FLASHINFER_CUDA_CHECK(cudaGetDevice(&device_id));
  int num_sms = 0;
  FLASHINFER_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
  int scheduler_sms = num_sms;
  if (params.total_blocks <= num_sms * 8) {
    scheduler_sms *=
        Is_causal ? kPersistentGridMultiplierCausal : kPersistentGridMultiplierNonCausal;
  }
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, scheduler_sms);
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));

  cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size,
                                             stream};

  cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params,
                                    scheduler_params);

  FLASHINFER_CUDA_CHECK(cudaGetLastError());
}

template <typename Kernel_traits, bool Is_causal, bool ReturnLSE>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;
  int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
  num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
  const bool use_single_tile = num_blocks_m >= kSingleTileSchedulerMBlocks;

  if (use_single_tile) {
    run_flash_fwd_with_scheduler<Kernel_traits, Is_causal, ReturnLSE,
                                 qk_mxfp8_pv_nvfp4_attention::SingleTileScheduler>(params, stream);
  } else {
    run_flash_fwd_with_scheduler<Kernel_traits, Is_causal, ReturnLSE,
                                 qk_mxfp8_pv_nvfp4_attention::StaticPersistentTileScheduler>(
        params, stream);
  }
}

/**
 * MHA Forward Dispatcher
 *
 *
 * @tparam T: FP4 pair type
 * @tparam Headdim: Head dimension (64 or 128)
 * @tparam O: Output type (bfloat16 or float16)
 */
template <typename T, int Headdim, typename O = cutlass::bfloat16_t, typename DS = float,
          bool ReturnLSE = true>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream) {
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    static_assert(Headdim == 128, "Only head dimension 128 is supported");
    static constexpr int kStages = Is_causal ? kStagesCausal : kStagesNonCausal;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, kStages, 1, false, T, O, DS>,
                  Is_causal, ReturnLSE>(params, stream);
  });
}

/**
 *
 * - Headdim: Head dimension (64 or 128)
 * - kStages: 3 (Pipeline stages for K/V)
 * - T: FP4 pair type
 * - O: Output type (BF16/FP16)
 *
 * Scheduler:
 *
 * Shared Memory:
 *
 */

}  // namespace qk_mxfp8_pv_nvfp4_attention
