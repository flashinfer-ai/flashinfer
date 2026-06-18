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
#include "../compute/epilogue/output_writer.cuh"
#include "../compute/producer/load_k.cuh"
#include "../compute/producer/load_q.cuh"
#include "../compute/producer/load_v.cuh"
#include "../kernel/attention_kernel.h"
#include "../kernel/scheduler.h"
#include "../kernel/traits.h"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "flashinfer/utils.cuh"

namespace nvfp4_attention {

template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using ElementSF = typename Kernel_traits::ElementSF;
  using ElementOut = typename Kernel_traits::ElementOut;
  using ElementDS = typename Kernel_traits::ElementDS;
  using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

  using CollectiveMainloop = nvfp4_attention::CollectiveMainloopFwd<Kernel_traits, Is_causal>;
  using CollectiveEpilogue = nvfp4_attention::CollectiveEpilogueFwd<Kernel_traits>;

  using Scheduler = nvfp4_attention::StaticPersistentTileScheduler;

  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {static_cast<Element const*>(params.q_ptr),
       {params.seqlen_q, params.d, params.h, params.b},
       {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},

       static_cast<Element const*>(params.k_ptr),
       {params.seqlen_k, params.d, params.h_k, params.b},
       {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},
       {params.unpadded_seqlen_k, params.d, params.h_k, params.b},

       static_cast<Element const*>(params.v_ptr),
       {params.d, params.seqlen_k, params.h_k, params.b},
       {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},

       static_cast<ElementSF const*>(params.sfq_ptr),
       {params.seqlen_q, params.d, params.h, params.b},
       static_cast<ElementSF const*>(params.sfk_ptr),
       {params.seqlen_k, params.d, params.h_k, params.b},
       static_cast<ElementSF const*>(params.sfv_ptr),
       {params.d, params.seqlen_k, params.h_k, params.b},

       static_cast<ElementDS const*>(params.delta_s_ptr),
       {params.seqlen_s, params.seqlen_k, params.h_k, params.b},
       {params.ds_row_stride, _1{}, params.ds_head_stride, params.ds_batch_stride},

       params.scale_softmax_log2});

  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({

          static_cast<ElementOut*>(params.o_ptr),
          {params.seqlen_q, params.d, params.h, params.b},
          {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},

          static_cast<float*>(params.softmax_lse_ptr),
          {_1{}, params.seqlen_q, params.h * params.seqlen_q},
      });

  int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
  num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});

  typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b, nullptr,
                                                  params.is_causal};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  void* kernel = (void*)nvfp4_attention::attention_kernel_ws<Kernel_traits, Is_causal, Scheduler>;

  int smem_size = sizeof(typename Kernel_traits::SharedStorage);
  if (smem_size >= 48 * 1024) {
    FLASHINFER_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  static constexpr int ctaSize = Kernel_traits::kNWarps * 32;

  params.m_block_divmod = cutlass::FastDivmod(num_blocks_m);
  params.total_blocks = num_blocks_m * params.h * params.b;

  int device_id = 0;
  FLASHINFER_CUDA_CHECK(cudaGetDevice(&device_id));
  int num_sms = 0;
  FLASHINFER_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, num_sms);
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));

  cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size,
                                             stream};

  cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params,
                                    scheduler_params);

  FLASHINFER_CUDA_CHECK(cudaGetLastError());
}

template <typename T, int Headdim, typename O = cutlass::bfloat16_t>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream) {
  using DeltaSType = float;

  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(params.per_block_mean, per_block, [&] {
      if constexpr (Headdim == 64 || Headdim == 128) {
        static constexpr int kStages = 3;
        static constexpr int kBlockN = 128;
        run_flash_fwd<
            Flash_fwd_kernel_traits<Headdim, 128, kBlockN, kStages, 1, per_block, T, O, DeltaSType>,
            Is_causal>(params, stream);
      } else {
        static_assert(Headdim == 64 || Headdim == 128, "Unsupported Headdim");
      }
    });
  });
}

}  // namespace nvfp4_attention
