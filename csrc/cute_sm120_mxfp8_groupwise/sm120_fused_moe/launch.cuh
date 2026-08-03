/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// clang-format off
#include "cute_sm120_mxfp8_groupwise/sm120_fused_moe/fp8_kernel_impl.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_fused_moe/mxfp8_kernel_impl.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/launch.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaling/launch.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaled {

template <typename KT>
void launch_fused_moe(typename KT::ElementA* ptr_A, typename KT::ElementB* ptr_B,
                      typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                      typename KT::SFConfig::ElementSFLoad* ptr_SFB, typename KT::ElementD* ptr_D,
                      int M, int N, int K, int num_experts, int32_t const* grouped_layout,
                      int num_sms, cudaStream_t stream = 0) {
  using Kernel = SM120BlockScaledFusedMoeGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, 2 * N, K,
                                           grouped_layout);
  auto problem_shape = make_shape(M, N, K, num_experts);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

}  // namespace sm120_blockscaled

namespace sm120_blockscaling {

template <typename KT>
void launch_fused_moe(typename KT::ElementA const* ptr_A, typename KT::ElementB const* ptr_B,
                      typename KT::ElementScale const* ptr_SFA,
                      typename KT::ElementScale const* ptr_SFB, typename KT::ElementD* ptr_D, int M,
                      int N, int K, int num_experts, int32_t const* grouped_layout, int num_sms,
                      cudaStream_t stream = 0) {
  using Kernel = SM120BlockScalingFusedMoeGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, 2 * N, K,
                                           grouped_layout);
  auto problem_shape = make_shape(M, N, K, num_experts);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

}  // namespace sm120_blockscaling
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
