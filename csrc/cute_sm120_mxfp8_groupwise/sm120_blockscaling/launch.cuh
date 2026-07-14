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
#include <cuda_runtime.h>

#include <cute/config.hpp>

#include <cutlass/device_kernel.h>

#include "kernel_impl.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaling {

inline int get_num_sms() {
  int device;
  cudaGetDevice(&device);
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  return num_sms;
}

template <typename KT, typename Kernel, typename L_T>
__forceinline__ typename Kernel::Arguments make_launch_args(
    typename KT::ElementA const* ptr_A, typename KT::ElementB const* ptr_B,
    typename KT::ElementScale const* ptr_SFA, typename KT::ElementScale const* ptr_SFB,
    typename KT::ElementD* ptr_D, int M, int N, int K, L_T grouped_layout) {
  if constexpr (KT::kSwapAB) {
    return typename Kernel::Arguments{
        ptr_B,   typename KT::ABLoadConfig::StrideA{int64_t(K), Int<1>{}, int64_t(N) * K},
        ptr_A,   typename KT::ABLoadConfig::StrideB{int64_t(K), Int<1>{}, int64_t(M) * K},
        ptr_SFB, ptr_SFA,
        ptr_D,   grouped_layout};
  } else {
    return typename Kernel::Arguments{
        ptr_A,   typename KT::ABLoadConfig::StrideA{int64_t(K), Int<1>{}, int64_t(M) * K},
        ptr_B,   typename KT::ABLoadConfig::StrideB{int64_t(K), Int<1>{}, int64_t(N) * K},
        ptr_SFA, ptr_SFB,
        ptr_D,   grouped_layout};
  }
}

template <typename Kernel>
__forceinline__ void launch_kernel(typename Kernel::Params const& params, int num_sms,
                                   cudaStream_t stream) {
  cudaLaunchConfig_t launch_config;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;

  launch_config.gridDim = Kernel::get_grid_shape(num_sms);
  launch_config.blockDim = Kernel::get_block_shape();
  launch_config.dynamicSmemBytes = Kernel::kSmemSize;
  launch_config.stream = stream;
  launch_config.attrs = attrs;
  launch_config.numAttrs = 1;

  auto kernel_ptr = &cutlass::device_kernel<Kernel>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Kernel::kSmemSize));
  CUTE_CHECK_ERROR(cudaLaunchKernelEx(&launch_config, kernel_ptr, params));
  CUTE_CHECK_ERROR(cudaGetLastError());
}

template <typename KT>
void launch_moe_gemm(typename KT::ElementA const* ptr_A, typename KT::ElementB const* ptr_B,
                     typename KT::ElementScale const* ptr_SFA,
                     typename KT::ElementScale const* ptr_SFB, typename KT::ElementD* ptr_D, int M,
                     int N, int K, int num_experts, int32_t const* grouped_layout, int num_sms,
                     cudaStream_t stream = 0) {
  using Kernel = SM120BlockScalingGemmKernel<KT>;
  auto args =
      make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, N, K, grouped_layout);
  auto problem_shape = make_shape(M, N, K, num_experts);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

template <typename KT>
void launch_gemm(typename KT::ElementA const* ptr_A, typename KT::ElementB const* ptr_B,
                 typename KT::ElementScale const* ptr_SFA, typename KT::ElementScale const* ptr_SFB,
                 typename KT::ElementD* ptr_D, int M, int N, int K, int num_sms,
                 cudaStream_t stream = 0) {
  using Kernel = SM120BlockScalingGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, N, K,
                                           /*grouped_layout=*/nullptr);
  auto problem_shape = make_shape(M, N, K, 1);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

template <typename KT>
void launch_bmm(typename KT::ElementA const* ptr_A, typename KT::ElementB const* ptr_B,
                typename KT::ElementScale const* ptr_SFA, typename KT::ElementScale const* ptr_SFB,
                typename KT::ElementD* ptr_D, int M, int N, int K, int L, int num_sms,
                cudaStream_t stream = 0) {
  using Kernel = SM120BlockScalingGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, N, K,
                                           /*grouped_layout=*/nullptr);
  auto problem_shape = make_shape(M, N, K, L);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

template <typename KT>
void launch_masked_gemm(typename KT::ElementA const* ptr_A, typename KT::ElementB const* ptr_B,
                        typename KT::ElementScale const* ptr_SFA,
                        typename KT::ElementScale const* ptr_SFB, typename KT::ElementD* ptr_D,
                        int max_m, int N, int K, int num_groups, int32_t const* masked_m,
                        int num_sms, cudaStream_t stream = 0) {
  using Kernel = SM120BlockScalingGemmKernel<KT>;
  auto args =
      make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, max_m, N, K, masked_m);
  auto problem_shape = make_shape(max_m, N, K, num_groups);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

}  // namespace sm120_blockscaling
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
