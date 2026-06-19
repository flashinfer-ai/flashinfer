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

#include "kernel_impl.cuh"
#include "scheduler.cuh"
#include "utils.cuh"

namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled {

inline int get_num_sms() {
  int device = 0;
  CUTE_CHECK_ERROR(cudaGetDevice(&device));
  int num_sms = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  return num_sms;
}

// Build Arguments for a launch. For SwapAB KT, swap A↔B + SFA↔SFB and use
// (M↔N)-inverted strides so the kernel sees user-N as kernel-M.
template <typename KT, typename Kernel, typename L_T>
__forceinline__ typename Kernel::Arguments make_launch_args(
    typename KT::ElementA* ptr_A, typename KT::ElementB* ptr_B,
    typename KT::SFConfig::ElementSFLoad* ptr_SFA, typename KT::SFConfig::ElementSFLoad* ptr_SFB,
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

// MoE GEMM (PsumLayout / ZeroPadding): per-expert padded A + grouped_layout int32 cumsum.
//   PsumLayout       : grouped_layout = [E]    cumsum aligned_m WITHOUT leading 0
//   ZeroPadding      : grouped_layout = [E+1]  cumsum actual_m  WITH    leading 0
template <typename KT>
void launch_moe_gemm(typename KT::ElementA* ptr_A, typename KT::ElementB* ptr_B,
                     typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                     typename KT::SFConfig::ElementSFLoad* ptr_SFB, typename KT::ElementD* ptr_D,
                     int M_padded, int N, int K, int num_experts, int32_t const* grouped_layout,
                     int num_sms, cudaStream_t stream = 0) {
  using Kernel = SM120BlockScaledGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M_padded, N, K,
                                           grouped_layout);
  auto problem_shape = make_shape(M_padded, N, K, num_experts);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

// Normal GEMM (kFlat, GemmType::Normal): single problem (M, N, K).
template <typename KT>
void launch_gemm(typename KT::ElementA* ptr_A, typename KT::ElementB* ptr_B,
                 typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                 typename KT::SFConfig::ElementSFLoad* ptr_SFB, typename KT::ElementD* ptr_D, int M,
                 int N, int K, int num_sms, cudaStream_t stream = 0) {
  using Kernel = SM120BlockScaledGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, N, K,
                                           /*grouped_layout=*/nullptr);
  auto problem_shape = make_shape(M, N, K, 1);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

// Batched GEMM (kFlat, GemmType::Batched): L same-shape contiguous batches.
template <typename KT>
void launch_bmm(typename KT::ElementA* ptr_A, typename KT::ElementB* ptr_B,
                typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                typename KT::SFConfig::ElementSFLoad* ptr_SFB, typename KT::ElementD* ptr_D, int M,
                int N, int K, int L, int num_sms, cudaStream_t stream = 0) {
  using Kernel = SM120BlockScaledGemmKernel<KT>;
  auto args = make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, M, N, K,
                                           /*grouped_layout=*/nullptr);
  auto problem_shape = make_shape(M, N, K, L);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

// Masked GEMM (flat, GemmType::MGroupedMasked): per-batch padded (max_m, N, K)
// with masked_m[b] = actual_m (NOT cumulative).
template <typename KT>
void launch_masked_gemm(typename KT::ElementA* ptr_A, typename KT::ElementB* ptr_B,
                        typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                        typename KT::SFConfig::ElementSFLoad* ptr_SFB, typename KT::ElementD* ptr_D,
                        int max_m, int N, int K, int num_groups, int32_t const* masked_m,
                        int num_sms, cudaStream_t stream = 0) {
  using Kernel = SM120BlockScaledGemmKernel<KT>;
  auto args =
      make_launch_args<KT, Kernel>(ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_D, max_m, N, K, masked_m);
  auto problem_shape = make_shape(max_m, N, K, num_groups);
  launch_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_sms, stream);
}

}  // namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled
