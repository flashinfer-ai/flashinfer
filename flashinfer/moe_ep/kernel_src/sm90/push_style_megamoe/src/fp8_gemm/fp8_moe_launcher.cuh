/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh>
#include <tuple>

#include "fp8_moe_jit.cuh"
#include "fp8_moe_scheduler.cuh"

namespace flashinfer::sm90_push_fp8 {

inline int fc1_fused_smem_size(int num_stages, int shape_k, int block_m, int block_n,
                               int block_k = 128) {
  int const smem_act_bf16 = block_m * block_n * 2;
  int const smem_out_fp8 = block_m * block_n;
  int const smem_a_per_stage = block_m * block_k;
  int const smem_scales_a_per_stage = block_m * 4;
  int const smem_b_per_stage = block_n * block_k;
  int const smem_scales_b = 2 * deep_gemm::jit::div_up(shape_k, block_k) * 4;
  int const smem_barriers = num_stages * 8 * 2;
  return smem_act_bf16 + smem_out_fp8 +
         num_stages * (smem_a_per_stage + smem_scales_a_per_stage + smem_b_per_stage) +
         deep_gemm::jit::div_up(smem_scales_b, 8) * 8 + smem_barriers;
}

inline std::tuple<int, int, int> fc1_fused_config(uint32_t expected_m, uint32_t shape_k) {
  constexpr int block_n = 128;
  constexpr int sm90_smem_capacity = 232448;
  int const block_m = expected_m <= 64 ? 64 : 128;
  for (int num_stages : {8, 7, 6, 5, 4}) {
    int const smem_size = fc1_fused_smem_size(num_stages, shape_k, block_m, block_n);
    if (smem_size <= sm90_smem_capacity) return {block_m, num_stages, smem_size};
  }
  return {block_m, 4, fc1_fused_smem_size(4, shape_k, block_m, block_n)};
}

inline bool fp8_moe_gemm_jit_cache_ready(int64_t expected_m, int shape_n, int shape_k,
                                         int num_problems) {
  using namespace tensorrt_llm::kernels::fp8_blockscale_gemm;
  if (!getDeepGemmEnabled()) return true;
  if (kNumDeviceSMs < 0) kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();

  constexpr uint32_t block_k = 128;
  uint32_t const m_per_expert_threshold = kNumDeviceSMs == 78 ? 64 : 32;
  bool const swap_ab = expected_m < m_per_expert_threshold;
  auto const [block_m, block_n, num_stages, num_tma_multicast, smem_size] =
      swap_ab ? deep_gemm::jit::get_best_gemm_config(
                    static_cast<uint32_t>(shape_n), static_cast<uint32_t>(expected_m),
                    static_cast<uint32_t>(shape_k), num_problems, kNumDeviceSMs, false, true)
              : deep_gemm::jit::get_best_gemm_config(
                    static_cast<uint32_t>(expected_m), static_cast<uint32_t>(shape_n),
                    static_cast<uint32_t>(shape_k), num_problems, kNumDeviceSMs);
  static_cast<void>(smem_size);
  return jit::grouped_kernel_cache_ready(
      static_cast<uint32_t>(shape_n), static_cast<uint32_t>(shape_k),
      static_cast<uint32_t>(block_m), static_cast<uint32_t>(block_n), block_k,
      static_cast<uint32_t>(num_problems), static_cast<uint32_t>(num_stages),
      static_cast<uint32_t>(num_tma_multicast), swap_ab);
}

inline bool fp8_moe_fc1_fused_jit_cache_ready(int64_t expected_m, int shape_n_interleaved,
                                              int shape_k, int num_problems) {
  constexpr uint32_t block_n = 128;
  constexpr uint32_t block_k = 128;
  auto const [block_m, num_stages, smem_size] =
      fc1_fused_config(static_cast<uint32_t>(expected_m), static_cast<uint32_t>(shape_k));
  static_cast<void>(smem_size);
  return jit::get_fc1_compiler().is_cached(
      static_cast<uint32_t>(shape_n_interleaved), static_cast<uint32_t>(shape_k),
      static_cast<uint32_t>(block_m), block_n, block_k, static_cast<uint32_t>(num_problems),
      static_cast<uint32_t>(num_stages));
}

template <typename LayoutIndexType>
inline void launch_fc1_fused(cudaKernel_t kernel, void* mat_a, void* mat_b, void* mat_d_fp8,
                             int64_t d_rows, float* sfa_out, float* scales_a, float* scales_b,
                             uint32_t shape_m, uint32_t shape_n_interleaved, uint32_t shape_k,
                             uint32_t block_m, uint32_t block_n, uint32_t block_k,
                             uint32_t num_groups, LayoutIndexType* problem_m_offsets,
                             cudaStream_t stream, int num_sms, uint32_t smem_size,
                             uint32_t max_shape_m_padded) {
  auto tma_a_desc = deep_gemm::make_2d_tma_a_desc(reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m,
                                                  shape_k, block_m, block_k, num_groups,
                                                  deep_gemm::GemmType::GroupedWithOffset);
  auto tma_b_desc = deep_gemm::make_2d_tma_b_desc(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n_interleaved, shape_k, block_n, block_k,
      num_groups, deep_gemm::GemmType::GroupedWithOffset);
  auto tma_scales_a_desc = deep_gemm::make_tma_scales_a_offset_desc(scales_a, max_shape_m_padded,
                                                                    shape_k, block_m, block_k);

  constexpr uint32_t kNumTMAThreads = 128;
  constexpr uint32_t kNumMathThreadsPerGroup = 128;
  DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size) == cudaSuccess);

  cudaLaunchConfig_t config{};
  config.gridDim = num_sms;
  config.blockDim = deep_gemm::get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(
      static_cast<int32_t>(block_m));
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, 1, 1};
  config.attrs = &attr;
  config.numAttrs = 1;

  Fp8MoeSchedulerInput input{};
  input.shape_m = shape_m;
  input.problem_m_offsets = problem_m_offsets;
  auto const status =
      cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_fp8_e4m3*>(mat_d_fp8), d_rows,
                         sfa_out, static_cast<int64_t>(max_shape_m_padded), scales_b, input,
                         tma_a_desc, tma_b_desc, tma_scales_a_desc);
  DG_HOST_ASSERT(status == cudaSuccess);
}

inline void fp8_moe_gemm(__nv_fp8_e4m3* mat_a, float* scales_a, __nv_fp8_e4m3* mat_b,
                         float* scales_b, __nv_bfloat16* mat_d, int64_t const* problem_m_offsets,
                         int num_problems, int64_t expected_m, int64_t max_shape_m,
                         int64_t max_shape_m_padded, int shape_n, int shape_k,
                         cudaStream_t stream) {
  using namespace tensorrt_llm::kernels::fp8_blockscale_gemm;
  fp8_grouped_gemm_run(nullptr, mat_a, scales_a, nullptr, mat_b, scales_b, mat_d, problem_m_offsets,
                       num_problems, expected_m, max_shape_m, max_shape_m_padded, shape_n, shape_k,
                       stream, false, false);
}

inline void fp8_moe_fc1_fused(__nv_fp8_e4m3* mat_a, float* scales_a, __nv_fp8_e4m3* mat_b,
                              float* scales_b, __nv_fp8_e4m3* mat_d_fp8, int64_t d_rows,
                              float* sfa_out, int64_t const* problem_m_offsets, int num_problems,
                              int64_t expected_m, int64_t max_shape_m, int64_t max_shape_m_padded,
                              int shape_n_interleaved, int shape_k, cudaStream_t stream) {
  using namespace tensorrt_llm::kernels::fp8_blockscale_gemm;
  if (kNumDeviceSMs < 0) kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
  TLLM_CHECK_WITH_INFO(getDeepGemmEnabled(),
                       "moe_gemm_fc1_fused requires the SM90 DeepGEMM JIT path");
  TLLM_CHECK_WITH_INFO(shape_n_interleaved % 256 == 0,
                       "moe_gemm_fc1_fused requires interleaved N divisible by 256");
  TLLM_CHECK_WITH_INFO(shape_k % 128 == 0, "moe_gemm_fc1_fused requires K divisible by 128");

  constexpr uint32_t block_n = 128;
  constexpr uint32_t block_k = 128;
  auto const [block_m, num_stages, smem_size] =
      fc1_fused_config(static_cast<uint32_t>(expected_m), static_cast<uint32_t>(shape_k));
  auto* runtime = jit::get_fc1_compiler().build(
      static_cast<uint32_t>(shape_n_interleaved), static_cast<uint32_t>(shape_k),
      static_cast<uint32_t>(block_m), block_n, block_k, static_cast<uint32_t>(num_problems),
      static_cast<uint32_t>(num_stages));
  auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
  launch_fc1_fused(kernel, mat_a, mat_b, mat_d_fp8, d_rows, sfa_out, scales_a, scales_b,
                   static_cast<uint32_t>(max_shape_m), static_cast<uint32_t>(shape_n_interleaved),
                   static_cast<uint32_t>(shape_k), static_cast<uint32_t>(block_m), block_n, block_k,
                   static_cast<uint32_t>(num_problems), const_cast<int64_t*>(problem_m_offsets),
                   stream, kNumDeviceSMs, static_cast<uint32_t>(smem_size),
                   static_cast<uint32_t>(max_shape_m_padded));
}

}  // namespace flashinfer::sm90_push_fp8
