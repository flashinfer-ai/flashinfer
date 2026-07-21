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
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cute/numeric/numeric_types.hpp>
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {

class CuteSm120Fp8GemmRunnerInterface {
 public:
  CuteSm120Fp8GemmRunnerInterface() {}

  virtual ~CuteSm120Fp8GemmRunnerInterface() {}

  virtual void gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int shape_m,
                                     int shape_n, int shape_k, float const* scales_a,
                                     float const* scales_b, cudaStream_t stream,
                                     int scale_granularity_m = 1, int scale_granularity_n = 128,
                                     int scale_granularity_k = 128) = 0;

  virtual void batch_gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int num_groups,
                                           int shape_m, int shape_n, int shape_k,
                                           float const* scales_a, float const* scales_b,
                                           cudaStream_t stream, int scale_granularity_m = 1,
                                           int scale_granularity_n = 128,
                                           int scale_granularity_k = 128) = 0;

  virtual void group_gemm_fp8_nt_groupwise_masked(void* D, void const* A, void const* B,
                                                  int32_t const* masked_m, int num_groups,
                                                  int max_m, int n, int k, cudaStream_t stream,
                                                  float const* SFA, float const* SFB,
                                                  int scale_granularity_m = 1,
                                                  int scale_granularity_n = 128,
                                                  int scale_granularity_k = 128) = 0;

  virtual void moe_gemm_fp8_nt_groupwise(void* D, void const* A, void const* B,
                                         int32_t const* token_offset, int num_groups,
                                         int max_shape_m, int shape_n, int shape_k,
                                         cudaStream_t stream, float const* SFA, float const* SFB,
                                         int scale_granularity_m = 1, int scale_granularity_n = 128,
                                         int scale_granularity_k = 128) = 0;

  virtual void group_gemm_fp8_nt_groupwise_contiguous(
      void* D, void const* A, void const* B, int32_t const* m_indices, int num_groups, int m,
      int shape_n, int shape_k, cudaStream_t stream, float const* SFA, float const* SFB,
      int scale_granularity_m = 1, int scale_granularity_n = 128, int scale_granularity_k = 128,
      bool use_psum_layout = true) = 0;
};

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
class CuteSm120Fp8GemmRunner : public CuteSm120Fp8GemmRunnerInterface {
 public:
  CuteSm120Fp8GemmRunner();
  ~CuteSm120Fp8GemmRunner();

  void gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int shape_m, int shape_n,
                             int shape_k, float const* scales_a, float const* scales_b,
                             cudaStream_t stream, int scale_granularity_m = 1,
                             int scale_granularity_n = 128, int scale_granularity_k = 128) override;

  void batch_gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int num_groups,
                                   int shape_m, int shape_n, int shape_k, float const* scales_a,
                                   float const* scales_b, cudaStream_t stream,
                                   int scale_granularity_m = 1, int scale_granularity_n = 128,
                                   int scale_granularity_k = 128) override;

  void group_gemm_fp8_nt_groupwise_masked(void* D, void const* A, void const* B,
                                          int32_t const* masked_m, int num_groups, int max_m, int n,
                                          int k, cudaStream_t stream, float const* SFA,
                                          float const* SFB, int scale_granularity_m = 1,
                                          int scale_granularity_n = 128,
                                          int scale_granularity_k = 128) override;

  void moe_gemm_fp8_nt_groupwise(void* D, void const* A, void const* B, int32_t const* token_offset,
                                 int num_groups, int max_shape_m, int shape_n, int shape_k,
                                 cudaStream_t stream, float const* SFA, float const* SFB,
                                 int scale_granularity_m = 1, int scale_granularity_n = 128,
                                 int scale_granularity_k = 128) override;

  void group_gemm_fp8_nt_groupwise_contiguous(
      void* D, void const* A, void const* B, int32_t const* m_indices, int num_groups, int m,
      int shape_n, int shape_k, cudaStream_t stream, float const* SFA, float const* SFB,
      int scale_granularity_m = 1, int scale_granularity_n = 128, int scale_granularity_k = 128,
      bool use_psum_layout = true) override;

 private:
  void gemm_fp8_nt_groupwise_impl(void* D, void const* A, void const* B, int shape_m, int shape_n,
                                  int shape_k, float const* SFA, float const* SFB,
                                  cudaStream_t stream);

  void batch_gemm_fp8_nt_groupwise_impl(void* D, void const* A, void const* B, int num_groups,
                                        int shape_m, int shape_n, int shape_k, float const* SFA,
                                        float const* SFB, cudaStream_t stream);

  void group_gemm_fp8_nt_groupwise_masked_impl(void* D, void const* A, void const* B,
                                               int32_t const* masked_m, int num_groups, int max_m,
                                               int n, int k, cudaStream_t stream, float const* SFA,
                                               float const* SFB);

  void moe_gemm_fp8_nt_groupwise_impl(void* D, void const* A, void const* B,
                                      int32_t const* token_offset, int num_groups, int max_shape_m,
                                      int shape_n, int shape_k, cudaStream_t stream,
                                      float const* SFA, float const* SFB);

  void group_gemm_fp8_nt_groupwise_contiguous_impl(void* D, void const* A, void const* B,
                                                   int32_t const* m_indices, int num_groups, int m,
                                                   int shape_n, int shape_k, cudaStream_t stream,
                                                   float const* SFA, float const* SFB,
                                                   bool use_psum_layout);
};

}  // namespace flashinfer::gemm::mxfp8_cute_sm120
