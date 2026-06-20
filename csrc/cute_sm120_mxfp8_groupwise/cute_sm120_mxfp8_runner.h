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
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cute/numeric/numeric_types.hpp>
#include <vector>

namespace flashinfer::gemm::mxfp8_cute_sm120 {

class CuteSm120Mxfp8GemmRunnerInterface {
 public:
  CuteSm120Mxfp8GemmRunnerInterface() {}

  virtual ~CuteSm120Mxfp8GemmRunnerInterface() {}

  virtual void moe_gemm_mxfp8_nt_groupwise(void* D, void const* A, void const* B,
                                           int32_t const* token_offset, int num_groups,
                                           int max_shape_m, int shape_n, int shape_k,
                                           cudaStream_t stream, int32_t const* SFA = nullptr,
                                           int32_t const* SFB = nullptr, int granK = 128) = 0;
};

template <typename ElementType, typename OutElementType, typename AccumElementType,
          typename BlockScaleElementType>
class CuteSm120Mxfp8GemmRunner : public CuteSm120Mxfp8GemmRunnerInterface {
 public:
  CuteSm120Mxfp8GemmRunner();
  ~CuteSm120Mxfp8GemmRunner();

  void moe_gemm_mxfp8_nt_groupwise(void* D, void const* A, void const* B,
                                   int32_t const* token_offset, int num_groups, int max_shape_m,
                                   int shape_n, int shape_k, cudaStream_t stream,
                                   int32_t const* SFA = nullptr, int32_t const* SFB = nullptr,
                                   int granK = 128) override;

 private:
  template <int GranK>
  void moe_gemm_mxfp8_nt_groupwise_impl(void* D, void const* A, void const* B,
                                        int32_t const* token_offset, int num_groups,
                                        int max_shape_m, int shape_n, int shape_k,
                                        cudaStream_t stream, int32_t const* SFA,
                                        int32_t const* SFB);
};

}  // namespace flashinfer::gemm::mxfp8_cute_sm120
