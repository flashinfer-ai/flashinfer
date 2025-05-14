/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/gemm/group_gemm_groupwise_sm100.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(input_dtype, output_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                                  \
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, c_type_out, [&] {                    \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(input_dtype, c_type_in,                           \
                                                 [&] { return __VA_ARGS__(); });                   \
    });                                                                                            \
  }()

#define DISPATCH_SCALE_GRANULARITY(scale_granularity_m, scale_granularity_n, scale_granularity_k, \
                                   SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, \
                                   ...)                                                           \
  [&]() -> bool {                                                                                 \
    if (scale_granularity_m == 1 && scale_granularity_n == 128 && scale_granularity_k == 128) {   \
      constexpr int SCALE_GRANULARITY_M = 1;                                                      \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      constexpr int SCALE_GRANULARITY_K = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    } else if (scale_granularity_m == 128 && scale_granularity_n == 128 &&                        \
               scale_granularity_k == 128) {                                                      \
      constexpr int SCALE_GRANULARITY_M = 128;                                                    \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      constexpr int SCALE_GRANULARITY_K = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    TORCH_CHECK(false, "Unsupported scale granularity");                                          \
    return false;                                                                                 \
  }()

void CutlassGroupGemmGroupwiseScaledSM100(at::Tensor int_workspace_buffer,
                                          at::Tensor float_workspace_buffer, at::Tensor A,
                                          at::Tensor B, at::Tensor SFA, at::Tensor SFB,
                                          at::Tensor C, at::Tensor m_indptr, int64_t cum_m,
                                          int64_t n, int64_t k, int64_t scale_granularity_m,
                                          int64_t scale_granularity_n,
                                          int64_t scale_granularity_k) {
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int batch_size = m_indptr.size(0) - 1;
  DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(A.scalar_type(), C.scalar_type(), c_type_in, c_type_out, [&] {
    return DISPATCH_SCALE_GRANULARITY(
        scale_granularity_m, scale_granularity_n, scale_granularity_k, SCALE_GRANULARITY_M,
        SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, [&] {
          using cutlass_t_in = cutlass_dtype_t<c_type_in>;
          using cutlass_t_out = cutlass_dtype_t<c_type_out>;
          auto status = flashinfer::gemm::CutlassGroupwiseScaledGroupGEMMSM100<
              SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K>(
              static_cast<int*>(int_workspace_buffer.data_ptr()),
              int_workspace_buffer.element_size() * int_workspace_buffer.size(0),
              static_cast<float*>(float_workspace_buffer.data_ptr()),
              float_workspace_buffer.element_size() * float_workspace_buffer.size(0),
              static_cast<cutlass_t_in*>(A.data_ptr()), static_cast<cutlass_t_in*>(B.data_ptr()),
              static_cast<float*>(SFA.data_ptr()), static_cast<float*>(SFB.data_ptr()),
              static_cast<cutlass_t_out*>(C.data_ptr()), static_cast<int*>(m_indptr.data_ptr()),
              cum_m, n, k, batch_size, stream);
          return true;
        });
  });
}
