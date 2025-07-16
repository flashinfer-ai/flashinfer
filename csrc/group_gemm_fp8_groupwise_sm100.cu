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

#include "pytorch_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_MMA_SM(mma_sm, MMA_SM, ...)  \
  [&]() -> bool {                             \
    if (mma_sm == 1) {                        \
      constexpr int MMA_SM = 1;               \
      return __VA_ARGS__();                   \
    } else if (mma_sm == 2) {                 \
      constexpr int MMA_SM = 2;               \
      return __VA_ARGS__();                   \
    }                                         \
    TORCH_CHECK(false, "Unsupported MMA SM"); \
    return false;                             \
  }()

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

#define DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, ...) \
  [&]() -> bool {                                                    \
    if (scale_major_mode == "K") {                                   \
      constexpr bool SCALE_MAJOR_K = true;                           \
      return __VA_ARGS__();                                          \
    } else if (scale_major_mode == "MN") {                           \
      constexpr bool SCALE_MAJOR_K = false;                          \
      return __VA_ARGS__();                                          \
    }                                                                \
    TORCH_CHECK(false, "Unsupported Scale Major Mode");              \
    return false;                                                    \
  }()

namespace flashinfer {
namespace group_gemm {

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassFP8GroupwiseScaledGroupGEMMSM100(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, DTypeIn* A, DTypeIn* B, float* SFA, float* SFB, DTypeOut* D,
    int* m_indptr, int max_m, int n, int k, int num_groups, cudaStream_t stream);

}  // namespace group_gemm
}  // namespace flashinfer

void CutlassGroupGemmFP8GroupwiseScaledSM100(
    at::Tensor int_workspace_buffer, at::Tensor float_workspace_buffer, at::Tensor A, at::Tensor B,
    at::Tensor SFA, at::Tensor SFB, at::Tensor D, at::Tensor m_indptr, int64_t n, int64_t k,
    int64_t scale_granularity_m, int64_t scale_granularity_n, int64_t scale_granularity_k,
    std::string scale_major_mode, int64_t mma_sm) {
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int num_groups = m_indptr.size(0) - 1;
  int max_m = SFA.size(1);
  DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(A.scalar_type(), D.scalar_type(), c_type_in, c_type_out, [&] {
    return DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, [&] {
      return DISPATCH_MMA_SM(mma_sm, MMA_SM, [&] {
        return DISPATCH_SCALE_GRANULARITY(
            scale_granularity_m, scale_granularity_n, scale_granularity_k, SCALE_GRANULARITY_M,
            SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, [&] {
              using cutlass_t_in = cutlass_dtype_t<c_type_in>;
              using cutlass_t_out = cutlass_dtype_t<c_type_out>;
              auto status = flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
                  SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, SCALE_MAJOR_K,
                  MMA_SM>(static_cast<int*>(int_workspace_buffer.data_ptr()),
                          int_workspace_buffer.element_size() * int_workspace_buffer.size(0),
                          static_cast<float*>(float_workspace_buffer.data_ptr()),
                          float_workspace_buffer.element_size() * float_workspace_buffer.size(0),
                          static_cast<cutlass_t_in*>(A.data_ptr()),
                          static_cast<cutlass_t_in*>(B.data_ptr()),
                          static_cast<float*>(SFA.data_ptr()), static_cast<float*>(SFB.data_ptr()),
                          static_cast<cutlass_t_out*>(D.data_ptr()),
                          static_cast<int*>(m_indptr.data_ptr()), max_m, n, k, num_groups, stream);
              return status == cudaSuccess;
            });
      });
    });
  });
}
