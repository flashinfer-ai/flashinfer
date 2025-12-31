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

#include "tvm_ffi_utils.h"

using namespace flashinfer;

#define DISPATCH_DLPACK_INPUT_OUTPUT_DTYPE(input_dtype, output_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                                 \
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output_dtype, c_type_out, [&] {                    \
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(input_dtype, c_type_in,                           \
                                                [&] { return __VA_ARGS__(); });                   \
    });                                                                                           \
  }()

#define DISPATCH_SCALE_GRANULARITY(scale_granularity_m, scale_granularity_n, scale_granularity_k, \
                                   SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, \
                                   ...)                                                           \
  [&]() -> bool {                                                                                 \
    constexpr int SCALE_GRANULARITY_K = 128;                                                      \
    if (scale_granularity_k != 128) {                                                             \
      TVM_FFI_ICHECK(false)                                                                       \
          << "SM120 requires scale_granularity_k=128. CUTLASS enforces ScaleGranularityK must "   \
             "equal tile shape K dimension (128 for both Cooperative and PingPong schedules).";   \
      return false;                                                                               \
    }                                                                                             \
    /* Match SM100's approach: support only (1,128,128) and (128,128,128) */                      \
    if (scale_granularity_m == 1 && scale_granularity_n == 128) {                                 \
      constexpr int SCALE_GRANULARITY_M = 1;                                                      \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    } else if (scale_granularity_m == 128 && scale_granularity_n == 128) {                        \
      constexpr int SCALE_GRANULARITY_M = 128;                                                    \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    TVM_FFI_ICHECK(false) << "SM120: Unsupported scale granularity combination ("                 \
                          << scale_granularity_m << "," << scale_granularity_n << ","             \
                          << scale_granularity_k << ")";                                          \
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
    TVM_FFI_ICHECK(false) << "Unsupported Scale Major Mode";         \
    return false;                                                    \
  }()

namespace flashinfer {
namespace group_gemm {

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          typename DTypeIn, typename DTypeOut>
cudaError_t CutlassFP8GroupwiseScaledGroupGEMMSM120(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, DTypeIn* A, DTypeIn* B, float* SFA, float* SFB, DTypeOut* D,
    int* m_indptr, int max_m, int n, int k, int num_groups, cudaStream_t stream);

}  // namespace group_gemm
}  // namespace flashinfer

void CutlassGroupGemmFP8GroupwiseScaledSM120(
    TensorView int_workspace_buffer, TensorView float_workspace_buffer, TensorView A, TensorView B,
    TensorView SFA, TensorView SFB, TensorView D, TensorView m_indptr, int64_t n, int64_t k,
    int64_t scale_granularity_m, int64_t scale_granularity_n, int64_t scale_granularity_k,
    std::string scale_major_mode) {
  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  auto stream = get_stream(D.device());
  int num_groups = m_indptr.size(0) - 1;

  // Ensure scales are contiguous
  // Note: We keep the original shape and let the kernel's layout handle interpretation
  CHECK_CONTIGUOUS(SFA);
  CHECK_CONTIGUOUS(SFB);

  // Get max_m from SFA shape
  int max_m = SFA.size(SFA.ndim() > 1 ? 1 : 0);

  DISPATCH_DLPACK_INPUT_OUTPUT_DTYPE(A.dtype(), D.dtype(), c_type_in, c_type_out, [&] {
    return DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, [&] {
      return DISPATCH_SCALE_GRANULARITY(
          scale_granularity_m, scale_granularity_n, scale_granularity_k, SCALE_GRANULARITY_M,
          SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, [&] {
            using cutlass_t_in = cutlass_dtype_t<c_type_in>;
            using cutlass_t_out = cutlass_dtype_t<c_type_out>;
            auto status = flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM120<
                SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, SCALE_MAJOR_K,
                cutlass_t_in, cutlass_t_out>(
                static_cast<int*>(int_workspace_buffer.data_ptr()),
                get_element_size(int_workspace_buffer) * int_workspace_buffer.size(0),
                static_cast<float*>(float_workspace_buffer.data_ptr()),
                get_element_size(float_workspace_buffer) * float_workspace_buffer.size(0),
                static_cast<cutlass_t_in*>(A.data_ptr()), static_cast<cutlass_t_in*>(B.data_ptr()),
                static_cast<float*>(SFA.data_ptr()), static_cast<float*>(SFB.data_ptr()),
                static_cast<cutlass_t_out*>(D.data_ptr()), static_cast<int*>(m_indptr.data_ptr()),
                max_m, n, k, num_groups, stream);
            return status == cudaSuccess;
          });
    });
  });
}
