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
    TVM_FFI_ICHECK(false) << "Unsupported scale granularity";                                     \
    return false;                                                                                 \
  }()

#define DISPATCH_MMA_SM(mma_sm, MMA_SM, ...)       \
  [&]() -> bool {                                  \
    if (mma_sm == 1) {                             \
      constexpr int MMA_SM = 1;                    \
      return __VA_ARGS__();                        \
    } else if (mma_sm == 2) {                      \
      constexpr int MMA_SM = 2;                    \
      return __VA_ARGS__();                        \
    }                                              \
    TVM_FFI_ICHECK(false) << "Unsupported MMA SM"; \
    return false;                                  \
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
namespace gemm {

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM100(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* C_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream);

}  // namespace gemm
}  // namespace flashinfer

void CutlassGemmGroupwiseScaledSM100(TensorView float_workspace_buffer, TensorView A, TensorView B,
                                     TensorView SFA, TensorView SFB, TensorView C,
                                     int64_t scale_granularity_m, int64_t scale_granularity_n,
                                     int64_t scale_granularity_k, std::string scale_major_mode,
                                     int64_t mma_sm) {
  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(C.device());
  DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, [&] {
    return DISPATCH_MMA_SM(mma_sm, MMA_SM, [&] {
      return DISPATCH_DLPACK_INPUT_OUTPUT_DTYPE(A.dtype(), C.dtype(), c_type_in, c_type_out, [&] {
        return DISPATCH_SCALE_GRANULARITY(
            scale_granularity_m, scale_granularity_n, scale_granularity_k, SCALE_GRANULARITY_M,
            SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, [&] {
              using cutlass_t_in = cutlass_dtype_t<c_type_in>;
              using cutlass_t_out = cutlass_dtype_t<c_type_out>;
              auto status = flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100<
                  SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, SCALE_MAJOR_K,
                  MMA_SM>(static_cast<float*>(float_workspace_buffer.data_ptr()),
                          get_element_size(float_workspace_buffer) * float_workspace_buffer.size(0),
                          static_cast<cutlass_t_in*>(A.data_ptr()),
                          static_cast<cutlass_t_in*>(B.data_ptr()),
                          static_cast<float*>(SFA.data_ptr()), static_cast<float*>(SFB.data_ptr()),
                          static_cast<cutlass_t_out*>(C.data_ptr()), A.size(0), B.size(0),
                          A.size(1), 1, stream);
              TVM_FFI_ICHECK_EQ(status, cudaSuccess)
                  << "Failed to run cutlass gemm groupwise scaled sm100"
                  << cudaGetErrorString(status);
              return true;
            });
      });
    });
  });
}
