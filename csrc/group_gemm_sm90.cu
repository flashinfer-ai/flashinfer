/*
 * Copyright (c) 2024 by FlashInfer team.
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

#define DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(input_dtype, output_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                                  \
    if (input_dtype == output_dtype) {                                                             \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type_in, [&] {                    \
        using c_type_out = c_type_in;                                                              \
        return __VA_ARGS__();                                                                      \
      });                                                                                          \
    } else {                                                                                       \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, c_type_out, [&] {                  \
        return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(input_dtype, c_type_in,                         \
                                                   [&] { return __VA_ARGS__(); });                 \
      });                                                                                          \
    }                                                                                              \
  }()

namespace flashinfer {
namespace group_gemm {

template <typename DTypeIn, typename DTypeOut>
cudaError_t CutlassSegmentGEMMSM90Run(void* float_buffer, size_t float_buffer_size_in_bytes,
                                      void* int_buffer, size_t int_buffer_size_in_bytes,
                                      void* all_problems, int64_t batch_size, void* x, void* w,
                                      void* y, void* x_stride, void* w_stride, void* y_stride,
                                      bool weight_column_major, cudaStream_t stream);

}  // namespace group_gemm
}  // namespace flashinfer

void CutlassSegmentGEMMSM90(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor all_problems, at::Tensor x_ptr, at::Tensor w_ptr,
                            at::Tensor y_ptr, at::Tensor x_stride, at::Tensor weight_stride,
                            at::Tensor y_stride, at::Tensor empty_x_data, at::Tensor empty_y_data,
                            bool weight_column_major) {
  unsigned int batch_size = x_ptr.size(0);
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(
      empty_x_data.scalar_type(), empty_y_data.scalar_type(), c_type_in, c_type_out, [&] {
        using cutlass_t_in = cutlass_dtype_t<c_type_in>;
        using cutlass_t_out = cutlass_dtype_t<c_type_out>;
        auto status =
            flashinfer::group_gemm::CutlassSegmentGEMMSM90Run<cutlass_t_in, cutlass_t_out>(
                float_workspace_buffer.data_ptr(),
                float_workspace_buffer.element_size() * float_workspace_buffer.size(0),
                int_workspace_buffer.data_ptr(),
                int_workspace_buffer.element_size() * int_workspace_buffer.size(0),
                all_problems.data_ptr(), batch_size, x_ptr.data_ptr(), w_ptr.data_ptr(),
                y_ptr.data_ptr(), x_stride.data_ptr(), weight_stride.data_ptr(),
                y_stride.data_ptr(), weight_column_major, stream);
        TORCH_CHECK(status == cudaSuccess,
                    "Failed to run CutlassSegmentGEMM: ", cudaGetErrorString(status));
        return true;
      });
}
