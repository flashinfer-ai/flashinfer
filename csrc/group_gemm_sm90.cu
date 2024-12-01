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
#include <flashinfer/gemm/group_gemm_sm90.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;
using namespace flashinfer::group_gemm;

void CutlassSegmentGEMMSM90(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor all_problems, at::Tensor x_ptr, at::Tensor w_ptr,
                            at::Tensor y_ptr, at::Tensor x_stride, at::Tensor weight_stride,
                            at::Tensor y_stride, at::Tensor empty_x_data, bool weight_column_major,
                            int64_t cuda_stream) {
  unsigned int batch_size = x_ptr.size(0);
  auto device = float_workspace_buffer.device();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(empty_x_data.scalar_type(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    auto status = CutlassSegmentGEMMSM90Run<cutlass_t, cutlass_t>(
        float_workspace_buffer.data_ptr(),
        float_workspace_buffer.element_size() * float_workspace_buffer.size(0),
        int_workspace_buffer.data_ptr(),
        int_workspace_buffer.element_size() * int_workspace_buffer.size(0), all_problems.data_ptr(),
        batch_size, x_ptr.data_ptr(), w_ptr.data_ptr(), y_ptr.data_ptr(), x_stride.data_ptr(),
        weight_stride.data_ptr(), y_stride.data_ptr(), weight_column_major, stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassSegmentGEMM: ", cudaGetErrorString(status));
    return true;
  });
}
