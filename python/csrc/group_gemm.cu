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
#include <flashinfer/gemm/group_gemm.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer::group_gemm;

void CutlassSegmentGEMM(torch::Tensor workspace_buffer, torch::Tensor all_problems,
                        torch::Tensor x_ptr, torch::Tensor w_ptr, torch::Tensor y_ptr,
                        torch::Tensor x_ld, torch::Tensor w_ld, torch::Tensor y_ld,
                        torch::Tensor empty_x_data, bool weight_column_major) {
  unsigned int batch_size = x_ptr.size(0);
  auto device = workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(empty_x_data.scalar_type(), c_type, [&] {
    using cutlass_t = typename cutlass_dtype<c_type>::value;
    auto status = CutlassSegmentGEMMRun<cutlass_t>(
        workspace_buffer.data_ptr(), workspace_buffer.element_size() * workspace_buffer.size(0),
        all_problems.data_ptr(), batch_size, x_ptr.data_ptr(), w_ptr.data_ptr(), y_ptr.data_ptr(),
        x_ld.data_ptr(), w_ld.data_ptr(), y_ld.data_ptr(), weight_column_major,
        torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassSegmentGEMM: ", cudaGetErrorString(status));
    return true;
  });
}
