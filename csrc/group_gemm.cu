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

using namespace flashinfer;
using namespace flashinfer::group_gemm;

void CutlassSegmentGEMM(at::Tensor workspace_buffer, at::Tensor all_problems, at::Tensor x_ptr,
                        at::Tensor w_ptr, at::Tensor y_ptr, at::Tensor x_ld, at::Tensor w_ld,
                        at::Tensor y_ld, at::Tensor empty_x_data, bool weight_column_major) {
  unsigned int batch_size = x_ptr.size(0);

  const c10::cuda::OptionalCUDAGuard device_guard(workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(empty_x_data.scalar_type(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    auto status = CutlassSegmentGEMMRun<cutlass_t>(
        workspace_buffer.data_ptr(), workspace_buffer.element_size() * workspace_buffer.size(0),
        all_problems.data_ptr(), batch_size, x_ptr.data_ptr(), w_ptr.data_ptr(), y_ptr.data_ptr(),
        x_ld.data_ptr(), w_ld.data_ptr(), y_ld.data_ptr(), weight_column_major, stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassSegmentGEMM: ", cudaGetErrorString(status));
    return true;
  });
}
