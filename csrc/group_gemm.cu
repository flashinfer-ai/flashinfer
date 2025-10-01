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

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using namespace flashinfer::group_gemm;

void CutlassSegmentGEMM(Tensor workspace_buffer, Tensor all_problems, Tensor x_ptr, Tensor w_ptr,
                        Tensor y_ptr, Tensor x_ld, Tensor w_ld, Tensor y_ld, Tensor empty_x_data,
                        bool weight_column_major) {
  unsigned int batch_size = x_ptr->shape[0];

  cudaSetDevice(workspace_buffer->device.device_id);
  const cudaStream_t stream = get_stream(workspace_buffer->device);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(empty_x_data->dtype, c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    auto status = CutlassSegmentGEMMRun<cutlass_t>(
        workspace_buffer->data, get_element_size(workspace_buffer) * workspace_buffer->shape[0],
        all_problems->data, batch_size, x_ptr->data, w_ptr->data, y_ptr->data, x_ld->data,
        w_ld->data, y_ld->data, weight_column_major, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "Failed to run CutlassSegmentGEMM: " << cudaGetErrorString(status);
    return true;
  });
}
