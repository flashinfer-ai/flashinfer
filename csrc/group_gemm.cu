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

void CutlassSegmentGEMM(TensorView workspace_buffer, TensorView all_problems, TensorView x_ptr,
                        TensorView w_ptr, TensorView y_ptr, TensorView x_ld, TensorView w_ld,
                        TensorView y_ld, TensorView empty_x_data, bool weight_column_major) {
  unsigned int batch_size = x_ptr.size(0);

  ffi::CUDADeviceGuard device_guard(workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(workspace_buffer.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(empty_x_data.dtype(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    auto status = CutlassSegmentGEMMRun<cutlass_t>(
        workspace_buffer.data_ptr(), get_element_size(workspace_buffer) * workspace_buffer.size(0),
        all_problems.data_ptr(), batch_size, x_ptr.data_ptr(), w_ptr.data_ptr(), y_ptr.data_ptr(),
        x_ld.data_ptr(), w_ld.data_ptr(), y_ld.data_ptr(), weight_column_major, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "Failed to run CutlassSegmentGEMM: " << cudaGetErrorString(status);
    return true;
  });
}
