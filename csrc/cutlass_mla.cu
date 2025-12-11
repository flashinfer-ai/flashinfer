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
#include <flashinfer/attention/cutlass_mla.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using namespace flashinfer::attention;

void CutlassMLAPagedAttention(ffi::TensorView workspace, ffi::TensorView out, ffi::TensorView lse,
                              ffi::TensorView q_nope_pe, ffi::TensorView ckv_kpe_cache,
                              ffi::TensorView kv_lens, ffi::TensorView page_table) {
  ffi::CUDADeviceGuard device_guard(q_nope_pe.device().device_id);
  const cudaStream_t stream = get_stream(q_nope_pe.device());

  int device_index = q_nope_pe.device().device_id;
  int batches = q_nope_pe.size(0);
  int page_count_per_seq = page_table.size(1);
  int page_count_total = ckv_kpe_cache.size(0);
  int page_size = ckv_kpe_cache.size(1);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q_nope_pe.dtype(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    auto status = runMla<cutlass_t>(
        workspace.data_ptr(), out.data_ptr(), lse.data_ptr(), q_nope_pe.data_ptr(),
        ckv_kpe_cache.data_ptr(), kv_lens.data_ptr(), page_table.data_ptr(), batches,
        page_count_per_seq, page_count_total, page_size, device_index, stream);

    TVM_FFI_ICHECK(status == cudaSuccess)
        << "Failed to run CutlassMLAPagedAttention: " << cudaGetErrorString(status);
    return true;
  });
}
