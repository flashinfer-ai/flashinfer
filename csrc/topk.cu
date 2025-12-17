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
#include <flashinfer/sampling.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

void radix_topk(TensorView input, TensorView output_indices, TensorView output_values,
                Optional<TensorView> maybe_row_states_buffer, int64_t top_k) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(output_values);
  CHECK_DIM(2, input);           // input: (batch_size, d)
  CHECK_DIM(2, output_indices);  // output_indices: (batch_size, top_k)
  CHECK_DIM(2, output_values);   // output_values: (batch_size, top_k)

  unsigned int batch_size = input.size(0);
  unsigned int d = input.size(1);

  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());

  cudaError_t status;
  auto dtype = input.dtype();

  // Get row_states_buffer if provided (for multi-CTA path)
  sampling::RadixRowState* row_states_ptr = nullptr;
  if (maybe_row_states_buffer.has_value()) {
    row_states_ptr =
        static_cast<sampling::RadixRowState*>(maybe_row_states_buffer.value().data_ptr());
  }

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKMultiCTA<c_type, int32_t>(
        static_cast<c_type*>(input.data_ptr()), static_cast<int32_t*>(output_indices.data_ptr()),
        static_cast<c_type*>(output_values.data_ptr()),
        nullptr,  // top_k_arr
        batch_size, static_cast<uint32_t>(top_k), d, row_states_ptr, stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "RadixTopK failed with error code " << cudaGetErrorString(status);
}
