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

void radix_topk(TensorView input, TensorView output_indices, Optional<TensorView> maybe_starts,
                Optional<TensorView> maybe_ends, int64_t top_k) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_indices);
  CHECK_DIM(2, input);           // input: (batch_size, d)
  CHECK_DIM(2, output_indices);  // output_indices: (batch_size, top_k)

  unsigned int batch_size = input.size(0);
  unsigned int d = input.size(1);

  bool has_starts = maybe_starts.has_value();
  bool has_ends = maybe_ends.has_value();

  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());

  cudaError_t status = sampling::RadixTopK<float, int32_t>(
      static_cast<float*>(input.data_ptr()), static_cast<int32_t*>(output_indices.data_ptr()),
      has_starts ? static_cast<int32_t*>(maybe_starts.value().data_ptr()) : nullptr,
      has_ends ? static_cast<int32_t*>(maybe_ends.value().data_ptr()) : nullptr, batch_size, d,
      static_cast<uint32_t>(top_k), stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "RadixTopK failed with error code " << cudaGetErrorString(status);
}
