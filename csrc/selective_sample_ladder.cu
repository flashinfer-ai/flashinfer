/*
 * Copyright (c) 2026 by FlashInfer team.
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
#include <flashinfer/attention/selective_sample_ladder.cuh>

#include "tvm_ffi_utils.h"

void selective_sample_ladder(TensorView samples, TensorView valid, TensorView ranks,
                             TensorView ends, TensorView thresholds, TensorView selected_bins,
                             TensorView ladder, TensorView populations, int64_t top_k,
                             int64_t capacity, int64_t count_segments) {
  CHECK_CUDA(samples);
  CHECK_INPUT_TYPE(samples, dl_float32);
  TVM_FFI_ICHECK_EQ(samples.ndim(), 2);
  TVM_FFI_ICHECK_EQ(samples.stride(1), 1);
  TVM_FFI_ICHECK_EQ(valid.ndim(), 1);
  // Scorer storage can include inactive tile rows; policy metadata is logical.
  const auto rows = valid.size(0);
  TVM_FFI_ICHECK_GE(samples.size(0), rows);
  for (const auto& tensor : {valid, ranks, ends, thresholds, selected_bins, ladder, populations}) {
    CHECK_INPUT(tensor);
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, samples.device().device_id);
    TVM_FFI_ICHECK_EQ(tensor.size(0), rows);
  }
  for (const auto& tensor : {valid, ranks, ends, selected_bins, populations}) {
    CHECK_INPUT_TYPE(tensor, dl_int32);
  }
  for (const auto& tensor : {thresholds, ladder}) {
    CHECK_INPUT_TYPE(tensor, dl_float32);
  }
  for (const auto& tensor : {valid, ranks, ends, thresholds, selected_bins}) {
    TVM_FFI_ICHECK_EQ(tensor.ndim(), 1);
  }
  for (const auto& tensor : {ladder, populations}) {
    TVM_FFI_ICHECK_EQ(tensor.ndim(), 2);
    TVM_FFI_ICHECK_EQ(tensor.size(1), 3);
  }
  TVM_FFI_ICHECK(top_k > 0 && top_k <= capacity && capacity <= INT32_MAX);
  TVM_FFI_ICHECK(count_segments > 0 && count_segments <= capacity);
  TVM_FFI_ICHECK(rows <= INT32_MAX);
  if (rows == 0) return;
  ffi::CUDADeviceGuard guard(samples.device().device_id);
  flashinfer::attention::SelectiveSampleLadder<<<rows, 128, 0, get_stream(samples.device())>>>(
      static_cast<const float*>(samples.data_ptr()), samples.stride(0),
      static_cast<const int32_t*>(valid.data_ptr()), static_cast<const int32_t*>(ranks.data_ptr()),
      static_cast<const int32_t*>(ends.data_ptr()), static_cast<float*>(thresholds.data_ptr()),
      static_cast<int32_t*>(selected_bins.data_ptr()), static_cast<float*>(ladder.data_ptr()),
      static_cast<int32_t*>(populations.data_ptr()), top_k, capacity, count_segments);
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, selective_sample_ladder);
