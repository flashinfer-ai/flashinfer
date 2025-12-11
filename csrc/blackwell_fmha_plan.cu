/*
 * Copyright (c) 2025 by FlashInfer team.
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

#include "flashinfer/attention/blackwell/plan.cuh"
#include "tvm_ffi_utils.h"

void blackwell_fmha_plan(TensorView qo_segment_offsets, TensorView kv_segment_offsets,
                         TensorView work_indptr, TensorView qo_tile_indices,
                         TensorView head_indices, TensorView batch_indices, int64_t qo_tile_size,
                         int64_t num_heads, int64_t num_buckets, bool causal) {
  ffi::CUDADeviceGuard device_guard(qo_segment_offsets.device().device_id);
  const cudaStream_t stream = get_stream(qo_tile_indices.device());
  int batch_size = qo_segment_offsets.size(0) - 1;

  auto status = flashinfer::plan_kernel_wrapper(
      static_cast<int*>(qo_segment_offsets.data_ptr()),
      static_cast<int*>(kv_segment_offsets.data_ptr()),
      /*qo_lens=*/nullptr,
      /*kv_lens=*/nullptr, static_cast<int*>(work_indptr.data_ptr()),
      static_cast<int*>(qo_tile_indices.data_ptr()), static_cast<int*>(head_indices.data_ptr()),
      static_cast<int*>(batch_indices.data_ptr()), qo_tile_size, batch_size, num_heads, num_buckets,
      causal, /*enable_pdl=*/true, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Failed to plan blackwell fmha" << cudaGetErrorString(status);
}
