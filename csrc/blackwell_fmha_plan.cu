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
#include "pytorch_extension_utils.h"

void blackwell_fmha_plan(at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets,
                         at::Tensor work_indptr, at::Tensor qo_tile_indices,
                         at::Tensor head_indices, at::Tensor batch_indices, int64_t qo_tile_size,
                         int64_t num_heads, int64_t num_buckets, bool causal) {
  const c10::cuda::OptionalCUDAGuard device_guard(qo_segment_offsets.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int batch_size = qo_segment_offsets.size(0) - 1;

  auto status = flashinfer::plan_kernel_wrapper(
      static_cast<int*>(qo_segment_offsets.data_ptr()),
      static_cast<int*>(kv_segment_offsets.data_ptr()),
      /*qo_lens=*/nullptr,
      /*kv_lens=*/nullptr, static_cast<int*>(work_indptr.data_ptr()),
      static_cast<int*>(qo_tile_indices.data_ptr()), static_cast<int*>(head_indices.data_ptr()),
      static_cast<int*>(batch_indices.data_ptr()), qo_tile_size, batch_size, num_heads, num_buckets,
      causal, /*enable_pdl=*/true, stream);
  TORCH_CHECK(status == cudaSuccess, "Failed to plan blackwell fmha", cudaGetErrorString(status));
}
