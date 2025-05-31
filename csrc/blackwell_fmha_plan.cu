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

void blackwell_fmha_plan(at::Tensor qo_lens, at::Tensor kv_lens, at::Tensor work_indptr,
                         at::Tensor qo_tile_indices, at::Tensor head_indices,
                         at::Tensor batch_indices, int64_t qo_tile_size, int64_t batch_size,
                         int64_t num_heads, int64_t num_buckets, cudaStream_t stream) {
  auto status = plan_kernel_wrapper(
      static_cast<int*>(qo_lens.data_ptr()), static_cast<int*>(kv_lens.data_ptr()),
      static_cast<int*>(work_indptr.data_ptr()), static_cast<int*>(qo_tile_indices.data_ptr()),
      static_cast<int*>(head_indices.data_ptr()), static_cast<int*>(batch_indices.data_ptr()),
      qo_tile_size, batch_size, num_heads, num_buckets,
      /*enable_pdl=*/true, stream);
  TORCH_CHECK(status == cudaSuccess, "Failed to plan blackwell fmha", cudaGetErrorString(status));
}
