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
#include <flashinfer/attention/scheduler.cuh>
#include <optional>

#include "batch_mla_sm90_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

at::Tensor BatchMLAPagedAttentionSM90Plan(at::Tensor float_workspace_buffer,
                                          at::Tensor int_workspace_buffer,
                                          at::Tensor page_locked_int_workspace_buffer,
                                          at::Tensor qo_indptr, at::Tensor kv_indptr,
                                          at::Tensor kv_len, int64_t num_heads, int64_t head_dim_o,
                                          bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  MLAPlanInfo plan_info;

  int batch_size = kv_len.size(0);

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  cudaError_t status =
      MLAPlan(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
              int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
              int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
              static_cast<IdType*>(kv_indptr.data_ptr()), static_cast<IdType*>(kv_len.data_ptr()),
              batch_size, num_heads, head_dim_o, causal, stream);

  TORCH_CHECK(status == cudaSuccess, "Failed to plan MLA, error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}
