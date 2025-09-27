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

#include "batch_mla_sm90_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;

Array<int64_t> BatchMLAPagedAttentionSM90Plan(Tensor float_workspace_buffer,
                                              Tensor int_workspace_buffer,
                                              Tensor page_locked_int_workspace_buffer,
                                              Tensor qo_indptr, Tensor kv_indptr, Tensor kv_len,
                                              int64_t num_heads, int64_t head_dim_o, bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * get_element_size(int_workspace_buffer);

  MLAPlanInfo plan_info;

  int batch_size = kv_len->shape[0];

  cudaSetDevice(float_workspace_buffer->device.device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer->device);

  cudaError_t status = MLAPlan(
      float_workspace_buffer->data, float_workspace_size_in_bytes, int_workspace_buffer->data,
      page_locked_int_workspace_buffer->data, int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data), static_cast<IdType*>(kv_indptr->data),
      static_cast<IdType*>(kv_len->data), batch_size, num_heads, head_dim_o, causal, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan MLA, error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}
