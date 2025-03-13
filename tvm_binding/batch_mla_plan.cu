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

#include "batch_mla_config.inc"
#include "tvm_binding_utils.h"

using namespace flashinfer;

IntTuple BatchMLAPagedAttentionPlan(DLTensor* float_workspace_buffer,
                                    DLTensor* int_workspace_buffer,
                                    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr,
                                    DLTensor* kv_indptr, IntTuple kv_len_arr, int64_t num_heads,
                                    int64_t head_dim_o, bool causal, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();
  std::vector<IdType> kv_len_vec{kv_len_arr->data, kv_len_arr->data + kv_len_arr->size};

  MLAPlanInfo plan_info;

  int batch_size = kv_len_vec.size();

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = MLAPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      kv_len_vec.data(), batch_size, num_heads, head_dim_o, causal, stream);

  CHECK(status == cudaSuccess) << "Failed to plan MLA, error: " << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}
