/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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
#include "batch_decode_config.inc"
#include "tvm_binding_utils.h"

IntTuple BatchDecodeWithPagedKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t pos_encoding_mode_code, int64_t window_left, int64_t head_dim_qk, int64_t head_dim_vo,
    DataType q_scalar_type, DataType kv_scalar_type, TVMStreamHandle cuda_stream);

void BatchDecodeWithPagedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* paged_kv_indptr, DLTensor* paged_kv_indices,
    DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset, DLTensor* paged_kv_rope_pos_offset,
    DLTensor* o, DLTensor* lse, int64_t pos_encoding_mode_code, int64_t kv_layout_code,
    int64_t window_left ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_decode_with_paged_kv_cache_plan,
                              BatchDecodeWithPagedKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_decode_with_paged_kv_cache_run, BatchDecodeWithPagedKVCacheRun);
