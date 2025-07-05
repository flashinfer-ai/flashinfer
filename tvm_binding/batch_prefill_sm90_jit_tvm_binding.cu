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
#include "batch_prefill_sm90_config.inc"
#include "tvm_binding_utils.h"

IntTuple BatchPrefillWithKVCacheSM90Plan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    IntTuple kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream);

void BatchPrefillWithRaggedKVCacheSM90Run(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* qo_indptr, DLTensor* kv_indptr,
    DLTensor* q_rope_offset, DLTensor* k_rope_offset, DLTensor* o, DLTensor* lse,
    int64_t mask_mode_code, int64_t pos_encoding_mode_code, int64_t layout,
    int64_t window_left ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

void BatchPrefillWithPagedKVCacheSM90Run(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* qo_indptr, DLTensor* paged_kv_indptr,
    DLTensor* paged_kv_indices, DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset,
    DLTensor* paged_kv_rope_pos_offset, DLTensor* o, DLTensor* lse, int64_t mask_mode_code,
    int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS,
    TVMStreamHandle cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_kv_cache_plan, BatchPrefillWithKVCacheSM90Plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_ragged_kv_cache_run,
                              BatchPrefillWithRaggedKVCacheSM90Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_paged_kv_cache_run,
                              BatchPrefillWithPagedKVCacheSM90Run);
