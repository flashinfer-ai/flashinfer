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
#include "batch_prefill_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchPrefillWithKVCachePlan(
    Tensor float_workspace_buffer, Tensor int_workspace_buffer,
    Tensor page_locked_int_workspace_buffer, Tensor qo_indptr, Tensor kv_indptr, Tensor kv_len_arr,
    int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk, int64_t head_dim_vo,
    bool causal, int64_t window_left, int64_t fixed_split_size, bool disable_split_kv);

void BatchPrefillWithRaggedKVCacheRun(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                      Array<int64_t> plan_info_vec, Tensor q, Tensor k, Tensor v,
                                      Tensor qo_indptr, Tensor kv_indptr, Tensor o,
                                      Optional<Tensor> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout, int64_t window_left,
                                      bool enable_pdl ADDITIONAL_FUNC_PARAMS);

void BatchPrefillWithPagedKVCacheRun(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                     Array<int64_t> plan_info_vec, Tensor q, Tensor paged_k_cache,
                                     Tensor paged_v_cache, Tensor qo_indptr, Tensor paged_kv_indptr,
                                     Tensor paged_kv_indices, Tensor paged_kv_last_page_len,
                                     Tensor o, Optional<Tensor> maybe_lse, int64_t mask_mode_code,
                                     int64_t layout, int64_t window_left,
                                     bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run, BatchPrefillWithRaggedKVCacheRun);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, BatchPrefillWithPagedKVCacheRun);
