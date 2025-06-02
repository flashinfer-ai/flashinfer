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
#include "batch_mla_config.inc"
#include "tvm_binding_utils.h"

IntTuple BatchMLAPagedAttentionPlan(DLTensor* float_workspace_buffer,
                                    DLTensor* int_workspace_buffer,
                                    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr,
                                    DLTensor* kv_indptr, IntTuple kv_len_arr, int64_t num_heads,
                                    int64_t head_dim_o, bool causal, TVMStreamHandle cuda_stream);

void BatchMLAPagedAttentionRun(DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
                               IntTuple plan_info_vec, DLTensor* q, DLTensor* kv_cache,
                               DLTensor* kv_indices, DLTensor* o, DLTensor* lse,
                               int64_t mask_mode_code, int64_t num_heads, int64_t page_size,
                               double sm_scale, TVMStreamHandle cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_mla_paged_attention_plan, BatchMLAPagedAttentionPlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_mla_paged_attention_run, BatchMLAPagedAttentionRun);
