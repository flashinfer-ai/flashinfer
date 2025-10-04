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
#include "batch_attention_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchPagedAttentionPlan(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                       Tensor page_locked_int_workspace_buffer, Tensor qo_indptr,
                                       Tensor kv_indptr, Tensor kv_len, int64_t batch_size,
                                       int64_t num_qo_heads, int64_t num_kv_heads,
                                       int64_t head_dim_o, bool causal);

void BatchPagedAttentionRun(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                            Array<int64_t> plan_info_vec, Tensor q, Tensor k_cache, Tensor v_cache,
                            Tensor kv_indices, Tensor o, Optional<Tensor> maybe_lse,
                            int64_t mask_mode_code, int64_t layout_code, int64_t num_qo_heads,
                            int64_t num_kv_heads, int64_t page_size, double v_scale,
                            double sm_scale,
                            double logits_soft_cap ADDITIONAL_FUNC_PARAMS PROFILER_FUNC_PARAMS);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, &BatchPagedAttentionPlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, &BatchPagedAttentionRun);
