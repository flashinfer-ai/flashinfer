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
#include "batch_mla_sm90_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchMLAPagedAttentionSM90Plan(Tensor float_workspace_buffer,
                                              Tensor int_workspace_buffer,
                                              Tensor page_locked_int_workspace_buffer,
                                              Tensor qo_indptr, Tensor kv_indptr, Tensor kv_len,
                                              int64_t num_heads, int64_t head_dim_o, bool causal);

void BatchMLAPagedAttentionSM90Run(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                   Tensor plan_info_vec, Tensor q_nope, Tensor q_pe,
                                   Tensor ckv_cache, Tensor kpe_cache, Tensor kv_indices, Tensor o,
                                   Optional<Tensor> maybe_lse, int64_t mask_mode_code,
                                   int64_t num_heads, int64_t page_size,
                                   double sm_scale ADDITIONAL_FUNC_PARAMS);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchMLAPagedAttentionSM90Plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, BatchMLAPagedAttentionSM90Run);
