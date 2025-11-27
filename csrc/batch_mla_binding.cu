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
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchMLAPagedAttentionPlan(TensorView float_workspace_buffer,
                                          TensorView int_workspace_buffer,
                                          TensorView page_locked_int_workspace_buffer,
                                          TensorView qo_indptr, TensorView kv_indptr,
                                          TensorView kv_len, int64_t num_heads, int64_t head_dim_o,
                                          bool causal);

void BatchMLAPagedAttentionRun(TensorView float_workspace_buffer, TensorView int_workspace_buffer,
                               Array<int64_t> plan_info_vec, TensorView q_nope, TensorView q_pe,
                               TensorView ckv_cache, TensorView kpe_cache, TensorView kv_indices,
                               TensorView o, Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                               int64_t num_heads, int64_t page_size, double sm_scale,
                               bool return_lse_base_on_e);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchMLAPagedAttentionPlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, BatchMLAPagedAttentionRun);
