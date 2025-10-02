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
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchDecodeWithPagedKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    TensorView empty_q_data, TensorView empty_kv_data);

void BatchDecodeWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                    TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                    TensorView q, TensorView paged_k_cache,
                                    TensorView paged_v_cache, TensorView paged_kv_indptr,
                                    TensorView paged_kv_indices, TensorView paged_kv_last_page_len,
                                    TensorView o, Optional<TensorView> maybe_lse,
                                    int64_t kv_layout_code, int64_t window_left,
                                    bool enable_pdl ADDITIONAL_FUNC_PARAMS);

// Batched decode with paged KV-Cache plan
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchDecodeWithPagedKVCachePlan);
// Batched decode with paged KV-Cache run
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, BatchDecodeWithPagedKVCacheRun);
