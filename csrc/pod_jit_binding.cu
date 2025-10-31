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
#include "pod_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> PODWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr_p, TensorView kv_indptr_p,
    int64_t total_num_rows_p, int64_t batch_size_p, TensorView qo_indptr_d, TensorView kv_indptr_d,
    int64_t total_num_rows_d, int64_t batch_size_d, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_vo, int64_t page_size, bool enable_cuda_graph);

void PODWithKVCacheTensorRun(
    // Shared params (match implementation in pod.cu)
    TensorView float_workspace_buffer_d, TensorView int_workspace_buffer_d,
    Array<int64_t> plan_info_vec, TensorView paged_k_cache, TensorView paged_v_cache,
    TensorView qo_indptr, TensorView paged_kv_indptr, TensorView paged_kv_indices,
    TensorView paged_kv_last_page_len, TensorView o, Optional<TensorView> maybe_lse, int64_t layout,
    // Prefill params
    TensorView q_p, int64_t mask_mode_code_p, int64_t window_left_p,
    Optional<TensorView> maybe_custom_mask_p, Optional<TensorView> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    TensorView q_d, int64_t mask_mode_code_d, int64_t window_left_d,
    Optional<TensorView> maybe_custom_mask_d, Optional<TensorView> maybe_mask_indptr_d,
    Optional<TensorView> maybe_alibi_slopes_d, double logits_soft_cap_d, double sm_scale_d,
    double rope_rcp_scale_d, double rope_rcp_theta_d, bool enable_pdl);

// Batch-request prefill attention with KV-Cache operator
TVM_FFI_DLL_EXPORT_TYPED_FUNC(PODWithKVCachePlan, PODWithKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(PODWithKVCacheTensorRun, PODWithKVCacheTensorRun);
