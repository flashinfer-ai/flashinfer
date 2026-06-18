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
#include "batch_pod_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

void batch_pod_with_kv_cache_tensor(
    // Prefill params
    TensorView float_workspace_buffer_p, TensorView int_workspace_buffer_p,
    Array<int64_t> plan_info_vec_p, TensorView q_p, TensorView paged_k_cache_p,
    TensorView paged_v_cache_p, TensorView qo_indptr_p, TensorView paged_kv_indptr_p,
    TensorView paged_kv_indices_p, TensorView paged_kv_last_page_len_p, TensorView o_p,
    Optional<TensorView> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, Optional<TensorView> maybe_custom_mask_p,
    Optional<TensorView> maybe_mask_indptr_p, Optional<TensorView> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    TensorView float_workspace_buffer_d, TensorView int_workspace_buffer_d,
    Array<int64_t> plan_info_vec_d, TensorView q_d, TensorView paged_k_cache_d,
    TensorView paged_v_cache_d, TensorView qo_indptr_d, TensorView paged_kv_indptr_d,
    TensorView paged_kv_indices_d, TensorView paged_kv_last_page_len_d, TensorView o_d,
    Optional<TensorView> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, Optional<TensorView> maybe_custom_mask_d,
    Optional<TensorView> maybe_mask_indptr_d, Optional<TensorView> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl, TensorView sm_aware_sched);

// Batch-request prefill attention with KV-Cache operator
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_pod_with_kv_cache_tensor, batch_pod_with_kv_cache_tensor);
