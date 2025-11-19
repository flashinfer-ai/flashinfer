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

void pod_with_kv_cache_tensor(
    // Prefill params
    TensorView q_p, TensorView k_p, TensorView v_p, TensorView tmp_p, TensorView o_p,
    Optional<TensorView> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, Optional<TensorView> maybe_custom_mask_p,
    Optional<TensorView> maybe_alibi_slopes_p, double logits_soft_cap_p, double sm_scale_p,
    double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    TensorView float_workspace_buffer_d, TensorView int_workspace_buffer_d,
    Array<int64_t> plan_info_vec, TensorView q_d, TensorView paged_k_cache_d,
    TensorView paged_v_cache_d, TensorView qo_indptr_d, TensorView paged_kv_indptr_d,
    TensorView paged_kv_indices_d, TensorView paged_kv_last_page_len_d, TensorView o_d,
    Optional<TensorView> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, Optional<TensorView> maybe_custom_mask_d,
    Optional<TensorView> maybe_mask_indptr_d, Optional<TensorView> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl);

// Single prefill, Batch-request decode attention with KV-Cache operator
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pod_with_kv_cache_tensor, pod_with_kv_cache_tensor);
