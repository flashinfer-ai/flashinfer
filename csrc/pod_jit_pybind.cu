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
#include "pytorch_extension_utils.h"

void PODWithKVCacheTensorRun(
    // Shared params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
    at::Tensor qo_indptr, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    int64_t layout,
    // Prefill params
    at::Tensor q_p, int64_t mask_mode_code_p, int64_t window_left_p,
    std::optional<at::Tensor> maybe_custom_mask_p, std::optional<at::Tensor> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor q_d, int64_t mask_mode_code_d, int64_t window_left_d,
    std::optional<at::Tensor> maybe_custom_mask_d, std::optional<at::Tensor> maybe_mask_indptr_d,
    std::optional<at::Tensor> maybe_alibi_slopes_d, double logits_soft_cap_d, double sm_scale_d,
    double rope_rcp_scale_d, double rope_rcp_theta_d, bool enable_pdl);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Batch-request prefill attention with KV-Cache operator
  m.def("PODWithKVCacheTensor", PODWithKVCacheTensorRun);
}
