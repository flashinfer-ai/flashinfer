/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <torch/extension.h>

torch::Tensor single_prefill_with_kv_cache(
    unsigned int mask_mode_code, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    std::optional<torch::Tensor> maybe_packed_custom_mask, torch::Tensor tmp,
    std::optional<torch::Tensor> maybe_alibi_slopes, unsigned int layout, int32_t window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<torch::Tensor> maybe_lse);

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    unsigned int head_dim, torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer, torch::Tensor qo_indptr,
    torch::Tensor kv_indptr, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph);

torch::Tensor BatchPrefillWithRaggedKVCacheRun(
    unsigned int mask_mode_code, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, std::vector<int64_t> plan_info_vec, torch::Tensor q,
    torch::Tensor k, torch::Tensor v, std::optional<torch::Tensor> maybe_custom_mask,
    std::optional<torch::Tensor> maybe_alibi_slopes, torch::Tensor qo_indptr,
    torch::Tensor kv_indptr, std::optional<torch::Tensor> maybe_qk_indptr, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<torch::Tensor> maybe_lse);

torch::Tensor BatchPrefillWithPagedKVCacheRun(
    unsigned int mask_mode_code, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, std::vector<int64_t> plan_info_vec, torch::Tensor q,
    torch::Tensor paged_k_cache, torch::Tensor paged_v_cache,
    std::optional<torch::Tensor> maybe_custom_mask, std::optional<torch::Tensor> maybe_alibi_slopes,
    torch::Tensor qo_indptr, torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, std::optional<torch::Tensor> maybe_qk_indptr,
    unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<torch::Tensor> maybe_lse);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
  m.def("batch_prefill_with_kv_cache_plan", &BatchPrefillWithKVCachePlan);
  m.def("batch_prefill_with_ragged_kv_cache_run", &BatchPrefillWithRaggedKVCacheRun);
  m.def("batch_prefill_with_paged_kv_cache_run", &BatchPrefillWithPagedKVCacheRun);
}
