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

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp,
                                          std::optional<torch::Tensor> alibi_slopes,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta);

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    bool use_logits_soft_cap, unsigned int head_dim, torch::Tensor empty_q_data,
    torch::Tensor empty_kv_data, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor indptr, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph);

std::vector<torch::Tensor> BatchDecodeWithPagedKVCacheRun(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, torch::Tensor q,
    std::optional<torch::Tensor> paged_kv_cache, std::optional<torch::Tensor> paged_k_cache,
    std::optional<torch::Tensor> paged_v_cache, torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices, torch::Tensor paged_kv_last_page_len,
    std::optional<torch::Tensor> alibi_slopes, unsigned int kv_layout_code, int window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_decode_with_kv_cache", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
  m.def("batch_decode_with_paged_kv_cache_plan", &BatchDecodeWithPagedKVCachePlan);
  m.def("batch_decode_with_paged_kv_cache_run", &BatchDecodeWithPagedKVCacheRun);
}
