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
#include "aot_extension_utils.h"

//========== decode ==========

void single_decode_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                 std::optional<at::Tensor> alibi_slopes, at::Tensor o,
                                 unsigned int layout, int window_left, float logits_soft_cap,
                                 float sm_scale, float rope_scale, float rope_theta,
                                 int64_t cuda_stream);

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    bool use_logits_soft_cap, unsigned int head_dim, at::Tensor empty_q_data,
    at::Tensor empty_kv_data, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph, int64_t cuda_stream);

void BatchDecodeWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, std::optional<at::Tensor> alibi_slopes, at::Tensor o,
    unsigned int kv_layout_code, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

//========== prefill ==========

void single_prefill_with_kv_cache(unsigned int mask_mode_code, at::Tensor q, at::Tensor k,
                                  at::Tensor v, std::optional<at::Tensor> maybe_packed_custom_mask,
                                  at::Tensor tmp, std::optional<at::Tensor> maybe_alibi_slopes,
                                  at::Tensor o, unsigned int layout, int32_t window_left,
                                  float logits_soft_cap, float sm_scale, float rope_scale,
                                  float rope_theta, std::optional<at::Tensor> maybe_lse,
                                  int64_t cuda_stream);

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    unsigned int head_dim, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    unsigned total_num_rows, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph, int64_t cuda_stream);

void BatchPrefillWithRaggedKVCacheRun(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes,
    at::Tensor qo_indptr, at::Tensor kv_indptr, std::optional<at::Tensor> maybe_qk_indptr,
    at::Tensor o, unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

void BatchPrefillWithPagedKVCacheRun(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, std::optional<at::Tensor> maybe_custom_mask,
    std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    std::optional<at::Tensor> maybe_qk_indptr, at::Tensor o, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

//========== pybind11 ==========

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // decode
  m.def("single_decode_with_kv_cache", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
  m.def("batch_decode_with_paged_kv_cache_plan", &BatchDecodeWithPagedKVCachePlan);
  m.def("batch_decode_with_paged_kv_cache_run", &BatchDecodeWithPagedKVCacheRun);

  // prefill
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
  m.def("batch_prefill_with_kv_cache_plan", &BatchPrefillWithKVCachePlan);
  m.def("batch_prefill_with_ragged_kv_cache_run", &BatchPrefillWithRaggedKVCacheRun);
  m.def("batch_prefill_with_paged_kv_cache_run", &BatchPrefillWithPagedKVCacheRun);
}
