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
#include "pytorch_extension_utils.h"

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    at::Tensor empty_q_data, at::Tensor empty_kv_data);

void BatchDecodeWithPagedKVCacheRun(at::Tensor float_workspace_buffer,
                                    at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                    at::Tensor q, at::Tensor paged_k_cache,
                                    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
                                    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
                                    at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                    int64_t kv_layout_code, int64_t window_left,
                                    bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Batched decode with paged KV-Cache plan
  m.def("plan", BatchDecodeWithPagedKVCachePlan);
  // Batched decode with paged KV-Cache run
  m.def("run", BatchDecodeWithPagedKVCacheRun);
}
