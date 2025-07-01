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
#include "batch_attention_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                   at::Tensor int_workspace_buffer,
                                   at::Tensor page_locked_int_workspace_buffer,
                                   at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len,
                                   int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
                                   int64_t head_dim_o, bool causal);

void BatchPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor plan_info_vec, at::Tensor q, at::Tensor k_cache,
                            at::Tensor v_cache, at::Tensor kv_indices, at::Tensor o,
                            std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                            int64_t layout_code, int64_t num_qo_heads, int64_t num_kv_heads,
                            int64_t page_size,
                            double sm_scale ADDITIONAL_FUNC_PARAMS PROFILER_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchPagedAttentionPlan);
  m.def("run", &BatchPagedAttentionRun);
}
