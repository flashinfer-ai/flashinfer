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
#include "batch_mla_config.inc"
#include "pytorch_extension_utils.h"

std::vector<int64_t> BatchMLAPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                                at::Tensor int_workspace_buffer,
                                                at::Tensor page_locked_int_workspace_buffer,
                                                at::Tensor qo_indptr, at::Tensor kv_indptr,
                                                at::Tensor kv_len, unsigned int num_heads,
                                                unsigned int head_dim_o, bool causal,
                                                int64_t cuda_stream);

void BatchMLAPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                               std::vector<int64_t> plan_info_vec, at::Tensor q_nope,
                               at::Tensor q_pe, at::Tensor ckv_cache, at::Tensor kpe_cache,
                               at::Tensor kv_indices, at::Tensor o,
                               std::optional<at::Tensor> maybe_lse, int mask_mode_code,
                               int num_heads, int page_size, float sm_scale, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchMLAPagedAttentionPlan, "Batch MLA Page Attention Plan");
  m.def("run", &BatchMLAPagedAttentionRun, "Batch MLA Page Attention Run");
}
