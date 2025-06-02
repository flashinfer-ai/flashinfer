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
#include "pytorch_extension_utils.h"

void FMHACutlassSM100Run(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                         at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets,
                         at::Tensor work_indptr, at::Tensor qo_tile_indices,
                         at::Tensor qo_head_indices, at::Tensor batch_indices, at::Tensor o,
                         std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                         double sm_scale, int64_t num_qo_heads, int64_t num_kv_heads,
                         int64_t head_dim_qk, int64_t head_dim_vo, int64_t max_qo_len);

void blackwell_fmha_plan(at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets,
                         at::Tensor work_indptr, at::Tensor qo_tile_indices,
                         at::Tensor head_indices, at::Tensor batch_indices, int64_t qo_tile_size,
                         int64_t num_heads, int64_t num_buckets, bool causal);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("run", FMHACutlassSM100Run);
  m.def("plan", blackwell_fmha_plan);
}
