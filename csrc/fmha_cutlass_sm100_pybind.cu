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
                         at::Tensor qo_lens, at::Tensor kv_lens, at::Tensor qo_segment_offsets,
                         at::Tensor kv_segment_offsets, at::Tensor o,
                         std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                         double sm_scale, int64_t num_qo_heads, int64_t num_kv_heads,
                         int64_t head_dim_qk, int64_t head_dim_vo, int64_t batch_size,
                         int64_t total_qo_len, int64_t total_kv_len, int64_t max_qo_len,
                         int64_t max_kv_len);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("run", FMHACutlassSM100Run); }
