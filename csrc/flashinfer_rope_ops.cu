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
#include <vector>

#include "pytorch_extension_utils.h"

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, unsigned int rotary_dim, bool interleave, float rope_scale,
                float rope_theta, int64_t cuda_stream);

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, unsigned int rotary_dim,
                        bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
                        float high_freq_factor, float old_context_length, int64_t cuda_stream);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                        float rope_scale, float rope_theta, int64_t cuda_stream);

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                                float rope_scale, float rope_theta, float low_freq_factor,
                                float high_freq_factor, float old_context_length,
                                int64_t cuda_stream);

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids, bool interleave, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_rope", &apply_rope, "Apply RoPE");
  m.def("apply_llama31_rope", &apply_llama31_rope, "Apply Llama 3.1 style RoPE");
  m.def("apply_rope_pos_ids", &apply_rope_pos_ids, "Apply RoPE with positional ids");
  m.def("apply_llama31_rope_pos_ids", &apply_llama31_rope_pos_ids,
        "Apply Llama 3.1 style RoPE with positional ids");
  m.def("apply_rope_pos_ids_cos_sin_cache", &apply_rope_pos_ids_cos_sin_cache,
        "Apply RoPE with positional ids and cosine/sine cache");
}
