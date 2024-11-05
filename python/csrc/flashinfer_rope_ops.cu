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

#include <vector>

void apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope, torch::Tensor k_rope,
                torch::Tensor indptr, torch::Tensor offsets, bool interleave, float rope_scale,
                float rope_theta);

void apply_llama31_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                        torch::Tensor k_rope, torch::Tensor indptr, torch::Tensor offsets,
                        bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
                        float high_freq_factor, float old_context_length);

void apply_rope_pos_ids(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                        torch::Tensor k_rope, torch::Tensor pos_ids, bool interleave,
                        float rope_scale, float rope_theta);

void apply_llama31_rope_pos_ids(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                                torch::Tensor k_rope, torch::Tensor pos_ids, bool interleave,
                                float rope_scale, float rope_theta, float low_freq_factor,
                                float high_freq_factor, float old_context_length);

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                                      torch::Tensor k_rope, torch::Tensor cos_cache,
                                      torch::Tensor sin_cache, torch::Tensor pos_ids,
                                      bool interleave);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_rope", &apply_rope, "Apply RoPE");
  m.def("apply_llama31_rope", &apply_llama31_rope, "Apply Llama 3.1 style RoPE");
  m.def("apply_rope_pos_ids", &apply_rope_pos_ids, "Apply RoPE with positional ids");
  m.def("apply_llama31_rope_pos_ids", &apply_llama31_rope_pos_ids,
        "Apply Llama 3.1 style RoPE with positional ids");
  m.def("apply_rope_pos_ids_cos_sin_cache", &apply_rope_pos_ids_cos_sin_cache,
        "Apply RoPE with positional ids and cosine/sine cache");
}
