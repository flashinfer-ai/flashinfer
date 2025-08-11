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
                at::Tensor offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta);

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta);

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length);

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids, bool interleave);

void mla_rope_quantize(at::Tensor q_rope_in, at::Tensor k_rope_in, at::Tensor q_nope_in,
                       at::Tensor k_nope_in, at::Tensor q_rope_out, at::Tensor k_rope_out,
                       at::Tensor q_nope_out, at::Tensor k_nope_out, at::Tensor cos_sin_cache,
                       at::Tensor pos_ids, double quant_scale_q, double quant_scale_kv,
                       bool interleave);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // "Apply RoPE"
  m.def("apply_rope", apply_rope);
  // "Apply Llama 3.1 style RoPE"
  m.def("apply_llama31_rope", apply_llama31_rope);
  // "Apply RoPE with positional ids"
  m.def("apply_rope_pos_ids", apply_rope_pos_ids);
  // "Apply Llama 3.1 style RoPE with positional ids"
  m.def("apply_llama31_rope_pos_ids", apply_llama31_rope_pos_ids);
  // "Apply RoPE with positional ids and cosine/sine cache"
  m.def("apply_rope_pos_ids_cos_sin_cache", apply_rope_pos_ids_cos_sin_cache);
  // "MLA RoPE Quantize"
  m.def("mla_rope_quantize", mla_rope_quantize);
}
