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
#pragma once

#include <cuda_fp8.h>

#include <stdexcept>
#include <algorithm>

#include "utils.h"

namespace cpu_reference {

using namespace flashinfer;

template <typename dtype_in, typename dtype_out, typename dtype_scale = float>
void sym_quant_per_head(const std::vector<dtype_in>& x_in, std::vector<dtype_out>& x_out,
                        std::vector<dtype_scale>& s_out, size_t len, size_t num_heads,
                        size_t head_dim, QKVLayout kv_layout, bool is_q = false) {
  assert(x_in.size() == x_out.size());
  assert(s_out.size() == num_heads);
  assert(x_in.size() == len * num_heads * head_dim);

  float o_max_val = std::numeric_limits<dtype_out>::max();
  float o_min_val = std::numeric_limits<dtype_out>::lowest();

  tensor_info_t info(len, len, num_heads, num_heads, kv_layout, head_dim);
  auto offset = [&](size_t token_idx, size_t head_idx, size_t feat_idx) {
    if (is_q) {
      return info.get_q_elem_offset(token_idx, head_idx, feat_idx);
    } else {
      return info.get_kv_elem_offset(token_idx, head_idx, feat_idx);
    }
  };
  for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
    float max_val = 0;
    for (size_t token_idx = 0; token_idx < len; ++token_idx) {
      for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
        max_val = std::max(max_val, std::abs(x_in[offset(token_idx, head_idx, feat_idx)]));
      }
    }
    s_out[head_idx] = dtype_scale(max_val / o_max_val);
    for (size_t token_idx = 0; token_idx < len; ++token_idx) {
      for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
        float q_x = float(x_in[offset(token_idx, head_idx, feat_idx)]) / float(s_out[head_idx]);
        q_x = std::clamp(q_x, o_min_val, o_max_val);
        x_out[offset(token_idx, head_idx, feat_idx)] = dtype_out(q_x);
      }
    }
  }
}

template <typename dtype_in, typename dtype_out>
std::vector<dtype_out> single_mha(const std::vector<dtype_in>& q, const std::vector<dtype_in>& k,
                                  const std::vector<dtype_in>& v, size_t qo_len, size_t kv_len,
                                  size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
                                  float sm_scale, bool causal = true,
                                  QKVLayout kv_layout = QKVLayout::kHND, float rope_scale = 1.f,
                                  float rope_theta = 1e4) {
  assert(qo_len <= kv_len);
  assert(num_q_heads % num_kv_heads == 0);

  size_t group_size = num_q_heads / num_kv_heads;
  std::vector<dtype_out> o(qo_len * num_q_heads * head_dim);
  std::vector<float> att(kv_len);

  tensor_info_t info(qo_len, kv_len, num_q_heads, num_kv_heads, kv_layout, head_dim);
  for (size_t qo_head_idx = 0; qo_head_idx < num_q_heads; ++qo_head_idx) {
    const size_t kv_head_idx = qo_head_idx / group_size;
    for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
      float max_val = -5e4;
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] = 0.;
        for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
          att[kv_idx] += float(q[info.get_q_elem_offset(q_idx, qo_head_idx, feat_idx)]) *
                         float(k[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]);
        }
        // apply mask
        if (causal && kv_idx > kv_len + q_idx - qo_len) {
          att[kv_idx] = -5e4;
        }
        max_val = std::max(max_val, att[kv_idx]);
      }
      // exp minus max
      float denom = 0;
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] = std::exp(att[kv_idx] * sm_scale - max_val * sm_scale);
        denom += att[kv_idx];
      }

      // divide by denom
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] /= denom;
      }

      for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
        float o_float = 0.;
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          o_float += att[kv_idx] * float(v[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]);
        }
        o[info.get_o_elem_offset(q_idx, qo_head_idx, feat_idx)] = dtype_out(o_float);
      }
    }
  }
  return std::move(o);
}

template <typename dtype_in, typename dtype_out, typename dtype_scale>
std::vector<dtype_out> single_fp8_mha(
    const std::vector<dtype_in>& q, const std::vector<dtype_in>& k, const std::vector<dtype_in>& v,
    const std::vector<dtype_scale>& q_scale, const std::vector<dtype_scale>& k_scale,
    const std::vector<dtype_scale>& v_scale, size_t qo_len, size_t kv_len, size_t num_q_heads,
    size_t num_kv_heads, size_t head_dim, float sm_scale, bool causal = true,
    QKVLayout kv_layout = QKVLayout::kHND, float rope_scale = 1.f, float rope_theta = 1e4) {
  static_assert(sizeof(dtype_in) == 1);
  float p_fp8_scale = std::numeric_limits<dtype_in>::max();

  assert(qo_len <= kv_len);
  assert(num_q_heads % num_kv_heads == 0);
  assert(q_scale.size() == num_q_heads);
  assert(k_scale.size() == num_kv_heads);
  assert(v_scale.size() == num_kv_heads);

  size_t group_size = num_q_heads / num_kv_heads;
  std::vector<dtype_out> o(qo_len * num_q_heads * head_dim);
  std::vector<float> att(kv_len);

  tensor_info_t info(qo_len, kv_len, num_q_heads, num_kv_heads, kv_layout, head_dim);
  for (size_t qo_head_idx = 0; qo_head_idx < num_q_heads; ++qo_head_idx) {
    const size_t kv_head_idx = qo_head_idx / group_size;
    for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
      float max_val = -5e4;
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] = 0.;
        for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
          att[kv_idx] += float(q[info.get_q_elem_offset(q_idx, qo_head_idx, feat_idx)]) *
                         float(k[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]);
        }

        // apply mask
        if (causal && kv_idx > kv_len + q_idx - qo_len) {
          att[kv_idx] = -5e4;
        }
        max_val = std::max(max_val, att[kv_idx]);
      }
      // exp minus max
      float denom = 0;
      float sm_scale_fused_dequantize_log2 =
          sm_scale * float(q_scale[qo_head_idx]) * float(k_scale[kv_head_idx]);
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] = std::exp(att[kv_idx] * sm_scale_fused_dequantize_log2 -
                               max_val * sm_scale_fused_dequantize_log2);
        denom += att[kv_idx];
      }

      // divide by denom
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] /= denom;
      }

      // Requantize
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        att[kv_idx] *= float(p_fp8_scale);
        att[kv_idx] = float(dtype_in(att[kv_idx]));
      }

      for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
        float o_float = 0.;
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          o_float += att[kv_idx] * float(v[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]);
        }
        // Dequantize
        o_float *= float(v_scale[kv_head_idx]) / float(p_fp8_scale);
        o[info.get_o_elem_offset(q_idx, qo_head_idx, feat_idx)] = dtype_out(o_float);
      }
    }
  }

  return std::move(o);
}

}  // namespace cpu_reference
