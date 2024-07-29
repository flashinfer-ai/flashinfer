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

#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <stdexcept>

#include "utils.h"

namespace cpu_reference {

using namespace flashinfer;

template <typename T>
inline std::vector<T> rms_norm(const T* input, const T* weight, size_t batch_size, size_t d,
                               float eps = 1e-5) {
  std::vector<T> output(batch_size * d);
  for (size_t i = 0; i < batch_size; ++i) {
    float sum = 0;
    for (size_t j = 0; j < d; ++j) {
      sum += float(input[i * d + j]) * float(input[i * d + j]);
    }
    float rms_rcp = 1.f / (std::sqrt(sum / float(d)) + eps);
    for (size_t j = 0; j < d; ++j) {
      output[i * d + j] = (float(input[i * d + j]) * rms_rcp) * float(weight[j]);
    }
  }
  return std::move(output);
}

template <typename T>
inline std::vector<T> exclusive_prefix_sum(const T* input, size_t batch_size, size_t d) {
  std::vector<T> output(batch_size * d);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < d; ++j) {
      output[i * d + j] = (j == 0) ? 0 : output[i * d + j - 1] + input[i * d + j - 1];
    }
  }
  return std::move(output);
}

template <typename T>
inline std::vector<float> apply_llama_rope(const T* input, size_t D, size_t offset,
                                           float rope_scale, float rope_theta) {
  std::vector<float> rst(D);
  std::vector<float> permuted_input(D);
  for (size_t k = 0; k < D; ++k) {
    permuted_input[k] = (k < D / 2) ? -float(input[k + D / 2]) : float(input[k - D / 2]);
  }

  for (size_t k = 0; k < D; ++k) {
    float inv_freq =
        (offset / rope_scale) / (std::pow(rope_theta, float(2 * (k % (D / 2))) / float(D)));
    float cos = std::cos(inv_freq);
    float sin = std::sin(inv_freq);
    rst[k] = cos * float(input[k]) + sin * permuted_input[k];
  }
  return std::move(rst);
}

template <typename dtype_q, typename dtype_kv, typename dtype_out>
std::vector<dtype_out> single_mha(const std::vector<dtype_q>& q, const std::vector<dtype_kv>& k,
                                  const std::vector<dtype_kv>& v, size_t qo_len, size_t kv_len,
                                  size_t num_qo_heads, size_t num_kv_heads, size_t head_dim,
                                  bool causal = true, QKVLayout kv_layout = QKVLayout::kHND,
                                  PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                  float rope_scale = 1.f, float rope_theta = 1e4) {
  assert(qo_len <= kv_len);
  assert(num_qo_heads % num_kv_heads == 0);
  float sm_scale = 1.f / std::sqrt(float(head_dim));
  std::vector<dtype_out> o(qo_len * num_qo_heads * head_dim);
  std::vector<float> att(kv_len);
  std::vector<float> q_rotary_local(head_dim);
  std::vector<float> k_rotary_local(head_dim);
  DISPATCH_head_dim(head_dim, HEAD_DIM, {
    tensor_info_t info(qo_len, kv_len, num_qo_heads, num_kv_heads, kv_layout, HEAD_DIM);
    for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      const size_t kv_head_idx = qo_head_idx / info.get_group_size();
      for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
        float max_val = -5e4;
        if (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
          q_rotary_local = std::move(cpu_reference::apply_llama_rope(
              q.data() + info.get_q_elem_offset(q_idx, qo_head_idx, 0), head_dim,
              q_idx + kv_len - qo_len, rope_scale, rope_theta));
        }
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] = 0.;
          switch (pos_encoding_mode) {
            case PosEncodingMode::kNone: {
              for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                att[kv_idx] += float(q[info.get_q_elem_offset(q_idx, qo_head_idx, feat_idx)]) *
                               float(k[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]) *
                               sm_scale;
              }
              break;
            }
            case PosEncodingMode::kRoPELlama: {
              k_rotary_local = std::move(cpu_reference::apply_llama_rope(
                  k.data() + info.get_kv_elem_offset(kv_idx, kv_head_idx, 0), head_dim, kv_idx,
                  rope_scale, rope_theta));
              for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                att[kv_idx] += q_rotary_local[feat_idx] * k_rotary_local[feat_idx] * sm_scale;
              }
              break;
            }
            default: {
              std::ostringstream err_msg;
              err_msg << "Unsupported rotary mode.";
              throw std::invalid_argument(err_msg.str());
            }
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
          att[kv_idx] = std::exp(att[kv_idx] - max_val);
          denom += att[kv_idx];
        }

        // divide by denom
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] /= denom;
        }

        for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
          float o_float = 0.;
          for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
            o_float +=
                att[kv_idx] * float(v[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]);
          }
          o[info.get_o_elem_offset(q_idx, qo_head_idx, feat_idx)] = dtype_out(o_float);
        }
      }
    }
  });
  return std::move(o);
}

template <typename T, typename IdxType>
void append_paged_kv_cache(paged_kv_t<PageStorage::kIndices, T, IdxType> page_cpu,
                           const std::vector<std::vector<T>>& keys,
                           const std::vector<std::vector<T>>& values,
                           const std::vector<IdxType>& append_indptr) {
  size_t batch_size = page_cpu.batch_size;
  size_t num_heads = page_cpu.num_heads;
  size_t head_dim = page_cpu.head_dim;
  size_t page_size = page_cpu.page_size;
  for (size_t i = 0; i < batch_size; ++i) {
    const std::vector<T>& ki = keys[i];
    const std::vector<T>& vi = values[i];
    size_t append_seq_len = append_indptr[i + 1] - append_indptr[i];
    size_t num_pages_i = page_cpu.indptr[i + 1] - page_cpu.indptr[i];
    size_t seq_len = (num_pages_i - 1) * page_size + page_cpu.last_page_len[i];
    assert(append_seq_len <= seq_len);
    size_t append_start = seq_len - append_seq_len;

    for (size_t j = 0; j < append_seq_len; ++j) {
      size_t page_seq_idx = j + append_start;
      size_t page_idx = page_cpu.indices[page_cpu.indptr[i] + page_seq_idx / page_size];
      size_t entry_idx = page_seq_idx % page_size;
      for (size_t h = 0; h < num_heads; ++h) {
        std::copy(ki.begin() + (j * num_heads + h) * head_dim,
                  ki.begin() + (j * num_heads + h + 1) * head_dim,
                  page_cpu.k_data + page_cpu.get_elem_offset(page_idx, h, entry_idx, 0));
        std::copy(vi.begin() + (j * num_heads + h) * head_dim,
                  vi.begin() + (j * num_heads + h + 1) * head_dim,
                  page_cpu.v_data + page_cpu.get_elem_offset(page_idx, h, entry_idx, 0));
      }
    }
  }
}

}  // namespace cpu_reference
