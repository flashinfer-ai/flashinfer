#pragma once

#include <flashinfer.cuh>

#include "utils.h"

namespace cpu_reference {

template <typename T>
inline std::vector<float> apply_rotary(const T* input, size_t D, size_t offset, float rope_scale,
                                       float rope_theta) {
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

template <typename dtype_in, typename dtype_out>
std::vector<dtype_out> single_mha(
    const std::vector<dtype_in>& q, const std::vector<dtype_in>& k, const std::vector<dtype_in>& v,
    size_t num_heads, size_t seq_len, size_t head_dim,
    flashinfer::QKVLayout qkv_layout = flashinfer::QKVLayout::kNHD,
    flashinfer::RotaryMode rotary_mode = flashinfer::RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4) {
  float sm_scale = 1.f / std::sqrt(float(head_dim));
  std::vector<dtype_out> o(num_heads * head_dim);
  std::vector<float> att(num_heads * seq_len);
  std::vector<float> q_rotary_local(head_dim);
  std::vector<float> k_rotary_local(head_dim);
  auto kv_offset = [&](size_t h, size_t n, size_t d) -> size_t {
    if (qkv_layout == flashinfer::QKVLayout::kNHD) {
      return n * num_heads * head_dim + h * head_dim + d;
    } else if (qkv_layout == flashinfer::QKVLayout::kHND) {
      return h * seq_len * head_dim + n * head_dim + d;
    } else {
      std::cerr << "Unsupported qkv layout." << std::endl;
      abort();
      return 0;
    }
  };
  for (size_t i = 0; i < num_heads; ++i) {
    float max_val = -INFINITY;
    if (rotary_mode == flashinfer::RotaryMode::kApplyRotary) {
      q_rotary_local = std::move(cpu_reference::apply_rotary(q.data() + i * head_dim, head_dim,
                                                             seq_len - 1, rope_scale, rope_theta));
    }
    for (size_t j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] = 0.;
      switch (rotary_mode) {
        case flashinfer::RotaryMode::kNone: {
          for (size_t k_ = 0; k_ < head_dim; ++k_) {
            att[i * seq_len + j] +=
                float(q[i * head_dim + k_]) * float(k[kv_offset(i, j, k_)]) * sm_scale;
          }
          break;
        }
        case flashinfer::RotaryMode::kApplyRotary: {
          k_rotary_local = std::move(cpu_reference::apply_rotary(
              k.data() + kv_offset(i, j, 0), head_dim, j, rope_scale, rope_theta));
          for (size_t k_ = 0; k_ < head_dim; ++k_) {
            att[i * seq_len + j] += q_rotary_local[k_] * k_rotary_local[k_] * sm_scale;
          }
          break;
        }
        default: {
          std::cerr << "Unsupported rotary mode." << std::endl;
          abort();
        }
      }
      max_val = std::max(max_val, att[i * seq_len + j]);
    }
    // exp minus max
    float denom = 0;
    for (size_t j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] = std::exp(att[i * seq_len + j] - max_val);
      denom += att[i * seq_len + j];
    }

    // divide by denom
    for (size_t j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] /= denom;
    }

    for (size_t k_ = 0; k_ < head_dim; ++k_) {
      float o_float = 0.;
      for (size_t j = 0; j < seq_len; ++j) {
        o_float += att[i * seq_len + j] * float(v[kv_offset(i, j, k_)]);
      }
      o[i * head_dim + k_] = dtype_out(o_float);
    }
  }
  return std::move(o);
}

}  // namespace cpu_reference