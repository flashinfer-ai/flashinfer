#include <gtest/gtest.h>

#include <flashinfer.cuh>

#include "utils.cuh"

template <typename T>
inline void cpu_inplace_apply_rotary(thrust::host_vector<T>& input, int offset, float inv_ratio) {
  int D = input.size();
  thrust::host_vector<T> permuted_input(input);
  for (int k = 0; k < D; ++k) {
    permuted_input[k] = (k < D / 2) ? -float(input[k + D / 2]) : float(input[k - D / 2]);
  }

  for (int k = 0; k < D; ++k) {
    float inv_freq = (offset * inv_ratio) * (std::pow(1e-4, float(2 * (k % (D / 2))) / float(D)));
    float cos = std::cos(inv_freq);
    float sin = std::sin(inv_freq);
    input[k] = cos * float(input[k]) + sin * float(permuted_input[k]);
  }
}

template <typename dtype_in, typename dtype_out>
thrust::host_vector<dtype_out> cpu_mha_reference(
    const thrust::host_vector<dtype_in>& q, thrust::host_vector<dtype_in>& k,
    const thrust::host_vector<dtype_in>& v, int num_heads, int seq_len, int head_dim,
    float sm_scale = 1.f, flashinfer::RotaryMode rotary_mode = flashinfer::RotaryMode::kNone,
    float rotary_pi_inv_ratio = 1.f) {
  thrust::host_vector<dtype_out> o(num_heads * head_dim);
  thrust::host_vector<float> att(num_heads * seq_len);
  thrust::host_vector<dtype_in> q_rotary_local(head_dim);
  thrust::host_vector<dtype_in> k_rotary_local(head_dim);
  for (int i = 0; i < num_heads; ++i) {
    float max_val = -INFINITY;
    if (rotary_mode == flashinfer::RotaryMode::kApplyRotary ||
        rotary_mode == flashinfer::RotaryMode::kApplyRotaryUpdateLastK) {
      thrust::copy_n(thrust::host, q.begin() + i * head_dim, head_dim, q_rotary_local.begin());
      cpu_inplace_apply_rotary(q_rotary_local, seq_len - 1, rotary_pi_inv_ratio);
    }
    for (int j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] = 0.;
      switch (rotary_mode) {
        case flashinfer::RotaryMode::kNone: {
          for (int k_ = 0; k_ < head_dim; ++k_) {
            att[i * seq_len + j] += float(q[i * head_dim + k_]) *
                                    float(k[j * num_heads * head_dim + i * head_dim + k_]) *
                                    sm_scale;
          }
          break;
        }
        case flashinfer::RotaryMode::kApplyRotary: {
          thrust::copy_n(thrust::host, k.begin() + j * num_heads * head_dim + i * head_dim,
                         head_dim, k_rotary_local.begin());
          cpu_inplace_apply_rotary(k_rotary_local, j, rotary_pi_inv_ratio);
          for (int k_ = 0; k_ < head_dim; ++k_) {
            att[i * seq_len + j] +=
                float(q_rotary_local[k_]) * float(k_rotary_local[k_]) * sm_scale;
          }
          break;
        }
        case flashinfer::RotaryMode::kApplyRotaryUpdateLastK: {
          thrust::copy_n(thrust::host, k.begin() + j * num_heads * head_dim + i * head_dim,
                         head_dim, k_rotary_local.begin());
          if (j == seq_len - 1) {
            cpu_inplace_apply_rotary(k_rotary_local, j, rotary_pi_inv_ratio);
            thrust::copy_n(thrust::host, k_rotary_local.begin(), head_dim,
                           k.begin() + j * num_heads * head_dim + i * head_dim);
          }
          for (int k_ = 0; k_ < head_dim; ++k_) {
            att[i * seq_len + j] +=
                float(q_rotary_local[k_]) * float(k_rotary_local[k_]) * sm_scale;
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
    for (int j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] = std::exp(att[i * seq_len + j] - max_val);
      denom += att[i * seq_len + j];
    }
    // divide by denom
    for (int j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] /= denom;
    }

    for (int k_ = 0; k_ < head_dim; ++k_) {
      float o_float = 0.;
      for (int j = 0; j < seq_len; ++j) {
        o_float += att[i * seq_len + j] * float(v[j * num_heads * head_dim + i * head_dim + k_]);
      }
      o[i * head_dim + k_] = dtype_out(o_float);
    }
  }
  return std::move(o);
}

template <typename T>
void _TestDecodingKernelCorrectness(int num_heads, int seq_len, int head_dim,
                                    flashinfer::RotaryMode rotary_mode) {
  thrust::device_vector<T> Q(num_heads * head_dim);
  thrust::device_vector<T> K(seq_len * num_heads * head_dim);
  thrust::device_vector<T> V(seq_len * num_heads * head_dim);
  thrust::device_vector<T> O(num_heads * head_dim);

  thrust::device_vector<float> m_global(num_heads * head_dim);
  thrust::device_vector<float> d_global(num_heads * head_dim);
  thrust::device_vector<int> mutex(num_heads * head_dim);

  // utils::thrust_normal_init(Q);
  utils::thrust_zero_init(Q);
  utils::thrust_normal_init(K);
  utils::thrust_normal_init(V);
  thrust::fill(m_global.begin(), m_global.end(), -INFINITY);
  thrust::fill(d_global.begin(), d_global.end(), 0.f);
  thrust::fill(mutex.begin(), mutex.end(), 0);

  thrust::host_vector<T> K_ref_host = K;
  thrust::host_vector<T> o_ref_host =
      cpu_mha_reference<T, T>(thrust::host_vector<T>(Q), K_ref_host, thrust::host_vector<T>(V),
                              num_heads, seq_len, head_dim, 1.f, rotary_mode);

  flashinfer::SingleDecodeWithKVCache(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
      thrust::raw_pointer_cast(m_global.data()), thrust::raw_pointer_cast(d_global.data()),
      thrust::raw_pointer_cast(mutex.data()), num_heads, seq_len, head_dim, 1.f, rotary_mode);

  thrust::host_vector<T> o_host = O;
  thrust::host_vector<T> K_host = K;
  thrust::host_vector<float> m_host = m_global;
  thrust::host_vector<float> d_host = d_global;

  int num_result_errors_atol_1e_3_rtol_1e_3 = 0, num_updated_k_errors_atol_1e_3_rtol_1e_3 = 0;
  for (int i = 0; i < num_heads * head_dim; ++i) {
    num_updated_k_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(K_host[(seq_len - 1) * num_heads * head_dim + i]),
                         float(K_ref_host[(seq_len - 1) * num_heads * head_dim + i]), 1e-3, 1e-3));
  }
  for (int i = 0; i < num_heads * head_dim; ++i) {
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref_host[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
            1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_heads * head_dim),
        updated_k_accuracy =
            1. - float(num_updated_k_errors_atol_1e_3_rtol_1e_3) / float(num_heads * head_dim);
  std::cout << "num_heads=" << num_heads << ", seq_len=" << seq_len << ", head_dim=" << head_dim
            << ", rotary_mode=" << flashinfer::RotaryModeToString(rotary_mode)
            << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << " "
            << ", updated_k accuracy (atol=1e-3, rtol=1e-3): " << updated_k_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_GT(updated_k_accuracy, 0.90) << "Updated K correctness test failed.";
}

TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTest) {
  for (int num_heads : {32}) {
    for (int seq_len : {1, 3, 9, 27, 81, 129, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
      for (int head_dim : {64, 128, 256}) {
        for (unsigned int rotary_mode : {0U, 1U, 2U}) {
          _TestDecodingKernelCorrectness<half>(num_heads, seq_len, head_dim,
                                               flashinfer::RotaryMode(rotary_mode));
        }
      }
    }
  }
}