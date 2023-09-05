#include <gtest/gtest.h>

#include <flashinfer.cuh>
#include <type_traits>

#include "utils.cuh"

template <typename T>
inline thrust::host_vector<float> cpu_apply_rotary(const T* input, size_t D, size_t offset,
                                                   float rope_scale, float rope_theta) {
  thrust::host_vector<float> rst(D);
  thrust::host_vector<float> permuted_input(D);
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
thrust::host_vector<dtype_out> cpu_mha_reference(
    const thrust::host_vector<dtype_in>& q, thrust::host_vector<dtype_in>& k,
    const thrust::host_vector<dtype_in>& v, size_t num_heads, size_t seq_len, size_t head_dim,
    flashinfer::QKVLayout qkv_layout = flashinfer::QKVLayout::kNHD,
    flashinfer::RotaryMode rotary_mode = flashinfer::RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4) {
  float sm_scale = 1.f / std::sqrt(float(head_dim));
  thrust::host_vector<dtype_out> o(num_heads * head_dim);
  thrust::host_vector<float> att(num_heads * seq_len);
  thrust::host_vector<float> q_rotary_local(head_dim);
  thrust::host_vector<float> k_rotary_local(head_dim);
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
      q_rotary_local = std::move(cpu_apply_rotary(thrust::raw_pointer_cast(q.data()) + i * head_dim,
                                                  head_dim, seq_len - 1, rope_scale, rope_theta));
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
          k_rotary_local =
              std::move(cpu_apply_rotary(thrust::raw_pointer_cast(k.data()) + kv_offset(i, j, 0),
                                         head_dim, j, rope_scale, rope_theta));
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

template <typename T>
void _TestDecodingKernelCorrectness(size_t num_heads, size_t seq_len, size_t head_dim,
                                    flashinfer::QKVLayout qkv_layout,
                                    flashinfer::RotaryMode rotary_mode) {
  thrust::device_vector<T> Q(num_heads * head_dim);
  thrust::device_vector<T> K(seq_len * num_heads * head_dim);
  thrust::device_vector<T> V(seq_len * num_heads * head_dim);
  thrust::device_vector<T> O(num_heads * head_dim);

  thrust::device_vector<float> tmp(512 * num_heads * head_dim);

  utils::thrust_normal_init(Q);
  utils::thrust_normal_init(K);
  utils::thrust_normal_init(V);
  utils::thrust_zero_init(O);

  auto Q_host = thrust::host_vector<T>(Q);
  auto K_host = thrust::host_vector<T>(K);
  auto V_host = thrust::host_vector<T>(V);

  thrust::host_vector<T> o_ref_host = cpu_mha_reference<T, T>(
      Q_host, K_host, V_host, num_heads, seq_len, head_dim, qkv_layout, rotary_mode);

  flashinfer::SingleDecodeWithKVCache(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
      thrust::raw_pointer_cast(tmp.data()), num_heads, seq_len, head_dim, qkv_layout, rotary_mode);

  thrust::host_vector<T> o_host = O;
  thrust::host_vector<float> tmp_host = tmp;

  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < num_heads * head_dim; ++i) {
    if (isnan(float(o_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref_host[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_heads * head_dim);
  std::cout << "num_heads=" << num_heads << ", seq_len=" << seq_len << ", head_dim=" << head_dim
            << ", qkv_layout=" << flashinfer::QKVLayoutToString(qkv_layout)
            << ", rotary_mode=" << flashinfer::RotaryModeToString(rotary_mode)
            << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected.";
}

template <typename T>
void TestDecodeKernelCorrectness() {
  for (size_t num_heads : {32}) {
    for (size_t seq_len : {1, 3, 9, 27, 81, 129, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
      for (size_t head_dim : {64, 128, 256}) {
        for (unsigned int qkv_layout : {0U, 1U}) {
          for (unsigned int rotary_mode : {0U, 1U}) {
            _TestDecodingKernelCorrectness<T>(num_heads, seq_len, head_dim,
                                              flashinfer::QKVLayout(qkv_layout),
                                              flashinfer::RotaryMode(rotary_mode));
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTestBF16) {
  TestDecodeKernelCorrectness<nv_bfloat16>();
}
TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTestFP16) {
  TestDecodeKernelCorrectness<half>();
}
TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTestE4M3) {
  TestDecodeKernelCorrectness<__nv_fp8_e4m3>();
}
TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTestE5M2) {
  TestDecodeKernelCorrectness<__nv_fp8_e5m2>();
}
TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTestFP32) {
  TestDecodeKernelCorrectness<float>();
}