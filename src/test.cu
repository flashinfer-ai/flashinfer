#include <gtest/gtest.h>

#include <flashinfer.cuh>

#include "utils.cuh"

template <typename dtype_in, typename dtype_out>
thrust::host_vector<dtype_out> cpu_mha_reference(thrust::host_vector<dtype_in> q,
                                                 thrust::host_vector<dtype_in> k,
                                                 thrust::host_vector<dtype_in> v, int num_heads,
                                                 int seq_len, int head_dim) {
  thrust::host_vector<dtype_out> o(num_heads * head_dim);
  thrust::host_vector<float> att(num_heads * seq_len);
  for (int i = 0; i < num_heads; ++i) {
    float max_val = -MAXFLOAT;
    for (int j = 0; j < seq_len; ++j) {
      att[i * seq_len + j] = 0.;
      for (int k_ = 0; k_ < head_dim; ++k_) {
        att[i * seq_len + j] +=
            float(q[i * head_dim + k_]) * float(k[j * num_heads * head_dim + i * head_dim + k_]);
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

void _TestDecodingKernelCorrectness(int num_heads, int seq_len, int head_dim) {
  thrust::device_vector<half> Q(num_heads * head_dim);
  thrust::device_vector<half> K(seq_len * num_heads * head_dim);
  thrust::device_vector<half> V(seq_len * num_heads * head_dim);
  thrust::device_vector<half> O(num_heads * head_dim);

  thrust::device_vector<float> m_global(num_heads * seq_len);
  thrust::device_vector<float> d_global(num_heads * seq_len);
  thrust::device_vector<int> mutex(num_heads * seq_len);

  utils::thrust_normal_init(Q);
  utils::thrust_normal_init(K);
  utils::thrust_normal_init(V);
  utils::thrust_zero_init(O);
  thrust::fill(m_global.begin(), m_global.end(), -10000);
  thrust::fill(d_global.begin(), d_global.end(), 0.f);
  thrust::fill(mutex.begin(), mutex.end(), 0);

  thrust::host_vector<half> o_ref_host =
      cpu_mha_reference<half, half>(thrust::host_vector<half>(Q), thrust::host_vector<half>(K),
                                    thrust::host_vector<half>(V), num_heads, seq_len, head_dim);

  flashinfer::decoding_dispatch(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
      thrust::raw_pointer_cast(m_global.data()), thrust::raw_pointer_cast(d_global.data()),
      thrust::raw_pointer_cast(mutex.data()), num_heads, seq_len, head_dim);

  thrust::host_vector<half> o_host = O;
  thrust::host_vector<float> m_host = m_global;
  thrust::host_vector<float> d_host = d_global;

  int num_errors_atol_1e_3_rtol_1e_3 = 0;
  for (int i = 0; i < num_heads * head_dim; ++i) {
    num_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref_host[i]), 1e-3, 1e-3));
  }
  float accurate_rate = 1. - float(num_errors_atol_1e_3_rtol_1e_3) / float(num_heads * head_dim);
  std::cout << "num_heads=" << num_heads << ", seq_len=" << seq_len << ", head_dim=" << head_dim
            << ", accuracy (atol=1e-3, rtol=1e-3): " << accurate_rate << std::endl;
  EXPECT_GT(accurate_rate, 0.98) << "Correctness test failed.";
}

TEST(FlashInferCorrectnessTest, DecodingKernelCorrectnessTest) {
  for (int num_heads : {32}) {
    for (int seq_len : {129, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
      //   for (int head_dim : {64, 128, 256}) {
      for (int head_dim : {128}) {
        _TestDecodingKernelCorrectness(num_heads, seq_len, head_dim);
      }
    }
  }
}