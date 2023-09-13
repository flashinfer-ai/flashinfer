#include <gtest/gtest.h>

#include <flashinfer.cuh>
#include <type_traits>

#include "cpu_reference.h"
#include "utils.h"

template <typename T>
void _TestDecodingKernelCorrectness(size_t num_heads, size_t seq_len, size_t head_dim,
                                    flashinfer::QKVLayout qkv_layout,
                                    flashinfer::RotaryMode rotary_mode) {
  std::vector<T> Q_host(num_heads * head_dim);
  std::vector<T> K_host(seq_len * num_heads * head_dim);
  std::vector<T> V_host(seq_len * num_heads * head_dim);
  std::vector<T> O_host(num_heads * head_dim);

  utils::vec_normal_(Q_host);
  utils::vec_normal_(K_host);
  utils::vec_normal_(V_host);
  utils::vec_zero_(O_host);

  thrust::device_vector<T> Q(Q_host);
  thrust::device_vector<T> K(K_host);
  thrust::device_vector<T> V(V_host);
  thrust::device_vector<T> O(O_host);
  thrust::device_vector<float> tmp(512 * num_heads * head_dim);

  std::vector<T> o_ref_host = cpu_reference::single_mha<T, T>(
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
void TestSingleDecodeKernelCorrectness() {
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

TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestBF16) {
  TestSingleDecodeKernelCorrectness<nv_bfloat16>();
}
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestFP16) {
  TestSingleDecodeKernelCorrectness<half>();
}
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestE4M3) {
  TestSingleDecodeKernelCorrectness<__nv_fp8_e4m3>();
}
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestE5M2) {
  TestSingleDecodeKernelCorrectness<__nv_fp8_e5m2>();
}
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestFP32) {
  TestSingleDecodeKernelCorrectness<float>();
}