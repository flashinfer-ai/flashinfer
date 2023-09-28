#include <gtest/gtest.h>

#include <flashinfer/prefill.cuh>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <typename DTypeIn, typename DTypeOut>
void _TestSinglePrefillKernelCorrectness(size_t qo_len, size_t kv_len, size_t num_heads,
                                         size_t head_dim, QKVLayout layout,
                                         RotaryMode rotary_mode) {
  std::vector<DTypeIn> q(qo_len * num_heads * head_dim);
  std::vector<DTypeIn> k(kv_len * num_heads * head_dim);
  std::vector<DTypeIn> v(kv_len * num_heads * head_dim);
  std::vector<DTypeOut> o(qo_len * num_heads * head_dim);

  utils::vec_normal_(q);
  utils::vec_normal_(k);
  utils::vec_normal_(v);
  utils::vec_zero_(o);

  thrust::device_vector<DTypeIn> q_d(q);
  thrust::device_vector<DTypeIn> k_d(k);
  thrust::device_vector<DTypeIn> v_d(v);
  thrust::device_vector<DTypeOut> o_d(o);
  thrust::device_vector<float> tmp_d(kv_len * num_heads * head_dim);

  cudaError_t status = flashinfer::SinglePrefillWithKVCache<DTypeIn, DTypeOut>(
      thrust::raw_pointer_cast(q_d.data()), thrust::raw_pointer_cast(k_d.data()),
      thrust::raw_pointer_cast(v_d.data()), thrust::raw_pointer_cast(o_d.data()),
      thrust::raw_pointer_cast(tmp_d.data()), num_heads, qo_len, kv_len, head_dim, layout,
      rotary_mode);

  thrust::host_vector<DTypeOut> o_h(o_d);
  std::vector<DTypeOut> o_ref = cpu_reference::single_mha<DTypeIn, DTypeOut>(
      q, k, v, qo_len, kv_len, num_heads, head_dim, layout, rotary_mode);
  size_t num_results_error_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;

  for (size_t i = 0; i < o_ref.size(); ++i) {
    if (isnan(float(o_h[i]))) {
      nan_detected = true;
    }
    num_results_error_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_ref[i]), float(o_h[i]), 1e-3, 1e-3));
    if (!utils::isclose(float(o_ref[i]), float(o_h[i]), 1e-3, 1e-3)) {
      std::cout << "row=" << i / head_dim << ", rol=" << i % head_dim << ", ref=" << float(o_ref[i])
                << ", result=" << float(o_h[i]) << std::endl;
    }
  }

  float result_accuracy = 1. - float(num_results_error_atol_1e_3_rtol_1e_3) / float(o_ref.size());
  std::cout << "num_heads=" << num_heads << ", qo_len=" << qo_len << ", kv_len=" << kv_len
            << ", head_dim=" << head_dim << ", layout=" << QKVLayoutToString(layout)
            << ", rotary_mode=" << RotaryModeToString(rotary_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_FALSE(nan_detected) << "Nan detected in the result.";
}

template <typename DTypeIn, typename DTypeOut>
void TestSinglePrefillKernelCorrectness() {
  for (size_t qo_len : {399, 400, 401}) {
    for (size_t kv_len : {533, 534, 535}) {
      for (size_t num_heads : {32}) {
        for (size_t head_dim : {64, 128}) {
          for (size_t rotary_mode : {0, 1}) {
            for (size_t layout : {0, 1}) {
              _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeOut>(
                  qo_len, kv_len, num_heads, head_dim, QKVLayout(layout), RotaryMode(rotary_mode));
            }
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, SinglePrefillKernelCorrectnessTestFP16) {
  TestSinglePrefillKernelCorrectness<half, half>();
}

TEST(FlashInferCorrectnessTest, SinglePrefillKernelCorrectnessTestBF16) {
  TestSinglePrefillKernelCorrectness<nv_bfloat16, nv_bfloat16>();
}