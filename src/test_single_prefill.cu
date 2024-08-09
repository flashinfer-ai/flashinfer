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
#include <gtest/gtest.h>

#include <cstdint>

#include "cpu_reference.h"
#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;

template <typename DTypeQ, typename DTypeKV, typename DTypeOut>
void _TestSinglePrefillKernelCorrectness(size_t qo_len, size_t kv_len, size_t num_qo_heads,
                                         size_t num_kv_heads, size_t head_dim, bool causal,
                                         QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
                                         bool allow_fp16_qk_reduction, float rtol = 1e-3,
                                         float atol = 1e-3) {
  std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeOut> o(qo_len * num_qo_heads * head_dim);

  utils::vec_normal_(q);
  utils::vec_normal_(k);
  utils::vec_normal_(v);
  utils::vec_zero_(o);

  thrust::device_vector<DTypeQ> q_d(q);
  thrust::device_vector<DTypeKV> k_d(k);
  thrust::device_vector<DTypeKV> v_d(v);
  thrust::device_vector<DTypeOut> o_d(o);
  thrust::device_vector<DTypeOut> tmp_d(16 * 1024 * 1024);

  cudaError_t status = flashinfer::SinglePrefillWithKVCache<DTypeQ, DTypeKV, DTypeOut>(
      thrust::raw_pointer_cast(q_d.data()), thrust::raw_pointer_cast(k_d.data()),
      thrust::raw_pointer_cast(v_d.data()), thrust::raw_pointer_cast(o_d.data()),
      thrust::raw_pointer_cast(tmp_d.data()),
      /*lse=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal, kv_layout,
      pos_encoding_mode, allow_fp16_qk_reduction);

  EXPECT_EQ(status, cudaSuccess) << "SinglePrefillWithKVCache kernel launch failed, error message: "
                                 << cudaGetErrorString(status);

  thrust::host_vector<DTypeOut> o_h(o_d);
  std::vector<DTypeOut> o_ref = cpu_reference::single_mha<DTypeQ, DTypeKV, DTypeOut>(
      q, k, v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, kv_layout,
      pos_encoding_mode);
  size_t num_results_error_atol = 0;
  bool nan_detected = false;

  for (size_t i = 0; i < o_ref.size(); ++i) {
    if (isnan(float(o_h[i]))) {
      nan_detected = true;
    }
    num_results_error_atol += (!utils::isclose(float(o_ref[i]), float(o_h[i]), rtol, atol));
    if (!utils::isclose(float(o_ref[i]), float(o_h[i]), rtol, atol)) {
      std::cout << "i=" << i << ", o_ref[i]=" << float(o_ref[i]) << ", o_h[i]=" << float(o_h[i])
                << std::endl;
    }
  }

  float result_accuracy = 1. - float(num_results_error_atol) / float(o_ref.size());
  std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", qo_len=" << qo_len << ", kv_len=" << kv_len << ", head_dim=" << head_dim
            << ", causal=" << causal << ", kv_layout=" << QKVLayoutToString(kv_layout)
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_FALSE(nan_detected) << "Nan detected in the result.";
}

template <typename DTypeIn, typename DTypeOut>
void TestSinglePrefillKernelLongContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t qo_len : {1, 31, 63, 127}) {
    for (size_t kv_len : {31717}) {
      for (size_t num_heads : {1}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              for (size_t kv_layout : {0, 1}) {
                _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeOut>(
                    qo_len, kv_len, num_heads, num_heads, head_dim, causal, QKVLayout(kv_layout),
                    PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DTypeKV>
void TestSinglePrefillFP8KernelLongContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t qo_len : {1, 31, 63, 127}) {
    for (size_t kv_len : {31717}) {
      for (size_t num_heads : {1}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool causal : {false, true}) {
            for (size_t pos_encoding_mode : {0}) {
              for (size_t kv_layout : {0, 1}) {
                _TestSinglePrefillKernelCorrectness<half, DTypeKV, half>(
                    qo_len, kv_len, num_heads, num_heads, head_dim, causal, QKVLayout(kv_layout),
                    PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DTypeIn, typename DTypeOut>
void TestSinglePrefillKernelShortContextCorrectness(bool allow_fp16_qk_reduction) {
  float rtol = std::is_same<DTypeOut, nv_bfloat16>::value ? 1e-2 : 1e-3;
  float atol = std::is_same<DTypeOut, nv_bfloat16>::value ? 1e-2 : 1e-3;
  for (size_t qkv_len : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t num_kv_heads : {4, 8, 32}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              for (size_t kv_layout : {0, 1}) {
                _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeOut>(
                    qkv_len, qkv_len, num_qo_heads, num_kv_heads, head_dim, causal,
                    QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
                    allow_fp16_qk_reduction, rtol, atol);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DTypeKV>
void TestSinglePrefillFP8KernelShortContextCorrectness(bool allow_fp16_qk_reduction) {
  float rtol = 1e-3;
  float atol = 1e-3;
  for (size_t qkv_len : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t num_kv_heads : {4, 8, 32}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool causal : {false, true}) {
            for (size_t pos_encoding_mode : {0}) {
              for (size_t kv_layout : {0, 1}) {
                _TestSinglePrefillKernelCorrectness<half, DTypeKV, half>(
                    qkv_len, qkv_len, num_qo_heads, num_kv_heads, head_dim, causal,
                    QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
                    allow_fp16_qk_reduction, rtol, atol);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DTypeIn, typename DTypeOut>
void TestSinglePrefillKernelCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t qo_len : {399, 400, 401}) {
    for (size_t kv_len : {533, 534, 535}) {
      for (size_t num_heads : {12}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              for (size_t kv_layout : {0, 1}) {
                _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeOut>(
                    qo_len, kv_len, num_heads, num_heads, head_dim, causal, QKVLayout(kv_layout),
                    PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
              }
            }
          }
        }
      }
    }
  }
}

template <typename DTypeKV>
void TestSinglePrefillFP8KernelCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t qo_len : {399, 400, 401}) {
    for (size_t kv_len : {533, 534, 535}) {
      for (size_t num_heads : {12}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool causal : {false, true}) {
            for (size_t pos_encoding_mode : {0}) {
              for (size_t kv_layout : {0, 1}) {
                _TestSinglePrefillKernelCorrectness<half, DTypeKV, half>(
                    qo_len, kv_len, num_heads, num_heads, head_dim, causal, QKVLayout(kv_layout),
                    PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
              }
            }
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelLongContextCorrectnessFP16) {
  TestSinglePrefillKernelLongContextCorrectness<half, half>(false);
}

TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelLongContextCorrectnessFP16QKHalfAccum) {
  TestSinglePrefillKernelLongContextCorrectness<half, half>(true);
}

TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelShortContextCorrectnessFP16) {
  TestSinglePrefillKernelShortContextCorrectness<half, half>(false);
}

TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelShortContextCorrectnessFP16QKHalfAccum) {
  TestSinglePrefillKernelShortContextCorrectness<half, half>(true);
}

TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestFP16) {
  TestSinglePrefillKernelCorrectness<half, half>(false);
}

TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestFP16QKHalfAccum) {
  TestSinglePrefillKernelCorrectness<half, half>(true);
}

#ifdef FLASHINFER_ENABLE_BF16
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelLongContextCorrectnessBF16) {
  TestSinglePrefillKernelLongContextCorrectness<nv_bfloat16, nv_bfloat16>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelShortContextCorrectnessBF16) {
  TestSinglePrefillKernelShortContextCorrectness<nv_bfloat16, nv_bfloat16>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestBF16) {
  TestSinglePrefillKernelCorrectness<nv_bfloat16, nv_bfloat16>(false);
}
#endif

#ifdef FLASHINFER_ENABLE_FP8
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelShortContextCorrectnessE4M3) {
  TestSinglePrefillFP8KernelShortContextCorrectness<__nv_fp8_e4m3>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelShortContextCorrectnessE5M2) {
  TestSinglePrefillFP8KernelShortContextCorrectness<__nv_fp8_e5m2>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestE4M3) {
  TestSinglePrefillFP8KernelCorrectness<__nv_fp8_e4m3>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestE5M2) {
  TestSinglePrefillFP8KernelCorrectness<__nv_fp8_e5m2>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelLongContextCorrectnessE4M3) {
  TestSinglePrefillFP8KernelLongContextCorrectness<__nv_fp8_e4m3>(false);
}
TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelLongContextCorrectnessE5M2) {
  TestSinglePrefillFP8KernelLongContextCorrectness<__nv_fp8_e5m2>(false);
}
#endif
