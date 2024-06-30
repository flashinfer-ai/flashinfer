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

#include <type_traits>

#include "cpu_reference.h"
#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;

template <typename DTypeQO, typename DTypeKV>
void _TestDecodingKernelCorrectness(size_t num_qo_heads, size_t num_kv_heads, size_t seq_len,
                                    size_t head_dim, QKVLayout kv_layout,
                                    PosEncodingMode pos_encoding_mode) {
  std::vector<DTypeQO> Q_host(num_qo_heads * head_dim);
  std::vector<DTypeKV> K_host(seq_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> V_host(seq_len * num_kv_heads * head_dim);
  std::vector<DTypeQO> O_host(num_qo_heads * head_dim);

  utils::vec_normal_(Q_host);
  utils::vec_normal_(K_host);
  utils::vec_normal_(V_host);
  utils::vec_zero_(O_host);

  thrust::device_vector<DTypeQO> Q(Q_host);
  thrust::device_vector<DTypeKV> K(K_host);
  thrust::device_vector<DTypeKV> V(V_host);
  thrust::device_vector<DTypeQO> O(O_host);
  thrust::device_vector<DTypeQO> tmp(32 * 1024 * 1024);
  std::vector<DTypeQO> o_ref_host;

  o_ref_host = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
      Q_host, K_host, V_host, 1, seq_len, num_qo_heads, num_kv_heads, head_dim, false, kv_layout,
      pos_encoding_mode);

  cudaError_t status = SingleDecodeWithKVCache<DTypeQO, DTypeKV, DTypeQO>(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
      thrust::raw_pointer_cast(tmp.data()), num_qo_heads, num_kv_heads, seq_len, head_dim,
      kv_layout, pos_encoding_mode);
  EXPECT_EQ(status, cudaSuccess) << "SingleDecodeWithKVCache kernel launch failed, error message: "
                                 << cudaGetErrorString(status);

  thrust::host_vector<DTypeQO> o_host = O;

  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < num_qo_heads * head_dim; ++i) {
    if (isnan(float(o_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref_host[i]), 1e-2, 1e-2));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_qo_heads * head_dim);
  std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", seq_len=" << seq_len << ", head_dim=" << head_dim
            << ", kv_layout=" << QKVLayoutToString(kv_layout)
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_FALSE(nan_detected) << "NaN detected.";
}

template <typename DTypeQO, typename DTypeKV>
void TestSingleDecodeKernelCorrectness() {
  for (size_t num_qo_heads : {32}) {
    for (size_t num_kv_heads : {4, 8, 32}) {
      for (size_t seq_len :
           {1, 3, 9, 27, 81, 129, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (unsigned int kv_layout : {0U, 1U}) {
            for (unsigned int pos_encoding_mode : {0U, 1U}) {
              _TestDecodingKernelCorrectness<DTypeQO, DTypeKV>(num_qo_heads, num_kv_heads, seq_len,
                                                               head_dim, QKVLayout(kv_layout),
                                                               PosEncodingMode(pos_encoding_mode));
            }
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestFP16) {
  TestSingleDecodeKernelCorrectness<half, half>();
}

#ifdef FLASHINFER_ENABLE_BF16
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestBF16) {
  TestSingleDecodeKernelCorrectness<nv_bfloat16, nv_bfloat16>();
}
#endif

#ifdef FLASHINFER_ENABLE_FP8
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestE4M3) {
  TestSingleDecodeKernelCorrectness<half, __nv_fp8_e4m3>();
}
TEST(FlashInferCorrectnessTest, SingleDecodeKernelCorrectnessTestE5M2) {
  TestSingleDecodeKernelCorrectness<half, __nv_fp8_e5m2>();
}
#endif
