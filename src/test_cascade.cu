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

#include <flashinfer/cascade.cuh>

#include "utils.h"

using namespace flashinfer;

bool is_prime(int x) {
  for (int i = 2; i < int(std::sqrt(x)); ++i) {
    if (x % i == 0) return false;
  }
  return true;
}

template <typename T>
void _TestMergeKernelCorrectness(size_t num_index_sets, size_t batch_size, size_t num_heads,
                                 size_t head_dim, bool sparse_s) {
  EXPECT_GT(num_index_sets, 1) << "num_index_sets must be greater than 1";
  std::vector<T> V_host(num_index_sets * batch_size * num_heads * head_dim);
  std::vector<float> V_host_f32(num_index_sets * batch_size * num_heads * head_dim);
  std::vector<float> S_host(num_index_sets * batch_size * num_heads);

  utils::vec_normal_(V_host);
  std::transform(V_host.begin(), V_host.end(), V_host_f32.begin(),
                 [](T x) { return static_cast<float>(x); });
  if (sparse_s) {
    for (uint32_t i = 0; i < num_index_sets; ++i) {
      float fill_val = is_prime(i) ? 10 : -10;
      for (uint32_t j = 0; j < batch_size; ++j) {
        for (uint32_t k = 0; k < num_heads; ++k) {
          S_host[i * batch_size * num_heads + j * num_heads + k] = fill_val;
        }
      }
    }
  } else {
    utils::vec_uniform_(S_host, -10, 10);
  }

  thrust::device_vector<T> V_device(V_host);
  thrust::device_vector<float> V_device_f32(V_host_f32);
  thrust::device_vector<float> S_device(S_host);

  thrust::device_vector<float> V_merged_0_device(batch_size * num_heads * head_dim);
  thrust::device_vector<float> S_merged_0_device(batch_size * num_heads);
  thrust::device_vector<T> V_merged_1_device(batch_size * num_heads * head_dim);
  thrust::device_vector<float> S_merged_1_device(batch_size * num_heads);

  // Method 0: use MergeState
  MergeState(thrust::raw_pointer_cast(V_device_f32.data()),
             thrust::raw_pointer_cast(S_device.data()),
             thrust::raw_pointer_cast(V_device_f32.data() + batch_size * num_heads * head_dim),
             thrust::raw_pointer_cast(S_device.data() + batch_size * num_heads),
             thrust::raw_pointer_cast(V_merged_0_device.data()),
             thrust::raw_pointer_cast(S_merged_0_device.data()), batch_size, num_heads, head_dim);
  for (uint i = 2; i < num_index_sets; ++i) {
    MergeStateInPlace(
        thrust::raw_pointer_cast(V_merged_0_device.data()),
        thrust::raw_pointer_cast(S_merged_0_device.data()),
        thrust::raw_pointer_cast(V_device_f32.data() + i * batch_size * num_heads * head_dim),
        thrust::raw_pointer_cast(S_device.data() + i * batch_size * num_heads), batch_size,
        num_heads, head_dim);
  }

  // Method 1: use MergeStates
  MergeStates(thrust::raw_pointer_cast(V_device.data()), thrust::raw_pointer_cast(S_device.data()),
              thrust::raw_pointer_cast(V_merged_1_device.data()),
              thrust::raw_pointer_cast(S_merged_1_device.data()), num_index_sets, batch_size,
              num_heads, head_dim);

  thrust::host_vector<float> V_merged_0_host(V_merged_0_device);
  thrust::host_vector<T> V_merged_1_host(V_merged_1_device);
  thrust::host_vector<float> S_merged_0_host(S_merged_0_device), S_merged_1_host(S_merged_1_device);
  size_t num_V_result_errors_atol_1e_3_rtol_1e_3 = 0, num_S_result_errors_atol_1e_3_rtol_1e_3 = 0;
  for (size_t i = 0; i < batch_size * num_heads * head_dim; ++i) {
    EXPECT_FALSE(std::isnan(float(V_merged_0_host[i]))) << "V_merged_0_host[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(V_merged_1_host[i]))) << "V_merged_1_host[" << i << "] is nan";
    num_V_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(V_merged_0_host[i]), float(V_merged_1_host[i]), 1e-3, 1e-3));
  }
  for (size_t i = 0; i < batch_size * num_heads; ++i) {
    EXPECT_FALSE(std::isnan(float(S_merged_0_host[i]))) << "S_merged_0_host[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(S_merged_1_host[i]))) << "S_merged_1_host[" << i << "] is nan";
    num_S_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(S_merged_0_host[i]), float(S_merged_1_host[i]), 1e-3, 1e-3));
  }
  float V_result_accuracy =
      1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) / (batch_size * num_heads * head_dim);
  float S_result_accuracy =
      1.0 - float(num_S_result_errors_atol_1e_3_rtol_1e_3) / (batch_size * num_heads);
  std::cout << "num_index_setes=" << num_index_sets << ", batch_size=" << batch_size
            << ", num_heads=" << num_heads << ", head_dim=" << head_dim << ", sparse_s=" << sparse_s
            << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
            << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy << std::endl;
  EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
  EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";
}

template <typename T>
void TestMergeKernelCorrectness() {
  for (size_t num_index_sets : {2, 9, 81, 513}) {
    for (size_t batch_size : {4, 16}) {
      for (size_t num_heads : {1, 21, 32}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool sparse_s : {false, true}) {
            _TestMergeKernelCorrectness<T>(num_index_sets, batch_size, num_heads, head_dim,
                                           sparse_s);
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, MergeKernelCorrectnessTestFP16) {
  TestMergeKernelCorrectness<half>();
}
