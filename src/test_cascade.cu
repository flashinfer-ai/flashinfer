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

template <typename T>
void _TestMergeKernelCorrectness(size_t num_index_sets, size_t batch_size, size_t num_heads,
                                 size_t head_dim) {
  EXPECT_GT(num_index_sets, 1) << "num_index_sets must be greater than 1";
  std::vector<T> V_host(num_index_sets * batch_size * num_heads * head_dim);
  std::vector<float> S_host(num_index_sets * batch_size * num_heads);

  // utils::vec_normal_(V_host);
  utils::vec_fill_<T>(V_host, 1.0);
  // utils::vec_uniform_(S_host, 5, 10);
  utils::vec_fill_<float>(S_host, 1.0);

  thrust::device_vector<T> V_device(V_host);
  thrust::device_vector<float> S_device(S_host);

  thrust::device_vector<T> V_merged_0_device(batch_size * num_heads * head_dim);
  thrust::device_vector<float> S_merged_0_device(batch_size * num_heads);
  thrust::device_vector<T> V_merged_1_device(batch_size * num_heads * head_dim);
  thrust::device_vector<float> S_merged_1_device(batch_size * num_heads);

  // Method 0: use MergeState
  MergeState(thrust::raw_pointer_cast(V_device.data()), thrust::raw_pointer_cast(S_device.data()),
             thrust::raw_pointer_cast(V_device.data() + batch_size * num_heads * head_dim),
             thrust::raw_pointer_cast(S_device.data() + batch_size * num_heads),
             thrust::raw_pointer_cast(V_merged_0_device.data()),
             thrust::raw_pointer_cast(S_merged_0_device.data()), batch_size, num_heads, head_dim);
  for (uint i = 2; i < num_index_sets; ++i) {
    MergeStateInPlace(
        thrust::raw_pointer_cast(V_merged_0_device.data()),
        thrust::raw_pointer_cast(S_merged_0_device.data()),
        thrust::raw_pointer_cast(V_device.data() + i * batch_size * num_heads * head_dim),
        thrust::raw_pointer_cast(S_device.data() + i * batch_size * num_heads), batch_size,
        num_heads, head_dim);
  }

  // Method 1: use MergeStates
  MergeStates(thrust::raw_pointer_cast(V_device.data()), thrust::raw_pointer_cast(S_device.data()),
              thrust::raw_pointer_cast(V_merged_1_device.data()),
              thrust::raw_pointer_cast(S_merged_1_device.data()), num_index_sets, batch_size,
              num_heads, head_dim);

  thrust::host_vector<T> V_merged_0_host(V_merged_0_device), V_merged_1_host(V_merged_1_device);
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
  float V_result_accuracy = 1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) /
                                      (num_index_sets * batch_size * num_heads * head_dim);
  float S_result_accuracy = 1.0 - float(num_S_result_errors_atol_1e_3_rtol_1e_3) /
                                      (num_index_sets * batch_size * num_heads);
  std::cout << "num_index_setes=" << num_index_sets << ", batch_size=" << batch_size
            << ", num_heads=" << num_heads << ", head_dim=" << head_dim
            << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
            << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy << std::endl;
  EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
  EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";
}

template <typename T>
void TestMergeKernelCorrectness() {
  for (size_t num_index_sets : {2, 9, 81}) {
    for (size_t batch_size : {4, 16}) {
      for (size_t num_heads : {1, 21, 32}) {
        for (size_t head_dim : {64, 128, 256}) {
          _TestMergeKernelCorrectness<T>(num_index_sets, batch_size, num_heads, head_dim);
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, MergeKernelCorrectnessTestFP16) {
  TestMergeKernelCorrectness<half>();
}
