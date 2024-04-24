/*
 * Copyright (c) 2024 by FlashInfer team.
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

#include <flashinfer/norm.cuh>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <typename T>
void _TestRMSNormCorrectness(uint32_t batch_size, uint32_t d) {
  std::vector<T> x_host(batch_size * d);
  std::vector<T> w_host(d);

  utils::vec_normal_(x_host);
  utils::vec_normal_(w_host);

  std::vector<T> y_ref_host =
      std::move(cpu_reference::rms_norm<T>(x_host.data(), w_host.data(), batch_size, d, 1e-5));

  thrust::device_vector<T> x_device(x_host);
  thrust::device_vector<T> w_device(w_host);
  thrust::device_vector<T> y_device(batch_size * d);

  cudaError_t status = norm::RMSNorm<T>(
      thrust::raw_pointer_cast(x_device.data()), thrust::raw_pointer_cast(w_device.data()),
      thrust::raw_pointer_cast(y_device.data()), batch_size, d, 1e-6);
  EXPECT_EQ(status, cudaSuccess) << "RMSNorm kernel launch failed, error message: "
                                 << cudaGetErrorString(status);

  thrust::host_vector<T> y_host(y_device);
  bool nan_detected = false;
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  for (uint i = 0; i < batch_size * d; i++) {
    if (isnan(float(y_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(y_host[i]), float(y_ref_host[i]), 1e-3, 1e-3));
    if (!utils::isclose(float(y_host[i]), float(y_ref_host[i]), 1e-3, 1e-3)) {
      std::cout << "i: " << i << ", y_host[i]: " << float(y_host[i])
                << ", y_ref_host[i]: " << float(y_ref_host[i]) << std::endl;
    }
  }
  float result_accuracy = 1.0f - float(num_result_errors_atol_1e_3_rtol_1e_3) / (batch_size * d);
  std::cout << "batch_size: " << batch_size << ", d: " << d
            << ", RMSNorm correctness: " << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.99f) << "RMSNorm correctness test failed";
  EXPECT_FALSE(nan_detected) << "Nan detected in RMSNorm output";
}

template <typename T>
void TestRMSNormCorrectness() {
  for (size_t batch_size : {1, 3, 7, 19, 733}) {
    for (size_t d : {37, 128, 512, 1002, 3072, 4096, 8192, 16384}) {
      _TestRMSNormCorrectness<T>(batch_size, d);
    }
  }
}

TEST(FlashInferCorrectnessTests, TestRMSNormFP16) { TestRMSNormCorrectness<half>(); }
