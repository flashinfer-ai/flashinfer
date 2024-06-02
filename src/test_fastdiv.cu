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

#include <flashinfer/fastdiv.cuh>

#include "utils.h"

using namespace flashinfer;

__global__ void test_fastdiv_kernel(uint_fastdiv fd, uint32_t* out) {
  uint32_t global_rank = blockIdx.x * blockDim.x + threadIdx.x;
  out[global_rank] = global_rank / fd;
}

void _TestFastDivU32Correctness(uint32_t d) {
  uint_fastdiv fd(d);
  thrust::device_vector<uint32_t> out(1024 * 1024);

  test_fastdiv_kernel<<<1024, 1024>>>(fd, thrust::raw_pointer_cast(out.data()));

  thrust::host_vector<uint32_t> out_h(out);

  for (size_t i = 0; i < out_h.size(); ++i) {
    EXPECT_EQ(out_h[i], i / d);
  }

  std::cout << "FastDivU32 correctness test passed for d = " << d << std::endl;
}

void TestFastDivU32Correctness() {
  for (uint32_t d = 1; d < 127; ++d) {
    _TestFastDivU32Correctness(d);
  }
}

TEST(FlashInferCorrectnessTest, TestFastDivU32Correctness) { TestFastDivU32Correctness(); }