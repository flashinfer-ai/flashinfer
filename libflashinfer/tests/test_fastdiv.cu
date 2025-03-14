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

#include "gtest/gtest.h"
#include "utils.h"

using namespace flashinfer;

__global__ void test_fastdiv_kernel_0(uint_fastdiv fd, uint32_t* q, uint32_t* r) {
  uint32_t global_rank = blockIdx.x * blockDim.x + threadIdx.x;
  q[global_rank] = global_rank / fd;
  r[global_rank] = global_rank % fd;
}

__global__ void test_fastdiv_kernel_1(uint_fastdiv fd, uint32_t* q, uint32_t* r) {
  uint32_t global_rank = blockIdx.x * blockDim.x + threadIdx.x;
  fd.divmod(global_rank, q[global_rank], r[global_rank]);
}

void _TestFastDivU32Correctness(uint32_t d) {
  uint_fastdiv fd(d);
  thrust::device_vector<uint32_t> q(1024 * 1024), r(1024 * 1024);

  {
    test_fastdiv_kernel_0<<<1024, 1024>>>(fd, thrust::raw_pointer_cast(q.data()),
                                          thrust::raw_pointer_cast(r.data()));

    thrust::host_vector<uint32_t> q_h(q), r_h(r);

    for (size_t i = 0; i < q_h.size(); ++i) {
      EXPECT_EQ(q_h[i], i / d);
      EXPECT_EQ(r_h[i], i % d);
    }
  }

  {
    test_fastdiv_kernel_1<<<1024, 1024>>>(fd, thrust::raw_pointer_cast(q.data()),
                                          thrust::raw_pointer_cast(r.data()));

    thrust::host_vector<uint32_t> q_h(q), r_h(r);

    for (size_t i = 0; i < q_h.size(); ++i) {
      EXPECT_EQ(q_h[i], i / d);
      EXPECT_EQ(r_h[i], i % d);
    }
  }

  std::cout << "FastDivU32 correctness test passed for d = " << d << std::endl;
}

void TestFastDivU32Correctness() {
  for (uint32_t d = 1; d < 127; ++d) {
    _TestFastDivU32Correctness(d);
  }
}

TEST(FlashInferCorrectnessTest, TestFastDivU32Correctness) { TestFastDivU32Correctness(); }
