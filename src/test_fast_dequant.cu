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
#include <thrust/detail/raw_pointer_cast.h>

#include <bitset>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

using namespace flashinfer;

template <typename dtype_f8, typename dtype_f16>
__global__ void test_fast_f8_f16_dequant(dtype_f8* f8, dtype_f16* f16) {
  size_t global_tidx = blockIdx.x * blockDim.x + threadIdx.x;
  vec_cast<dtype_f16, dtype_f8>::cast<8>(f16 + global_tidx * 8, f8 + global_tidx * 8);
}

template <typename dtype_f8, typename dtype_f16>
void TestFastDequant() {
  std::vector<dtype_f8> f8_h(1024);
  utils::vec_normal_(f8_h);
  std::vector<dtype_f16> f16_h_ref(1024);
  for (uint32_t i = 0; i < 1024; ++i) {
    f16_h_ref[i] = static_cast<dtype_f16>(f8_h[i]);
  }

  thrust::device_vector<dtype_f8> f8_d(f8_h);
  thrust::device_vector<dtype_f16> f16_d(1024);

  test_fast_f8_f16_dequant<dtype_f8, dtype_f16>
      <<<1, 128>>>(thrust::raw_pointer_cast(f8_d.data()), thrust::raw_pointer_cast(f16_d.data()));

  cudaError_t err = cudaGetLastError();
  EXPECT_EQ(err, cudaSuccess);

  thrust::host_vector<dtype_f16> f16_h(f16_d);
  for (uint32_t i = 0; i < 1024; ++i) {
    if (f16_h[i] != f16_h_ref[i]) {
      printf("mismatch at i=%d: out=%x ref=%x\n", i, *(uint16_t*)(f16_h.data() + i),
             *(uint16_t*)(f16_h_ref.data() + i));
    }
    EXPECT_EQ(f16_h[i], f16_h_ref[i]);
  }
}

TEST(FlashInferCorrectnessTest, TestFastDequantCorrectnessE4M3ToFloat16) {
  TestFastDequant<__nv_fp8_e4m3, half>();
}
TEST(FlashInferCorrectnessTest, TestFastDequantCorrectnessE5M2ToFloat16) {
  TestFastDequant<__nv_fp8_e5m2, half>();
}
TEST(FlashInferCorrectnessTest, TestFastDequantCorrectnessE4M3ToBFloat16) {
  TestFastDequant<__nv_fp8_e4m3, __nv_bfloat16>();
}
TEST(FlashInferCorrectnessTest, TestFastDequantCorrectnessE5M2ToBFloat16) {
  TestFastDequant<__nv_fp8_e5m2, __nv_bfloat16>();
}
