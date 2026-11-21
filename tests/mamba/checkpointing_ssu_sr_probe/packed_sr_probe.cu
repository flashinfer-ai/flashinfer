// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cuda_runtime.h>
#include "flashinfer/mamba/conversion.cuh"
#include "original_fp8_scalar.cuh"

// clang-format on
using flashinfer::mamba::conversion::bitrev16;
using flashinfer::mamba::conversion::cvt_rs_e4m3x4_f32;

__global__ void flashinfer_test_original_scalar_sr_reference(uint32_t const* a, uint32_t const* b,
                                                             uint32_t const* random, uint32_t* out,
                                                             int64_t count) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    uint32_t const r = random[i], low = r & 65535u, high = r >> 16;
    uint32_t const pa = flashinfer_test_fp8_original::cvt_rs_e4m3_sw(__uint_as_float(a[i]), low);
    uint32_t const pb =
        flashinfer_test_fp8_original::cvt_rs_e4m3_sw(__uint_as_float(b[i]), bitrev16(low));
    uint32_t const pc =
        flashinfer_test_fp8_original::cvt_rs_e4m3_sw(__uint_as_float(a[i] ^ 0x80000000u), high);
    uint32_t const pd = flashinfer_test_fp8_original::cvt_rs_e4m3_sw(
        __uint_as_float(b[i] ^ 0x80000000u), bitrev16(high));
    out[i] = pa | (pb << 8) | (pc << 16) | (pd << 24);
  }
}
__global__ void flashinfer_test_packed_software_sr_candidate(uint32_t const* a, uint32_t const* b,
                                                             uint32_t const* random, uint32_t* out,
                                                             int64_t count) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x)
    out[i] = cvt_rs_e4m3x4_f32(__uint_as_float(a[i]), __uint_as_float(b[i]),
                               __uint_as_float(a[i] ^ 0x80000000u),
                               __uint_as_float(b[i] ^ 0x80000000u), random[i]);
}
extern "C" cudaError_t launch_exact_packed_sr_probe(uint32_t const* a, uint32_t const* b,
                                                    uint32_t const* r, uint32_t* reference,
                                                    uint32_t* candidate, int64_t count,
                                                    cudaStream_t stream) {
  if (!count) return cudaSuccess;
  int const blocks = int((count + 255) / 256 < 4096 ? (count + 255) / 256 : 4096);
  flashinfer_test_original_scalar_sr_reference<<<blocks, 256, 0, stream>>>(a, b, r, reference,
                                                                           count);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  flashinfer_test_packed_software_sr_candidate<<<blocks, 256, 0, stream>>>(a, b, r, candidate,
                                                                           count);
  return cudaGetLastError();
}
