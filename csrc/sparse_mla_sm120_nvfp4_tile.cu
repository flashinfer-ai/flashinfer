/*
 * Copyright (c) 2026 by FlashInfer team.
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

// Standalone M16N32K64 NVFP4 tile used to validate the QK/PV operand and
// scale-factor mappings before they are wired into the sparse orchestrator.

#include <cuda_runtime.h>

#include <cstdint>
#include <flashinfer/attention/sparse_mla_sm120/arch/ldmatrix_sm120.cuh>
#include <flashinfer/attention/sparse_mla_sm120/arch/mma_sm120_nvfp4.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/d2_load_b_nvfp4.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int kM = 16;
constexpr int kN = 32;
constexpr int kK = 64;
constexpr int kSFVecSize = 16;
constexpr int kPackedKBytes = kK / 2;
constexpr int kPvN = 8;
constexpr int kPvPackedDims = kPvN / 2;

__global__ void NVFP4M16N32K64Kernel(const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
                                     const uint8_t* sfb, float* output, int iterations) {
  __shared__ __align__(16) uint8_t sm_a[kM][kPackedKBytes];
  __shared__ __align__(16) uint8_t sm_b[kN][kPackedKBytes];

  // Physical byte copies preserve both packed E2M1 nibbles and mirror the
  // row-major shared-memory layout used by sparse MLA.
  for (int word = threadIdx.x; word < sizeof(sm_a) / sizeof(uint32_t); word += blockDim.x) {
    reinterpret_cast<uint32_t*>(sm_a)[word] = reinterpret_cast<const uint32_t*>(a)[word];
  }
  for (int word = threadIdx.x; word < sizeof(sm_b) / sizeof(uint32_t); word += blockDim.x) {
    reinterpret_cast<uint32_t*>(sm_b)[word] = reinterpret_cast<const uint32_t*>(b)[word];
  }
  __syncthreads();

  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int a_scale_row = gid + (lane & 1) * 8;
  const uint32_t scale_a =
      *reinterpret_cast<const uint32_t*>(sfa + a_scale_row * (kK / kSFVecSize));

  uint32_t a0, a1, a2, a3;
  ldmatrix_load_A_fp8(a0, a1, a2, a3, &sm_a[0][0], kPackedKBytes, lane);

#pragma unroll
  for (int nt = 0; nt < kN / 8; ++nt) {
    uint32_t b0, b1;
    ldmatrix_load_B_fp8(b0, b1, &sm_b[nt * 8][0], kPackedKBytes, lane);
    const uint32_t scale_b =
        *reinterpret_cast<const uint32_t*>(sfb + (nt * 8 + gid) * (kK / kSFVecSize));
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int iter = 0; iter < iterations; ++iter) {
      MmaNvfp4Result r =
          mma_nvfp4_block_scaled_m16n8k64(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, scale_a, scale_b);
      c0 = r.d0;
      c1 = r.d1;
      c2 = r.d2;
      c3 = r.d3;
    }
    const int column = nt * 8 + tid * 2;
    output[gid * kN + column] = c0;
    output[gid * kN + column + 1] = c1;
    output[(gid + 8) * kN + column] = c2;
    output[(gid + 8) * kN + column + 1] = c3;
  }
}

__global__ void NVFP4M16N8K64CandidateMajorKernel(const uint8_t* a,
                                                  const uint8_t* candidate_major_b,
                                                  const uint8_t* sfa, const uint8_t* sfb,
                                                  float* output, int iterations) {
  __shared__ __align__(16) uint8_t sm_a[kM][kPackedKBytes];
  __shared__ __align__(16) uint8_t sm_b[kK][kPvPackedDims];

  for (int word = threadIdx.x; word < sizeof(sm_a) / sizeof(uint32_t); word += blockDim.x) {
    reinterpret_cast<uint32_t*>(sm_a)[word] = reinterpret_cast<const uint32_t*>(a)[word];
  }
  for (int word = threadIdx.x; word < sizeof(sm_b) / sizeof(uint32_t); word += blockDim.x) {
    reinterpret_cast<uint32_t*>(sm_b)[word] =
        reinterpret_cast<const uint32_t*>(candidate_major_b)[word];
  }
  __syncthreads();

  const int lane = threadIdx.x;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int a_scale_row = gid + (lane & 1) * 8;
  const uint32_t scale_a =
      *reinterpret_cast<const uint32_t*>(sfa + a_scale_row * (kK / kSFVecSize));
  const uint32_t scale_b = *reinterpret_cast<const uint32_t*>(sfb + gid * (kK / kSFVecSize));

  uint32_t a0, a1, a2, a3, b0, b1;
  ldmatrix_load_A_fp8(a0, a1, a2, a3, &sm_a[0][0], kPackedKBytes, lane);
  d2_load_b_nvfp4<kPvPackedDims>(b0, b1, &sm_b[0][0], 0, 0, lane);

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int iter = 0; iter < iterations; ++iter) {
    MmaNvfp4Result r =
        mma_nvfp4_block_scaled_m16n8k64(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, scale_a, scale_b);
    c0 = r.d0;
    c1 = r.d1;
    c2 = r.d2;
    c3 = r.d3;
  }
  const int column = tid * 2;
  output[gid * kPvN + column] = c0;
  output[gid * kPvN + column + 1] = c1;
  output[(gid + 8) * kPvN + column] = c2;
  output[(gid + 8) * kPvN + column + 1] = c3;
}

namespace {

void check_tensor_2d(const TensorView& tensor, DLDataType dtype, int rows, int columns,
                     const char* name) {
  CHECK_CUDA(tensor);
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name << " has an invalid dtype";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 2) << name << " must be 2D";
  TVM_FFI_ICHECK_EQ(tensor.size(0), rows) << name << " row mismatch";
  TVM_FFI_ICHECK_EQ(tensor.size(1), columns) << name << " column mismatch";
}

}  // namespace

void SparseMlaSm120NVFP4M16N32K64(TensorView a, TensorView b, TensorView sfa, TensorView sfb,
                                  TensorView output, int64_t iterations) {
  check_tensor_2d(a, dl_uint8, kM, kK / 2, "a");
  check_tensor_2d(b, dl_uint8, kN, kK / 2, "b");
  check_tensor_2d(sfa, dl_float8_e4m3fn, kM, kK / kSFVecSize, "sfa");
  check_tensor_2d(sfb, dl_float8_e4m3fn, kN, kK / kSFVecSize, "sfb");
  check_tensor_2d(output, dl_float32, kM, kN, "output");
  TVM_FFI_ICHECK_GT(iterations, 0) << "iterations must be positive";
  TVM_FFI_ICHECK_LE(iterations, 1 << 20) << "iterations is unreasonably large";
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, sfa);
  CHECK_DEVICE(a, sfb);
  CHECK_DEVICE(a, output);

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  cudaStream_t stream = get_stream(a.device());
  NVFP4M16N32K64Kernel<<<1, 32, 0, stream>>>(
      static_cast<const uint8_t*>(a.data_ptr()), static_cast<const uint8_t*>(b.data_ptr()),
      static_cast<const uint8_t*>(sfa.data_ptr()), static_cast<const uint8_t*>(sfb.data_ptr()),
      static_cast<float*>(output.data_ptr()), static_cast<int>(iterations));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "NVFP4 M16N32K64 launch failed: " << cudaGetErrorString(status);
}

void SparseMlaSm120NVFP4M16N8K64CandidateMajor(TensorView a, TensorView b, TensorView sfa,
                                               TensorView sfb, TensorView output,
                                               int64_t iterations) {
  check_tensor_2d(a, dl_uint8, kM, kK / 2, "a");
  check_tensor_2d(b, dl_uint8, kK, kPvN / 2, "b");
  check_tensor_2d(sfa, dl_float8_e4m3fn, kM, kK / kSFVecSize, "sfa");
  check_tensor_2d(sfb, dl_float8_e4m3fn, kPvN, kK / kSFVecSize, "sfb");
  check_tensor_2d(output, dl_float32, kM, kPvN, "output");
  TVM_FFI_ICHECK_GT(iterations, 0) << "iterations must be positive";
  TVM_FFI_ICHECK_LE(iterations, 1 << 20) << "iterations is unreasonably large";
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, sfa);
  CHECK_DEVICE(a, sfb);
  CHECK_DEVICE(a, output);

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  cudaStream_t stream = get_stream(a.device());
  NVFP4M16N8K64CandidateMajorKernel<<<1, 32, 0, stream>>>(
      static_cast<const uint8_t*>(a.data_ptr()), static_cast<const uint8_t*>(b.data_ptr()),
      static_cast<const uint8_t*>(sfa.data_ptr()), static_cast<const uint8_t*>(sfb.data_ptr()),
      static_cast<float*>(output.data_ptr()), static_cast<int>(iterations));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "NVFP4 candidate-major M16N8K64 launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_nvfp4_m16n32k64,
                              flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4M16N32K64);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sparse_mla_sm120_nvfp4_m16n8k64_candidate_major,
    flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4M16N8K64CandidateMajor);
