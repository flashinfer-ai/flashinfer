/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include "cake_bf16_bmm_declarations_sm103a.cuh"
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace blackwell_bf16_bmm {

namespace {

#ifndef FLASHINFER_BLACKWELL_BF16_BMM_TARGET_MINOR
#error "FLASHINFER_BLACKWELL_BF16_BMM_TARGET_MINOR must be defined by the JIT/AOT spec"
#endif

constexpr int kTargetMinor = FLASHINFER_BLACKWELL_BF16_BMM_TARGET_MINOR;
static_assert(kTargetMinor == 0 || kTargetMinor == 3,
              "CAKE BF16 BMM target must be exact SM100a or SM103a");

constexpr int kOutBf16 = 0;
constexpr int kOutF16 = 1;
constexpr int kOutF32 = 2;

enum class Route : int {
  kGenericK64 = 0,
  kGenericK256 = 1,
  kGenericK1024 = 2,
  kK256M32N40Bf16 = 3,
  kK256M32N40F16 = 4,
  kK256M32N40F32 = 5,
  kK256M128N64Bf16 = 6,
  kK256M128N64F16 = 7,
  kK256M128N64F32 = 8,
  kK1024M16N1024Bf16 = 9,
  kK1024M16N1024F16 = 10,
  kK1024M16N1024F32 = 11,
  kK1024N16M8Tail = 12,
};

struct LaunchSpec {
  const void* kernel;
  dim3 grid;
  int threads;
  int dynamic_smem_bytes;
  Route route;
  bool use_pdl;
};

struct Problem {
  int batch_size;
  int m;
  int n;
  int k;
  int out_type;
  int a_stride_b;
  int a_stride_m;
  int a_stride_k;
  int b_stride_b;
  int b_stride_n;
  int b_stride_k;
};

void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

void CheckTarget(int device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == kTargetMinor)
      << "this CAKE BF16 BMM module was compiled for exact compute capability 10." << kTargetMinor
      << ", got " << major << "." << minor;
}

int CheckedInt(int64_t value, const char* name) {
  TVM_FFI_ICHECK_GE(value, 0) << name << " must be non-negative";
  TVM_FFI_ICHECK_LE(value, std::numeric_limits<int>::max())
      << name << " exceeds the generated kernel's int32 ABI";
  return static_cast<int>(value);
}

void CheckDataAlignment(const TensorView& tensor, const char* name) {
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  if (address % 16 != 0) {
    TVM_FFI_THROW(ValueError) << name << " data pointer must be 16-byte aligned";
  }
}

bool ByteRangesOverlap(const TensorView& lhs, std::uint64_t lhs_bytes, const TensorView& rhs,
                       std::uint64_t rhs_bytes) {
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data_ptr());
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data_ptr());
  TVM_FFI_ICHECK_LE(lhs_bytes, std::numeric_limits<std::uintptr_t>::max() - lhs_begin)
      << "lhs tensor byte range overflows uintptr_t";
  TVM_FFI_ICHECK_LE(rhs_bytes, std::numeric_limits<std::uintptr_t>::max() - rhs_begin)
      << "rhs tensor byte range overflows uintptr_t";
  const auto lhs_end = lhs_begin + lhs_bytes;
  const auto rhs_end = rhs_begin + rhs_bytes;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

int OutputType(const TensorView& out) {
  if (out.dtype() == dl_bfloat16) {
    return kOutBf16;
  }
  if (out.dtype() == dl_float16) {
    return kOutF16;
  }
  if (out.dtype() == dl_float32) {
    return kOutF32;
  }
  TVM_FFI_THROW(ValueError) << "CAKE BF16 BMM output must be bfloat16, float16, or float32";
  return -1;
}

const void* SelectByOutputType(int out_type, const void* bf16_kernel, const void* f16_kernel,
                               const void* f32_kernel) {
  if (out_type == kOutBf16) {
    return bf16_kernel;
  }
  if (out_type == kOutF16) {
    return f16_kernel;
  }
  return f32_kernel;
}

Route RouteByOutputType(int out_type, Route bf16_route, Route f16_route, Route f32_route) {
  if (out_type == kOutBf16) {
    return bf16_route;
  }
  if (out_type == kOutF16) {
    return f16_route;
  }
  return f32_route;
}

LaunchSpec SelectLaunch(const Problem& problem) {
  if (problem.batch_size == 16 && problem.k == 256 && problem.m == 128 && problem.n == 80 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_44eb4a7b8a86325202c6),
      dim3((problem.m + 31) / 32, (problem.n + 39) / 40, problem.batch_size),
      160, 36864, static_cast<Route>(3), true};
  }
  if (problem.batch_size == 16 && problem.k == 256 && problem.m == 128 && problem.n == 80 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_a13456761a57e20eb43d),
      dim3((problem.m + 31) / 32, (problem.n + 39) / 40, problem.batch_size),
      160, 36864, static_cast<Route>(4), true};
  }
  if (problem.batch_size == 16 && problem.k == 256 && problem.m == 128 && problem.n == 80 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_4c2925f941c18e9ba383),
      dim3((problem.m + 31) / 32, (problem.n + 39) / 40, problem.batch_size),
      160, 36864, static_cast<Route>(5), true};
  }
  if (problem.batch_size == 16 && problem.k == 256 && problem.m == 128 && problem.n == 64 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_fdbd1b90786b149d7d2e),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(6), true};
  }
  if (problem.batch_size == 16 && problem.k == 256 && problem.m == 128 && problem.n == 64 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_af7495a29467dba4178c),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(7), true};
  }
  if (problem.batch_size == 16 && problem.k == 256 && problem.m == 128 && problem.n == 64 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_869f43afdb8a948398b7),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(8), true};
  }
  if (problem.batch_size == 4 && problem.k == 1024 && problem.m == 16 && problem.n == 1024 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_e16dc73fa76da4f687c4),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 98304, static_cast<Route>(9), true};
  }
  if (problem.batch_size == 4 && problem.k == 1024 && problem.m == 16 && problem.n == 1024 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_4c33ad0ae6c78c36746c),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 98304, static_cast<Route>(10), true};
  }
  if (problem.batch_size == 4 && problem.k == 1024 && problem.m == 16 && problem.n == 1024 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_2be765abed4a60c000a7),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 98304, static_cast<Route>(11), true};
  }
  if (problem.batch_size == 2 && problem.k == 1024 && problem.m == 8 && problem.n == 1024 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_0aa6610c5790bb7e78a8),
      dim3((problem.m + 15) / 16, (problem.n + 15) / 16, problem.batch_size),
      64, 65536, static_cast<Route>(12), true};
  }
  if (problem.k == 64 && problem.m == 128 && problem.n == 64 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_05210b13930b347b10e5),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 128 && problem.n == 64 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_c76c2166a5d5c53f4d72),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 128 && problem.n == 64 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_16a6b9251e408d683478),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 48 && problem.n == 64 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_c23b198fd16a09495a3c),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 48 && problem.n == 64 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_b7b3dfc29cbe1e75cea8),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 48 && problem.n == 64 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_9589f85cc4b9aafde12a),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 128 && problem.n == 80 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_97ac58f8c28e3a9f7ccd),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 128 && problem.n == 80 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_dfb467d2ac52721f071a),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 128 && problem.n == 80 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_59b89d08f864e7dfae86),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 48 && problem.n == 80 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_3c2820b9896ee6f117ab),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 48 && problem.n == 80 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_78ff57242075aae196f4),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 64 && problem.m == 48 && problem.n == 80 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_b751bc0ab613774d2a8d),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 256 && problem.m == 128 && problem.n == 64 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_fdbd1b90786b149d7d2e),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 128 && problem.n == 64 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_af7495a29467dba4178c),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 128 && problem.n == 64 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_869f43afdb8a948398b7),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 48 && problem.n == 64 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_be0a25527b6a907a6b9f),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 48 && problem.n == 64 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_214585a1ff53e7aa90e4),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 48 && problem.n == 64 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_40eba7ca5a71bfe0c7b8),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 128 && problem.n == 80 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_1355ce28cd6e7bb672aa),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 128 && problem.n == 80 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_5a2711b7cb94fd49c706),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 128 && problem.n == 80 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_dade0fe0e8e644a42572),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 48 && problem.n == 80 && problem.out_type == 0) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_706b766a7aedb03057b4),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 48 && problem.n == 80 && problem.out_type == 1) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_20eb25725fe130422815),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 256 && problem.m == 48 && problem.n == 80 && problem.out_type == 2) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_2d6361a99d62f3f52a21),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 64) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_b051749a89c3c70704d1),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 6144, static_cast<Route>(0), true};
  }
  if (problem.k == 256) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_a1dc7b1314ac48d86dbf),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 24576, static_cast<Route>(1), true};
  }
  if (problem.k == 1024) {
    return {reinterpret_cast<const void*>(kernel_cake_bf16_bmm_9a3f18b71438e9f341b6),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128, 98304, static_cast<Route>(2), true};
  }
  TVM_FFI_THROW(ValueError) << "No generated CAKE BF16 BMM route for this problem";
  return {};
}

Problem ValidateProblem(const TensorView& A, const TensorView& B, const TensorView& out) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CUDA(out);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, out);
  CHECK_DEVICE(A, B);
  CHECK_DEVICE(A, out);

  CheckDataAlignment(A, "A");
  CheckDataAlignment(B, "B");
  CheckDataAlignment(out, "out");

  TVM_FFI_ICHECK_EQ(A.dtype(), dl_bfloat16) << "A must be bfloat16";
  TVM_FFI_ICHECK_EQ(B.dtype(), dl_bfloat16) << "B must be bfloat16";

  Problem problem;
  problem.batch_size = CheckedInt(A.size(0), "batch size");
  problem.m = CheckedInt(A.size(1), "M");
  problem.k = CheckedInt(A.size(2), "K");
  problem.n = CheckedInt(B.size(2), "N");
  problem.out_type = OutputType(out);

  TVM_FFI_ICHECK_GT(problem.batch_size, 0) << "batch size must be positive";
  TVM_FFI_ICHECK_GT(problem.m, 0) << "M must be positive";
  TVM_FFI_ICHECK_GT(problem.n, 0) << "N must be positive";
  TVM_FFI_ICHECK_LE(problem.batch_size, 65535) << "batch size exceeds CUDA grid.z";
  TVM_FFI_ICHECK_LE((static_cast<int64_t>(problem.n) + 31) / 32, 65535)
      << "N exceeds CUDA grid.y for the narrowest dispatcher tile";
  TVM_FFI_ICHECK_EQ(problem.n % 8, 0) << "CAKE BF16 BMM requires N to be a multiple of 8";
  TVM_FFI_ICHECK(problem.k == 64 || problem.k == 256 || problem.k == 1024)
      << "CAKE BF16 BMM requires K to be one of {64, 256, 1024}";

  TVM_FFI_ICHECK_EQ(B.size(0), problem.batch_size) << "A and B batch sizes must match";
  TVM_FFI_ICHECK_EQ(B.size(1), problem.k) << "A K and B K dimensions must match";
  TVM_FFI_ICHECK_EQ(out.size(0), problem.batch_size) << "out batch size mismatch";
  TVM_FFI_ICHECK_EQ(out.size(1), problem.m) << "out M dimension mismatch";
  TVM_FFI_ICHECK_EQ(out.size(2), problem.n) << "out N dimension mismatch";

  TVM_FFI_ICHECK_EQ(A.stride(2), 1) << "A must be row-major in K";
  TVM_FFI_ICHECK_EQ(A.stride(1), problem.k) << "A must have exact row-major [B,M,K] strides";
  TVM_FFI_ICHECK_EQ(A.stride(0), static_cast<int64_t>(problem.m) * problem.k)
      << "A must have exact row-major [B,M,K] strides";

  TVM_FFI_ICHECK_EQ(B.stride(1), 1) << "B must be the exact column-major/transposed [B,K,N] view";
  TVM_FFI_ICHECK_EQ(B.stride(2), problem.k)
      << "B must be the exact column-major/transposed [B,K,N] view";
  TVM_FFI_ICHECK_EQ(B.stride(0), static_cast<int64_t>(problem.k) * problem.n)
      << "B must be the exact column-major/transposed [B,K,N] view";

  TVM_FFI_ICHECK_EQ(out.stride(2), 1) << "out must be contiguous row-major";
  TVM_FFI_ICHECK_EQ(out.stride(1), problem.n) << "out must be contiguous row-major";
  TVM_FFI_ICHECK_EQ(out.stride(0), static_cast<int64_t>(problem.m) * problem.n)
      << "out must be contiguous row-major";

  problem.a_stride_b = CheckedInt(A.stride(0), "A batch stride");
  problem.a_stride_m = CheckedInt(A.stride(1), "A row stride");
  problem.a_stride_k = CheckedInt(A.stride(2), "A K stride");
  problem.b_stride_b = CheckedInt(B.stride(0), "B batch stride");
  problem.b_stride_n = CheckedInt(B.stride(2), "B N stride");
  problem.b_stride_k = CheckedInt(B.stride(1), "B K stride");
  const int64_t a_element_count = static_cast<int64_t>(problem.batch_size) * problem.m * problem.k;
  const int64_t b_element_count = static_cast<int64_t>(problem.batch_size) * problem.k * problem.n;
  CheckedInt(a_element_count, "A element count");
  CheckedInt(b_element_count, "B element count");
  const int out_element_bytes = problem.out_type == kOutF32 ? 4 : 2;
  const int64_t out_byte_span =
      static_cast<int64_t>(problem.batch_size) * problem.m * problem.n * out_element_bytes;
  CheckedInt(out_byte_span, "output byte span");
  if (ByteRangesOverlap(out, out_byte_span, A, a_element_count * 2)) {
    TVM_FFI_THROW(ValueError) << "out must not overlap A";
  }
  if (ByteRangesOverlap(out, out_byte_span, B, b_element_count * 2)) {
    TVM_FFI_THROW(ValueError) << "out must not overlap B";
  }
  return problem;
}

}  // namespace

void Run(TensorView A, TensorView B, TensorView out) {
  Problem problem = ValidateProblem(A, B, out);
  const LaunchSpec launch = SelectLaunch(problem);

  ffi::CUDADeviceGuard device_guard(A.device().device_id);
  CheckTarget(A.device().device_id);
  cudaError_t status = cudaFuncSetAttribute(
      launch.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch.dynamic_smem_bytes);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Failed to set CAKE BF16 BMM dynamic shared memory: " << cudaGetErrorString(status);

  auto* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
  auto* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
  auto* out_ptr = static_cast<uint8_t*>(out.data_ptr());
  void* args[] = {
      &a_ptr,
      &b_ptr,
      &out_ptr,
      &problem.m,
      &problem.n,
      &problem.a_stride_b,
      &problem.a_stride_m,
      &problem.a_stride_k,
      &problem.b_stride_b,
      &problem.b_stride_n,
      &problem.b_stride_k,
      &problem.out_type,
  };
  if (launch.use_pdl) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = launch.grid;
    config.blockDim = dim3(launch.threads);
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream = get_stream(A.device());
    config.attrs = &attribute;
    config.numAttrs = 1;
    status = cudaLaunchKernelExC(&config, launch.kernel, args);
  } else {
    status = cudaLaunchKernel(launch.kernel, launch.grid, dim3(launch.threads), args,
                              launch.dynamic_smem_bytes, get_stream(A.device()));
  }
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Failed to launch CAKE BF16 BMM: " << cudaGetErrorString(status);
}

int RouteOf(TensorView A, TensorView B, TensorView out) {
  const Problem problem = ValidateProblem(A, B, out);
  CheckTarget(A.device().device_id);
  return static_cast<int>(SelectLaunch(problem).route);
}

}  // namespace blackwell_bf16_bmm
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_bf16_bmm::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(route_of, flashinfer::blackwell_bf16_bmm::RouteOf);
