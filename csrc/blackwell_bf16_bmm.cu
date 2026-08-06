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
#include <flashinfer/gemm/blackwell_bf16_bmm.cuh>
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
  if (problem.k == 256 && problem.batch_size == 16 && problem.m == 128 && problem.n == 80) {
    return {
        SelectByOutputType(
            problem.out_type,
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o0_fixed),
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o1_fixed),
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o2_fixed)),
        dim3((problem.m + 31) / 32, (problem.n + 39) / 40, problem.batch_size),
        160,
        36864,
        RouteByOutputType(problem.out_type, Route::kK256M32N40Bf16, Route::kK256M32N40F16,
                          Route::kK256M32N40F32),
    };
  }
  if (problem.k == 1024 && problem.batch_size == 4 && problem.m == 16 && problem.n == 1024) {
    return {
        SelectByOutputType(
            problem.out_type,
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o0_fixed),
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o1_fixed),
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o2_fixed)),
        dim3(1, 32, problem.batch_size),
        128,
        98304,
        RouteByOutputType(problem.out_type, Route::kK1024M16N1024Bf16, Route::kK1024M16N1024F16,
                          Route::kK1024M16N1024F32),
    };
  }
  if (problem.k == 1024 && problem.batch_size == 2 && problem.m == 8 && problem.n == 1024 &&
      problem.out_type == kOutBf16) {
    return {
        reinterpret_cast<const void*>(
            kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_n16_m8_tail),
        dim3(1, 64, problem.batch_size),
        64,
        65536,
        Route::kK1024N16M8Tail,
    };
  }
  if (problem.k == 256 && problem.batch_size == 16 && problem.m == 128 && problem.n == 64) {
    return {
        SelectByOutputType(
            problem.out_type,
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o0_fixed),
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o1_fixed),
            reinterpret_cast<const void*>(
                kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o2_fixed)),
        dim3(8, 2, problem.batch_size),
        128,
        24576,
        RouteByOutputType(problem.out_type, Route::kK256M128N64Bf16, Route::kK256M128N64F16,
                          Route::kK256M128N64F32),
    };
  }
  if (problem.k == 64) {
    return {
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k64),
        dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
        128,
        6144,
        Route::kGenericK64,
    };
  }
  if (problem.k == 256) {
    return {
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256),
        dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
        128,
        24576,
        Route::kGenericK256,
    };
  }
  return {
      reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024),
      dim3((problem.m + 15) / 16, (problem.n + 31) / 32, problem.batch_size),
      128,
      98304,
      Route::kGenericK1024,
  };
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
  status = cudaLaunchKernel(launch.kernel, launch.grid, dim3(launch.threads), args,
                            launch.dynamic_smem_bytes, get_stream(A.device()));
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
