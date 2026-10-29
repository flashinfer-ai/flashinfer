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

// The kernel and launcher sources, the registry and build.json are exporter
// output, hand-folded here pending exporter support: programs that differ only
// in their constants or lever text share one translation unit, and each
// build.json entry selects its program through its own -D definitions.
// Shared tvm-ffi launch helpers for the generated Cake kernels of one export.
#ifndef CAKE_DSA_INDEXER_TOPK_HOST_COMMON_CUH_
#define CAKE_DSA_INDEXER_TOPK_HOST_COMMON_CUH_
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cake_ffi_checks.h"

// A binding translation unit shared by several programs declares and launches
// CAKE_PROGRAM_KERNEL and exports its entry points from namespace CAKE_PROGRAM_SHIM;
// the file defaults both to the program it was exported for, and every other
// program passes -DCAKE_PROGRAM_KERNEL / -DCAKE_PROGRAM_SHIM on the compile line.
// The finalize bindings take their block size and dynamic shared memory from the
// THREADS / SMEM_TOTAL constants their build.json entries already pass to the
// kernel.  CAKE_STRINGIZE spells a macro's value inside a string literal;
// CAKE_UNSIGNED appends the unsigned suffix to a macro's integer value.  No
// check condition names a macro: TVM_FFI_CHECK stringizes its condition unexpanded.
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#define CAKE_UNSIGNED_(x) x##u
#define CAKE_UNSIGNED(x) CAKE_UNSIGNED_(x)

namespace cake_host_shim_common {

// Opt the linked kernel into its dynamic shared memory once for every device
// that accepts it. Function-local static initialization already serializes
// the first call; no lock or per-device cache is needed.
inline bool CakeSetMaxDynamicSmem(const void* symbol, int smem_bytes) {
  cudaKernel_t kernel = nullptr;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetKernel(&kernel, symbol));
  int device_count = 0;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  int configured = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaKernelSetAttributeForDevice(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes, device) ==
        cudaSuccess) {
      ++configured;
    } else {
      (void)cudaGetLastError();
    }
  }
  TVM_FFI_CHECK(configured > 0, RuntimeError)
      << "no CUDA device accepts " << smem_bytes << " B of dynamic shared memory";
  return true;
}

}  // namespace cake_host_shim_common

#endif  // CAKE_DSA_INDEXER_TOPK_HOST_COMMON_CUH_

// Text every scan binding carried verbatim, included from inside the binding's
// namespace at the position the exported text had it (the header's own section
// above is skipped there by its include guard).  The TMA descriptor encoders
// differ between the scan bodies only in their box line, so each box shape is one
// part:
//   CAKE_HOST_COMMON_PART 1 / 2 / 3  EncodeTma_Q, box {64u, 128u, 2u} / {64u, 192u, 2u} / {64u, 256u, 2u}
//   CAKE_HOST_COMMON_PART 4 / 5      EncodeTma_K, box {64u, 128u, 1u} / {64u, 128u, 2u}
//   CAKE_HOST_COMMON_PART 6 / 7 / 8  EncodeTma_W, box {32u, 4u} / {32u, 6u} / {32u, 8u}
// Only whole functions are hoisted; the opening of Run stays in each binding.
#ifdef CAKE_HOST_COMMON_PART
#if CAKE_HOST_COMMON_PART == 1
// 3D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "Q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'Q' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 2u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2] << ") exceeds resolved global dims for 'Q'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 2
// 3D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "Q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'Q' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 192u, 2u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2] << ") exceeds resolved global dims for 'Q'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 3
// 3D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "Q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'Q' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 256u, 2u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2] << ") exceeds resolved global dims for 'Q'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 4
// 3D TMA descriptor for buffer 'K' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'K' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'K' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "K");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'K' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2] << ") exceeds resolved global dims for 'K'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'K' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'K') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 5
// 3D TMA descriptor for buffer 'K' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'K' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'K' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "K");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'K' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 2u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2] << ") exceeds resolved global dims for 'K'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'K' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'K') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 6
// 2D TMA descriptor for buffer 'W' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_W(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'W' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'W' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'W' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "W");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'W' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'W' resolved a non-positive global dim";
  uint32_t box_dim[2] = {32u, 4u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ") exceeds resolved global dims for 'W'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
  };
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'W') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 7
// 2D TMA descriptor for buffer 'W' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_W(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'W' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'W' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'W' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "W");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'W' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'W' resolved a non-positive global dim";
  uint32_t box_dim[2] = {32u, 6u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ") exceeds resolved global dims for 'W'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
  };
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'W') failed: CUresult=" << (int)r;
  return tm;
}
#elif CAKE_HOST_COMMON_PART == 8
// 2D TMA descriptor for buffer 'W' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_W(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'W' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'W' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'W' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "W");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'W' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'W' resolved a non-positive global dim";
  uint32_t box_dim[2] = {32u, 8u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ") exceeds resolved global dims for 'W'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'W' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
  };
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'W') failed: CUresult=" << (int)r;
  return tm;
}
#else
#error "CAKE_HOST_COMMON_PART must be 1 to 8"
#endif
#undef CAKE_HOST_COMMON_PART
#endif
