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
// Rendered by the Cake source exporter — do not edit.
// One-call host sequence of the Cake all-gather matmul route
// 'float16_ws8_k_major_sm_100a': NVLink barrier, bridge event, per-peer push copies with a
// per-chunk readiness epoch (or, for small padded M, the SM push kernel
// chained to the barrier and the main kernel through programmatic dependent
// launch), tcgen05 main kernel. The main kernel computes 256-wide output
// tiles; a class that ships the 128-wide twin (kHasMainN128) runs the twin
// when the adapter's table-driven rule selects block_n = 128 for the call.
// The tensor-map encoders below are the generated main launchers'; the
// kernels are the route's device units.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "cake_all_gather_matmul_host_common.cuh"

#ifndef CAKE_PEER_POINTER_TABLE_DECLARED
#define CAKE_PEER_POINTER_TABLE_DECLARED
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable {
  T* ptrs[Capacity];
};
#endif
extern "C" __global__ void kernel_cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9(
    int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags);
extern "C" __global__ void kernel_cake_all_gather_matmul_8dea171063652cd775f8(
    int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags);
extern "C" __global__ void kernel_cake_all_gather_matmul_40242e9c64d6e66af6f5(
    unsigned int* __restrict__ inp, long long* __restrict__ payload_peers,
    long long* __restrict__ signal_peers, unsigned int* __restrict__ counters,
    unsigned int ready_target, int rank, int rows, int chunk_rows, int num_chunks);
extern "C" __global__ void kernel_cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3(
    unsigned int* __restrict__ inp, long long* __restrict__ payload_peers,
    long long* __restrict__ signal_peers, unsigned int* __restrict__ counters,
    unsigned int ready_target, int rank, int rows, int chunk_rows, int num_chunks);
extern "C" __global__ void kernel_cake_all_gather_matmul_f85e6d8165a64ac49252(
    const __grid_constant__ CUtensorMap A_local, const __grid_constant__ CUtensorMap A_scratch,
    const __grid_constant__ CUtensorMap B, __half* __restrict__ C,
    __half* __restrict__ scratch_payload, unsigned int* __restrict__ ready,
    unsigned int ready_target, int rank, int M, int scratch_pitch, int signal_rows, int n_tiles,
    int remote_order);
extern "C" __global__ void kernel_cake_all_gather_matmul_6e81f55424d75f150b3b(
    const __grid_constant__ CUtensorMap A_local, const __grid_constant__ CUtensorMap A_scratch,
    const __grid_constant__ CUtensorMap B, __half* __restrict__ C,
    __half* __restrict__ scratch_payload, unsigned int* __restrict__ ready,
    unsigned int ready_target, int rank, int M, int scratch_pitch, int signal_rows, int n_tiles,
    int remote_order);

namespace cake_host_shim_seq_5e7dedbca8872e49 {

using namespace cake_host_shim_common;
using tvm::ffi::TensorView;

template <int Capacity>
struct alignas(16) CakePeerPointerTableHost {
  uint64_t ptrs[Capacity];
};
static_assert(sizeof(CakePeerPointerTableHost<8>) == 64,
              "eight-entry peer table CUDA ABI must be 64 bytes");
static_assert(sizeof(CakePeerPointerTableHost<32>) == 256,
              "32-entry peer table CUDA ABI must be 256 bytes");
static_assert(alignof(CakePeerPointerTableHost<8>) == 16,
              "peer table CUDA ABI must be 16-byte aligned");
static_assert(alignof(CakePeerPointerTableHost<32>) == 16,
              "peer table CUDA ABI must be 16-byte aligned");

// 3D TMA descriptor for buffer 'A_local' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_A_local(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'A_local' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'A_local' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source 'A_local' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "A_local");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source 'A_local' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'A_local' extent " << d1 << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'A_local' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2]
      << ") exceeds resolved global dims for 'A_local'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 1 zero while global dimension 1 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 2 zero while global dimension 2 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                                      const_cast<void*>(tensor_base), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'A_local') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'A_scratch' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_A_scratch(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'A_scratch' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'A_scratch' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source 'A_scratch' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "A_scratch");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source 'A_scratch' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'A_scratch' extent " << d1 << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 1u};
  TVM_FFI_CHECK(
      box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2],
      ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2]
      << ") exceeds resolved global dims for 'A_scratch'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 1 zero while global dimension 1 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 2 zero while global dimension 2 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                                      const_cast<void*>(tensor_base), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'A_scratch') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'B' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_B(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'B' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'B' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source 'B' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "B");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source 'B' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'B' extent " << d1 << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'B' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 256u, 1u};
  TVM_FFI_CHECK(
      box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2],
      ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2]
      << ") exceeds resolved global dims for 'B'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'B' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'B' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                                      const_cast<void*>(tensor_base), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'B') failed: CUresult=" << (int)r;
  return tm;
}

// Tensor-map encoders of the 128-wide main twin, which reads B through
// n128::EncodeTma_B: its K-major B box differs from the N256 main's (64 x 128
// rows); its N-major box is identical and keeps the one code path. A class
// without the twin renders no n128 block, so n128 is opened first with a
// using-directive nominating the sibling namespace n128_absent: qualified
// lookup of n128::EncodeTma_B reaches that stand-in only when n128 itself
// declares no EncodeTma_B, and no call reaches it (resolve_call admits
// block_n = 128 only with the twin). A sibling rather than a nested namespace
// because a using-directive shows the nominated names to unqualified lookup
// as if declared in the nearest namespace enclosing both; from inside the
// twin's block that is the sequence namespace, not n128, so an unqualified
// EncodeTma_B(...) written there is never ambiguous with the twin's own.
namespace n128_absent {
inline CUtensorMap EncodeTma_B(const TensorView&) {
  TVM_FFI_THROW(RuntimeError) << "this route ships no 128-wide main";
  return CUtensorMap{};
}
}  // namespace n128_absent
namespace n128 {
using namespace n128_absent;
}  // namespace n128

namespace n128 {
// 3D TMA descriptor for buffer 'A_local' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_A_local(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'A_local' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'A_local' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source 'A_local' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "A_local");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source 'A_local' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'A_local' extent " << d1 << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'A_local' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2]
      << ") exceeds resolved global dims for 'A_local'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 1 zero while global dimension 1 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 2 zero while global dimension 2 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_local' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                                      const_cast<void*>(tensor_base), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'A_local') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'A_scratch' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_A_scratch(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'A_scratch' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'A_scratch' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source 'A_scratch' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "A_scratch");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source 'A_scratch' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'A_scratch' extent " << d1 << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 1u};
  TVM_FFI_CHECK(
      box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2],
      ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2]
      << ") exceeds resolved global dims for 'A_scratch'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 1 zero while global dimension 1 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 2 zero while global dimension 2 is "
         "not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A_scratch' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                                      const_cast<void*>(tensor_base), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'A_scratch') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'B' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_B(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'B' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'B' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError) << "TMA source 'B' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "B");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError) << "TMA source 'B' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'B' extent " << d1 << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'B' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 128u, 1u};
  TVM_FFI_CHECK(
      box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2],
      ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2]
      << ") exceeds resolved global dims for 'B'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'B' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'B' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'B' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
                                      const_cast<void*>(tensor_base), global_dim, global_strides,
                                      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'B') failed: CUresult=" << (int)r;
  return tm;
}
}  // namespace n128

constexpr int64_t kK = 8192;
constexpr int64_t kBlockM = 128;
constexpr int64_t kChunkRows = 2432;
constexpr int64_t kBlockN = 256;
// Whether the route class ships the 128-wide main twin: block_n = 128 is admitted
// only then (N itself stays a multiple of kBlockN, the public contract).
constexpr bool kHasMainN128 = true;
// The main program is the arrival-ordered persistent schedule (one CTA per SM,
// grid chosen by the API); both push transports land a peer's chunks in order,
// so the remote tile order is peer-major (0).
constexpr int32_t kRemoteOrder = 0;
// Padded local rows up to which the SM push kernel replaces the copy-engine pushes,
// and the widest N it serves at a padded row count (the GEMM waits behind the SM
// push; a wider GEMM overlaps the copy-engine pushes instead).
constexpr int64_t kSmPushMaxRows = 512;
constexpr int64_t kSmPushColsIntercept = 12288;
constexpr int64_t kSmPushColsPerRow = 16;
constexpr int64_t sm_push_max_cols(int64_t padded_rows) {
  return kSmPushColsIntercept - kSmPushColsPerRow * padded_rows;
}
constexpr int64_t kWorldSize = 8;
constexpr bool kMainBeforePushes = false;
constexpr unsigned kBarrierThreads = 32u;
constexpr unsigned kMainThreads = 192u;
constexpr unsigned kMainDynamicSmem = 197632u;
constexpr unsigned kMainN128DynamicSmem = 197632u;

// A tensor map is a pure function of the source's address, extents and
// strides, so the encoder output is cached on that key: the bound weight and
// the symmetric scratch hit on every call, the activation hits whenever the
// caller's allocator hands back a known buffer.
struct TensorMapKey {
  uint64_t base = 0;
  int32_t ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};
  int64_t strides[4] = {0, 0, 0, 0};

  static TensorMapKey of(const TensorView& t) {
    TensorMapKey key;
    key.base = reinterpret_cast<uint64_t>(t.data_ptr());
    key.ndim = static_cast<int32_t>(t.ndim());
    for (int32_t i = 0; i < key.ndim && i < 4; ++i) {
      key.shape[i] = t.size(i);
      key.strides[i] = t.stride(i);
    }
    return key;
  }

  bool operator==(const TensorMapKey& other) const {
    if (base != other.base || ndim != other.ndim) return false;
    for (int32_t i = 0; i < 4; ++i) {
      if (shape[i] != other.shape[i] || strides[i] != other.strides[i]) return false;
    }
    return true;
  }
};

template <int Entries>
struct TensorMapCache {
  TensorMapKey keys[Entries];
  CUtensorMap maps[Entries];
  bool valid[Entries] = {};
  int next = 0;

  template <typename Encode>
  const CUtensorMap& get(const TensorView& t, Encode encode) {
    TensorMapKey key = TensorMapKey::of(t);
    for (int i = 0; i < Entries; ++i) {
      if (valid[i] && keys[i] == key) return maps[i];
    }
    int slot = next;
    next = (next + 1) % Entries;
    maps[slot] = encode(t);
    keys[slot] = key;
    valid[slot] = true;
    return maps[slot];
  }
};

// The two bridge events of one communication stream, created once.
struct BridgeEvents {
  cudaEvent_t to_comm = nullptr;
  cudaEvent_t to_main = nullptr;
};

inline BridgeEvents& bridge_events(cudaStream_t comm_stream) {
  static std::mutex mutex;
  static std::unordered_map<cudaStream_t, BridgeEvents> table;
  std::lock_guard<std::mutex> guard(mutex);
  auto it = table.find(comm_stream);
  if (it == table.end()) {
    BridgeEvents events;
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&events.to_comm, cudaEventDisableTiming));
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&events.to_main, cudaEventDisableTiming));
    it = table.emplace(comm_stream, events).first;
  }
  return it->second;
}

inline void launch_barrier(int64_t phase, int32_t world, int32_t rank,
                           CakePeerPointerTableHost<8>& flags, cudaStream_t stream) {
  void* kargs[] = {&world, &rank, &flags};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1u, 1u, 1u);
  config.blockDim = dim3(kBarrierThreads, 1u, 1u);
  config.dynamicSmemBytes = 0u;
  config.stream = stream;
  const void* kernel = phase == 0
                           ? reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_bdfeeb2b9ea3cd522fb9)
                           : reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_8dea171063652cd775f8);
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for the phase " << phase
      << " barrier failed: " << cudaGetErrorString(status);
}

// One main program of the route: the kernel computing BlockN-wide output tiles
// and its dynamic smem footprint. The N256 main and its N128 twin are distinct
// symbols with distinct footprints, so each instantiation raises its own
// kernel's max dynamic smem attribute once (one function-local flag per
// symbol; a flag shared by both would leave the second at the default limit).
// For a class without the twin the N128 entry names the N256 main with a 0-byte
// footprint and is never reached: it throws before its static would lower that
// kernel's dynamic smem limit.
struct MainProgram {
  const void* kernel;
  unsigned dynamic_smem;
};

template <int64_t BlockN>
inline MainProgram main_program() {
  static_assert(BlockN == kBlockN || BlockN == 128, "the route has 256- and 128-wide mains only");
  MainProgram program{};
  if constexpr (BlockN == 128) {
    if constexpr (!kHasMainN128) {
      TVM_FFI_THROW(RuntimeError) << "this route ships no 128-wide main";
    }
    program.kernel = reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_6e81f55424d75f150b3b);
    program.dynamic_smem = kMainN128DynamicSmem;
  } else {
    program.kernel = reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_f85e6d8165a64ac49252);
    program.dynamic_smem = kMainDynamicSmem;
  }
  static const bool smem_ready =
      CakeSetMaxDynamicSmem(program.kernel, static_cast<int>(program.dynamic_smem));
  (void)smem_ready;
  return program;
}

inline void launch_main(const void* kernel, unsigned dynamic_smem, const CUtensorMap& a_local,
                        const CUtensorMap& a_scratch, const CUtensorMap& b, void* c,
                        void* scratch_payload, unsigned int* ready, uint32_t ready_target,
                        int32_t rank, int32_t m, int32_t scratch_pitch, int32_t signal_rows,
                        int32_t n_tiles, int32_t remote_order, dim3 grid, cudaStream_t stream,
                        bool programmatic_launch) {
  CUtensorMap p_a_local = a_local;
  CUtensorMap p_a_scratch = a_scratch;
  CUtensorMap p_b = b;
  void* kargs[] = {&p_a_local, &p_a_scratch, &p_b,          &c, &scratch_payload,
                   &ready,     &ready_target, &rank, &m, &scratch_pitch,
                   &signal_rows, &n_tiles, &remote_order};
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(kMainThreads, 1u, 1u);
  config.dynamicSmemBytes = dynamic_smem;
  config.stream = stream;
  // SM push route: the main grid may dispatch as soon as the push grid is
  // resident (it gates every remote chunk on its readiness epoch).
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = programmatic_launch ? &attribute : nullptr;
  config.numAttrs = programmatic_launch ? 1u : 0u;
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for the main kernel failed: " << cudaGetErrorString(status);
}

struct Call {
  DLDevice dev;
  cudaStream_t main_stream;
  cudaStream_t comm_stream;
  int32_t rank;
  int32_t rows;
  int32_t pitch;
  uint32_t ready_target;
  int64_t padded_rows;
  int64_t cols;
  int64_t block_n;
  int64_t chunk_rows;
  int64_t num_chunks;
  dim3 grid;
  CakePeerPointerTableHost<8> flags;
};

inline Call resolve_call(const TensorView& inp, const TensorView& scratch, const TensorView& weight,
                         const TensorView& out, const TensorView& ready,
                         const tvm::ffi::Shape& flag_peers, int64_t rank, int64_t rows,
                         int64_t pitch, int64_t phase, int64_t ready_target, int64_t grid_x,
                         int64_t grid_y, int64_t grid_z, int64_t block_n, int64_t main_stream,
                         int64_t comm_stream) {
  Call call;
  call.dev = inp.device();
  check_cuda_tensor(inp, "A_local");
  check_dtype(inp, DLDataType{kDLFloat, 16, 1}, "A_local");
  check_contiguous(inp, "A_local");
  check_cuda_tensor(scratch, "A_scratch");
  check_dtype(scratch, DLDataType{kDLFloat, 16, 1}, "A_scratch");
  check_contiguous(scratch, "A_scratch");
  check_cuda_tensor(weight, "B");
  check_dtype(weight, DLDataType{kDLFloat, 16, 1}, "B");
  // K-major weight: the TMA encoder validates its dense leading dims and row stride.
  check_cuda_tensor(out, "C");
  check_dtype(out, DLDataType{kDLFloat, 16, 1}, "C");
  check_contiguous(out, "C");
  check_cuda_tensor(ready, "ready");
  check_dtype(ready, DLDataType{kDLUInt, 32, 1}, "ready");
  check_contiguous(ready, "ready");
  check_same_device(scratch, inp, "A_scratch", "A_local");
  check_same_device(weight, inp, "B", "A_local");
  check_same_device(out, inp, "C", "A_local");
  check_same_device(ready, inp, "ready", "A_local");
  TVM_FFI_CHECK(inp.ndim() == 2 && inp.size(1) == kK, ValueError)
      << "A_local must be a [M, " << kK << "] tensor";
  TVM_FFI_CHECK(out.ndim() == 2 && out.size(1) > 0 && out.size(1) % kBlockN == 0, ValueError)
      << "C must be a [world_size * M, N] tensor with N a positive multiple of " << kBlockN;
  TVM_FFI_CHECK(rows > 0 && rows == inp.size(0) && rows <= 2147483647LL, ValueError)
      << "M=" << rows << " must equal the A_local row count " << inp.size(0);
  TVM_FFI_CHECK(scratch.ndim() == 3 && scratch.size(0) == kWorldSize && scratch.size(2) == kK,
                ValueError)
      << "A_scratch must be a [" << kWorldSize << ", pitch, " << kK << "] tensor";
  TVM_FFI_CHECK(pitch == scratch.size(1) && pitch % kBlockM == 0, ValueError)
      << "scratch_pitch=" << pitch << " must equal the scratch pitch " << scratch.size(1)
      << " and be a multiple of " << kBlockM;
  TVM_FFI_CHECK(phase == 0 || phase == 1, ValueError) << "phase must be 0 or 1, got " << phase;
  TVM_FFI_CHECK(ready_target >= 0LL && ready_target <= 4294967295LL, ValueError)
      << "scalar 'ready_target' value " << ready_target << " is outside u32 range";
  TVM_FFI_CHECK(rank >= 0 && rank < kWorldSize, ValueError)
      << "rank " << rank << " is outside world size " << kWorldSize;
  TVM_FFI_CHECK(flag_peers.size() == kWorldSize, ValueError)
      << "peer table 'pg_flags' requires exactly " << kWorldSize << " entries";
  for (int64_t i = 0; i < flag_peers.size(); ++i) {
    TVM_FFI_CHECK(flag_peers[i] != 0, ValueError) << "peer table 'pg_flags' contains a null address";
    call.flags.ptrs[i] = static_cast<uint64_t>(flag_peers[i]);
  }
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y << ", "
      << grid_z << ")";
  call.main_stream = reinterpret_cast<cudaStream_t>(main_stream);
  call.comm_stream = reinterpret_cast<cudaStream_t>(comm_stream);
  TVM_FFI_CHECK(call.comm_stream != nullptr && call.comm_stream != call.main_stream, ValueError)
      << "the communication stream must be a distinct non-default stream";
  call.rank = static_cast<int32_t>(rank);
  call.rows = static_cast<int32_t>(rows);
  call.pitch = static_cast<int32_t>(pitch);
  call.ready_target = static_cast<uint32_t>(ready_target);
  call.padded_rows = (rows + kBlockM - 1) / kBlockM * kBlockM;
  call.cols = out.size(1);
  call.chunk_rows = call.padded_rows < kChunkRows ? call.padded_rows : kChunkRows;
  call.num_chunks = (call.padded_rows + call.chunk_rows - 1) / call.chunk_rows;
  TVM_FFI_CHECK(call.padded_rows <= pitch, ValueError)
      << "the workspace holds " << pitch << " rows per peer, the call needs " << call.padded_rows;
  // The main tile width of the call: kBlockN, or 128 when the class ships the
  // twin and the adapter's rule selected it; the persistent grid never exceeds
  // the output tile count at that width.
  TVM_FFI_CHECK(block_n == kBlockN || (block_n == 128 && kHasMainN128), ValueError)
      << "block_n=" << block_n << " is not a main tile width of this route (" << kBlockN
      << (kHasMainN128 ? " or 128)" : "; it ships no 128-wide main)");
  TVM_FFI_CHECK(call.cols % block_n == 0, ValueError)
      << "N=" << call.cols << " must be a multiple of the main tile width block_n=" << block_n;
  const int64_t total_tiles = kWorldSize * (call.padded_rows / kBlockM) * (call.cols / block_n);
  TVM_FFI_CHECK(grid_x <= total_tiles, ValueError)
      << "grid_x=" << grid_x << " exceeds the " << total_tiles
      << " output tiles of the call at block_n=" << block_n;
  call.block_n = block_n;
  call.grid = dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
                   static_cast<uint32_t>(grid_z));
  return call;
}

inline void launch_main_for(const Call& call, const TensorView& inp, const TensorView& scratch,
                            const TensorView& weight, const TensorView& out,
                            const TensorView& ready, bool programmatic_launch = false) {
  static thread_local TensorMapCache<4> a_local_cache;
  static thread_local TensorMapCache<2> a_scratch_cache;
  static thread_local TensorMapCache<2> b_cache;
  // The N128 twin reads B through its own map (its K-major box is 64 x 128; the
  // N-major map is identical): the two B maps never share a cache, since a
  // K-major N256 map handed to the twin hangs under PDL instead of failing.
  static thread_local TensorMapCache<2> b128_cache;
  const bool n128_main = call.block_n == 128;
  const CUtensorMap& a_local = a_local_cache.get(inp, EncodeTma_A_local);
  const CUtensorMap& a_scratch = a_scratch_cache.get(scratch, EncodeTma_A_scratch);
  const CUtensorMap& b =
      n128_main ? b128_cache.get(weight, n128::EncodeTma_B) : b_cache.get(weight, EncodeTma_B);
  const MainProgram program = n128_main ? main_program<128>() : main_program<kBlockN>();
  // One readiness signal per push chunk; the output tile count along N comes
  // from the caller-owned output ([world * rows, N]) at the call's tile width.
  launch_main(program.kernel, program.dynamic_smem, a_local, a_scratch, b, out.data_ptr(),
              scratch.data_ptr(), static_cast<unsigned int*>(ready.data_ptr()), call.ready_target,
              call.rank, call.rows, call.pitch, static_cast<int32_t>(call.chunk_rows),
              static_cast<int32_t>(out.size(1) / call.block_n), kRemoteOrder, call.grid,
              call.main_stream, programmatic_launch);
}

// Copy-engine route: barrier(phase); comm waits main; for every peer, for
// every chunk: push the valid rows of the chunk and publish its epoch; main
// kernel on the caller's stream (it consumes chunks as their epochs land);
// the caller's stream then joins the pushes. kMainBeforePushes places the
// main kernel launch before the push loop (the GPU starts it while the host
// is still issuing this rank's pushes) or after it; the adapter chooses per
// world size from measurement.
void Run(TensorView inp, TensorView scratch, TensorView weight, TensorView out, TensorView ready,
         tvm::ffi::Shape flag_peers, tvm::ffi::Shape peer_scratch, tvm::ffi::Shape peer_signals,
         int64_t rank, int64_t rows, int64_t pitch, int64_t phase, int64_t ready_target,
         int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_n, int64_t main_stream,
         int64_t comm_stream) {
  tvm::ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  Call call = resolve_call(inp, scratch, weight, out, ready, flag_peers, rank, rows, pitch, phase,
                           ready_target, grid_x, grid_y, grid_z, block_n, main_stream, comm_stream);
  TVM_FFI_CHECK(peer_scratch.size() == kWorldSize - 1 && peer_signals.size() == kWorldSize - 1,
                ValueError)
      << "peer scratch and signal tables need exactly " << (kWorldSize - 1) << " entries";
  for (int64_t i = 0; i < peer_scratch.size(); ++i) {
    TVM_FFI_CHECK(peer_scratch[i] != 0 && peer_signals[i] != 0, ValueError)
        << "peer scratch/signal tables contain a null address";
  }
  BridgeEvents& events = bridge_events(call.comm_stream);
  const int64_t row_bytes = kK * static_cast<int64_t>(inp.dtype().bits / 8);
  const char* source = static_cast<const char*>(inp.data_ptr());

  launch_barrier(phase, static_cast<int32_t>(kWorldSize), call.rank, call.flags, call.main_stream);
  TVM_FFI_CHECK_CUDA_ERROR(cudaEventRecord(events.to_comm, call.main_stream));
  TVM_FFI_CHECK_CUDA_ERROR(cudaStreamWaitEvent(call.comm_stream, events.to_comm, 0));
  if constexpr (kMainBeforePushes) {
    launch_main_for(call, inp, scratch, weight, out, ready);
  }
  for (int64_t peer = 0; peer < peer_scratch.size(); ++peer) {
    char* destination = reinterpret_cast<char*>(static_cast<uintptr_t>(peer_scratch[peer]));
    const CUdeviceptr signal_row =
        static_cast<CUdeviceptr>(peer_signals[peer]) +
        static_cast<CUdeviceptr>(call.rank) * call.num_chunks * sizeof(uint32_t);
    for (int64_t chunk = 0; chunk < call.num_chunks; ++chunk) {
      const int64_t begin = chunk * call.chunk_rows;
      const int64_t end = begin + call.chunk_rows < rows ? begin + call.chunk_rows : rows;
      if (end > begin) {
        TVM_FFI_CHECK_CUDA_ERROR(cudaMemcpyAsync(destination + begin * row_bytes,
                                                 source + begin * row_bytes,
                                                 static_cast<size_t>((end - begin) * row_bytes),
                                                 cudaMemcpyDeviceToDevice, call.comm_stream));
      }
      CUresult written = cuStreamWriteValue32(reinterpret_cast<CUstream>(call.comm_stream),
                                              signal_row + chunk * sizeof(uint32_t),
                                              call.ready_target, 0);
      TVM_FFI_CHECK(written == CUDA_SUCCESS, RuntimeError)
          << "cuStreamWriteValue32 for the chunk " << chunk << " epoch failed: CUresult="
          << static_cast<int>(written);
    }
  }
  if constexpr (!kMainBeforePushes) {
    launch_main_for(call, inp, scratch, weight, out, ready);
  }
  TVM_FFI_CHECK_CUDA_ERROR(cudaEventRecord(events.to_main, call.comm_stream));
  TVM_FFI_CHECK_CUDA_ERROR(cudaStreamWaitEvent(call.main_stream, events.to_main, 0));
}

constexpr unsigned kPushThreads = 128u;
// The SM push launch is chosen per call from the per-peer payload of the padded local shard
// (padded rows x K x element bytes). Below kPushPayloadThresholdBytes a call takes the 32-CTA
// launch: kPushCtasPerPeer CTAs per remote peer of the push program that keeps one 16-byte vector
// in flight per lane. At or above it a call takes the wide launch: kPushWideCtasPerPeer CTAs per
// peer of the push program that keeps eight vectors in flight per lane (fewer CTAs leave most SMs
// to the main grid, which then starts at the barrier, while the deeper memory-level parallelism
// keeps the NVLink egress; measured faster only on the large payloads). Both programs write the
// same 16-byte vectors to the same addresses and the readiness ticket chain counts gridDim.x CTAs
// at runtime, so the outputs are identical whichever launch a call takes.
constexpr unsigned kPushCtasPerPeer = 32u;
constexpr unsigned kPushWideCtasPerPeer = 8u;
constexpr int64_t kPushPayloadThresholdBytes = 4194304;
// Preferred shared-memory carveout of both push kernels (cudaFuncAttributePreferredSharedMemoryCarveout;
// -1 = driver default). The push kernels use no shared memory, but their CTAs land on the SMs before
// the main grid dispatches (programmatic dependent launch) and an SM's L1/shared split is fixed while
// any CTA is resident: with the driver default a 192 KiB main CTA cannot land on an SM hosting a push
// CTA until it drains. The maximum-shared preference lets the main CTAs co-reside with the push CTAs.
constexpr int kPushCarveout = -1;

// Set one push kernel's carveout once for every device that accepts it: a function attribute is
// per device and one process may serve several devices through RunPush, so this mirrors the
// per-device form CakeSetMaxDynamicSmem uses for the main kernel (cudaGetKernel +
// cudaKernelSetAttributeForDevice over cudaGetDeviceCount) instead of a current-device-only
// attribute call. The attribute calls are not stream operations, so the static-once form of the
// caller is graph-capture safe.
inline void set_push_carveout_for(const void* function, const char* label) {
  cudaKernel_t kernel = nullptr;
  cudaError_t status = cudaGetKernel(&kernel, function);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaGetKernel for the " << label << " failed: " << cudaGetErrorString(status);
  int device_count = 0;
  status = cudaGetDeviceCount(&device_count);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaGetDeviceCount for the " << label << " carveout failed: " << cudaGetErrorString(status);
  int configured = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaKernelSetAttributeForDevice(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, kPushCarveout, device) == cudaSuccess) {
      ++configured;
    } else {
      (void)cudaGetLastError();
    }
  }
  TVM_FFI_CHECK(configured > 0, RuntimeError)
      << "no CUDA device accepts PreferredSharedMemoryCarveout=" << kPushCarveout << " for the "
      << label;
}

// Both push kernels carry the same carveout; function-local static initialization in RunPush
// serializes the first call.
inline bool set_push_carveout() {
  if constexpr (kPushCarveout < 0) {
    return true;
  }
  set_push_carveout_for(reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_40242e9c64d6e66af6f5), "peer push");
  set_push_carveout_for(reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3), "wide peer push");
  return true;
}

// SM push route (padded M <= kSmPushMaxRows, N <= sm_push_max_cols(padded M)):
// barrier(phase); the push kernel
// (launched with the programmatic-dependent-launch attribute: its grid is
// dispatched while the barrier rendezvous is in flight and waits in hardware
// for the barrier's completion) copies every chunk of the local shard into
// every peer's scratch with system-scope 16-byte stores and publishes each
// chunk's epoch (one launch for all peers, no communication stream); the main
// kernel follows on the same stream with the same attribute (the push grid
// triggers its dispatch once every push CTA is resident; the GEMM gates each
// remote chunk on its epoch, so no second barrier is needed). The main kernel
// is the one of the call's block_n, as on the copy-engine route.
void RunPush(TensorView inp, TensorView scratch, TensorView weight, TensorView out,
             TensorView ready, tvm::ffi::Shape flag_peers, TensorView payload_peers,
             TensorView signal_peers, TensorView counters, int64_t rank, int64_t rows,
             int64_t pitch, int64_t phase, int64_t ready_target, int64_t grid_x, int64_t grid_y,
             int64_t grid_z, int64_t block_n, int64_t main_stream, int64_t comm_stream) {
  tvm::ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  Call call = resolve_call(inp, scratch, weight, out, ready, flag_peers, rank, rows, pitch, phase,
                           ready_target, grid_x, grid_y, grid_z, block_n, main_stream, comm_stream);
  check_cuda_tensor(payload_peers, "payload_peers");
  check_dtype(payload_peers, DLDataType{kDLInt, 64, 1}, "payload_peers");
  check_contiguous(payload_peers, "payload_peers");
  check_cuda_tensor(signal_peers, "signal_peers");
  check_dtype(signal_peers, DLDataType{kDLInt, 64, 1}, "signal_peers");
  check_contiguous(signal_peers, "signal_peers");
  check_cuda_tensor(counters, "counters");
  check_dtype(counters, DLDataType{kDLUInt, 32, 1}, "counters");
  check_contiguous(counters, "counters");
  check_same_device(payload_peers, inp, "payload_peers", "A_local");
  check_same_device(signal_peers, inp, "signal_peers", "A_local");
  check_same_device(counters, inp, "counters", "A_local");
  TVM_FFI_CHECK(payload_peers.numel() == kWorldSize - 1 && signal_peers.numel() == kWorldSize - 1,
                ValueError)
      << "push tables need exactly " << (kWorldSize - 1) << " entries";
  TVM_FFI_CHECK(counters.numel() >= (kWorldSize - 1) * call.num_chunks, ValueError)
      << "push counters need at least " << ((kWorldSize - 1) * call.num_chunks) << " words";
  TVM_FFI_CHECK(call.padded_rows <= kSmPushMaxRows, ValueError)
      << "the SM push route serves at most " << kSmPushMaxRows << " padded rows, got "
      << call.padded_rows;
  TVM_FFI_CHECK(call.cols <= sm_push_max_cols(call.padded_rows), ValueError)
      << "the SM push route serves at most N=" << sm_push_max_cols(call.padded_rows) << " at "
      << call.padded_rows << " padded rows, got N=" << call.cols;

  launch_barrier(phase, static_cast<int32_t>(kWorldSize), call.rank, call.flags, call.main_stream);
  {
    unsigned int* p_inp = static_cast<unsigned int*>(inp.data_ptr());
    long long* p_payload = static_cast<long long*>(payload_peers.data_ptr());
    long long* p_signal = static_cast<long long*>(signal_peers.data_ptr());
    unsigned int* p_counters = static_cast<unsigned int*>(counters.data_ptr());
    unsigned int v_ready_target = call.ready_target;
    int32_t v_rank = call.rank;
    int32_t v_rows = call.rows;
    int32_t v_chunk_rows = static_cast<int32_t>(call.chunk_rows);
    int32_t v_num_chunks = static_cast<int32_t>(call.num_chunks);
    void* kargs[] = {&p_inp,  &p_payload, &p_signal,     &p_counters,  &v_ready_target,
                     &v_rank, &v_rows,    &v_chunk_rows, &v_num_chunks};
    static const bool carveout_ready = set_push_carveout();
    (void)carveout_ready;
    // The launch of this call: the per-peer payload of the padded shard selects the grid and the
    // push program (both take the same arguments).
    const int64_t push_payload_bytes =
        call.padded_rows * kK * static_cast<int64_t>(inp.dtype().bits / 8);
    const bool wide_push = push_payload_bytes >= kPushPayloadThresholdBytes;
    const void* push_kernel = wide_push ? reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_f6b8a82f4b6bbe6b34f3)
                                        : reinterpret_cast<const void*>(kernel_cake_all_gather_matmul_40242e9c64d6e66af6f5);
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(wide_push ? kPushWideCtasPerPeer : kPushCtasPerPeer,
                          static_cast<uint32_t>(kWorldSize - 1), 1u);
    config.blockDim = dim3(kPushThreads, 1u, 1u);
    config.dynamicSmemBytes = 0u;
    config.stream = call.main_stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1u;
    cudaError_t status = cudaLaunchKernelExC(&config, push_kernel, kargs);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "cudaLaunchKernelExC for the peer push failed: " << cudaGetErrorString(status);
  }
  launch_main_for(call, inp, scratch, weight, out, ready, /*programmatic_launch=*/true);
}

}  // namespace cake_host_shim_seq_5e7dedbca8872e49

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_seq_5e7dedbca8872e49::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_push, cake_host_shim_seq_5e7dedbca8872e49::RunPush);
