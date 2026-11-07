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

// Shared tvm-ffi host binding of the `kmaj_p_main` argument plan: a hand fold of the exporter's per-program
// bindings, pending exporter regeneration.  The record's compile_flags select the program:
//   -DCAKE_PROGRAM_KERNEL (the kernel symbol its device source defines)
//   -DCAKE_BIND_SMEM (dynamic shared memory bytes)
//   -DCAKE_BIND_PAGED (paged or flat K / V descriptors)
//   -DCAKE_BIND_GQA (the query-group size the Q descriptor is tiled for)
// The preprocessed text equals the exporter's binding of that program apart from the file / line of the host checks.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

#ifndef CAKE_PROGRAM_KERNEL
#error "CAKE_PROGRAM_KERNEL (the kernel symbol the record's device source defines) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_SMEM
#error "CAKE_BIND_SMEM (dynamic shared memory bytes of the launch) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_PAGED
#error "CAKE_BIND_PAGED (1 for the paged-KV program, 0 for flat KV) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_GQA
#error "CAKE_BIND_GQA (the route's query-group size: 0, 8 or 16) must be defined on the compile line"
#endif
// Stringise / paste the expansion of a macro argument (the kernel name string and the binding
// namespace take the value the exporter's per-program binding spelled).
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#define CAKE_PASTE_(a, b) a##b
#define CAKE_PASTE(a, b) CAKE_PASTE_(a, b)
#define CAKE_BINDING_NS CAKE_PASTE(cake_binding_, CAKE_PROGRAM_KERNEL)

extern "C" __global__ void CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse);

namespace CAKE_BINDING_NS {

using tvm::ffi::TensorView;

constexpr const char* KERNEL_NAME = CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL);

void CheckCuda(cudaError_t status, const char* what) {
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << KERNEL_NAME << ": " << what << " failed: " << cudaGetErrorString(status);
}

void CheckCu(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    TVM_FFI_THROW(RuntimeError) << KERNEL_NAME << ": " << what << " failed with "
                                << (name ? name : "an unknown CUresult") << " (" << static_cast<int>(result)
                                << "): " << (message ? message : "");
  }
}

void* DataPtr(const TensorView& t) { return static_cast<char*>(t.data_ptr()) + t.byte_offset(); }

void CheckUnitInnerStride(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << KERNEL_NAME << ": '" << name << "' needs a unit innermost stride, got " << t.stride(-1);
}

#if CAKE_BIND_PAGED
// The product of the extents above the trailing `trailing` dims.
int64_t NumelAbove(const TensorView& t, int trailing) {
  int64_t product = 1;
  for (int axis = 0; axis < t.ndim() - trailing; ++axis) product *= t.size(axis);
  return product;
}

#endif
int64_t NonZero(int64_t divisor) {
  TVM_FFI_CHECK(divisor != 0, ValueError) << KERNEL_NAME << ": a descriptor formula divides by zero";
  return divisor;
}

int64_t CDiv(int64_t a, int64_t b) { return a / NonZero(b); }

int64_t CMod(int64_t a, int64_t b) { return a % NonZero(b); }

cuuint64_t Extent(int64_t value, const char* descriptor) {
  TVM_FFI_CHECK(value > 0, ValueError)
      << KERNEL_NAME << ": an extent of descriptor '" << descriptor << "' resolves to " << value;
  return static_cast<cuuint64_t>(value);
}

cuuint64_t Stride(int64_t value, const char* descriptor) {
  TVM_FFI_CHECK(value >= 0, ValueError)
      << KERNEL_NAME << ": a stride of descriptor '" << descriptor << "' resolves to " << value;
  return static_cast<cuuint64_t>(value);
}

CUtensorMap EncodeTma_q(const TensorView& t) {
#if CAKE_BIND_GQA == 16
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'q' takes a source of at least 3 dims, got " << t.ndim();
#else
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'q' takes a source of at least 2 dims, got " << t.ndim();
#endif
  CheckUnitInnerStride(t, "q");
  int64_t d1 = t.size(-1);
#if CAKE_BIND_GQA == 16
  int64_t d2 = t.size(-2);
  int64_t d3 = t.size(-3);
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'q' requires (CMod(d1, 64) == 0)";
  cuuint64_t global_dim[4] = {Extent(64, "q"), Extent(d2, "q"), Extent(CDiv(d1, 64), "q"), Extent(d3, "q")};
  cuuint64_t global_strides[3] = {Stride((d1 * 2), "q"), Stride(128, "q"), Stride(((d2 * d1) * 2), "q")};
  cuuint32_t box_dim[4] = {64u, 16u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
#else
  int64_t outer1 = NumelAbove(t, 1);
  check_dense_leading_dims(t, 1, "q");
  int64_t s2 = t.stride(-2) * 2;
  cuuint64_t global_dim[2] = {Extent(d1, "q"), Extent(outer1, "q")};
  cuuint64_t global_strides[1] = {Stride(s2, "q")};
  cuuint32_t box_dim[2] = {64u, 1u};
  cuuint32_t elem_strides[2] = {1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1], ValueError)
#endif
      << KERNEL_NAME << ": the box of descriptor 'q' exceeds its source extents";
#if CAKE_BIND_GQA == 16
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
#else
  TVM_FFI_CHECK(global_strides[0] % 16 == 0, ValueError)
#endif
      << KERNEL_NAME << ": the strides of descriptor 'q' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'q' must be 16-byte aligned";
  CUtensorMap map{};
#if CAKE_BIND_GQA == 16
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, address, global_dim, global_strides,
#else
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, address, global_dim, global_strides,
#endif
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of q");
  return map;
}

CUtensorMap EncodeTma_k(const TensorView& t) {
#if CAKE_BIND_PAGED
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'k' takes a source of at least 2 dims, got " << t.ndim();
#else
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'k' takes a source of at least 3 dims, got " << t.ndim();
#endif
  CheckUnitInnerStride(t, "k");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
#if CAKE_BIND_PAGED
  int64_t outer2 = NumelAbove(t, 2);
#else
  int64_t d3 = t.size(-3);
#endif
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'k' requires (CMod(d1, 64) == 0)";
#if CAKE_BIND_PAGED
  cuuint64_t global_dim[4] = {Extent(64, "k"), Extent(d2, "k"), Extent(CDiv(d1, 64), "k"), Extent(outer2, "k")};
  cuuint64_t global_strides[3] = {Stride((d1 * 2), "k"), Stride(128, "k"), Stride(((d2 * d1) * 2), "k")};
#else
  cuuint64_t global_dim[4] = {Extent(64, "k"), Extent(d3, "k"), Extent(CDiv(d1, 64), "k"), Extent(d2, "k")};
  cuuint64_t global_strides[3] = {Stride(((d2 * d1) * 2), "k"), Stride(128, "k"), Stride((d1 * 2), "k")};
#endif
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'k' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'k' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'k' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of k");
  return map;
}

CUtensorMap EncodeTma_v(const TensorView& t) {
#if CAKE_BIND_PAGED
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'v' takes a source of at least 2 dims, got " << t.ndim();
#else
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'v' takes a source of at least 3 dims, got " << t.ndim();
#endif
  CheckUnitInnerStride(t, "v");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
#if CAKE_BIND_PAGED
  int64_t outer2 = NumelAbove(t, 2);
#else
  int64_t d3 = t.size(-3);
#endif
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'v' requires (CMod(d1, 64) == 0)";
#if CAKE_BIND_PAGED
  cuuint64_t global_dim[4] = {Extent(64, "v"), Extent(d2, "v"), Extent(CDiv(d1, 64), "v"), Extent(outer2, "v")};
  cuuint64_t global_strides[3] = {Stride((d1 * 2), "v"), Stride(128, "v"), Stride(((d2 * d1) * 2), "v")};
#else
  cuuint64_t global_dim[4] = {Extent(64, "v"), Extent(d3, "v"), Extent(CDiv(d1, 64), "v"), Extent(d2, "v")};
  cuuint64_t global_strides[3] = {Stride(((d2 * d1) * 2), "v"), Stride(128, "v"), Stride((d1 * 2), "v")};
#endif
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'v' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'v' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'v' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of v");
  return map;
}

void Run(TensorView a_q, TensorView a_k, TensorView a_v, TensorView a_scheduler_metadata, TensorView a_k2q_row_ptr, TensorView a_k2q_qsplit_indices, TensorView a_partial_o, TensorView a_partial_lse, TensorView a_partial_temperature_lse, TensorView a_cu_seqlens_q, TensorView a_cu_seqlens_k, TensorView a_q_offsets, TensorView a_kv_lens, TensorView a_page_table, int64_t a_total_q, int64_t a_num_q_heads, int64_t a_num_kv_heads, int64_t a_total_rows, int64_t a_nnz_per_head, int64_t a_num_work_items, int64_t a_topk, int64_t a_max_pages, int64_t a_causal, int64_t a_derive_q_offset, double a_softmax_scale_log2, double a_lse_temperature_scale, int64_t a_return_temperature_lse, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = a_q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(a_q, "q");
  check_dtype(a_q, DLDataType{kDLBfloat, 16, 1}, "q");
  check_cuda_tensor(a_k, "k");
  check_dtype(a_k, DLDataType{kDLBfloat, 16, 1}, "k");
  check_cuda_tensor(a_v, "v");
  check_dtype(a_v, DLDataType{kDLBfloat, 16, 1}, "v");
  check_cuda_tensor(a_scheduler_metadata, "scheduler_metadata");
  check_dtype(a_scheduler_metadata, DLDataType{kDLInt, 32, 1}, "scheduler_metadata");
  check_cuda_tensor(a_k2q_row_ptr, "k2q_row_ptr");
  check_dtype(a_k2q_row_ptr, DLDataType{kDLInt, 32, 1}, "k2q_row_ptr");
  check_cuda_tensor(a_k2q_qsplit_indices, "k2q_qsplit_indices");
  check_dtype(a_k2q_qsplit_indices, DLDataType{kDLInt, 32, 1}, "k2q_qsplit_indices");
  check_cuda_tensor(a_partial_o, "partial_o");
  check_dtype(a_partial_o, DLDataType{kDLBfloat, 16, 1}, "partial_o");
  check_cuda_tensor(a_partial_lse, "partial_lse");
  check_dtype(a_partial_lse, DLDataType{kDLFloat, 32, 1}, "partial_lse");
  check_cuda_tensor(a_partial_temperature_lse, "partial_temperature_lse");
  check_dtype(a_partial_temperature_lse, DLDataType{kDLFloat, 32, 1}, "partial_temperature_lse");
  check_cuda_tensor(a_cu_seqlens_q, "cu_seqlens_q");
  check_dtype(a_cu_seqlens_q, DLDataType{kDLInt, 32, 1}, "cu_seqlens_q");
  check_cuda_tensor(a_cu_seqlens_k, "cu_seqlens_k");
  check_dtype(a_cu_seqlens_k, DLDataType{kDLInt, 32, 1}, "cu_seqlens_k");
  check_cuda_tensor(a_q_offsets, "q_offsets");
  check_dtype(a_q_offsets, DLDataType{kDLInt, 32, 1}, "q_offsets");
  check_cuda_tensor(a_kv_lens, "kv_lens");
  check_dtype(a_kv_lens, DLDataType{kDLInt, 32, 1}, "kv_lens");
  check_cuda_tensor(a_page_table, "page_table");
  check_dtype(a_page_table, DLDataType{kDLInt, 32, 1}, "page_table");
  TVM_FFI_CHECK(a_total_q >= -2147483648LL && a_total_q <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'total_q' = " << a_total_q
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_q_heads >= -2147483648LL && a_num_q_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_q_heads' = " << a_num_q_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_kv_heads >= -2147483648LL && a_num_kv_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_kv_heads' = " << a_num_kv_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_total_rows >= -2147483648LL && a_total_rows <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'total_rows' = " << a_total_rows
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_nnz_per_head >= -2147483648LL && a_nnz_per_head <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'nnz_per_head' = " << a_nnz_per_head
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_work_items >= -2147483648LL && a_num_work_items <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_work_items' = " << a_num_work_items
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_topk >= -2147483648LL && a_topk <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'topk' = " << a_topk
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_max_pages >= -2147483648LL && a_max_pages <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'max_pages' = " << a_max_pages
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_causal >= -2147483648LL && a_causal <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'causal' = " << a_causal
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_derive_q_offset >= -2147483648LL && a_derive_q_offset <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'derive_q_offset' = " << a_derive_q_offset
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_return_temperature_lse >= -2147483648LL && a_return_temperature_lse <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'return_temperature_lse' = " << a_return_temperature_lse
      << " is outside the i32 range [-2147483648, 2147483647]";
#if CAKE_BIND_GQA == 16
  check_contiguous(a_q, "q");
#endif
  check_contiguous(a_k, "k");
  check_same_device(a_k, a_q, "k", "q");
  check_contiguous(a_v, "v");
  check_same_device(a_v, a_q, "v", "q");
  check_contiguous(a_scheduler_metadata, "scheduler_metadata");
  check_same_device(a_scheduler_metadata, a_q, "scheduler_metadata", "q");
  check_contiguous(a_k2q_row_ptr, "k2q_row_ptr");
  check_same_device(a_k2q_row_ptr, a_q, "k2q_row_ptr", "q");
  check_contiguous(a_k2q_qsplit_indices, "k2q_qsplit_indices");
  check_same_device(a_k2q_qsplit_indices, a_q, "k2q_qsplit_indices", "q");
  check_contiguous(a_partial_o, "partial_o");
  check_same_device(a_partial_o, a_q, "partial_o", "q");
  check_contiguous(a_partial_lse, "partial_lse");
  check_same_device(a_partial_lse, a_q, "partial_lse", "q");
  check_contiguous(a_partial_temperature_lse, "partial_temperature_lse");
  check_same_device(a_partial_temperature_lse, a_q, "partial_temperature_lse", "q");
  check_contiguous(a_cu_seqlens_q, "cu_seqlens_q");
  check_same_device(a_cu_seqlens_q, a_q, "cu_seqlens_q", "q");
  check_contiguous(a_cu_seqlens_k, "cu_seqlens_k");
  check_same_device(a_cu_seqlens_k, a_q, "cu_seqlens_k", "q");
  check_contiguous(a_q_offsets, "q_offsets");
  check_same_device(a_q_offsets, a_q, "q_offsets", "q");
  check_contiguous(a_kv_lens, "kv_lens");
  check_same_device(a_kv_lens, a_q, "kv_lens", "q");
  check_contiguous(a_page_table, "page_table");
  check_same_device(a_page_table, a_q, "page_table", "q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") must be positive";
  TVM_FFI_CHECK(grid_x <= 2147483647 && grid_y <= 65535 && grid_z <= 65535, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") exceeds CUDA's limits";
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  {
    // Once per thread per device: the kernel's function attributes hold per device; the tensor-map encoders need this thread's context of the device bound.
    static thread_local uint64_t prepared_devices = 0;
    const uint64_t device_bit = dev.device_id < 64 ? uint64_t{1} << dev.device_id : uint64_t{0};
    if (device_bit == 0 || (prepared_devices & device_bit) == 0) {
      CheckCuda(cudaSetDevice(dev.device_id), "cudaSetDevice");
      CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), cudaFuncAttributeMaxDynamicSharedMemorySize, CAKE_BIND_SMEM), "cudaFuncSetAttribute");
      prepared_devices |= device_bit;
    }
  }
  CUtensorMap v_q = EncodeTma_q(a_q);
  CUtensorMap v_k = EncodeTma_k(a_k);
  CUtensorMap v_v = EncodeTma_v(a_v);
  void* v_scheduler_metadata = DataPtr(a_scheduler_metadata);
  void* v_k2q_row_ptr = DataPtr(a_k2q_row_ptr);
  void* v_k2q_qsplit_indices = DataPtr(a_k2q_qsplit_indices);
  void* v_partial_o = DataPtr(a_partial_o);
  void* v_partial_lse = DataPtr(a_partial_lse);
  void* v_partial_temperature_lse = DataPtr(a_partial_temperature_lse);
  void* v_cu_seqlens_q = DataPtr(a_cu_seqlens_q);
  void* v_cu_seqlens_k = DataPtr(a_cu_seqlens_k);
  void* v_q_offsets = DataPtr(a_q_offsets);
  void* v_kv_lens = DataPtr(a_kv_lens);
  void* v_page_table = DataPtr(a_page_table);
  int32_t v_total_q = static_cast<int32_t>(a_total_q);
  int32_t v_num_q_heads = static_cast<int32_t>(a_num_q_heads);
  int32_t v_num_kv_heads = static_cast<int32_t>(a_num_kv_heads);
  int32_t v_total_rows = static_cast<int32_t>(a_total_rows);
  int32_t v_nnz_per_head = static_cast<int32_t>(a_nnz_per_head);
  int32_t v_num_work_items = static_cast<int32_t>(a_num_work_items);
  int32_t v_topk = static_cast<int32_t>(a_topk);
  int32_t v_max_pages = static_cast<int32_t>(a_max_pages);
  int32_t v_causal = static_cast<int32_t>(a_causal);
  int32_t v_derive_q_offset = static_cast<int32_t>(a_derive_q_offset);
  float v_softmax_scale_log2 = static_cast<float>(a_softmax_scale_log2);
  float v_lse_temperature_scale = static_cast<float>(a_lse_temperature_scale);
  int32_t v_return_temperature_lse = static_cast<int32_t>(a_return_temperature_lse);
  void* kernel_args[] = {&v_q, &v_k, &v_v, &v_scheduler_metadata, &v_k2q_row_ptr, &v_k2q_qsplit_indices, &v_partial_o, &v_partial_lse, &v_partial_temperature_lse, &v_cu_seqlens_q, &v_cu_seqlens_k, &v_q_offsets, &v_kv_lens, &v_page_table, &v_total_q, &v_num_q_heads, &v_num_kv_heads, &v_total_rows, &v_nnz_per_head, &v_num_work_items, &v_topk, &v_max_pages, &v_causal, &v_derive_q_offset, &v_softmax_scale_log2, &v_lse_temperature_scale, &v_return_temperature_lse};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z));
  config.blockDim = dim3(512, 1, 1);
  config.dynamicSmemBytes = CAKE_BIND_SMEM;
  config.stream = stream;
  CheckCuda(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kernel_args),
            "cudaLaunchKernelExC");
}

}  // namespace CAKE_BINDING_NS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_BINDING_NS::Run);
