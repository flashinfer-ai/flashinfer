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

// Shared tvm-ffi host binding of the `decode_m16` argument plan: a hand fold of the exporter's per-program
// bindings, pending exporter regeneration.  The record's compile_flags select the program:
//   -DCAKE_PROGRAM_KERNEL (the kernel symbol its device source defines)
//   -DCAKE_BIND_SMEM (dynamic shared memory bytes)
//   -DCAKE_BIND_PAGED (paged or flat K / V descriptors)
//   -DCAKE_BIND_KV_FP8 (float8_e4m3fn K / V)
//   -DCAKE_BIND_Q_FP16 (float16 Q / O)
// The preprocessed text equals the exporter's binding of that program apart from the file / line of the host checks.
#include <cuda_runtime.h>
#include <cuda.h>
#if CAKE_BIND_Q_FP16
#include <cuda_fp16.h>
#define CAKE_Q_T __half
#define CAKE_Q_TMA_DTYPE CU_TENSOR_MAP_DATA_TYPE_FLOAT16
#define CAKE_Q_DL kDLFloat
#else
#include <cuda_bf16.h>
#define CAKE_Q_T __nv_bfloat16
#define CAKE_Q_TMA_DTYPE CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
#define CAKE_Q_DL kDLBfloat
#endif

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
#ifndef CAKE_BIND_KV_FP8
#error "CAKE_BIND_KV_FP8 (1 when K / V are float8_e4m3fn, else 0) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_Q_FP16
#error "CAKE_BIND_Q_FP16 (1 when Q / O are float16, 0 for bfloat16) must be defined on the compile line"
#endif
// Stringise / paste the expansion of a macro argument (the kernel name string and the binding
// namespace take the value the exporter's per-program binding spelled).
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#define CAKE_PASTE_(a, b) a##b
#define CAKE_PASTE(a, b) CAKE_PASTE_(a, b)
#define CAKE_BINDING_NS CAKE_PASTE(cake_binding_, CAKE_PROGRAM_KERNEL)

extern "C" __global__ void CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap Q_prefill, CAKE_Q_T* __restrict__ Q_prefill_raw, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap K_prefill_pair, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, const __grid_constant__ CUtensorMap V_prefill_pair, const __grid_constant__ CUtensorMap KV, CAKE_Q_T* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ qo_indptr, int* __restrict__ kv_indptr, int* __restrict__ kv_len_arr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int* __restrict__ task_q_tile, int* __restrict__ task_split, int* __restrict__ task_kv_tile_begin, int* __restrict__ task_kv_tile_end, int* __restrict__ task_qo_begin, int* __restrict__ task_qo_end, int* __restrict__ task_page_begin, int* __restrict__ task_page_end, int* __restrict__ status, int num_requests, int num_q_heads, int num_kv_heads, int max_kv_tiles, int max_splits, int max_task_claims, float softmax_scale_log2, float k_global_scale, float v_global_scale, int attention_mode, int is_causal, int derive_q_offset, int record_tasks, int msa_max_pages, int msa_split_policy);

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

// The product of the extents above the trailing `trailing` dims.
int64_t NumelAbove(const TensorView& t, int trailing) {
  int64_t product = 1;
  for (int axis = 0; axis < t.ndim() - trailing; ++axis) product *= t.size(axis);
  return product;
}

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

CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'Q' takes a source of at least 2 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "Q");
  int64_t d1 = t.size(-1);
  int64_t outer1 = NumelAbove(t, 1);
  check_dense_leading_dims(t, 1, "Q");
  int64_t s2 = t.stride(-2) * 2;
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'Q' requires (CMod(d1, 64) == 0)";
  cuuint64_t global_dim[3] = {Extent(64, "Q"), Extent(outer1, "Q"), Extent(CDiv(d1, 64), "Q")};
  cuuint64_t global_strides[2] = {Stride(s2, "Q"), Stride(128, "Q")};
  cuuint32_t box_dim[3] = {64u, 16u, 2u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'Q' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'Q' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'Q' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 3, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of Q");
  return map;
}

CUtensorMap EncodeTma_Q_prefill(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'Q_prefill' takes a source of at least 3 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "Q_prefill");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
  int64_t d3 = t.size(-3);
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'Q_prefill' requires (CMod(d1, 64) == 0)";
  cuuint64_t global_dim[4] = {Extent(64, "Q_prefill"), Extent(d3, "Q_prefill"), Extent(d2, "Q_prefill"), Extent(CDiv(d1, 64), "Q_prefill")};
  cuuint64_t global_strides[3] = {Stride(((d2 * d1) * 2), "Q_prefill"), Stride((d1 * 2), "Q_prefill"), Stride(128, "Q_prefill")};
  cuuint32_t box_dim[4] = {64u, 128u, 1u, 2u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'Q_prefill' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'Q_prefill' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'Q_prefill' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 4, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of Q_prefill");
  return map;
}

CUtensorMap EncodeTma_K(const TensorView& t) {
#if CAKE_BIND_PAGED
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'K' takes a source of at least 2 dims, got " << t.ndim();
#else
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'K' takes a source of at least 3 dims, got " << t.ndim();
#endif
  CheckUnitInnerStride(t, "K");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
#if CAKE_BIND_PAGED
  int64_t outer2 = NumelAbove(t, 2);
#else
  int64_t d3 = t.size(-3);
#endif
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'K' requires (CMod(d1, 64) == 0)";
#if CAKE_BIND_KV_FP8
#if CAKE_BIND_PAGED
  cuuint64_t global_dim[3] = {Extent(128, "K"), Extent(d2, "K"), Extent(outer2, "K")};
  cuuint64_t global_strides[2] = {Stride(d1, "K"), Stride((d2 * d1), "K")};
  cuuint32_t box_dim[3] = {128u, 64u, 1u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2], ValueError)
#else
  cuuint64_t global_dim[3] = {Extent(128, "K"), Extent(d3, "K"), Extent(d2, "K")};
  cuuint64_t global_strides[2] = {Stride((d2 * d1), "K"), Stride(d1, "K")};
  cuuint32_t box_dim[3] = {128u, 64u, 1u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2], ValueError)
#endif
#else
#if CAKE_BIND_PAGED
  cuuint64_t global_dim[4] = {Extent(64, "K"), Extent(d2, "K"), Extent(CDiv(d1, 64), "K"), Extent(outer2, "K")};
  cuuint64_t global_strides[3] = {Stride((d1 * 2), "K"), Stride(128, "K"), Stride(((d2 * d1) * 2), "K")};
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
#else
  cuuint64_t global_dim[4] = {Extent(64, "K"), Extent(d3, "K"), Extent(CDiv(d1, 64), "K"), Extent(d2, "K")};
  cuuint64_t global_strides[3] = {Stride(((d2 * d1) * 2), "K"), Stride(128, "K"), Stride((d1 * 2), "K")};
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
#endif
#endif
      << KERNEL_NAME << ": the box of descriptor 'K' exceeds its source extents";
#if CAKE_BIND_KV_FP8
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0, ValueError)
#else
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
#endif
      << KERNEL_NAME << ": the strides of descriptor 'K' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'K' must be 16-byte aligned";
  CUtensorMap map{};
#if CAKE_BIND_KV_FP8
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
#else
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 4, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
#endif
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of K");
  return map;
}

CUtensorMap EncodeTma_K_scale(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << KERNEL_NAME << ": descriptor 'K_scale' takes a source of at least 4 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "K_scale");
  int64_t d3 = t.size(-3);
  int64_t d4 = t.size(-4);
  int64_t s3 = t.stride(-3) * 1;
  int64_t s4 = t.stride(-4) * 1;
  cuuint64_t global_dim[3] = {Extent(1024, "K_scale"), Extent(d3, "K_scale"), Extent(d4, "K_scale")};
  cuuint64_t global_strides[2] = {Stride(s3, "K_scale"), Stride(s4, "K_scale")};
  cuuint32_t box_dim[3] = {256u, 1u, 1u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'K_scale' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'K_scale' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'K_scale' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of K_scale");
  return map;
}

CUtensorMap EncodeTma_K_prefill_pair(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'K_prefill_pair' takes a source of at least 3 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "K_prefill_pair");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
  int64_t d3 = t.size(-3);
  int64_t outer3 = NumelAbove(t, 3);
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'K_prefill_pair' requires (CMod(d1, 64) == 0)";
  cuuint64_t global_dim[5] = {Extent(64, "K_prefill_pair"), Extent(d2, "K_prefill_pair"), Extent(outer3, "K_prefill_pair"), Extent(CDiv(d1, 64), "K_prefill_pair"), Extent(d3, "K_prefill_pair")};
  cuuint64_t global_strides[4] = {Stride((d1 * 2), "K_prefill_pair"), Stride((((d3 * d2) * d1) * 2), "K_prefill_pair"), Stride(128, "K_prefill_pair"), Stride(((d2 * d1) * 2), "K_prefill_pair")};
  cuuint32_t box_dim[5] = {64u, 64u, 2u, 2u, 1u};
  cuuint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3] && box_dim[4] <= global_dim[4], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'K_prefill_pair' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0 && global_strides[3] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'K_prefill_pair' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'K_prefill_pair' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 5, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of K_prefill_pair");
  return map;
}

CUtensorMap EncodeTma_V(const TensorView& t) {
#if CAKE_BIND_PAGED
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'V' takes a source of at least 2 dims, got " << t.ndim();
#else
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'V' takes a source of at least 3 dims, got " << t.ndim();
#endif
  CheckUnitInnerStride(t, "V");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
#if CAKE_BIND_PAGED
  int64_t outer2 = NumelAbove(t, 2);
#else
  int64_t d3 = t.size(-3);
#endif
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'V' requires (CMod(d1, 64) == 0)";
#if CAKE_BIND_KV_FP8
#if CAKE_BIND_PAGED
  cuuint64_t global_dim[3] = {Extent(128, "V"), Extent(d2, "V"), Extent(outer2, "V")};
  cuuint64_t global_strides[2] = {Stride(d1, "V"), Stride((d2 * d1), "V")};
  cuuint32_t box_dim[3] = {128u, 64u, 1u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2], ValueError)
#else
  cuuint64_t global_dim[3] = {Extent(128, "V"), Extent(d3, "V"), Extent(d2, "V")};
  cuuint64_t global_strides[2] = {Stride((d2 * d1), "V"), Stride(d1, "V")};
  cuuint32_t box_dim[3] = {128u, 64u, 1u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2], ValueError)
#endif
#else
#if CAKE_BIND_PAGED
  cuuint64_t global_dim[4] = {Extent(64, "V"), Extent(d2, "V"), Extent(CDiv(d1, 64), "V"), Extent(outer2, "V")};
  cuuint64_t global_strides[3] = {Stride((d1 * 2), "V"), Stride(128, "V"), Stride(((d2 * d1) * 2), "V")};
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
#else
  cuuint64_t global_dim[4] = {Extent(64, "V"), Extent(d3, "V"), Extent(CDiv(d1, 64), "V"), Extent(d2, "V")};
  cuuint64_t global_strides[3] = {Stride(((d2 * d1) * 2), "V"), Stride(128, "V"), Stride((d1 * 2), "V")};
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
#endif
#endif
      << KERNEL_NAME << ": the box of descriptor 'V' exceeds its source extents";
#if CAKE_BIND_KV_FP8
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0, ValueError)
#else
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
#endif
      << KERNEL_NAME << ": the strides of descriptor 'V' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'V' must be 16-byte aligned";
  CUtensorMap map{};
#if CAKE_BIND_KV_FP8
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
#else
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 4, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
#endif
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of V");
  return map;
}

CUtensorMap EncodeTma_V_scale(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << KERNEL_NAME << ": descriptor 'V_scale' takes a source of at least 4 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "V_scale");
  int64_t d3 = t.size(-3);
  int64_t d4 = t.size(-4);
  int64_t s3 = t.stride(-3) * 1;
  int64_t s4 = t.stride(-4) * 1;
  cuuint64_t global_dim[3] = {Extent(1024, "V_scale"), Extent(d3, "V_scale"), Extent(d4, "V_scale")};
  cuuint64_t global_strides[2] = {Stride(s3, "V_scale"), Stride(s4, "V_scale")};
  cuuint32_t box_dim[3] = {256u, 1u, 1u};
  cuuint32_t elem_strides[3] = {1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'V_scale' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'V_scale' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'V_scale' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of V_scale");
  return map;
}

CUtensorMap EncodeTma_V_prefill_pair(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'V_prefill_pair' takes a source of at least 3 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "V_prefill_pair");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
  int64_t d3 = t.size(-3);
  int64_t outer3 = NumelAbove(t, 3);
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'V_prefill_pair' requires (CMod(d1, 64) == 0)";
  cuuint64_t global_dim[5] = {Extent(64, "V_prefill_pair"), Extent(d2, "V_prefill_pair"), Extent(outer3, "V_prefill_pair"), Extent(CDiv(d1, 64), "V_prefill_pair"), Extent(d3, "V_prefill_pair")};
  cuuint64_t global_strides[4] = {Stride((d1 * 2), "V_prefill_pair"), Stride((((d3 * d2) * d1) * 2), "V_prefill_pair"), Stride(128, "V_prefill_pair"), Stride(((d2 * d1) * 2), "V_prefill_pair")};
  cuuint32_t box_dim[5] = {64u, 64u, 2u, 2u, 1u};
  cuuint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3] && box_dim[4] <= global_dim[4], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'V_prefill_pair' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0 && global_strides[3] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'V_prefill_pair' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'V_prefill_pair' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 5, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of V_prefill_pair");
  return map;
}

CUtensorMap EncodeTma_KV(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'KV' takes a source of at least 2 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "KV");
  int64_t d1 = t.size(-1);
  int64_t outer1 = NumelAbove(t, 1);
  check_dense_leading_dims(t, 1, "KV");
  int64_t s2 = t.stride(-2) * 2;
  cuuint64_t global_dim[2] = {Extent(d1, "KV"), Extent(outer1, "KV")};
  cuuint64_t global_strides[1] = {Stride(s2, "KV")};
  cuuint32_t box_dim[2] = {64u, 1u};
  cuuint32_t elem_strides[2] = {1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'KV' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'KV' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'KV' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CAKE_Q_TMA_DTYPE, 2, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of KV");
  return map;
}

void Run(TensorView a_Q, TensorView a_Q_prefill, TensorView a_Q_prefill_raw, TensorView a_K, TensorView a_K_scale, TensorView a_K_prefill_pair, TensorView a_V, TensorView a_V_scale, TensorView a_V_prefill_pair, TensorView a_KV, TensorView a_O, TensorView a_partial_O, TensorView a_partial_M, TensorView a_partial_D, TensorView a_split_completion, TensorView a_msa_lse, TensorView a_kv_indices, TensorView a_qo_indptr, TensorView a_kv_indptr, TensorView a_kv_len_arr, TensorView a_task_kind, TensorView a_task_request, TensorView a_task_kv_head, TensorView a_task_q_tile, TensorView a_task_split, TensorView a_task_kv_tile_begin, TensorView a_task_kv_tile_end, TensorView a_task_qo_begin, TensorView a_task_qo_end, TensorView a_task_page_begin, TensorView a_task_page_end, TensorView a_status, int64_t a_num_requests, int64_t a_num_q_heads, int64_t a_num_kv_heads, int64_t a_max_kv_tiles, int64_t a_max_splits, int64_t a_max_task_claims, double a_softmax_scale_log2, double a_k_global_scale, double a_v_global_scale, int64_t a_attention_mode, int64_t a_is_causal, int64_t a_derive_q_offset, int64_t a_record_tasks, int64_t a_msa_max_pages, int64_t a_msa_split_policy, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = a_Q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(a_Q, "Q");
  check_dtype(a_Q, DLDataType{CAKE_Q_DL, 16, 1}, "Q");
  check_cuda_tensor(a_Q_prefill, "Q_prefill");
  check_dtype(a_Q_prefill, DLDataType{CAKE_Q_DL, 16, 1}, "Q_prefill");
  check_cuda_tensor(a_Q_prefill_raw, "Q_prefill_raw");
  check_dtype(a_Q_prefill_raw, DLDataType{CAKE_Q_DL, 16, 1}, "Q_prefill_raw");
  check_cuda_tensor(a_K, "K");
#if CAKE_BIND_KV_FP8
  check_dtype(a_K, DLDataType{kDLUInt, 8, 1}, "K");
#else
  check_dtype(a_K, DLDataType{CAKE_Q_DL, 16, 1}, "K");
#endif
  check_cuda_tensor(a_K_scale, "K_scale");
  check_dtype(a_K_scale, DLDataType{kDLUInt, 8, 1}, "K_scale");
  check_cuda_tensor(a_K_prefill_pair, "K_prefill_pair");
  check_dtype(a_K_prefill_pair, DLDataType{CAKE_Q_DL, 16, 1}, "K_prefill_pair");
  check_cuda_tensor(a_V, "V");
#if CAKE_BIND_KV_FP8
  check_dtype(a_V, DLDataType{kDLUInt, 8, 1}, "V");
#else
  check_dtype(a_V, DLDataType{CAKE_Q_DL, 16, 1}, "V");
#endif
  check_cuda_tensor(a_V_scale, "V_scale");
  check_dtype(a_V_scale, DLDataType{kDLUInt, 8, 1}, "V_scale");
  check_cuda_tensor(a_V_prefill_pair, "V_prefill_pair");
  check_dtype(a_V_prefill_pair, DLDataType{CAKE_Q_DL, 16, 1}, "V_prefill_pair");
  check_cuda_tensor(a_KV, "KV");
  check_dtype(a_KV, DLDataType{CAKE_Q_DL, 16, 1}, "KV");
  check_cuda_tensor(a_O, "O");
  check_dtype(a_O, DLDataType{CAKE_Q_DL, 16, 1}, "O");
  check_cuda_tensor(a_partial_O, "partial_O");
  check_dtype(a_partial_O, DLDataType{kDLFloat, 32, 1}, "partial_O");
  check_cuda_tensor(a_partial_M, "partial_M");
  check_dtype(a_partial_M, DLDataType{kDLFloat, 32, 1}, "partial_M");
  check_cuda_tensor(a_partial_D, "partial_D");
  check_dtype(a_partial_D, DLDataType{kDLFloat, 32, 1}, "partial_D");
  check_cuda_tensor(a_split_completion, "split_completion");
  check_dtype(a_split_completion, DLDataType{kDLInt, 32, 1}, "split_completion");
  check_cuda_tensor(a_msa_lse, "msa_lse");
  check_dtype(a_msa_lse, DLDataType{kDLFloat, 32, 1}, "msa_lse");
  check_cuda_tensor(a_kv_indices, "kv_indices");
  check_dtype(a_kv_indices, DLDataType{kDLInt, 32, 1}, "kv_indices");
  check_cuda_tensor(a_qo_indptr, "qo_indptr");
  check_dtype(a_qo_indptr, DLDataType{kDLInt, 32, 1}, "qo_indptr");
  check_cuda_tensor(a_kv_indptr, "kv_indptr");
  check_dtype(a_kv_indptr, DLDataType{kDLInt, 32, 1}, "kv_indptr");
  check_cuda_tensor(a_kv_len_arr, "kv_len_arr");
  check_dtype(a_kv_len_arr, DLDataType{kDLInt, 32, 1}, "kv_len_arr");
  check_cuda_tensor(a_task_kind, "task_kind");
  check_dtype(a_task_kind, DLDataType{kDLInt, 32, 1}, "task_kind");
  check_cuda_tensor(a_task_request, "task_request");
  check_dtype(a_task_request, DLDataType{kDLInt, 32, 1}, "task_request");
  check_cuda_tensor(a_task_kv_head, "task_kv_head");
  check_dtype(a_task_kv_head, DLDataType{kDLInt, 32, 1}, "task_kv_head");
  check_cuda_tensor(a_task_q_tile, "task_q_tile");
  check_dtype(a_task_q_tile, DLDataType{kDLInt, 32, 1}, "task_q_tile");
  check_cuda_tensor(a_task_split, "task_split");
  check_dtype(a_task_split, DLDataType{kDLInt, 32, 1}, "task_split");
  check_cuda_tensor(a_task_kv_tile_begin, "task_kv_tile_begin");
  check_dtype(a_task_kv_tile_begin, DLDataType{kDLInt, 32, 1}, "task_kv_tile_begin");
  check_cuda_tensor(a_task_kv_tile_end, "task_kv_tile_end");
  check_dtype(a_task_kv_tile_end, DLDataType{kDLInt, 32, 1}, "task_kv_tile_end");
  check_cuda_tensor(a_task_qo_begin, "task_qo_begin");
  check_dtype(a_task_qo_begin, DLDataType{kDLInt, 32, 1}, "task_qo_begin");
  check_cuda_tensor(a_task_qo_end, "task_qo_end");
  check_dtype(a_task_qo_end, DLDataType{kDLInt, 32, 1}, "task_qo_end");
  check_cuda_tensor(a_task_page_begin, "task_page_begin");
  check_dtype(a_task_page_begin, DLDataType{kDLInt, 32, 1}, "task_page_begin");
  check_cuda_tensor(a_task_page_end, "task_page_end");
  check_dtype(a_task_page_end, DLDataType{kDLInt, 32, 1}, "task_page_end");
  check_cuda_tensor(a_status, "status");
  check_dtype(a_status, DLDataType{kDLInt, 32, 1}, "status");
  TVM_FFI_CHECK(a_num_requests >= -2147483648LL && a_num_requests <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_requests' = " << a_num_requests
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_q_heads >= -2147483648LL && a_num_q_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_q_heads' = " << a_num_q_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_kv_heads >= -2147483648LL && a_num_kv_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_kv_heads' = " << a_num_kv_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_max_kv_tiles >= -2147483648LL && a_max_kv_tiles <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'max_kv_tiles' = " << a_max_kv_tiles
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_max_splits >= -2147483648LL && a_max_splits <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'max_splits' = " << a_max_splits
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_max_task_claims >= -2147483648LL && a_max_task_claims <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'max_task_claims' = " << a_max_task_claims
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_attention_mode >= -2147483648LL && a_attention_mode <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'attention_mode' = " << a_attention_mode
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_is_causal >= -2147483648LL && a_is_causal <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'is_causal' = " << a_is_causal
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_derive_q_offset >= -2147483648LL && a_derive_q_offset <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'derive_q_offset' = " << a_derive_q_offset
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_record_tasks >= -2147483648LL && a_record_tasks <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'record_tasks' = " << a_record_tasks
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_msa_max_pages >= -2147483648LL && a_msa_max_pages <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'msa_max_pages' = " << a_msa_max_pages
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_msa_split_policy >= -2147483648LL && a_msa_split_policy <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'msa_split_policy' = " << a_msa_split_policy
      << " is outside the i32 range [-2147483648, 2147483647]";
  check_contiguous(a_Q_prefill, "Q_prefill");
  check_same_device(a_Q_prefill, a_Q, "Q_prefill", "Q");
  check_contiguous(a_Q_prefill_raw, "Q_prefill_raw");
  check_same_device(a_Q_prefill_raw, a_Q, "Q_prefill_raw", "Q");
  check_contiguous(a_K, "K");
  check_same_device(a_K, a_Q, "K", "Q");
  check_same_device(a_K_scale, a_Q, "K_scale", "Q");
  check_contiguous(a_K_prefill_pair, "K_prefill_pair");
  check_same_device(a_K_prefill_pair, a_Q, "K_prefill_pair", "Q");
  check_contiguous(a_V, "V");
  check_same_device(a_V, a_Q, "V", "Q");
  check_same_device(a_V_scale, a_Q, "V_scale", "Q");
  check_contiguous(a_V_prefill_pair, "V_prefill_pair");
  check_same_device(a_V_prefill_pair, a_Q, "V_prefill_pair", "Q");
  check_same_device(a_KV, a_Q, "KV", "Q");
  check_contiguous(a_O, "O");
  check_same_device(a_O, a_Q, "O", "Q");
  check_contiguous(a_partial_O, "partial_O");
  check_same_device(a_partial_O, a_Q, "partial_O", "Q");
  check_contiguous(a_partial_M, "partial_M");
  check_same_device(a_partial_M, a_Q, "partial_M", "Q");
  check_contiguous(a_partial_D, "partial_D");
  check_same_device(a_partial_D, a_Q, "partial_D", "Q");
  check_contiguous(a_split_completion, "split_completion");
  check_same_device(a_split_completion, a_Q, "split_completion", "Q");
  check_contiguous(a_msa_lse, "msa_lse");
  check_same_device(a_msa_lse, a_Q, "msa_lse", "Q");
  check_contiguous(a_kv_indices, "kv_indices");
  check_same_device(a_kv_indices, a_Q, "kv_indices", "Q");
  check_contiguous(a_qo_indptr, "qo_indptr");
  check_same_device(a_qo_indptr, a_Q, "qo_indptr", "Q");
  check_contiguous(a_kv_indptr, "kv_indptr");
  check_same_device(a_kv_indptr, a_Q, "kv_indptr", "Q");
  check_contiguous(a_kv_len_arr, "kv_len_arr");
  check_same_device(a_kv_len_arr, a_Q, "kv_len_arr", "Q");
  check_contiguous(a_task_kind, "task_kind");
  check_same_device(a_task_kind, a_Q, "task_kind", "Q");
  check_contiguous(a_task_request, "task_request");
  check_same_device(a_task_request, a_Q, "task_request", "Q");
  check_contiguous(a_task_kv_head, "task_kv_head");
  check_same_device(a_task_kv_head, a_Q, "task_kv_head", "Q");
  check_contiguous(a_task_q_tile, "task_q_tile");
  check_same_device(a_task_q_tile, a_Q, "task_q_tile", "Q");
  check_contiguous(a_task_split, "task_split");
  check_same_device(a_task_split, a_Q, "task_split", "Q");
  check_contiguous(a_task_kv_tile_begin, "task_kv_tile_begin");
  check_same_device(a_task_kv_tile_begin, a_Q, "task_kv_tile_begin", "Q");
  check_contiguous(a_task_kv_tile_end, "task_kv_tile_end");
  check_same_device(a_task_kv_tile_end, a_Q, "task_kv_tile_end", "Q");
  check_contiguous(a_task_qo_begin, "task_qo_begin");
  check_same_device(a_task_qo_begin, a_Q, "task_qo_begin", "Q");
  check_contiguous(a_task_qo_end, "task_qo_end");
  check_same_device(a_task_qo_end, a_Q, "task_qo_end", "Q");
  check_contiguous(a_task_page_begin, "task_page_begin");
  check_same_device(a_task_page_begin, a_Q, "task_page_begin", "Q");
  check_contiguous(a_task_page_end, "task_page_end");
  check_same_device(a_task_page_end, a_Q, "task_page_end", "Q");
  check_contiguous(a_status, "status");
  check_same_device(a_status, a_Q, "status", "Q");
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
  CUtensorMap v_Q = EncodeTma_Q(a_Q);
  CUtensorMap v_Q_prefill = EncodeTma_Q_prefill(a_Q_prefill);
  void* v_Q_prefill_raw = DataPtr(a_Q_prefill_raw);
  CUtensorMap v_K = EncodeTma_K(a_K);
  CUtensorMap v_K_scale = EncodeTma_K_scale(a_K_scale);
  CUtensorMap v_K_prefill_pair = EncodeTma_K_prefill_pair(a_K_prefill_pair);
  CUtensorMap v_V = EncodeTma_V(a_V);
  CUtensorMap v_V_scale = EncodeTma_V_scale(a_V_scale);
  CUtensorMap v_V_prefill_pair = EncodeTma_V_prefill_pair(a_V_prefill_pair);
  CUtensorMap v_KV = EncodeTma_KV(a_KV);
  void* v_O = DataPtr(a_O);
  void* v_partial_O = DataPtr(a_partial_O);
  void* v_partial_M = DataPtr(a_partial_M);
  void* v_partial_D = DataPtr(a_partial_D);
  void* v_split_completion = DataPtr(a_split_completion);
  void* v_msa_lse = DataPtr(a_msa_lse);
  void* v_kv_indices = DataPtr(a_kv_indices);
  void* v_qo_indptr = DataPtr(a_qo_indptr);
  void* v_kv_indptr = DataPtr(a_kv_indptr);
  void* v_kv_len_arr = DataPtr(a_kv_len_arr);
  void* v_task_kind = DataPtr(a_task_kind);
  void* v_task_request = DataPtr(a_task_request);
  void* v_task_kv_head = DataPtr(a_task_kv_head);
  void* v_task_q_tile = DataPtr(a_task_q_tile);
  void* v_task_split = DataPtr(a_task_split);
  void* v_task_kv_tile_begin = DataPtr(a_task_kv_tile_begin);
  void* v_task_kv_tile_end = DataPtr(a_task_kv_tile_end);
  void* v_task_qo_begin = DataPtr(a_task_qo_begin);
  void* v_task_qo_end = DataPtr(a_task_qo_end);
  void* v_task_page_begin = DataPtr(a_task_page_begin);
  void* v_task_page_end = DataPtr(a_task_page_end);
  void* v_status = DataPtr(a_status);
  int32_t v_num_requests = static_cast<int32_t>(a_num_requests);
  int32_t v_num_q_heads = static_cast<int32_t>(a_num_q_heads);
  int32_t v_num_kv_heads = static_cast<int32_t>(a_num_kv_heads);
  int32_t v_max_kv_tiles = static_cast<int32_t>(a_max_kv_tiles);
  int32_t v_max_splits = static_cast<int32_t>(a_max_splits);
  int32_t v_max_task_claims = static_cast<int32_t>(a_max_task_claims);
  float v_softmax_scale_log2 = static_cast<float>(a_softmax_scale_log2);
  float v_k_global_scale = static_cast<float>(a_k_global_scale);
  float v_v_global_scale = static_cast<float>(a_v_global_scale);
  int32_t v_attention_mode = static_cast<int32_t>(a_attention_mode);
  int32_t v_is_causal = static_cast<int32_t>(a_is_causal);
  int32_t v_derive_q_offset = static_cast<int32_t>(a_derive_q_offset);
  int32_t v_record_tasks = static_cast<int32_t>(a_record_tasks);
  int32_t v_msa_max_pages = static_cast<int32_t>(a_msa_max_pages);
  int32_t v_msa_split_policy = static_cast<int32_t>(a_msa_split_policy);
  void* kernel_args[] = {&v_Q, &v_Q_prefill, &v_Q_prefill_raw, &v_K, &v_K_scale, &v_K_prefill_pair, &v_V, &v_V_scale, &v_V_prefill_pair, &v_KV, &v_O, &v_partial_O, &v_partial_M, &v_partial_D, &v_split_completion, &v_msa_lse, &v_kv_indices, &v_qo_indptr, &v_kv_indptr, &v_kv_len_arr, &v_task_kind, &v_task_request, &v_task_kv_head, &v_task_q_tile, &v_task_split, &v_task_kv_tile_begin, &v_task_kv_tile_end, &v_task_qo_begin, &v_task_qo_end, &v_task_page_begin, &v_task_page_end, &v_status, &v_num_requests, &v_num_q_heads, &v_num_kv_heads, &v_max_kv_tiles, &v_max_splits, &v_max_task_claims, &v_softmax_scale_log2, &v_k_global_scale, &v_v_global_scale, &v_attention_mode, &v_is_causal, &v_derive_q_offset, &v_record_tasks, &v_msa_max_pages, &v_msa_split_policy};
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
