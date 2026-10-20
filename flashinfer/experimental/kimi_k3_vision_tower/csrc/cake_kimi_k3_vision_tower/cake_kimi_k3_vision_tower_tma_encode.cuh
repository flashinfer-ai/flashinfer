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

// Parameterised TMA-descriptor encoder of the shared launcher
// (cake_kimi_k3_vision_tower_launcher.cu): exporter output, hand-folded here
// pending exporter support.  The export writes one EncodeTma_<buffer> function
// per TMA buffer of every binding; the three function texts below (one per
// descriptor rank) are those functions with the buffer name, the box extents
// and the axis order as parameters.  The launcher includes this file once per
// descriptor, inside its host-shim namespace, after defining
//
//   CAKE_TMA_NAME          the buffer name (function EncodeTma_<name>, messages)
//   CAKE_TMA_RANK          2, 3 or 4
//   CAKE_TMA_BOX_1         rank 3 / 4: the second box extent (128, 64 or 32)
//   CAKE_TMA_CHECK_BOX_1   rank 3: 1 when the second extent is checked against
//                          the resolved global extent (the B operand), else 0
//   CAKE_TMA_BOX_3         rank 4: the fourth box extent (2 or 1)
//   CAKE_TMA_AXES          rank 4: 0 for the (64, d3, d2, d1 / 64) axis order
//                          of Q / K / O_tma, 1 for the (64, d3, d1 / 64, d2)
//                          order of V
//   CAKE_TMA_PAIR_FORM     rank 2: 1 for the A map of the position-table GEMMs
//                          ((d1, ceil(d2 / 2)) rows), 2 for its A2 map
//                          ((2 * d1, d2 / 2))
//
// and undefines them again at the end.  The helper macros are defined once;
// a check condition never names a parameter (the check macro stringizes its
// condition), so every message stays the exported text.
#ifndef CAKE_KIMI_K3_VISION_TOWER_TMA_ENCODE_CUH_
#define CAKE_KIMI_K3_VISION_TOWER_TMA_ENCODE_CUH_
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#define CAKE_UNSIGNED_(x) x##u
#define CAKE_UNSIGNED(x) CAKE_UNSIGNED_(x)
#define CAKE_CONCAT_(a, b) a##b
#define CAKE_CONCAT(a, b) CAKE_CONCAT_(a, b)
#endif  // CAKE_KIMI_K3_VISION_TOWER_TMA_ENCODE_CUH_

#ifdef CAKE_TMA_NAME
#if CAKE_TMA_RANK == 2

// 2-D TMA descriptor of the A / A2 maps of the position-table GEMMs (rows in pairs).
inline CUtensorMap CAKE_CONCAT(EncodeTma_, CAKE_TMA_NAME)(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' physical strides must be positive";
#if CAKE_TMA_PAIR_FORM == 1
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(((d2 + 2 - 1) / 2))};
#elif CAKE_TMA_PAIR_FORM == 2
  uint64_t global_dim[2] = {(uint64_t)((2 * d1)), (uint64_t)((d2 / 2))};
#else
#error "CAKE_TMA_PAIR_FORM must be 1 (A) or 2 (A2)"
#endif
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0], ValueError)
      << "TMA box (64, 128) exceeds resolved global dims for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "'";
  int64_t carrier_stride_0 = (2 * s2);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {64u, 128u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, '" CAKE_STRINGIZE(CAKE_TMA_NAME) "') failed: CUresult=" << (int)r;
  return tm;
}

#elif CAKE_TMA_RANK == 3

// 3-D TMA descriptor of a (rows, 64-column panels) BF16 operand.
inline CUtensorMap CAKE_CONCAT(EncodeTma_, CAKE_TMA_NAME)(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, CAKE_STRINGIZE(CAKE_TMA_NAME));
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved a non-positive global dim";
#if CAKE_TMA_CHECK_BOX_1 && CAKE_TMA_BOX_1 == 128
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
#elif CAKE_TMA_CHECK_BOX_1 && CAKE_TMA_BOX_1 == 64
  TVM_FFI_CHECK(64u <= global_dim[0] && 64u <= global_dim[1] && 1u <= global_dim[2], ValueError)
#elif CAKE_TMA_CHECK_BOX_1 && CAKE_TMA_BOX_1 == 32
  TVM_FFI_CHECK(64u <= global_dim[0] && 32u <= global_dim[1] && 1u <= global_dim[2], ValueError)
#elif CAKE_TMA_CHECK_BOX_1
#error "CAKE_TMA_BOX_1 must be 128, 64 or 32 when its extent is checked"
#else
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2], ValueError)
#endif
      << "TMA box (64, " CAKE_STRINGIZE(CAKE_TMA_BOX_1) ", 1) exceeds resolved global dims for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, CAKE_UNSIGNED(CAKE_TMA_BOX_1), 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, '" CAKE_STRINGIZE(CAKE_TMA_NAME) "') failed: CUresult=" << (int)r;
  return tm;
}

#elif CAKE_TMA_RANK == 4

// 4-D TMA descriptor of a (rows, heads, 64-column panels) BF16 operand.
inline CUtensorMap CAKE_CONCAT(EncodeTma_, CAKE_TMA_NAME)(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' trailing dims must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' extent " << d1
      << " must divide exactly by " << 64;
#if CAKE_TMA_AXES == 0
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d3), (uint64_t)(d2), (uint64_t)((d1 / 64))};
#elif CAKE_TMA_AXES == 1
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d3), (uint64_t)((d1 / 64)), (uint64_t)(d2)};
#else
#error "CAKE_TMA_AXES must be 0 (Q / K / O_tma) or 1 (V)"
#endif
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved a non-positive global dim";
#if CAKE_TMA_BOX_3 == 2
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
#elif CAKE_TMA_BOX_3 == 1
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
#else
#error "CAKE_TMA_BOX_3 must be 2 or 1"
#endif
      << "TMA box (64, " CAKE_STRINGIZE(CAKE_TMA_BOX_1) ", 1, " CAKE_STRINGIZE(CAKE_TMA_BOX_3) ") exceeds resolved global dims for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 1 to a non-whole-byte offset";
#if CAKE_TMA_AXES == 0
  int64_t carrier_stride_1 = d1;
#else
  int64_t carrier_stride_1 = 64;
#endif
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 2 to a non-whole-byte offset";
#if CAKE_TMA_AXES == 0
  int64_t carrier_stride_2 = 64;
#else
  int64_t carrier_stride_2 = d1;
#endif
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for '" CAKE_STRINGIZE(CAKE_TMA_NAME) "' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, CAKE_UNSIGNED(CAKE_TMA_BOX_1), 1u, CAKE_UNSIGNED(CAKE_TMA_BOX_3)};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, '" CAKE_STRINGIZE(CAKE_TMA_NAME) "') failed: CUresult=" << (int)r;
  return tm;
}

#else
#error "CAKE_TMA_RANK must be 2, 3 or 4"
#endif
#undef CAKE_TMA_NAME
#undef CAKE_TMA_RANK
#undef CAKE_TMA_BOX_1
#undef CAKE_TMA_CHECK_BOX_1
#undef CAKE_TMA_BOX_3
#undef CAKE_TMA_AXES
#undef CAKE_TMA_PAIR_FORM
#endif  // CAKE_TMA_NAME
