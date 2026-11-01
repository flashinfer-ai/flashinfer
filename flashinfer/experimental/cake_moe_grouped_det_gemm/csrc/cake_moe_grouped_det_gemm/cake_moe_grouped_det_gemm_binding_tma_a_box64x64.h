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

// TMA descriptor encoder for buffer 'A', box {64, 64, 1}: the two weight-gradient main bindings
// share it (exporter output, hand-folded here); included once inside each host-shim namespace.

// 3D TMA descriptor for buffer 'A' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_A(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'A' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'A' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'A' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  check_dense_leading_dims(t, 1, "A");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'A' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'A' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'A' resolved a non-positive global dim";
  uint32_t box_dim[3] = {64u, 64u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "TMA box (" << box_dim[0] << ", " << box_dim[1] << ", " << box_dim[2] << ") exceeds resolved global dims for 'A'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'A' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'A' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'A' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'A' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'A' resolved global stride 2 to a non-whole-byte offset";
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
      << "cuTensorMapEncodeTiled (3D, 'A') failed: CUresult=" << (int)r;
  return tm;
}
