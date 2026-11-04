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

// The tensor-map encoder of descriptor 'out' shared by the bindings that take it; included once per binding inside its namespace, after the check helpers.

CUtensorMap EncodeTma_out(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << KERNEL_NAME << ": descriptor 'out' takes a source of at least 3 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "out");
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
  int64_t d3 = t.size(-3);
  int64_t s2 = t.stride(-2) * 2;
  int64_t s3 = t.stride(-3) * 2;
  TVM_FFI_CHECK((CMod(d1, 64) == 0), ValueError)
      << KERNEL_NAME << ": descriptor 'out' requires (CMod(d1, 64) == 0)";
  cuuint64_t global_dim[4] = {Extent(64, "out"), Extent(d2, "out"), Extent(CDiv(d1, 64), "out"), Extent(d3, "out")};
  cuuint64_t global_strides[3] = {Stride(s2, "out"), Stride(128, "out"), Stride(s3, "out")};
  cuuint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1] && box_dim[2] <= global_dim[2] && box_dim[3] <= global_dim[3], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'out' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0 && global_strides[1] % 16 == 0 && global_strides[2] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'out' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'out' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of out");
  return map;
}
