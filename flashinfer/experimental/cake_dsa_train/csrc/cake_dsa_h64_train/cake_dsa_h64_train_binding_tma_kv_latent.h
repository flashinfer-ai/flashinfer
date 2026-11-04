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

// The tensor-map encoder of descriptor 'kv_latent' shared by the bindings that take it; included once per binding inside its namespace, after the check helpers.

CUtensorMap EncodeTma_kv_latent(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << KERNEL_NAME << ": descriptor 'kv_latent' takes a source of at least 2 dims, got " << t.ndim();
  CheckUnitInnerStride(t, "kv_latent");
  int64_t d1 = t.size(-1);
  int64_t outer1 = NumelAbove(t, 1);
  check_dense_leading_dims(t, 1, "kv_latent");
  int64_t s2 = t.stride(-2) * 2;
  cuuint64_t global_dim[2] = {Extent(d1, "kv_latent"), Extent(outer1, "kv_latent")};
  cuuint64_t global_strides[1] = {Stride(s2, "kv_latent")};
  cuuint32_t box_dim[2] = {64u, 1u};
  cuuint32_t elem_strides[2] = {1u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1], ValueError)
      << KERNEL_NAME << ": the box of descriptor 'kv_latent' exceeds its source extents";
  TVM_FFI_CHECK(global_strides[0] % 16 == 0, ValueError)
      << KERNEL_NAME << ": the strides of descriptor 'kv_latent' must be multiples of 16 bytes";
  void* address = DataPtr(t);
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << KERNEL_NAME << ": the address of descriptor 'kv_latent' must be 16-byte aligned";
  CUtensorMap map{};
  CheckCu(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, address, global_dim, global_strides,
                                 box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled of kv_latent");
  return map;
}
