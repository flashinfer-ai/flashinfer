/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#pragma once

#include <flashinfer/conv3d/patchshift/common.cuh>
#include <flashinfer/conv3d/patchshift/weight_layout.cuh>

// Host-side TMA descriptor construction and deterministic packed-weight
// indexing. Driver errors are returned to the framework binding; library code
// must never terminate the hosting process.

namespace flashinfer::conv3d::patchshift::host {

using patchshift::Element;
using patchshift::TensorMap;

inline CUresult MakeInputMap(TensorMap* map, Element* input, int n, int d, int h, int w, int c_size,
                             int input_q, int input_p) {
  cuuint64_t dims[5] = {cuuint64_t(c_size), cuuint64_t(w), cuuint64_t(h), cuuint64_t(d),
                        cuuint64_t(n)};
  cuuint64_t strides[4] = {
      cuuint64_t(c_size) * sizeof(Element), cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element),
      cuuint64_t(h) * cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element),
      cuuint64_t(d) * cuuint64_t(h) * cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element)};
  cuuint32_t box[5] = {16u, uint32_t(input_q), uint32_t(input_p), 1u, 1u};
  cuuint32_t elem_stride[5] = {1u, 1u, 1u, 1u, 1u};
  return cuTensorMapEncodeTiled(map, patchshift::kTensorMapDataType, 5, input, dims, strides, box,
                                elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult MakeInputC32Map(TensorMap* map, Element* input, int n, int d, int h, int w,
                                int c_size, int input_q, int input_p) {
  cuuint64_t dims[5] = {cuuint64_t(c_size), cuuint64_t(w), cuuint64_t(h), cuuint64_t(d),
                        cuuint64_t(n)};
  cuuint64_t strides[4] = {
      cuuint64_t(c_size) * sizeof(Element), cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element),
      cuuint64_t(h) * cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element),
      cuuint64_t(d) * cuuint64_t(h) * cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element)};
  cuuint32_t box[5] = {32u, uint32_t(input_q), uint32_t(input_p), 1u, 1u};
  cuuint32_t elem_stride[5] = {1u, 1u, 1u, 1u, 1u};
  return cuTensorMapEncodeTiled(map, patchshift::kTensorMapDataType, 5, input, dims, strides, box,
                                elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult MakeInputC64Map(TensorMap* map, Element* input, int n, int d, int h, int w,
                                int c_size, int input_q, int input_p) {
  cuuint64_t dims[5] = {cuuint64_t(c_size), cuuint64_t(w), cuuint64_t(h), cuuint64_t(d),
                        cuuint64_t(n)};
  cuuint64_t strides[4] = {
      cuuint64_t(c_size) * sizeof(Element), cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element),
      cuuint64_t(h) * cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element),
      cuuint64_t(d) * cuuint64_t(h) * cuuint64_t(w) * cuuint64_t(c_size) * sizeof(Element)};
  cuuint32_t box[5] = {64u, uint32_t(input_q), uint32_t(input_p), 1u, 1u};
  cuuint32_t elem_stride[5] = {1u, 1u, 1u, 1u, 1u};
  return cuTensorMapEncodeTiled(map, patchshift::kTensorMapDataType, 5, input, dims, strides, box,
                                elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult MakePackedWeightMap(TensorMap* map, Element* weight, int task_count, int tile_m,
                                    int k16_groups) {
  // Generic packed task layout, fastest to slowest:
  //   [K8, tile_m, K16-plane2, (kw3,K16-subgroup), (tile,sg,kh)]
  // One transaction publishes all three kw matrices for one filter row.  Its
  // size follows tile_m: 6/12/24 KiB for M32/M64/M128 when k16_groups=2.
  // The layout matches each retained packed A-row stage byte-for-byte.
  int kw_group_extent = 3 * k16_groups;
  cuuint64_t dims[5] = {8u, uint64_t(tile_m), 2u, uint64_t(kw_group_extent), uint64_t(task_count)};
  cuuint64_t strides[4] = {uint64_t(8 * sizeof(Element)), uint64_t(8 * tile_m * sizeof(Element)),
                           uint64_t(8 * tile_m * 2 * sizeof(Element)),
                           uint64_t(8 * tile_m * 2 * kw_group_extent * sizeof(Element))};
  cuuint32_t box[5] = {8u, uint32_t(tile_m), 2u, uint32_t(kw_group_extent), 1u};
  cuuint32_t elem_stride[5] = {1u, 1u, 1u, 1u, 1u};
  return cuTensorMapEncodeTiled(map, patchshift::kTensorMapDataType, 5, weight, dims, strides, box,
                                elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace flashinfer::conv3d::patchshift::host
