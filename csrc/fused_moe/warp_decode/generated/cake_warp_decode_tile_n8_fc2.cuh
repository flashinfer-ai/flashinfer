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

// Host side of the tile-N8 FC2 programs (warp_decode::TileN8Fc2): launch arguments, tensor maps, grid and work
// counter. The generated manifest and the clamped E256 manifest launch them.
#pragma once
#include <cuda.h>

#include <algorithm>
#include <cstdint>

namespace flashinfer::warp_decode::generated {
namespace {  // DSO-local, as the generated host implementation

// The work counter pair of a tile-N8 FC2 program: the int32 words kTileN8Fc2CounterOffset and
// kTileN8Fc2CounterOffset + 1 after the FC2 workfeed counter, in the unused tail of that counter's workspace slot.
// The program re-arms the pair at the end of every launch, so it is zeroed once, when the workspace is prepared.
inline constexpr int32_t kTileN8Fc2CounterOffset = 4;
// The clamped E256 rows hold the pair in a region of this size appended to their workspace.
inline constexpr int64_t kTileN8Fc2ClampedCounterBytes = 256;

struct TileN8Fc2Args {
  CUtensorMap A;
  CUtensorMap B;
  CUtensorMap SFA;
  CUtensorMap SFB;
  CUtensorMap C;
  float* scale_c;
  int* tile_expert;
  int* tile_mn_limit;
  int* num_tiles;
  int* work_counter;
  int grid_m;
  int max_tiles;
  int K_tiles;
};

inline bool EncodeTileN8Fc2Map(CUtensorMap* out, CUtensorMapDataType type, uint32_t rank, const void* data,
                               const uint64_t* dims, const uint64_t* strides_bytes, const uint32_t* box,
                               CUtensorMapSwizzle swizzle) {
  const uint32_t element_stride[5] = {1u, 1u, 1u, 1u, 1u};
  const CUresult result = cuTensorMapEncodeTiled(
      out, type, rank, const_cast<void*>(data), dims, strides_bytes, box, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS;
}

// The tensor maps and sizes of one tile-N8 FC2 launch over `extent` route tiles of E experts, hidden size H and
// intermediate size I:
//   A    FC2 weights [E, H, I/2] (E2M1 pairs), box 256 x 128.
//   B    FC1 output [extent, 8, I/2] (E2M1 pairs), box 256 x 8; the gather program loads one row per box (256 x 1).
//   SFA  FC2 weight scales [E * H/128, I/64, 2, 256], box 256 x 2 x 4.
//   SFB  FC1 output scales [extent, I/64, 32], box 32 x 4.
//   C    partials [extent * 8, H] BF16. The program stores a transposed 8-column tile through dims (H, 8, 2^31,
//        2^31) with byte strides (2H, 2^36 - 2H, 2H), box 64 x 8; the gather program stores single rows through
//        the plain [extent * 8, H] map, box 64 x 1.
inline bool EncodeTileN8Fc2(TileN8Fc2Args* args, bool gather, const void* weights, const void* weight_scales,
                            const void* activations, const void* activation_scales, const void* partials,
                            int64_t experts, int64_t hidden, int64_t intermediate, int64_t extent) {
  if (args == nullptr || weights == nullptr || weight_scales == nullptr || activations == nullptr ||
      activation_scales == nullptr || partials == nullptr || experts <= 0 || hidden % 128 != 0 ||
      intermediate % 64 != 0 || extent <= 0)
    return false;
  const uint64_t e = static_cast<uint64_t>(experts);
  const uint64_t h = static_cast<uint64_t>(hidden);
  const uint64_t i = static_cast<uint64_t>(intermediate);
  const uint64_t x = static_cast<uint64_t>(extent);
  {
    const uint64_t dims[3] = {i, h, e};
    const uint64_t strides[2] = {i / 2, h * (i / 2)};
    const uint32_t box[3] = {256u, 128u, 1u};
    if (!EncodeTileN8Fc2Map(&args->A, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, weights, dims, strides, box,
                            CU_TENSOR_MAP_SWIZZLE_128B))
      return false;
  }
  {
    const uint64_t dims[3] = {i, 8u, x};
    const uint64_t strides[2] = {i / 2, 8u * (i / 2)};
    const uint32_t box[3] = {256u, gather ? 1u : 8u, 1u};
    if (!EncodeTileN8Fc2Map(&args->B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, activations, dims, strides, box,
                            CU_TENSOR_MAP_SWIZZLE_128B))
      return false;
  }
  {
    const uint64_t dims[4] = {256u, 2u, i / 64, e * (h / 128)};
    const uint64_t strides[3] = {256u, 512u, 512u * (i / 64)};
    const uint32_t box[4] = {256u, 2u, 4u, 1u};
    if (!EncodeTileN8Fc2Map(&args->SFA, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, weight_scales, dims, strides, box,
                            CU_TENSOR_MAP_SWIZZLE_NONE))
      return false;
  }
  {
    const uint64_t dims[3] = {32u, i / 64, x};
    const uint64_t strides[2] = {32u, 32u * (i / 64)};
    const uint32_t box[3] = {32u, 4u, 1u};
    if (!EncodeTileN8Fc2Map(&args->SFB, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, activation_scales, dims, strides, box,
                            CU_TENSOR_MAP_SWIZZLE_NONE))
      return false;
  }
  const uint64_t row = 2u * h;
  if (gather) {
    const uint64_t dims[2] = {h, x * 8u};
    const uint64_t strides[1] = {row};
    const uint32_t box[2] = {64u, 1u};
    if (!EncodeTileN8Fc2Map(&args->C, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, partials, dims, strides, box,
                            CU_TENSOR_MAP_SWIZZLE_128B))
      return false;
  } else {
    const uint64_t dims[4] = {h, 8u, 2147483648ull, 2147483648ull};
    const uint64_t strides[3] = {row, 68719476736ull - row, row};
    const uint32_t box[4] = {64u, 8u, 1u, 1u};
    if (!EncodeTileN8Fc2Map(&args->C, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, partials, dims, strides, box,
                            CU_TENSOR_MAP_SWIZZLE_128B))
      return false;
  }
  args->grid_m = static_cast<int>(hidden / 128);
  args->max_tiles = static_cast<int>(extent);
  args->K_tiles = static_cast<int>((intermediate + 255) / 256);
  return true;
}

// One CTA per SM, at most one per work item.
inline uint32_t TileN8Fc2Ctas(int64_t grid_m, int64_t extent, int32_t sm_count) {
  return static_cast<uint32_t>(
      std::max<int64_t>(1, std::min<int64_t>(sm_count > 0 ? sm_count : 1, grid_m * extent)));
}

}  // namespace
}  // namespace flashinfer::warp_decode::generated
