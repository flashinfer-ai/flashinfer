// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION.
// Based on the NVIDIA/vLLM FP4 quantization implementation.

// Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad,
// csrc/nvfp4/cuda/nvfp4_residue_pack16.cu. See resinfer_kext/README.md.
#pragma once
#include "kext_utils.cuh"
namespace flashinfer::resinfer {

namespace pack16 {

__device__ __forceinline__ float pick_residue_value(float2 const (&residue)[4], int idx) {
  // Missing mask bits contribute zero; valid masks preserve donor arithmetic.
  if (idx < 0) return 0.0f;
  int pairIdx = idx >> 1;
  return (idx & 1) == 0 ? residue[pairIdx].x : residue[pairIdx].y;
}

template <int R2, bool HAS_TAIL>
__device__ __forceinline__ uint32_t* residue_data_offset(int rowIdx, int colIdx, int numMainCols,
                                                         int numTotalCols, uint32_t* out) {
  constexpr int CHUNKS_PER_WARP = 32 * R2 / 8;
  int64_t u32PerRow = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
  int64_t baseU32 = numMainCols / CVT_FP4_ELTS_PER_UINT32;
  int64_t rowBase = static_cast<int64_t>(rowIdx) * u32PerRow + baseU32;
  if constexpr (R2 == 8) {
    return out + rowBase + colIdx;
  } else {
    int warpId = colIdx / 32;
    int laneId = colIdx % 32;
    if (laneId >= CHUNKS_PER_WARP) return nullptr;
    int chunkIdx = warpId * CHUNKS_PER_WARP + laneId;
    if constexpr (HAS_TAIL) {
      if (chunkIdx >= (numTotalCols - numMainCols) / CVT_FP4_ELTS_PER_UINT32) {
        return nullptr;
      }
    }
    return out + rowBase + chunkIdx;
  }
}

template <class SFType, int R2>
__device__ __forceinline__ SFType* residue_sf_offset(int rowIdx, int colIdx, int numTotalCols,
                                                     int numMainCols, SFType* base) {
  constexpr int THREADS_PER_SF_GROUP = 16 / R2;
  int warpId = colIdx / 32;
  int laneId = colIdx % 32;
  if (laneId % THREADS_PER_SF_GROUP != 0) return nullptr;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int residueElementIdx = numMainCols + warpId * 32 * R2 + sfGroupId * 16;
  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;
  int64_t SFOffset = compute_tiled_sf_offset<SFType>(rowIdx, kIdx, numTotalCols);
  return base + SFOffset;
}

template <class Type, bool UE8M0_SF>
__device__ __forceinline__ u32x2 main_quant16(PackedVec<Type>& loVec, PackedVec<Type>& hiVec,
                                              float SFScaleVal, uint8_t* SFout,
                                              float2 (&loResidue)[4], float2 (&hiResidue)[4]) {
  auto localMax = __habs2(loVec.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(loVec.elts[i]));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(hiVec.elts[i]));
  }
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 loOriginal[4];
  float2 hiOriginal[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    loOriginal[i] = packed_vec_to_float2(loVec, i);
    hiOriginal[i] = packed_vec_to_float2(hiVec, i);
    loResidue[i].x = loOriginal[i].x * outputScale;
    loResidue[i].y = loOriginal[i].y * outputScale;
    hiResidue[i].x = hiOriginal[i].x * outputScale;
    hiResidue[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 packed = fp32x16_to_e2m1<true>(loResidue, hiResidue);

  float inverseOutputScale = outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    loResidue[i].x = loOriginal[i].x - loResidue[i].x * inverseOutputScale;
    loResidue[i].y = loOriginal[i].y - loResidue[i].y * inverseOutputScale;
    hiResidue[i].x = hiOriginal[i].x - hiResidue[i].x * inverseOutputScale;
    hiResidue[i].y = hiOriginal[i].y - hiResidue[i].y * inverseOutputScale;
  }
  return packed;
}

}  // namespace pack16

template <class Type, bool UE8M0_SF, bool HAS_TAIL>
__global__ void __launch_bounds__(512, 2)
    cvt_fp16_to_fp4_residue_pack16_r0125_no_swap(int32_t numRows, int32_t numTotalCols,
                                                 int32_t numMainCols, Type const* in,
                                                 float const* SFScale, uint32_t* out,
                                                 uint32_t* SFout, uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

  if constexpr (!HAS_TAIL) {
    if (colIdx >= numMainCols / 16) return;
  }
  bool validCol = !HAS_TAIL || colIdx < numMainCols / 16;
  uint8_t mask0 = validCol ? channelIndices[colIdx * 2] : 0x01;
  uint8_t mask1 = validCol ? channelIndices[colIdx * 2 + 1] : 0x01;
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow && validCol) {
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16, loVec,
                        hiVec);
    }

    auto sfOut = validCol ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(rowIdx, colIdx,
                                                                            numTotalCols, SFout)
                          : nullptr;
    auto residueSFOut = validCol ? pack16::residue_sf_offset<uint8_t, 2>(
                                       rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart)
                                 : nullptr;

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed =
        pack16::main_quant16<Type, UE8M0_SF>(loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    int i1 = __ffs(mask1) - 1;
    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(hiResidue, i1);

    float localMaxResidue = fmaxf(fabsf(rv0), fabsf(rv1));
    float sfGroupMaxResidue = warp_group_max<8>(localMaxResidue, laneId);
    float residueSFValue = compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);
    rv0 *= residueOutputScale;
    rv1 *= residueOutputScale;

    float2 residueChunk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int sourceThread = (laneId < 8 ? laneId : 0) * 4 + i;
      residueChunk[i] = make_float2(__shfl_sync(uint32_t(-1), rv0, sourceThread),
                                    __shfl_sync(uint32_t(-1), rv1, sourceThread));
    }

    if (validRow && validCol) {
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut =
          pack16::residue_data_offset<2, HAS_TAIL>(rowIdx, colIdx, numMainCols, numTotalCols, out);
      if (residueDataOut) {
        *residueDataOut = fp32_vec_to_e2m1(residueChunk);
      }
    }
  }
}

template <class Type, bool UE8M0_SF, bool HAS_TAIL>
__global__ void __launch_bounds__(512, 2)
    cvt_fp16_to_fp4_residue_pack16_r025_no_swap(int32_t numRows, int32_t numTotalCols,
                                                int32_t numMainCols, Type const* in,
                                                float const* SFScale, uint32_t* out,
                                                uint32_t* SFout, uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

  if constexpr (!HAS_TAIL) {
    if (colIdx >= numMainCols / 16) return;
  }
  bool validCol = !HAS_TAIL || colIdx < numMainCols / 16;
  uint8_t mask0 = validCol ? channelIndices[colIdx * 2] : 0x03;
  uint8_t mask1 = validCol ? channelIndices[colIdx * 2 + 1] : 0x03;
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow && validCol) {
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16, loVec,
                        hiVec);
    }

    auto sfOut = validCol ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(rowIdx, colIdx,
                                                                            numTotalCols, SFout)
                          : nullptr;
    auto residueSFOut = validCol ? pack16::residue_sf_offset<uint8_t, 4>(
                                       rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart)
                                 : nullptr;

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed =
        pack16::main_quant16<Type, UE8M0_SF>(loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    int i1 = __ffs(mask0 & (mask0 - 1)) - 1;
    int i2 = __ffs(mask1) - 1;
    int i3 = __ffs(mask1 & (mask1 - 1)) - 1;
    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(loResidue, i1);
    float rv2 = pack16::pick_residue_value(hiResidue, i2);
    float rv3 = pack16::pick_residue_value(hiResidue, i3);

    float localMaxResidue = fmaxf(fmaxf(fabsf(rv0), fabsf(rv1)), fmaxf(fabsf(rv2), fabsf(rv3)));
    float sfGroupMaxResidue = warp_group_max<4>(localMaxResidue, laneId);
    float residueSFValue = compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);
    rv0 *= residueOutputScale;
    rv1 *= residueOutputScale;
    rv2 *= residueOutputScale;
    rv3 *= residueOutputScale;

    int writerGroup = laneId < 16 ? laneId : 0;
    float2 residueChunk[4];
    residueChunk[0] = make_float2(__shfl_sync(uint32_t(-1), rv0, writerGroup * 2),
                                  __shfl_sync(uint32_t(-1), rv1, writerGroup * 2));
    residueChunk[1] = make_float2(__shfl_sync(uint32_t(-1), rv2, writerGroup * 2),
                                  __shfl_sync(uint32_t(-1), rv3, writerGroup * 2));
    residueChunk[2] = make_float2(__shfl_sync(uint32_t(-1), rv0, writerGroup * 2 + 1),
                                  __shfl_sync(uint32_t(-1), rv1, writerGroup * 2 + 1));
    residueChunk[3] = make_float2(__shfl_sync(uint32_t(-1), rv2, writerGroup * 2 + 1),
                                  __shfl_sync(uint32_t(-1), rv3, writerGroup * 2 + 1));

    if (validRow && validCol) {
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut =
          pack16::residue_data_offset<4, HAS_TAIL>(rowIdx, colIdx, numMainCols, numTotalCols, out);
      if (residueDataOut) {
        *residueDataOut = fp32_vec_to_e2m1(residueChunk);
      }
    }
  }
}

template <class Type, bool UE8M0_SF, bool HAS_TAIL>
__global__ void __launch_bounds__(512, 2)
    cvt_fp16_to_fp4_residue_pack16_r050_no_swap(int32_t numRows, int32_t numTotalCols,
                                                int32_t numMainCols, Type const* in,
                                                float const* SFScale, uint32_t* out,
                                                uint32_t* SFout, uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

  if constexpr (!HAS_TAIL) {
    if (colIdx >= numMainCols / 16) return;
  }
  bool validCol = !HAS_TAIL || colIdx < numMainCols / 16;
  uint8_t mask0 = validCol ? channelIndices[colIdx * 2] : 0x0f;
  uint8_t mask1 = validCol ? channelIndices[colIdx * 2 + 1] : 0x0f;
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow && validCol) {
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16, loVec,
                        hiVec);
    }

    auto sfOut = validCol ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(rowIdx, colIdx,
                                                                            numTotalCols, SFout)
                          : nullptr;
    auto residueSFOut = validCol ? pack16::residue_sf_offset<uint8_t, 8>(
                                       rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart)
                                 : nullptr;

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed =
        pack16::main_quant16<Type, UE8M0_SF>(loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    uint8_t rem = mask0 & (mask0 - 1);
    int i1 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int i2 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int i3 = __ffs(rem) - 1;
    int j0 = __ffs(mask1) - 1;
    rem = mask1 & (mask1 - 1);
    int j1 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int j2 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int j3 = __ffs(rem) - 1;

    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(loResidue, i1);
    float rv2 = pack16::pick_residue_value(loResidue, i2);
    float rv3 = pack16::pick_residue_value(loResidue, i3);
    float rv4 = pack16::pick_residue_value(hiResidue, j0);
    float rv5 = pack16::pick_residue_value(hiResidue, j1);
    float rv6 = pack16::pick_residue_value(hiResidue, j2);
    float rv7 = pack16::pick_residue_value(hiResidue, j3);

    float localMaxResidue =
        fmaxf(fmaxf(fmaxf(fabsf(rv0), fabsf(rv1)), fmaxf(fabsf(rv2), fabsf(rv3))),
              fmaxf(fmaxf(fabsf(rv4), fabsf(rv5)), fmaxf(fabsf(rv6), fabsf(rv7))));
    float sfGroupMaxResidue = warp_group_max<2>(localMaxResidue, laneId);
    float residueSFValue = compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

    float2 residueChunk[4];
    residueChunk[0] = make_float2(rv0 * residueOutputScale, rv1 * residueOutputScale);
    residueChunk[1] = make_float2(rv2 * residueOutputScale, rv3 * residueOutputScale);
    residueChunk[2] = make_float2(rv4 * residueOutputScale, rv5 * residueOutputScale);
    residueChunk[3] = make_float2(rv6 * residueOutputScale, rv7 * residueOutputScale);

    if (validRow && validCol) {
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut =
          pack16::residue_data_offset<8, HAS_TAIL>(rowIdx, colIdx, numMainCols, numTotalCols, out);
      *residueDataOut = fp32_vec_to_e2m1(residueChunk);
    }
  }
}

}  // namespace flashinfer::resinfer
