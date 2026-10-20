// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION.
// Based on the NVIDIA/vLLM FP4 quantization implementation.

// Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad,
// csrc/nvfp4/cuda/nvfp4_residue_pack8.cu. See resinfer_kext/README.md.
#pragma once
#include "kext_utils.cuh"
namespace flashinfer::resinfer {

__device__ __forceinline__ float pick_pack8_residue_value(
    float2 const (&residue)[CVT_FP4_ELTS_PER_THREAD / 2], int idx) {
  // Missing mask bits contribute zero; valid masks preserve donor arithmetic.
  if (idx < 0) return 0.0f;
  int pairIdx = idx >> 1;
  return (idx & 1) == 0 ? residue[pairIdx].x : residue[pairIdx].y;
}

__device__ __forceinline__ uint32_t pack_pack8_residue_group_r0125(float residue0, int laneId) {
  int writerGroup = laneId < 4 ? laneId : 0;
  int sourceFirstThread = writerGroup * 8;

  float2 residueChunk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int sourceThread0 = sourceFirstThread + i * 2;
    int sourceThread1 = sourceThread0 + 1;
    float val0 = __shfl_sync(uint32_t(-1), residue0, sourceThread0);
    float val1 = __shfl_sync(uint32_t(-1), residue0, sourceThread1);
    residueChunk[i] = make_float2(val0, val1);
  }
  return fp32_vec_to_e2m1(residueChunk);
}

__device__ __forceinline__ uint32_t pack_pack8_residue_group_r025(float residue0, float residue1,
                                                                  int laneId) {
  int writerGroup = laneId < 8 ? laneId : 0;
  int sourceFirstThread = writerGroup * 4;

  float2 residueChunk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int sourceThread = sourceFirstThread + i;

    float val0 = __shfl_sync(uint32_t(-1), residue0, sourceThread);
    float val1 = __shfl_sync(uint32_t(-1), residue1, sourceThread);
    residueChunk[i] = make_float2(val0, val1);
  }
  return fp32_vec_to_e2m1(residueChunk);
}

__device__ __forceinline__ uint32_t pack_pack8_residue_group_r050(float residue0, float residue1,
                                                                  float residue2, float residue3,
                                                                  int laneId) {
  int writerGroup = laneId < 16 ? laneId : 0;
  int sourceFirstThread = writerGroup * 2;

  float2 residueChunk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float vals[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int residueIdx = i * 2 + j;
      int sourceThread = sourceFirstThread + (residueIdx >> 2);
      int sourceOffset = residueIdx & 3;
      float sourceValue;
      if (sourceOffset == 0) {
        sourceValue = residue0;
      } else if (sourceOffset == 1) {
        sourceValue = residue1;
      } else if (sourceOffset == 2) {
        sourceValue = residue2;
      } else {
        sourceValue = residue3;
      }
      vals[j] = __shfl_sync(uint32_t(-1), sourceValue, sourceThread);
    }
    residueChunk[i] = make_float2(vals[0], vals[1]);
  }
  return fp32_vec_to_e2m1(residueChunk);
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_pack8_r0125(PackedVec<Type>& vec,
                                                                     float SFScaleVal,
                                                                     uint8_t* SFout, uint8_t mask,
                                                                     uint32_t* residueDataOut,
                                                                     uint8_t* residueSFOut) {
  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = fp2Vals[i];
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(fp2Vals);

  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    fp2Vals[i].x *= inverseOutputScale;
    fp2Vals[i].y *= inverseOutputScale;
    fp2Vals[i].x = fp2ValsOriginal[i].x - fp2Vals[i].x;
    fp2Vals[i].y = fp2ValsOriginal[i].y - fp2Vals[i].y;
  }

  int i0 = __ffs(mask) - 1;
  float residue0 = pick_pack8_residue_value(fp2Vals, i0);

  float localMaxResidue = fabsf(residue0);
  int laneId = threadIdx.x % 32;
  float sfGroupMaxResidue = warp_group_max<16>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  uint32_t packedResidue = pack_pack8_residue_group_r0125(residue0 * residueOutputScale, laneId);
  if (residueDataOut) {
    *residueDataOut = packedResidue;
  }

  return e2m1Vec;
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_pack8_r025(PackedVec<Type>& vec,
                                                                    float SFScaleVal,
                                                                    uint8_t* SFout, uint8_t mask,
                                                                    uint32_t* residueDataOut,
                                                                    uint8_t* residueSFOut) {
  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = fp2Vals[i];
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(fp2Vals);

  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    fp2Vals[i].x *= inverseOutputScale;
    fp2Vals[i].y *= inverseOutputScale;
    fp2Vals[i].x = fp2ValsOriginal[i].x - fp2Vals[i].x;
    fp2Vals[i].y = fp2ValsOriginal[i].y - fp2Vals[i].y;
  }

  int i0 = __ffs(mask) - 1;
  int i1 = __ffs(mask & (mask - 1)) - 1;
  float residue0 = pick_pack8_residue_value(fp2Vals, i0);
  float residue1 = pick_pack8_residue_value(fp2Vals, i1);

  float localMaxResidue = fmaxf(fabsf(residue0), fabsf(residue1));
  int laneId = threadIdx.x % 32;
  float sfGroupMaxResidue = warp_group_max<8>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  uint32_t packedResidue = pack_pack8_residue_group_r025(residue0 * residueOutputScale,
                                                         residue1 * residueOutputScale, laneId);
  if (residueDataOut) {
    *residueDataOut = packedResidue;
  }

  return e2m1Vec;
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_pack8_r050(PackedVec<Type>& vec,
                                                                    float SFScaleVal,
                                                                    uint8_t* SFout, uint8_t mask,
                                                                    uint32_t* residueDataOut,
                                                                    uint8_t* residueSFOut) {
  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = fp2Vals[i];
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(fp2Vals);

  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    fp2Vals[i].x *= inverseOutputScale;
    fp2Vals[i].y *= inverseOutputScale;
    fp2Vals[i].x = fp2ValsOriginal[i].x - fp2Vals[i].x;
    fp2Vals[i].y = fp2ValsOriginal[i].y - fp2Vals[i].y;
  }

  int i0 = __ffs(mask) - 1;
  uint8_t remaining = mask & (mask - 1);
  int i1 = __ffs(remaining) - 1;
  remaining = remaining & (remaining - 1);
  int i2 = __ffs(remaining) - 1;
  remaining = remaining & (remaining - 1);
  int i3 = __ffs(remaining) - 1;

  float residue0 = pick_pack8_residue_value(fp2Vals, i0);
  float residue1 = pick_pack8_residue_value(fp2Vals, i1);
  float residue2 = pick_pack8_residue_value(fp2Vals, i2);
  float residue3 = pick_pack8_residue_value(fp2Vals, i3);

  float localMaxResidue =
      fmaxf(fmaxf(fabsf(residue0), fabsf(residue1)), fmaxf(fabsf(residue2), fabsf(residue3)));
  int laneId = threadIdx.x % 32;
  float sfGroupMaxResidue = warp_group_max<4>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  uint32_t packedResidue = pack_pack8_residue_group_r050(
      residue0 * residueOutputScale, residue1 * residueOutputScale, residue2 * residueOutputScale,
      residue3 * residueOutputScale, laneId);
  if (residueDataOut) {
    *residueDataOut = packedResidue;
  }

  return e2m1Vec;
}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 4)
    cvt_fp16_to_fp4_residue_pack8_r0125_no_swap(int32_t numRows, int32_t numTotalCols,
                                                int32_t numMainCols, Type const* in,
                                                float const* SFScale, uint32_t* out,
                                                uint32_t* SFout, uint8_t const* channelIndices) {
  using PackedVecT = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static constexpr int RESIDUE_PER_8_ELTS = 1;

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / CVT_FP4_ELTS_PER_THREAD) {
    return;
  }

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVecT inVec{};
    if (validRow) {
      int64_t inOffset =
          static_cast<int64_t>(rowIdx) * (numMainCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      inVec = reinterpret_cast<PackedVecT const*>(in)[inOffset];
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx, colIdx, numTotalCols, SFout);

    uint32_t* outPos = nullptr;
    uint32_t* residueDataOut = nullptr;
    if (validRow) {
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      outPos = out + outOffset;
      residueDataOut = cvt_residue_data_get_offset<uint32_t, RESIDUE_PER_8_ELTS>(
          rowIdx, colIdx, numMainCols, numTotalCols, out);
    }

    auto residueSFOut = cvt_residue_sf_get_offset<uint8_t, RESIDUE_PER_8_ELTS>(
        rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    uint32_t packed = cvt_warp_fp16_to_fp4_pack8_r0125<Type, UE8M0_SF>(
        inVec, SFScaleVal, sfOut, channelIndices[colIdx], residueDataOut, residueSFOut);
    if (outPos) {
      *outPos = packed;
    }
  }
}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 4)
    cvt_fp16_to_fp4_residue_pack8_r025_no_swap(int32_t numRows, int32_t numTotalCols,
                                               int32_t numMainCols, Type const* in,
                                               float const* SFScale, uint32_t* out, uint32_t* SFout,
                                               uint8_t const* channelIndices) {
  using PackedVecT = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static constexpr int RESIDUE_PER_8_ELTS = 2;

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / CVT_FP4_ELTS_PER_THREAD) {
    return;
  }

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVecT inVec{};
    if (validRow) {
      int64_t inOffset =
          static_cast<int64_t>(rowIdx) * (numMainCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      inVec = reinterpret_cast<PackedVecT const*>(in)[inOffset];
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx, colIdx, numTotalCols, SFout);

    uint32_t* outPos = nullptr;
    uint32_t* residueDataOut = nullptr;
    if (validRow) {
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      outPos = out + outOffset;
      residueDataOut = cvt_residue_data_get_offset<uint32_t, RESIDUE_PER_8_ELTS>(
          rowIdx, colIdx, numMainCols, numTotalCols, out);
    }

    auto residueSFOut = cvt_residue_sf_get_offset<uint8_t, RESIDUE_PER_8_ELTS>(
        rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    uint32_t packed = cvt_warp_fp16_to_fp4_pack8_r025<Type, UE8M0_SF>(
        inVec, SFScaleVal, sfOut, channelIndices[colIdx], residueDataOut, residueSFOut);
    if (outPos) {
      *outPos = packed;
    }
  }
}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 4)
    cvt_fp16_to_fp4_residue_pack8_r050_no_swap(int32_t numRows, int32_t numTotalCols,
                                               int32_t numMainCols, Type const* in,
                                               float const* SFScale, uint32_t* out, uint32_t* SFout,
                                               uint8_t const* channelIndices) {
  using PackedVecT = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static constexpr int RESIDUE_PER_8_ELTS = 4;

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / CVT_FP4_ELTS_PER_THREAD) {
    return;
  }

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVecT inVec{};
    if (validRow) {
      int64_t inOffset =
          static_cast<int64_t>(rowIdx) * (numMainCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      inVec = reinterpret_cast<PackedVecT const*>(in)[inOffset];
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx, colIdx, numTotalCols, SFout);

    uint32_t* outPos = nullptr;
    uint32_t* residueDataOut = nullptr;
    if (validRow) {
      int64_t outOffset =
          static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      outPos = out + outOffset;
      residueDataOut = cvt_residue_data_get_offset<uint32_t, RESIDUE_PER_8_ELTS>(
          rowIdx, colIdx, numMainCols, numTotalCols, out);
    }

    auto residueSFOut = cvt_residue_sf_get_offset<uint8_t, RESIDUE_PER_8_ELTS>(
        rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    uint32_t packed = cvt_warp_fp16_to_fp4_pack8_r050<Type, UE8M0_SF>(
        inVec, SFScaleVal, sfOut, channelIndices[colIdx], residueDataOut, residueSFOut);
    if (outPos) {
      *outPos = packed;
    }
  }
}

}  // namespace flashinfer::resinfer
