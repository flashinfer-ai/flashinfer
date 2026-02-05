/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <optional>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/quantization.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm {
namespace kernels {

////////////////////////////////////////////////////////////////////////////////////////////////////
// DstVec type traits for quantization

template <typename T, int NUM_ELTS>
struct DstVec {
  static_assert("not implemented.");
};

template <>
struct DstVec<float2, 2> {
  using Type = uint32_t;
};

template <>
struct DstVec<half2, 4> {
  using Type = uint2;
};

#ifdef ENABLE_BF16

template <>
struct DstVec<__nv_bfloat162, 4> {
  using Type = uint2;
};

#endif  // ENABLE_BF16

template <typename T>
struct DstVec<T, 4> {
  static_assert(sizeof(T) == 4, "not implemented.");
  using Type = uint32_t;
};

template <typename T>
struct DstVec<T, 8> {
  static_assert(sizeof(T) == 2, "not implemented.");
  using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function of getting the absMax of all elements in the vector after clamping.
// Pack two elements in order to use possible hmax2 instructions.
template <typename T>
inline __device__ void clampAndAbsMax(T& localMax, uint4& vec, T const clampMin, T const clampMax) {
  static constexpr int NUM_ELTS = sizeof(uint4) / sizeof(T);

#pragma unroll
  for (int i = 0; i < NUM_ELTS; ++i) {
    T& val = reinterpret_cast<T*>(&vec)[i];
    val = cuda_clamp(val, clampMin, clampMax);
    localMax = cuda_max(localMax, cuda_abs(val));
  }
}

// Helper function of quantizing the vector and storing it to global memory.
// Pack two elements in order to use fast convert instructions.
template <typename T, typename QuantT, bool USE_SMEM>
inline __device__ void quantizeAndStore(QuantT* dstPtr, uint4 vec, T const clampMin,
                                        T const clampMax, float const scaleOrigQuant) {
  static constexpr int NUM_ELTS = sizeof(uint4) / sizeof(T);

  using DstVecType = typename DstVec<T, NUM_ELTS>::Type;
  DstVecType dstVec;
#pragma unroll
  for (int i = 0; i < NUM_ELTS; ++i) {
    T val = reinterpret_cast<T*>(&vec)[i];
    // Values loaded from smem has already been clamped.
    if constexpr (!USE_SMEM) {
      val = cuda_clamp(val, clampMin, clampMax);
    }
    float2 val2 = cuda_cast<float2>(val);
    val2.x *= scaleOrigQuant;
    val2.y *= scaleOrigQuant;
    QuantT quantVal = cuda_cast<QuantT>(val2);
    reinterpret_cast<QuantT*>(&dstVec)[i] = quantVal;
  }
  // Store to destination buffer.
  *reinterpret_cast<DstVecType*>(dstPtr) = dstVec;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4/MXFP8 Conversion Functions

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]),
        "f"(array[6]), "f"(array[7]));
  return val;
#else
  // static_assert(false, "not supported.");
  return 0;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x),
        "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  // static_assert(false, "not supported.");
  return 0;
#endif
}

// Convert 8 float2 values into 16 e2m1 values (represented as one uint64_t).
inline __device__ uint64_t fp32_vec_to_e2m1(float2 (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint64_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      ".reg .b8 byte4;\n"
      ".reg .b8 byte5;\n"
      ".reg .b8 byte6;\n"
      ".reg .b8 byte7;\n"
      ".reg .b32 val0;\n"
      ".reg .b32 val1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0,  %2,  %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1,  %4,  %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2,  %6,  %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3,  %8,  %7;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte4, %10,  %9;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte5, %12, %11;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte6, %14, %13;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte7, %16, %15;\n"
      "mov.b32 val0, {byte0, byte1, byte2, byte3};\n"
      "mov.b32 val1, {byte4, byte5, byte6, byte7};\n"
      "mov.b64 %0, {val0, val1};\n"
      "}"
      : "=l"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x),
        "f"(array[2].y), "f"(array[3].x), "f"(array[3].y), "f"(array[4].x), "f"(array[4].y),
        "f"(array[5].x), "f"(array[5].y), "f"(array[6].x), "f"(array[6].y), "f"(array[7].x),
        "f"(array[7].y));
  return val;
#else
  // static_assert(false, "not supported.");
  return 0;
#endif
}

inline __device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

// Convert 4 float2 values into 8 e4m3 values (represented as one uint64_t).
inline __device__ uint64_t fp32_vec_to_e4m3(float2 (&array)[4]) {
  union {
    uint64_t val;
    __nv_fp8x2_e4m3 elts[4];
  } u;

  static_assert(sizeof(u.val) == sizeof(u.elts),
                "Expected to alias uint64_t and __nv_fp8x2_e4m3[4]");

  u.elts[0] = __nv_fp8x2_e4m3(array[0]);
  u.elts[1] = __nv_fp8x2_e4m3(array[1]);
  u.elts[2] = __nv_fp8x2_e4m3(array[2]);
  u.elts[3] = __nv_fp8x2_e4m3(array[3]);
  return u.val;
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float exp2f_rcp(uint8_t exp) {
  constexpr uint32_t FP32_EXPONENT_BIAS = 127;
  return (exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Type converters for packed vectors

template <class Type>
struct TypeConverter {
  using PackedType = void;
};

template <>
struct TypeConverter<half> {
  using PackedType = half2;
};

#ifdef ENABLE_BF16
template <>
struct TypeConverter<__nv_bfloat16> {
  using PackedType = __nv_bfloat162;
};
#endif

// Define a packed data type parameterized by the number of elements.
// For half/bf16: uses half2/bfloat162, so NUM_ELTS elements require NUM_ELTS/2 pairs.
// For FP8: uses __nv_fp8x2_e4m3, so NUM_ELTS elements require NUM_ELTS/2 pairs.
template <class Type, int NUM_ELTS = 8>
struct PackedVec {
  typename TypeConverter<Type>::PackedType elts[NUM_ELTS / 2];
  static_assert(sizeof(elts) == sizeof(Type) * NUM_ELTS,
                "Vector size should match the number of elements per thread.");
};

// Specialization for FP8 with default 16 elements
template <int NUM_ELTS>
struct PackedVec<__nv_fp8_e4m3, NUM_ELTS> {
  __nv_fp8x2_e4m3 elts[NUM_ELTS / 2];
  static_assert(sizeof(elts) == sizeof(__nv_fp8_e4m3) * NUM_ELTS,
                "Vector size should match the number of elements per thread.");
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Quantization helper functions

// Quantizes the provided PackedVec into the uint32_t or uint64_t output
template <class Type, int SF_VEC_SIZE, int CVT_ELTS_PER_THREAD, bool UE8M0_SF>
__device__ std::conditional_t<CVT_ELTS_PER_THREAD == 16, uint64_t, uint32_t> cvt_warp_fp16_to_fp4(
    PackedVec<Type, CVT_ELTS_PER_THREAD>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_ELTS_PER_THREAD == 8 || CVT_ELTS_PER_THREAD == 16,
                "CVT_ELTS_PER_THREAD must be 8 or 16");

  using ReturnType = std::conditional_t<CVT_ELTS_PER_THREAD == 16, uint64_t, uint32_t>;

  // Get absolute maximum values among the local 8 values.
  auto localMax = cuda_abs(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
    localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
  // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
  if constexpr (CVT_NUM_THREADS_PER_SF >= 2) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(cuda_max(localMax.x, localMax.y));

  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  float outputScale;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    // Scale the max value to the range of E2m1.
    vecMax *= reciprocal_approximate_ftz(6.0f);
    tmp.__x = __nv_cvt_float_to_e8m0(vecMax, __NV_SATFINITE, cudaRoundPosInf);

    fp8SFVal = tmp.__x;
    outputScale = vecMax != 0 ? exp2f_rcp(fp8SFVal) : 0.0f;
  } else {
    // Get the SF (max value of the vector / max value of e2m1).
    // maximum value of e2m1 = 6.0.
    // TODO: use half as compute data type.
    auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));

    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
    // Get the output scale.
    // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal)) * reciprocal(SFScaleVal))
    outputScale = vecMax != 0
                      ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
  }

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  ReturnType e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

template <class Type, int SF_VEC_SIZE, int CVT_ELTS_PER_THREAD, bool UE8M0_SF>
__device__ uint64_t cvt_warp_fp8_to_fp4(PackedVec<Type, CVT_ELTS_PER_THREAD>& vec, float SFScaleVal,
                                        uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  // Because the return value is a uint64_t, we need to ensure that the CVT_ELTS_PER_THREAD is 16.
  static_assert(CVT_ELTS_PER_THREAD == 16, "CVT_ELTS_PER_THREAD must be 16");

  float const dequant_to_fp16_scale = 6.f * reciprocal_approximate_ftz(SFScaleVal);

  // Dequant fp8 to fp16
  __half2 vec_half2[8];
#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    float2 tmp = static_cast<float2>(vec.elts[i]);
    tmp.x *= dequant_to_fp16_scale;
    tmp.y *= dequant_to_fp16_scale;
    vec_half2[i] = __float22half2_rn(tmp);
  }

  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec_half2[0]);
  // Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec_half2[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
  if constexpr (CVT_NUM_THREADS_PER_SF == 2) {
    // For block 32, we need to reduce the local max across two threads.
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }

  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  float SFValueNarrow;
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValueNarrow = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValueNarrow = static_cast<float>(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
  float outputScale = SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValueNarrow) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = __half22float2(vec_half2[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint64_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

// Quantizes the provided PackedVec into the uint64_t output
template <class Type, int SF_VEC_SIZE, int CVT_ELTS_PER_THREAD>
__device__ uint64_t cvt_warp_fp16_to_mxfp8(PackedVec<Type, CVT_ELTS_PER_THREAD>& vec,
                                           uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = cuda_abs(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
    localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
  // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
  if constexpr (CVT_NUM_THREADS_PER_SF >= 2) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(cuda_max(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of mxfp8).
  float SFValue = vecMax * reciprocal_approximate_ftz(448.0f);
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  __nv_fp8_e8m0 tmpSFVal;
  tmpSFVal.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmpSFVal);
  fp8SFVal = tmpSFVal.__x;
  // Get the output scale (reciprocal of the SFValue).
  // Note: Check SFValue != 0 (not vecMax != 0) because E8M0 conversion can underflow
  // very small vecMax values to zero. Using vecMax != 0 would cause division by zero
  // (reciprocal of 0 = infinity), leading to NaN when multiplied with denormal inputs.
  float outputScale = SFValue != 0.f ? reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e4m3 values.
  uint64_t e4m3Vec = fp32_vec_to_e4m3(fp2Vals);

  // Write the e4m3 values to global memory.
  return e4m3Vec;
#else
  return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Scale factor offset calculation functions

inline __device__ __host__ int64_t get_sf_out_offset_128x4(std::optional<int> batchIdx, int mIdx,
                                                           int kIdx, std::optional<int> numRows,
                                                           int numColVecs) {
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  // batched tensor
  // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4 * innerKStride;  // 4

  // M tile layout [32, 4] is column-major.
  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * innerMStride;  // 16

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * outerMStride;  // 512

  // SF vector size 16 or 32. We round the "numCols" up to a multiple of 64 or 128.
  // It is the same as rounding the "numColVecs" up to a multiple of 4.
  int32_t numKTiles = (numColVecs + 4 - 1) / 4;

  int32_t mTileIdx = mIdx / (32 * 4);
  int64_t mTileStride = numKTiles * kTileStride;

  // Each SF block has 128 rows so pad rows to the multiple of 128.
  int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
  int64_t bTileStride = numMTiles * mTileStride;

  // Compute the global offset.
  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;

  return SFOffset;
}

inline __device__ __host__ int64_t get_sf_out_offset_8x4(std::optional<int> batchIdx, int mIdx,
                                                         int kIdx, std::optional<int> numRows,
                                                         int numCols) {
  // SF layout [numMTiles, numKTiles, 8 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, innerMIdx, innerKIdx]

  // batched tensor
  // SF layout [numBTiles, numMTiles, numKTiles, 8 (mTile), 4(kTile)]
  // --> index [bTileIdx, mTileIdx, kTileIdx, innerMIdx, innerKIdx]
  const int32_t mTile = 8;
  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int32_t innerMIdx = (mIdx % mTile);
  int64_t mStride = 4 * innerKStride;

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = mTile * mStride;

  int32_t numKTiles = (numCols + 4 - 1) / 4;
  int32_t mTileIdx = mIdx / mTile;
  int64_t mTileStride = numKTiles * kTileStride;

  int32_t numMTiles = (numRows.value_or(0) + 8 - 1) / 8;
  int64_t bTileStride = numMTiles * mTileStride;

  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + innerMIdx * mStride + innerKIdx * innerKStride;

  return SFOffset;
}

template <class SFType, int CVT_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx,
                                                int colVecIdx, std::optional<int> numRows,
                                                int numColVecs, SFType* SFout,
                                                QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2 ||
                CVT_NUM_THREADS_PER_SF == 4);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_NUM_THREADS_PER_SF == 0) {
    if (layout == QuantizationSFLayout::SWIZZLED_128x4 ||
        layout == QuantizationSFLayout::SWIZZLED_8x4) {
      // SF vector index (16 elements share one SF in the K dimension).
      // numRows and numCols are unpadded.
      int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = layout == QuantizationSFLayout::SWIZZLED_128x4
                          ? get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numColVecs)
                          : get_sf_out_offset_8x4(batchIdx, mIdx, kIdx, numRows, numColVecs);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == QuantizationSFLayout::LINEAR) {
      // Linear row-major layout, no padding required.
      int32_t KTileIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;

      int32_t numKTiles = numColVecs;
      int64_t mTileStride = numKTiles;

      int64_t BTileStride = numRows.value_or(0) * mTileStride;

      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}

template <class SFType, int CVT_FP4_SF_VEC_SIZE, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numCols,
                                                       SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 16.
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // Compute the global offset.
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
                       innerMIdx * innerMStride + innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

__device__ __forceinline__ float silu(const float& val) { return val / (1.0f + __expf(-val)); }

template <class Type, int CVT_ELTS_PER_THREAD>
inline __device__ void silu_and_mul(PackedVec<Type, CVT_ELTS_PER_THREAD>& x_vec,
                                    const PackedVec<Type, CVT_ELTS_PER_THREAD>& y_vec) {
  float2 x[CVT_ELTS_PER_THREAD / 2];
  float2 y[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      x[i] = __half22float2(x_vec.elts[i]);
      y[i] = __half22float2(y_vec.elts[i]);
      x[i].x = silu(x[i].x) * y[i].x;
      x[i].y = silu(x[i].y) * y[i].y;
      x_vec.elts[i] = __float22half2_rn(x[i]);
    } else {
      x[i] = __bfloat1622float2(x_vec.elts[i]);
      y[i] = __bfloat1622float2(y_vec.elts[i]);
      x[i].x = silu(x[i].x) * y[i].x;
      x[i].y = silu(x[i].y) * y[i].y;
      x_vec.elts[i] = __float22bfloat162_rn(x[i]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for quantization kernel with TMA in high throughput mode

template <typename FuncT>
struct PatternVisitor {
  FuncT func;

  __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

  __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }
};

template <class InputType>
struct TmaKernelTraits;

// Base template for 2-byte types (half, __nv_bfloat16)
// 2 bytes per element, 16 elements per thread = 32 bytes = 2 float4s
template <class T>
struct TmaKernelTraitsTwoBytes {
  using InputType = T;
  using SmemType = T;

  static constexpr int TMA_ROW_TILE = 16;
  static constexpr int TMA_COL_TILE = 64;  // 64 elements = 128 bytes
  static constexpr int NUM_STAGES = 4;
  static constexpr int SMEM_ROWS = TMA_ROW_TILE;      // Must match TMA_ROW_TILE for TMA loads
  static constexpr int SMEM_COLS = 8 * TMA_COL_TILE;  // 8 warps * 64 cols
  static constexpr int THREADS_PER_ROW = 4;           // laneIdx % 4
  static constexpr int ROWS_PER_WARP = 8;             // 32 / 4
  static constexpr int ROW_ITERATIONS = TMA_ROW_TILE / ROWS_PER_WARP;  // 2
  static constexpr int ELTS_PER_THREAD = 16;
  static constexpr int NUM_CONSUMER_WARPS = 8;

  static constexpr size_t SMEM_DATA_SIZE = NUM_STAGES * SMEM_ROWS * SMEM_COLS * sizeof(SmemType);
  static constexpr int SMEM_STAGE_SIZE = SMEM_ROWS * SMEM_COLS;

  // Thread indexing helper - encapsulates all index calculations
  struct ThreadIndexing {
    int const colIdxLocal;    // Thread's local column index within warp tile (constant)
    int const rowIdxLocal;    // Thread's local row index within warp (constant)
    int const baseColIdx;     // Base column index for this thread (constant)
    int const baseColVecIdx;  // Base column vector index (constant)
    int colIdx;               // Thread's global column index (in elements)
    int colVecIdx;            // Thread's column index in SF vector units

    __device__ ThreadIndexing(int laneIdx, int consumerWarpIdx)
        : colIdxLocal(laneIdx % THREADS_PER_ROW),
          rowIdxLocal(laneIdx / THREADS_PER_ROW),
          baseColIdx(consumerWarpIdx * TMA_COL_TILE + colIdxLocal * ELTS_PER_THREAD),
          baseColVecIdx(consumerWarpIdx * (TMA_COL_TILE / ELTS_PER_THREAD) + colIdxLocal),
          colIdx(baseColIdx),
          colVecIdx(baseColVecIdx) {}

    __device__ void reset() {
      colIdx = baseColIdx;
      colVecIdx = baseColVecIdx;
    }

    __device__ void advance_col() {
      colIdx += NUM_CONSUMER_WARPS * TMA_COL_TILE;
      colVecIdx = colIdx / ELTS_PER_THREAD;
    }
  };

  // Load input vector from shared memory for 2-byte types
  // Uses SWIZZLE_128B indexing, loads 2 float4s (32 bytes = 16 elements)
  template <typename PackedVecT>
  __device__ static PackedVecT load_input_vec(float4 const* base_float4, int threadRowIdxLocal,
                                              int threadColIdxLocal) {
    // Compute swizzled indices for SWIZZLE_128B
    int swizzled_col = threadColIdxLocal * 2;  // Each thread reads 2 float4s
    int col_after_swizzle_0 = threadRowIdxLocal ^ swizzled_col;
    int col_after_swizzle_1 = threadRowIdxLocal ^ (swizzled_col + 1);
    int float4_idx_0 = threadRowIdxLocal * TMA_COL_TILE / 8 + col_after_swizzle_0;
    int float4_idx_1 = threadRowIdxLocal * TMA_COL_TILE / 8 + col_after_swizzle_1;

    // Load 2 float4s (32 bytes)
    float4 load_data[2];
    load_data[0] = base_float4[float4_idx_0];
    load_data[1] = base_float4[float4_idx_1];
    return reinterpret_cast<PackedVecT&>(load_data[0]);
  }
};

// Specialization for half
template <>
struct TmaKernelTraits<half> : TmaKernelTraitsTwoBytes<half> {};

// Specialization for BF16
#ifdef ENABLE_BF16
template <>
struct TmaKernelTraits<__nv_bfloat16> : TmaKernelTraitsTwoBytes<__nv_bfloat16> {};
#endif

// Specialization for FP8 input (FP8_TO_FP4 native)
// FP8: 1 byte per element, 16 elements per thread = 16 bytes = 1 float4
template <>
struct TmaKernelTraits<__nv_fp8_e4m3> {
  using InputType = __nv_fp8_e4m3;
  using SmemType = __nv_fp8_e4m3;

  static constexpr int TMA_ROW_TILE = 8;
  static constexpr int TMA_COL_TILE = 128;  // 128 FP8 elements = 128 bytes
  static constexpr int NUM_STAGES = 6;
  static constexpr int SMEM_ROWS = TMA_ROW_TILE;      // Must match TMA_ROW_TILE for TMA loads
  static constexpr int SMEM_COLS = 8 * TMA_COL_TILE;  // 8 warps * 128 cols
  static constexpr int THREADS_PER_ROW = 8;           // laneIdx % 8
  static constexpr int ROWS_PER_WARP = 4;             // 32 / 8
  static constexpr int ROW_ITERATIONS = TMA_ROW_TILE / ROWS_PER_WARP;  // 2
  static constexpr int ELTS_PER_THREAD = 16;
  static constexpr int NUM_CONSUMER_WARPS = 8;

  static constexpr size_t SMEM_DATA_SIZE = NUM_STAGES * SMEM_ROWS * SMEM_COLS * sizeof(SmemType);
  static constexpr int SMEM_STAGE_SIZE = SMEM_ROWS * SMEM_COLS;

  // Thread indexing helper - encapsulates all index calculations
  struct ThreadIndexing {
    int const colIdxLocal;    // Thread's local column index within warp tile (constant)
    int const rowIdxLocal;    // Thread's local row index within warp (constant)
    int const baseColIdx;     // Base column index for this thread (constant)
    int const baseColVecIdx;  // Base column vector index (constant)
    int colIdx;               // Thread's global column index (in elements)
    int colVecIdx;            // Thread's column index in SF vector units

    __device__ ThreadIndexing(int laneIdx, int consumerWarpIdx)
        : colIdxLocal(laneIdx % THREADS_PER_ROW),
          rowIdxLocal(laneIdx / THREADS_PER_ROW),
          baseColIdx(consumerWarpIdx * TMA_COL_TILE + colIdxLocal * ELTS_PER_THREAD),
          baseColVecIdx(consumerWarpIdx * (TMA_COL_TILE / ELTS_PER_THREAD) + colIdxLocal),
          colIdx(baseColIdx),
          colVecIdx(baseColVecIdx) {}

    __device__ void reset() {
      colIdx = baseColIdx;
      colVecIdx = baseColVecIdx;
    }

    __device__ void advance_col() {
      colIdx += NUM_CONSUMER_WARPS * TMA_COL_TILE;
      colVecIdx = colIdx / ELTS_PER_THREAD;
    }
  };

  // Load input vector from shared memory for FP8
  // Uses linear indexing (no swizzle), loads 1 float4 (16 bytes = 16 FP8 elements)
  template <typename PackedVecT>
  __device__ static PackedVecT load_input_vec(float4 const* base_float4, int threadRowIdxLocal,
                                              int threadColIdxLocal) {
    // Linear indexing: compute float4 offset directly
    int float4_idx = threadRowIdxLocal * (TMA_COL_TILE / 16) + threadColIdxLocal;

    // Load 1 float4 (16 bytes)
    float4 load_data = base_float4[float4_idx];
    return reinterpret_cast<PackedVecT&>(load_data);
  }
};

// Shared memory size constants (for kernel launch)
constexpr size_t TMA_BARRIER_SECTION_SIZE = 1024;  // Reserved for barriers (aligned)

template <class InputType>
constexpr size_t get_tma_smem_size() {
  return TMA_BARRIER_SECTION_SIZE + TmaKernelTraits<InputType>::SMEM_DATA_SIZE;
}

}  // namespace kernels
}  // namespace tensorrt_llm
