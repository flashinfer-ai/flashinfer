// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

// Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad, csrc/common/fp4/nvfp4_utils.cuh.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
namespace flashinfer::resinfer {
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
template <typename T>
struct TypeConverter {
  using Type = half2;
};

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
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
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1_with_dequant(float2 (&in_out_buffer)[4]) {
  uint32_t packed_u32;
  uint32_t f16x2_0, f16x2_1, f16x2_2, f16x2_3;

  asm volatile(
      "{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b0, %6,  %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b1, %8,  %7;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b2, %10, %9;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b3, %12, %11;\n"
      "  cvt.rn.f16x2.e2m1x2           %0, b0;\n"
      "  cvt.rn.f16x2.e2m1x2           %1, b1;\n"
      "  cvt.rn.f16x2.e2m1x2           %2, b2;\n"
      "  cvt.rn.f16x2.e2m1x2           %3, b3;\n"
      "  mov.b32 %4, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(f16x2_0), "=r"(f16x2_1), "=r"(f16x2_2), "=r"(f16x2_3), "=r"(packed_u32)
      : "f"(in_out_buffer[0].x), "f"(in_out_buffer[0].y), "f"(in_out_buffer[1].x),
        "f"(in_out_buffer[1].y), "f"(in_out_buffer[2].x), "f"(in_out_buffer[2].y),
        "f"(in_out_buffer[3].x), "f"(in_out_buffer[3].y));

  union U32Half2 {
    uint32_t u;
    __half2 h2;
  };

  auto residual_inplace = [&](uint32_t f16x2_u32, float2& inout) {
    U32Half2 v;
    v.u = f16x2_u32;

    __half hx = __low2half(v.h2);
    __half hy = __high2half(v.h2);

    float hat_x = __half2float(hx);
    float hat_y = __half2float(hy);

    inout.x = hat_x;
    inout.y = hat_y;
  };

  residual_inplace(f16x2_0, in_out_buffer[0]);
  residual_inplace(f16x2_1, in_out_buffer[1]);
  residual_inplace(f16x2_2, in_out_buffer[2]);
  residual_inplace(f16x2_3, in_out_buffer[3]);

  return packed_u32;
}

inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float compute_sf_from_max(float maxVal, float SFScaleVal) {
  return SFScaleVal * (maxVal * reciprocal_approximate_ftz(6.0f));
}

template <bool UE8M0_SF>
__device__ __forceinline__ float encode_sf_to_fp8(float SFValue, uint8_t* out_fp8) {
  uint8_t fp8;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8 = tmp & 0xff;
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8) = tmp;
    SFValue = float(tmp);
  }
  if (out_fp8) *out_fp8 = fp8;
  return SFValue;
}

__device__ __forceinline__ float compute_output_scale_precise(float SFValue, float SFScaleVal) {
  return (SFValue != 0.0f) ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;
}

struct u32x2 {
  uint32_t lo, hi;
};

template <class Type>
__device__ __forceinline__ float2 packed_vec_to_float2(PackedVec<Type> const& vec, int index) {
  if constexpr (std::is_same_v<Type, half>) {
    return __half22float2(vec.elts[index]);
  } else {
    static_assert(std::is_same_v<Type, __nv_bfloat16>);
    return __bfloat1622float2(vec.elts[index]);
  }
}

template <bool WITH_DEQUANT>
__device__ __forceinline__ u32x2 fp32x16_to_e2m1(float2 (&loValues)[4], float2 (&hiValues)[4]) {
  u32x2 packed;
  if constexpr (WITH_DEQUANT) {
    packed.lo = fp32x8_to_e2m1_with_dequant(loValues);
    packed.hi = fp32x8_to_e2m1_with_dequant(hiValues);
  } else {
    packed.lo = fp32_vec_to_e2m1(loValues);
    packed.hi = fp32_vec_to_e2m1(hiValues);
  }
  return packed;
}

template <class Type>
__device__ __forceinline__ void load_packed_vec16(Type const* ptr, PackedVec<Type>& lo,
                                                  PackedVec<Type>& hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  union {
    uint32_t u[8];
    struct {
      PackedVec<Type> lo, hi;
    } v;
  } tmp;
  asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(tmp.u[0]), "=r"(tmp.u[1]), "=r"(tmp.u[2]), "=r"(tmp.u[3]), "=r"(tmp.u[4]),
                 "=r"(tmp.u[5]), "=r"(tmp.u[6]), "=r"(tmp.u[7])
               : "l"(ptr));
  lo = tmp.v.lo;
  hi = tmp.v.hi;
#else
  lo = reinterpret_cast<PackedVec<Type> const*>(ptr)[0];
  hi = reinterpret_cast<PackedVec<Type> const*>(ptr)[1];
#endif
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ u32x2 cvt_warp_fp16_to_fp4_mext_r1_fast16(
    PackedVec<Type>& loVec, PackedVec<Type>& hiVec, float SFScaleVal, float SFScaleResVal,
    uint8_t* SFout, uint32_t* residueDataOut, uint8_t* residueSFOut) {
  static_assert(CVT_FP4_ELTS_PER_THREAD == 8,
                "pack16 transitional path assumes 8-element base helper.");

  auto localMax = __habs2(loVec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(loVec.elts[i]));
  }

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(hiVec.elts[i]));
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 loOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 loScaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    loOriginal[i] = packed_vec_to_float2(loVec, i);
    hiOriginal[i] = packed_vec_to_float2(hiVec, i);
    loScaled[i].x = loOriginal[i].x * outputScale;
    loScaled[i].y = loOriginal[i].y * outputScale;
    hiScaled[i].x = hiOriginal[i].x * outputScale;
    hiScaled[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 mainPacked = fp32x16_to_e2m1<true>(loScaled, hiScaled);

  float inverseOutputScale = outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

  float2 loResidue[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiResidue[CVT_FP4_ELTS_PER_THREAD / 2];
  float localMaxResidue = 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float loDeqX = loScaled[i].x * inverseOutputScale;
    float loDeqY = loScaled[i].y * inverseOutputScale;
    float hiDeqX = hiScaled[i].x * inverseOutputScale;
    float hiDeqY = hiScaled[i].y * inverseOutputScale;

    loResidue[i].x = loOriginal[i].x - loDeqX;
    loResidue[i].y = loOriginal[i].y - loDeqY;
    hiResidue[i].x = hiOriginal[i].x - hiDeqX;
    hiResidue[i].y = hiOriginal[i].y - hiDeqY;

    localMaxResidue = fmaxf(localMaxResidue, fmaxf(fabsf(loResidue[i].x), fabsf(loResidue[i].y)));
    localMaxResidue = fmaxf(localMaxResidue, fmaxf(fabsf(hiResidue[i].x), fabsf(hiResidue[i].y)));
  }

  float residueSFValue = compute_sf_from_max(localMaxResidue, SFScaleResVal);
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleResVal);

  float2 loResidueScaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiResidueScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    loResidueScaled[i].x = loResidue[i].x * residueOutputScale;
    loResidueScaled[i].y = loResidue[i].y * residueOutputScale;
    hiResidueScaled[i].x = hiResidue[i].x * residueOutputScale;
    hiResidueScaled[i].y = hiResidue[i].y * residueOutputScale;
  }

  if (residueDataOut) {
    *reinterpret_cast<uint2*>(residueDataOut) =
        make_uint2(fp32_vec_to_e2m1(loResidueScaled), fp32_vec_to_e2m1(hiResidueScaled));
  }

  return mainPacked;
}

template <class SFType, int SF_VEC_SIZE = CVT_FP4_SF_VEC_SIZE>
__device__ inline int64_t compute_tiled_sf_offset(int32_t mIdx, int32_t kIdx, int32_t numTotalCols,
                                                  bool swizzle = true) {
  if (!swizzle) {
    return int64_t(mIdx) * (numTotalCols / SF_VEC_SIZE) + kIdx;
  }
  int32_t mTileIdx = mIdx / (32 * 4);
  constexpr int factor = SF_VEC_SIZE * 4;
  int64_t numKTiles = (int64_t(numTotalCols) + factor - 1) / factor;
  int64_t mTileStride = numKTiles * 32 * 4 * 4;

  int32_t kTileIdx = (kIdx / 4);

  int64_t kTileStride = 32 * 4 * 4;

  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * 4;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4;

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  return mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
         innerMIdx * innerMStride + innerKIdx * innerKStride;
}

}  // namespace flashinfer::resinfer
