/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Fused smooth + NVFP4 quantize: apply the per-input-channel pre_quant_scale AND NVFP4-quantize in
// ONE pass over the input, eliminating the separate x_hat = x*s elementwise pass. Self-contained
// port of TensorRT-LLM's kernels/nvfp4SmoothQuantize.cu: the device helpers it reused from
// trtllm's kernels/quantization.cuh (cvt_warp_fp16_to_fp4, PackedVec, cvt_quant_get_sf_out_offset,
// get_sf_out_offset_128x4, ...) are copied verbatim below so the xq+SF output stays byte-identical
// to fp4_quantize(x*s); the only addition is the per-channel multiply before the block-amax +
// quantize. Differences vs the TRT-LLM original: PDL comes in as a function parameter (instead of
// an env probe) and the *_SMOOTH_QUANT_THREADS/*_SMOOTH_QUANT_BLOCKS_PER_SM env overrides are
// dropped (the defaults they fell back to are hardcoded).
#pragma once

#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace flashinfer {
namespace gemm {

namespace smooth_quantize_detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers copied verbatim from TensorRT-LLM common/cudaTypeUtils.cuh (only the variants the
// kernels below instantiate: bfloat16/bfloat162 plus the generic templates they specialize).

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

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

template <typename T>
__device__ inline T cuda_abs(T val) {
  assert(false);
  return {};
}

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
  return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
  return __habs2(val);
}
#endif

// Binary maximum: compute the max of two values.
template <typename T>
__device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2) {
  return __hmax2(val1, val2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers copied verbatim from TensorRT-LLM kernels/quantization.cuh (FP4 quantization section).

constexpr int CVT_ELTS_PER_THREAD = 8;

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

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
  static_assert(sizeof(elts) == sizeof(Type) * CVT_ELTS_PER_THREAD,
                "Vector size should match the number of elements per thread.");
};

// Quantizes the provided PackedVec into the uint32_t output.
// Port note: only the UE4M3 scale-factor path (UE8M0_SF == false) is kept; the removed
// "if constexpr (UE8M0_SF)" branch emitted no code for the instantiation this port uses.
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  static_assert(!UE8M0_SF, "this port only supports the UE4M3 scale-factor path");
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
  localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(cuda_max(localMax.x, localMax.y));

  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  float outputScale;
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
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

// Port note: batch support dropped -- the original takes std::optional<int> batchIdx/numRows, but
// every call site of this port passes std::nullopt/0, making the batch term (batchIdx *
// bTileStride) identically zero. Byte-identical for the single-batch case this port serves.
inline __host__ __device__ int64_t get_sf_out_offset_128x4(int mIdx, int kIdx, int numColVecs) {
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

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

  // Compute the global offset.
  int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
                     innerMIdx * innerMStride + innerKIdx * innerKStride;

  return SFOffset;
}

// Port note: hardcoded to the SWIZZLED 128x4 layout (the only layout this port dispatches on) and
// batch support dropped as above; otherwise verbatim.
template <class SFType, int CVT_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_get_sf_out_offset(int rowIdx, int colVecIdx, int numColVecs,
                                                SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Each thread holds one vector.
  static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2 ||
                CVT_NUM_THREADS_PER_SF == 4);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    // numRows and numCols are unpadded.
    int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    auto SFOffset = get_sf_out_offset_128x4(mIdx, kIdx, numColVecs);
    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// The fused smooth-quantize kernels, ported from TensorRT-LLM kernels/nvfp4SmoothQuantize.cu.

// bf16, NVFP4 (UE4M3 SF, SF_VEC_SIZE=16), swizzled layout, single batch.
constexpr int SF_VEC_SIZE = 16;
using Type = __nv_bfloat16;
constexpr int ELTS_PER_THREAD = CVT_ELTS_PER_THREAD;
using SmoothPackedVec = PackedVec<Type>;
constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
constexpr int FAST_ELTS_PER_THREAD = SF_VEC_SIZE;

// Two of these make one complete 16-element NVFP4 scale block. Keeping the load granularity at
// 128 bits avoids imposing a stronger alignment requirement than the stock quantizer.
union alignas(16) Bf16x8 {
  uint4 bits;
  __nv_bfloat162 elts[4];
};

static_assert(sizeof(Bf16x8) == 16);

// trtllm's PadUpFn is a function-like macro (quantization.h); use a plain
// host+device helper here instead.
__host__ __device__ inline int padUp(int x, int y) { return (x + y - 1) / y * y; }

__device__ __forceinline__ void loadBf16x8(Type const* ptr, Bf16x8& result) {
  result.bits = *reinterpret_cast<uint4 const*>(ptr);
}

__device__ __forceinline__ uint64_t quantizeSmoothed16(Bf16x8& lo, Bf16x8& hi, Bf16x8 const& pqsLo,
                                                       Bf16x8 const& pqsHi, float SFScaleVal,
                                                       uint8_t* SFout) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo.elts[i] = __hmul2(lo.elts[i], pqsLo.elts[i]);
    hi.elts[i] = __hmul2(hi.elts[i], pqsHi.elts[i]);
  }

  // Match the legacy even lane's reduction order: reduce each 8-element half independently, then
  // merge the high half into the low half. For finite BF16 values this produces the same scale for
  // both halves.
  auto loMax = cuda_abs(lo.elts[0]);
  auto hiMax = cuda_abs(hi.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    loMax = cuda_max(loMax, cuda_abs(lo.elts[i]));
    hiMax = cuda_max(hiMax, cuda_abs(hi.elts[i]));
  }
  auto const localMax = cuda_max(hiMax, loMax);
  float const vecMax = float(cuda_max(localMax.x, localMax.y));

  // This is deliberately kept instruction-for-instruction equivalent to cvt_warp_fp16_to_fp4's
  // UE4M3 path.
  auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  uint8_t const fp8SFVal = tmp.__x;
  SFValue = static_cast<float>(tmp);
  float const outputScale =
      vecMax != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
                  : 0.0f;

  if (SFout != nullptr) *SFout = fp8SFVal;

  float2 fp2Vals[FAST_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fp2Vals[i] = __bfloat1622float2(lo.elts[i]);
    fp2Vals[i + 4] = __bfloat1622float2(hi.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
    fp2Vals[i + 4].x *= outputScale;
    fp2Vals[i + 4].y *= outputScale;
  }
  return fp32_vec_to_e2m1(fp2Vals);
}

template <int NumCols, int RowsPerCta>
__global__ void __launch_bounds__(512, 4)
    smooth_quantize_fast_kernel(int numRows, Type const* __restrict__ in,
                                Type const* __restrict__ pqs, float const* __restrict__ SFScale,
                                uint64_t* __restrict__ out, uint8_t* __restrict__ SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(NumCols % (4 * SF_VEC_SIZE) == 0);
  constexpr int NumSfCols = NumCols / SF_VEC_SIZE;
  constexpr int NumSfGroups = NumSfCols / 4;
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  cudaGridDependencySynchronize();

  // Hot path: map a fixed number of complete rows onto each CTA. K=3072 uses two rows and 384
  // threads, so every thread owns exactly one 16-value scale block without a warp shuffle.
  for (int rowBase = blockIdx.x * RowsPerCta; rowBase < numRows;
       rowBase += gridDim.x * RowsPerCta) {
    int const rowsRemaining = numRows - rowBase;
    int const rowsThisCta = rowsRemaining < RowsPerCta ? rowsRemaining : RowsPerCta;
    int const workItems = rowsThisCta * NumSfCols;
    for (int item = threadIdx.x; item < workItems; item += blockDim.x) {
      int const rowOffset = item / NumSfCols;
      int const sfCol = item - rowOffset * NumSfCols;
      int const row = rowBase + rowOffset;
      int64_t const vecOffset = static_cast<int64_t>(row) * NumSfCols + sfCol;

      Type const* xPtr = in + vecOffset * FAST_ELTS_PER_THREAD;
      Type const* pqsPtr = pqs + sfCol * FAST_ELTS_PER_THREAD;
      Bf16x8 xLo;
      Bf16x8 xHi;
      Bf16x8 pqsLo;
      Bf16x8 pqsHi;
      loadBf16x8(xPtr, xLo);
      loadBf16x8(xPtr + 8, xHi);
      loadBf16x8(pqsPtr, pqsLo);
      loadBf16x8(pqsPtr + 8, pqsHi);

      int64_t const sfOffset = get_sf_out_offset_128x4(row, sfCol, NumSfCols);
      out[vecOffset] = quantizeSmoothed16(xLo, xHi, pqsLo, pqsHi, SFScaleVal, SFout + sfOffset);
    }
  }

  // Cold path: only scale factors have padded rows. Four consecutive SF columns are contiguous in
  // the 128x4 layout, so initialize them with one aligned 32-bit store instead of four byte stores.
  int const numPaddedRows = padUp(numRows, 128);
  int64_t const numPaddingStores = static_cast<int64_t>(numPaddedRows - numRows) * NumSfGroups;
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < numPaddingStores; item += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int const paddingRow = static_cast<int>(item / NumSfGroups);
    int const sfGroup = static_cast<int>(item - static_cast<int64_t>(paddingRow) * NumSfGroups);
    int const row = numRows + paddingRow;
    int const sfCol = sfGroup * 4;
    int64_t const sfOffset = get_sf_out_offset_128x4(row, sfCol, NumSfCols);
    *reinterpret_cast<uint32_t*>(SFout + sfOffset) = 0u;
  }

  cudaTriggerProgrammaticLaunchCompletion();
#else
  // Fail loudly instead of silently leaving out/SFout uninitialized if a build for an
  // unsupported architecture is ever launched.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("nvfp4_smooth_quantize requires SM100 or newer\n");
    __trap();
  }
#endif
}

__global__ void __launch_bounds__(512, 4)
    smooth_quantize_legacy_kernel(int numRows, int numCols, int numPaddedCols, Type const* in,
                                  Type const* pqs, float const* SFScale, uint32_t* out,
                                  uint32_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  int const numPaddedRowsForSf = padUp(numRows, 128);
  int const numColsForSf = padUp(numPaddedCols, 4 * SF_VEC_SIZE);
  int const numColThreads = numCols / ELTS_PER_THREAD;
  int const numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
  int const numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

  cudaGridDependencySynchronize();
  for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x) {
    bool const isRowPadding = (rowIdx >= numRows);
    for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x) {
      auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
          rowIdx, colIdx, numPaddedCols / SF_VEC_SIZE, SFout);

      if (isRowPadding || colIdx >= numColThreads) {
        if (sf_out != nullptr) sf_out[0] = 0x00;
        if (!isRowPadding && colIdx >= numColThreads && colIdx < numPaddedColThreads)
          reinterpret_cast<uint32_t*>(
              out)[static_cast<int64_t>(rowIdx) * numPaddedColThreads + colIdx] = 0u;
        continue;
      }

      int64_t const inOffset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
      int64_t const outOffset = static_cast<int64_t>(rowIdx) * numPaddedColThreads + colIdx;
      SmoothPackedVec in_vec = reinterpret_cast<SmoothPackedVec const*>(in)[inOffset];
      // --- the fusion: smooth by the per-channel pre_quant_scale (broadcast over rows) ---
      SmoothPackedVec p_vec = reinterpret_cast<SmoothPackedVec const*>(pqs)[colIdx];
#pragma unroll
      for (int i = 0; i < ELTS_PER_THREAD / 2; i++)
        in_vec.elts[i] = __hmul2(in_vec.elts[i], p_vec.elts[i]);
      reinterpret_cast<uint32_t*>(out)[outOffset] =
          cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, false>(in_vec, SFScaleVal, sf_out);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("nvfp4_smooth_quantize requires SM100 or newer\n");
    __trap();
  }
#endif
}

template <int NumCols, int RowsPerCta>
void launchSmoothQuantizeFast(void* out, void* sfOut, void const* in, void const* pqs,
                              float const* sfScale, int numRows, int multiProcessorCount,
                              int blockThreads, int blocksPerSm, bool enablePDL,
                              cudaStream_t stream) {
  constexpr int NumSfGroups = NumCols / (4 * SF_VEC_SIZE);
  int const numPaddedRows = padUp(numRows, 128);
  int64_t const hotCtas = (static_cast<int64_t>(numRows) + RowsPerCta - 1) / RowsPerCta;
  int64_t const numPaddingStores = static_cast<int64_t>(numPaddedRows - numRows) * NumSfGroups;
  int64_t const paddingCtas = (numPaddingStores + blockThreads - 1) / blockThreads;
  int64_t const wantedCtas = std::max(hotCtas, paddingCtas);
  int64_t const maxCtas = static_cast<int64_t>(multiProcessorCount) * blocksPerSm;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned int>(std::min(wantedCtas, maxCtas)));
  cfg.blockDim = dim3(blockThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  auto* kernel = &smooth_quantize_fast_kernel<NumCols, RowsPerCta>;
  cudaLaunchKernelEx(&cfg, kernel, numRows, reinterpret_cast<Type const*>(in),
                     reinterpret_cast<Type const*>(pqs), sfScale, reinterpret_cast<uint64_t*>(out),
                     reinterpret_cast<uint8_t*>(sfOut));
}

}  // namespace smooth_quantize_detail

// Fused smooth + NVFP4 quantize: (out, sf_out) = NVFP4-quantize(in * pqs) in a single pass over
// in, folding the per-input-channel pre_quant_scale smoothing into the quantize. Byte-identical to
// fp4_quantize(in * pqs) (same cvt_warp_fp16_to_fp4 + swizzled SF layout), so the residual GEMM
// consumes the output unchanged. in [m, n] bf16, pqs [n] bf16, sf_scale f32[1] (the per-tensor
// global scale). out [m, n/2] uint8 (packed e2m1), sf_out swizzled UE4M3 block scales (vec size
// 16). SM100+ only.
inline void nvfp4_smooth_quantize(void* out, void* sf_out, void const* in, void const* pqs,
                                  float const* sf_scale, int m, int n, int multiProcessorCount,
                                  cudaStream_t stream, bool enable_pdl) {
  using namespace smooth_quantize_detail;

  if (m == 0 || n == 0) return;

  bool const enablePDL = enable_pdl;
  bool const useFastPath = (n == 3072 || n == 12288);
  if (useFastPath) {
    // Same-node SM100 sweeps over the Qwen image-token M values select 192 threads for K=3072
    // and 256 for K=12288. A grid cap of eight CTAs per SM is best for both.
    int const blockThreads = n == 3072 ? 192 : 256;
    int const blocksPerSm = 8;

    if (n == 3072)
      launchSmoothQuantizeFast<3072, 2>(out, sf_out, in, pqs, sf_scale, m, multiProcessorCount,
                                        blockThreads, blocksPerSm, enablePDL, stream);
    else
      launchSmoothQuantizeFast<12288, 1>(out, sf_out, in, pqs, sf_scale, m, multiProcessorCount,
                                         blockThreads, blocksPerSm, enablePDL, stream);
    return;
  }

  dim3 block(std::min(n / ELTS_PER_THREAD, 512));
  int const numBlocksPerSM = std::max(1, 2048 / int(block.x));
  dim3 grid(std::min(padUp(m, 128), multiProcessorCount * numBlocksPerSM));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  // No column padding here (n is the padded width); the residual GEMM and the SF layout use n.
  cudaLaunchKernelEx(&cfg, smooth_quantize_legacy_kernel, m, n, n,
                     reinterpret_cast<Type const*>(in), reinterpret_cast<Type const*>(pqs),
                     sf_scale, reinterpret_cast<uint32_t*>(out),
                     reinterpret_cast<uint32_t*>(sf_out));
}

}  // namespace gemm
}  // namespace flashinfer
