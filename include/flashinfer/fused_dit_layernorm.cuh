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
#ifndef FLASHINFER_FUSED_DIT_LAYERNORM_CUH_
#define FLASHINFER_FUSED_DIT_LAYERNORM_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// float2 packed math: SM100+ has native f2math::fadd2/f2math::fmul2/f2math::ffma2.
// For pre-SM100, we provide wrappers under a namespace to avoid ODR collision
// with host-side stubs in CUDA 13.0's sm_100_rt.h.
namespace f2math {
__device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return ::__fadd2_rn(a, b);
#else
  return make_float2(a.x + b.x, a.y + b.y);
#endif
}
__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return ::__fmul2_rn(a, b);
#else
  return make_float2(a.x * b.x, a.y * b.y);
#endif
}
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return ::__ffma2_rn(a, b, c);
#else
  return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
#endif
}
__device__ __forceinline__ float fadd(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
__device__ __forceinline__ float fmaf_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fmaf_rn(a, b, c);
#else
  return fmaf(a, b, c);
#endif
}
__device__ __forceinline__ float frsqrt(float a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __frsqrt_rn(a);
#else
  return rsqrtf(a);
#endif
}
}  // namespace f2math

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <optional>
#include <type_traits>

namespace flashinfer {
namespace fused_layernorm {

#define FLASHINFER_FUSED_LN_CHECK(condition)                                                 \
  do {                                                                                       \
    if (!(condition)) {                                                                      \
      fprintf(stderr, "FLASHINFER_FUSED_LN_CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #condition);                                                                   \
      abort();                                                                               \
    }                                                                                        \
  } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 1: FwdParam — kernel parameter struct
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct FwdParam {
  uint32_t* normed_output;
  T* output;
  const T* input;
  const T* residual;
  int batch_size;
  int num_rows;
  const float epsilon;
  const float* gamma;
  const float* beta;
  const T* gate;
  const float* gate_bias;
  const T* scale;
  const float* scale_bias;
  const T* shift;
  const float* shift_bias;
  float* sf_scale;
  float* input_sf_scale;
  uint32_t* sf_out;
  int n;
  __host__ __device__ FwdParam(T* output_, uint32_t* normed_output_, const T* input_,
                               const T* residual_, int batch_size_, int num_rows_,
                               const float epsilon_, const float* gamma_, const float* beta_,
                               const T* gate_, const float* gate_bias_, const T* scale_,
                               const float* scale_bias_, const T* shift_, const float* shift_bias_,
                               int n_ = 0, float* sf_scale_ = nullptr,
                               float* input_sf_scale_ = nullptr, uint32_t* sf_out_ = nullptr)
      : normed_output(normed_output_),
        output(output_),
        input(input_),
        residual(residual_),
        batch_size(batch_size_),
        num_rows(num_rows_),
        epsilon(epsilon_),
        gamma(gamma_),
        beta(beta_),
        gate(gate_),
        gate_bias(gate_bias_),
        scale(scale_),
        scale_bias(scale_bias_),
        shift(shift_),
        shift_bias(shift_bias_),
        sf_scale(sf_scale_),
        input_sf_scale(input_sf_scale_),
        sf_out(sf_out_),
        n(n_) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 2: Packed data types
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int PACK_SIZE>
struct PackType {
  using type = float;
};

template <typename T>
struct PackType<T, 1> {
  struct __CUDA_ALIGN__(std::alignment_of_v<T>) type {
    T array[1];
  };
};

template <>
struct PackType<__nv_bfloat16, 2> {
  struct __CUDA_ALIGN__(4) type {
    __nv_bfloat16 array[2];
  };
};

template <>
struct PackType<__nv_bfloat16, 4> {
  struct __CUDA_ALIGN__(8) type {
    __nv_bfloat16 array[4];
  };
};

template <>
struct PackType<__nv_bfloat16, 8> {
  struct __CUDA_ALIGN__(16) type {
    __nv_bfloat16 array[8];
  };
};

template <>
struct PackType<__nv_bfloat16, 16> {
  struct __CUDA_ALIGN__(32) type {
    __nv_bfloat16 array[16];
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 3: Quantization helpers (SM100+ Blackwell only for NVFP4/MXFP8)
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class QuantizationSFLayout { SWIZZLED, LINEAR };

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
  return 0;
#endif
}

inline __device__ uint64_t fp32_vec_to_e4m3(float2 (&array)[4]) {
  union {
    uint64_t val;
    __nv_fp8x2_e4m3 elts[4];
  } u;
  u.elts[0] = __nv_fp8x2_e4m3(array[0]);
  u.elts[1] = __nv_fp8x2_e4m3(array[1]);
  u.elts[2] = __nv_fp8x2_e4m3(array[2]);
  u.elts[3] = __nv_fp8x2_e4m3(array[3]);
  return u.val;
}

inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

inline __host__ __device__ int64_t get_sf_out_offset_128x4(std::optional<int> batchIdx, int mIdx,
                                                           int kIdx, std::optional<int> numRows,
                                                           int numColVecs) {
  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;
  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4 * innerKStride;
  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * innerMStride;
  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * outerMStride;
  int32_t numKTiles = (numColVecs + 4 - 1) / 4;
  int32_t mTileIdx = mIdx / (32 * 4);
  int64_t mTileStride = numKTiles * kTileStride;
  int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
  int64_t bTileStride = numMTiles * mTileStride;
  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;
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
  if (threadIdx.x % CVT_NUM_THREADS_PER_SF == 0) {
    if (layout == QuantizationSFLayout::SWIZZLED) {
      int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;
      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numColVecs);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == QuantizationSFLayout::LINEAR) {
      int32_t KTileIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
      int32_t numKTiles = numColVecs;
      int64_t mTileStride = numKTiles;
      int64_t BTileStride = numRows.value_or(0) * mTileStride;
      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    }
  }
#endif
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 4: FP4 and FP8 converters (SM100+ Blackwell only)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, bool UE8M0_SF = false, typename = void>
struct FP4Converter;

template <typename TIn, bool UE8M0_SF>
struct FP4Converter<
    TIn, UE8M0_SF,
    std::enable_if_t<std::is_same_v<TIn, half> || std::is_same_v<TIn, __nv_bfloat16>>> {
  static constexpr int ELTS_PER_THREAD = 8;
  static constexpr int SF_VEC_SIZE = 16;
  static constexpr int THREADS_PER_WARP = 32;

  float const SFScaleVal;
  int const numCols;
  int const numRows;
  uint32_t* const SFout;
  uint32_t* const out;

  template <typename Param>
  __device__ __forceinline__
  FP4Converter<TIn, UE8M0_SF,
               std::enable_if_t<std::is_same_v<TIn, half> || std::is_same_v<TIn, __nv_bfloat16>>>(
      Param const& p)
      : SFScaleVal(p.sf_scale == nullptr ? 1.0f : p.sf_scale[0]),
        numCols(p.n),
        numRows(p.num_rows),
        SFout(p.sf_out),
        out(p.normed_output) {}

  template <size_t ELTS_PER_THREAD_T, typename TPacked>
  __device__ __forceinline__ void post_process(int batchIdx, int rowIdx, int n_base,
                                               TPacked packed_input) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    static_assert(sizeof(TPacked) == sizeof(TIn) * ELTS_PER_THREAD_T, "Vec size mismatch.");
    static constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD_T;
    static_assert(NUM_THREADS_PER_SF == 2);

    int globalRowIdx = batchIdx * numRows + rowIdx;
    int colIdx = n_base / ELTS_PER_THREAD_T;

    auto localMax = __habs2({packed_input.array[0], packed_input.array[1]});
#pragma unroll
    for (int i = 2; i < static_cast<int>(ELTS_PER_THREAD_T); i += 2) {
      localMax = __hmax2(localMax, __habs2({packed_input.array[i], packed_input.array[i + 1]}));
    }
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);

    float vecMax;
    if constexpr (std::is_same_v<TIn, __nv_bfloat16>) {
      vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));
    } else {
      vecMax = __half2float(__hmax(localMax.x, localMax.y));
    }

    float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    uint8_t fp8SFVal;
    if constexpr (UE8M0_SF) {
      __nv_fp8_e8m0 tmp;
      tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
      SFValue = static_cast<float>(tmp);
      fp8SFVal = tmp.__x;
    } else {
      __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
      fp8SFVal = tmp.__x;
      SFValue = static_cast<float>(tmp);
    }

    auto SFOffset = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(
        batchIdx, rowIdx, colIdx, numRows, numCols / SF_VEC_SIZE, SFout,
        QuantizationSFLayout::SWIZZLED);
    *SFOffset = fp8SFVal;

    float outputScale =
        vecMax != 0.f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
    float2 fp2Vals[ELTS_PER_THREAD_T / 2];
#pragma unroll
    for (int i = 0; i < static_cast<int>(ELTS_PER_THREAD_T); i += 2) {
      if constexpr (std::is_same_v<TIn, __nv_bfloat16>) {
        fp2Vals[i / 2] = __bfloat1622float2({packed_input.array[i], packed_input.array[i + 1]});
      } else {
        fp2Vals[i / 2] = __half22float2({packed_input.array[i], packed_input.array[i + 1]});
      }
      fp2Vals[i / 2].x *= outputScale;
      fp2Vals[i / 2].y *= outputScale;
    }
    uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
    int64_t outOffset = globalRowIdx * (numCols / static_cast<int>(ELTS_PER_THREAD_T)) + colIdx;
    out[outOffset] = e2m1Vec;
#else
    printf("FP4 output is not supported pre-Blackwell!\n");
#endif
  }
};

template <typename T, typename = void>
struct FP8Converter;

template <typename TIn>
struct FP8Converter<
    TIn, std::enable_if_t<std::is_same_v<TIn, half> || std::is_same_v<TIn, __nv_bfloat16>>> {
  static constexpr int ELTS_PER_THREAD = 8;
  static constexpr int SF_VEC_SIZE = 32;
  static constexpr int THREADS_PER_WARP = 32;

  int const numCols;
  int const numRows;
  uint32_t* const SFout;
  uint64_t* const out;

  template <typename Param>
  __device__ __forceinline__ FP8Converter<
      TIn, std::enable_if_t<std::is_same_v<TIn, half> || std::is_same_v<TIn, __nv_bfloat16>>>(
      Param const& p)
      : numCols(p.n),
        numRows(p.num_rows),
        SFout(p.sf_out),
        out(reinterpret_cast<uint64_t*>(p.normed_output)) {}

  template <size_t ELTS_PER_THREAD_T, typename TPacked>
  __device__ __forceinline__ void post_process(int batchIdx, int rowIdx, int n_base,
                                               TPacked packed_input) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(sizeof(TPacked) == sizeof(TIn) * ELTS_PER_THREAD_T, "Vec size mismatch.");
    static constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD_T;
    static_assert(NUM_THREADS_PER_SF == 4);

    int globalRowIdx = batchIdx * numRows + rowIdx;
    int colIdx = n_base / static_cast<int>(ELTS_PER_THREAD_T);

    auto localMax = __habs2({packed_input.array[0], packed_input.array[1]});
#pragma unroll
    for (int i = 2; i < static_cast<int>(ELTS_PER_THREAD_T); i += 2) {
      localMax = __hmax2(localMax, __habs2({packed_input.array[i], packed_input.array[i + 1]}));
    }
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    if constexpr (NUM_THREADS_PER_SF == 4) {
      localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
    }

    float vecMax;
    if constexpr (std::is_same_v<TIn, __nv_bfloat16>) {
      vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));
    } else {
      vecMax = __half2float(__hmax(localMax.x, localMax.y));
    }

    float SFValue = vecMax * reciprocal_approximate_ftz(448.0f);
    uint8_t fp8SFVal;
    __nv_fp8_e8m0 tmpSFVal;
    tmpSFVal.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmpSFVal);
    fp8SFVal = tmpSFVal.__x;
    float outputScale = vecMax != 0.f ? reciprocal_approximate_ftz(SFValue) : 0.0f;

    auto SFOffset = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(
        batchIdx, rowIdx, colIdx, numRows, numCols / SF_VEC_SIZE, SFout,
        QuantizationSFLayout::SWIZZLED);
    *SFOffset = fp8SFVal;

    float2 fp2Vals[ELTS_PER_THREAD_T / 2];
#pragma unroll
    for (int i = 0; i < static_cast<int>(ELTS_PER_THREAD_T); i += 2) {
      if constexpr (std::is_same_v<TIn, __nv_bfloat16>) {
        fp2Vals[i / 2] = __bfloat1622float2({packed_input.array[i], packed_input.array[i + 1]});
      } else {
        fp2Vals[i / 2] = __half22float2({packed_input.array[i], packed_input.array[i + 1]});
      }
      fp2Vals[i / 2].x *= outputScale;
      fp2Vals[i / 2].y *= outputScale;
    }
    uint64_t e4m3Vec = fp32_vec_to_e4m3(fp2Vals);
    int64_t outOffset = globalRowIdx * (numCols / static_cast<int>(ELTS_PER_THREAD_T)) + colIdx;
    out[outOffset] = e4m3Vec;
#else
    printf("MXFP8 output is not supported pre-Blackwell!\n");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 5: Mode enum and traits
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class FusedLayerNormMode {
  GATE_RESIDUAL_GAMMA_BETA,
  RESIDUAL_SCALE_SHIFT,
  GATE_RESIDUAL_SCALE_SHIFT,
};

template <FusedLayerNormMode mode>
struct LayerNormModeTrait {
  static constexpr bool use_gate = (mode == FusedLayerNormMode::GATE_RESIDUAL_GAMMA_BETA) ||
                                   (mode == FusedLayerNormMode::GATE_RESIDUAL_SCALE_SHIFT);
  static constexpr bool use_gamma_beta = (mode == FusedLayerNormMode::GATE_RESIDUAL_GAMMA_BETA);
  static constexpr bool use_scale_shift = (mode == FusedLayerNormMode::RESIDUAL_SCALE_SHIFT) ||
                                          (mode == FusedLayerNormMode::GATE_RESIDUAL_SCALE_SHIFT);
};

enum class OutputFormat { BF16 = 0, NVFP4 = 1, MXFP8 = 2 };

using InputType = __nv_bfloat16;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 6: Main fused layernorm kernel
//
// Gate/scale/shift inputs use a stride of 6*hidden_dim in the row dimension,
// matching WAN's temb.chunk(6, dim=2) pattern.
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int gate_shift_scale_stride = 6;

struct __align__(32) float2_4 {
  float2 data[4];
};

struct Float2Sum {
  __device__ __forceinline__ float2 operator()(const float2& a, const float2& b) const {
    return f2math::fadd2(a, b);
  }
};

template <typename T, const int block_size, const int hidden_size, const FusedLayerNormMode mode,
          OutputFormat output_format, bool USE_INPUT_SF_SCALE>
__global__ void meta_fused_layernorm(FwdParam<T> param) {
  constexpr int hidden_size_pack2 = hidden_size / 2;
  constexpr int elem_per_thread = hidden_size_pack2 / block_size;
  static_assert(elem_per_thread == 4, "elem_per_thread must be 4");
  constexpr bool kUseGate = LayerNormModeTrait<mode>::use_gate;
  constexpr bool kUseGammaBeta = LayerNormModeTrait<mode>::use_gamma_beta;
  constexpr bool kUseScaleShift = LayerNormModeTrait<mode>::use_scale_shift;

  using BlockReduce = cub::BlockReduce<float2, block_size>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  __shared__ float2 s_mean;
  __shared__ float2 s_inv_std;

  __nv_bfloat162* normalized_output_base = reinterpret_cast<__nv_bfloat162*>(param.normed_output);
  __nv_bfloat162* residual_out_base = reinterpret_cast<__nv_bfloat162*>(param.output);
  const __nv_bfloat162* input_base = reinterpret_cast<const __nv_bfloat162*>(param.input);
  const __nv_bfloat162* residual_base = reinterpret_cast<const __nv_bfloat162*>(param.residual);

  const __nv_bfloat162* gate_base =
      kUseGate ? reinterpret_cast<const __nv_bfloat162*>(param.gate) : nullptr;
  const __nv_bfloat162* scale_base =
      kUseScaleShift ? reinterpret_cast<const __nv_bfloat162*>(param.scale) : nullptr;
  const __nv_bfloat162* shift_base =
      kUseScaleShift ? reinterpret_cast<const __nv_bfloat162*>(param.shift) : nullptr;

  const float2* gamma_ptr = kUseGammaBeta ? reinterpret_cast<const float2*>(param.gamma) : nullptr;
  const float2* beta_ptr = kUseGammaBeta ? reinterpret_cast<const float2*>(param.beta) : nullptr;
  const float2* gate_bias_ptr =
      kUseGate ? reinterpret_cast<const float2*>(param.gate_bias) : nullptr;
  const float2* scale_bias_ptr =
      kUseScaleShift ? reinterpret_cast<const float2*>(param.scale_bias) : nullptr;
  const float2* shift_bias_ptr =
      kUseScaleShift ? reinterpret_cast<const float2*>(param.shift_bias) : nullptr;

  float input_sf_scale = USE_INPUT_SF_SCALE ? param.input_sf_scale[0] : 1.0f;
  float2 input_sf_scale_val = make_float2(input_sf_scale, input_sf_scale);

  int i = threadIdx.x * 4;

  float2 gamma[elem_per_thread];
  float2 beta[elem_per_thread];
  float2 gate_bias[elem_per_thread];
  float2 scale_bias[elem_per_thread];
  float2 shift_bias[elem_per_thread];

  if constexpr (kUseGammaBeta) {
    float2_4 tmp_gamma = reinterpret_cast<const float2_4*>(&gamma_ptr[i])[0];
    float2_4 tmp_beta = reinterpret_cast<const float2_4*>(&beta_ptr[i])[0];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      gamma[k] = tmp_gamma.data[k];
      beta[k] = tmp_beta.data[k];
    }
  }

  if constexpr (kUseGate) {
    float2_4 tmp_gate_bias = reinterpret_cast<const float2_4*>(&gate_bias_ptr[i])[0];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      gate_bias[k] = tmp_gate_bias.data[k];
    }
  }

  if constexpr (kUseScaleShift) {
    float2_4 tmp_scale_bias = reinterpret_cast<const float2_4*>(&scale_bias_ptr[i])[0];
    float2_4 tmp_shift_bias = reinterpret_cast<const float2_4*>(&shift_bias_ptr[i])[0];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      scale_bias[k] = tmp_scale_bias.data[k];
      shift_bias[k] = tmp_shift_bias.data[k];
    }
  }

#pragma unroll
  for (int batch_id = 0; batch_id < param.batch_size; batch_id++) {
    int row_id = blockIdx.x;
    int gate_shift_scale_index =
        batch_id * param.num_rows * hidden_size_pack2 * gate_shift_scale_stride +
        row_id * hidden_size_pack2 * gate_shift_scale_stride + i;

    const size_t batch_offset = batch_id * param.num_rows * hidden_size_pack2;
    const __nv_bfloat162* input_ptr = input_base + batch_offset;
    const __nv_bfloat162* residual_ptr = residual_base + batch_offset;
    __nv_bfloat162* residual_out_ptr = residual_out_base + batch_offset;
    __nv_bfloat162* normalized_output_ptr = normalized_output_base + batch_offset;

    float2 input_val[elem_per_thread];
    float2 residual_val[elem_per_thread];

    int index = row_id * hidden_size_pack2 + i;
    uint4 input_tmp = reinterpret_cast<uint4 const*>(&input_ptr[index])[0];
    uint32_t input_components[4] = {input_tmp.x, input_tmp.y, input_tmp.z, input_tmp.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      input_val[k] =
          __bfloat1622float2(reinterpret_cast<__nv_bfloat162 const&>(input_components[k]));
    }

    uint4 residual_tmp =
        residual_base ? reinterpret_cast<uint4 const*>(&residual_ptr[index])[0] : uint4{0, 0, 0, 0};
    uint32_t residual_components[4] = {residual_tmp.x, residual_tmp.y, residual_tmp.z,
                                       residual_tmp.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      residual_val[k] =
          __bfloat1622float2(reinterpret_cast<__nv_bfloat162 const&>(residual_components[k]));
    }

    if constexpr (kUseGate) {
      uint4 tmp_gate = reinterpret_cast<uint4 const*>(&gate_base[gate_shift_scale_index])[0];
      uint32_t gate_components[4] = {tmp_gate.x, tmp_gate.y, tmp_gate.z, tmp_gate.w};
      float2 gate_val[elem_per_thread];
#pragma unroll
      for (int k = 0; k < 4; k++) {
        __nv_bfloat162 const& gate_component =
            reinterpret_cast<__nv_bfloat162 const&>(gate_components[k]);
        gate_val[k] = __bfloat1622float2(gate_component);
      }

      if constexpr (USE_INPUT_SF_SCALE) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          gate_val[k] = f2math::fadd2(gate_val[k], gate_bias[k]);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          gate_val[k] = f2math::fmul2(gate_val[k], input_sf_scale_val);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          input_val[k] = f2math::ffma2(input_val[k], gate_val[k], residual_val[k]);
        }
      } else {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          input_val[k] = f2math::ffma2(input_val[k], f2math::fadd2(gate_val[k], gate_bias[k]),
                                       residual_val[k]);
        }
      }
    } else {
      if constexpr (USE_INPUT_SF_SCALE) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          input_val[k] = f2math::ffma2(input_val[k], input_sf_scale_val, residual_val[k]);
        }
      } else {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          input_val[k] = f2math::fadd2(input_val[k], residual_val[k]);
        }
      }
    }

    __nv_bfloat162 bf16_residual[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bf16_residual[k] = __float22bfloat162_rn(input_val[k]);
    }
    uint4 packed_residual_output;
    packed_residual_output.x = *reinterpret_cast<uint32_t*>(&bf16_residual[0]);
    packed_residual_output.y = *reinterpret_cast<uint32_t*>(&bf16_residual[1]);
    packed_residual_output.z = *reinterpret_cast<uint32_t*>(&bf16_residual[2]);
    packed_residual_output.w = *reinterpret_cast<uint32_t*>(&bf16_residual[3]);
    reinterpret_cast<uint4*>(&residual_out_ptr[index])[0] = packed_residual_output;

    float2 r_inv_std;
    float2 r_mean;

    float sum = 0.0f;
    float sum_sq = 0.0f;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 val = input_val[k];
      sum = f2math::fadd(f2math::fadd(sum, val.x), val.y);
      sum_sq = f2math::fmaf_rn(val.y, val.y, f2math::fmaf_rn(val.x, val.x, sum_sq));
    }

    float2 thread_sums = make_float2(sum, sum_sq);
    float2 block_sums = BlockReduce(reduceStore).Reduce(thread_sums, Float2Sum());

    if (threadIdx.x == 0) {
      constexpr float inv_hidden_size = 1.0f / hidden_size;
      float mean_val = block_sums.x * inv_hidden_size;
      float mean_sq = block_sums.y * inv_hidden_size;
      float variance = f2math::fmaf_rn(-mean_val, mean_val, mean_sq);
      float inv_std = f2math::frsqrt(f2math::fadd(variance, param.epsilon));
      s_mean = make_float2(mean_val, mean_val);
      s_inv_std = make_float2(inv_std, inv_std);
    }
    __syncthreads();

    r_mean = s_mean;
    r_inv_std = s_inv_std;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      input_val[k] = f2math::ffma2(make_float2(-1.0f, -1.0f), r_mean, input_val[k]);
    }

    if constexpr (kUseGammaBeta) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 scaled_inv = f2math::fmul2(r_inv_std, gamma[k]);
        input_val[k] = f2math::ffma2(input_val[k], scaled_inv, beta[k]);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        input_val[k] = f2math::fmul2(input_val[k], r_inv_std);
      }
    }

    if constexpr (kUseScaleShift) {
      uint4 tmp_scale = reinterpret_cast<uint4 const*>(&scale_base[gate_shift_scale_index])[0];
      uint32_t scale_components[4] = {tmp_scale.x, tmp_scale.y, tmp_scale.z, tmp_scale.w};
      float2 scale_val[elem_per_thread];
#pragma unroll
      for (int k = 0; k < 4; k++) {
        __nv_bfloat162 const& scale_component =
            reinterpret_cast<__nv_bfloat162 const&>(scale_components[k]);
        scale_val[k] = __bfloat1622float2(scale_component);
      }

      uint4 tmp_shift = reinterpret_cast<uint4 const*>(&shift_base[gate_shift_scale_index])[0];
      uint32_t shift_components[4] = {tmp_shift.x, tmp_shift.y, tmp_shift.z, tmp_shift.w};
      float2 shift_val[elem_per_thread];
#pragma unroll
      for (int k = 0; k < 4; k++) {
        __nv_bfloat162 const& shift_component =
            reinterpret_cast<__nv_bfloat162 const&>(shift_components[k]);
        shift_val[k] = __bfloat1622float2(shift_component);
      }

#pragma unroll
      for (int k = 0; k < 4; k++) {
        input_val[k] = f2math::ffma2(
            input_val[k],
            f2math::fadd2(make_float2(1.0f, 1.0f), f2math::fadd2(scale_val[k], scale_bias[k])),
            f2math::fadd2(shift_val[k], shift_bias[k]));
      }
    }

    __nv_bfloat162 bf16_vals[elem_per_thread];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bf16_vals[k] = __float22bfloat162_rn(input_val[k]);
    }

    if constexpr (output_format == OutputFormat::BF16) {
      uint4 packed_output;
      packed_output.x = *reinterpret_cast<uint32_t*>(&bf16_vals[0]);
      packed_output.y = *reinterpret_cast<uint32_t*>(&bf16_vals[1]);
      packed_output.z = *reinterpret_cast<uint32_t*>(&bf16_vals[2]);
      packed_output.w = *reinterpret_cast<uint32_t*>(&bf16_vals[3]);
      reinterpret_cast<uint4*>(&normalized_output_ptr[index])[0] = packed_output;
    } else {
      typename PackType<InputType, 8>::type normed_output;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        normed_output.array[k * 2 + 0] = bf16_vals[k].x;
        normed_output.array[k * 2 + 1] = bf16_vals[k].y;
      }
      int n_base = threadIdx.x * 8;
      using ConverterType = std::conditional_t<output_format == OutputFormat::MXFP8,
                                               FP8Converter<InputType>, FP4Converter<InputType>>;
      ConverterType converter(param);
      converter.template post_process<8, decltype(normed_output)>(batch_id, row_id, n_base,
                                                                  normed_output);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 7: Host launcher functions
////////////////////////////////////////////////////////////////////////////////////////////////////

template <FusedLayerNormMode mode, OutputFormat output_format>
void layernorm_mode_impl(__nv_bfloat16* d_output, __nv_bfloat16* d_norm_output,
                         __nv_bfloat16* d_input, const float* d_gamma, const float* d_beta,
                         const __nv_bfloat16* d_gate, const float* d_gate_bias,
                         const __nv_bfloat16* d_residual, const __nv_bfloat16* d_scale,
                         const float* d_scale_bias, const __nv_bfloat16* d_shift,
                         const float* d_shift_bias, const float epsilon, const int batch_size,
                         const int num_rows, const int hidden_size, cudaStream_t stream,
                         uint32_t* d_sf_out, float* d_sf_scale, float* d_input_sf_scale) {
  constexpr int block_size = 384;
  static_assert((3072 / 2) % block_size == 0, "block_size must divide hidden_size/2");

  FwdParam<InputType> param(d_output, reinterpret_cast<uint32_t*>(d_norm_output), d_input,
                            d_residual, batch_size, num_rows, epsilon, d_gamma, d_beta, d_gate,
                            d_gate_bias, d_scale, d_scale_bias, d_shift, d_shift_bias, hidden_size,
                            d_sf_scale, d_input_sf_scale, d_sf_out);
  if (d_input_sf_scale) {
    meta_fused_layernorm<InputType, block_size, 3072, mode, output_format, true>
        <<<num_rows, block_size, 0, stream>>>(param);
  } else {
    meta_fused_layernorm<InputType, block_size, 3072, mode, output_format, false>
        <<<num_rows, block_size, 0, stream>>>(param);
  }
}

template <FusedLayerNormMode mode>
void layernorm_mode(__nv_bfloat16* d_output, __nv_bfloat16* d_norm_output, __nv_bfloat16* d_input,
                    const float* d_gamma, const float* d_beta, const __nv_bfloat16* d_gate,
                    const float* d_gate_bias, const __nv_bfloat16* d_residual,
                    const __nv_bfloat16* d_scale, const float* d_scale_bias,
                    const __nv_bfloat16* d_shift, const float* d_shift_bias, const float epsilon,
                    const int batch_size, const int num_rows, const int hidden_size,
                    cudaStream_t stream, OutputFormat output_format, uint32_t* d_sf_out,
                    float* d_sf_scale, float* d_input_sf_scale) {
  switch (output_format) {
    case OutputFormat::BF16:
      layernorm_mode_impl<mode, OutputFormat::BF16>(
          d_output, d_norm_output, d_input, d_gamma, d_beta, d_gate, d_gate_bias, d_residual,
          d_scale, d_scale_bias, d_shift, d_shift_bias, epsilon, batch_size, num_rows, hidden_size,
          stream, nullptr, nullptr, d_input_sf_scale);
      break;
    case OutputFormat::NVFP4:
      layernorm_mode_impl<mode, OutputFormat::NVFP4>(
          d_output, d_norm_output, d_input, d_gamma, d_beta, d_gate, d_gate_bias, d_residual,
          d_scale, d_scale_bias, d_shift, d_shift_bias, epsilon, batch_size, num_rows, hidden_size,
          stream, d_sf_out, d_sf_scale, d_input_sf_scale);
      break;
    case OutputFormat::MXFP8:
      layernorm_mode_impl<mode, OutputFormat::MXFP8>(
          d_output, d_norm_output, d_input, d_gamma, d_beta, d_gate, d_gate_bias, d_residual,
          d_scale, d_scale_bias, d_shift, d_shift_bias, epsilon, batch_size, num_rows, hidden_size,
          stream, d_sf_out, d_sf_scale, d_input_sf_scale);
      break;
  }
}

inline void launchFusedLayerNorm(__nv_bfloat16* d_output, __nv_bfloat16* d_norm_output,
                                 __nv_bfloat16* d_input, const float* d_gamma, const float* d_beta,
                                 const __nv_bfloat16* d_gate, const float* d_gate_bias,
                                 const __nv_bfloat16* d_residual, const __nv_bfloat16* d_scale,
                                 const float* d_scale_bias, const __nv_bfloat16* d_shift,
                                 const float* d_shift_bias, const float epsilon,
                                 const int batch_size, const int num_rows, const int hidden_size,
                                 cudaStream_t stream, FusedLayerNormMode mode,
                                 OutputFormat output_format, uint32_t* d_sf_out = nullptr,
                                 float* d_sf_scale = nullptr, float* d_input_sf_scale = nullptr) {
  FLASHINFER_FUSED_LN_CHECK(hidden_size == 3072);

  switch (mode) {
    case FusedLayerNormMode::RESIDUAL_SCALE_SHIFT:
      layernorm_mode<FusedLayerNormMode::RESIDUAL_SCALE_SHIFT>(
          d_output, d_norm_output, d_input, d_gamma, d_beta, d_gate, d_gate_bias, d_residual,
          d_scale, d_scale_bias, d_shift, d_shift_bias, epsilon, batch_size, num_rows, hidden_size,
          stream, output_format, d_sf_out, d_sf_scale, d_input_sf_scale);
      break;
    case FusedLayerNormMode::GATE_RESIDUAL_SCALE_SHIFT:
      layernorm_mode<FusedLayerNormMode::GATE_RESIDUAL_SCALE_SHIFT>(
          d_output, d_norm_output, d_input, d_gamma, d_beta, d_gate, d_gate_bias, d_residual,
          d_scale, d_scale_bias, d_shift, d_shift_bias, epsilon, batch_size, num_rows, hidden_size,
          stream, output_format, d_sf_out, d_sf_scale, d_input_sf_scale);
      break;
    case FusedLayerNormMode::GATE_RESIDUAL_GAMMA_BETA:
    default:
      layernorm_mode<FusedLayerNormMode::GATE_RESIDUAL_GAMMA_BETA>(
          d_output, d_norm_output, d_input, d_gamma, d_beta, d_gate, d_gate_bias, d_residual,
          d_scale, d_scale_bias, d_shift, d_shift_bias, epsilon, batch_size, num_rows, hidden_size,
          stream, output_format, d_sf_out, d_sf_scale, d_input_sf_scale);
      break;
  }
}

}  // namespace fused_layernorm
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_DIT_LAYERNORM_CUH_
