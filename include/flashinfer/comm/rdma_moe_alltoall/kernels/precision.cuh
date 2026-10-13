/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

#include <tuple>

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"

namespace deep_ep {
namespace extensions {
// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

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
  EP_DEVICE_ASSERT(false);
  return 0;
#endif
}

// Convert 8 e2m1 values (represented as one uint32_t) into 8 float32 values.
inline __device__ void e2m1_to_fp32_vec(uint32_t e2m1Vec, float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t out_fp16[4];
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2   %0, byte0;\n"
      "cvt.rn.f16x2.e2m1x2   %1, byte1;\n"
      "cvt.rn.f16x2.e2m1x2   %2, byte2;\n"
      "cvt.rn.f16x2.e2m1x2   %3, byte3;\n"
      "}"
      : "=r"(out_fp16[0]), "=r"(out_fp16[1]), "=r"(out_fp16[2]), "=r"(out_fp16[3])
      : "r"(e2m1Vec));

  // Convert FP16x2 values to float2 values using vectorized conversion
  float2 res0 = __half22float2(reinterpret_cast<__half2&>(out_fp16[0]));
  float2 res1 = __half22float2(reinterpret_cast<__half2&>(out_fp16[1]));
  float2 res2 = __half22float2(reinterpret_cast<__half2&>(out_fp16[2]));
  float2 res3 = __half22float2(reinterpret_cast<__half2&>(out_fp16[3]));

  array[0] = res0.x;
  array[1] = res0.y;
  array[2] = res1.x;
  array[3] = res1.y;
  array[4] = res2.x;
  array[5] = res2.y;
  array[6] = res3.x;
  array[7] = res3.y;
#else
  // Fallback for older architectures
  static float const kE2M1ToFloatArray[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  for (int i = 0; i < 8; i++) {
    uint8_t e2m1Val = (e2m1Vec >> (i * 4)) & 0xF;
    bool signBit = e2m1Val & 8;
    auto absValue = e2m1Val & 7;
    float result = kE2M1ToFloatArray[absValue];
    if (signBit) result = -result;
    array[i] = result;
  }
#endif
}

__device__ std::tuple<uint32_t, uint8_t> quantize_bf16_to_nvfp4(const int4& packed_input_bf16,
                                                                const float gloabl_scale) {
  const auto bf16x2_vals = reinterpret_cast<const __nv_bfloat162*>(&packed_input_bf16);
  auto local_max = __habs2(bf16x2_vals[0]);
  for (int j = 1; j < 4; ++j) {
    local_max = __hmax2(local_max, __habs2(bf16x2_vals[j]));
  }
  local_max = __hmax2(local_max, __shfl_xor_sync(~0, local_max, 1));
  float max_val = static_cast<float>(__hmax(local_max.x, local_max.y));
  float scale_val = gloabl_scale * (max_val * reciprocal_approximate_ftz(6.0f));
  float scale_val_narrow;
  uint8_t scale_val_fp8;
  __nv_fp8_e4m3 tmp = static_cast<__nv_fp8_e4m3>(scale_val);
  scale_val_narrow = static_cast<float>(tmp);
  scale_val_fp8 = tmp.__x;
  float output_scale =
      scale_val != 0
          ? reciprocal_approximate_ftz(scale_val_narrow * reciprocal_approximate_ftz(gloabl_scale))
          : 0.f;
  float input_vec_fp32[8];
  for (int j = 0; j < 8; ++j) {
    input_vec_fp32[j] =
        static_cast<float>(reinterpret_cast<const nv_bfloat16*>(&packed_input_bf16)[j]) *
        output_scale;
  }
  uint32_t e2m1_vec = fp32_vec_to_e2m1(input_vec_fp32);
  return {e2m1_vec, scale_val_fp8};
}

template <bool ApplyWeight>
__device__ void dequantize_nvfp4_to_bf16(const uint32_t& packed_input_nvfp4,
                                         const float global_scale, const uint8_t scale_val_fp8,
                                         float (&output_fp32)[8], const float weight = 1.f) {
  float global_scale_reciprocal = 1.f / global_scale;
  if (__isinf(global_scale_reciprocal)) {
    global_scale_reciprocal = 1.f;
  }
  __nv_fp8_e4m3 tmp;
  tmp.__x = scale_val_fp8;
  float scale_val_fp32 = static_cast<float>(tmp);
  scale_val_fp32 *= global_scale_reciprocal;
  e2m1_to_fp32_vec(packed_input_nvfp4, output_fp32);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if constexpr (ApplyWeight) {
      output_fp32[j] = output_fp32[j] * scale_val_fp32 * weight;
    } else {
      output_fp32[j] = output_fp32[j] * scale_val_fp32;
    }
  }
}

__device__ std::tuple<int2, float> quantize_bf16_to_fp8(const int4& packed_input_bf16) {
  constexpr int kNumElemsPerRead = 8;
  auto bf16_values = reinterpret_cast<const nv_bfloat16*>(&packed_input_bf16);
  float fp32_values[kNumElemsPerRead];
  float amax = kFP8Margin, scale, scale_inv;
#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; ++j) {
    fp32_values[j] = static_cast<float>(bf16_values[j]);
    amax = fmaxf(amax, fabsf(fp32_values[j]));
  }
  amax = half_warp_reduce_max(amax);
  calculate_fp8_scales(amax, scale, scale_inv, false);
  int2 int2_value;
  auto fp8_values = reinterpret_cast<__nv_fp8_e4m3*>(&int2_value);
#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; ++j) {
    fp8_values[j] = static_cast<__nv_fp8_e4m3>(fp32_values[j] * scale);
  }
  return {int2_value, scale_inv};
}

template <bool ApplyWeight>
__device__ void dequantize_fp8_to_bf16(const int2& packed_input_fp8, const float scale_val,
                                       float (&output_fp32)[8], const float weight = 1.f) {
  constexpr int kNumElemsPerRead = 8;
  auto scale = extract_required_scale_format<false>(scale_val);
  auto fp8_values = reinterpret_cast<const __nv_fp8_e4m3*>(&packed_input_fp8);
#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; ++j) {
    if constexpr (ApplyWeight) {
      output_fp32[j] = static_cast<float>(fp8_values[j]) * scale * weight;
    } else {
      output_fp32[j] = static_cast<float>(fp8_values[j]) * scale;
    }
  }
}

enum class LowPrecisionType : int { FP8 = 0, NVFP4 = 1, BF16 = 2 };

template <LowPrecisionType kPrecision>
struct LowPrecisionTypeTraits {};
template <>
struct LowPrecisionTypeTraits<LowPrecisionType::NVFP4> {
  using ScaleType = uint8_t;
  using CommVecType = uint32_t;
  static constexpr int kNumElemsPerUnit = 2;
  static constexpr int kUnitBytes = 1;
  static constexpr int kNumElemsPerScale = 16;
  static constexpr int kBytesPerScale = sizeof(ScaleType);
  static constexpr bool kUseGlobalScale = true;

  __device__ static std::tuple<CommVecType, ScaleType> quantize(const int4& int4_value,
                                                                const float global_scale_val) {
    return quantize_bf16_to_nvfp4(int4_value, global_scale_val);
  }
  __device__ static void dequantize(const CommVecType& comm_vec, const float global_scale_val,
                                    const ScaleType scale_vec, float (&output_fp32)[8],
                                    const float weight = 1.f) {
    dequantize_nvfp4_to_bf16<true>(comm_vec, global_scale_val, scale_vec, output_fp32, weight);
  }
};
template <>
struct LowPrecisionTypeTraits<LowPrecisionType::FP8> {
  using ScaleType = float;
  using CommVecType = int2;
  static constexpr int kNumElemsPerUnit = 1;
  static constexpr int kUnitBytes = 1;
  static constexpr int kNumElemsPerScale = 128;
  static constexpr int kBytesPerScale = sizeof(ScaleType);
  static constexpr bool kUseGlobalScale = false;
  __device__ static std::tuple<CommVecType, ScaleType> quantize(const int4& int4_value,
                                                                const float global_scale_val) {
    return quantize_bf16_to_fp8(int4_value);
  }
  __device__ static void dequantize(const CommVecType& comm_vec, const float global_scale_val,
                                    const ScaleType scale_vec, float (&output_fp32)[8],
                                    const float weight = 1.f) {
    dequantize_fp8_to_bf16<true>(comm_vec, scale_vec, output_fp32, weight);
  }
};

template <>
struct LowPrecisionTypeTraits<LowPrecisionType::BF16> {
  using ScaleType = float;  // Placeholder only; BF16 messages have no scales.
  using CommVecType = int4;
  static constexpr int kNumElemsPerUnit = 1;
  static constexpr int kUnitBytes = sizeof(nv_bfloat16);
  static constexpr int kNumElemsPerScale = 8;  // Keeps index divisors nonzero.
  static constexpr int kBytesPerScale = 0;
  static constexpr bool kUseGlobalScale = false;

  __device__ static std::tuple<CommVecType, ScaleType> quantize(const int4& value, const float) {
    return {value, 0.f};
  }
  __device__ static void dequantize(const CommVecType& value, const float, const ScaleType,
                                    float (&output_fp32)[8], const float weight = 1.f) {
    const auto* values = reinterpret_cast<const nv_bfloat16*>(&value);
#pragma unroll
    for (int j = 0; j < 8; ++j) output_fp32[j] = static_cast<float>(values[j]) * weight;
  }
};

}  // namespace extensions
}  // namespace deep_ep
