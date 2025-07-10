// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once
#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

#include <float.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <math.h>

#include <type_traits>

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

__host__ __device__ inline __hip_bfloat162 __float2bfloat162_rn(const float a)
{
    return __hip_bfloat162{__float2bfloat16(a), __float2bfloat16(a)};
}

FLASHINFER_INLINE __hip_bfloat162 make_bfloat162(const __hip_bfloat16 x,
                                                 const __hip_bfloat16 y)
{
    __hip_bfloat162 t;
    t.x = x;
    t.y = y;
    return t;
}

namespace flashinfer
{

#define FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 < 120400) &&    \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
// CUDA version < 12.4 and GPU architecture < 80

FLASHINFER_INLINE __hip_bfloat16 __hmul(const __hip_bfloat16 a,
                                        const __hip_bfloat16 b)
{
    __hip_bfloat16 val;
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    // avoid ftz in device code
    val = __float2bfloat16(__fmaf_ieee_rn(fa, fb, -0.0f));
    return val;
}

FLASHINFER_INLINE __hip_bfloat162 __hmul2(const __hip_bfloat162 a,
                                          const __hip_bfloat162 b)
{
    __hip_bfloat162 val;
    val.x = __hmul(a.x, b.x);
    val.y = __hmul(a.y, b.y);
    return val;
}

FLASHINFER_INLINE __hip_bfloat162 __floats2bfloat162_rn(const float a,
                                                        const float b)
{
    __hip_bfloat162 val;
    val = __hip_bfloat162(__float2bfloat16(a), __float2bfloat16(b));
    return val;
}

FLASHINFER_INLINE __hip_bfloat162 __float22bfloat162_rn(const float2 a)
{
    __hip_bfloat162 val = __float22bfloat162_rn(a.x, a.y);
    return val;
}
FLASHINFER_INLINE float2 __bfloat1622float2(const __hip_bfloat162 a)
{
    float hi_float;
    float lo_float;
    // lo_float = __internal_bfloat162float(((__gpu_bfloat162_raw)a).x);
    // hi_float = __internal_bfloat162float(((__gpu_bfloat162_raw)a).y);
    lo_float = __bfloat1622float2(a.x);
    hi_float = __bfloat1622float2(a.y);
    return make_float2(lo_float, hi_float);
}
#endif

/******************* vec_t type cast *******************/

template <typename dst_t, typename src_t> struct vec_cast
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(dst_t *dst, const src_t *src)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            dst[i] = (dst_t)src[i];
        }
    }
};

template <> struct vec_cast<float, half>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(float *dst, const half *src)
    {
        if constexpr (vec_size == 1) {
            // dst[0] = (float)src[0];
            dst[0] = __half2float(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                ((float2 *)dst)[i] = __half22float2(((half2 *)src)[i]);
            }
        }
    }
};

template <> struct vec_cast<half, float>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(half *dst, const float *src)
    {
        if constexpr (vec_size == 1) {
            dst[0] = __float2half(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                ((half2 *)dst)[i] = __float22half2_rn(((float2 *)src)[i]);
            }
        }
    }
};

template <typename T> constexpr FLASHINFER_INLINE int get_exponent_bits()
{
    if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz>) {
        return 4;
    }
    else if constexpr (std::is_same_v<T, __hip_fp8_e5m2_fnuz>) {
        return 5;
    }
    else if constexpr (std::is_same_v<T, half>) {
        return 5;
    }
    else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        return 8;
    }
}

template <typename T> constexpr FLASHINFER_INLINE int get_mantissa_bits()
{
    if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz>) {
        return 3;
    }
    else if constexpr (std::is_same_v<T, __hip_fp8_e5m2_fnuz>) {
        return 2;
    }
    else if constexpr (std::is_same_v<T, half>) {
        return 11;
    }
    else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        return 7;
    }
}

/*!
 * \brief Fallback to software fast dequant implementation if hardware
 * dequantization is not available.
 * \note Inspired by Marlin's fast dequantization, but here we don't have to
 * permute weights order.
 * \ref
 * https://github.com/vllm-project/vllm/blob/6dffa4b0a6120159ef2fe44d695a46817aff65bc/csrc/quantization/fp8/fp8_marlin.cu#L120
 */
template <typename fp8_dtype, typename fp16_dtype>
__device__ void fast_dequant_f8f16x4(uint32_t *input, uint2 *output)
{
    uint32_t q = *input;
    if constexpr (std::is_same_v<fp8_dtype, __hip_fp8_e5m2_fnuz> &&
                  std::is_same_v<fp16_dtype, half>)
    {
        output->x = __byte_perm(0U, q, 0x5140);
        output->y = __byte_perm(0U, q, 0x7362);
    }
    else {
        constexpr int FP8_EXPONENT = get_exponent_bits<fp8_dtype>();
        constexpr int FP8_MANTISSA = get_mantissa_bits<fp8_dtype>();
        constexpr int FP16_EXPONENT = get_exponent_bits<fp16_dtype>();

        constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
        // Calculate MASK for extracting mantissa and exponent
        // XXX: duplicate defs of `MASK1` and `MASK2`,
        // in the HIP file "include/hip/amd_detail/amd_device_functions.h".
        constexpr int MASK1_orig = 0x80000000;
        constexpr int MASK2_orig = MASK1_orig >> (FP8_EXPONENT + FP8_MANTISSA);
        constexpr int MASK3 = MASK2_orig & 0x7fffffff;
        constexpr int MASK = MASK3 | (MASK3 >> 16);
        q = __byte_perm(q, q, 0x1302);

        // Extract and shift FP8 values to FP16 format
        uint32_t Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
        uint32_t Out2 =
            ((q << 8) & 0x80008000) | (((q << 8) & MASK) >> RIGHT_SHIFT);

        constexpr int BIAS_OFFSET =
            (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
        // Construct and apply exponent bias
        if constexpr (std::is_same_v<fp16_dtype, half>) {
            const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

            // Convert to half2 and apply bias
            *(half2 *)&(output->x) =
                __hmul2(*reinterpret_cast<const half2 *>(&Out1), bias_reg);
            *(half2 *)&(output->y) =
                __hmul2(*reinterpret_cast<const half2 *>(&Out2), bias_reg);
        }
        else {
            constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
            const __hip_bfloat162 bias_reg =
                __float2bfloat162_rn(*reinterpret_cast<const float *>(&BIAS));
            // Convert to bfloat162 and apply bias
            *(__hip_bfloat162 *)&(output->x) = __hmul2(
                *reinterpret_cast<const __hip_bfloat162 *>(&Out1), bias_reg);
            *(__hip_bfloat162 *)&(output->y) = __hmul2(
                *reinterpret_cast<const __hip_bfloat162 *>(&Out2), bias_reg);
        }
    }
}

template <> struct vec_cast<__hip_bfloat16, __hip_fp8_e4m3_fnuz>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(__hip_bfloat16 *dst,
                                       const __hip_fp8_e4m3_fnuz *src)
    {
        if constexpr (vec_size == 1) {
            dst[0] = __hip_bfloat16(src[0]);
        }
        else if constexpr (vec_size == 2) {
            dst[0] = __hip_bfloat16(src[0]);
            dst[1] = __hip_bfloat16(src[1]);
        }
        else {
            static_assert(vec_size % 4 == 0,
                          "vec_size must be a multiple of 4");
#pragma unroll
            for (uint32_t i = 0; i < vec_size / 4; ++i) {
                fast_dequant_f8f16x4<__hip_fp8_e4m3_fnuz, __hip_bfloat16>(
                    (uint32_t *)&src[i * 4], (uint2 *)&dst[i * 4]);
            }
        }
    }
};

template <> struct vec_cast<__hip_bfloat16, __hip_fp8_e5m2_fnuz>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(__hip_bfloat16 *dst,
                                       const __hip_fp8_e5m2_fnuz *src)
    {
        if constexpr (vec_size == 1) {
            dst[0] = __hip_bfloat16(src[0]);
        }
        else if constexpr (vec_size == 2) {
            dst[0] = __hip_bfloat16(src[0]);
            dst[1] = __hip_bfloat16(src[1]);
        }
        else {
            static_assert(vec_size % 4 == 0,
                          "vec_size must be a multiple of 4");
#pragma unroll
            for (uint32_t i = 0; i < vec_size / 4; ++i) {
                fast_dequant_f8f16x4<__hip_fp8_e5m2_fnuz, __hip_bfloat16>(
                    (uint32_t *)&src[i * 4], (uint2 *)&dst[i * 4]);
            }
        }
    }
};

// Function to convert float to e4m3
__device__ uint8_t convert_f32_to_e4m3(float val)
{
    // Define the range of e4m3
    // 1. Minimum representable value for e4m3
    // 2. Binary 1000.000 in e4m3
    // 3. FLT_MIN is not suitable for e4m3 because e4m3 has a much smaller
    // dynamic range.
    float min_e4m3 = -8.0f;
    // 1. Maximum representable value for e4m3
    // 2. Binary 0111.111 in e4m3
    // FLT_MAX far exceeds the maximum value representable in e4m3.
    float max_e4m3 = 7.875f;

    // Saturate the value to the e4m3 range
    val = fminf(fmaxf(val, min_e4m3), max_e4m3);

    // Perform conversion
    // Decompose into mantissa and exponent
    int exp;
    float mantissa = frexpf(val, &exp);

    // Encode sign bit
    uint8_t sign = (mantissa < 0) ? 0x80 : 0x00;

    // Normalize mantissa and encode exponent
    mantissa =
        fabsf(mantissa) * 16.0f; // Scale mantissa for e4m3's 3-bit precision
    uint8_t exponent = static_cast<uint8_t>(exp + 7); // Bias of 7 for e4m3

    // Quantize mantissa
    // Apply round-to-nearest-even to the mantissa
    uint8_t quant_mantissa = static_cast<uint8_t>(roundf(mantissa)) & 0x07;

    // Combine into 8 bits: [sign][exponent][mantissa]
    return sign | (exponent << 3) | quant_mantissa;
}

__device__ __half2 convert_uint32_to_half2(uint32_t input)
{
    // Extract the low and high 16 bits
    uint16_t low_val = input & 0xFFFF;
    uint16_t high_val = (input >> 16) & 0xFFFF;
    // Convert to __half
    __half low_half = __float2half(static_cast<float>(low_val));
    __half high_half = __float2half(static_cast<float>(high_val));
    // Pack into __half2
    return __halves2half2(low_half, high_half);
}

// Convert f16x2 (__half2) to e4m3x2 (packed 16-bit)
__device__ uint16_t convert_f16x2_to_e4m3x2(__half2 x)
{
    float f32_0 = __half2float(__low2half(x));
    float f32_1 = __half2float(__high2half(x));
    uint8_t e4m3_0 = convert_f32_to_e4m3(f32_0);
    uint8_t e4m3_1 = convert_f32_to_e4m3(f32_1);
    return (static_cast<uint16_t>(e4m3_1) << 8) | e4m3_0;
}

template <> struct vec_cast<__hip_fp8_e4m3_fnuz, half>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(__hip_fp8_e4m3_fnuz *dst,
                                       const half *src)
    {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
        if constexpr (vec_size == 1) {
            dst[0] = __hip_fp8_e4m3_fnuz(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                uint16_t y;
                uint32_t x = *(uint32_t *)&src[i * 2];
                __half2 x_h2 = convert_uint32_to_half2(x);
                y = convert_f16x2_to_e4m3x2(x_h2);

                *(uint16_t *)&dst[i * 2] = y;
            }
        }
#else
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            dst[i] = __hip_fp8_e4m3_fnuz(src[i]);
        }
#endif // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    }
};

__device__ uint16_t convert_f16x2_to_e5m2x2(uint32_t x)
{
    // Unpack the two 16-bit half-precision floats from the input
    // Extract lower 16 bits
    __half h1 = __ushort_as_half(x & 0xFFFF);
    // Extract upper 16 bits
    __half h2 = __ushort_as_half((x >> 16) & 0xFFFF);

#if 0
  // Alternative with `__uint2half_rn`
  uint16_t val1 = x & 0xFFFF;  // Lower 16 bits
  uint16_t val2 = (x >> 16) & 0xFFFF; // Upper 16 bits
  __half h1 = __uint2half_rn(val1);
  __half h2 = __uint2half_rn(val2);
#endif

    // Define the range of e5m2
    // Minimum representable value for e5m2
    const float min_e5m2 = -8.0f;
    // Maximum representable value for e5m2
    const float max_e5m2 = 7.75f;

    // Helper lambda for conversion
    auto f32_to_e5m2 = [min_e5m2, max_e5m2](float val) -> uint8_t {
        // Saturate the val
        val = fminf(fmaxf(val, min_e5m2), max_e5m2);

        // Decompose into mantissa and exponent
        int exp;
        float mantissa = frexpf(val, &exp);

        // Encode sign bit
        uint8_t sign = (mantissa < 0) ? 0x10 : 0x00; // Sign in bit 4
        mantissa = fabsf(mantissa);

        // Normalize mantissa and encode exponent
        mantissa *= 4.0f; // Scale for 2-bit mantissa
        uint8_t exponent = static_cast<uint8_t>(exp + 7); // Apply bias for e5m2

        // Apply round-to-nearest-even
        uint8_t quant_mantissa = static_cast<uint8_t>(roundf(mantissa)) & 0x03;

        // Combine into 5 bits: [sign][exponent][mantissa]
        return sign | (exponent << 2) | quant_mantissa;
    };

    // Convert the two __half values to e5m2
    uint8_t e5m2_1 = f32_to_e5m2(__half2float(h1));
    uint8_t e5m2_2 = f32_to_e5m2(__half2float(h2));

    // Pack the two e5m2 values into a single 16-bit output
    return (e5m2_2 << 8) | e5m2_1;
}
#endif

template <> struct vec_cast<__hip_fp8_e5m2_fnuz, half>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(__hip_fp8_e5m2_fnuz *dst,
                                       const half *src)
    {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
        if constexpr (vec_size == 1) {
            dst[0] = __hip_fp8_e5m2_fnuz(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                uint16_t y;
                uint32_t x = *(uint32_t *)&src[i * 2];
                y = convert_f16x2_to_e5m2x2(x);
                *(uint16_t *)&dst[i * 2] = y;
            }
        }
#else
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            dst[i] = __hip_fp8_e5m2_fnuz(src[i]);
        }
#endif // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    }
};

__device__ uint32_t convert_e4m3x2_to_f16x2(uint16_t x)
{
    // Extract two e4m3 values from the 16-bit input
    uint8_t e4m3_1 = x & 0xFF;        // Lower 8 bits
    uint8_t e4m3_2 = (x >> 8) & 0xFF; // Upper 8 bits

    // Decode e4m3 to float
    auto e4m3_to_f32 = [](uint8_t e4m3) -> float {
        // Extract sign, exponent, and mantissa
        int sign = (e4m3 & 0x80) ? -1 : 1;
        int exponent = ((e4m3 >> 3) & 0x0F) - 7; // 4-bit exponent with bias 7
        int mantissa = e4m3 & 0x07;              // 3-bit mantissa

        // Handle special case: zero
        if (exponent == -7 && mantissa == 0) {
            return 0.0f;
        }

        // Convert to float
        float f32_val = sign * ldexpf(1.0f + mantissa / 8.0f, exponent);
        return f32_val;
    };

    float f1 = e4m3_to_f32(e4m3_1);
    float f2 = e4m3_to_f32(e4m3_2);

    // Convert float to IEEE f16
    __half h1 = __float2half_rn(f1);
    __half h2 = __float2half_rn(f2);

    // Pack the two f16 values into a single uint32_t
    uint32_t f16x2 = (__half_as_ushort(h2) << 16) | __half_as_ushort(h1);
    return f16x2;
}

template <> struct vec_cast<half, __hip_fp8_e4m3_fnuz>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(half *dst,
                                       const __hip_fp8_e4m3_fnuz *src)
    {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
        if constexpr (vec_size == 1) {
            dst[0] = half(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                uint32_t y;
                uint16_t x = *(uint16_t *)&src[i * 2];
                y = convert_e4m3x2_to_f16x2(x);

                *(uint32_t *)&dst[i * 2] = y;
            }
        }
#else
        if constexpr (vec_size == 1) {
            dst[0] = half(src[0]);
        }
        else if constexpr (vec_size == 2) {
            dst[0] = half(src[0]);
            dst[1] = half(src[1]);
        }
        else {
            static_assert(vec_size % 4 == 0,
                          "vec_size must be a multiple of 4");
#pragma unroll
            for (uint32_t i = 0; i < vec_size / 4; ++i) {
                fast_dequant_f8f16x4<__hip_fp8_e4m3_fnuz, half>(
                    (uint32_t *)&src[i * 4], (uint2 *)&dst[i * 4]);
            }
        }
#endif // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    }
};

__device__ uint32_t convert_e5m2x2_to_f16x2(uint16_t x)
{
    // Extract two e5m2 values from the 16-bit input
    uint8_t e5m2_1 = x & 0xFF;        // Lower 8 bits
    uint8_t e5m2_2 = (x >> 8) & 0xFF; // Upper 8 bits

    // Decode e5m2 to float
    auto e5m2_to_f32 = [](uint8_t e5m2) -> float {
        // Extract sign, exponent, and mantissa
        int sign = (e5m2 & 0x80) ? -1 : 1;        // Sign bit
        int exponent = ((e5m2 >> 2) & 0x1F) - 15; // 5-bit exponent with bias 15
        int mantissa = e5m2 & 0x03;               // 2-bit mantissa

        // Handle special case: zero
        if (exponent == -15 && mantissa == 0) {
            return 0.0f;
        }

        // Convert to float
        float value = sign * ldexpf(1.0f + mantissa / 4.0f, exponent);
        return value;
    };

    float f1 = e5m2_to_f32(e5m2_1);
    float f2 = e5m2_to_f32(e5m2_2);

    // Convert float to IEEE f16
    __half h1 = __float2half_rn(f1);
    __half h2 = __float2half_rn(f2);

    // Pack the two f16 values into a single uint32_t
    uint32_t f16x2 = (__half_as_ushort(h2) << 16) | __half_as_ushort(h1);
    return f16x2;
}

template <> struct vec_cast<half, __hip_fp8_e5m2_fnuz>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(half *dst,
                                       const __hip_fp8_e5m2_fnuz *src)
    {
#ifdef FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
        if constexpr (vec_size == 1) {
            dst[0] = half(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                uint32_t y;
                uint16_t x = *(uint16_t *)&src[i * 2];
                y = convert_e5m2x2_to_f16x2(x);
                *(uint32_t *)&dst[i * 2] = y;
            }
        }
#else
        if constexpr (vec_size == 1) {
            dst[0] = half(src[0]);
        }
        else if constexpr (vec_size == 2) {
            dst[0] = half(src[0]);
            dst[1] = half(src[1]);
        }
        else {
            static_assert(vec_size % 4 == 0,
                          "vec_size must be a multiple of 4");
#pragma unroll
            for (uint32_t i = 0; i < vec_size / 4; ++i) {
                fast_dequant_f8f16x4<__hip_fp8_e5m2_fnuz, half>(
                    (uint32_t *)&src[i * 4], (uint2 *)&dst[i * 4]);
            }
        }
#endif // FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
    }
};

template <> struct vec_cast<float, __hip_bfloat16>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(float *dst, const __hip_bfloat16 *src)
    {
        if constexpr (vec_size == 1) {
            dst[0] = (float)src[0];
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                ((float2 *)dst)[i] =
                    __bfloat1622float2(((__hip_bfloat162 *)src)[i]);
            }
        }
    }
};

template <> struct vec_cast<__hip_bfloat16, float>
{
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(__hip_bfloat16 *dst, const float *src)
    {
        if constexpr (vec_size == 1) {
            dst[0] = __hip_bfloat16(src[0]);
        }
        else {
#pragma unroll
            for (size_t i = 0; i < vec_size / 2; ++i) {
                ((__hip_bfloat162 *)dst)[i] =
                    __float22bfloat162_rn(((float2 *)src)[i]);
            }
        }
    }
};

template <typename float_t, size_t vec_size> struct vec_t
{
    FLASHINFER_INLINE float_t &operator[](size_t i);
    FLASHINFER_INLINE const float_t &operator[](size_t i) const;
    FLASHINFER_INLINE void fill(float_t val);
    FLASHINFER_INLINE void load(const float_t *ptr);
    FLASHINFER_INLINE void store(float_t *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src);
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr);
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const;
    FLASHINFER_INLINE static void memcpy(float_t *dst, const float_t *src);
    FLASHINFER_INLINE float_t *ptr();
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size> &dst,
                                      const vec_t<src_float_t, vec_size> &src)
{

    vec_cast<tgt_float_t, src_float_t>::template cast<vec_size>(
        dst.ptr(), const_cast<vec_t<src_float_t, vec_size> *>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size> &dst,
                                      const src_float_t *src_ptr)
{
    if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
        dst.load(src_ptr);
    }
    else {
        vec_t<src_float_t, vec_size> tmp;
        tmp.load(src_ptr);
        dst.cast_from(tmp);
    }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(tgt_float_t *dst_ptr,
                                       const vec_t<src_float_t, vec_size> &src)
{
    if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
        src.store(dst_ptr);
    }
    else {
        vec_t<tgt_float_t, vec_size> tmp;
        tmp.cast_from(src);
        tmp.store(dst_ptr);
    }
}

/******************* vec_t<__hip_fp8_e4m3_fnuz> *******************/

// __hip_fp8_e4m3_fnuz x 1
template <> struct vec_t<__hip_fp8_e4m3_fnuz, 1>
{
    __hip_fp8_e4m3_fnuz data;

    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e4m3_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e4m3_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e4m3_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e4m3_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e4m3_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 1> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e4m3_fnuz *dst,
                                         const __hip_fp8_e4m3_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 1>::fill(__hip_fp8_e4m3_fnuz val)
{
    data = val;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 1>::load(const __hip_fp8_e4m3_fnuz *ptr)
{
    data = *ptr;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 1>::store(__hip_fp8_e4m3_fnuz *ptr) const
{
    *ptr = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 1>::memcpy(__hip_fp8_e4m3_fnuz *dst,
                                      const __hip_fp8_e4m3_fnuz *src)
{
    *dst = *src;
}

// __hip_fp8_e4m3_fnuz x 2
template <> struct vec_t<__hip_fp8_e4m3_fnuz, 2>
{
    __hip_fp8x2_e4m3_fnuz data;

    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e4m3_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e4m3_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e4m3_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e4m3_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e4m3_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 2> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_fp8_e4m3_fnuz *dst,
                                         const __hip_fp8_e4m3_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 2>::fill(__hip_fp8_e4m3_fnuz val)
{
    data.__x =
        (__hip_fp8x2_storage_t(val.__x) << 8) | __hip_fp8x2_storage_t(val.__x);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 2>::load(const __hip_fp8_e4m3_fnuz *ptr)
{
    data = *((__hip_fp8x2_e4m3_fnuz *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 2>::store(__hip_fp8_e4m3_fnuz *ptr) const
{
    *((__hip_fp8x2_e4m3_fnuz *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 2>::memcpy(__hip_fp8_e4m3_fnuz *dst,
                                      const __hip_fp8_e4m3_fnuz *src)
{
    *((__hip_fp8x2_e4m3_fnuz *)dst) = *((__hip_fp8x2_e4m3_fnuz *)src);
}

// __hip_fp8_e4m3_fnuz x 4

template <> struct vec_t<__hip_fp8_e4m3_fnuz, 4>
{
    __hip_fp8x4_e4m3_fnuz data;

    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e4m3_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e4m3_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e4m3_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e4m3_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e4m3_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 4> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e4m3_fnuz *dst,
                                         const __hip_fp8_e4m3_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 4>::fill(__hip_fp8_e4m3_fnuz val)
{
    data.__x = (__hip_fp8x4_storage_t(val.__x) << 24) |
               (__hip_fp8x4_storage_t(val.__x) << 16) |
               (__hip_fp8x4_storage_t(val.__x) << 8) |
               __hip_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 4>::load(const __hip_fp8_e4m3_fnuz *ptr)
{
    data = *((__hip_fp8x4_e4m3_fnuz *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 4>::store(__hip_fp8_e4m3_fnuz *ptr) const
{
    *((__hip_fp8x4_e4m3_fnuz *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 4>::memcpy(__hip_fp8_e4m3_fnuz *dst,
                                      const __hip_fp8_e4m3_fnuz *src)
{
    *((__hip_fp8x4_e4m3_fnuz *)dst) = *((__hip_fp8x4_e4m3_fnuz *)src);
}

// __hip_fp8_e4m3_fnuz x 8

template <> struct vec_t<__hip_fp8_e4m3_fnuz, 8>
{
    uint2 data;

    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e4m3_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e4m3_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e4m3_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e4m3_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e4m3_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e4m3_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 8> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e4m3_fnuz *dst,
                                         const __hip_fp8_e4m3_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 8>::fill(__hip_fp8_e4m3_fnuz val)
{
    ((__hip_fp8x4_e4m3_fnuz *)(&data.x))->__x =
        (__hip_fp8x4_storage_t(val.__x) << 24) |
        (__hip_fp8x4_storage_t(val.__x) << 16) |
        (__hip_fp8x4_storage_t(val.__x) << 8) | __hip_fp8x4_storage_t(val.__x);
    ((__hip_fp8x4_e4m3_fnuz *)(&data.y))->__x =
        (__hip_fp8x4_storage_t(val.__x) << 24) |
        (__hip_fp8x4_storage_t(val.__x) << 16) |
        (__hip_fp8x4_storage_t(val.__x) << 8) | __hip_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 8>::load(const __hip_fp8_e4m3_fnuz *ptr)
{
    data = *((uint2 *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 8>::store(__hip_fp8_e4m3_fnuz *ptr) const
{
    *((uint2 *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e4m3_fnuz, 8>::memcpy(__hip_fp8_e4m3_fnuz *dst,
                                      const __hip_fp8_e4m3_fnuz *src)
{
    *((uint2 *)dst) = *((uint2 *)src);
}

// __hip_fp8_e4m3_fnuz x 16 or more
template <size_t vec_size> struct vec_t<__hip_fp8_e4m3_fnuz, vec_size>
{
    uint4 data[vec_size / 16];

    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e4m3_fnuz *)data)[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e4m3_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e4m3_fnuz *)data)[i];
    }
    FLASHINFER_INLINE __hip_fp8_e4m3_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e4m3_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e4m3_fnuz val)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            ((__hip_fp8x4_e4m3_fnuz *)(&(data[i].x)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
            ((__hip_fp8x4_e4m3_fnuz *)(&(data[i].y)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
            ((__hip_fp8x4_e4m3_fnuz *)(&(data[i].z)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
            ((__hip_fp8x4_e4m3_fnuz *)(&(data[i].w)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
        }
    }
    FLASHINFER_INLINE void load(const __hip_fp8_e4m3_fnuz *ptr)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            data[i] = ((uint4 *)ptr)[i];
        }
    }
    FLASHINFER_INLINE void store(__hip_fp8_e4m3_fnuz *ptr) const
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            ((uint4 *)ptr)[i] = data[i];
        }
    }
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e4m3_fnuz *dst,
                                         const __hip_fp8_e4m3_fnuz *src)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            ((uint4 *)dst)[i] = ((uint4 *)src)[i];
        }
    }
};

/******************* vec_t<__hip_fp8_e5m2_fnuz> *******************/

// __hip_fp8_e5m2_fnuz x 1
template <> struct vec_t<__hip_fp8_e5m2_fnuz, 1>
{
    __hip_fp8_e5m2_fnuz data;

    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e5m2_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e5m2_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e5m2_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e5m2_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e5m2_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 1> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e5m2_fnuz *dst,
                                         const __hip_fp8_e5m2_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 1>::fill(__hip_fp8_e5m2_fnuz val)
{
    data = val;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 1>::load(const __hip_fp8_e5m2_fnuz *ptr)
{
    data = *ptr;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 1>::store(__hip_fp8_e5m2_fnuz *ptr) const
{
    *ptr = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 1>::memcpy(__hip_fp8_e5m2_fnuz *dst,
                                      const __hip_fp8_e5m2_fnuz *src)
{
    *dst = *src;
}

// __hip_fp8_e5m2_fnuz x 2
template <> struct vec_t<__hip_fp8_e5m2_fnuz, 2>
{
    __hip_fp8x2_e5m2_fnuz data;

    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e5m2_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e5m2_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e5m2_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e5m2_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e5m2_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 2> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e5m2_fnuz *dst,
                                         const __hip_fp8_e5m2_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 2>::fill(__hip_fp8_e5m2_fnuz val)
{
    data.__x =
        (__hip_fp8x2_storage_t(val.__x) << 8) | __hip_fp8x2_storage_t(val.__x);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 2>::load(const __hip_fp8_e5m2_fnuz *ptr)
{
    data = *((__hip_fp8x2_e5m2_fnuz *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 2>::store(__hip_fp8_e5m2_fnuz *ptr) const
{
    *((__hip_fp8x2_e5m2_fnuz *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 2>::memcpy(__hip_fp8_e5m2_fnuz *dst,
                                      const __hip_fp8_e5m2_fnuz *src)
{
    *((__hip_fp8x2_e5m2_fnuz *)dst) = *((__hip_fp8x2_e5m2_fnuz *)src);
}

// __hip_fp8_e5m2_fnuz x 4

template <> struct vec_t<__hip_fp8_e5m2_fnuz, 4>
{
    __hip_fp8x4_e5m2_fnuz data;

    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e5m2_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e5m2_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e5m2_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e5m2_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e5m2_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 4> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(__hip_fp8_e5m2_fnuz *dst,
                                         const __hip_fp8_e5m2_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 4>::fill(__hip_fp8_e5m2_fnuz val)
{
    data.__x = (__hip_fp8x4_storage_t(val.__x) << 24) |
               (__hip_fp8x4_storage_t(val.__x) << 16) |
               (__hip_fp8x4_storage_t(val.__x) << 8) |
               __hip_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 4>::load(const __hip_fp8_e5m2_fnuz *ptr)
{
    data = *((__hip_fp8x4_e5m2_fnuz *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 4>::store(__hip_fp8_e5m2_fnuz *ptr) const
{
    *((__hip_fp8x4_e5m2_fnuz *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 4>::memcpy(__hip_fp8_e5m2_fnuz *dst,
                                      const __hip_fp8_e5m2_fnuz *src)
{
    *((__hip_fp8x4_e5m2_fnuz *)dst) = *((__hip_fp8x4_e5m2_fnuz *)src);
}

// __hip_fp8_e5m2_fnuz x 8

template <> struct vec_t<__hip_fp8_e5m2_fnuz, 8>
{
    uint2 data;

    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e5m2_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e5m2_fnuz *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e5m2_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e5m2_fnuz val);
    FLASHINFER_INLINE void load(const __hip_fp8_e5m2_fnuz *ptr);
    FLASHINFER_INLINE void store(__hip_fp8_e5m2_fnuz *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 8> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_fp8_e5m2_fnuz *dst,
                                         const __hip_fp8_e5m2_fnuz *src);
};

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 8>::fill(__hip_fp8_e5m2_fnuz val)
{
    ((__hip_fp8x4_e5m2_fnuz *)(&data.x))->__x =
        (__hip_fp8x4_storage_t(val.__x) << 24) |
        (__hip_fp8x4_storage_t(val.__x) << 16) |
        (__hip_fp8x4_storage_t(val.__x) << 8) | __hip_fp8x4_storage_t(val.__x);
    ((__hip_fp8x4_e5m2_fnuz *)(&data.y))->__x =
        (__hip_fp8x4_storage_t(val.__x) << 24) |
        (__hip_fp8x4_storage_t(val.__x) << 16) |
        (__hip_fp8x4_storage_t(val.__x) << 8) | __hip_fp8x4_storage_t(val.__x);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 8>::load(const __hip_fp8_e5m2_fnuz *ptr)
{
    data = *((uint2 *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 8>::store(__hip_fp8_e5m2_fnuz *ptr) const
{
    *((uint2 *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_fp8_e5m2_fnuz, 8>::memcpy(__hip_fp8_e5m2_fnuz *dst,
                                      const __hip_fp8_e5m2_fnuz *src)
{
    *((uint2 *)dst) = *((uint2 *)src);
}

// __hip_fp8_e5m2_fnuz x 16 or more

template <size_t vec_size> struct vec_t<__hip_fp8_e5m2_fnuz, vec_size>
{
    uint4 data[vec_size / 16];

    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz &operator[](size_t i)
    {
        return ((__hip_fp8_e5m2_fnuz *)data)[i];
    }
    FLASHINFER_INLINE const __hip_fp8_e5m2_fnuz &operator[](size_t i) const
    {
        return ((const __hip_fp8_e5m2_fnuz *)data)[i];
    }
    FLASHINFER_INLINE __hip_fp8_e5m2_fnuz *ptr()
    {
        return reinterpret_cast<__hip_fp8_e5m2_fnuz *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_fp8_e5m2_fnuz val)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            ((__hip_fp8x4_e5m2_fnuz *)(&(data[i].x)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
            ((__hip_fp8x4_e5m2_fnuz *)(&(data[i].y)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
            ((__hip_fp8x4_e5m2_fnuz *)(&(data[i].z)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
            ((__hip_fp8x4_e5m2_fnuz *)(&(data[i].w)))->__x =
                (__hip_fp8x4_storage_t(val.__x) << 24) |
                (__hip_fp8x4_storage_t(val.__x) << 16) |
                (__hip_fp8x4_storage_t(val.__x) << 8) |
                __hip_fp8x4_storage_t(val.__x);
        }
    }
    FLASHINFER_INLINE void load(const __hip_fp8_e5m2_fnuz *ptr)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            data[i] = ((uint4 *)ptr)[i];
        }
    }
    FLASHINFER_INLINE void store(__hip_fp8_e5m2_fnuz *ptr) const
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            ((uint4 *)ptr)[i] = data[i];
        }
    }
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_fp8_e5m2_fnuz *dst,
                                         const __hip_fp8_e5m2_fnuz *src)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 16; ++i) {
            ((uint4 *)dst)[i] = ((uint4 *)src)[i];
        }
    }
};

/******************* vec_t<half> *******************/

// half x 1
template <> struct vec_t<half, 1>
{
    half data;

    FLASHINFER_INLINE half &operator[](size_t i)
    {
        return ((half *)(&data))[i];
    }
    FLASHINFER_INLINE const half &operator[](size_t i) const
    {
        return ((const half *)(&data))[i];
    }
    FLASHINFER_INLINE half *ptr() { return reinterpret_cast<half *>(&data); }
    FLASHINFER_INLINE void fill(half val);
    FLASHINFER_INLINE void load(const half *ptr);
    FLASHINFER_INLINE void store(half *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 1> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(half *dst, const half *src);
};

FLASHINFER_INLINE void vec_t<half, 1>::fill(half val) { data = val; }

FLASHINFER_INLINE void vec_t<half, 1>::load(const half *ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<half, 1>::store(half *ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<half, 1>::memcpy(half *dst, const half *src)
{
    *dst = *src;
}

// half x 2
template <> struct vec_t<half, 2>
{
    half2 data;

    FLASHINFER_INLINE half &operator[](size_t i)
    {
        return ((half *)(&data))[i];
    }
    FLASHINFER_INLINE const half &operator[](size_t i) const
    {
        return ((const half *)(&data))[i];
    }
    FLASHINFER_INLINE half *ptr() { return reinterpret_cast<half *>(&data); }
    FLASHINFER_INLINE void fill(half val);
    FLASHINFER_INLINE void load(const half *ptr);
    FLASHINFER_INLINE void store(half *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 2> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }

    FLASHINFER_INLINE static void memcpy(half *dst, const half *src);
};

FLASHINFER_INLINE void vec_t<half, 2>::fill(half val)
{
    data = make_half2(val, val);
}

FLASHINFER_INLINE void vec_t<half, 2>::load(const half *ptr)
{
    data = *((half2 *)ptr);
}

FLASHINFER_INLINE void vec_t<half, 2>::store(half *ptr) const
{
    *((half2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<half, 2>::memcpy(half *dst, const half *src)
{
    *((half2 *)dst) = *((half2 *)src);
}

// half x 4

template <> struct vec_t<half, 4>
{
    uint2 data;

    FLASHINFER_INLINE half &operator[](size_t i)
    {
        return ((half *)(&data))[i];
    }
    FLASHINFER_INLINE const half &operator[](size_t i) const
    {
        return ((const half *)(&data))[i];
    }
    FLASHINFER_INLINE half *ptr() { return reinterpret_cast<half *>(&data); }
    FLASHINFER_INLINE void fill(half val);
    FLASHINFER_INLINE void load(const half *ptr);
    FLASHINFER_INLINE void store(half *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 4> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(half *dst, const half *src);
};

FLASHINFER_INLINE void vec_t<half, 4>::fill(half val)
{
    *(half2 *)(&data.x) = make_half2(val, val);
    *(half2 *)(&data.y) = make_half2(val, val);
}

FLASHINFER_INLINE void vec_t<half, 4>::load(const half *ptr)
{
    data = *((uint2 *)ptr);
}

FLASHINFER_INLINE void vec_t<half, 4>::store(half *ptr) const
{
    *((uint2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<half, 4>::memcpy(half *dst, const half *src)
{
    *((uint2 *)dst) = *((uint2 *)src);
}

// half x 8 or more

template <size_t vec_size> struct vec_t<half, vec_size>
{
    uint4 data[vec_size / 8];
    FLASHINFER_INLINE half &operator[](size_t i) { return ((half *)data)[i]; }
    FLASHINFER_INLINE const half &operator[](size_t i) const
    {
        return ((const half *)data)[i];
    }
    FLASHINFER_INLINE half *ptr() { return reinterpret_cast<half *>(&data); }
    FLASHINFER_INLINE void fill(half val)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            *(half2 *)(&(data[i].x)) = make_half2(val, val);
            *(half2 *)(&(data[i].y)) = make_half2(val, val);
            *(half2 *)(&(data[i].z)) = make_half2(val, val);
            *(half2 *)(&(data[i].w)) = make_half2(val, val);
        }
    }
    FLASHINFER_INLINE void load(const half *ptr)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            data[i] = ((uint4 *)ptr)[i];
        }
    }
    FLASHINFER_INLINE void store(half *ptr) const
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            ((uint4 *)ptr)[i] = data[i];
        }
    }
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(half *dst, const half *src)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            ((uint4 *)dst)[i] = ((uint4 *)src)[i];
        }
    }
};

/******************* vec_t<__hip_bfloat16> *******************/

// __hip_bfloat16 x 1
template <> struct vec_t<__hip_bfloat16, 1>
{
    __hip_bfloat16 data;
    FLASHINFER_INLINE __hip_bfloat16 &operator[](size_t i)
    {
        return ((__hip_bfloat16 *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_bfloat16 &operator[](size_t i) const
    {
        return ((const __hip_bfloat16 *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_bfloat16 *ptr()
    {
        return reinterpret_cast<__hip_bfloat16 *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_bfloat16 val);
    FLASHINFER_INLINE void load(const __hip_bfloat16 *ptr);
    FLASHINFER_INLINE void store(__hip_bfloat16 *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 1> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_bfloat16 *dst,
                                         const __hip_bfloat16 *src);
};

FLASHINFER_INLINE void vec_t<__hip_bfloat16, 1>::fill(__hip_bfloat16 val)
{
    data = val;
}

FLASHINFER_INLINE void vec_t<__hip_bfloat16, 1>::load(const __hip_bfloat16 *ptr)
{
    data = *ptr;
}

FLASHINFER_INLINE void
vec_t<__hip_bfloat16, 1>::store(__hip_bfloat16 *ptr) const
{
    *ptr = data;
}

FLASHINFER_INLINE void
vec_t<__hip_bfloat16, 1>::memcpy(__hip_bfloat16 *dst, const __hip_bfloat16 *src)
{
    *dst = *src;
}

// __hip_bfloat16 x 2
template <> struct vec_t<__hip_bfloat16, 2>
{
    __hip_bfloat162 data;

    FLASHINFER_INLINE __hip_bfloat16 &operator[](size_t i)
    {
        return ((__hip_bfloat16 *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_bfloat16 &operator[](size_t i) const
    {
        return ((const __hip_bfloat16 *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_bfloat16 *ptr()
    {
        return reinterpret_cast<__hip_bfloat16 *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_bfloat16 val);
    FLASHINFER_INLINE void load(const __hip_bfloat16 *ptr);
    FLASHINFER_INLINE void store(__hip_bfloat16 *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 2> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_bfloat16 *dst,
                                         const __hip_bfloat16 *src);
};

FLASHINFER_INLINE void vec_t<__hip_bfloat16, 2>::fill(__hip_bfloat16 val)
{
    data = make_bfloat162(val, val);
}

FLASHINFER_INLINE void vec_t<__hip_bfloat16, 2>::load(const __hip_bfloat16 *ptr)
{
    data = *((__hip_bfloat162 *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_bfloat16, 2>::store(__hip_bfloat16 *ptr) const
{
    *((__hip_bfloat162 *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_bfloat16, 2>::memcpy(__hip_bfloat16 *dst, const __hip_bfloat16 *src)
{
    *((__hip_bfloat162 *)dst) = *((__hip_bfloat162 *)src);
}

// __hip_bfloat16 x 4

template <> struct vec_t<__hip_bfloat16, 4>
{
    uint2 data;

    FLASHINFER_INLINE __hip_bfloat16 &operator[](size_t i)
    {
        return ((__hip_bfloat16 *)(&data))[i];
    }
    FLASHINFER_INLINE const __hip_bfloat16 &operator[](size_t i) const
    {
        return ((const __hip_bfloat16 *)(&data))[i];
    }
    FLASHINFER_INLINE __hip_bfloat16 *ptr()
    {
        return reinterpret_cast<__hip_bfloat16 *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_bfloat16 val);
    FLASHINFER_INLINE void load(const __hip_bfloat16 *ptr);
    FLASHINFER_INLINE void store(__hip_bfloat16 *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 4> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_bfloat16 *dst,
                                         const __hip_bfloat16 *src);
};

FLASHINFER_INLINE void vec_t<__hip_bfloat16, 4>::fill(__hip_bfloat16 val)
{
    *(__hip_bfloat162 *)(&data.x) = make_bfloat162(val, val);
    *(__hip_bfloat162 *)(&data.y) = make_bfloat162(val, val);
}

FLASHINFER_INLINE void vec_t<__hip_bfloat16, 4>::load(const __hip_bfloat16 *ptr)
{
    data = *((uint2 *)ptr);
}

FLASHINFER_INLINE void
vec_t<__hip_bfloat16, 4>::store(__hip_bfloat16 *ptr) const
{
    *((uint2 *)ptr) = data;
}

FLASHINFER_INLINE void
vec_t<__hip_bfloat16, 4>::memcpy(__hip_bfloat16 *dst, const __hip_bfloat16 *src)
{
    *((uint2 *)dst) = *((uint2 *)src);
}

// __hip_bfloat16 x 8 or more

template <size_t vec_size> struct vec_t<__hip_bfloat16, vec_size>
{
    uint4 data[vec_size / 8];

    FLASHINFER_INLINE __hip_bfloat16 &operator[](size_t i)
    {
        return ((__hip_bfloat16 *)data)[i];
    }
    FLASHINFER_INLINE const __hip_bfloat16 &operator[](size_t i) const
    {
        return ((const __hip_bfloat16 *)data)[i];
    }
    FLASHINFER_INLINE __hip_bfloat16 *ptr()
    {
        return reinterpret_cast<__hip_bfloat16 *>(&data);
    }
    FLASHINFER_INLINE void fill(__hip_bfloat16 val)
    {
#pragma unoll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            *(__hip_bfloat162 *)(&(data[i].x)) = make_bfloat162(val, val);
            *(__hip_bfloat162 *)(&(data[i].y)) = make_bfloat162(val, val);
            *(__hip_bfloat162 *)(&(data[i].z)) = make_bfloat162(val, val);
            *(__hip_bfloat162 *)(&(data[i].w)) = make_bfloat162(val, val);
        }
    }
    FLASHINFER_INLINE void load(const __hip_bfloat16 *ptr)
    {
#pragma unoll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            data[i] = ((uint4 *)ptr)[i];
        }
    }
    FLASHINFER_INLINE void store(__hip_bfloat16 *ptr) const
    {
#pragma unoll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            ((uint4 *)ptr)[i] = data[i];
        }
    }
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(__hip_bfloat16 *dst,
                                         const __hip_bfloat16 *src)
    {
#pragma unoll
        for (size_t i = 0; i < vec_size / 8; ++i) {
            ((uint4 *)dst)[i] = ((uint4 *)src)[i];
        }
    }
};

/******************* vec_t<float> *******************/

// float x 1

template <> struct vec_t<float, 1>
{
    float data;

    FLASHINFER_INLINE float &operator[](size_t i)
    {
        return ((float *)(&data))[i];
    }
    FLASHINFER_INLINE const float &operator[](size_t i) const
    {
        return ((const float *)(&data))[i];
    }
    FLASHINFER_INLINE float *ptr() { return reinterpret_cast<float *>(&data); }
    FLASHINFER_INLINE void fill(float val);
    FLASHINFER_INLINE void load(const float *ptr);
    FLASHINFER_INLINE void store(float *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 1> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(float *dst, const float *src);
};

FLASHINFER_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

FLASHINFER_INLINE void vec_t<float, 1>::load(const float *ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<float, 1>::store(float *ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<float, 1>::memcpy(float *dst, const float *src)
{
    *dst = *src;
}

// float x 2

template <> struct vec_t<float, 2>
{
    float2 data;

    FLASHINFER_INLINE float &operator[](size_t i)
    {
        return ((float *)(&data))[i];
    }
    FLASHINFER_INLINE const float &operator[](size_t i) const
    {
        return ((const float *)(&data))[i];
    }
    FLASHINFER_INLINE float *ptr() { return reinterpret_cast<float *>(&data); }
    FLASHINFER_INLINE void fill(float val);
    FLASHINFER_INLINE void load(const float *ptr);
    FLASHINFER_INLINE void store(float *ptr) const;
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, 2> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(float *dst, const float *src);
};

FLASHINFER_INLINE void vec_t<float, 2>::fill(float val)
{
    data = make_float2(val, val);
}

FLASHINFER_INLINE void vec_t<float, 2>::load(const float *ptr)
{
    data = *((float2 *)ptr);
}

FLASHINFER_INLINE void vec_t<float, 2>::store(float *ptr) const
{
    *((float2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<float, 2>::memcpy(float *dst, const float *src)
{
    *((float2 *)dst) = *((float2 *)src);
}

// float x 4 or more
template <size_t vec_size> struct vec_t<float, vec_size>
{
    float4 data[vec_size / 4];

    FLASHINFER_INLINE float &operator[](size_t i)
    {
        return ((float *)(data))[i];
    }
    FLASHINFER_INLINE const float &operator[](size_t i) const
    {
        return ((const float *)(data))[i];
    }
    FLASHINFER_INLINE float *ptr() { return reinterpret_cast<float *>(&data); }
    FLASHINFER_INLINE void fill(float val)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 4; ++i) {
            data[i] = make_float4(val, val, val, val);
        }
    }
    FLASHINFER_INLINE void load(const float *ptr)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 4; ++i) {
            data[i] = ((float4 *)ptr)[i];
        }
    }
    FLASHINFER_INLINE void store(float *ptr) const
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 4; ++i) {
            ((float4 *)ptr)[i] = data[i];
        }
    }
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size> &src)
    {
        cast_from_impl(*this, src);
    }
    template <typename T> FLASHINFER_INLINE void cast_load(const T *ptr)
    {
        cast_load_impl(*this, ptr);
    }
    template <typename T> FLASHINFER_INLINE void cast_store(T *ptr) const
    {
        cast_store_impl(ptr, *this);
    }
    FLASHINFER_INLINE static void memcpy(float *dst, const float *src)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size / 4; ++i) {
            ((float4 *)dst)[i] = ((float4 *)src)[i];
        }
    }
};

} // namespace flashinfer
