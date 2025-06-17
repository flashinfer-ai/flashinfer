// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once
#define HIP_ENABLE_WARP_SYNC_BUILTINS

#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_platform.h"
#include <cstdint>

namespace flashinfer::math
{

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

template <typename T> __forceinline__ __device__ T ptx_exp2(T x);

/// @brief Wrapper for computing 2 ^ x. We currently do not support a direct
/// equivalent of __exp2f()
/// @param x Input power to exponentiate
/// @return Computes 2 ^ x
template <> __forceinline__ __device__ float ptx_exp2<float>(float x)
{
    return __exp10f(x * __log10f(2.0f)); // Writing 2^x = 10 ^ (x * log_10(2))
}

/// @brief Wrapper for computing 2 ^ x. We currently do not support a direct
/// equivalent of __exp2f()
/// @param x Input power to exponentiate
/// @return Computes 2 ^ x
template <> __forceinline__ __device__ __half ptx_exp2<__half>(__half x)
{
    return hexp2(x);
}

/// @brief Wrapper for computing 2 ^ x. We currently do not support a direct
/// equivalent of __exp2f()
/// @param x Vector of two half dtypes to exponentiate
/// @return Computes 2 ^ x
template <> __forceinline__ __device__ __half2 ptx_exp2<__half2>(__half2 x)
{
    return half2(ptx_exp2(x.x), ptx_exp2(x.y));
}

/// @brief Compute log2
/// @param x Input param - float dtype
/// @return Log2
__forceinline__ __device__ float ptx_log2(float x) { return __log2f(x); }

/// @brief Compute 1/x
/// @param x Input param - float dtype
/// @return Returns 1 / x in round-to-nearest-even mod.
__forceinline__ __device__ float ptx_rcp(float x) { return __frcp_rn(x); }

template <typename T>
__forceinline__ __device__ T shfl_xor_sync(T x, int lane_mask)
{
    // FIXME (diptorupd): The shfl_xor_sync is used to implement a butterfly
    // reduction pattern. The caller in decode.cuh most likely assumes that the
    // warp size is 32 and the lane_mask is going from 16, 8, 4, 2, 1.
    // Given that AMDGPU for CDNA3 has a warp size of 64, the lane_mask based on
    // the warp size of 32 might lead to incorrect exchanges between the
    // threads. The issue requires further investigation, for now I have hard
    // coded the warp size to 32 when calling shfl_xor.
    return __shfl_xor(x, lane_mask, 32);
}

/// @brief Wrapper for math intrinsic 1/sqrt(x)
/// @param x Input param - float dtype
/// @return Returns 1 / sqrt(x) in round to nearest even mode
__forceinline__ __device__ float rsqrt(float x) { return __frsqrt_rn(x); }

template <typename T> __forceinline__ __device__ T tanh(T x);

/// @brief Compute tanhf(x)
/// @param x Input param - float dtype
/// @return Returns tanhf(x)
/// @note ROCm6.3 does not have a fast tanh or instrincs to support this
template <> __forceinline__ __device__ float tanh<float>(float x)
{
    return tanhf(x);
}

/// @brief A utility function to compute tanh for half dtype
/// @param x Input param - half
/// @return Hyperbolic tangent of x
template <> __forceinline__ __device__ __half tanh<__half>(__half x)
{
    return __float2half(tanh(__half2float(x)));
}

/// @brief Compute hyperbolic tangent for a vector of two half dtype
/// @param x Vector of two half dtypes
/// @return Hyperbolic tangent of x
template <> __forceinline__ __device__ __half2 tanh<__half2>(__half2 x)
{
    return __half2(tanh(x.x), tanh(x.y));
}

} // namespace flashinfer::math
#endif  // FLASHINFER_MATH_CUH_
