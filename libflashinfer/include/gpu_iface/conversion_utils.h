// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace fi::con
{
template <typename DTypeIn, typename DTypeOut>
__host__ __device__ __inline__ DTypeOut explicit_casting(DTypeIn value)
{
    return DTypeOut(value);
}

template <>
__host__ __device__ __inline__ float
explicit_casting<__half, float>(__half value)
{
    return __half2float(value);
}

template <>
__host__ __device__ __inline__ float
explicit_casting<__hip_bfloat16, float>(__hip_bfloat16 value)
{
    return __bfloat162float(value);
}

template <>
__host__ __device__ __inline__ __half
explicit_casting<float, __half>(float value)
{
    return __float2half(value);
}

template <>
__host__ __device__ __inline__ __hip_bfloat16
explicit_casting<__half, __hip_bfloat16>(__half value)
{
    return __float2bfloat16(__half2float(value));
}

template <>
__host__ __device__ __inline__ float explicit_casting<float, float>(float value)
{
    return value;
}

template <>
__host__ __device__ __inline__ __half
explicit_casting<__half, __half>(__half value)
{
    return value;
}

template <>
__host__ __device__ __inline__ __hip_bfloat16
explicit_casting<__hip_bfloat16, __hip_bfloat16>(__hip_bfloat16 value)
{
    return value;
}
} // namespace fi::con
