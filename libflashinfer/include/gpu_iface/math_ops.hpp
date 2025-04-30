// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "macros.hpp"

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/math.cuh"
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/math.hip"
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace math
{

// Use inline namespaces to bring in the right implementations
#if defined(PLATFORM_CUDA_DEVICE)
inline namespace cuda_math
{
// Re-export CUDA math functions with same names
using flashinfer::math::half2_as_uint32;
using flashinfer::math::inf;
using flashinfer::math::log2e;
using flashinfer::math::loge2;
using flashinfer::math::ptx_exp2;
using flashinfer::math::ptx_log2;
using flashinfer::math::ptx_rcp;
using flashinfer::math::rsqrt;
using flashinfer::math::shfl_xor_sync;
using flashinfer::math::tanh;
using flashinfer::math::uint32_as_half2;

// Add other functions as needed
} // namespace cuda_math
#elif defined(PLATFORM_HIP_DEVICE)
inline namespace hip_math
{
// Re-export HIP math functions with same names
using flashinfer::math::inf;
using flashinfer::math::log2e;
using flashinfer::math::loge2;
using flashinfer::math::ptx_exp2;
using flashinfer::math::ptx_log2;
using flashinfer::math::rsqrt;
using flashinfer::math::shfl_xor;
using flashinfer::math::tanh;
// Add other function mappings as needed
} // namespace hip_math
#else
#error "Unreachable: One of HIP or CUDA backend must be enabled."
// Add other functions as needed
#endif

} // namespace math
} // namespace gpu_iface
} // namespace flashinfer
