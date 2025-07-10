// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "macros.hpp"

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/math.cuh"
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/math.hip.h"
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace math
{
#if defined(PLATFORM_CUDA_DEVICE)
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

#elif defined(PLATFORM_HIP_DEVICE)
using flashinfer::math::inf;
using flashinfer::math::log2e;
using flashinfer::math::loge2;
using flashinfer::math::ptx_exp2;
using flashinfer::math::ptx_log2;
using flashinfer::math::ptx_rcp;
using flashinfer::math::rsqrt;
using flashinfer::math::shfl_xor_sync;
using flashinfer::math::tanh;

#else
#error "Unreachable: One of HIP or CUDA backend must be enabled."
// Add other functions as needed
#endif

} // namespace math
} // namespace gpu_iface
} // namespace flashinfer
