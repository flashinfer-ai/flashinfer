// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once
#include "macros.hpp"
#include <iostream>

#include "gpu_runtime_compat.hpp"

namespace flashinfer
{
namespace gpu_iface
{

// Platform-agnostic stream type
#if defined(PLATFORM_CUDA_DEVICE)
constexpr int kWarpSize = 32;

#elif defined(PLATFORM_HIP_DEVICE)
constexpr int kWarpSize = 64;

#endif

} // namespace gpu_iface
} // namespace flashinfer
