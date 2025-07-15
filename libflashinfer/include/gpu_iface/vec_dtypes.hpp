// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "platform.hpp"
#include <float.h>
#include <math.h>

namespace flashinfer
{
namespace gpu_iface
{
namespace vec_dtypes
{

// Include the appropriate backend implementation
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/vec_dtypes.cuh"
namespace detail = flashinfer::gpu_iface::vec_dtypes::detail::cuda;
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/vec_dtypes_hip.h"
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1
namespace detail = flashinfer::gpu_iface::vec_dtypes::detail::hip;
#endif

// Re-export types and functions from the appropriate backend
// This allows code to use flashinfer::gpu_iface::vec_dtypes::vec_t<float, 4>
using detail::vec_t;

} // namespace vec_dtypes
} // namespace gpu_iface
} // namespace flashinfer
