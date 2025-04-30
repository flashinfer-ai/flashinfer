// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// gpu_iface/cooperative_groups.hpp
#pragma once
#include "macros.hpp"

#if defined(PLATFORM_CUDA_DEVICE)
#include <cooperative_groups.h>
namespace flashinfer
{
namespace gpu_iface
{
namespace cg = ::cooperative_groups;
} // namespace gpu_iface
} // namespace flashinfer

#elif defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_cooperative_groups.h>
namespace flashinfer
{
namespace gpu_iface
{
namespace cg = ::cooperative_groups;
} // namespace gpu_iface
} // namespace flashinfer
#endif
