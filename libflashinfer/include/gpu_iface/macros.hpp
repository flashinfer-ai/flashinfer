// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// Platform detection
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIP__)
#define PLATFORM_HIP_DEVICE
// FIXME: Temporarily setting __forceinline__ to inline as amdclang++ 6.4 throws
// an error when __forceinline__ is used.
#define __forceinline__ inline
#elif defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define PLATFORM_CUDA_DEVICE
#endif

// Common attributes with FlashInfer (FI) specific naming
#define FI_HOST_DEVICE_QUAL __host__ __device__
#define FI_DEVICE_QUAL __device__
#define FI_HOST_QUAL __host__
#define FI_GLOBAL_QUAL __global__
#define FI_SHARED_QUAL __shared__
#define FI_CONSTANT_QUAL __constant__
