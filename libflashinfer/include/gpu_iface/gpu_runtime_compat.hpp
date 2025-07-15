// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

/// gpu_runtime_compat.hpp - Minimal compatibility layer for CUDA/HIP
/// differences

#pragma once
#include "macros.hpp"

// Include appropriate runtime
#if defined(PLATFORM_CUDA_DEVICE)
#include <cuda_runtime.h>
#elif defined(PLATFORM_HIP_DEVICE)
#include <hip/hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

// Basic type mappings
#if defined(PLATFORM_CUDA_DEVICE)
#define gpuError_t cudaError_t
#define gpuStream_t cudaStream_t
#elif defined(PLATFORM_HIP_DEVICE)
#define gpuError_t hipError_t
#define gpuStream_t hipStream_t
#endif

// Kernel launch and attributes (these actually differ in name)
#if defined(PLATFORM_CUDA_DEVICE)
#define gpuGetDevice cudaGetDevice
#define gpuLaunchKernel cudaLaunchKernel
#define gpuFuncSetAttribute cudaFuncSetAttribute
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#elif defined(PLATFORM_HIP_DEVICE)
#define gpuGetDevice hipGetDevice
#define gpuLaunchKernel hipLaunchKernel
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#endif

#if defined(PLATFORM_CUDA_DEVICE)
#define gpuMemCpy cudaMemcpy
#define gpuMemCpyAsync cudaMemcpyAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#elif defined(PLATFORM_HIP_DEVICE)
#define gpuMemCpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif

// Function attribute enums (these have different names)
#if defined(PLATFORM_CUDA_DEVICE)
#define gpuFuncAttributeMaxDynamicSharedMemorySize                             \
    cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncAttributePreferredSharedMemoryCarveout                          \
    cudaFuncAttributePreferredSharedMemoryCarveout
#elif defined(PLATFORM_HIP_DEVICE)
#define gpuFuncAttributeMaxDynamicSharedMemorySize                             \
    hipFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncAttributePreferredSharedMemoryCarveout                          \
    hipFuncAttributePreferredSharedMemoryCarveout
#endif

// Device attribute enums (different names)
#if defined(PLATFORM_CUDA_DEVICE)
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define gpuDevAttrMaxSharedMemoryPerMultiProcessor                             \
    cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor                           \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor
#elif defined(PLATFORM_HIP_DEVICE)
#define gpuDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define gpuDevAttrMaxSharedMemoryPerMultiProcessor                             \
    hipDeviceAttributeMaxSharedMemPerMultiprocessor
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor                           \
    hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif

// Error handling (for FI_GPU_CALL)
#if defined(PLATFORM_CUDA_DEVICE)
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#elif defined(PLATFORM_HIP_DEVICE)
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#endif

// CUDA error checking macro (replaces FLASHINFER_CUDA_CALL)
#define FI_GPU_CALL(call)                                                      \
    do {                                                                       \
        gpuError_t err = (call);                                               \
        if (err != gpuSuccess) {                                               \
            std::ostringstream err_msg;                                        \
            err_msg << "GPU error: " << gpuGetErrorString(err) << " at "       \
                    << __FILE__ << ":" << __LINE__;                            \
            throw std::runtime_error(err_msg.str());                           \
        }                                                                      \
    } while (0)
