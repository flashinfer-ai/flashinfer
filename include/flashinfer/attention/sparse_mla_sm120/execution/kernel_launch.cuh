// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cuda_runtime.h>

namespace flashinfer::sparse_mla_sm120 {

// Shared CUDA launch setup for decode and prefill; no attention-route semantics.
constexpr int kMaxCachedCudaDevices = 32;

template <typename Kernel>
cudaError_t configure_dynamic_smem_per_device(Kernel kernel, size_t smem_bytes,
                                              bool (&configured)[kMaxCachedCudaDevices]) {
  if (smem_bytes <= 48 * 1024) return cudaSuccess;

  int device = 0;
  const cudaError_t device_error = cudaGetDevice(&device);
  if (device_error != cudaSuccess) return device_error;
  const bool cacheable_device = device >= 0 && device < kMaxCachedCudaDevices;
  if (cacheable_device && configured[device]) return cudaSuccess;

  const cudaError_t rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem_bytes));
  if (rc == cudaSuccess && cacheable_device) configured[device] = true;
  return rc;
}

}  // namespace flashinfer::sparse_mla_sm120
