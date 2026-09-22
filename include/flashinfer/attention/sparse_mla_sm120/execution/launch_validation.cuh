// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

namespace flashinfer::sparse_mla_sm120::execution {

template <typename Kernel>
cudaError_t validate_mixed_smem(Kernel kernel, size_t dynamic_bytes) {
  cudaFuncAttributes attributes{};
  cudaError_t result = cudaFuncGetAttributes(&attributes, kernel);
  if (result != cudaSuccess) return result;
  int device = 0, limit = 0;
  result = cudaGetDevice(&device);
  if (result != cudaSuccess) return result;
  result = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (result != cudaSuccess) return result;
  return dynamic_bytes + attributes.sharedSizeBytes <= size_t(limit)
             ? cudaSuccess
             : cudaErrorInvalidConfiguration;
}

}  // namespace flashinfer::sparse_mla_sm120::execution
