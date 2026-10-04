// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

namespace flashinfer::sparse_mla_sm120 {

struct PrefillLaunchResult {
  bool supported;
  cudaError_t error;
  const char* operation;

  PrefillLaunchResult(bool supported)
      : supported(supported), error(cudaSuccess), operation(nullptr) {}
  PrefillLaunchResult(cudaError_t error, const char* operation)
      : supported(true), error(error), operation(operation) {}
};

}  // namespace flashinfer::sparse_mla_sm120
