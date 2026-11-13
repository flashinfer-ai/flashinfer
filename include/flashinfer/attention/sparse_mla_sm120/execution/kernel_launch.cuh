// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
