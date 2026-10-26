/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cuda/atomic>

namespace flashinfer::gemm {

// The caller zeroes the counter on the stream before launching the prefix.
// Each quantize CTA publishes all its stores; the GEMM acquires the complete
// release sequence before loading XQ/SF through TMA. LoRA-down has its own PDL wait.
struct SvdquantQuantizeBarrierSm120 {
  unsigned int* counter = nullptr;
  unsigned int blocks = 0;

  /// Publish this CTA's completed quantize stores with a device-scope release.
  /// All CTA threads must call together; a null counter disables the barrier.
  __device__ void arrive() const {
    if (counter != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ready(*counter);
        ready.fetch_add(1, cuda::memory_order_release);
      }
    }
  }

  /// Acquire all quantize CTA releases through a non-null, per-launch counter.
  /// The caller must fence the async proxy before issuing dependent TMA loads.
  __device__ void wait() const {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ready(*counter);
    while (ready.load(cuda::memory_order_acquire) != blocks) {
    }
  }
};

}  // namespace flashinfer::gemm
