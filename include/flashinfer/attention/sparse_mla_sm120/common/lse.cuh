// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

namespace flashinfer::sparse_mla_sm120 {

// Only final LSE stores change base; split softmax state stays in log2 space.
// Preserve the empty-row sentinel used by the sparse kernels exactly.
__device__ __forceinline__ float scale_output_lse(float lse, float scale) {
  return lse == -1e30f ? lse : scale * lse;
}

}  // namespace flashinfer::sparse_mla_sm120
