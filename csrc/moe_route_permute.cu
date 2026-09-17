// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <cstdint>

#include "flashinfer/exception.h"
#include "flashinfer/moe_route_permute.cuh"
#include "tvm_ffi_utils.h"

void flashinfer_moe_route_permute_small(int64_t x, int64_t ids, int64_t routed, int64_t inverse,
                                        int64_t offsets, int64_t tokens, int64_t top_k,
                                        int64_t hidden, int64_t experts, int64_t stream) {
  FLASHINFER_CHECK(tokens > 0 && tokens <= 512 && top_k > 0 && top_k <= 512 / tokens &&
                       hidden > 0 && hidden % 8 == 0 && hidden <= 16384 && experts > 0 &&
                       experts <= 4096,
                   "Small MoE route/permute requires 1..512 routed rows and aligned BF16 width");
  FLASHINFER_CHECK(x && ids && routed && inverse && offsets && x % 16 == 0 && routed % 16 == 0 &&
                       ids % 4 == 0 && inverse % 4 == 0 && offsets % 4 == 0,
                   "Small MoE route/permute received null or misaligned storage");
  flashinfer::moeRoutePermuteSmallKernel<<<tokens * top_k, 256, 0,
                                           reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint4*>(x), reinterpret_cast<const int32_t*>(ids),
      reinterpret_cast<uint4*>(routed), reinterpret_cast<int32_t*>(inverse),
      reinterpret_cast<int32_t*>(offsets), tokens * top_k, top_k, hidden / 8, experts);
  const auto status = cudaGetLastError();
  FLASHINFER_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_route_permute_small,
                              flashinfer_moe_route_permute_small);
