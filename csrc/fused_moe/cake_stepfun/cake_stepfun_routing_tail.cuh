/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

// Benign routing tail for GEMM units that do not bound the tiles they acquire through cluster
// launch control by num_non_exiting_ctas (KernelSpec::bounds_acquired_tiles == false). The routing
// stage writes cta_idx_xy_to_batch_idx / cta_idx_xy_to_mn_limit for the first *numNonExitingCtas
// CTAs and permuted_idx_to_token_idx for their token slots only; a running CTA of such a unit
// processes every cancelled CTA it acquires, so the entries in [*numNonExitingCtas, gridN) must
// describe a benign tile: expert 0, zero valid rows (mn_limit = tile * tileN) and padded token
// slots (-1). The kernel runs in-stream between routing and the GEMM with the GEMM's
// programmatic-dependent-launch attribute; it waits for routing before reading the count and
// releases the GEMM once the tail is written. Shared by the FC1 and FC2 runners.

#include <cuda_runtime.h>

#include <cstdint>

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {
namespace cake_stepfun {

constexpr unsigned kRoutingTailBlocks = 4;
constexpr unsigned kRoutingTailThreads = 256;

// routeMap may be nullptr when the consuming stage reads no token slots (FC2).
static __global__ void __launch_bounds__(kRoutingTailThreads)
    cake_stepfun_routing_tail_kernel(int32_t* tileExpert, int32_t* tileMnLimit, int32_t* routeMap,
                                     int32_t const* numNonExitingCtas, int32_t gridN,
                                     int32_t tileN) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int32_t const first = *numNonExitingCtas;
  int32_t const stride = static_cast<int32_t>(gridDim.x * blockDim.x);
  int32_t const lane = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  for (int32_t tile = first + lane; tile < gridN; tile += stride) {
    tileExpert[tile] = 0;
    tileMnLimit[tile] = tile * tileN;
  }
  if (routeMap != nullptr) {
    for (int32_t slot = first * tileN + lane; slot < gridN * tileN; slot += stride) {
      routeMap[slot] = -1;
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

inline cudaError_t launchRoutingTail(int32_t* tileExpert, int32_t* tileMnLimit, int32_t* routeMap,
                                     int32_t const* numNonExitingCtas, int32_t gridN, int32_t tileN,
                                     bool pdl, cudaStream_t stream) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kRoutingTailBlocks, 1u, 1u);
  config.blockDim = dim3(kRoutingTailThreads, 1u, 1u);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = pdl ? 1u : 0u;
  return cudaLaunchKernelEx(&config, cake_stepfun_routing_tail_kernel, tileExpert, tileMnLimit,
                            routeMap, numNonExitingCtas, gridN, tileN);
}

}  // namespace cake_stepfun
}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
