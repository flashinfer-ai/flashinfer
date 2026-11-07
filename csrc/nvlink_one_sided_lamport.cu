/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Launchers of the Lamport transport of the NVLink one-sided MoE all-to-all: the only
// translation unit that instantiates its kernels (flashinfer/comm/moe_alltoall_lamport.cuh).

#include "flashinfer/comm/moe_alltoall_lamport.cuh"
#include "tensorrt_llm/common/envUtils.h"

// Runs the statements __VA_ARGS__ with TOP_K_CAPACITY and FIXED_TOP_K of the kernels serving
// top_k: top_k 6, 8, 10, 16 and 22 have their own kernels, the others share a kernel with a
// runtime top_k of at most 8, 16 or 24.
#define MOE_A2A_LAMPORT_TOP_K_CASE(CAPACITY, FIXED, ...) \
  {                                                      \
    constexpr int TOP_K_CAPACITY = CAPACITY;             \
    constexpr bool FIXED_TOP_K = FIXED;                  \
    __VA_ARGS__;                                         \
  }
#define MOE_A2A_LAMPORT_SWITCH_TOP_K(top_k, ...)       \
  if ((top_k) == 6)                                    \
    MOE_A2A_LAMPORT_TOP_K_CASE(6, true, __VA_ARGS__)   \
  else if ((top_k) == 8)                               \
    MOE_A2A_LAMPORT_TOP_K_CASE(8, true, __VA_ARGS__)   \
  else if ((top_k) == 10)                              \
    MOE_A2A_LAMPORT_TOP_K_CASE(10, true, __VA_ARGS__)  \
  else if ((top_k) == 16)                              \
    MOE_A2A_LAMPORT_TOP_K_CASE(16, true, __VA_ARGS__)  \
  else if ((top_k) == 22)                              \
    MOE_A2A_LAMPORT_TOP_K_CASE(22, true, __VA_ARGS__)  \
  else if ((top_k) <= 8)                               \
    MOE_A2A_LAMPORT_TOP_K_CASE(8, false, __VA_ARGS__)  \
  else if ((top_k) <= 16)                              \
    MOE_A2A_LAMPORT_TOP_K_CASE(16, false, __VA_ARGS__) \
  else                                                 \
    MOE_A2A_LAMPORT_TOP_K_CASE(24, false, __VA_ARGS__)

namespace flashinfer {
namespace moe_alltoall_lamport {
namespace {

using tensorrt_llm::common::launchWithPdlWhenEnabled;

// 1 in the code built for an architecture the kernels support.
__device__ int g_kernels_compiled = MOE_A2A_LAMPORT_COMPILE;

template <int TOP_K_CAPACITY, bool FIXED_TOP_K, bool ENABLE_RANK_MASK>
void launch_dispatch(int payload_kind, DispatchParams const& params, bool enable_pdl,
                     cudaStream_t stream) {
  void (*kernel)(DispatchParams) =
      payload_kind == kKindBf16
          ? moeA2ALamportDispatchKernel<TOP_K_CAPACITY, FIXED_TOP_K, kKindBf16, ENABLE_RANK_MASK>
      : payload_kind == kKindFp8
          ? moeA2ALamportDispatchKernel<TOP_K_CAPACITY, FIXED_TOP_K, kKindFp8, ENABLE_RANK_MASK>
          : moeA2ALamportDispatchKernel<TOP_K_CAPACITY, FIXED_TOP_K, kKindNvfp4, ENABLE_RANK_MASK>;
  launchWithPdlWhenEnabled("moeA2ALamportDispatchKernel", enable_pdl, kernel,
                           dim3(params.max_tokens_per_rank * params.blocks_per_token),
                           dim3(kNumThreads), 0, stream, params);
}

template <int TOP_K_CAPACITY, bool FIXED_TOP_K, bool ENABLE_RANK_MASK>
void launch_combine(CombineParams const& params, bool enable_pdl, cudaStream_t stream) {
  launchWithPdlWhenEnabled(
      "moeA2ALamportCombineKernel", enable_pdl,
      moeA2ALamportCombineKernel<TOP_K_CAPACITY, FIXED_TOP_K, ENABLE_RANK_MASK>,
      dim3(params.max_tokens_per_rank * params.blocks_per_token), dim3(kNumThreads), 0, stream,
      params);
}

}  // namespace

bool moe_a2a_lamport_kernels_available() {
  int compiled = 0;
  if (cudaMemcpyFromSymbol(&compiled, g_kernels_compiled, sizeof(compiled)) != cudaSuccess) {
    cudaGetLastError();  // no code for this device at all
    return false;
  }
  return compiled != 0;
}

void moe_a2a_lamport_dispatch_launch(int payload_kind, DispatchParams const& params,
                                     bool enable_rank_mask, bool enable_pdl, cudaStream_t stream) {
  MOE_A2A_LAMPORT_SWITCH_TOP_K(params.top_k, {
    if (enable_rank_mask) {
      launch_dispatch<TOP_K_CAPACITY, FIXED_TOP_K, true>(payload_kind, params, enable_pdl, stream);
    } else {
      launch_dispatch<TOP_K_CAPACITY, FIXED_TOP_K, false>(payload_kind, params, enable_pdl, stream);
    }
  })
}

void moe_a2a_lamport_combine_launch(CombineParams const& params, bool enable_rank_mask,
                                    bool enable_pdl, cudaStream_t stream) {
  MOE_A2A_LAMPORT_SWITCH_TOP_K(params.top_k, {
    if (enable_rank_mask) {
      launch_combine<TOP_K_CAPACITY, FIXED_TOP_K, true>(params, enable_pdl, stream);
    } else {
      launch_combine<TOP_K_CAPACITY, FIXED_TOP_K, false>(params, enable_pdl, stream);
    }
  })
}

}  // namespace moe_alltoall_lamport
}  // namespace flashinfer
