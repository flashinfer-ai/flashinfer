/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_FUSED_MOE_HASH_TOPK_CUH_
#define FLASHINFER_FUSED_MOE_HASH_TOPK_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

// DSv4 hash-based MoE routing.
//
// Ported from SGLang's `deepseek_v4/hash_topk.cuh`. DSv4-Pro hash-MoE layers
// select experts from a precomputed token-to-expert table (`tid2eid`) rather
// than running a dynamic top-k. Routing is therefore an O(1) table lookup plus
// a sqrt-softplus score normalization. One warp processes one token.
//
// This header is framework-agnostic (raw pointers only). The launcher path
// from PyTorch tensors lives in `csrc/fused_moe/hash_topk.cu`.

namespace flashinfer {
namespace fused_moe {

constexpr uint32_t kHashTopKWarpThreads = 32;

// Numerically stable sqrt(softplus(x)) = sqrt(log(1 + exp(x))).
__device__ __forceinline__ float act_sqrt_softplus(float x) {
  const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(softplus);
}

// Full-warp butterfly sum. Lanes outside the active range contribute 0.
__device__ __forceinline__ float hash_topk_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kHashTopKWarpThreads / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

struct MoEHashTopKParams {
  const float* __restrict__ router_logits;  // [num_tokens, num_routed_experts] fp32
  const int64_t* __restrict__ input_id;     // [num_tokens] int64
  const int32_t* __restrict__ tid2eid;      // [vocab, topk] int32
  int32_t* __restrict__ topk_ids;           // [num_tokens, topk_fused] int32
  float* __restrict__ topk_weights;         // [num_tokens, topk_fused] fp32
  uint32_t num_tokens;
  uint32_t topk;
  uint32_t num_routed_experts;
  uint32_t num_shared_experts;
  float routed_scaling_factor;
};

template <bool kUsePDL>
__global__ void moe_hash_topk_fused(const MoEHashTopKParams __grid_constant__ params) {
  const uint32_t topk_fused = params.topk + params.num_shared_experts;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / kHashTopKWarpThreads;
  const uint32_t lane_id = tid % kHashTopKWarpThreads;
  // A whole warp shares one warp_id, so all lanes return together.
  if (warp_id >= params.num_tokens) return;

  // Safe to prefetch the token id for the whole warp.
  const int64_t token_id = params.input_id[warp_id];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;");
  }
#endif

  float routed_weight = 0.0f;
  int32_t expert_id = 0;
  if (lane_id < params.topk) {
    expert_id = params.tid2eid[token_id * params.topk + lane_id];
    routed_weight =
        act_sqrt_softplus(params.router_logits[warp_id * params.num_routed_experts + expert_id]);
  }

  const float routed_sum = hash_topk_warp_reduce_sum(routed_weight);
  if (lane_id < topk_fused) {
    const bool is_shared = lane_id >= params.topk;
    const uint32_t output_offset = warp_id * topk_fused + lane_id;
    params.topk_ids[output_offset] =
        is_shared ? static_cast<int32_t>(params.num_routed_experts + lane_id - params.topk)
                  : expert_id;
    params.topk_weights[output_offset] =
        is_shared ? (1.0f / params.routed_scaling_factor) : (routed_weight / routed_sum);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
#endif
}

// Framework-agnostic launcher. `enable_pdl` is honored only on SM90+.
inline cudaError_t LaunchHashTopK(const float* router_logits, const int64_t* input_id,
                                  const int32_t* tid2eid, int32_t* topk_ids, float* topk_weights,
                                  uint32_t num_tokens, uint32_t topk, uint32_t num_routed_experts,
                                  uint32_t num_shared_experts, float routed_scaling_factor,
                                  bool enable_pdl, cudaStream_t stream) {
  MoEHashTopKParams params{router_logits,
                           input_id,
                           tid2eid,
                           topk_ids,
                           topk_weights,
                           num_tokens,
                           topk,
                           num_routed_experts,
                           num_shared_experts,
                           routed_scaling_factor};

  constexpr uint32_t kBlockSize = 128;
  constexpr uint32_t kNumWarps = kBlockSize / kHashTopKWarpThreads;  // 4
  const uint32_t num_blocks = (num_tokens + kNumWarps - 1) / kNumWarps;

  cudaLaunchConfig_t config;
  config.gridDim = num_blocks;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (enable_pdl) {
    return cudaLaunchKernelEx(&config, moe_hash_topk_fused<true>, params);
  }
  return cudaLaunchKernelEx(&config, moe_hash_topk_fused<false>, params);
}

}  // namespace fused_moe
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_HASH_TOPK_CUH_
