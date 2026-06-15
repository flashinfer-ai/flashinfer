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

/*!
 * \file hash_topk.cuh
 * \brief DSv4 hash-based MoE expert routing.
 *
 * Ported from SGLang (Apache-2.0):
 *   sgl-project/sglang
 *   python/sglang/jit_kernel/csrc/deepseek_v4/hash_topk.cuh
 *
 * DSv4-Pro hash-MoE layers select experts from a precomputed token-to-expert
 * table (`tid2eid`) rather than running a dynamic top-k, so routing is an O(1)
 * table lookup plus a sqrt-softplus score normalization. One warp processes one
 * token.
 *
 * This header is framework-agnostic (raw pointers only). The launcher path from
 * PyTorch tensors lives in `csrc/fused_moe/hash_topk.cu`.
 */

namespace flashinfer {
namespace fused_moe {

constexpr uint32_t kHashTopKWarpThreads = 32;

/*!
 * \brief Numerically stable sqrt(softplus(x)) = sqrt(log(1 + exp(x))).
 * \param x The input logit.
 */
__device__ __forceinline__ float act_sqrt_softplus(float x) {
  const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(softplus);
}

/*!
 * \brief Full-warp butterfly sum. Lanes outside the active range contribute 0.
 * \param val The per-lane value to reduce.
 */
__device__ __forceinline__ float hash_topk_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kHashTopKWarpThreads / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

/*!
 * \brief Parameters for the hash top-k kernel. All pointers are device pointers
 *   into contiguous, row-major tensors on a single device.
 */
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

/*!
 * \brief Hash top-k routing kernel: one warp per token.
 * \tparam kUsePDL Whether to use programmatic dependent launch (SM90+).
 * \param params Kernel parameters.
 *
 * \note Behavior is undefined if `input_id[t]` is outside `[0, vocab)` or if any
 *   table entry `tid2eid[...]` is outside `[0, num_routed_experts)`. Callers are
 *   responsible for supplying a valid table.
 */
template <bool kUsePDL>
__global__ void moe_hash_topk_fused(const MoEHashTopKParams __grid_constant__ params) {
  const uint32_t topk_fused = params.topk + params.num_shared_experts;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / kHashTopKWarpThreads;
  const uint32_t lane_id = tid % kHashTopKWarpThreads;
  // A whole warp shares one warp_id, so all lanes return together.
  if (warp_id >= params.num_tokens) return;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Wait before the dependent loads. The "memory" clobber prevents the compiler
  // from hoisting loads/stores across the dependency barrier.
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif

  const int64_t token_id = params.input_id[warp_id];
  // 64-bit row base: warp_id * num_routed_experts can exceed 2^32 for large
  // token counts, so widen before the multiply to avoid index overflow.
  const int64_t logits_row = static_cast<int64_t>(warp_id) * params.num_routed_experts;

  float routed_weight = 0.0f;
  int32_t expert_id = 0;
  if (lane_id < params.topk) {
    expert_id = params.tid2eid[token_id * params.topk + lane_id];
    routed_weight = act_sqrt_softplus(params.router_logits[logits_row + expert_id]);
  }

  const float routed_sum = hash_topk_warp_reduce_sum(routed_weight);
  if (lane_id < topk_fused) {
    const bool is_shared = lane_id >= params.topk;
    const int64_t output_offset = static_cast<int64_t>(warp_id) * topk_fused + lane_id;
    params.topk_ids[output_offset] =
        is_shared ? static_cast<int32_t>(params.num_routed_experts + lane_id - params.topk)
                  : expert_id;
    params.topk_weights[output_offset] =
        is_shared ? (1.0f / params.routed_scaling_factor) : (routed_weight / routed_sum);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // The "memory" clobber prevents the compiler from sinking the output stores
  // above past the launch-dependents signal.
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

/*!
 * \brief Framework-agnostic launcher for the hash top-k kernel.
 * \param router_logits Device pointer, [num_tokens, num_routed_experts] fp32.
 * \param input_id Device pointer, [num_tokens] int64.
 * \param tid2eid Device pointer, [vocab, topk] int32.
 * \param topk_ids Device pointer, [num_tokens, topk_fused] int32 (output).
 * \param topk_weights Device pointer, [num_tokens, topk_fused] fp32 (output).
 * \param num_tokens Number of tokens.
 * \param topk Number of routed experts per token.
 * \param num_routed_experts Total number of routed experts.
 * \param num_shared_experts Number of fused shared experts (0 or 1).
 * \param routed_scaling_factor Scaling factor for the shared-expert weight.
 * \param enable_pdl Whether to enable programmatic dependent launch (SM90+).
 * \param stream The CUDA stream.
 * \return cudaSuccess on success (a no-op success for num_tokens == 0).
 */
inline cudaError_t LaunchHashTopK(const float* router_logits, const int64_t* input_id,
                                  const int32_t* tid2eid, int32_t* topk_ids, float* topk_weights,
                                  uint32_t num_tokens, uint32_t topk, uint32_t num_routed_experts,
                                  uint32_t num_shared_experts, float routed_scaling_factor,
                                  bool enable_pdl, cudaStream_t stream) {
  if (num_tokens == 0) {
    return cudaSuccess;
  }

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
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = enable_pdl ? 1 : 0;
  config.attrs = enable_pdl ? attrs : nullptr;

  if (enable_pdl) {
    return cudaLaunchKernelEx(&config, moe_hash_topk_fused<true>, params);
  }
  return cudaLaunchKernelEx(&config, moe_hash_topk_fused<false>, params);
}

}  // namespace fused_moe
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_HASH_TOPK_CUH_
