/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <algorithm>

#define FLASHINFER_ROUTING_CUSTOM_BLOCK_GROUP
#include "trtllm_fused_moe_routing_custom.cuh"

namespace moe::dev::routing::routingCustom {

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched. A false return means no routing output was written;
// the caller (run) must not proceed to the downstream pipeline in that case.
bool launchBlockKernel(Data const& data, void* stream) {
  uint32_t const numThreadsBlock =
      std::min(1024u, static_cast<uint32_t>(queryDispatchedMaxExperts(data)));
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesBlockKernel, 1, numThreadsBlock,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
  return queryPolicyHasCompiledTier(data);
}

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched (see launchBlockKernel).
bool launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream) {
  int32_t const maxExperts = queryDispatchedMaxExperts(data);
  int const numSlots = data.mNumTokens * maxExperts;
  int const smemSize = numSlots + numSlots * 2 + 128 +
                       2 * (maxExperts / WarpSize) * static_cast<int>(sizeof(int32_t));
  int const threads =
      std::min(std::max(data.mNumTokens * static_cast<int>(WarpSize), maxExperts), 1024);

  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesDynBlockKernel, 1, threads, smemSize, stream);
  return queryPolicyHasCompiledTier(data);
}

}  // namespace moe::dev::routing::routingCustom
