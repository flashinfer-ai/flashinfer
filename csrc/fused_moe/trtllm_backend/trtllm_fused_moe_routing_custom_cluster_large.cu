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

#define FLASHINFER_ROUTING_CUSTOM_CLUSTER_LARGE
#include "trtllm_fused_moe_routing_custom.cuh"

namespace moe::dev::routing::routingCustom {

bool launchClusterKernelBlockDim1024(Data const& data, void* stream) {
  bool const useNoOpSoftmaxScores = data.mPtrScores != nullptr &&
                                    data.mPreprocessType == RoutingPreprocessType::None &&
                                    data.mPostprocessType == RoutingPostprocessType::Softmax;
  if (useNoOpSoftmaxScores) {
    return launchClusterKernelForPolicy<ClusterBlockDim1024, NoOpPreprocess, SoftmaxPostprocess>(
        data, stream);
  }

  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
  return queryPolicyHasCompiledTier(data);
}

}  // namespace moe::dev::routing::routingCustom
