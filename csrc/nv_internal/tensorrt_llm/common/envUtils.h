/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "flashinfer/exception.h"
#include "tensorrt_llm/common/quantization.h"

namespace tensorrt_llm::common {
// Useful when you want to inject some debug code controllable with env var.
std::optional<int32_t> getIntEnv(char const* name);

std::optional<size_t> getUInt64Env(char const* name);

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels();

// Whether XQA JIT is enabled.
//
// Returns the value of TRTLLM_ENABLE_XQA_JIT env var. If such env var doesn't exist, std::nullopt
// is returned.
std::optional<bool> getEnvEnableXQAJIT();

// 0 means to use heuristics.
std::optional<int32_t> getEnvXqaBlocksPerSequence();

// Whether use tileSizeKv64 for multiCtasKvMode of trtllm-gen kernels.
bool getEnvUseTileSizeKv64ForTrtllmGen();

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug();

int getEnvMmhaBlocksPerSequence();

int getEnvMmhaKernelBlockSize();

bool getEnvUseUCXKvCache();

bool getEnvUseMPIKvCache();

std::string getEnvUCXInterface();

bool getEnvDisaggLayerwise();

bool getEnvParallelCacheSend();

bool getEnvRequestKVCacheConcurrent();

bool getEnvDisableKVCacheTransferOverlap();

bool getEnvEnableReceiveKVCacheParallel();

std::string const& getEnvKVCacheTimeOutputPath();

bool getEnvTryZCopyForKVCacheTransfer();

// Force deterministic behavior for all kernels.
bool getEnvForceDeterministic();

// Force deterministic behavior for MoE plugin.
bool getEnvForceDeterministicMOE();

// Force deterministic behavior for attention plugin.
bool getEnvForceDeterministicAttention();

// Force deterministic behavior for all reduce plugin.
bool getEnvForceDeterministicAllReduce();

// Return the workspace size for custom all reduce kernels.
// This only works when force deterministic is enabled.
size_t getEnvAllReduceWorkspaceSize();

size_t getEnvKVCacheRecvBufferCount();

bool getEnvKVCacheTransferUseAsyncBuffer();

size_t getEnvKVCacheSendMaxConcurrenceNum();

size_t getEnvMemSizeForKVCacheTransferBuffer();

// TODO: For DEV purpose temporarily.
// Block size (threads per block) for MoE A2A Dispatch kernels (default 256 if unset or invalid)
int getEnvMoeA2ADispatchBlockSize();
// Block size (threads per block) for MoE A2A Combine kernels (default 256 if unset or invalid)
int getEnvMoeA2ACombineBlockSize();

// Disable the fast fp4 quantization math and align with the TransformerEngine
bool getEnvDisableFP4QuantFastMath();

// Enable the NVFP4 4over6 scale-candidate quantization mode.
bool getEnvNVFP4Use4Over6();

// Select the candidate error used by the NVFP4 4over6 scale-candidate path.
NVFP44Over6ErrMode getEnvNVFP44Over6ErrMode();

// Enable fast math in the NVFP4 4over6 scale-candidate error path.
bool getEnvNVFP44Over6ErrUseFastMath();

// Use 256 instead of 448 for the NVFP4 4over6 E4M3 scaling convention.
bool getEnvNVFP44Over6E4M3Use256();

template <typename KernelFn, typename... Args>
inline void launchWithPdlWhenEnabled(char const* name, bool enable_pdl, KernelFn kernelFn,
                                     dim3 grid, dim3 block, size_t dynamicShmSize,
                                     cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t kernelConfig;
  kernelConfig.gridDim = grid;
  kernelConfig.blockDim = block;
  kernelConfig.dynamicSmemBytes = dynamicShmSize;
  kernelConfig.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  kernelConfig.attrs = attrs;
  kernelConfig.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&kernelConfig, kernelFn, std::forward<Args>(args)...);
  FLASHINFER_CHECK(e == cudaSuccess, "cudaLaunchKernelEx (", name,
                   ") failed: ", cudaGetErrorString(e));
}

}  // namespace tensorrt_llm::common
