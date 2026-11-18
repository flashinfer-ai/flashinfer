/*
 * Copyright (c) 2026 by FlashInfer team.
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
// CUDA helpers shared by the launchers of the Nemotron 3.5 Super NVFP4-KV kernels.
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string>

#include "decode_params.h"
#include "flashinfer/exception.h"
#include "host.h"

namespace flashinfer {
namespace cake_nvfp4 {

inline void cudaCheck(cudaError_t status, char const* what) {
  if (status == cudaSuccess) return;
  FLASHINFER_ERROR(std::string("cake-nvfp4 ") + what + ": " + cudaGetErrorString(status));
}

inline void driverCheck(CUresult result, char const* what) {
  if (result == CUDA_SUCCESS) return;
  char const* message = nullptr;
  cuGetErrorString(result, &message);
  FLASHINFER_ERROR(std::string("cake-nvfp4 ") + what + ": " +
                   (message ? message : std::to_string(static_cast<int>(result))));
}

// 4-D tiled TMA descriptor. `strides` are the byte strides of dimensions 1 to 3.
inline CUtensorMap tensorMap(void const* ptr, CUtensorMapDataType dtype,
                             std::array<uint64_t, 4> shape, std::array<uint64_t, 3> strides,
                             std::array<uint32_t, 4> box, CUtensorMapSwizzle swizzle,
                             CUtensorMapL2promotion l2Promotion) {
  CUtensorMap result{};
  std::array<uint32_t, 4> elementStrides{1, 1, 1, 1};
  driverCheck(cuTensorMapEncodeTiled(&result, dtype, 4, const_cast<void*>(ptr), shape.data(),
                                     strides.data(), box.data(), elementStrides.data(),
                                     CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "cuTensorMapEncodeTiled");
  return result;
}

// Enqueues one kernel. `overlapsPrevious` lets it start before the previous kernel on the stream
// has finished (programmatic dependent launch); the kernel then waits where it needs the data.
template <class Kernel, class... Args>
void launchKernel(Kernel kernel, dim3 grid, int threads, int sharedBytes, cudaStream_t stream,
                  bool overlapsPrevious, Args const&... args) {
  if (sharedBytes > 0)
    cudaCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes),
        "cudaFuncSetAttribute");
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = sharedBytes;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = overlapsPrevious ? &attribute : nullptr;
  config.numAttrs = overlapsPrevious ? 1 : 0;
  cudaCheck(cudaLaunchKernelEx(&config, kernel, args...), "cudaLaunchKernelEx");
}

// Fills the parameter block of a decode kernel.
inline DecodeParams makeDecodeParams(Call const& c, DecodePlan const& plan) {
  DecodeParams p{};
  uint64_t const pages = c.poolPages;
  uint64_t const rows = c.sumQ;
  auto const none = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  p.tmaQ = tensorMap(c.q, CU_TENSOR_MAP_DATA_TYPE_UINT8, {128, 16, 2, rows}, {128, 2048, 4096},
                     {128, 16, 1, uint32_t(plan.g)}, CU_TENSOR_MAP_SWIZZLE_128B, none);
  p.tmaK = tensorMap(c.k, CU_TENSOR_MAP_DATA_TYPE_UINT8, {64, 128, 2, pages},
                     {64, uint64_t(c.kStrideHead), uint64_t(c.kStridePage)}, {64, 128, 1, 1},
                     CU_TENSOR_MAP_SWIZZLE_64B, none);
  p.tmaV = tensorMap(c.v, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, {128, 128, 2, pages},
                     {64, uint64_t(c.vStrideHead), uint64_t(c.vStridePage)}, {128, 128, 1, 1},
                     CU_TENSOR_MAP_SWIZZLE_128B, none);
  p.tmaKSf = tensorMap(c.kSf, CU_TENSOR_MAP_DATA_TYPE_UINT8, {128, 8, 2, pages},
                       {128, uint64_t(c.kSfStrideHead), uint64_t(c.kSfStridePage)}, {128, 8, 1, 1},
                       CU_TENSOR_MAP_SWIZZLE_NONE, none);
  p.tmaVSf = tensorMap(c.vSf, CU_TENSOR_MAP_DATA_TYPE_UINT8, {128, 8, 2, pages},
                       {128, uint64_t(c.vSfStrideHead), uint64_t(c.vSfStridePage)}, {128, 8, 1, 1},
                       CU_TENSOR_MAP_SWIZZLE_NONE, none);
  p.out = static_cast<uint8_t*>(c.out);
  p.ws = static_cast<float*>(c.workspace);
  p.pageTable = c.pageTable;
  p.seqLens = c.seqLensKv;
  p.cumSeqLensQ = plan.ragged ? c.cumSeqLensQ : nullptr;
  p.wsStatsOffset = plan.stats;
  p.wsRangeOffset = plan.ranges;
  p.wsSegsOffset = plan.segments;
  p.maxSegsPerCta = plan.virt + 1;
  p.numVirt = plan.virt;
  p.numGroups = plan.groups;
  p.queryTokens = c.maxQ;
  p.maxPagesPerSeq = c.maxPagesPerSeq;
  p.scaleSoftmaxLog2 = c.scaleSoftmaxLog2;
  p.outputScale = c.outputScale;
  p.scaleSoftmaxLog2Ptr = c.scaleSoftmaxLog2Ptr;
  p.outputScalePtr = c.outputScalePtr;
  // Mode 1: the combine kernel overlaps the main kernel. Mode 0 keeps the two ordered.
  p.pdlMode = c.enablePdl ? 1 : 0;
  return p;
}

}  // namespace cake_nvfp4
}  // namespace flashinfer
