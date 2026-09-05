/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <flashinfer/conv3d/patchshift/common.cuh>
#include <flashinfer/conv3d/patchshift/problem.cuh>

namespace flashinfer::conv3d::patchshift::host {

using patchshift::Conv3dProblem;
using patchshift::Element;
using patchshift::TensorMap;

enum class StatusDomain : int { kSuccess = 0, kCudaRuntime = 1, kCudaDriver = 2 };

struct Status {
  StatusDomain domain;
  int code;

  constexpr bool ok() const { return domain == StatusDomain::kSuccess; }
  static constexpr Status Success() { return {StatusDomain::kSuccess, 0}; }
  static constexpr Status Cuda(cudaError_t error) {
    return error == cudaSuccess ? Success()
                                : Status{StatusDomain::kCudaRuntime, static_cast<int>(error)};
  }
  static constexpr Status Driver(CUresult error) {
    return error == CUDA_SUCCESS ? Success()
                                 : Status{StatusDomain::kCudaDriver, static_cast<int>(error)};
  }
};

struct alignas(128) DescriptorWorkspace {
  TensorMap input_m128;
  TensorMap input_hybrid_c32;
  TensorMap input_compact_p32;
  TensorMap input_compact_q8;
  TensorMap input_compact_q4;
  TensorMap input_compact_p1_c64;
  TensorMap input_id40_ptail_c64;
  TensorMap input_id40_qtail_c64;
  TensorMap weight_m128;
  TensorMap input_m64;
  TensorMap input_m64_compact_q4;
  TensorMap weight_m64;
};

static_assert(sizeof(TensorMap) == 128);
static_assert(sizeof(DescriptorWorkspace) == 12 * sizeof(TensorMap));

enum class LaunchPart : int { kAll = 0, kMain = 1, kAuxiliary = 2 };

enum class ConcurrencyMode : int { kSequential = 0, kDisjointMainAuxiliary = 1 };

Status PackWeights(const Element* weight, Element* packed_m128, Element* packed_m64,
                   Element* packed_m32, const Conv3dProblem& problem, int64_t stride_k,
                   int64_t stride_c, int64_t stride_t, int64_t stride_r, int64_t stride_s,
                   cudaStream_t stream);

Status PrepareDescriptors(DescriptorWorkspace* workspace, Element* input, Element* packed_m128,
                          Element* packed_m64, Element* packed_m32, const Conv3dProblem& problem,
                          int multi_processor_count, cudaStream_t stream);

ConcurrencyMode GetConcurrencyMode(const Conv3dProblem& problem, int multi_processor_count);

Status UpdateInputMaps(DescriptorWorkspace* workspace, Element* input, const Conv3dProblem& problem,
                       int multi_processor_count, cudaStream_t stream);

Status Launch(DescriptorWorkspace* workspace, Element* input, Element* output,
              const Conv3dProblem& problem, int multi_processor_count, cudaStream_t stream,
              LaunchPart part = LaunchPart::kAll);

}  // namespace flashinfer::conv3d::patchshift::host
