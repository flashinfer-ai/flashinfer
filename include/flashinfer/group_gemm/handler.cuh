/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_GROUP_GEMM_HANDLER_CUH_
#define FLASHINFER_GROUP_GEMM_HANDLER_CUH_

#include <cstddef>

#include "../allocator.h"
#include "../utils.cuh"
#include "group_gemm_cutlass.cuh"
#include "group_gemm_lora.cuh"
#include "group_gemv.cuh"

namespace flashinfer {

namespace group_gemm {

enum class GroupGEMMKernelConfig {
  kGeneral,  // large d_in, d_out
  kShrink,   // large d_in, small d_out
  kExpand,   // small d_in, large d_out
};

class CutlassSegmentGEMMHandler {
 public:
  void RegisterWorkspace(void* buffer, size_t size) {
    buffer_ = buffer;
    workspace_size_in_bytes_ = size;
  }

  void* GetWorkspace() const { return buffer_; }

  size_t GetWorkspaceSizeInBytes() const { return workspace_size_in_bytes_; }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  CutlassSegmentGEMMHandler() {}

  ~CutlassSegmentGEMMHandler() {}

 private:
  void* buffer_;
  size_t workspace_size_in_bytes_;
  cudaStream_t stream_;
};

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_HANDLER_CUH_
