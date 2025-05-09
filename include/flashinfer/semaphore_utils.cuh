/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_SEMAPHORE_UTILS_CUH
#define FLASHINFER_SEMAPHORE_UTILS_CUH

#include <cuda_runtime.h>

#include "utils.cuh"

namespace flashinfer {

template <typename T>
__global__ void zero_gmem_semaphore(T* semaphore, int size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    semaphore[i] = 0;
  }
}

template <typename T>
cudaError_t zero_gmem_semaphore_launcher(T* semaphore, int size, bool enable_pdl,
                                         cudaStream_t stream) {
  cudaLaunchConfig_t config = {0};
  config.gridDim = 1;
  config.blockDim = 128;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, zero_gmem_semaphore<T>, semaphore, size));

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_SEMAPHORE_UTILS_CUH
