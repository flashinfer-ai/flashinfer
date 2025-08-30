/*
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef FLASHINFER_TRTLLM_FMHA_LSE_CUH
#define FLASHINFER_TRTLLM_FMHA_LSE_CUH

#include <cuda.h>

#include "../../math.cuh"
#include "../../utils.cuh"

namespace flashinfer {

__global__ void ComputeLSEFromMDKernel(float2* __restrict__ md, float* __restrict__ lse, int n) {
  int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_idx >= n) return;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  float2 md_elem = md[elem_idx];
  float m = md_elem.x;
  float d = md_elem.y;
  lse[elem_idx] = math::log2e * m + math::ptx_log2(d);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

inline cudaError_t ComputeLSEFromMD(float2* md, float* lse, int n, bool launch_with_pdl,
                                    cudaStream_t stream) {
  int num_threads = std::min(1024, UpPowerOfTwo(n));
  int num_blocks = ceil_div(n, num_threads);
  cudaLaunchConfig_t config;
  config.gridDim = num_blocks;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = launch_with_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, ComputeLSEFromMDKernel, md, lse, n));
  return cudaSuccess;
}

};  // namespace flashinfer

#endif  // FLASHINFER_TRTLLM_FMHA_LSE_CUH
