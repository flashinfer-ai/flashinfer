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

// md is a contiguous [num_tokens, num_heads_q] float2 buffer (max/sum per (token, head)).
// lse is a user-provided [num_tokens, num_heads_q] float buffer with arbitrary strides:
// lse[token_idx, head_idx] lives at
//     lse + token_idx * lse_stride_tokens + head_idx * lse_stride_heads.
// Each thread handles one (token, head) pair indexed by elem_idx in [0, num_tokens*num_heads_q).
__global__ void ComputeLSEFromMDKernel(float2* __restrict__ md, float* __restrict__ lse,
                                       int num_heads_q, int64_t lse_stride_tokens,
                                       int64_t lse_stride_heads, int n) {
  int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_idx >= n) return;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  float2 md_elem = md[elem_idx];
  float m = md_elem.x;
  float d = md_elem.y;
  int token_idx = elem_idx / num_heads_q;
  int head_idx = elem_idx - token_idx * num_heads_q;
  int64_t out_idx = static_cast<int64_t>(token_idx) * lse_stride_tokens +
                    static_cast<int64_t>(head_idx) * lse_stride_heads;
  lse[out_idx] = math::log2e * m + math::ptx_log2(d);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

inline cudaError_t ComputeLSEFromMD(float2* md, float* lse, int num_tokens, int num_heads_q,
                                    int64_t lse_stride_tokens, int64_t lse_stride_heads,
                                    bool launch_with_pdl, cudaStream_t stream) {
  int n = num_tokens * num_heads_q;
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

  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, ComputeLSEFromMDKernel, md, lse, num_heads_q,
                                          lse_stride_tokens, lse_stride_heads, n));
  return cudaSuccess;
}

};  // namespace flashinfer

#endif  // FLASHINFER_TRTLLM_FMHA_LSE_CUH
