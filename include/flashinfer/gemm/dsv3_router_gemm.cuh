/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace flashinfer::trtllm_dsv3_router_gemm {
// Custom FMA implementation using PTX assembly instructions
__device__ __forceinline__ void fma(float2& d, float2 const& a, float2 const& b, float2 const& c) {
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
}

// Convert 8 bfloat16 values from a uint4 to float array - optimized conversion
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec, float* dst) {
  __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));

#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

// One block per expert column; the block reduces the full K extent for all
// kNumTokens rows at once. kHiddenDim must be a whole number of K iterations
// (VPT * kBlockSize elements, i.e. a multiple of 1024 for bf16 with a 128-thread
// block), which is what lets every load be a fully-coalesced 16B vector load.
template <typename Tin, typename Tout, int kBlockSize, int VPT, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(kBlockSize, 1) void router_gemm_kernel(Tout* out, Tin const* mat_a,
                                                                    Tin const* mat_b) {
  // Each block handles one expert column
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  // Constants for this kernel
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  static_assert(kHiddenDim % k_elems_per_k_iteration == 0,
                "kHiddenDim must be a whole number of K iterations (VPT * kBlockSize)");
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;  // Total K iterations

  // Initialize accumulators for all M rows
  float acc[kNumTokens] = {};

  // Shared memory for warp-level reduction.
  //
  // The final cross-warp reduction below has lane `l` walk row `l` of this
  // array. With an unpadded row stride of kNumWarps (4 for a 128-thread block)
  // every one of those lanes lands in the same 4-bank group, so the reads
  // serialize. Padding the row by one float staggers each lane onto a distinct
  // bank.
  //
  // This mirrors SGLang's copy of the kernel so the two do not drift. Measured
  // on B200 it makes no difference at router shapes -- the kernel is bound on
  // streaming the weight column, and this reduction is over kNumWarps values
  // for at most 16 tokens -- so it is kept for parity, not for speed.
  constexpr int kSmemPad = (kNumTokens > 8) ? 1 : 0;
  __shared__ float sm_reduction[kNumTokens][kNumWarps + kSmemPad];

  // B matrix is in column-major order, so we can directly load a column for the n_idx expert
  Tin const* b_col = mat_b + n_idx * kHiddenDim;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Process the GEMM in chunks
  int k_base = tid * VPT;
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++, k_base += k_elems_per_k_iteration) {
    // Load B matrix values using vector load (8 bf16 values)
    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);

    // Convert B values to float
    float b_float[VPT];
    bf16_uint4_to_float8<VPT>(b_vec, b_float);

// Process each token
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      // Load both rows of A matrix using vector loads
      uint4 a_vec = *reinterpret_cast<uint4 const*>(mat_a + (m_idx * kHiddenDim) + k_base);

      // Convert A values to float
      float a_float[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_float);

// Process elements in this chunk
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        float a = a_float[k];
        float b = b_float[k];
        acc[m_idx] += a * b;
      }
    }
  }

  // Perform warp-level reduction
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

// Perform warp-level reduction using optimized butterfly pattern
#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = acc[m];

    // Butterfly reduction pattern
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    // Only the first thread in each warp stores to shared memory
    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }

  __syncthreads();

  // Final reduction across warps. One lane per token (kNumTokens <= 16 < 32, so
  // a single warp covers every row) instead of serializing all M rows on tid 0.
  if (warpId == 0 && laneId < kNumTokens) {
    float final_sum = 0.0f;

// Sum across the kNumWarps
#pragma unroll
    for (int w = 0; w < kNumWarps; w++) {
      final_sum += sm_reduction[laneId][w];
    }

    // Write final result
    out[laneId * kNumExperts + n_idx] = static_cast<Tout>(final_sum);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}
}  // namespace flashinfer::trtllm_dsv3_router_gemm
