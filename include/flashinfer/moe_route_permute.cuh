// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef FLASHINFER_MOE_ROUTE_PERMUTE_CUH_
#define FLASHINFER_MOE_ROUTE_PERMUTE_CUH_

// Stable-rank router + BF16 permutation for small routed-row counts.
// Writes only routed rows, inverse indices and expert offsets. This is not a
// replacement for the generic moe_sort API and its additional metadata.
// IDs must already name valid local experts in [0, experts).
// FI's existing routing/offset/inverse contracts define the interface. This is a
// new implementation, not a copy of the NVIDIA TRT-LLM baseline routing kernel.
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {

__global__ void moeRoutePermuteSmallKernel(const uint4* __restrict__ x,
                                           const int32_t* __restrict__ ids,
                                           uint4* __restrict__ routed,
                                           int32_t* __restrict__ inverse,
                                           int32_t* __restrict__ offsets, int rows, int top_k,
                                           int vectors, int experts) {
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  const int expert = ids[row];
  int rank = 0, less = 0, through = 0, next = experts;
  for (int j = thread; j < rows; j += 256) {
    const int other = ids[j];
    less += other < expert;
    through += other <= expert;
    rank += other < expert || (other == expert && j < row);
    if (other > expert && other < next) next = other;
  }
  for (int delta = 16; delta > 0; delta /= 2) {
    rank += __shfl_down_sync(0xffffffff, rank, delta);
    less += __shfl_down_sync(0xffffffff, less, delta);
    through += __shfl_down_sync(0xffffffff, through, delta);
    const int n = __shfl_down_sync(0xffffffff, next, delta);
    next = next < n ? next : n;
  }
  __shared__ int partial[4][8];
  __shared__ int total[4];
  if (thread % 32 == 0) {
    partial[0][thread / 32] = rank;
    partial[1][thread / 32] = less;
    partial[2][thread / 32] = through;
    partial[3][thread / 32] = next;
  }
  __syncthreads();
  if (thread < 32) {
    rank = thread < 8 ? partial[0][thread] : 0;
    less = thread < 8 ? partial[1][thread] : 0;
    through = thread < 8 ? partial[2][thread] : 0;
    next = thread < 8 ? partial[3][thread] : experts;
    for (int delta = 16; delta > 0; delta /= 2) {
      rank += __shfl_down_sync(0xffffffff, rank, delta);
      less += __shfl_down_sync(0xffffffff, less, delta);
      through += __shfl_down_sync(0xffffffff, through, delta);
      const int n = __shfl_down_sync(0xffffffff, next, delta);
      next = next < n ? next : n;
    }
    if (thread == 0) {
      total[0] = rank;
      total[1] = less;
      total[2] = through;
      total[3] = next;
    }
  }
  __syncthreads();
  rank = total[0];
  less = total[1];
  through = total[2];
  next = total[3];
  if (thread == 0) inverse[row] = rank;
  // Exactly the first occurrence of each expert owns its following offset gap.
  // The smallest active expert additionally fills the leading empty groups.
  // These ranges are disjoint, including offsets[experts], without atomics.
  if (rank == less) {
    const int begin = less == 0 ? 0 : expert + 1;
    for (int e = begin + thread; e <= next; e += 256) {
      offsets[e] = e <= expert ? 0 : through;
    }
  }
  for (int col = thread; col < vectors; col += 256) {
    routed[rank * vectors + col] = x[(row / top_k) * vectors + col];
  }
}

}  // namespace flashinfer

#endif  // FLASHINFER_MOE_ROUTE_PERMUTE_CUH_
