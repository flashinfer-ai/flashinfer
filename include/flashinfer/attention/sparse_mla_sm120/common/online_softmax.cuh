// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../arch/common.cuh"

// Online softmax state for one head, maintained in smem (m) and registers (per-warp l).
//
// The attention kernel uses a distributed online softmax:
//   - Each warp computes local max over its ENTRIES_PER_WARP entries
//   - Cross-warp max reduction via smem reduce_buf
//   - Global max update → rescale acc_o by exp2(old_m - new_m)
//   - Each warp computes local sum of exp weights
//   - Cross-warp sum reduction via smem sum_reduce_buf (or same buf via O1 overlap)
//   - l_smem accumulates total sum

// Warp-local max of 4 QK values, reduced across tid lanes (tid = lane & 3)
__device__ __forceinline__ void softmax_warp_max(float s[4], float& lm0, float& lm1) {
  lm0 = fmaxf(s[0], s[1]);
  lm1 = fmaxf(s[2], s[3]);
  lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1));
  lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
  lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1));
  lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));
}

// Warp-local sum of exp weights
__device__ __forceinline__ void softmax_warp_sum(float w0, float w1, float w2, float w3, float& ls0,
                                                 float& ls1) {
  ls0 = w0 + w1;
  ls1 = w2 + w3;
  ls0 += __shfl_xor_sync(0xffffffff, ls0, 1);
  ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
  ls1 += __shfl_xor_sync(0xffffffff, ls1, 1);
  ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
}

// Compute LSE (log-sum-exp) in log2 space for the merge kernel
__device__ __forceinline__ float softmax_lse(float m, float l) {
  return (l > 0.f) ? (log2f(l) + m) : -1e30f;
}
