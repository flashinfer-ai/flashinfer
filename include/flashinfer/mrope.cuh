/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 by FlashInfer team.
 */
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
__device__ __forceinline__ float MropeFma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ half MropeFma(half a, half b, half c) { return __hfma(a, b, c); }
__device__ __forceinline__ __nv_bfloat16 MropeFma(__nv_bfloat16 a, __nv_bfloat16 b,
                                                  __nv_bfloat16 c) {
  return __hfma(a, b, c);
}
// Pair-parallel kernel: each thread owns both components of one pair. Exact in-place
// operation is safe because no other thread reads or writes that pair.
template <typename T, typename C, typename I, bool Dense = false, bool Strided = false>
__global__ void MropePairKernel(const T* q, const T* k, T* qo, T* ko, const I* positions,
                                const C* cache, int64_t tokens, int hq, int hk, int dim, int rotary,
                                int st, int sh, int sw, bool axis_interleaved, int pair_bits,
                                int64_t q_stride = 0, int64_t k_stride = 0,
                                int64_t position_stride = 0) {
  const int64_t token = blockIdx.x;
  const int pair_index = blockIdx.y * blockDim.x + threadIdx.x;
  // Use shifts for the compact rotary pitch. Keep dense indexing when padding
  // would increase the work beyond the original full-head mapping.
  int head, pair;
  if constexpr (Dense) {
    head = pair_index / (dim / 2);
    pair = pair_index % (dim / 2);
  } else {
    head = pair_index >> pair_bits;
    pair = pair_index & ((1 << pair_bits) - 1);
  }
  if (head >= hq + hk || pair >= dim / 2) return;
  const bool is_q = head < hq;
  const int heads = is_q ? hq : hk;
  const int local_head = is_q ? head : head - hq;
  const int64_t row_stride = Strided ? (is_q ? q_stride : k_stride) : heads * dim;
  const int64_t offset = token * row_stride + local_head * dim;
  const T* input = (is_q ? q : k) + offset;
  T* output = (is_q ? qo : ko) + offset;
  if (pair >= rotary / 2) {
    // Tail pairs are adjacent regardless of the rotary pairing convention.
    if (output != input) {
      output[2 * pair] = input[2 * pair];
      output[2 * pair + 1] = input[2 * pair + 1];
    }
    return;
  }
  int axis;
  if (axis_interleaved) {
    axis = pair % 3 == 1 && pair < 3 * sh ? 1 : pair % 3 == 2 && pair < 3 * sw ? 2 : 0;
  } else {
    axis = pair < st ? 0 : pair < st + sh ? 1 : 2;
  }
  // Value bounds are a caller precondition, as for existing RoPE cache APIs.
  const int64_t position =
      positions[static_cast<int64_t>(axis) * (Strided ? position_stride : tokens) + token];
  const C c = cache[position * rotary + pair];
  const C s = cache[position * rotary + pair + rotary / 2];
  const int first = pair;
  const int second = pair + rotary / 2;
  const C a = static_cast<C>(input[first]);
  const C b = static_cast<C>(input[second]);
  // Match the pinned vLLM Triton PTX: round b*s and b*c to cache precision,
  // then fuse a*c - rounded(b*s) and a*s + rounded(b*c), respectively.
  const C bs = b * s;
  const C bc = b * c;
  output[first] = static_cast<T>(MropeFma(a, c, -bs));
  output[second] = static_cast<T>(MropeFma(a, s, bc));
}
// Reuse each cache pair across heads instead of reloading it in several CTAs.
// The binding selects this in-place 16-Q/8-K-head, full D=128 specialization
// only after rejecting Q/K overlap, which makes the restrict qualifiers valid.
template <typename T, typename C, typename I, int Threads, bool Strided = false>
__global__ void MropeToken128(T* __restrict__ q, T* __restrict__ k, const I* positions,
                              const C* cache, int64_t tokens, int st, int sh, int sw,
                              bool axis_interleaved, int64_t q_stride = 0, int64_t k_stride = 0,
                              int64_t position_stride = 0) {
  const int64_t token = blockIdx.x;
  const int pair = threadIdx.x & 63;
  const int first_head = threadIdx.x >> 6;
  constexpr int head_step = Threads / 64;
  const int axis = axis_interleaved ? (pair % 3 == 1 && pair < 3 * sh   ? 1
                                       : pair % 3 == 2 && pair < 3 * sw ? 2
                                                                        : 0)
                                    : (pair < st        ? 0
                                       : pair < st + sh ? 1
                                                        : 2);
  const int64_t position =
      positions[static_cast<int64_t>(axis) * (Strided ? position_stride : tokens) + token];
  const C c = cache[position * 128 + pair];
  const C s = cache[position * 128 + pair + 64];
#pragma unroll
  for (int i = 0; i < 16 / head_step; ++i) {
    const int head = first_head + i * head_step;
    const int64_t offset = token * (Strided ? q_stride : 16 * 128) + head * 128 + pair;
    const C a = static_cast<C>(q[offset]);
    const C b = static_cast<C>(q[offset + 64]);
    const C bs = b * s;
    const C bc = b * c;
    q[offset] = static_cast<T>(MropeFma(a, c, -bs));
    q[offset + 64] = static_cast<T>(MropeFma(a, s, bc));
  }
#pragma unroll
  for (int i = 0; i < 8 / head_step; ++i) {
    const int head = first_head + i * head_step;
    const int64_t offset = token * (Strided ? k_stride : 8 * 128) + head * 128 + pair;
    const C a = static_cast<C>(k[offset]);
    const C b = static_cast<C>(k[offset + 64]);
    const C bs = b * s;
    const C bc = b * c;
    k[offset] = static_cast<T>(MropeFma(a, c, -bs));
    k[offset + 64] = static_cast<T>(MropeFma(a, s, bc));
  }
}
// One token per CTA; each thread retains its cache pair across Q/K heads.
// Used for bounded-head D=128 prefill. Small batches retain the pair mapping
// to expose more CTAs, and the tuned 16/8 full-rotation kernel stays separate.
template <typename T, typename C, typename I, int Rotary, int Threads = 128>
__global__ void MropeTokenPrefill(T* __restrict__ q, T* __restrict__ k, const I* positions,
                                  const C* cache, int hq, int hk, int st, int sh, int sw,
                                  bool axis_interleaved, int64_t q_stride, int64_t k_stride,
                                  int64_t position_stride) {
  constexpr int half_rotary = Rotary / 2;
  constexpr int head_step = Threads / half_rotary;
  const int pair = threadIdx.x % half_rotary;
  const int first_head = threadIdx.x / half_rotary;
  const int64_t token = blockIdx.x;
  const int axis = axis_interleaved ? (pair % 3 == 1 && pair < 3 * sh   ? 1
                                       : pair % 3 == 2 && pair < 3 * sw ? 2
                                                                        : 0)
                                    : (pair < st        ? 0
                                       : pair < st + sh ? 1
                                                        : 2);
  const int64_t position = positions[static_cast<int64_t>(axis) * position_stride + token];
  const C c = cache[position * Rotary + pair];
  const C s = cache[position * Rotary + pair + half_rotary];
  for (int head = first_head; head < hq; head += head_step) {
    const int64_t offset = token * q_stride + head * 128 + pair;
    const C a = static_cast<C>(q[offset]);
    const C b = static_cast<C>(q[offset + half_rotary]);
    const C bs = b * s, bc = b * c;
    q[offset] = static_cast<T>(MropeFma(a, c, -bs));
    q[offset + half_rotary] = static_cast<T>(MropeFma(a, s, bc));
  }
  for (int head = first_head; head < hk; head += head_step) {
    const int64_t offset = token * k_stride + head * 128 + pair;
    const C a = static_cast<C>(k[offset]);
    const C b = static_cast<C>(k[offset + half_rotary]);
    const C bs = b * s, bc = b * c;
    k[offset] = static_cast<T>(MropeFma(a, c, -bs));
    k[offset + half_rotary] = static_cast<T>(MropeFma(a, s, bc));
  }
}
}  // namespace flashinfer
