/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::glm5_tp8::exact_select {

// Total ordering of finite FP32 values. Signed zeros may be selected in either
// order under the numeric top-k contract; this key consistently prefers +0.
__device__ __forceinline__ uint32_t score_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

// Exact selection from a complete, immutable global candidate array. No shared
// candidate list and no compaction: each pass rescans only the matching prefix.
// Preconditions: unique valid candidate IDs, finite scores, 0 <= k <= count,
// all CTA threads participate, hist[257] and cursors[2] are private to this CTA.
// The caller has already published candidates/scores and appends this result
// after its strictly higher coarse bins. Output order and cutoff ties are free.
// On return cursors[0] is exactly the number of output IDs, including ties.
template <class Sync>
__device__ __forceinline__ void exact_topk(int tid, int threads, const float* scores,
                                           const int32_t* candidates, int count, int k,
                                           int32_t* output, uint32_t* hist, uint32_t* cursors,
                                           Sync sync) {
  if (k == 0) {
    if (tid == 0) cursors[0] = 0;
    sync();
    return;
  }
  uint32_t prefix = 0, mask = 0;
  int remaining = k;
#pragma unroll
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int bin = tid; bin < 256; bin += threads) hist[bin] = 0;
    sync();
    for (int i = tid; i < count; i += threads) {
      const uint32_t key = score_key(scores[candidates[i]]);
      if ((key & mask) == prefix) atomicAdd(hist + ((key >> shift) & 255u), 1u);
    }
    sync();
    if (tid == 0) {
      uint32_t above = 0;
      for (int bin = 255; bin >= 0; --bin) {
        const uint32_t at = hist[bin];
        // Inclusive boundary also handles remaining == matching population.
        if (above + at >= static_cast<uint32_t>(remaining)) {
          cursors[0] = static_cast<uint32_t>(bin);
          cursors[1] = above;
          break;
        }
        above += at;
      }
    }
    sync();
    prefix |= cursors[0] << shift;
    mask |= 255u << shift;
    remaining -= static_cast<int>(cursors[1]);
    sync();  // Protect uniform register copies before the next cursor write.
  }
  const int strictly_above = k - remaining;
  if (tid == 0) cursors[0] = cursors[1] = 0;
  sync();
  for (int i = tid; i < count; i += threads) {
    const int32_t id = candidates[i];
    const uint32_t key = score_key(scores[id]);
    if (key > prefix) {
      const uint32_t slot = atomicAdd(cursors, 1u);
      output[slot] = id;
    } else if (key == prefix) {
      const uint32_t slot = atomicAdd(cursors + 1, 1u);
      if (slot < static_cast<uint32_t>(remaining)) output[strictly_above + slot] = id;
    }
  }
  sync();
  if (tid == 0) {
    const uint32_t ties = cursors[1] < static_cast<uint32_t>(remaining)
                              ? cursors[1]
                              : static_cast<uint32_t>(remaining);
    cursors[0] += ties;
  }
  sync();
}

}  // namespace flashinfer::glm5_tp8::exact_select
