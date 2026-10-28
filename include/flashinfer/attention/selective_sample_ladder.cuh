/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_SELECTIVE_SAMPLE_LADDER_CUH_
#define FLASHINFER_ATTENTION_SELECTIVE_SAMPLE_LADDER_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace flashinfer {
namespace attention {

__device__ __forceinline__ uint16_t SampleLadderKey(float value) {
  uint16_t bits = __half_as_ushort(__float2half_rn(value));
  uint16_t key = bits;
  if (!(bits & 0x8000)) key = (bits ^ 0xffff) & 0x7fff;
  return key;
}

__device__ __forceinline__ int SampleLadderBin(float value) { return SampleLadderKey(value) >> 5; }

// Descending FP32 keys let atomicMax publish the lowest actual score in a
// selected FP16 sub-bin, without rounding that conservative floor upward.
__device__ __forceinline__ uint32_t SampleLadderScoreKey(float value) {
  const uint32_t bits = __float_as_uint(value);
  return bits & 0x80000000u ? bits : (~bits & 0x7fffffffu);
}

__device__ __forceinline__ float SampleLadderScoreFromKey(uint32_t key) {
  return __uint_as_float(key & 0x80000000u ? key : (~key & 0x7fffffffu));
}

__device__ __forceinline__ float SampleLadderBinFloor(int bin) {
  uint16_t key = min(0xfc00, (bin << 5) + 31);
  uint16_t bits = key;
  if (!(key & 0x8000)) bits = (key ^ 0xffff) & 0x7fff;
  return __half2float(__ushort_as_half(bits));
}

// The histogram and all three crossings share one CTA. The sampled populations
// remain estimates: exact selection and repair must still verify the result.
static __global__ void SelectiveSampleLadder(const float* samples, int64_t sample_stride,
                                             const int32_t* valid, const int32_t* ranks,
                                             const int32_t* ends, float* thresholds,
                                             int32_t* selected_bins, float* ladder,
                                             int32_t* populations, int top_k, int capacity,
                                             int count_segments) {
  constexpr int kThreads = 128;
  constexpr int kBins = 2048;
  constexpr int kItems = kBins / kThreads;
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int histogram[kBins];
  __shared__ int chosen_bins[3];
  __shared__ int before_bins[3];
  __shared__ int refine_row;
  __shared__ int fine_histogram[3][32];
  __shared__ uint32_t fine_floors[3][32];
  const int tid = threadIdx.x, row = blockIdx.x, count = valid[row];
  for (int bin = tid; bin < kBins; bin += kThreads) histogram[bin] = 0;
  if (tid < 96) {
    fine_histogram[tid / 32][tid % 32] = 0;
    fine_floors[tid / 32][tid % 32] = 0;
  }
  __syncthreads();
  for (int item = tid; item < count; item += kThreads) {
    atomicAdd(histogram + SampleLadderBin(samples[row * sample_stride + item]), 1);
  }
  __syncthreads();
  int counts[kItems], prefix[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) counts[i] = histogram[tid * kItems + i];
  Scan(scan_storage).ExclusiveSum(counts, prefix);
#pragma unroll
  for (int rung = 0; rung < 3; ++rung) {
    const int target = max(1, min(count, ranks[row]) / (4 >> rung));
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      if (prefix[i] < target && prefix[i] + counts[i] >= target) {
        const int bin = tid * kItems + i;
        chosen_bins[rung] = bin;
        before_bins[rung] = prefix[i];
        ladder[row * 3 + rung] = SampleLadderBinFloor(bin);
        populations[row * 3 + rung] = prefix[i] + counts[i];
      }
    }
    if (count == 0 && tid == 0) {
      chosen_bins[rung] = kBins - 1;
      before_bins[rung] = 0;
      ladder[row * 3 + rung] = -CUDART_INF_F;
      populations[row * 3 + rung] = 0;
    }
  }
  __syncthreads();
  // One CTA owns one row. Skip the second sample pass only when at least
  // one coarse rung already passes the existing population/bank confidence
  // checks. This gate uses samples, never oracle score or repair information.
  if (tid == 0) {
    refine_row = count > 0 && ends[row] >= top_k;
#pragma unroll
    for (int rung = 0; rung < 3; ++rung) {
      const float hits = populations[row * 3 + rung];
      const float sigma = sqrtf(fmaxf(hits, 1.f));
      const float scale = float(ends[row]) / float(max(count, 1));
      const float lower = fmaxf(0.f, hits - 3.f * sigma) * scale;
      const float upper = (hits + 3.f * sigma) * scale;
      const float bank_mean = upper / float(count_segments);
      const float bank_upper = fmaf(3.f, sqrtf(fmaxf(bank_mean, 1.f)), bank_mean);
      if (lower >= top_k && bank_upper < capacity / count_segments) refine_row = 0;
    }
  }
  __syncthreads();
  if (refine_row) {
    // Refine only the three boundary bins, reusing the existing sample scores.
    // Candidate scoring still runs once. Keep chosen_bins coarse: underfilled
    // candidate rows use that encoding to expand the exact-repair threshold.
    for (int item = tid; item < count; item += kThreads) {
      const float score = samples[row * sample_stride + item];
      const uint16_t key = SampleLadderKey(score);
      const int bin = key >> 5, fine = key & 31;
#pragma unroll
      for (int rung = 0; rung < 3; ++rung) {
        if (bin == chosen_bins[rung]) {
          atomicAdd(&fine_histogram[rung][fine], 1);
          atomicMax(&fine_floors[rung][fine], SampleLadderScoreKey(score));
        }
      }
    }
    __syncthreads();
    if (tid < 96) {
      const int rung = tid / 32, lane = tid % 32;
      const int own = fine_histogram[rung][lane];
      int inclusive = own;
#pragma unroll
      for (int delta = 1; delta < 32; delta <<= 1) {
        const int prior = __shfl_up_sync(0xffffffffu, inclusive, delta);
        if (lane >= delta) inclusive += prior;
      }
      const int target = max(1, min(count, ranks[row]) / (4 >> rung));
      const int before = before_bins[rung] + inclusive - own;
      if (count > 0 && before < target && before + own >= target) {
        ladder[row * 3 + rung] = SampleLadderScoreFromKey(fine_floors[rung][lane]);
        populations[row * 3 + rung] = before + own;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    int chosen = 2;
#pragma unroll
    for (int rung = 2; rung >= 0; --rung) {
      const float hits = populations[row * 3 + rung];
      const float sigma = sqrtf(fmaxf(hits, 1.f));
      const float scale = float(ends[row]) / float(max(count, 1));
      const float lower = fmaxf(0.f, hits - 3.f * sigma) * scale;
      const float upper = (hits + 3.f * sigma) * scale;
      const float bank_mean = upper / float(count_segments);
      // Fuse the bank-population estimate and its three-sigma safety margin.
      const float bank_upper = fmaf(3.f, sqrtf(fmaxf(bank_mean, 1.f)), bank_mean);
      if (lower >= top_k && bank_upper < capacity / count_segments) chosen = rung;
    }
    const bool eligible = count > 0 && ends[row] >= top_k;
    thresholds[row] = eligible ? ladder[row * 3 + chosen] : -CUDART_INF_F;
    selected_bins[row] = eligible ? chosen_bins[chosen] : kBins - 1;
  }
}

}  // namespace attention
}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_SELECTIVE_SAMPLE_LADDER_CUH_
