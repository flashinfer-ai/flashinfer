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

__device__ __forceinline__ int SampleLadderBin(float value) {
  uint16_t bits = __half_as_ushort(__float2half_rn(value));
  uint16_t key = bits;
  if (!(bits & 0x8000)) key = (bits ^ 0xffff) & 0x7fff;
  return key >> 5;
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
  const int tid = threadIdx.x, row = blockIdx.x, count = valid[row];
  for (int bin = tid; bin < kBins; bin += kThreads) histogram[bin] = 0;
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
        ladder[row * 3 + rung] = SampleLadderBinFloor(bin);
        populations[row * 3 + rung] = prefix[i] + counts[i];
      }
    }
    if (count == 0 && tid == 0) {
      chosen_bins[rung] = kBins - 1;
      ladder[row * 3 + rung] = -CUDART_INF_F;
      populations[row * 3 + rung] = 0;
    }
  }
  __syncthreads();
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
