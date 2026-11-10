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
#pragma once

#include <math_constants.h>

#include <cstdint>
#include <cub/cub.cuh>

namespace flashinfer {
namespace sampling {

// Exclude both the compensated helper and its reduction storage by default:
// that variant regresses sampling performance even on untied rows.
#ifndef FLASHINFER_SAMPLING_USE_COMPENSATED_MASS
#define FLASHINFER_SAMPLING_USE_COMPENSATED_MASS 0
#endif

#if FLASHINFER_SAMPLING_USE_COMPENSATED_MASS
struct SamplingBoundaryMass {
  double above;
  uint32_t equal_count;
  uint32_t inexact;

  __device__ SamplingBoundaryMass operator+(const SamplingBoundaryMass& rhs) const {
    return {above + rhs.above, equal_count + rhs.equal_count, inexact | rhs.inexact};
  }
};

// Explicit IEEE operations preserve compensation under --use_fast_math,
// including subnormal residuals (no FTZ or reassociation).
__device__ __forceinline__ float SamplingMassAdd(float a, float b) {
  float result;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float SamplingMassSub(float a, float b) {
  float result;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float SamplingMassTwoSum(float a, float b, float& error) {
  const float sum = SamplingMassAdd(a, b);
  const float b_virtual = SamplingMassSub(sum, a);
  error = SamplingMassAdd(SamplingMassSub(a, SamplingMassSub(sum, b_virtual)),
                          SamplingMassSub(b, b_virtual));
  return sum;
}

// Float subtraction can turn an integral boundary quota into n + one ulp.
// Recheck tied boundaries in double, leaving unique-pivot scans unchanged.
template <uint32_t BLOCK_THREADS, typename DType>
__device__ __forceinline__ SamplingBoundaryMass GetSamplingBoundaryMass(const DType* row,
                                                                        uint32_t size,
                                                                        float boundary) {
  using Reduce = cub::BlockReduce<SamplingBoundaryMass, BLOCK_THREADS>;
  __shared__ typename Reduce::TempStorage temporary;
  __shared__ SamplingBoundaryMass result;
  SamplingBoundaryMass local{0, 0, 0};
  float sum = 0, correction = 0;
  for (uint32_t i = threadIdx.x; i < size; i += BLOCK_THREADS) {
    const float value = row[i];
    float error, lost;
    sum = SamplingMassTwoSum(sum, value > boundary ? value : 0.f, error);
    correction = SamplingMassTwoSum(correction, error, lost);
    local.inexact |= (__float_as_uint(lost) & 0x7fffffffu) != 0;
    local.equal_count += value == boundary;
  }
  double correction_double;
  asm("cvt.f64.f32 %0, %1;" : "=d"(correction_double) : "f"(correction));
  local.above = double(sum) + correction_double;
  const auto reduced = Reduce(temporary).Sum(local);
  if (threadIdx.x == 0) result = reduced;
  __syncthreads();
  // A two-float expansion is exact until its residual itself loses bits.
  // Rare wide-dynamic-range rows retain the original double-scan accuracy.
  if (result.inexact) {
    local.above = 0;
    for (uint32_t i = threadIdx.x; i < size; i += BLOCK_THREADS) {
      const float value = row[i];
      local.above += value > boundary ? double(value) : 0;
    }
    const auto exact = Reduce(temporary).Sum(local);
    if (threadIdx.x == 0) result = exact;
    __syncthreads();
  }
  return result;
}

#else
struct SamplingBoundaryMass {
  double above;
  uint32_t equal_count;

  __device__ SamplingBoundaryMass operator+(const SamplingBoundaryMass& rhs) const {
    return {above + rhs.above, equal_count + rhs.equal_count};
  }
};

// Float subtraction can turn an integral boundary quota into n + one ulp.
// Recheck tied boundaries in double, leaving unique-pivot scans unchanged.
template <uint32_t BLOCK_THREADS, typename DType>
__device__ __forceinline__ SamplingBoundaryMass GetSamplingBoundaryMass(const DType* row,
                                                                        uint32_t size,
                                                                        float boundary) {
  using Reduce = cub::BlockReduce<SamplingBoundaryMass, BLOCK_THREADS>;
  __shared__ typename Reduce::TempStorage temporary;
  __shared__ SamplingBoundaryMass result;
  SamplingBoundaryMass local{0, 0};
  for (uint32_t i = threadIdx.x; i < size; i += BLOCK_THREADS) {
    const float value = row[i];
    local.above += value > boundary ? double(value) : 0;
    local.equal_count += value == boundary;
  }
  const auto reduced = Reduce(temporary).Sum(local);
  if (threadIdx.x == 0) result = reduced;
  __syncthreads();
  return result;
}

#endif

// Conservative error allowance for a nonnegative float sum: at most one
// sequential addition per lane item, plus the block reduction and comparison.
template <uint32_t BLOCK_THREADS>
__device__ __forceinline__ float SamplingMassError(float mass, uint32_t size) {
  const float nu = (float((size + BLOCK_THREADS - 1) / BLOCK_THREADS) + 32.f) * 0x1p-24f;
  // Convert the usual true-sum error bound to one using the computed mass.
  return nu < 0.5f ? mass * (nu / (1.f - 2.f * nu)) : CUDART_INF_F;
}

__device__ __forceinline__ uint32_t SamplingBoundaryQuota(double target, double above,
                                                          double boundary, uint32_t count) {
  if (target <= above || boundary <= 0) return 0;
  uint32_t take = min(count, uint32_t(ceil((target - above) / boundary)));
  // Use the defining prefix condition at exact cumulative-mass boundaries.
  while (take > 0 && above + (take - 1) * boundary >= target) --take;
  while (take < count && above + take * boundary < target) ++take;
  return take;
}

// Return the exclusive original-index limit for the first `take` matching items.
// Tiles follow index order, unlike a scan of thread-strided per-thread totals.
// Callers skip this scan when the complete boundary group can be retained.
template <uint32_t BLOCK_THREADS, bool BISECT = false, typename IsTie>
__device__ __forceinline__ uint32_t FindSamplingTieIndexLimit(uint32_t size, uint32_t take,
                                                              IsTie is_tie,
                                                              uint32_t index_base = 0) {
  if (take == 0) return index_base;
  if constexpr (BISECT) {
    // AIR scans global memory over a whole vocabulary. Count successively
    // smaller contiguous halves, then scan only the final tile; this avoids
    // one block-wide prefix scan per vocabulary tile on sparse bf16 ties.
    using Reduce = cub::BlockReduce<uint32_t, BLOCK_THREADS>;
    __shared__ typename Reduce::TempStorage reduce;
    __shared__ uint32_t left_count;
    uint32_t low = 0, high = size;
    while (high - low > BLOCK_THREADS) {
      const uint32_t middle = low + (high - low) / 2;
      uint32_t local = 0;
      for (uint32_t i = low + threadIdx.x; i < middle; i += BLOCK_THREADS) local += is_tie(i);
      const uint32_t count = Reduce(reduce).Sum(local);
      if (threadIdx.x == 0) left_count = count;
      __syncthreads();
      const uint32_t total = left_count;
      __syncthreads();
      if (take <= total) {
        high = middle;
      } else {
        low = middle;
        take -= total;
      }
    }
    return FindSamplingTieIndexLimit<BLOCK_THREADS>(
        high - low, take, [is_tie, low](uint32_t i) { return is_tie(low + i); }, index_base + low);
  }
  using Scan = cub::BlockScan<uint32_t, BLOCK_THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ uint32_t limit;
  if (threadIdx.x == 0) limit = UINT32_MAX;
  __syncthreads();
  uint32_t processed = 0;
  for (uint32_t base = 0; base < size; base += BLOCK_THREADS) {
    const uint32_t index = base + threadIdx.x;
    const uint32_t match = index < size && is_tie(index);
    uint32_t prefix, total;
    Scan(scan).ExclusiveSum(match, prefix, total);
    if (match && processed + prefix == take - 1) limit = index_base + index + 1;
    __syncthreads();
    const uint32_t result = limit;
    __syncthreads();
    if (result != UINT32_MAX) return result;
    processed += total;
  }
  return UINT32_MAX;
}

}  // namespace sampling
}  // namespace flashinfer
