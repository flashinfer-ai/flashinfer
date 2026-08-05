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

#include <cstdint>
#include <cub/cub.cuh>

namespace flashinfer {
namespace sampling {

struct SamplingBoundaryMass {
  double above, total;
  uint32_t equal_count;

  __device__ SamplingBoundaryMass operator+(const SamplingBoundaryMass& rhs) const {
    return {above + rhs.above, total + rhs.total, equal_count + rhs.equal_count};
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
  SamplingBoundaryMass local{0, 0, 0};
  for (uint32_t i = threadIdx.x; i < size; i += BLOCK_THREADS) {
    const float value = row[i];
    local.above += value > boundary ? double(value) : 0;
    local.total += double(value);
    local.equal_count += value == boundary;
  }
  const auto reduced = Reduce(temporary).Sum(local);
  if (threadIdx.x == 0) result = reduced;
  __syncthreads();
  return result;
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
template <uint32_t BLOCK_THREADS, typename IsTie>
__device__ __forceinline__ uint32_t FindSamplingTieIndexLimit(uint32_t size, uint32_t take,
                                                              IsTie is_tie,
                                                              uint32_t index_base = 0) {
  if (take == 0) return index_base;
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
