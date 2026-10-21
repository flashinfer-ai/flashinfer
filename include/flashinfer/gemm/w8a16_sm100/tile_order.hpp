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
#ifndef FLASHINFER_GEMM_W8A16_SM100_TILE_ORDER_HPP_
#define FLASHINFER_GEMM_W8A16_SM100_TILE_ORDER_HPP_

// Tile walk order and TMA cache hints of the Blackwell W8A16 kernels.  Pure functions of the
// problem shape.

#include <cstdint>

#include "cute/arch/copy_sm90_desc.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "flashinfer/gemm/w8a16_common.hpp"

namespace flashinfer::gemm {

// Which axis consecutive clusters walk fastest: CUTLASS's heuristic.
inline constexpr cutlass::gemm::kernel::detail::RasterOrderOptions W8A16Sm100Raster =
    cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;

// L2 eviction priority of the weight (A) and activation (B) TMA loads: normal for both.
inline constexpr uint64_t W8A16Sm100WeightCacheHint =
    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
inline constexpr uint64_t W8A16Sm100ActCacheHint =
    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);

// L2 tile group (CUTLASS's max_swizzle_size).  With a group S the in-flight clusters cover
// S tiles along the fast raster axis and the rest along the slow one, so a (weight band x
// activation band) working set stays in L2 instead of a line of tiles.  It matters when the
// weight does not fit L2 and there are many activation tiles to re-stream it for: an fp8
// weight of >= 64 MB (half of L2) at M >= 4096 is grouped, anything else keeps the plain
// raster (on a grid only a couple of waves deep the group scrambles the wave packing).  The
// group size scales with 1/K so that the band of S weight tiles (S x 256 rows x K bytes)
// stays around 10-26 MB: S = 24 MB / (256 K) rounded to the nearest integer, clamped to
// [2, 16] (16 at K=5120, 8 at 12800, 4 at 25600).  The sm100 scheduler does not pad the
// tile grid to the group (residual tiles are simply not remapped), so no padding guard is
// needed.
inline int W8A16Sm100PickSwizzle(int M, int N, int K) {
  const long long weight_bytes = static_cast<long long>(N) * K;  // e4m3: one byte each
  constexpr long long HalfL2 = 64ll << 20;
  if (weight_bytes < HalfL2 || M < 4096) return 0;
  const long long band_bytes = 24ll << 20;
  const long long s = (band_bytes + 128ll * K) / (256ll * K);
  return static_cast<int>(s < 2 ? 2 : s > 16 ? 16 : s);
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM100_TILE_ORDER_HPP_
