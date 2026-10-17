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
// problem shape; the measurements behind each rule are in benchmarks/w8a16_fp8_gemm_ladder.md
// (versions B15, B16).  Until version B18 these were overridable through environment
// variables for benchmarking; the tuned values are now the only values.

#include <cstdint>

#include "cute/arch/copy_sm90_desc.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

namespace flashinfer::gemm {

// Which axis consecutive clusters walk fastest: CUTLASS's heuristic.  "m" (weight tiles
// fastest) and "n" were both measured in version B15 and neither beat it on the grid.
inline constexpr cutlass::gemm::kernel::detail::RasterOrderOptions kW8A16Sm100Raster =
    cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;

// L2 eviction priority of the weight (A) and activation (B) TMA loads: normal for both.
// Evict-last on the weight and evict-first on the activation were tried on Wed 14 Oct 2026
// for the widest rows whose weight fits L2 (mlp.gate_up tp2, 65 MB) but is evicted by the
// activation stream; neither moved the time.
inline constexpr uint64_t kW8A16Sm100WeightCacheHint =
    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
inline constexpr uint64_t kW8A16Sm100ActCacheHint =
    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);

// Version B15 (Wed 14 Oct 2026, B200, free-running clocks).  With a swizzle S the
// in-flight clusters cover S tiles along the fast raster axis and the rest along the slow
// one, so a (weight band x activation band) working set stays in L2 instead of a line of
// tiles.  Where it matters: a weight that does not fit L2 and many activation tiles to
// re-stream it for.  mlp.gate_up tp1 (N=25600, 131 MB fp8 weight) at M=44160 / 88320 went
// from kernel / cuBLAS 1.20 / 1.18 (256x192) and 1.52 / 1.43 (256x128) to 0.99 / 1.01 and
// 1.07 / 1.04 with swizzle 8 (16 the same within noise; 2 and 4 half-way); mlp.down tp2
// (65 MB) gained 8-16% at those M.  Where it hurts: at M <= 2048 the swizzle only
// scrambles the wave packing of a grid that is a couple of waves deep, and the long-K rows
// lost 15-21% (mlp.down tp1 / tp2 at M=1024: 172 -> 209 us, 85 -> 101 us, both repeats).
// Rows whose weight fits L2 (und.q tp2, gen.qkv tp1) do not move either way at large M.
// So: grouped when the weight is at least half of L2 and there are >= 32 activation tiles,
// else the plain raster.  The sm100 scheduler does not pad the tile grid to the swizzle
// (residual tiles are simply not remapped), so no padding guard is needed.
inline int W8A16Sm100PickSwizzle(int M, int N, int K) {
  const long long weight_bytes = static_cast<long long>(N) * K;  // e4m3: one byte each
  constexpr long long kHalfL2 = 64ll << 20;
  if (weight_bytes < kHalfL2 || M < 4096) return 0;
  // Version B16: the group size scales with 1/K.  ncu on umb-b200-240 (cycles constant at
  // 19.2 M on every setting, the SM clock following the DRAM bytes one to one):
  //   mlp.down tp1  K=25600 M=88320 256x192: swizzle 2 / 4 / 8 / 16 -> DRAM 36 / 26 / 39 / 69 GB,
  //                 clock 1.22 / 1.27 / 1.20 / 1.08 GHz, time 15.7 / 15.1 / 16.0 / 17.9 ms
  //                 (cuBLAS 20.7 GB, 14.7 ms);
  //   mlp.gate_up tp2 K=5120 M=44160 256x128: 4 / 8 / 16 / 32 -> 4.1 / 3.2 / 1.9 / 3.7 GB,
  //                 time 3.82 / 3.75 / 3.69 / 3.88 ms (cuBLAS 2.4 GB, 3.56 ms).
  // The band of S weight tiles (S x 256 rows x K bytes) wants to stay around 10-26 MB:
  // S = 24 MB / (256 K) rounded to the nearest integer, clamped to [2, 16] (16 at K=5120,
  // 8 at 12800, 4 at 25600; the sm100 scheduler takes any group size).  Steady-state
  // check on mlp.down tp1 M=88320 (`b17_fine`): 3 / 4 / 5 / 6 -> 1.04 / 1.04 / 1.02 / 1.03
  // of cuBLAS, i.e. flat within noise around the DRAM minimum at 4-5.
  const long long band_bytes = 24ll << 20;
  const long long s = (band_bytes + 128ll * K) / (256ll * K);
  return static_cast<int>(s < 2 ? 2 : s > 16 ? 16 : s);
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM100_TILE_ORDER_HPP_
