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
#ifndef FLASHINFER_GEMM_W8A16_SM90_TILE_ORDER_HPP_
#define FLASHINFER_GEMM_W8A16_SM90_TILE_ORDER_HPP_

// Tile walk order of the Hopper persistent schedulers: which operand's tiles consecutive
// CTAs take, and how many tiles are grouped so that the operand slab they share stays in L2.
// Pure functions of the problem shape; the measurements behind each rule are in
// benchmarks/w8a16_fp8_gemm_ladder.md.

#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "flashinfer/gemm/w8a16_common.hpp"

namespace flashinfer::gemm {

// Raster "m": consecutive CTAs walk the weight tiles of one activation tile (the activation
// tile stays in L2, the weight streams).  CUTLASS's heuristic picks "n" (walk the activation
// tiles of one weight tile) whenever there are at least as many weight tiles as activation
// tiles (N >= M/2), and for this operand layout that is 2x slower once the activation
// outgrows L2; "m" is never worse than the heuristic on the benchmark grid.
inline constexpr cutlass::gemm::kernel::detail::RasterOrderOptions kW8A16Sm90Raster =
    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;

// L2 tile-group size handed to CUTLASS's own schedulers (the stream-K variants and any
// cluster-2 tile): 1, i.e. none.  CUTLASS's persistent and stream-K schedulers round BOTH
// tile counts up to a multiple of (swizzle x cluster) and run the padding tiles (TMA
// zero-fills them, the epilogue drops them; see round_up in tile_scheduler_params.h): with
// swizzle 8 that is 10 -> 16 weight tiles on N=1280, 1.6x the MMA work.  With raster "m" the
// group buys nothing on top, so it is off; the plain persistent variants get their L2
// grouping from W8A16Sm90GroupedTileScheduler instead, which does not pad
// (W8A16Sm90PickGroup below).
inline constexpr int kW8A16Sm90CutlassSwizzle = 1;

// L2 group of the grouped scheduler.  Ungrouped, the 132 CTAs in flight cover about one
// activation tile and every activation tile re-reads the whole weight; that only matters
// once the weight is a large share of the 50 MB L2 and there are many activation tiles.
// Weight >= 40 MB and M >= 16384: 4 activation tiles per group when there are >= 64 weight
// tiles, else 2 (DRAM reads 2.0-2.7x lower, 1-3% faster).  Smaller weights stay ungrouped
// (the group only adds activation re-reads), and so does M < 16384 (the group cuts DRAM
// reads 20-30% there but the time does not move).
inline int W8A16Sm90PickGroup(int M, int N, int K, int tile_w) {
  const long long weight_bytes = 1LL * N * K;
  const long long tiles_w = W8A16Tiles(N, tile_w);
  if (weight_bytes < 40LL * 1000 * 1000 || M < 16384) return 1;
  return tiles_w >= 64 ? 4 : 2;
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM90_TILE_ORDER_HPP_
