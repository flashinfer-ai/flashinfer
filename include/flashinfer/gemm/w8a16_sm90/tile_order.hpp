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
// benchmarks/w8a16_fp8_gemm_ladder.md (versions H1, H3, H4).  Until version H10 these were
// overridable through environment variables for benchmarking; the tuned values are now the
// only values.

#include "cutlass/gemm/kernel/tile_scheduler.hpp"

namespace flashinfer::gemm {

// Raster "m": consecutive CTAs walk the weight tiles of one activation tile (the activation
// tile stays in L2, the weight streams).  CUTLASS's heuristic picks "n" (walk the activation
// tiles of one weight tile) whenever there are at least as many weight tiles as activation
// tiles (N >= M/2), and for this operand layout that is 2x slower once the activation
// outgrows L2: mlp.gate_up tp1 M=44160 31.1 vs 14.9 ms, mlp.down tp1 33.2 vs 14.5, gen.qkv
// tp1 12.9 vs 5.8 (ncu, locked clocks, Tue 13 Oct 2026); "m" is never worse than the
// heuristic on the 144-row table.
inline constexpr cutlass::gemm::kernel::detail::RasterOrderOptions kW8A16Sm90Raster =
    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;

// L2 tile-group size handed to CUTLASS's own schedulers (the stream-K variants and any
// cluster-2 tile): 1, i.e. none.  CUTLASS's persistent and stream-K schedulers round BOTH
// tile counts up to a multiple of (swizzle x cluster) and run the padding tiles (TMA
// zero-fills them, the epilogue drops them; see round_up in tile_scheduler_params.h).  With
// swizzle 8 that is 10 -> 16 weight tiles on N=1280 (ncu, Tue 13 Oct 2026: 1.6x the HGMMA
// count, 247 vs 162 us) and 4 -> 8 activation tiles on M=1024.  Version H1 had made 8 the
// default because it recovered 12% on gen.qkv tp1 M=8192; that was the heuristic's "n"
// raster being papered over by the grouping.  With raster "m" the group buys nothing (1.046
// vs 1.054 ms on that shape, equal on five others), so it is off; the plain persistent
// variants get their L2 grouping from W8A16Sm90GroupedTileScheduler instead, which does not
// pad (W8A16Sm90PickGroup below).
inline constexpr int kW8A16Sm90CutlassSwizzle = 1;

// L2 group of the grouped scheduler (version H4, ncu locked-clock sweep of the 48 rows at
// M = 44160 / 88320, Wed 14 Oct 2026): ungrouped, the 132 CTAs in flight cover ~one
// activation tile and every activation tile re-reads the whole weight; that only matters once
// the weight is a large share of the 50 MB L2.  N*K >= 40 MB: 4 activation tiles per group
// when there are >= 64 weight tiles (und.q / gen.qkv / mlp.gate_up tp1, mlp.gate_up tp2:
// DRAM reads 2.0-2.7x lower, 1-3% faster), else 2 (mlp.down tp1 / tp2, attn.out tp1).
// Smaller weights stay ungrouped: there the group only adds activation re-reads (und.q tp4
// 0.67 -> 0.87 GB at group 4).  M below 16384 stays ungrouped too: at M = 4096 / 8192 the
// group cuts DRAM reads 20-30% but the time does not move (+-1%), so those rows keep H3's
// exact schedule.
inline int W8A16Sm90PickGroup(int M, int N, int K, int tile_w) {
  const long long weight_bytes = 1LL * N * K;
  const int tiles_w = (N + tile_w - 1) / tile_w;
  if (weight_bytes < 40LL * 1000 * 1000 || M < 16384) return 1;
  return tiles_w >= 64 ? 4 : 2;
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM90_TILE_ORDER_HPP_
