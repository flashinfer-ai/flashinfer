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
#ifndef FLASHINFER_GEMM_W8A16_SM100_DISPATCH_HPP_
#define FLASHINFER_GEMM_W8A16_SM100_DISPATCH_HPP_

// Variant selection for the Blackwell W8A16 GEMM: the public `algo` codes, the split-K and
// stream-K rules, the wave-fill cost model that picks a tile per problem, and the switch
// tables that instantiate the chosen W8A16Sm100Gemm.  Policy only; the kernels are in
// gemm.hpp.

#include <algorithm>

#include "flashinfer/gemm/w8a16_sm100/gemm.hpp"

namespace flashinfer::gemm {

// Tile codes.  The numeric values are the public `algo` codes of the Python op
// (`flashinfer.gemm.w8a16_fp8_gemm(..., algo=...)`), so they must not change.
enum class W8A16Sm100Tile {
  kAuto,
  k2SM256x128 = 1,
  k1SM128x128 = 2,
  k2SM256x256 = 3,
  k2SM256x192 = 4,
  k2SM256x128SplitK = 5,   // version B12: 256x128 tile, batched K-slices + fp32 reduce
  k2SM256x128StreamK = 6,  // version B13: 256x128 tile under the stream-K scheduler
  k2SM256x160 = 7          // version B14: 160-row tile for wave quantisation
  // Codes 8-13 were the B16 / B18 / B20 experiments (cluster (2,2,1) weight multicast on
  // the 128 / 192 tiles, a 224-row tile with a 64-deep k-tile, SMEM-staged A for the 192 /
  // 128 tiles, a 160-row tile with a 64-deep k-tile); all measured slower than the tiles
  // above on every row and were removed from the build (ladder doc, "losers").
};
constexpr int kW8A16Sm100MaxTileCode = 7;
constexpr int kW8A16Sm100MaxSplits = 8;

inline W8A16Sm100Tile W8A16Sm100TileFromCode(int code) {
  return (code >= 1 && code <= kW8A16Sm100MaxTileCode) ? static_cast<W8A16Sm100Tile>(code)
                                                       : W8A16Sm100Tile::kAuto;
}

// Split count for the 256x128 2SM tile (version B12).  Two cases split, both from the
// back-to-back per-M sweeps of Mon 12 Oct 2026:
//   1. an under-filled grid, fewer cluster tiles than half the resident clusters:
//      the largest split that still fits in one wave (every cluster gets at most one
//      K-slice; the und.kv rows, 8..64 tiles on 74 clusters, run 1.3-1.8x faster);
//   2. a long K (>= 100 k-tiles) whose partial last wave the split removes: the
//      smallest split that keeps >= 50 k-tiles per slice and cuts the modelled
//      mainloop (waves x k-tiles per slice) by at least 15% (mlp.down at M=1024,
//      K=25600: 160 tiles = 2.2 waves, 8% faster split 4-5 ways).
// Short-K problems never split: the fp32 partials (one M x N write and read per slice)
// cost more than the fill buys (attn.out / gen.qkv / mlp.down at K <= 8192 lose 5-25%).
// Every K-slice is a whole number of 128-deep k-tiles (all batches share one slice
// length) and at most kW8A16Sm100MaxSplits slices.  1 = do not split.
inline int W8A16Sm100SplitK(int M, int N, int K, int num_sms) {
  if (K % 128 != 0) return 1;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long units = W8A16Sm100Tiles(N, 256) * W8A16Sm100Tiles(M, 128);
  const int ktiles = K / 128;
  if (units * 2 <= clusters) {
    int best = 1;
    for (int s = 2; s <= kW8A16Sm100MaxSplits; ++s) {
      if (ktiles % s != 0 || ktiles / s < 2 || units * s > clusters) continue;
      best = s;
    }
    return best;
  }
  if (ktiles >= 100) {
    const long long waves1 = (units + clusters - 1) / clusters;
    for (int s = 2; s <= kW8A16Sm100MaxSplits; ++s) {
      if (ktiles % s != 0 || ktiles / s < 50) continue;
      const long long waves = (units * s + clusters - 1) / clusters;
      if (waves * (ktiles / s) * 100 <= waves1 * ktiles * 85) return s;
    }
  }
  return 1;
}

// Split count actually used by the split-K variant: the model's choice, or, when the
// variant is forced through `algo` on a grid the model would not split, the smallest
// s >= 2 that divides the k-tile count.  0 = K cannot be split (fewer than 2 k-tiles).
inline int W8A16Sm100SplitsFor(int M, int N, int K, int num_sms) {
  const int s = W8A16Sm100SplitK(M, N, K, num_sms);
  if (s >= 2) return s;
  const int ktiles = K / 128;
  if (K % 128 != 0 || ktiles < 2) return 0;
  for (int c = 2; c <= kW8A16Sm100MaxSplits; ++c) {
    if (ktiles % c == 0) return c;
  }
  return 0;
}

// Version B13: stream-K pays only where the tail wave is most of the grid and the tiles
// are long.  Every K fragment pays a pipeline prologue and a 128 KB fp32 accumulator
// round trip through global memory, about as much as 8-10 k-tiles of mainloop, so a
// tile has to be >= 64 k-tiles deep before cutting it is cheaper than idling (measured
// Tue 13 Oct 2026: N=5120 at M=1024, 160 tiles = 2.16 waves: K=2048 1.53x slower,
// K=5120 1.04x, K=8192 0.93x, K=12800 0.87x, K=25600 0.82x against the plain tile).
// With four or more waves the saved fraction of a wave is too small (M=2048 rows lose
// 5-11%), and under-filled grids (<= half the clusters busy) belong to split-K, whose
// batched slices skip the per-fragment handshake.  So: more than half the clusters
// busy, at most three waves with the last one at most a third full, >= 64 k-tiles.
inline bool W8A16Sm100UseStreamK(int M, int N, int K, int num_sms) {
  if (K % 128 != 0 || K / 128 < 64) return false;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long units = W8A16Sm100Tiles(N, 256) * W8A16Sm100Tiles(M, 128);
  if (units * 2 <= clusters) return false;
  const long long waves = (units + clusters - 1) / clusters;
  const long long tail = units - (waves - 1) * clusters;  // tiles in the last wave
  return waves >= 2 && waves <= 3 && tail * 3 <= clusters;
}

// `algo` is the caller's tile code (0 = auto); an explicit code wins over the cost model.
inline W8A16Sm100Tile W8A16Sm100PickTile(int M, int N, int K, int num_sms, int algo = 0) {
  const W8A16Sm100Tile forced = W8A16Sm100TileFromCode(algo);
  if (forced != W8A16Sm100Tile::kAuto) return forced;
  // Version B17: a weight of <= 128 rows (und.kv tp8, N=128) fills only half of the 2SM
  // tile's 256 weight rows, so every 2SM variant runs twice the MMA for the same bytes;
  // the 1SM 128x128 tile has no padding there.  M=88320, 5 interleaved rounds, kernel /
  // cuBLAS: auto (256x128) 1.09-1.13, 1SM 128x128 1.00-1.01, split-K 1.20-1.27, stream-K
  // 1.13-1.20 (`flush_*_undkv8`).  Small M on that row stays with split-K (B12).
  if (N <= 128 && M >= 16384) return W8A16Sm100Tile::k1SM128x128;
  // Version B13: a mostly-empty last wave over deep tiles is balanced by stream-K.
  if (W8A16Sm100UseStreamK(M, N, K, num_sms)) return W8A16Sm100Tile::k2SM256x128StreamK;
  // Version B12: under-filled grids and long-K partial waves split K instead of choosing
  // between tiles; see W8A16Sm100SplitK.
  if (W8A16Sm100SplitK(M, N, K, num_sms) > 1) return W8A16Sm100Tile::k2SM256x128SplitK;

  // Version B11 refit (native converter, version B10): with the convert off the critical
  // path the 192-row tile's advantage shrinks to a few percent (eff 1.22 -> 1.02) and the
  // 256x256 smem-A tile loses on every one of the 144 benchmark rows, so it is no
  // longer picked (it stays reachable through `algo`).  Fitted on three back-to-back
  // sweeps of all tiles (geomean model pick / per-row oracle 1.025, the B9 constants
  // gave 1.040).  The 1SM 128x128 tile never wins once the per-M classes are measured
  // in isolation (its apparent wins were thermal noise of the full-table sweep), so it
  // is `algo`-only as well.
  //
  // Version B14: the 160-row tile joins the menu as a third tile *height*, so the model
  // can land the tile count near a whole number of waves (the taller tile's extra rows
  // in the last M-tile are masked for free).  Its factor (0.96) was fitted on the
  // back-to-back M=1024 / 2048 sweeps of Tue 13 Oct 2026 together with the B13 all-M
  // 128/192 data: geomean model pick / oracle 1.010; any value in 0.94-1.00 scores the
  // same, 1.02 for the 192-row tile keeps the large-M picks of B11.  On the N=5120
  // M=1024 rows (160 tiles of 128 rows = 2.16 waves vs 140 tiles of 160 rows = 1.89
  // waves) the 160-row tile is 11-16% faster; it also takes several M=2048 rows with
  // N <= 4096.
  (void)K;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long w_tiles = W8A16Sm100Tiles(N, 256);
  auto cost = [&](int tile_a, double eff) {
    const long long units = w_tiles * W8A16Sm100Tiles(M, tile_a);
    return static_cast<double>((units + clusters - 1) / clusters) * tile_a / eff;
  };
  const double c128 = cost(128, 1.00);
  const double c160 = cost(160, 0.96);
  const double c192 = cost(192, 1.02);
  if (c192 <= c128 && c192 <= c160) return W8A16Sm100Tile::k2SM256x192;
  if (c160 < c128) return W8A16Sm100Tile::k2SM256x160;
  return W8A16Sm100Tile::k2SM256x128;
}

template <typename T>
size_t W8A16Fp8GemmSm100WorkspaceBytes(int M, int N, int K, int device_id, int num_sms,
                                       int algo = 0) {
  switch (W8A16Sm100PickTile(M, N, K, num_sms, algo)) {
    case W8A16Sm100Tile::k2SM256x128SplitK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, true>::WorkspaceBytes(
          M, N, K, device_id, num_sms, std::max(2, W8A16Sm100SplitsFor(M, N, K, num_sms)));
    case W8A16Sm100Tile::k2SM256x128StreamK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, false, true>::WorkspaceBytes(
          M, N, K, device_id, num_sms);
    case W8A16Sm100Tile::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Tile::k2SM256x256:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Tile::k2SM256x192:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Tile::k2SM256x160:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x160>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    default:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
  }
}

template <typename T>
cudaError_t W8A16Fp8GemmSm100(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                              const T* bias, T* out, int M, int N, int K, void* workspace,
                              size_t workspace_bytes, int device_id, int num_sms,
                              cudaStream_t stream, int algo = 0) {
  switch (W8A16Sm100PickTile(M, N, K, num_sms, algo)) {
    case W8A16Sm100Tile::k2SM256x128SplitK: {
      const int splits = W8A16Sm100SplitsFor(M, N, K, num_sms);
      if (splits < 2) return cudaErrorNotSupported;
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, true>::Run(
          act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes, device_id, num_sms,
          stream, splits);
    }
    case W8A16Sm100Tile::k2SM256x128StreamK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, false, true>::Run(
          act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes, device_id, num_sms,
          stream);
    case W8A16Sm100Tile::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Tile::k2SM256x256:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Tile::k2SM256x192:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Tile::k2SM256x160:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x160>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    default:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
  }
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM100_DISPATCH_HPP_
