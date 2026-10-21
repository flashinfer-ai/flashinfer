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
#ifndef FLASHINFER_GEMM_W8A16_SM90_DISPATCH_HPP_
#define FLASHINFER_GEMM_W8A16_SM90_DISPATCH_HPP_

// Variant selection for the Hopper W8A16 GEMM: the public `algo` codes, the split-K slice
// count, the cost model that picks a variant per problem, and the one table that maps a
// variant to its W8A16Sm90Gemm.  Policy only; the kernels are in gemm.hpp.
//
// Sizing and execution go through the same W8A16Sm90Plan (W8A16Sm90Resolve) and the same
// variant table (W8A16Sm90Visit), so the workspace query and the launch agree on the
// variant, on whether the request is valid, and on the kernel behind the code.

#include <algorithm>
#include <cmath>

#include "flashinfer/gemm/w8a16_sm90/gemm.hpp"

namespace flashinfer::gemm {

// Hopper `algo` codes of the Python op (`flashinfer.gemm.w8a16_fp8_gemm(..., algo=...)`
// on sm_90a); they must not change.  0 = the launcher's choice, 2.. = one CUTLASS variant.
// Code 1 is reserved and rejected by the launcher.
enum class W8A16Sm90Algo {
  Auto = 0,
  Reserved1 = 1,
  Tile128x256 = 2,
  Tile128x192 = 3,
  Tile128x160 = 4,
  Tile128x128 = 5,
  Tile64x256 = 6,
  Tile128x256StreamK = 7,
  Tile128x192StreamK = 8,
  Tile64x128 = 9,
  Tile128x128SplitK = 10,
  Tile128x256SplitK = 11,
};
constexpr int W8A16Sm90MaxAlgoCode = 11;

inline bool W8A16Sm90IsSplitK(W8A16Sm90Algo a) {
  return a == W8A16Sm90Algo::Tile128x128SplitK || a == W8A16Sm90Algo::Tile128x256SplitK;
}

// Split-K slicing: a whole number of 64-deep k-tiles per slice, all slices equal, at least
// W8A16Sm90::MinKTilesPerSplit k-tiles each, at most W8A16Sm90::MaxSplits slices.  The count
// comes from the cost model (W8A16Sm90SplitUs).

// Largest legal split <= want that divides the k-tile count (the batched problem needs
// equal slices); 1 when K has fewer than 2 * W8A16Sm90::MinKTilesPerSplit k-tiles.
inline int W8A16Sm90LegalSplits(int K, int want) {
  const int ktiles = K / W8A16Sm90::TileK;
  if (K % W8A16Sm90::TileK != 0) return 1;
  for (int s = std::min(want, W8A16Sm90::MaxSplits); s >= 2; --s) {
    if (ktiles % s == 0 && ktiles / s >= W8A16Sm90::MinKTilesPerSplit) return s;
  }
  return 1;
}

// Cost model for M <= 8192 over the CUTLASS menu.  Each candidate's time, in us, from three
// terms:
//  * per tile: c_tile * K (+ an epilogue of 2 us x activation rows / 256 for the cooperative
//    tiles; the ping-pong 64-row tiles overlap it with the sibling tile's mainloop).  c_tile
//    is the time per unit of K of one tile alone on an SM (W8A16Sm90TileCosts below; a
//    ping-pong SM runs two of its tiles in that time, so its tile count is still divided by
//    the SM count, not twice it).
//  * plain tiles: ceil(tiles / SMs) waves of that.
//  * stream-K: CUTLASS's own decomposition (get_num_sk_tiles / get_num_sk_units): data-
//    parallel when there is no quantisation, K has <= 8 k-tiles, or the tail is >= half a
//    wave; else the last up-to-two waves of tiles are cut into equal k-iteration units over
//    min(SMs, sk k-tiles / 8) CTAs, at 1.1x the per-iteration cost, + 8 us fixed + 2 us per
//    peer per tile (the fix-up).
//  * split-K s (equal slices of >= 8 k-tiles, <= 8): ceil(tiles x s / SMs) waves of a K/s
//    tile + 2 us + 2e-6 us per fp32 partial element (write + reduce read); the partials are
//    capped at W8A16Sm90::MaxPartialsBytes.
struct W8A16Sm90TileCost {
  W8A16Sm90Algo algo;  // the plain variant of this tile
  int tile_w, tile_a;
  double ns_per_k;  // c_tile
  bool pingpong;
};

// The tiles the cost model ranks, in candidate order (ties go to the earlier entry).
inline constexpr W8A16Sm90TileCost W8A16Sm90TileCosts[] = {
    {W8A16Sm90Algo::Tile128x256, W8A16Sm90::CooperativeW, W8A16Sm90::TileA256, 11.37, false},
    {W8A16Sm90Algo::Tile128x192, W8A16Sm90::CooperativeW, W8A16Sm90::TileA192, 9.00, false},
    {W8A16Sm90Algo::Tile128x128, W8A16Sm90::CooperativeW, W8A16Sm90::TileA128, 7.45, false},
    {W8A16Sm90Algo::Tile64x256, W8A16Sm90::PingpongW, W8A16Sm90::TileA256, 6.15, true},
    {W8A16Sm90Algo::Tile64x128, W8A16Sm90::PingpongW, W8A16Sm90::TileA128, 4.50, true}};

// The ranked tile a variant runs on (its own for a plain variant, the underlying tile for a
// stream-K or split-K one).  Only for variants the cost model ranks.
inline const W8A16Sm90TileCost& W8A16Sm90TileOf(W8A16Sm90Algo a) {
  switch (a) {
    case W8A16Sm90Algo::Tile128x256StreamK:
    case W8A16Sm90Algo::Tile128x256SplitK:
      return W8A16Sm90TileCosts[0];
    case W8A16Sm90Algo::Tile128x192StreamK:
      return W8A16Sm90TileCosts[1];
    case W8A16Sm90Algo::Tile128x128SplitK:
      return W8A16Sm90TileCosts[2];
    default:
      for (const auto& t : W8A16Sm90TileCosts) {
        if (t.algo == a) return t;
      }
      return W8A16Sm90TileCosts[0];
  }
}

inline double W8A16Sm90TileUs(const W8A16Sm90TileCost& t, double k_len) {
  const double epi = t.pingpong ? 0.0 : 2.0 * t.tile_a / W8A16Sm90::TileA256;
  return t.ns_per_k * 1e-3 * k_len + epi;
}

// Output tiles of a tile shape, in double: the admitted M and N reach INT32_MAX each.
inline double W8A16Sm90TilesOf(const W8A16Sm90TileCost& t, int M, int N) {
  return static_cast<double>(W8A16Tiles(N, t.tile_w)) *
         static_cast<double>(W8A16Tiles(M, t.tile_a));
}

// Modelled time of tile `t` under split-K with s slices; 1e30 when the partials exceed the
// cap (compared in double: 4 * M * N * s overflows 64 bits at the admitted dimensions).
inline double W8A16Sm90SplitUs(const W8A16Sm90TileCost& t, int M, int N, int K, int num_sms,
                               int s) {
  if (4.0 * M * N * s > static_cast<double>(W8A16Sm90::MaxPartialsBytes)) return 1e30;
  return std::ceil(W8A16Sm90TilesOf(t, M, N) * s / std::max(num_sms, 1)) *
             W8A16Sm90TileUs(t, static_cast<double>(K) / s) +
         2.0 + 2e-6 * static_cast<double>(M) * N * s;
}

// Split count for a split-K variant over tile `t`: the s with the smallest modelled time
// among the legal splits; 1 when none is legal.
inline int W8A16Sm90PickSplits(const W8A16Sm90TileCost& t, int M, int N, int K, int num_sms) {
  int best = 1;
  double best_us = 1e30;
  for (int s = 2; s <= W8A16Sm90::MaxSplits; ++s) {
    if (W8A16Sm90LegalSplits(K, s) != s) continue;
    const double us = W8A16Sm90SplitUs(t, M, N, K, num_sms, s);
    if (us < best_us) {
      best_us = us;
      best = s;
    }
  }
  return best;
}

// The split actually run by a split-K variant: the model's, or when it is forced through
// `algo` on a grid the model would not split, the smallest legal s >= 2; 1 = cannot split.
inline int W8A16Sm90SplitsFor(const W8A16Sm90TileCost& t, int M, int N, int K, int num_sms) {
  const int s = W8A16Sm90PickSplits(t, M, N, K, num_sms);
  if (s >= 2) return s;
  for (int c = 2; c <= W8A16Sm90::MaxSplits; ++c)
    if (W8A16Sm90LegalSplits(K, c) == c) return c;
  return 1;
}

inline W8A16Sm90Algo W8A16Sm90PickByCost(int M, int N, int K, int num_sms) {
  using A = W8A16Sm90Algo;
  const double sms = std::max(num_sms, 1);
  const int ktiles = K / W8A16Sm90::TileK;
  A best = A::Tile128x256;
  double best_us = 1e30;
  auto consider = [&](A a, double us) {
    if (us < best_us) {
      best_us = us;
      best = a;
    }
  };
  for (const auto& t : W8A16Sm90TileCosts)
    consider(t.algo, std::ceil(W8A16Sm90TilesOf(t, M, N) / sms) * W8A16Sm90TileUs(t, K));
  // Stream-K over 128x256 / 128x192.
  for (A a : {A::Tile128x256StreamK, A::Tile128x192StreamK}) {
    const auto& t = W8A16Sm90TileOf(a);
    const double tiles = W8A16Sm90TilesOf(t, M, N);
    const double per_tile = W8A16Sm90TileUs(t, K);
    const double full = std::floor(tiles / sms), total = std::ceil(tiles / sms);
    const double tail = tiles - full * sms;
    double us;
    if (full == total || ktiles <= 8 || (full >= 1 && 2 * tail >= sms)) {
      us = total * per_tile;
    } else {
      const double dp_waves = full > 1 ? full - 1 : 0;
      const double sk_tiles = tiles - dp_waves * sms;
      const double units = std::max(1.0, std::min(sms, std::floor(sk_tiles * ktiles / 8.0)));
      us = dp_waves * per_tile + sk_tiles * ktiles / units * (per_tile / ktiles) * 1.1 + 8.0 +
           2.0 * units / sk_tiles;
    }
    consider(a, us);
  }
  // Split-K over 128x128 / 128x256.
  for (A a : {A::Tile128x128SplitK, A::Tile128x256SplitK}) {
    const auto& t = W8A16Sm90TileOf(a);
    for (int s = 2; s <= W8A16Sm90::MaxSplits; ++s) {
      if (W8A16Sm90LegalSplits(K, s) != s) continue;
      consider(a, W8A16Sm90SplitUs(t, M, N, K, num_sms, s));
    }
  }
  return best;
}

// The launcher's choice for a problem.  M <= 8192: the cost model above.  Above that,
// wave-count rules over the 128x256 tile grid: 128x256 when the last wave is >= 0.8 full,
// stream-K for a sliver of a last wave, the 64x256 ping-pong tile at short K or under 3
// waves, 128x192 for N <= 512, all with the padding-free L2 grouping of tile_order.hpp.
inline W8A16Sm90Algo W8A16Sm90PickAlgo(int M, int N, int K, int num_sms) {
  const long long units =
      W8A16Tiles(N, W8A16Sm90::CooperativeW) * W8A16Tiles(M, W8A16Sm90::TileA256);
  const double waves = static_cast<double>(units) / std::max(num_sms, 1);
  const double frac = waves - std::floor(waves);
  if (M <= 8192) return W8A16Sm90PickByCost(M, N, K, num_sms);
  if (N <= 512) {
    if (waves < 0.75) return W8A16Sm90Algo::Tile128x128;
    // 2-4 weight tiles at large M: the 128x192 tile.
    if (M >= 16384) return W8A16Sm90Algo::Tile128x192;
    return W8A16Sm90Algo::Tile128x256;
  }
  // Short K at large M (16 k-tiles per tile): the 64x256 ping-pong tile, whose two tiles in
  // flight per SM hide the epilogue behind the short mainloop; at long K the single-tile
  // cooperative mainloop wins.  The weight is small here, so W8A16Sm90PickGroup leaves it
  // ungrouped.
  if (K <= 1024 && M >= 16384) return W8A16Sm90Algo::Tile64x256;
  if (waves <= 1.5) {
    if (K >= 4096 && waves >= 0.7) return W8A16Sm90Algo::Tile128x256StreamK;
    if (N <= 2048 || K <= 3200) return W8A16Sm90Algo::Tile64x256;
    return W8A16Sm90Algo::Tile128x256;
  }
  if (frac >= 0.8) return W8A16Sm90Algo::Tile128x256;
  // A sliver of a last wave is worth spreading along K only while it is a visible share
  // of the run (0.1 of 262 waves is not; stream-K's fix-up traffic is).
  if (frac < 0.15 && K >= 2048 && waves <= 16.0) return W8A16Sm90Algo::Tile128x256StreamK;
  if (waves <= 3.0) return W8A16Sm90Algo::Tile64x256;
  return W8A16Sm90Algo::Tile128x256;
}

// What runs for one problem: the variant and, for the split-K variants, the slice count.
struct W8A16Sm90Plan {
  W8A16Sm90Algo algo;
  int splits;  // >= 2 for the split-K variants, 1 otherwise
};

// Resolves the caller's `algo` code (0 = the launcher's choice, 2..W8A16Sm90MaxAlgoCode =
// that variant) and the problem to the plan both the workspace query and the launch use.
//   cudaErrorInvalidValue: `algo_code` is not a variant (1 is reserved).
//   cudaErrorNotSupported: a forced split-K variant on a K that cannot be sliced (see
//                          W8A16Sm90LegalSplits); the launcher's own choice never is.
inline cudaError_t W8A16Sm90Resolve(int M, int N, int K, int num_sms, int64_t algo_code,
                                    W8A16Sm90Plan* plan) {
  if (algo_code == 0) {
    plan->algo = W8A16Sm90PickAlgo(M, N, K, num_sms);
  } else if (algo_code >= 2 && algo_code <= W8A16Sm90MaxAlgoCode) {
    plan->algo = static_cast<W8A16Sm90Algo>(algo_code);
  } else {
    return cudaErrorInvalidValue;
  }
  plan->splits = 1;
  if (W8A16Sm90IsSplitK(plan->algo)) {
    plan->splits = W8A16Sm90SplitsFor(W8A16Sm90TileOf(plan->algo), M, N, K, num_sms);
    if (plan->splits < 2) return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

// The one algo -> kernel table: calls `f(W8A16VariantTag<V>{})` with the W8A16Sm90Gemm of
// the variant and returns its cudaError_t; cudaErrorInvalidValue for a code that is no
// variant.
template <typename T, typename F>
cudaError_t W8A16Sm90Visit(W8A16Sm90Algo algo, F&& f) {
  using D = W8A16Decomposition;
  switch (algo) {
    case W8A16Sm90Algo::Tile128x256:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256>>{});
    case W8A16Sm90Algo::Tile128x192:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x192>>{});
    case W8A16Sm90Algo::Tile128x160:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x160>>{});
    case W8A16Sm90Algo::Tile128x128:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x128>>{});
    case W8A16Sm90Algo::Tile64x256:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg64x256>>{});
    case W8A16Sm90Algo::Tile64x128:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg64x128>>{});
    case W8A16Sm90Algo::Tile128x256StreamK:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256, D::StreamK>>{});
    case W8A16Sm90Algo::Tile128x192StreamK:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x192, D::StreamK>>{});
    case W8A16Sm90Algo::Tile128x128SplitK:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x128, D::SplitK>>{});
    case W8A16Sm90Algo::Tile128x256SplitK:
      return f(W8A16VariantTag<W8A16Sm90Gemm<T, W8A16Sm90Cfg128x256, D::SplitK>>{});
    default:
      return cudaErrorInvalidValue;
  }
}

// Workspace bytes of the plan's variant for this problem.  cudaErrorInvalidValue when the
// size leaves the signed 64-bit range (see W8A16WorkspaceOf).
template <typename T>
cudaError_t W8A16Fp8GemmSm90WorkspaceBytes(const W8A16Sm90Plan& plan, int M, int N, int K,
                                           int device_id, int num_sms, size_t* bytes) {
  return W8A16Sm90Visit<T>(plan.algo, [&](auto tag) {
    using V = typename decltype(tag)::type;
    return W8A16WorkspaceBytesOf<V>(M, N, K, device_id, num_sms, V::IsSplitK ? plan.splits : 1,
                                    bytes);
  });
}

// Runs the plan's variant (see W8A16RunVariant for the errors).
template <typename T>
cudaError_t W8A16Fp8GemmSm90(const W8A16Sm90Plan& plan, const T* act, const __nv_fp8_e4m3* weight,
                             const float* scale, const T* bias, T* out, int M, int N, int K,
                             void* workspace, size_t workspace_bytes, int device_id, int num_sms,
                             cudaStream_t stream) {
  return W8A16Sm90Visit<T>(plan.algo, [&](auto tag) {
    using V = typename decltype(tag)::type;
    return W8A16RunVariant<V>(act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes,
                              device_id, num_sms, stream, V::IsSplitK ? plan.splits : 1);
  });
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM90_DISPATCH_HPP_
