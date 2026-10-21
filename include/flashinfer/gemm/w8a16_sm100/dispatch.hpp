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
// stream-K rules, the wave-fill cost model that picks a tile per problem, and the one table
// that maps a variant to its W8A16Sm100Gemm.  Policy only; the kernels are in gemm.hpp.
//
// Sizing and execution go through the same W8A16Sm100Plan (W8A16Sm100Resolve) and the same
// variant table (W8A16Sm100Visit), so the workspace query and the launch agree on the
// variant, on whether the request is valid, and on the kernel behind the code.

#include <algorithm>

#include "flashinfer/gemm/w8a16_sm100/gemm.hpp"

namespace flashinfer::gemm {

// Blackwell `algo` codes of the Python op (`flashinfer.gemm.w8a16_fp8_gemm(..., algo=...)`
// on sm_100a); they must not change.  0 = the launcher's choice, 1.. = one variant (a tile
// and, for 5 / 6, a decomposition).  Codes 8-13 are reserved; never reuse them.
enum class W8A16Sm100Algo {
  Auto = 0,
  TwoSM256x128 = 1,
  OneSM128x128 = 2,
  TwoSM256x256 = 3,
  TwoSM256x192 = 4,
  TwoSM256x128SplitK = 5,
  TwoSM256x128StreamK = 6,
  TwoSM256x160 = 7,
};
constexpr int W8A16Sm100MaxAlgoCode = 7;
constexpr int W8A16Sm100MaxSplits = 8;

// Split count for the 256x128 2SM tile.  Two cases split:
//   1. an under-filled grid, fewer cluster tiles than half the resident clusters: the
//      largest split that still fits in one wave (every cluster gets at most one K-slice);
//   2. a long K (>= 100 k-tiles) whose partial last wave the split removes: the smallest
//      split that keeps >= 50 k-tiles per slice and cuts the modelled mainloop (waves x
//      k-tiles per slice) by at least 15%.
// A filled grid with a short K does not split: the fp32 partials (one M x N write and read
// per slice) cost more than the fill buys.  Case 1 does split a short K (down to two k-tiles
// per slice) because there the idle clusters are free.  Every K-slice is a whole number of
// 128-deep k-tiles (all batches share one slice length) and at most W8A16Sm100MaxSplits
// slices.  1 = do not split.
inline int W8A16Sm100SplitK(int M, int N, int K, int num_sms) {
  if (K % 128 != 0) return 1;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long units = W8A16Tiles(N, 256) * W8A16Tiles(M, 128);
  const int ktiles = K / 128;
  if (units * 2 <= clusters) {
    int best = 1;
    for (int s = 2; s <= W8A16Sm100MaxSplits; ++s) {
      if (ktiles % s != 0 || ktiles / s < 2 || units * s > clusters) continue;
      best = s;
    }
    return best;
  }
  if (ktiles >= 100) {
    const long long waves1 = (units + clusters - 1) / clusters;
    for (int s = 2; s <= W8A16Sm100MaxSplits; ++s) {
      if (ktiles % s != 0 || ktiles / s < 50) continue;
      const long long waves = (units * s + clusters - 1) / clusters;
      // waves x (ktiles / s) <= 0.85 x waves1 x ktiles, with the common factor ktiles / s
      // divided out: the full products overflow 64 bits at large M, N and K together
      // (units up to 2^47, ktiles up to 2^24), the reduced ones stay below 2^58.
      if (waves * 100 <= waves1 * s * 85) return s;
    }
  }
  return 1;
}

// Split count actually used by the split-K variant: the model's choice, or, when the
// variant is forced through `algo` on a grid the model would not split, the smallest
// s >= 2 that divides the k-tile count.  0 = K cannot be split (not a multiple of 128, or
// a k-tile count with no divisor in 2..W8A16Sm100MaxSplits).
inline int W8A16Sm100SplitsFor(int M, int N, int K, int num_sms) {
  const int s = W8A16Sm100SplitK(M, N, K, num_sms);
  if (s >= 2) return s;
  const int ktiles = K / 128;
  if (K % 128 != 0 || ktiles < 2) return 0;
  for (int c = 2; c <= W8A16Sm100MaxSplits; ++c) {
    if (ktiles % c == 0) return c;
  }
  return 0;
}

// Stream-K pays only where the tail wave is most of the grid and the tiles are long.  Every
// K fragment pays a pipeline prologue and a 128 KB fp32 accumulator round trip through
// global memory, about as much as 8-10 k-tiles of mainloop, so a tile has to be >= 64
// k-tiles deep before cutting it is cheaper than idling.  With four or more waves the saved
// fraction of a wave is too small, and under-filled grids (<= half the clusters busy) belong
// to split-K, whose batched slices skip the per-fragment handshake.  So: more than half the
// clusters busy, at most three waves with the last one at most a third full, >= 64 k-tiles.
inline bool W8A16Sm100UseStreamK(int M, int N, int K, int num_sms) {
  if (K % 128 != 0 || K / 128 < 64) return false;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long units = W8A16Tiles(N, 256) * W8A16Tiles(M, 128);
  if (units * 2 <= clusters) return false;
  const long long waves = (units + clusters - 1) / clusters;
  const long long tail = units - (waves - 1) * clusters;  // tiles in the last wave
  return waves >= 2 && waves <= 3 && tail * 3 <= clusters;
}

// The launcher's choice for a problem.
inline W8A16Sm100Algo W8A16Sm100PickAlgo(int M, int N, int K, int num_sms) {
  // A weight of <= 128 rows fills only half of the 2SM tile's 256 weight rows, so every 2SM
  // variant runs twice the MMA for the same bytes; the 1SM 128x128 tile has no padding
  // there.  Small M on such a row stays with split-K below.
  if (N <= 128 && M >= 16384) return W8A16Sm100Algo::OneSM128x128;
  // A mostly-empty last wave over deep tiles is balanced by stream-K.
  if (W8A16Sm100UseStreamK(M, N, K, num_sms)) return W8A16Sm100Algo::TwoSM256x128StreamK;
  // Under-filled grids and long-K partial waves split K instead of choosing between tiles.
  if (W8A16Sm100SplitK(M, N, K, num_sms) > 1) return W8A16Sm100Algo::TwoSM256x128SplitK;

  // Wave-fill model over the three 2SM tile heights: waves x rows per tile / efficiency,
  // with per-height efficiency factors (128: 1.00, 160: 0.96, 192: 1.02).  The 192-row
  // tile's advantage is a few percent, so it wins only where its tile count fills the waves
  // better; the taller tile's extra rows in the last M-tile are masked for free.  The
  // 256x256 smem-A tile and the 1SM tile are not candidates here (both stay reachable
  // through `algo`).
  (void)K;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long w_tiles = W8A16Tiles(N, 256);
  auto cost = [&](int tile_a, double eff) {
    const long long units = w_tiles * W8A16Tiles(M, tile_a);
    return static_cast<double>((units + clusters - 1) / clusters) * tile_a / eff;
  };
  const double c128 = cost(128, 1.00);
  const double c160 = cost(160, 0.96);
  const double c192 = cost(192, 1.02);
  if (c192 <= c128 && c192 <= c160) return W8A16Sm100Algo::TwoSM256x192;
  if (c160 < c128) return W8A16Sm100Algo::TwoSM256x160;
  return W8A16Sm100Algo::TwoSM256x128;
}

// What runs for one problem: the variant and, for the split-K variant, the slice count.
struct W8A16Sm100Plan {
  W8A16Sm100Algo algo;
  int splits;  // >= 2 for the split-K variant, 1 otherwise
};

// Resolves the caller's `algo` code (0 = the launcher's choice, 1..W8A16Sm100MaxAlgoCode =
// that variant) and the problem to the plan both the workspace query and the launch use.
//   cudaErrorInvalidValue: `algo_code` is not a variant.
//   cudaErrorNotSupported: the split-K variant forced on a K that cannot be sliced (see
//                          W8A16Sm100SplitsFor); the launcher's own choice never is.
inline cudaError_t W8A16Sm100Resolve(int M, int N, int K, int num_sms, int64_t algo_code,
                                     W8A16Sm100Plan* plan) {
  if (algo_code == 0) {
    plan->algo = W8A16Sm100PickAlgo(M, N, K, num_sms);
  } else if (algo_code >= 1 && algo_code <= W8A16Sm100MaxAlgoCode) {
    plan->algo = static_cast<W8A16Sm100Algo>(algo_code);
  } else {
    return cudaErrorInvalidValue;
  }
  plan->splits = 1;
  if (plan->algo == W8A16Sm100Algo::TwoSM256x128SplitK) {
    plan->splits = W8A16Sm100SplitsFor(M, N, K, num_sms);
    if (plan->splits < 2) return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

// The one algo -> kernel table: calls `f(W8A16VariantTag<V>{})` with the W8A16Sm100Gemm of
// the variant and returns its cudaError_t; cudaErrorInvalidValue for a code that is no
// variant.
template <typename T, typename F>
cudaError_t W8A16Sm100Visit(W8A16Sm100Algo algo, F&& f) {
  using D = W8A16Decomposition;
  switch (algo) {
    case W8A16Sm100Algo::TwoSM256x128:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>>{});
    case W8A16Sm100Algo::OneSM128x128:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>>{});
    case W8A16Sm100Algo::TwoSM256x256:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>>{});
    case W8A16Sm100Algo::TwoSM256x192:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>>{});
    case W8A16Sm100Algo::TwoSM256x128SplitK:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, D::SplitK>>{});
    case W8A16Sm100Algo::TwoSM256x128StreamK:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, D::StreamK>>{});
    case W8A16Sm100Algo::TwoSM256x160:
      return f(W8A16VariantTag<W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x160>>{});
    default:
      return cudaErrorInvalidValue;
  }
}

// Workspace bytes of the plan's variant for this problem.  cudaErrorInvalidValue when the
// size leaves the signed 64-bit range (see W8A16WorkspaceOf).
template <typename T>
cudaError_t W8A16Fp8GemmSm100WorkspaceBytes(const W8A16Sm100Plan& plan, int M, int N, int K,
                                            int device_id, int num_sms, size_t* bytes) {
  return W8A16Sm100Visit<T>(plan.algo, [&](auto tag) {
    using V = typename decltype(tag)::type;
    return W8A16WorkspaceBytesOf<V>(M, N, K, device_id, num_sms, V::IsSplitK ? plan.splits : 1,
                                    bytes);
  });
}

// Runs the plan's variant (see W8A16RunVariant for the errors).
template <typename T>
cudaError_t W8A16Fp8GemmSm100(const W8A16Sm100Plan& plan, const T* act, const __nv_fp8_e4m3* weight,
                              const float* scale, const T* bias, T* out, int M, int N, int K,
                              void* workspace, size_t workspace_bytes, int device_id, int num_sms,
                              cudaStream_t stream) {
  return W8A16Sm100Visit<T>(plan.algo, [&](auto tag) {
    using V = typename decltype(tag)::type;
    return W8A16RunVariant<V>(act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes,
                              device_id, num_sms, stream, V::IsSplitK ? plan.splits : 1);
  });
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM100_DISPATCH_HPP_
