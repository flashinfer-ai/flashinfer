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
//
// Sizing and execution go through the same W8A16Sm100Plan (W8A16Sm100Resolve), so the
// workspace query and the launch agree on the variant and on whether the request is valid.

#include <algorithm>

#include "flashinfer/gemm/w8a16_sm100/gemm.hpp"

namespace flashinfer::gemm {

// Blackwell `algo` codes of the Python op (`flashinfer.gemm.w8a16_fp8_gemm(..., algo=...)`
// on sm_100a); they must not change.  0 = the launcher's choice, 1.. = one variant (a tile
// and, for 5 / 6, a decomposition).  Codes 8-13 belonged to variants that measured slower on
// every row and were removed; never reuse them.
enum class W8A16Sm100Algo {
  kAuto = 0,
  k2SM256x128 = 1,
  k1SM128x128 = 2,
  k2SM256x256 = 3,
  k2SM256x192 = 4,
  k2SM256x128SplitK = 5,
  k2SM256x128StreamK = 6,
  k2SM256x160 = 7,
};
constexpr int kW8A16Sm100MaxAlgoCode = 7;
constexpr int kW8A16Sm100MaxSplits = 8;

// Split count for the 256x128 2SM tile.  Two cases split:
//   1. an under-filled grid, fewer cluster tiles than half the resident clusters: the
//      largest split that still fits in one wave (every cluster gets at most one K-slice;
//      the narrow-N decode rows run 1.3-1.8x faster);
//   2. a long K (>= 100 k-tiles) whose partial last wave the split removes: the smallest
//      split that keeps >= 50 k-tiles per slice and cuts the modelled mainloop (waves x
//      k-tiles per slice) by at least 15%.
// Short-K problems never split: the fp32 partials (one M x N write and read per slice)
// cost more than the fill buys (5-25% slower at K <= 8192).  Every K-slice is a whole number
// of 128-deep k-tiles (all batches share one slice length) and at most kW8A16Sm100MaxSplits
// slices.  1 = do not split.
inline int W8A16Sm100SplitK(int M, int N, int K, int num_sms) {
  if (K % 128 != 0) return 1;
  const long long clusters = num_sms > 1 ? num_sms / 2 : 1;
  const long long units = W8A16Tiles(N, 256) * W8A16Tiles(M, 128);
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
// s >= 2 that divides the k-tile count.  0 = K cannot be split (not a multiple of 128, or
// a k-tile count with no divisor in 2..kW8A16Sm100MaxSplits).
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

// Stream-K pays only where the tail wave is most of the grid and the tiles are long.  Every
// K fragment pays a pipeline prologue and a 128 KB fp32 accumulator round trip through
// global memory, about as much as 8-10 k-tiles of mainloop, so a tile has to be >= 64
// k-tiles deep before cutting it is cheaper than idling (at 2.16 waves: K=2048 1.53x slower
// than the plain tile, K=8192 0.93x, K=25600 0.82x).  With four or more waves the saved
// fraction of a wave is too small (5-11% slower), and under-filled grids (<= half the
// clusters busy) belong to split-K, whose batched slices skip the per-fragment handshake.
// So: more than half the clusters busy, at most three waves with the last one at most a
// third full, >= 64 k-tiles.
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
  // there (kernel / cuBLAS 1.00-1.01 vs 1.09-1.27 for the 2SM variants at large M).  Small
  // M on such a row stays with split-K below.
  if (N <= 128 && M >= 16384) return W8A16Sm100Algo::k1SM128x128;
  // A mostly-empty last wave over deep tiles is balanced by stream-K.
  if (W8A16Sm100UseStreamK(M, N, K, num_sms)) return W8A16Sm100Algo::k2SM256x128StreamK;
  // Under-filled grids and long-K partial waves split K instead of choosing between tiles.
  if (W8A16Sm100SplitK(M, N, K, num_sms) > 1) return W8A16Sm100Algo::k2SM256x128SplitK;

  // Wave-fill model over the three 2SM tile heights: waves x rows per tile / efficiency,
  // with the efficiency factors (128: 1.00, 160: 0.96, 192: 1.02) fitted on back-to-back
  // sweeps of all tiles (geomean model pick / per-row oracle 1.010; any 160-row factor in
  // 0.94-1.00 scores the same).  With the native converter the 192-row tile's advantage is
  // a few percent, so it wins only where its tile count fills the waves better; the taller
  // tile's extra rows in the last M-tile are masked for free.  The 256x256 smem-A tile loses
  // on every benchmark row and the 1SM tile never wins outside the N <= 128 rule above, so
  // neither is a candidate here (both stay reachable through `algo`).
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
  if (c192 <= c128 && c192 <= c160) return W8A16Sm100Algo::k2SM256x192;
  if (c160 < c128) return W8A16Sm100Algo::k2SM256x160;
  return W8A16Sm100Algo::k2SM256x128;
}

// What runs for one problem: the variant and, for the split-K variant, the slice count.
struct W8A16Sm100Plan {
  W8A16Sm100Algo algo;
  int splits;  // >= 2 for the split-K variant, 1 otherwise
};

// Resolves the caller's `algo` code (0 = the launcher's choice, 1..kW8A16Sm100MaxAlgoCode =
// that variant) and the problem to the plan both the workspace query and the launch use.
//   cudaErrorInvalidValue: `algo_code` is not a variant.
//   cudaErrorNotSupported: the split-K variant forced on a K that cannot be sliced (see
//                          W8A16Sm100SplitsFor); the launcher's own choice never is.
inline cudaError_t W8A16Sm100Resolve(int M, int N, int K, int num_sms, int64_t algo_code,
                                     W8A16Sm100Plan* plan) {
  if (algo_code == 0) {
    plan->algo = W8A16Sm100PickAlgo(M, N, K, num_sms);
  } else if (algo_code >= 1 && algo_code <= kW8A16Sm100MaxAlgoCode) {
    plan->algo = static_cast<W8A16Sm100Algo>(algo_code);
  } else {
    return cudaErrorInvalidValue;
  }
  plan->splits = 1;
  if (plan->algo == W8A16Sm100Algo::k2SM256x128SplitK) {
    plan->splits = W8A16Sm100SplitsFor(M, N, K, num_sms);
    if (plan->splits < 2) return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

// Workspace bytes of the plan's variant for this problem.
template <typename T>
size_t W8A16Fp8GemmSm100WorkspaceBytes(const W8A16Sm100Plan& plan, int M, int N, int K,
                                       int device_id, int num_sms) {
  using D = W8A16Decomposition;
  switch (plan.algo) {
    case W8A16Sm100Algo::k2SM256x128SplitK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, D::SplitK>::WorkspaceBytes(
          M, N, K, device_id, num_sms, plan.splits);
    case W8A16Sm100Algo::k2SM256x128StreamK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, D::StreamK>::WorkspaceBytes(
          M, N, K, device_id, num_sms);
    case W8A16Sm100Algo::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Algo::k2SM256x256:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Algo::k2SM256x192:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Algo::k2SM256x160:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x160>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    case W8A16Sm100Algo::k2SM256x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>::WorkspaceBytes(M, N, K, device_id,
                                                                        num_sms);
    default:
      return 0;
  }
}

template <typename T>
cudaError_t W8A16Fp8GemmSm100(const W8A16Sm100Plan& plan, const T* act, const __nv_fp8_e4m3* weight,
                              const float* scale, const T* bias, T* out, int M, int N, int K,
                              void* workspace, size_t workspace_bytes, int device_id, int num_sms,
                              cudaStream_t stream) {
  using D = W8A16Decomposition;
  switch (plan.algo) {
    case W8A16Sm100Algo::k2SM256x128SplitK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, D::SplitK>::Run(
          act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes, device_id, num_sms,
          stream, plan.splits);
    case W8A16Sm100Algo::k2SM256x128StreamK:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128, D::StreamK>::Run(
          act, weight, scale, bias, out, M, N, K, workspace, workspace_bytes, device_id, num_sms,
          stream);
    case W8A16Sm100Algo::k1SM128x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg1SM128x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Algo::k2SM256x256:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x256>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Algo::k2SM256x192:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x192>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Algo::k2SM256x160:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x160>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    case W8A16Sm100Algo::k2SM256x128:
      return W8A16Sm100Gemm<T, W8A16Sm100Cfg2SM256x128>::Run(act, weight, scale, bias, out, M, N, K,
                                                             workspace, workspace_bytes, device_id,
                                                             num_sms, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_SM100_DISPATCH_HPP_
