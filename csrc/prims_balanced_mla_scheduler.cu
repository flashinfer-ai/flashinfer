/***************************************************************************************************
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// PrimsTS balanced host scheduler. It runs between graph replays and emits the packed work and\n//
// compact combine descriptors consumed by the balanced PrimsTS kernels.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <queue>
#include <vector>

#include "flashinfer/exception.h"
#include "prims_balanced_mla_scheduler.cuh"

namespace flashinfer {
namespace prims_ts {

// The descriptor buffers are owned by the replay-stable PrimsTS plan.

////////////////////////////////////////////////////////////////////////////////////////////////////
// ===== Balanced scheduler (host) =====
//
// The production balanced-scheduler emitter. Python selects an explicitly calibrated cost model;
// this implementation applies cost-derived Y floors and exact emitted-placement scoring, then
// writes the compact descriptors consumed by the PrimsTS producer and reducer. Planning is not
// capture-safe (D2H sequence lengths when needed, host compute, and H2D descriptor copies), but its
// output buffers have stable addresses for graph replay. Validated against the Python reference.
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace {
inline void checkCuda(cudaError_t status, char const* operation) {
  FLASHINFER_CHECK(status == cudaSuccess, operation, "failed:", cudaGetErrorString(status));
}

inline int32_t ceilDivNonnegativeI32(int32_t value, int32_t divisor) {
  FLASHINFER_CHECK(value >= 0 && divisor > 0,
                   "Balanced scheduler ceil division requires a non-negative value and positive "
                   "divisor");
  return static_cast<int32_t>((static_cast<int64_t>(value) + divisor - 1) / divisor);
}

struct HostChunk {
  int32_t reqIdx, startBlock, endBlock, numPieces, splitBeginIdx, splitGlobalIdx;
  bool isSplit;
};

// Split-penalty semantics: splitting a request into k pieces costs an EXTRA
// gamma0 + gamma1*k over the whole request; each piece is charged gamma1 + gamma0/k. (No cF term —
// provably irrelevant to the ordering; dropped from the model.)
inline int64_t hostPieceCost(int32_t blocks, bool split, int32_t numPieces, int32_t cB, int32_t cR,
                             int32_t g0, int32_t g1) {
  int64_t c = static_cast<int64_t>(cR) + static_cast<int64_t>(blocks) * cB;
  if (split) c += static_cast<int64_t>(g1) + static_cast<int64_t>(g0) / std::max(numPieces, 1);
  return c;
}

struct HostCost {
  int32_t cB, cR, gamma0, gamma1;
  // Additive combine model (see hostCombineCost). {0, 0} falls back to the legacy per-piece
  // gammas; either nonzero component defines an explicit separate-kernel model.
  int32_t comb0 = 0, comb1 = 0;
};

// Cost of the separate combine kernel, ADDED to the main kernel's makespan.
//
// The gamma0/gamma1 terms above charge the split penalty per piece INSIDE the makespan max, i.e. as
// if the combine ran concurrently with the main kernel. It does not: it is a second kernel launched
// after it, so its duration lands on the critical path. Charging it inside the max makes splitting
// look nearly free and the search over-splits (measured TP2 uniform_8192_bs2: the scheduler picks
// Y=1 / 128 pieces at 28.70 us, while Y=2 / 64 pieces is 24.60 us -- main barely moves, 13.18 ->
// 11.72, while combine grows 9.07 -> 14.70).
//
// Scaling: the combine grid is (batchSize, maxNumCtasQ*ncr, numCtasForAllHeads) and each CTA loops
// over ITS OWN request's pieces, so the cost tracks the MAX pieces of any split request, not the
// total piece count. Fitting red_us against max-pieces on four measured Y-curves gives a per-family
// constant that holds across very different shapes (Swaps: 163/187/163 ns/piece on uniform_2048,
// uniform_8192 and ragged_131072, max error 0.5-3.6%; Keeps ~368 ns/piece, ~2x for its 4x larger
// tileSizeQ). Requests that are not split never enter the combine.
inline int64_t hostCombineCost(int32_t maxPiecesPerSplitReq, int32_t comb0, int32_t comb1) {
  if (maxPiecesPerSplitReq < 2) return 0;  // nothing split -> combine CTAs all exit immediately
  return static_cast<int64_t>(comb0) + static_cast<int64_t>(comb1) * maxPiecesPerSplitReq;
}

// NOT MODELLED: a bandwidth roofline on the main kernel's makespan (totalKvBytes / BW_eff), on the
// theory that the schedule must stream every KV byte however it is sliced, so without a floor the
// LPT term keeps falling as pieces shrink and the model believes arbitrarily fine splits keep
// paying off. It was implemented and REJECTED; recorded here so it is not re-attempted blind.
//
// The motivating measurement: at the finest split each cell bottoms out at (graph-replay main_us)
// 37.7 MB / 12.16 us = 3.10 TB/s (uniform_8192_bs4) and 155.1 MB / 45.93 us = 3.38 TB/s
// (ragged_131072_bs8) -- the same regime across a 4x footprint difference, one constant fitting
// both floors within ~7%. Not an L2-capacity effect either: an ncu cold-cache A/B (--cache-control
// all vs none) showed the L2-resident 37.7 MB cell is no slower with L2 flushed (24.96 vs 26.66
// us), so a two-regime cache model is not warranted.
//
// A/B at 3300 bytes/ns on the same build (roof on vs off, total us): uniform_8192_bs4 22.56 ->
// 20.53 (helps, lands exactly on the measured optimum), but ragged_131072_bs8 65.14 -> 83.75
// (+29%, a case the balanced scheduler exists to win); uniform_8192_bs2 and uniform_2048_bs4
// unchanged. Why it fails: a DEVICE-level floor assumes attainable bandwidth is independent of how
// many partitions are live. On the ragged cell at coarse Y only ~32-43 pieces are live out of
// P=148, so attainable bandwidth is a fraction of the device's and the true floor is far above
// totalBytes/BW_eff; the model then stops valuing finer splits and drifts coarse. Making the floor
// parallelism-aware (dividing by activePieces/P) overshoots the other way (predicts 217 us where 85
// us is measured). The 3.10 / 3.38 TB/s is better explained as cR + cB arithmetic than as a
// bandwidth ceiling -- both cells sit at ~40% of HBM peak, i.e. not saturated -- and per-CTA
// streaming is already captured by cB (bytesPerBlock / ~94 GB/s ~= 3.06 us, which IS the fitted
// cB), so a device roof adds nothing real.

inline int32_t costSearchY(std::vector<int32_t> const& nb, int32_t N, int32_t P, int32_t bMax,
                           int32_t baseY, int32_t yMin, int32_t yCap, HostCost const& cm,
                           int32_t cF);

// -------------------------------------------------------------------------------------
// Modular scheduler primitives shared by candidate scoring and final descriptor emission.
// Keeping placement policy in these helpers prevents the cost search from optimizing a layout
// different from the one that the kernel ultimately consumes.
// -------------------------------------------------------------------------------------

// computeY. Y_avg = ceil(total/P); Y_brk = ceil((gamma0 + 2*gamma1)/cB) + 3;
// Y_min is cost-derived and Y_cap keeps piece counts <= 126.
// Y = max(Y_avg + Y_brk, Y_min, Y_cap, 1).
struct HostYBreakdown {
  int32_t Y, Y_brk, Y_min, Y_cap;
};
inline HostYBreakdown computeY(int64_t total, int32_t P, int32_t bMax, HostCost const& cm) {
  int64_t const Y_avg = (total + P - 1) / P;
  int64_t const Y_brk =
      (cm.cB > 0) ? ((static_cast<int64_t>(cm.gamma0) + 2LL * cm.gamma1 + cm.cB - 1) / cm.cB + 3)
                  : 1000000000;
  int64_t const cBpos = std::max(cm.cB, 1);
  int64_t const Y_min =
      (static_cast<int64_t>(cm.cR) + cm.gamma1 + cm.gamma0 / 2 + cBpos - 1) / cBpos + 1;
  int64_t const Y_cap = (static_cast<int64_t>(bMax) + 125) / 126;
  int64_t const Y = std::max({Y_avg + Y_brk, Y_min, Y_cap, int64_t{1}});
  FLASHINFER_CHECK(Y <= INT32_MAX, "Balanced scheduler target piece tiles exceeds int32 capacity");
  return {static_cast<int32_t>(Y), static_cast<int32_t>(Y_brk), static_cast<int32_t>(Y_min),
          static_cast<int32_t>(Y_cap)};
}

// LPT (Longest Processing Time first): min-heap over (runningCost, pid); the largest remaining
// request goes onto the least-loaded partition. No split. Within-partition ordering is deferred
// to emitDescriptors.
inline void lptAssign(std::vector<std::vector<HostChunk>>* part, std::vector<int64_t>& run,
                      std::vector<int32_t>& idx, std::vector<int32_t> const& nb, HostCost const& cm,
                      int32_t cF, int32_t P, bool alreadySorted = false) {
  if (!alreadySorted)
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
      int64_t ca = hostPieceCost(nb[a], false, 0, cm.cB, cm.cR, cm.gamma0, cm.gamma1),
              cb = hostPieceCost(nb[b], false, 0, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
      return ca != cb ? ca > cb : a < b;
    });
  using E = std::pair<int64_t, int32_t>;
  std::priority_queue<E, std::vector<E>, std::greater<E>> loadOrder;
  for (int32_t pid = 0; pid < P; ++pid) loadOrder.push({run[pid], pid});
  for (int32_t i : idx) {
    auto [c, pid] = loadOrder.top();
    loadOrder.pop();
    if (c == 0) c = cF;
    c += hostPieceCost(nb[i], false, 0, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
    run[pid] = c;
    if (part != nullptr) (*part)[pid].push_back({i, 0, nb[i], 1, 0, 0, false});
    loadOrder.push({c, pid});
  }
}

// Equal-split long requests: each request with b > Y is split into k = min(ceil(b/Y), 127, P)
// equal pieces placed on the k least-loaded partitions. splitGlobalIdx temporarily holds the
// local piece index j; finalizeSplitMetadata/emitDescriptors resolve it to the global index.
inline void equalSplitLong(std::vector<std::vector<HostChunk>>* part, std::vector<int64_t>& run,
                           std::vector<int32_t>& idx, std::vector<int32_t> const& nb,
                           HostCost const& cm, int32_t cF, int32_t P, int32_t Y,
                           bool chargeSplitPenalty, bool alreadySorted = false) {
  if (!alreadySorted)
    std::sort(idx.begin(), idx.end(),
              [&](int32_t a, int32_t b) { return nb[a] != nb[b] ? nb[a] > nb[b] : a < b; });
  using E = std::pair<int64_t, int32_t>;
  std::priority_queue<E, std::vector<E>, std::greater<E>> loadOrder;
  for (int32_t pid = 0; pid < P; ++pid) loadOrder.push({run[pid], pid});
  std::vector<int32_t> selected;
  selected.reserve(P);
  for (int32_t i : idx) {
    int32_t b = nb[i], k = std::min({ceilDivNonnegativeI32(b, Y), 127, P});
    bool const isSplit = k > 1;
    selected.clear();
    for (int32_t j = 0; j < k; ++j) {
      selected.push_back(loadOrder.top().second);
      loadOrder.pop();
    }
    std::sort(selected.begin(), selected.end());
    int32_t base = b / k, rem = b - base * k, offb = 0;
    for (int32_t j = 0; j < k; ++j) {
      int32_t pid = selected[j], piece = base + (j < rem ? 1 : 0);
      if (run[pid] == 0) run[pid] = cF;
      run[pid] += hostPieceCost(piece, chargeSplitPenalty && isSplit, k, cm.cB, cm.cR, cm.gamma0,
                                cm.gamma1);
      if (part != nullptr) (*part)[pid].push_back({i, offb, offb + piece, k, 0, j, isSplit});
      offb += piece;
    }
    for (int32_t pid : selected) loadOrder.push({run[pid], pid});
  }
}

// Place exactly one target candidate. Shorts use load-aware LPT; split requests are processed from
// longest to shortest and each uses the least-loaded distinct partitions. The final emitter sorts
// each partition by request coordinate, retaining deterministic within-partition locality without
// changing this load assignment. Candidate scoring calls the same lower-level assignment
// primitives with pre-sorted request lists and without materializing descriptors.
inline void placeForTarget(std::vector<std::vector<HostChunk>>* part, std::vector<int64_t>& run,
                           std::vector<int32_t> const& nb, HostCost const& cm, int32_t cF,
                           int32_t P, int32_t Y, bool chargeSplitPenalty) {
  std::vector<int32_t> shorts, longs;
  for (int32_t i = 0; i < static_cast<int32_t>(nb.size()); ++i) {
    if (nb[i] <= 0) continue;
    (nb[i] > Y ? longs : shorts).push_back(i);
  }
  if (!shorts.empty()) lptAssign(part, run, shorts, nb, cm, cF, P);
  if (!longs.empty()) equalSplitLong(part, run, longs, nb, cm, cF, P, Y, chargeSplitPenalty);
}

// Cost-driven Y selection (recalibrated families, nz <= 2*P): evaluate candidate piece sizes
// Y_c = ceil(bMax/k), k = 1..min(P,127), and score the exact placement emitted for that target.
// The coarser target wins ties. Host-only; runs once per step outside graph replay.
inline int32_t costSearchY(std::vector<int32_t> const& nb, int32_t N, int32_t P, int32_t bMax,
                           int32_t baseY, int32_t yMin, int32_t yCap, HostCost const& cm,
                           int32_t cF) {
  std::vector<int32_t> cands;
  cands.push_back(baseY);
  int32_t const kMax = std::min(P, 127);
  int32_t nzCount = 0;
  for (int32_t i = 0; i < N; ++i)
    if (nb[i] > 0) ++nzCount;
  for (int32_t k = 1; k <= kMax; ++k) {
    // Keep the efficiency floor disabled while candidate pieces still fit on otherwise-idle
    // partitions. In that region extra total work does not extend the modeled critical path.
    int32_t const floorY = (static_cast<int64_t>(k) * nzCount <= P) ? 1 : yMin;
    cands.push_back(std::max({ceilDivNonnegativeI32(bMax, k), floorY, yCap, 1}));
  }
  std::sort(cands.begin(), cands.end(), std::greater<int32_t>());
  cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

  bool const useAdditive = cm.comb0 != 0 || cm.comb1 != 0;
  int64_t bestCost = INT64_MAX;
  int32_t bestY = baseY;
  std::vector<int32_t> pieceSignature(N, 0), previousSignature;
  std::vector<int32_t> lptOrder, longOrder, shorts, longs;
  lptOrder.reserve(nzCount);
  for (int32_t i = 0; i < N; ++i)
    if (nb[i] > 0) lptOrder.push_back(i);
  longOrder = lptOrder;
  std::sort(lptOrder.begin(), lptOrder.end(), [&](int32_t a, int32_t b) {
    int64_t const ca = hostPieceCost(nb[a], false, 0, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
    int64_t const cb = hostPieceCost(nb[b], false, 0, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
    return ca != cb ? ca > cb : a < b;
  });
  std::sort(longOrder.begin(), longOrder.end(),
            [&](int32_t a, int32_t b) { return nb[a] != nb[b] ? nb[a] > nb[b] : a < b; });
  shorts.reserve(nzCount);
  longs.reserve(nzCount);
  std::vector<int64_t> candidateRun(P, 0);
  for (int32_t yc : cands) {
    int64_t descriptors = 0;
    int64_t totalPieceCost = 0, largestPieceCost = 0;
    int32_t maxPieces = 0;
    for (int32_t i = 0; i < N; ++i) {
      int32_t const b = nb[i];
      int32_t const pieces =
          b > yc ? std::min({ceilDivNonnegativeI32(b, yc), 127, P}) : (b > 0 ? 1 : 0);
      pieceSignature[i] = pieces;
      descriptors += pieces;
      maxPieces = std::max(maxPieces, pieces);
      if (pieces == 1) {
        int64_t const pieceCost = hostPieceCost(b, false, 1, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
        totalPieceCost += pieceCost;
        largestPieceCost = std::max(largestPieceCost, pieceCost);
      } else if (pieces > 1) {
        int32_t const base = b / pieces, remainder = b - base * pieces;
        int64_t const smallCost =
            hostPieceCost(base, !useAdditive, pieces, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
        int64_t const largeCost =
            hostPieceCost(base + 1, !useAdditive, pieces, cm.cB, cm.cR, cm.gamma0, cm.gamma1);
        totalPieceCost += static_cast<int64_t>(pieces - remainder) * smallCost +
                          static_cast<int64_t>(remainder) * largeCost;
        largestPieceCost = std::max(largestPieceCost, remainder > 0 ? largeCost : smallCost);
      }
    }
    // Candidates are visited from coarsest to finest, so descriptor demand is monotonic. Equal
    // per-request piece counts also imply identical equal-split sizes and placement; retain the
    // first (coarsest) target for the established tie break without simulating it again.
    if (descriptors > static_cast<int64_t>(nzCount) + P) break;
    if (!previousSignature.empty() && pieceSignature == previousSignature) continue;
    previousSignature = pieceSignature;
    int64_t const optimisticMakespan = std::max((totalPieceCost + P - 1) / P, largestPieceCost);
    int64_t const optimisticTotal =
        optimisticMakespan + (useAdditive ? hostCombineCost(maxPieces, cm.comb0, cm.comb1) : 0);
    // Candidates are ordered coarse-to-fine, so an equal objective cannot win the tie break.
    if (optimisticTotal >= bestCost) continue;
    std::fill(candidateRun.begin(), candidateRun.end(), 0);
    shorts.clear();
    longs.clear();
    for (int32_t i : lptOrder)
      if (nb[i] <= yc) shorts.push_back(i);
    for (int32_t i : longOrder)
      if (nb[i] > yc) longs.push_back(i);
    if (!shorts.empty())
      lptAssign(nullptr, candidateRun, shorts, nb, cm, cF, P, /*alreadySorted=*/true);
    if (!longs.empty())
      equalSplitLong(nullptr, candidateRun, longs, nb, cm, cF, P, yc,
                     /*chargeSplitPenalty=*/!useAdditive, /*alreadySorted=*/true);
    int64_t const makespan = *std::max_element(candidateRun.begin(), candidateRun.end());
    int64_t const total =
        makespan + (useAdditive ? hostCombineCost(maxPieces, cm.comb0, cm.comb1) : int64_t{0});
    if (total < bestCost || (total == bestCost && yc > bestY)) {
      bestCost = total;
      bestY = yc;
    }
  }
  return bestY;
}

// Finalize split metadata: per-request piece count + splitBegin prefix sum over reqIdx.
inline void finalizeSplitMetadata(std::vector<std::vector<HostChunk>> const& part, int32_t N,
                                  std::vector<int32_t>& pieces, std::vector<int32_t>& splitBegin) {
  pieces.assign(N, 0);
  for (auto const& pp : part)
    for (auto const& c : pp)
      if (c.isSplit) pieces[c.reqIdx]++;
  splitBegin.assign(N, 0);
  int32_t cum = 0;
  for (int32_t i = 0; i < N; ++i)
    if (pieces[i] >= 2) {
      splitBegin[i] = cum;
      cum += pieces[i];
    }
  // Hard ABI limit, NOT an internal invariant: splitGlobalIdx/splitBeginIdx are 12-bit fields in
  // BalancedWorkDescriptor::splitInfo, so packing beyond 4096 total pieces truncates silently and
  // the combine then reads the wrong partials. The bound is structurally hard to reach (peak ~96
  // pieces at batch 4096), which is exactly why it must be a check rather than a comment.
  FLASHINFER_CHECK(cum <= 4096,
                   "Balanced scheduler: %d split pieces exceed the 4096-piece cap of the 12-bit "
                   "splitInfo fields",
                   cum);
}

// Emit partition-grouped descriptors + offsets. Sorts each partition by (reqIdx, startBlock)
// for L2 locality and resolves splitGlobalIdx (local j) to the global index splitBegin + j.
inline void emitDescriptors(std::vector<std::vector<HostChunk>>& part, int32_t P,
                            std::vector<int32_t> const& splitBegin,
                            std::vector<BalancedWorkDescriptor>& desc, std::vector<int32_t>& off) {
  off.assign(P + 1, 0);
  for (int32_t pid = 0; pid < P; ++pid) {
    off[pid] = (int32_t)desc.size();
    std::sort(part[pid].begin(), part[pid].end(), [](HostChunk const& a, HostChunk const& b) {
      return a.reqIdx != b.reqIdx ? a.reqIdx < b.reqIdx : a.startBlock < b.startBlock;
    });
    for (auto& c : part[pid]) {
      BalancedWorkDescriptor d;
      d.reqIdx = c.reqIdx;
      d.startBlock = c.startBlock;
      d.endBlock = c.endBlock;
      d.splitInfo = c.isSplit ? packSplitInfo(1, splitBegin[c.reqIdx] + c.splitGlobalIdx,
                                              c.numPieces, splitBegin[c.reqIdx])
                              : 0;
      desc.push_back(d);
    }
  }
  off[P] = (int32_t)desc.size();
}
}  // namespace

void runBalancedSchedHost(BalancedSchedParams const& p, int32_t* numCombineDescDev,
                          int32_t maxTotalSplits) {
  int32_t const N = p.batchSize, P = p.numSmParts, stepKv = p.blockSizeN;
  // Python requires an exact device calibration and supplies its selected
  // family/dtype/workload-bucket model on every call.
  HostCost const cm{p.costPerBlock,   p.fixedPieceCost,   p.splitFixedCost,
                    p.splitPieceCost, p.combineFixedCost, p.combinePieceCost};
  int32_t const cF = 0;

  // seqLensKv is the scheduler's only device input. If the caller provides it host-side
  // (seqLensOnHost) we read it directly — no D2H, no stream sync. Otherwise D2H+sync.
  std::vector<int32_t> seq(N);
  if (p.seqLensOnHost) {
    std::copy(p.seqLensKvPtr, p.seqLensKvPtr + N, seq.data());
  } else {
    checkCuda(cudaMemcpyAsync(seq.data(), p.seqLensKvPtr, N * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, p.stream),
              "balanced scheduler sequence-length copy");
    checkCuda(cudaStreamSynchronize(p.stream), "balanced scheduler sequence-length synchronize");
  }

  std::vector<int32_t> nb(N, 0);
  int64_t total = 0;
  int32_t bMax = 0, bMin = INT32_MAX, nz = 0;
  for (int32_t i = 0; i < N; ++i) {
    int32_t const s = seq[i];
    nb[i] = (s > 0) ? ceilDivNonnegativeI32(s, stepKv) : 0;
    if (nb[i] > 0) {
      total += nb[i];
      bMax = std::max(bMax, nb[i]);
      bMin = std::min(bMin, nb[i]);
      ++nz;
    }
  }

  // Inactive requests need no producer descriptor. The fixed compact reducer
  // grid strides across logical request slots and deterministically publishes
  // zero output for every slot whose live K/V length is zero.

  std::vector<int32_t> off(P + 1, 0);
  if (nz == 0) {
    checkCuda(cudaMemcpyAsync(p.workDescriptorOffsetsPtr, off.data(), (P + 1) * sizeof(int32_t),
                              cudaMemcpyHostToDevice, p.stream),
              "balanced scheduler empty partition-offset copy");
    int32_t zero = 0;
    checkCuda(cudaMemcpyAsync(numCombineDescDev, &zero, sizeof(int32_t), cudaMemcpyHostToDevice,
                              p.stream),
              "balanced scheduler empty combine-count copy");
    checkCuda(cudaStreamSynchronize(p.stream), "balanced scheduler empty-plan synchronize");
    if (p.planMetadataHostPtr != nullptr) {
      p.planMetadataHostPtr[0] = 0;
      p.planMetadataHostPtr[1] = 0;
      p.planMetadataHostPtr[2] = 0;
    }
    return;
  }
  // computeY. Returns Y plus Y_brk/Y_min/Y_cap (used by the uniform-spread test and
  // the cost search below).
  HostYBreakdown const yb = computeY(total, P, bMax, cm);
  int32_t Y = yb.Y;
  bool const usesForcedTarget = p.forcedTargetPieceTiles > 0;
  if (usesForcedTarget) Y = p.forcedTargetPieceTiles;
  // Cost-driven Y in the few-requests regime (nz <= 2*P) fixes piece-count
  // quantization, idle-SM under-splitting, and the split-vs-no-split decision.
  // When it runs, the uniform fast path below is skipped because it already
  // compared the k=1 layout.
  bool const usesCostSearch = !usesForcedTarget && nz > 0 && nz <= 2 * P;
  if (usesCostSearch) Y = costSearchY(nb, N, P, bMax, Y, yb.Y_min, yb.Y_cap, cm, cF);

  std::vector<std::vector<HostChunk>> part(P);
  std::vector<int64_t> run(P, 0);

  std::vector<int32_t> nzIdx;
  for (int32_t i = 0; i < N; ++i)
    if (nb[i] > 0) nzIdx.push_back(i);
  // Uniform fast path: when spread is tight (bMax - bMin <= Y_brk) and there are enough nonzero
  // requests (2*nz >= P), splitting only adds gamma overhead — emit a no-split layout. The cost
  // search (when it ran) already compared the k=1 layout, so it must not be overridden here.
  bool const spreadUniform = (bMax - bMin) <= yb.Y_brk;
  bool const enough = 2LL * nz >= P;
  if (spreadUniform && enough && !usesCostSearch && !usesForcedTarget) {
    lptAssign(&part, run, nzIdx, nb, cm, cF, P);
  } else {
    bool const useAdditive = cm.comb0 != 0 || cm.comb1 != 0;
    placeForTarget(&part, run, nb, cm, cF, P, Y, /*chargeSplitPenalty=*/!useAdditive);
  }
  // Finalize split metadata (per-request piece count + splitBegin prefix sum) then emit the
  // partition-grouped descriptors + offsets and the per-split-request combine descriptors.
  std::vector<int32_t> pieces, splitBegin;
  finalizeSplitMetadata(part, N, pieces, splitBegin);

  std::vector<BalancedWorkDescriptor> desc;
  std::vector<BalancedCombineDescriptor> comb;
  desc.reserve(N + P);
  emitDescriptors(part, P, splitBegin, desc, off);
  for (int32_t i = 0; i < N; ++i)
    if (pieces[i] >= 2) comb.push_back({i, pieces[i], splitBegin[i], 0});
  int32_t const numComb = (int32_t)comb.size();

  int32_t const activeWorkCap = nz + P;
  FLASHINFER_CHECK((int32_t)desc.size() <= activeWorkCap,
                   "Balanced scheduler: the schedule needs %d work descriptors but the active + "
                   "P limit is %d",
                   (int32_t)desc.size(), activeWorkCap);
  int32_t const partialCap =
      static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(N) + P, 2LL * P));
  int32_t splitPartialCount = 0;
  for (int32_t i = 0; i < N; ++i)
    if (pieces[i] >= 2) splitPartialCount += pieces[i];
  FLASHINFER_CHECK(splitPartialCount <= partialCap,
                   "Balanced scheduler: the schedule needs %d split partials but the compact "
                   "partial buffer holds %d",
                   splitPartialCount, partialCap);

  if (p.planMetadataHostPtr != nullptr) {
    p.planMetadataHostPtr[0] = static_cast<int32_t>(desc.size());
    p.planMetadataHostPtr[1] = Y;
    p.planMetadataHostPtr[2] = numComb;
  }

  // The descriptor buffers are caller-owned and carry no size, so writing past them is silent
  // device-memory corruption. maxTotalSplits is the capacity the caller allocated the work buffer
  // with; it was previously accepted as a parameter and never read, which read as a bounds check
  // without being one. A declared capacity on the params takes precedence when present.
  int32_t const workCap = p.workDescriptorCapacity > 0 ? p.workDescriptorCapacity : maxTotalSplits;
  FLASHINFER_CHECK(workCap <= 0 || (int32_t)desc.size() <= workCap,
                   "Balanced scheduler: the schedule needs %d work descriptors but the caller's "
                   "buffer holds %d. Size it with getBalancedWorkspaceSizes().",
                   (int32_t)desc.size(), workCap);
  FLASHINFER_CHECK(p.combineDescriptorCapacity <= 0 || numComb <= p.combineDescriptorCapacity,
                   "Balanced scheduler: the schedule needs %d combine descriptors but the caller's "
                   "buffer holds %d. Size it with getBalancedWorkspaceSizes().",
                   numComb, p.combineDescriptorCapacity);

  checkCuda(cudaMemcpyAsync(p.workDescriptorPtr, desc.data(),
                            desc.size() * sizeof(BalancedWorkDescriptor), cudaMemcpyHostToDevice,
                            p.stream),
            "balanced scheduler work-descriptor copy");
  checkCuda(cudaMemcpyAsync(p.workDescriptorOffsetsPtr, off.data(), (P + 1) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, p.stream),
            "balanced scheduler partition-offset copy");
  if (numComb > 0)
    checkCuda(cudaMemcpyAsync(p.combineDescriptorPtr, comb.data(),
                              numComb * sizeof(BalancedCombineDescriptor), cudaMemcpyHostToDevice,
                              p.stream),
              "balanced scheduler combine-descriptor copy");
  checkCuda(cudaMemcpyAsync(numCombineDescDev, &numComb, sizeof(int32_t), cudaMemcpyHostToDevice,
                            p.stream),
            "balanced scheduler combine-count copy");
  checkCuda(cudaStreamSynchronize(p.stream), "balanced scheduler plan synchronize");
}

}  // namespace prims_ts
}  // namespace flashinfer
