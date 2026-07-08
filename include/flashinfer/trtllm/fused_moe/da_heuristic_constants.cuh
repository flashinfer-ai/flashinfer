/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace flashinfer {
namespace da_heuristic {

// Maximum number of candidate tile sizes carried by one selector state.
constexpr int kMaxTiles = 8;

// Maximum normalized routing-count exemplars stored for one selector state.
// Keep in sync with MAX_EXEMPLARS in
// flashinfer/fused_moe/dist_aware/da_profile.py.
constexpr int kMaxExemplars = 8;
// Largest local-expert count vector accepted by the k-NN selector.
constexpr int kMaxKnnExperts = 512;
// Threads used by the one-block selector and histogram kernel.
constexpr int kKnnSelectorBlockThreads = 256;
// Expanded routing entries each selector thread processes per histogram pass.
constexpr int kKnnSelectorHistogramItemsPerThread = 8;
// Histogram storage for all experts plus the out-of-range sentinel bin.
constexpr int kKnnSelectorHistogramBins = kMaxKnnExperts + 1;
// Threads used by each block of the split histogram implementation.
constexpr int kKnnSplitHistogramBlockThreads = 256;
// Expanded routing entries each split-histogram thread processes per pass.
constexpr int kKnnSplitHistogramItemsPerThread = 8;
// Small routing workloads use the single-block selector; larger workloads use
// the split implementation. The boundary is 4096 expanded routing entries
// (512 tokens at top_k=8).
// Minimum expanded routing entries that select the split histogram path.
constexpr int kKnnSplitHistogramMinElements = 4096;
// Upper bound on split histogram blocks to limit temporary-work scaling.
constexpr int kKnnSplitHistogramMaxBlocks = 64;

}  // namespace da_heuristic
}  // namespace flashinfer
