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
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <flashinfer/cake_blackwell_softmax.cuh>
#include <flashinfer/sampling.cuh>
#include <limits>
#include <vector>

#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

constexpr int kBootstrapThreads = 256;
constexpr int kRowwiseThreads = 512;
constexpr int kWarpThreads = 128;
constexpr int kWarpRowsPerCta = 4;
constexpr int kMaxSplits = 64;
constexpr size_t kBootstrapDynamicSmemBytes = 128;
constexpr size_t kRowwiseDynamicSmemBytes = 128;
constexpr size_t kWarpDynamicSmemBytes = 0;
constexpr int kCachedThreads = 256;
constexpr int kCachedVectorElements = 8;
constexpr int kCachedTileElements = kCachedThreads * kCachedVectorElements;
constexpr int kCachedPortableClusterSize = 8;
constexpr int kCachedMaxClusterSize = 16;
constexpr int kCachedMaxTiles = 8;
constexpr uint32_t kCachedMaxVocabSize =
    static_cast<uint32_t>(kCachedMaxClusterSize) * kCachedMaxTiles * kCachedTileElements;
constexpr uintptr_t kCachedAlignmentBytes = 32;
// An 8-tile launch whose rows exceed one wave of resident persistent clusters
// R_P takes the persistent kernel (the clusters stay resident and walk the
// rows) or the shared-tile kernel (4 register tiles and up to 4 shared-memory
// tiles per CTA, four resident CTAs per SM, one cluster per row; R_S resident
// clusters), by this rule (measured on GB300 and B200):
// - a chunk of at most 5 tiles takes the shared-tile kernel, except
//   - an 8-CTA chunk whose rows fill at least 4/5 of the persistent kernel's
//     trips and whose fifth tile is more than 1/4 (sm_100) or at least 3/8
//     (sm_103) full, from three persistent trips on when the fifth tile is
//     at least 3/8 full and from 2.5 rows per persistent cluster on
//     otherwise, below 7 rows per shared-tile cluster; on sm_100 also at
//     three persistent trips from 10/9 rows per shared-tile cluster on, and,
//     when the fifth tile is at most 279/1024 full, below 15/2 rows per
//     shared-tile cluster;
//   - a 16-CTA chunk whose rows fill at least 9/10 of three persistent trips;
// - a full 8-tile chunk takes the persistent kernel, unless the rows fit one
//   wave of shared-tile clusters while filling less than a fraction N / D of
//   the persistent kernel's trips (rows * D < ceil(rows / R_P) * R_P * N; 4/5
//   for 16-CTA clusters, 3/4 for 8-CTA clusters), or reach 10 on sm_100 and 8
//   on sm_103 (16-CTA clusters) or 7 (8-CTA clusters) rows per shared-tile
//   cluster;
// - a 6- or 7-tile chunk takes the persistent kernel below 4 rows per
//   shared-tile cluster unless the same trip rule holds; a 6-tile chunk or a
//   16-CTA chunk on sm_103 takes it from 2.5 rows per persistent cluster on
//   instead; on sm_100 a 16-CTA chunk keeps it past 4 rows per shared-tile
//   cluster up to 20 (6 tiles) or 22 (7 tiles) persistent trips; on sm_103 a
//   6-tile 8-CTA chunk takes it from 20/9 rows per persistent cluster on.
constexpr int kSharedTilesAnyRowsMaxTiles = 5;
constexpr uint64_t kPersistentTripFillNumC16 = 4;
constexpr uint64_t kPersistentTripFillDenC16 = 5;
constexpr uint64_t kPersistentTripFillNumC8 = 3;
constexpr uint64_t kPersistentTripFillDenC8 = 4;
constexpr uint64_t kSharedTilesRowsPerClusterFullChunkC16Sm100 = 10;
constexpr uint64_t kSharedTilesRowsPerClusterFullChunkC16Sm103 = 8;
constexpr uint64_t kSharedTilesRowsPerClusterFullChunkC8 = 7;
constexpr uint64_t kFifthTileMaxThinEighthsSm100 = 2;
constexpr uint64_t kFifthTileMinFillEighthsSm103 = 3;
constexpr uint64_t kFifthTileEarlyPersistentEighthsSm100 = 3;
constexpr uint64_t kFiveTileSharedTilesMaxTrips = 2;
constexpr uint64_t kNarrowChunkC16PersistentTrips = 3;
constexpr uint64_t kNarrowChunkC16TripFillNum = 9;
constexpr uint64_t kNarrowChunkC16TripFillDen = 10;
constexpr uint64_t kPersistentRowsPerClusterNum = 5;
constexpr uint64_t kPersistentRowsPerClusterDen = 2;
constexpr uint64_t kSharedTilesRowsPerCluster = 4;
// sm_100, 5-tile 8-CTA chunks whose fifth tile is more than 2/8 full: three
// persistent trips take the persistent kernel from 10/9 rows per shared-tile
// cluster on, and a fifth tile at most 279/1024 full keeps it below 15/2 rows
// per shared-tile cluster.
constexpr uint64_t kFiveTileThreeTrips = 3;
constexpr uint64_t kFiveTileThreeTripRowsNumSm100 = 10;
constexpr uint64_t kFiveTileThreeTripRowsDenSm100 = 9;
constexpr uint64_t kFifthTileLateMaxNumSm100 = 279;
constexpr uint64_t kFifthTileLateMaxDenSm100 = 1024;
constexpr uint64_t kFiveTileLateRowsNumSm100 = 15;
constexpr uint64_t kFiveTileLateRowsDenSm100 = 2;
// sm_100, 6- and 7-tile 16-CTA chunks: the persistent kernel's trip cap past 4
// rows per shared-tile cluster, raised to the persistent kernel's 24-trip fast
// path when the last tile of the chunk holds at most 6/8 (6 tiles) or 4/8
// (7 tiles) of a tile, unless the temperature is per row.
constexpr uint64_t kSixTileC16MaxTripsSm100 = 20;
constexpr uint64_t kSevenTileC16MaxTripsSm100 = 22;
constexpr uint64_t kPartialTileC16MaxTripsSm100 = 24;
constexpr uint32_t kSixTilePartialMaxElementsSm100 = 1536;
constexpr uint32_t kSevenTilePartialMaxElementsSm100 = 1024;
// sm_103, 6-tile 8-CTA chunks below 2.5 rows per persistent cluster: the
// persistent kernel from 20/9 rows per persistent cluster on (80 to 89 rows on
// GB300).
constexpr uint64_t kSixTileC8PersistentRowsNumSm103 = 20;
constexpr uint64_t kSixTileC8PersistentRowsDenSm103 = 9;
// The balanced chunk (see select_cached_launch): rounding, the minimum
// cut of the whole-tile chunk (1/32; 1/16 past one wave, and for a one-wave
// 4-tile 16-CTA row kernel), the waves it covers, and the minimum cut (1/10)
// of a 16-CTA 8-tile chunk (in one wave on sm_100).
constexpr uint32_t kBalancedChunkElements = 32;
constexpr uint64_t kBalancedChunkMinCutDen = 32;
constexpr uint64_t kBalancedChunkMultiWaveCutDen = 16;
constexpr uint64_t kBalancedChunkMaxWaves = 5;
constexpr uint64_t kBalancedChunkSixteenCtaMinCutDen = 10;
// sm_103: the cuts at which the 8-CTA shared-tile and persistent kernels take
// the balanced chunk: at least 1/MinCutDen and under 1/MaxCutDen of the
// whole-tile chunk (1/1: no upper bound). The shared-tile kernel takes it at
// any cut the rule above allows, the persistent kernel from a cut of 1/10 on.
constexpr uint64_t kBalancedChunkEightCtaSharedMinCutDenSm103 = 32;
constexpr uint64_t kBalancedChunkEightCtaSharedMaxCutDenSm103 = 1;
constexpr uint64_t kBalancedChunkEightCtaPersistentMinCutDenSm103 = 10;
constexpr uint64_t kBalancedChunkEightCtaPersistentMaxCutDenSm103 = 1;
constexpr int kMaxDevices = 64;

enum class ParameterKind : int {
  kNone = 0,
  kScalar = 1,
  kPerRow = 2,
};

// Stable numeric values are intentionally exposed by softmax_route for tests
// and benchmark evidence. Keep this selection in C++ so an API-level
// correctness test cannot silently pass through OnlineSoftmax fallback.
enum class SoftmaxRoute : int64_t {
  kFallback = 0,
  kWarp = 1,
  kRowwise = 2,
  kBootstrap = 3,
  // Value 4 is not assigned; the values stay stable for tests and benchmarks.
  kCachedCluster = 5,
  kCachedClusterPersistent = 6,
  kCachedClusterSharedTiles = 7,
};

struct CachedVariant {
  int cluster_ctas;
  int max_tiles;
};

// The row-caching cluster family needs 32-byte aligned rows: 256-bit
// vector accesses on both buffers and a vocabulary that is a multiple of 8.
bool use_cached_cluster_kernel(uint32_t vocab_size) {
  return vocab_size % kCachedVectorElements == 0 && vocab_size <= kCachedMaxVocabSize;
}

// Every generated route issues 128- or 256-bit vector accesses relative to
// the buffer base, so a base pointer that is not 32-byte aligned (a
// storage-offset view) must take the OnlineSoftmax fallback.
bool buffers_aligned(const void* logits, const void* output) {
  return reinterpret_cast<uintptr_t>(logits) % kCachedAlignmentBytes == 0 &&
         (output == nullptr || reinterpret_cast<uintptr_t>(output) % kCachedAlignmentBytes == 0);
}

// Whole tiles per CTA covering the row.
int cached_chunk_elements(uint32_t vocab_size, int cluster_ctas) {
  const uint32_t per_cta = ceil_div(vocab_size, static_cast<uint32_t>(cluster_ctas));
  return static_cast<int>(ceil_div(per_cta, static_cast<uint32_t>(kCachedTileElements)) *
                          kCachedTileElements);
}

cudaError_t row_resident_clusters(int cluster_ctas, bool pdl, int device, int* clusters);
cudaError_t ten_cta_row_resident_clusters(bool pdl, int device, int* clusters);
cudaError_t shared_tile_resident_clusters(int cluster_ctas, bool pdl, int device, int* clusters);
cudaError_t one_cta_per_sm_clusters(int cluster_ctas, int device, int* clusters);

// The wide kernels' cluster sizes, the larger first, and the narrowest of
// their register chunks that covers the row's chunk: one register tile costs
// about 0.7 % at equal work.
constexpr int kWideClusterSizes[] = {12, 10};
// A 10-CTA chunk is 1.6 times the 16-CTA chunk; the 10-CTA kernel takes the
// batch only when the 16-CTA chunk exceeds 6 tiles (measured on B200).
constexpr int kTenCtaMinSixteenChunkTiles = 7;
// sm_103: the 8-CTA shared-tile kernel slows down past 7/10 of its resident
// clusters (between 51 and 52 rows on GB300).
constexpr uint64_t kSharedTilesSlowRowsNumSm103 = 7;
constexpr uint64_t kSharedTilesSlowRowsDenSm103 = 10;
// sm_103: a 6-tile 8-CTA chunk takes the two-wave 4-tile 16-CTA kernel past 3/4
// of the resident 8-CTA shared-tile clusters, up to 7/4 of the resident 4-tile
// 16-CTA clusters (56 and 57 rows on GB300).
constexpr uint64_t kSixTileTwoWavesSharedNumSm103 = 3;
constexpr uint64_t kSixTileTwoWavesSharedDenSm103 = 4;
constexpr uint64_t kSixTileTwoWavesRowsNumSm103 = 7;
constexpr uint64_t kSixTileTwoWavesRowsDenSm103 = 4;
// sm_103: a 5- to 7-tile 16-CTA chunk whose 10-CTA chunk holds at most 10
// tiles takes the 10-CTA kernel from 11/14 of its resident clusters up to all
// of them (22 to 28 rows on GB300), unless the temperature is per row.
constexpr int kTenCtaRowsMaxTilesSm103 = 10;
constexpr uint64_t kTenCtaRowsNumSm103 = 11;
constexpr uint64_t kTenCtaRowsDenSm103 = 14;
// sm_100: a 4-tile 4-CTA chunk keeps spreading to 8 CTAs at twice as many CTAs
// as SMs from 14 tiles of vocabulary, and up to 8 CTAs past that from 15.
constexpr uint32_t kSpreadFourTileMinVocabTilesSm100 = 14;
constexpr uint32_t kSpreadFourTilePastMinVocabTilesSm100 = 15;
constexpr uint64_t kSpreadFourTilePastMaxCtasSm100 = 8;

int wide_max_tiles(uint32_t vocab_size, int cluster_ctas) {
  const int tiles = cached_chunk_elements(vocab_size, cluster_ctas) / kCachedTileElements;
  if (tiles <= 8) {
    return 8;
  }
  if (tiles <= 10) {
    return 10;
  }
  return cluster_ctas == 12 ? 11 : 13;
}

// Shape policy measured on GB300/B200: four resident 32 KiB
// CTAs per SM stream best, so a row is split into the smallest cluster (at most
// 8 CTAs) whose 32 KiB chunks cover it; rows wider than 256 KiB fall back to
// 64 KiB chunks so the cluster stays within 16 CTAs. Short grids (fewer CTAs
// than twice the SM count) spread each row over more CTAs until the machine
// is covered twice or every CTA would own less than one tile. When spreading
// leaves every CTA at most four tiles, the 4-tile specialization (four
// resident CTAs per SM) replaces the 8-tile one. A spread stops before a
// cluster size with fewer resident 4-tile clusters than rows, where the excess
// clusters would run a second wave, when measured faster: before 16 CTAs if
// 8-CTA chunks hold at most four tiles, before 8 CTAs if 4-CTA chunks hold at
// most three; on sm_100 a 4-tile 4-CTA chunk of at least 14 tiles of
// vocabulary still spreads to 8 CTAs at twice as many CTAs as SMs, and one of
// at least 15 tiles up to 8 CTAs past that. A 6- to 8-tile 8-CTA chunk whose
// 16-CTA chunk holds at most four tiles takes the 4-tile 16-CTA kernel in two
// waves while its rows stay within a multiple of the resident 4-tile 16-CTA
// clusters: on sm_100 9/5 for a full 8-tile chunk and 10/7 for a 6- or 7-tile
// chunk; on sm_103 3/2 for a full 8-tile chunk, or 5/3 past 7/10 of the
// resident 8-CTA shared-tile clusters, and 7/4 for a 6-tile chunk past 3/4 of
// them. On sm_103 a 5- to 7-tile 16-CTA chunk whose 10-CTA chunk holds at most
// 10 tiles takes the 10-CTA kernel from 11/14 of its resident clusters up to
// all of them, unless the temperature is per row.
// A batch of 5-8-tile 16-CTA chunks with one row more than 16-CTA
// clusters fit one CTA per SM would put two CTAs on the SMs of its last
// cluster; it takes the wide kernel of 12 CTAs, or else 10 for a 16-CTA chunk
// of more than 6 tiles, when that many clusters of it fit one CTA per SM
// (measured faster on B200 and GB300; with two or more rows over, the 16-CTA
// launch is as fast or faster).
cudaError_t select_cached_variant(uint32_t rows, uint32_t vocab_size, int sm_count,
                                  ParameterKind parameter_kind, bool pdl, int device,
                                  CachedVariant* variant) {
  constexpr int kClusterSizes[] = {1, 2, 4, 8, 16};
  constexpr int kTileOptions[] = {4, 8};
  CachedVariant small_chunk{0, 0};
  CachedVariant wide_chunk{0, 0};
  CachedVariant any{0, 0};
  for (int cluster_ctas : kClusterSizes) {
    for (int max_tiles : kTileOptions) {
      const uint64_t capacity =
          static_cast<uint64_t>(cluster_ctas) * max_tiles * kCachedTileElements;
      if (capacity < vocab_size) {
        continue;
      }
      if (any.cluster_ctas == 0) {
        any = {cluster_ctas, max_tiles};
      }
      if (max_tiles == 4 && cluster_ctas <= kCachedPortableClusterSize &&
          small_chunk.cluster_ctas == 0) {
        small_chunk = {cluster_ctas, max_tiles};
      }
      if (max_tiles == 8 && wide_chunk.cluster_ctas == 0) {
        wide_chunk = {cluster_ctas, max_tiles};
      }
    }
  }
  CachedVariant selected = small_chunk.cluster_ctas != 0  ? small_chunk
                           : wide_chunk.cluster_ctas != 0 ? wide_chunk
                                                          : any;
  const uint64_t useful_ctas = std::max<uint64_t>(
      1, ceil_div(static_cast<uint64_t>(vocab_size), static_cast<uint64_t>(kCachedTileElements)));
  while (static_cast<uint64_t>(rows) * selected.cluster_ctas <
             2 * static_cast<uint64_t>(sm_count) &&
         selected.cluster_ctas < kCachedMaxClusterSize &&
         static_cast<uint64_t>(selected.cluster_ctas) * 2 <= useful_ctas) {
    const int next_ctas = selected.cluster_ctas * 2;
    const int chunk_tiles =
        cached_chunk_elements(vocab_size, selected.cluster_ctas) / kCachedTileElements;
    const int max_capped_tiles = next_ctas == kCachedMaxClusterSize        ? 4
                                 : next_ctas == kCachedPortableClusterSize ? 3
                                                                           : 0;
    if (chunk_tiles <= max_capped_tiles) {
      int resident = 0;
      const cudaError_t status = row_resident_clusters(next_ctas, pdl, device, &resident);
      if (status != cudaSuccess) {
        return status;
      }
      if (resident > 0 && rows > static_cast<uint32_t>(resident)) {
        break;
      }
    }
    selected.cluster_ctas *= 2;
  }
  if (selected.cluster_ctas == 4 && selected.max_tiles == 4 &&
      vocab_size >= kSpreadFourTileMinVocabTilesSm100 * kCachedTileElements) {
    const uint64_t ctas = static_cast<uint64_t>(rows) * selected.cluster_ctas;
    const uint64_t twice_sms = 2 * static_cast<uint64_t>(sm_count);
    if (ctas <= twice_sms ||
        (ctas <= twice_sms + kSpreadFourTilePastMaxCtasSm100 &&
         vocab_size >= kSpreadFourTilePastMinVocabTilesSm100 * kCachedTileElements)) {
      int device_minor = 0;
      const cudaError_t status =
          cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor, device);
      if (status != cudaSuccess) {
        return status;
      }
      if (device_minor == 0) {
        selected.cluster_ctas = kCachedPortableClusterSize;
      }
    }
  }
  if (selected.max_tiles == kCachedMaxTiles &&
      cached_chunk_elements(vocab_size, selected.cluster_ctas) <= 4 * kCachedTileElements) {
    selected.max_tiles = 4;
  }
  const int portable_chunk_tiles =
      cached_chunk_elements(vocab_size, kCachedPortableClusterSize) / kCachedTileElements;
  if (selected.cluster_ctas == kCachedPortableClusterSize &&
      selected.max_tiles == kCachedMaxTiles && portable_chunk_tiles > kSharedTilesAnyRowsMaxTiles &&
      cached_chunk_elements(vocab_size, kCachedMaxClusterSize) <= 4 * kCachedTileElements) {
    int device_minor = 0;
    cudaError_t status =
        cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor, device);
    if (status != cudaSuccess) {
      return status;
    }
    int resident = 0;
    status = row_resident_clusters(kCachedMaxClusterSize, pdl, device, &resident);
    if (status != cudaSuccess) {
      return status;
    }
    const uint64_t rows64 = rows;
    const uint64_t resident64 = static_cast<uint64_t>(resident > 0 ? resident : 0);
    uint64_t rows_num = 0;
    uint64_t rows_den = 1;
    if (device_minor == 0) {
      rows_num = portable_chunk_tiles == kCachedMaxTiles ? 9 : 10;
      rows_den = portable_chunk_tiles == kCachedMaxTiles ? 5 : 7;
    } else if (portable_chunk_tiles == kCachedMaxTiles) {
      rows_num = 3;
      rows_den = 2;
    }
    bool two_waves = rows64 * rows_den <= rows_num * resident64;
    if (!two_waves && device_minor != 0 &&
        (portable_chunk_tiles == kCachedMaxTiles || portable_chunk_tiles == 6) && resident64 > 0) {
      int shared_resident = 0;
      status =
          shared_tile_resident_clusters(kCachedPortableClusterSize, pdl, device, &shared_resident);
      if (status != cudaSuccess) {
        return status;
      }
      const bool six_tiles = portable_chunk_tiles != kCachedMaxTiles;
      const uint64_t shared_num =
          six_tiles ? kSixTileTwoWavesSharedNumSm103 : kSharedTilesSlowRowsNumSm103;
      const uint64_t shared_den =
          six_tiles ? kSixTileTwoWavesSharedDenSm103 : kSharedTilesSlowRowsDenSm103;
      rows_num = six_tiles ? kSixTileTwoWavesRowsNumSm103 : 5;
      rows_den = six_tiles ? kSixTileTwoWavesRowsDenSm103 : 3;
      two_waves = shared_resident > 0 &&
                  rows64 * shared_den > shared_num * static_cast<uint64_t>(shared_resident) &&
                  rows64 * rows_den <= rows_num * resident64;
    }
    if (two_waves) {
      selected = {kCachedMaxClusterSize, 4};
    }
  }
  if (selected.cluster_ctas == kCachedMaxClusterSize && selected.max_tiles == kCachedMaxTiles) {
    int sixteen = 0;
    cudaError_t status = one_cta_per_sm_clusters(kCachedMaxClusterSize, device, &sixteen);
    if (status != cudaSuccess) {
      return status;
    }
    if (parameter_kind != ParameterKind::kPerRow &&
        wide_max_tiles(vocab_size, 10) <= kTenCtaRowsMaxTilesSm103) {
      int device_minor = 0;
      status = cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor, device);
      if (status != cudaSuccess) {
        return status;
      }
      int ten = 0;
      if (device_minor != 0) {
        status = ten_cta_row_resident_clusters(pdl, device, &ten);
        if (status != cudaSuccess) {
          return status;
        }
      }
      const uint64_t rows64 = rows;
      const uint64_t ten64 = static_cast<uint64_t>(ten > 0 ? ten : 0);
      if (ten64 > 0 && rows64 * kTenCtaRowsDenSm103 >= kTenCtaRowsNumSm103 * ten64 &&
          rows64 <= ten64) {
        *variant = {10, wide_max_tiles(vocab_size, 10)};
        return cudaSuccess;
      }
    }
    if (sixteen > 0 && rows == static_cast<uint32_t>(sixteen) + 1) {
      for (int wide_ctas : kWideClusterSizes) {
        int fit = 0;
        status = one_cta_per_sm_clusters(wide_ctas, device, &fit);
        if (status != cudaSuccess) {
          return status;
        }
        if (rows <= static_cast<uint32_t>(fit)) {
          if (wide_ctas != 10 ||
              cached_chunk_elements(vocab_size, kCachedMaxClusterSize) / kCachedTileElements >=
                  kTenCtaMinSixteenChunkTiles) {
            selected = {wide_ctas, wide_max_tiles(vocab_size, wide_ctas)};
          }
          break;
        }
      }
    }
  }
  *variant = selected;
  return cudaSuccess;
}

// Each row-caching cluster kernel NAME has a twin NAME_pdl for PDL launches:
// it waits on the prerequisite grid at entry and triggers its dependents at
// exit.
const void* cached_cluster_kernel(CachedVariant variant, bool pdl) {
#define FLASHINFER_CACHED_SOFTMAX_CASE(C, T)                                               \
  if (variant.cluster_ctas == C && variant.max_tiles == T) {                               \
    return pdl ? reinterpret_cast<const void*>(                                            \
                     kernel_flashinfer_blackwell_softmax_cached_cluster_c##C##_t##T##_pdl) \
               : reinterpret_cast<const void*>(                                            \
                     kernel_flashinfer_blackwell_softmax_cached_cluster_c##C##_t##T);      \
  }
  FLASHINFER_CACHED_SOFTMAX_CASE(1, 4)
  FLASHINFER_CACHED_SOFTMAX_CASE(2, 4)
  FLASHINFER_CACHED_SOFTMAX_CASE(4, 4)
  FLASHINFER_CACHED_SOFTMAX_CASE(8, 4)
  FLASHINFER_CACHED_SOFTMAX_CASE(8, 8)
  FLASHINFER_CACHED_SOFTMAX_CASE(16, 4)
  FLASHINFER_CACHED_SOFTMAX_CASE(16, 8)
  FLASHINFER_CACHED_SOFTMAX_CASE(10, 8)
  FLASHINFER_CACHED_SOFTMAX_CASE(10, 10)
  FLASHINFER_CACHED_SOFTMAX_CASE(10, 13)
  FLASHINFER_CACHED_SOFTMAX_CASE(12, 8)
  FLASHINFER_CACHED_SOFTMAX_CASE(12, 10)
  FLASHINFER_CACHED_SOFTMAX_CASE(12, 11)
#undef FLASHINFER_CACHED_SOFTMAX_CASE
  return nullptr;
}

// Dynamic shared memory of each generated kernel (its SMEM_TOTAL): the
// exchange mbarrier, then from byte 1024 the CTA's table of 8 warp pairs and
// the cluster's table of 8 * C pairs. The persistent kernels add the 8-tile
// buffer and keep two of each table.
size_t cached_dynamic_smem_bytes(int cluster_ctas) {
  switch (cluster_ctas) {
    case 1:
      return 128;
    case 2:
      return 1280;
    case 4:
      return 1408;
    case 8:
      return 1664;
    case 10:
      return 1792;
    case 12:
      return 1920;
    default:
      return 2176;
  }
}

size_t persistent_dynamic_smem_bytes(int cluster_ctas) { return cluster_ctas == 8 ? 67712 : 68736; }

// The shared-tile kernels add the 4-tile buffer and its mbarriers.
size_t shared_tiles_dynamic_smem_bytes(int cluster_ctas) {
  return cluster_ctas == 8 ? 34432 : 34944;
}

// The persistent kernels serve the 8-tile variants of 8 and 16 CTAs.
const void* cached_persistent_kernel(CachedVariant variant) {
  if (variant.max_tiles != kCachedMaxTiles) {
    return nullptr;
  }
  if (variant.cluster_ctas == 8) {
    return reinterpret_cast<const void*>(
        kernel_flashinfer_blackwell_softmax_cached_cluster_persistent_c8_t8);
  }
  if (variant.cluster_ctas == 16) {
    return reinterpret_cast<const void*>(
        kernel_flashinfer_blackwell_softmax_cached_cluster_persistent_c16_t8);
  }
  return nullptr;
}

// The shared-tile kernels serve the 8-tile variants of 8 and 16 CTAs.
const void* cached_shared_tiles_kernel(CachedVariant variant, bool pdl) {
  if (variant.max_tiles != kCachedMaxTiles) {
    return nullptr;
  }
  if (variant.cluster_ctas == 8) {
    return pdl ? reinterpret_cast<const void*>(
                     kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t4_s4_pdl)
               : reinterpret_cast<const void*>(
                     kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t4_s4);
  }
  if (variant.cluster_ctas == 16) {
    return pdl ? reinterpret_cast<const void*>(
                     kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t4_s4_pdl)
               : reinterpret_cast<const void*>(
                     kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t4_s4);
  }
  return nullptr;
}

// Function attributes of a cluster launch: non-portable clusters above 8 CTAs,
// and dynamic shared memory above the 48 KiB default.
cudaError_t configure_cached_kernel(const void* kernel, int cluster_ctas,
                                    size_t dynamic_smem_bytes) {
  cudaError_t status = cudaSuccess;
  if (cluster_ctas > kCachedPortableClusterSize) {
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if (status == cudaSuccess && dynamic_smem_bytes > 48 * 1024) {
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem_bytes));
  }
  return status;
}

// Resident clusters per device of the persistent and shared-tile kernels of 8
// and 16 CTAs per PDL mode (a shared-tile kernel or its _pdl twin; the same
// persistent kernel in both modes), plus one (0: not yet queried). The query
// also sets the kernel's function attributes on that device.
enum ResidentKernel : int {
  kResidentPersistent = 0,
  kResidentSharedTiles = 1,
};
std::atomic<int> g_resident_clusters[kMaxDevices][2][2][2];

// Resident clusters of a cluster kernel on the current device, kept in *cached
// (plus one; 0: not yet queried) unless it is null.
cudaError_t query_resident_clusters(std::atomic<int>* cached, const void* kernel, int cluster_ctas,
                                    size_t dynamic_smem_bytes, int* clusters) {
  if (cached != nullptr) {
    const int value = cached->load(std::memory_order_relaxed);
    if (value > 0) {
      *clusters = value - 1;
      return cudaSuccess;
    }
  }
  cudaError_t status = configure_cached_kernel(kernel, cluster_ctas, dynamic_smem_bytes);
  if (status != cudaSuccess) {
    return status;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned int>(cluster_ctas));
  config.blockDim = dim3(kCachedThreads);
  config.dynamicSmemBytes = dynamic_smem_bytes;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = static_cast<unsigned int>(cluster_ctas);
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  int value = 0;
  status = cudaOccupancyMaxActiveClusters(&value, kernel, &config);
  if (status != cudaSuccess) {
    return status;
  }
  if (cached != nullptr) {
    cached->store(value + 1, std::memory_order_relaxed);
  }
  *clusters = value;
  return cudaSuccess;
}

cudaError_t resident_clusters(ResidentKernel which, const void* kernel, int cluster_ctas, bool pdl,
                              size_t dynamic_smem_bytes, int device, int* clusters) {
  std::atomic<int>* cached = device >= 0 && device < kMaxDevices
                                 ? &g_resident_clusters[device][which][cluster_ctas == 16][pdl]
                                 : nullptr;
  return query_resident_clusters(cached, kernel, cluster_ctas, dynamic_smem_bytes, clusters);
}

// Never launched: its resident clusters at the opt-in shared memory maximum
// are the clusters of that size that fit one CTA per SM.
__global__ void cached_one_cta_per_sm_probe() {}

// Clusters of 10, 12 and 16 CTAs that fit one CTA per SM per device, plus one
// (0: not yet queried).
std::atomic<int> g_one_cta_per_sm_clusters[kMaxDevices][3];

cudaError_t one_cta_per_sm_clusters(int cluster_ctas, int device, int* clusters) {
  std::atomic<int>* cached = device >= 0 && device < kMaxDevices
                                 ? &g_one_cta_per_sm_clusters[device][cluster_ctas == 10   ? 0
                                                                      : cluster_ctas == 12 ? 1
                                                                                           : 2]
                                 : nullptr;
  if (cached != nullptr) {
    const int value = cached->load(std::memory_order_relaxed);
    if (value > 0) {
      *clusters = value - 1;
      return cudaSuccess;
    }
  }
  int smem_optin = 0;
  const cudaError_t status =
      cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) {
    return status;
  }
  return query_resident_clusters(cached, reinterpret_cast<const void*>(cached_one_cta_per_sm_probe),
                                 cluster_ctas, static_cast<size_t>(smem_optin), clusters);
}

// Resident clusters per device of the 4-tile row kernels of 8 and 16 CTAs and
// their PDL twins, plus one (0: not yet queried).
std::atomic<int> g_row_resident_clusters[kMaxDevices][2][2];

cudaError_t row_resident_clusters(int cluster_ctas, bool pdl, int device, int* clusters) {
  std::atomic<int>* cached = device >= 0 && device < kMaxDevices
                                 ? &g_row_resident_clusters[device][cluster_ctas == 16][pdl]
                                 : nullptr;
  return query_resident_clusters(cached, cached_cluster_kernel({cluster_ctas, 4}, pdl),
                                 cluster_ctas, cached_dynamic_smem_bytes(cluster_ctas), clusters);
}

// Resident clusters per device of the 8-tile 10-CTA row kernel and its PDL
// twin, plus one (0: not yet queried).
std::atomic<int> g_ten_cta_row_resident_clusters[kMaxDevices][2];

cudaError_t ten_cta_row_resident_clusters(bool pdl, int device, int* clusters) {
  std::atomic<int>* cached =
      device >= 0 && device < kMaxDevices ? &g_ten_cta_row_resident_clusters[device][pdl] : nullptr;
  return query_resident_clusters(cached, cached_cluster_kernel({10, 8}, pdl), 10,
                                 cached_dynamic_smem_bytes(10), clusters);
}

// Resident clusters of the shared-tile kernel of 8 or 16 CTAs (or its _pdl
// twin) on a device; the cache slot select_cached_launch fills.
cudaError_t shared_tile_resident_clusters(int cluster_ctas, bool pdl, int device, int* clusters) {
  return resident_clusters(
      kResidentSharedTiles, cached_shared_tiles_kernel({cluster_ctas, kCachedMaxTiles}, pdl),
      cluster_ctas, pdl, shared_tiles_dynamic_smem_bytes(cluster_ctas), device, clusters);
}

struct CachedLaunch {
  CachedVariant variant;
  const void* kernel;
  uint64_t clusters;
  size_t dynamic_smem_bytes;
  SoftmaxRoute route;
  // Elements of the row per CTA (set by select_cached_launch).
  uint32_t chunk_elements = 0;
};

// One cluster per row. An 8-tile variant whose rows exceed the resident
// persistent clusters takes the shared-tile kernel (one cluster per row) or
// the persistent kernel on every resident cluster, by the rule above.
cudaError_t select_cached_route(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind,
                                bool pdl, int device, CachedLaunch* launch) {
  int sm_count = 0;
  cudaError_t status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }
  CachedVariant variant{};
  status = select_cached_variant(rows, vocab_size, sm_count, parameter_kind, pdl, device, &variant);
  if (status != cudaSuccess) {
    return status;
  }
  *launch = {variant, cached_cluster_kernel(variant, pdl), rows,
             cached_dynamic_smem_bytes(variant.cluster_ctas), SoftmaxRoute::kCachedCluster};
  if (launch->kernel == nullptr) {
    return cudaErrorNotSupported;
  }
  const void* persistent = cached_persistent_kernel(variant);
  if (persistent == nullptr) {
    return cudaSuccess;
  }
  const size_t persistent_smem = persistent_dynamic_smem_bytes(variant.cluster_ctas);
  int resident = 0;
  status = resident_clusters(kResidentPersistent, persistent, variant.cluster_ctas, pdl,
                             persistent_smem, device, &resident);
  if (status != cudaSuccess) {
    return status;
  }
  if (resident <= 0 || rows <= static_cast<uint32_t>(resident)) {
    return cudaSuccess;
  }
  const void* shared_tiles = cached_shared_tiles_kernel(variant, pdl);
  const size_t shared_tiles_smem = shared_tiles_dynamic_smem_bytes(variant.cluster_ctas);
  int shared_resident = 0;
  status = resident_clusters(kResidentSharedTiles, shared_tiles, variant.cluster_ctas, pdl,
                             shared_tiles_smem, device, &shared_resident);
  if (status != cudaSuccess) {
    return status;
  }
  int device_minor = 0;
  status = cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) {
    return status;
  }
  const uint32_t chunk_elements = cached_chunk_elements(vocab_size, variant.cluster_ctas);
  const uint64_t rows64 = rows;
  const uint64_t persistent_clusters = static_cast<uint64_t>(resident);
  const uint64_t shared_clusters = static_cast<uint64_t>(shared_resident > 0 ? shared_resident : 0);
  const bool under_persistent =
      rows64 * kPersistentRowsPerClusterDen < persistent_clusters * kPersistentRowsPerClusterNum;
  const uint64_t trips = (rows64 + persistent_clusters - 1) / persistent_clusters;
  const bool sixteen = variant.cluster_ctas == kCachedMaxClusterSize;
  const bool short_last_trip =
      rows64 <= shared_clusters &&
      rows64 * (sixteen ? kPersistentTripFillDenC16 : kPersistentTripFillDenC8) <
          trips * persistent_clusters *
              (sixteen ? kPersistentTripFillNumC16 : kPersistentTripFillNumC8);
  const bool short_trips =
      rows64 * kPersistentTripFillDenC16 < trips * persistent_clusters * kPersistentTripFillNumC16;
  const int chunk_tiles = static_cast<int>(chunk_elements / kCachedTileElements);
  bool use_shared_tiles = true;
  if (chunk_tiles <= kSharedTilesAnyRowsMaxTiles) {
    // Eight times the row's share of the chunk's last tile, in elements: the
    // tile is thin when at most 2/8 full on sm_100 or under 3/8 full on sm_103.
    const uint64_t per_cta = ceil_div(vocab_size, static_cast<uint32_t>(variant.cluster_ctas));
    const uint64_t last_tile_x8 =
        (per_cta - static_cast<uint64_t>(chunk_tiles - 1) * kCachedTileElements) * 8;
    const bool thin_last_tile =
        device_minor == 0 ? last_tile_x8 <= kFifthTileMaxThinEighthsSm100 * kCachedTileElements
                          : last_tile_x8 < kFifthTileMinFillEighthsSm103 * kCachedTileElements;
    if (sixteen) {
      use_shared_tiles = !(trips == kNarrowChunkC16PersistentTrips &&
                           rows64 * kNarrowChunkC16TripFillDen >=
                               trips * persistent_clusters * kNarrowChunkC16TripFillNum);
    } else {
      // A fifth tile at least 3/8 full (on sm_103 every fifth tile that is
      // not thin) keeps the shared-tile kernel for up to two persistent trips.
      const bool early_persistent =
          device_minor != 0 ||
          last_tile_x8 >= kFifthTileEarlyPersistentEighthsSm100 * kCachedTileElements;
      bool few_rows = early_persistent ? trips <= kFiveTileSharedTilesMaxTrips : under_persistent;
      bool short_rows = short_trips;
      bool many_rows = rows64 >= kSharedTilesRowsPerClusterFullChunkC8 * shared_clusters;
      if (device_minor == 0 && !thin_last_tile) {
        if (trips == kFiveTileThreeTrips && rows64 * kFiveTileThreeTripRowsDenSm100 >=
                                                kFiveTileThreeTripRowsNumSm100 * shared_clusters) {
          few_rows = false;
          short_rows = false;
        }
        if (last_tile_x8 * kFifthTileLateMaxDenSm100 <=
                kFifthTileLateMaxNumSm100 * 8 * kCachedTileElements &&
            rows64 * kFiveTileLateRowsDenSm100 < kFiveTileLateRowsNumSm100 * shared_clusters) {
          many_rows = false;
        }
      }
      use_shared_tiles = few_rows || short_rows || thin_last_tile || many_rows;
    }
  } else if (chunk_tiles == kCachedMaxTiles) {
    const uint64_t rows_per_cluster = !sixteen ? kSharedTilesRowsPerClusterFullChunkC8
                                      : device_minor == 0
                                          ? kSharedTilesRowsPerClusterFullChunkC16Sm100
                                          : kSharedTilesRowsPerClusterFullChunkC16Sm103;
    use_shared_tiles = short_last_trip || rows64 >= rows_per_cluster * shared_clusters;
  } else {
    const uint32_t per_cta = ceil_div(vocab_size, static_cast<uint32_t>(variant.cluster_ctas));
    const uint32_t last_tile = per_cta % kCachedTileElements;
    const bool partial_last_tile =
        parameter_kind != ParameterKind::kPerRow && last_tile != 0 &&
        last_tile <= (chunk_tiles == 6 ? kSixTilePartialMaxElementsSm100
                                       : kSevenTilePartialMaxElementsSm100);
    const uint64_t max_trips = partial_last_tile  ? kPartialTileC16MaxTripsSm100
                               : chunk_tiles == 6 ? kSixTileC16MaxTripsSm100
                                                  : kSevenTileC16MaxTripsSm100;
    const bool many_rows = rows64 >= kSharedTilesRowsPerCluster * shared_clusters &&
                           !(device_minor == 0 && sixteen && trips <= max_trips);
    bool few_rows =
        device_minor == 3 && (chunk_tiles == 6 || sixteen) ? under_persistent : short_last_trip;
    if (device_minor == 3 && !sixteen && chunk_tiles == 6 &&
        rows64 * kSixTileC8PersistentRowsDenSm103 >=
            kSixTileC8PersistentRowsNumSm103 * persistent_clusters) {
      few_rows = false;
    }
    use_shared_tiles = few_rows || many_rows;
  }
  if (shared_resident > 0 && use_shared_tiles) {
    *launch = {variant, shared_tiles, rows, shared_tiles_smem,
               SoftmaxRoute::kCachedClusterSharedTiles};
  } else if (shared_resident > 0 || !under_persistent) {
    *launch = {variant, persistent, persistent_clusters, persistent_smem,
               SoftmaxRoute::kCachedClusterPersistent};
  }
  return cudaSuccess;
}

// The launch: select_cached_route's kernel and the chunk each CTA covers. A CTA
// covers whole tiles, except that the chunk is ceil(V / C) rounded up to 32
// elements (CTA starts 128-byte aligned within the row; the same tile count)
// when that balanced chunk is more than 1/32 shorter than the whole-tile chunk
// and the rows fit one wave of the launched kernel's resident clusters, or it
// is more than 1/16 shorter and the rows fit five waves, for these kernels
// (measured on B200 and GB300): the 8-CTA shared-tile kernel; the 8-CTA
// persistent kernel on sm_100, or on sm_103 with a cut of at least 1/10; the
// 4-tile 16-CTA row kernel past one wave or with a cut of more than 1/16; the
// 16-CTA shared-tile and persistent kernels with a cut of at least 1/10, or on
// sm_100 past one wave. The other kernels keep whole tiles.
cudaError_t select_cached_launch(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind,
                                 bool pdl, int device, CachedLaunch* launch) {
  cudaError_t status = select_cached_route(rows, vocab_size, parameter_kind, pdl, device, launch);
  if (status != cudaSuccess) {
    return status;
  }
  const int cluster_ctas = launch->variant.cluster_ctas;
  const uint32_t whole = static_cast<uint32_t>(cached_chunk_elements(vocab_size, cluster_ctas));
  launch->chunk_elements = whole;
  const bool shared_or_persistent = launch->route == SoftmaxRoute::kCachedClusterSharedTiles ||
                                    launch->route == SoftmaxRoute::kCachedClusterPersistent;
  const bool eight_cta_chunk = shared_or_persistent && cluster_ctas == kCachedPortableClusterSize;
  const bool sixteen_cta_chunk = shared_or_persistent && cluster_ctas == kCachedMaxClusterSize &&
                                 launch->variant.max_tiles == kCachedMaxTiles;
  const bool sixteen_cta_rows = launch->route == SoftmaxRoute::kCachedCluster &&
                                cluster_ctas == kCachedMaxClusterSize &&
                                launch->variant.max_tiles == 4;
  if (!eight_cta_chunk && !sixteen_cta_chunk && !sixteen_cta_rows) {
    return cudaSuccess;
  }
  int device_minor = 0;
  status = cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) {
    return status;
  }
  const uint32_t balanced =
      ceil_div(ceil_div(vocab_size, static_cast<uint32_t>(cluster_ctas)), kBalancedChunkElements) *
      kBalancedChunkElements;
  const uint64_t cut = whole - balanced;
  if (cut * kBalancedChunkMinCutDen <= whole) {
    return cudaSuccess;
  }
  if (eight_cta_chunk && device_minor != 0) {
    const bool persistent = launch->route == SoftmaxRoute::kCachedClusterPersistent;
    const uint64_t min_cut_den = persistent ? kBalancedChunkEightCtaPersistentMinCutDenSm103
                                            : kBalancedChunkEightCtaSharedMinCutDenSm103;
    const uint64_t max_cut_den = persistent ? kBalancedChunkEightCtaPersistentMaxCutDenSm103
                                            : kBalancedChunkEightCtaSharedMaxCutDenSm103;
    if (cut * min_cut_den < whole || cut * max_cut_den >= whole) {
      return cudaSuccess;
    }
  }
  // Resident clusters of the launched kernel.
  uint64_t resident = 0;
  if (launch->route == SoftmaxRoute::kCachedClusterPersistent) {
    resident = launch->clusters;
  } else if (launch->route == SoftmaxRoute::kCachedClusterSharedTiles) {
    int shared = 0;
    status = shared_tile_resident_clusters(cluster_ctas, pdl, device, &shared);
    if (status != cudaSuccess) {
      return status;
    }
    resident = static_cast<uint64_t>(std::max(shared, 0));
  } else {
    int row = 0;
    status = row_resident_clusters(cluster_ctas, pdl, device, &row);
    if (status != cudaSuccess) {
      return status;
    }
    resident = static_cast<uint64_t>(std::max(row, 0));
  }
  const uint64_t rows64 = rows;
  const bool large_cut = cut * kBalancedChunkMultiWaveCutDen > whole;
  if (resident == 0 ||
      !(rows64 <= resident || (large_cut && rows64 <= kBalancedChunkMaxWaves * resident))) {
    return cudaSuccess;
  }
  if (sixteen_cta_rows && !(rows64 > resident || large_cut)) {
    return cudaSuccess;
  }
  if (sixteen_cta_chunk && cut * kBalancedChunkSixteenCtaMinCutDen < whole &&
      (device_minor != 0 || rows64 <= resident)) {
    return cudaSuccess;
  }
  launch->chunk_elements = balanced;
  return cudaSuccess;
}

bool use_warp_kernel(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind) {
  return rows <= 128 && vocab_size <= 257 &&
         (parameter_kind == ParameterKind::kScalar || parameter_kind == ParameterKind::kPerRow);
}

bool use_rowwise_kernel(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind) {
  const bool small_low_row = rows <= 32 && vocab_size <= 16384;
  const bool dense_aligned_mid_row = rows > 128 && rows <= 384 && vocab_size >= 24576 &&
                                     vocab_size <= 256000 && vocab_size % 4 == 0 &&
                                     parameter_kind == ParameterKind::kNone;
  const bool dense_aligned_high_row_narrow = rows > 384 && rows <= 1024 && vocab_size >= 24576 &&
                                             vocab_size <= 32000 && vocab_size % 4 == 0;
  const bool dense_aligned_high_row_wide = rows > 512 && rows <= 1024 && vocab_size > 32000 &&
                                           vocab_size <= 64000 && vocab_size % 4 == 0 &&
                                           parameter_kind == ParameterKind::kNone;
  const bool measured_large_odd = rows > 128 && rows <= 512 && vocab_size >= 24576 &&
                                  vocab_size <= 131072 && vocab_size % 4 != 0;
  return small_low_row || dense_aligned_mid_row || dense_aligned_high_row_narrow ||
         dense_aligned_high_row_wide || measured_large_odd;
}

SoftmaxRoute select_softmax_route(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind,
                                  bool aligned_buffers) {
  if (rows == 0 || vocab_size == 0 || !aligned_buffers ||
      static_cast<uint64_t>(rows) * vocab_size >
          static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return SoftmaxRoute::kFallback;
  }
  if (use_warp_kernel(rows, vocab_size, parameter_kind)) {
    return SoftmaxRoute::kWarp;
  }
  if (use_cached_cluster_kernel(vocab_size)) {
    return SoftmaxRoute::kCachedCluster;
  }
  if (use_rowwise_kernel(rows, vocab_size, parameter_kind)) {
    return SoftmaxRoute::kRowwise;
  }
  return SoftmaxRoute::kBootstrap;
}

cudaError_t launch_noncooperative_kernel(const void* kernel, dim3 grid, dim3 block, void** args,
                                         size_t dynamic_smem_bytes, bool enable_pdl,
                                         cudaStream_t stream) {
  if (!enable_pdl) {
    return cudaLaunchKernel(kernel, grid, block, args, dynamic_smem_bytes, stream);
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t launch_cached_cluster_kernel(const void* kernel, int cluster_ctas, dim3 grid,
                                         dim3 block, void** args, size_t dynamic_smem_bytes,
                                         bool enable_pdl, cudaStream_t stream) {
  if (cluster_ctas > kCachedPortableClusterSize || dynamic_smem_bytes > 48 * 1024) {
    cudaError_t status = configure_cached_kernel(kernel, cluster_ctas, dynamic_smem_bytes);
    if (status != cudaSuccess) {
      return status;
    }
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attributes[3]{};
  unsigned int num_attrs = 0;
  if (cluster_ctas > 1) {
    // The kernels carry compile-time __cluster_dims__; the explicit attribute
    // must match it. Spread placement mirrors the audited Cake launch.
    attributes[num_attrs].id = cudaLaunchAttributeClusterDimension;
    attributes[num_attrs].val.clusterDim.x = static_cast<unsigned int>(cluster_ctas);
    attributes[num_attrs].val.clusterDim.y = 1;
    attributes[num_attrs].val.clusterDim.z = 1;
    ++num_attrs;
    attributes[num_attrs].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attributes[num_attrs].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
    ++num_attrs;
  }
  if (enable_pdl) {
    attributes[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  config.attrs = attributes;
  config.numAttrs = num_attrs;
  return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t launch_cooperative_kernel(const void* kernel, dim3 grid, dim3 block, void** args,
                                      size_t dynamic_smem_bytes, bool enable_pdl,
                                      cudaStream_t stream) {
  if (!enable_pdl) {
    return cudaLaunchCooperativeKernel(kernel, grid, block, args, dynamic_smem_bytes, stream);
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeCooperative;
  attributes[0].val.cooperative = 1;
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attributes;
  config.numAttrs = 2;
  return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t launch_blackwell_softmax(float* logits, float* output, float* temperature_arr,
                                     float temperature_val, ParameterKind parameter_kind,
                                     uint32_t rows, uint32_t vocab_size, void* workspace,
                                     size_t workspace_bytes, bool enable_pdl, bool is_sm103,
                                     SoftmaxRoute* selected_route, cudaStream_t stream) {
  const SoftmaxRoute route =
      select_softmax_route(rows, vocab_size, parameter_kind, buffers_aligned(logits, output));
  *selected_route = route;
  if (route == SoftmaxRoute::kFallback) {
    return cudaErrorNotSupported;
  }

  float* parameter = temperature_arr != nullptr ? temperature_arr : logits;
  int rows_i = static_cast<int>(rows);
  int vocab_size_i = static_cast<int>(vocab_size);
  // Keep the existing sm_100a launch path unchanged. The high bit is an
  // sm_103a integration-only PDL flag; adapted kernels strip it before
  // evaluating the original 0/1/2 parameter-kind ABI. The row-caching cluster
  // kernels instead have _pdl twins and take the plain parameter kind.
  const bool launch_with_pdl = is_sm103 && enable_pdl;
  int parameter_kind_i = static_cast<int>(parameter_kind) | (launch_with_pdl ? 4 : 0);

  if (route == SoftmaxRoute::kWarp) {
    void* args[] = {&logits,       &parameter,        &output,         &rows_i,
                    &vocab_size_i, &parameter_kind_i, &temperature_val};
    return launch_noncooperative_kernel(
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_followup_warp),
        dim3(ceil_div(rows, static_cast<uint32_t>(kWarpRowsPerCta))), dim3(kWarpThreads), args,
        kWarpDynamicSmemBytes, launch_with_pdl, stream);
  }

  if (route == SoftmaxRoute::kCachedCluster) {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    CachedLaunch launch{};
    if (status == cudaSuccess) {
      status =
          select_cached_launch(rows, vocab_size, parameter_kind, launch_with_pdl, device, &launch);
    }
    if (status != cudaSuccess) {
      return status;
    }
    *selected_route = launch.route;
    const uint64_t grid = launch.clusters * launch.variant.cluster_ctas;
    if (grid > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return cudaErrorNotSupported;
    }
    int chunk_elements_i = static_cast<int>(launch.chunk_elements);
    // The row-caching cluster kernels take PDL as their _pdl twins, the
    // persistent kernels as the flag.
    int cached_parameter_kind_i = launch.route == SoftmaxRoute::kCachedClusterPersistent
                                      ? parameter_kind_i
                                      : static_cast<int>(parameter_kind);
    void* args[] = {&logits,
                    &parameter,
                    &output,
                    &rows_i,
                    &vocab_size_i,
                    &chunk_elements_i,
                    &cached_parameter_kind_i,
                    &temperature_val};
    return launch_cached_cluster_kernel(launch.kernel, launch.variant.cluster_ctas,
                                        dim3(static_cast<unsigned int>(grid)), dim3(kCachedThreads),
                                        args, launch.dynamic_smem_bytes, launch_with_pdl, stream);
  }

  if (route == SoftmaxRoute::kRowwise) {
    void* args[] = {&logits,       &parameter,        &output,         &rows_i,
                    &vocab_size_i, &parameter_kind_i, &temperature_val};
    return launch_noncooperative_kernel(
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_followup_rowwise),
        dim3(rows), dim3(kRowwiseThreads), args, kRowwiseDynamicSmemBytes, launch_with_pdl, stream);
  }

  int active_blocks_per_sm = 0;
  cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel_flashinfer_blackwell_softmax_bootstrap_seed, kBootstrapThreads,
      kBootstrapDynamicSmemBytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (active_blocks_per_sm <= 0) {
    return cudaErrorNotSupported;
  }

  int sm_count = 0;
  int device = 0;
  if ((status = cudaGetDevice(&device)) != cudaSuccess ||
      (status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device)) !=
          cudaSuccess) {
    return status;
  }

  const uint64_t tiles_per_row =
      ceil_div(static_cast<uint64_t>(vocab_size), static_cast<uint64_t>(kBootstrapThreads));
  const uint64_t cooperative_capacity =
      static_cast<uint64_t>(active_blocks_per_sm) * static_cast<uint64_t>(sm_count);
  const uint64_t provisional_grid = std::max<uint64_t>(
      1, std::min<uint64_t>(static_cast<uint64_t>(rows) * tiles_per_row, cooperative_capacity));
  const uint64_t split_capacity =
      provisional_grid >= rows ? provisional_grid / static_cast<uint64_t>(rows) : 1;
  const int splits = static_cast<int>(
      std::max<uint64_t>(1, std::min<uint64_t>({tiles_per_row, kMaxSplits, split_capacity})));
  const uint64_t grid = std::max<uint64_t>(
      1, std::min<uint64_t>(provisional_grid, static_cast<uint64_t>(rows) * splits));
  const size_t partial_count = static_cast<size_t>(rows) * static_cast<size_t>(splits);
  const size_t required_workspace_bytes = partial_count * 2 * sizeof(float);
  if (required_workspace_bytes > workspace_bytes) {
    return cudaErrorNotSupported;
  }

  auto* partial_max = static_cast<float*>(workspace);
  auto* partial_sum = partial_max + partial_count;
  int splits_i = splits;
  void* args[] = {&logits, &parameter,    &output,   &partial_max,      &partial_sum,
                  &rows_i, &vocab_size_i, &splits_i, &parameter_kind_i, &temperature_val};
  return launch_cooperative_kernel(
      reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_bootstrap_seed),
      dim3(static_cast<uint32_t>(grid)), dim3(kBootstrapThreads), args, kBootstrapDynamicSmemBytes,
      launch_with_pdl, stream);
}

ParameterKind validate_logits_and_temperature(TensorView logits,
                                              Optional<TensorView> maybe_temperature_arr,
                                              bool temperature_is_none) {
  CHECK_INPUT(logits);
  CHECK_DIM(2, logits);
  CHECK_INPUT_TYPE(logits, dl_float32);
  TVM_FFI_ICHECK_GT(logits.size(0), 0) << "logits must contain at least one row";
  TVM_FFI_ICHECK_GT(logits.size(1), 0) << "logits must contain at least one vocabulary entry";
  TVM_FFI_ICHECK_LE(static_cast<uint64_t>(logits.size(0)),
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
  TVM_FFI_ICHECK_LE(static_cast<uint64_t>(logits.size(1)),
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

  if (maybe_temperature_arr.has_value()) {
    const TensorView temperature_arr = maybe_temperature_arr.value();
    CHECK_INPUT(temperature_arr);
    CHECK_DIM(1, temperature_arr);
    CHECK_INPUT_TYPE(temperature_arr, dl_float32);
    CHECK_DEVICE(temperature_arr, logits);
    TVM_FFI_ICHECK_EQ(temperature_arr.size(0), logits.size(0))
        << "temperature length must equal logits.size(0)";
    TVM_FFI_ICHECK(!temperature_is_none)
        << "temperature_is_none must be false when a temperature tensor is provided";
    return ParameterKind::kPerRow;
  }
  return temperature_is_none ? ParameterKind::kNone : ParameterKind::kScalar;
}

void validate_softmax_io(TensorView workspace_buffer, TensorView logits, TensorView output) {
  CHECK_INPUT(workspace_buffer);
  CHECK_DIM(1, workspace_buffer);
  CHECK_DEVICE(workspace_buffer, logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, output);
  CHECK_INPUT_TYPE(output, dl_float32);
  CHECK_DEVICE(output, logits);
  CHECK_SHAPE(output, logits);
  TVM_FFI_ICHECK_NE(output.data_ptr(), logits.data_ptr())
      << "output must be fresh and must not alias logits";
}

// The route these logits take; a cached-cluster route also fills *cached_launch
// when it is not null.
SoftmaxRoute query_softmax_route(TensorView logits, Optional<TensorView> maybe_temperature_arr,
                                 double temperature_val, bool enable_pdl, bool temperature_is_none,
                                 CachedLaunch* cached_launch = nullptr) {
  const ParameterKind parameter_kind =
      validate_logits_and_temperature(logits, maybe_temperature_arr, temperature_is_none);
  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  int device_major = 0;
  int device_minor = 0;
  cudaError_t status = cudaDeviceGetAttribute(&device_major, cudaDevAttrComputeCapabilityMajor,
                                              logits.device().device_id);
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor,
                                    logits.device().device_id);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax route query failed with error code " << cudaGetErrorString(status);
  if (device_major != 10 || (device_minor != 0 && device_minor != 3)) {
    return SoftmaxRoute::kFallback;
  }
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const SoftmaxRoute route = select_softmax_route(
      rows, vocab_size, parameter_kind, buffers_aligned(logits.data_ptr(), /*output=*/nullptr));
  if (route != SoftmaxRoute::kCachedCluster) {
    return route;
  }
  CachedLaunch launch{};
  // Mirrors launch_blackwell_softmax: only sm_103a launches with PDL.
  const bool launch_with_pdl = enable_pdl && device_minor == 3;
  status = select_cached_launch(rows, vocab_size, parameter_kind, launch_with_pdl,
                                logits.device().device_id, &launch);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax route query failed with error code " << cudaGetErrorString(status);
  if (cached_launch != nullptr) {
    *cached_launch = launch;
  }
  return launch.route;
}

// The cached-cluster kernel these logits would launch, for tests: {cluster
// CTAs, max tiles, 1 if its symbol is a _pdl twin, its resident clusters on
// this device, the resident clusters of the 4-tile kernel twice its size (0
// past 16 CTAs), the elements of the row per CTA}; empty for the other routes.
Array<int64_t> query_cached_kernel(TensorView logits, Optional<TensorView> maybe_temperature_arr,
                                   double temperature_val, bool enable_pdl,
                                   bool temperature_is_none) {
  CachedLaunch launch{};
  query_softmax_route(logits, maybe_temperature_arr, temperature_val, enable_pdl,
                      temperature_is_none, &launch);
  if (launch.kernel == nullptr) {
    return Array<int64_t>();
  }
  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  const char* name = nullptr;
  int resident = 0;
  int next_resident = 0;
  cudaError_t status = cudaFuncGetName(&name, launch.kernel);
  if (status == cudaSuccess) {
    status = query_resident_clusters(nullptr, launch.kernel, launch.variant.cluster_ctas,
                                     launch.dynamic_smem_bytes, &resident);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax kernel query failed with error code " << cudaGetErrorString(status);
  const size_t length = std::strlen(name);
  const bool pdl_twin = length > 4 && std::strcmp(name + length - 4, "_pdl") == 0;
  const int next_ctas = launch.variant.cluster_ctas * 2;
  if (next_ctas <= kCachedMaxClusterSize) {
    status =
        query_resident_clusters(nullptr, cached_cluster_kernel({next_ctas, 4}, pdl_twin), next_ctas,
                                cached_dynamic_smem_bytes(next_ctas), &next_resident);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax kernel query failed with error code " << cudaGetErrorString(status);
  return Array<int64_t>(std::vector<int64_t>{launch.variant.cluster_ctas, launch.variant.max_tiles,
                                             pdl_twin ? 1 : 0, resident, next_resident,
                                             static_cast<int64_t>(launch.chunk_elements)});
}

void blackwell_softmax_impl(TensorView workspace_buffer, TensorView logits, TensorView output,
                            Optional<TensorView> maybe_temperature_arr, double temperature_val,
                            bool enable_pdl, bool temperature_is_none, bool is_sm103) {
  const ParameterKind parameter_kind =
      validate_logits_and_temperature(logits, maybe_temperature_arr, temperature_is_none);
  validate_softmax_io(workspace_buffer, logits, output);

  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const bool has_temperature_arr = maybe_temperature_arr.has_value();

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());
  auto* logits_ptr = static_cast<float*>(logits.data_ptr());
  auto* output_ptr = static_cast<float*>(output.data_ptr());
  auto* temperature_ptr =
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr.value().data_ptr()) : nullptr;
  const size_t workspace_bytes = get_element_size(workspace_buffer) * workspace_buffer.size(0);

  SoftmaxRoute selected_route = SoftmaxRoute::kFallback;
  cudaError_t status = launch_blackwell_softmax(
      logits_ptr, output_ptr, temperature_ptr, static_cast<float>(temperature_val), parameter_kind,
      rows, vocab_size, workspace_buffer.data_ptr(), workspace_bytes, enable_pdl, is_sm103,
      &selected_route, stream);
  if (status == cudaErrorNotSupported) {
    status = sampling::OnlineSoftmax<float>(logits_ptr, output_ptr, rows, vocab_size,
                                            temperature_ptr, static_cast<float>(temperature_val),
                                            workspace_buffer.data_ptr(), workspace_bytes,
                                            enable_pdl, stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax route " << static_cast<int64_t>(selected_route)
      << " failed with error code " << cudaGetErrorString(status);
}

}  // namespace

void blackwell_softmax(TensorView workspace_buffer, TensorView logits, TensorView output,
                       Optional<TensorView> maybe_temperature_arr, double temperature_val,
                       bool enable_pdl, bool temperature_is_none, bool is_sm103) {
  blackwell_softmax_impl(workspace_buffer, logits, output, maybe_temperature_arr, temperature_val,
                         enable_pdl, temperature_is_none, is_sm103);
}

int64_t blackwell_softmax_route(TensorView logits, Optional<TensorView> maybe_temperature_arr,
                                double temperature_val, bool enable_pdl, bool temperature_is_none) {
  return static_cast<int64_t>(query_softmax_route(logits, maybe_temperature_arr, temperature_val,
                                                  enable_pdl, temperature_is_none));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, blackwell_softmax);
Array<int64_t> blackwell_softmax_cached_kernel(TensorView logits,
                                               Optional<TensorView> maybe_temperature_arr,
                                               double temperature_val, bool enable_pdl,
                                               bool temperature_is_none) {
  return query_cached_kernel(logits, maybe_temperature_arr, temperature_val, enable_pdl,
                             temperature_is_none);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax_route, blackwell_softmax_route);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax_cached_kernel, blackwell_softmax_cached_kernel);
