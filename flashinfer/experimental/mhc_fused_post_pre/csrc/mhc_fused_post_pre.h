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
#pragma once

// Host-side interface of the fused mHC post -> pre (+ input RMSNorm) kernels
// (hidden size 4096, four residual streams, SM103 thread-block clusters of 16 CTAs).

#include <cuda.h>
#include <cuda_runtime.h>

namespace flashinfer {
namespace mhc_fpp {

constexpr int kHidden = 4096;                        // hidden size H
constexpr int kStreams = 4;                          // mHC residual streams (hc_mult)
constexpr int kMixWidth = 24;                        // hc_mult * (2 + hc_mult) projection rows
constexpr int kClusterSize = 16;                     // CTAs per cluster (non-portable cluster size)
constexpr int kSliceWidth = kHidden / kClusterSize;  // hidden columns owned by one CTA
constexpr int kTileTokens = 16;                      // max tokens per cluster tile

// Raw operands of one call; every tensor is dense row-major (see the binding's checks).
struct LaunchArgs {
  const void* x;                // [M, H] bf16
  const void* residual;         // [M, 4, H] bf16
  const float* post_layer_mix;  // [M, 4] fp32
  const float* comb_res_mix;    // [M, 4, 4] fp32
  const float* fn;              // [24, 4 * H] fp32
  const float* hc_scale;        // [3] fp32
  const float* hc_base;         // [24] fp32
  const void* norm_weight;      // [H] bf16
  void* residual_cur;           // [M, 4, H] bf16 (out)
  float* post_mix_cur;          // [M, 4] fp32 (out)
  float* comb_mix_cur;          // [M, 4, 4] fp32 (out)
  void* layer_input_cur;        // [M, H] bf16 (out)
  int num_tokens;
  int tile_tokens;
  int num_tiles;
};

// TMA descriptors of x, residual and residual_cur for one (pointer, num_tokens, tile) triple.
struct TensorMaps {
  CUtensorMap x;
  CUtensorMap residual;
  CUtensorMap residual_cur;
};

// Kernel variants. The regime thresholds and register-split rules below are fixed per
// variant; plan_launch() applies them exactly and reports kUnsupported for the
// (device-dependent) corner cases whose variant is not compiled.
enum class Variant : int {
  kUnsupported = -1,
  kDecodeNarrow = 0,    // one tile per cluster, <= 8 tokens per tile
  kDecodeWide = 1,      // one tile per cluster, 9..16 tokens per tile
  kStream = 2,          // several tiles per cluster, default register split
  kStreamLongRegs = 3,  // several tiles per cluster, 112-register elementwise split
  kShortStream = 4,     // 1024 <= num_tokens < 1800
  kCrossover = 5,       // 1800 <= num_tokens <= 2048
  kLongStream = 6,      // num_tokens >= 6245
};
constexpr int kNumVariants = 7;

// Co-resident cluster capacity of each kernel family on one device
// (cudaOccupancyMaxActiveClusters with 16-CTA clusters).
struct ClusterLimits {
  int decode;
  int stream;
  int short_stream;
  int crossover;
  int long_stream;
};

struct LaunchPlan {
  Variant variant;
  int tile_tokens;
  int num_tiles;
  int num_clusters;
};

inline LaunchPlan plan_launch(int num_tokens, const ClusterLimits& limits) {
  const LaunchPlan unsupported{Variant::kUnsupported, 0, 0, 0};
  if (num_tokens <= 0 || limits.decode <= 0) return unsupported;
  const int m = num_tokens;
  const int tiles16 = (m + kTileTokens - 1) / kTileTokens;
  const bool long_regime = m >= 6245;
  const bool crossover_regime = m >= 1800 && m <= 2048;
  const bool short_regime = m >= 1024 && m < 1800;
  if (!long_regime && !crossover_regime && !short_regime) {
    if (tiles16 <= limits.decode) {
      // Decode: exactly one cluster per tile; spread the tokens over every resident cluster.
      int tm = tiles16 >= limits.decode ? kTileTokens : (m + limits.decode - 1) / limits.decode;
      if (tm < 1) tm = 1;
      if (tm > kTileTokens) tm = kTileTokens;
      const int tiles = (m + tm - 1) / tm;
      const int clusters = tiles < limits.decode ? tiles : limits.decode;
      return {tm <= 8 ? Variant::kDecodeNarrow : Variant::kDecodeWide, tm, tiles, clusters};
    }
    if (limits.stream <= 0) return unsupported;
    const int clusters = tiles16 < limits.stream ? tiles16 : limits.stream;
    const bool long_regs = tiles16 >= 20 * clusters;
    return {long_regs ? Variant::kStreamLongRegs : Variant::kStream, kTileTokens, tiles16,
            clusters};
  }
  // These regimes always use their stream kernel (their decode kernel needs
  // ceil(num_tokens / 16) <= decode capacity, i.e. 64+ co-resident clusters); only the one
  // register split they reach on real devices is compiled.
  if (tiles16 <= limits.decode) return unsupported;
  const int capacity = long_regime ? limits.long_stream
                                   : (crossover_regime ? limits.crossover : limits.short_stream);
  if (capacity <= 0) return unsupported;
  const int clusters = tiles16 < capacity ? tiles16 : capacity;
  const bool long_regs = tiles16 >= (crossover_regime ? 32 : 20) * clusters;
  if (long_regs != long_regime) return unsupported;
  return {long_regime ? Variant::kLongStream
                      : (crossover_regime ? Variant::kCrossover : Variant::kShortStream),
          kTileTokens, tiles16, clusters};
}

// Per-regime entry points (one translation unit each). prepare_* sets the kernels'
// dynamic-shared-memory / non-portable-cluster attributes on the CURRENT device and returns
// the co-resident cluster capacity; it must run once per device before the first launch
// (outside CUDA graph capture). launch_* enqueue one cluster launch on `stream`.
cudaError_t prepare_decode_stream(int* decode_clusters, int* stream_clusters);
cudaError_t launch_decode(const LaunchArgs& args, int num_clusters, const TensorMaps& maps,
                          cudaStream_t stream);
cudaError_t launch_stream(const LaunchArgs& args, int num_clusters, bool long_regs,
                          const TensorMaps& maps, cudaStream_t stream);
cudaError_t prepare_short_stream(int* stream_clusters);
cudaError_t launch_short_stream(const LaunchArgs& args, int num_clusters, const TensorMaps& maps,
                                cudaStream_t stream);
cudaError_t prepare_crossover(int* stream_clusters);
cudaError_t launch_crossover(const LaunchArgs& args, int num_clusters, const TensorMaps& maps,
                             cudaStream_t stream);
cudaError_t prepare_long_stream(int* stream_clusters);
cudaError_t launch_long_stream(const LaunchArgs& args, int num_clusters, const TensorMaps& maps,
                               cudaStream_t stream);

}  // namespace mhc_fpp
}  // namespace flashinfer
