/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace flashinfer::warp_decode {

#ifndef FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR
#error "FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR must select exact SM100a (0) or SM103a (3)"
#elif FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR != 0 && FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR != 3
#error "FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR must be 0 or 3"
#endif

inline constexpr uint32_t kGeneratedContractVersion = 1;
inline constexpr int32_t kMaximumTokens = 32;
inline constexpr int32_t kPackedWorkfeedCtas = 152;

enum class Target : uint8_t {
  kSm100a = 0,
  kSm103a = 3,
};

inline constexpr Target kTarget =
    FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR == 0 ? Target::kSm100a : Target::kSm103a;

enum class Geometry : uint8_t {
  kUnsupported = 0,
  kH2048I512E512K10,
  kH2048I1536E60K4,
  kH6144I1536E192K4,
  kH4096I2048E256K6,
  kH2560I768E384K4,
  kH2048I768E128K8,
  kH4096I1536E128K8,
  kH2048I512E256K8,
  kH4096I1024E512K10,
  kH3072I1536E256K8,
  kH6144I3072E128K4,
  kH3584I3072E896K16,
  // Sharded per-partition slices (tensor-parallel expert shards).
  kH4096I512E512K10,
  kH4096I256E512K10,
  kH3072I768E256K8,
  kH3072I384E256K8,
};

enum class Activation : uint8_t {
  kSwiGLU = 0,
  kSiLU,
  kSwiGLUParameterized,
  kSiTU,
};

enum class RouteLayout : uint8_t {
  kDirect = 0,
  kGpuPacked,
};

enum class RoutePacker : uint8_t {
  kNone = 0,
  kE64Scan1,
  kE64Scan2,
  kGeneral,
  kExpertOrder,
  kSortedShort,
  kCountRank,
  // The packed route tables are derived inside the persistent FC1 prologue;
  // the graph has no route_pack launch and FC1/FC2 re-arm their own workfeeds.
  kFusedFc1,
};

enum class Fc1Schedule : uint8_t {
  kStatic = 0,
  kPersistent,
  kPersistentDeviceWorkfeed,
  kPersistentPaddedScaleDeviceWorkfeed,
  kPersistentExpertOrder,
  kPersistentMetadataPublication,
  kPersistentEarlySfbDeviceWorkfeed,
};

enum class Fc2Schedule : uint8_t {
  kRouteParallelK256 = 0,
  kRouteParallelK512DeviceWorkfeed,
  kRouteParallelK768K96,
  kRouteParallelK768K96PaddedScale,
  kRouteParallelK512,
  kImmutableWeightPrefill,
  kRouteParallelK512Stage5DeviceWorkfeed,
  kRouteParallelK512MmaU2DeviceWorkfeed,
};

struct Shape {
  int32_t num_tokens;
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t num_experts;
  int32_t local_num_experts;
  int32_t top_k;
};

struct Schedule {
  bool supported;
  Geometry geometry;
  RouteLayout route_layout;
  RoutePacker route_packer;
  Fc1Schedule fc1;
  Fc2Schedule fc2;
  int32_t finalize_threads;
  int32_t finalize_unroll;
  int32_t workfeed_ctas;
};

constexpr bool IsGeometry(const Shape& shape, int32_t hidden_size, int32_t intermediate_size,
                          int32_t num_experts, int32_t top_k) {
  return shape.hidden_size == hidden_size && shape.intermediate_size == intermediate_size &&
         shape.num_experts == num_experts && shape.local_num_experts == num_experts &&
         shape.top_k == top_k;
}

// True when `tokens` equals one of `token_counts`.
template <typename... TokenCounts>
constexpr bool TokensIn(int32_t tokens, TokenCounts... token_counts) {
  return ((tokens == token_counts) || ...);
}

// The fused route-pack K256 FC2 rows that run the linear-grid, unthrottled workfeed program:
// one 1-D grid of min(grid_m * route tiles, workfeed CTAs) CTAs, each starting at the tile of
// its block index, instead of the grid_m x (workfeed CTAs / grid_m) pool, with a five-deep work
// ring and every route tile's metadata read once into a 512-row shared-memory table. Both
// targets run it on the H4096 E512 slices at every token count and on the H3072 I384 slice at T8
// and T10..T32. The other rows of that FC2 keep the pool program.
constexpr bool LinearK256Fc2(Target target, const Shape& shape) {
  (void)target;
  const int32_t t = shape.num_tokens;
  if (IsGeometry(shape, 4096, 256, 512, 10) || IsGeometry(shape, 4096, 512, 512, 10)) {
    return t >= 1 && t <= 32;
  }
  return IsGeometry(shape, 3072, 384, 256, 8) && (t == 8 || (t >= 10 && t <= 32));
}
// The linear-grid program's shared-memory table holds the metadata of this many route tiles.
inline constexpr int32_t kLinearK256Fc2MaxRouteTiles = 512;

constexpr Activation ActivationForGeometry(Geometry geometry) {
  if (geometry == Geometry::kH6144I1536E192K4) return Activation::kSiLU;
  if (geometry == Geometry::kH6144I3072E128K4 ||
      geometry == Geometry::kH4096I2048E256K6) return Activation::kSwiGLUParameterized;
  if (geometry == Geometry::kH3584I3072E896K16) return Activation::kSiTU;
  return Activation::kSwiGLU;
}

constexpr int32_t Gemm1WeightRows(const Shape& shape, const Schedule& schedule) {
  return ActivationForGeometry(schedule.geometry) == Activation::kSiLU
             ? shape.intermediate_size
             : 2 * shape.intermediate_size;
}

// H2048 I768 rows and sharded slices whose route packing runs inside the FC1 prologue
// (RoutePacker::kFusedFc1): H2048 I768 sm_103a T8..T32 and sm_100a T16..T32 (sm_103a T1..T7 and
// sm_100a T1..T15 stay on their direct routes). SelectSm103aSchedule picks the same packer
// directly for its fused route-pack K512 MMA-U2 rows.
constexpr bool IsFusedRoutePackRow(Target target, const Shape& shape) {
  if (shape.num_tokens < 1 || shape.num_tokens > 32) return false;
  // Sharded TP slices (Qwen3.5-397B TP2/TP4, MiniMax-M2 TP2/TP4): both targets fuse every
  // token count.
  if (IsGeometry(shape, 4096, 512, 512, 10) || IsGeometry(shape, 4096, 256, 512, 10) ||
      IsGeometry(shape, 3072, 768, 256, 8) || IsGeometry(shape, 3072, 384, 256, 8)) {
    return true;
  }
  if (!IsGeometry(shape, 2048, 768, 128, 8)) return false;
  return (target == Target::kSm103a && shape.num_tokens >= 8) ||
         (target == Target::kSm100a && shape.num_tokens >= 16);
}

constexpr RoutePacker PackedRoutePacker(Target target, const Shape& shape) {
  return IsFusedRoutePackRow(target, shape) ? RoutePacker::kFusedFc1 : RoutePacker::kGeneral;
}

constexpr Schedule UnsupportedSchedule() {
  return {false,
          Geometry::kUnsupported,
          RouteLayout::kDirect,
          RoutePacker::kNone,
          Fc1Schedule::kStatic,
          Fc2Schedule::kRouteParallelK256,
          0,
          0,
          0};
}

constexpr Schedule SelectAdditionalDirectSchedule(const Shape& shape) {
  Geometry geometry = Geometry::kUnsupported;
  if (IsGeometry(shape, 2048, 768, 128, 8)) {
    geometry = Geometry::kH2048I768E128K8;
  } else if (IsGeometry(shape, 4096, 1536, 128, 8)) {
    geometry = Geometry::kH4096I1536E128K8;
  } else if (IsGeometry(shape, 2048, 512, 256, 8)) {
    geometry = Geometry::kH2048I512E256K8;
  } else if (IsGeometry(shape, 4096, 1024, 512, 10)) {
    geometry = Geometry::kH4096I1024E512K10;
  } else if (IsGeometry(shape, 3072, 1536, 256, 8)) {
    geometry = Geometry::kH3072I1536E256K8;
  } else if (IsGeometry(shape, 6144, 3072, 128, 4)) {
    geometry = Geometry::kH6144I3072E128K4;
  } else if (IsGeometry(shape, 3584, 3072, 896, 16)) {
    geometry = Geometry::kH3584I3072E896K16;
  } else if (IsGeometry(shape, 4096, 512, 512, 10)) {
    geometry = Geometry::kH4096I512E512K10;
  } else if (IsGeometry(shape, 4096, 256, 512, 10)) {
    geometry = Geometry::kH4096I256E512K10;
  } else if (IsGeometry(shape, 3072, 768, 256, 8)) {
    geometry = Geometry::kH3072I768E256K8;
  } else if (IsGeometry(shape, 3072, 384, 256, 8)) {
    geometry = Geometry::kH3072I384E256K8;
  } else {
    return UnsupportedSchedule();
  }
  return {true,
          geometry,
          RouteLayout::kDirect,
          RoutePacker::kNone,
          geometry == Geometry::kH3584I3072E896K16 ? Fc1Schedule::kStatic
                                                   : Fc1Schedule::kPersistent,
          Fc2Schedule::kRouteParallelK256,
          128,
          4,
          0};
}

// These selectors are the public calibration boundary. The generated manifest
// supplies implementations for these choices but must not independently
// reinterpret a shape, architecture, activation, or token boundary.
constexpr Schedule SelectSm103aSchedule(const Shape& shape) {
  if (shape.num_tokens < 1 || shape.num_tokens > kMaximumTokens) {
    return UnsupportedSchedule();
  }

  // The packed rows that run the fused route-pack FC1 (no route_pack launch) and the K512 MMA-U2
  // device-workfeed FC2. Each geometry's block precedes its other blocks, which keep the
  // geometry's remaining rows. Each row keeps the workfeed it was measured with: 144 CTAs, or 0
  // (the device's SM count).
  if (IsGeometry(shape, 4096, 1536, 128, 8) &&
      (shape.num_tokens >= 6 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH4096I1536E128K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            TokensIn(shape.num_tokens, 6, 7, 12, 13, 15, 16, 17, 19, 20, 21, 23) ? 0 : 144};
  }

  if (IsGeometry(shape, 3072, 1536, 256, 8) &&
      (shape.num_tokens >= 15 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH3072I1536E256K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            shape.num_tokens == 20 ? 0 : 144};
  }

  if (IsGeometry(shape, 4096, 1024, 512, 10) &&
      (shape.num_tokens >= 5 && shape.num_tokens <= 25)) {
    return {true,
            Geometry::kH4096I1024E512K10,
            RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            TokensIn(shape.num_tokens, 6, 7, 8, 15) ? 0 : 144};
  }

  if (IsGeometry(shape, 4096, 1024, 512, 10) && shape.num_tokens == 1) {
    return {true,
            Geometry::kH4096I1024E512K10,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  if (IsGeometry(shape, 4096, 1024, 512, 10) &&
      (shape.num_tokens >= 8 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH4096I1024E512K10,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 512, 256, 8) &&
      TokensIn(shape.num_tokens, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32)) {
    return {true,
            Geometry::kH2048I512E256K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            TokensIn(shape.num_tokens, 20, 22, 24) ? 0 : 144};
  }

  if (IsGeometry(shape, 2048, 512, 256, 8) &&
      (shape.num_tokens >= 16 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH2048I512E256K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512Stage5DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) && shape.num_tokens == 8) {
    // Round 12: the last direct Qwen3-30B row joins the fused route-pack K256
    // metadata-prefetch workfeed route of T10..T19 (fixed top-8 finalizer).
    return {true, Geometry::kH2048I768E128K8, RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm103a, shape), Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) &&
      (shape.num_tokens == 9 || shape.num_tokens == 10)) {
    return {true, Geometry::kH2048I768E128K8, RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm103a, shape), Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed, 128, 4, 144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) &&
      shape.num_tokens >= 10 && shape.num_tokens <= 19) {
    return {true,
            Geometry::kH2048I768E128K8,
            RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm103a, shape),
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) && shape.num_tokens >= 20 && shape.num_tokens <= 32) {
    return {true,
            Geometry::kH2048I768E128K8,
            RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm103a, shape),
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            144};
  }

  // H6144 I3072 T1..T32: the fused route-pack FC1 with the parameterized-SwiGLU epilogue (the
  // generated manifest selects those programs for this geometry only) and the K512 MMA-U2
  // device-workfeed FC2.
  if (IsGeometry(shape, 6144, 3072, 128, 4) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH6144I3072E128K4,
            RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2560, 768, 384, 4)) {
    return {true,
            Geometry::kH2560I768E384K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  if (IsGeometry(shape, 2048, 512, 512, 10)) {
    return {true,
            Geometry::kH2048I512E512K10,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            shape.num_tokens <= 3 ? 128 : 32,
            4,
            0};
  }

  if (IsGeometry(shape, 2048, 1536, 60, 4) && shape.num_tokens >= 12) {
    return {true,
            Geometry::kH2048I1536E60K4,
            RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            TokensIn(shape.num_tokens, 12, 16, 19, 23, 25, 26, 32) ? 0 : 144};
  }

  if (IsGeometry(shape, 2048, 1536, 60, 4)) {
    if (shape.num_tokens < 5) {
      return {true,
              Geometry::kH2048I1536E60K4,
              RouteLayout::kDirect,
              RoutePacker::kNone,
              Fc1Schedule::kPersistent,
              Fc2Schedule::kRouteParallelK768K96,
              128,
              4,
              0};
    }
    if (shape.num_tokens < 11) {
      return {true,
              Geometry::kH2048I1536E60K4,
              RouteLayout::kDirect,
              RoutePacker::kNone,
              Fc1Schedule::kPersistent,
              Fc2Schedule::kRouteParallelK768K96PaddedScale,
              128,
              4,
              0};
    }
    if (shape.num_tokens == 11) {
      return {true,
              Geometry::kH2048I1536E60K4,
              RouteLayout::kGpuPacked,
              RoutePacker::kE64Scan1,
              Fc1Schedule::kPersistentPaddedScaleDeviceWorkfeed,
              Fc2Schedule::kRouteParallelK768K96PaddedScale,
              128,
              4,
              kPackedWorkfeedCtas};
    }
  }

  if (IsGeometry(shape, 6144, 1536, 192, 4)) {
    return {true,
            Geometry::kH6144I1536E192K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  // Sharded TP slices: fused route packing + metadata-prefetch K256 device-workfeed FC2 on every
  // token count (calibrated against the official trtllm-gen peer on GB300).
  if (IsGeometry(shape, 4096, 512, 512, 10) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH4096I512E512K10, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }
  if (IsGeometry(shape, 4096, 256, 512, 10) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH4096I256E512K10, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }
  if (IsGeometry(shape, 3072, 768, 256, 8) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH3072I768E256K8, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }
  if (IsGeometry(shape, 3072, 384, 256, 8) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH3072I384E256K8, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }

  return SelectAdditionalDirectSchedule(shape);
}

constexpr Schedule SelectSm100aSchedule(const Shape& shape) {
  if (shape.num_tokens < 1 || shape.num_tokens > kMaximumTokens) {
    return UnsupportedSchedule();
  }

  if (IsGeometry(shape, 4096, 2048, 256, 6)) {
    // Fixed top-k6 finalizers do not use the generic unroll/workfeed fields.
    if (shape.num_tokens <= 8) {
      const bool ordered = shape.num_tokens >= 8;
      return {true,
              Geometry::kH4096I2048E256K6,
              RouteLayout::kDirect,
              ordered ? RoutePacker::kExpertOrder : RoutePacker::kNone,
              ordered ? Fc1Schedule::kPersistentExpertOrder : Fc1Schedule::kPersistent,
              Fc2Schedule::kRouteParallelK512,
              128,
              0,
              0};
    }
    const RoutePacker planner = shape.num_tokens <= 9 ? RoutePacker::kCountRank
                                : shape.num_tokens <= 10 ? RoutePacker::kSortedShort
                                : (shape.num_tokens == 11 || shape.num_tokens == 23)
                                    ? RoutePacker::kCountRank
                                    : RoutePacker::kGeneral;
    return {true,
            Geometry::kH4096I2048E256K6,
            RouteLayout::kGpuPacked,
            planner,
            Fc1Schedule::kPersistentMetadataPublication,
            Fc2Schedule::kImmutableWeightPrefill,
            128,
            0,
            0};
  }

  // H6144 I3072 T4: the fused route-pack FC1 with the parameterized-SwiGLU epilogue (the
  // generated manifest selects that program for this geometry only) and the K512 MMA-U2
  // device-workfeed FC2, as on sm_103a.
  if (IsGeometry(shape, 6144, 3072, 128, 4) && shape.num_tokens == 4) {
    return {true, Geometry::kH6144I3072E128K4, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed, 128, 4, 144};
  }

  if (IsGeometry(shape, 6144, 3072, 128, 4) &&
      (shape.num_tokens >= 4 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH6144I3072E128K4, RouteLayout::kGpuPacked,
            RoutePacker::kGeneral, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed, 128, 4, 144};
  }

  if (IsGeometry(shape, 4096, 1024, 512, 10) && shape.num_tokens == 1) {
    return {true,
            Geometry::kH4096I1024E512K10,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  if (IsGeometry(shape, 4096, 1024, 512, 10) &&
      (shape.num_tokens >= 8 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH4096I1024E512K10,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
            128,
            4,
            0};  // Resolve the source's unspecified workfeed from the device.
  }

  if (IsGeometry(shape, 3072, 1536, 256, 8) &&
      (shape.num_tokens >= 16 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH3072I1536E256K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 4096, 1536, 128, 8) &&
      (shape.num_tokens >= 12 && shape.num_tokens <= 32)) {
    return {true,
            Geometry::kH4096I1536E128K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 512, 256, 8) &&
      shape.num_tokens >= 16 && shape.num_tokens <= 32) {
    return {true,
            Geometry::kH2048I512E256K8,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) &&
      (shape.num_tokens == 18 || shape.num_tokens == 19)) {
    return {true,
            Geometry::kH2048I768E128K8,
            RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm100a, shape),
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) &&
      (shape.num_tokens == 16 || shape.num_tokens == 17 ||
       (shape.num_tokens >= 20 && shape.num_tokens <= 31))) {
    return {true,
            Geometry::kH2048I768E128K8,
            RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm100a, shape),
            Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2048, 768, 128, 8) && shape.num_tokens == 32) {
    return {true,
            Geometry::kH2048I768E128K8,
            RouteLayout::kGpuPacked,
            PackedRoutePacker(Target::kSm100a, shape),
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed,
            128,
            4,
            144};
  }

  if (IsGeometry(shape, 2560, 768, 384, 4)) {
    return {true,
            Geometry::kH2560I768E384K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  if (IsGeometry(shape, 2048, 512, 512, 10)) {
    return {true,
            Geometry::kH2048I512E512K10,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            shape.num_tokens <= 3 ? 128 : 32,
            4,
            0};
  }

  if (IsGeometry(shape, 2048, 1536, 60, 4)) {
    // T12..T14 join the packed general route of T15..T32.
    if (shape.num_tokens >= 12) {
      return {true,
              Geometry::kH2048I1536E60K4,
              RouteLayout::kGpuPacked,
              RoutePacker::kGeneral,
              Fc1Schedule::kPersistentDeviceWorkfeed,
              Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
              128,
              4,
              144};
    }
    return {true,
            Geometry::kH2048I1536E60K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  if (IsGeometry(shape, 6144, 1536, 192, 4)) {
    return {true,
            Geometry::kH6144I1536E192K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  // Sharded TP slices: fused route packing + metadata-prefetch K256 device-workfeed FC2 with the
  // L2 evict-first weight streams on every token count (calibrated against the official
  // trtllm-gen peer on B200).
  if (IsGeometry(shape, 4096, 512, 512, 10) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH4096I512E512K10, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }
  if (IsGeometry(shape, 4096, 256, 512, 10) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH4096I256E512K10, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }
  if (IsGeometry(shape, 3072, 768, 256, 8) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH3072I768E256K8, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }
  if (IsGeometry(shape, 3072, 384, 256, 8) && (shape.num_tokens >= 1 && shape.num_tokens <= 32)) {
    return {true, Geometry::kH3072I384E256K8, RouteLayout::kGpuPacked,
            RoutePacker::kFusedFc1, Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK256, 128, 4, 144};
  }

  return SelectAdditionalDirectSchedule(shape);
}

constexpr Schedule SelectSchedule(const Shape& shape) {
  return kTarget == Target::kSm100a ? SelectSm100aSchedule(shape) : SelectSm103aSchedule(shape);
}

// Every linear-grid row is a supported fused route-pack K256 FC2 row of its target whose route
// tiles (the packed workspace's count) fit the program's metadata table: the host picks the FC2
// program and arms its workfeed counter from that schedule and the predicate.
constexpr bool LinearK256Fc2RowsAreFusedK256(Target target) {
  for (int32_t t = 0; t <= 64; ++t) {
    const Shape shapes[] = {Shape{t, 4096, 256, 512, 512, 10}, Shape{t, 4096, 512, 512, 512, 10},
                            Shape{t, 3072, 384, 256, 256, 8}};
    for (const Shape& shape : shapes) {
      if (!LinearK256Fc2(target, shape)) continue;
      const Schedule s =
          target == Target::kSm100a ? SelectSm100aSchedule(shape) : SelectSm103aSchedule(shape);
      if (!s.supported || s.route_layout != RouteLayout::kGpuPacked ||
          s.route_packer != RoutePacker::kFusedFc1 || s.fc2 != Fc2Schedule::kRouteParallelK256)
        return false;
      const int32_t routes = shape.num_tokens * shape.top_k;
      const int32_t nonempty = routes < shape.num_experts ? routes : shape.num_experts;
      if (nonempty + (routes - nonempty) / 8 > kLinearK256Fc2MaxRouteTiles) return false;
    }
  }
  return true;
}
static_assert(LinearK256Fc2RowsAreFusedK256(Target::kSm100a));
static_assert(LinearK256Fc2RowsAreFusedK256(Target::kSm103a));
static_assert(!LinearK256Fc2(Target::kSm103a, {7, 3072, 384, 256, 256, 8}));
static_assert(LinearK256Fc2(Target::kSm103a, {8, 3072, 384, 256, 256, 8}));
static_assert(!LinearK256Fc2(Target::kSm103a, {9, 3072, 384, 256, 256, 8}));
static_assert(LinearK256Fc2(Target::kSm103a, {10, 3072, 384, 256, 256, 8}));
static_assert(LinearK256Fc2(Target::kSm103a, {32, 3072, 384, 256, 256, 8}));
static_assert(!LinearK256Fc2(Target::kSm103a, {33, 3072, 384, 256, 256, 8}));
static_assert(!LinearK256Fc2(Target::kSm100a, {7, 3072, 384, 256, 256, 8}));
static_assert(LinearK256Fc2(Target::kSm100a, {8, 3072, 384, 256, 256, 8}));
static_assert(!LinearK256Fc2(Target::kSm100a, {9, 3072, 384, 256, 256, 8}));
static_assert(LinearK256Fc2(Target::kSm100a, {10, 3072, 384, 256, 256, 8}));
static_assert(LinearK256Fc2(Target::kSm100a, {32, 3072, 384, 256, 256, 8}));
static_assert(!LinearK256Fc2(Target::kSm103a, {8, 3072, 768, 256, 256, 8}));
static_assert(!LinearK256Fc2(Target::kSm100a, {8, 3072, 768, 256, 256, 8}));

constexpr Shape E512Shape(int32_t tokens) { return {tokens, 2048, 512, 512, 512, 10}; }
constexpr Shape E60Shape(int32_t tokens) { return {tokens, 2048, 1536, 60, 60, 4}; }
constexpr Shape E192SiluShape(int32_t tokens) { return {tokens, 6144, 1536, 192, 192, 4}; }
constexpr Shape E384Shape(int32_t tokens) { return {tokens, 2560, 768, 384, 384, 4}; }
constexpr Shape Q30Shape(int32_t tokens) { return {tokens, 2048, 768, 128, 128, 8}; }

// Fused route packing (no route_pack launch): sm103 Q30 T8..T32 and sm100 Q30 T16..T32.
static_assert(SelectSm103aSchedule(Q30Shape(7)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(Q30Shape(7)).route_packer == RoutePacker::kNone);
static_assert(SelectSm103aSchedule(Q30Shape(8)).route_layout == RouteLayout::kGpuPacked);
static_assert(SelectSm103aSchedule(Q30Shape(8)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(Q30Shape(8)).fc1 == Fc1Schedule::kPersistentDeviceWorkfeed);
static_assert(SelectSm103aSchedule(Q30Shape(8)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm103aSchedule(Q30Shape(8)).workfeed_ctas == 144);
static_assert(SelectSm103aSchedule(Q30Shape(9)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(Q30Shape(9)).fc1 == Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed);
static_assert(SelectSm103aSchedule(Q30Shape(9)).fc2 ==
              Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed);
static_assert(SelectSm103aSchedule(Q30Shape(11)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(Q30Shape(11)).fc1 == Fc1Schedule::kPersistentDeviceWorkfeed);
static_assert(SelectSm103aSchedule(Q30Shape(11)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm103aSchedule(Q30Shape(20)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(Q30Shape(31)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(Q30Shape(31)).fc2 ==
              Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed);
static_assert(SelectSm103aSchedule(Q30Shape(32)).route_packer == RoutePacker::kFusedFc1);
static_assert(!SelectSm103aSchedule(Q30Shape(33)).supported);
static_assert(SelectSm103aSchedule(Q30Shape(31)).workfeed_ctas == 144);
static_assert(SelectSm100aSchedule(Q30Shape(9)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(Q30Shape(9)).route_packer == RoutePacker::kNone);
static_assert(SelectSm100aSchedule(Q30Shape(15)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(Q30Shape(15)).route_packer == RoutePacker::kNone);
static_assert(SelectSm100aSchedule(Q30Shape(16)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm100aSchedule(Q30Shape(16)).fc1 == Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed);
static_assert(SelectSm100aSchedule(Q30Shape(16)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm100aSchedule(Q30Shape(18)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm100aSchedule(Q30Shape(18)).fc1 == Fc1Schedule::kPersistentDeviceWorkfeed);
static_assert(SelectSm100aSchedule(Q30Shape(31)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm100aSchedule(Q30Shape(32)).route_packer == RoutePacker::kFusedFc1);
static_assert(!SelectSm100aSchedule(Q30Shape(33)).supported);
static_assert(!IsFusedRoutePackRow(Target::kSm100a, Q30Shape(15)));
static_assert(IsFusedRoutePackRow(Target::kSm100a, Q30Shape(16)));
static_assert(!IsFusedRoutePackRow(Target::kSm103a, {11, 2048, 768, 128, 64, 8}));
static_assert(IsFusedRoutePackRow(Target::kSm100a, {1, 4096, 512, 512, 512, 10}));
static_assert(IsFusedRoutePackRow(Target::kSm103a, {32, 3072, 384, 256, 256, 8}));
static_assert(SelectSm100aSchedule({1, 4096, 256, 512, 512, 10}).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm100aSchedule({1, 4096, 256, 512, 512, 10}).fc1 == Fc1Schedule::kPersistentDeviceWorkfeed);
static_assert(SelectSm100aSchedule({32, 3072, 768, 256, 256, 8}).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm100aSchedule({32, 3072, 768, 256, 256, 8}).workfeed_ctas == 144);
static_assert(SelectSm103aSchedule({16, 4096, 512, 512, 512, 10}).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule({16, 4096, 512, 512, 512, 10}).workfeed_ctas == 144);

// The sm_103a fused route-pack FC1 + K512 MMA-U2 device-workfeed FC2 rows: representative rows
// of each run with their workfeed, and the rows beside them that keep their schedules.
// Every row, geometry field included, is checked further below: CheckPublicBoundaries walks
// the five other geometries and Sm103aE60FusedK512MmaU2Rows the E60 geometry.
constexpr bool SelectsSm103aFusedK512MmaU2(const Shape& shape, int32_t workfeed_ctas) {
  const Schedule s = SelectSm103aSchedule(shape);
  return s.supported && s.route_layout == RouteLayout::kGpuPacked &&
         s.route_packer == RoutePacker::kFusedFc1 &&
         s.fc1 == Fc1Schedule::kPersistentDeviceWorkfeed &&
         s.fc2 == Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed && s.finalize_threads == 128 &&
         s.finalize_unroll == 4 && s.workfeed_ctas == workfeed_ctas;
}
// The sm_100a fused route-pack FC1 + K512 MMA-U2 device-workfeed FC2 rows that
// Sm100aFusedK512MmaU2WorkfeedCtas lists (the H2048 I768 fused rows keep their own checks).
constexpr bool SelectsSm100aFusedK512MmaU2(const Shape& shape, int32_t workfeed_ctas) {
  const Schedule s = SelectSm100aSchedule(shape);
  return s.supported && s.route_layout == RouteLayout::kGpuPacked &&
         s.route_packer == RoutePacker::kFusedFc1 &&
         s.fc1 == Fc1Schedule::kPersistentDeviceWorkfeed &&
         s.fc2 == Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed && s.finalize_threads == 128 &&
         s.finalize_unroll == 4 && s.workfeed_ctas == workfeed_ctas;
}
static_assert(SelectSm103aSchedule({5, 4096, 1536, 128, 128, 8}).route_layout ==
              RouteLayout::kDirect);
static_assert(SelectsSm103aFusedK512MmaU2({6, 4096, 1536, 128, 128, 8}, 0));
static_assert(SelectsSm103aFusedK512MmaU2({8, 4096, 1536, 128, 128, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({24, 4096, 1536, 128, 128, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({25, 4096, 1536, 128, 128, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({32, 4096, 1536, 128, 128, 8}, 144));
static_assert(SelectSm103aSchedule({14, 3072, 1536, 256, 256, 8}).route_layout ==
              RouteLayout::kDirect);
static_assert(SelectsSm103aFusedK512MmaU2({15, 3072, 1536, 256, 256, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({20, 3072, 1536, 256, 256, 8}, 0));
static_assert(SelectsSm103aFusedK512MmaU2({21, 3072, 1536, 256, 256, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({32, 3072, 1536, 256, 256, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({16, 2048, 512, 256, 256, 8}, 144));
static_assert(SelectSm103aSchedule({18, 2048, 512, 256, 256, 8}).fc2 ==
              Fc2Schedule::kRouteParallelK512Stage5DeviceWorkfeed);
static_assert(SelectsSm103aFusedK512MmaU2({19, 2048, 512, 256, 256, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({20, 2048, 512, 256, 256, 8}, 0));
static_assert(SelectsSm103aFusedK512MmaU2({27, 2048, 512, 256, 256, 8}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({28, 2048, 512, 256, 256, 8}, 144));
static_assert(SelectSm103aSchedule({30, 2048, 512, 256, 256, 8}).fc2 ==
              Fc2Schedule::kRouteParallelK512Stage5DeviceWorkfeed);
static_assert(SelectSm103aSchedule({31, 2048, 512, 256, 256, 8}).fc2 ==
              Fc2Schedule::kRouteParallelK512Stage5DeviceWorkfeed);
static_assert(SelectsSm103aFusedK512MmaU2({32, 2048, 512, 256, 256, 8}, 144));
static_assert(SelectSm103aSchedule({4, 4096, 1024, 512, 512, 10}).route_layout ==
              RouteLayout::kDirect);
static_assert(SelectsSm103aFusedK512MmaU2({5, 4096, 1024, 512, 512, 10}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({6, 4096, 1024, 512, 512, 10}, 0));
static_assert(SelectsSm103aFusedK512MmaU2({8, 4096, 1024, 512, 512, 10}, 0));
static_assert(SelectsSm103aFusedK512MmaU2({9, 4096, 1024, 512, 512, 10}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({15, 4096, 1024, 512, 512, 10}, 0));
static_assert(SelectsSm103aFusedK512MmaU2({25, 4096, 1024, 512, 512, 10}, 144));
static_assert(SelectSm103aSchedule({26, 4096, 1024, 512, 512, 10}).fc1 ==
              Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed);
static_assert(SelectsSm103aFusedK512MmaU2({1, 6144, 3072, 128, 128, 4}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({4, 6144, 3072, 128, 128, 4}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({15, 6144, 3072, 128, 128, 4}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({16, 6144, 3072, 128, 128, 4}, 144));
static_assert(SelectsSm103aFusedK512MmaU2({32, 6144, 3072, 128, 128, 4}, 144));
static_assert(SelectSm100aSchedule({1, 6144, 3072, 128, 128, 4}).route_layout ==
              RouteLayout::kDirect);
static_assert(SelectsSm100aFusedK512MmaU2({4, 6144, 3072, 128, 128, 4}, 144));
static_assert(SelectSm100aSchedule({5, 6144, 3072, 128, 128, 4}).route_packer ==
              RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule({8, 6144, 3072, 128, 128, 4}).route_packer ==
              RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule({9, 6144, 3072, 128, 128, 4}).route_packer ==
              RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule({10, 6144, 3072, 128, 128, 4}).route_packer ==
              RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule({32, 6144, 3072, 128, 128, 4}).route_packer ==
              RoutePacker::kGeneral);

// Compile-time boundary tests keep the public policy stable even before the
// generated kernel inventory is present.
static_assert(!SelectSm103aSchedule(E512Shape(0)).supported);
static_assert(SelectSm103aSchedule(E512Shape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E512Shape(2)).finalize_threads == 128);
static_assert(SelectSm103aSchedule(E512Shape(3)).finalize_threads == 128);
static_assert(SelectSm103aSchedule(E512Shape(4)).finalize_threads == 32);
static_assert(SelectSm100aSchedule(E512Shape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E512Shape(3)).finalize_threads == 128);
static_assert(SelectSm100aSchedule(E512Shape(4)).finalize_threads == 32);
static_assert(SelectSm103aSchedule(E512Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E512Shape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(E512Shape(32)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm103aSchedule(E512Shape(32)).finalize_threads == 32);
static_assert(!SelectSm103aSchedule(E512Shape(33)).supported);
static_assert(SelectSm103aSchedule(E60Shape(4)).fc2 == Fc2Schedule::kRouteParallelK768K96);
static_assert(SelectSm103aSchedule(E60Shape(5)).fc2 ==
              Fc2Schedule::kRouteParallelK768K96PaddedScale);
static_assert(SelectSm103aSchedule(E60Shape(10)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(E60Shape(11)).route_packer == RoutePacker::kE64Scan1);
static_assert(SelectSm103aSchedule(E60Shape(12)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(E60Shape(16)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(E60Shape(17)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectsSm103aFusedK512MmaU2(E60Shape(20), 144));
static_assert(SelectsSm103aFusedK512MmaU2(E60Shape(27), 144));
static_assert(SelectSm103aSchedule(E60Shape(32)).route_packer == RoutePacker::kFusedFc1);
static_assert(SelectSm103aSchedule(E192SiluShape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E192SiluShape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E192SiluShape(32)).fc1 == Fc1Schedule::kPersistent);
static_assert(ActivationForGeometry(SelectSm103aSchedule(E192SiluShape(1)).geometry) ==
              Activation::kSiLU);
static_assert(Gemm1WeightRows(E192SiluShape(1), SelectSm103aSchedule(E192SiluShape(1))) == 1536);
static_assert(Gemm1WeightRows(E512Shape(1), SelectSm103aSchedule(E512Shape(1))) == 1024);
static_assert(SelectSm100aSchedule(E512Shape(23)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E60Shape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E60Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E60Shape(10)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E60Shape(11)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E60Shape(12)).route_layout == RouteLayout::kGpuPacked);
static_assert(SelectSm100aSchedule(E60Shape(12)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule(E60Shape(12)).workfeed_ctas == 144);
static_assert(SelectSm100aSchedule(E60Shape(13)).route_layout == RouteLayout::kGpuPacked);
static_assert(SelectSm100aSchedule(E60Shape(13)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule(E60Shape(13)).workfeed_ctas == 144);
static_assert(SelectSm100aSchedule(E60Shape(14)).route_layout == RouteLayout::kGpuPacked);
static_assert(SelectSm100aSchedule(E60Shape(14)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule(E60Shape(14)).workfeed_ctas == 144);
static_assert(SelectSm100aSchedule(E60Shape(14)).fc2 == Fc2Schedule::kRouteParallelK512DeviceWorkfeed);
static_assert(SelectSm100aSchedule(E60Shape(15)).route_layout == RouteLayout::kGpuPacked);
static_assert(SelectSm100aSchedule(E60Shape(15)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule(E60Shape(15)).fc1 == Fc1Schedule::kPersistentDeviceWorkfeed);
static_assert(SelectSm100aSchedule(E60Shape(15)).fc2 == Fc2Schedule::kRouteParallelK512DeviceWorkfeed);
static_assert(SelectSm100aSchedule(E60Shape(32)).workfeed_ctas == 144);
static_assert(SelectSm100aSchedule(E192SiluShape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E192SiluShape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E192SiluShape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E192SiluShape(32)).fc1 == Fc1Schedule::kPersistent);
static_assert(!SelectSm103aSchedule({1, 2048, 512, 511, 511, 10}).supported);
static_assert(!SelectSm100aSchedule({1, 2048, 1536, 60, 60, 5}).supported);
static_assert(SelectSm100aSchedule(E384Shape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E384Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E384Shape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E384Shape(32)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm103aSchedule(E384Shape(1)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E384Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E384Shape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(E384Shape(32)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(ActivationForGeometry(SelectSm103aSchedule(E384Shape(1)).geometry) ==
              Activation::kSwiGLU);
static_assert(Gemm1WeightRows(E384Shape(1), SelectSm103aSchedule(E384Shape(1))) == 1536);
static_assert(!SelectSm100aSchedule(E384Shape(0)).supported);
static_assert(!SelectSm103aSchedule(E384Shape(33)).supported);

// The workfeed CTAs of the sm_103a rows that run the fused route-pack FC1 and the K512 MMA-U2
// device-workfeed FC2 (0: the device's SM count), restated for the boundary checks below; -1 for
// every other row.
constexpr int32_t Sm103aFusedK512MmaU2WorkfeedCtas(Geometry geometry, int32_t tokens) {
  switch (geometry) {
    case Geometry::kH4096I1536E128K8:
      if (TokensIn(tokens, 6, 7, 12, 13, 15, 16, 17, 19, 20, 21, 23)) return 0;
      return tokens >= 8 && tokens <= 32 ? 144 : -1;
    case Geometry::kH3072I1536E256K8:
      if (tokens == 20) return 0;
      return tokens >= 15 && tokens <= 32 ? 144 : -1;
    case Geometry::kH4096I1024E512K10:
      if (TokensIn(tokens, 6, 7, 8, 15)) return 0;
      return tokens >= 5 && tokens <= 25 ? 144 : -1;
    case Geometry::kH2048I512E256K8:
      if (TokensIn(tokens, 20, 22, 24)) return 0;
      return TokensIn(tokens, 16, 17, 19, 21, 23, 25, 26, 27, 28, 29, 32) ? 144 : -1;
    case Geometry::kH2048I1536E60K4:
      if (TokensIn(tokens, 12, 16, 19, 23, 25, 26, 32)) return 0;
      return tokens >= 12 && tokens <= 32 ? 144 : -1;
    case Geometry::kH6144I3072E128K4:
      return tokens >= 1 && tokens <= 32 ? 144 : -1;
    default:
      return -1;
  }
}

// The workfeed CTAs of the sm_100a rows that run the fused route-pack FC1 and the K512 MMA-U2
// device-workfeed FC2 and are walked by CheckPublicBoundaries (0: the device's SM count); -1 for
// every other row. The H2048 I768 fused rows keep their own branches there.
constexpr int32_t Sm100aFusedK512MmaU2WorkfeedCtas(Geometry geometry, int32_t tokens) {
  switch (geometry) {
    case Geometry::kH6144I3072E128K4:
      return tokens == 4 ? 144 : -1;
    default:
      return -1;
  }
}

// Every sm_103a E60 row from T12 selects the E60 geometry and runs the fused route-pack K512
// MMA-U2 schedule with the workfeed listed above.
constexpr bool Sm103aE60FusedK512MmaU2Rows() {
  for (int32_t t = 12; t <= 32; ++t) {
    const Schedule s = SelectSm103aSchedule(E60Shape(t));
    const int32_t workfeed = Sm103aFusedK512MmaU2WorkfeedCtas(Geometry::kH2048I1536E60K4, t);
    if (!s.supported || s.geometry != Geometry::kH2048I1536E60K4) return false;
    if (workfeed < 0 || !SelectsSm103aFusedK512MmaU2(E60Shape(t), workfeed)) return false;
  }
  return true;
}
static_assert(Sm103aE60FusedK512MmaU2Rows());

// The fused route-pack FC1 programs carry the plain SwiGLU epilogue, except two sm_103a programs
// with the parameterized-SwiGLU epilogue (they differ only in the weight cache hint) that the
// generated manifest selects for the H6144 I3072 geometry only. So no row of the SiLU, SiTU or
// clamped SwiGLU (H4096 I2048) geometries selects RoutePacker::kFusedFc1 on either target, and an
// H6144 I3072 row selects it exactly where Sm100aFusedK512MmaU2WorkfeedCtas (sm_100a) and
// Sm103aFusedK512MmaU2WorkfeedCtas (sm_103a) list it.
constexpr bool FusedFc1RowsMatchTheirEpilogue() {
  const Shape plain_only[] = {E192SiluShape(0), Shape{0, 4096, 2048, 256, 256, 6},
                              Shape{0, 3584, 3072, 896, 896, 16}};
  for (Shape shape : plain_only) {
    for (int32_t t = 0; t <= 33; ++t) {
      shape.num_tokens = t;
      if (SelectSm100aSchedule(shape).route_packer == RoutePacker::kFusedFc1 ||
          SelectSm103aSchedule(shape).route_packer == RoutePacker::kFusedFc1) {
        return false;
      }
    }
  }
  for (int32_t t = 0; t <= 33; ++t) {
    const Shape shape{t, 6144, 3072, 128, 128, 4};
    const bool fused = SelectSm103aSchedule(shape).route_packer == RoutePacker::kFusedFc1;
    const bool fused100 = SelectSm100aSchedule(shape).route_packer == RoutePacker::kFusedFc1;
    if (fused100 != (Sm100aFusedK512MmaU2WorkfeedCtas(Geometry::kH6144I3072E128K4, t) >= 0) ||
        fused != (Sm103aFusedK512MmaU2WorkfeedCtas(Geometry::kH6144I3072E128K4, t) >= 0)) {
      return false;
    }
  }
  return true;
}
static_assert(FusedFc1RowsMatchTheirEpilogue());

constexpr bool CheckPublicBoundaries(Shape shape, Geometry geometry,
                                     Activation activation = Activation::kSwiGLU,
                                     bool static_fc1 = false) {
  for (int32_t tokens = 0; tokens <= 33; ++tokens) {
    shape.num_tokens = tokens;
    for (int32_t target = 0; target < 2; ++target) {
      const Schedule schedule =
          target == 0 ? SelectSm100aSchedule(shape) : SelectSm103aSchedule(shape);
      const RoutePacker packed_packer =
          PackedRoutePacker(target == 0 ? Target::kSm100a : Target::kSm103a, shape);
      const int32_t fused_k512_workfeed =
          target == 1 ? Sm103aFusedK512MmaU2WorkfeedCtas(geometry, tokens)
                      : Sm100aFusedK512MmaU2WorkfeedCtas(geometry, tokens);
      if (tokens == 0 || tokens == 33) {
        if (schedule.supported) return false;
      } else if (fused_k512_workfeed >= 0) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kFusedFc1 ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != fused_k512_workfeed) return false;
      } else if (geometry == Geometry::kH2048I768E128K8 &&
                 ((target == 1 && tokens >= 8 && tokens <= 19) ||
                  (target == 0 && (tokens == 16 || tokens == 17 || (tokens >= 20 && tokens <= 31))))) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != packed_packer ||
            schedule.fc1 != (target == 0 || tokens == 9 || tokens == 10 ? Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed
                                         : Fc1Schedule::kPersistentDeviceWorkfeed) ||
            schedule.fc2 != (target == 1 && (tokens == 9 || tokens == 10)
                                 ? Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed
                                 : Fc2Schedule::kRouteParallelK256) ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (target == 0 && geometry == Geometry::kH6144I3072E128K4 &&
                 (tokens >= 4 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kGeneral ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (geometry == Geometry::kH2048I768E128K8 &&
                 (tokens == 32 || (target == 1 && tokens >= 20 && tokens <= 31)
                  || (target == 0 && (tokens == 18 || tokens == 19)))) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != packed_packer ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512MmaU2DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (geometry == Geometry::kH2048I512E256K8 &&
                 (tokens >= 16 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kGeneral ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != (target == 0 ? Fc2Schedule::kRouteParallelK512DeviceWorkfeed
                                         : Fc2Schedule::kRouteParallelK512Stage5DeviceWorkfeed) ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (target == 0 && geometry == Geometry::kH4096I1536E128K8 &&
                 (tokens >= 12 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kGeneral ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (geometry == Geometry::kH3072I1536E256K8 && (tokens >= 16 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kGeneral ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (target == 0 && geometry == Geometry::kH4096I1024E512K10 &&
                 (tokens >= 8 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kGeneral ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 0) return false;
      } else if (target == 1 && geometry == Geometry::kH4096I1024E512K10 &&
                 (tokens >= 8 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kGeneral ||
            schedule.fc1 != Fc1Schedule::kPersistentEarlySfbDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK512DeviceWorkfeed ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (geometry == Geometry::kH4096I512E512K10 && (tokens >= 1 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kFusedFc1 ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK256 ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (geometry == Geometry::kH4096I256E512K10 && (tokens >= 1 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kFusedFc1 ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK256 ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if ((geometry == Geometry::kH3072I768E256K8 || geometry == Geometry::kH3072I384E256K8) && (tokens >= 1 && tokens <= 32)) {
        if (!schedule.supported || schedule.geometry != geometry ||
            ActivationForGeometry(geometry) != activation ||
            schedule.route_layout != RouteLayout::kGpuPacked ||
            schedule.route_packer != RoutePacker::kFusedFc1 ||
            schedule.fc1 != Fc1Schedule::kPersistentDeviceWorkfeed ||
            schedule.fc2 != Fc2Schedule::kRouteParallelK256 ||
            schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
            schedule.workfeed_ctas != 144) return false;
      } else if (!schedule.supported || schedule.geometry != geometry ||
                 ActivationForGeometry(geometry) != activation ||
                 schedule.route_layout != RouteLayout::kDirect ||
                 schedule.route_packer != RoutePacker::kNone ||
                 schedule.fc1 != (static_fc1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent) ||
                 schedule.fc2 != Fc2Schedule::kRouteParallelK256 ||
                 schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
                 schedule.workfeed_ctas != 0) {
        return false;
      }
    }
  }
  return true;
}

static_assert(CheckPublicBoundaries({1, 2048, 768, 128, 128, 8}, Geometry::kH2048I768E128K8));
static_assert(CheckPublicBoundaries({1, 4096, 1536, 128, 128, 8}, Geometry::kH4096I1536E128K8));
static_assert(CheckPublicBoundaries({1, 2048, 512, 256, 256, 8}, Geometry::kH2048I512E256K8));
static_assert(CheckPublicBoundaries({1, 4096, 1024, 512, 512, 10}, Geometry::kH4096I1024E512K10));
static_assert(CheckPublicBoundaries({1, 3072, 1536, 256, 256, 8}, Geometry::kH3072I1536E256K8));
static_assert(CheckPublicBoundaries({1, 6144, 3072, 128, 128, 4}, Geometry::kH6144I3072E128K4,
                                    Activation::kSwiGLUParameterized));
static_assert(CheckPublicBoundaries({1, 3584, 3072, 896, 896, 16}, Geometry::kH3584I3072E896K16,
                                    Activation::kSiTU, true));
static_assert(CheckPublicBoundaries({1, 4096, 512, 512, 512, 10}, Geometry::kH4096I512E512K10));
static_assert(CheckPublicBoundaries({1, 4096, 256, 512, 512, 10}, Geometry::kH4096I256E512K10));
static_assert(CheckPublicBoundaries({1, 3072, 768, 256, 256, 8}, Geometry::kH3072I768E256K8));
static_assert(CheckPublicBoundaries({1, 3072, 384, 256, 256, 8}, Geometry::kH3072I384E256K8));
static_assert(!SelectSm100aSchedule({1, 2048, 768, 128, 64, 8}).supported);
static_assert(!SelectSm103aSchedule({1, 2048, 768, 128, 128, 4}).supported);

struct Invocation {
  Shape shape;
  void* output;
  void* workspace;
  const void* hidden_states_q;
  const void* hidden_states_scale;
  const void* topk_ids;
  const void* topk_weights;
  const void* gemm1_weights;
  const void* gemm1_weights_scale;
  const void* gemm2_weights;
  const void* gemm2_weights_scale;
  const void* output1_scale_scalar;
  const void* output1_scale_gate_scalar;
  const void* output2_scale_scalar;
  size_t workspace_bytes;
  const void* gemm1_alpha = nullptr;
  const void* gemm1_beta = nullptr;
  const void* gemm1_clamp_limit = nullptr;
};

enum class StatusDomain : uint8_t {
  kSuccess = 0,
  kCudaRuntime,
  kCudaDriver,
  kInvalidManifest,
};

struct ManifestStatus {
  StatusDomain domain;
  int32_t code;
  const char* operation;

  constexpr bool Ok() const { return domain == StatusDomain::kSuccess; }
  static constexpr ManifestStatus Success() { return {StatusDomain::kSuccess, 0, nullptr}; }
};

// A generated submit thunk owns the concrete kernel signature. Its opaque
// argument object may contain CUtensorMap values and must remain live for the
// duration of SubmitExtendedKernel. This preserves the by-value grid-constant
// tensor-map ABI instead of substituting device descriptor pointers.
using KernelSubmit = cudaError_t (*)(const cudaLaunchConfig_t*, const void*);

struct KernelLaunch {
  const char* name;
  dim3 grid;
  dim3 block;
  dim3 cluster;
  size_t dynamic_smem_bytes;
  bool programmatic_dependent_launch;
  bool cooperative;
  bool spread_cluster;
  KernelSubmit submit;
  const void* arguments;
  // Negative means no launch-local preference.
  int preferred_shared_memory_carveout = -1;
};

using LaunchVisitor = void (*)(const KernelLaunch&, void*);

template <typename Kernel, typename... Args>
inline cudaError_t SubmitExtendedKernel(const cudaLaunchConfig_t* config, Kernel kernel,
                                        Args&&... args) {
  return cudaLaunchKernelEx(config, kernel, std::forward<Args>(args)...);
}

}  // namespace flashinfer::warp_decode
