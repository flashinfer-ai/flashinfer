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
  kH2560I768E384K4,
  kH2048I768E128K8,
  kH4096I1536E128K8,
  kH2048I512E256K8,
  kH4096I1024E512K10,
  kH3072I1536E256K8,
  kH6144I3072E128K4,
  kH3584I3072E896K16,
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
};

enum class Fc1Schedule : uint8_t {
  kStatic = 0,
  kPersistent,
  kPersistentDeviceWorkfeed,
  kPersistentPaddedScaleDeviceWorkfeed,
};

enum class Fc2Schedule : uint8_t {
  kRouteParallelK256 = 0,
  kRouteParallelK512DeviceWorkfeed,
  kRouteParallelK768K96,
  kRouteParallelK768K96PaddedScale,
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

constexpr Activation ActivationForGeometry(Geometry geometry) {
  if (geometry == Geometry::kH6144I1536E192K4) return Activation::kSiLU;
  if (geometry == Geometry::kH6144I3072E128K4) return Activation::kSwiGLUParameterized;
  if (geometry == Geometry::kH3584I3072E896K16) return Activation::kSiTU;
  return Activation::kSwiGLU;
}

constexpr int32_t Gemm1WeightRows(const Shape& shape, const Schedule& schedule) {
  return ActivationForGeometry(schedule.geometry) == Activation::kSiLU
             ? shape.intermediate_size
             : 2 * shape.intermediate_size;
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
  } else {
    return UnsupportedSchedule();
  }
  return {true,
          geometry,
          RouteLayout::kDirect,
          RoutePacker::kNone,
          shape.num_tokens == 1 || geometry == Geometry::kH3584I3072E896K16
              ? Fc1Schedule::kStatic
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

  if (IsGeometry(shape, 2560, 768, 384, 4)) {
    return {true,
            Geometry::kH2560I768E384K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  if (IsGeometry(shape, 2048, 512, 512, 10)) {
    if (shape.num_tokens < 23) {
      return {true,
              Geometry::kH2048I512E512K10,
              RouteLayout::kDirect,
              RoutePacker::kNone,
              shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
              Fc2Schedule::kRouteParallelK256,
              32,
              4,
              0};
    }
    return {true,
            Geometry::kH2048I512E512K10,
            RouteLayout::kGpuPacked,
            RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK512DeviceWorkfeed,
            32,
            4,
            kPackedWorkfeedCtas};
  }

  if (IsGeometry(shape, 2048, 1536, 60, 4)) {
    if (shape.num_tokens < 8) {
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
    return {true,
            Geometry::kH2048I1536E60K4,
            RouteLayout::kGpuPacked,
            shape.num_tokens <= 16 ? RoutePacker::kE64Scan2 : RoutePacker::kGeneral,
            Fc1Schedule::kPersistentDeviceWorkfeed,
            Fc2Schedule::kRouteParallelK768K96,
            128,
            4,
            kPackedWorkfeedCtas};
  }

  if (IsGeometry(shape, 6144, 1536, 192, 4)) {
    return {true,
            Geometry::kH6144I1536E192K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  return SelectAdditionalDirectSchedule(shape);
}

constexpr Schedule SelectSm100aSchedule(const Shape& shape) {
  if (shape.num_tokens < 1 || shape.num_tokens > kMaximumTokens) {
    return UnsupportedSchedule();
  }

  if (IsGeometry(shape, 2560, 768, 384, 4)) {
    return {true,
            Geometry::kH2560I768E384K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
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
            shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            32,
            4,
            0};
  }

  if (IsGeometry(shape, 2048, 1536, 60, 4)) {
    if (shape.num_tokens >= 20) {
      return {true,
              Geometry::kH2048I1536E60K4,
              RouteLayout::kGpuPacked,
              RoutePacker::kGeneral,
              Fc1Schedule::kPersistentDeviceWorkfeed,
              Fc2Schedule::kRouteParallelK256,
              128,
              4,
              kPackedWorkfeedCtas};
    }
    return {true,
            Geometry::kH2048I1536E60K4,
            RouteLayout::kDirect,
            RoutePacker::kNone,
            shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
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
            shape.num_tokens == 1 ? Fc1Schedule::kStatic : Fc1Schedule::kPersistent,
            Fc2Schedule::kRouteParallelK256,
            128,
            4,
            0};
  }

  return SelectAdditionalDirectSchedule(shape);
}

constexpr Schedule SelectSchedule(const Shape& shape) {
  return kTarget == Target::kSm100a ? SelectSm100aSchedule(shape) : SelectSm103aSchedule(shape);
}

constexpr Shape E512Shape(int32_t tokens) { return {tokens, 2048, 512, 512, 512, 10}; }
constexpr Shape E60Shape(int32_t tokens) { return {tokens, 2048, 1536, 60, 60, 4}; }
constexpr Shape E192SiluShape(int32_t tokens) { return {tokens, 6144, 1536, 192, 192, 4}; }
constexpr Shape E384Shape(int32_t tokens) { return {tokens, 2560, 768, 384, 384, 4}; }

// Compile-time boundary tests keep the public policy stable even before the
// generated kernel inventory is present.
static_assert(!SelectSm103aSchedule(E512Shape(0)).supported);
static_assert(SelectSm103aSchedule(E512Shape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSm103aSchedule(E512Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E512Shape(22)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(E512Shape(23)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm103aSchedule(E512Shape(32)).fc2 ==
              Fc2Schedule::kRouteParallelK512DeviceWorkfeed);
static_assert(!SelectSm103aSchedule(E512Shape(33)).supported);
static_assert(SelectSm103aSchedule(E60Shape(7)).fc2 == Fc2Schedule::kRouteParallelK768K96);
static_assert(SelectSm103aSchedule(E60Shape(8)).fc2 ==
              Fc2Schedule::kRouteParallelK768K96PaddedScale);
static_assert(SelectSm103aSchedule(E60Shape(10)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(E60Shape(11)).route_packer == RoutePacker::kE64Scan1);
static_assert(SelectSm103aSchedule(E60Shape(12)).route_packer == RoutePacker::kE64Scan2);
static_assert(SelectSm103aSchedule(E60Shape(16)).route_packer == RoutePacker::kE64Scan2);
static_assert(SelectSm103aSchedule(E60Shape(17)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm103aSchedule(E60Shape(32)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm103aSchedule(E192SiluShape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSm103aSchedule(E192SiluShape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E192SiluShape(32)).fc1 == Fc1Schedule::kPersistent);
static_assert(ActivationForGeometry(SelectSm103aSchedule(E192SiluShape(1)).geometry) ==
              Activation::kSiLU);
static_assert(Gemm1WeightRows(E192SiluShape(1), SelectSm103aSchedule(E192SiluShape(1))) == 1536);
static_assert(Gemm1WeightRows(E512Shape(1), SelectSm103aSchedule(E512Shape(1))) == 1024);
static_assert(SelectSm100aSchedule(E512Shape(23)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E60Shape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSm100aSchedule(E60Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E60Shape(11)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E60Shape(19)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E60Shape(20)).route_layout == RouteLayout::kGpuPacked);
static_assert(SelectSm100aSchedule(E60Shape(20)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSm100aSchedule(E60Shape(20)).fc1 == Fc1Schedule::kPersistentDeviceWorkfeed);
static_assert(SelectSm100aSchedule(E60Shape(20)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm100aSchedule(E60Shape(20)).workfeed_ctas == 152);
static_assert(SelectSm100aSchedule(E192SiluShape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E192SiluShape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSm100aSchedule(E192SiluShape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E192SiluShape(32)).fc1 == Fc1Schedule::kPersistent);
static_assert(!SelectSm103aSchedule({1, 2048, 512, 511, 511, 10}).supported);
static_assert(!SelectSm100aSchedule({1, 2048, 1536, 60, 60, 5}).supported);
static_assert(SelectSm100aSchedule(E384Shape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSm100aSchedule(E384Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm100aSchedule(E384Shape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm100aSchedule(E384Shape(32)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(SelectSm103aSchedule(E384Shape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSm103aSchedule(E384Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSm103aSchedule(E384Shape(32)).route_layout == RouteLayout::kDirect);
static_assert(SelectSm103aSchedule(E384Shape(32)).fc2 == Fc2Schedule::kRouteParallelK256);
static_assert(ActivationForGeometry(SelectSm103aSchedule(E384Shape(1)).geometry) ==
              Activation::kSwiGLU);
static_assert(Gemm1WeightRows(E384Shape(1), SelectSm103aSchedule(E384Shape(1))) == 1536);
static_assert(!SelectSm100aSchedule(E384Shape(0)).supported);
static_assert(!SelectSm103aSchedule(E384Shape(33)).supported);

constexpr bool CheckDirectBoundaries(Shape shape, Geometry geometry,
                                     Activation activation = Activation::kSwiGLU,
                                     bool static_fc1 = false) {
  for (int32_t tokens = 0; tokens <= 33; ++tokens) {
    shape.num_tokens = tokens;
    for (int32_t target = 0; target < 2; ++target) {
      const Schedule schedule =
          target == 0 ? SelectSm100aSchedule(shape) : SelectSm103aSchedule(shape);
      if (tokens == 0 || tokens == 33) {
        if (schedule.supported) return false;
      } else if (!schedule.supported || schedule.geometry != geometry ||
                 ActivationForGeometry(geometry) != activation ||
                 schedule.route_layout != RouteLayout::kDirect ||
                 schedule.route_packer != RoutePacker::kNone ||
                 schedule.fc1 != (tokens == 1 || static_fc1 ? Fc1Schedule::kStatic
                                                            : Fc1Schedule::kPersistent) ||
                 schedule.fc2 != Fc2Schedule::kRouteParallelK256 ||
                 schedule.finalize_threads != 128 || schedule.finalize_unroll != 4 ||
                 schedule.workfeed_ctas != 0) {
        return false;
      }
    }
  }
  return true;
}

static_assert(CheckDirectBoundaries({1, 2048, 768, 128, 128, 8}, Geometry::kH2048I768E128K8));
static_assert(CheckDirectBoundaries({1, 4096, 1536, 128, 128, 8}, Geometry::kH4096I1536E128K8));
static_assert(CheckDirectBoundaries({1, 2048, 512, 256, 256, 8}, Geometry::kH2048I512E256K8));
static_assert(CheckDirectBoundaries({1, 4096, 1024, 512, 512, 10}, Geometry::kH4096I1024E512K10));
static_assert(CheckDirectBoundaries({1, 3072, 1536, 256, 256, 8}, Geometry::kH3072I1536E256K8));
static_assert(CheckDirectBoundaries({1, 6144, 3072, 128, 128, 4}, Geometry::kH6144I3072E128K4,
                                    Activation::kSwiGLUParameterized));
static_assert(CheckDirectBoundaries({1, 3584, 3072, 896, 896, 16}, Geometry::kH3584I3072E896K16,
                                    Activation::kSiTU, true));
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
  bool allow_oversized_smem;
  bool cooperative;
  bool spread_cluster;
  KernelSubmit submit;
  const void* arguments;
};

using LaunchVisitor = void (*)(const KernelLaunch&, void*);

template <typename Kernel, typename... Args>
inline cudaError_t SubmitExtendedKernel(const cudaLaunchConfig_t* config, Kernel kernel,
                                        Args&&... args) {
  return cudaLaunchKernelEx(config, kernel, std::forward<Args>(args)...);
}

}  // namespace flashinfer::warp_decode
