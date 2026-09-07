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

inline constexpr uint32_t kGeneratedContractVersion = 1;
inline constexpr int32_t kMaximumTokens = 32;
inline constexpr int32_t kPackedWorkfeedCtas = 152;

enum class Geometry : uint8_t {
  kUnsupported = 0,
  kH2048I512E512K10,
  kH2048I1536E60K4,
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

constexpr bool IsGeometry(const Shape& shape, int32_t intermediate_size, int32_t num_experts,
                          int32_t top_k) {
  return shape.hidden_size == 2048 && shape.intermediate_size == intermediate_size &&
         shape.num_experts == num_experts && shape.local_num_experts == num_experts &&
         shape.top_k == top_k;
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

// This selector is the public calibration boundary. The generated manifest
// supplies implementations for these choices but must not independently
// reinterpret a shape or token boundary.
constexpr Schedule SelectSchedule(const Shape& shape) {
  if (shape.num_tokens < 1 || shape.num_tokens > kMaximumTokens) {
    return UnsupportedSchedule();
  }

  if (IsGeometry(shape, 512, 512, 10)) {
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

  if (IsGeometry(shape, 1536, 60, 4)) {
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

  return UnsupportedSchedule();
}

constexpr Shape E512Shape(int32_t tokens) { return {tokens, 2048, 512, 512, 512, 10}; }
constexpr Shape E60Shape(int32_t tokens) { return {tokens, 2048, 1536, 60, 60, 4}; }

// Compile-time boundary tests keep the public policy stable even before the
// generated kernel inventory is present.
static_assert(!SelectSchedule(E512Shape(0)).supported);
static_assert(SelectSchedule(E512Shape(1)).fc1 == Fc1Schedule::kStatic);
static_assert(SelectSchedule(E512Shape(2)).fc1 == Fc1Schedule::kPersistent);
static_assert(SelectSchedule(E512Shape(22)).route_layout == RouteLayout::kDirect);
static_assert(SelectSchedule(E512Shape(23)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSchedule(E512Shape(32)).fc2 == Fc2Schedule::kRouteParallelK512DeviceWorkfeed);
static_assert(!SelectSchedule(E512Shape(33)).supported);
static_assert(SelectSchedule(E60Shape(7)).fc2 == Fc2Schedule::kRouteParallelK768K96);
static_assert(SelectSchedule(E60Shape(8)).fc2 == Fc2Schedule::kRouteParallelK768K96PaddedScale);
static_assert(SelectSchedule(E60Shape(10)).route_layout == RouteLayout::kDirect);
static_assert(SelectSchedule(E60Shape(11)).route_packer == RoutePacker::kE64Scan1);
static_assert(SelectSchedule(E60Shape(12)).route_packer == RoutePacker::kE64Scan2);
static_assert(SelectSchedule(E60Shape(16)).route_packer == RoutePacker::kE64Scan2);
static_assert(SelectSchedule(E60Shape(17)).route_packer == RoutePacker::kGeneral);
static_assert(SelectSchedule(E60Shape(32)).route_packer == RoutePacker::kGeneral);
static_assert(!SelectSchedule({1, 2048, 512, 511, 511, 10}).supported);
static_assert(!SelectSchedule({1, 2048, 1536, 60, 60, 5}).supported);

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
