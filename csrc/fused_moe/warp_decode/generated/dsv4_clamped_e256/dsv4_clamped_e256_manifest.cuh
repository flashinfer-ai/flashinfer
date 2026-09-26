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

// Generated from the selected SM100 route inventory.
#pragma once
#include "../../cake_warp_decode_contract.cuh"
#include "declarations.cuh"
#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#if FLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR == 0
namespace flashinfer::warp_decode::generated::dsv4_clamped_e256 {
inline ManifestStatus Invalid(const char* operation) {
  return {StatusDomain::kInvalidManifest, 1, operation};
}
inline ManifestStatus Runtime(cudaError_t error, const char* operation) {
  return {StatusDomain::kCudaRuntime, static_cast<int32_t>(error), operation};
}
inline bool Supported(const Shape& shape) {
  return IsGeometry(shape, 4096, 2048, 256, 6) && shape.num_tokens >= 1 && shape.num_tokens <= 32;
}
inline int64_t WorkspaceSize(const Shape& shape, const Schedule&) {
  if (!Supported(shape)) return -1;
  switch (shape.num_tokens) {
    case 1: return 449792;
    case 2: return 898560;
    case 3: return 1347328;
    case 4: return 1795840;
    case 5: return 2244608;
    case 6: return 2693376;
    case 7: return 3142400;
    case 8: return 3590912;
    case 9: return 4039936;
    case 10: return 4488704;
    case 11: return 4938240;
    case 12: return 5386752;
    case 13: return 5835520;
    case 14: return 6284288;
    case 15: return 6733056;
    case 16: return 7181568;
    case 17: return 7630336;
    case 18: return 8079104;
    case 19: return 8527872;
    case 20: return 8976384;
    case 21: return 9425152;
    case 22: return 9874688;
    case 23: return 10323456;
    case 24: return 10771968;
    case 25: return 11220736;
    case 26: return 11669504;
    case 27: return 12118272;
    case 28: return 12566784;
    case 29: return 13015552;
    case 30: return 13464320;
    case 31: return 13913088;
    case 32: return 14361600;
    default: return -1;
  }
}

inline ManifestStatus PrepareWorkspace(const Invocation& inv, const Schedule& schedule, cudaStream_t stream) {
  const int64_t bytes = WorkspaceSize(inv.shape, schedule);
  if (bytes < 0 || inv.workspace == nullptr || inv.workspace_bytes < static_cast<size_t>(bytes))
    return Invalid("clamped E256 workspace size");
  std::vector<std::pair<size_t, std::vector<int32_t>>> initializers;
  switch (inv.shape.num_tokens) {
    case 1:
      initializers.push_back({55296u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
      initializers.push_back({55552u, {1, 9, 17, 25, 33, 41}});
      initializers.push_back({55808u, {6}});
      initializers.push_back({56064u, {192}});
      initializers.push_back({449536u, {192}});
      break;
    case 2:
      initializers.push_back({110592u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
      initializers.push_back({111104u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89}});
      initializers.push_back({111360u, {12}});
      initializers.push_back({111616u, {384}});
      initializers.push_back({898304u, {384}});
      break;
    case 3:
      initializers.push_back({165888u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}});
      initializers.push_back({166656u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137}});
      initializers.push_back({166912u, {18}});
      initializers.push_back({167168u, {576}});
      initializers.push_back({1347072u, {576}});
      break;
    case 4:
      initializers.push_back({221184u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}});
      initializers.push_back({221952u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185}});
      initializers.push_back({222208u, {24}});
      initializers.push_back({222464u, {768}});
      initializers.push_back({1795584u, {768}});
      break;
    case 5:
      initializers.push_back({276480u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}});
      initializers.push_back({277504u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233}});
      initializers.push_back({277760u, {30}});
      initializers.push_back({278016u, {960}});
      initializers.push_back({2244352u, {960}});
      break;
    case 6:
      initializers.push_back({331776u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}});
      initializers.push_back({333056u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281}});
      initializers.push_back({333312u, {36}});
      initializers.push_back({333568u, {1152}});
      initializers.push_back({2693120u, {1152}});
      break;
    case 7:
      initializers.push_back({387328u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}});
      initializers.push_back({388864u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329}});
      initializers.push_back({389120u, {42}});
      initializers.push_back({389376u, {1344}});
      initializers.push_back({3142144u, {1344}});
      break;
    case 8:
      initializers.push_back({442624u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}});
      initializers.push_back({444160u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345, 353, 361, 369, 377}});
      initializers.push_back({444416u, {48}});
      initializers.push_back({444672u, {1536}});
      initializers.push_back({3590656u, {1536}});
      break;
    default: break;
  }
  cudaError_t error = cudaMemsetAsync(inv.workspace, 0, static_cast<size_t>(bytes), stream);
  for (const auto& item : initializers) {
    if (error != cudaSuccess) break;
    error = cudaMemcpyAsync(static_cast<uint8_t*>(inv.workspace) + item.first, item.second.data(),
                            item.second.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  }
  // Keep host initializer storage alive until every accepted copy completes.
  const cudaError_t completed = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 workspace initialization");
  if (completed != cudaSuccess) return Runtime(completed, "clamped E256 workspace synchronization");
  return ManifestStatus::Success();
}

struct DeviceState { bool ready = false; };
inline std::array<DeviceState, 64>& DeviceStates() { static std::array<DeviceState, 64> states{}; return states; }
inline std::mutex& DeviceMutex() { static std::mutex mutex; return mutex; }
inline ManifestStatus EnsureDeviceReady(int32_t device, bool allow_initialization) {
  if (device < 0 || device >= 64) return Invalid("clamped E256 device index");
  int current = -1;
  cudaError_t current_status = cudaGetDevice(&current);
  if (current_status != cudaSuccess) return Runtime(current_status, "clamped E256 current device");
  if (current != device) return Invalid("clamped E256 current device mismatch");
  std::lock_guard<std::mutex> lock(DeviceMutex());
  auto& state = DeviceStates()[device];
  if (state.ready) return ManifestStatus::Success();
  if (!allow_initialization) return Invalid("clamped E256 device not prepared");
  cudaError_t error = cudaSuccess;
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100), cudaFuncAttributeMaxDynamicSharedMemorySize, 64128);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready_expert_order), cudaFuncAttributeMaxDynamicSharedMemorySize, 199040);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready), cudaFuncAttributeMaxDynamicSharedMemorySize, 199040);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe), cudaFuncAttributeMaxDynamicSharedMemorySize, 159872);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100), cudaFuncAttributeMaxDynamicSharedMemorySize, 85120);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  state.ready = true;
  return ManifestStatus::Success();
}

struct Args_device_00 {
  int* route_experts;
  int* route_map;
  int* tile_expert;
  int* tile_mn_limit;
  int* route_slots;
  int* num_non_exiting_ctas;
  int* fc1_work_counter;
  int* fc2_work_counter;
  int route_count;
  int top_k;
  int local_expert_offset;
  int num_experts;
  int fc1_initial_work;
  int fc2_initial_work;
};
inline cudaError_t Submit_device_00(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_00*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_e256_route_pack,
      args.route_experts, args.route_map, args.tile_expert, args.tile_mn_limit, args.route_slots, args.num_non_exiting_ctas, args.fc1_work_counter, args.fc2_work_counter, args.route_count, args.top_k, args.local_expert_offset, args.num_experts, args.fc1_initial_work, args.fc2_initial_work);
}

struct Args_device_01 {
  CUtensorMap A;
  uint8_t* B;
  CUtensorMap SFA;
  uint8_t* SFB;
  CUtensorMap C;
  uint8_t* SFC;
  int* route_map;
  int* tile_expert;
  int* tile_mn_limit;
  int* num_non_exiting_ctas;
  int* work_counter;
  float* scale_c;
  float* scale_gate;
  float* clamp_limit;
  float* act_alpha;
  float* act_beta;
  int M_out;
  int K;
  int grid_m;
  int grid_n;
  int K_tiles;
  uint8_t* SFA_raw;
  uint8_t* C_raw;
};
inline cudaError_t Submit_device_01(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_01*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100,
      args.A, args.B, args.SFA, args.SFB, args.C, args.SFC, args.route_map, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.scale_c, args.scale_gate, args.clamp_limit, args.act_alpha, args.act_beta, args.M_out, args.K, args.grid_m, args.grid_n, args.K_tiles, args.SFA_raw, args.C_raw);
}

struct Args_device_02 {
  CUtensorMap A;
  uint8_t* B;
  CUtensorMap SFA;
  uint8_t* SFB;
  CUtensorMap C;
  uint8_t* SFC;
  int* route_map;
  int* tile_expert;
  int* tile_mn_limit;
  int* num_non_exiting_ctas;
  int* work_counter;
  float* scale_c;
  float* scale_gate;
  float* clamp_limit;
  float* act_alpha;
  float* act_beta;
  int M_out;
  int K;
  int grid_m;
  int grid_n;
  int K_tiles;
  int* route_order;
};
inline cudaError_t Submit_device_02(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_02*>(opaque);
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready_expert_order,
      args.A, args.B, args.SFA, args.SFB, args.C, args.SFC, args.route_map, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.scale_c, args.scale_gate, args.clamp_limit, args.act_alpha, args.act_beta, args.M_out, args.K, args.grid_m, args.grid_n, args.K_tiles, args.route_order);
}

struct Args_device_03 {
  __nv_bfloat16* route_outputs;
  __nv_bfloat16* route_weights;
  int* route_slots;
  __nv_bfloat16* output;
  int num_tokens;
  int route_stride;
  int M;
};
inline cudaError_t Submit_device_03(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_03*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100,
      args.route_outputs, args.route_weights, args.route_slots, args.output, args.num_tokens, args.route_stride, args.M);
}

struct Args_device_04 {
  __nv_bfloat16* route_outputs;
  __nv_bfloat16* route_weights;
  __nv_bfloat16* output;
  int num_tokens;
  int route_stride;
  int M;
};
inline cudaError_t Submit_device_04(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_04*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100,
      args.route_outputs, args.route_weights, args.output, args.num_tokens, args.route_stride, args.M);
}

struct Args_device_05 {
  int* route_experts;
  int* route_map;
  int* tile_expert;
  int* tile_mn_limit;
  int* route_slots;
  int* num_non_exiting_ctas;
  int* fc1_work_counter;
  int* fc2_work_counter;
  int route_count;
  int top_k;
  int local_expert_offset;
  int num_experts;
  int fc1_initial_work;
  int fc2_initial_work;
};
inline cudaError_t Submit_device_05(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_05*>(opaque);
  return SubmitExtendedKernel(config, kernel_e256_route_tile_plan_count_rank_reuse,
      args.route_experts, args.route_map, args.tile_expert, args.tile_mn_limit, args.route_slots, args.num_non_exiting_ctas, args.fc1_work_counter, args.fc2_work_counter, args.route_count, args.top_k, args.local_expert_offset, args.num_experts, args.fc1_initial_work, args.fc2_initial_work);
}

struct Args_device_06 {
  CUtensorMap A;
  uint8_t* B;
  CUtensorMap SFA;
  uint8_t* SFB;
  CUtensorMap C;
  uint8_t* SFC;
  int* route_map;
  int* tile_expert;
  int* tile_mn_limit;
  int* num_non_exiting_ctas;
  int* work_counter;
  float* scale_c;
  float* scale_gate;
  float* clamp_limit;
  float* act_alpha;
  float* act_beta;
  int M_out;
  int K;
  int grid_m;
  int grid_n;
  int K_tiles;
};
inline cudaError_t Submit_device_06(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_06*>(opaque);
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready,
      args.A, args.B, args.SFA, args.SFB, args.C, args.SFC, args.route_map, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.scale_c, args.scale_gate, args.clamp_limit, args.act_alpha, args.act_beta, args.M_out, args.K, args.grid_m, args.grid_n, args.K_tiles);
}

struct Args_device_07 {
  CUtensorMap A;
  CUtensorMap B;
  CUtensorMap SFA;
  CUtensorMap SFB;
  CUtensorMap C_tma;
  __nv_bfloat16* C;
  float* scale_c;
  int* tile_expert;
  int* tile_mn_limit;
  int* num_non_exiting_ctas;
  int* work_counter;
  int M;
  int K;
  int grid_m;
  int grid_n;
  int K_tiles;
};
inline cudaError_t Submit_device_07(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_07*>(opaque);
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe,
      args.A, args.B, args.SFA, args.SFB, args.C_tma, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.M, args.K, args.grid_m, args.grid_n, args.K_tiles);
}

struct Args_device_08 {
  int* route_experts;
  int* route_order;
  int route_count;
};
inline cudaError_t Submit_device_08(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_08*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_t7_direct_route_order,
      args.route_experts, args.route_order, args.route_count);
}

struct Args_device_09 {
  CUtensorMap A;
  CUtensorMap B;
  CUtensorMap SFA;
  uint8_t* SFB;
  CUtensorMap C_tma;
  __nv_bfloat16* C;
  float* scale_c;
  int* tile_expert;
  int* tile_mn_limit;
  int M;
  int K;
  int grid_m;
  int grid_n;
  int K_tiles;
  int* total_tiles;
};
inline cudaError_t Submit_device_09(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_09*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100,
      args.A, args.B, args.SFA, args.SFB, args.C_tma, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.M, args.K, args.grid_m, args.grid_n, args.K_tiles, args.total_tiles);
}

struct Args_device_10 {
  int* route_experts;
  int* route_map;
  int* tile_expert;
  int* tile_mn_limit;
  int* route_slots;
  int* num_non_exiting_ctas;
  int* fc1_work_counter;
  int* fc2_work_counter;
  int route_count;
  int top_k;
  int local_expert_offset;
  int num_experts;
  int fc1_initial_work;
  int fc2_initial_work;
};
inline cudaError_t Submit_device_10(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_10*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_sorted_short_route_pack,
      args.route_experts, args.route_map, args.tile_expert, args.tile_mn_limit, args.route_slots, args.num_non_exiting_ctas, args.fc1_work_counter, args.fc2_work_counter, args.route_count, args.top_k, args.local_expert_offset, args.num_experts, args.fc1_initial_work, args.fc2_initial_work);
}

inline void ForEachLaunch(const Invocation& inv, const Schedule& schedule, LaunchVisitor visitor, void* context) {
  const int64_t required = WorkspaceSize(inv.shape, schedule);
  if (required < 0 || inv.workspace == nullptr || inv.workspace_bytes < static_cast<size_t>(required) || !visitor)
    throw std::runtime_error("invalid clamped E256 invocation");
  switch (inv.shape.num_tokens) {
  case 1: {
    Args_device_06 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_0_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_0_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_0_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_0_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_0_dims, map_0_strides, map_0_box, map_0_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_1_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_1_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_1_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_1_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_1_dims, map_1_strides, map_1_box, map_1_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_2_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_2_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_2_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_2_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_2_dims, map_2_strides, map_2_box, map_2_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 49152u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 55296u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 55552u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 55808u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 56064u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(6);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_3_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_3_strides[] = {1024ull, 4194304ull};
      const uint32_t map_3_box[] = {256u, 128u, 1u};
      const uint32_t map_3_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_3_dims, map_3_strides, map_3_box, map_3_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_4_dims[] = {2048ull, 8ull, 6ull};
      const uint64_t map_4_strides[] = {1024ull, 8192ull};
      const uint32_t map_4_box[] = {256u, 8u, 1u};
      const uint32_t map_4_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_4_dims, map_4_strides, map_4_box, map_4_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_5_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_5_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_5_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_5_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_5_dims, map_5_strides, map_5_box, map_5_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_6_dims[] = {32ull, 32ull, 6ull};
      const uint64_t map_6_strides[] = {32ull, 1024ull};
      const uint32_t map_6_box[] = {32u, 8u, 1u};
      const uint32_t map_6_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 49152u) + 0, map_6_dims, map_6_strides, map_6_box, map_6_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_7_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_7_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_7_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_7_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 56320u) + 0, map_7_dims, map_7_strides, map_7_box, map_7_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 56320u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 55552u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 55808u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 449536u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(6);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 56320u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(1);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 6, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 6, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 1, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 2: {
    Args_device_06 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_8_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_8_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_8_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_8_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_8_dims, map_8_strides, map_8_box, map_8_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_9_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_9_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_9_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_9_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_9_dims, map_9_strides, map_9_box, map_9_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_10_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_10_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_10_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_10_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_10_dims, map_10_strides, map_10_box, map_10_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 98304u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 110592u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 111104u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 111360u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 111616u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(12);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_11_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_11_strides[] = {1024ull, 4194304ull};
      const uint32_t map_11_box[] = {256u, 128u, 1u};
      const uint32_t map_11_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_11_dims, map_11_strides, map_11_box, map_11_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_12_dims[] = {2048ull, 8ull, 12ull};
      const uint64_t map_12_strides[] = {1024ull, 8192ull};
      const uint32_t map_12_box[] = {256u, 8u, 1u};
      const uint32_t map_12_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_12_dims, map_12_strides, map_12_box, map_12_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_13_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_13_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_13_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_13_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_13_dims, map_13_strides, map_13_box, map_13_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_14_dims[] = {32ull, 32ull, 12ull};
      const uint64_t map_14_strides[] = {32ull, 1024ull};
      const uint32_t map_14_box[] = {32u, 8u, 1u};
      const uint32_t map_14_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 98304u) + 0, map_14_dims, map_14_strides, map_14_box, map_14_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_15_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_15_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_15_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_15_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 111872u) + 0, map_15_dims, map_15_strides, map_15_box, map_15_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 111872u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 111104u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 111360u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 898304u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(12);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 111872u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(2);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 12, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 12, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 2, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 3: {
    Args_device_06 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_16_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_16_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_16_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_16_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_16_dims, map_16_strides, map_16_box, map_16_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_17_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_17_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_17_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_17_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_17_dims, map_17_strides, map_17_box, map_17_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_18_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_18_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_18_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_18_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_18_dims, map_18_strides, map_18_box, map_18_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 147456u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 165888u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 166656u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 166912u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 167168u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(18);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_19_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_19_strides[] = {1024ull, 4194304ull};
      const uint32_t map_19_box[] = {256u, 128u, 1u};
      const uint32_t map_19_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_19_dims, map_19_strides, map_19_box, map_19_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_20_dims[] = {2048ull, 8ull, 18ull};
      const uint64_t map_20_strides[] = {1024ull, 8192ull};
      const uint32_t map_20_box[] = {256u, 8u, 1u};
      const uint32_t map_20_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_20_dims, map_20_strides, map_20_box, map_20_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_21_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_21_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_21_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_21_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_21_dims, map_21_strides, map_21_box, map_21_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_22_dims[] = {32ull, 32ull, 18ull};
      const uint64_t map_22_strides[] = {32ull, 1024ull};
      const uint32_t map_22_box[] = {32u, 8u, 1u};
      const uint32_t map_22_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 147456u) + 0, map_22_dims, map_22_strides, map_22_box, map_22_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_23_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_23_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_23_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_23_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 167424u) + 0, map_23_dims, map_23_strides, map_23_box, map_23_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 167424u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 166656u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 166912u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 1347072u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(18);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 167424u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(3);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 18, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 18, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 3, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 4: {
    Args_device_06 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_24_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_24_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_24_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_24_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_24_dims, map_24_strides, map_24_box, map_24_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_25_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_25_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_25_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_25_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_25_dims, map_25_strides, map_25_box, map_25_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_26_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_26_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_26_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_26_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_26_dims, map_26_strides, map_26_box, map_26_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 196608u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 221184u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 221952u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 222208u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 222464u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(24);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_27_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_27_strides[] = {1024ull, 4194304ull};
      const uint32_t map_27_box[] = {256u, 128u, 1u};
      const uint32_t map_27_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_27_dims, map_27_strides, map_27_box, map_27_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_28_dims[] = {2048ull, 8ull, 24ull};
      const uint64_t map_28_strides[] = {1024ull, 8192ull};
      const uint32_t map_28_box[] = {256u, 8u, 1u};
      const uint32_t map_28_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_28_dims, map_28_strides, map_28_box, map_28_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_29_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_29_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_29_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_29_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_29_dims, map_29_strides, map_29_box, map_29_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_30_dims[] = {32ull, 32ull, 24ull};
      const uint64_t map_30_strides[] = {32ull, 1024ull};
      const uint32_t map_30_box[] = {32u, 8u, 1u};
      const uint32_t map_30_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 196608u) + 0, map_30_dims, map_30_strides, map_30_box, map_30_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_31_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_31_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_31_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_31_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 222720u) + 0, map_31_dims, map_31_strides, map_31_box, map_31_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 222720u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 221952u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 222208u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 1795584u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(24);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 222720u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(4);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 24, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 24, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 4, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 5: {
    Args_device_06 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_32_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_32_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_32_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_32_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_32_dims, map_32_strides, map_32_box, map_32_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_33_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_33_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_33_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_33_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_33_dims, map_33_strides, map_33_box, map_33_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_34_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_34_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_34_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_34_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_34_dims, map_34_strides, map_34_box, map_34_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 245760u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 276480u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 277504u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 277760u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 278016u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(30);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_35_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_35_strides[] = {1024ull, 4194304ull};
      const uint32_t map_35_box[] = {256u, 128u, 1u};
      const uint32_t map_35_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_35_dims, map_35_strides, map_35_box, map_35_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_36_dims[] = {2048ull, 8ull, 30ull};
      const uint64_t map_36_strides[] = {1024ull, 8192ull};
      const uint32_t map_36_box[] = {256u, 8u, 1u};
      const uint32_t map_36_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_36_dims, map_36_strides, map_36_box, map_36_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_37_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_37_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_37_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_37_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_37_dims, map_37_strides, map_37_box, map_37_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_38_dims[] = {32ull, 32ull, 30ull};
      const uint64_t map_38_strides[] = {32ull, 1024ull};
      const uint32_t map_38_box[] = {32u, 8u, 1u};
      const uint32_t map_38_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 245760u) + 0, map_38_dims, map_38_strides, map_38_box, map_38_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_39_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_39_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_39_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_39_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 278272u) + 0, map_39_dims, map_39_strides, map_39_box, map_39_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 278272u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 277504u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 277760u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2244352u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(30);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 278272u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(5);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 30, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 30, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 5, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 6: {
    Args_device_06 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_40_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_40_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_40_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_40_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_40_dims, map_40_strides, map_40_box, map_40_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_41_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_41_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_41_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_41_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_41_dims, map_41_strides, map_41_box, map_41_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_42_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_42_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_42_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_42_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_42_dims, map_42_strides, map_42_box, map_42_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 294912u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 331776u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 333056u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 333312u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 333568u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(36);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_43_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_43_strides[] = {1024ull, 4194304ull};
      const uint32_t map_43_box[] = {256u, 128u, 1u};
      const uint32_t map_43_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_43_dims, map_43_strides, map_43_box, map_43_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_44_dims[] = {2048ull, 8ull, 36ull};
      const uint64_t map_44_strides[] = {1024ull, 8192ull};
      const uint32_t map_44_box[] = {256u, 8u, 1u};
      const uint32_t map_44_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_44_dims, map_44_strides, map_44_box, map_44_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_45_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_45_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_45_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_45_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_45_dims, map_45_strides, map_45_box, map_45_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_46_dims[] = {32ull, 32ull, 36ull};
      const uint64_t map_46_strides[] = {32ull, 1024ull};
      const uint32_t map_46_box[] = {32u, 8u, 1u};
      const uint32_t map_46_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 294912u) + 0, map_46_dims, map_46_strides, map_46_box, map_46_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_47_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_47_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_47_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_47_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 333824u) + 0, map_47_dims, map_47_strides, map_47_box, map_47_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 333824u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 333056u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 333312u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2693120u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(36);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 333824u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(6);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 36, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 36, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 6, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 7: {
    Args_device_08 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_order = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.route_count = static_cast<int>(42);
    Args_device_02 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_48_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_48_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_48_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_48_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_48_dims, map_48_strides, map_48_box, map_48_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_49_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_49_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_49_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_49_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_49_dims, map_49_strides, map_49_box, map_49_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_50_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_50_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_50_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_50_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 256u) + 0, map_50_dims, map_50_strides, map_50_box, map_50_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 344320u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 387328u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 388864u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 389120u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 389376u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(42);
    args_1.K_tiles = static_cast<int>(8);
    args_1.route_order = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    Args_device_07 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_51_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_51_strides[] = {1024ull, 4194304ull};
      const uint32_t map_51_box[] = {256u, 128u, 1u};
      const uint32_t map_51_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_51_dims, map_51_strides, map_51_box, map_51_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_52_dims[] = {2048ull, 8ull, 42ull};
      const uint64_t map_52_strides[] = {1024ull, 8192ull};
      const uint32_t map_52_box[] = {256u, 8u, 1u};
      const uint32_t map_52_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 256u) + 0, map_52_dims, map_52_strides, map_52_box, map_52_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_53_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_53_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_53_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_53_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_53_dims, map_53_strides, map_53_box, map_53_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_54_dims[] = {32ull, 32ull, 42ull};
      const uint64_t map_54_strides[] = {32ull, 1024ull};
      const uint32_t map_54_box[] = {32u, 8u, 1u};
      const uint32_t map_54_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 344320u) + 0, map_54_dims, map_54_strides, map_54_box, map_54_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFB) == sizeof(encoded));
      std::memcpy(&args_2.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_55_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_55_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_55_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_55_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 389632u) + 0, map_55_dims, map_55_strides, map_55_box, map_55_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 389632u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 388864u));
    args_2.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 389120u));
    args_2.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3142144u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(42);
    args_2.K_tiles = static_cast<int>(4);
    Args_device_04 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 389632u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(7);
    args_3.route_stride = static_cast<int>(32768);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_t7_direct_route_order", dim3(1, 1, 1), dim3(64, 1, 1), dim3(1, 1, 1),
      128u, true, false, false, false,
      &Submit_device_08, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready_expert_order", dim3(32, 42, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_02, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 42, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_2, -1};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 7, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 8: {
    Args_device_08 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_order = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.route_count = static_cast<int>(48);
    Args_device_02 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_56_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_56_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_56_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_56_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_56_dims, map_56_strides, map_56_box, map_56_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_57_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_57_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_57_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_57_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_57_dims, map_57_strides, map_57_box, map_57_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_58_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_58_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_58_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_58_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 256u) + 0, map_58_dims, map_58_strides, map_58_box, map_58_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 393472u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 442624u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 444160u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 444416u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 444672u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(48);
    args_1.K_tiles = static_cast<int>(8);
    args_1.route_order = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    Args_device_07 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_59_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_59_strides[] = {1024ull, 4194304ull};
      const uint32_t map_59_box[] = {256u, 128u, 1u};
      const uint32_t map_59_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_59_dims, map_59_strides, map_59_box, map_59_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_60_dims[] = {2048ull, 8ull, 48ull};
      const uint64_t map_60_strides[] = {1024ull, 8192ull};
      const uint32_t map_60_box[] = {256u, 8u, 1u};
      const uint32_t map_60_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 256u) + 0, map_60_dims, map_60_strides, map_60_box, map_60_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_61_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_61_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_61_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_61_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_61_dims, map_61_strides, map_61_box, map_61_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_62_dims[] = {32ull, 32ull, 48ull};
      const uint64_t map_62_strides[] = {32ull, 1024ull};
      const uint32_t map_62_box[] = {32u, 8u, 1u};
      const uint32_t map_62_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 393472u) + 0, map_62_dims, map_62_strides, map_62_box, map_62_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFB) == sizeof(encoded));
      std::memcpy(&args_2.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_63_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_63_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_63_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_63_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 444928u) + 0, map_63_dims, map_63_strides, map_63_box, map_63_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 444928u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 444160u));
    args_2.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 444416u));
    args_2.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3590656u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(48);
    args_2.K_tiles = static_cast<int>(4);
    Args_device_04 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 444928u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(8);
    args_3.route_stride = static_cast<int>(32768);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_t7_direct_route_order", dim3(1, 1, 1), dim3(64, 1, 1), dim3(1, 1, 1),
      128u, true, false, false, false,
      &Submit_device_08, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready_expert_order", dim3(32, 48, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false, false,
      &Submit_device_02, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_direct_stg_probe", dim3(32, 48, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      159872u, true, false, false, false,
      &Submit_device_07, &args_2, -1};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 8, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_04, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 9: {
    Args_device_10 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 1792u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_0.route_count = static_cast<int>(54);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_64_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_64_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_64_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_64_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_64_dims, map_64_strides, map_64_box, map_64_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_65_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_65_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_65_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_65_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_65_dims, map_65_strides, map_65_box, map_65_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_66_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_66_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_66_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_66_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 3328u) + 0, map_66_dims, map_66_strides, map_66_box, map_66_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 445696u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 1792u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(54);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_67_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_67_strides[] = {1024ull, 4194304ull};
      const uint32_t map_67_box[] = {256u, 128u, 1u};
      const uint32_t map_67_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_67_dims, map_67_strides, map_67_box, map_67_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_68_dims[] = {2048ull, 8ull, 54ull};
      const uint64_t map_68_strides[] = {1024ull, 8192ull};
      const uint32_t map_68_box[] = {256u, 8u, 1u};
      const uint32_t map_68_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 3328u) + 0, map_68_dims, map_68_strides, map_68_box, map_68_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_69_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_69_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_69_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_69_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_69_dims, map_69_strides, map_69_box, map_69_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 445696u));
    {
      CUtensorMap encoded{};
      const uint64_t map_70_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_70_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_70_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_70_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 500992u) + 0, map_70_dims, map_70_strides, map_70_box, map_70_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 500992u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 1792u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(54);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 500992u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(9);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_sorted_short_route_pack", dim3(1, 1, 1), dim3(64, 1, 1), dim3(1, 1, 1),
      128u, true, false, false, false,
      &Submit_device_10, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 54, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 54, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 9, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 10: {
    Args_device_10 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_0.route_count = static_cast<int>(60);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_71_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_71_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_71_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_71_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_71_dims, map_71_strides, map_71_box, map_71_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_72_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_72_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_72_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_72_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_72_dims, map_72_strides, map_72_box, map_72_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_73_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_73_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_73_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_73_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 3584u) + 0, map_73_dims, map_73_strides, map_73_box, map_73_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 495104u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(60);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_74_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_74_strides[] = {1024ull, 4194304ull};
      const uint32_t map_74_box[] = {256u, 128u, 1u};
      const uint32_t map_74_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_74_dims, map_74_strides, map_74_box, map_74_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_75_dims[] = {2048ull, 8ull, 60ull};
      const uint64_t map_75_strides[] = {1024ull, 8192ull};
      const uint32_t map_75_box[] = {256u, 8u, 1u};
      const uint32_t map_75_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 3584u) + 0, map_75_dims, map_75_strides, map_75_box, map_75_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_76_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_76_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_76_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_76_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_76_dims, map_76_strides, map_76_box, map_76_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 495104u));
    {
      CUtensorMap encoded{};
      const uint64_t map_77_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_77_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_77_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_77_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 556544u) + 0, map_77_dims, map_77_strides, map_77_box, map_77_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 556544u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(60);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 556544u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(10);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_sorted_short_route_pack", dim3(1, 1, 1), dim3(64, 1, 1), dim3(1, 1, 1),
      128u, true, false, false, false,
      &Submit_device_10, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 60, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 60, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 10, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 11: {
    Args_device_05 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.route_count = static_cast<int>(66);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_78_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_78_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_78_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_78_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_78_dims, map_78_strides, map_78_box, map_78_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_79_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_79_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_79_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_79_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_79_dims, map_79_strides, map_79_box, map_79_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_80_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_80_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_80_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_80_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 4608u) + 0, map_80_dims, map_80_strides, map_80_box, map_80_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 545280u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(66);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_81_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_81_strides[] = {1024ull, 4194304ull};
      const uint32_t map_81_box[] = {256u, 128u, 1u};
      const uint32_t map_81_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_81_dims, map_81_strides, map_81_box, map_81_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_82_dims[] = {2048ull, 8ull, 66ull};
      const uint64_t map_82_strides[] = {1024ull, 8192ull};
      const uint32_t map_82_box[] = {256u, 8u, 1u};
      const uint32_t map_82_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 4608u) + 0, map_82_dims, map_82_strides, map_82_box, map_82_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_83_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_83_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_83_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_83_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_83_dims, map_83_strides, map_83_box, map_83_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 545280u));
    {
      CUtensorMap encoded{};
      const uint64_t map_84_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_84_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_84_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_84_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 612864u) + 0, map_84_dims, map_84_strides, map_84_box, map_84_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 612864u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(66);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 612864u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(11);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 66, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 66, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 11, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 12: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.route_count = static_cast<int>(72);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_85_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_85_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_85_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_85_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_85_dims, map_85_strides, map_85_box, map_85_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_86_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_86_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_86_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_86_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_86_dims, map_86_strides, map_86_box, map_86_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_87_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_87_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_87_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_87_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 4608u) + 0, map_87_dims, map_87_strides, map_87_box, map_87_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 594432u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(72);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_88_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_88_strides[] = {1024ull, 4194304ull};
      const uint32_t map_88_box[] = {256u, 128u, 1u};
      const uint32_t map_88_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_88_dims, map_88_strides, map_88_box, map_88_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_89_dims[] = {2048ull, 8ull, 72ull};
      const uint64_t map_89_strides[] = {1024ull, 8192ull};
      const uint32_t map_89_box[] = {256u, 8u, 1u};
      const uint32_t map_89_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 4608u) + 0, map_89_dims, map_89_strides, map_89_box, map_89_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_90_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_90_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_90_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_90_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_90_dims, map_90_strides, map_90_box, map_90_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 594432u));
    {
      CUtensorMap encoded{};
      const uint64_t map_91_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_91_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_91_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_91_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 668160u) + 0, map_91_dims, map_91_strides, map_91_box, map_91_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 668160u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(72);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 668160u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(12);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 72, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 72, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 12, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 13: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.route_count = static_cast<int>(78);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_92_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_92_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_92_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_92_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_92_dims, map_92_strides, map_92_box, map_92_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_93_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_93_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_93_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_93_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_93_dims, map_93_strides, map_93_box, map_93_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_94_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_94_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_94_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_94_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 4864u) + 0, map_94_dims, map_94_strides, map_94_box, map_94_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 643840u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(78);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_95_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_95_strides[] = {1024ull, 4194304ull};
      const uint32_t map_95_box[] = {256u, 128u, 1u};
      const uint32_t map_95_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_95_dims, map_95_strides, map_95_box, map_95_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_96_dims[] = {2048ull, 8ull, 78ull};
      const uint64_t map_96_strides[] = {1024ull, 8192ull};
      const uint32_t map_96_box[] = {256u, 8u, 1u};
      const uint32_t map_96_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 4864u) + 0, map_96_dims, map_96_strides, map_96_box, map_96_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_97_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_97_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_97_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_97_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_97_dims, map_97_strides, map_97_box, map_97_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 643840u));
    {
      CUtensorMap encoded{};
      const uint64_t map_98_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_98_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_98_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_98_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 723712u) + 0, map_98_dims, map_98_strides, map_98_box, map_98_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 723712u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(78);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 723712u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(13);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 78, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 78, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 13, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 14: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.route_count = static_cast<int>(84);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_99_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_99_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_99_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_99_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_99_dims, map_99_strides, map_99_box, map_99_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_100_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_100_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_100_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_100_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_100_dims, map_100_strides, map_100_box, map_100_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_101_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_101_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_101_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_101_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5120u) + 0, map_101_dims, map_101_strides, map_101_box, map_101_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 693248u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(84);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_102_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_102_strides[] = {1024ull, 4194304ull};
      const uint32_t map_102_box[] = {256u, 128u, 1u};
      const uint32_t map_102_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_102_dims, map_102_strides, map_102_box, map_102_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_103_dims[] = {2048ull, 8ull, 84ull};
      const uint64_t map_103_strides[] = {1024ull, 8192ull};
      const uint32_t map_103_box[] = {256u, 8u, 1u};
      const uint32_t map_103_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 5120u) + 0, map_103_dims, map_103_strides, map_103_box, map_103_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_104_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_104_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_104_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_104_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_104_dims, map_104_strides, map_104_box, map_104_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 693248u));
    {
      CUtensorMap encoded{};
      const uint64_t map_105_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_105_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_105_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_105_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 779264u) + 0, map_105_dims, map_105_strides, map_105_box, map_105_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 779264u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(84);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 779264u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(14);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 84, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 84, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 14, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 15: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.route_count = static_cast<int>(90);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_106_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_106_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_106_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_106_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_106_dims, map_106_strides, map_106_box, map_106_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_107_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_107_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_107_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_107_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_107_dims, map_107_strides, map_107_box, map_107_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_108_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_108_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_108_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_108_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5376u) + 0, map_108_dims, map_108_strides, map_108_box, map_108_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 742656u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(90);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_109_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_109_strides[] = {1024ull, 4194304ull};
      const uint32_t map_109_box[] = {256u, 128u, 1u};
      const uint32_t map_109_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_109_dims, map_109_strides, map_109_box, map_109_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_110_dims[] = {2048ull, 8ull, 90ull};
      const uint64_t map_110_strides[] = {1024ull, 8192ull};
      const uint32_t map_110_box[] = {256u, 8u, 1u};
      const uint32_t map_110_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 5376u) + 0, map_110_dims, map_110_strides, map_110_box, map_110_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_111_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_111_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_111_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_111_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_111_dims, map_111_strides, map_111_box, map_111_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 742656u));
    {
      CUtensorMap encoded{};
      const uint64_t map_112_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_112_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_112_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_112_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 834816u) + 0, map_112_dims, map_112_strides, map_112_box, map_112_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 834816u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(90);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 834816u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(15);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 90, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 90, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 15, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 16: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.route_count = static_cast<int>(96);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_113_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_113_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_113_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_113_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_113_dims, map_113_strides, map_113_box, map_113_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_114_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_114_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_114_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_114_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_114_dims, map_114_strides, map_114_box, map_114_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_115_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_115_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_115_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_115_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5376u) + 0, map_115_dims, map_115_strides, map_115_box, map_115_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 791808u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(96);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_116_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_116_strides[] = {1024ull, 4194304ull};
      const uint32_t map_116_box[] = {256u, 128u, 1u};
      const uint32_t map_116_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_116_dims, map_116_strides, map_116_box, map_116_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_117_dims[] = {2048ull, 8ull, 96ull};
      const uint64_t map_117_strides[] = {1024ull, 8192ull};
      const uint32_t map_117_box[] = {256u, 8u, 1u};
      const uint32_t map_117_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 5376u) + 0, map_117_dims, map_117_strides, map_117_box, map_117_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_118_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_118_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_118_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_118_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_118_dims, map_118_strides, map_118_box, map_118_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 791808u));
    {
      CUtensorMap encoded{};
      const uint64_t map_119_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_119_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_119_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_119_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 890112u) + 0, map_119_dims, map_119_strides, map_119_box, map_119_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 890112u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(96);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 890112u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(16);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 96, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 96, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 16, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 17: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.route_count = static_cast<int>(102);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_120_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_120_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_120_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_120_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_120_dims, map_120_strides, map_120_box, map_120_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_121_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_121_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_121_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_121_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_121_dims, map_121_strides, map_121_box, map_121_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_122_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_122_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_122_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_122_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5632u) + 0, map_122_dims, map_122_strides, map_122_box, map_122_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 841216u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(102);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_123_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_123_strides[] = {1024ull, 4194304ull};
      const uint32_t map_123_box[] = {256u, 128u, 1u};
      const uint32_t map_123_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_123_dims, map_123_strides, map_123_box, map_123_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_124_dims[] = {2048ull, 8ull, 102ull};
      const uint64_t map_124_strides[] = {1024ull, 8192ull};
      const uint32_t map_124_box[] = {256u, 8u, 1u};
      const uint32_t map_124_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 5632u) + 0, map_124_dims, map_124_strides, map_124_box, map_124_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_125_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_125_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_125_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_125_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_125_dims, map_125_strides, map_125_box, map_125_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 841216u));
    {
      CUtensorMap encoded{};
      const uint64_t map_126_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_126_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_126_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_126_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 945664u) + 0, map_126_dims, map_126_strides, map_126_box, map_126_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 945664u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(102);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 945664u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(17);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 102, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 102, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 17, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 18: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_0.route_count = static_cast<int>(108);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_127_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_127_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_127_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_127_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_127_dims, map_127_strides, map_127_box, map_127_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_128_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_128_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_128_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_128_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_128_dims, map_128_strides, map_128_box, map_128_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_129_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_129_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_129_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_129_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5888u) + 0, map_129_dims, map_129_strides, map_129_box, map_129_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 890624u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(108);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_130_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_130_strides[] = {1024ull, 4194304ull};
      const uint32_t map_130_box[] = {256u, 128u, 1u};
      const uint32_t map_130_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_130_dims, map_130_strides, map_130_box, map_130_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_131_dims[] = {2048ull, 8ull, 108ull};
      const uint64_t map_131_strides[] = {1024ull, 8192ull};
      const uint32_t map_131_box[] = {256u, 8u, 1u};
      const uint32_t map_131_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 5888u) + 0, map_131_dims, map_131_strides, map_131_box, map_131_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_132_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_132_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_132_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_132_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_132_dims, map_132_strides, map_132_box, map_132_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 890624u));
    {
      CUtensorMap encoded{};
      const uint64_t map_133_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_133_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_133_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_133_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1001216u) + 0, map_133_dims, map_133_strides, map_133_box, map_133_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1001216u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(108);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1001216u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(18);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 108, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 108, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 18, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 19: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_0.route_count = static_cast<int>(114);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_134_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_134_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_134_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_134_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_134_dims, map_134_strides, map_134_box, map_134_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_135_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_135_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_135_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_135_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_135_dims, map_135_strides, map_135_box, map_135_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_136_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_136_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_136_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_136_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 6144u) + 0, map_136_dims, map_136_strides, map_136_box, map_136_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 940032u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(114);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_137_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_137_strides[] = {1024ull, 4194304ull};
      const uint32_t map_137_box[] = {256u, 128u, 1u};
      const uint32_t map_137_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_137_dims, map_137_strides, map_137_box, map_137_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_138_dims[] = {2048ull, 8ull, 114ull};
      const uint64_t map_138_strides[] = {1024ull, 8192ull};
      const uint32_t map_138_box[] = {256u, 8u, 1u};
      const uint32_t map_138_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 6144u) + 0, map_138_dims, map_138_strides, map_138_box, map_138_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_139_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_139_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_139_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_139_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_139_dims, map_139_strides, map_139_box, map_139_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 940032u));
    {
      CUtensorMap encoded{};
      const uint64_t map_140_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_140_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_140_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_140_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1056768u) + 0, map_140_dims, map_140_strides, map_140_box, map_140_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1056768u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(114);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1056768u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(19);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 114, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 114, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 19, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 20: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_0.route_count = static_cast<int>(120);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_141_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_141_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_141_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_141_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_141_dims, map_141_strides, map_141_box, map_141_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_142_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_142_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_142_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_142_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_142_dims, map_142_strides, map_142_box, map_142_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_143_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_143_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_143_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_143_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 6144u) + 0, map_143_dims, map_143_strides, map_143_box, map_143_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 989184u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(120);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_144_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_144_strides[] = {1024ull, 4194304ull};
      const uint32_t map_144_box[] = {256u, 128u, 1u};
      const uint32_t map_144_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_144_dims, map_144_strides, map_144_box, map_144_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_145_dims[] = {2048ull, 8ull, 120ull};
      const uint64_t map_145_strides[] = {1024ull, 8192ull};
      const uint32_t map_145_box[] = {256u, 8u, 1u};
      const uint32_t map_145_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 6144u) + 0, map_145_dims, map_145_strides, map_145_box, map_145_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_146_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_146_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_146_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_146_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_146_dims, map_146_strides, map_146_box, map_146_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 989184u));
    {
      CUtensorMap encoded{};
      const uint64_t map_147_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_147_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_147_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_147_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1112064u) + 0, map_147_dims, map_147_strides, map_147_box, map_147_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1112064u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(120);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1112064u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(20);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 120, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 120, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 20, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 21: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.route_count = static_cast<int>(126);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_148_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_148_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_148_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_148_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_148_dims, map_148_strides, map_148_box, map_148_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_149_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_149_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_149_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_149_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_149_dims, map_149_strides, map_149_box, map_149_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_150_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_150_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_150_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_150_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 6400u) + 0, map_150_dims, map_150_strides, map_150_box, map_150_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1038592u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(126);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_151_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_151_strides[] = {1024ull, 4194304ull};
      const uint32_t map_151_box[] = {256u, 128u, 1u};
      const uint32_t map_151_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_151_dims, map_151_strides, map_151_box, map_151_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_152_dims[] = {2048ull, 8ull, 126ull};
      const uint64_t map_152_strides[] = {1024ull, 8192ull};
      const uint32_t map_152_box[] = {256u, 8u, 1u};
      const uint32_t map_152_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 6400u) + 0, map_152_dims, map_152_strides, map_152_box, map_152_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_153_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_153_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_153_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_153_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_153_dims, map_153_strides, map_153_box, map_153_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1038592u));
    {
      CUtensorMap encoded{};
      const uint64_t map_154_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_154_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_154_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_154_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1167616u) + 0, map_154_dims, map_154_strides, map_154_box, map_154_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1167616u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(126);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1167616u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(21);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 126, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 126, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 21, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 22: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_0.route_count = static_cast<int>(132);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_155_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_155_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_155_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_155_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_155_dims, map_155_strides, map_155_box, map_155_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_156_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_156_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_156_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_156_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_156_dims, map_156_strides, map_156_box, map_156_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_157_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_157_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_157_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_157_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7424u) + 0, map_157_dims, map_157_strides, map_157_box, map_157_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1088768u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(132);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_158_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_158_strides[] = {1024ull, 4194304ull};
      const uint32_t map_158_box[] = {256u, 128u, 1u};
      const uint32_t map_158_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_158_dims, map_158_strides, map_158_box, map_158_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_159_dims[] = {2048ull, 8ull, 132ull};
      const uint64_t map_159_strides[] = {1024ull, 8192ull};
      const uint32_t map_159_box[] = {256u, 8u, 1u};
      const uint32_t map_159_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 7424u) + 0, map_159_dims, map_159_strides, map_159_box, map_159_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_160_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_160_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_160_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_160_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_160_dims, map_160_strides, map_160_box, map_160_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1088768u));
    {
      CUtensorMap encoded{};
      const uint64_t map_161_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_161_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_161_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_161_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1223936u) + 0, map_161_dims, map_161_strides, map_161_box, map_161_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1223936u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(132);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1223936u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(22);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 132, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 132, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 22, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 23: {
    Args_device_05 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_0.route_count = static_cast<int>(138);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_162_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_162_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_162_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_162_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_162_dims, map_162_strides, map_162_box, map_162_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_163_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_163_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_163_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_163_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_163_dims, map_163_strides, map_163_box, map_163_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_164_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_164_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_164_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_164_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7680u) + 0, map_164_dims, map_164_strides, map_164_box, map_164_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1138176u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(138);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_165_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_165_strides[] = {1024ull, 4194304ull};
      const uint32_t map_165_box[] = {256u, 128u, 1u};
      const uint32_t map_165_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_165_dims, map_165_strides, map_165_box, map_165_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_166_dims[] = {2048ull, 8ull, 138ull};
      const uint64_t map_166_strides[] = {1024ull, 8192ull};
      const uint32_t map_166_box[] = {256u, 8u, 1u};
      const uint32_t map_166_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 7680u) + 0, map_166_dims, map_166_strides, map_166_box, map_166_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_167_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_167_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_167_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_167_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_167_dims, map_167_strides, map_167_box, map_167_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1138176u));
    {
      CUtensorMap encoded{};
      const uint64_t map_168_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_168_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_168_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_168_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1279488u) + 0, map_168_dims, map_168_strides, map_168_box, map_168_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1279488u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(138);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1279488u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(23);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 138, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 138, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 23, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 24: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_0.route_count = static_cast<int>(144);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_169_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_169_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_169_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_169_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_169_dims, map_169_strides, map_169_box, map_169_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_170_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_170_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_170_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_170_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_170_dims, map_170_strides, map_170_box, map_170_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_171_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_171_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_171_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_171_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7680u) + 0, map_171_dims, map_171_strides, map_171_box, map_171_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1187328u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(144);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_172_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_172_strides[] = {1024ull, 4194304ull};
      const uint32_t map_172_box[] = {256u, 128u, 1u};
      const uint32_t map_172_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_172_dims, map_172_strides, map_172_box, map_172_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_173_dims[] = {2048ull, 8ull, 144ull};
      const uint64_t map_173_strides[] = {1024ull, 8192ull};
      const uint32_t map_173_box[] = {256u, 8u, 1u};
      const uint32_t map_173_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 7680u) + 0, map_173_dims, map_173_strides, map_173_box, map_173_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_174_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_174_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_174_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_174_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_174_dims, map_174_strides, map_174_box, map_174_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1187328u));
    {
      CUtensorMap encoded{};
      const uint64_t map_175_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_175_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_175_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_175_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1334784u) + 0, map_175_dims, map_175_strides, map_175_box, map_175_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1334784u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(144);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1334784u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(24);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 144, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 144, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 24, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 25: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_0.route_count = static_cast<int>(150);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_176_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_176_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_176_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_176_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_176_dims, map_176_strides, map_176_box, map_176_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_177_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_177_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_177_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_177_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_177_dims, map_177_strides, map_177_box, map_177_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_178_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_178_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_178_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_178_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7936u) + 0, map_178_dims, map_178_strides, map_178_box, map_178_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1236736u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(150);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_179_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_179_strides[] = {1024ull, 4194304ull};
      const uint32_t map_179_box[] = {256u, 128u, 1u};
      const uint32_t map_179_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_179_dims, map_179_strides, map_179_box, map_179_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_180_dims[] = {2048ull, 8ull, 150ull};
      const uint64_t map_180_strides[] = {1024ull, 8192ull};
      const uint32_t map_180_box[] = {256u, 8u, 1u};
      const uint32_t map_180_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 7936u) + 0, map_180_dims, map_180_strides, map_180_box, map_180_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_181_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_181_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_181_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_181_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_181_dims, map_181_strides, map_181_box, map_181_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1236736u));
    {
      CUtensorMap encoded{};
      const uint64_t map_182_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_182_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_182_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_182_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1390336u) + 0, map_182_dims, map_182_strides, map_182_box, map_182_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1390336u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(150);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1390336u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(25);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 150, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 150, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 25, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 26: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_0.route_count = static_cast<int>(156);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_183_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_183_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_183_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_183_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_183_dims, map_183_strides, map_183_box, map_183_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_184_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_184_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_184_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_184_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_184_dims, map_184_strides, map_184_box, map_184_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_185_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_185_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_185_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_185_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8192u) + 0, map_185_dims, map_185_strides, map_185_box, map_185_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1286144u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(156);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_186_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_186_strides[] = {1024ull, 4194304ull};
      const uint32_t map_186_box[] = {256u, 128u, 1u};
      const uint32_t map_186_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_186_dims, map_186_strides, map_186_box, map_186_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_187_dims[] = {2048ull, 8ull, 156ull};
      const uint64_t map_187_strides[] = {1024ull, 8192ull};
      const uint32_t map_187_box[] = {256u, 8u, 1u};
      const uint32_t map_187_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 8192u) + 0, map_187_dims, map_187_strides, map_187_box, map_187_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_188_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_188_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_188_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_188_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_188_dims, map_188_strides, map_188_box, map_188_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1286144u));
    {
      CUtensorMap encoded{};
      const uint64_t map_189_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_189_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_189_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_189_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1445888u) + 0, map_189_dims, map_189_strides, map_189_box, map_189_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1445888u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(156);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1445888u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(26);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 156, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 156, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 26, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 27: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    args_0.route_count = static_cast<int>(162);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_190_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_190_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_190_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_190_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_190_dims, map_190_strides, map_190_box, map_190_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_191_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_191_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_191_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_191_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_191_dims, map_191_strides, map_191_box, map_191_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_192_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_192_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_192_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_192_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8448u) + 0, map_192_dims, map_192_strides, map_192_box, map_192_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1335552u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(162);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_193_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_193_strides[] = {1024ull, 4194304ull};
      const uint32_t map_193_box[] = {256u, 128u, 1u};
      const uint32_t map_193_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_193_dims, map_193_strides, map_193_box, map_193_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_194_dims[] = {2048ull, 8ull, 162ull};
      const uint64_t map_194_strides[] = {1024ull, 8192ull};
      const uint32_t map_194_box[] = {256u, 8u, 1u};
      const uint32_t map_194_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 8448u) + 0, map_194_dims, map_194_strides, map_194_box, map_194_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_195_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_195_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_195_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_195_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_195_dims, map_195_strides, map_195_box, map_195_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1335552u));
    {
      CUtensorMap encoded{};
      const uint64_t map_196_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_196_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_196_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_196_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1501440u) + 0, map_196_dims, map_196_strides, map_196_box, map_196_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1501440u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(162);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1501440u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(27);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 162, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 162, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 27, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 28: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    args_0.route_count = static_cast<int>(168);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_197_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_197_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_197_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_197_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_197_dims, map_197_strides, map_197_box, map_197_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_198_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_198_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_198_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_198_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_198_dims, map_198_strides, map_198_box, map_198_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_199_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_199_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_199_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_199_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8448u) + 0, map_199_dims, map_199_strides, map_199_box, map_199_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1384704u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(168);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_200_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_200_strides[] = {1024ull, 4194304ull};
      const uint32_t map_200_box[] = {256u, 128u, 1u};
      const uint32_t map_200_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_200_dims, map_200_strides, map_200_box, map_200_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_201_dims[] = {2048ull, 8ull, 168ull};
      const uint64_t map_201_strides[] = {1024ull, 8192ull};
      const uint32_t map_201_box[] = {256u, 8u, 1u};
      const uint32_t map_201_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 8448u) + 0, map_201_dims, map_201_strides, map_201_box, map_201_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_202_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_202_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_202_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_202_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_202_dims, map_202_strides, map_202_box, map_202_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1384704u));
    {
      CUtensorMap encoded{};
      const uint64_t map_203_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_203_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_203_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_203_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1556736u) + 0, map_203_dims, map_203_strides, map_203_box, map_203_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1556736u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(168);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1556736u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(28);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 168, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 168, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 28, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 29: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_0.route_count = static_cast<int>(174);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_204_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_204_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_204_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_204_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_204_dims, map_204_strides, map_204_box, map_204_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_205_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_205_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_205_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_205_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_205_dims, map_205_strides, map_205_box, map_205_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_206_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_206_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_206_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_206_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8704u) + 0, map_206_dims, map_206_strides, map_206_box, map_206_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1434112u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(174);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 8704u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_207_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_207_strides[] = {1024ull, 4194304ull};
      const uint32_t map_207_box[] = {256u, 128u, 1u};
      const uint32_t map_207_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_207_dims, map_207_strides, map_207_box, map_207_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_208_dims[] = {2048ull, 8ull, 174ull};
      const uint64_t map_208_strides[] = {1024ull, 8192ull};
      const uint32_t map_208_box[] = {256u, 8u, 1u};
      const uint32_t map_208_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 8704u) + 0, map_208_dims, map_208_strides, map_208_box, map_208_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_209_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_209_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_209_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_209_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_209_dims, map_209_strides, map_209_box, map_209_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1434112u));
    {
      CUtensorMap encoded{};
      const uint64_t map_210_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_210_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_210_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_210_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1612288u) + 0, map_210_dims, map_210_strides, map_210_box, map_210_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1612288u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(174);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7936u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1612288u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(29);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 174, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 174, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 29, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 30: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8704u));
    args_0.route_count = static_cast<int>(180);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_211_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_211_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_211_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_211_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_211_dims, map_211_strides, map_211_box, map_211_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_212_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_212_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_212_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_212_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_212_dims, map_212_strides, map_212_box, map_212_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_213_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_213_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_213_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_213_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8960u) + 0, map_213_dims, map_213_strides, map_213_box, map_213_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1483520u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(180);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 8960u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_214_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_214_strides[] = {1024ull, 4194304ull};
      const uint32_t map_214_box[] = {256u, 128u, 1u};
      const uint32_t map_214_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_214_dims, map_214_strides, map_214_box, map_214_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_215_dims[] = {2048ull, 8ull, 180ull};
      const uint64_t map_215_strides[] = {1024ull, 8192ull};
      const uint32_t map_215_box[] = {256u, 8u, 1u};
      const uint32_t map_215_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 8960u) + 0, map_215_dims, map_215_strides, map_215_box, map_215_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_216_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_216_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_216_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_216_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_216_dims, map_216_strides, map_216_box, map_216_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1483520u));
    {
      CUtensorMap encoded{};
      const uint64_t map_217_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_217_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_217_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_217_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1667840u) + 0, map_217_dims, map_217_strides, map_217_box, map_217_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1667840u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(180);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8192u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1667840u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(30);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 180, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 180, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 30, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 31: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8704u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8960u));
    args_0.route_count = static_cast<int>(186);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_218_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_218_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_218_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_218_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_218_dims, map_218_strides, map_218_box, map_218_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_219_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_219_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_219_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_219_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_219_dims, map_219_strides, map_219_box, map_219_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_220_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_220_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_220_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_220_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 9216u) + 0, map_220_dims, map_220_strides, map_220_box, map_220_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1532928u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8704u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(186);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 9216u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_221_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_221_strides[] = {1024ull, 4194304ull};
      const uint32_t map_221_box[] = {256u, 128u, 1u};
      const uint32_t map_221_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_221_dims, map_221_strides, map_221_box, map_221_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_222_dims[] = {2048ull, 8ull, 186ull};
      const uint64_t map_222_strides[] = {1024ull, 8192ull};
      const uint32_t map_222_box[] = {256u, 8u, 1u};
      const uint32_t map_222_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 9216u) + 0, map_222_dims, map_222_strides, map_222_box, map_222_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_223_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_223_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_223_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_223_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_223_dims, map_223_strides, map_223_box, map_223_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1532928u));
    {
      CUtensorMap encoded{};
      const uint64_t map_224_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_224_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_224_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_224_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1723392u) + 0, map_224_dims, map_224_strides, map_224_box, map_224_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1723392u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(186);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1723392u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(31);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 186, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 186, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 31, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 32: {
    Args_device_00 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_0.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_0.fc1_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8704u));
    args_0.fc2_work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8960u));
    args_0.route_count = static_cast<int>(192);
    args_0.top_k = static_cast<int>(6);
    args_0.local_expert_offset = static_cast<int>(0);
    args_0.num_experts = static_cast<int>(256);
    args_0.fc1_initial_work = static_cast<int>(0);
    args_0.fc2_initial_work = static_cast<int>(0);
    Args_device_01 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_225_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_225_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_225_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_225_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_225_dims, map_225_strides, map_225_box, map_225_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_226_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_226_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_226_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_226_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_226_dims, map_226_strides, map_226_box, map_226_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_227_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_227_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_227_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_227_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 9216u) + 0, map_227_dims, map_227_strides, map_227_box, map_227_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C) == sizeof(encoded));
      std::memcpy(&args_1.C, &encoded, sizeof(encoded));
    }
    args_1.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1582080u));
    args_1.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8704u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_1.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_1.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_1.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_1.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_1.M_out = static_cast<int>(2048);
    args_1.K = static_cast<int>(4096);
    args_1.grid_m = static_cast<int>(64);
    args_1.grid_n = static_cast<int>(192);
    args_1.K_tiles = static_cast<int>(8);
    args_1.SFA_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0));
    args_1.C_raw = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 9216u));
    Args_device_09 args_2{};
    {
      CUtensorMap encoded{};
      const uint64_t map_228_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_228_strides[] = {1024ull, 4194304ull};
      const uint32_t map_228_box[] = {256u, 128u, 1u};
      const uint32_t map_228_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_228_dims, map_228_strides, map_228_box, map_228_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.A) == sizeof(encoded));
      std::memcpy(&args_2.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_229_dims[] = {2048ull, 8ull, 192ull};
      const uint64_t map_229_strides[] = {1024ull, 8192ull};
      const uint32_t map_229_box[] = {256u, 8u, 1u};
      const uint32_t map_229_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 9216u) + 0, map_229_dims, map_229_strides, map_229_box, map_229_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.B) == sizeof(encoded));
      std::memcpy(&args_2.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_230_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_230_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_230_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_230_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_230_dims, map_230_strides, map_230_box, map_230_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.SFA) == sizeof(encoded));
      std::memcpy(&args_2.SFA, &encoded, sizeof(encoded));
    }
    args_2.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 1582080u));
    {
      CUtensorMap encoded{};
      const uint64_t map_231_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_231_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_231_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_231_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 1778688u) + 0, map_231_dims, map_231_strides, map_231_box, map_231_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_2.C_tma) == sizeof(encoded));
      std::memcpy(&args_2.C_tma, &encoded, sizeof(encoded));
    }
    args_2.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1778688u));
    args_2.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_2.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_2.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_2.M = static_cast<int>(4096);
    args_2.K = static_cast<int>(2048);
    args_2.grid_m = static_cast<int>(32);
    args_2.grid_n = static_cast<int>(192);
    args_2.K_tiles = static_cast<int>(4);
    args_2.total_tiles = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 8448u));
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1778688u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(32);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_dsv4_e256_route_pack", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false, false,
      &Submit_device_00, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100", dim3(64, 192, 1), dim3(384, 1, 1), dim3(2, 1, 1),
      64128u, true, false, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_packed_fc2_stg_v15_probe_sm100", dim3(32, 192, 1), dim3(416, 1, 1), dim3(2, 1, 1),
      85120u, true, false, false, true,
      &Submit_device_09, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 32, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  default: throw std::runtime_error("unsupported clamped E256 token count");
  }
}
}  // namespace flashinfer::warp_decode::generated::dsv4_clamped_e256
#endif
