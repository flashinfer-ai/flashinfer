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
#include "../cake_warp_decode_tile_n8_fc2.cuh"
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
// The rows whose FC2 the tile-N8 FC2 program kK512Stage5 runs (SelectTileN8Fc2). Their workspace ends with
// kTileN8Fc2ClampedCounterBytes more bytes for its work counter pair, which PrepareWorkspace zeroes.
inline bool TileN8Fc2Row(const Shape& shape) {
  return SelectTileN8Fc2(Target::kSm100a, shape) == TileN8Fc2::kK512Stage5;
}
inline int64_t WorkspaceSize(const Shape& shape, const Schedule&) {
  if (!Supported(shape)) return -1;
  const int64_t counter_bytes = TileN8Fc2Row(shape) ? kTileN8Fc2ClampedCounterBytes : 0;
  switch (shape.num_tokens) {
    case 1: return 449792 + counter_bytes;
    case 2: return 898560 + counter_bytes;
    case 3: return 1347328 + counter_bytes;
    case 4: return 1795840 + counter_bytes;
    case 5: return 2244608 + counter_bytes;
    case 6: return 2693376 + counter_bytes;
    case 7: return 3142144 + counter_bytes;
    case 8: return 3590912 + counter_bytes;
    case 9: return 4039936 + counter_bytes;
    case 10: return 4488704 + counter_bytes;
    case 11: return 4938240 + counter_bytes;
    case 12: return 5386752 + counter_bytes;
    case 13: return 5835520 + counter_bytes;
    case 14: return 6284288 + counter_bytes;
    case 15: return 6733056 + counter_bytes;
    case 16: return 7181568 + counter_bytes;
    case 17: return 7630336 + counter_bytes;
    case 18: return 8079104 + counter_bytes;
    case 19: return 8527872 + counter_bytes;
    case 20: return 8976384 + counter_bytes;
    case 21: return 9425152 + counter_bytes;
    case 22: return 9874688 + counter_bytes;
    case 23: return 10323456 + counter_bytes;
    case 24: return 10771968 + counter_bytes;
    case 25: return 11220736 + counter_bytes;
    case 26: return 11669504 + counter_bytes;
    case 27: return 12118272 + counter_bytes;
    case 28: return 12566784 + counter_bytes;
    case 29: return 13015552 + counter_bytes;
    case 30: return 13464320 + counter_bytes;
    case 31: return 13913088 + counter_bytes;
    case 32: return 14361600 + counter_bytes;
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
      initializers.push_back({387072u, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}});
      initializers.push_back({388608u, {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329}});
      initializers.push_back({388864u, {42}});
      initializers.push_back({389120u, {1344}});
      initializers.push_back({3141888u, {1344}});
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

struct DeviceState { bool ready = false; int32_t sm_count = 0; };
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
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_dsv4_fc2_terminal_recycle_oob_v21_sm100), cudaFuncAttributeMaxDynamicSharedMemorySize, 85120);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready_expert_order), cudaFuncAttributeMaxDynamicSharedMemorySize, 199040);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100), cudaFuncAttributeMaxDynamicSharedMemorySize, 64128);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready), cudaFuncAttributeMaxDynamicSharedMemorySize, 199040);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe), cudaFuncAttributeMaxDynamicSharedMemorySize, 81536);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_dsv4_fc2_terminal_recycle_v21_sm100), cudaFuncAttributeMaxDynamicSharedMemorySize, 85120);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe_loader_bypass_v23), cudaFuncAttributeMaxDynamicSharedMemorySize, 81536);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready), cudaFuncAttributeMaxDynamicSharedMemorySize, 199040);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready_expert_order), cudaFuncAttributeMaxDynamicSharedMemorySize, 199040);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  error = cudaDeviceGetAttribute(&state.sm_count, cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 SM count");
  error = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_cake_warp_decode_4352a5beff13b20b2912), cudaFuncAttributeMaxDynamicSharedMemorySize, 199808);
  if (error != cudaSuccess) return Runtime(error, "clamped E256 dynamic shared memory");
  state.ready = true;
  return ManifestStatus::Success();
}
// The SM count EnsureDeviceReady recorded for the current device; 0 when that device is not ready.
inline int32_t ReadySmCount() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return 0;
  std::lock_guard<std::mutex> lock(DeviceMutex());
  const auto& state = DeviceStates()[device];
  return state.ready ? state.sm_count : 0;
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

struct Args_device_11 {
  CUtensorMap A;
  CUtensorMap tmap_b_rows_1;
  CUtensorMap tmap_b_rows_2;
  CUtensorMap tmap_b_rows_3;
  CUtensorMap tmap_b_rows_4;
  CUtensorMap tmap_b_rows_5;
  CUtensorMap tmap_b_rows_6;
  CUtensorMap tmap_b_rows_7;
  CUtensorMap tmap_b_rows_8;
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
inline cudaError_t Submit_device_11(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_11*>(opaque);
  return SubmitExtendedKernel(config, kernel_dsv4_fc2_terminal_recycle_oob_v21_sm100,
      args.A, args.tmap_b_rows_1, args.tmap_b_rows_2, args.tmap_b_rows_3, args.tmap_b_rows_4, args.tmap_b_rows_5, args.tmap_b_rows_6, args.tmap_b_rows_7, args.tmap_b_rows_8, args.SFA, args.SFB, args.C_tma, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.M, args.K, args.grid_m, args.grid_n, args.K_tiles, args.total_tiles);
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
  return SubmitExtendedKernel(config, kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100,
      args.A, args.B, args.SFA, args.SFB, args.C, args.SFC, args.route_map, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.scale_c, args.scale_gate, args.clamp_limit, args.act_alpha, args.act_beta, args.M_out, args.K, args.grid_m, args.grid_n, args.K_tiles, args.SFA_raw, args.C_raw);
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
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe,
      args.A, args.B, args.SFA, args.SFB, args.C_tma, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.M, args.K, args.grid_m, args.grid_n, args.K_tiles);
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
  return SubmitExtendedKernel(config, kernel_e256_route_tile_plan_count_rank_reuse_early_pdl,
      args.route_experts, args.route_map, args.tile_expert, args.tile_mn_limit, args.route_slots, args.num_non_exiting_ctas, args.fc1_work_counter, args.fc2_work_counter, args.route_count, args.top_k, args.local_expert_offset, args.num_experts, args.fc1_initial_work, args.fc2_initial_work);
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
  return SubmitExtendedKernel(config, kernel_dsv4_fc2_terminal_recycle_v21_sm100,
      args.A, args.B, args.SFA, args.SFB, args.C_tma, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.M, args.K, args.grid_m, args.grid_n, args.K_tiles, args.total_tiles);
}

struct Args_device_12 {
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
inline cudaError_t Submit_device_12(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_12*>(opaque);
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe_loader_bypass_v23,
      args.A, args.B, args.SFA, args.SFB, args.C_tma, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.M, args.K, args.grid_m, args.grid_n, args.K_tiles);
}

struct Args_device_13 {
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
inline cudaError_t Submit_device_13(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_13*>(opaque);
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready,
      args.A, args.B, args.SFA, args.SFB, args.C, args.SFC, args.route_map, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.scale_c, args.scale_gate, args.clamp_limit, args.act_alpha, args.act_beta, args.M_out, args.K, args.grid_m, args.grid_n, args.K_tiles);
}

struct Args_device_14 {
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
inline cudaError_t Submit_device_14(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const Args_device_14*>(opaque);
  return SubmitExtendedKernel(config, kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready_expert_order,
      args.A, args.B, args.SFA, args.SFB, args.C, args.SFC, args.route_map, args.tile_expert, args.tile_mn_limit, args.num_non_exiting_ctas, args.work_counter, args.scale_c, args.scale_gate, args.clamp_limit, args.act_alpha, args.act_beta, args.M_out, args.K, args.grid_m, args.grid_n, args.K_tiles, args.route_order);
}

// The tile-N8 FC2 program kK512Stage5 on a clamped row: the route inventory's FC1 output (B), its scales (SFB)
// and partials (C) of `extent` route tiles at these workspace offsets, its route tile tables, and the work
// counter pair in the region WorkspaceSize appends.
struct TileN8Fc2Regions {
  size_t activations, activation_scales, partials, tile_mn_limit, num_tiles;
  int64_t extent;
};
inline cudaError_t SubmitTileN8Fc2(const cudaLaunchConfig_t* config, const void* opaque) {
  const auto& args = *static_cast<const TileN8Fc2Args*>(opaque);
  return SubmitExtendedKernel(config, kernel_cake_warp_decode_4352a5beff13b20b2912,
      args.A, args.B, args.SFA, args.SFB, args.C, args.scale_c, args.tile_expert, args.tile_mn_limit, args.num_tiles,
      args.work_counter, args.grid_m, args.max_tiles, args.K_tiles);
}
inline void VisitTileN8Fc2(const Invocation& inv, int64_t workspace_bytes, const TileN8Fc2Regions& regions,
                           int* tile_expert, int carveout, LaunchVisitor visitor, void* context) {
  uint8_t* workspace = static_cast<uint8_t*>(inv.workspace);
  TileN8Fc2Args args{};
  if (!EncodeTileN8Fc2(&args, false, inv.gemm2_weights, inv.gemm2_weights_scale, workspace + regions.activations,
                       workspace + regions.activation_scales, workspace + regions.partials, 256, 4096, 2048,
                       regions.extent))
    throw std::runtime_error("clamped E256 tensor map encoding failed");
  args.scale_c = reinterpret_cast<float*>(const_cast<void*>(inv.output2_scale_scalar));
  args.tile_expert = tile_expert;
  args.tile_mn_limit = reinterpret_cast<int*>(workspace + regions.tile_mn_limit);
  args.num_tiles = reinterpret_cast<int*>(workspace + regions.num_tiles);
  args.work_counter = reinterpret_cast<int*>(workspace + workspace_bytes - kTileN8Fc2ClampedCounterBytes);
  KernelLaunch launch{
      "kernel_cake_warp_decode_4352a5beff13b20b2912", dim3(TileN8Fc2Ctas(32, regions.extent, ReadySmCount()), 1, 1),
      dim3(512, 1, 1), dim3(1, 1, 1), 199808u, true, false, false, &SubmitTileN8Fc2, &args, carveout};
  visitor(launch, context);
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
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 56320u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(1);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 6, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{0u, 49152u, 56320u, 55552u, 55808u, 6},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0)), 100, visitor, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 1, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
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
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 111872u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(2);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_mma_u2_ready", dim3(32, 12, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{0u, 98304u, 111872u, 111104u, 111360u, 12},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0)), 100, visitor, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 2, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
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
      199040u, true, false, false,
      &Submit_device_06, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe", dim3(32, 18, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      81536u, true, false, false,
      &Submit_device_07, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 3, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 4: {
    Args_device_13 args_0{};
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
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready", dim3(32, 24, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_13, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe", dim3(32, 24, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      81536u, true, false, false,
      &Submit_device_07, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 4, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 5: {
    Args_device_13 args_0{};
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
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready", dim3(32, 30, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_13, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe", dim3(32, 30, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      81536u, true, false, false,
      &Submit_device_07, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 5, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 6: {
    Args_device_13 args_0{};
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
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready", dim3(32, 36, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_13, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe", dim3(32, 36, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      81536u, true, false, false,
      &Submit_device_07, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 6, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 7: {
    Args_device_13 args_0{};
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x0_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_c3d7x0_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_c3d7x0_box[] = {128u, 128u, 1u, 1u};
      const uint32_t map_c3d7x0_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_c3d7x0_dims, map_c3d7x0_strides, map_c3d7x0_box, map_c3d7x0_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.A) == sizeof(encoded));
      std::memcpy(&args_0.A, &encoded, sizeof(encoded));
    }
    args_0.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x1_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_c3d7x1_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_c3d7x1_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_c3d7x1_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_c3d7x1_dims, map_c3d7x1_strides, map_c3d7x1_box, map_c3d7x1_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.SFA) == sizeof(encoded));
      std::memcpy(&args_0.SFA, &encoded, sizeof(encoded));
    }
    args_0.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x2_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_c3d7x2_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_c3d7x2_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_c3d7x2_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_c3d7x2_dims, map_c3d7x2_strides, map_c3d7x2_box, map_c3d7x2_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_0.C) == sizeof(encoded));
      std::memcpy(&args_0.C, &encoded, sizeof(encoded));
    }
    args_0.SFC = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(inv.workspace) + 344064u));
    args_0.route_map = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 387072u));
    args_0.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 388608u));
    args_0.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 388864u));
    args_0.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 389120u));
    args_0.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_scalar)) + 0));
    args_0.scale_gate = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output1_scale_gate_scalar)) + 0));
    args_0.clamp_limit = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_clamp_limit)) + 0));
    args_0.act_alpha = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_alpha)) + 0));
    args_0.act_beta = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_beta)) + 0));
    args_0.M_out = static_cast<int>(2048);
    args_0.K = static_cast<int>(4096);
    args_0.grid_m = static_cast<int>(32);
    args_0.grid_n = static_cast<int>(42);
    args_0.K_tiles = static_cast<int>(8);
    Args_device_07 args_1{};
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x3_dims[] = {2048ull, 4096ull, 256ull};
      const uint64_t map_c3d7x3_strides[] = {1024ull, 4194304ull};
      const uint32_t map_c3d7x3_box[] = {256u, 128u, 1u};
      const uint32_t map_c3d7x3_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights)) + 0) + 0, map_c3d7x3_dims, map_c3d7x3_strides, map_c3d7x3_box, map_c3d7x3_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x4_dims[] = {2048ull, 8ull, 42ull};
      const uint64_t map_c3d7x4_strides[] = {1024ull, 8192ull};
      const uint32_t map_c3d7x4_box[] = {256u, 8u, 1u};
      const uint32_t map_c3d7x4_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3,
          (static_cast<uint8_t*>(inv.workspace) + 0u) + 0, map_c3d7x4_dims, map_c3d7x4_strides, map_c3d7x4_box, map_c3d7x4_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.B) == sizeof(encoded));
      std::memcpy(&args_1.B, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x5_dims[] = {256ull, 2ull, 32ull, 8192ull};
      const uint64_t map_c3d7x5_strides[] = {256ull, 512ull, 16384ull};
      const uint32_t map_c3d7x5_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_c3d7x5_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm2_weights_scale)) + 0) + 0, map_c3d7x5_dims, map_c3d7x5_strides, map_c3d7x5_box, map_c3d7x5_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x6_dims[] = {32ull, 32ull, 42ull};
      const uint64_t map_c3d7x6_strides[] = {32ull, 1024ull};
      const uint32_t map_c3d7x6_box[] = {32u, 8u, 1u};
      const uint32_t map_c3d7x6_elem[] = {1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          (static_cast<uint8_t*>(inv.workspace) + 344064u) + 0, map_c3d7x6_dims, map_c3d7x6_strides, map_c3d7x6_box, map_c3d7x6_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFB) == sizeof(encoded));
      std::memcpy(&args_1.SFB, &encoded, sizeof(encoded));
    }
    {
      CUtensorMap encoded{};
      const uint64_t map_c3d7x7_dims[] = {4096ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_c3d7x7_strides[] = {8192ull, 68719468544ull, 8192ull};
      const uint32_t map_c3d7x7_box[] = {64u, 8u, 1u, 1u};
      const uint32_t map_c3d7x7_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          (static_cast<uint8_t*>(inv.workspace) + 389376u) + 0, map_c3d7x7_dims, map_c3d7x7_strides, map_c3d7x7_box, map_c3d7x7_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.C_tma) == sizeof(encoded));
      std::memcpy(&args_1.C_tma, &encoded, sizeof(encoded));
    }
    args_1.C = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 389376u));
    args_1.scale_c = reinterpret_cast<float*>((static_cast<uint8_t*>(const_cast<void*>(inv.output2_scale_scalar)) + 0));
    args_1.tile_expert = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_1.tile_mn_limit = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 388608u));
    args_1.num_non_exiting_ctas = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 388864u));
    args_1.work_counter = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3141888u));
    args_1.M = static_cast<int>(4096);
    args_1.K = static_cast<int>(2048);
    args_1.grid_m = static_cast<int>(32);
    args_1.grid_n = static_cast<int>(42);
    args_1.K_tiles = static_cast<int>(4);
    Args_device_04 args_2{};
    args_2.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 389376u));
    args_2.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_2.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_2.num_tokens = static_cast<int>(7);
    args_2.route_stride = static_cast<int>(32768);
    args_2.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready", dim3(32, 42, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_13, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe", dim3(32, 42, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      81536u, true, false, false,
      &Submit_device_07, &args_1, 100};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 7, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_04, &args_2, -1};
    visitor(launch_2, context);
    return;
  }
  case 8: {
    Args_device_08 args_0{};
    args_0.route_experts = reinterpret_cast<int*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_ids)) + 0));
    args_0.route_order = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 0u));
    args_0.route_count = static_cast<int>(48);
    Args_device_14 args_1{};
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
      128u, true, false, false,
      &Submit_device_08, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_trtllm_moe_bmm_tile_n8_fc1_persistent_nvfp4_cooperative_init_mma_u2_ready_expert_order", dim3(32, 48, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      199040u, true, false, false,
      &Submit_device_14, &args_1, -1};
    visitor(launch_1, context);
    KernelLaunch launch_2{
      "kernel_trtllm_moe_bmm_tile_n8_fc2_nvfp4_k512_two_cta_s2_uniform_regs_direct_stg_probe", dim3(32, 48, 1), dim3(512, 1, 1), dim3(1, 1, 1),
      81536u, true, false, false,
      &Submit_device_07, &args_2, 100};
    visitor(launch_2, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_direct_finalize_weight_preload_sm100", dim3(16, 8, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_04, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 9: {
    Args_device_05 args_0{};
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
      const uint64_t map_c3p9x0_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_c3p9x0_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_c3p9x0_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_c3p9x0_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_c3p9x0_dims, map_c3p9x0_strides, map_c3p9x0_box, map_c3p9x0_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_c3p9x1_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_c3p9x1_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_c3p9x1_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_c3p9x1_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_c3p9x1_dims, map_c3p9x1_strides, map_c3p9x1_box, map_c3p9x1_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_c3p9x2_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_c3p9x2_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_c3p9x2_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_c3p9x2_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 3328u) + 0, map_c3p9x2_dims, map_c3p9x2_strides, map_c3p9x2_box, map_c3p9x2_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 500992u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(9);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 54, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{3328u, 445696u, 500992u, 2048u, 2560u, 54},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 1792u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 9, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 10: {
    Args_device_05 args_0{};
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
      const uint64_t map_72_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_72_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_72_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_72_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_72_dims, map_72_strides, map_72_box, map_72_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_73_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_73_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_73_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_73_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_73_dims, map_73_strides, map_73_box, map_73_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_74_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_74_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_74_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_74_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 3584u) + 0, map_74_dims, map_74_strides, map_74_box, map_74_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 556544u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(10);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 60, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{3584u, 495104u, 556544u, 2304u, 2816u, 60},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2048u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 10, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
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
      const uint64_t map_79_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_79_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_79_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_79_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_79_dims, map_79_strides, map_79_box, map_79_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_80_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_80_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_80_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_80_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_80_dims, map_80_strides, map_80_box, map_80_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_81_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_81_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_81_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_81_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 4608u) + 0, map_81_dims, map_81_strides, map_81_box, map_81_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 612864u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(11);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 66, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{4608u, 545280u, 612864u, 2816u, 3840u, 66},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 11, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 12: {
    Args_device_05 args_0{};
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
      const uint64_t map_86_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_86_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_86_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_86_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_86_dims, map_86_strides, map_86_box, map_86_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_87_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_87_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_87_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_87_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_87_dims, map_87_strides, map_87_box, map_87_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_88_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_88_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_88_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_88_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 4608u) + 0, map_88_dims, map_88_strides, map_88_box, map_88_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 668160u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(12);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 72, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{4608u, 594432u, 668160u, 2816u, 3840u, 72},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2304u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 12, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 13: {
    Args_device_05 args_0{};
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
      const uint64_t map_100_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_100_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_100_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_100_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_100_dims, map_100_strides, map_100_box, map_100_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_101_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_101_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_101_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_101_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_101_dims, map_101_strides, map_101_box, map_101_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_102_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_102_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_102_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_102_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 4864u) + 0, map_102_dims, map_102_strides, map_102_box, map_102_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 723712u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(13);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 78, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{4864u, 643840u, 723712u, 3072u, 4096u, 78},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2560u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 13, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 14: {
    Args_device_05 args_0{};
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
      const uint64_t map_107_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_107_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_107_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_107_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_107_dims, map_107_strides, map_107_box, map_107_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_108_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_108_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_108_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_108_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_108_dims, map_108_strides, map_108_box, map_108_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_109_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_109_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_109_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_109_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5120u) + 0, map_109_dims, map_109_strides, map_109_box, map_109_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 779264u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(14);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 84, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{5120u, 693248u, 779264u, 3328u, 4352u, 84},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 2816u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 14, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 15: {
    Args_device_05 args_0{};
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
      const uint64_t map_121_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_121_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_121_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_121_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_121_dims, map_121_strides, map_121_box, map_121_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_122_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_122_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_122_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_122_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_122_dims, map_122_strides, map_122_box, map_122_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_123_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_123_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_123_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_123_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5376u) + 0, map_123_dims, map_123_strides, map_123_box, map_123_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 834816u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(15);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 90, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{5376u, 742656u, 834816u, 3584u, 4608u, 90},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 15, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 16: {
    Args_device_05 args_0{};
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
      const uint64_t map_135_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_135_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_135_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_135_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_135_dims, map_135_strides, map_135_box, map_135_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_136_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_136_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_136_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_136_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_136_dims, map_136_strides, map_136_box, map_136_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_137_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_137_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_137_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_137_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5376u) + 0, map_137_dims, map_137_strides, map_137_box, map_137_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 890112u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(16);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 96, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{5376u, 791808u, 890112u, 3584u, 4608u, 96},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3072u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 16, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 17: {
    Args_device_05 args_0{};
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
      const uint64_t map_142_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_142_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_142_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_142_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_142_dims, map_142_strides, map_142_box, map_142_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_143_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_143_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_143_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_143_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_143_dims, map_143_strides, map_143_box, map_143_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_144_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_144_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_144_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_144_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5632u) + 0, map_144_dims, map_144_strides, map_144_box, map_144_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 945664u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(17);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 102, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{5632u, 841216u, 945664u, 3840u, 4864u, 102},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3328u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 17, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 18: {
    Args_device_05 args_0{};
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
      const uint64_t map_156_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_156_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_156_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_156_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_156_dims, map_156_strides, map_156_box, map_156_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_157_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_157_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_157_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_157_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_157_dims, map_157_strides, map_157_box, map_157_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_158_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_158_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_158_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_158_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 5888u) + 0, map_158_dims, map_158_strides, map_158_box, map_158_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1001216u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(18);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 108, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{5888u, 890624u, 1001216u, 4096u, 5120u, 108},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3584u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 18, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 19: {
    Args_device_05 args_0{};
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
      const uint64_t map_170_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_170_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_170_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_170_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_170_dims, map_170_strides, map_170_box, map_170_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_171_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_171_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_171_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_171_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_171_dims, map_171_strides, map_171_box, map_171_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_172_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_172_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_172_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_172_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 6144u) + 0, map_172_dims, map_172_strides, map_172_box, map_172_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1056768u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(19);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 114, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{6144u, 940032u, 1056768u, 4352u, 5376u, 114},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 19, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 20: {
    Args_device_05 args_0{};
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
      const uint64_t map_184_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_184_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_184_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_184_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_184_dims, map_184_strides, map_184_box, map_184_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_185_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_185_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_185_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_185_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_185_dims, map_185_strides, map_185_box, map_185_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_186_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_186_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_186_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_186_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 6144u) + 0, map_186_dims, map_186_strides, map_186_box, map_186_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1112064u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(20);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 120, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{6144u, 989184u, 1112064u, 4352u, 5376u, 120},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 3840u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 20, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 21: {
    Args_device_05 args_0{};
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
      const uint64_t map_191_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_191_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_191_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_191_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_191_dims, map_191_strides, map_191_box, map_191_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_192_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_192_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_192_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_192_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_192_dims, map_192_strides, map_192_box, map_192_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_193_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_193_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_193_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_193_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 6400u) + 0, map_193_dims, map_193_strides, map_193_box, map_193_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1167616u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(21);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 126, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{6400u, 1038592u, 1167616u, 4608u, 5632u, 126},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4096u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 21, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 22: {
    Args_device_05 args_0{};
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
      const uint64_t map_198_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_198_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_198_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_198_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_198_dims, map_198_strides, map_198_box, map_198_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_199_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_199_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_199_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_199_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_199_dims, map_199_strides, map_199_box, map_199_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_200_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_200_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_200_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_200_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7424u) + 0, map_200_dims, map_200_strides, map_200_box, map_200_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1223936u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(22);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 132, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{7424u, 1088768u, 1223936u, 5120u, 6656u, 132},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4352u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 22, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
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
      const uint64_t map_212_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_212_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_212_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_212_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_212_dims, map_212_strides, map_212_box, map_212_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_213_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_213_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_213_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_213_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_213_dims, map_213_strides, map_213_box, map_213_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_214_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_214_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_214_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_214_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7680u) + 0, map_214_dims, map_214_strides, map_214_box, map_214_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1279488u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(23);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 138, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{7680u, 1138176u, 1279488u, 5376u, 6912u, 138},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 23, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 24: {
    Args_device_05 args_0{};
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
      const uint64_t map_219_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_219_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_219_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_219_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_219_dims, map_219_strides, map_219_box, map_219_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_220_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_220_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_220_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_220_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_220_dims, map_220_strides, map_220_box, map_220_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_221_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_221_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_221_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_221_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7680u) + 0, map_221_dims, map_221_strides, map_221_box, map_221_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1334784u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(24);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 144, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{7680u, 1187328u, 1334784u, 5376u, 6912u, 144},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4608u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 24, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 25: {
    Args_device_05 args_0{};
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
      const uint64_t map_233_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_233_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_233_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_233_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_233_dims, map_233_strides, map_233_box, map_233_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_234_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_234_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_234_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_234_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_234_dims, map_234_strides, map_234_box, map_234_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_235_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_235_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_235_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_235_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 7936u) + 0, map_235_dims, map_235_strides, map_235_box, map_235_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1390336u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6400u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(25);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 150, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{7936u, 1236736u, 1390336u, 5632u, 7168u, 150},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 4864u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 25, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 26: {
    Args_device_05 args_0{};
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
      const uint64_t map_240_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_240_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_240_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_240_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_240_dims, map_240_strides, map_240_box, map_240_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_241_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_241_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_241_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_241_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_241_dims, map_241_strides, map_241_box, map_241_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_242_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_242_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_242_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_242_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8192u) + 0, map_242_dims, map_242_strides, map_242_box, map_242_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1445888u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6656u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(26);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 156, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{8192u, 1286144u, 1445888u, 5888u, 7424u, 156},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5120u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 26, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 27: {
    Args_device_05 args_0{};
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
      const uint64_t map_254_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_254_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_254_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_254_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_254_dims, map_254_strides, map_254_box, map_254_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_255_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_255_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_255_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_255_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_255_dims, map_255_strides, map_255_box, map_255_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_256_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_256_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_256_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_256_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8448u) + 0, map_256_dims, map_256_strides, map_256_box, map_256_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1501440u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(27);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 162, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{8448u, 1335552u, 1501440u, 6144u, 7680u, 162},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 27, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 28: {
    Args_device_05 args_0{};
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
      const uint64_t map_268_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_268_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_268_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_268_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_268_dims, map_268_strides, map_268_box, map_268_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_269_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_269_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_269_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_269_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_269_dims, map_269_strides, map_269_box, map_269_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_270_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_270_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_270_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_270_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8448u) + 0, map_270_dims, map_270_strides, map_270_box, map_270_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1556736u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6912u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(28);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 168, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{8448u, 1384704u, 1556736u, 6144u, 7680u, 168},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5376u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 28, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 29: {
    Args_device_05 args_0{};
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
      const uint64_t map_282_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_282_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_282_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_282_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_282_dims, map_282_strides, map_282_box, map_282_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_283_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_283_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_283_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_283_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_283_dims, map_283_strides, map_283_box, map_283_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_284_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_284_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_284_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_284_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8704u) + 0, map_284_dims, map_284_strides, map_284_box, map_284_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1612288u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7168u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(29);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 174, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{8704u, 1434112u, 1612288u, 6400u, 7936u, 174},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5632u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 29, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 30: {
    Args_device_05 args_0{};
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
      const uint64_t map_289_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_289_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_289_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_289_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_289_dims, map_289_strides, map_289_box, map_289_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_290_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_290_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_290_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_290_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_290_dims, map_290_strides, map_290_box, map_290_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_291_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_291_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_291_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_291_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 8960u) + 0, map_291_dims, map_291_strides, map_291_box, map_291_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1667840u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7424u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(30);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 180, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{8960u, 1483520u, 1667840u, 6656u, 8192u, 180},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 5888u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 30, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 31: {
    Args_device_05 args_0{};
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
      const uint64_t map_296_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_296_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_296_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_296_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_296_dims, map_296_strides, map_296_box, map_296_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_297_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_297_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_297_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_297_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_297_dims, map_297_strides, map_297_box, map_297_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_298_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_298_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_298_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_298_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 9216u) + 0, map_298_dims, map_298_strides, map_298_box, map_298_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1723392u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(31);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 186, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{9216u, 1532928u, 1723392u, 6912u, 8448u, 186},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 31, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  case 32: {
    Args_device_05 args_0{};
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
      const uint64_t map_310_dims[] = {128ull, 4096ull, 16ull, 256ull};
      const uint64_t map_310_strides[] = {2048ull, 128ull, 8388608ull};
      const uint32_t map_310_box[] = {128u, 64u, 2u, 1u};
      const uint32_t map_310_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights)) + 0) + 0, map_310_dims, map_310_strides, map_310_box, map_310_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.A) == sizeof(encoded));
      std::memcpy(&args_1.A, &encoded, sizeof(encoded));
    }
    args_1.B = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_q)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_311_dims[] = {256ull, 2ull, 64ull, 8192ull};
      const uint64_t map_311_strides[] = {256ull, 512ull, 32768ull};
      const uint32_t map_311_box[] = {256u, 2u, 8u, 1u};
      const uint32_t map_311_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
          (static_cast<uint8_t*>(const_cast<void*>(inv.gemm1_weights_scale)) + 0) + 0, map_311_dims, map_311_strides, map_311_box, map_311_elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (status != CUDA_SUCCESS) throw std::runtime_error("clamped E256 tensor map encoding failed");
      static_assert(sizeof(args_1.SFA) == sizeof(encoded));
      std::memcpy(&args_1.SFA, &encoded, sizeof(encoded));
    }
    args_1.SFB = reinterpret_cast<uint8_t*>((static_cast<uint8_t*>(const_cast<void*>(inv.hidden_states_scale)) + 0));
    {
      CUtensorMap encoded{};
      const uint64_t map_312_dims[] = {2048ull, 8ull, 2147483648ull, 2147483648ull};
      const uint64_t map_312_strides[] = {1024ull, 17179868160ull, 1024ull};
      const uint32_t map_312_box[] = {32u, 8u, 1u, 1u};
      const uint32_t map_312_elem[] = {1u, 1u, 1u, 1u};
      const CUresult status = cuTensorMapEncodeTiled(&encoded, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4,
          (static_cast<uint8_t*>(inv.workspace) + 9216u) + 0, map_312_dims, map_312_strides, map_312_box, map_312_elem,
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
    Args_device_03 args_3{};
    args_3.route_outputs = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(inv.workspace) + 1778688u));
    args_3.route_weights = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.topk_weights)) + 0));
    args_3.route_slots = reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 7680u));
    args_3.output = reinterpret_cast<__nv_bfloat16*>((static_cast<uint8_t*>(const_cast<void*>(inv.output)) + 0));
    args_3.num_tokens = static_cast<int>(32);
    args_3.route_stride = static_cast<int>(4096);
    args_3.M = static_cast<int>(4096);
    KernelLaunch launch_0{
      "kernel_e256_route_tile_plan_count_rank_reuse_early_pdl", dim3(1, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1),
      6272u, true, false, false,
      &Submit_device_05, &args_0, -1};
    visitor(launch_0, context);
    KernelLaunch launch_1{
      "kernel_dsv4_flash_moe_fc1_joint_sfb_cursor_v20_sm100", dim3(64, 192, 1), dim3(352, 1, 1), dim3(2, 1, 1),
      64128u, true, false, true,
      &Submit_device_01, &args_1, 100};
    visitor(launch_1, context);
    VisitTileN8Fc2(inv, required, TileN8Fc2Regions{9216u, 1582080u, 1778688u, 6912u, 8448u, 192},
                   reinterpret_cast<int*>((static_cast<uint8_t*>(inv.workspace) + 6144u)), 100, visitor, context);
    KernelLaunch launch_3{
      "kernel_dsv4_flash_moe_5184_finalize_weight_preload_sm100", dim3(16, 32, 1), dim3(128, 1, 1), dim3(1, 1, 1),
      0u, true, false, false,
      &Submit_device_03, &args_3, -1};
    visitor(launch_3, context);
    return;
  }
  default: throw std::runtime_error("unsupported clamped E256 token count");
  }
}
}  // namespace flashinfer::warp_decode::generated::dsv4_clamped_e256
#endif
