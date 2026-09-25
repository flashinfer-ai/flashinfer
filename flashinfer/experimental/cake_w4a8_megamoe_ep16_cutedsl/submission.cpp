// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <cstdint>

namespace w4a8_cluster_submission {
struct State {
  CUfunction function;
  void** arguments;
  CUlaunchAttribute attributes[2]{};
  CUlaunchConfig config{};
};
int64_t Create(int64_t function, int64_t arguments, int64_t stream, int64_t gx, int64_t gy,
               int64_t gz, int64_t bx, int64_t by, int64_t bz, int64_t smem, int64_t cx, int64_t cy,
               int64_t cz, int64_t spread) {
  auto* state = new State{};
  state->function = reinterpret_cast<CUfunction>(static_cast<uintptr_t>(function));
  state->arguments = reinterpret_cast<void**>(static_cast<uintptr_t>(arguments));
  auto& cfg = state->config;
  cfg.gridDimX = gx;
  cfg.gridDimY = gy;
  cfg.gridDimZ = gz;
  cfg.blockDimX = bx;
  cfg.blockDimY = by;
  cfg.blockDimZ = bz;
  cfg.sharedMemBytes = smem;
  cfg.hStream = reinterpret_cast<CUstream>(static_cast<uintptr_t>(stream));
  state->attributes[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  state->attributes[0].value.clusterDim.x = cx;
  state->attributes[0].value.clusterDim.y = cy;
  state->attributes[0].value.clusterDim.z = cz;
  cfg.attrs = state->attributes;
  cfg.numAttrs = 1;
  if (spread) {
    state->attributes[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    state->attributes[1].value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    cfg.numAttrs = 2;
  }
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(state));
}
void Run(int64_t handle) {
  auto* state = reinterpret_cast<State*>(static_cast<uintptr_t>(handle));
  const CUresult result =
      cuLaunchKernelEx(&state->config, state->function, state->arguments, nullptr);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "native cuLaunchKernelEx failed: " << static_cast<int>(result);
}
int64_t Inspect(int64_t handle, int64_t field) {
  auto* state = reinterpret_cast<State*>(static_cast<uintptr_t>(handle));
  const auto& cfg = state->config;
  switch (field) {
    case 0:
      return reinterpret_cast<uintptr_t>(state->function);
    case 1:
      return reinterpret_cast<uintptr_t>(state->arguments);
    case 2:
      return reinterpret_cast<uintptr_t>(cfg.hStream);
    case 3:
      return cfg.gridDimX;
    case 4:
      return cfg.gridDimY;
    case 5:
      return cfg.gridDimZ;
    case 6:
      return cfg.blockDimX;
    case 7:
      return cfg.blockDimY;
    case 8:
      return cfg.blockDimZ;
    case 9:
      return cfg.sharedMemBytes;
    case 10:
      return cfg.attrs[0].value.clusterDim.x;
    case 11:
      return cfg.attrs[0].value.clusterDim.y;
    case 12:
      return cfg.attrs[0].value.clusterDim.z;
    case 13:
      return cfg.numAttrs == 2;
    case 14:
      return cfg.attrs[0].id;
    case 15:
      return cfg.numAttrs == 2 ? cfg.attrs[1].id : -1;
    case 16:
      return cfg.numAttrs == 2 ? cfg.attrs[1].value.clusterSchedulingPolicyPreference : -1;
    default:
      TVM_FFI_THROW(ValueError) << "invalid config field";
  }
  return -1;
}
void Destroy(int64_t handle) { delete reinterpret_cast<State*>(static_cast<uintptr_t>(handle)); }
}  // namespace w4a8_cluster_submission
TVM_FFI_DLL_EXPORT_TYPED_FUNC(create, w4a8_cluster_submission::Create);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, w4a8_cluster_submission::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(inspect, w4a8_cluster_submission::Inspect);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(destroy, w4a8_cluster_submission::Destroy);
