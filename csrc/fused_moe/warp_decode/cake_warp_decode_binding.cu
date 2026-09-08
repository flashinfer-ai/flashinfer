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

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "cake_warp_decode_contract.cuh"
#include "tvm_ffi_utils.h"

#if !__has_include("generated/cake_warp_decode_generated_manifest.cuh")
#error \
    "generated/cake_warp_decode_generated_manifest.cuh is required; generate the kernel manifest before building this module"
#endif
#include "generated/cake_warp_decode_generated_manifest.cuh"

namespace flashinfer::warp_decode {

using tvm::ffi::TensorView;

static_assert(generated::kContractVersion == kGeneratedContractVersion,
              "warp-decode generated manifest contract version mismatch");
static_assert(generated::kUsesGridConstantTensorMapAbi,
              "warp-decode kernels require by-value grid-constant tensor maps");
static_assert(generated::kMaximumLaunchCount == 4,
              "warp-decode adaptive schedules contain three or four launches");
static_assert(generated::kConfiguresDynamicSharedMemory,
              "warp-decode generated manifest must configure every kernel's dynamic shared memory");
static_assert(generated::kWorkspaceAlignment >= alignof(CUtensorMap),
              "warp-decode workspace alignment must cover CUDA tensor maps");

namespace {

struct NamedTensor {
  const TensorView* tensor;
  const char* name;
};

struct AddressRange {
  uintptr_t begin;
  uintptr_t end;
};

void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

void CheckManifestStatus(const ManifestStatus& status) {
  if (status.Ok()) {
    return;
  }
  const char* operation =
      status.operation == nullptr ? "generated manifest operation" : status.operation;
  if (status.domain == StatusDomain::kCudaRuntime) {
    const auto error = static_cast<cudaError_t>(status.code);
    TVM_FFI_ICHECK(false) << operation << " failed: " << cudaGetErrorString(error);
  }
  if (status.domain == StatusDomain::kCudaDriver) {
    const char* description = nullptr;
    cuGetErrorString(static_cast<CUresult>(status.code), &description);
    TVM_FFI_ICHECK(false) << operation << " failed: "
                          << (description == nullptr ? "unknown CUDA driver error" : description);
  }
  TVM_FFI_ICHECK(false) << operation
                        << " rejected the generated kernel manifest, code=" << status.code;
}

void CheckSm103a(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(computeCapabilityMajor)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(computeCapabilityMinor)");
  TVM_FFI_ICHECK(major == 10 && minor == 3)
      << "cake warp decode requires exact compute capability 10.3, got " << major << "." << minor;
}

void CheckDtype(const TensorView& tensor, const char* name, DLDataType expected) {
  const DLDataType actual = tensor.dtype();
  TVM_FFI_ICHECK(actual.code == expected.code && actual.bits == expected.bits &&
                 actual.lanes == expected.lanes)
      << name << " dtype mismatch: expected (code=" << static_cast<int>(expected.code)
      << ", bits=" << static_cast<int>(expected.bits)
      << ", lanes=" << static_cast<int>(expected.lanes)
      << "), got (code=" << static_cast<int>(actual.code)
      << ", bits=" << static_cast<int>(actual.bits) << ", lanes=" << static_cast<int>(actual.lanes)
      << ")";
}

void CheckTensor(const TensorView& tensor, const char* name, int32_t device_id, DLDataType dtype) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on cuda:" << device_id << ", got cuda:" << tensor.device().device_id;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  CheckDtype(tensor, name, dtype);
}

void CheckE4m3Storage(const TensorView& tensor, const char* name, int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on cuda:" << device_id << ", got cuda:" << tensor.device().device_id;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  const DLDataType actual = tensor.dtype();
  const bool e4m3 = actual.code == dl_float8_e4m3fn.code && actual.bits == dl_float8_e4m3fn.bits &&
                    actual.lanes == dl_float8_e4m3fn.lanes;
  const bool raw_bytes = actual.code == dl_uint8.code && actual.bits == dl_uint8.bits &&
                         actual.lanes == dl_uint8.lanes;
  TVM_FFI_ICHECK(e4m3 || raw_bytes)
      << name << " must expose E4M3 bytes as float8_e4m3fn or uint8 storage";
}

void CheckShape(const TensorView& tensor, const char* name,
                std::initializer_list<int64_t> expected) {
  TVM_FFI_ICHECK(tensor.ndim() == static_cast<int32_t>(expected.size()))
      << name << " rank mismatch: expected " << expected.size() << ", got " << tensor.ndim();
  int32_t axis = 0;
  for (int64_t extent : expected) {
    TVM_FFI_ICHECK(tensor.size(axis) == extent) << name << " dimension " << axis << " must equal "
                                                << extent << ", got " << tensor.size(axis);
    ++axis;
  }
}

size_t TensorStorageBytes(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK(tensor.numel() >= 0) << name << " has a negative element count";
  const uint64_t bits_per_element =
      static_cast<uint64_t>(tensor.dtype().bits) * static_cast<uint64_t>(tensor.dtype().lanes);
  const uint64_t elements = static_cast<uint64_t>(tensor.numel());
  TVM_FFI_ICHECK(bits_per_element > 0 &&
                 elements <= (std::numeric_limits<uint64_t>::max() - 7) / bits_per_element)
      << name << " storage size overflows uint64";
  const uint64_t bytes = (elements * bits_per_element + 7) / 8;
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<size_t>::max())
      << name << " storage size overflows size_t";
  return static_cast<size_t>(bytes);
}

AddressRange TensorRange(const TensorView& tensor, const char* name) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const size_t bytes = TensorStorageBytes(tensor, name);
  TVM_FFI_ICHECK(bytes == 0 || begin != 0) << name << " has null storage";
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " address range overflows uintptr_t";
  return {begin, begin + bytes};
}

void CheckNoOverlap(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                    const char* rhs_name) {
  const AddressRange a = TensorRange(lhs, lhs_name);
  const AddressRange b = TensorRange(rhs, rhs_name);
  TVM_FFI_ICHECK(a.end <= b.begin || b.end <= a.begin)
      << lhs_name << " must not overlap " << rhs_name;
}

void CheckAlignment(const TensorView& tensor, const char* name, size_t alignment) {
  TVM_FFI_ICHECK(alignment > 0 && (alignment & (alignment - 1)) == 0)
      << "generated manifest supplied a non-power-of-two alignment";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0)
      << name << " data pointer must be aligned to " << alignment << " bytes";
}

Shape CheckedShape(int64_t num_tokens, int64_t hidden_size, int64_t intermediate_size,
                   int64_t num_experts, int64_t top_k) {
  for (const auto& value : {std::pair<int64_t, const char*>{num_tokens, "num_tokens"},
                            {hidden_size, "hidden_size"},
                            {intermediate_size, "intermediate_size"},
                            {num_experts, "num_experts"},
                            {top_k, "top_k"}}) {
    TVM_FFI_ICHECK(value.first >= 1 && value.first <= std::numeric_limits<int32_t>::max())
        << value.second << " must fit a positive int32, got " << value.first;
  }
  Shape shape{static_cast<int32_t>(num_tokens),        static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(intermediate_size), static_cast<int32_t>(num_experts),
              static_cast<int32_t>(num_experts),       static_cast<int32_t>(top_k)};
  TVM_FFI_ICHECK(SelectSchedule(shape).supported)
      << "cake warp decode supports only (H=2048, I=512, E=512, top_k=10) or "
         "(H=2048, I=1536, E=60, top_k=4), with 1 <= num_tokens <= 32";
  return shape;
}

int64_t CheckedWorkspaceSize(const Shape& shape, const Schedule& schedule) {
  const int64_t bytes = generated::WorkspaceSize(shape, schedule);
  TVM_FFI_ICHECK(bytes > 0) << "generated manifest returned an invalid workspace size: " << bytes;
  TVM_FFI_ICHECK(static_cast<uint64_t>(bytes) <= std::numeric_limits<size_t>::max())
      << "generated workspace size does not fit size_t";
  return bytes;
}

void CheckWorkspace(const TensorView& workspace, int32_t device_id, int64_t required_bytes) {
  CheckTensor(workspace, "workspace_u8", device_id, dl_uint8);
  TVM_FFI_ICHECK(workspace.ndim() == 1) << "workspace_u8 must be a one-dimensional byte buffer";
  TVM_FFI_ICHECK(workspace.numel() >= required_bytes)
      << "workspace_u8 requires at least " << required_bytes << " bytes, got " << workspace.numel();
  CheckAlignment(workspace, "workspace_u8", generated::kWorkspaceAlignment);
}

struct LaunchContext {
  cudaStream_t stream;
  int32_t launch_count;
};

struct WorkspaceKey {
  int32_t device_id;
  uintptr_t pointer;
  size_t bytes;
  Shape shape;
  Schedule schedule;

  bool operator==(const WorkspaceKey& other) const {
    return device_id == other.device_id && pointer == other.pointer && bytes == other.bytes &&
           shape.num_tokens == other.shape.num_tokens &&
           shape.hidden_size == other.shape.hidden_size &&
           shape.intermediate_size == other.shape.intermediate_size &&
           shape.num_experts == other.shape.num_experts &&
           shape.local_num_experts == other.shape.local_num_experts &&
           shape.top_k == other.shape.top_k && schedule.supported == other.schedule.supported &&
           schedule.geometry == other.schedule.geometry &&
           schedule.route_layout == other.schedule.route_layout &&
           schedule.route_packer == other.schedule.route_packer &&
           schedule.fc1 == other.schedule.fc1 && schedule.fc2 == other.schedule.fc2 &&
           schedule.finalize_threads == other.schedule.finalize_threads &&
           schedule.finalize_unroll == other.schedule.finalize_unroll &&
           schedule.workfeed_ctas == other.schedule.workfeed_ctas;
  }
};

struct WorkspaceKeyHash {
  size_t operator()(const WorkspaceKey& key) const {
    size_t seed = 0xcbf29ce484222325ULL;
    const auto mix = [&seed](uint64_t value) {
      seed ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    mix(static_cast<uint32_t>(key.device_id));
    mix(key.pointer);
    mix(key.bytes);
    mix(static_cast<uint32_t>(key.shape.num_tokens));
    mix(static_cast<uint32_t>(key.shape.hidden_size));
    mix(static_cast<uint32_t>(key.shape.intermediate_size));
    mix(static_cast<uint32_t>(key.shape.num_experts));
    mix(static_cast<uint32_t>(key.shape.local_num_experts));
    mix(static_cast<uint32_t>(key.shape.top_k));
    mix(static_cast<uint8_t>(key.schedule.geometry));
    mix(static_cast<uint8_t>(key.schedule.route_layout));
    mix(static_cast<uint8_t>(key.schedule.route_packer));
    mix(static_cast<uint8_t>(key.schedule.fc1));
    mix(static_cast<uint8_t>(key.schedule.fc2));
    mix(static_cast<uint32_t>(key.schedule.finalize_threads));
    mix(static_cast<uint32_t>(key.schedule.finalize_unroll));
    mix(static_cast<uint32_t>(key.schedule.workfeed_ctas));
    return seed;
  }
};

struct WorkspaceAddressKey {
  int32_t device_id;
  uintptr_t pointer;

  bool operator==(const WorkspaceAddressKey& other) const {
    return device_id == other.device_id && pointer == other.pointer;
  }
};

struct WorkspaceAddressKeyHash {
  size_t operator()(const WorkspaceAddressKey& key) const {
    size_t seed = 0xcbf29ce484222325ULL;
    seed ^= static_cast<size_t>(static_cast<uint32_t>(key.device_id)) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    seed ^= static_cast<size_t>(key.pointer) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
  }
};

WorkspaceKey MakeWorkspaceKey(const Invocation& invocation, const Schedule& schedule,
                              int32_t device_id) {
  return {device_id, reinterpret_cast<uintptr_t>(invocation.workspace), invocation.workspace_bytes,
          invocation.shape, schedule};
}

std::mutex prepared_workspaces_mutex;
struct PreparedWorkspaceState {
  int64_t receipt;
  cudaEvent_t completion;
  bool has_submission;
  bool poisoned;
};

std::unordered_map<WorkspaceKey, PreparedWorkspaceState, WorkspaceKeyHash> prepared_workspaces;
// External event nodes captured by CUDA Graphs retain the event handle, not a
// C++ owner. Keep one stable handle per device/address for process lifetime so
// releasing a receipt cannot leave a still-live GraphExec with a dangling
// handle. The bounded address pool also fails closed under allocator churn.
std::unordered_map<WorkspaceAddressKey, cudaEvent_t, WorkspaceAddressKeyHash>
    workspace_completion_events;
int64_t next_workspace_receipt = 1;

cudaEvent_t GetWorkspaceCompletionEvent(int32_t device_id, uintptr_t pointer) {
  const WorkspaceAddressKey address{device_id, pointer};
  const auto existing = workspace_completion_events.find(address);
  if (existing != workspace_completion_events.end()) {
    return existing->second;
  }
  constexpr size_t kMaximumWorkspaceCompletionEvents = 4096;
  TVM_FFI_ICHECK(workspace_completion_events.size() < kMaximumWorkspaceCompletionEvents)
      << "warp-decode workspace completion-event address capacity is exhausted";
  cudaEvent_t completion = nullptr;
  CheckCuda(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming),
            "cudaEventCreateWithFlags(workspace completion)");
  workspace_completion_events.emplace(address, completion);
  return completion;
}

void SynchronizeWorkspaceSubmissions(const PreparedWorkspaceState& state, const char* operation) {
  TVM_FFI_ICHECK(!state.poisoned)
      << "warp-decode workspace is quarantined because accepted GPU work could not "
         "be covered by a completion event; refusing to release or re-prepare it";
  if (state.has_submission) {
    CheckCuda(cudaEventSynchronize(state.completion), operation);
  }
}

int64_t PrepareWorkspaceAlways(const Invocation& invocation, const Schedule& schedule,
                               int32_t device_id, cudaStream_t stream) {
  const WorkspaceKey key = MakeWorkspaceKey(invocation, schedule, device_id);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
            "cudaStreamIsCapturing(workspace prepare)");
  TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
      << "warp-decode workspace preparation must run outside CUDA Graph capture";
  std::lock_guard<std::mutex> lock(prepared_workspaces_mutex);
  // A pointer has exactly one live preparation generation on a device. Remove
  // every prior shape/schedule for the address before initialization begins;
  // a failed re-prepare therefore cannot leave an older receipt usable for a
  // workspace whose contents may already have been partially overwritten.
  for (auto existing = prepared_workspaces.begin(); existing != prepared_workspaces.end();) {
    if (existing->first.device_id == device_id && existing->first.pointer == key.pointer) {
      SynchronizeWorkspaceSubmissions(existing->second,
                                      "cudaEventSynchronize(previous workspace submissions)");
      existing = prepared_workspaces.erase(existing);
    } else {
      ++existing;
    }
  }
  constexpr size_t kMaximumPreparedWorkspaceReceipts = 4096;
  TVM_FFI_ICHECK(prepared_workspaces.size() < kMaximumPreparedWorkspaceReceipts)
      << "warp-decode prepared-workspace receipt capacity is exhausted";
  CheckManifestStatus(generated::PrepareWorkspace(invocation, schedule, stream));

  // Preparation is deliberately outside timed and capture regions. Complete it
  // before returning so a subsequent launch on any CUDA stream has a concrete
  // happens-before edge without adding an external-event node to captured graphs.
  cudaEvent_t ready = nullptr;
  CheckCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
            "cudaEventCreateWithFlags(workspace ready)");
  CheckCuda(cudaEventRecord(ready, stream), "cudaEventRecord(workspace ready)");
  CheckCuda(cudaEventSynchronize(ready), "cudaEventSynchronize(workspace ready)");
  CheckCuda(cudaEventDestroy(ready), "cudaEventDestroy(workspace ready)");

  TVM_FFI_ICHECK(next_workspace_receipt > 0 &&
                 next_workspace_receipt < std::numeric_limits<int64_t>::max())
      << "warp-decode workspace receipt counter is exhausted";
  const cudaEvent_t completion = GetWorkspaceCompletionEvent(device_id, key.pointer);
  const int64_t receipt = next_workspace_receipt++;
  prepared_workspaces.insert_or_assign(key,
                                       PreparedWorkspaceState{receipt, completion, false, false});
  return receipt;
}

PreparedWorkspaceState& RequirePreparedWorkspace(const Invocation& invocation,
                                                 const Schedule& schedule, int32_t device_id,
                                                 int64_t receipt) {
  const WorkspaceKey key = MakeWorkspaceKey(invocation, schedule, device_id);
  auto existing = prepared_workspaces.find(key);
  TVM_FFI_ICHECK(existing != prepared_workspaces.end() && existing->second.receipt == receipt)
      << "workspace_u8 does not have the current preparation receipt for this warp-decode "
         "shape; prepare this exact workspace outside CUDA Graph capture";
  TVM_FFI_ICHECK(!existing->second.poisoned)
      << "workspace_u8 is quarantined because its last accepted GPU submission "
         "could not be covered by a completion event";
  return existing->second;
}

void ReleaseWorkspaceReceipt(int64_t receipt) {
  TVM_FFI_ICHECK(receipt > 0)
      << "warp-decode workspace release requires a positive preparation receipt";
  std::lock_guard<std::mutex> lock(prepared_workspaces_mutex);
  for (auto existing = prepared_workspaces.begin(); existing != prepared_workspaces.end();
       ++existing) {
    if (existing->second.receipt == receipt) {
      ffi::CUDADeviceGuard device_guard(existing->first.device_id);
      SynchronizeWorkspaceSubmissions(existing->second,
                                      "cudaEventSynchronize(released workspace submissions)");
      prepared_workspaces.erase(existing);
      return;
    }
  }
  TVM_FFI_ICHECK(false)
      << "warp-decode workspace release received an unknown or already released receipt";
}

void LaunchOne(const KernelLaunch& launch, void* opaque_context) {
  auto* context = static_cast<LaunchContext*>(opaque_context);
  TVM_FFI_ICHECK(context != nullptr) << "warp-decode launch context is null";
  TVM_FFI_ICHECK(context->launch_count < static_cast<int32_t>(generated::kMaximumLaunchCount))
      << "generated manifest emitted too many launches";
  TVM_FFI_ICHECK(launch.name != nullptr && launch.name[0] != '\0')
      << "generated manifest emitted an unnamed kernel";
  TVM_FFI_ICHECK(launch.submit != nullptr && launch.arguments != nullptr)
      << launch.name << " has an incomplete generated submit packet";
  TVM_FFI_ICHECK(launch.grid.x > 0 && launch.grid.y > 0 && launch.grid.z > 0)
      << launch.name << " has an invalid grid";
  TVM_FFI_ICHECK(launch.block.x > 0 && launch.block.y > 0 && launch.block.z > 0 &&
                 static_cast<uint64_t>(launch.block.x) * launch.block.y * launch.block.z <= 1024)
      << launch.name << " has an invalid block";
  TVM_FFI_ICHECK(launch.programmatic_dependent_launch)
      << launch.name << " does not declare the required programmatic dependent launch contract";

  std::array<cudaLaunchAttribute, 5> attributes{};
  uint32_t attribute_count = 0;

  attributes[attribute_count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[attribute_count].val.programmaticStreamSerializationAllowed = 1;
  ++attribute_count;

  const bool clustered = launch.cluster.x > 1 || launch.cluster.y > 1 || launch.cluster.z > 1;
  if (clustered) {
    TVM_FFI_ICHECK(launch.cluster.x > 0 && launch.cluster.y > 0 && launch.cluster.z > 0)
        << launch.name << " has an invalid cluster dimension";
    attributes[attribute_count].id = cudaLaunchAttributeClusterDimension;
    attributes[attribute_count].val.clusterDim.x = launch.cluster.x;
    attributes[attribute_count].val.clusterDim.y = launch.cluster.y;
    attributes[attribute_count].val.clusterDim.z = launch.cluster.z;
    ++attribute_count;
    if (launch.spread_cluster) {
      attributes[attribute_count].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
      attributes[attribute_count].val.clusterSchedulingPolicyPreference =
          cudaClusterSchedulingPolicySpread;
      ++attribute_count;
    }
  } else {
    TVM_FFI_ICHECK(!launch.spread_cluster)
        << launch.name << " requests spread scheduling without a cluster";
  }
  if (launch.cooperative) {
    attributes[attribute_count].id = cudaLaunchAttributeCooperative;
    attributes[attribute_count].val.cooperative = 1;
    ++attribute_count;
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
  if (launch.allow_oversized_smem) {
    attributes[attribute_count].id = cudaLaunchAttributeSharedMemoryMode;
    attributes[attribute_count].val.sharedMemoryMode = cudaSharedMemoryModeAllowNonPortable;
    ++attribute_count;
  }
#else
  // Older runtimes use the per-function MaxDynamicSharedMemorySize attribute
  // configured by generated::EnsureDeviceReady. CUDA 13.2 added the explicit
  // per-launch non-portable shared-memory mode used above.
  (void)launch.allow_oversized_smem;
#endif

  cudaLaunchConfig_t config{};
  config.gridDim = launch.grid;
  config.blockDim = launch.block;
  config.dynamicSmemBytes = launch.dynamic_smem_bytes;
  config.stream = context->stream;
  config.attrs = attributes.data();
  config.numAttrs = attribute_count;

  CheckCuda(launch.submit(&config, launch.arguments), launch.name);
  ++context->launch_count;
}

void CheckAllTensorArguments(
    const TensorView& output_bf16, const TensorView& workspace_u8,
    const TensorView& hidden_states_q_u8, const TensorView& hidden_states_scale_e4m3,
    const TensorView& topk_ids_i32, const TensorView& topk_weights_bf16,
    const TensorView& gemm1_weights_u8, const TensorView& gemm1_weights_scale_e4m3,
    const TensorView& gemm2_weights_u8, const TensorView& gemm2_weights_scale_e4m3,
    const TensorView& output1_scale_scalar_f32, const TensorView& output1_scale_gate_scalar_f32,
    const TensorView& output2_scale_scalar_f32, const Shape& shape, int64_t workspace_bytes) {
  const int32_t device_id = output_bf16.device().device_id;
  const int64_t tokens = shape.num_tokens;
  const int64_t hidden = shape.hidden_size;
  const int64_t intermediate = shape.intermediate_size;
  const int64_t experts = shape.num_experts;
  const int64_t top_k = shape.top_k;

  CheckTensor(output_bf16, "output_bf16", device_id, dl_bfloat16);
  CheckWorkspace(workspace_u8, device_id, workspace_bytes);
  CheckTensor(hidden_states_q_u8, "hidden_states_q_u8", device_id, dl_uint8);
  CheckE4m3Storage(hidden_states_scale_e4m3, "hidden_states_scale_e4m3", device_id);
  CheckTensor(topk_ids_i32, "topk_ids_i32", device_id, dl_int32);
  CheckTensor(topk_weights_bf16, "topk_weights_bf16", device_id, dl_bfloat16);
  CheckTensor(gemm1_weights_u8, "gemm1_weights_u8", device_id, dl_uint8);
  CheckE4m3Storage(gemm1_weights_scale_e4m3, "gemm1_weights_scale_e4m3", device_id);
  CheckTensor(gemm2_weights_u8, "gemm2_weights_u8", device_id, dl_uint8);
  CheckE4m3Storage(gemm2_weights_scale_e4m3, "gemm2_weights_scale_e4m3", device_id);
  CheckTensor(output1_scale_scalar_f32, "output1_scale_scalar_f32", device_id, dl_float32);
  CheckTensor(output1_scale_gate_scalar_f32, "output1_scale_gate_scalar_f32", device_id,
              dl_float32);
  CheckTensor(output2_scale_scalar_f32, "output2_scale_scalar_f32", device_id, dl_float32);

  CheckShape(output_bf16, "output_bf16", {tokens, hidden});
  CheckShape(hidden_states_q_u8, "hidden_states_q_u8", {tokens, hidden / 2});
  CheckShape(hidden_states_scale_e4m3, "hidden_states_scale_e4m3", {tokens, hidden / 16});
  CheckShape(topk_ids_i32, "topk_ids_i32", {tokens, top_k});
  CheckShape(topk_weights_bf16, "topk_weights_bf16", {tokens, top_k});
  CheckShape(gemm1_weights_u8, "gemm1_weights_u8", {experts, 2 * intermediate, hidden / 2});
  CheckShape(gemm1_weights_scale_e4m3, "gemm1_weights_scale_e4m3",
             {experts, 2 * intermediate, hidden / 16});
  CheckShape(gemm2_weights_u8, "gemm2_weights_u8", {experts, hidden, intermediate / 2});
  CheckShape(gemm2_weights_scale_e4m3, "gemm2_weights_scale_e4m3",
             {experts, hidden, intermediate / 16});
  CheckShape(output1_scale_scalar_f32, "output1_scale_scalar_f32", {experts});
  CheckShape(output1_scale_gate_scalar_f32, "output1_scale_gate_scalar_f32", {experts});
  CheckShape(output2_scale_scalar_f32, "output2_scale_scalar_f32", {experts});

  CheckAlignment(output_bf16, "output_bf16", 16);
  CheckAlignment(hidden_states_q_u8, "hidden_states_q_u8", 16);
  CheckAlignment(hidden_states_scale_e4m3, "hidden_states_scale_e4m3", 16);
  CheckAlignment(gemm1_weights_u8, "gemm1_weights_u8", 16);
  CheckAlignment(gemm1_weights_scale_e4m3, "gemm1_weights_scale_e4m3", 16);
  CheckAlignment(gemm2_weights_u8, "gemm2_weights_u8", 16);
  CheckAlignment(gemm2_weights_scale_e4m3, "gemm2_weights_scale_e4m3", 16);

  const std::array<NamedTensor, 11> read_only{{
      {&hidden_states_q_u8, "hidden_states_q_u8"},
      {&hidden_states_scale_e4m3, "hidden_states_scale_e4m3"},
      {&topk_ids_i32, "topk_ids_i32"},
      {&topk_weights_bf16, "topk_weights_bf16"},
      {&gemm1_weights_u8, "gemm1_weights_u8"},
      {&gemm1_weights_scale_e4m3, "gemm1_weights_scale_e4m3"},
      {&gemm2_weights_u8, "gemm2_weights_u8"},
      {&gemm2_weights_scale_e4m3, "gemm2_weights_scale_e4m3"},
      {&output1_scale_scalar_f32, "output1_scale_scalar_f32"},
      {&output1_scale_gate_scalar_f32, "output1_scale_gate_scalar_f32"},
      {&output2_scale_scalar_f32, "output2_scale_scalar_f32"},
  }};
  CheckNoOverlap(output_bf16, "output_bf16", workspace_u8, "workspace_u8");
  for (const NamedTensor& input : read_only) {
    CheckNoOverlap(output_bf16, "output_bf16", *input.tensor, input.name);
    CheckNoOverlap(workspace_u8, "workspace_u8", *input.tensor, input.name);
  }
}

Invocation MakeInvocation(
    const TensorView& output_bf16, const TensorView& workspace_u8,
    const TensorView& hidden_states_q_u8, const TensorView& hidden_states_scale_e4m3,
    const TensorView& topk_ids_i32, const TensorView& topk_weights_bf16,
    const TensorView& gemm1_weights_u8, const TensorView& gemm1_weights_scale_e4m3,
    const TensorView& gemm2_weights_u8, const TensorView& gemm2_weights_scale_e4m3,
    const TensorView& output1_scale_scalar_f32, const TensorView& output1_scale_gate_scalar_f32,
    const TensorView& output2_scale_scalar_f32, const Shape& shape) {
  return {shape,
          output_bf16.data_ptr(),
          workspace_u8.data_ptr(),
          hidden_states_q_u8.data_ptr(),
          hidden_states_scale_e4m3.data_ptr(),
          topk_ids_i32.data_ptr(),
          topk_weights_bf16.data_ptr(),
          gemm1_weights_u8.data_ptr(),
          gemm1_weights_scale_e4m3.data_ptr(),
          gemm2_weights_u8.data_ptr(),
          gemm2_weights_scale_e4m3.data_ptr(),
          output1_scale_scalar_f32.data_ptr(),
          output1_scale_gate_scalar_f32.data_ptr(),
          output2_scale_scalar_f32.data_ptr(),
          TensorStorageBytes(workspace_u8, "workspace_u8")};
}

}  // namespace

int64_t WorkspaceSize(int64_t num_tokens, int64_t hidden_size, int64_t intermediate_size,
                      int64_t num_experts, int64_t top_k) {
  const Shape shape = CheckedShape(num_tokens, hidden_size, intermediate_size, num_experts, top_k);
  return CheckedWorkspaceSize(shape, SelectSchedule(shape));
}

int64_t PrepareWorkspace(TensorView workspace_u8, int64_t num_tokens, int64_t hidden_size,
                         int64_t intermediate_size, int64_t num_experts, int64_t top_k) {
  TVM_FFI_ICHECK(workspace_u8.device().device_type == kDLCUDA)
      << "workspace_u8 must be a CUDA tensor";
  const Shape shape = CheckedShape(num_tokens, hidden_size, intermediate_size, num_experts, top_k);
  const Schedule schedule = SelectSchedule(shape);
  const int64_t workspace_bytes = CheckedWorkspaceSize(shape, schedule);
  const int32_t device_id = workspace_u8.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckSm103a(device_id);
  CheckWorkspace(workspace_u8, device_id, workspace_bytes);

  const cudaStream_t stream = get_current_stream();
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
  TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
      << "cake warp-decode workspace preparation must run outside CUDA Graph capture";
  CheckManifestStatus(generated::EnsureDeviceReady(device_id, true));

  const Invocation invocation{shape,   nullptr, workspace_u8.data_ptr(),
                              nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr,
                              nullptr, nullptr, TensorStorageBytes(workspace_u8, "workspace_u8")};
  return PrepareWorkspaceAlways(invocation, schedule, device_id, stream);
}

void Run(TensorView output_bf16, TensorView workspace_u8, TensorView hidden_states_q_u8,
         TensorView hidden_states_scale_e4m3, TensorView topk_ids_i32, TensorView topk_weights_bf16,
         TensorView gemm1_weights_u8, TensorView gemm1_weights_scale_e4m3,
         TensorView gemm2_weights_u8, TensorView gemm2_weights_scale_e4m3,
         TensorView output1_scale_scalar_f32, TensorView output1_scale_gate_scalar_f32,
         TensorView output2_scale_scalar_f32, int64_t workspace_receipt, bool enable_pdl) {
  TVM_FFI_ICHECK(enable_pdl)
      << "cake warp decode requires programmatic dependent launch; enable_pdl must be true";
  TVM_FFI_ICHECK(output_bf16.device().device_type == kDLCUDA)
      << "output_bf16 must be a CUDA tensor";
  TVM_FFI_ICHECK(output_bf16.ndim() == 2) << "output_bf16 must have rank 2";
  TVM_FFI_ICHECK(gemm2_weights_u8.ndim() == 3) << "gemm2_weights_u8 must have rank 3";
  TVM_FFI_ICHECK(topk_ids_i32.ndim() == 2) << "topk_ids_i32 must have rank 2";

  const int64_t intermediate_size = gemm2_weights_u8.size(2) * 2;
  const Shape shape = CheckedShape(output_bf16.size(0), output_bf16.size(1), intermediate_size,
                                   gemm2_weights_u8.size(0), topk_ids_i32.size(1));
  const Schedule schedule = SelectSchedule(shape);
  const int64_t workspace_bytes = CheckedWorkspaceSize(shape, schedule);

  const int32_t device_id = output_bf16.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckSm103a(device_id);
  CheckAllTensorArguments(output_bf16, workspace_u8, hidden_states_q_u8, hidden_states_scale_e4m3,
                          topk_ids_i32, topk_weights_bf16, gemm1_weights_u8,
                          gemm1_weights_scale_e4m3, gemm2_weights_u8, gemm2_weights_scale_e4m3,
                          output1_scale_scalar_f32, output1_scale_gate_scalar_f32,
                          output2_scale_scalar_f32, shape, workspace_bytes);

  const cudaStream_t stream = get_current_stream();
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
            "cudaStreamIsCapturing(workspace submission)");
  const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;
  CheckManifestStatus(generated::EnsureDeviceReady(device_id, false));

  const Invocation invocation =
      MakeInvocation(output_bf16, workspace_u8, hidden_states_q_u8, hidden_states_scale_e4m3,
                     topk_ids_i32, topk_weights_bf16, gemm1_weights_u8, gemm1_weights_scale_e4m3,
                     gemm2_weights_u8, gemm2_weights_scale_e4m3, output1_scale_scalar_f32,
                     output1_scale_gate_scalar_f32, output2_scale_scalar_f32, shape);
  // Keep the registry lock across the complete host-side submission transaction:
  // receipt validation, dependency insertion, kernel submission, and completion
  // recording. GPU execution stays asynchronous. External event nodes make the
  // dependency observable across CUDA Graph capture and replay without retaining
  // a borrowed raw stream handle.
  std::lock_guard<std::mutex> lock(prepared_workspaces_mutex);
  PreparedWorkspaceState& workspace =
      RequirePreparedWorkspace(invocation, schedule, device_id, workspace_receipt);
  if (workspace.has_submission) {
    CheckCuda(
        cudaStreamWaitEvent(stream, workspace.completion, is_capturing ? cudaEventWaitExternal : 0),
        "cudaStreamWaitEvent(previous workspace submission)");
  }
  LaunchContext context{stream, 0};
  try {
    generated::ForEachLaunch(invocation, schedule, LaunchOne, &context);
  } catch (...) {
    // A malformed manifest can fail after earlier launches were accepted. Make
    // those partial submissions visible to release/re-prepare before propagating
    // the original failure.
    if (context.launch_count > 0) {
      workspace.poisoned = true;
      const cudaError_t status =
          is_capturing
              ? cudaEventRecordWithFlags(workspace.completion, stream, cudaEventRecordExternal)
              : cudaEventRecord(workspace.completion, stream);
      if (status == cudaSuccess) {
        workspace.has_submission = true;
        workspace.poisoned = false;
      }
    }
    throw;
  }
  // Once GPU work has been accepted, loss of its retirement event makes the
  // address unsafe to overwrite or free. Poison before recording and clear
  // only after the external event is known to cover the complete submission.
  workspace.poisoned = true;
  const cudaError_t completion_status =
      is_capturing ? cudaEventRecordWithFlags(workspace.completion, stream, cudaEventRecordExternal)
                   : cudaEventRecord(workspace.completion, stream);
  if (completion_status == cudaSuccess) {
    workspace.has_submission = true;
    workspace.poisoned = false;
  }
  CheckCuda(completion_status, "cudaEventRecord(workspace completion)");
  const int32_t expected_launches = schedule.route_layout == RouteLayout::kDirect ? 3 : 4;
  TVM_FFI_ICHECK(context.launch_count == expected_launches)
      << "generated manifest emitted " << context.launch_count
      << " launches for a schedule that requires " << expected_launches;
}

}  // namespace flashinfer::warp_decode

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode_workspace_size,
                              flashinfer::warp_decode::WorkspaceSize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode_prepare_workspace,
                              flashinfer::warp_decode::PrepareWorkspace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode_release_workspace,
                              flashinfer::warp_decode::ReleaseWorkspaceReceipt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_moe_warp_decode, flashinfer::warp_decode::Run);
