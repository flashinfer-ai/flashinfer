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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

using tvm::ffi::Array;
using tvm::ffi::String;

namespace {

constexpr int kRequiredCudaVersion = 13040;
constexpr int kLocalityDomainCount = 2;

thread_local std::string g_last_error;

std::string CuError(CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &description));
  std::ostringstream os;
  os << (name == nullptr ? "CUDA driver error" : name) << " (" << static_cast<int>(result) << ")";
  if (description != nullptr) {
    os << ": " << description;
  }
  return os.str();
}

std::string RuntimeError(cudaError_t result) {
  std::ostringstream os;
  os << cudaGetErrorName(result) << " (" << static_cast<int>(result)
     << "): " << cudaGetErrorString(result);
  return os.str();
}

void SetLastError(std::string message) { g_last_error = std::move(message); }

class RuntimeDeviceGuard {
 public:
  explicit RuntimeDeviceGuard(int device) : requested_device_(device) {
    cudaError_t result = cudaGetDevice(&previous_device_);
    if (result != cudaSuccess) {
      error_ = "cudaGetDevice failed: " + RuntimeError(result);
      return;
    }
    if (previous_device_ != requested_device_) {
      result = cudaSetDevice(requested_device_);
      if (result != cudaSuccess) {
        error_ = "cudaSetDevice failed: " + RuntimeError(result);
        return;
      }
      restore_ = true;
    }
    // Ensure that PyTorch's primary context exists before using the driver API.
    result = cudaFree(nullptr);
    if (result != cudaSuccess) {
      error_ = "CUDA context initialization failed: " + RuntimeError(result);
      return;
    }
    valid_ = true;
  }

  RuntimeDeviceGuard(RuntimeDeviceGuard const&) = delete;
  RuntimeDeviceGuard& operator=(RuntimeDeviceGuard const&) = delete;

  ~RuntimeDeviceGuard() {
    if (restore_) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }

  bool valid() const { return valid_; }
  std::string const& error() const { return error_; }

 private:
  int requested_device_{};
  int previous_device_{};
  bool restore_{};
  bool valid_{};
  std::string error_;
};

#if CUDA_VERSION >= 13040

template <typename Proc>
bool LoadDriverProc(char const* name, Proc* proc, std::string* error) {
  void* address = nullptr;
  CUdriverProcAddressQueryResult query_result{};
  CUresult result = cuGetProcAddress(name, &address, kRequiredCudaVersion,
                                     CU_GET_PROC_ADDRESS_DEFAULT, &query_result);
  if (result != CUDA_SUCCESS || query_result != CU_GET_PROC_ADDRESS_SUCCESS || address == nullptr) {
    std::ostringstream os;
    os << "CUDA 13.4 driver entry point " << name << " is unavailable: " << CuError(result)
       << ", query status=" << static_cast<int>(query_result);
    *error = os.str();
    *proc = nullptr;
    return false;
  }
  *proc = reinterpret_cast<Proc>(address);
  return true;
}

struct GreenContextApi {
  using DeviceGetDevResource = CUresult(CUDAAPI*)(CUdevice device, CUdevResource* resource,
                                                  CUdevResourceType type);
  using DevSmResourceSplit = CUresult(CUDAAPI*)(CUdevResource* result, unsigned int nb_groups,
                                                CUdevResource const* input,
                                                CUdevResource* remainder, unsigned int flags,
                                                CU_DEV_SM_RESOURCE_GROUP_PARAMS* group_params);
  using DevResourceGenerateDesc = CUresult(CUDAAPI*)(CUdevResourceDesc* desc,
                                                     CUdevResource* resources,
                                                     unsigned int nb_resources);
  using GreenCtxCreate = CUresult(CUDAAPI*)(CUgreenCtx* green_context, CUdevResourceDesc desc,
                                            CUdevice device, unsigned int flags);
  using GreenCtxDestroy = CUresult(CUDAAPI*)(CUgreenCtx green_context);
  using GreenCtxStreamCreate = CUresult(CUDAAPI*)(CUstream* stream, CUgreenCtx green_context,
                                                  unsigned int flags, int priority);

  bool Load(std::string* error) {
    return LoadDriverProc("cuDeviceGetDevResource", &device_get_dev_resource, error) &&
           LoadDriverProc("cuDevSmResourceSplit", &dev_sm_resource_split, error) &&
           LoadDriverProc("cuDevResourceGenerateDesc", &dev_resource_generate_desc, error) &&
           LoadDriverProc("cuGreenCtxCreate", &green_ctx_create, error) &&
           LoadDriverProc("cuGreenCtxDestroy", &green_ctx_destroy, error) &&
           LoadDriverProc("cuGreenCtxStreamCreate", &green_ctx_stream_create, error);
  }

  DeviceGetDevResource device_get_dev_resource{};
  DevSmResourceSplit dev_sm_resource_split{};
  DevResourceGenerateDesc dev_resource_generate_desc{};
  GreenCtxCreate green_ctx_create{};
  GreenCtxDestroy green_ctx_destroy{};
  GreenCtxStreamCreate green_ctx_stream_create{};
};

std::string ProbeImpl(int device) {
  RuntimeDeviceGuard guard(device);
  if (!guard.valid()) {
    return guard.error();
  }

  int driver_version = 0;
  CUresult result = cuDriverGetVersion(&driver_version);
  if (result != CUDA_SUCCESS) {
    return "cuDriverGetVersion failed: " + CuError(result);
  }
  if (driver_version < kRequiredCudaVersion) {
    std::ostringstream os;
    os << "locality domains require a CUDA 13.4-compatible driver, but cuDriverGetVersion "
          "reported "
       << driver_version;
    return os.str();
  }

  CUdevice cu_device{};
  result = cuDeviceGet(&cu_device, device);
  if (result != CUDA_SUCCESS) {
    return "cuDeviceGet failed for CUDA device " + std::to_string(device) + ": " + CuError(result);
  }

  int domain_count = 0;
  result =
      cuDeviceGetAttribute(&domain_count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device);
  if (result != CUDA_SUCCESS) {
    return "failed to query CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT: " + CuError(result);
  }
  if (domain_count < kLocalityDomainCount) {
    std::ostringstream os;
    os << "locality domains require at least " << kLocalityDomainCount << " domains, but device "
       << device << " exposes " << domain_count;
    return os.str();
  }

  int domain_sm_count = 0;
  result = cuDeviceGetAttribute(
      &domain_sm_count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, cu_device);
  if (result != CUDA_SUCCESS) {
    return "failed to query CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT: " +
           CuError(result);
  }
  if (domain_sm_count <= 0) {
    return "device reported a non-positive locality-domain multiprocessor count";
  }

  GreenContextApi api;
  std::string error;
  if (!api.Load(&error)) {
    return error;
  }
  return {};
}

struct StreamState {
  std::array<CUdevResource, kLocalityDomainCount> resources{};
  std::array<CUdevResourceDesc, kLocalityDomainCount> descriptors{};
  std::array<CUgreenCtx, kLocalityDomainCount> contexts{};
  std::array<CUstream, kLocalityDomainCount> streams{};
  std::array<unsigned int, kLocalityDomainCount> sm_counts{};
  unsigned int remainder_sm_count{};
  unsigned int sm_coscheduled_alignment{};
};

using StreamKey = std::pair<int, bool>;

std::map<StreamKey, StreamState*>& StreamStates() {
  // CUDA objects may be torn down after Python finalization starts. Keeping the
  // state alive until process exit avoids destructor-order races with PyTorch.
  static auto* states = new std::map<StreamKey, StreamState*>;
  return *states;
}

std::mutex& StreamStatesMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

void DestroyPartialStreamState(StreamState* state, GreenContextApi const& api) {
  for (CUstream& stream : state->streams) {
    if (stream != nullptr) {
      static_cast<void>(cuStreamDestroy(stream));
      stream = nullptr;
    }
  }
  if (api.green_ctx_destroy != nullptr) {
    for (CUgreenCtx& context : state->contexts) {
      if (context != nullptr) {
        static_cast<void>(api.green_ctx_destroy(context));
        context = nullptr;
      }
    }
  }
}

StreamState* CreateStreamState(int device, bool backfill, std::string* error) {
  RuntimeDeviceGuard guard(device);
  if (!guard.valid()) {
    *error = guard.error();
    return nullptr;
  }

  CUdevice cu_device{};
  CUresult result = cuDeviceGet(&cu_device, device);
  if (result != CUDA_SUCCESS) {
    *error = "cuDeviceGet failed: " + CuError(result);
    return nullptr;
  }

  int domain_sm_count = 0;
  result = cuDeviceGetAttribute(
      &domain_sm_count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, cu_device);
  if (result != CUDA_SUCCESS || domain_sm_count <= 0) {
    *error = "failed to query locality-domain SM count: " + CuError(result);
    return nullptr;
  }

  GreenContextApi api;
  if (!api.Load(error)) {
    return nullptr;
  }

  CUdevResource full_resource{};
  result = api.device_get_dev_resource(cu_device, &full_resource, CU_DEV_RESOURCE_TYPE_SM);
  if (result != CUDA_SUCCESS) {
    *error = "cuDeviceGetDevResource failed: " + CuError(result);
    return nullptr;
  }
  if (full_resource.type != CU_DEV_RESOURCE_TYPE_SM || full_resource.sm.smCount == 0) {
    *error = "device returned an invalid full-device SM resource";
    return nullptr;
  }

  unsigned int const total_sm_count = full_resource.sm.smCount;
  unsigned int const sm_alignment =
      full_resource.sm.smCoscheduledAlignment == 0 ? 2 : full_resource.sm.smCoscheduledAlignment;
  if (backfill && (total_sm_count < 4 || total_sm_count % 4 != 0)) {
    std::ostringstream os;
    os << "backfill locality split requires a total SM count divisible by 4 and at least 4, got "
       << total_sm_count << "; use sm_split=\"strict\"";
    *error = os.str();
    return nullptr;
  }
  if (!backfill &&
      static_cast<unsigned int>(kLocalityDomainCount * domain_sm_count) > total_sm_count) {
    std::ostringstream os;
    os << "strict locality split reported " << domain_sm_count
       << " SMs per domain, which exceeds total_sm_count=" << total_sm_count;
    *error = os.str();
    return nullptr;
  }
  if (!backfill && static_cast<unsigned int>(domain_sm_count) < sm_alignment) {
    std::ostringstream os;
    os << "strict locality split cannot form a valid group from " << domain_sm_count
       << " SMs per domain with co-schedule alignment " << sm_alignment;
    *error = os.str();
    return nullptr;
  }

  CUdevResource workqueue_config{};
  result = api.device_get_dev_resource(cu_device, &workqueue_config,
                                       CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG);
  if (result != CUDA_SUCCESS) {
    *error = "cuDeviceGetDevResource(WORKQUEUE_CONFIG) failed: " + CuError(result);
    return nullptr;
  }
  if (workqueue_config.type != CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG) {
    *error = "device returned an invalid workqueue configuration resource";
    return nullptr;
  }
  workqueue_config.wqConfig.sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED;

  auto* state = new StreamState;
  state->sm_coscheduled_alignment = sm_alignment;
  CUdevResource remainder{};
  std::array<CU_DEV_SM_RESOURCE_GROUP_PARAMS, kLocalityDomainCount> group_params{};
  for (int domain = 0; domain < kLocalityDomainCount; ++domain) {
    group_params[domain].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
    group_params[domain].localityDomainId = static_cast<unsigned int>(domain);
    if (backfill) {
      group_params[domain].smCount =
          total_sm_count / static_cast<unsigned int>(kLocalityDomainCount);
      group_params[domain].flags |= CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
    } else {
      group_params[domain].coscheduledSmCount = sm_alignment;
    }
  }

  result = api.dev_sm_resource_split(state->resources.data(), kLocalityDomainCount, &full_resource,
                                     &remainder, /*flags=*/0, group_params.data());
  if (result != CUDA_SUCCESS) {
    std::ostringstream os;
    os << (backfill ? "backfill" : "strict")
       << " locality-domain SM split failed for total_sm_count=" << total_sm_count << ": "
       << CuError(result);
    if (backfill) {
      os << "; use sm_split=\"strict\"";
    }
    *error = os.str();
    delete state;
    return nullptr;
  }

  state->remainder_sm_count = remainder.type == CU_DEV_RESOURCE_TYPE_SM ? remainder.sm.smCount : 0;
  for (int domain = 0; domain < kLocalityDomainCount; ++domain) {
    CUdevResource const& resource = state->resources[domain];
    unsigned int const actual_sm_count = resource.sm.smCount;
    bool const count_is_valid =
        backfill
            ? actual_sm_count == total_sm_count / static_cast<unsigned int>(kLocalityDomainCount)
            : actual_sm_count >= sm_alignment && actual_sm_count % sm_alignment == 0 &&
                  actual_sm_count <= static_cast<unsigned int>(domain_sm_count);
    state->sm_counts[domain] = resource.sm.smCount;
    if (resource.type != CU_DEV_RESOURCE_TYPE_SM || !count_is_valid ||
        resource.sm.localityDomainId != static_cast<unsigned int>(domain) ||
        (resource.sm.flags & CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID) == 0) {
      std::ostringstream os;
      os << (backfill ? "backfill" : "strict")
         << " split returned an invalid resource for locality domain " << domain
         << ": total=" << total_sm_count << ", expected=";
      if (backfill) {
        os << total_sm_count / static_cast<unsigned int>(kLocalityDomainCount);
      } else {
        os << "aligned multiple of " << sm_alignment << " in [" << sm_alignment << ", "
           << domain_sm_count << ']';
      }
      os << ", actual=" << resource.sm.smCount
         << ", locality_domain=" << resource.sm.localityDomainId << ", flags=" << resource.sm.flags;
      *error = os.str();
      delete state;
      return nullptr;
    }
  }
  if (!backfill && state->sm_counts[0] != state->sm_counts[1]) {
    std::ostringstream os;
    os << "strict split returned asymmetric locality-domain groups: domain0=" << state->sm_counts[0]
       << ", domain1=" << state->sm_counts[1] << ", co-schedule alignment=" << sm_alignment;
    *error = os.str();
    delete state;
    return nullptr;
  }

  unsigned int const expected_remainder =
      backfill ? 0 : total_sm_count - state->sm_counts[0] - state->sm_counts[1];
  if (state->remainder_sm_count != expected_remainder) {
    std::ostringstream os;
    os << (backfill ? "backfill" : "strict")
       << " split returned an unexpected remainder: total=" << total_sm_count
       << ", domain0=" << state->sm_counts[0] << ", domain1=" << state->sm_counts[1]
       << ", remainder=" << state->remainder_sm_count << ", expected=" << expected_remainder;
    *error = os.str();
    delete state;
    return nullptr;
  }

  for (int domain = 0; domain < kLocalityDomainCount; ++domain) {
    std::array<CUdevResource, 2> descriptor_resources = {state->resources[domain],
                                                         workqueue_config};
    result = api.dev_resource_generate_desc(
        &state->descriptors[domain], descriptor_resources.data(),
        /*nb_resources=*/static_cast<unsigned int>(descriptor_resources.size()));
    if (result == CUDA_SUCCESS) {
      result = api.green_ctx_create(&state->contexts[domain], state->descriptors[domain], cu_device,
                                    CU_GREEN_CTX_DEFAULT_STREAM);
    }
    if (result == CUDA_SUCCESS) {
      result = api.green_ctx_stream_create(&state->streams[domain], state->contexts[domain],
                                           CU_STREAM_NON_BLOCKING, /*priority=*/0);
    }
    if (result != CUDA_SUCCESS) {
      std::ostringstream os;
      os << "failed to create green-context stream for locality domain " << domain << ": "
         << CuError(result);
      *error = os.str();
      DestroyPartialStreamState(state, api);
      delete state;
      return nullptr;
    }
  }
  return state;
}

struct AllocationKey {
  int device{};
  CUdeviceptr pointer{};

  bool operator<(AllocationKey const& other) const {
    return device < other.device || (device == other.device && pointer < other.pointer);
  }
};

struct AllocationRecord {
  size_t aligned_size{};
  CUcontext context{};
  bool mapped{true};
};

std::map<AllocationKey, AllocationRecord>& Allocations() {
  static auto* allocations = new std::map<AllocationKey, AllocationRecord>;
  return *allocations;
}

std::mutex& AllocationsMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

void* LocalizedAlloc(size_t size, int device, int locality_domain) {
  g_last_error.clear();
  if (size == 0) {
    return nullptr;
  }

  RuntimeDeviceGuard guard(device);
  if (!guard.valid()) {
    SetLastError(guard.error());
    return nullptr;
  }

  CUcontext context{};
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS || context == nullptr) {
    SetLastError("failed to resolve the current CUDA context: " + CuError(result));
    return nullptr;
  }

  CUdevice cu_device{};
  result = cuDeviceGet(&cu_device, device);
  if (result != CUDA_SUCCESS) {
    SetLastError("cuDeviceGet failed: " + CuError(result));
    return nullptr;
  }

  int domain_count = 0;
  result =
      cuDeviceGetAttribute(&domain_count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device);
  if (result != CUDA_SUCCESS || locality_domain < 0 || locality_domain >= domain_count) {
    SetLastError("invalid locality domain " + std::to_string(locality_domain) +
                 " for CUDA device " + std::to_string(device));
    return nullptr;
  }
  if (device < 0 || device > std::numeric_limits<unsigned char>::max() ||
      locality_domain > std::numeric_limits<unsigned char>::max()) {
    SetLastError("device or locality-domain ordinal exceeds the CUDA 13.4 VMM field width");
    return nullptr;
  }

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  prop.location.localized.deviceId = static_cast<unsigned char>(device);
  prop.location.localized.localityDomainId = static_cast<unsigned char>(locality_domain);

  size_t granularity = 0;
  result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (result != CUDA_SUCCESS || granularity == 0) {
    SetLastError("cuMemGetAllocationGranularity failed: " + CuError(result));
    return nullptr;
  }
  if (size > std::numeric_limits<size_t>::max() - (granularity - 1)) {
    SetLastError("localized allocation size overflows the CUDA VMM granularity");
    return nullptr;
  }
  size_t const aligned_size = ((size + granularity - 1) / granularity) * granularity;

  CUdeviceptr address{};
  CUmemGenericAllocationHandle handle{};
  bool mapped = false;
  result = cuMemAddressReserve(&address, aligned_size, /*alignment=*/0, /*addr=*/0, /*flags=*/0);
  if (result == CUDA_SUCCESS) {
    result = cuMemCreate(&handle, aligned_size, &prop, /*flags=*/0);
  }
  if (result == CUDA_SUCCESS) {
    result = cuMemMap(address, aligned_size, /*offset=*/0, handle, /*flags=*/0);
    mapped = result == CUDA_SUCCESS;
  }
  if (result == CUDA_SUCCESS) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    result = cuMemSetAccess(address, aligned_size, &access, /*count=*/1);
  }
  if (result == CUDA_SUCCESS) {
    result = cuMemRelease(handle);
    if (result == CUDA_SUCCESS) {
      handle = 0;
    }
  }
  if (result == CUDA_SUCCESS) {
    try {
      std::lock_guard<std::mutex> lock(AllocationsMutex());
      bool const inserted =
          Allocations()
              .emplace(AllocationKey{device, address}, AllocationRecord{aligned_size, context})
              .second;
      if (!inserted) {
        result = CUDA_ERROR_INVALID_VALUE;
      }
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }

  if (result != CUDA_SUCCESS) {
    if (mapped) {
      static_cast<void>(cuMemUnmap(address, aligned_size));
    }
    if (handle != 0) {
      static_cast<void>(cuMemRelease(handle));
    }
    if (address != 0) {
      static_cast<void>(cuMemAddressFree(address, aligned_size));
    }
    SetLastError("localized VMM allocation in domain " + std::to_string(locality_domain) +
                 " failed: " + CuError(result));
    return nullptr;
  }
  return reinterpret_cast<void*>(address);
}

void LocalizedFree(void* ptr, int device) {
  g_last_error.clear();
  if (ptr == nullptr) {
    return;
  }

  RuntimeDeviceGuard guard(device);
  if (!guard.valid()) {
    SetLastError(guard.error());
    return;
  }

  CUdeviceptr const address = reinterpret_cast<CUdeviceptr>(ptr);
  std::lock_guard<std::mutex> lock(AllocationsMutex());
  auto& allocations = Allocations();
  auto it = allocations.find(AllocationKey{device, address});
  if (it == allocations.end()) {
    SetLastError("localized allocator received an unknown pointer");
    return;
  }
  AllocationRecord& record = it->second;

  CUcontext current_context{};
  CUresult result = cuCtxGetCurrent(&current_context);
  if (result == CUDA_ERROR_DEINITIALIZED) {
    allocations.erase(it);
    return;
  }
  if (result != CUDA_SUCCESS || current_context != record.context) {
    SetLastError("cannot free localized memory from a different CUDA context");
    return;
  }
  if (record.mapped) {
    result = cuMemUnmap(address, record.aligned_size);
    if (result == CUDA_SUCCESS) {
      record.mapped = false;
    }
  }
  if (result == CUDA_SUCCESS && !record.mapped) {
    result = cuMemAddressFree(address, record.aligned_size);
  }
  if (result == CUDA_SUCCESS || result == CUDA_ERROR_DEINITIALIZED) {
    allocations.erase(it);
  } else {
    SetLastError("localized VMM free failed: " + CuError(result));
  }
}

#endif  // CUDA_VERSION >= 13040

String Probe(int64_t device) {
#if CUDA_VERSION >= 13040
  return String(ProbeImpl(static_cast<int>(device)));
#else
  std::ostringstream os;
  os << "locality domains require CUDA 13.4 headers, but this module was built with CUDA "
     << CUDA_VERSION;
  return String(os.str());
#endif
}

int64_t DomainCount(int64_t device) {
#if CUDA_VERSION >= 13040
  g_last_error.clear();
  RuntimeDeviceGuard guard(static_cast<int>(device));
  if (!guard.valid()) {
    SetLastError(guard.error());
    return -1;
  }
  CUdevice cu_device{};
  int count = 0;
  CUresult result = cuDeviceGet(&cu_device, static_cast<int>(device));
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device);
  }
  if (result != CUDA_SUCCESS) {
    SetLastError("failed to query locality-domain count: " + CuError(result));
    return -1;
  }
  return count;
#else
  static_cast<void>(device);
  return 0;
#endif
}

Array<int64_t> CreateLocalizedStreams(int64_t device, bool backfill) {
#if CUDA_VERSION >= 13040
  std::string const unsupported = ProbeImpl(static_cast<int>(device));
  TVM_FFI_ICHECK(unsupported.empty()) << unsupported;

  std::lock_guard<std::mutex> lock(StreamStatesMutex());
  StreamKey const key{static_cast<int>(device), backfill};
  StreamState* state = nullptr;
  auto& states = StreamStates();
  auto it = states.find(key);
  if (it == states.end()) {
    std::string error;
    state = CreateStreamState(static_cast<int>(device), backfill, &error);
    TVM_FFI_ICHECK(state != nullptr) << error;
    states.emplace(key, state);
  } else {
    state = it->second;
  }
  Array<int64_t> result;
  result.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(state->streams[0])));
  result.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(state->streams[1])));
  result.push_back(static_cast<int64_t>(state->sm_counts[0]));
  result.push_back(static_cast<int64_t>(state->sm_counts[1]));
  result.push_back(static_cast<int64_t>(state->remainder_sm_count));
  result.push_back(static_cast<int64_t>(state->sm_coscheduled_alignment));
  return result;
#else
  static_cast<void>(device);
  static_cast<void>(backfill);
  TVM_FFI_ICHECK(false) << "locality domains require CUDA 13.4 headers; module CUDA_VERSION="
                        << CUDA_VERSION;
  return {};
#endif
}

int64_t PointerDomain(int64_t pointer) {
#if CUDA_VERSION >= 13040
  int locality_domain = -1;
  CUresult result =
      cuPointerGetAttribute(&locality_domain, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                            static_cast<CUdeviceptr>(pointer));
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL) failed: "
      << CuError(result);
  return static_cast<int64_t>(locality_domain);
#else
  static_cast<void>(pointer);
  TVM_FFI_ICHECK(false) << "pointer locality queries require CUDA 13.4 headers";
  return -1;
#endif
}

String LastError() { return String(g_last_error); }

}  // namespace

extern "C" void* flashinfer_locality_domain0_alloc(size_t size, int device,
                                                   cudaStream_t stream) noexcept {
  static_cast<void>(stream);
#if CUDA_VERSION >= 13040
  try {
    return LocalizedAlloc(size, device, 0);
  } catch (std::exception const& error) {
    try {
      SetLastError(std::string("localized allocation callback failed: ") + error.what());
    } catch (...) {
    }
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  static_cast<void>(size);
  static_cast<void>(device);
  SetLastError("locality domains require CUDA 13.4 headers");
  return nullptr;
#endif
}

extern "C" void* flashinfer_locality_domain1_alloc(size_t size, int device,
                                                   cudaStream_t stream) noexcept {
  static_cast<void>(stream);
#if CUDA_VERSION >= 13040
  try {
    return LocalizedAlloc(size, device, 1);
  } catch (std::exception const& error) {
    try {
      SetLastError(std::string("localized allocation callback failed: ") + error.what());
    } catch (...) {
    }
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  static_cast<void>(size);
  static_cast<void>(device);
  SetLastError("locality domains require CUDA 13.4 headers");
  return nullptr;
#endif
}

extern "C" void flashinfer_locality_domain_free(void* ptr, size_t size, int device,
                                                cudaStream_t stream) noexcept {
  static_cast<void>(size);
  static_cast<void>(stream);
#if CUDA_VERSION >= 13040
  try {
    LocalizedFree(ptr, device);
  } catch (std::exception const& error) {
    try {
      SetLastError(std::string("localized free callback failed: ") + error.what());
    } catch (...) {
    }
  } catch (...) {
  }
#else
  static_cast<void>(ptr);
  static_cast<void>(device);
#endif
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(probe, Probe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(domain_count, DomainCount);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_localized_streams, CreateLocalizedStreams);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pointer_domain, PointerDomain);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(last_error, LastError);
