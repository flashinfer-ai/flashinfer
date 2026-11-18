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

// Split locality-domain allocation of the Cake locality runtime: one virtual
// address range whose bytes [0, cut) are memory of locality domain 0 and whose
// bytes [cut, size) are memory of locality domain 1. The allocation is returned
// as a uint8 tensor whose deleter unmaps and releases it, and it stays in a
// registry keyed by its base pointer until then.

#include <cuda.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#if CUDA_VERSION < 13040
#error "the Cake locality runtime needs CUDA 13.4 or newer headers"
#endif

using tvm::ffi::Array;
using tvm::ffi::Tensor;

namespace {

constexpr int kRequiredCudaVersion = 13040;
constexpr int kLocalityDomainCount = 2;

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

// Makes `device` current (and its primary context exist) for the scope; on exit
// the thread's state is as it was on entry. Only the driver is used: the
// runtime's selected device of the thread is never changed.
//
// When the caller's current context (also a non-primary one) is on `device`, it
// stays current. Otherwise (another device's context, or none, for example in a
// garbage-collection thread) `device`'s primary context is retained through the
// driver and made current, and on exit the caller's context is made current
// again (or none). Selecting `device` through the runtime instead would leave
// the runtime reporting `device` after a caller that pushed its own context pops
// it, while its next call binds another device's primary context; callers that
// skip cudaSetDevice when the reported device already matches would then
// allocate on that device. The first primary context reference taken this way
// for a device is kept for the process lifetime, so the context (and the context
// id a split records) outlives the scope even when the runtime holds no
// reference to it.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    if (cuCtxGetCurrent(&previous_context_) != CUDA_SUCCESS) {
      previous_context_ = nullptr;
    }
    if (previous_context_ != nullptr) {
      CUdevice current{};
      CUdevice wanted{};
      if (cuCtxGetDevice(&current) == CUDA_SUCCESS &&
          cuDeviceGet(&wanted, device) == CUDA_SUCCESS && current == wanted) {
        ok_ = true;  // the caller's context is on `device`: keep it
        return;
      }
    }
    EnterPrimary(device);
  }
  DeviceScope(DeviceScope const&) = delete;
  DeviceScope& operator=(DeviceScope const&) = delete;
  ~DeviceScope() {
    if (restore_) {
      // nullptr restores a thread that had no current context.
      static_cast<void>(cuCtxSetCurrent(previous_context_));
    }
    if (release_) {
      static_cast<void>(cuDevicePrimaryCtxRelease(primary_device_));
    }
  }
  bool ok() const { return ok_; }

 private:
  void EnterPrimary(int device) {
    CUcontext primary{};
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&primary_device_, device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&primary, primary_device_) != CUDA_SUCCESS) {
      return;
    }
    release_ = !KeepFirstReference(device);
    restore_ = true;
    ok_ = cuCtxSetCurrent(primary) == CUDA_SUCCESS;
  }

  // True the first time it is called for `device` in the process.
  static bool KeepFirstReference(int device) {
    static auto* mutex = new std::mutex;
    static auto* kept = new std::set<int>;
    std::lock_guard<std::mutex> lock(*mutex);
    return kept->insert(device).second;
  }

  CUcontext previous_context_{};
  CUdevice primary_device_{};
  bool ok_{};
  bool restore_{};
  bool release_{};
};

template <typename Proc>
void LoadDriverProc(char const* name, Proc* proc) {
  void* address = nullptr;
  CUdriverProcAddressQueryResult query_result{};
  CUresult result = cuGetProcAddress(name, &address, kRequiredCudaVersion,
                                     CU_GET_PROC_ADDRESS_DEFAULT, &query_result);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && query_result == CU_GET_PROC_ADDRESS_SUCCESS &&
                 address != nullptr)
      << "CUDA 13.4 driver entry point " << name << " is unavailable: " << CuError(result)
      << ", query status=" << static_cast<int>(query_result);
  *proc = reinterpret_cast<Proc>(address);
}

struct VmmApi {
  using AddressReserve = CUresult(CUDAAPI*)(CUdeviceptr* ptr, size_t size, size_t alignment,
                                            CUdeviceptr addr, unsigned long long flags);
  using AddressFree = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size);
  using Create = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle* handle, size_t size,
                                    CUmemAllocationProp const* prop, unsigned long long flags);
  using Release = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle handle);
  using Map = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size, size_t offset,
                                 CUmemGenericAllocationHandle handle, unsigned long long flags);
  using Unmap = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size);
  using SetAccess = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size, CUmemAccessDesc const* desc,
                                       size_t count);
  using Granularity = CUresult(CUDAAPI*)(size_t* granularity, CUmemAllocationProp const* prop,
                                         CUmemAllocationGranularity_flags option);

  VmmApi() {
    LoadDriverProc("cuMemAddressReserve", &address_reserve);
    LoadDriverProc("cuMemAddressFree", &address_free);
    LoadDriverProc("cuMemCreate", &create);
    LoadDriverProc("cuMemRelease", &release);
    LoadDriverProc("cuMemMap", &map);
    LoadDriverProc("cuMemUnmap", &unmap);
    LoadDriverProc("cuMemSetAccess", &set_access);
    LoadDriverProc("cuMemGetAllocationGranularity", &granularity);
  }

  AddressReserve address_reserve{};
  AddressFree address_free{};
  Create create{};
  Release release{};
  Map map{};
  Unmap unmap{};
  SetAccess set_access{};
  Granularity granularity{};
};

VmmApi const& Api() {
  static VmmApi const api;
  return api;
}

CUmemAllocationProp DomainProp(int device, int domain) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  prop.location.localized.deviceId = static_cast<unsigned char>(device);
  prop.location.localized.localityDomainId = static_cast<unsigned char>(domain);
  return prop;
}

void CheckDevice(int device) {
  TVM_FFI_ICHECK(device >= 0 && device <= std::numeric_limits<unsigned char>::max())
      << "device ordinal " << device << " exceeds the CUDA VMM field width";
  CUdevice cu_device{};
  CUresult result = cuDeviceGet(&cu_device, device);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS) << "cuDeviceGet failed: " << CuError(result);
  int domain_count = 0;
  result =
      cuDeviceGetAttribute(&domain_count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "failed to query CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT: " << CuError(result);
  TVM_FFI_ICHECK(domain_count >= kLocalityDomainCount)
      << "device " << device << " exposes " << domain_count << " locality domains; "
      << kLocalityDomainCount << " are required";
}

// The larger of the two domains' minimum granularities.
size_t Granularity(int device) {
  size_t unit = 0;
  for (int domain = 0; domain < kLocalityDomainCount; ++domain) {
    CUmemAllocationProp const prop = DomainProp(device, domain);
    size_t granularity = 0;
    CUresult result = Api().granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    TVM_FFI_ICHECK(result == CUDA_SUCCESS && granularity != 0)
        << "cuMemGetAllocationGranularity failed for locality domain " << domain << ": "
        << CuError(result);
    unit = granularity > unit ? granularity : unit;
  }
  return unit;
}

struct Split {
  int device{};
  size_t nbytes{};    // bytes the tensor holds
  size_t size{};      // reserved bytes: nbytes rounded up to the granularity
  size_t cut{};       // bytes [0, cut) from the base are on domain 0
  size_t boundary{};  // physical boundary: cut, or size when cut == nbytes
  CUcontext context{};
  unsigned long long context_id{};  // cuCtxGetId of `context`: unique, unlike a handle
};

std::map<CUdeviceptr, Split>& Splits() {
  // Kept until process exit, as the locality_domain allocator does, so that late
  // deleters never touch a destroyed map.
  static auto* splits = new std::map<CUdeviceptr, Split>;
  return *splits;
}

std::mutex& SplitsMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

// The (offset, size, domain) pieces of a split allocation.
std::array<std::array<size_t, 3>, kLocalityDomainCount> Pieces(size_t boundary, size_t size) {
  return {{{0, boundary, 0}, {boundary, size - boundary, 1}}};
}

CUdeviceptr SplitAlloc(int device, size_t nbytes, size_t cut) {
  TVM_FFI_ICHECK(nbytes > 0) << "a split allocation needs at least one byte";
  TVM_FFI_ICHECK(cut <= nbytes) << "cut " << cut << " exceeds the allocation of " << nbytes
                                << " bytes";
  DeviceScope scope(device);
  TVM_FFI_ICHECK(scope.ok()) << "cannot make CUDA device " << device << " current";
  CheckDevice(device);
  CUcontext context{};
  CUresult result = cuCtxGetCurrent(&context);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && context != nullptr)
      << "failed to resolve the current CUDA context: " << CuError(result);
  unsigned long long context_id = 0;
  result = cuCtxGetId(context, &context_id);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "failed to read the current CUDA context id: " << CuError(result);

  size_t const unit = Granularity(device);
  TVM_FFI_ICHECK(nbytes <= std::numeric_limits<size_t>::max() - (unit - 1))
      << "split allocation size overflows the CUDA VMM granularity";
  size_t const size = (nbytes + unit - 1) / unit * unit;
  TVM_FFI_ICHECK(cut == nbytes || cut % unit == 0)
      << "cut " << cut << " is not a multiple of the allocation granularity " << unit;
  size_t const boundary = cut == nbytes ? size : cut;

  VmmApi const& api = Api();
  CUdeviceptr address{};
  result = api.address_reserve(&address, size, /*alignment=*/0, /*addr=*/0, /*flags=*/0);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS) << "cuMemAddressReserve failed: " << CuError(result);

  auto const pieces = Pieces(boundary, size);
  std::array<bool, kLocalityDomainCount> mapped{};
  for (size_t index = 0; index < pieces.size() && result == CUDA_SUCCESS; ++index) {
    size_t const offset = pieces[index][0];
    size_t const length = pieces[index][1];
    int const domain = static_cast<int>(pieces[index][2]);
    if (length == 0) {
      continue;
    }
    CUmemAllocationProp const prop = DomainProp(device, domain);
    CUmemGenericAllocationHandle handle{};
    result = api.create(&handle, length, &prop, /*flags=*/0);
    if (result != CUDA_SUCCESS) {
      break;
    }
    result = api.map(address + offset, length, /*offset=*/0, handle, /*flags=*/0);
    mapped[index] = result == CUDA_SUCCESS;
    // The mapping keeps the memory; it is freed when the range is unmapped.
    CUresult const released = api.release(handle);
    if (result == CUDA_SUCCESS) {
      result = released;
    }
  }
  if (result == CUDA_SUCCESS) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    result = api.set_access(address, size, &access, /*count=*/1);
  }
  if (result == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(SplitsMutex());
    try {
      bool const inserted =
          Splits()
              .emplace(address, Split{device, nbytes, size, cut, boundary, context, context_id})
              .second;
      if (!inserted) {
        result = CUDA_ERROR_INVALID_VALUE;
      }
    } catch (...) {
      // No registry entry (std::bad_alloc): release the mapped range below.
      result = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  if (result != CUDA_SUCCESS) {
    for (size_t index = 0; index < pieces.size(); ++index) {
      if (mapped[index]) {
        static_cast<void>(api.unmap(address + pieces[index][0], pieces[index][1]));
      }
    }
    static_cast<void>(api.address_free(address, size));
    TVM_FFI_THROW(RuntimeError) << "split locality-domain allocation of " << nbytes
                                << " bytes (cut " << cut << ") failed: " << CuError(result);
  }
  return address;
}

// Makes `context` current for the scope when another context is current, and
// makes the caller's context current again on exit.
class ContextScope {
 public:
  ContextScope() = default;
  ContextScope(ContextScope const&) = delete;
  ContextScope& operator=(ContextScope const&) = delete;
  bool Push(CUcontext context) {
    pushed_ = cuCtxPushCurrent(context) == CUDA_SUCCESS;
    return pushed_;
  }
  ~ContextScope() {
    if (pushed_) {
      CUcontext popped{};
      static_cast<void>(cuCtxPopCurrent(&popped));
    }
  }

 private:
  bool pushed_{};
};

// Frees a split allocation; called by the tensor deleter, so it never throws.
// It runs in the context that made the allocation when that context still exists
// (when another context is current, that context is synchronized first and is
// current again afterwards).
// When it was destroyed, the range is unmapped and released in the device's
// current context: VMM memory belongs to the process, not to a context, and the
// stale context handle (which may name another context now) is never made
// current. The registry lock is not held while the contexts synchronize. Work of
// any other context (neither the one that made the allocation nor the one that
// frees it) that reads or writes the allocation must be complete before its
// last tensor is freed. A CUDA graph that still reads the allocation faults on
// its next replay, as the range is unmapped.
void SplitFree(CUdeviceptr address) noexcept {
  try {
    Split split{};
    {
      std::lock_guard<std::mutex> lock(SplitsMutex());
      auto it = Splits().find(address);
      if (it == Splits().end()) {
        return;
      }
      split = it->second;
    }
    DeviceScope scope(split.device);
    CUcontext current{};
    CUresult result = cuCtxGetCurrent(&current);
    if (result == CUDA_ERROR_DEINITIALIZED || !scope.ok()) {
      std::lock_guard<std::mutex> lock(SplitsMutex());
      Splits().erase(address);
      return;
    }
    unsigned long long current_id = 0;
    if (result == CUDA_SUCCESS && current != nullptr) {
      result = cuCtxGetId(current, &current_id);
    }
    if (result != CUDA_SUCCESS || current == nullptr) {
      return;  // no current context to read: leave it mapped rather than corrupt it
    }
    ContextScope context_scope;  // declared after `scope`: restored before the device
    // Ids, not handles, are compared: a destroyed context's handle can name a new one.
    if (current_id != split.context_id) {
      unsigned long long id = 0;
      bool const alive = cuCtxGetId(split.context, &id) == CUDA_SUCCESS && id == split.context_id;
      if (alive) {
        // The freeing context's work may read the memory too: wait for it here,
        // and for the allocation's context below.
        static_cast<void>(cuCtxSynchronize());
        if (!context_scope.Push(split.context)) {
          return;  // cannot enter the allocation's context: leave it mapped
        }
      }
      // Not alive: the allocation's context was destroyed; free in the current one.
    }
    {
      // The range leaves the registry before it is unmapped: no lookup finds it.
      std::lock_guard<std::mutex> lock(SplitsMutex());
      Splits().erase(address);
    }
    // As cudaFree does: no kernel still in flight may read the memory being unmapped.
    static_cast<void>(cuCtxSynchronize());
    VmmApi const& api = Api();
    for (auto const& piece : Pieces(split.boundary, split.size)) {
      if (piece[1] != 0) {
        static_cast<void>(api.unmap(address + piece[0], piece[1]));
      }
    }
    static_cast<void>(api.address_free(address, split.size));
  } catch (...) {
  }
}

struct SplitNDAlloc {
  CUdeviceptr address{};
  void AllocData(DLTensor* tensor) { tensor->data = reinterpret_cast<void*>(address); }
  void FreeData(DLTensor* tensor) noexcept {
    SplitFree(reinterpret_cast<CUdeviceptr>(tensor->data));
  }
};

Tensor SplitEmpty(int64_t nbytes, int64_t cut, int64_t device) {
  TVM_FFI_ICHECK(nbytes > 0 && cut >= 0 && cut <= nbytes)
      << "invalid split allocation: nbytes=" << nbytes << ", cut=" << cut;
  CUdeviceptr const address =
      SplitAlloc(static_cast<int>(device), static_cast<size_t>(nbytes), static_cast<size_t>(cut));
  DLDataType const dtype{kDLUInt, 8, 1};
  DLDevice const dl_device{kDLCUDA, static_cast<int32_t>(device)};
  try {
    return Tensor::FromNDAlloc(SplitNDAlloc{address}, tvm::ffi::Shape({nbytes}), dtype, dl_device);
  } catch (...) {
    // No tensor owns the allocation (its deleter never ran): release it here.
    SplitFree(address);
    throw;
  }
}

// [device, nbytes, size, cut] of the split allocation based at `pointer`, or [].
Array<int64_t> SplitInfo(int64_t pointer) {
  Array<int64_t> info;
  std::lock_guard<std::mutex> lock(SplitsMutex());
  auto const& splits = Splits();
  auto it = splits.find(static_cast<CUdeviceptr>(pointer));
  if (it != splits.end()) {
    info.push_back(it->second.device);
    info.push_back(static_cast<int64_t>(it->second.nbytes));
    info.push_back(static_cast<int64_t>(it->second.size));
    info.push_back(static_cast<int64_t>(it->second.cut));
  }
  return info;
}

int64_t SplitGranularity(int64_t device) {
  DeviceScope scope(static_cast<int>(device));
  TVM_FFI_ICHECK(scope.ok()) << "cannot make CUDA device " << device << " current";
  CheckDevice(static_cast<int>(device));
  return static_cast<int64_t>(Granularity(static_cast<int>(device)));
}

int64_t PointerDomain(int64_t pointer, int64_t device) {
  DeviceScope scope(static_cast<int>(device));
  TVM_FFI_ICHECK(scope.ok()) << "cannot make CUDA device " << device << " current";
  int domain = -1;
  CUresult result = cuPointerGetAttribute(&domain, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                                          static_cast<CUdeviceptr>(pointer));
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL) failed: "
      << CuError(result);
  return static_cast<int64_t>(domain);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_locality_split_empty, SplitEmpty);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_locality_split_info, SplitInfo);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_locality_granularity, SplitGranularity);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_locality_pointer_domain, PointerDomain);
