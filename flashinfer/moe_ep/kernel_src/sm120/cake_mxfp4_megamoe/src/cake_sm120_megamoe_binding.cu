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

// Host binding for the Cake SM120 MXFP4 x MXFP8 hybrid-transport MegaMoE kernel.
//
// Four concerns, all driven by the sealed manifest on the Python side:
//
//   1. Transport session: adopt the caller's ncclComm_t (torch ProcessGroupNCCL),
//      allocate + register the symmetric windows, resolve the load/store (LSA)
//      peer pointer of every window for every reachable peer, and -- when the
//      communicator offers GIN -- create the device communicator (ncclDevComm)
//      with the exact GIN requirements.  Pairs are later served by SELF, LSA or
//      GIN according to a per-pair transport table chosen on the Python side.
//   2. Setup-time transport probe: one timed all-to-all round over a candidate
//      table row (LSA bulk copies and/or GIN puts), used to score the
//      candidate tables.  Never on the submit path.
//   3. TMA descriptors: encode 2-D tiled / native-FP4 CUtensorMap into
//      caller-owned 128-byte device carriers (pointer-TMA ABI).
//   4. Cooperative launch: load a generated cubin and launch it with a
//      manifest-ordered argument list marshalled by Python.
//
// The submit path (launch) performs no CUDA allocation and owns no CUDA
// graph: it forwards to cuLaunchCooperativeKernel on the caller's stream.

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

namespace flashinfer::cake_sm120_megamoe {

using tvm::ffi::TensorView;

namespace {

constexpr int kExpectedNcclVersion = 23007;  // NCCL 2.30.7 (Device API: LSA + GIN)
constexpr int64_t kTensorMapBytes = 128;

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorName(status)
                                        << " (" << cudaGetErrorString(status) << ")";
}

inline void CheckCu(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &text);
  TVM_FFI_ICHECK(false) << operation << " failed: " << (name ? name : "CUresult") << " ("
                        << (text ? text : "unknown") << ")";
}

inline void CheckNccl(ncclResult_t status, const char* operation) {
  TVM_FFI_ICHECK(status == ncclSuccess) << operation << " failed: " << ncclGetErrorString(status);
}

// NCCL_GIN_ENABLE=0 switches GIN off but ncclCommQueryProperties still reports a GIN type, and
// ncclDevCommCreate with GIN requirements then faults inside NCCL 2.30.7 (integer division by
// zero in ncclGinDevCommSetup).  NCCL loads nccl.conf into the environment during communicator
// init, so after init the environment also reflects a setting made there.
inline bool GinDisabledByEnvironment() {
  const char* value = std::getenv("NCCL_GIN_ENABLE");
  if (value == nullptr || *value == '\0') return false;
  char* end = nullptr;
  long long parsed = std::strtoll(value, &end, 0);
  return end != value && parsed == 0;
}

inline void CheckCpuInt64(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCPU) << name << " must be a CPU tensor";
  TVM_FFI_ICHECK(t.ndim() == 1) << name << " must be 1-D";
  TVM_FFI_ICHECK(t.dtype().code == kDLInt && t.dtype().bits == 64) << name << " must be int64";
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
}

inline void CheckCpuFloat64(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCPU) << name << " must be a CPU tensor";
  TVM_FFI_ICHECK(t.ndim() == 1) << name << " must be 1-D";
  TVM_FFI_ICHECK(t.dtype().code == kDLFloat && t.dtype().bits == 64) << name << " must be float64";
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
}

inline void CheckCudaInt64(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(t.ndim() == 1) << name << " must be 1-D";
  TVM_FFI_ICHECK(t.dtype().code == kDLInt && t.dtype().bits == 64) << name << " must be int64";
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
}

inline void CheckCpuInt8(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCPU) << name << " must be a CPU tensor";
  TVM_FFI_ICHECK(t.ndim() == 1) << name << " must be 1-D";
  TVM_FFI_ICHECK((t.dtype().code == kDLInt || t.dtype().code == kDLUInt) && t.dtype().bits == 8)
      << name << " must be int8/uint8";
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
}

inline cudaStream_t StreamOf(int64_t device, int64_t stream) {
  return stream != 0 ? reinterpret_cast<cudaStream_t>(stream)
                     : get_stream(DLDevice{kDLCUDA, static_cast<int32_t>(device)});
}

// ---------------------------------------------------------------------------
// 1. Transport session
// ---------------------------------------------------------------------------

struct Window {
  std::size_t bytes = 0;
  void* pointer = nullptr;
  ncclWindow_t window = nullptr;
};

// One rank's registered windows + LSA peer pointers + optional GIN device communicator.
// The ProcessGroupNCCL communicator is borrowed, never finalized or destroyed here.
class TransportSession {
 public:
  TransportSession(std::uintptr_t comm_pointer, int rank, int world_size, int device,
                   const std::vector<std::int64_t>& window_bytes, int gin_context_count,
                   int gin_signal_count, int gin_world_barrier_count,
                   bool gin_strong_signals_required)
      : comm_(reinterpret_cast<ncclComm_t>(comm_pointer)),
        rank_(rank),
        world_size_(world_size),
        device_(device) {
    TVM_FFI_ICHECK(comm_pointer != 0) << "NCCL communicator pointer cannot be null";
    TVM_FFI_ICHECK(rank >= 0 && world_size > 0 && rank < world_size)
        << "invalid rank/world-size pair";
    TVM_FFI_ICHECK(gin_context_count > 0 && gin_signal_count >= 0 && gin_world_barrier_count >= 0)
        << "invalid GIN requirements";

    int runtime_version = 0;
    CheckNccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
    TVM_FFI_ICHECK(runtime_version == kExpectedNcclVersion)
        << "Cake SM120 MegaMoE requires NCCL 2.30.7 (" << kExpectedNcclVersion << "), observed "
        << runtime_version;

    int current_device = -1;
    CheckCuda(cudaGetDevice(&current_device), "cudaGetDevice");
    TVM_FFI_ICHECK(current_device == device)
        << "current CUDA device " << current_device << " differs from expected " << device;

    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    CheckNccl(ncclCommQueryProperties(comm_, &properties), "ncclCommQueryProperties");
    TVM_FFI_ICHECK(properties.rank == rank && properties.nRanks == world_size &&
                   properties.cudaDev == device)
        << "NCCL communicator identity mismatch: rank=" << properties.rank
        << " world=" << properties.nRanks << " device=" << properties.cudaDev;
    TVM_FFI_ICHECK(properties.deviceApiSupport)
        << "NCCL communicator does not support the Device API";
    gin_type_ = static_cast<int>(properties.ginType);
    gin_available_ = properties.ginType != NCCL_GIN_TYPE_NONE;
    if (gin_available_ && GinDisabledByEnvironment()) {
      std::fprintf(stderr,
                   "[cake_sm120_megamoe] rank %d: NCCL_GIN_ENABLE=0, continuing without GIN\n",
                   rank_);
      gin_available_ = false;
    }
    lsa_team_count_ = properties.nLsaTeams;

    try {
      windows_.reserve(window_bytes.size());
      for (std::int64_t raw_bytes : window_bytes) {
        TVM_FFI_ICHECK(raw_bytes > 0) << "window byte count must be positive";
        Window w;
        w.bytes = static_cast<std::size_t>(raw_bytes);
        CheckNccl(ncclMemAlloc(&w.pointer, w.bytes), "ncclMemAlloc");
        // Record the allocation before any further step that can throw, so Release() frees it.
        windows_.push_back(w);
        Window& stored = windows_.back();
        ncclResult_t registered = ncclCommWindowRegister(comm_, stored.pointer, stored.bytes,
                                                         &stored.window, NCCL_WIN_COLL_SYMMETRIC);
        // Without GIN, NCCL can only register the window if every peer is load/store reachable.
        TVM_FFI_ICHECK(registered == ncclSuccess)
            << "ncclCommWindowRegister failed: " << ncclGetErrorString(registered)
            << (gin_available_ ? ""
                               : " (GIN is unavailable or disabled, so every EP peer must be "
                                 "load/store reachable)");
        CheckCuda(cudaMemset(stored.pointer, 0, stored.bytes), "cudaMemset(window)");
      }

      // LSA reachability: a peer is reachable only if every window maps.  Unreachable peers
      // keep pointer 0; the Python table validator forbids selecting LSA for them.
      lsa_reachable_.assign(world_size_, 1);
      peer_pointers_.assign(windows_.size() * world_size_, 0);
      for (std::size_t wi = 0; wi < windows_.size(); ++wi) {
        for (int peer = 0; peer < world_size_; ++peer) {
          void* p = nullptr;
          ncclResult_t rc = ncclGetPeerDevicePointer(windows_[wi].window, 0, peer, &p);
          if (rc != ncclSuccess || p == nullptr) {
            lsa_reachable_[peer] = 0;
            continue;
          }
          peer_pointers_[wi * world_size_ + peer] = reinterpret_cast<std::uintptr_t>(p);
        }
      }
      for (int peer = 0; peer < world_size_; ++peer) {
        if (lsa_reachable_[peer]) continue;
        for (std::size_t wi = 0; wi < windows_.size(); ++wi)
          peer_pointers_[wi * world_size_ + peer] = 0;
      }
      TVM_FFI_ICHECK(lsa_reachable_[rank_]) << "self window is not load/store mapped";

      if (gin_available_) {
        ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        requirements.ginContextCount = gin_context_count;
        requirements.ginSignalCount = gin_signal_count;
        requirements.worldGinBarrierCount = gin_world_barrier_count;
        requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        requirements.ginStrongSignalsRequired = gin_strong_signals_required;
        requirements.ginVaSignalsRequired = false;
        CheckNccl(ncclDevCommCreate(comm_, &requirements, &host_dev_comm_), "ncclDevCommCreate");
        dev_comm_created_ = true;
        TVM_FFI_ICHECK(host_dev_comm_.ginContextCount > 0)
            << "ncclDevCommCreate returned zero GIN contexts";
        // Setup-time allocation (not on the submit path): the kernel takes
        // `ncclDevComm const*` and reads it from global memory.
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_dev_comm_), sizeof(host_dev_comm_)),
                  "cudaMalloc(device_dev_comm)");
        CheckCuda(cudaMemcpy(device_dev_comm_, &host_dev_comm_, sizeof(host_dev_comm_),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(device_dev_comm)");
      }
      for (int peer = 0; peer < world_size_; ++peer) {
        TVM_FFI_ICHECK(lsa_reachable_[peer] || gin_available_)
            << "peer " << peer << " is neither load/store reachable nor GIN reachable";
      }
      CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(session init)");
    } catch (...) {
      Release();
      throw;
    }
  }

  TransportSession(const TransportSession&) = delete;
  TransportSession& operator=(const TransportSession&) = delete;
  ~TransportSession() { Release(); }

  const Window& window(int64_t index) const {
    RequireOpen();
    TVM_FFI_ICHECK(index >= 0 && static_cast<std::size_t>(index) < windows_.size())
        << "window index out of range: " << index;
    return windows_[static_cast<std::size_t>(index)];
  }
  const std::vector<std::uintptr_t>& peer_pointers() const {
    RequireOpen();
    return peer_pointers_;
  }
  const std::vector<int>& lsa_reachable() const {
    RequireOpen();
    return lsa_reachable_;
  }
  std::uintptr_t device_comm_pointer() const {
    RequireOpen();
    return reinterpret_cast<std::uintptr_t>(device_dev_comm_);
  }
  int world_size() const { return world_size_; }
  bool gin_available() const { return gin_available_; }
  int gin_type() const { return gin_type_; }
  int lsa_team_count() const { return lsa_team_count_; }
  int context_count() const { return gin_available_ ? host_dev_comm_.ginContextCount : 0; }
  int signal_count() const { return gin_available_ ? host_dev_comm_.ginSignalCount : 0; }
  std::size_t window_count() const { return windows_.size(); }
  bool closed() const { return closed_; }

  void Close() {
    if (closed_) return;
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(close)");
    Release();
  }

 private:
  void RequireOpen() const { TVM_FFI_ICHECK(!closed_) << "transport session is closed"; }

  // Best-effort teardown: destructors cannot throw.
  void Release() noexcept {
    if (closed_) return;
    closed_ = true;
    if (device_dev_comm_ != nullptr) {
      (void)cudaFree(device_dev_comm_);
      device_dev_comm_ = nullptr;
    }
    if (dev_comm_created_) {
      (void)ncclDevCommDestroy(comm_, &host_dev_comm_);
      dev_comm_created_ = false;
    }
    for (auto it = windows_.rbegin(); it != windows_.rend(); ++it) {
      if (it->window != nullptr) (void)ncclCommWindowDeregister(comm_, it->window);
      if (it->pointer != nullptr) (void)ncclMemFree(it->pointer);
      it->window = nullptr;
      it->pointer = nullptr;
    }
    windows_.clear();
  }

  ncclComm_t comm_ = nullptr;
  int rank_ = -1;
  int world_size_ = 0;
  int device_ = -1;
  bool gin_available_ = false;
  int gin_type_ = 0;
  int lsa_team_count_ = 0;
  std::vector<Window> windows_;
  std::vector<std::uintptr_t> peer_pointers_;
  std::vector<int> lsa_reachable_;
  ncclDevComm host_dev_comm_{};
  ncclDevComm* device_dev_comm_ = nullptr;
  bool dev_comm_created_ = false;
  bool closed_ = false;
};

std::mutex g_registry_mutex;
std::unordered_map<int64_t, std::unique_ptr<TransportSession>> g_sessions;
int64_t g_next_session_handle = 1;

TransportSession& SessionOf(int64_t handle) {
  std::lock_guard<std::mutex> guard(g_registry_mutex);
  auto found = g_sessions.find(handle);
  TVM_FFI_ICHECK(found != g_sessions.end()) << "unknown transport session handle " << handle;
  return *found->second;
}

}  // namespace

// window_bytes: CPU int64 [n]. Returns an opaque session handle (int64).
int64_t transport_session_create(int64_t nccl_comm_pointer, int64_t rank, int64_t world_size,
                                 int64_t device, TensorView window_bytes, int64_t gin_context_count,
                                 int64_t gin_signal_count, int64_t gin_world_barrier_count,
                                 bool gin_strong_signals_required) {
  CheckCpuInt64(window_bytes, "window_bytes");
  std::vector<std::int64_t> bytes(static_cast<std::size_t>(window_bytes.shape()[0]));
  std::memcpy(bytes.data(), window_bytes.data_ptr(), bytes.size() * sizeof(std::int64_t));
  TVM_FFI_ICHECK(!bytes.empty()) << "transport session requires at least one window";
  auto session = std::make_unique<TransportSession>(
      static_cast<std::uintptr_t>(nccl_comm_pointer), static_cast<int>(rank),
      static_cast<int>(world_size), static_cast<int>(device), bytes,
      static_cast<int>(gin_context_count), static_cast<int>(gin_signal_count),
      static_cast<int>(gin_world_barrier_count), gin_strong_signals_required);
  std::lock_guard<std::mutex> guard(g_registry_mutex);
  const int64_t handle = g_next_session_handle++;
  g_sessions.emplace(handle, std::move(session));
  return handle;
}

int64_t transport_session_window_pointer(int64_t handle, int64_t index) {
  return static_cast<int64_t>(
      reinterpret_cast<std::uintptr_t>(SessionOf(handle).window(index).pointer));
}

int64_t transport_session_window_handle(int64_t handle, int64_t index) {
  return static_cast<int64_t>(
      reinterpret_cast<std::uintptr_t>(SessionOf(handle).window(index).window));
}

int64_t transport_session_window_bytes(int64_t handle, int64_t index) {
  return static_cast<int64_t>(SessionOf(handle).window(index).bytes);
}

// Setup-time helper: zero one window on the given stream (0 = FFI current stream of `device`).
void transport_session_window_zero(int64_t handle, int64_t index, int64_t device, int64_t stream) {
  const Window& w = SessionOf(handle).window(index);
  CheckCuda(cudaMemsetAsync(w.pointer, 0, w.bytes, StreamOf(device, stream)),
            "cudaMemsetAsync(window)");
}

// 0 when the communicator offers no GIN transport (LSA-only deployment).
int64_t transport_session_dev_comm_pointer(int64_t handle) {
  return static_cast<int64_t>(SessionOf(handle).device_comm_pointer());
}

// out: CPU int64 [window_count * world_size], window-major LSA peer pointers (0 = unmapped).
void transport_session_peer_pointers(int64_t handle, TensorView out) {
  CheckCpuInt64(out, "out");
  const auto& pointers = SessionOf(handle).peer_pointers();
  TVM_FFI_ICHECK(out.shape()[0] == static_cast<int64_t>(pointers.size()))
      << "peer pointer output needs " << pointers.size() << " slots";
  auto* p = static_cast<int64_t*>(out.data_ptr());
  for (std::size_t i = 0; i < pointers.size(); ++i) p[i] = static_cast<int64_t>(pointers[i]);
}

// out: CPU int64 [world_size], 1 = every window of that peer is load/store mapped.
void transport_session_lsa_reachable(int64_t handle, TensorView out) {
  CheckCpuInt64(out, "out");
  const auto& reach = SessionOf(handle).lsa_reachable();
  TVM_FFI_ICHECK(out.shape()[0] == static_cast<int64_t>(reach.size()))
      << "reachability output needs " << reach.size() << " slots";
  auto* p = static_cast<int64_t*>(out.data_ptr());
  for (std::size_t i = 0; i < reach.size(); ++i) p[i] = reach[i];
}

// out: CPU int64 [>= 8] = [nccl_version, gin_available, gin_type, lsa_team_count,
//                          gin_context_count, gin_signal_count, window_count, closed].
void transport_session_properties(int64_t handle, TensorView out) {
  CheckCpuInt64(out, "out");
  TVM_FFI_ICHECK(out.shape()[0] >= 8) << "properties output needs >= 8 slots";
  TransportSession& s = SessionOf(handle);
  auto* p = static_cast<int64_t*>(out.data_ptr());
  p[0] = kExpectedNcclVersion;
  p[1] = s.gin_available() ? 1 : 0;
  p[2] = s.gin_type();
  p[3] = s.lsa_team_count();
  p[4] = s.context_count();
  p[5] = s.signal_count();
  p[6] = static_cast<int64_t>(s.window_count());
  p[7] = s.closed() ? 1 : 0;
}

void transport_session_close(int64_t handle) {
  std::unique_ptr<TransportSession> session;
  {
    std::lock_guard<std::mutex> guard(g_registry_mutex);
    auto found = g_sessions.find(handle);
    TVM_FFI_ICHECK(found != g_sessions.end()) << "unknown transport session handle " << handle;
    session = std::move(found->second);
    g_sessions.erase(found);
  }
  session->Close();
}

// ---------------------------------------------------------------------------
// 2. Setup-time transport probe
// ---------------------------------------------------------------------------
//
// One CTA per destination; LSA destinations stream 8 KB blocks with a 3-stage
// cp.async.bulk ring (the shape of the kernel's copier warps); GIN destinations
// issue chunked ncclGin puts from one elected thread.  The last CTA
// (blockIdx.x == world - 1) is the waiter: it first runs a device-side
// rendezvous with every peer it talks to this round (a 4-byte ping on a
// dedicated barrier signal over that pair's own transport), takes a
// %globaltimer stamp, releases the senders through a device-global `go` flag,
// and stamps again once every sender CTA finished and every receive signal
// arrived.  Inter-rank launch skew is therefore outside the measured window.

namespace probe {

constexpr int kSelf = 0, kLsa = 1, kGin = 2;
// 4 copier warps x 3 stages x 8 KB = 96 KB of staging per destination CTA.
constexpr int kStageBytes = 8192, kStages = 3, kWarps = 4;
constexpr int kBlockWarps = 8;     // block width of every probe CTA
constexpr int kMaxWorld = 64;      // ProbeArgs array capacity / largest supported world
constexpr int kSignalStride = 32;  // u32 words per LSA signal slot (kernel LSA_SIGNAL_STRIDE)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned phase) {
  asm volatile(
      "{\n .reg .pred p;\n PROBE_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n "
      "@!p bra PROBE_WAIT;\n}" ::"r"(smem_u32(b)),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* smem, const void* gsrc, unsigned bytes,
                                         uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_u32(smem)),
      "l"(gsrc), "r"(bytes), "r"(smem_u32(b))
      : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* gdst, const void* smem, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gdst),
               "r"(smem_u32(smem)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read2() {
  asm volatile("cp.async.bulk.wait_group.read 2;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void st_release_sys_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct ProbeArgs {
  ncclDevComm const* comm;
  const unsigned char* src;  // local dispatch_payload_out
  ncclWindow_t src_win;
  ncclWindow_t dst_win;                    // dispatch_payload_inbox (symmetric)
  unsigned long long dst_ptr[kMaxWorld];   // LSA-mapped peer dispatch_payload_inbox (0 = unmapped)
  unsigned long long peer_sig[kMaxWorld];  // LSA-mapped peer signal window
  uint32_t* my_sig;                        // local signal window
  int row[kMaxWorld];                      // transport per destination
  int rank, world;
  unsigned long long bytes_per_dest;
  unsigned long long dst_slot_bytes;      // per-source slot stride inside the inbox
  int signal_base;                        // data signals [base, base + world); barrier signals
                                          // [base + world, base + 2 * world)
  uint32_t epoch;                         // expected cumulative LSA signal value after this round
  unsigned chunk_bytes;                   // GIN put granularity
  const unsigned long long* gin_targets;  // per-source cumulative data-signal target
  const unsigned long long* gin_barrier_targets;  // per-source cumulative barrier target
  unsigned int* go_flag;                          // device scratch, host-zeroed each round
  unsigned int* done_ctas;                        // device scratch, host-zeroed each round
  unsigned long long* elapsed_ns_out;             // device scratch, 1 element
};

extern __shared__ __align__(1024) unsigned char probe_smem[];

__global__ void probe_send(ProbeArgs a) {
  if (blockIdx.x == static_cast<unsigned>(a.world) - 1) {
    const int w = threadIdx.x / 32;
    // Every warp must finish its receive waits before warp 0 stamps t1.
    __shared__ int recv_done;
    if (threadIdx.x == 0) recv_done = 0;
    __syncthreads();
    if (threadIdx.x % 32) return;

    const int barrier_base = a.signal_base + a.world;
    const unsigned long long dst_off = static_cast<unsigned long long>(a.rank) * a.dst_slot_bytes;
    unsigned long long t0 = 0;

    if (w == 0) {
      // Rendezvous phase 1: ping every peer this round talks to, over that pair's transport.
      for (int p = 0; p < a.world; ++p) {
        if (p == a.rank) continue;
        const int kind = a.row[p];
        if (kind == kLsa) {
          uint32_t v = *reinterpret_cast<const uint32_t*>(a.src);
          uint32_t* dst4 =
              reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(a.dst_ptr[p]) + dst_off);
          *dst4 = v;
          __threadfence_system();
          uint32_t* sig =
              reinterpret_cast<uint32_t*>(a.peer_sig[p]) + (barrier_base + a.rank) * kSignalStride;
          st_release_sys_u32(sig, a.epoch);
        } else if (kind == kGin) {
          ncclGin gin(*a.comm, 0);
          gin.put(ncclTeamWorld(*a.comm), p, a.dst_win, static_cast<size_t>(dst_off), a.src_win,
                  static_cast<size_t>(0), static_cast<size_t>(4),
                  ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(barrier_base + a.rank), 1},
                  ncclGin_None{}, ncclCoopThread());
        }
      }
      // Rendezvous phase 2: wait for those peers' pings.
      for (int p = 0; p < a.world; ++p) {
        if (p == a.rank) continue;
        const int kind = a.row[p];
        if (kind == kLsa) {
          const uint32_t* sig = a.my_sig + (barrier_base + p) * kSignalStride;
          while (static_cast<int32_t>(ld_acquire_sys_u32(sig) - a.epoch) < 0) {
          }
        } else if (kind == kGin) {
          ncclGin gin(*a.comm, 0);
          gin.waitSignal(ncclCoopThread(), static_cast<ncclGinSignal_t>(barrier_base + p),
                         a.gin_barrier_targets[p], 64, cuda::memory_order_acquire);
        }
      }
      t0 = globaltimer();
      __threadfence();
      st_release_gpu_u32(a.go_flag, 1u);
    }

    // Receive waits: warp w owns sources w, w + kBlockWarps, ...
    for (int src = w; src < a.world; src += kBlockWarps) {
      if (src == a.rank) continue;
      const int kind = a.row[src];
      if (kind == kLsa) {
        const uint32_t* sig = a.my_sig + (a.signal_base + src) * kSignalStride;
        while (static_cast<int32_t>(ld_acquire_sys_u32(sig) - a.epoch) < 0) {
        }
      } else if (kind == kGin) {
        ncclGin gin(*a.comm, 0);
        gin.waitSignal(ncclCoopThread(), static_cast<ncclGinSignal_t>(a.signal_base + src),
                       a.gin_targets[src], 64, cuda::memory_order_acquire);
      }
    }
    atomicAdd(&recv_done, 1);

    if (w == 0) {
      while (atomicAdd(&recv_done, 0) < kBlockWarps) {
      }
      while (atomicAdd(a.done_ctas, 0u) < static_cast<unsigned>(a.world - 1)) {
      }
      unsigned long long t1 = globaltimer();
      *a.elapsed_ns_out = t1 - t0;
    }
    return;
  }

  const int dest = (a.rank + 1 + blockIdx.x) % a.world;
  const int kind = a.row[dest];
  const unsigned long long dst_off = static_cast<unsigned long long>(a.rank) * a.dst_slot_bytes;
  const int w = threadIdx.x / 32;

  __shared__ int done;
  if (threadIdx.x == 0) done = 0;
  __syncthreads();
  if (threadIdx.x % 32) return;
  if (w >= kWarps) return;  // only kWarps copier warps do real work

  while (ld_acquire_gpu_u32(a.go_flag) == 0) {
  }  // gate the first transfer on the rendezvous

  if (kind == kLsa) {
    unsigned char* my = probe_smem + static_cast<size_t>(w) * kStages * kStageBytes;
    uint64_t* bars = reinterpret_cast<uint64_t*>(probe_smem + static_cast<size_t>(kWarps) *
                                                                  kStages * kStageBytes) +
                     w * kStages;
    for (int s = 0; s < kStages; ++s) mbar_init(&bars[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __threadfence_block();
    unsigned char* dst = reinterpret_cast<unsigned char*>(a.dst_ptr[dest]) + dst_off;
    unsigned n = 0;
    for (unsigned long long off = static_cast<unsigned long long>(w) * kStageBytes;
         off < a.bytes_per_dest;
         off += static_cast<unsigned long long>(kWarps) * kStageBytes, ++n) {
      unsigned len = static_cast<unsigned>(
          min(static_cast<unsigned long long>(kStageBytes), a.bytes_per_dest - off));
      int s = n % kStages;
      unsigned phase = (n / kStages) & 1;
      if (n >= static_cast<unsigned>(kStages)) bulk_wait_read2();
      mbar_expect_tx(&bars[s], len);
      bulk_g2s(my + static_cast<size_t>(s) * kStageBytes, a.src + off, len, &bars[s]);
      mbar_wait(&bars[s], phase);
      fence_proxy_async();
      bulk_s2g(dst + off, my + static_cast<size_t>(s) * kStageBytes, len);
      bulk_commit();
    }
    bulk_wait0();
    __threadfence_system();
    atomicAdd(&done, 1);
    if (w == 0) {
      while (atomicAdd(&done, 0) < kWarps) {
      }
      __threadfence_system();
      uint32_t* sig =
          reinterpret_cast<uint32_t*>(a.peer_sig[dest]) + (a.signal_base + a.rank) * kSignalStride;
      st_release_sys_u32(sig, a.epoch);
      atomicAdd(a.done_ctas, 1u);
    }
  } else if (kind == kGin) {
    if (w == 0) {
      ncclGin gin(*a.comm, 0);
      for (unsigned long long off = 0; off < a.bytes_per_dest; off += a.chunk_bytes) {
        size_t len = static_cast<size_t>(
            min(static_cast<unsigned long long>(a.chunk_bytes), a.bytes_per_dest - off));
        __threadfence_system();
        gin.put(ncclTeamWorld(*a.comm), dest, a.dst_win, static_cast<size_t>(dst_off + off),
                a.src_win, static_cast<size_t>(off), len,
                ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(a.signal_base + a.rank), 1},
                ncclGin_None{}, ncclCoopThread());
      }
      atomicAdd(a.done_ctas, 1u);
    }
  } else {
    // SELF this round: still counts toward done_ctas so the waiter's gate is satisfiable.
    if (w == 0) atomicAdd(a.done_ctas, 1u);
  }
}

__global__ void probe_read_gin_signals(ncclDevComm const* comm, int signal_base, int world,
                                       unsigned long long* out) {
  if (threadIdx.x >= static_cast<unsigned>(world)) return;
  ncclGin gin(*comm, 0);
  out[threadIdx.x] = gin.readSignal(static_cast<ncclGinSignal_t>(signal_base + threadIdx.x), 64,
                                    cuda::memory_order_acquire);
}

}  // namespace probe

// One probe round.  `peer_inbox_ptrs`, `peer_signal_ptrs`, `row`: CPU int64 [world];
// `gin_targets`, `gin_barrier_targets`: CUDA int64 [world] (cumulative expectations);
// `go`, `done`, `elapsed_ns`: device scratch pointers (zeroed by the caller).
// Writes {device_ms, host_ms} into `out` (CPU float64 [2]).  device_ms is measured on device
// between the rendezvous and receive completion; host_ms (event time around the launch,
// including launch skew) is diagnostic only.  The wait is a bounded poll (60 s) so a stuck
// handshake raises instead of hanging forever.
void probe_round(int64_t dev_comm, int64_t src, int64_t src_win, int64_t dst_win,
                 TensorView peer_inbox_ptrs, TensorView peer_signal_ptrs, int64_t my_signal_ptr,
                 TensorView row, int64_t rank, int64_t world, int64_t bytes_per_dest,
                 int64_t dst_slot_bytes, int64_t signal_base, int64_t signal_stride, int64_t epoch,
                 int64_t chunk_bytes, TensorView gin_targets, TensorView gin_barrier_targets,
                 int64_t go_ptr, int64_t done_ptr, int64_t elapsed_ns_ptr, int64_t device,
                 int64_t stream, TensorView out) {
  using namespace probe;
  TVM_FFI_ICHECK(world >= 1 && world <= kMaxWorld)
      << "probe supports 1 <= world <= " << kMaxWorld << " (got " << world << ")";
  TVM_FFI_ICHECK(signal_stride == kSignalStride)
      << "probe signal_stride " << signal_stride << " does not match the compiled stride "
      << kSignalStride;
  CheckCpuInt64(peer_inbox_ptrs, "peer_inbox_ptrs");
  CheckCpuInt64(peer_signal_ptrs, "peer_signal_ptrs");
  CheckCpuInt64(row, "row");
  CheckCudaInt64(gin_targets, "gin_targets");
  CheckCudaInt64(gin_barrier_targets, "gin_barrier_targets");
  CheckCpuFloat64(out, "out");
  TVM_FFI_ICHECK(peer_inbox_ptrs.shape()[0] == world && peer_signal_ptrs.shape()[0] == world &&
                 row.shape()[0] == world && gin_targets.shape()[0] == world &&
                 gin_barrier_targets.shape()[0] == world && out.shape()[0] >= 2)
      << "probe argument extents do not match world";

  ProbeArgs a{};
  a.comm = reinterpret_cast<ncclDevComm const*>(dev_comm);
  a.src = reinterpret_cast<const unsigned char*>(src);
  a.src_win = reinterpret_cast<ncclWindow_t>(src_win);
  a.dst_win = reinterpret_cast<ncclWindow_t>(dst_win);
  const auto* inbox = static_cast<const int64_t*>(peer_inbox_ptrs.data_ptr());
  const auto* sigs = static_cast<const int64_t*>(peer_signal_ptrs.data_ptr());
  const auto* kinds = static_cast<const int64_t*>(row.data_ptr());
  for (int i = 0; i < world; ++i) {
    a.dst_ptr[i] = static_cast<unsigned long long>(inbox[i]);
    a.peer_sig[i] = static_cast<unsigned long long>(sigs[i]);
    a.row[i] = static_cast<int>(kinds[i]);
  }
  a.my_sig = reinterpret_cast<uint32_t*>(my_signal_ptr);
  a.rank = static_cast<int>(rank);
  a.world = static_cast<int>(world);
  a.bytes_per_dest = static_cast<unsigned long long>(bytes_per_dest);
  a.dst_slot_bytes = static_cast<unsigned long long>(dst_slot_bytes);
  a.signal_base = static_cast<int>(signal_base);
  a.epoch = static_cast<uint32_t>(epoch);
  a.chunk_bytes = static_cast<unsigned>(chunk_bytes);
  a.gin_targets = static_cast<const unsigned long long*>(gin_targets.data_ptr());
  a.gin_barrier_targets = static_cast<const unsigned long long*>(gin_barrier_targets.data_ptr());
  a.go_flag = reinterpret_cast<unsigned int*>(go_ptr);
  a.done_ctas = reinterpret_cast<unsigned int*>(done_ptr);
  a.elapsed_ns_out = reinterpret_cast<unsigned long long*>(elapsed_ns_ptr);

  const size_t smem = static_cast<size_t>(kWarps) * kStages * kStageBytes +
                      kWarps * kStages * sizeof(uint64_t) + 1024;
  CheckCuda(cudaFuncSetAttribute(probe_send, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)),
            "cudaFuncSetAttribute(probe_send)");
  cudaStream_t s0 = StreamOf(device, stream);
  cudaEvent_t e0, e1;
  CheckCuda(cudaEventCreate(&e0), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&e1), "cudaEventCreate");
  CheckCuda(cudaEventRecord(e0, s0), "cudaEventRecord");
  probe_send<<<static_cast<unsigned>(world), 32 * kBlockWarps, smem, s0>>>(a);
  CheckCuda(cudaGetLastError(), "probe_send launch");
  CheckCuda(cudaEventRecord(e1, s0), "cudaEventRecord");

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  cudaError_t status;
  while ((status = cudaStreamQuery(s0)) == cudaErrorNotReady) {
    if (std::chrono::steady_clock::now() >= deadline) {
      // The kernel is left running; the caller is expected to abort the process.
      (void)cudaEventDestroy(e0);
      (void)cudaEventDestroy(e1);
      TVM_FFI_ICHECK(false) << "transport probe round timed out";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  CheckCuda(status, "probe_send");
  float host_ms = 0.f;
  CheckCuda(cudaEventElapsedTime(&host_ms, e0, e1), "cudaEventElapsedTime");
  CheckCuda(cudaEventDestroy(e0), "cudaEventDestroy");
  CheckCuda(cudaEventDestroy(e1), "cudaEventDestroy");
  unsigned long long elapsed_ns = 0;
  CheckCuda(cudaMemcpy(&elapsed_ns, reinterpret_cast<void*>(elapsed_ns_ptr), sizeof(elapsed_ns),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(elapsed_ns)");
  auto* result = static_cast<double*>(out.data_ptr());
  result[0] = static_cast<double>(elapsed_ns) / 1e6;
  result[1] = static_cast<double>(host_ms);
}

// Reads the cumulative GIN signal values [signal_base, signal_base + world) into `out`
// (CUDA int64 [world]) and synchronizes the device.
void probe_read_gin_signals(int64_t dev_comm, int64_t signal_base, int64_t world, TensorView out,
                            int64_t device, int64_t stream) {
  TVM_FFI_ICHECK(world >= 1 && world <= probe::kMaxWorld)
      << "probe supports 1 <= world <= " << probe::kMaxWorld << " (got " << world << ")";
  CheckCudaInt64(out, "out");
  TVM_FFI_ICHECK(out.shape()[0] == world) << "signal output must have world entries";
  cudaStream_t s0 = StreamOf(device, stream);
  probe::probe_read_gin_signals<<<1, static_cast<unsigned>(((world + 31) / 32) * 32), 0, s0>>>(
      reinterpret_cast<ncclDevComm const*>(dev_comm), static_cast<int>(signal_base),
      static_cast<int>(world), static_cast<unsigned long long*>(out.data_ptr()));
  CheckCuda(cudaGetLastError(), "probe_read_gin_signals launch");
  CheckCuda(cudaStreamSynchronize(s0), "cudaStreamSynchronize(probe_read_gin_signals)");
}

// ---------------------------------------------------------------------------
// 3. TMA descriptors (pointer ABI: 128-byte device carriers)
// ---------------------------------------------------------------------------

namespace {

CUtensorMapDataType TmaDtype(int64_t code) {
  // Codes mirror the manifest descriptor dtype names; see the backend's _TMA_DTYPE_CODE.
  switch (code) {
    case 0:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case 1:
      return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    case 2:
      return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case 3:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case 4:
      return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    case 5:
      return CU_TENSOR_MAP_DATA_TYPE_INT64;
    case 6:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case 7:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case 8:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case 9:
      return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;  // native packed E2M1
    default:
      TVM_FFI_ICHECK(false) << "unknown TMA dtype code " << code;
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
}

CUtensorMapSwizzle TmaSwizzle(int64_t code) {
  switch (code) {
    case 0:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 1:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 2:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 3:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    case 4:
      return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    default:
      TVM_FFI_ICHECK(false) << "unknown TMA swizzle code " << code;
      return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

CUtensorMapL2promotion TmaL2(int64_t code) {
  switch (code) {
    case 0:
      return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case 1:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case 2:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case 3:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    default:
      TVM_FFI_ICHECK(false) << "unknown TMA L2 promotion code " << code;
      return CU_TENSOR_MAP_L2_PROMOTION_NONE;
  }
}

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFn GetEncodeTiled() {
  static EncodeTiledFn fn = [] {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSuccess;
    CheckCuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &symbol, 12000,
                                               cudaEnableDefault, &query),
              "cudaGetDriverEntryPointByVersion(cuTensorMapEncodeTiled)");
    TVM_FFI_ICHECK(symbol != nullptr && query == cudaDriverEntryPointSuccess)
        << "cuTensorMapEncodeTiled is unavailable in this driver";
    return reinterpret_cast<EncodeTiledFn>(symbol);
  }();
  return fn;
}

}  // namespace

// Encode one canonical 2-D tiled descriptor over `source` flattened to
// [rows, cols] (cols = innermost extent; unit inner stride; dense leading
// fold is validated by Python) and copy the 128 bytes into `carrier`
// (device uint8, >= 128 bytes, 128-byte aligned). Setup-time operation.
// The copy is ordered on `stream` (a cudaStream_t as int64, 0 = FFI current
// stream of `device`), the stream that allocated the carrier and launches the
// kernel: a plain cudaMemcpy runs on the legacy default stream, which a
// non-blocking caller stream does not wait for.
//
// For native FP4 (dtype code 9) `cols_elems` is the element count (2 x bytes)
// and `row_stride_bytes` the byte pitch; Python computes both.
void tma_encode_2d(TensorView source, int64_t dtype_code, int64_t cols_elems, int64_t rows,
                   int64_t row_stride_bytes, int64_t box0, int64_t box1, int64_t swizzle_code,
                   int64_t l2_code, TensorView carrier, int64_t device, int64_t stream) {
  CHECK_CUDA(source);
  CHECK_CUDA(carrier);
  TVM_FFI_ICHECK(carrier.dtype().bits == 8 && carrier.ndim() == 1 &&
                 carrier.shape()[0] >= kTensorMapBytes)
      << "carrier must be a device uint8[>=128] tensor";
  TVM_FFI_ICHECK(reinterpret_cast<std::uintptr_t>(carrier.data_ptr()) % 128 == 0)
      << "carrier must be 128-byte aligned";
  TVM_FFI_ICHECK(reinterpret_cast<std::uintptr_t>(source.data_ptr()) % 16 == 0)
      << "TMA source must be 16-byte aligned";
  TVM_FFI_ICHECK(cols_elems > 0 && rows > 0 && box0 > 0 && box1 > 0 && box0 <= cols_elems &&
                 box1 <= rows)
      << "TMA box (" << box0 << "," << box1 << ") does not fit source (" << cols_elems << ","
      << rows << ")";
  TVM_FFI_ICHECK(row_stride_bytes % 16 == 0) << "TMA row pitch must be a multiple of 16 bytes";

  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols_elems), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_stride_bytes)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(box0), static_cast<cuuint32_t>(box1)};
  const cuuint32_t elem_strides[2] = {1, 1};
  alignas(64) CUtensorMap tensor_map;
  CheckCu(
      GetEncodeTiled()(&tensor_map, TmaDtype(dtype_code), 2, source.data_ptr(), dims, strides, box,
                       elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, TmaSwizzle(swizzle_code),
                       TmaL2(l2_code), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  static_assert(sizeof(CUtensorMap) == kTensorMapBytes, "CUtensorMap must be 128 bytes");
  // Pageable source: cudaMemcpyAsync stages it before returning (the stack copy may go away).
  CheckCuda(cudaMemcpyAsync(carrier.data_ptr(), &tensor_map, sizeof(tensor_map),
                            cudaMemcpyHostToDevice, StreamOf(device, stream)),
            "cudaMemcpyAsync(tensor map carrier)");
}

// ---------------------------------------------------------------------------
// 4. Module load + cooperative launch
// ---------------------------------------------------------------------------

namespace {

struct LoadedFunction {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  int dynamic_smem_opt_in = 0;
};

std::mutex g_function_mutex;
std::unordered_map<int64_t, LoadedFunction> g_functions;
int64_t g_next_function_handle = 1;

LoadedFunction& FunctionOf(int64_t handle) {
  std::lock_guard<std::mutex> guard(g_function_mutex);
  auto found = g_functions.find(handle);
  TVM_FFI_ICHECK(found != g_functions.end()) << "unknown kernel function handle " << handle;
  return found->second;
}

void EnsureDynamicSmem(LoadedFunction& f, int smem_bytes) {
  if (smem_bytes <= 48 * 1024 || smem_bytes <= f.dynamic_smem_opt_in) return;
  CUdevice dev;
  CheckCu(cuCtxGetDevice(&dev), "cuCtxGetDevice");
  int optin_max = 0;
  CheckCu(
      cuDeviceGetAttribute(&optin_max, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev),
      "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)");
  TVM_FFI_ICHECK(smem_bytes <= optin_max)
      << "dynamic shared memory " << smem_bytes << " B exceeds the opt-in ceiling " << optin_max;
  CheckCu(
      cuFuncSetAttribute(f.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes),
      "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
  f.dynamic_smem_opt_in = smem_bytes;
}

}  // namespace

// cubin: CPU uint8 [n] holding the cubin image. Returns a function handle.
int64_t module_load_function(TensorView cubin, tvm::ffi::String symbol) {
  CheckCpuInt8(cubin, "cubin");
  // PyTorch does not push a primary driver context; make sure one exists.
  CheckCuda(cudaFree(nullptr), "cudaFree(nullptr) (context init)");
  LoadedFunction f;
  CheckCu(cuModuleLoadData(&f.module, cubin.data_ptr()), "cuModuleLoadData");
  CheckCu(cuModuleGetFunction(&f.function, f.module, symbol.c_str()), "cuModuleGetFunction");
  std::lock_guard<std::mutex> guard(g_function_mutex);
  const int64_t handle = g_next_function_handle++;
  g_functions.emplace(handle, f);
  return handle;
}

int64_t function_num_regs(int64_t handle) {
  int value = 0;
  CheckCu(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_NUM_REGS, FunctionOf(handle).function),
          "cuFuncGetAttribute(NUM_REGS)");
  return value;
}

int64_t function_local_bytes(int64_t handle) {
  int value = 0;
  CheckCu(
      cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, FunctionOf(handle).function),
      "cuFuncGetAttribute(LOCAL_SIZE_BYTES)");
  return value;
}

int64_t function_max_active_blocks(int64_t handle, int64_t block_size, int64_t smem_bytes) {
  LoadedFunction& f = FunctionOf(handle);
  EnsureDynamicSmem(f, static_cast<int>(smem_bytes));
  int blocks = 0;
  CheckCu(cuOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, f.function, static_cast<int>(block_size), static_cast<size_t>(smem_bytes)),
          "cuOccupancyMaxActiveBlocksPerMultiprocessor");
  return blocks;
}

// Cooperative launch. `arg_values` (CPU int64 [n]) carries every kernel
// argument as a raw little-endian integer (pointers, int/uint scalars; float
// scalars pre-bitcast by Python) and `arg_widths` (CPU int8 [n]) the byte
// width of each physical argument (8 for pointers/int64, 4 for int32/uint32,
// 2/1 for narrower). The argument order is the manifest's physical order.
// `stream` is a cudaStream_t as int64 (0 = FFI current stream of `device`).
// No allocation, no graph ownership: one cuLaunchCooperativeKernel.
void launch_cooperative(int64_t handle, int64_t grid_x, int64_t block_x, int64_t smem_bytes,
                        TensorView arg_values, TensorView arg_widths, int64_t device,
                        int64_t stream) {
  CheckCpuInt64(arg_values, "arg_values");
  CheckCpuInt8(arg_widths, "arg_widths");
  const int64_t n = arg_values.shape()[0];
  TVM_FFI_ICHECK(arg_widths.shape()[0] == n) << "arg_values / arg_widths length mismatch";
  TVM_FFI_ICHECK(n > 0 && n <= 256) << "unsupported kernel argument count " << n;
  LoadedFunction& f = FunctionOf(handle);
  EnsureDynamicSmem(f, static_cast<int>(smem_bytes));

  // Every argument gets an 8-byte, 8-byte-aligned slot; cuLaunch reads
  // `width` bytes from each slot as declared by the kernel signature.
  alignas(8) int64_t storage[256];
  void* pointers[256];
  const auto* values = static_cast<const int64_t*>(arg_values.data_ptr());
  const auto* widths = static_cast<const int8_t*>(arg_widths.data_ptr());
  for (int64_t i = 0; i < n; ++i) {
    TVM_FFI_ICHECK(widths[i] == 1 || widths[i] == 2 || widths[i] == 4 || widths[i] == 8)
        << "argument " << i << " has unsupported width " << static_cast<int>(widths[i]);
    storage[i] = values[i];
    pointers[i] = &storage[i];
  }
  cudaStream_t cuda_stream = StreamOf(device, stream);
  CheckCu(cuLaunchCooperativeKernel(
              f.function, static_cast<unsigned>(grid_x), 1, 1, static_cast<unsigned>(block_x), 1, 1,
              static_cast<unsigned>(smem_bytes), reinterpret_cast<CUstream>(cuda_stream), pointers),
          "cuLaunchCooperativeKernel");
}

int64_t nccl_runtime_version() {
  int version = 0;
  CheckNccl(ncclGetVersion(&version), "ncclGetVersion");
  return version;
}

}  // namespace flashinfer::cake_sm120_megamoe

#define CAKE_MEGAMOE_EXPORT(name) \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, flashinfer::cake_sm120_megamoe::name);

CAKE_MEGAMOE_EXPORT(transport_session_create)
CAKE_MEGAMOE_EXPORT(transport_session_window_pointer)
CAKE_MEGAMOE_EXPORT(transport_session_window_handle)
CAKE_MEGAMOE_EXPORT(transport_session_window_bytes)
CAKE_MEGAMOE_EXPORT(transport_session_window_zero)
CAKE_MEGAMOE_EXPORT(transport_session_dev_comm_pointer)
CAKE_MEGAMOE_EXPORT(transport_session_peer_pointers)
CAKE_MEGAMOE_EXPORT(transport_session_lsa_reachable)
CAKE_MEGAMOE_EXPORT(transport_session_properties)
CAKE_MEGAMOE_EXPORT(transport_session_close)
CAKE_MEGAMOE_EXPORT(probe_round)
CAKE_MEGAMOE_EXPORT(probe_read_gin_signals)
CAKE_MEGAMOE_EXPORT(tma_encode_2d)
CAKE_MEGAMOE_EXPORT(module_load_function)
CAKE_MEGAMOE_EXPORT(function_num_regs)
CAKE_MEGAMOE_EXPORT(function_local_bytes)
CAKE_MEGAMOE_EXPORT(function_max_active_blocks)
CAKE_MEGAMOE_EXPORT(launch_cooperative)
CAKE_MEGAMOE_EXPORT(nccl_runtime_version)
