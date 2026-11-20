// NCCL-window transport of the SM90 MegaMoE kernels: symmetric windows allocated
// with ncclMemAlloc and registered on the expert-parallel communicator, their
// load/store peer pointers inside the NVLink domain, and the NCCL device
// communicator whose GPU-initiated network puts carry tokens across nodes.
// The communicator belongs to the caller's process group and is only borrowed.
#include <dlfcn.h>
#include <nccl.h>
#include <nccl_device.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

void check_nccl(ncclResult_t result, const char* what) {
  TVM_FFI_ICHECK(result == ncclSuccess) << what << " failed: " << ncclGetErrorString(result);
}

void check_cuda(cudaError_t error, const char* what) {
  TVM_FFI_ICHECK(error == cudaSuccess) << what << " failed: " << cudaGetErrorString(error);
}

struct Window {
  size_t bytes = 0;
  void* pointer = nullptr;
  ncclWindow_t window = nullptr;
  // Load/store pointer of this window on every rank of the communicator, 0 when
  // the rank is outside the NVLink domain (reachable through GIN only).
  std::vector<int64_t> peers;
};

struct Session {
  ncclComm_t comm = nullptr;
  int rank = 0, world_size = 0, device = 0;
  std::vector<Window> windows;
  bool has_dev_comm = false;
  ncclDevComm_t host_dev_comm{};
  ncclDevComm_t* device_dev_comm = nullptr;

  ~Session() { release(); }

  void release() {
    if (comm == nullptr) return;
    cudaDeviceSynchronize();
    for (auto& w : windows) {
      if (w.window != nullptr) ncclCommWindowDeregister(comm, w.window);
      if (w.pointer != nullptr) ncclMemFree(w.pointer);
      w.window = nullptr;
      w.pointer = nullptr;
    }
    windows.clear();
    if (has_dev_comm) ncclDevCommDestroy(comm, &host_dev_comm);
    has_dev_comm = false;
    if (device_dev_comm != nullptr) cudaFree(device_dev_comm);
    device_dev_comm = nullptr;
    comm = nullptr;
  }
};

std::mutex g_mutex;
std::unordered_map<int64_t, std::unique_ptr<Session>> g_sessions;
int64_t g_next_handle = 1;

Session& session_of(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_sessions.find(handle);
  TVM_FFI_ICHECK(it != g_sessions.end()) << "unknown transport session " << handle;
  return *it->second;
}

const Window& window_of(Session& s, int64_t index) {
  TVM_FFI_ICHECK(index >= 0 && index < static_cast<int64_t>(s.windows.size()))
      << "window index " << index << " out of range";
  return s.windows[static_cast<size_t>(index)];
}

}  // namespace

// Borrow the communicator and, when `gin_contexts` is positive, create the
// device communicator with that many GIN contexts (one queue pair per peer
// each), `gin_signals` signals and `gin_barriers` world barriers.
int64_t megamoe_sm90_transport_create(int64_t comm_pointer, int64_t rank, int64_t world_size,
                                      int64_t device, int64_t gin_contexts, int64_t gin_signals,
                                      int64_t gin_barriers) {
  TVM_FFI_ICHECK(comm_pointer != 0) << "the NCCL communicator pointer is null";
  TVM_FFI_ICHECK(rank >= 0 && world_size > 0 && rank < world_size) << "invalid rank or world size";
  TVM_FFI_ICHECK(gin_contexts >= 0 && gin_signals >= 0 && gin_barriers >= 0)
      << "GIN requirements must be non-negative";
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  TVM_FFI_ICHECK(current_device == device)
      << "current device " << current_device << " differs from the transport device " << device;
  int runtime_version = 0;
  check_nccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
  TVM_FFI_ICHECK(runtime_version == NCCL_VERSION_CODE)
      << "the NCCL runtime (" << runtime_version
      << ") differs from the headers this module was built with (" << NCCL_VERSION_CODE << ")";

  auto s = std::make_unique<Session>();
  s->comm = reinterpret_cast<ncclComm_t>(comm_pointer);
  s->rank = static_cast<int>(rank);
  s->world_size = static_cast<int>(world_size);
  s->device = static_cast<int>(device);
  if (gin_contexts > 0) {
    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.ginContextCount = static_cast<int>(gin_contexts);
    requirements.ginSignalCount = static_cast<int>(gin_signals > 0 ? gin_signals : 1);
    requirements.worldGinBarrierCount = static_cast<int>(gin_barriers);
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    requirements.ginStrongSignalsRequired = true;
    check_nccl(ncclDevCommCreate(s->comm, &requirements, &s->host_dev_comm), "ncclDevCommCreate");
    s->has_dev_comm = true;
    TVM_FFI_ICHECK(s->host_dev_comm.ginContextCount > 0)
        << "the communicator offers no GIN context (no GPU-initiated network path between the "
           "ranks)";
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s->device_dev_comm), sizeof(ncclDevComm_t)),
               "cudaMalloc(dev comm)");
    check_cuda(cudaMemcpy(s->device_dev_comm, &s->host_dev_comm, sizeof(ncclDevComm_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(dev comm)");
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  const int64_t handle = g_next_handle++;
  g_sessions.emplace(handle, std::move(s));
  return handle;
}

// Collective: every rank allocates a window of the same size in the same order.
// Returns the window index. The window starts zeroed.
int64_t megamoe_sm90_transport_alloc_window(int64_t handle, int64_t bytes) {
  Session& s = session_of(handle);
  TVM_FFI_ICHECK(bytes > 0) << "window size must be positive";
  Window w;
  w.bytes = static_cast<size_t>((bytes + NCCL_WIN_REQUIRED_ALIGNMENT - 1) /
                                NCCL_WIN_REQUIRED_ALIGNMENT * NCCL_WIN_REQUIRED_ALIGNMENT);
  check_nccl(ncclMemAlloc(&w.pointer, w.bytes), "ncclMemAlloc");
  const ncclResult_t registered =
      ncclCommWindowRegister(s.comm, w.pointer, w.bytes, &w.window, NCCL_WIN_COLL_SYMMETRIC);
  if (registered != ncclSuccess) {
    ncclMemFree(w.pointer);
    TVM_FFI_ICHECK(false) << "ncclCommWindowRegister failed: " << ncclGetErrorString(registered)
                          << " (without GIN every rank must be load/store reachable)";
  }
  check_cuda(cudaMemset(w.pointer, 0, w.bytes), "cudaMemset(window)");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(window)");
  w.peers.assign(static_cast<size_t>(s.world_size), 0);
  for (int peer = 0; peer < s.world_size; ++peer) {
    if (peer == s.rank) {
      w.peers[peer] = reinterpret_cast<int64_t>(w.pointer);
      continue;
    }
    void* pointer = nullptr;
    if (ncclGetPeerDevicePointer(w.window, 0, peer, &pointer) == ncclSuccess && pointer != nullptr)
      w.peers[peer] = reinterpret_cast<int64_t>(pointer);
  }
  s.windows.push_back(std::move(w));
  return static_cast<int64_t>(s.windows.size()) - 1;
}

int64_t megamoe_sm90_transport_window_pointer(int64_t handle, int64_t index) {
  return reinterpret_cast<int64_t>(window_of(session_of(handle), index).pointer);
}

int64_t megamoe_sm90_transport_window_bytes(int64_t handle, int64_t index) {
  return static_cast<int64_t>(window_of(session_of(handle), index).bytes);
}

// `ncclWindow_t` of the window, for GIN puts; never dereferenced on the host.
int64_t megamoe_sm90_transport_window_handle(int64_t handle, int64_t index) {
  return reinterpret_cast<int64_t>(window_of(session_of(handle), index).window);
}

tvm::ffi::Array<int64_t> megamoe_sm90_transport_peer_pointers(int64_t handle, int64_t index) {
  const Window& w = window_of(session_of(handle), index);
  return tvm::ffi::Array<int64_t>(w.peers.begin(), w.peers.end());
}

// Device address of the `ncclDevComm` copy the kernels read, 0 without GIN.
int64_t megamoe_sm90_transport_dev_comm_pointer(int64_t handle) {
  return reinterpret_cast<int64_t>(session_of(handle).device_dev_comm);
}

int64_t megamoe_sm90_transport_gin_context_count(int64_t handle) {
  Session& s = session_of(handle);
  return s.has_dev_comm ? s.host_dev_comm.ginContextCount : 0;
}

int64_t megamoe_sm90_transport_nccl_version() {
  int version = 0;
  check_nccl(ncclGetVersion(&version), "ncclGetVersion");
  return version;
}

// Teardown only: synchronizes the device, deregisters and frees every window
// and destroys the device communicator. The borrowed communicator is untouched.
void megamoe_sm90_transport_close(int64_t handle) {
  std::unique_ptr<Session> s;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_sessions.find(handle);
    TVM_FFI_ICHECK(it != g_sessions.end()) << "unknown transport session " << handle;
    s = std::move(it->second);
    g_sessions.erase(it);
  }
  s->release();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_create, megamoe_sm90_transport_create);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_alloc_window,
                              megamoe_sm90_transport_alloc_window);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_window_pointer,
                              megamoe_sm90_transport_window_pointer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_window_bytes,
                              megamoe_sm90_transport_window_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_window_handle,
                              megamoe_sm90_transport_window_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_peer_pointers,
                              megamoe_sm90_transport_peer_pointers);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_dev_comm_pointer,
                              megamoe_sm90_transport_dev_comm_pointer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_gin_context_count,
                              megamoe_sm90_transport_gin_context_count);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_nccl_version,
                              megamoe_sm90_transport_nccl_version);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_transport_close, megamoe_sm90_transport_close);
