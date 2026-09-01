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

#pragma once

// Host half of the experimental PCIe Ulysses backend: verbs/mlx5 RC-QP
// transport, buffer registration, and copy-engine scheduling. It speaks
// TVM-FFI types, so it is a private header of csrc/ulysses_pcie.cu, not an
// include/ kernel.

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flashinfer/comm/ulysses_pcie.cuh"

namespace flashinfer {
namespace comm {
namespace ulysses_pcie {

using tvm::ffi::Array;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

constexpr int kWorld = 8;
constexpr int kPort = 1;
constexpr int64_t kMaxInterleavedStride = 65535;
constexpr auto kTimeout = std::chrono::seconds(10);

struct GroupWire {
  uint32_t qpn[kWorld]{};
  uint32_t psn[kWorld]{};
  uint32_t mtu = 0;
  uint8_t gid[16]{};
};

struct BufferWire {
  uint64_t address = 0;
  uint32_t rkey = 0;
  uint32_t destination_rkey[kWorld]{};
  cudaIpcMemHandle_t ipc{};
  cudaIpcMemHandle_t signal_ipc{};
};

template <typename T>
Array<int64_t> Encode(const T& value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  Array<int64_t> result;
  for (size_t i = 0; i < sizeof(T); ++i) result.push_back(bytes[i]);
  return result;
}

template <typename T>
T DecodeAt(const Array<int64_t>& bytes, size_t offset) {
  TVM_FFI_ICHECK_LE(offset + sizeof(T), bytes.size()) << "truncated PCIe metadata";
  T result{};
  auto* output = reinterpret_cast<uint8_t*>(&result);
  for (size_t i = 0; i < sizeof(T); ++i) {
    const int64_t value = bytes[offset + i];
    TVM_FFI_ICHECK(value >= 0 && value <= 255) << "invalid PCIe metadata byte";
    output[i] = static_cast<uint8_t>(value);
  }
  return result;
}

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << operation << ": " << cudaGetErrorString(status);
}

inline cudaError_t QueryEventUntil(cudaEvent_t event,
                                   std::chrono::steady_clock::time_point deadline) noexcept {
  while (true) {
    const cudaError_t status = cudaEventQuery(event);
    if (status != cudaErrorNotReady) return status;
    if (std::chrono::steady_clock::now() >= deadline) return cudaErrorNotReady;
    std::this_thread::yield();
  }
}

// Blocks in the driver rather than polling: a cudaEventQuery spin loop on the
// hot barrier path floods CUDA API traces. Unbounded on purpose — a barrier
// that never completes means a peer process died, which by design is the
// launcher's problem (see the design doc). A peer that aborts while running
// publishes its signal, so the barrier still completes and the snapshot read
// after this call reports it; a stalled NIC stays bounded by kTimeout in
// Poll. QueryEventUntil above remains for the cold teardown and abort paths.
inline void WaitEvent(cudaEvent_t event, const char* operation) {
  CheckCuda(cudaEventSynchronize(event), operation);
}

inline void CheckVerbs(int status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, 0) << operation << ": " << std::strerror(status > 0 ? status : errno);
}

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device) noexcept : target_(device) {
    if (cudaGetDevice(&previous_) != cudaSuccess) return;
    if (previous_ != target_ && cudaSetDevice(target_) != cudaSuccess) return;
    active_ = true;
  }

  ~ScopedCudaDevice() noexcept {
    if (active_ && previous_ != target_) cudaSetDevice(previous_);
  }

  bool active() const { return active_; }

 private:
  int previous_ = 0;
  int target_ = 0;
  bool active_ = false;
};

inline void CheckNoOverlap(const void* input, int64_t input_bytes, const void* output,
                           int64_t output_bytes) {
  const auto* input_begin = static_cast<const char*>(input);
  const auto* output_begin = static_cast<const char*>(output);
  TVM_FFI_ICHECK(input_begin + input_bytes <= output_begin ||
                 output_begin + output_bytes <= input_begin)
      << "PCIe Ulysses input and output byte ranges must not overlap";
}

inline ibv_mr* RegisterGpuMr(ibv_pd* pd, void* pointer, size_t bytes, int access) {
  TVM_FFI_ICHECK(pointer != nullptr) << "cannot register a null GPU pointer";
  // Two registration paths: legacy cudaMalloc pointers need SYNC_MEMOPS set
  // before ibv_reg_mr; VMM-backed allocations (e.g. torch symmetric memory)
  // reject both and go through dma-buf. SYNC_MEMOPS is allocation-scoped, so
  // only allocations this transport owns are registered.
  unsigned int sync_memops = 1;
  const CUresult sync_status = cuPointerSetAttribute(&sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                                     reinterpret_cast<CUdeviceptr>(pointer));
  int direct_errno = 0;
  if (sync_status == CUDA_SUCCESS) {
    if (auto* mr = ibv_reg_mr(pd, pointer, bytes, access)) return mr;
    direct_errno = errno;
  }

  const long page_size_value = ::sysconf(_SC_PAGESIZE);
  TVM_FFI_ICHECK_GT(page_size_value, 0) << "could not read the host page size";
  const auto page_size = static_cast<size_t>(page_size_value);
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  const size_t offset = address % page_size;
  TVM_FFI_ICHECK_LE(bytes, std::numeric_limits<size_t>::max() - offset)
      << "dma-buf export size exceeds size_t range";
  const size_t span = offset + bytes;
  const size_t remainder = span % page_size;
  const size_t padding = remainder == 0 ? 0 : page_size - remainder;
  TVM_FFI_ICHECK_LE(span, std::numeric_limits<size_t>::max() - padding)
      << "dma-buf export size exceeds size_t range";
  const CUdeviceptr base = static_cast<CUdeviceptr>(address - offset);
  const size_t export_bytes = span + padding;
  int fd = -1;
  const CUresult export_status = cuMemGetHandleForAddressRange(
      &fd, base, export_bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  TVM_FFI_ICHECK_EQ(export_status, CUDA_SUCCESS)
      << "direct GPU MR registration failed"
      << (direct_errno == 0 ? "" : std::string(": ") + std::strerror(direct_errno))
      << ", and the allocation cannot be exported as dma-buf (CUresult " << export_status << ")";
  auto* mr = ibv_reg_dmabuf_mr(pd, offset, bytes, address, fd, access);
  const int dmabuf_errno = errno;
  ::close(fd);
  TVM_FFI_ICHECK(mr != nullptr) << "ibv_reg_dmabuf_mr failed: " << std::strerror(dmabuf_errno);
  return mr;
}

class Transport;

class Buffer {
 public:
  Transport* transport = nullptr;
  int mode = 0;
  int64_t batch = 0;
  int64_t seq = 0;
  int64_t heads = 0;
  int64_t dim = 0;
  int64_t element_size = 0;
  DLDataType dtype{};
  void* output_pointer = nullptr;
  int64_t output_bytes = 0;
  // Bytes actually registered (MR / IPC export / landing). The bound
  // geometry above may describe fewer bytes: a slot is registered once at
  // the communicator's declared capacity and re-bound per call.
  int64_t capacity_bytes = 0;
  std::unique_ptr<Tensor> output_owner;
  ibv_mr* output_mr = nullptr;
  std::array<mlx5dv_mkey*, kWorld> destination_mkeys{};
  ibv_mr* landing_mr = nullptr;
  std::array<mlx5dv_mkey*, kWorld> landing_source_mkeys{};
  // The staging buffer the NIC reads out of, held as an owning Tensor so its
  // lifetime is tied to this registration rather than to any caller.
  std::unique_ptr<Tensor> landing_owner;
  void* input_landing = nullptr;
  std::array<BufferWire, kWorld> peers{};
  std::array<void*, kWorld> peer_pointers{};
  uint64_t* signals = nullptr;
  std::array<uint64_t*, kWorld> peer_signals{};
  // The barrier epoch, advanced by the barrier kernel itself (see AdvanceEpoch).
  uint64_t* epoch_device = nullptr;
  // Host-side counter that tags this exchange's RDMA work requests; the
  // hybrid route is host-synchronous anyway.
  uint64_t epoch = 0;
  bool connected = false;
  bool disconnect_required = false;
  bool imports_closed = true;
  bool released = false;

  void Disconnect();
  void Release();
  ~Buffer();
};

class Transport {
 public:
  int rank = 0;
  int world_size = 0;
  int device = 0;
  std::array<int, kWorld> numa_nodes{};
  std::string nic_name;
  int gid_index = -1;
  int write_ordering = cudaGPUDirectRDMAWritesOrderingNone;
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  std::array<ibv_qp*, kWorld> qps{};
  std::array<ibv_qp_ex*, kWorld> qpxs{};
  std::array<mlx5dv_qp_ex*, kWorld> mlx5_qpxs{};
  GroupWire local{};
  std::array<GroupWire, kWorld> peers{};
  uint64_t next_wr_id = 1;
  std::array<uint64_t, kWorld> outstanding_send_wrs{};
  std::array<uint64_t, kWorld> outstanding_recv_wrs{};
  std::vector<cudaStream_t> streams;
  cudaEvent_t input_ready = nullptr;
  std::vector<cudaEvent_t> copy_done;
  cudaStream_t abort_stream = nullptr;
  cudaEvent_t abort_done = nullptr;
  uint64_t* abort_snapshot = nullptr;
  std::unordered_map<void*, std::unique_ptr<Buffer>> buffers;
  bool use_rdma = true;
  bool connected = false;
  bool failed = false;
  bool phase_inflight = false;
  bool teardown_safe = true;
  bool unsafe_release = false;

  void ValidatePlannedGid(ibv_port_attr* port_out, ibv_gid* gid_out) const {
    TVM_FFI_ICHECK(context != nullptr)
        << "cannot validate the planned GID without an open verbs context";
    ibv_port_attr port{};
    CheckVerbs(ibv_query_port(context, kPort, &port), "ibv_query_port");
    TVM_FFI_ICHECK_EQ(port.state, IBV_PORT_ACTIVE) << nic_name << " port 1 is not active";
    TVM_FFI_ICHECK_EQ(port.link_layer, IBV_LINK_LAYER_ETHERNET)
        << nic_name << " port 1 is not Ethernet/RoCE";
    TVM_FFI_ICHECK_LT(gid_index, port.gid_tbl_len)
        << "configured GID index " << gid_index << " is outside " << nic_name
        << " port 1 table length " << port.gid_tbl_len;

    ibv_gid_entry gid_entry{};
    CheckVerbs(ibv_query_gid_ex(context, kPort, gid_index, &gid_entry, 0), "ibv_query_gid_ex");
    TVM_FFI_ICHECK_EQ(gid_entry.port_num, kPort) << "GID entry changed port during initialization";
    TVM_FFI_ICHECK_EQ(gid_entry.gid_index, gid_index)
        << "GID entry changed index during initialization";
    TVM_FFI_ICHECK_EQ(gid_entry.gid_type, IBV_GID_TYPE_ROCE_V2)
        << "selected GID is no longer RoCE v2";
    TVM_FFI_ICHECK_NE(gid_entry.ndev_ifindex, 0)
        << "selected GID no longer has an associated netdev";
    const auto* raw = gid_entry.gid.raw;
    const bool ipv4_mapped = std::all_of(raw, raw + 10, [](uint8_t byte) { return byte == 0; }) &&
                             raw[10] == 0xff && raw[11] == 0xff;
    const bool ipv4_nonzero =
        std::any_of(raw + 12, raw + 16, [](uint8_t byte) { return byte != 0; });
    TVM_FFI_ICHECK(ipv4_mapped && ipv4_nonzero)
        << "selected GID is no longer a non-zero IPv4-mapped address";

    if (port_out != nullptr) *port_out = port;
    if (gid_out != nullptr) std::memcpy(gid_out, &gid_entry.gid, sizeof(gid_entry.gid));
  }

  // Validated in the member initializer: on failure nothing is constructed
  // and the catch below (whose Release() indexes kWorld-sized arrays by
  // world_size) never runs.
  static int CheckedWorldSize(int world_size) {
    TVM_FFI_ICHECK(world_size == 1 || world_size == 2 || world_size == 4 || world_size == 8)
        << "PCIe Ulysses world_size must be 1, 2, 4, or 8";
    return world_size;
  }

  Transport(int rank_arg, int world_size_arg, int device_arg, const Array<int64_t>& numa,
            std::string nic_name_arg, bool use_rdma_arg, int gid_index_arg)
      : rank(rank_arg),
        world_size(CheckedWorldSize(world_size_arg)),
        device(device_arg),
        nic_name(std::move(nic_name_arg)),
        gid_index(gid_index_arg),
        use_rdma(use_rdma_arg) {
    try {
      TVM_FFI_ICHECK(rank >= 0 && rank < world_size) << "invalid rank";
      TVM_FFI_ICHECK_EQ(numa.size(), static_cast<size_t>(world_size))
          << "expected one NUMA node per rank";
      for (int peer = 0; peer < world_size; ++peer) numa_nodes[peer] = static_cast<int>(numa[peer]);

      ScopedCudaDevice device_guard(device);
      streams.resize(2);
      copy_done.resize(streams.size());
      CheckCuda(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming),
                "cudaEventCreateWithFlags(input ready)");
      for (size_t index = 0; index < streams.size(); ++index) {
        CheckCuda(cudaStreamCreateWithFlags(&streams[index], cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
        CheckCuda(cudaEventCreateWithFlags(&copy_done[index], cudaEventDisableTiming),
                  "cudaEventCreateWithFlags(copy done)");
      }

      if (!use_rdma) {
        TVM_FFI_ICHECK_EQ(gid_index, -1) << "all-P2P PCIe Ulysses must not select an RDMA GID";
        return;
      }
      // The numa array carries route-group ids: a peer in another group is
      // reached over RDMA. Hybrid passes physical NUMA nodes (8 ranks, 4+4);
      // the all-RDMA route passes singleton groups at any world size.
      const int local_count = static_cast<int>(
          std::count(numa_nodes.begin(), numa_nodes.begin() + world_size, numa_nodes[rank]));
      TVM_FFI_ICHECK(local_count == 1 || (world_size == kWorld && local_count == 4))
          << "RDMA PCIe Ulysses requires singleton route groups or an 8-rank 4+4 split";
      TVM_FFI_ICHECK(!nic_name.empty()) << "RDMA PCIe Ulysses requires an mlx5 NIC";
      TVM_FFI_ICHECK_GE(gid_index, 0) << "RDMA PCIe Ulysses requires an explicit GID index";

      int least_priority = 0;
      int greatest_priority = 0;
      CheckCuda(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
                "cudaDeviceGetStreamPriorityRange");
      CheckCuda(
          cudaStreamCreateWithPriority(&abort_stream, cudaStreamNonBlocking, greatest_priority),
          "cudaStreamCreateWithPriority(abort)");
      CheckCuda(cudaEventCreateWithFlags(&abort_done, cudaEventDisableTiming),
                "cudaEventCreateWithFlags(abort done)");
      CheckCuda(
          cudaMallocHost(reinterpret_cast<void**>(&abort_snapshot), world_size * sizeof(uint64_t)),
          "cudaMallocHost(abort snapshot)");
      std::memset(abort_snapshot, 0, world_size * sizeof(uint64_t));

      CheckCuda(
          cudaDeviceGetAttribute(&write_ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device),
          "cudaDeviceGetAttribute(GPUDirectRDMAWritesOrdering)");
      if (write_ordering < cudaGPUDirectRDMAWritesOrderingOwner) {
        int flush_options = 0;
        CheckCuda(cudaDeviceGetAttribute(&flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                                         device),
                  "cudaDeviceGetAttribute(GPUDirectRDMAFlushWritesOptions)");
        TVM_FFI_ICHECK(flush_options & cudaFlushGPUDirectRDMAWritesOptionHost)
            << "RDMA PCIe Ulysses requires cudaDeviceFlushGPUDirectRDMAWrites(), but device "
            << device << " does not advertise the host flush option";
      }

      int count = 0;
      ibv_device** list = ibv_get_device_list(&count);
      TVM_FFI_ICHECK(list != nullptr) << "ibv_get_device_list failed: " << std::strerror(errno);
      mlx5dv_context_attr context_attr{};
      context_attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
      for (int i = 0; i < count; ++i) {
        if (nic_name == ibv_get_device_name(list[i])) {
          context = mlx5dv_open_device(list[i], &context_attr);
          break;
        }
      }
      ibv_free_device_list(list);
      TVM_FFI_ICHECK(context != nullptr)
          << "cannot open " << nic_name << " with DEVX: " << std::strerror(errno);
      pd = ibv_alloc_pd(context);
      TVM_FFI_ICHECK(pd != nullptr) << "ibv_alloc_pd failed: " << std::strerror(errno);
      cq = ibv_create_cq(context, 256, nullptr, nullptr, 0);
      TVM_FFI_ICHECK(cq != nullptr) << "ibv_create_cq failed: " << std::strerror(errno);

      ibv_port_attr port{};
      ibv_gid gid{};
      ValidatePlannedGid(&port, &gid);
      local.mtu = port.active_mtu;
      std::memcpy(local.gid, &gid, sizeof(gid));

      for (int peer = 0; peer < world_size; ++peer) {
        if (!Cross(peer)) continue;
        ibv_qp_init_attr_ex qp_attr{};
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.cap.max_send_wr = 128;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 128;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        qp_attr.pd = pd;
        qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
        mlx5dv_qp_init_attr dv_attr{};
        dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
        dv_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MKEY_CONFIGURE;
        auto* qp = mlx5dv_create_qp(context, &qp_attr, &dv_attr);
        TVM_FFI_ICHECK(qp != nullptr) << "mlx5dv_create_qp failed: " << std::strerror(errno);
        qps[peer] = qp;
        qpxs[peer] = ibv_qp_to_qp_ex(qp);
        mlx5_qpxs[peer] = mlx5dv_qp_ex_from_ibv_qp_ex(qpxs[peer]);
        TVM_FFI_ICHECK(qpxs[peer] != nullptr && mlx5_qpxs[peer] != nullptr)
            << "cannot create extended mlx5 QP";
        local.qpn[peer] = qp->qp_num;
        local.psn[peer] = 0x120000 + rank * 0x1000 + peer * 0x10;
      }

    } catch (...) {
      Release();
      throw;
    }
  }

  // Leak on purpose: destroying a PD or MR while a peer still holds an import
  // is undefined behaviour, and this runs on a noexcept destructor path.
  void LeakAndPoison(const char* reason) noexcept {
    std::fprintf(stderr,
                 "FlashInfer PCIe Ulysses: %s; verbs and CUDA resources are "
                 "intentionally leaked. Call close() on every rank before "
                 "dropping the communicator.\n",
                 reason);
    for (auto& entry : buffers) entry.second.release();
    buffers.clear();
  }

  void Release() noexcept {
    ScopedCudaDevice device_guard(device);
    if (!TeardownSafe()) {
      return LeakAndPoison("outstanding RDMA GPU work could not be bounded before teardown");
    }
    for (const auto& [pointer, buffer] : buffers) {
      if (buffer->imports_closed) continue;
      return LeakAndPoison("a transport was torn down while an output still had peer imports");
    }
    if (unsafe_release) {
      return LeakAndPoison("an earlier resource teardown lost its retry ledger");
    }
    // Buffer MRs/MKeys refer to this PD and must be destroyed first.
    try {
      for (auto& [pointer, buffer] : buffers) buffer->Release();
    } catch (...) {
      return LeakAndPoison("Buffer teardown failed");
    }
    buffers.clear();
    for (auto& event : copy_done) {
      if (event != nullptr) cudaEventDestroy(event);
      event = nullptr;
    }
    copy_done.clear();
    if (input_ready != nullptr) cudaEventDestroy(input_ready);
    input_ready = nullptr;
    if (abort_done != nullptr) cudaEventDestroy(abort_done);
    abort_done = nullptr;
    for (auto& stream : streams) {
      if (stream != nullptr) cudaStreamDestroy(stream);
      stream = nullptr;
    }
    streams.clear();
    if (abort_stream != nullptr) cudaStreamDestroy(abort_stream);
    abort_stream = nullptr;
    if (abort_snapshot != nullptr) cudaFreeHost(abort_snapshot);
    abort_snapshot = nullptr;
    for (auto*& qp : qps) {
      if (qp != nullptr) ibv_destroy_qp(qp);
      qp = nullptr;
    }
    if (cq != nullptr) ibv_destroy_cq(cq);
    cq = nullptr;
    if (pd != nullptr) ibv_dealloc_pd(pd);
    pd = nullptr;
    if (context != nullptr) ibv_close_device(context);
    context = nullptr;
  }

  ~Transport() { Release(); }

  bool Cross(int peer) const { return numa_nodes[peer] != numa_nodes[rank]; }

  void EnsureHealthy() const {
    TVM_FFI_ICHECK(!failed) << "PCIe Ulysses transport is poisoned by an earlier native failure";
  }

  bool TeardownSafe() const noexcept {
    return teardown_safe && !phase_inflight && !unsafe_release && OutstandingWrs() == 0;
  }

  uint64_t NewWrId(int peer, bool receive) {
    TVM_FFI_ICHECK(peer >= 0 && peer < world_size) << "invalid WR peer " << peer;
    // Keep the peer and direction in the completion itself.  The shared CQ can
    // then retire the exact per-QP ledger even while several QPs are being
    // flushed after a failure.
    constexpr uint64_t kPeerMask = 0x0f;
    constexpr uint64_t kReceiveBit = 0x10;
    constexpr unsigned kMetadataBits = 8;
    TVM_FFI_ICHECK_LT(next_wr_id, uint64_t{1} << (64 - kMetadataBits))
        << "PCIe Ulysses WR id space exhausted";
    const uint64_t result =
        (next_wr_id++ << kMetadataBits) | (receive ? kReceiveBit : 0) | static_cast<uint64_t>(peer);
    TVM_FFI_ICHECK_EQ(result & kPeerMask, static_cast<uint64_t>(peer));
    return result;
  }

  bool RetireCompletion(const ibv_wc& completion) noexcept {
    constexpr uint64_t kPeerMask = 0x0f;
    constexpr uint64_t kReceiveBit = 0x10;
    const int peer = static_cast<int>(completion.wr_id & kPeerMask);
    if (peer < 0 || peer >= world_size) return false;
    auto& outstanding = (completion.wr_id & kReceiveBit) != 0 ? outstanding_recv_wrs[peer]
                                                              : outstanding_send_wrs[peer];
    if (outstanding == 0) return false;
    --outstanding;
    return true;
  }

  uint64_t OutstandingWrs() const noexcept {
    uint64_t result = 0;
    for (int peer = 0; peer < world_size; ++peer) {
      result += outstanding_send_wrs[peer];
      result += outstanding_recv_wrs[peer];
    }
    return result;
  }

  // The hybrid route is deliberately host-synchronous. The abort readback is
  // included in the phase event so the snapshot is complete when the wait
  // returns.
  void RunHybridBarrier(Buffer* buffer, cudaStream_t stream, bool opening) {
    TVM_FFI_ICHECK(use_rdma) << "abort-aware barriers are RDMA-route-only";
    TVM_FFI_ICHECK(abort_snapshot != nullptr) << "missing RDMA abort snapshot";
    phase_inflight = true;
    CheckCuda(EnqueueBarrier(buffer->signals, buffer->peer_signals.data(), world_size, rank,
                             buffer->epoch_device, stream),
              opening ? "enqueue RDMA opening barrier" : "enqueue RDMA closing barrier");
    CheckCuda(cudaMemcpyAsync(abort_snapshot, buffer->signals + world_size,
                              world_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(RDMA abort snapshot)");
    CheckCuda(cudaEventRecord(input_ready, stream),
              opening ? "cudaEventRecord(RDMA opening)" : "cudaEventRecord(RDMA closing)");
    WaitEvent(input_ready,
              opening ? "wait for RDMA opening barrier" : "wait for RDMA closing barrier");
    phase_inflight = false;
    for (int peer = 0; peer < world_size; ++peer) {
      TVM_FFI_ICHECK_NE(abort_snapshot[peer], kAbortSignal)
          << "RDMA PCIe Ulysses peer " << peer << " aborted the exchange";
    }
  }

  bool TryPublishAbort(Buffer* buffer) noexcept {
    if (!use_rdma || buffer == nullptr || abort_stream == nullptr || abort_done == nullptr)
      return false;
    if (EnqueueAbort(buffer->peer_signals.data(), world_size, rank, abort_stream) != cudaSuccess)
      return false;
    if (cudaEventRecord(abort_done, abort_stream) != cudaSuccess) return false;
    return QueryEventUntil(abort_done, std::chrono::steady_clock::now() + kTimeout) == cudaSuccess;
  }

  // Record every stream before teardown.  A successful common-deadline wait
  // proves the abort-released barrier and all partially queued local copies no
  // longer reference IPC mappings or registered allocations.
  bool TryDrainHybrid(cudaStream_t current) noexcept {
    if (input_ready == nullptr || cudaEventRecord(input_ready, current) != cudaSuccess)
      return false;
    for (size_t index = 0; index < streams.size(); ++index) {
      if (copy_done[index] == nullptr ||
          cudaEventRecord(copy_done[index], streams[index]) != cudaSuccess)
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    if (QueryEventUntil(input_ready, deadline) != cudaSuccess) return false;
    for (const auto event : copy_done)
      if (QueryEventUntil(event, deadline) != cudaSuccess) return false;
    phase_inflight = false;
    return true;
  }

  bool Quiesce() noexcept {
    failed = true;
    bool safe = true;
    for (int peer = 0; peer < world_size; ++peer) {
      auto* qp = qps[peer];
      if (qp == nullptr) continue;
      ibv_qp_attr attr{};
      attr.qp_state = IBV_QPS_ERR;
      const int status = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
      if (status != 0 && (outstanding_send_wrs[peer] != 0 || outstanding_recv_wrs[peer] != 0)) {
        safe = false;
      }
    }
    if (cq == nullptr) {
      safe = safe && OutstandingWrs() == 0;
      if (!safe) teardown_safe = false;
      return safe;
    }

    // Every posted WR is signaled and carries its QP/direction in wr_id. After
    // moving the QPs to ERR, wait for each one's ordinary or flush CQE — an
    // empty poll is not proof. Registered memory may only be released once the
    // ledger is empty.
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (OutstandingWrs() != 0 && std::chrono::steady_clock::now() < deadline) {
      ibv_wc entries[kWorld]{};
      const int count = ibv_poll_cq(cq, kWorld, entries);
      if (count < 0) {
        safe = false;
        break;
      }
      if (count == 0) {
        std::this_thread::yield();
        continue;
      }
      for (int index = 0; index < count; ++index) {
        if (!RetireCompletion(entries[index])) safe = false;
        if (entries[index].status != IBV_WC_SUCCESS &&
            entries[index].status != IBV_WC_WR_FLUSH_ERR) {
          safe = false;
        }
      }
    }
    if (OutstandingWrs() != 0) safe = false;
    if (!safe) teardown_safe = false;
    return safe;
  }

  void AbortAndQuiesce(Buffer* buffer, cudaStream_t current) noexcept {
    failed = true;
    const bool abort_published = TryPublishAbort(buffer);
    const bool streams_drained = TryDrainHybrid(current);
    const bool verbs_drained = Quiesce();
    if (!abort_published || !streams_drained || !verbs_drained) teardown_safe = false;
  }

  template <typename Body>
  void RetireOnFailure(Body&& body) {
    EnsureHealthy();
    try {
      body();
    } catch (...) {
      Quiesce();
      throw;
    }
  }

  // With expected_receives == 0 completions are only counted. Otherwise every
  // receive must carry this exchange's immediate tag; a completion from a
  // previous epoch would otherwise let the closing barrier publish before
  // peer data landed.
  void Poll(int expected_sends, int expected_receives = 0, uint32_t immediate = 0) {
    EnsureHealthy();
    int sends = 0;
    int receives = 0;
    unsigned empty_polls = 0;
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (sends < expected_sends || receives < expected_receives) {
      ibv_wc entries[kWorld]{};
      const int count = ibv_poll_cq(cq, kWorld, entries);
      if (count < 0) teardown_safe = false;
      TVM_FFI_ICHECK_GE(count, 0) << "ibv_poll_cq failed";
      bool ledger_ok = true;
      for (int i = 0; i < count; ++i) ledger_ok = RetireCompletion(entries[i]) && ledger_ok;
      if (!ledger_ok) teardown_safe = false;
      TVM_FFI_ICHECK(ledger_ok) << "mlx5 completion does not match an outstanding WR";
      for (int i = 0; i < count; ++i) {
        const auto& entry = entries[i];
        if (entry.status != IBV_WC_SUCCESS) teardown_safe = false;
        TVM_FFI_ICHECK_EQ(entry.status, IBV_WC_SUCCESS)
            << "mlx5 completion failed: " << ibv_wc_status_str(entry.status)
            << " vendor_err=" << entry.vendor_err;
        if (expected_receives == 0) {
          ++sends;
        } else if (entry.opcode == IBV_WC_RDMA_WRITE) {
          ++sends;
        } else if (entry.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          TVM_FFI_ICHECK(entry.wc_flags & IBV_WC_WITH_IMM)
              << "mlx5 receive completion is missing immediate data";
          TVM_FFI_ICHECK_EQ(ntohl(entry.imm_data), immediate)
              << "mlx5 receive completion belongs to another exchange epoch";
          ++receives;
        } else {
          TVM_FFI_ICHECK(false) << "unexpected mlx5 completion opcode " << entry.opcode;
        }
      }
      TVM_FFI_ICHECK_LE(sends, expected_sends) << "too many mlx5 send completions";
      TVM_FFI_ICHECK_LE(receives, expected_receives) << "too many mlx5 receive completions";
      if (count == 0 && ++empty_polls == 1024) {
        TVM_FFI_ICHECK(std::chrono::steady_clock::now() < deadline)
            << "timed out waiting for mlx5 completions";
        empty_polls = 0;
      }
    }
  }

  mlx5dv_mkey* CreateMkey() {
    mlx5dv_mkey_init_attr attr{};
    attr.pd = pd;
    attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    attr.max_entries = 2;
    auto* result = mlx5dv_create_mkey(&attr);
    TVM_FFI_ICHECK(result != nullptr) << "mlx5dv_create_mkey failed: " << std::strerror(errno);
    return result;
  }

  void ConfigureMkey(int peer, mlx5dv_mkey* mkey, uint32_t access, uint64_t address, uint32_t width,
                     uint32_t skip, uint32_t rows, uint32_t lkey) {
    mlx5dv_mkey_conf_attr config{};
    mlx5dv_mr_interleaved layout{};
    layout.addr = address;
    layout.bytes_count = width;
    layout.bytes_skip = skip;
    layout.lkey = lkey;
    ibv_wr_start(qpxs[peer]);
    qpxs[peer]->wr_id = NewWrId(peer, false);
    qpxs[peer]->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    mlx5dv_wr_mkey_configure(mlx5_qpxs[peer], mkey, 2, &config);
    mlx5dv_wr_set_mkey_access_flags(mlx5_qpxs[peer], access);
    mlx5dv_wr_set_mkey_layout_interleaved(mlx5_qpxs[peer], rows, 1, &layout);
    const int status = ibv_wr_complete(qpxs[peer]);
    if (status == 0) ++outstanding_send_wrs[peer];
    TVM_FFI_ICHECK_EQ(status, 0) << "configure interleaved MKey failed: " << std::strerror(status);
  }

  void PostWrite(int peer, uint32_t local_key, uint64_t local_address, uint32_t bytes,
                 uint32_t remote_key, uint64_t remote_address, uint32_t immediate) {
    auto* qp = qpxs[peer];
    ibv_wr_start(qp);
    qp->wr_id = NewWrId(peer, false);
    qp->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write_imm(qp, remote_key, remote_address, htonl(immediate));
    ibv_wr_set_sge(qp, local_key, local_address, bytes);
    const int status = ibv_wr_complete(qp);
    if (status == 0) ++outstanding_send_wrs[peer];
    TVM_FFI_ICHECK_EQ(status, 0) << "post RDMA write failed: " << std::strerror(status);
  }

  void PostReceive(int peer) {
    ibv_recv_wr request{};
    request.wr_id = NewWrId(peer, true);
    ibv_recv_wr* bad = nullptr;
    const int status = ibv_post_recv(qps[peer], &request, &bad);
    if (status == 0) ++outstanding_recv_wrs[peer];
    TVM_FFI_ICHECK_EQ(status, 0) << "post RDMA receive failed: " << std::strerror(status);
  }

  int CrossCount() const {
    int count = 0;
    for (int peer = 0; peer < world_size; ++peer)
      if (Cross(peer)) ++count;
    return count;
  }

  void Connect(const Array<int64_t>& flat) {
    EnsureHealthy();
    TVM_FFI_ICHECK(!connected) << "PCIe Ulysses transport is already connected";
    TVM_FFI_ICHECK_EQ(flat.size(), static_cast<size_t>(world_size * sizeof(GroupWire)))
        << "invalid group metadata length";
    for (int peer = 0; peer < world_size; ++peer)
      peers[peer] = DecodeAt<GroupWire>(flat, peer * sizeof(GroupWire));

    if (!use_rdma) {
      connected = true;
      return;
    }

    RetireOnFailure([&] {
      // The rank-ordered connection metadata freezes the selected source GID
      // and active MTU. Revalidate both immediately before the QPs consume the
      // GID table index; a sysfs/netdev reconfiguration between construction
      // and INIT->RTR must fail here rather than at the first data WR.
      ibv_port_attr current_port{};
      ibv_gid current_gid{};
      ValidatePlannedGid(&current_port, &current_gid);
      TVM_FFI_ICHECK_EQ(current_port.active_mtu, local.mtu)
          << "selected RDMA port MTU changed after connection metadata exchange";
      TVM_FFI_ICHECK_EQ(std::memcmp(current_gid.raw, local.gid, sizeof(local.gid)), 0)
          << "selected GID changed after connection metadata exchange";
      for (int peer = 0; peer < world_size; ++peer) {
        auto* qp = qps[peer];
        if (qp == nullptr) continue;
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kPort;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        CheckVerbs(
            ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
            "QP RESET->INIT");

        attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = static_cast<ibv_mtu>(std::min(local.mtu, peers[peer].mtu));
        attr.dest_qp_num = peers[peer].qpn[rank];
        attr.rq_psn = peers[peer].psn[rank];
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = kPort;
        std::memcpy(&attr.ah_attr.grh.dgid, peers[peer].gid, 16);
        attr.ah_attr.grh.sgid_index = gid_index;
        attr.ah_attr.grh.hop_limit = 64;
        CheckVerbs(
            ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
            "QP INIT->RTR");

        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 18;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = local.psn[peer];
        attr.max_rd_atomic = 1;
        CheckVerbs(ibv_modify_qp(qp, &attr,
                                 IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                     IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
                   "QP RTR->RTS");
      }
    });
    connected = true;
  }
};

inline void Buffer::Disconnect() {
  if (imports_closed) return;
  connected = false;

  cudaError_t first_error = cudaSuccess;
  auto close_import = [&](void*& pointer, const void* local) {
    if (pointer == nullptr || pointer == local) return;
    const cudaError_t status = cudaIpcCloseMemHandle(pointer);
    if (status == cudaSuccess) {
      pointer = nullptr;
    } else if (first_error == cudaSuccess) {
      first_error = status;
    }
  };

  for (auto*& pointer : peer_pointers) close_import(pointer, output_pointer);
  for (auto*& pointer : peer_signals) {
    void* untyped = pointer;
    close_import(untyped, signals);
    pointer = static_cast<uint64_t*>(untyped);
  }
  if (first_error == cudaSuccess) {
    imports_closed = true;
  } else {
    CheckCuda(first_error, "cudaIpcCloseMemHandle(PCIe Ulysses output imports)");
  }
}

inline void Buffer::Release() {
  if (released) return;
  TVM_FFI_ICHECK(imports_closed)
      << "disconnect PCIe Ulysses peer imports before releasing a Buffer";
  ScopedCudaDevice device_guard(transport == nullptr ? 0 : transport->device);

  for (auto*& mkey : landing_source_mkeys) {
    if (mkey == nullptr) continue;
    const int status = mlx5dv_destroy_mkey(mkey);
    if (status == 0) mkey = nullptr;
    CheckVerbs(status, "mlx5dv_destroy_mkey(input landing)");
  }
  if (landing_mr != nullptr) {
    const int status = ibv_dereg_mr(landing_mr);
    if (status == 0) landing_mr = nullptr;
    CheckVerbs(status, "ibv_dereg_mr(input landing)");
  }
  // Only after the MKeys and the MR above are gone: nothing may still be
  // registered over the landing Tensor when it is released.
  input_landing = nullptr;
  landing_owner.reset();

  for (auto*& mkey : destination_mkeys) {
    if (mkey == nullptr) continue;
    const int status = mlx5dv_destroy_mkey(mkey);
    if (status == 0) mkey = nullptr;
    CheckVerbs(status, "mlx5dv_destroy_mkey(output)");
  }
  if (output_mr != nullptr) {
    const int status = ibv_dereg_mr(output_mr);
    if (status == 0) output_mr = nullptr;
    CheckVerbs(status, "ibv_dereg_mr(output)");
  }
  if (epoch_device != nullptr) {
    const cudaError_t status = cudaFree(epoch_device);
    if (status == cudaSuccess) epoch_device = nullptr;
    CheckCuda(status, "cudaFree(PCIe Ulysses epoch)");
  }
  if (signals != nullptr) {
    const cudaError_t status = cudaFree(signals);
    if (status == cudaSuccess) signals = nullptr;
    CheckCuda(status, "cudaFree(PCIe Ulysses signals)");
  }
  output_owner.reset();
  released = true;
}

inline Buffer::~Buffer() {
  bool safe = imports_closed && (transport == nullptr || transport->TeardownSafe());
  if (safe) {
    try {
      Release();
      return;
    } catch (...) {
      safe = false;
    }
  }
  std::fprintf(stderr,
               "FlashInfer PCIe Ulysses: refusing unsafe Buffer teardown for output %p; native "
               "resources and their Tensor owners are intentionally leaked\n",
               output_pointer);
  if (transport != nullptr) transport->unsafe_release = true;
  // Both allocations may still carry a live registration, so neither may be
  // freed here; leaking them is the whole point of this path.
  output_owner.release();
  landing_owner.release();
}

inline Transport* AsTransport(int64_t handle) {
  auto* result = reinterpret_cast<Transport*>(handle);
  TVM_FFI_ICHECK(result != nullptr) << "null PCIe Ulysses handle";
  return result;
}

inline Buffer* FindBufferByPointer(Transport* transport, void* pointer) {
  auto it = transport->buffers.find(pointer);
  TVM_FFI_ICHECK(it != transport->buffers.end())
      << "output is not registered with this PCIe Ulysses transport";
  return it->second.get();
}

inline uint32_t CheckedPayload(int64_t bytes, int world_size) {
  TVM_FFI_ICHECK_EQ(bytes % world_size, 0) << "tensor bytes must be divisible by world_size";
  const int64_t payload = bytes / world_size;
  TVM_FFI_ICHECK_GT(payload, 0) << "per-peer payload must be positive";
  TVM_FFI_ICHECK_LE(payload, UINT32_MAX) << "per-peer payload exceeds mlx5 WR limit";
  return static_cast<uint32_t>(payload);
}

struct MkeyGeometry {
  uint32_t rows = 0;
  uint32_t width = 0;
  uint32_t skip = 0;
  uint64_t pitch = 0;
};

inline MkeyGeometry GetMkeyGeometry(int mode, int64_t batch, int64_t seq, int64_t heads,
                                    int64_t dim, int64_t element_size, int world_size) {
  int64_t rows = 0;
  int64_t width = 0;
  int64_t pitch = 0;
  if (mode == 0) {
    rows = batch * seq;
    width = (heads / world_size) * dim * element_size;
    pitch = heads * dim * element_size;
  } else {
    rows = batch * (seq / world_size);
    width = heads * dim * element_size;
    pitch = heads * world_size * dim * element_size;
  }
  TVM_FFI_ICHECK_LE(rows, UINT32_MAX) << "mlx5 MKey row count exceeds UINT32_MAX";
  TVM_FFI_ICHECK_LE(width, UINT32_MAX) << "mlx5 MKey width exceeds UINT32_MAX";
  TVM_FFI_ICHECK_LE(width, pitch) << "mlx5 MKey width exceeds pitch";
  TVM_FFI_ICHECK_LE(pitch, kMaxInterleavedStride)
      << "mlx5 MKey pitch exceeds the 65535-byte interleaved-stride limit";
  const int64_t skip = pitch - width;
  TVM_FFI_ICHECK_LE(skip, UINT32_MAX) << "mlx5 MKey skip exceeds UINT32_MAX";
  return {static_cast<uint32_t>(rows), static_cast<uint32_t>(width), static_cast<uint32_t>(skip),
          static_cast<uint64_t>(pitch)};
}

inline uint64_t MkeyPeerAddress(const void* pointer, int peer, int64_t bytes,
                                const MkeyGeometry& geometry, const char* value) {
  const uint64_t peer_offset = uint64_t{static_cast<uint32_t>(peer)} * geometry.width;
  const uint64_t end = peer_offset + uint64_t{geometry.rows - 1} * geometry.pitch + geometry.width;
  TVM_FFI_ICHECK_LE(end, static_cast<uint64_t>(bytes))
      << value << " interleaved layout exceeds registered memory";
  return reinterpret_cast<uint64_t>(pointer) + peer_offset;
}

inline void RegisterInitialInput(Transport* transport, Buffer* buffer, int64_t input_bytes,
                                 int64_t capacity_bytes, std::unique_ptr<Tensor> landing_owner);
inline void ConfigureMkeys(Transport* transport, Buffer* buffer, const void* pointer, int64_t bytes,
                           ibv_mr* mr, std::array<mlx5dv_mkey*, kWorld>& mkeys, uint32_t access,
                           const char* what);

inline void RegisterOutput(Transport* transport, const Tensor& input, TensorView output,
                           int64_t mode, std::unique_ptr<Tensor> output_owner,
                           std::unique_ptr<Tensor> landing_owner, int64_t capacity_bytes,
                           int64_t element_size, int64_t input_bytes, int64_t output_bytes) {
  transport->EnsureHealthy();
  TVM_FFI_ICHECK(transport->connected) << "PCIe Ulysses transport is not connected";
  TVM_FFI_ICHECK_EQ(input.ndim(), 4) << "PCIe Ulysses input must be 4-D";
  const int64_t batch = input.size(0);
  const int64_t seq = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t dim = input.size(3);
  TVM_FFI_ICHECK_EQ(input.dtype(), output.dtype()) << "input/output dtype mismatch";
  TVM_FFI_ICHECK_EQ(input.device().device_id, transport->device)
      << "input is on the wrong CUDA device";
  TVM_FFI_ICHECK_EQ(output.device().device_id, transport->device)
      << "output is on the wrong CUDA device";
  TVM_FFI_ICHECK_EQ(input_bytes, output_bytes) << "input/output byte size mismatch";
  CheckNoOverlap(input.data_ptr(), input_bytes, output.data_ptr(), output_bytes);
  TVM_FFI_ICHECK(transport->buffers.find(output.data_ptr()) == transport->buffers.end())
      << "output pointer is already registered with this PCIe Ulysses transport";
  if (capacity_bytes <= 0) capacity_bytes = output_bytes;
  TVM_FFI_ICHECK_GE(capacity_bytes, output_bytes)
      << "registered capacity is smaller than the operand it must hold";

  auto buffer = std::make_unique<Buffer>();
  buffer->transport = transport;
  buffer->mode = static_cast<int>(mode);
  buffer->batch = batch;
  buffer->seq = seq;
  buffer->heads = heads;
  buffer->dim = dim;
  buffer->element_size = element_size;
  buffer->dtype = output.dtype();
  buffer->output_pointer = output.data_ptr();
  buffer->output_bytes = output_bytes;
  buffer->capacity_bytes = capacity_bytes;
  buffer->output_owner = std::move(output_owner);
  buffer->peer_pointers[transport->rank] = buffer->output_pointer;

  // [0, W) holds normal barrier epochs.  [W, 2W) is a sticky abort half where
  // rank r is the only writer of slot r in every peer allocation.
  const size_t signal_bytes = 2 * transport->world_size * sizeof(uint64_t);
  CheckCuda(cudaMalloc(&buffer->signals, signal_bytes), "cudaMalloc(PCIe Ulysses signals)");
  CheckCuda(cudaMemset(buffer->signals, 0, signal_bytes), "cudaMemset(PCIe Ulysses signals)");
  CheckCuda(cudaMalloc(&buffer->epoch_device, sizeof(uint64_t)), "cudaMalloc(PCIe Ulysses epoch)");
  CheckCuda(cudaMemset(buffer->epoch_device, 0, sizeof(uint64_t)),
            "cudaMemset(PCIe Ulysses epoch)");
  buffer->peer_signals[transport->rank] = buffer->signals;

  if (transport->use_rdma) {
    transport->RetireOnFailure([&] {
      // Register the whole capacity once. The interleaved MKeys below address a
      // prefix of it and are the only shape-dependent state; re-binding them is
      // a local UMR post, whereas re-registering would pin pages again.
      buffer->output_mr =
          RegisterGpuMr(transport->pd, buffer->output_pointer, static_cast<size_t>(capacity_bytes),
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
      if (mode == 1) {
        for (int peer = 0; peer < transport->world_size; ++peer) {
          if (!transport->Cross(peer)) continue;
          buffer->destination_mkeys[peer] = transport->CreateMkey();
        }
        ConfigureMkeys(transport, buffer.get(), buffer->output_pointer, buffer->output_bytes,
                       buffer->output_mr, buffer->destination_mkeys,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
                       "gather_heads destination");
      }
    });
    RegisterInitialInput(transport, buffer.get(), input_bytes, capacity_bytes,
                         std::move(landing_owner));
  }

  const bool inserted =
      transport->buffers.emplace(buffer->output_pointer, std::move(buffer)).second;
  TVM_FFI_ICHECK(inserted) << "output pointer was concurrently registered";
}

// Point every cross-NUMA peer's interleaved MKey at the current geometry over
// `pointer`/`bytes`: one local UMR post per peer (see BindGeometry).
inline void ConfigureMkeys(Transport* transport, Buffer* buffer, const void* pointer, int64_t bytes,
                           ibv_mr* mr, std::array<mlx5dv_mkey*, kWorld>& mkeys, uint32_t access,
                           const char* what) {
  if (mr == nullptr) return;
  const MkeyGeometry geometry =
      GetMkeyGeometry(buffer->mode, buffer->batch, buffer->seq, buffer->heads, buffer->dim,
                      buffer->element_size, transport->world_size);
  TVM_FFI_ICHECK_EQ(uint64_t{geometry.rows} * geometry.width,
                    CheckedPayload(bytes, transport->world_size))
      << what << " MKey geometry does not match per-peer payload";
  int configured = 0;
  for (int peer = 0; peer < transport->world_size; ++peer) {
    if (!transport->Cross(peer)) continue;
    TVM_FFI_ICHECK(mkeys[peer] != nullptr) << "missing " << what << " MKey";
    transport->ConfigureMkey(peer, mkeys[peer], access,
                             MkeyPeerAddress(pointer, peer, bytes, geometry, what), geometry.width,
                             geometry.skip, geometry.rows, mr->lkey);
    ++configured;
  }
  transport->Poll(configured);
}

// The NIC reads out of a transport-owned staging buffer, never caller memory:
// SYNC_MEMOPS applies to the whole backing allocation, and that allocation
// must outlive the MR.
inline void RegisterInitialInput(Transport* transport, Buffer* buffer, int64_t input_bytes,
                                 int64_t capacity_bytes, std::unique_ptr<Tensor> landing_owner) {
  try {
    TVM_FFI_ICHECK(landing_owner != nullptr && landing_owner->data_ptr() != nullptr)
        << "the RDMA routes need an input landing allocation";
    buffer->landing_owner = std::move(landing_owner);
    buffer->input_landing = buffer->landing_owner->data_ptr();
    transport->RetireOnFailure([&] {
      buffer->landing_mr =
          RegisterGpuMr(transport->pd, buffer->input_landing, static_cast<size_t>(capacity_bytes),
                        IBV_ACCESS_LOCAL_WRITE);
      if (buffer->mode != 0) return;
      for (int peer = 0; peer < transport->world_size; ++peer) {
        if (!transport->Cross(peer)) continue;
        buffer->landing_source_mkeys[peer] = transport->CreateMkey();
      }
      ConfigureMkeys(transport, buffer, buffer->input_landing, input_bytes, buffer->landing_mr,
                     buffer->landing_source_mkeys, 0, "scatter_heads source");
    });
  } catch (...) {
    transport->Quiesce();
    throw;
  }
}

inline const void* BindInput(Transport* transport, Buffer* buffer, TensorView input,
                             int64_t input_bytes, cudaStream_t current) {
  if (!transport->use_rdma) return input.data_ptr();
  TVM_FFI_ICHECK_LE(input_bytes, buffer->capacity_bytes)
      << "input exceeds the capacity this PCIe Ulysses slot was registered for";
  TVM_FFI_ICHECK(buffer->landing_mr != nullptr && buffer->input_landing != nullptr)
      << "missing PCIe Ulysses input registration";
  // A caller that produced its operand straight into the landing buffer has
  // nothing to stage. The test is exact pointer equality and cannot be relaxed
  // to "somewhere inside the landing region": on the scatter path the NIC reads
  // through the landing MKeys no matter what this returns (local_addresses is
  // never filled in), while the copy engine reads whatever comes back here. A
  // source that merely overlapped would send the NIC's peers stale bytes and
  // the copy engine's fresh ones, with nothing to report it.
  if (input.data_ptr() == buffer->input_landing) return buffer->input_landing;
  const auto* landing_begin = static_cast<const char*>(buffer->input_landing);
  const auto* source_begin = static_cast<const char*>(input.data_ptr());
  TVM_FFI_ICHECK(source_begin + input_bytes <= landing_begin ||
                 source_begin >= landing_begin + buffer->capacity_bytes)
      << "PCIe Ulysses input overlaps this slot's landing buffer without being "
         "it; pass the tensor returned by input_buffer() unmodified, or an "
         "operand allocated elsewhere";
  CheckCuda(cudaMemcpyAsync(buffer->input_landing, input.data_ptr(),
                            static_cast<size_t>(input_bytes), cudaMemcpyDeviceToDevice, current),
            "cudaMemcpyAsync(PCIe Ulysses input staging)");
  return buffer->input_landing;
}

// Re-point a registered slot at a new operand shape (the "plan" step).
//
// Registration is sized once from the declared capacity; only the interleaved
// MKeys depend on the shape. An mlx5 MKey keeps its lkey/rkey across
// reconfiguration, so already-exchanged rkeys stay valid and a rebind is a
// local UMR post per cross-NUMA peer — no collective.
inline void BindGeometry(Transport* transport, Buffer* buffer, int64_t mode, int64_t batch,
                         int64_t seq, int64_t heads, int64_t dim, int64_t element_size,
                         int64_t output_bytes) {
  TVM_FFI_ICHECK_EQ(buffer->mode, mode) << "output was registered for another operation";
  TVM_FFI_ICHECK_LE(output_bytes, buffer->capacity_bytes)
      << "operand exceeds the capacity this PCIe Ulysses slot was registered for";
  if (buffer->batch == batch && buffer->seq == seq && buffer->heads == heads &&
      buffer->dim == dim && buffer->element_size == element_size) {
    return;
  }
  buffer->batch = batch;
  buffer->seq = seq;
  buffer->heads = heads;
  buffer->dim = dim;
  buffer->element_size = element_size;
  buffer->output_bytes = output_bytes;
  if (!transport->use_rdma) return;
  // Runs only inside exchange()'s hybrid failure envelope: the outer catch
  // must publish the group abort before quiescing partially posted UMR work,
  // so do not wrap this in RetireOnFailure.
  if (buffer->mode == 1) {
    ConfigureMkeys(transport, buffer, buffer->output_pointer, output_bytes, buffer->output_mr,
                   buffer->destination_mkeys,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
                   "gather_heads destination");
  } else {
    ConfigureMkeys(transport, buffer, buffer->input_landing, output_bytes, buffer->landing_mr,
                   buffer->landing_source_mkeys, 0, "scatter_heads source");
  }
}

inline void EnqueueCopies(Transport* transport, Buffer* buffer, const void* input, int mode,
                          int64_t batch, int64_t seq, int64_t heads, int64_t dim,
                          int64_t element_size, cudaStream_t current, bool all_peers,
                          bool record_input_ready = true) {
  if (record_input_ready)
    CheckCuda(cudaEventRecord(transport->input_ready, current), "cudaEventRecord(input ready)");
  std::array<bool, kWorld> used{};
  int copy_index = 0;
  // Callers filter cross-NUMA peers; every peer reaching here is copied to
  // directly.
  auto enqueue_peer = [&](int peer) {
    const size_t stream_index = copy_index++ % transport->streams.size();
    auto stream = transport->streams[stream_index];
    if (!used[stream_index]) {
      CheckCuda(cudaStreamWaitEvent(stream, transport->input_ready, 0),
                "cudaStreamWaitEvent(input ready)");
      used[stream_index] = true;
    }
    CheckCuda(EnqueuePeerCopy(input, buffer->peer_pointers[peer], mode, batch, seq, heads, dim,
                              element_size, transport->rank, peer, transport->world_size, stream),
              "cudaMemcpy2DAsync(local Ulysses peer)");
  };
  // XOR-shifted order spreads the first destination across ranks instead of
  // pointing every copy engine at the lowest-numbered peer at once.
  for (int step = 1; step < transport->world_size; ++step) {
    const int peer = transport->rank ^ step;
    if (!all_peers && transport->Cross(peer)) continue;
    enqueue_peer(peer);
  }
  enqueue_peer(transport->rank);
  for (size_t index = 0; index < transport->streams.size(); ++index) {
    if (!used[index]) continue;
    CheckCuda(cudaEventRecord(transport->copy_done[index], transport->streams[index]),
              "cudaEventRecord(copy done)");
    CheckCuda(cudaStreamWaitEvent(current, transport->copy_done[index], 0),
              "cudaStreamWaitEvent(copy done)");
  }
}

}  // namespace ulysses_pcie
}  // namespace comm
}  // namespace flashinfer
