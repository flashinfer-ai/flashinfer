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

#include <tvm/ffi/container/tuple.h>

#include <cstdint>

#include "tvm_ffi_utils.h"
#include "ulysses_pcie_transport.cuh"

using tvm::ffi::Array;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::Tuple;
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace fi = flashinfer::comm::ulysses_pcie;

using fi::Buffer;
using fi::Transport;

namespace {

// Framework boundary: TVM-FFI tensors in, raw pointers/extents/byte counts
// out. The device kernels in include/flashinfer/comm/ulysses_pcie.cuh stay
// framework-agnostic; the host transport speaks TVM-FFI and lives beside this
// translation unit.

struct CudaMallocNDAlloc {
  int device = 0;

  void AllocData(DLTensor* tensor) {
    ffi::CUDADeviceGuard device_guard(static_cast<int>(device));
    const size_t bytes = tvm::ffi::GetDataSize(*tensor);
    fi::CheckCuda(cudaMalloc(&tensor->data, bytes), "cudaMalloc(PCIe Ulysses output)");
  }
  void FreeData(DLTensor* tensor) noexcept {
    fi::ScopedCudaDevice device_guard(device);
    if (device_guard.active() && tensor->data != nullptr) cudaFree(tensor->data);
  }
};
// Allocation identity is verified once at registration (fi::RegisterOutput),
// and the Buffer holds an owning reference, so the address cannot be recycled
// underneath us.
Buffer* FindBuffer(Transport* transport, TensorView output) {
  return fi::FindBufferByPointer(transport, output.data_ptr());
}

void FlushHybridWrites(Transport* transport) {
  if (transport->write_ordering >= cudaGPUDirectRDMAWritesOrderingOwner) return;
  fi::CheckCuda(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                   cudaFlushGPUDirectRDMAWritesToOwner),
                "cudaDeviceFlushGPUDirectRDMAWrites");
}

int64_t ValidateGeometry(TensorView tensor, int64_t mode, int64_t batch, int64_t seq, int64_t heads,
                         int64_t dim, int world_size, bool output, bool use_rdma) {
  CHECK_INPUT(tensor);
  TVM_FFI_ICHECK(mode == 0 || mode == 1) << "mode must be 0 or 1";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << "PCIe Ulysses operands must be 4-D";
  TVM_FFI_ICHECK(batch > 0 && seq > 0 && heads > 0 && dim > 0)
      << "PCIe Ulysses dimensions must be positive";
  if (use_rdma) {
    TVM_FFI_ICHECK_EQ(batch, 1) << "RDMA PCIe Ulysses routes currently support batch=1";
  }
  if (mode == 0) {
    TVM_FFI_ICHECK_EQ(heads % world_size, 0) << "heads must be divisible by world_size";
  } else {
    TVM_FFI_ICHECK_EQ(seq % world_size, 0) << "sequence must be divisible by world_size";
  }
  const int64_t out_seq = mode == 0 ? seq * world_size : seq / world_size;
  const int64_t out_heads = mode == 0 ? heads / world_size : heads * world_size;
  const int64_t sizes[4] = {
      batch,
      output ? out_seq : seq,
      output ? out_heads : heads,
      dim,
  };
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_ICHECK_EQ(tensor.size(axis), sizes[axis]) << "unexpected operand shape";
  }
  // The transport moves bytes (copy engines and RDMA writes); any element
  // size the pitch math and MKey stride limit can express is acceptable.
  const int64_t element_size = get_element_size(tensor);
  TVM_FFI_ICHECK(element_size == 1 || element_size == 2 || element_size == 4)
      << "PCIe Ulysses supports 1-, 2- and 4-byte element types, got " << element_size << " bytes";
  return tensor.numel() * element_size;
}

}  // namespace

// Returns (handle, rank-local connection metadata to all_gather and feed back
// through connect).
Tuple<int64_t, Array<int64_t>> init_ulysses_pcie(int64_t rank, int64_t world_size, int64_t device,
                                                 Array<int64_t> numa_nodes, String nic_name,
                                                 int64_t use_rdma, int64_t gid_index) {
  TVM_FFI_ICHECK(use_rdma == 0 || use_rdma == 1) << "use_rdma must be 0 or 1";
  TVM_FFI_ICHECK_GE(gid_index, -1) << "gid_index must be -1 or non-negative";
  auto* transport =
      new Transport(static_cast<int>(rank), static_cast<int>(world_size), static_cast<int>(device),
                    numa_nodes, std::string(nic_name), use_rdma != 0, static_cast<int>(gid_index));
  return Tuple(reinterpret_cast<fptr_t>(transport), fi::Encode(transport->local));
}

void connect_ulysses_pcie(fptr_t handle, Array<int64_t> flat_metadata) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  transport->Connect(flat_metadata);
}

Tuple<Tensor, Array<int64_t>> allocate_ulysses_pcie_output(fptr_t handle, Tensor input,
                                                           int64_t mode,
                                                           int64_t capacity_elements) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  transport->EnsureHealthy();
  TVM_FFI_ICHECK_EQ(input.ndim(), 4) << "PCIe Ulysses input must be 4-D";
  const int64_t batch = input.size(0);
  const int64_t seq = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t dim = input.size(3);
  const int64_t input_bytes = ValidateGeometry(input, mode, batch, seq, heads, dim,
                                               transport->world_size, false, transport->use_rdma);
  const int64_t out_seq = mode == 0 ? seq * transport->world_size : seq / transport->world_size;
  const int64_t out_heads =
      mode == 0 ? heads / transport->world_size : heads * transport->world_size;
  const int64_t element_size = get_element_size(input);
  const int64_t elements = batch * out_seq * out_heads * dim;
  TVM_FFI_ICHECK_GE(capacity_elements, elements)
      << "declared capacity is smaller than the operand this call would produce";
  const int64_t capacity_bytes = capacity_elements * element_size;

  Tensor storage =
      Tensor::FromNDAlloc(CudaMallocNDAlloc{transport->device},
                          tvm::ffi::Shape({capacity_elements}), input.dtype(), input.device());
  // Describe the initial geometry over that storage. Null strides mean compact
  // row-major, which is what every operand on this path is.
  int64_t view_shape[4] = {batch, out_seq, out_heads, dim};
  DLTensor view = *storage.GetDLTensorPtr();
  view.ndim = 4;
  view.shape = view_shape;
  view.strides = nullptr;

  auto storage_owner = std::make_unique<Tensor>(storage);
  // The hybrid route also needs a landing buffer this transport owns, because
  // the NIC never reads out of caller memory. Allocated here, beside the output,
  // so both are ordinary tensors the caller can be handed.
  std::unique_ptr<Tensor> landing_owner;
  if (transport->use_rdma) {
    landing_owner = std::make_unique<Tensor>(
        Tensor::FromNDAlloc(CudaMallocNDAlloc{transport->device},
                            tvm::ffi::Shape({capacity_elements}), input.dtype(), input.device()));
  }
  fi::RegisterOutput(transport, input, TensorView(&view), mode, std::move(storage_owner),
                     std::move(landing_owner), capacity_bytes, element_size, input_bytes,
                     batch * out_seq * out_heads * dim * element_size);

  // Rank-local registration record to all_gather and feed back through
  // connect_output.
  auto* buffer = fi::FindBufferByPointer(transport, storage.data_ptr());
  fi::BufferWire wire{};
  wire.address = reinterpret_cast<uint64_t>(buffer->output_pointer);
  if (buffer->output_mr != nullptr) wire.rkey = buffer->output_mr->rkey;
  fi::CheckCuda(cudaIpcGetMemHandle(&wire.ipc, buffer->output_pointer),
                "cudaIpcGetMemHandle(output)");
  fi::CheckCuda(cudaIpcGetMemHandle(&wire.signal_ipc, buffer->signals),
                "cudaIpcGetMemHandle(signals)");
  for (int peer = 0; peer < transport->world_size; ++peer)
    if (buffer->destination_mkeys[peer] != nullptr)
      wire.destination_rkey[peer] = buffer->destination_mkeys[peer]->rkey;
  return Tuple(storage, fi::Encode(wire));
}

void connect_ulysses_pcie_output(fptr_t handle, TensorView output, Array<int64_t> flat_metadata) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  transport->EnsureHealthy();
  auto* buffer = FindBuffer(transport, output);
  TVM_FFI_ICHECK(!buffer->disconnect_required)
      << "a disconnected PCIe Ulysses output cannot be reconnected";
  TVM_FFI_ICHECK(!buffer->connected) << "PCIe Ulysses output is already connected";
  TVM_FFI_ICHECK_EQ(flat_metadata.size(),
                    static_cast<size_t>(transport->world_size * sizeof(fi::BufferWire)))
      << "invalid output metadata length";
  std::array<fi::BufferWire, fi::kWorld> peers{};
  std::array<void*, fi::kWorld> payloads{};
  std::array<uint64_t*, fi::kWorld> signals{};
  for (int peer = 0; peer < transport->world_size; ++peer)
    peers[peer] = fi::DecodeAt<fi::BufferWire>(flat_metadata, peer * sizeof(fi::BufferWire));
  payloads[transport->rank] = buffer->output_pointer;
  signals[transport->rank] = buffer->signals;
  try {
    for (int peer = 0; peer < transport->world_size; ++peer) {
      if (peer == transport->rank) continue;
      if (!(transport->use_rdma && transport->Cross(peer))) {
        fi::CheckCuda(
            cudaIpcOpenMemHandle(&payloads[peer], peers[peer].ipc, cudaIpcMemLazyEnablePeerAccess),
            "cudaIpcOpenMemHandle(output)");
      }
      // Opening and closing epochs are full-group GPU barriers even when the
      // payload crosses NUMA domains through RDMA, so every rank pair needs
      // CUDA peer access regardless of which route carries the payload.
      fi::CheckCuda(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&signals[peer]),
                                         peers[peer].signal_ipc, cudaIpcMemLazyEnablePeerAccess),
                    "cudaIpcOpenMemHandle(signals)");
    }
  } catch (...) {
    const std::exception_ptr original = std::current_exception();
    // Make every successfully opened handle reachable by the idempotent
    // disconnect path before attempting rollback. If one close itself fails,
    // the Buffer remains registered and a later disconnect can retry it —
    // and the original exception, not the close failure, must propagate.
    buffer->peer_pointers = payloads;
    buffer->peer_signals = signals;
    buffer->disconnect_required = true;
    buffer->imports_closed = false;
    try {
      buffer->Disconnect();
    } catch (...) {
    }
    std::rethrow_exception(original);
  }
  buffer->peers = peers;
  buffer->peer_pointers = payloads;
  buffer->peer_signals = signals;
  buffer->connected = true;
  buffer->disconnect_required = true;
  buffer->imports_closed = false;
}

// The transport-owned input staging buffer behind one registered output.
//
// Handed out flat at capacity, like the output storage itself, so the caller
// views it per call. It stays alive as long as the output is registered: the
// buffer owns it through a refcounted Tensor, and this only adds a reference.
Tensor ulysses_pcie_input_landing(fptr_t handle, TensorView output) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  transport->EnsureHealthy();
  TVM_FFI_ICHECK(transport->use_rdma)
      << "only the RDMA PCIe Ulysses routes stage input through a landing buffer; "
         "the all-P2P route reads the caller's operand in place already";
  auto* buffer = FindBuffer(transport, output);
  TVM_FFI_ICHECK(buffer->landing_owner != nullptr && buffer->input_landing != nullptr)
      << "this PCIe Ulysses output has no registered input landing buffer";
  return *buffer->landing_owner;
}

void ulysses_pcie_exchange(fptr_t handle, TensorView input, TensorView output, int64_t mode,
                           int64_t batch, int64_t seq, int64_t heads, int64_t dim) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  transport->EnsureHealthy();
  auto* buffer = FindBuffer(transport, output);
  TVM_FFI_ICHECK(buffer->connected) << "PCIe Ulysses output is not connected";
  const int64_t input_bytes = ValidateGeometry(input, mode, batch, seq, heads, dim,
                                               transport->world_size, false, transport->use_rdma);
  const int64_t output_bytes = ValidateGeometry(output, mode, batch, seq, heads, dim,
                                                transport->world_size, true, transport->use_rdma);
  TVM_FFI_ICHECK_EQ(input.dtype(), output.dtype()) << "input/output dtype mismatch";
  TVM_FFI_ICHECK_EQ(input.dtype(), buffer->dtype)
      << "input dtype differs from the registered PCIe Ulysses geometry";
  TVM_FFI_ICHECK_EQ(output.dtype(), buffer->dtype)
      << "output dtype differs from the registered PCIe Ulysses geometry";
  TVM_FFI_ICHECK_EQ(input.device().device_id, transport->device)
      << "input is on the wrong CUDA device";
  TVM_FFI_ICHECK_EQ(output.device().device_id, transport->device)
      << "output is on the wrong CUDA device";
  TVM_FFI_ICHECK_EQ(input_bytes, output_bytes) << "input/output byte size mismatch";
  fi::CheckNoOverlap(input.data_ptr(), input_bytes, output.data_ptr(), output_bytes);
  const auto current = get_stream(input.device());

  if (!transport->use_rdma) {
    fi::BindGeometry(transport, buffer, mode, batch, seq, heads, dim, get_element_size(input),
                     output_bytes);
    const void* source = fi::BindInput(transport, buffer, input, input_bytes, current);
    try {
      fi::CheckCuda(
          fi::EnqueueBarrier(buffer->signals, buffer->peer_signals.data(), transport->world_size,
                             transport->rank, buffer->epoch_device, current),
          "enqueue PCIe Ulysses opening barrier");
      fi::EnqueueCopies(transport, buffer, source, mode, batch, seq, heads, dim,
                        get_element_size(input), current, true);
      fi::CheckCuda(
          fi::EnqueueBarrier(buffer->signals, buffer->peer_signals.data(), transport->world_size,
                             transport->rank, buffer->epoch_device, current),
          "enqueue PCIe Ulysses closing barrier");
    } catch (...) {
      transport->failed = true;
      // The all-P2P barrier has no abort protocol: once an opening barrier may
      // have been enqueued, a failed rank can leave peers spinning. The Python
      // safety vote must keep every rank out of cudaDeviceSynchronize.
      transport->teardown_safe = false;
      throw;
    }
    return;
  }

  try {
    // Keep every shape rebind and staging operation inside the hybrid failure
    // envelope. A local UMR/CUDA setup failure can then publish the sticky abort
    // immediately instead of making peers consume the full barrier timeout.
    fi::BindGeometry(transport, buffer, mode, batch, seq, heads, dim, get_element_size(input),
                     output_bytes);
    const void* source = fi::BindInput(transport, buffer, input, input_bytes, current);

    const uint32_t payload = fi::CheckedPayload(input_bytes, transport->world_size);
    const uint64_t payload64 = payload;
    const fi::MkeyGeometry geometry =
        fi::GetMkeyGeometry(static_cast<int>(mode), batch, seq, heads, dim, get_element_size(input),
                            transport->world_size);
    TVM_FFI_ICHECK_EQ(uint64_t{geometry.rows} * geometry.width, payload64)
        << "RDMA MKey geometry does not match per-peer payload";

    std::array<uint32_t, fi::kWorld> local_keys{};
    std::array<uint64_t, fi::kWorld> local_addresses{};
    std::array<uint32_t, fi::kWorld> remote_keys{};
    std::array<uint64_t, fi::kWorld> remote_addresses{};
    // The NIC always reads out of this transport's own staging buffer, so there is
    // one registration to name here rather than a choice between two.
    const auto& active_source_mkeys = buffer->landing_source_mkeys;
    auto* active_input_mr = buffer->landing_mr;
    for (int peer = 0; peer < transport->world_size; ++peer) {
      if (!transport->Cross(peer)) continue;
      TVM_FFI_ICHECK(transport->qps[peer] != nullptr && transport->qpxs[peer] != nullptr)
          << "missing mlx5 QP for peer " << peer;
      if (mode == 0) {
        TVM_FFI_ICHECK(active_source_mkeys[peer] != nullptr)
            << "missing scatter_heads source MKey for peer " << peer;
        local_keys[peer] = active_source_mkeys[peer]->lkey;
        TVM_FFI_ICHECK(buffer->peers[peer].address != 0 && buffer->peers[peer].rkey != 0)
            << "missing scatter_heads output registration for peer " << peer;
        const uint64_t remote_offset = uint64_t(transport->rank) * payload64;
        TVM_FFI_ICHECK_LE(remote_offset + payload64, static_cast<uint64_t>(buffer->output_bytes))
            << "scatter_heads remote payload exceeds output buffer";
        remote_addresses[peer] = buffer->peers[peer].address + remote_offset;
        remote_keys[peer] = buffer->peers[peer].rkey;
      } else {
        TVM_FFI_ICHECK(active_input_mr != nullptr) << "missing RDMA input registration";
        local_keys[peer] = active_input_mr->lkey;
        const uint64_t local_offset = uint64_t(peer) * payload64;
        TVM_FFI_ICHECK_LE(local_offset + payload64, static_cast<uint64_t>(input_bytes))
            << "gather_heads local payload exceeds input buffer";
        local_addresses[peer] = reinterpret_cast<uint64_t>(source) + local_offset;
        remote_keys[peer] = buffer->peers[peer].destination_rkey[transport->rank];
        TVM_FFI_ICHECK_NE(remote_keys[peer], 0)
            << "missing gather_heads destination MKey for peer " << peer;
      }
    }

    // Only the RDMA work-request tag needs a host-visible sequence number; the
    // barriers advance their own epoch in device memory.
    const uint64_t exchange_tag = ++buffer->epoch;
    const uint32_t payload_tag = static_cast<uint32_t>(exchange_tag & 0x3fffffffU) | 0x80000000U;
    // A successful bounded opening proves both that every rank has retired its
    // previous output and that no sticky abort was published before this phase.
    transport->RunHybridBarrier(buffer, current, true);

    const int expected = transport->CrossCount();
    for (int peer = 0; peer < transport->world_size; ++peer)
      if (transport->Cross(peer)) transport->PostReceive(peer);
    for (int peer = 0; peer < transport->world_size; ++peer) {
      if (!transport->Cross(peer)) continue;
      transport->PostWrite(peer, local_keys[peer], local_addresses[peer], payload,
                           remote_keys[peer], remote_addresses[peer], payload_tag);
    }
    fi::EnqueueCopies(transport, buffer, source, mode, batch, seq, heads, dim,
                      get_element_size(input), current, false, false);

    // Closing is a success vote, so it must not be published until every send
    // and receive CQE is verified and GPUDirect writes are visible to the GPU.
    transport->Poll(expected, expected, payload_tag);
    FlushHybridWrites(transport);
    transport->RunHybridBarrier(buffer, current, false);
  } catch (...) {
    transport->AbortAndQuiesce(buffer, current);
    throw;
  }
}

int64_t ulysses_pcie_teardown_safe(fptr_t handle) {
  auto* transport = fi::AsTransport(handle);
  return transport->TeardownSafe() ? 1 : 0;
}

void DisconnectOutput(Buffer* buffer) {
  if (buffer->imports_closed) return;
  TVM_FFI_ICHECK(buffer->transport->TeardownSafe())
      << "PCIe Ulysses cannot disconnect after unbounded native GPU work; terminate the process";
  // Python's staged close synchronizes the device (group-wide) before this
  // stage; do not reorder those stages.
  buffer->Disconnect();
}

void disconnect_ulysses_pcie_output_ptr(fptr_t handle, int64_t pointer) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  void* address = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
  DisconnectOutput(fi::FindBufferByPointer(transport, address));
}

void dispose_ulysses_pcie_output_ptr(fptr_t handle, int64_t pointer) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  TVM_FFI_ICHECK(transport->TeardownSafe())
      << "PCIe Ulysses output cannot be disposed after unbounded native GPU work";
  void* address = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
  auto* buffer = fi::FindBufferByPointer(transport, address);
  TVM_FFI_ICHECK(buffer->imports_closed)
      << "disconnect_ulysses_pcie_output_ptr must complete on every rank before output disposal";
  buffer->Release();
  const size_t removed = transport->buffers.erase(address);
  TVM_FFI_ICHECK_EQ(removed, 1) << "PCIe Ulysses output pointer is not registered";
}

void dispose_ulysses_pcie(fptr_t handle) {
  auto* transport = fi::AsTransport(handle);
  ffi::CUDADeviceGuard device_guard(transport->device);
  TVM_FFI_ICHECK(transport->TeardownSafe())
      << "PCIe Ulysses transport cannot be disposed after unbounded native GPU work";
  TVM_FFI_ICHECK(!transport->unsafe_release)
      << "PCIe Ulysses transport has an unrecoverable native teardown ledger";
  for (const auto& [pointer, buffer] : transport->buffers) {
    TVM_FFI_ICHECK(buffer->imports_closed)
        << "disconnect all PCIe Ulysses outputs on every rank before transport disposal; output "
        << pointer << " still has peer imports";
  }
  for (auto it = transport->buffers.begin(); it != transport->buffers.end();) {
    it->second->Release();
    it = transport->buffers.erase(it);
  }
  TVM_FFI_ICHECK(!transport->unsafe_release)
      << "PCIe Ulysses transport teardown lost a native resource ledger";
  delete transport;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_ulysses_pcie, init_ulysses_pcie);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(connect_ulysses_pcie, connect_ulysses_pcie);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allocate_ulysses_pcie_output, allocate_ulysses_pcie_output);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(connect_ulysses_pcie_output, connect_ulysses_pcie_output);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_pcie_input_landing, ulysses_pcie_input_landing);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_pcie_exchange, ulysses_pcie_exchange);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_pcie_teardown_safe, ulysses_pcie_teardown_safe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(disconnect_ulysses_pcie_output_ptr,
                              disconnect_ulysses_pcie_output_ptr);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispose_ulysses_pcie_output_ptr, dispose_ulysses_pcie_output_ptr);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispose_ulysses_pcie, dispose_ulysses_pcie);
