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

#ifndef FLASHINFER_COMM_ULYSSES_PCIE_CUH_
#define FLASHINFER_COMM_ULYSSES_PCIE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

// Device half of the experimental PCIe Ulysses backend: the epoch barrier
// kernels and the raw-pointer enqueue helpers.  The host verbs/mlx5 transport
// that drives them speaks TVM-FFI types and therefore lives with its binding
// in csrc/ulysses_pcie_transport.cuh, per the include//csrc split in CLAUDE.md.

namespace flashinfer {
namespace comm {
namespace ulysses_pcie {

struct PeerSignalPointers {
  uint64_t* values[8];
};

constexpr uint64_t kAbortSignal = ~uint64_t{0};

__device__ __forceinline__ void PublishSignal(uint64_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t AcquireSignal(const uint64_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

// Advance the epoch in device memory rather than taking it as a launch
// argument: a launch argument is baked into a CUDA graph, so a replay would
// re-publish an already-passed epoch and the barrier would not synchronize.
//
// Every lane must reach the shuffle, so this runs before the barrier body's
// peer-range early return.
__device__ __forceinline__ uint64_t AdvanceEpoch(uint64_t* counter) {
  uint64_t epoch = 0;
  if (threadIdx.x == 0) {
    epoch = atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL) + 1ULL;
  }
  return __shfl_sync(0xffffffffu, epoch, 0);
}

// Each warp lane owns one peer: publish this rank's epoch into the peer's
// slot, then spin until the peer's epoch lands locally. The sticky abort half
// lets any lane terminate the barrier via __any_sync; on the all-P2P route it
// is never written, so the body degenerates to a plain epoch barrier with no
// recovery protocol. All 32 lanes stay active for the sync intrinsics.
__device__ __forceinline__ void UlyssesPcieBarrierBody(uint64_t* local, PeerSignalPointers peers,
                                                       int world_size, int rank, uint64_t epoch) {
  const int peer = threadIdx.x;
  bool aborted = peer < world_size && AcquireSignal(local + world_size + peer) == kAbortSignal;
  if (__any_sync(0xffffffffu, aborted)) return;

  if (peer < world_size) PublishSignal(peers.values[peer] + rank, epoch);
  while (true) {
    aborted = peer < world_size && AcquireSignal(local + world_size + peer) == kAbortSignal;
    if (__any_sync(0xffffffffu, aborted)) return;
    const bool ready = peer >= world_size || AcquireSignal(local + peer) >= epoch;
    if (__all_sync(0xffffffffu, ready)) return;
  }
}

// `static`: a non-template __global__ in a header would otherwise have
// external linkage and collide across translation units.
static __global__ void UlyssesPcieBarrier(uint64_t* local, PeerSignalPointers peers, int world_size,
                                          int rank, uint64_t* counter) {
  UlyssesPcieBarrierBody(local, peers, world_size, rank, AdvanceEpoch(counter));
}

// Every rank owns one slot in the abort half of every peer's allocation.  A
// sticky store avoids both remote atomics and a race with the normal epoch half.
static __global__ void UlyssesPciePublishAbort(PeerSignalPointers peers, int world_size, int rank) {
  const int peer = threadIdx.x;
  if (peer < world_size) PublishSignal(peers.values[peer] + world_size + rank, kAbortSignal);
}

inline cudaError_t FillPeerSignals(PeerSignalPointers& peers, uint64_t* const* peer_signals,
                                   int world_size) {
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer_signals[peer] == nullptr) return cudaErrorInvalidDevicePointer;
    peers.values[peer] = peer_signals[peer];
  }
  return cudaSuccess;
}

// The barrier is always full-group, including on the hybrid route: an exchange
// is only safe to start once every peer has finished consuming the previous
// output, and only safe to finish once every peer has written this one, whether
// those bytes arrived over a peer copy or over the NIC.
inline cudaError_t EnqueueBarrier(uint64_t* local, uint64_t* const* peer_signals, int world_size,
                                  int rank, uint64_t* counter, cudaStream_t stream) {
  PeerSignalPointers peers{};
  if (const auto status = FillPeerSignals(peers, peer_signals, world_size); status != cudaSuccess)
    return status;
  UlyssesPcieBarrier<<<1, 32, 0, stream>>>(local, peers, world_size, rank, counter);
  return cudaGetLastError();
}

inline cudaError_t EnqueueAbort(uint64_t* const* peer_signals, int world_size, int rank,
                                cudaStream_t stream) {
  PeerSignalPointers peers{};
  if (const auto status = FillPeerSignals(peers, peer_signals, world_size); status != cudaSuccess)
    return status;
  UlyssesPciePublishAbort<<<1, 32, 0, stream>>>(peers, world_size, rank);
  return cudaGetLastError();
}

// Enqueue the portion of a Ulysses layout transform sent from this rank to
// one same-NUMA peer. Cross-NUMA peers use mlx5 UMRs in the host transport.
inline cudaError_t EnqueuePeerCopy(const void* input, void* peer_output, int mode, int64_t batch,
                                   int64_t seq, int64_t heads, int64_t dim, int64_t element_size,
                                   int rank, int peer, int world_size, cudaStream_t stream) {
  const auto* source = static_cast<const uint8_t*>(input);
  auto* destination = static_cast<uint8_t*>(peer_output);
  const int64_t width = (mode == 0 ? heads / world_size : heads) * dim * element_size;

  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    int64_t src_offset, dst_offset, src_pitch, dst_pitch, rows;
    if (mode == 0) {
      const int64_t local_heads = heads / world_size;
      src_offset = (batch_index * seq * heads + peer * local_heads) * dim * element_size;
      dst_offset = (batch_index * seq * world_size + rank * seq) * local_heads * dim * element_size;
      src_pitch = heads * dim * element_size;
      dst_pitch = local_heads * dim * element_size;
      rows = seq;
    } else {
      const int64_t local_seq = seq / world_size;
      const int64_t global_heads = heads * world_size;
      src_offset = (batch_index * seq + peer * local_seq) * heads * dim * element_size;
      dst_offset = (batch_index * local_seq * global_heads + rank * heads) * dim * element_size;
      src_pitch = heads * dim * element_size;
      dst_pitch = global_heads * dim * element_size;
      rows = local_seq;
    }
    const cudaError_t status =
        cudaMemcpy2DAsync(destination + dst_offset, dst_pitch, source + src_offset, src_pitch,
                          width, rows, cudaMemcpyDefault, stream);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

}  // namespace ulysses_pcie
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_ULYSSES_PCIE_CUH_
