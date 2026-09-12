/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_CE_MEMOP_CUH_
#define FLASHINFER_COMM_PCIE_IPC_CE_MEMOP_CUH_
#include <cuda.h>

#include <type_traits>

#include "pcie_ipc_ce_ring.cuh"
#include "pcie_ipc_ce_sm120.cuh"

#define FI_CE_CHECK(expr)               \
  do {                                  \
    const cudaError_t err = (expr);     \
    if (err != cudaSuccess) return err; \
  } while (0)
namespace flashinfer {
namespace comm {
namespace pcie_ipc {
// Variant 6 retains the flat ring's reduction order and scratch addresses.
// Producer writes use the default stream-scoped system fence. The consumer
// acquires the binary flag on an SM and clears it before consuming the payload.
// The original end handshake follows all consumption and protects slot reuse.
// No CUDA stream waits on a memory value: local dependencies remain CUDA events.
inline cudaError_t ce_launch_publish(const CeResources& ce, cudaStream_t stream, int32_t* flag,
                                     int32_t* counter) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1200
  if (ce.memop_enabled) {
    kernel_pcie_ipc_ce_sm120_publish<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(flag),
                                                          reinterpret_cast<uint32_t*>(counter));
  } else
#endif
  {
    ce_publish_flag_kernel<<<1, 1, 0, stream>>>(flag, counter);
  }
  return cudaGetLastError();
}

inline cudaError_t ce_launch_wait(const CeResources& ce, cudaStream_t stream, int32_t* flag,
                                  int32_t* counter) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1200
  if (ce.memop_enabled) {
    kernel_pcie_ipc_ce_sm120_wait<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(flag),
                                                       reinterpret_cast<uint32_t*>(counter));
  } else
#endif
  {
    ce_wait_flag_kernel<<<1, 1, 0, stream>>>(flag, counter);
  }
  return cudaGetLastError();
}

template <typename T>
inline cudaError_t ce_launch_add(const CeResources& ce, unsigned grid, int threads,
                                 cudaStream_t stream, uint4* out, const uint4* a, const uint4* b,
                                 int64_t packs) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1200
  if (ce.memop_enabled) {
    if constexpr (std::is_same<T, nv_bfloat16>::value) {
      kernel_pcie_ipc_ce_sm120_add_bf16<<<grid, threads, 0, stream>>>(
          reinterpret_cast<uint32_t*>(out), reinterpret_cast<const uint32_t*>(a),
          reinterpret_cast<const uint32_t*>(b), packs);
    } else {
      kernel_pcie_ipc_ce_sm120_add_f16<<<grid, threads, 0, stream>>>(
          reinterpret_cast<uint32_t*>(out), reinterpret_cast<const uint32_t*>(a),
          reinterpret_cast<const uint32_t*>(b), packs);
    }
  } else
#endif
  {
    ce_add_kernel<T><<<grid, threads, 0, stream>>>(out, a, b, packs);
  }
  return cudaGetLastError();
}

// Per-rank allocation: 2*(world_size-1) cache-line-separated flags, zeroed once.
inline size_t ce_binary_flag_bytes(int world_size) {
  return static_cast<size_t>(2 * (world_size - 1)) * 128u;
}
inline cudaError_t ce_binary_write(cudaStream_t stream, int32_t* flag, uint32_t value) {
  CUresult rc = cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                                     reinterpret_cast<CUdeviceptr>(flag), value, 0);
  return rc == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}
inline cudaError_t ce_binary_wait_clear(const CeResources& ce, cudaStream_t stream, int32_t* flag) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1200
  kernel_pcie_ipc_ce_sm120_binary_wait_clear<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(flag));
  return cudaGetLastError();
#else
  return cudaErrorInvalidDeviceFunction;
#endif
}

template <typename T>
cudaError_t ce_ring_all_reduce_memop(const T* input, T* output, int64_t numel,
                                     const PeerViews& views, int rank, int world_size,
                                     const WorkspaceLayout& layout, const CeResources& ce,
                                     int pieces_hint, int threads, cudaStream_t stream) {
  constexpr int kPackElems = PackTraits<T>::kPackElems;
  const int64_t shard_elems = numel / world_size;
  const size_t shard_bytes = static_cast<size_t>(shard_elems) * sizeof(T);
  const int pieces = ce_pick_pieces(shard_elems, shard_bytes, kPackElems, pieces_hint);
  if (!ce.memop_enabled || pieces != 1) {
    return ce_ring_all_reduce_flat(input, output, numel, views, rank, world_size, layout, ce,
                                   pieces_hint, threads, stream);
  }
  for (int peer = 0; peer < world_size; ++peer) {
    if (ce.binary_flags[peer] == nullptr) return cudaErrorInvalidValue;
  }
  const int64_t piece_elems = shard_elems / pieces;
  const size_t piece_bytes = static_cast<size_t>(piece_elems) * sizeof(T);
  const size_t slot_stride = detail::ce_slot_stride(layout, world_size);
  const int steps = 2 * (world_size - 1);
  const int next = (rank + 1) % world_size;
  const int prev = (rank - 1 + world_size) % world_size;

  const int64_t packs = piece_elems / kPackElems;
  const unsigned grid =
      static_cast<unsigned>(packs < threads ? 1 : (packs + threads - 1) / threads);
  const unsigned add_grid = grid > 64u ? 64u : grid;

  auto flag_at = [&](int peer, int slot) {
    return reinterpret_cast<int32_t*>(views.ce_flags[peer] +
                                      static_cast<uint64_t>(slot) * kCeFlagStride);
  };
  auto scratch_at = [&](int peer, int k, int p) {
    return reinterpret_cast<char*>(views.ce_scratch[peer]) +
           static_cast<ptrdiff_t>(k) * static_cast<ptrdiff_t>(slot_stride) +
           static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(piece_bytes);
  };

  // Copy/publish overlap the caller's incoming wait/add or local copy.
  // The only waits on this side stream are CUDA-visible add-done events.
  FI_CE_CHECK(cudaEventRecord(ce.input_ready, stream));
  FI_CE_CHECK(cudaStreamWaitEvent(ce.copy_stream, ce.input_ready));
  auto binary_flag_at = [&](int peer, int slot) {
    return reinterpret_cast<int32_t*>(reinterpret_cast<char*>(ce.binary_flags[peer]) +
                                      static_cast<size_t>(slot) * 128u);
  };
  for (int k = 0; k < steps; ++k) {
    const bool reduce_phase = k < world_size - 1;
    int send_c, recv_c;
    if (reduce_phase) {
      send_c = ((rank - k) % world_size + world_size) % world_size;
      recv_c = ((rank - k - 1) % world_size + world_size) % world_size;
    } else {
      const int kk = k - (world_size - 1);
      send_c = ((rank + 1 - kk) % world_size + world_size) % world_size;
      recv_c = ((rank - kk) % world_size + world_size) % world_size;
    }
    for (int p = 0; p < pieces; ++p) {
      const int slot = k * pieces + p;
      const int64_t off = static_cast<int64_t>(send_c) * shard_elems + p * piece_elems;
      const void* src =
          k == 0 ? static_cast<const void*>(input + off) : static_cast<const void*>(output + off);

      if (k > 0) {
        FI_CE_CHECK(cudaStreamWaitEvent(ce.copy_stream, ce.add_done[p]));
      }
      FI_CE_CHECK(cudaMemcpyAsync(scratch_at(next, k, p), src, piece_bytes,
                                  cudaMemcpyDeviceToDevice, ce.copy_stream));
      // Default flags fence all preceding writes in the producer stream.
      FI_CE_CHECK(ce_binary_write(ce.copy_stream, binary_flag_at(next, slot), 1));
      FI_CE_CHECK(ce_binary_wait_clear(ce, stream, binary_flag_at(rank, slot)));

      const int64_t roff = static_cast<int64_t>(recv_c) * shard_elems + p * piece_elems;
      auto* landed = scratch_at(rank, k, p);
      if (reduce_phase) {
        FI_CE_CHECK(ce_launch_add<T>(ce, add_grid, threads, stream,
                                     reinterpret_cast<uint4*>(output + roff),
                                     reinterpret_cast<const uint4*>(input + roff),
                                     reinterpret_cast<const uint4*>(landed), packs));
      } else {
        FI_CE_CHECK(
            cudaMemcpyAsync(output + roff, landed, piece_bytes, cudaMemcpyDeviceToDevice, stream));
      }
      FI_CE_CHECK(cudaEventRecord(ce.add_done[p], stream));
    }
  }

  FI_CE_CHECK(cudaEventRecord(ce.copy_done, ce.copy_stream));
  FI_CE_CHECK(cudaStreamWaitEvent(stream, ce.copy_done));

  // The wait leaf clears before consumption; this original end handshake
  // follows both clear and consumption, and protects next-call slot reuse.
  const int hs = ce_handshake_slot(world_size);
  FI_CE_CHECK(ce_launch_publish(ce, stream, flag_at(prev, hs), views.ce_send_counters + hs));
  FI_CE_CHECK(ce_launch_wait(ce, stream, flag_at(rank, hs), views.ce_wait_counters + hs));
  return cudaGetLastError();
}

}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer
#undef FI_CE_CHECK
#endif
