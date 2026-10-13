/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

#include <nccl_device.h>

#include <cstdint>

namespace deep_ep {

// Every signal ID has one deterministic context: local expert modulo context
// count in the legacy path, or the explicit channel in the coalesced path.
// Epochs live on the device so captured CUDA graphs can replay without resetting
// a signal which a faster peer may already be incrementing.
struct GinLaunch {
  ncclDevComm comm;
  ncclWindow_t window;
  void* base;
  uint64_t* epochs;
  int signal_base;
  bool coalesced = false;
  bool bf16_local_partial = false;
  bool bf16_local_partial_planned = false;
  // Typed NVFP4 legacy buffers pass the exact GDAKI completion admission.
  // Untyped and coalesced buffers retain their existing transport protocol.
  bool nvfp4_legacy_rank_dedup = false;
  // Isolated typed NVFP4 EP4/8 legacy packet protocol; default paths stay unchanged.
  bool nvfp4_rank_packet = false;
  const float* planned_input_weights = nullptr;
  int64_t* planned_rows = nullptr;
  float* planned_owner_weights = nullptr;
  int* planned_tokens = nullptr;
  int* planned_counts = nullptr;
  int* planned_source_ids = nullptr;
  float* planned_source_weights = nullptr;
  int* planned_source_slots = nullptr;
  void* partial_meta_send = nullptr;
  void* partial_meta_recv = nullptr;
  int* partial_lookup = nullptr;
  size_t partial_meta_stride = 0;
  int partial_signal_base = 0;
  int num_ctas = 0;
  // A fixed per-operation subset of the reserved contexts. Never mutate
  // comm.ginContextCount: NCCL owns that resource configuration.
  int active_context_count = 0;
  // Registered peer vectors: L expert counts normally; planned rank fanout
  // uses L+1 entries (unique packet count, then all L expert counts).
  int* send_counts = nullptr;  // Indexed by destination rank.
  int* recv_counts = nullptr;  // Indexed by source rank.
  // Planned rank fanout only: one local mutable cursor per (source, expert),
  // owned by the current ping-pong slot and initialized after count publication.
  int* planned_fanout_cursors = nullptr;
};

#ifdef __CUDACC__
__device__ __forceinline__ size_t gin_offset(const GinLaunch& launch, const void* ptr) {
  return reinterpret_cast<const char*>(ptr) - static_cast<const char*>(launch.base);
}

__device__ __forceinline__ ncclGin gin_context(const GinLaunch& launch, int local_expert) {
  return ncclGin(launch.comm, local_expert % launch.active_context_count);
}

// All lanes enter after producing the payload. The transport posts one request,
// retaining the original kernel's warp-per-token routing and packing algorithm.
__device__ __forceinline__ void gin_put_warp(const GinLaunch& launch, const void* dst,
                                             const void* src, size_t bytes, int peer,
                                             int local_expert) {
  if (peer == launch.comm.rank) {
    // Match the legacy NVSHMEM path: self traffic is a local vector copy.
    // Publication is joined by the dispatch scoreboard / combine barrier.
    __syncwarp();
    const auto* input = static_cast<const int4*>(src);
    auto* output = const_cast<int4*>(static_cast<const int4*>(dst));
    for (size_t i = threadIdx.x % 32; i < bytes / sizeof(int4); i += 32) output[i] = input[i];
    __syncwarp();
    return;
  }
  gin_context(launch, local_expert)
      .put(ncclTeamWorld(launch.comm), peer, launch.window, gin_offset(launch, dst), launch.window,
           gin_offset(launch, src), bytes, ncclGin_None{}, ncclGin_None{}, ncclCoopWarp{});
}

__device__ __forceinline__ void gin_wait_epoch(const GinLaunch& launch, int local_expert,
                                               int signal_index) {
  const int index = launch.signal_base + signal_index;
  const uint64_t expected = launch.epochs[index] + 1;
  gin_context(launch, local_expert).waitSignal(ncclCoopThread{}, index, expected);
  launch.epochs[index] = expected;
}

__device__ __forceinline__ void gin_finish_peer(const GinLaunch& launch, int peer,
                                                int local_expert) {
  auto gin = gin_context(launch, local_expert);
  ncclGinRequest_t request;
  gin.flushAsync(ncclTeamWorld(launch.comm), peer, &request);
  gin.wait(request);
}
#endif

}  // namespace deep_ep
