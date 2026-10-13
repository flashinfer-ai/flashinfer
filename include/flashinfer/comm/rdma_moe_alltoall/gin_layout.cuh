/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "config.hpp"

namespace deep_ep {

// GIN-only memory layout. The inherited NVSHMEM layout remains unchanged.
// precision: -1 preserves its BF16 upper bound and two send slots; explicit
// 0/1/2 selects FP8/NVFP4/BF16 and shares one send slot across both operations.
// NCCL 2.30.7 GDAKI source completion is checked by the host before reuse.
struct GinArenaLayout {
  LowLatencyBuffer buffers[2];
  size_t total_bytes = 0, send_bytes = 0, recv_bytes = 0, metadata_bytes = 0;
  size_t dispatch_msg_bytes = 0, combine_msg_bytes = 0;
  size_t data_capacity_rows = 0;

  static size_t aligned(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
  }

  GinArenaLayout(void* arena, int capacity, int hidden, int ranks, int experts, int precision,
                 bool planned = false, bool compact_rank = false, bool nvfp4_rank_packet = false) {
    const size_t h = size_t(hidden), c = size_t(capacity), e = size_t(experts);
    const bool rank_capacity = compact_rank && planned && precision == 2;
    data_capacity_rows = (rank_capacity ? size_t(ranks) : e) * c;
    const size_t payload = precision == 0 ? h : precision == 1 ? h / 2 : 2 * h;
    const size_t scales = precision == 0 ? (h / 128) * sizeof(float) : precision == 1 ? h / 16 : 0;
    dispatch_msg_bytes = (planned                               ? 80
                          : nvfp4_rank_packet && precision == 1 ? 48
                                                                : 16) +
                         payload + scales;
    combine_msg_bytes = aligned(payload + scales + (precision == 1 ? sizeof(float) : 0), 16);
    metadata_bytes = aligned(e * sizeof(int), 128);
    if (precision < 0) {
      LowLatencyLayout legacy(arena, capacity, hidden, ranks, experts);
      buffers[0] = legacy.buffers[0];
      buffers[1] = legacy.buffers[1];
      total_bytes = get_low_latency_rdma_size_hint(capacity, hidden, ranks, experts);
      send_bytes = std::max(c * (16 + 2 * h), e * c * 2 * h);
      recv_bytes = e * c * (16 + 2 * h);
      return;
    }
    // Planned BF16 rank references retain immutable 16-byte headers. Reserve
    // the worst K=8 tail explicitly; for E>=2 the combine arena dominates it.
    // Rank-fanout sends full token records, including its hook/grid fallback;
    // only legacy planned dispatch can use the per-expert reference tail.
    const size_t reference_tail = planned && experts >= 2 && !rank_capacity ? c * 8 * 16 : 0;
    send_bytes = aligned(
        std::max(c * dispatch_msg_bytes + reference_tail, data_capacity_rows * combine_msg_bytes),
        128);
    recv_bytes = aligned(data_capacity_rows * std::max(dispatch_msg_bytes, combine_msg_bytes), 128);
    total_bytes =
        aligned(2 * metadata_bytes + send_bytes + 2 * recv_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
    const auto address = reinterpret_cast<uintptr_t>(arena);
    auto* send = reinterpret_cast<void*>(address + 2 * metadata_bytes);
    for (int slot = 0; slot != 2; ++slot) {
      auto* recv = reinterpret_cast<void*>(address + 2 * metadata_bytes + send_bytes +
                                           size_t(slot) * recv_bytes);
      auto* metadata = reinterpret_cast<int*>(address + size_t(slot) * metadata_bytes);
      buffers[slot] = {experts, send,     recv, metadata,         send,
                       recv,    metadata, send, combine_msg_bytes};
    }
  }
};

}  // namespace deep_ep
