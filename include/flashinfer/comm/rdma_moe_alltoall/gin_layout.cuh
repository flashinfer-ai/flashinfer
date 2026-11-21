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
// 0/1/2 selects the combine codec and shares one send slot across both operations.
// Dispatch can independently select a typed codec or an opaque fixed-byte row.
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
                 bool planned = false, bool compact_rank = false, bool dispatch_rank_packet = false,
                 int dispatch_precision = -2, int data_bytes = 0, int scale_bytes = 0) {
    const size_t h = size_t(hidden), c = size_t(capacity), e = size_t(experts);
    const bool rank_capacity = compact_rank && planned && precision == 2;
    data_capacity_rows = (rank_capacity ? size_t(ranks) : e) * c;
    const int dp = dispatch_precision == -2 ? precision : dispatch_precision;
    const size_t payload = precision == 0 ? h : precision == 1 ? h / 2 : 2 * h;
    const size_t scales = precision == 0 ? (h / 128) * sizeof(float) : precision == 1 ? h / 16 : 0;
    const size_t dispatch_payload = dp == 3   ? aligned(size_t(data_bytes), 16)
                                    : dp == 0 ? h
                                    : dp == 1 ? h / 2
                                              : 2 * h;
    const size_t dispatch_scales = dp == 3   ? aligned(size_t(scale_bytes), 16)
                                   : dp == 0 ? (h / 128) * sizeof(float)
                                   : dp == 1 ? h / 16
                                             : 0;
    dispatch_msg_bytes = (planned                ? 80
                          : dispatch_rank_packet ? 48
                                                 : 16) +
                         dispatch_payload + dispatch_scales;
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
    // BF16 dispatch references retain immutable 16-byte headers. Reserve K=8
    // explicitly: with independent codecs, the combine arena may be smaller.
    // Compact planned rank fanout only sends full token records.
    const size_t reference_tail = dp == 2 && experts >= 2 && !rank_capacity ? c * 8 * 16 : 0;
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
