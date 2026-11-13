/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

// Included inside deep_ep::extensions after conversion helpers and precision traits.
// These kernels retain the legacy expert-major public layout and wire format,
// but partition work independently of the number of experts.

constexpr int kGinChannelThreads = 256;

__device__ __forceinline__ void gin_channels_publish(const GinLaunch& launch, int num_local_experts,
                                                     bool dispatch, bool finish_source = true) {
  const int lane = threadIdx.x % 32;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int num_warps = gridDim.x * blockDim.x / 32;
  const int num_ranks = launch.comm.nRanks;

  // The caller's grid barrier joined all payload producers. Each channel
  // signals on its own QPs, so the public strong-signal ordering contract
  // covers every payload even when writes originate from different CTAs.
  // A flush on one QP followed by a signal on another QP is insufficient:
  // NCCL only promises that flush releases source buffers, not remote data.
  // Independent peers must not serialize behind one channel's publisher.
  // Assign one warp to each (channel, peer) QP instead.
  for (int task = warp; task < launch.active_context_count * num_ranks; task += num_warps) {
    const int channel = task / num_ranks;
    const int peer = task % num_ranks;
    if (peer == launch.comm.rank) {
      if (dispatch && channel == 0) {
        for (int i = lane; i < num_local_experts; i += 32)
          launch.recv_counts[peer * num_local_experts + i] =
              launch.send_counts[peer * num_local_experts + i];
      }
      continue;
    }
    auto gin = gin_context(launch, channel);
    const auto signal = ncclGin_StrongSignalInc{
        static_cast<ncclGinSignal_t>(launch.signal_base + channel * num_ranks + launch.comm.rank)};
    if (dispatch && channel == 0) {
      gin.put(ncclTeamWorld(launch.comm), peer, launch.window,
              gin_offset(launch, launch.recv_counts + launch.comm.rank * num_local_experts),
              launch.window, gin_offset(launch, launch.send_counts + peer * num_local_experts),
              num_local_experts * sizeof(int), signal, ncclGin_None{}, ncclCoopWarp{});
    } else {
      gin.signal(ncclTeamWorld(launch.comm), peer, signal, ncclCoopWarp{});
    }
    // This QP's completion wait relies on the pinned NCCL 2.30.7 GDAKI
    // implementation, not on a stronger public flush contract.
    // gin/gdaki/gin_gdaki.h::flushAsyncImpl snapshots the QP-global
    // sq_rsvd_index, and waitImpl polls CQ completion through that tail.
    // The preceding grid join therefore includes every producer's puts,
    // even though one lane waits on the QP. Coalesced mode rejects other
    // backends. Remote visibility still comes from each QP's strong signal.
    if (lane == 0 && finish_source) gin_finish_peer(launch, peer, channel);
    __syncwarp();
  }
}

__device__ __forceinline__ void gin_channels_wait(const GinLaunch& launch) {
  const int lane = threadIdx.x % 32;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int num_warps = gridDim.x * blockDim.x / 32;
  const int num_ranks = launch.comm.nRanks;
  for (int task = warp; task < launch.active_context_count * num_ranks; task += num_warps) {
    const int channel = task / num_ranks;
    const int peer = task % num_ranks;
    if (lane == 0 && peer != launch.comm.rank) {
      gin_wait_epoch(launch, channel, task);
    }
  }
  // This also publishes self copies made by gin_channels_publish. In a
  // receive hook they were already joined by the preceding send kernel.
  cg::this_grid().sync();
}

// One waiter owns each (channel, source) epoch. Channel zero also publishes
// that source's complete expert-count vector. Its lane-zero acquire is followed
// by a full-warp boundary and system-acquire count loads before reserving ranges.
// The existing final grid join publishes those ranges/cursors and joins every
// payload channel, so no packet is consumed before all of its writes are visible.
__device__ __forceinline__ void gin_planned_fanout_wait_and_reserve(
    const GinLaunch& launch, int local_experts, int capacity, int* packed_recv_count,
    int64_t* packed_recv_layout_range, int* cumulative_local_expert_recv_stats) {
  const int lane = threadIdx.x % 32;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int num_warps = gridDim.x * blockDim.x / 32;
  const int num_ranks = launch.comm.nRanks;
  const int stride = local_experts + 1;
  for (int task = warp; task < launch.active_context_count * num_ranks; task += num_warps) {
    const int channel = task / num_ranks, peer = task % num_ranks;
    if (lane == 0 && peer != launch.comm.rank) gin_wait_epoch(launch, channel, task);
    // Self count publication uses the identical task-to-warp mapping in
    // gin_channels_publish; its local copies finish before this boundary too.
    __syncwarp();
    if (channel == 0) {
      if (lane == 0) {
        // The wire packet slot is also this private handle's partial slot.
        // Publish zero too: a replay with no packets must retire old entries.
        const int packets = ld_acquire_sys_global(launch.recv_counts + peer * stride);
        EP_DEVICE_ASSERT(packets >= 0 && packets <= capacity);
        launch.planned_counts[peer] = packets;
      }
      for (int expert = lane; expert < local_experts; expert += 32) {
        const int count = ld_acquire_sys_global(launch.recv_counts + peer * stride + 1 + expert);
        EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
        const int begin = atomicAdd(packed_recv_count + expert, count);
        EP_DEVICE_ASSERT(begin >= 0 && size_t(begin) + count <= size_t(num_ranks) * capacity);
        packed_recv_layout_range[expert * num_ranks + peer] = pack2<int, int64_t>(count, begin);
        launch.planned_fanout_cursors[peer * local_experts + expert] = begin;
        if (cumulative_local_expert_recv_stats != nullptr)
          atomicAdd(cumulative_local_expert_recv_stats + expert, count);
      }
    }
    // No lane advances to another channel/source while other lanes still
    // reserve this source's expert ranges, including the partial final warp.
    __syncwarp();
  }
  cg::this_grid().sync();
}

// Fused small-EP path: one complete warp owns a source. Individual lanes
// exclusively advance that source's distinct channel epochs. No other waiter
// runs for this call. A local ready count publishes every reserved range/cursor.
__device__ __forceinline__ void gin_planned_publish_source_ready(
    const GinLaunch& launch, int source, int local_experts, int capacity, int* packed_recv_count,
    int64_t* packed_recv_layout_range, int* cumulative_local_expert_recv_stats) {
  const int lane = threadIdx.x % 32;
  const int num_ranks = launch.comm.nRanks;
  const int stride = local_experts + 1;
  if (source != launch.comm.rank) {
    for (int channel = lane; channel < launch.active_context_count; channel += 32)
      gin_wait_epoch(launch, channel, channel * num_ranks + source);
  }
  __syncwarp();
  // Local payload and send counts were completed by the producer grid join.
  // Do not race the self count copy still running in gin_channels_publish.
  const int* counts = source == launch.comm.rank ? launch.send_counts + source * stride
                                                 : launch.recv_counts + source * stride;
  for (int expert = lane; expert < local_experts; expert += 32) {
    const int count = ld_acquire_sys_global(counts + 1 + expert);
    EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
    const int begin = atomicAdd(packed_recv_count + expert, count);
    EP_DEVICE_ASSERT(begin >= 0 && size_t(begin) + count <= size_t(num_ranks) * capacity);
    packed_recv_layout_range[expert * num_ranks + source] = pack2<int, int64_t>(count, begin);
    launch.planned_fanout_cursors[source * local_experts + expert] = begin;
    if (cumulative_local_expert_recv_stats != nullptr)
      atomicAdd(cumulative_local_expert_recv_stats + expert, count);
  }
  // These outputs are local to this GPU. Every writer completes its device
  // fence before lane zero release-publishes the count (including empty U=0).
  __threadfence();
  __syncwarp();
  if (lane == 0) {
    const int packets = ld_acquire_sys_global(counts);
    EP_DEVICE_ASSERT(packets >= 0 && packets <= capacity);
    st_release_sys_global(launch.planned_counts + source, packets);
  }
  __syncwarp();
}

__device__ __forceinline__ int gin_planned_acquire_source_ready(const int* ready) {
  int count;
  do {
    // The memory clobber also prevents compiler hoisting of ordinary range
    // loads ahead of the acquire; the GPU acquire alone would not express it.
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(count) : "l"(ready) : "memory");
  } while (count < 0);
  return count;
}

// Isolated NVFP4 packet protocol. Public expert-major rows/ranges remain the
// old Combine contract; only native ready/cursors reuse the BF16 fanout helpers.
// A packet contains token/T/K/flag, eight expert IDs, packed NVFP4, and scales.
// No weights or BF16 partial plan are transported or constructed.
template <int kHidden>
__global__ __launch_bounds__(kGinChannelThreads, 1) void dispatch_channels_nvfp4_rank_packet(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* cumulative_local_expert_recv_stats, void* rdma_recv_x, void* rdma_x, const void* x,
    const void* x_scales, const int* topk_idx, int num_tokens, int capacity, int num_topk,
    int num_experts, int rank, int num_ranks, int phases, GinLaunch launch) {
  constexpr size_t kHeaderBytes = 48;
  constexpr size_t kDataVecs = size_t(kHidden) / 2 / sizeof(int4);
  constexpr size_t kScaleVecs = size_t(kHidden) / 16 / sizeof(int4);
  constexpr size_t kMsgBytes = kHeaderBytes + size_t(kHidden) / 2 + size_t(kHidden) / 16;
  constexpr int kWarps = kGinChannelThreads / 32;
  const int lane = threadIdx.x % 32, local_warp = threadIdx.x / 32;
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_warp = global_thread / 32;
  const int total_threads = gridDim.x * blockDim.x;
  const int local_experts = num_experts / num_ranks;
  const int count_stride = local_experts + 1;
  const bool source_ready =
      phases == (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE) && gridDim.x > 1;
  const int data_ctas = int(gridDim.x) - int(source_ready);
  auto grid = cg::this_grid();
  EP_DEVICE_ASSERT(launch.nvfp4_rank_packet && !launch.bf16_local_partial_planned &&
                   (num_ranks == 4 || num_ranks == 8) && num_topk >= 1 && num_topk <= 8 &&
                   launch.planned_counts != nullptr && launch.planned_fanout_cursors != nullptr);
  if (phases & LOW_LATENCY_SEND_PHASE) {
    for (int source = global_thread; source < num_ranks; source += total_threads)
      launch.planned_counts[source] = -1;
    for (int entry = global_thread; entry < num_experts + num_ranks; entry += total_threads)
      launch.send_counts[entry] = 0;
    for (int expert = global_thread; expert < local_experts; expert += total_threads)
      packed_recv_count[expert] = 0;
    grid.sync();

    for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
      auto* message = static_cast<uint8_t*>(rdma_x) + size_t(token) * kMsgBytes;
      if (threadIdx.x == 0)
        *reinterpret_cast<int4*>(message) = make_int4(token, num_tokens, num_topk, 0);
      if (threadIdx.x < 8) {
        const int expert = int(threadIdx.x) < num_topk
                               ? __ldg(topk_idx + size_t(token) * num_topk + threadIdx.x)
                               : -1;
        EP_DEVICE_ASSERT(expert < num_experts);
        reinterpret_cast<int*>(message + 16)[threadIdx.x] = expert;
      }
      auto* data = reinterpret_cast<int4*>(message + kHeaderBytes);
      auto* scales = data + kDataVecs;
      for (size_t vector = threadIdx.x; vector < kDataVecs; vector += blockDim.x)
        data[vector] = __ldg(static_cast<const int4*>(x) + size_t(token) * kDataVecs + vector);
      for (size_t vector = threadIdx.x; vector < kScaleVecs; vector += blockDim.x)
        scales[vector] =
            __ldg(static_cast<const int4*>(x_scales) + size_t(token) * kScaleVecs + vector);
      __syncthreads();

      const int expert =
          local_warp < num_topk ? __ldg(topk_idx + size_t(token) * num_topk + local_warp) : -1;
      if (expert >= 0) {
        const int owner = expert / local_experts;
        if (lane == 0)
          atomicAdd(launch.send_counts + owner * count_stride + 1 + expert % local_experts, 1);
        bool first_owner = true;
        for (int k = 0; k < local_warp; ++k) {
          const int previous = __ldg(topk_idx + size_t(token) * num_topk + k);
          if (previous >= 0 && previous / local_experts == owner) first_owner = false;
        }
        if (first_owner) {
          int slot = lane == 0 ? atomicAdd(launch.send_counts + owner * count_stride, 1) : 0;
          slot = __shfl_sync(0xffffffff, slot, 0);
          EP_DEVICE_ASSERT(slot >= 0 && slot < capacity);
          auto* dst =
              static_cast<uint8_t*>(rdma_recv_x) + (size_t(rank) * capacity + slot) * kMsgBytes;
          gin_put_warp(launch, dst, message, kMsgBytes, owner, global_warp);
        }
      }
      __syncthreads();
    }
    // All packet producers finish posting before any channel signals. The
    // exact pinned GDAKI helper retains inline source completion for every QP.
    grid.sync();
    gin_channels_publish(launch, count_stride, true);
  }
  if (!(phases & LOW_LATENCY_RECV_PHASE)) return;

  if (source_ready) {
    if (int(blockIdx.x) == int(gridDim.x) - 1) {
      if (local_warp < num_ranks)
        gin_planned_publish_source_ready(launch, local_warp, local_experts, capacity,
                                         packed_recv_count, packed_recv_layout_range,
                                         cumulative_local_expert_recv_stats);
      // No later grid join. Kernel completion joins every publisher's source
      // wait, every source leader and every public-output writer.
      return;
    }
  } else {
    // Receive hooks use the same packet wire format. SEND has completed all
    // source reads; this helper waits all remote channels and publishes ranges.
    gin_planned_fanout_wait_and_reserve(launch, local_experts, capacity, packed_recv_count,
                                        packed_recv_layout_range,
                                        cumulative_local_expert_recv_stats);
  }
  const size_t packet_warp = size_t(local_warp) * data_ctas + blockIdx.x;
  const size_t packet_warps = size_t(data_ctas) * kWarps;
  size_t packet_total = 0, packet_start = packet_warp, packet_stride = packet_warps;
  int ready_source = -1;
  if (source_ready) {
    ready_source = int(packet_warp % num_ranks);
    packet_start = packet_warp / num_ranks;
    packet_stride = (packet_warps + num_ranks - 1 - ready_source) / num_ranks;
    const int count = gin_planned_acquire_source_ready(launch.planned_counts + ready_source);
    EP_DEVICE_ASSERT(count <= capacity);
    packet_total = size_t(count);
  } else {
    for (int source = 0; source < num_ranks; ++source) {
      const int count = ld_acquire_sys_global(launch.recv_counts + source * count_stride);
      EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
      packet_total += count;
    }
  }
  for (size_t packet = packet_start; packet < packet_total; packet += packet_stride) {
    size_t source_packet = packet;
    int source = ready_source;
    if (!source_ready) {
      source = 0;
      for (; source < num_ranks; ++source) {
        const int count = ld_acquire_sys_global(launch.recv_counts + source * count_stride);
        if (source_packet < size_t(count)) break;
        source_packet -= count;
      }
    }
    EP_DEVICE_ASSERT(source < num_ranks && source_packet < size_t(capacity));
    const auto* message = static_cast<const uint8_t*>(rdma_recv_x) +
                          (size_t(source) * capacity + source_packet) * kMsgBytes;
    const int4 header = ld_nc_global(reinterpret_cast<const int4*>(message));
    EP_DEVICE_ASSERT(header.w == 0 && header.x >= 0 && header.x < header.y &&
                     header.y <= capacity && header.z >= 1 && header.z <= 8);
    const int expert =
        lane < header.z ? ld_nc_global(reinterpret_cast<const int*>(message + 16) + lane) : -1;
    EP_DEVICE_ASSERT(expert < num_experts);
    const bool selected = expert >= 0 && expert / local_experts == rank;
    const unsigned selected_mask = __ballot_sync(0xffffffff, selected);
    EP_DEVICE_ASSERT(selected_mask != 0);
    uint64_t row = 0;
    if (selected) {
      const int local_expert = expert % local_experts;
      const int slot =
          atomicAdd(launch.planned_fanout_cursors + source * local_experts + local_expert, 1);
      const int64_t range = packed_recv_layout_range[local_expert * num_ranks + source];
      const int count = int(range & 0xffffffff), begin = int(uint64_t(range) >> 32);
      EP_DEVICE_ASSERT(slot >= begin && size_t(slot) < size_t(begin) + count);
      row = size_t(local_expert) * num_ranks * capacity + slot;
      packed_recv_src_info[row] = header.x;
    }
    uint64_t row0 = 0;
    if (selected_mask & (1u << 0)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 0);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 0);
      row0 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row1 = 0;
    if (selected_mask & (1u << 1)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 1);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 1);
      row1 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row2 = 0;
    if (selected_mask & (1u << 2)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 2);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 2);
      row2 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row3 = 0;
    if (selected_mask & (1u << 3)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 3);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 3);
      row3 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row4 = 0;
    if (selected_mask & (1u << 4)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 4);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 4);
      row4 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row5 = 0;
    if (selected_mask & (1u << 5)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 5);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 5);
      row5 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row6 = 0;
    if (selected_mask & (1u << 6)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 6);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 6);
      row6 = (uint64_t(hi) << 32) | lo;
    }
    uint64_t row7 = 0;
    if (selected_mask & (1u << 7)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 7);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 7);
      row7 = (uint64_t(hi) << 32) | lo;
    }
    const auto* data = reinterpret_cast<const int4*>(message + kHeaderBytes);
    for (size_t vector = lane; vector < kDataVecs; vector += 32) {
      const int4 value = ld_nc_global(data + vector);
      if (selected_mask & (1u << 0))
        st_na_global(static_cast<int4*>(packed_recv_x) + row0 * kDataVecs + vector, value);
      if (selected_mask & (1u << 1))
        st_na_global(static_cast<int4*>(packed_recv_x) + row1 * kDataVecs + vector, value);
      if (selected_mask & (1u << 2))
        st_na_global(static_cast<int4*>(packed_recv_x) + row2 * kDataVecs + vector, value);
      if (selected_mask & (1u << 3))
        st_na_global(static_cast<int4*>(packed_recv_x) + row3 * kDataVecs + vector, value);
      if (selected_mask & (1u << 4))
        st_na_global(static_cast<int4*>(packed_recv_x) + row4 * kDataVecs + vector, value);
      if (selected_mask & (1u << 5))
        st_na_global(static_cast<int4*>(packed_recv_x) + row5 * kDataVecs + vector, value);
      if (selected_mask & (1u << 6))
        st_na_global(static_cast<int4*>(packed_recv_x) + row6 * kDataVecs + vector, value);
      if (selected_mask & (1u << 7))
        st_na_global(static_cast<int4*>(packed_recv_x) + row7 * kDataVecs + vector, value);
    }
    const auto* scales = reinterpret_cast<const int4*>(message + kHeaderBytes) + kDataVecs;
    for (size_t vector = lane; vector < kScaleVecs; vector += 32) {
      const int4 value = ld_nc_global(scales + vector);
      if (selected_mask & (1u << 0))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row0 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 1))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row1 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 2))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row2 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 3))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row3 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 4))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row4 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 5))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row5 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 6))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row6 * kScaleVecs + vector, value);
      if (selected_mask & (1u << 7))
        st_na_global(static_cast<int4*>(packed_recv_x_scales) + row7 * kScaleVecs + vector, value);
    }
  }
}

// Planned BF16 only: one full record per destination token, complete expert
// counts, then a single packet traversal that fans each loaded vector out to
// its local expert rows. Public rows stay contiguous for each (source, expert).
template <int kHidden>
__global__ __launch_bounds__(kGinChannelThreads, 1) void dispatch_channels_planned_rank_fanout(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* cumulative_local_expert_recv_stats, void* rdma_recv_x, void* rdma_x, const void* x,
    const void* x_scales, const int* topk_idx, int num_tokens, int capacity, int num_topk,
    int num_experts, int rank, int num_ranks, int phases, GinLaunch launch) {
  constexpr size_t kHeaderBytes = 80;
  constexpr size_t kDataVecs = size_t(kHidden) * sizeof(nv_bfloat16) / sizeof(int4);
  constexpr size_t kMsgBytes = kHeaderBytes + size_t(kHidden) * sizeof(nv_bfloat16);
  constexpr int kWarps = kGinChannelThreads / 32;
  const int lane = threadIdx.x % 32, local_warp = threadIdx.x / 32;
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_warp = global_thread / 32;
  const int total_threads = gridDim.x * blockDim.x;
  const int local_experts = num_experts / num_ranks;
  const int count_stride = local_experts + 1;
  const bool source_ready = phases == (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE) &&
                            num_ranks <= kWarps && gridDim.x > 1;
  const bool defer_source_completion =
      phases == (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE) && gridDim.x > 1 && !source_ready;
  // Reserve enough tail CTAs to keep one warp per QP at the normal recipe,
  // but always retain at least one data CTA for very small configured grids.
  const int flush_ctas =
      defer_source_completion
          ? min((launch.active_context_count * num_ranks + kWarps - 1) / kWarps, int(gridDim.x) - 1)
          : 0;
  const int data_ctas = int(gridDim.x) - (source_ready ? 1 : flush_ctas);
  auto grid = cg::this_grid();

  EP_DEVICE_ASSERT(launch.bf16_local_partial_planned && num_topk >= 1 && num_topk <= kWarps &&
                   launch.planned_fanout_cursors != nullptr);
  if (phases & LOW_LATENCY_SEND_PHASE) {
    // The plan is private to this dispatch handle, including during Graph
    // replay. Only local receive expansion writes these cells. Reuse the
    // existing initial grid join to publish all invalid-row markers before
    // any receive writer, also for empty/all-negative sources and recv hooks.
    for (size_t cell = global_thread; cell < size_t(num_ranks) * capacity * 8;
         cell += total_threads)
      launch.planned_rows[cell] = -1;
    // Each first-owner send below fills exactly one [owner, token] cell.
    // These are handle-owned values, not references into reusable wire data.
    for (size_t cell = global_thread; cell < size_t(num_ranks) * capacity; cell += total_threads)
      launch.planned_source_slots[cell] = -1;
    if (source_ready) {
      for (int source = global_thread; source < num_ranks; source += total_threads)
        launch.planned_counts[source] = -1;
    }
    for (int entry = global_thread; entry < num_experts + num_ranks; entry += total_threads)
      launch.send_counts[entry] = 0;
    for (int expert = global_thread; expert < local_experts; expert += total_threads)
      packed_recv_count[expert] = 0;
    grid.sync();

    if (num_ranks <= kWarps) {
      // Every nonempty subset of two adjacent records is contiguous. Pack
      // each record once, then one owner warp sends its selected one or two
      // records without a peer-specific staging area or another barrier.
      const size_t pairs = (size_t(num_tokens) + 1) / 2;
      for (size_t pair = blockIdx.x; pair < pairs; pair += gridDim.x) {
        const size_t first_token = pair * 2;
        const int pair_tokens = first_token + 1 < size_t(num_tokens) ? 2 : 1;
        if (int(threadIdx.x) < pair_tokens) {
          const size_t token = first_token + threadIdx.x;
          auto* message = static_cast<uint8_t*>(rdma_x) + token * kMsgBytes;
          planned_pack_header(message, int(token), num_tokens, num_topk, topk_idx, launch);
        }
        // Each payload starts after its own 80-byte header. Treating both
        // payloads as one contiguous array would overwrite the second header.
        for (size_t i = threadIdx.x; i < size_t(pair_tokens) * kDataVecs; i += blockDim.x) {
          const size_t token = first_token + i / kDataVecs;
          const size_t vector = i % kDataVecs;
          auto* data = reinterpret_cast<int4*>(static_cast<uint8_t*>(rdma_x) + token * kMsgBytes +
                                               kHeaderBytes);
          data[vector] = __ldg(static_cast<const int4*>(x) + token * kDataVecs + vector);
        }
        __syncthreads();

        if (local_warp < num_ranks) {
          const int peer = local_warp;
          const int token_in_pair = lane / 8, k = lane % 8;
          int expert = -1;
          if (lane < 16 && token_in_pair < pair_tokens && k < num_topk) {
            expert = __ldg(topk_idx + (first_token + token_in_pair) * num_topk + k);
            EP_DEVICE_ASSERT(expert < num_experts);
          }
          const bool selected = expert >= 0 && expert / local_experts == peer;
          if (selected)
            atomicAdd(launch.send_counts + peer * count_stride + 1 + expert % local_experts, 1);
          // All 32 lanes participate, including inactive K/tail/upper lanes.
          const unsigned selected_mask = __ballot_sync(0xffffffff, selected);
          const bool has_first = (selected_mask & 0xffu) != 0;
          const bool has_second = (selected_mask & 0xff00u) != 0;
          const int records = int(has_first) + int(has_second);
          if (records != 0) {
            int slot = lane == 0 ? atomicAdd(launch.send_counts + peer * count_stride, records) : 0;
            slot = __shfl_sync(0xffffffff, slot, 0);
            EP_DEVICE_ASSERT(slot >= 0 && size_t(slot) + records <= size_t(capacity));
            const size_t first_selected = first_token + (has_first ? 0 : 1);
            if (lane == 0) {
              launch.planned_source_slots[size_t(peer) * capacity + first_selected] = slot;
              if (records == 2)
                launch.planned_source_slots[size_t(peer) * capacity + first_selected + 1] =
                    slot + 1;
            }
            auto* dst =
                static_cast<uint8_t*>(rdma_recv_x) + (size_t(rank) * capacity + slot) * kMsgBytes;
            const auto* source = static_cast<const uint8_t*>(rdma_x) + first_selected * kMsgBytes;
            // pair <= (INT_MAX - 1)/2 and peer < 8, so this seed fits int.
            gin_put_warp(launch, dst, source, size_t(records) * kMsgBytes, peer, int(pair) + peer);
          }
        }
        // These token records remain immutable until all QPs are completed.
        __syncthreads();
      }
    } else {
      for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
        auto* message = static_cast<uint8_t*>(rdma_x) + size_t(token) * kMsgBytes;
        if (threadIdx.x == 0)
          planned_pack_header(message, token, num_tokens, num_topk, topk_idx, launch);
        auto* data = reinterpret_cast<int4*>(message + kHeaderBytes);
        for (size_t i = threadIdx.x; i < kDataVecs; i += blockDim.x)
          data[i] = __ldg(static_cast<const int4*>(x) + size_t(token) * kDataVecs + i);
        __syncthreads();

        if (local_warp < num_topk) {
          const int expert = __ldg(topk_idx + size_t(token) * num_topk + local_warp);
          if (expert >= 0) {
            EP_DEVICE_ASSERT(expert < num_experts);
            const int peer = expert / local_experts;
            // Every valid route contributes exactly one expert count, while only
            // the first route to this owner allocates/sends the shared full record.
            if (lane == 0)
              atomicAdd(launch.send_counts + peer * count_stride + 1 + expert % local_experts, 1);
            bool first_owner = true;
            for (int previous = 0; previous < local_warp; ++previous) {
              const int earlier = __ldg(topk_idx + size_t(token) * num_topk + previous);
              if (earlier >= 0 && earlier / local_experts == peer) first_owner = false;
            }
            if (first_owner) {
              int slot = lane == 0 ? atomicAdd(launch.send_counts + peer * count_stride, 1) : 0;
              slot = __shfl_sync(0xffffffff, slot, 0);
              EP_DEVICE_ASSERT(slot >= 0 && slot < capacity);
              if (lane == 0) launch.planned_source_slots[size_t(peer) * capacity + token] = slot;
              auto* dst =
                  static_cast<uint8_t*>(rdma_recv_x) + (size_t(rank) * capacity + slot) * kMsgBytes;
              gin_put_warp(launch, dst, message, kMsgBytes, peer, global_warp);
            }
          }
        }
        // Full records are per-token and immutable through all QP completions.
        __syncthreads();
      }
    }
    grid.sync();
    // The helper's vector length is generic: each peer now gets U followed by
    // all L expert counts. Channel signals/flushes are otherwise unchanged.
    // Preserve every strong signal. Source-ready uses original inline QP
    // completion: other CTAs can receive while publisher CTAs finish sources.
    // The larger-EP fused fallback retains the old deferred-flush tail CTAs;
    // hooks and one-CTA grids retain their original inline completion.
    gin_channels_publish(launch, count_stride, true, !defer_source_completion);
  }
  if (!(phases & LOW_LATENCY_RECV_PHASE)) return;

  if (source_ready) {
    if (int(blockIdx.x) == int(gridDim.x) - 1) {
      if (local_warp < num_ranks)
        gin_planned_publish_source_ready(launch, local_warp, local_experts, capacity,
                                         packed_recv_count, packed_recv_layout_range,
                                         cumulative_local_expert_recv_stats);
      // There are no later grid joins. Main-kernel completion includes every
      // source leader, inline publisher source wait, and packet expansion.
      return;
    }
  } else {
    gin_planned_fanout_wait_and_reserve(launch, local_experts, capacity, packed_recv_count,
                                        packed_recv_layout_range,
                                        cumulative_local_expert_recv_stats);
  }
  if (defer_source_completion && int(blockIdx.x) >= data_ctas) {
    // The existing grid join above includes all payload/count/strong-signal
    // posts. The pinned 2.30.7 GDAKI flush snapshots the persistent QP-global
    // reserved tail. There are no later grid joins: kernel completion joins
    // these waiters with all data CTAs before any same-stream arena reuse.
    const int flush_warp = (int(blockIdx.x) - data_ctas) * kWarps + local_warp;
    for (int task = flush_warp; task < launch.active_context_count * num_ranks;
         task += flush_ctas * kWarps) {
      const int channel = task / num_ranks, peer = task % num_ranks;
      if (lane == 0 && peer != rank) gin_finish_peer(launch, peer, channel);
      __syncwarp();
    }
    return;
  }
  // Transpose the global warp ordinal to spread a small packet population over
  // CTAs rather than filling all eight warps of the first CTA first.
  const size_t packet_warp = size_t(local_warp) * data_ctas + blockIdx.x;
  const size_t packet_warps = size_t(data_ctas) * kWarps;
  size_t packet_total = 0, packet_start = packet_warp, packet_stride = packet_warps;
  int ready_source = -1;
  if (source_ready) {
    ready_source = int(packet_warp % num_ranks);
    packet_start = packet_warp / num_ranks;
    // Non-power-of-two EP groups do not divide the number of data warps.
    // Count exactly the warps assigned to this source, including the remainder.
    packet_stride = (packet_warps + num_ranks - 1 - ready_source) / num_ranks;
    const int count = gin_planned_acquire_source_ready(launch.planned_counts + ready_source);
    EP_DEVICE_ASSERT(count <= capacity);
    packet_total = size_t(count);
  } else {
    for (int peer = 0; peer < num_ranks; ++peer) {
      const int count = ld_acquire_sys_global(launch.recv_counts + peer * count_stride);
      EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
      packet_total += count;
    }
  }
  for (size_t packet = packet_start; packet < packet_total; packet += packet_stride) {
    size_t source_packet = packet;
    int peer = ready_source;
    if (!source_ready) {
      peer = 0;
      for (; peer < num_ranks; ++peer) {
        const int count = ld_acquire_sys_global(launch.recv_counts + peer * count_stride);
        if (source_packet < size_t(count)) break;
        source_packet -= count;
      }
    }
    EP_DEVICE_ASSERT(peer < num_ranks && source_packet < size_t(capacity));
    const auto* message = static_cast<const uint8_t*>(rdma_recv_x) +
                          (size_t(peer) * capacity + source_packet) * kMsgBytes;
    const int4 header = ld_nc_global(reinterpret_cast<const int4*>(message));
    EP_DEVICE_ASSERT(header.w == 0 && header.x >= 0 && header.x < header.y &&
                     header.y <= capacity && header.z >= 1 && header.z <= 8);
    // One packet warp owns this [source, wire-slot] entry. Its inverse was
    // stored by the sender's first-owner warp, so neither side needs a sorted
    // token prefix scan. Kernel completion publishes this plan and all rows.
    if (lane == 0) launch.planned_tokens[size_t(peer) * capacity + source_packet] = header.x;
    const int expert_id =
        lane < header.z ? ld_nc_global(reinterpret_cast<const int*>(message + 16) + lane) : -1;
    EP_DEVICE_ASSERT(expert_id < num_experts);
    const bool selected = expert_id >= 0 && expert_id / local_experts == rank;
    const unsigned selected_mask = __ballot_sync(0xffffffff, selected);
    EP_DEVICE_ASSERT(selected_mask != 0);
    // Each selected lane owns one original-k plan cell. Duplicate positive
    // experts are outside the public routing contract, as in the parent path.
    uint64_t row = 0;
    if (selected) {
      const int expert = expert_id % local_experts;
      const int slot = atomicAdd(launch.planned_fanout_cursors + peer * local_experts + expert, 1);
      const int64_t range = packed_recv_layout_range[expert * num_ranks + peer];
      const int count = int(range & 0xffffffff), begin = int(uint64_t(range) >> 32);
      EP_DEVICE_ASSERT(slot >= begin && size_t(slot) < size_t(begin) + count);
      row = (size_t(expert) * num_ranks * capacity) + slot;
      packed_recv_src_info[row] = header.x;
      const size_t cell = (size_t(peer) * capacity + header.x) * 8 + lane;
      launch.planned_rows[cell] = int64_t(row);
      launch.planned_owner_weights[cell] =
          ld_nc_global(reinterpret_cast<const float*>(message + 48) + lane);
    }
    // The selected public rows remain fixed for the whole packet. Keep eight
    // named bases instead of a dynamically indexed register array: broadcast
    // each selected 64-bit row once, before the payload-vector loop.
    size_t base0 = 0;
    if (selected_mask & (1u << 0)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 0);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 0);
      base0 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base1 = 0;
    if (selected_mask & (1u << 1)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 1);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 1);
      base1 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base2 = 0;
    if (selected_mask & (1u << 2)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 2);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 2);
      base2 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base3 = 0;
    if (selected_mask & (1u << 3)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 3);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 3);
      base3 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base4 = 0;
    if (selected_mask & (1u << 4)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 4);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 4);
      base4 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base5 = 0;
    if (selected_mask & (1u << 5)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 5);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 5);
      base5 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base6 = 0;
    if (selected_mask & (1u << 6)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 6);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 6);
      base6 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    size_t base7 = 0;
    if (selected_mask & (1u << 7)) {
      const uint32_t lo = __shfl_sync(0xffffffff, uint32_t(row), 7);
      const uint32_t hi = __shfl_sync(0xffffffff, uint32_t(row >> 32), 7);
      base7 = ((uint64_t(hi) << 32) | lo) * kDataVecs;
    }
    const auto* data = reinterpret_cast<const int4*>(message + kHeaderBytes);
    auto* output = static_cast<int4*>(packed_recv_x);
    for (size_t i = lane; i < kDataVecs; i += 32) {
      const int4 value = ld_nc_global(data + i);
      // Preserve original ascending-k store order and one load per vector.
      if (selected_mask & (1u << 0)) output[base0 + i] = value;
      if (selected_mask & (1u << 1)) output[base1 + i] = value;
      if (selected_mask & (1u << 2)) output[base2 + i] = value;
      if (selected_mask & (1u << 3)) output[base3 + i] = value;
      if (selected_mask & (1u << 4)) output[base4 + i] = value;
      if (selected_mask & (1u << 5)) output[base5 + i] = value;
      if (selected_mask & (1u << 6)) output[base6 + i] = value;
      if (selected_mask & (1u << 7)) output[base7 + i] = value;
    }
  }
}

template <typename Precision, int kHidden>
__global__ __launch_bounds__(kGinChannelThreads, 1) void dispatch_channels(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* cumulative_local_expert_recv_stats, void* rdma_recv_x, void* rdma_x, const void* x,
    const void* x_scales, const int* topk_idx, int num_tokens, int capacity, int num_topk,
    int num_experts, int rank, int num_ranks, int phases, GinLaunch launch) {
  // Byte and row offsets use size_t: expert-major arenas exceed 2 GiB at
  // larger capacities even with NVFP4 payloads.
  constexpr size_t kHiddenBytes = kHidden / Precision::kNumElemsPerUnit * Precision::kUnitBytes;
  constexpr size_t kScalesBytes =
      kHidden / Precision::kNumElemsPerScale * Precision::kBytesPerScale;
  constexpr size_t kDataVecs = kHiddenBytes / sizeof(int4);
  constexpr size_t kScaleVecs = kScalesBytes / sizeof(int4);
  const size_t header_bytes = launch.bf16_local_partial_planned ? 80 : sizeof(int4);
  const size_t kMsgBytes = header_bytes + kHiddenBytes + kScalesBytes;
  const int lane = threadIdx.x % 32;
  const int local_warp = threadIdx.x / 32;
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_warp = global_thread / 32;
  const int total_threads = gridDim.x * blockDim.x;
  const int total_warps = total_threads / 32;
  const int local_experts = num_experts / num_ranks;
  auto grid = cg::this_grid();
  // BF16 references fit in the registered send arena's idle combine space:
  // C * (16 + 2H + 16K) <= E * C * 2H for E >= 2, H >= 4096, K <= 8.
  const bool rank_dedup = Precision::kBytesPerScale == 0 && num_experts >= 2;
  __shared__ int route_slots[kGinChannelThreads / 32];

  if (phases & LOW_LATENCY_SEND_PHASE) {
    for (int expert = global_thread; expert < num_experts; expert += total_threads)
      launch.send_counts[expert] = 0;
    for (int expert = global_thread; expert < local_experts; expert += total_threads)
      packed_recv_count[expert] = 0;
    grid.sync();

    // Eight warps cooperate on each token, then route its eight expert
    // copies concurrently. A small grid can iterate over arbitrarily many
    // tokens; it no longer needs one warp group per expert.
    for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
      int expert = -1, slot = 0;
      if (rank_dedup && local_warp < num_topk) {
        expert = __ldg(topk_idx + token * num_topk + local_warp);
        if (expert >= 0) {
          EP_DEVICE_ASSERT(expert < num_experts);
          if (lane == 0) {
            slot = atomicAdd(launch.send_counts + expert, 1);
            route_slots[local_warp] = slot;
          }
          slot = __shfl_sync(0xffffffff, slot, 0);
          EP_DEVICE_ASSERT(slot < capacity);
        }
      }
      auto* message = static_cast<uint8_t*>(rdma_x) + token * kMsgBytes;
      auto* data = reinterpret_cast<int4*>(message + header_bytes);
      if (threadIdx.x == 0) {
        if (launch.bf16_local_partial_planned)
          planned_pack_header(message, token, num_tokens, num_topk, topk_idx, launch);
        else
          *reinterpret_cast<int4*>(message) = make_int4(token, 0, 0, 0);
      }
      for (int i = threadIdx.x; i < kDataVecs; i += blockDim.x)
        data[i] = __ldg(static_cast<const int4*>(x) + token * kDataVecs + i);
      if constexpr (kScalesBytes > 0) {
        auto* scales = reinterpret_cast<int4*>(message + header_bytes + kHiddenBytes);
        for (int i = threadIdx.x; i < kScaleVecs; i += blockDim.x)
          scales[i] = __ldg(static_cast<const int4*>(x_scales) + token * kScaleVecs + i);
      }
      __syncthreads();

      if (local_warp < num_topk) {
        if (!rank_dedup) expert = __ldg(topk_idx + token * num_topk + local_warp);
        if (expert >= 0) {
          EP_DEVICE_ASSERT(expert < num_experts);
          if (!rank_dedup) {
            slot = lane == 0 ? atomicAdd(launch.send_counts + expert, 1) : 0;
            slot = __shfl_sync(0xffffffff, slot, 0);
          }
          EP_DEVICE_ASSERT(slot < capacity);
          const int peer = expert / local_experts;
          const int local_expert = expert % local_experts;
          const uint8_t* source = message;
          size_t bytes = kMsgBytes;
          if (rank_dedup && peer != rank) {
            int leader = local_warp, leader_expert = local_expert;
            for (int previous = 0; previous < local_warp; ++previous) {
              const int earlier = __ldg(topk_idx + token * num_topk + previous);
              if (earlier >= 0 && earlier / local_experts == peer) {
                leader = previous;
                leader_expert = earlier % local_experts;
                break;
              }
            }
            if (leader != local_warp) {
              auto* reference = reinterpret_cast<int4*>(static_cast<uint8_t*>(rdma_x) +
                                                        size_t(capacity) * kMsgBytes) +
                                size_t(token) * num_topk + local_warp;
              if (lane == 0) *reference = make_int4(token, leader_expert, route_slots[leader], 1);
              __syncwarp();
              source = reinterpret_cast<const uint8_t*>(reference);
              bytes = sizeof(int4);
            }
          }
          auto* dst = static_cast<uint8_t*>(rdma_recv_x) +
                      ((size_t(local_expert) * num_ranks + rank) * capacity + slot) * kMsgBytes;
          gin_put_warp(launch, dst, source, bytes, peer, global_warp);
        }
      }
      // This existing join also prevents the next token from overwriting
      // a leader slot before all followers have built immutable headers.
      __syncthreads();
    }
    grid.sync();
    gin_channels_publish(launch, local_experts, true);
  }
  if (!(phases & LOW_LATENCY_RECV_PHASE)) return;

  gin_channels_wait(launch);
  for (int task = global_warp; task < num_experts; task += total_warps) {
    const int peer = task / local_experts;
    const int expert = task % local_experts;
    int count = 0, begin = 0;
    if (lane == 0) {
      count = ld_acquire_sys_global(launch.recv_counts + task);
      EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
      begin = atomicAdd(packed_recv_count + expert, count);
      packed_recv_layout_range[expert * num_ranks + peer] = pack2<int, int64_t>(count, begin);
      if (cumulative_local_expert_recv_stats != nullptr)
        atomicAdd(cumulative_local_expert_recv_stats + expert, count);
    }
    count = __shfl_sync(0xffffffff, count, 0);
    begin = __shfl_sync(0xffffffff, begin, 0);
    const auto* src = static_cast<const uint8_t*>(rdma_recv_x) +
                      (size_t(expert) * num_ranks + peer) * capacity * kMsgBytes;
    const size_t output_base = size_t(expert) * num_ranks * capacity;
    for (int token = 0; token < count; ++token) {
      const auto* message = src + token * kMsgBytes;
      if (lane == 0)
        packed_recv_src_info[output_base + begin + token] =
            ld_nc_global(reinterpret_cast<const int*>(message));
      const auto* payload = message;
      if (rank_dedup) {
        // gin_channels_wait has joined every context's strong signal. A
        // leader on another expert/channel is therefore already visible.
        const int4 header = ld_nc_global(reinterpret_cast<const int4*>(message));
        if (header.w != 0) {
          EP_DEVICE_ASSERT(header.w == 1 && header.y >= 0 && header.y < local_experts);
          EP_DEVICE_ASSERT(header.z >= 0 && header.z < capacity);
          payload = static_cast<const uint8_t*>(rdma_recv_x) +
                    ((size_t(header.y) * num_ranks + peer) * capacity + header.z) * kMsgBytes;
          EP_DEVICE_ASSERT(ld_nc_global(reinterpret_cast<const int*>(payload)) == header.x);
        }
      }
      if (launch.bf16_local_partial_planned && lane == 0)
        planned_record_row(payload, peer, rank * local_experts + expert,
                           output_base + begin + token, capacity, num_experts, launch);
      const auto* data = reinterpret_cast<const int4*>(payload + header_bytes);
      auto* out_data =
          static_cast<int4*>(packed_recv_x) + (output_base + begin + token) * kDataVecs;
      for (int i = lane; i < kDataVecs; i += 32) out_data[i] = ld_nc_global(data + i);
      if constexpr (kScalesBytes > 0) {
        const auto* scales = reinterpret_cast<const int4*>(message + header_bytes + kHiddenBytes);
        auto* out_scales =
            static_cast<int4*>(packed_recv_x_scales) + (output_base + begin + token) * kScaleVecs;
        for (int i = lane; i < kScaleVecs; i += 32) out_scales[i] = ld_nc_global(scales + i);
      }
    }
  }
}

template <typename Precision, int kHidden>
__device__ __forceinline__ void gin_combine_send_row(
    void* rdma_recv_x, void* rdma_send_x, const void* x, const float* global_scale_per_token,
    const int* src_info, int capacity, int num_ranks, int local_experts, int rank, int expert,
    int token, int peer, int channel, const GinLaunch& launch) {
  using CommVecType = typename Precision::CommVecType;
  using ScaleType = typename Precision::ScaleType;
  constexpr int kElements = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kThreadsPerScale = Precision::kNumElemsPerScale / kElements;
  constexpr int kInputVecs = kHidden / kElements;
  constexpr size_t kCommBytes = kHidden / Precision::kNumElemsPerUnit * Precision::kUnitBytes;
  constexpr size_t kScaleBytes = kHidden / Precision::kNumElemsPerScale * Precision::kBytesPerScale;
  constexpr size_t kGlobalBytes = Precision::kUseGlobalScale ? sizeof(float) : 0;
  constexpr size_t kMsgBytes =
      (kCommBytes + kScaleBytes + kGlobalBytes + sizeof(int4) - 1) / sizeof(int4) * sizeof(int4);
  const int lane = threadIdx.x % 32;
  const size_t row = size_t(expert) * num_ranks * capacity + token;
  auto* message = static_cast<uint8_t*>(rdma_send_x) + row * kMsgBytes;
  auto* data = reinterpret_cast<CommVecType*>(message);
  float global_scale = 0.f;
  if constexpr (Precision::kUseGlobalScale) {
    global_scale = __ldg(global_scale_per_token + row);
    if (lane == 0) *reinterpret_cast<float*>(message + kCommBytes + kScaleBytes) = global_scale;
  }
  const auto* input = static_cast<const int4*>(x) + row * kInputVecs;
  for (int i = lane; i < kInputVecs; i += 32) {
    auto [value, scale] = Precision::quantize(__ldg(input + i), global_scale);
    data[i] = value;
    if constexpr (Precision::kBytesPerScale > 0) {
      if (i % kThreadsPerScale == 0)
        reinterpret_cast<ScaleType*>(message + kCommBytes)[i / kThreadsPerScale] = scale;
    }
  }
  const int source_token = __ldg(src_info + row);
  const int global_expert = rank * local_experts + expert;
  auto* dst = static_cast<uint8_t*>(rdma_recv_x) +
              (size_t(global_expert) * capacity + source_token) * kMsgBytes;
  gin_put_warp(launch, dst, message, kMsgBytes, peer, channel);
}

template <typename Precision, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kGinChannelThreads, 1) void combine_channels(
    void* combined_x, void* rdma_recv_x, void* rdma_send_x, const void* x,
    const float* global_scale_per_token, const int* topk_idx, const float* topk_weights,
    const int* src_info, const int64_t* layout_range, int num_tokens, int num_topk, int capacity,
    int num_experts, int rank, int num_ranks, int phases, GinLaunch launch) {
  using CommVecType = typename Precision::CommVecType;
  using ScaleType = typename Precision::ScaleType;
  constexpr int kElements = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kThreadsPerScale = Precision::kNumElemsPerScale / kElements;
  constexpr int kInputVecs = kHidden / kElements;
  constexpr size_t kCommBytes = kHidden / Precision::kNumElemsPerUnit * Precision::kUnitBytes;
  constexpr size_t kScaleBytes = kHidden / Precision::kNumElemsPerScale * Precision::kBytesPerScale;
  constexpr size_t kGlobalBytes = Precision::kUseGlobalScale ? sizeof(float) : 0;
  constexpr size_t kMsgBytes =
      (kCommBytes + kScaleBytes + kGlobalBytes + sizeof(int4) - 1) / sizeof(int4) * sizeof(int4);
  const int lane = threadIdx.x % 32;
  const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int total_warps = gridDim.x * blockDim.x / 32;
  const int local_experts = num_experts / num_ranks;
  auto grid = cg::this_grid();

  if (phases & LOW_LATENCY_SEND_PHASE) {
    if (num_ranks <= 32) {
      // Dispatch compacts every expert's rows into [0, total), but the
      // source-rank ranges can arrive in any order. Distribute whole rows
      // across several warps per expert instead of serializing all rows
      // belonging to one (expert, source rank) pair in a single warp.
      // Transpose the warp numbering so first rows occupy different CTAs
      // even for tiny token counts. The same numbering balances QPs when
      // most active work belongs to warp zero of each CTA.
      const int row_warp = (threadIdx.x / 32) * gridDim.x + blockIdx.x;
      const int row_begin = row_warp / local_experts;
      for (int expert = row_warp % local_experts; expert < local_experts; expert += total_warps) {
        int count = 0, begin = 0;
        if (lane < num_ranks)
          unpack2(__ldg(layout_range + expert * num_ranks + lane), count, begin);
        const int total = warp_reduce_sum(count);
        const int row_stride = max(1, (total_warps + local_experts - 1 - expert) / local_experts);
        for (int token = row_begin; token < total; token += row_stride) {
          const unsigned peers = __ballot_sync(
              0xffffffff, lane < num_ranks && token >= begin && token < begin + count);
          // Empty ranges never match; no ordering by peer is assumed.
          EP_DEVICE_ASSERT(__popc(peers) == 1);
          const int peer = __ffs(peers) - 1;
          gin_combine_send_row<Precision, kHidden>(
              rdma_recv_x, rdma_send_x, x, global_scale_per_token, src_info, capacity, num_ranks,
              local_experts, rank, expert, token, peer, row_warp, launch);
        }
      }
    } else {
      // Preserve the original pair scheduler for EP groups wider than a
      // warp. The row scheduler above requires one lane per source rank.
      for (int task = global_warp; task < num_experts; task += total_warps) {
        const int peer = task / local_experts;
        const int expert = task % local_experts;
        int count, begin;
        unpack2(__ldg(layout_range + expert * num_ranks + peer), count, begin);
        for (int token = begin; token < begin + count; ++token)
          gin_combine_send_row<Precision, kHidden>(
              rdma_recv_x, rdma_send_x, x, global_scale_per_token, src_info, capacity, num_ranks,
              local_experts, rank, expert, token, peer, global_warp, launch);
      }
    }
    grid.sync();
    gin_channels_publish(launch, local_experts, false);
  }
  if (!(phases & LOW_LATENCY_RECV_PHASE)) return;

  gin_channels_wait(launch);
  // Tile the hidden dimension as well as tokens so a tiny batch can use
  // several CTAs per token. A task owns one disjoint group of 256 int4 output
  // vectors; each active thread still sums top-k experts in their original
  // order. No communication or cross-CTA reduction occurs in this epilogue.
  constexpr int kRecvTiles = (kInputVecs + kGinChannelThreads - 1) / kGinChannelThreads;
  for (int64_t task = blockIdx.x; task < int64_t(num_tokens) * kRecvTiles; task += gridDim.x) {
    const int token = static_cast<int>(task / kRecvTiles);
    const int i = static_cast<int>(task % kRecvTiles) * kGinChannelThreads + threadIdx.x;
    // The last H7168 tile has 128 active threads. Skip metadata loads for
    // its other threads; all required grid synchronization is already done.
    if (i >= kInputVecs) continue;
    int experts[kNumMaxTopk];
    float weights[kNumMaxTopk];
#pragma unroll
    for (int k = 0; k < num_topk; ++k) {
      experts[k] = __ldg(topk_idx + token * num_topk + k);
      weights[k] = __ldg(topk_weights + token * num_topk + k);
    }
    float sum[kElements] = {0.f};
#pragma unroll
    for (int k = 0; k < num_topk; ++k)
      if (experts[k] >= 0) {
        const auto* message = static_cast<const uint8_t*>(rdma_recv_x) +
                              (size_t(experts[k]) * capacity + token) * kMsgBytes;
        float global_scale = 0.f;
        if constexpr (Precision::kUseGlobalScale)
          global_scale =
              ld_nc_global(reinterpret_cast<const float*>(message + kCommBytes + kScaleBytes));
        const auto value = ld_nc_global(reinterpret_cast<const CommVecType*>(message) + i);
        ScaleType scale{};
        if constexpr (Precision::kBytesPerScale > 0)
          scale = ld_nc_global(reinterpret_cast<const ScaleType*>(message + kCommBytes) +
                               i / kThreadsPerScale);
        float values[kElements];
        Precision::dequantize(value, global_scale, scale, values, weights[k]);
#pragma unroll
        for (int j = 0; j < kElements; ++j) sum[j] += values[j];
      }
    int4 result;
#pragma unroll
    for (int j = 0; j < kElements; ++j)
      reinterpret_cast<nv_bfloat16*>(&result)[j] = static_cast<nv_bfloat16>(sum[j]);
    static_cast<int4*>(combined_x)[size_t(token) * kInputVecs + i] = result;
  }
}
