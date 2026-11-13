/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

// Exactly one packing thread owns this immutable token record. The existing
// dispatch packing barrier publishes it to every routing warp before any put.
__device__ __forceinline__ void planned_pack_header(void* message, int token, int tokens, int topk,
                                                    const int* ids, const GinLaunch& launch) {
  *reinterpret_cast<int4*>(message) = make_int4(token, tokens, topk, 0);
  auto* header_ids = reinterpret_cast<int*>(static_cast<uint8_t*>(message) + 16);
  auto* header_weights = reinterpret_cast<float*>(static_cast<uint8_t*>(message) + 48);
  for (int k = 0; k < 8; ++k) {
    const int expert = k < topk ? __ldg(ids + size_t(token) * topk + k) : -1;
    const float weight =
        k < topk ? __ldg(launch.planned_input_weights + size_t(token) * topk + k) : 0.f;
    header_ids[k] = expert;
    header_weights[k] = weight;
    if (k < topk) {
      launch.planned_source_ids[size_t(token) * topk + k] = expert;
      launch.planned_source_weights[size_t(token) * topk + k] = weight;
    }
  }
}

// Distinct expert IDs are the dispatch contract: each (source,token,k) has
// exactly one receiving expert/warp. No shared per-token metadata is rewritten.
__device__ __forceinline__ void planned_record_row(const uint8_t* full_message, int source,
                                                   int expert, size_t row, int capacity,
                                                   int experts, const GinLaunch& launch) {
  const int4 header = ld_nc_global(reinterpret_cast<const int4*>(full_message));
  EP_DEVICE_ASSERT(header.w == 0 && header.x >= 0 && header.x < header.y && header.y <= capacity &&
                   header.z >= 1 && header.z <= 8);
  const auto* ids = reinterpret_cast<const int*>(full_message + 16);
  const auto* weights = reinterpret_cast<const float*>(full_message + 48);
  int matched = -1;
  for (int k = 0; k < header.z; ++k) {
    const int id = ld_nc_global(ids + k);
    EP_DEVICE_ASSERT(id < experts);
    if (id == expert) {
      EP_DEVICE_ASSERT(matched == -1);
      matched = k;
    }
  }
  EP_DEVICE_ASSERT(matched >= 0);
  const size_t cell = (size_t(source) * capacity + header.x) * 8 + matched;
  launch.planned_rows[cell] = int64_t(row);
  launch.planned_owner_weights[cell] = ld_nc_global(weights + matched);
}

// This same-stream D epilogue runs after all dispatch receive writers finish.
// A handle owns its outputs; no plan points into the reusable registered arena.
// Both sides number selected tokens in ascending source-token order.
__global__ void finalize_planned_dispatch_kernel(int tokens, int topk, int capacity, int ranks,
                                                 int local_experts, GinLaunch launch) {
  const int task = blockIdx.x;
  const bool producing = task < ranks;
  const int peer = task % ranks;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ int warp_totals[8];
  int carry = 0;
  for (int base = 0; base < capacity; base += blockDim.x) {
    const int token = base + threadIdx.x;
    bool selected = false;
    if (token < capacity) {
      if (producing) {
        for (int k = 0; k < 8; ++k)
          selected |= launch.planned_rows[(size_t(peer) * capacity + token) * 8 + k] >= 0;
      } else if (token < tokens) {
        for (int k = 0; k < topk; ++k) {
          const int expert = launch.planned_source_ids[size_t(token) * topk + k];
          selected |= expert >= 0 && expert / local_experts == peer;
        }
      }
    }
    const unsigned mask = __ballot_sync(0xffffffff, selected);
    if (lane == 0) warp_totals[warp] = __popc(mask);
    __syncthreads();
    int prefix = __popc(mask & ((1u << lane) - 1)), total = 0;
    for (int w = 0; w < blockDim.x / 32; ++w) {
      if (w < warp) prefix += warp_totals[w];
      total += warp_totals[w];
    }
    if (token < capacity) {
      if (producing) {
        if (selected) launch.planned_tokens[size_t(peer) * capacity + carry + prefix] = token;
      } else {
        launch.planned_source_slots[size_t(peer) * capacity + token] =
            selected ? carry + prefix : -1;
      }
    }
    carry += total;
    __syncthreads();
  }
  if (producing && threadIdx.x == 0) launch.planned_counts[peer] = carry;
}

void finalize_planned_dispatch(int tokens, int topk, int capacity, int ranks, int local_experts,
                               cudaStream_t stream, const GinLaunch& launch) {
  finalize_planned_dispatch_kernel<<<2 * ranks, 256, 0, stream>>>(tokens, topk, capacity, ranks,
                                                                  local_experts, launch);
}

template <int kHidden>
__global__ __launch_bounds__(256, 1) void combine_bf16_local_partial_planned(
    void* combined_x, void* rdma_recv_x, void* rdma_send_x, const void* x, int num_tokens,
    int capacity, int num_topk, int num_experts, int rank, int num_ranks, int phases,
    GinLaunch launch) {
  constexpr int kVecElements = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kPayloadVectors = kHidden / kVecElements;
  constexpr int kChunkTokens = 2;
  static_assert(kPayloadVectors % 32 == 0, "final reduction requires complete token warps");
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  const int lane = threadIdx.x % 32, warp = thread / 32, warps = threads / 32;
  const int local_experts = num_experts / num_ranks;
  const int signal_base = launch.partial_signal_base + num_ranks;
  const bool defer_source_completion =
      phases == (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE) && gridDim.x > 1;
  // One tail CTA has eight independent warp waiters: all remote QPs fit
  // concurrently at EP4/EP8. Larger worlds retain coverage by a warp stride.
  const int reduce_ctas = int(gridDim.x) - (defer_source_completion ? 1 : 0);
  auto grid = cg::this_grid();
  __shared__ size_t route_vector_offsets[kChunkTokens][8];
  __shared__ float route_weights[kChunkTokens][8];
  __shared__ int route_counts[kChunkTokens];

  if (phases & LOW_LATENCY_SEND_PHASE) {
    const int chunks = (capacity + kChunkTokens - 1) / kChunkTokens;
    for (int task = blockIdx.x; task < num_ranks * chunks; task += gridDim.x) {
      const int peer = (task % num_ranks + rank) % num_ranks;
      const int start = (task / num_ranks) * kChunkTokens;
      const int count = __ldg(launch.planned_counts + peer);
      EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
      const int end = min(start + kChunkTokens, count);
      // One full warp handles two tokens in eight-lane groups. Inactive
      // lanes still participate in the ballot, but never read a plan cell.
      // The group-local prefix preserves original-k accumulation order.
      if (threadIdx.x < 32) {
        const int chunk_token = lane / 8, original_k = lane % 8;
        const int slot = start + chunk_token;
        bool valid = false;
        size_t vector_offset = 0;
        float weight = 0.f;
        if (chunk_token < kChunkTokens && slot < end) {
          const int token = __ldg(launch.planned_tokens + size_t(peer) * capacity + slot);
          EP_DEVICE_ASSERT(token >= 0 && token < capacity);
          const size_t cell = (size_t(peer) * capacity + token) * 8 + original_k;
          const int64_t row = __ldg(launch.planned_rows + cell);
          if (row >= 0) {
            EP_DEVICE_ASSERT(size_t(row) < size_t(num_experts) * capacity);
            valid = true;
            vector_offset = size_t(row) * kPayloadVectors;
            weight = __ldg(launch.planned_owner_weights + cell);
          }
        }
        const unsigned valid_lanes = __ballot_sync(0xffffffff, valid);
        if (chunk_token < kChunkTokens) {
          const unsigned group_mask = (valid_lanes >> (chunk_token * 8)) & 0xffu;
          if (valid) {
            const int compact_k = __popc(group_mask & ((1u << original_k) - 1));
            route_vector_offsets[chunk_token][compact_k] = vector_offset;
            route_weights[chunk_token][compact_k] = weight;
          }
          if (original_k == 0) {
            const int routes = __popc(group_mask);
            if (slot < end) EP_DEVICE_ASSERT(routes > 0);
            route_counts[chunk_token] = routes;
          }
        }
      }
      __syncthreads();
      // Self partials have the same [rank, slot] destination as the original
      // self put. Store there directly with all producer threads; remote
      // owners use the unchanged registered staging and put path below.
      auto* partial_output = reinterpret_cast<int4*>(peer == rank ? rdma_recv_x : rdma_send_x);
      for (size_t item = threadIdx.x; start < end && item < size_t(end - start) * kPayloadVectors;
           item += blockDim.x) {
        const int chunk_token = item / kPayloadVectors, vector = item % kPayloadVectors;
        const int slot = start + chunk_token;
        float sum[kVecElements] = {0.f};
        for (int k = 0; k < route_counts[chunk_token]; ++k) {
          const int4 value = __ldg(reinterpret_cast<const int4*>(x) +
                                   route_vector_offsets[chunk_token][k] + vector);
          const float weight = route_weights[chunk_token][k];
          for (int j = 0; j < kVecElements; ++j) {
            const float leaf = float(reinterpret_cast<const nv_bfloat16*>(&value)[j]);
            const nv_bfloat16 weighted = static_cast<nv_bfloat16>(__fmul_rn(leaf, weight));
            sum[j] = __fadd_rn(sum[j], float(weighted));
          }
        }
        int4 partial;
        for (int j = 0; j < kVecElements; ++j)
          reinterpret_cast<nv_bfloat16*>(&partial)[j] = static_cast<nv_bfloat16>(sum[j]);
        partial_output[(size_t(peer) * capacity + slot) * kPayloadVectors + vector] = partial;
      }
      __syncthreads();
      if (threadIdx.x < 32 && start < end && peer != rank) {
        auto* dst = reinterpret_cast<int4*>(rdma_recv_x) +
                    (size_t(rank) * capacity + start) * kPayloadVectors;
        const auto* src = reinterpret_cast<const int4*>(rdma_send_x) +
                          (size_t(peer) * capacity + start) * kPayloadVectors;
        gin_put_warp(launch, dst, src, size_t(end - start) * kHidden * 2, peer, rank);
      }
      __syncthreads();
    }
    grid.sync();
    for (int peer = warp; peer < num_ranks; peer += warps) {
      if (lane == 0 && peer != rank) {
        gin_context(launch, rank)
            .signal(ncclTeamWorld(launch.comm), peer,
                    ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(signal_base + rank)});
        if (!defer_source_completion) gin_finish_peer(launch, peer, rank);
      }
      __syncwarp();
    }
    // Fused launches preserve every strong signal but finish source reads in
    // the tail CTA below, concurrently with final reduction. Send-only hooks
    // retain inline completion before the same-stream receive launch.
  }
  if (!(phases & LOW_LATENCY_RECV_PHASE)) return;
  for (int owner = warp; owner < num_ranks; owner += warps)
    if (lane == 0 && owner != rank) partial_wait_epoch_absolute(launch, owner, signal_base + owner);
  grid.sync();

  if (defer_source_completion && int(blockIdx.x) == reduce_ctas) {
    // Every payload/signal post precedes the original joins above. NCCL
    // 2.30.7 GDAKI flush snapshots the same persistent QP-global reserved tail;
    // waiting here releases all registered staging before kernel completion.
    // There is no later grid barrier: kernel completion joins these waiters
    // with every output writer, including for empty/all-negative inputs.
    for (int peer = threadIdx.x / 32; peer < num_ranks; peer += blockDim.x / 32) {
      if (lane == 0 && peer != rank) gin_finish_peer(launch, peer, rank);
      __syncwarp();
    }
    return;
  }
  // Drop only the dedicated completion CTA from the work stride. The start
  // ordinal remains unchanged for all remaining CTAs, preserving full warps.
  const int reduce_threads = reduce_ctas * blockDim.x;
  for (size_t item = thread; item < size_t(num_tokens) * kPayloadVectors; item += reduce_threads) {
    const int token = item / kPayloadVectors, vector = item % kPayloadVectors;
    const int expert =
        lane < num_topk ? __ldg(launch.planned_source_ids + size_t(token) * num_topk + lane) : -1;
    if (expert >= 0) EP_DEVICE_ASSERT(expert < num_experts);
    const int owner = expert >= 0 ? expert / local_experts : -1;
    const unsigned owner_lanes = __match_any_sync(0xffffffff, owner);
    const bool is_leader = owner >= 0 && lane == __ffs(owner_lanes) - 1;
    unsigned leaders = __ballot_sync(0xffffffff, is_leader);
    uint64_t source_vector_base = 0;
    if (is_leader) {
      const int slot = __ldg(launch.planned_source_slots + size_t(owner) * capacity + token);
      EP_DEVICE_ASSERT(slot >= 0 && slot < capacity);
      source_vector_base = (size_t(owner) * capacity + slot) * kPayloadVectors;
    }
    float sum[kVecElements] = {0.f};
    while (leaders) {
      const int leader = __ffs(leaders) - 1;
      const uint32_t low =
          __shfl_sync(0xffffffff, static_cast<uint32_t>(source_vector_base), leader);
      const uint32_t high =
          __shfl_sync(0xffffffff, static_cast<uint32_t>(source_vector_base >> 32), leader);
      const uint64_t base = (uint64_t(high) << 32) | low;
      const int4 value = ld_nc_global(reinterpret_cast<const int4*>(rdma_recv_x) + base + vector);
      for (int j = 0; j < kVecElements; ++j)
        sum[j] = __fadd_rn(sum[j], float(reinterpret_cast<const nv_bfloat16*>(&value)[j]));
      leaders &= leaders - 1;
    }
    int4 output;
    for (int j = 0; j < kVecElements; ++j)
      reinterpret_cast<nv_bfloat16*>(&output)[j] = static_cast<nv_bfloat16>(sum[j]);
    reinterpret_cast<int4*>(combined_x)[item] = output;
  }
}
