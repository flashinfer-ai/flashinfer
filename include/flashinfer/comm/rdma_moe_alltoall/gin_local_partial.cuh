/*
 * Experimental explicit BF16 local-partial reduction. Derived infrastructure
 * from DeepEP (MIT); see LICENSE.deepep.txt. Default FP32 combine is unchanged.
 */
#pragma once

// Included inside deep_ep::extensions. One metadata row holds all eight picks;
// the message header carries actual token and top-k counts, including zero T.
struct alignas(16) PartialRoute {
  int experts[8];
  float weights[8];
};
static_assert(sizeof(PartialRoute) == 64, "fixed metadata record");

// Every thread enters. Ascending-token compact slots are independently
// reproducible at the owner and source, so partial payloads need no token header.
__device__ void partial_build_slots(const PartialRoute* routes, int tokens, int topk, int owner,
                                    int local_experts, int* slots, int* warp_totals) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  int carry = 0;
  for (int base = 0; base < tokens; base += blockDim.x) {
    const int token = base + threadIdx.x;
    bool selected = false;
    if (token < tokens)
      for (int k = 0; k < topk; ++k) {
        const int expert = ld_nc_global(&routes[token].experts[k]);
        selected |= expert >= 0 && expert / local_experts == owner;
      }
    const unsigned mask = __ballot_sync(0xffffffff, selected);
    if (lane == 0) warp_totals[warp] = __popc(mask);
    __syncthreads();
    int prefix = __popc(mask & ((1u << lane) - 1));
    int total = 0;
    for (int w = 0; w < blockDim.x / 32; ++w) {
      if (w < warp) prefix += warp_totals[w];
      total += warp_totals[w];
    }
    if (token < tokens) slots[token] = selected ? carry + prefix : -1;
    carry += total;
    __syncthreads();
  }
}

// Only the signal namespace varies between metadata and partial traffic. Keep
// the original launch descriptor in parameter storage instead of copying it.
__device__ __forceinline__ void partial_wait_epoch_absolute(const GinLaunch& launch, int context,
                                                            int index) {
  const uint64_t expected = launch.epochs[index] + 1;
  gin_context(launch, context).waitSignal(ncclCoopThread{}, index, expected);
  launch.epochs[index] = expected;
}

template <int kHidden>
__global__ __launch_bounds__(256, 1) void combine_bf16_local_partial(
    void* combined_x, void* rdma_recv_x, void* rdma_send_x, const void* x, const int* topk_idx,
    const float* topk_weights, const int* src_info, const int64_t* layout_range, int num_tokens,
    int capacity, int num_topk, int num_experts, int rank, int num_ranks, int phases,
    GinLaunch launch) {
  constexpr int kVecElements = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kPayloadVectors = kHidden / kVecElements;
  static_assert(kPayloadVectors % 32 == 0, "final reduction requires complete token warps");
  constexpr int kChunkTokens = 8;
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  const int lane = threadIdx.x % 32;
  const int warp = thread / 32;
  const int warps = threads / 32;
  const int local_experts = num_experts / num_ranks;
  auto grid = cg::this_grid();
  const int meta_signal_base = launch.partial_signal_base;
  const int partial_signal_base = launch.partial_signal_base + num_ranks;
  auto* send_meta = static_cast<uint8_t*>(launch.partial_meta_send);
  auto* own_routes = reinterpret_cast<PartialRoute*>(send_meta + 16);
  auto* recv_meta = static_cast<uint8_t*>(launch.partial_meta_recv);
  auto* lookup = launch.partial_lookup;
  auto* owner_slots = lookup + size_t(num_experts) * capacity;
  auto* source_slots = owner_slots + size_t(num_ranks) * capacity;
  __shared__ int warp_totals[8];
  __shared__ int first_slot, selected_rows;
  // Resolve each token's ordered local routes once per chunk, rather than once
  // per hidden vector. All entries are consumed after the existing block join.
  __shared__ size_t route_vector_offsets[kChunkTokens][8];
  __shared__ float route_weights[kChunkTokens][8];
  __shared__ int route_counts[kChunkTokens];

  if (phases & LOW_LATENCY_SEND_PHASE) {
    // Scratch is local, per receive slot. A current route absent from the
    // original dispatch must fail instead of reading an uninitialized row.
    for (size_t i = thread; i < size_t(num_experts) * capacity; i += threads) lookup[i] = -1;
    if (thread == 0) *reinterpret_cast<int4*>(send_meta) = make_int4(num_tokens, num_topk, 0, 0);
    for (int token = thread; token < num_tokens; token += threads)
      for (int k = 0; k < 8; ++k) {
        own_routes[token].experts[k] = k < num_topk ? __ldg(topk_idx + token * num_topk + k) : -1;
        own_routes[token].weights[k] =
            k < num_topk ? __ldg(topk_weights + token * num_topk + k) : 0.f;
      }
    grid.sync();

    // The message is shared and immutable until every peer's source-completion
    // wait has finished. Metadata traffic is part of this combine kernel.
    for (int peer = warp; peer < num_ranks; peer += warps) {
      auto* dst = recv_meta + size_t(rank) * launch.partial_meta_stride;
      gin_put_warp(launch, dst, send_meta, 16 + size_t(num_tokens) * 64, peer, 0);
      if (lane == 0 && peer != rank) {
        gin_context(launch, 0).signal(
            ncclTeamWorld(launch.comm), peer,
            ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(meta_signal_base + rank)});
        gin_finish_peer(launch, peer, 0);
      }
      __syncwarp();
    }
    // Dispatch preserves each source segment's original compact row order.
    // Build only a local token -> expert-row map; public handle remains five fields.
    for (int pair = warp; pair < num_experts; pair += warps) {
      const int expert = pair / num_ranks, source = pair % num_ranks;
      int count, begin;
      unpack2(__ldg(layout_range + expert * num_ranks + source), count, begin);
      EP_DEVICE_ASSERT(count >= 0 && count <= capacity);
      EP_DEVICE_ASSERT(begin >= 0 && begin + count <= num_ranks * capacity);
      for (int i = lane; i < count; i += 32) {
        const int row = begin + i;
        const int token = __ldg(src_info + size_t(expert) * num_ranks * capacity + row);
        EP_DEVICE_ASSERT(token >= 0 && token < capacity);
        lookup[(size_t(expert) * num_ranks + source) * capacity + token] = row;
      }
    }
    for (int source = warp; source < num_ranks; source += warps)
      if (lane == 0 && source != rank)
        partial_wait_epoch_absolute(launch, 0, meta_signal_base + source);
    grid.sync();

    // First half plans compact partials produced here for each source. Second
    // half plans the same slots expected from each owner for this rank's tokens.
    for (int task = blockIdx.x; task < 2 * num_ranks; task += gridDim.x) {
      const bool producing = task < num_ranks;
      const int peer = task % num_ranks;
      const auto* message =
          producing ? recv_meta + size_t(peer) * launch.partial_meta_stride : send_meta;
      const int4 header = ld_nc_global(reinterpret_cast<const int4*>(message));
      EP_DEVICE_ASSERT(header.x >= 0 && header.x <= capacity && header.y >= 1 && header.y <= 8);
      partial_build_slots(reinterpret_cast<const PartialRoute*>(message + 16), header.x, header.y,
                          producing ? rank : peer, local_experts,
                          (producing ? owner_slots : source_slots) + size_t(peer) * capacity,
                          warp_totals);
    }
    grid.sync();

    const int chunks = (capacity + kChunkTokens - 1) / kChunkTokens;
    for (int task = blockIdx.x; task < num_ranks * chunks; task += gridDim.x) {
      // Interleave source peers within each chunk; rotate their visitation by
      // owner rank so all ranks do not begin with the same destination peer.
      const int peer = (task % num_ranks + rank) % num_ranks;
      const int start = (task / num_ranks) * kChunkTokens;
      const auto* message = recv_meta + size_t(peer) * launch.partial_meta_stride;
      const int4 header = ld_nc_global(reinterpret_cast<const int4*>(message));
      const auto* routes = reinterpret_cast<const PartialRoute*>(message + 16);
      const int end = min(start + kChunkTokens, header.x);
      if (threadIdx.x == 0) {
        first_slot = -1;
        selected_rows = 0;
        for (int token = start; token < end; ++token) {
          const int slot = owner_slots[size_t(peer) * capacity + token];
          if (slot >= 0) {
            if (first_slot < 0) first_slot = slot;
            ++selected_rows;
          }
        }
      }
      if (threadIdx.x < kChunkTokens) {
        const int chunk_token = threadIdx.x;
        const int token = start + chunk_token;
        int count = 0;
        if (token < end) {
          for (int k = 0; k < header.y; ++k) {
            const int expert = ld_nc_global(&routes[token].experts[k]);
            if (expert < 0 || expert / local_experts != rank) continue;
            EP_DEVICE_ASSERT(expert < num_experts);
            const int local = expert % local_experts;
            const int row = lookup[(size_t(local) * num_ranks + peer) * capacity + token];
            EP_DEVICE_ASSERT(row >= 0 && row < num_ranks * capacity);
            route_vector_offsets[chunk_token][count] =
                (size_t(local) * num_ranks * capacity + row) * kPayloadVectors;
            route_weights[chunk_token][count] = ld_nc_global(&routes[token].weights[k]);
            ++count;
          }
        }
        route_counts[chunk_token] = count;
      }
      __syncthreads();
      for (size_t item = threadIdx.x; start < end && item < size_t(end - start) * kPayloadVectors;
           item += blockDim.x) {
        const int token = start + item / kPayloadVectors;
        const int vector = item % kPayloadVectors;
        const int slot = owner_slots[size_t(peer) * capacity + token];
        if (slot < 0) continue;
        float sum[kVecElements] = {0.f};
        const int chunk_token = token - start;
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
        reinterpret_cast<int4*>(
            rdma_send_x)[(size_t(peer) * capacity + slot) * kPayloadVectors + vector] = partial;
      }
      __syncthreads();
      if (threadIdx.x < 32 && selected_rows) {
        auto* dst = reinterpret_cast<int4*>(rdma_recv_x) +
                    (size_t(rank) * capacity + first_slot) * kPayloadVectors;
        const auto* src = reinterpret_cast<const int4*>(rdma_send_x) +
                          (size_t(peer) * capacity + first_slot) * kPayloadVectors;
        gin_put_warp(launch, dst, src, size_t(selected_rows) * kHidden * 2, peer, rank);
      }
      __syncthreads();
    }
    // All producers use owner-rank % active-contexts for each destination QP.
    // Join before the unique signal/flush owner snapshots its reserved tail.
    grid.sync();
    for (int peer = warp; peer < num_ranks; peer += warps) {
      if (lane == 0 && peer != rank) {
        gin_context(launch, rank)
            .signal(
                ncclTeamWorld(launch.comm), peer,
                ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(partial_signal_base + rank)});
        gin_finish_peer(launch, peer, rank);
      }
      __syncwarp();
    }
    grid.sync();
  }
  if (!(phases & LOW_LATENCY_RECV_PHASE)) return;
  for (int owner = warp; owner < num_ranks; owner += warps)
    if (lane == 0 && owner != rank)
      partial_wait_epoch_absolute(launch, owner, partial_signal_base + owner);
  grid.sync();

  // Each full warp processes one token: vectors/token and the grid stride
  // are multiples of 32, so both the loop predicate and token are warp-uniform,
  // including the final iteration. Parse one route per lane, then visit owner
  // leaders from the lowest route lane to preserve first-owner appearance order.
  for (size_t item = thread; item < size_t(num_tokens) * kPayloadVectors; item += threads) {
    const int token = item / kPayloadVectors, vector = item % kPayloadVectors;
    const int expert = lane < num_topk ? __ldg(topk_idx + token * num_topk + lane) : -1;
    if (expert >= 0) EP_DEVICE_ASSERT(expert < num_experts);
    const int owner = expert >= 0 ? expert / local_experts : -1;
    const unsigned owner_lanes = __match_any_sync(0xffffffff, owner);
    const bool is_leader = owner >= 0 && lane == __ffs(owner_lanes) - 1;
    unsigned leaders = __ballot_sync(0xffffffff, is_leader);
    uint64_t source_vector_base = 0;
    if (is_leader) {
      const int slot = source_slots[size_t(owner) * capacity + token];
      EP_DEVICE_ASSERT(slot >= 0 && slot < capacity);
      source_vector_base = (size_t(owner) * capacity + slot) * kPayloadVectors;
    }
    float sum[kVecElements] = {0.f};
    while (leaders) {
      const int leader = __ffs(leaders) - 1;
      const uint32_t base_low =
          __shfl_sync(0xffffffff, static_cast<uint32_t>(source_vector_base), leader);
      const uint32_t base_high =
          __shfl_sync(0xffffffff, static_cast<uint32_t>(source_vector_base >> 32), leader);
      const uint64_t base = (uint64_t(base_high) << 32) | base_low;
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
