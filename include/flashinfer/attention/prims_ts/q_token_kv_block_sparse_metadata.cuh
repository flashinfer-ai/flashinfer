/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLASHINFER_ATTENTION_PRIMS_TS_Q_TOKEN_KV_BLOCK_SPARSE_METADATA_CUH_
#define FLASHINFER_ATTENTION_PRIMS_TS_Q_TOKEN_KV_BLOCK_SPARSE_METADATA_CUH_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <flashinfer/fastdiv.cuh>
#include <type_traits>
#include <unordered_map>

namespace flashinfer {
namespace attention {
namespace prims_ts {

constexpr int kQTokenKvBlockSparseMaxBlockTopK = 512;
constexpr int kQTokenKvBlockSparseMembershipsPerWord = 4;
constexpr int kQTokenKvBlockSparseQ1BlockThreads = 256;
constexpr int kQTokenKvBlockSparseBitmapBlockThreads = 512;

template <int GroupSize>
struct QTokenKvBlockSparseTouchedMetadataKernelTraits {
  static_assert(GroupSize >= 3 && GroupSize <= 8);
  static constexpr int kMaxCandidates = GroupSize * (kQTokenKvBlockSparseMaxBlockTopK + 1);
  // Bound collective width and keep more keys local to each thread. This
  // reduces cross-warp work for large unions without a batch-specific recipe.
  static constexpr int kBlockThreads = 256;
  static constexpr int kItemsPerThread = (kMaxCandidates + kBlockThreads - 1) / kBlockThreads;
};

// Keep enough independent warps to fill a Blackwell wave when decode exposes
// only a few query groups.  The per-group traits keep schedule tuning separate
// from metadata semantics while covering G * (512 selected blocks + one
// causal-tail block).
template <>
struct QTokenKvBlockSparseTouchedMetadataKernelTraits<2> {
  static constexpr int kBlockThreads = 384;
  static constexpr int kItemsPerThread = 3;
};

template <typename PositionType>
struct QTokenKvBlockSparseTouchedMetadataParams {
  const int32_t* block_indices;
  const int32_t* block_table;
  const int32_t* token_to_request;
  const PositionType* query_positions;
  // Null for the fixed layout.  Packed routes use [groups + 1] row offsets.
  const int32_t* qo_indptr;

  int32_t* q_token_kv_block_sparse_page_indices;
  // Four uint8 query-membership masks are packed in each int32 word.
  int32_t* q_token_kv_block_sparse_page_memberships;
  int32_t* seq_lens;

  int64_t block_indices_row_stride;
  int64_t block_indices_column_stride;
  int64_t block_indices_head_stride;
  int64_t block_table_request_stride;
  int64_t block_table_page_stride;

  int32_t rows;
  int32_t groups;
  int32_t num_requests;
  int32_t page_table_width;
  int32_t block_topk;
  int32_t page_capacity;
  int32_t membership_words;
  int32_t max_seq_len_kv;
  int32_t model_block_bound;
  // G9-G32: the entry-map prefix in sparse blocks (the model bound) when the
  // launch takes the membership-entry map, else 0 (set by the launcher).
  int32_t entry_map_blocks;
  int32_t model_radix_end_bit;
  int32_t sparse_block_size;
  int32_t sparse_block_shift;
  int32_t fragment_size;
  int32_t fragments_per_block;

  uint_fastdiv candidates_per_query;
  uint_fastdiv subpages_per_storage_page;
  uint_fastdiv pattern_heads;
  int32_t locator_page_stride;
  bool release_pdl;
  // True: a complete current block competes inside top-k; append only a partial tail.
  // False: exclude the current block from top-k and always append its visible tokens.
  bool count_current_block_in_topk;
  // Nonzero for request-aligned fixed Q; query_positions then stores KV lengths.
  // Rows map to (request, query offset) with this precomputed divisor.
  uint_fastdiv fixed_query_len;
  // Optional longest-first dispatch order for persistent attention: [routes]
  // route IDs by descending KV128 tiles, written by the last CTA to finish,
  // and its completion counter (zero between launches). Null disables both.
  int32_t* route_order;
  int32_t* route_order_counter;
  // Shortest routes ordered last, longest first (two SM waves); the others
  // keep index order.
  int32_t route_order_tail;
};

namespace detail {

template <typename PositionType>
__device__ __forceinline__ int64_t
QueryPosition(const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params, int32_t row) {
  const uint32_t query_len = params.fixed_query_len;
  if (query_len > 0) {
    uint32_t request;
    uint32_t offset;
    params.fixed_query_len.divmod(static_cast<uint32_t>(row), request, offset);
    const int64_t length = params.query_positions[request];
    return length > 0 ? length - query_len + offset : -1;
  }
  return static_cast<int64_t>(params.query_positions[row]);
}

// Excluding the current block leaves it outside the selection budget even
// when its last token completes the block.
template <typename PositionType>
__device__ __forceinline__ int64_t SelectedBlockBound(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params, int64_t visible) {
  return (visible - (!params.count_current_block_in_topk && visible > 0)) >>
         params.sparse_block_shift;
}

struct QTokenKvBlockSparseMembershipSegment {
  uint32_t logical_block;
  uint32_t memberships;
};

// Inclusive segmented OR for keys that have already been radix sorted.  The
// operation is associative over this monotonic-key domain, which is the input
// contract consumed by CUB BlockScan.  It gives every run-end lane the exact
// membership OR without a serial walk, even when an input row has duplicates.
struct QTokenKvBlockSparseMembershipSegmentedOr {
  __device__ __forceinline__ QTokenKvBlockSparseMembershipSegment
  operator()(const QTokenKvBlockSparseMembershipSegment& left,
             const QTokenKvBlockSparseMembershipSegment& right) const {
    return {right.logical_block, left.logical_block == right.logical_block
                                     ? left.memberships | right.memberships
                                     : right.memberships};
  }
};

template <int BlockThreads, int ItemsPerThread>
using QTokenKvBlockSparseKeySort = cub::BlockRadixSort<uint32_t, BlockThreads, ItemsPerThread>;

template <int BlockThreads>
using QTokenKvBlockSparseSegmentScan =
    cub::BlockScan<QTokenKvBlockSparseMembershipSegment, BlockThreads>;

template <int BlockThreads>
using QTokenKvBlockSparseOutputRankScan = cub::BlockScan<int, BlockThreads>;

template <int BlockThreads, int ItemsPerThread>
union QTokenKvBlockSparseCollectiveTempStorage {
  typename QTokenKvBlockSparseKeySort<BlockThreads, ItemsPerThread>::TempStorage key_sort;
  typename QTokenKvBlockSparseSegmentScan<BlockThreads>::TempStorage membership_scan;
  typename QTokenKvBlockSparseOutputRankScan<BlockThreads>::TempStorage output_rank_scan;
};

struct QTokenKvBlockSparseRouteState {
  int32_t valid;
  int32_t request;
  int32_t first_row;
  int32_t query_count;
  int64_t first_position;
  int64_t last_position;
};

template <int BlockThreads, int ItemsPerThread>
struct QTokenKvBlockSparseTouchedMetadataSharedStorage {
  static constexpr int kSortCapacity = BlockThreads * ItemsPerThread;

  QTokenKvBlockSparseCollectiveTempStorage<BlockThreads, ItemsPerThread> temp;
  uint32_t sorted_logical_blocks[kSortCapacity];
  QTokenKvBlockSparseRouteState route;
  int32_t union_pages;
};

template <typename PositionType, int GroupSize, bool PackedQuery>
__device__ __forceinline__ void InitRoute(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params,
    QTokenKvBlockSparseRouteState* route) {
  // One lane validates each query, avoiding serial dependent global loads
  // on lane zero while preserving the complete route contract.
  if (threadIdx.x >= 32) {
    return;
  }

  int32_t first_row;
  int32_t row_end;
  bool valid;
  const uint32_t group = blockIdx.x / params.pattern_heads;
  if constexpr (PackedQuery) {
    first_row = params.qo_indptr[group];
    row_end = params.qo_indptr[group + 1];
    valid = first_row >= 0 && row_end > first_row && row_end <= params.rows &&
            row_end - first_row <= GroupSize;
  } else {
    first_row = static_cast<int32_t>(group) * GroupSize;
    row_end = first_row + GroupSize;
    valid = row_end <= params.rows;
  }

  const int32_t query_count = valid ? row_end - first_row : 0;
  const int32_t lane = threadIdx.x;
  int32_t lane_request = -1;
  int64_t lane_position = -1;
  if (lane < query_count) {
    lane_request =
        static_cast<uint32_t>(params.fixed_query_len) > 0
            ? static_cast<int32_t>(static_cast<uint32_t>(first_row + lane) / params.fixed_query_len)
            : params.token_to_request[first_row + lane];
    lane_position = QueryPosition(params, first_row + lane);
  }
  const int32_t request = __shfl_sync(0xffffffffu, lane_request, 0);
  const int64_t first_position = __shfl_sync(0xffffffffu, static_cast<long long>(lane_position), 0);
  const int64_t last_position = __shfl_sync(0xffffffffu, static_cast<long long>(lane_position),
                                            query_count > 0 ? query_count - 1 : 0);
  // Validate endpoints before subtracting or forming expected positions, so
  // malformed Int64 positions cannot cause signed overflow.
  valid = valid && request >= 0 && request < params.num_requests && first_position >= 0 &&
          first_position < params.max_seq_len_kv && last_position >= first_position &&
          last_position < params.max_seq_len_kv &&
          last_position - first_position == query_count - 1;
  const bool lane_valid = !valid || lane >= query_count ||
                          (lane_request == request && lane_position == first_position + lane);
  valid = valid && __all_sync(0xffffffffu, lane_valid);

  if (lane == 0) {
    route->valid = valid;
    route->request = request;
    route->first_row = first_row;
    route->query_count = valid ? query_count : 0;
    route->first_position = first_position;
    route->last_position = last_position;
  }
}

// Permit the dependent attention grid to launch. Correctness does not depend
// on this placement: the attention dependency wait covers completion and
// visibility of every metadata CTA's output stores. Each kernel calls it after
// its last output store, so attention CTAs overlap only this grid's tail rather
// than idling resident through the whole metadata grid; measured on GB300, the
// long residency slowed attention's later TMA issue and output stores under a
// cold L2 by about 5 us on single-wave decode shapes.
template <typename PositionType>
__device__ __forceinline__ void ReleasePdlDependents(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.release_pdl) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

// Buckets of the balanced route order: one per KV128 tile count. Unions of
// 127 or more tiles (16K+ tokens) share the longest bucket, so the order is
// longest first only up to that length; within a bucket it is arbitrary.
constexpr int kQTokenKvBlockSparseRouteOrderBins = 128;

// Shared-memory ints the route order needs for ``routes`` routes: bins, warp
// sums plus a total, and a one-bit-per-route tail mask.
__host__ __device__ constexpr int QTokenKvBlockSparseRouteOrderSmemInts(int routes) {
  return kQTokenKvBlockSparseRouteOrderBins + 33 + (routes + 31) / 32;
}

// Dynamic shared memory of a metadata launch of ``bytes`` that also writes
// the route order: room for the tail mask when that fits the per-CTA limit,
// else only bins and warp sums (WriteLongestFirstRouteOrder then orders every
// route longest first).
inline size_t QTokenKvBlockSparseRouteOrderLaunchBytes(size_t bytes, const int32_t* route_order,
                                                       int64_t routes, int max_bytes) {
  if (route_order == nullptr) return bytes;
  const size_t order_bytes =
      static_cast<size_t>(QTokenKvBlockSparseRouteOrderSmemInts(static_cast<int>(routes))) *
      sizeof(int);
  const size_t sort_bytes = (kQTokenKvBlockSparseRouteOrderBins + 33) * sizeof(int);
  return std::max(bytes, order_bytes <= static_cast<size_t>(max_bytes) ? order_bytes : sort_bytes);
}

// Called by every CTA after its outputs. The CTA whose completion RMW returns
// routes - 1 writes route_order and rearms the counter. Persistent attention
// takes routes in that order: the ``route_order_tail`` shortest routes come
// last, longest first, so the final waves drain evenly; all other routes keep
// their index order, so concurrently running routes stay close in the
// request (L2 reuse of their K/V). ``smem`` holds ``smem_ints`` ints the CTA
// no longer uses; with fewer than the tail mask needs, every route takes its
// longest-first slot. Slots within one tile bucket follow atomic arrival; the
// order is always a permutation of the routes.
template <typename PositionType>
__device__ __forceinline__ void WriteLongestFirstRouteOrder(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params, int* smem,
    int smem_ints) {
  constexpr int kBins = kQTokenKvBlockSparseRouteOrderBins;
  if (params.route_order == nullptr) return;
  const int routes = static_cast<int>(gridDim.x);
  int last = 0;
  if (threadIdx.x == 0) {
    // Release publishes this CTA's seq_lens; the last CTA acquires all others.
    cuda::atomic_ref<int, cuda::thread_scope_device> done(*params.route_order_counter);
    last = done.fetch_add(1, cuda::memory_order_acq_rel) == routes - 1;
  }
  if (!__syncthreads_or(last)) return;
  const bool keep_index_order = smem_ints >= QTokenKvBlockSparseRouteOrderSmemInts(routes);
  const int tail_start = keep_index_order ? routes - min(params.route_order_tail, routes) : 0;
  int* bins = smem;
  int* warp_sums = smem + kBins;
  unsigned* tail_mask = reinterpret_cast<unsigned*>(smem + kBins + 33);
  auto bin = [&](int route) {
    const int tiles = (__ldcg(params.seq_lens + route) + 127) >> 7;
    return kBins - 1 - min(tiles, kBins - 1);  // bin 0 holds the longest routes
  };
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) bins[b] = 0;
  if (keep_index_order) {
    for (int w = threadIdx.x; w < (routes + 31) / 32; w += blockDim.x) tail_mask[w] = 0;
  }
  __syncthreads();
  for (int r = threadIdx.x; r < routes; r += blockDim.x) atomicAdd(bins + bin(r), 1);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp == 0) {
    // Exclusive scan of the bin counts: each lane owns kBins / 32 bins.
    int counts[kBins / 32];
    int sum = 0;
#pragma unroll
    for (int i = 0; i < kBins / 32; ++i) {
      counts[i] = bins[lane * (kBins / 32) + i];
      sum += counts[i];
    }
    int inclusive = sum;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int lower = __shfl_up_sync(0xffffffffu, inclusive, d);
      if (lane >= d) inclusive += lower;
    }
    int start = inclusive - sum;
#pragma unroll
    for (int i = 0; i < kBins / 32; ++i) {
      bins[lane * (kBins / 32) + i] = start;
      start += counts[i];
    }
  }
  __syncthreads();
  // Longest-first slots; the tail keeps them, the rest is compacted below.
  for (int r = threadIdx.x; r < routes; r += blockDim.x) {
    const int slot = atomicAdd(bins + bin(r), 1);
    if (slot >= tail_start) {
      params.route_order[slot] = r;
      if (keep_index_order) atomicOr(tail_mask + (r >> 5), 1u << (r & 31));
    }
  }
  if (tail_start > 0) {
    // Stable compaction of the non-tail routes into [0, tail_start).
    __syncthreads();
    const int warps = static_cast<int>(blockDim.x) >> 5;
    int base = 0;
    for (int chunk = 0; chunk < routes; chunk += blockDim.x) {
      const int r = chunk + static_cast<int>(threadIdx.x);
      const bool keep = r < routes && !((tail_mask[r >> 5] >> (r & 31)) & 1u);
      const unsigned ballot = __ballot_sync(0xffffffffu, keep);
      if (lane == 0) warp_sums[warp] = __popc(ballot);
      __syncthreads();
      if (warp == 0) {
        const int count = lane < warps ? warp_sums[lane] : 0;
        int inclusive = count;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const int lower = __shfl_up_sync(0xffffffffu, inclusive, d);
          if (lane >= d) inclusive += lower;
        }
        if (lane < warps) warp_sums[lane] = inclusive - count;
        if (lane == 31) warp_sums[32] = inclusive;
      }
      __syncthreads();
      if (keep) {
        params.route_order[base + warp_sums[warp] + __popc(ballot & ((1u << lane) - 1u))] = r;
      }
      base += warp_sums[32];
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) *params.route_order_counter = 0;
}

template <typename PositionType>
__device__ __forceinline__ int32_t
ResolveFragment(const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params,
                const QTokenKvBlockSparseRouteState& route, uint32_t logical_fragment) {
  uint32_t storage_page;
  uint32_t subpage;
  params.subpages_per_storage_page.divmod(logical_fragment, storage_page, subpage);
  if (storage_page < static_cast<uint32_t>(params.page_table_width)) {
    const int32_t physical_page =
        params.block_table[static_cast<int64_t>(route.request) * params.block_table_request_stride +
                           static_cast<int64_t>(storage_page) * params.block_table_page_stride];
    if (physical_page >= 0) {
      return static_cast<int32_t>(static_cast<uint32_t>(physical_page) *
                                      static_cast<uint32_t>(params.locator_page_stride) +
                                  subpage);
    }
  }
  return -1;
}

template <typename PositionType>
__device__ __forceinline__ void StorePageMetadata(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params,
    const QTokenKvBlockSparseRouteState& route, uint32_t logical_block, uint8_t membership,
    int32_t output_rank, int32_t* group_indices, uint8_t* group_memberships) {
  // Sort semantic blocks once. Only split their output when a physical cache
  // boundary requires smaller TMA fragments; adjacent logical pages need not
  // be adjacent physical pages.
  for (int fragment = 0; fragment < params.fragments_per_block; ++fragment) {
    const int rank = output_rank * params.fragments_per_block + fragment;
    const uint32_t logical_fragment = logical_block * params.fragments_per_block + fragment;
    const int32_t locator =
        static_cast<int64_t>(logical_fragment) * params.fragment_size <= route.last_position
            ? ResolveFragment(params, route, logical_fragment)
            : -1;
    uint32_t fragment_membership = membership;
    if (params.fragments_per_block > 1) {
      const int64_t first_visible_query =
          static_cast<int64_t>(logical_fragment) * params.fragment_size - route.first_position;
      if (first_visible_query > 0) {
        fragment_membership &= first_visible_query < 8 ? (0xffu << first_visible_query) : 0u;
      }
    }
    group_indices[rank] = locator;
    group_memberships[rank] = locator >= 0 ? static_cast<uint8_t>(fragment_membership) : 0;
  }
}

template <typename PositionType, bool PackedQuery>
__global__
__launch_bounds__(kQTokenKvBlockSparseQ1BlockThreads) void QTokenKvBlockSparseQ1MetadataKernel(
    const __grid_constant__ QTokenKvBlockSparseTouchedMetadataParams<PositionType> params) {
  __shared__ QTokenKvBlockSparseRouteState route;
  if (threadIdx.x == 0) {
    route = {0, -1, 0, 0, -1, -1};
  }
  __syncthreads();
  InitRoute<PositionType, 1, PackedQuery>(params, &route);
  __syncthreads();

  int32_t* row_indices = params.q_token_kv_block_sparse_page_indices +
                         static_cast<int64_t>(blockIdx.x) * params.page_capacity;
  const int32_t visible = route.valid ? static_cast<int32_t>(route.first_position + 1) : 0;
  const int32_t causal_blocks = SelectedBlockBound(params, visible);
  const int32_t complete_blocks = min(causal_blocks, params.block_topk);
  const int32_t tail = visible - causal_blocks * params.sparse_block_size;
  const int32_t compact_length = complete_blocks * params.sparse_block_size + tail;
  const int32_t live_fragments = (compact_length + params.fragment_size - 1) / params.fragment_size;
  const uint32_t head = blockIdx.x % params.pattern_heads;

  // Initialize the complete capacity: rounded speculative page staging may
  // read padding before its token predicates apply.
  for (int rank = threadIdx.x; rank < params.page_capacity;
       rank += kQTokenKvBlockSparseQ1BlockThreads) {
    int32_t locator = -1;
    if (route.valid && rank < live_fragments) {
      const int block_rank = rank / params.fragments_per_block;
      int32_t logical = causal_blocks;
      if (block_rank < complete_blocks) {
        logical =
            params.block_indices[static_cast<int64_t>(route.first_row) *
                                     params.block_indices_row_stride +
                                 static_cast<int64_t>(head) * params.block_indices_head_stride +
                                 static_cast<int64_t>(block_rank) *
                                     params.block_indices_column_stride];
        if (logical < 0 || logical >= causal_blocks) logical = -1;
      }
      if (logical >= 0 && logical < params.model_block_bound) {
        locator = ResolveFragment(params, route,
                                  static_cast<uint32_t>(logical) * params.fragments_per_block +
                                      rank % params.fragments_per_block);
      }
    }
    row_indices[rank] = locator;
  }
  if (threadIdx.x == 0) {
    params.seq_lens[blockIdx.x] = max(compact_length, 1);
  }
  __syncthreads();
  ReleasePdlDependents(params);
}

// Call after publishing the live membership bytes with a CTA barrier.
// Both union algorithms emit the same eight-slot/32-fragment tile, including
// zero unused query slots/tail bits and a complete zero tile for an inert route.
template <typename PositionType, int BlockThreads, int GroupSize>
__device__ __forceinline__ void StoreQueryMajorMemberships(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params,
    const uint8_t* group_memberships, int union_pages) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int tiles = union_pages == 0 ? 1 : (union_pages + 31) / 32;
  uint32_t* query_words =
      reinterpret_cast<uint32_t*>(params.q_token_kv_block_sparse_page_memberships) +
      static_cast<int64_t>(blockIdx.x) * params.membership_words;
  for (int tile = warp; tile < tiles; tile += BlockThreads / 32) {
    const int page = tile * 32 + lane;
    const uint32_t member = page < union_pages ? group_memberships[page] : 0;
    uint32_t own_word = 0;
#pragma unroll
    for (int query = 0; query < GroupSize; ++query) {
      const uint32_t word = __ballot_sync(0xffffffffu, (member & (1u << query)) != 0);
      if (lane == query) own_word = word;
    }
    if (lane < 8) query_words[tile * 8 + lane] = own_word;
  }
  __syncthreads();
}

template <typename PositionType, int GroupSize, bool QueryMajor = false>
__device__ __forceinline__ int BuildTouchedUnion(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params,
    QTokenKvBlockSparseTouchedMetadataSharedStorage<
        QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kBlockThreads,
        QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kItemsPerThread>& shared,
    uint32_t active_logical_capacity, int active_radix_end_bit, uint32_t low_mask,
    int32_t* group_indices, uint8_t* group_memberships) {
  constexpr int kBlockThreads =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kBlockThreads;
  constexpr int kItemsPerThread =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kItemsPerThread;
  constexpr int kSortCapacity = kBlockThreads * kItemsPerThread;

  uint32_t encoded_keys[kItemsPerThread];
  // Gather every independent candidate before validating its value. This
  // preserves the causal read bounds while exposing memory-level parallelism
  // across each thread's items instead of consuming one load at a time.
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const uint32_t candidate_rank = item * kBlockThreads + threadIdx.x;
    uint32_t query;
    uint32_t query_item;
    params.candidates_per_query.divmod(candidate_rank, query, query_item);
    int32_t selected_block = -1;
    if (shared.route.valid && query < static_cast<uint32_t>(shared.route.query_count) &&
        candidate_rank < static_cast<uint32_t>(GroupSize * (params.block_topk + 1))) {
      const int64_t visible_tokens = shared.route.first_position + query + 1;
      const int64_t complete_block_count = SelectedBlockBound(params, visible_tokens);
      const int32_t selected_count = static_cast<int32_t>(
          complete_block_count < params.block_topk ? complete_block_count : params.block_topk);
      if (query_item < static_cast<uint32_t>(selected_count)) {
        const int32_t row = shared.route.first_row + query;
        selected_block =
            params.block_indices[static_cast<int64_t>(row) * params.block_indices_row_stride +
                                 static_cast<int64_t>(blockIdx.x % params.pattern_heads) *
                                     params.block_indices_head_stride +
                                 static_cast<int64_t>(query_item) *
                                     params.block_indices_column_stride];
      }
    }
    encoded_keys[item] = static_cast<uint32_t>(selected_block);
  }
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const uint32_t candidate_rank = item * kBlockThreads + threadIdx.x;
    uint32_t query;
    uint32_t query_item;
    params.candidates_per_query.divmod(candidate_rank, query, query_item);

    int64_t logical_block = -1;
    if (shared.route.valid && query < static_cast<uint32_t>(shared.route.query_count) &&
        candidate_rank < static_cast<uint32_t>(GroupSize * (params.block_topk + 1))) {
      const int64_t visible_tokens = shared.route.first_position + query + 1;
      const int64_t complete_block_count = SelectedBlockBound(params, visible_tokens);
      const int32_t selected_count = static_cast<int32_t>(
          complete_block_count < params.block_topk ? complete_block_count : params.block_topk);
      if (query_item < static_cast<uint32_t>(selected_count)) {
        const int32_t selected_block = static_cast<int32_t>(encoded_keys[item]);
        // Selected IDs may only name complete causal blocks. The partial
        // causal tail is synthesized separately and must remain the final
        // logical block for attention's tail-only mask.
        if (selected_block >= 0 && selected_block < complete_block_count &&
            selected_block < params.model_block_bound) {
          logical_block = selected_block;
        }
      } else if (query_item == static_cast<uint32_t>(params.block_topk) &&
                 (!params.count_current_block_in_topk ||
                  (visible_tokens & (params.sparse_block_size - 1)) != 0)) {
        logical_block = SelectedBlockBound(params, visible_tokens);
      }
    }

    const bool live =
        logical_block >= 0 && logical_block < static_cast<int64_t>(active_logical_capacity);
    encoded_keys[item] =
        live ? static_cast<uint32_t>(logical_block) | (query << params.model_radix_end_bit)
             : low_mask;
  }

  QTokenKvBlockSparseKeySort<kBlockThreads, kItemsPerThread>(shared.temp.key_sort)
      .Sort(encoded_keys, 0, active_radix_end_bit);

  QTokenKvBlockSparseMembershipSegment segments[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = threadIdx.x * kItemsPerThread + item;
    const uint32_t logical_block = encoded_keys[item] & low_mask;
    const bool live = logical_block < active_logical_capacity;
    const uint32_t query = encoded_keys[item] >> params.model_radix_end_bit;
    shared.sorted_logical_blocks[rank] = logical_block;
    segments[item] = {logical_block, live ? uint32_t{1} << query : 0};
  }
  __syncthreads();

  // Exact duplicate handling: the segmented scan propagates the membership
  // OR across each complete equal-key run in logarithmic collective depth.
  QTokenKvBlockSparseSegmentScan<kBlockThreads>(shared.temp.membership_scan)
      .InclusiveScan(segments, segments, QTokenKvBlockSparseMembershipSegmentedOr{});
  __syncthreads();

  int unique_flags[kItemsPerThread];
  int local_unique_count = 0;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = threadIdx.x * kItemsPerThread + item;
    const uint32_t logical_block = segments[item].logical_block;
    const bool unique_end =
        logical_block < active_logical_capacity &&
        (rank + 1 == kSortCapacity || shared.sorted_logical_blocks[rank + 1] != logical_block);
    unique_flags[item] = unique_end;
    local_unique_count += unique_end;
  }

  int thread_output_begin = 0;
  int union_pages = 0;
  QTokenKvBlockSparseOutputRankScan<kBlockThreads>(shared.temp.output_rank_scan)
      .ExclusiveSum(local_unique_count, thread_output_begin, union_pages);

  if constexpr (QueryMajor) {
    // Every neighbor-key read must finish before membership bytes reuse the
    // now-dead sorted-key storage. Do not depend on CUB's internal barriers.
    __syncthreads();
  }

  int local_output_rank = 0;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    if (unique_flags[item]) {
      StorePageMetadata(params, shared.route, segments[item].logical_block,
                        static_cast<uint8_t>(segments[item].memberships),
                        thread_output_begin + local_output_rank, group_indices, group_memberships);
      ++local_output_rank;
    }
  }
  return union_pages;
}

template <typename PositionType, int GroupSize, bool PackedQuery, bool QueryMajor = false>
__global__ __launch_bounds__(
    QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::
        kBlockThreads) void QTokenKvBlockSparseTouchedMetadataKernel(const __grid_constant__
                                                                         QTokenKvBlockSparseTouchedMetadataParams<
                                                                             PositionType>
                                                                             params) {
  constexpr int kBlockThreads =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kBlockThreads;
  constexpr int kItemsPerThread =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kItemsPerThread;
  constexpr int kSortCapacity = kBlockThreads * kItemsPerThread;
  constexpr int kMaximumCandidates = GroupSize * (kQTokenKvBlockSparseMaxBlockTopK + 1);
  static_assert(kSortCapacity >= kMaximumCandidates);
  static_assert(GroupSize <= 8, "query membership is stored in one byte");

  __shared__ QTokenKvBlockSparseTouchedMetadataSharedStorage<kBlockThreads, kItemsPerThread> shared;

  // Initialize CTA-local state before reading semantic inputs. This metadata
  // kernel has no producer dependency; it releases attention after its outputs.
  if (threadIdx.x == 0) {
    shared.route = {0, -1, 0, 0, -1, -1};
    shared.union_pages = 0;
  }
  __syncthreads();
  InitRoute<PositionType, GroupSize, PackedQuery>(params, &shared.route);
  __syncthreads();

  const int64_t causal_block_bound =
      shared.route.valid
          ? (shared.route.last_position + params.sparse_block_size) >> params.sparse_block_shift
          : 0;
  const uint32_t active_logical_capacity = static_cast<uint32_t>(
      causal_block_bound < params.model_block_bound ? causal_block_bound
                                                    : params.model_block_bound);
  int32_t* group_indices = params.q_token_kv_block_sparse_page_indices +
                           static_cast<int64_t>(blockIdx.x) * params.page_capacity;
  uint8_t* group_memberships =
      reinterpret_cast<uint8_t*>(params.q_token_kv_block_sparse_page_memberships +
                                 static_cast<int64_t>(blockIdx.x) * params.membership_words);
  if constexpr (QueryMajor) {
    static_assert(GroupSize >= 2 && GroupSize <= 8);
    static_assert(sizeof(shared.sorted_logical_blocks) >= kMaximumCandidates);
    // The private entry point requires one page-4 fragment per semantic block.
    // No extra shared allocation or intermediate GMEM membership write is needed.
    group_memberships = reinterpret_cast<uint8_t*>(shared.sorted_logical_blocks);
  }

  // bit_width(N) leaves an all-ones sentinel strictly above every live
  // [0, N) key, including when N is a power of two.
  const int active_radix_end_bit_unclamped =
      active_logical_capacity == 0 ? 1 : 32 - __clz(active_logical_capacity);
  const int active_radix_end_bit = active_radix_end_bit_unclamped < params.model_radix_end_bit
                                       ? active_radix_end_bit_unclamped
                                       : params.model_radix_end_bit;
  const uint32_t low_mask = (uint32_t{1} << active_radix_end_bit) - 1;
  const int union_pages = BuildTouchedUnion<PositionType, GroupSize, QueryMajor>(
      params, shared, active_logical_capacity, active_radix_end_bit, low_mask, group_indices,
      group_memberships);
  if (threadIdx.x == 0) {
    shared.union_pages = union_pages;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (shared.route.valid && shared.union_pages > 0) {
      const int tail_tokens =
          static_cast<int>((shared.route.last_position + 1) & (params.sparse_block_size - 1));
      const int tail_padding = tail_tokens == 0 ? 0 : params.sparse_block_size - tail_tokens;
      params.seq_lens[blockIdx.x] = shared.union_pages * params.sparse_block_size - tail_padding;
      // Attention reads packed Uint32 words. Initialize only the last word's
      // padding bytes so that its masked load never reads unwritten memory.
      for (int byte = shared.union_pages * params.fragments_per_block;
           byte % kQTokenKvBlockSparseMembershipsPerWord != 0; ++byte) {
        group_memberships[byte] = 0;
      }
    } else {
      // Attention requires one addressable sentinel entry for an inert route.
      group_indices[0] = -1;
      reinterpret_cast<uint32_t*>(group_memberships)[0] = 0;
      params.seq_lens[blockIdx.x] = 1;
    }
  }
  __syncthreads();

  if constexpr (QueryMajor) {
    StoreQueryMajorMemberships<PositionType, kBlockThreads, GroupSize>(params, group_memberships,
                                                                       shared.union_pages);
  }
  ReleasePdlDependents(params);
  static_assert(sizeof(shared) >= (kQTokenKvBlockSparseRouteOrderBins + 33) * sizeof(int));
  WriteLongestFirstRouteOrder(params, reinterpret_cast<int*>(&shared),
                              static_cast<int>(sizeof(shared) / sizeof(int)));
}

// ---------------------------------------------------------------------------
// Shared-memory map union for grouped routes.
//
// One CTA per route. The route's visible causal prefix of logical sparse
// blocks is mapped in dynamic shared memory sized per launch from the model
// bound, so the union is formed by scattering at most G * (topk + 1)
// candidates with shared-memory atomics, and the ascending compact list falls
// out of a single block-wide prefix sum over per-word counts. Two granularities
// share the same buffer and are chosen per route from its causal prefix:
//
//   * byte map (prefix <= kQTokenKvBlockSparseByteMapMaxBlocks): one byte per
//     block holds the OR of the query-membership bits, so the compaction pass
//     emits both the block and its membership directly. Four blocks per word
//     keep the scatter atomics nearly conflict-free.
//   * bit map (longer prefixes): one bit per block (1M tokens = 32 KiB) plus a
//     per-thread summary bit per map word, so the count, compaction, and rank
//     passes touch only the (at most G * (topk + 1)) non-empty words; each
//     candidate then recovers its block's compact rank from the prefix
//     (popcount of the preceding non-empty words and lower bits) and ORs its
//     query bit into the compact entry's membership byte.
//
// Per-thread word ranges are contiguous (so the compact list stays ascending)
// but stored with an odd padded stride, keeping the clear/count/compaction
// loops free of shared-memory bank conflicts at large maps. This replaces the
// three-pass block radix sort and two collective scans of the bounded sort
// path with work proportional to the visible logical prefix; models whose map
// would not fit the device's opt-in dynamic shared memory (about 6M tokens on
// SM100) keep the sort path.
// ---------------------------------------------------------------------------
// Longest causal prefix (in sparse blocks) mapped at byte granularity.
constexpr int kQTokenKvBlockSparseByteMapMaxBlocks = 32768;
// Bit-map summary capacity: up to 128 map words per thread (the opt-in
// shared-memory limit keeps every supported model well below this).
constexpr int kQTokenKvBlockSparseBitmapSummaryWordsPerThread = 4;
constexpr int kQTokenKvBlockSparseBitmapMaxWordsPerThread =
    kQTokenKvBlockSparseBitmapSummaryWordsPerThread * 32;

// Threads of the bit-map kernel. G32 routes carry 32 x 513 candidates and
// their compact union already limits the kernel to one CTA per SM, so 1024
// threads halve the per-thread candidate loop for free (GB300: 57 -> 32 us at
// 4K queries). G <= 16 keep 512 threads and two CTAs per SM.
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseBitmapThreads() {
  return GroupSize > 16 ? 1024 : kQTokenKvBlockSparseBitmapBlockThreads;
}

// Minimum resident bit-map CTAs per SM the compiler must allow. G <= 4 routes
// keep four 512-thread CTAs (at most 32 registers): at three, 1K routes need a
// third wave (GB300, G4, 4K queries: 14.6 -> 24.6 us). Zero leaves larger
// groups to the compiler's default (an explicit 1 raised G8 from 40 to 56
// registers, one CTA fewer per SM).
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseBitmapMinBlocks() {
  return GroupSize <= 4 ? 4 : 0;
}

template <int GroupSize>
struct QTokenKvBlockSparseBitmapSharedStorage {
  static constexpr int kMaxUnionPages = GroupSize * (kQTokenKvBlockSparseMaxBlockTopK + 1);
  static constexpr int kThreads = QTokenKvBlockSparseBitmapThreads<GroupSize>();

  typename cub::BlockScan<int, kThreads>::TempStorage scan;
  // Ascending compact union: bits [0, 24) logical block, bits [24, 32) query
  // membership (written directly by the byte map, OR-reduced by the bit map).
  uint32_t compact_union[kMaxUnionPages];
  // G9-G32: membership bits of queries 8..31 per compact entry (planes 1-3).
  uint32_t compact_membership_hi[GroupSize > 8 ? kMaxUnionPages : 1];
  // Compact rank of the first entry in each thread's contiguous word range.
  int32_t thread_rank_begin[kThreads];
  QTokenKvBlockSparseRouteState route;
  int32_t union_pages;
  // Per query of the route: complete selectable blocks, and the partial
  // causal tail block (or -1), computed once before the scatter.
  int32_t query_complete_blocks[GroupSize];
  int32_t query_tail_block[GroupSize];
};

// Reserve the larger of the full bit map and the byte map capped at the
// byte-granularity prefix limit. Runtime prefixes may use either layout;
// `bit_only` (G9-G32, which never take the byte map) reserves the bit map.
__host__ __device__ constexpr int QTokenKvBlockSparseMapWords(int32_t blocks,
                                                              bool bit_only = false) {
  const int bit_words = (blocks + 31) / 32;
  if (bit_only) {
    return bit_words;
  }
  const int byte_blocks =
      blocks < kQTokenKvBlockSparseByteMapMaxBlocks ? blocks : kQTokenKvBlockSparseByteMapMaxBlocks;
  const int byte_words = (byte_blocks + 3) / 4;
  return bit_words > byte_words ? bit_words : byte_words;
}

// Membership-entry map: one entry of EntryBits query bits per logical block
// (a byte for G <= 8, 16 bits for G9-G16, a word for G17-G32), so the scatter
// records query membership directly and the compaction emits it. G <= 8 caps
// it at the byte-map prefix limit (longer routes take the bit map); G9-G32
// take it for a whole launch when the model's prefix fits without lowering
// occupancy (else the bit map with its membership pass).
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseEntryBits() {
  return GroupSize <= 8 ? 8 : GroupSize <= 16 ? 16 : 32;
}
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseEntryShift() {
  return GroupSize <= 8 ? 2 : GroupSize <= 16 ? 1 : 0;  // log2(entries per word)
}
// Map words reserved for a model prefix whose entry-map routes stop at
// `entry_blocks` (the bit map covers every longer route).
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseEntryMapWords(int32_t blocks,
                                                                   int32_t entry_blocks) {
  const int bit_words = (blocks + 31) / 32;
  const int capped = blocks < entry_blocks ? blocks : entry_blocks;
  constexpr int kEntries = 1 << QTokenKvBlockSparseEntryShift<GroupSize>();
  const int entry_words = capped > 0 ? (capped + kEntries - 1) / kEntries : 0;
  return bit_words > entry_words ? bit_words : entry_words;
}
// Longest G9-G32 model prefix considered for the entry map (G16: 64 KiB,
// G32: 128 KiB maps; the launcher also requires unchanged occupancy).
constexpr int kQTokenKvBlockSparseWideEntryMapMaxBlocks = 32768;

// Words each thread owns for a map of `map_words`, and the padded storage
// stride between consecutive threads' ranges (odd, so lanes touch distinct
// banks when they walk their ranges in lockstep).
__host__ __device__ constexpr int QTokenKvBlockSparseWordsPerThread(
    int map_words, int threads = kQTokenKvBlockSparseBitmapBlockThreads) {
  return (map_words + threads - 1) / threads;
}
__host__ __device__ constexpr int QTokenKvBlockSparseMapStride(int words_per_thread) {
  return words_per_thread + ((words_per_thread & 1) == 0 ? 1 : 0);
}

template <int GroupSize>
__host__ __device__ constexpr size_t QTokenKvBlockSparseBitmapFixedSmemBytes() {
  return (sizeof(QTokenKvBlockSparseBitmapSharedStorage<GroupSize>) + 15) / 16 * 16;
}

// Entry-map prefix limit of a launch: fixed for G <= 8 (byte map), chosen by
// the launcher for G9-G32 (`params.entry_map_blocks`, 0 = bit map only).
template <int GroupSize>
__host__ __device__ constexpr int32_t QTokenKvBlockSparseEntryMapBlocks(int32_t wide_entry_blocks) {
  return GroupSize <= 8 ? kQTokenKvBlockSparseByteMapMaxBlocks : wide_entry_blocks;
}

// Padded map stride sized for the model's full logical prefix.
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseBitmapModelStride(int32_t model_block_bound,
                                                                       int32_t entry_blocks) {
  return QTokenKvBlockSparseMapStride(QTokenKvBlockSparseWordsPerThread(
      QTokenKvBlockSparseEntryMapWords<GroupSize>(model_block_bound, entry_blocks),
      QTokenKvBlockSparseBitmapThreads<GroupSize>()));
}

// Padded stride of the bit map alone (layout of the word-rank array).
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseBitmapBitStride(int32_t model_block_bound) {
  return QTokenKvBlockSparseMapStride(
      QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(model_block_bound, true),
                                        QTokenKvBlockSparseBitmapThreads<GroupSize>()));
}

// Summary words per thread: one bit per bit-map word of the thread's range.
template <int GroupSize>
__host__ __device__ constexpr int QTokenKvBlockSparseBitmapSummaryWords(int32_t model_block_bound) {
  const int bit_words_per_thread =
      QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(model_block_bound, true),
                                        QTokenKvBlockSparseBitmapThreads<GroupSize>());
  return bit_words_per_thread <= 32 ? 1 : (bit_words_per_thread + 31) / 32;
}

// Dynamic shared memory of the map kernel: the fixed storage followed by the
// padded map sized for the model's full logical prefix.
template <int GroupSize>
__host__ __device__ constexpr size_t QTokenKvBlockSparseBitmapSmemBytes(int32_t model_block_bound,
                                                                        bool with_word_rank,
                                                                        int32_t entry_blocks) {
  constexpr int kThreads = QTokenKvBlockSparseBitmapThreads<GroupSize>();
  const size_t map_bytes =
      static_cast<size_t>(kThreads) *
      QTokenKvBlockSparseBitmapModelStride<GroupSize>(model_block_bound, entry_blocks) *
      sizeof(uint32_t);
  // Bit-map path only (prefixes past the entry-map limit): one summary bit per
  // map word of each thread's range marking words that received a candidate,
  // so the count, compaction, and rank passes visit only non-empty words.
  const bool bit_map_path = model_block_bound > entry_blocks;
  const size_t summary_bytes =
      bit_map_path ? static_cast<size_t>(kThreads) *
                         QTokenKvBlockSparseBitmapSummaryWords<GroupSize>(model_block_bound) *
                         sizeof(uint32_t)
                   : 0;
  // Optional, bit-map path only: the first compact rank of each non-empty
  // bit-map word (16 bits; ranks stay below G * (top-k + 1)), laid out like the
  // bit map, for O(1) rank lookups in the membership pass.
  const size_t word_rank_bytes =
      bit_map_path && with_word_rank
          ? static_cast<size_t>(kThreads) *
                QTokenKvBlockSparseBitmapBitStride<GroupSize>(model_block_bound) * sizeof(uint16_t)
          : 0;
  return QTokenKvBlockSparseBitmapFixedSmemBytes<GroupSize>() + (map_bytes + 15) / 16 * 16 +
         (summary_bytes + 15) / 16 * 16 + (word_rank_bytes + 15) / 16 * 16;
}

// Map-kernel footprint for a model prefix: with the word-rank array unless it
// does not fit or lowers the CTAs per SM (the summary-walk instantiation then
// runs). Returns zero bytes when neither footprint fits.
struct QTokenKvBlockSparseBitmapFootprint {
  size_t bytes = 0;
  int ctas_per_sm = 0;
  bool word_rank = true;
  int32_t entry_blocks = 0;  // G9-G32: entry-map launch (params.entry_map_blocks), else 0
};

template <int GroupSize, typename Kernel>
cudaError_t SelectQTokenKvBlockSparseBitmapFootprint(Kernel rank_kernel, Kernel walk_kernel,
                                                     int threads, int32_t model_block_bound,
                                                     int max_dynamic_smem_bytes,
                                                     QTokenKvBlockSparseBitmapFootprint* out,
                                                     int32_t entry_blocks) {
  const size_t with_rank =
      QTokenKvBlockSparseBitmapSmemBytes<GroupSize>(model_block_bound, true, entry_blocks);
  const size_t without_rank =
      QTokenKvBlockSparseBitmapSmemBytes<GroupSize>(model_block_bound, false, entry_blocks);
  *out = {};
  out->entry_blocks = entry_blocks;
  if (with_rank == without_rank) {
    // Byte-map-only prefix (G <= 8): no rank array, either instantiation works.
    if (with_rank <= static_cast<size_t>(max_dynamic_smem_bytes)) {
      cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &out->ctas_per_sm, rank_kernel, threads, with_rank);
      if (status != cudaSuccess) return status;
      out->bytes = with_rank;
    }
    return cudaSuccess;
  }
  if (without_rank <= static_cast<size_t>(max_dynamic_smem_bytes)) {
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &out->ctas_per_sm, walk_kernel, threads, without_rank);
    if (status != cudaSuccess) return status;
    out->bytes = without_rank;
    out->word_rank = false;
  }
  if (with_rank <= static_cast<size_t>(max_dynamic_smem_bytes)) {
    int with_rank_ctas = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&with_rank_ctas, rank_kernel,
                                                                       threads, with_rank);
    if (status != cudaSuccess) return status;
    if (with_rank_ctas > 0 && with_rank_ctas >= out->ctas_per_sm) {
      out->bytes = with_rank;
      out->ctas_per_sm = with_rank_ctas;
      out->word_rank = true;
    }
  }
  return cudaSuccess;
}

// WordRank: the launch reserved the word-rank array (O(1) membership ranks);
// otherwise ranks walk the owner's word summaries.
// Map path of every route: chosen per route by its causal prefix (G <= 8), or
// fixed per launch for G9-G32, so each path compiles on its own (G32 runs at
// the 1024-thread register cap, where a shared body slows both).
enum QTokenKvBlockSparseMapMode : int { kMapPerRoute = 0, kMapEntryOnly = 1, kMapBitOnly = 2 };

template <typename PositionType, int GroupSize, bool PackedQuery, bool QueryMajor = false,
          bool WordRank = true, int MapMode = kMapPerRoute>
__global__ void __launch_bounds__(QTokenKvBlockSparseBitmapThreads<GroupSize>(),
                                  QTokenKvBlockSparseBitmapMinBlocks<GroupSize>())
    QTokenKvBlockSparseBitmapMetadataKernel(
        const __grid_constant__ QTokenKvBlockSparseTouchedMetadataParams<PositionType> params) {
  const uint32_t group = blockIdx.x / params.pattern_heads;
  const uint32_t head = blockIdx.x % params.pattern_heads;
  constexpr int kBlockThreads = QTokenKvBlockSparseBitmapThreads<GroupSize>();
  constexpr int kMaximumCandidates = GroupSize * (kQTokenKvBlockSparseMaxBlockTopK + 1);
  constexpr int kItemsPerThread = (kMaximumCandidates + kBlockThreads - 1) / kBlockThreads;
  // G <= 8 stores one membership byte per fragment; larger groups store
  // ceil(G / 8) byte planes per route row (queries 8p..8p+7 in plane p, each
  // membership_words / planes words).
  static_assert(GroupSize <= 32, "query membership is stored in at most four byte planes");
  constexpr int kPlanes = (GroupSize + 7) / 8;
  static_assert(kBlockThreads * 32 * 128 <= (1 << 24),
                "bit-map prefixes up to the shared-memory limit keep the block in 24 bits");
  using SharedStorage = QTokenKvBlockSparseBitmapSharedStorage<GroupSize>;
  // Candidate enumeration. When the warps split evenly over the group's
  // queries, each warp serves one query (warp-constant row, bounds, and
  // membership bit; no division). Otherwise candidates are query-major over
  // the whole CTA. Either way a thread's items cover query_item in
  // [0, block_topk] once per query (block_topk is the partial causal tail).
  constexpr int kWarps = kBlockThreads / 32;
  constexpr bool kWarpQuery = kWarps % GroupSize == 0;
  constexpr int kLanesPerQuery = kWarpQuery ? kBlockThreads / GroupSize : 1;
  static_assert(
      !kWarpQuery || kItemsPerThread * kLanesPerQuery >= kQTokenKvBlockSparseMaxBlockTopK + 1,
      "per-query lanes cover every candidate");
  const uint32_t warp_query = kWarpQuery ? threadIdx.x / kLanesPerQuery : 0u;
  const uint32_t lane_in_query = kWarpQuery ? threadIdx.x - warp_query * kLanesPerQuery : 0u;
  auto candidate = [&](int item, uint32_t& query, uint32_t& query_item) {
    if constexpr (kWarpQuery) {
      query = warp_query;
      query_item = static_cast<uint32_t>(item * kLanesPerQuery) + lane_in_query;
    } else {
      params.candidates_per_query.divmod(item * kBlockThreads + threadIdx.x, query, query_item);
    }
  };

  extern __shared__ __align__(16) unsigned char q_token_kv_block_sparse_map_smem[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(q_token_kv_block_sparse_map_smem);
  uint32_t* block_map = reinterpret_cast<uint32_t*>(
      q_token_kv_block_sparse_map_smem + QTokenKvBlockSparseBitmapFixedSmemBytes<GroupSize>());
  // Bit-map word summaries follow the model-sized padded map (allocated only
  // when the model bound exceeds the byte-map prefix limit).
  const int32_t entry_map_blocks =
      QTokenKvBlockSparseEntryMapBlocks<GroupSize>(params.entry_map_blocks);
  constexpr int kEntryBits = QTokenKvBlockSparseEntryBits<GroupSize>();
  constexpr int kEntryShift = QTokenKvBlockSparseEntryShift<GroupSize>();
  constexpr uint32_t kEntryMask = (1u << kEntryShift) - 1u;
  const int model_map_stride =
      QTokenKvBlockSparseBitmapModelStride<GroupSize>(params.model_block_bound, entry_map_blocks);
  uint32_t* thread_word_summary =
      block_map + (static_cast<size_t>(kBlockThreads) * model_map_stride + 3) / 4 * 4;
  const int summary_stride =
      QTokenKvBlockSparseBitmapSummaryWords<GroupSize>(params.model_block_bound);
  // First compact rank of each non-empty bit-map word (bit-map path; laid out
  // like the bit map), present when WordRank.
  static_assert(SharedStorage::kMaxUnionPages <= 65536, "compact ranks fit 16 bits");
  uint16_t* word_rank_base = reinterpret_cast<uint16_t*>(
      thread_word_summary + (static_cast<size_t>(kBlockThreads) * summary_stride + 3) / 4 * 4);

  if (threadIdx.x == 0) {
    shared.route = {0, -1, 0, 0, -1, -1};
    shared.union_pages = 0;
  }
  if constexpr (GroupSize > 8) {
    // The bit-map membership pass ORs into these (the entry map overwrites
    // them); cleared here, where the stores overlap the route's loads.
    for (int i = threadIdx.x; i < SharedStorage::kMaxUnionPages; i += kBlockThreads) {
      shared.compact_membership_hi[i] = 0u;
    }
  }

  // Every warp first prefetches its fixed-layout rows; the first warp then
  // resolves the route while those loads are in flight. The ignored indexer
  // suffix need not be initialized, so bound those loads by each row's visible
  // completed-block prefix.
  // Request-aligned fixed Q: the host requires fixed_query_len % GroupSize == 0,
  // so a group never crosses a request and one KV-length load per thread gives
  // every row's position, instead of one dependent division and load per item.
  // An empty request (length <= 0) keeps a sentinel far below zero, so every
  // row's position stays negative and is rejected below, as QueryPosition does.
  const uint32_t fixed_query_len = params.fixed_query_len;
  int64_t group_first_position = -1;
  if constexpr (!PackedQuery) {
    const int64_t first_row = static_cast<int64_t>(group) * GroupSize;
    if (fixed_query_len > 0 && first_row < params.rows) {
      uint32_t request;
      uint32_t offset;
      params.fixed_query_len.divmod(static_cast<uint32_t>(first_row), request, offset);
      const int64_t length = params.query_positions[request];
      group_first_position = length > 0 ? length - fixed_query_len + offset : INT64_MIN / 2;
    }
  }
  int32_t prefetched_blocks[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    prefetched_blocks[item] = -1;
    if constexpr (!PackedQuery) {
      uint32_t query;
      uint32_t query_item;
      candidate(item, query, query_item);
      const int64_t row = static_cast<int64_t>(group) * GroupSize + query;
      if (query < static_cast<uint32_t>(GroupSize) &&
          query_item < static_cast<uint32_t>(params.block_topk) && row < params.rows) {
        const int64_t position = fixed_query_len > 0
                                     ? group_first_position + query
                                     : QueryPosition(params, static_cast<int32_t>(row));
        if (position >= 0 && position < params.max_seq_len_kv &&
            query_item < static_cast<uint64_t>(SelectedBlockBound(params, position + 1))) {
          prefetched_blocks[item] =
              params.block_indices[row * params.block_indices_row_stride +
                                   static_cast<int64_t>(head) * params.block_indices_head_stride +
                                   static_cast<int64_t>(query_item) *
                                       params.block_indices_column_stride];
        }
      }
    }
  }
  // Packed Q: every thread reads the route's row range and issues its
  // selected-block loads before the route is validated, so their round trips
  // overlap the validation's dependent request/position loads and barrier.
  // Rows always hold block_topk columns (-1 padded), so loading past a query's
  // visible prefix stays in bounds; the scatter uses a loaded value only for a
  // valid route, whose rows are exactly these.
  if constexpr (PackedQuery) {
    const int32_t first_row = params.qo_indptr[group];
    const int32_t row_end = params.qo_indptr[group + 1];
    if (first_row >= 0 && row_end > first_row && row_end <= params.rows &&
        row_end - first_row <= GroupSize) {
      const uint32_t query_count = static_cast<uint32_t>(row_end - first_row);
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        uint32_t query;
        uint32_t query_item;
        candidate(item, query, query_item);
        if (query < query_count && query_item < static_cast<uint32_t>(params.block_topk)) {
          const int32_t row = first_row + static_cast<int32_t>(query);
          prefetched_blocks[item] =
              params.block_indices[static_cast<int64_t>(row) * params.block_indices_row_stride +
                                   static_cast<int64_t>(head) * params.block_indices_head_stride +
                                   static_cast<int64_t>(query_item) *
                                       params.block_indices_column_stride];
        }
      }
    }
  }
  InitRoute<PositionType, GroupSize, PackedQuery>(params, &shared.route);
  __syncthreads();

  const int64_t causal_block_bound =
      shared.route.valid
          ? (shared.route.last_position + params.sparse_block_size) >> params.sparse_block_shift
          : 0;
  const uint32_t active_logical_capacity = static_cast<uint32_t>(
      causal_block_bound < params.model_block_bound ? causal_block_bound
                                                    : params.model_block_bound);
  int32_t* group_indices = params.q_token_kv_block_sparse_page_indices +
                           static_cast<int64_t>(blockIdx.x) * params.page_capacity;
  uint8_t* group_memberships =
      reinterpret_cast<uint8_t*>(params.q_token_kv_block_sparse_page_memberships +
                                 static_cast<int64_t>(blockIdx.x) * params.membership_words);
  // Plane p (queries 8p..8p+7) starts p * plane_bytes after plane 0.
  const int plane_bytes = GroupSize > 8 ? params.membership_words / kPlanes * 4 : 0;

  // Granularity and geometry of this route's map (CTA-uniform). Only the
  // visible prefix is cleared, scattered, and scanned. The entry map holds each
  // block's query membership (no membership pass); the bit map only presence.
  const bool use_entry_map = MapMode == kMapEntryOnly ||
                             (MapMode == kMapPerRoute &&
                              active_logical_capacity <= static_cast<uint32_t>(entry_map_blocks));
  const int blocks_per_word = use_entry_map ? (1 << kEntryShift) : 32;
  const int map_words = static_cast<int>((active_logical_capacity + blocks_per_word - 1) /
                                         static_cast<uint32_t>(blocks_per_word));
  const int words_per_thread = QTokenKvBlockSparseWordsPerThread(map_words, kBlockThreads);
  const int map_stride = QTokenKvBlockSparseMapStride(words_per_thread);
  // Storage index of logical map word `word` (owner thread range + offset).
  auto map_index = [&](int word) -> int {
    if (words_per_thread == 1) {
      return word;
    }
    const int owner = word / words_per_thread;
    return owner * map_stride + (word - owner * words_per_thread);
  };
  // Owner thread and offset of a logical map word; one word per thread (the
  // common case) needs no division.
  auto owner_offset = [&](int word, int& owner, int& offset) {
    if (words_per_thread == 1) {
      owner = word;
      offset = 0;
    } else {
      owner = word / words_per_thread;
      offset = word - owner * words_per_thread;
    }
  };
  const int word_begin = threadIdx.x * words_per_thread;
  const int word_end =
      word_begin + words_per_thread < map_words ? word_begin + words_per_thread : map_words;
  const int storage_begin = threadIdx.x * map_stride;

  if (threadIdx.x < static_cast<unsigned>(GroupSize)) {
    int32_t complete = 0;
    int32_t tail = -1;
    if (shared.route.valid && threadIdx.x < static_cast<unsigned>(shared.route.query_count)) {
      const int64_t visible_tokens = shared.route.first_position + threadIdx.x + 1;
      complete = static_cast<int32_t>(SelectedBlockBound(params, visible_tokens));
      if (!params.count_current_block_in_topk ||
          (visible_tokens & (params.sparse_block_size - 1)) != 0) {
        tail = complete;
      }
    }
    shared.query_complete_blocks[threadIdx.x] = complete;
    shared.query_tail_block[threadIdx.x] = tail;
  }

  // Clear the active map's padded storage span with coalesced 16-byte stores
  // (the fixed storage keeps the map 16-byte aligned); only the span the
  // active prefix can touch is cleared.
  {
    const int active_span_words = kBlockThreads * map_stride;
    uint4* map_vec = reinterpret_cast<uint4*>(block_map);
    for (int chunk = threadIdx.x; chunk < (active_span_words + 3) / 4; chunk += kBlockThreads) {
      map_vec[chunk] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  const int active_summary_words = (words_per_thread + 31) >> 5;
  uint32_t* my_summary = thread_word_summary + threadIdx.x * summary_stride;
  if (!use_entry_map) {
#pragma unroll
    for (int k = 0; k < kQTokenKvBlockSparseBitmapSummaryWordsPerThread; ++k) {
      if (k >= active_summary_words) break;
      my_summary[k] = 0u;
    }
  }
  __syncthreads();

  // Scatter every live candidate. The candidate enumeration matches the sort
  // path exactly: selected complete blocks first, then the synthesized partial
  // causal tail per query. Accepted candidates stay in registers for the
  // bit-map membership pass.
  int32_t candidate_blocks[kItemsPerThread];
  uint32_t candidate_queries[kItemsPerThread];
  // Warp-constant bounds of the warp's query (one-query-per-warp enumeration).
  const int32_t warp_complete_blocks = kWarpQuery ? shared.query_complete_blocks[warp_query] : 0;
  const int32_t warp_tail_block = kWarpQuery ? shared.query_tail_block[warp_query] : -1;
  if constexpr (kWarpQuery) {
    // One query per warp: the route and bound state live in registers (the
    // shared-memory atomics below would otherwise force per-item reloads), and
    // the loop is compiled per map granularity and one-word-per-thread case.
    const bool warp_live =
        shared.route.valid && warp_query < static_cast<uint32_t>(shared.route.query_count);
    const uint32_t topk = static_cast<uint32_t>(params.block_topk);
    const uint32_t complete = static_cast<uint32_t>(warp_complete_blocks);
    const uint32_t selected_count = complete < topk ? complete : topk;
    uint32_t block_bound = complete < static_cast<uint32_t>(params.model_block_bound)
                               ? complete
                               : static_cast<uint32_t>(params.model_block_bound);
    block_bound = block_bound < active_logical_capacity ? block_bound : active_logical_capacity;
    const int32_t tail_block =
        warp_tail_block >= 0 && static_cast<uint32_t>(warp_tail_block) < active_logical_capacity
            ? warp_tail_block
            : -1;
    auto scatter = [&](auto one_word, auto byte_map) {
      constexpr bool kOneWord = decltype(one_word)::value;
      constexpr bool kByteMap = decltype(byte_map)::value;
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const uint32_t query_item = static_cast<uint32_t>(item * kLanesPerQuery) + lane_in_query;
        const int32_t selected = prefetched_blocks[item];
        int32_t block = query_item < selected_count && static_cast<uint32_t>(selected) < block_bound
                            ? selected
                            : -1;
        if (query_item == topk) {
          block = tail_block;
        }
        candidate_blocks[item] = block;
        if (block >= 0) {
          const uint32_t b = static_cast<uint32_t>(block);
          if constexpr (kByteMap) {
            const int word = static_cast<int>(b >> kEntryShift);
            atomicOr(&block_map[kOneWord ? word : map_index(word)],
                     (uint32_t{1} << warp_query) << ((b & kEntryMask) * kEntryBits));
          } else {
            int owner = static_cast<int>(b >> 5);
            int offset = 0;
            if constexpr (!kOneWord) {
              owner_offset(static_cast<int>(b >> 5), owner, offset);
            }
            atomicOr(&block_map[owner * map_stride + offset], uint32_t{1} << (b & 31u));
          }
        }
      }
    };
    if (!warp_live) {
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        candidate_blocks[item] = -1;
      }
    } else if (use_entry_map) {
      if (words_per_thread == 1) {
        scatter(std::true_type{}, std::true_type{});
      } else {
        scatter(std::false_type{}, std::true_type{});
      }
    } else if (words_per_thread == 1) {
      scatter(std::true_type{}, std::false_type{});
    } else {
      scatter(std::false_type{}, std::false_type{});
    }
  } else {
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      uint32_t query;
      uint32_t query_item;
      candidate(item, query, query_item);

      int32_t logical_block = -1;
      if (shared.route.valid && query < static_cast<uint32_t>(shared.route.query_count) &&
          query < static_cast<uint32_t>(GroupSize) &&
          query_item <= static_cast<uint32_t>(params.block_topk)) {
        const int32_t complete_block_count =
            kWarpQuery ? warp_complete_blocks : shared.query_complete_blocks[query];
        const int32_t selected_count =
            complete_block_count < params.block_topk ? complete_block_count : params.block_topk;
        if (query_item < static_cast<uint32_t>(selected_count)) {
          const int32_t selected_block = prefetched_blocks[item];
          if (selected_block >= 0 && selected_block < complete_block_count &&
              selected_block < params.model_block_bound) {
            logical_block = selected_block;
          }
        } else if (query_item == static_cast<uint32_t>(params.block_topk)) {
          logical_block = kWarpQuery ? warp_tail_block : shared.query_tail_block[query];
        }
      }
      candidate_blocks[item] = -1;
      candidate_queries[item] = query;
      if (logical_block >= 0 && static_cast<uint32_t>(logical_block) < active_logical_capacity) {
        const uint32_t block = static_cast<uint32_t>(logical_block);
        candidate_blocks[item] = static_cast<int32_t>(block);
        if (use_entry_map) {
          atomicOr(&block_map[map_index(block >> kEntryShift)],
                   (uint32_t{1} << query) << ((block & kEntryMask) * kEntryBits));
        } else {
          int owner;
          int offset;
          owner_offset(static_cast<int>(block >> 5), owner, offset);
          atomicOr(&block_map[owner * map_stride + offset], uint32_t{1} << (block & 31u));
        }
      }
    }
  }
  __syncthreads();

  // Each owner marks its own non-empty words (no per-candidate summary atomics).
  if (!use_entry_map) {
    for (int offset = 0; offset < word_end - word_begin; ++offset) {
      if (block_map[storage_begin + offset] != 0u) {
        my_summary[offset >> 5] |= uint32_t{1} << (offset & 31);
      }
    }
  }
  // Each thread owns a contiguous word range so the compact list stays in
  // ascending logical order; a block-wide exclusive sum ranks its entries.
  int local_unique_count = 0;
  if (use_entry_map) {
    for (int word = word_begin; word < word_end; ++word) {
      const uint32_t value = block_map[storage_begin + (word - word_begin)];
      if constexpr (kEntryBits == 8) {
        local_unique_count += __popc(__vsetne4(value, 0u));
      } else if constexpr (kEntryBits == 16) {
        local_unique_count += __popc(__vsetne2(value, 0u));
      } else {
        local_unique_count += value != 0u;
      }
    }
  } else {
#pragma unroll
    for (int k = 0; k < kQTokenKvBlockSparseBitmapSummaryWordsPerThread; ++k) {
      if (k >= active_summary_words) break;
      uint32_t marks = my_summary[k];
      while (marks != 0u) {
        const int offset = k * 32 + __ffs(static_cast<int>(marks)) - 1;
        local_unique_count += __popc(block_map[storage_begin + offset]);
        marks &= marks - 1u;
      }
    }
  }
  int thread_output_begin = 0;
  int union_pages = 0;
  cub::BlockScan<int, kBlockThreads>(shared.scan)
      .ExclusiveSum(local_unique_count, thread_output_begin, union_pages);
  shared.thread_rank_begin[threadIdx.x] = thread_output_begin;

  int output_rank = thread_output_begin;
  if (use_entry_map) {
    for (int word = word_begin; word < word_end; ++word) {
      const uint32_t value = block_map[storage_begin + (word - word_begin)];
      if (value == 0u) {
        continue;
      }
#pragma unroll
      for (int entry = 0; entry < (1 << kEntryShift); ++entry) {
        uint32_t membership = value;
        if constexpr (kEntryBits < 32) {
          membership = (value >> (entry * kEntryBits)) & ((1u << kEntryBits) - 1u);
        }
        if (membership != 0u) {
          shared.compact_union[output_rank] =
              static_cast<uint32_t>((word << kEntryShift) + entry) | (membership << 24);
          if constexpr (GroupSize > 8) {
            shared.compact_membership_hi[output_rank] = membership >> 8;
          }
          ++output_rank;
        }
      }
    }
  } else {
#pragma unroll
    for (int k = 0; k < kQTokenKvBlockSparseBitmapSummaryWordsPerThread; ++k) {
      if (k >= active_summary_words) break;
      uint32_t marks = my_summary[k];
      while (marks != 0u) {
        const int offset = k * 32 + __ffs(static_cast<int>(marks)) - 1;
        marks &= marks - 1u;
        const int word = word_begin + offset;
        uint32_t bits = block_map[storage_begin + offset];
        if constexpr (WordRank) {
          word_rank_base[storage_begin + offset] = static_cast<uint16_t>(output_rank);
        }
        while (bits != 0u) {
          const int bit = __ffs(static_cast<int>(bits)) - 1;
          shared.compact_union[output_rank++] = static_cast<uint32_t>(word * 32 + bit);
          bits &= bits - 1u;
        }
      }
    }
  }
  if (threadIdx.x == 0) {
    shared.union_pages = union_pages;
  }
  __syncthreads();

  if (!use_entry_map) {
    // Bit-map membership pass: each accepted candidate recovers its block's
    // compact rank (owner thread's rank prefix + popcount of the preceding
    // words and lower bits) and ORs its query bit into that entry's high byte.
    // Compact rank of a bit-map block from its owner thread and word offset.
    auto rank_of = [&](int block, int owner, int offset) -> int {
      const int owner_storage = owner * map_stride;
      int rank;
      if constexpr (WordRank) {
        rank = static_cast<int>(word_rank_base[owner_storage + offset]);
      } else {
        // Owner's rank prefix plus the non-empty words of its range below this word.
        rank = shared.thread_rank_begin[owner];
        const uint32_t* owner_summary = thread_word_summary + owner * summary_stride;
        for (int k = 0; k <= (offset >> 5); ++k) {
          uint32_t marks = owner_summary[k];
          if (k == (offset >> 5)) {
            marks &= (uint32_t{1} << (offset & 31)) - 1u;
          }
          while (marks != 0u) {
            rank += __popc(block_map[owner_storage + k * 32 + __ffs(static_cast<int>(marks)) - 1]);
            marks &= marks - 1u;
          }
        }
      }
      return rank +
             __popc(block_map[owner_storage + offset] & ((uint32_t{1} << (block & 31)) - 1u));
    };
    if constexpr (kWarpQuery) {
      // The warp's query bit and its membership plane are warp-constant.
      const bool low_plane = GroupSize <= 8 || warp_query < 8u;
      uint32_t* member_words = low_plane ? shared.compact_union : shared.compact_membership_hi;
      const uint32_t member_bit =
          low_plane ? (uint32_t{1} << warp_query) << 24 : uint32_t{1} << (warp_query - 8u);
      auto member = [&](auto one_word) {
        constexpr bool kOneWord = decltype(one_word)::value;
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
          const int32_t block = candidate_blocks[item];
          if (block >= 0) {
            int owner = block >> 5;
            int offset = 0;
            if constexpr (!kOneWord) {
              owner_offset(block >> 5, owner, offset);
            }
            atomicOr(&member_words[rank_of(block, owner, offset)], member_bit);
          }
        }
      };
      if (words_per_thread == 1) {
        member(std::true_type{});
      } else {
        member(std::false_type{});
      }
    } else {
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int32_t block = candidate_blocks[item];
        if (block >= 0) {
          int owner;
          int offset;
          owner_offset(block >> 5, owner, offset);
          const int rank = rank_of(block, owner, offset);
          const uint32_t query = candidate_queries[item];
          if (GroupSize <= 8 || query < 8u) {
            atomicOr(&shared.compact_union[rank], (uint32_t{1} << query) << 24);
          } else {
            atomicOr(&shared.compact_membership_hi[rank], uint32_t{1} << (query - 8u));
          }
        }
      }
    }
    __syncthreads();
  }

  // The map and its summaries are dead after membership reduction. Reuse
  // that storage for private query bytes instead of writing an intermediate
  // GMEM representation. The byte-map allocation covers every live union.
  if constexpr (QueryMajor) {
    static_assert(GroupSize >= 2 && GroupSize <= 8);
    group_memberships = reinterpret_cast<uint8_t*>(block_map);
  }
  // Resolve independent page-table loads before publishing each output batch:
  // issue the whole batch's loads before consuming any page ID, so that their
  // round trips overlap. A semantic block may cross multiple noncontiguous
  // physical cache pages.
  constexpr int kEmitBatch = 4;
  const int union_fragments = union_pages * params.fragments_per_block;
  const int fragment_shift = __ffs(params.fragments_per_block) - 1;
  for (int rank_base = 0; rank_base < union_fragments; rank_base += kEmitBatch * kBlockThreads) {
    int32_t physical_pages[kEmitBatch];
    uint32_t subpages[kEmitBatch];
    uint32_t memberships[kEmitBatch];
#pragma unroll
    for (int j = 0; j < kEmitBatch; ++j) {
      const int rank = rank_base + j * kBlockThreads + threadIdx.x;
      physical_pages[j] = -1;
      subpages[j] = 0;
      memberships[j] = 0;
      if (rank < union_fragments) {
        const uint32_t entry = shared.compact_union[rank >> fragment_shift];
        const uint32_t fragment =
            ((entry & 0xFFFFFFu) << fragment_shift) + (rank & (params.fragments_per_block - 1));
        const int64_t first_token = static_cast<int64_t>(fragment) * params.fragment_size;
        uint32_t storage_page;
        params.subpages_per_storage_page.divmod(fragment, storage_page, subpages[j]);
        if (first_token <= shared.route.last_position &&
            storage_page < static_cast<uint32_t>(params.page_table_width)) {
          physical_pages[j] =
              params
                  .block_table[static_cast<int64_t>(shared.route.request) *
                                   params.block_table_request_stride +
                               static_cast<int64_t>(storage_page) * params.block_table_page_stride];
        }
        uint32_t membership = entry >> 24;
        if constexpr (GroupSize > 8) {
          membership |= shared.compact_membership_hi[rank >> fragment_shift] << 8;
        }
        const int64_t first_visible_query = first_token - shared.route.first_position;
        if (first_visible_query > 0) {
          membership &= first_visible_query < GroupSize ? (~0u << first_visible_query) : 0u;
        }
        memberships[j] = membership;
      }
    }
#pragma unroll
    for (int j = 0; j < kEmitBatch; ++j) {
      const int rank = rank_base + j * kBlockThreads + threadIdx.x;
      if (rank < union_fragments) {
        const bool mapped = physical_pages[j] >= 0;
        group_indices[rank] =
            mapped ? static_cast<int32_t>(static_cast<uint32_t>(physical_pages[j]) *
                                              static_cast<uint32_t>(params.locator_page_stride) +
                                          subpages[j])
                   : -1;
        group_memberships[rank] = mapped ? static_cast<uint8_t>(memberships[j]) : 0;
#pragma unroll
        for (int plane = 1; plane < kPlanes; ++plane) {
          group_memberships[plane * plane_bytes + rank] =
              mapped ? static_cast<uint8_t>(memberships[j] >> (8 * plane)) : 0;
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (shared.route.valid && shared.union_pages > 0) {
      const int tail_tokens =
          static_cast<int>((shared.route.last_position + 1) & (params.sparse_block_size - 1));
      const int tail_padding = tail_tokens == 0 ? 0 : params.sparse_block_size - tail_tokens;
      params.seq_lens[blockIdx.x] = shared.union_pages * params.sparse_block_size - tail_padding;
      for (int byte = shared.union_pages * params.fragments_per_block;
           byte % kQTokenKvBlockSparseMembershipsPerWord != 0; ++byte) {
        for (int plane = 0; plane < kPlanes; ++plane) {
          group_memberships[plane * plane_bytes + byte] = 0;
        }
      }
    } else {
      group_indices[0] = -1;
      for (int plane = 0; plane < kPlanes; ++plane) {
        reinterpret_cast<uint32_t*>(group_memberships + plane * plane_bytes)[0] = 0;
      }
      params.seq_lens[blockIdx.x] = 1;
    }
  }
  __syncthreads();
  if constexpr (QueryMajor) {
    StoreQueryMajorMemberships<PositionType, kBlockThreads, GroupSize>(params, group_memberships,
                                                                       shared.union_pages);
  }
  ReleasePdlDependents(params);
  uint32_t dynamic_smem_bytes;
  asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem_bytes));
  WriteLongestFirstRouteOrder(params, reinterpret_cast<int*>(q_token_kv_block_sparse_map_smem),
                              static_cast<int>(dynamic_smem_bytes / sizeof(int)));
}

// Cached per host thread, device, and kernel specialization: no shared mutable
// host state, device-count limit, or repeated setup in a prepared launch.
struct QTokenKvBlockSparseUnionLaunchState {
  bool device_initialized = false;
  bool map_initialized = false;
  int max_dynamic_smem_bytes = 0;
  int sm_count = 0;
  int sort_ctas_per_sm = 0;
  // Per model block bound: map-kernel footprint and rank mode.
  std::unordered_map<int32_t, QTokenKvBlockSparseBitmapFootprint> map_footprints;
};

// Smallest dynamic SMEM a G9-G32 metadata launch can use for a model block
// bound: the entry map when the prefix allows it, or the bit map without the
// word-rank array. -1 when the bit map exceeds the per-thread word limit.
template <int GroupSize>
int64_t QTokenKvBlockSparseWideMetadataMinSmemBytes(int32_t model_block_bound) {
  static_assert(GroupSize > 8 && GroupSize <= 32);
  constexpr int kThreads = QTokenKvBlockSparseBitmapThreads<GroupSize>();
  if (QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(model_block_bound, true),
                                        kThreads) > kQTokenKvBlockSparseBitmapMaxWordsPerThread) {
    return -1;
  }
  size_t bytes = QTokenKvBlockSparseBitmapSmemBytes<GroupSize>(model_block_bound, false, 0);
  if (model_block_bound <= kQTokenKvBlockSparseWideEntryMapMaxBlocks) {
    bytes = std::min(bytes, QTokenKvBlockSparseBitmapSmemBytes<GroupSize>(model_block_bound, false,
                                                                          model_block_bound));
  }
  return static_cast<int64_t>(bytes);
}

// G9-G32 (two to four membership planes) only has the bit-map kernel.
template <typename PositionType, int GroupSize, bool PackedQuery>
cudaError_t LaunchQTokenKvBlockSparseWideMetadataTyped(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, cudaStream_t stream) {
  static_assert(GroupSize > 8 && GroupSize <= 32);
  auto map_kernel = QTokenKvBlockSparseBitmapMetadataKernel<PositionType, GroupSize, PackedQuery,
                                                            false, true, kMapBitOnly>;
  auto walk_kernel = QTokenKvBlockSparseBitmapMetadataKernel<PositionType, GroupSize, PackedQuery,
                                                             false, false, kMapBitOnly>;
  auto entry_kernel = QTokenKvBlockSparseBitmapMetadataKernel<PositionType, GroupSize, PackedQuery,
                                                              false, true, kMapEntryOnly>;
  constexpr int kThreads = QTokenKvBlockSparseBitmapThreads<GroupSize>();
  const int map_words_per_thread = QTokenKvBlockSparseWordsPerThread(
      QTokenKvBlockSparseMapWords(params.model_block_bound, true), kThreads);
  if (map_words_per_thread > kQTokenKvBlockSparseBitmapMaxWordsPerThread) {
    return cudaErrorInvalidValue;
  }
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  static thread_local std::unordered_map<int, QTokenKvBlockSparseUnionLaunchState> devices;
  auto& state = devices[device];
  if (!state.map_initialized) {
    status = cudaDeviceGetAttribute(&state.max_dynamic_smem_bytes,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    for (auto kernel : {map_kernel, walk_kernel, entry_kernel}) {
      status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    state.max_dynamic_smem_bytes);
      if (status != cudaSuccess) return status;
    }
    state.map_initialized = true;
  }
  auto footprint = state.map_footprints.find(params.model_block_bound);
  if (footprint == state.map_footprints.end()) {
    // The bit map, or the entry map over the model's whole prefix when that
    // fits without lowering the CTAs per SM (no membership pass).
    QTokenKvBlockSparseBitmapFootprint selected;
    status = SelectQTokenKvBlockSparseBitmapFootprint<GroupSize>(
        map_kernel, walk_kernel, kThreads, params.model_block_bound, state.max_dynamic_smem_bytes,
        &selected, 0);
    if (status != cudaSuccess) return status;
    if (params.model_block_bound <= kQTokenKvBlockSparseWideEntryMapMaxBlocks) {
      QTokenKvBlockSparseBitmapFootprint entry;
      status = SelectQTokenKvBlockSparseBitmapFootprint<GroupSize>(
          entry_kernel, entry_kernel, kThreads, params.model_block_bound,
          state.max_dynamic_smem_bytes, &entry, params.model_block_bound);
      if (status != cudaSuccess) return status;
      if (entry.bytes != 0 && entry.ctas_per_sm > 0 && entry.ctas_per_sm >= selected.ctas_per_sm) {
        selected = entry;
      }
    }
    footprint = state.map_footprints.emplace(params.model_block_bound, selected).first;
  }
  const QTokenKvBlockSparseBitmapFootprint& selected = footprint->second;
  if (selected.bytes == 0 || selected.ctas_per_sm <= 0) {
    return cudaErrorInvalidValue;
  }
  params.entry_map_blocks = selected.entry_blocks;
  const int64_t routes =
      static_cast<int64_t>(params.groups) * static_cast<uint32_t>(params.pattern_heads);
  auto kernel = selected.entry_blocks > 0 ? entry_kernel
                : selected.word_rank      ? map_kernel
                                          : walk_kernel;
  const size_t smem_bytes = QTokenKvBlockSparseRouteOrderLaunchBytes(
      selected.bytes, params.route_order, routes, state.max_dynamic_smem_bytes);
  kernel<<<routes, kThreads, smem_bytes, stream>>>(params);
  return cudaGetLastError();
}

template <typename PositionType, int GroupSize, bool PackedQuery, bool QueryMajor = false>
cudaError_t LaunchQTokenKvBlockSparseTouchedMetadataTyped(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, cudaStream_t stream) {
  constexpr int kSortBlockThreads =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kBlockThreads;
  auto map_kernel = QTokenKvBlockSparseBitmapMetadataKernel<PositionType, GroupSize, PackedQuery,
                                                            QueryMajor, true>;
  auto walk_kernel = QTokenKvBlockSparseBitmapMetadataKernel<PositionType, GroupSize, PackedQuery,
                                                             QueryMajor, false>;
  auto sort_kernel =
      QTokenKvBlockSparseTouchedMetadataKernel<PositionType, GroupSize, PackedQuery, QueryMajor>;

  const int64_t routes =
      static_cast<int64_t>(params.groups) * static_cast<uint32_t>(params.pattern_heads);
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  static thread_local std::unordered_map<int, QTokenKvBlockSparseUnionLaunchState> devices;
  auto& state = devices[device];
  if (!state.device_initialized) {
    status = cudaDeviceGetAttribute(&state.max_dynamic_smem_bytes,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(&state.sm_count, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&state.sort_ctas_per_sm, sort_kernel,
                                                           kSortBlockThreads, 0);
    if (status != cudaSuccess) return status;
    state.device_initialized = true;
  }

  const int map_words_per_thread =
      QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(params.model_block_bound));
  bool use_map = map_words_per_thread <= kQTokenKvBlockSparseBitmapMaxWordsPerThread;
  QTokenKvBlockSparseBitmapFootprint selected;
  if (use_map) {
    if (!state.map_initialized) {
      // Always opt in to the same maximum, not a per-launch value that another
      // host thread could lower between configuration and launch.
      for (auto kernel : {map_kernel, walk_kernel}) {
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      state.max_dynamic_smem_bytes);
        if (status != cudaSuccess) return status;
      }
      state.map_initialized = true;
    }
    auto footprint = state.map_footprints.find(params.model_block_bound);
    if (footprint == state.map_footprints.end()) {
      status = SelectQTokenKvBlockSparseBitmapFootprint<GroupSize>(
          map_kernel, walk_kernel, kQTokenKvBlockSparseBitmapBlockThreads, params.model_block_bound,
          state.max_dynamic_smem_bytes, &selected, kQTokenKvBlockSparseByteMapMaxBlocks);
      if (status != cudaSuccess) return status;
      footprint = state.map_footprints.emplace(params.model_block_bound, selected).first;
    }
    selected = footprint->second;
    const int map_ctas = selected.ctas_per_sm;
    use_map = selected.bytes != 0 && map_ctas > 0;
    if (use_map && state.sort_ctas_per_sm > 0) {
      // Use actual occupancy (including registers and the GroupSize-specific
      // sort topology), cached once per footprint rather than queried per run.
      const int64_t map_slots = static_cast<int64_t>(state.sm_count) * map_ctas;
      const int64_t sort_slots = static_cast<int64_t>(state.sm_count) * state.sort_ctas_per_sm;
      use_map = (routes + map_slots - 1) / map_slots <= (routes + sort_slots - 1) / sort_slots;
    }
  }
  if (use_map) {
    const size_t smem_bytes = QTokenKvBlockSparseRouteOrderLaunchBytes(
        selected.bytes, params.route_order, routes, state.max_dynamic_smem_bytes);
    (selected.word_rank
         ? map_kernel
         : walk_kernel)<<<routes, kQTokenKvBlockSparseBitmapBlockThreads, smem_bytes, stream>>>(
        params);
  } else {
    sort_kernel<<<routes, kSortBlockThreads, 0, stream>>>(params);
  }
  return cudaGetLastError();
}

template <typename PositionType, bool PackedQuery>
cudaError_t LaunchQTokenKvBlockSparseQ1MetadataTyped(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, cudaStream_t stream) {
  auto kernel = QTokenKvBlockSparseQ1MetadataKernel<PositionType, PackedQuery>;
  kernel<<<params.groups* static_cast<uint32_t>(params.pattern_heads),
           kQTokenKvBlockSparseQ1BlockThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace detail

template <typename PositionType, bool PackedQuery, bool QueryMajor = false>
cudaError_t LaunchQTokenKvBlockSparseTouchedMetadata(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, int32_t group_size,
    cudaStream_t stream) {
  static_assert(std::is_same_v<PositionType, int32_t> || std::is_same_v<PositionType, int64_t>);
  if (params.groups == 0) {
    return cudaSuccess;
  }
  if constexpr (QueryMajor) {
    if (group_size < 2 || group_size > 8 || params.sparse_block_size != 4 ||
        params.fragment_size != 4) {
      return cudaErrorInvalidValue;
    }
  }
  switch (group_size) {
    case 1:
      return detail::LaunchQTokenKvBlockSparseQ1MetadataTyped<PositionType, PackedQuery>(params,
                                                                                         stream);
    case 2:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 2, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 3:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 3, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 4:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 4, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 5:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 5, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 6:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 6, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 7:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 7, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 8:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 8, PackedQuery,
                                                                   QueryMajor>(params, stream);
    case 16:
      if constexpr (QueryMajor) {
        return cudaErrorInvalidValue;
      } else {
        return detail::LaunchQTokenKvBlockSparseWideMetadataTyped<PositionType, 16, PackedQuery>(
            params, stream);
      }
    case 32:
      if constexpr (QueryMajor) {
        return cudaErrorInvalidValue;
      } else {
        return detail::LaunchQTokenKvBlockSparseWideMetadataTyped<PositionType, 32, PackedQuery>(
            params, stream);
      }
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace prims_ts
}  // namespace attention
}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PRIMS_TS_Q_TOKEN_KV_BLOCK_SPARSE_METADATA_CUH_
