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

#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
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
};

namespace detail {

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
    lane_request = params.token_to_request[first_row + lane];
    lane_position = static_cast<int64_t>(params.query_positions[first_row + lane]);
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

// Permit the attention prologue to start early. Its dependency wait still
// covers completion and visibility of every metadata CTA's output stores.
template <typename PositionType>
__device__ __forceinline__ void ReleasePdlDependents(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.release_pdl) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
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
  ReleasePdlDependents(params);
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
  const int32_t causal_blocks = visible >> params.sparse_block_shift;
  const int32_t complete_blocks = min(causal_blocks, params.block_topk);
  const int32_t tail = visible & (params.sparse_block_size - 1);
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
}

// Call after publishing the live membership bytes with a CTA barrier.
// Both union algorithms emit the same eight-query/32-page tile, including
// zero tail bits and a complete zero tile for an inert route.
template <typename PositionType, int BlockThreads>
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
    for (int query = 0; query < 8; ++query) {
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
      const int64_t complete_block_count = visible_tokens >> params.sparse_block_shift;
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
      const int64_t complete_block_count = (visible_tokens >> params.sparse_block_shift);
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
                 (visible_tokens & (params.sparse_block_size - 1)) != 0) {
        logical_block = (visible_tokens >> params.sparse_block_shift);
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
  ReleasePdlDependents(params);
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
  // kernel has no producer dependency. Its entry release lets the attention
  // prologue overlap; the consumer wait still covers all output stores.
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
    static_assert(GroupSize == 8);
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
    StoreQueryMajorMemberships<PositionType, kBlockThreads>(params, group_memberships,
                                                            shared.union_pages);
  }
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

template <int GroupSize>
struct QTokenKvBlockSparseBitmapSharedStorage {
  static constexpr int kMaxUnionPages = GroupSize * (kQTokenKvBlockSparseMaxBlockTopK + 1);

  typename cub::BlockScan<int, kQTokenKvBlockSparseBitmapBlockThreads>::TempStorage scan;
  // Ascending compact union: bits [0, 24) logical block, bits [24, 32) query
  // membership (written directly by the byte map, OR-reduced by the bit map).
  uint32_t compact_union[kMaxUnionPages];
  // Compact rank of the first entry in each thread's contiguous word range.
  int32_t thread_rank_begin[kQTokenKvBlockSparseBitmapBlockThreads];
  QTokenKvBlockSparseRouteState route;
  int32_t union_pages;
};

// Reserve the larger of the full bit map and the byte map capped at the
// byte-granularity prefix limit. Runtime prefixes may use either layout.
__host__ __device__ constexpr int QTokenKvBlockSparseMapWords(int32_t blocks) {
  const int bit_words = (blocks + 31) / 32;
  const int byte_blocks =
      blocks < kQTokenKvBlockSparseByteMapMaxBlocks ? blocks : kQTokenKvBlockSparseByteMapMaxBlocks;
  const int byte_words = (byte_blocks + 3) / 4;
  return bit_words > byte_words ? bit_words : byte_words;
}

// Words each thread owns for a map of `map_words`, and the padded storage
// stride between consecutive threads' ranges (odd, so lanes touch distinct
// banks when they walk their ranges in lockstep).
__host__ __device__ constexpr int QTokenKvBlockSparseWordsPerThread(int map_words) {
  return (map_words + kQTokenKvBlockSparseBitmapBlockThreads - 1) /
         kQTokenKvBlockSparseBitmapBlockThreads;
}
__host__ __device__ constexpr int QTokenKvBlockSparseMapStride(int words_per_thread) {
  return words_per_thread + ((words_per_thread & 1) == 0 ? 1 : 0);
}

template <int GroupSize>
__host__ __device__ constexpr size_t QTokenKvBlockSparseBitmapFixedSmemBytes() {
  return (sizeof(QTokenKvBlockSparseBitmapSharedStorage<GroupSize>) + 15) / 16 * 16;
}

// Dynamic shared memory of the map kernel: the fixed storage followed by the
// padded map sized for the model's full logical prefix.
template <int GroupSize>
__host__ __device__ constexpr size_t QTokenKvBlockSparseBitmapSmemBytes(int32_t model_block_bound) {
  const int stride = QTokenKvBlockSparseMapStride(
      QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(model_block_bound)));
  const size_t map_bytes =
      static_cast<size_t>(kQTokenKvBlockSparseBitmapBlockThreads) * stride * sizeof(uint32_t);
  // Bit-map path only (prefixes past the byte-map limit): one summary bit per
  // map word of each thread's range marking words that received a candidate,
  // so the count, compaction, and rank passes visit only non-empty words.
  const size_t summary_bytes = model_block_bound > kQTokenKvBlockSparseByteMapMaxBlocks
                                   ? static_cast<size_t>(kQTokenKvBlockSparseBitmapBlockThreads) *
                                         kQTokenKvBlockSparseBitmapSummaryWordsPerThread *
                                         sizeof(uint32_t)
                                   : 0;
  return QTokenKvBlockSparseBitmapFixedSmemBytes<GroupSize>() + (map_bytes + 15) / 16 * 16 +
         summary_bytes;
}

template <typename PositionType, int GroupSize, bool PackedQuery, bool QueryMajor = false>
__global__
__launch_bounds__(kQTokenKvBlockSparseBitmapBlockThreads) void QTokenKvBlockSparseBitmapMetadataKernel(
    const __grid_constant__ QTokenKvBlockSparseTouchedMetadataParams<PositionType> params) {
  ReleasePdlDependents(params);
  const uint32_t group = blockIdx.x / params.pattern_heads;
  const uint32_t head = blockIdx.x % params.pattern_heads;
  constexpr int kBlockThreads = kQTokenKvBlockSparseBitmapBlockThreads;
  constexpr int kMaximumCandidates = GroupSize * (kQTokenKvBlockSparseMaxBlockTopK + 1);
  constexpr int kItemsPerThread = (kMaximumCandidates + kBlockThreads - 1) / kBlockThreads;
  static_assert(GroupSize <= 8, "query membership is stored in one byte");
  static_assert(kQTokenKvBlockSparseBitmapBlockThreads * 32 * 128 <= (1 << 24),
                "bit-map prefixes up to the shared-memory limit keep the block in 24 bits");
  using SharedStorage = QTokenKvBlockSparseBitmapSharedStorage<GroupSize>;

  extern __shared__ __align__(16) unsigned char q_token_kv_block_sparse_map_smem[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(q_token_kv_block_sparse_map_smem);
  uint32_t* block_map = reinterpret_cast<uint32_t*>(
      q_token_kv_block_sparse_map_smem + QTokenKvBlockSparseBitmapFixedSmemBytes<GroupSize>());
  // Bit-map word summaries follow the model-sized padded map (allocated only
  // when the model bound exceeds the byte-map prefix limit).
  const int model_map_stride = QTokenKvBlockSparseMapStride(
      QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(params.model_block_bound)));
  uint32_t* thread_word_summary =
      block_map + (static_cast<size_t>(kBlockThreads) * model_map_stride + 3) / 4 * 4;

  if (threadIdx.x == 0) {
    shared.route = {0, -1, 0, 0, -1, -1};
    shared.union_pages = 0;
  }

  // The first warp resolves the route while the other warps prefetch their
  // fixed-layout rows. The ignored indexer suffix need not be initialized, so
  // bound those loads by each row's visible completed-block prefix.
  InitRoute<PositionType, GroupSize, PackedQuery>(params, &shared.route);
  int32_t prefetched_blocks[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    prefetched_blocks[item] = -1;
    if constexpr (!PackedQuery) {
      const uint32_t candidate_rank = item * kBlockThreads + threadIdx.x;
      uint32_t query;
      uint32_t query_item;
      params.candidates_per_query.divmod(candidate_rank, query, query_item);
      const int64_t row = static_cast<int64_t>(group) * GroupSize + query;
      if (query < static_cast<uint32_t>(GroupSize) &&
          query_item < static_cast<uint32_t>(params.block_topk) && row < params.rows) {
        const int64_t position = static_cast<int64_t>(params.query_positions[row]);
        if (position >= 0 && position < params.max_seq_len_kv &&
            query_item < static_cast<uint64_t>((position + 1) >> params.sparse_block_shift)) {
          prefetched_blocks[item] =
              params.block_indices[row * params.block_indices_row_stride +
                                   static_cast<int64_t>(head) * params.block_indices_head_stride +
                                   static_cast<int64_t>(query_item) *
                                       params.block_indices_column_stride];
        }
      }
    }
  }
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

  // Granularity and geometry of this route's map (CTA-uniform). Only the
  // visible prefix is cleared, scattered, and scanned.
  const bool use_byte_map =
      active_logical_capacity <= static_cast<uint32_t>(kQTokenKvBlockSparseByteMapMaxBlocks);
  const int blocks_per_word = use_byte_map ? 4 : 32;
  const int map_words = static_cast<int>((active_logical_capacity + blocks_per_word - 1) /
                                         static_cast<uint32_t>(blocks_per_word));
  const int words_per_thread = QTokenKvBlockSparseWordsPerThread(map_words);
  const int map_stride = QTokenKvBlockSparseMapStride(words_per_thread);
  // Storage index of logical map word `word` (owner thread range + offset).
  auto map_index = [&](int word) -> int {
    if (words_per_thread == 1) {
      return word;
    }
    const int owner = word / words_per_thread;
    return owner * map_stride + (word - owner * words_per_thread);
  };
  const int word_begin = threadIdx.x * words_per_thread;
  const int word_end =
      word_begin + words_per_thread < map_words ? word_begin + words_per_thread : map_words;
  const int storage_begin = threadIdx.x * map_stride;

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
  constexpr int kSummaryWords = kQTokenKvBlockSparseBitmapSummaryWordsPerThread;
  uint32_t* my_summary = thread_word_summary + threadIdx.x * kSummaryWords;
  if (!use_byte_map) {
#pragma unroll
    for (int k = 0; k < kSummaryWords; ++k) {
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
      const int64_t complete_block_count = visible_tokens >> params.sparse_block_shift;
      const int32_t selected_count = static_cast<int32_t>(
          complete_block_count < params.block_topk ? complete_block_count : params.block_topk);
      if (query_item < static_cast<uint32_t>(selected_count)) {
        int32_t selected_block = prefetched_blocks[item];
        if constexpr (PackedQuery) {
          const int32_t row = shared.route.first_row + query;
          selected_block =
              params.block_indices[static_cast<int64_t>(row) * params.block_indices_row_stride +
                                   static_cast<int64_t>(head) * params.block_indices_head_stride +
                                   static_cast<int64_t>(query_item) *
                                       params.block_indices_column_stride];
        }
        if (selected_block >= 0 && selected_block < complete_block_count &&
            selected_block < params.model_block_bound) {
          logical_block = selected_block;
        }
      } else if (query_item == static_cast<uint32_t>(params.block_topk) &&
                 (visible_tokens & (params.sparse_block_size - 1)) != 0) {
        logical_block = visible_tokens >> params.sparse_block_shift;
      }
    }
    candidate_blocks[item] = -1;
    candidate_queries[item] = query;
    if (logical_block >= 0 && logical_block < static_cast<int64_t>(active_logical_capacity)) {
      const uint32_t block = static_cast<uint32_t>(logical_block);
      candidate_blocks[item] = static_cast<int32_t>(block);
      if (use_byte_map) {
        atomicOr(&block_map[map_index(block >> 2)], (uint32_t{1} << query) << ((block & 3u) * 8u));
      } else {
        const int word = static_cast<int>(block >> 5);
        const int owner = word / words_per_thread;
        const int offset = word - owner * words_per_thread;
        atomicOr(&block_map[owner * map_stride + offset], uint32_t{1} << (block & 31u));
        atomicOr(&thread_word_summary[owner * kSummaryWords + (offset >> 5)],
                 uint32_t{1} << (offset & 31));
      }
    }
  }
  __syncthreads();

  // Each thread owns a contiguous word range so the compact list stays in
  // ascending logical order; a block-wide exclusive sum ranks its entries.
  int local_unique_count = 0;
  if (use_byte_map) {
    for (int word = word_begin; word < word_end; ++word) {
      local_unique_count += __popc(__vsetne4(block_map[storage_begin + (word - word_begin)], 0u));
    }
  } else {
#pragma unroll
    for (int k = 0; k < kSummaryWords; ++k) {
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
  if (use_byte_map) {
    for (int word = word_begin; word < word_end; ++word) {
      const uint32_t value = block_map[storage_begin + (word - word_begin)];
      if (value == 0u) {
        continue;
      }
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const uint32_t membership = (value >> (byte * 8)) & 0xFFu;
        if (membership != 0u) {
          shared.compact_union[output_rank++] =
              static_cast<uint32_t>(word * 4 + byte) | (membership << 24);
        }
      }
    }
  } else {
#pragma unroll
    for (int k = 0; k < kSummaryWords; ++k) {
      uint32_t marks = my_summary[k];
      while (marks != 0u) {
        const int offset = k * 32 + __ffs(static_cast<int>(marks)) - 1;
        marks &= marks - 1u;
        const int word = word_begin + offset;
        uint32_t bits = block_map[storage_begin + offset];
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

  if (!use_byte_map) {
    // Bit-map membership pass: each accepted candidate recovers its block's
    // compact rank (owner thread's rank prefix + popcount of the preceding
    // words and lower bits) and ORs its query bit into that entry's high byte.
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int32_t block = candidate_blocks[item];
      if (block >= 0) {
        const int word = block >> 5;
        const int owner = word / words_per_thread;
        const int offset = word - owner * words_per_thread;
        int rank = shared.thread_rank_begin[owner];
        const int owner_storage = owner * map_stride;
        const uint32_t* owner_summary = thread_word_summary + owner * kSummaryWords;
        // Non-empty words of the owner's range below this word.
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
        rank += __popc(block_map[owner_storage + offset] & ((uint32_t{1} << (block & 31)) - 1u));
        atomicOr(&shared.compact_union[rank], (uint32_t{1} << candidate_queries[item]) << 24);
      }
    }
    __syncthreads();
  }

  // The map and its summaries are dead after membership reduction. Reuse
  // that storage for private query bytes instead of writing an intermediate
  // GMEM representation. The byte-map allocation covers every live union.
  if constexpr (QueryMajor) {
    static_assert(GroupSize == 8);
    group_memberships = reinterpret_cast<uint8_t*>(block_map);
  }
  // Resolve independent page-table loads before publishing each output batch.
  // A semantic block may cross multiple noncontiguous physical cache pages.
  constexpr int kEmitBatch = 4;
  const int union_fragments = union_pages * params.fragments_per_block;
  const int fragment_shift = __ffs(params.fragments_per_block) - 1;
  for (int rank_base = 0; rank_base < union_fragments; rank_base += kEmitBatch * kBlockThreads) {
    int32_t locators[kEmitBatch];
    uint32_t memberships[kEmitBatch];
#pragma unroll
    for (int j = 0; j < kEmitBatch; ++j) {
      const int rank = rank_base + j * kBlockThreads + threadIdx.x;
      locators[j] = -1;
      memberships[j] = 0;
      if (rank < union_fragments) {
        const uint32_t entry = shared.compact_union[rank >> fragment_shift];
        const uint32_t fragment =
            ((entry & 0xFFFFFFu) << fragment_shift) + (rank & (params.fragments_per_block - 1));
        const int64_t first_token = static_cast<int64_t>(fragment) * params.fragment_size;
        if (first_token <= shared.route.last_position) {
          locators[j] = ResolveFragment(params, shared.route, fragment);
        }
        uint32_t membership = entry >> 24;
        const int64_t first_visible_query = first_token - shared.route.first_position;
        if (first_visible_query > 0) {
          membership &= first_visible_query < 8 ? (0xffu << first_visible_query) : 0u;
        }
        memberships[j] = locators[j] >= 0 ? membership : 0;
      }
    }
#pragma unroll
    for (int j = 0; j < kEmitBatch; ++j) {
      const int rank = rank_base + j * kBlockThreads + threadIdx.x;
      if (rank < union_fragments) {
        group_indices[rank] = locators[j];
        group_memberships[rank] = static_cast<uint8_t>(memberships[j]);
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
        group_memberships[byte] = 0;
      }
    } else {
      group_indices[0] = -1;
      reinterpret_cast<uint32_t*>(group_memberships)[0] = 0;
      params.seq_lens[blockIdx.x] = 1;
    }
  }
  __syncthreads();
  if constexpr (QueryMajor) {
    StoreQueryMajorMemberships<PositionType, kBlockThreads>(params, group_memberships,
                                                            shared.union_pages);
  }
}

// Cached per host thread, device, and kernel specialization: no shared mutable
// host state, device-count limit, or repeated setup in a prepared launch.
struct QTokenKvBlockSparseUnionLaunchState {
  bool device_initialized = false;
  bool map_initialized = false;
  int max_dynamic_smem_bytes = 0;
  int sm_count = 0;
  int sort_ctas_per_sm = 0;
  std::unordered_map<size_t, int> map_ctas_per_sm;
};

template <typename PositionType, int GroupSize, bool PackedQuery, bool QueryMajor = false>
cudaError_t LaunchQTokenKvBlockSparseTouchedMetadataTyped(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, cudaStream_t stream) {
  constexpr int kSortBlockThreads =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kBlockThreads;
  auto map_kernel =
      QTokenKvBlockSparseBitmapMetadataKernel<PositionType, GroupSize, PackedQuery, QueryMajor>;
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

  const size_t map_smem_bytes =
      QTokenKvBlockSparseBitmapSmemBytes<GroupSize>(params.model_block_bound);
  const int map_words_per_thread =
      QTokenKvBlockSparseWordsPerThread(QTokenKvBlockSparseMapWords(params.model_block_bound));
  bool use_map = map_smem_bytes <= static_cast<size_t>(state.max_dynamic_smem_bytes) &&
                 map_words_per_thread <= kQTokenKvBlockSparseBitmapMaxWordsPerThread;
  if (use_map) {
    if (!state.map_initialized) {
      // Always opt in to the same maximum, not a per-launch value that another
      // host thread could lower between configuration and launch.
      status = cudaFuncSetAttribute(map_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    state.max_dynamic_smem_bytes);
      if (status != cudaSuccess) return status;
      state.map_initialized = true;
    }
    auto occupancy = state.map_ctas_per_sm.find(map_smem_bytes);
    if (occupancy == state.map_ctas_per_sm.end()) {
      int active_ctas = 0;
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_ctas, map_kernel, kQTokenKvBlockSparseBitmapBlockThreads, map_smem_bytes);
      if (status != cudaSuccess) return status;
      occupancy = state.map_ctas_per_sm.emplace(map_smem_bytes, active_ctas).first;
    }
    const int map_ctas = occupancy->second;
    use_map = map_ctas > 0;
    if (use_map && state.sort_ctas_per_sm > 0) {
      // Use actual occupancy (including registers and the GroupSize-specific
      // sort topology), cached once per footprint rather than queried per run.
      const int64_t map_slots = static_cast<int64_t>(state.sm_count) * map_ctas;
      const int64_t sort_slots = static_cast<int64_t>(state.sm_count) * state.sort_ctas_per_sm;
      use_map = (routes + map_slots - 1) / map_slots <= (routes + sort_slots - 1) / sort_slots;
    }
  }
  if (use_map) {
    map_kernel<<<routes, kQTokenKvBlockSparseBitmapBlockThreads, map_smem_bytes, stream>>>(params);
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
    if (group_size != 8 || params.sparse_block_size != 4 || params.fragment_size != 4) {
      return cudaErrorInvalidValue;
    }
    return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 8, PackedQuery,
                                                                 true>(params, stream);
  }
  switch (group_size) {
    case 1:
      return detail::LaunchQTokenKvBlockSparseQ1MetadataTyped<PositionType, PackedQuery>(params,
                                                                                         stream);
    case 2:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 2, PackedQuery>(
          params, stream);
    case 3:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 3, PackedQuery>(
          params, stream);
    case 4:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 4, PackedQuery>(
          params, stream);
    case 5:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 5, PackedQuery>(
          params, stream);
    case 6:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 6, PackedQuery>(
          params, stream);
    case 7:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 7, PackedQuery>(
          params, stream);
    case 8:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 8, PackedQuery>(
          params, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace prims_ts
}  // namespace attention
}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PRIMS_TS_Q_TOKEN_KV_BLOCK_SPARSE_METADATA_CUH_
