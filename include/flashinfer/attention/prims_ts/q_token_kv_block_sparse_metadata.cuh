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

namespace flashinfer {
namespace attention {
namespace prims_ts {

constexpr int kQTokenKvBlockSparseSparseBlockSize = 4;
constexpr int kQTokenKvBlockSparseMaxBlockTopK = 512;
constexpr int kQTokenKvBlockSparseMembershipsPerWord = 4;
constexpr int kQTokenKvBlockSparseQ1BlockThreads = 256;

template <int GroupSize>
struct QTokenKvBlockSparseTouchedMetadataKernelTraits;

// Keep enough independent warps to fill a Blackwell wave when decode exposes
// only a few query groups.  The per-group traits keep schedule tuning separate
// from metadata semantics while covering G * (512 selected blocks + one
// causal-tail block).
template <>
struct QTokenKvBlockSparseTouchedMetadataKernelTraits<2> {
  static constexpr int kBlockThreads = 384;
  static constexpr int kItemsPerThread = 3;
};

template <>
struct QTokenKvBlockSparseTouchedMetadataKernelTraits<4> {
  static constexpr int kBlockThreads = 544;
  static constexpr int kItemsPerThread = 4;
};

template <>
struct QTokenKvBlockSparseTouchedMetadataKernelTraits<5> {
  static constexpr int kBlockThreads = 672;
  static constexpr int kItemsPerThread = 4;
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

  uint_fastdiv candidates_per_query;
  uint_fastdiv subpages_per_storage_page;
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
  if (threadIdx.x != 0) {
    return;
  }

  int32_t first_row;
  int32_t row_end;
  bool valid;
  if constexpr (PackedQuery) {
    first_row = params.qo_indptr[blockIdx.x];
    row_end = params.qo_indptr[blockIdx.x + 1];
    valid = first_row >= 0 && row_end > first_row && row_end <= params.rows &&
            row_end - first_row <= GroupSize;
  } else {
    first_row = static_cast<int32_t>(blockIdx.x) * GroupSize;
    row_end = first_row + GroupSize;
    valid = row_end <= params.rows;
  }

  int32_t request = -1;
  int64_t first_position = -1;
  int64_t last_position = -1;
  if (valid) {
    request = params.token_to_request[first_row];
    first_position = static_cast<int64_t>(params.query_positions[first_row]);
    last_position = static_cast<int64_t>(params.query_positions[row_end - 1]);
    // Bound both endpoints before subtracting. Besides expressing the route
    // contract directly, this avoids signed overflow for malformed Int64
    // positions supplied to the synchronization-free device validator.
    valid = request >= 0 && request < params.num_requests && first_position >= 0 &&
            first_position < params.max_seq_len_kv && last_position >= first_position &&
            last_position < params.max_seq_len_kv &&
            last_position - first_position == (row_end - first_row) - 1;

#pragma unroll
    for (int query = 1; query < GroupSize && valid; ++query) {
      if (query < row_end - first_row) {
        valid = params.token_to_request[first_row + query] == request &&
                static_cast<int64_t>(params.query_positions[first_row + query]) ==
                    first_position + query;
      }
    }
  }

  route->valid = valid;
  route->request = request;
  route->first_row = first_row;
  route->query_count = valid ? row_end - first_row : 0;
  route->first_position = first_position;
  route->last_position = last_position;
}

template <typename PositionType>
__device__ __forceinline__ void StorePageMetadata(
    const QTokenKvBlockSparseTouchedMetadataParams<PositionType>& params,
    const QTokenKvBlockSparseRouteState& route, uint32_t logical_block, uint8_t membership,
    int32_t output_rank, int32_t* group_indices, uint8_t* group_memberships) {
  uint32_t storage_page;
  uint32_t subpage;
  params.subpages_per_storage_page.divmod(logical_block, storage_page, subpage);

  int32_t locator = -1;
  uint8_t output_membership = 0;
  if (storage_page < static_cast<uint32_t>(params.page_table_width)) {
    const int32_t physical_page =
        params.block_table[static_cast<int64_t>(route.request) * params.block_table_request_stride +
                           static_cast<int64_t>(storage_page) * params.block_table_page_stride];
    if (physical_page >= 0) {
      // The prepared plan proves that the cache's largest encoded locator
      // fits signed Int32. Dense block-table entries are trusted cache-page
      // IDs under the same contract, so no per-union-page Int64 proof belongs
      // in this hot path.
      locator = static_cast<int32_t>(static_cast<uint32_t>(physical_page) *
                                         static_cast<uint32_t>(params.subpages_per_storage_page) +
                                     subpage);
      output_membership = membership;
    }
  }
  group_indices[output_rank] = locator;
  group_memberships[output_rank] = output_membership;
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
  const int64_t visible_tokens = route.valid ? route.first_position + 1 : 0;
  const int32_t complete_blocks =
      static_cast<int32_t>(visible_tokens / kQTokenKvBlockSparseSparseBlockSize < params.block_topk
                               ? visible_tokens / kQTokenKvBlockSparseSparseBlockSize
                               : params.block_topk);

  // Initialize the complete fixed-capacity row on every replay. Attention may
  // speculatively load a rounded page tile before token-lane predicates apply.
  for (int32_t output_rank = threadIdx.x; output_rank < params.page_capacity;
       output_rank += kQTokenKvBlockSparseQ1BlockThreads) {
    int32_t locator = -1;
    if (route.valid && output_rank < complete_blocks) {
      const int32_t logical_block = params.block_indices[static_cast<int64_t>(route.first_row) *
                                                             params.block_indices_row_stride +
                                                         static_cast<int64_t>(output_rank) *
                                                             params.block_indices_column_stride];
      if (logical_block >= 0 &&
          logical_block < visible_tokens / kQTokenKvBlockSparseSparseBlockSize &&
          logical_block < params.model_block_bound) {
        uint32_t storage_page;
        uint32_t subpage;
        params.subpages_per_storage_page.divmod(static_cast<uint32_t>(logical_block), storage_page,
                                                subpage);
        if (storage_page < static_cast<uint32_t>(params.page_table_width)) {
          const int32_t physical_page =
              params
                  .block_table[static_cast<int64_t>(route.request) *
                                   params.block_table_request_stride +
                               static_cast<int64_t>(storage_page) * params.block_table_page_stride];
          if (physical_page >= 0) {
            locator =
                static_cast<int32_t>(static_cast<uint32_t>(physical_page) *
                                         static_cast<uint32_t>(params.subpages_per_storage_page) +
                                     subpage);
          }
        }
      }
    }
    row_indices[output_rank] = locator;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int32_t tail_tokens =
        route.valid ? static_cast<int32_t>(visible_tokens % kQTokenKvBlockSparseSparseBlockSize)
                    : 0;
    if (tail_tokens != 0) {
      const uint32_t tail_logical_block =
          static_cast<uint32_t>(visible_tokens / kQTokenKvBlockSparseSparseBlockSize);
      uint32_t storage_page;
      uint32_t subpage;
      params.subpages_per_storage_page.divmod(tail_logical_block, storage_page, subpage);
      if (storage_page < static_cast<uint32_t>(params.page_table_width)) {
        const int32_t physical_page =
            params.block_table[static_cast<int64_t>(route.request) *
                                   params.block_table_request_stride +
                               static_cast<int64_t>(storage_page) * params.block_table_page_stride];
        if (physical_page >= 0) {
          row_indices[complete_blocks] =
              static_cast<int32_t>(static_cast<uint32_t>(physical_page) *
                                       static_cast<uint32_t>(params.subpages_per_storage_page) +
                                   subpage);
        }
      }
    }
    const int32_t compact_length =
        complete_blocks * kQTokenKvBlockSparseSparseBlockSize + tail_tokens;
    params.seq_lens[blockIdx.x] = route.valid ? (compact_length > 0 ? compact_length : 1) : 1;
  }
  __syncthreads();

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.release_pdl) {
    // Keep the CTA-uniform CUDA builtin semantics. Repeated thread-level PTX
    // invocations have no additional effect after this CTA has signaled.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

template <typename PositionType, int GroupSize>
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
      const int64_t complete_block_count = visible_tokens / kQTokenKvBlockSparseSparseBlockSize;
      const int32_t selected_count = static_cast<int32_t>(
          complete_block_count < params.block_topk ? complete_block_count : params.block_topk);
      if (query_item < static_cast<uint32_t>(selected_count)) {
        const int32_t row = shared.route.first_row + query;
        const int32_t selected_block =
            params.block_indices[static_cast<int64_t>(row) * params.block_indices_row_stride +
                                 static_cast<int64_t>(query_item) *
                                     params.block_indices_column_stride];
        // Selected IDs may only name complete causal blocks. The partial
        // causal tail is synthesized separately and must remain the final
        // logical block for attention's tail-only mask.
        if (selected_block >= 0 && selected_block < complete_block_count &&
            selected_block < params.model_block_bound) {
          logical_block = selected_block;
        }
      } else if (query_item == static_cast<uint32_t>(params.block_topk) &&
                 visible_tokens % kQTokenKvBlockSparseSparseBlockSize != 0) {
        logical_block = visible_tokens / kQTokenKvBlockSparseSparseBlockSize;
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

template <typename PositionType, int GroupSize, bool PackedQuery>
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
  // kernel has no producer dependency; its terminal release may launch the
  // prepared attention consumer.
  if (threadIdx.x == 0) {
    shared.route = {0, -1, 0, 0, -1, -1};
    shared.union_pages = 0;
  }
  __syncthreads();
  InitRoute<PositionType, GroupSize, PackedQuery>(params, &shared.route);
  __syncthreads();

  const int64_t causal_block_bound =
      shared.route.valid ? (shared.route.last_position + kQTokenKvBlockSparseSparseBlockSize) /
                               kQTokenKvBlockSparseSparseBlockSize
                         : 0;
  const uint32_t active_logical_capacity = static_cast<uint32_t>(
      causal_block_bound < params.model_block_bound ? causal_block_bound
                                                    : params.model_block_bound);
  int32_t* group_indices = params.q_token_kv_block_sparse_page_indices +
                           static_cast<int64_t>(blockIdx.x) * params.page_capacity;
  uint8_t* group_memberships =
      reinterpret_cast<uint8_t*>(params.q_token_kv_block_sparse_page_memberships +
                                 static_cast<int64_t>(blockIdx.x) * params.membership_words);

  // bit_width(N) leaves an all-ones sentinel strictly above every live
  // [0, N) key, including when N is a power of two.
  const int active_radix_end_bit_unclamped =
      active_logical_capacity == 0 ? 1 : 32 - __clz(active_logical_capacity);
  const int active_radix_end_bit = active_radix_end_bit_unclamped < params.model_radix_end_bit
                                       ? active_radix_end_bit_unclamped
                                       : params.model_radix_end_bit;
  const uint32_t low_mask = (uint32_t{1} << active_radix_end_bit) - 1;
  const int union_pages = BuildTouchedUnion<PositionType, GroupSize>(
      params, shared, active_logical_capacity, active_radix_end_bit, low_mask, group_indices,
      group_memberships);
  if (threadIdx.x == 0) {
    shared.union_pages = union_pages;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (shared.route.valid && shared.union_pages > 0) {
      const int tail_tokens =
          static_cast<int>((shared.route.last_position + 1) % kQTokenKvBlockSparseSparseBlockSize);
      const int tail_padding =
          tail_tokens == 0 ? 0 : kQTokenKvBlockSparseSparseBlockSize - tail_tokens;
      params.seq_lens[blockIdx.x] =
          shared.union_pages * kQTokenKvBlockSparseSparseBlockSize - tail_padding;
      // Attention reads packed Uint32 words. Initialize only the last word's
      // padding bytes so that its masked load never reads unwritten memory.
      for (int byte = shared.union_pages; byte % kQTokenKvBlockSparseMembershipsPerWord != 0;
           ++byte) {
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

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.release_pdl) {
    // Every CTA, including an inert packed route, executes the release after
    // all metadata stores and CTA barriers. Every thread executes the
    // CTA-scoped signal uniformly; repeated invocations have no extra effect.
    // The dependent attention grid's wait establishes visibility.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

template <typename PositionType, int GroupSize, bool PackedQuery>
cudaError_t LaunchQTokenKvBlockSparseTouchedMetadataTyped(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, cudaStream_t stream) {
  constexpr int kBlockThreads =
      QTokenKvBlockSparseTouchedMetadataKernelTraits<GroupSize>::kBlockThreads;
  auto kernel = QTokenKvBlockSparseTouchedMetadataKernel<PositionType, GroupSize, PackedQuery>;
  kernel<<<params.groups, kBlockThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

template <typename PositionType, bool PackedQuery>
cudaError_t LaunchQTokenKvBlockSparseQ1MetadataTyped(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, cudaStream_t stream) {
  auto kernel = QTokenKvBlockSparseQ1MetadataKernel<PositionType, PackedQuery>;
  kernel<<<params.groups, kQTokenKvBlockSparseQ1BlockThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace detail

template <typename PositionType, bool PackedQuery>
cudaError_t LaunchQTokenKvBlockSparseTouchedMetadata(
    QTokenKvBlockSparseTouchedMetadataParams<PositionType> params, int32_t group_size,
    cudaStream_t stream) {
  static_assert(std::is_same_v<PositionType, int32_t> || std::is_same_v<PositionType, int64_t>);
  if (params.groups == 0) {
    return cudaSuccess;
  }
  switch (group_size) {
    case 1:
      return detail::LaunchQTokenKvBlockSparseQ1MetadataTyped<PositionType, PackedQuery>(params,
                                                                                         stream);
    case 2:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 2, PackedQuery>(
          params, stream);
    case 4:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 4, PackedQuery>(
          params, stream);
    case 5:
      return detail::LaunchQTokenKvBlockSparseTouchedMetadataTyped<PositionType, 5, PackedQuery>(
          params, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace prims_ts
}  // namespace attention
}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PRIMS_TS_Q_TOKEN_KV_BLOCK_SPARSE_METADATA_CUH_
