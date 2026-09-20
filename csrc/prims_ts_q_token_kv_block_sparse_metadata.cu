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

#include <algorithm>
#include <cstdint>
#include <flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.cuh>
#include <limits>
#include <numeric>

#include "tvm_ffi_utils.h"

namespace {

using flashinfer::attention::prims_ts::kQTokenKvBlockSparseMaxBlockTopK;
using flashinfer::attention::prims_ts::kQTokenKvBlockSparseMembershipsPerWord;
using flashinfer::attention::prims_ts::LaunchQTokenKvBlockSparseTouchedMetadata;
using flashinfer::attention::prims_ts::QTokenKvBlockSparseTouchedMetadataParams;

struct QTokenKvBlockSparseMetadataGeometry {
  int32_t rows;
  int32_t groups;
  int32_t num_requests;
  int32_t page_table_width;
  int32_t block_topk;
  int32_t page_capacity;
  int32_t membership_words;
  int32_t storage_page_size;
  int32_t max_seq_len_kv;
  int32_t model_block_bound;
  int32_t model_radix_end_bit;
  int32_t sparse_block_size;
  int32_t fragment_size;
  int32_t pattern_heads;
};

void CheckInt32CUDA(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(tensor.dtype()), int32_code)
      << name << " must have dtype int32";
}

void CheckSameDevice(const TensorView& reference, const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, reference.device().device_type)
      << name << " must be on the same device as block_indices";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, reference.device().device_id)
      << name << " must be on the same device as block_indices";
}

int32_t BitWidth(uint32_t value) {
  int32_t bits = 0;
  for (; value != 0; value >>= 1) {
    ++bits;
  }
  return bits;
}

template <bool PackedQuery, bool QueryMajor = false>
QTokenKvBlockSparseMetadataGeometry ValidateInputs(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, const TensorView* qo_indptr,
    TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv) {
  CheckInt32CUDA(block_indices, "block_indices");
  CheckInt32CUDA(block_table, "block_table");
  CheckInt32CUDA(token_to_request, "token_to_request");
  CheckInt32CUDA(q_token_kv_block_sparse_page_indices, "q_token_kv_block_sparse_page_indices");
  CheckInt32CUDA(q_token_kv_block_sparse_page_memberships,
                 "q_token_kv_block_sparse_page_memberships");
  CheckInt32CUDA(seq_lens, "seq_lens");
  TVM_FFI_ICHECK_EQ(query_positions.device().device_type, kDLCUDA)
      << "query_positions must be a CUDA tensor";
  const int64_t position_dtype = encode_dlpack_dtype(query_positions.dtype());
  TVM_FFI_ICHECK(position_dtype == int32_code || position_dtype == int64_code)
      << "query_positions must have dtype int32 or int64";

  CheckSameDevice(block_indices, block_table, "block_table");
  CheckSameDevice(block_indices, token_to_request, "token_to_request");
  CheckSameDevice(block_indices, query_positions, "query_positions");
  CheckSameDevice(block_indices, q_token_kv_block_sparse_page_indices,
                  "q_token_kv_block_sparse_page_indices");
  CheckSameDevice(block_indices, q_token_kv_block_sparse_page_memberships,
                  "q_token_kv_block_sparse_page_memberships");
  CheckSameDevice(block_indices, seq_lens, "seq_lens");
  if constexpr (PackedQuery) {
    TVM_FFI_ICHECK(qo_indptr != nullptr)
        << "packed QToken-KvBlock-Sparse-Attention requires qo_indptr";
    CheckInt32CUDA(*qo_indptr, "qo_indptr");
    CheckSameDevice(block_indices, *qo_indptr, "qo_indptr");
  }

  TVM_FFI_ICHECK(group_size >= 1 && group_size <= 8) << "group_size must be in [1, 8]";
  TVM_FFI_ICHECK(sparse_block_size >= 4 && sparse_block_size <= 128 &&
                 (sparse_block_size & (sparse_block_size - 1)) == 0)
      << "sparse_block_size must be 4, 8, 16, 32, 64, or 128";
  TVM_FFI_ICHECK(storage_page_size >= 4 && storage_page_size % 4 == 0 &&
                 storage_page_size <= std::numeric_limits<int32_t>::max())
      << "storage_page_size must be an int32 positive multiple of four";
  const int64_t fragment_size = std::gcd(storage_page_size, sparse_block_size);
  TVM_FFI_ICHECK(max_seq_len_kv > 0 && max_seq_len_kv <= std::numeric_limits<int32_t>::max())
      << "max_seq_len_kv must fit positive int32";
  const int64_t expected_model_block_bound =
      (max_seq_len_kv + sparse_block_size - 1) / sparse_block_size;

  TVM_FFI_ICHECK(block_indices.ndim() == 2 || block_indices.ndim() == 3)
      << "block_indices must be [T,K] or [T,Hkv,K]";
  const int topk_axis = block_indices.ndim() - 1;
  const int64_t pattern_heads = block_indices.ndim() == 3 ? block_indices.size(1) : 1;
  TVM_FFI_ICHECK(pattern_heads > 0 && pattern_heads <= std::numeric_limits<int32_t>::max())
      << "pattern head count must fit positive int32";
  TVM_FFI_ICHECK(block_indices.size(0) >= 0 &&
                 block_indices.size(0) <= std::numeric_limits<int32_t>::max())
      << "block_indices row count must fit int32";
  TVM_FFI_ICHECK(block_indices.size(topk_axis) > 0 &&
                 block_indices.size(topk_axis) <= kQTokenKvBlockSparseMaxBlockTopK)
      << "block_indices topk must be in [1, 512]";
  TVM_FFI_ICHECK_EQ(block_indices.stride(topk_axis), 1) << "block_indices rows must be contiguous";

  const int64_t rows = block_indices.size(0);
  const int64_t block_topk = block_indices.size(topk_axis);
  TVM_FFI_ICHECK_EQ(token_to_request.ndim(), 1) << "token_to_request must be rank one";
  TVM_FFI_ICHECK_EQ(token_to_request.size(0), rows)
      << "token_to_request must have one entry per query row";
  TVM_FFI_ICHECK_EQ(token_to_request.stride(0), 1) << "token_to_request must be contiguous";
  TVM_FFI_ICHECK_EQ(query_positions.ndim(), 1) << "query_positions must be rank one";
  TVM_FFI_ICHECK_EQ(query_positions.size(0), rows)
      << "query_positions must have one entry per query row";
  TVM_FFI_ICHECK_EQ(query_positions.stride(0), 1) << "query_positions must be contiguous";

  TVM_FFI_ICHECK_EQ(block_table.ndim(), 2) << "block_table must be rank two";
  TVM_FFI_ICHECK(block_table.size(0) > 0 && block_table.size(1) > 0)
      << "block_table must be nonempty";
  TVM_FFI_ICHECK(block_table.size(0) <= std::numeric_limits<int32_t>::max() &&
                 block_table.size(1) <= std::numeric_limits<int32_t>::max())
      << "block_table dimensions must fit int32";
  TVM_FFI_ICHECK_EQ(block_table.stride(1), 1) << "block_table rows must be contiguous";
  TVM_FFI_ICHECK_GE(block_table.stride(0), block_table.size(1))
      << "block_table must be a dense row-strided page table, not CSR storage";

  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1) << "seq_lens must be rank one";
  TVM_FFI_ICHECK(seq_lens.size(0) >= 0 && seq_lens.size(0) <= std::numeric_limits<int32_t>::max())
      << "query-group count must fit int32";
  TVM_FFI_ICHECK(seq_lens.IsContiguous()) << "seq_lens must be contiguous";
  TVM_FFI_ICHECK_EQ(seq_lens.size(0) % pattern_heads, 0)
      << "metadata rows must be divisible by pattern head count";
  const int64_t groups = seq_lens.size(0) / pattern_heads;
  const int64_t metadata_rows = groups * pattern_heads;
  if constexpr (PackedQuery) {
    TVM_FFI_ICHECK_EQ(qo_indptr->ndim(), 1) << "qo_indptr must be rank one";
    TVM_FFI_ICHECK_EQ(qo_indptr->size(0), groups + 1) << "qo_indptr must have groups + 1 entries";
    TVM_FFI_ICHECK(qo_indptr->IsContiguous()) << "qo_indptr must be contiguous";
    TVM_FFI_ICHECK(groups > 0 && rows > 0)
        << "packed QToken-KvBlock-Sparse-Attention requires at least one nonempty route";
  } else {
    TVM_FFI_ICHECK_EQ(rows, groups * group_size)
        << "fixed QToken-KvBlock-Sparse-Attention rows must equal groups * group_size";
  }
  const int64_t max_block_capacity = group_size * (block_topk + 1);
  const int64_t min_block_capacity = std::min(max_block_capacity, expected_model_block_bound);
  TVM_FFI_ICHECK_EQ(q_token_kv_block_sparse_page_indices.ndim(), 2)
      << "q_token_kv_block_sparse_page_indices must be rank two";
  const int64_t page_capacity = q_token_kv_block_sparse_page_indices.size(1);
  int64_t minimum_pages = min_block_capacity * (sparse_block_size / fragment_size);
  int64_t maximum_pages = max_block_capacity * (sparse_block_size / fragment_size);
  if constexpr (QueryMajor) {
    TVM_FFI_ICHECK(group_size == 8 && sparse_block_size == 4 && fragment_size == 4)
        << "query-major metadata requires G8 and page-4 sparse blocks";
    minimum_pages = (minimum_pages + 31) / 32 * 32;
    maximum_pages = (maximum_pages + 31) / 32 * 32;
    TVM_FFI_ICHECK_EQ(page_capacity % 32, 0) << "query-major capacity must cover complete tiles";
  }
  TVM_FFI_ICHECK(page_capacity >= minimum_pages && page_capacity <= maximum_pages)
      << "sparse fragment capacity must cover the logical union bound";
  const int64_t membership_words =
      group_size == 1 ? 0
                      : (page_capacity + kQTokenKvBlockSparseMembershipsPerWord - 1) /
                            kQTokenKvBlockSparseMembershipsPerWord;
  TVM_FFI_ICHECK(page_capacity <= std::numeric_limits<int32_t>::max())
      << "QToken-KvBlock-Sparse-Attention page capacity must fit int32";
  TVM_FFI_ICHECK_EQ(q_token_kv_block_sparse_page_indices.size(0), metadata_rows)
      << "q_token_kv_block_sparse_page_indices group count mismatch";
  TVM_FFI_ICHECK(q_token_kv_block_sparse_page_indices.IsContiguous())
      << "q_token_kv_block_sparse_page_indices must be contiguous";
  TVM_FFI_ICHECK_EQ(q_token_kv_block_sparse_page_memberships.ndim(), 2)
      << "q_token_kv_block_sparse_page_memberships must be rank two";
  TVM_FFI_ICHECK_EQ(q_token_kv_block_sparse_page_memberships.size(0), metadata_rows)
      << "q_token_kv_block_sparse_page_memberships group count mismatch";
  TVM_FFI_ICHECK_EQ(q_token_kv_block_sparse_page_memberships.size(1), membership_words)
      << "q_token_kv_block_sparse_page_memberships must pack four membership bytes per int32 word";
  // DLPack does not require meaningful strides for a zero-element tensor, and
  // some FFI conversions therefore report [groups, 0] as non-contiguous. Q1
  // deliberately uses that shape because the direct route has no membership
  // table; no kernel dereferences its pointer.
  TVM_FFI_ICHECK(membership_words == 0 || q_token_kv_block_sparse_page_memberships.IsContiguous())
      << "q_token_kv_block_sparse_page_memberships must be contiguous";

  TVM_FFI_ICHECK_LE(max_seq_len_kv, block_table.size(1) * storage_page_size)
      << "the dense block table does not address max_seq_len_kv";
  const int32_t model_radix_end_bit = BitWidth(static_cast<uint32_t>(expected_model_block_bound));
  // The sorted key retains a three-bit query tag above the radix-sorted
  // logical-ID field. Reserve those tag bits and an out-of-range sentinel.
  TVM_FFI_ICHECK_LE(model_radix_end_bit, 29)
      << "model_block_bound leaves no room for the three-bit query tag";

  return {static_cast<int32_t>(rows),
          static_cast<int32_t>(groups),
          static_cast<int32_t>(block_table.size(0)),
          static_cast<int32_t>(block_table.size(1)),
          static_cast<int32_t>(block_topk),
          static_cast<int32_t>(page_capacity),
          static_cast<int32_t>(membership_words),
          static_cast<int32_t>(storage_page_size),
          static_cast<int32_t>(max_seq_len_kv),
          static_cast<int32_t>(expected_model_block_bound),
          model_radix_end_bit,
          static_cast<int32_t>(sparse_block_size),
          static_cast<int32_t>(fragment_size),
          static_cast<int32_t>(pattern_heads)};
}

template <typename PositionType, bool PackedQuery, bool QueryMajor = false>
void Launch(TensorView block_indices, TensorView block_table, TensorView token_to_request,
            TensorView query_positions, const TensorView* qo_indptr,
            TensorView q_token_kv_block_sparse_page_indices,
            TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens,
            const QTokenKvBlockSparseMetadataGeometry& geometry, int32_t group_size,
            bool release_pdl) {
  const int topk_axis = block_indices.ndim() - 1;
  QTokenKvBlockSparseTouchedMetadataParams<PositionType> params{
      static_cast<const int32_t*>(block_indices.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()),
      static_cast<const int32_t*>(token_to_request.data_ptr()),
      static_cast<const PositionType*>(query_positions.data_ptr()),
      PackedQuery ? static_cast<const int32_t*>(qo_indptr->data_ptr()) : nullptr,
      static_cast<int32_t*>(q_token_kv_block_sparse_page_indices.data_ptr()),
      static_cast<int32_t*>(q_token_kv_block_sparse_page_memberships.data_ptr()),
      static_cast<int32_t*>(seq_lens.data_ptr()),
      block_indices.stride(0),
      block_indices.stride(topk_axis),
      block_indices.ndim() == 3 ? block_indices.stride(1) : 0,
      block_table.stride(0),
      block_table.stride(1),
      geometry.rows,
      geometry.groups,
      geometry.num_requests,
      geometry.page_table_width,
      geometry.block_topk,
      geometry.page_capacity,
      geometry.membership_words,
      geometry.max_seq_len_kv,
      geometry.model_block_bound,
      geometry.model_radix_end_bit,
      geometry.sparse_block_size,
      BitWidth(geometry.sparse_block_size) - 1,
      geometry.fragment_size,
      geometry.sparse_block_size / geometry.fragment_size,
      flashinfer::uint_fastdiv(static_cast<uint32_t>(geometry.block_topk + 1)),
      flashinfer::uint_fastdiv(
          static_cast<uint32_t>(geometry.storage_page_size / geometry.fragment_size)),
      flashinfer::uint_fastdiv(static_cast<uint32_t>(geometry.pattern_heads)),
      release_pdl};

  ffi::CUDADeviceGuard device_guard(block_indices.device().device_id);
  const cudaStream_t stream = get_stream(block_indices.device());
  const cudaError_t status =
      LaunchQTokenKvBlockSparseTouchedMetadata<PositionType, PackedQuery, QueryMajor>(
          params, group_size, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "PrimTS QToken-KvBlock-Sparse-Attention metadata launch failed: "
      << cudaGetErrorString(status);
}

template <bool PackedQuery, bool QueryMajor = false>
void Run(TensorView block_indices, TensorView block_table, TensorView token_to_request,
         TensorView query_positions, const TensorView* qo_indptr,
         TensorView q_token_kv_block_sparse_page_indices,
         TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens,
         int64_t group_size, int64_t storage_page_size, int64_t sparse_block_size,
         int64_t max_seq_len_kv, bool release_pdl) {
  const QTokenKvBlockSparseMetadataGeometry geometry = ValidateInputs<PackedQuery, QueryMajor>(
      block_indices, block_table, token_to_request, query_positions, qo_indptr,
      q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens,
      group_size, storage_page_size, sparse_block_size, max_seq_len_kv);

  if (encode_dlpack_dtype(query_positions.dtype()) == int32_code) {
    Launch<int32_t, PackedQuery, QueryMajor>(
        block_indices, block_table, token_to_request, query_positions, qo_indptr,
        q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens,
        geometry, static_cast<int32_t>(group_size), release_pdl);
  } else {
    Launch<int64_t, PackedQuery, QueryMajor>(
        block_indices, block_table, token_to_request, query_positions, qo_indptr,
        q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens,
        geometry, static_cast<int32_t>(group_size), release_pdl);
  }
}

}  // namespace

void PrimsTSQTokenKvBlockSparseMetadataRunFixed(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv,
    bool release_pdl) {
  Run<false>(block_indices, block_table, token_to_request, query_positions, nullptr,
             q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships,
             seq_lens, group_size, storage_page_size, sparse_block_size, max_seq_len_kv,
             release_pdl);
}

void PrimsTSQTokenKvBlockSparseMetadataRunPacked(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, TensorView qo_indptr,
    TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv,
    bool release_pdl) {
  Run<true>(block_indices, block_table, token_to_request, query_positions, &qo_indptr,
            q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships,
            seq_lens, group_size, storage_page_size, sparse_block_size, max_seq_len_kv,
            release_pdl);
}

// Prepared-plan entries; standalone metadata APIs retain byte membership.
void PrimsTSQTokenKvBlockSparseMetadataRunFixedQueryMajor(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv,
    bool release_pdl) {
  Run<false, true>(block_indices, block_table, token_to_request, query_positions, nullptr,
                   q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships,
                   seq_lens, group_size, storage_page_size, sparse_block_size, max_seq_len_kv,
                   release_pdl);
}

void PrimsTSQTokenKvBlockSparseMetadataRunPackedQueryMajor(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, TensorView qo_indptr,
    TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv,
    bool release_pdl) {
  Run<true, true>(block_indices, block_table, token_to_request, query_positions, &qo_indptr,
                  q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships,
                  seq_lens, group_size, storage_page_size, sparse_block_size, max_seq_len_kv,
                  release_pdl);
}
