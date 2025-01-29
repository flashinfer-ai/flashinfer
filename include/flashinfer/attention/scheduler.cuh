/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_SCHEDULER_CUH_
#define FLASHINFER_ATTENTION_SCHEDULER_CUH_

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

#include "../allocator.h"
#include "../exception.h"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "heap.h"

namespace flashinfer {

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params);

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params);

/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 *   after we partition Paged KV-Cache into multiple chunks on KV sequence length
 *   dimension.
 * \tparam IdType A template type indicates the index data type
 * \param max_grid_size The maximum grid size of the kernel
 * \param gdy gridDim.y
 * \param num_pages The number of pages per request in the batch
 * \param max_num_pages_per_batch_lb The pre-set lower bound of maximum number of
 *   pages per batch, default to 1
 * \return (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 *   the new batch size after the partition.
 */
template <typename IdType>
inline auto PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    const uint32_t max_grid_size, const uint32_t gdy, const std::vector<IdType>& num_pages,
    const uint32_t min_num_pages_per_batch = 1) {
  uint32_t low = min_num_pages_per_batch, high = 0;
  for (const IdType& elem : num_pages) {
    high = max(high, elem);
  }
  uint32_t new_batch_size;
  while (low < high) {
    uint32_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (const IdType& elem : num_pages) {
      new_batch_size += ceil_div(elem, mid);
    }
    if (new_batch_size * gdy > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (const IdType& elem : num_pages) {
    new_batch_size += ceil_div(std::max(elem, 1), low);
  }
  return std::make_tuple(low, new_batch_size);
}

inline auto PrefillBinarySearchKVChunkSize(const bool enable_cuda_graph,
                                           const uint32_t max_batch_size_if_split,
                                           const std::vector<int64_t>& packed_qo_len_arr,
                                           const std::vector<int64_t>& kv_len_arr,
                                           const uint32_t qo_chunk_size,
                                           const uint32_t min_kv_chunk_size = 1) {
  const int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 1;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }

  int64_t low = min_kv_chunk_size;
  int64_t high = max_kv_len;
  constexpr int64_t min_kv_len = 1;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    int64_t new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                        ceil_div(std::max(kv_len_arr[i], min_kv_len), mid);
    }
    if (new_batch_size > max_batch_size_if_split) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return std::make_tuple(enable_cuda_graph || low < max_kv_len, low);
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param split_kv Whether to split the KV cache into multiple chunks
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param pos_encoding_mode The positional encoding mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;
    gdy = num_kv_heads;
    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                          vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }
    return cudaSuccess;
  })
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
    constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
    constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

    constexpr uint32_t bdy = 8;
    constexpr uint32_t tile_size_qo_heads = 2;
    constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    const uint32_t gdy = ceil_div(num_qo_heads, qo_heads_per_block);

    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }

    return cudaSuccess;
  });
}

/*!
 * \brief Partition Paged KV-Cache into multiple chunks on KV sequence length
 * \tparam IdType A template type indicates the index data type
 * \param old_batch_size The batch size of the old Paged KV-Cache
 * \param old_page_indptr_h The host-side page indptr of the old Paged KV-Cache
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_paged_kv_d The device-side new Paged KV-Cache
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename IdType>
inline auto DecodeSplitKVIndptr(IdType* indptr_h, uint32_t batch_size, uint32_t kv_chunk_size) {
  std::vector<IdType> request_indices, kv_tile_indices, o_indptr;
  o_indptr.push_back(0);

  for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    uint32_t num_tiles_kv = ceil_div(
        std::max<uint32_t>(indptr_h[batch_idx + 1] - indptr_h[batch_idx], 1U), kv_chunk_size);
    for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
      request_indices.push_back(batch_idx);
      kv_tile_indices.push_back(kv_tile_idx);
    }
    o_indptr.push_back(o_indptr.back() + num_tiles_kv);
  }

  return std::make_tuple(request_indices, kv_tile_indices, o_indptr);
}

struct DecodePlanInfo {
  int64_t padded_batch_size;
  int64_t v_offset;
  int64_t s_offset;
  int64_t request_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t o_indptr_offset;
  int64_t block_valid_mask_offset;
  int64_t kv_chunk_size_ptr_offset;
  bool enable_cuda_graph;
  bool split_kv;

  DecodePlanInfo()
      : padded_batch_size(0),
        v_offset(0),
        s_offset(0),
        request_indices_offset(0),
        kv_tile_indices_offset(0),
        o_indptr_offset(0),
        block_valid_mask_offset(0),
        kv_chunk_size_ptr_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert DecodePlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            v_offset,
            s_offset,
            request_indices_offset,
            kv_tile_indices_offset,
            o_indptr_offset,
            block_valid_mask_offset,
            kv_chunk_size_ptr_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to DecodePlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 10) {
      std::ostringstream err_msg;
      err_msg << "DecodePlanInfo::FromVector: vec.size() should be 10, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    v_offset = vec[1];
    s_offset = vec[2];
    request_indices_offset = vec[3];
    kv_tile_indices_offset = vec[4];
    o_indptr_offset = vec[5];
    block_valid_mask_offset = vec[6];
    kv_chunk_size_ptr_offset = vec[7];
    enable_cuda_graph = vec[8];
    split_kv = vec[9];
  }
};

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params, typename WorkEstimationFunc>
inline cudaError_t DecodePlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                              void* int_buffer, void* page_locked_int_buffer,
                              size_t int_workspace_size_in_bytes, DecodePlanInfo& plan_info,
                              typename Params::IdType* indptr_h, uint32_t batch_size,
                              uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                              cudaStream_t stream, WorkEstimationFunc work_estimation_func) {
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  bool split_kv;
  uint32_t max_grid_size, kv_chunk_size_in_pages, new_batch_size, gdy;

  FLASHINFER_CUDA_CALL(work_estimation_func(split_kv, max_grid_size, kv_chunk_size_in_pages,
                                            new_batch_size, gdy, batch_size, indptr_h, num_qo_heads,
                                            page_size, enable_cuda_graph, stream));
  size_t padded_batch_size;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.split_kv = split_kv;
  padded_batch_size =
      (enable_cuda_graph) ? (split_kv ? max_grid_size / gdy : batch_size) : new_batch_size;
  plan_info.padded_batch_size = padded_batch_size;

  auto [request_indices_vec, kv_tile_indices_vec, o_indptr_vec] =
      DecodeSplitKVIndptr(indptr_h, batch_size, kv_chunk_size_in_pages);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      padded_batch_size * sizeof(IdType), 16, "batch_decode_request_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      padded_batch_size * sizeof(IdType), 16, "batch_decode_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(
      (padded_batch_size + 1) * sizeof(IdType), 16, "batch_decode_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_decode_kv_chunk_size_ptr");
  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size_in_pages * page_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * HEAD_DIM * sizeof(DTypeO), 16, "batch_decode_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * sizeof(float), 16, "batch_decode_tmp_s");

    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        padded_batch_size * sizeof(bool), 16, "batch_decode_block_valid_mask");
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

inline uint32_t DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim) {
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    return 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        // avg_packed_qo_len <= 64
        return 64;
      } else {
        // avg_packed_qo_len <= 16
        return 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
      return 64;
    }
  }
}

template <typename IdType>
inline auto PrefillSplitQOKVIndptr(IdType* qo_indptr_h, IdType* kv_indptr_h,
                                   uint32_t total_num_rows, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, uint32_t max_batch_size_if_split,
                                   bool enable_cuda_graph) {
  std::vector<IdType> request_indices, qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr;
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;

  // step 1: determine packed_qo_len_arr and verify qo_indptr contents.
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    if (packed_qo_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    if (kv_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_indptr[" << i + 1 << "]" << kv_indptr_h[i + 1] << " - kv_indptr[" << i << "]"
              << kv_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
  }

  // step 2: determine cta_tile_q, kv_chunk_size and total_num_tiles_q
  const uint32_t min_kv_chunk_size = std::max((128 / page_size), 1U);
  uint32_t cta_tile_q;
  uint32_t total_num_tiles_q;
  if (enable_cuda_graph) {
    // When CUDA graphs are enabled, the lengths of sequences determined by
    // qo_indptr_h can vary. We assume that the dummy data based on which
    // the CUDA graph is created fixes the maximum number of tokens.
    const uint64_t max_seq_len = total_num_rows - batch_size + 1;
    uint64_t max_qo_len = uint64_t(max_seq_len) * gqa_group_size;
    cta_tile_q = DetermineCtaTileQ(max_qo_len, head_dim);

    // Find an upper bound for the number of tiles, derived from the total
    // number of rows and the batch size.  The sum of qo lengths rounded
    // up to cta_tile_q will not exceed this number derived from the total
    // number of rows.
    total_num_tiles_q = ceil_div(total_num_rows * gqa_group_size, cta_tile_q) + batch_size - 1;
  } else {
    int64_t sum_packed_qo_len = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      sum_packed_qo_len += packed_qo_len_arr[i];
    }
    const int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
    cta_tile_q = DetermineCtaTileQ(avg_packed_qo_len, head_dim);

    total_num_tiles_q = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      total_num_tiles_q += ceil_div(packed_qo_len_arr[i], cta_tile_q);
    }
  }

  auto [split_kv, kv_chunk_size] =
      PrefillBinarySearchKVChunkSize(enable_cuda_graph, max_batch_size_if_split, packed_qo_len_arr,
                                     kv_len_arr, cta_tile_q, min_kv_chunk_size);

  // step 3: split qo_indptr and kv_indptr
  uint32_t new_batch_size = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    const int64_t packed_qo_len = packed_qo_len_arr[request_idx];
    const int64_t kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    const int64_t num_tiles_q = ceil_div(packed_qo_len, cta_tile_q);
    const int64_t num_tiles_kv = ceil_div(kv_len, kv_chunk_size);

    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
        new_batch_size += 1;
        request_indices.push_back(request_idx);
        qo_tile_indices.push_back(q_tile_idx);
        kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      merge_indptr.push_back(merge_indptr.back() + num_tiles_kv);
    }
    o_indptr.push_back(o_indptr.back() + qo_len * num_tiles_kv);
  }

  const size_t padded_batch_size =
      enable_cuda_graph ? std::max(max_batch_size_if_split, total_num_tiles_q) : new_batch_size;
  FLASHINFER_CHECK(new_batch_size <= padded_batch_size,
                   "new batch size should not exceed padded batch size");

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return std::make_tuple(split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size,
                         std::move(request_indices), std::move(qo_tile_indices),
                         std::move(kv_tile_indices), std::move(merge_indptr), std::move(o_indptr));
}

struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t total_num_rows_offset;
  int64_t cta_tile_q;
  int64_t request_indices_offset;
  int64_t qo_tile_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t merge_indptr_offset;
  int64_t o_indptr_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t v_offset;
  int64_t s_offset;
  int64_t block_valid_mask_offset;
  bool enable_cuda_graph;
  bool split_kv;

  PrefillPlanInfo()
      : padded_batch_size(0),
        total_num_rows(0),
        total_num_rows_offset(0),
        cta_tile_q(0),
        request_indices_offset(0),
        qo_tile_indices_offset(0),
        kv_tile_indices_offset(0),
        merge_indptr_offset(0),
        o_indptr_offset(0),
        kv_chunk_size_ptr_offset(0),
        v_offset(0),
        s_offset(0),
        block_valid_mask_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert PrefillPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            total_num_rows,
            total_num_rows_offset,
            cta_tile_q,
            request_indices_offset,
            qo_tile_indices_offset,
            kv_tile_indices_offset,
            merge_indptr_offset,
            o_indptr_offset,
            kv_chunk_size_ptr_offset,
            v_offset,
            s_offset,
            block_valid_mask_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to PrefillPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 15) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 14, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    total_num_rows_offset = vec[2];
    cta_tile_q = vec[3];
    request_indices_offset = vec[4];
    qo_tile_indices_offset = vec[5];
    kv_tile_indices_offset = vec[6];
    merge_indptr_offset = vec[7];
    o_indptr_offset = vec[8];
    kv_chunk_size_ptr_offset = vec[9];
    v_offset = vec[10];
    s_offset = vec[11];
    block_valid_mask_offset = vec[12];
    enable_cuda_graph = vec[13];
    split_kv = vec[14];
  }
};

template <typename IdType>
inline cudaError_t PrefillPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                               void* int_buffer, void* page_locked_int_buffer,
                               size_t int_workspace_size_in_bytes, PrefillPlanInfo& plan_info,
                               IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t total_num_rows,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t head_dim, uint32_t page_size, bool enable_cuda_graph,
                               uint32_t sizeof_dtype_o, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  uint32_t max_batch_size_if_split = max_grid_size / num_kv_heads;

  // step 2: determine kv_chunk_size
  auto [split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size, request_indices_vec,
        qo_tile_indices_vec, kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec] =
      PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                             num_kv_heads, head_dim, page_size, max_batch_size_if_split,
                             enable_cuda_graph);

  plan_info.cta_tile_q = cta_tile_q;
  plan_info.total_num_rows = total_num_rows;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.padded_batch_size = padded_batch_size;
  plan_info.split_kv = split_kv;

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_request_indices");
  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_qo_tile_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * (batch_size + 1),
                                                                 16, "batch_prefill_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");

  if (plan_info.enable_cuda_graph) {
    plan_info.total_num_rows_offset =
        int_allocator.aligned_alloc_offset(sizeof(uint32_t), 16, "batch_prefill_total_num_rows");
    uint32_t* total_num_rows_h =
        GetPtrFromBaseOffset<uint32_t>(page_locked_int_buffer, plan_info.total_num_rows_offset);
    *total_num_rows_h = qo_indptr_h[batch_size];
  }

  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * head_dim * sizeof_dtype_o, 16,
        "batch_prefill_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * sizeof(float), 16, "batch_prefill_tmp_s");
    plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * (plan_info.total_num_rows + 1), 16, "batch_prefill_merge_indptr");
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");

    IdType* merge_indptr_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), merge_indptr_h);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  return cudaSuccess;
}

inline float cost_function(int qo_len, int kv_len, int group_size) {
  return 2 * float(qo_len) * float(group_size) + kv_len;
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec, int size_after_flatten) {
  std::vector<T> result;
  result.reserve(size_after_flatten);
  for (const auto& inner_vec : vec) {
    result.insert(result.end(), inner_vec.begin(), inner_vec.end());
  }
  return std::move(result);
}

struct PrefillPlanSM90Info {
  int64_t qo_tile_indices_offset;
  int64_t qo_indptr_offset;
  int64_t kv_indptr_offset;
  int64_t qo_len_offset;
  int64_t kv_len_offset;
  int64_t head_indices_offset;
  int64_t work_indptr_offset;
  bool same_schedule_for_all_heads;

  PrefillPlanSM90Info()
      : qo_tile_indices_offset(0),
        qo_indptr_offset(0),
        kv_indptr_offset(0),
        qo_len_offset(0),
        kv_len_offset(0),
        head_indices_offset(0),
        work_indptr_offset(0),
        same_schedule_for_all_heads(false) {}

  // convert PrefillPlanSM90Info to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {qo_tile_indices_offset, qo_indptr_offset,
            kv_indptr_offset,       qo_len_offset,
            kv_len_offset,          head_indices_offset,
            work_indptr_offset,     same_schedule_for_all_heads};
  }

  // From std::vector<int64_t> to PrefillPlanSM90Info
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 8) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanSM90Info::FromVector: vec.size() should be 8, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    qo_tile_indices_offset = vec[0];
    qo_indptr_offset = vec[1];
    kv_indptr_offset = vec[2];
    qo_len_offset = vec[3];
    kv_len_offset = vec[4];
    head_indices_offset = vec[5];
    work_indptr_offset = vec[6];
    same_schedule_for_all_heads = vec[7];
  }
};

template <typename IdType>
inline cudaError_t PrefillSM90Plan(void* float_buffer, size_t float_workspace_size_in_bytes,
                                   void* int_buffer, void* page_locked_int_buffer,
                                   size_t int_workspace_size_in_bytes,
                                   PrefillPlanSM90Info& plan_info, IdType* qo_indptr_h,
                                   IdType* kv_indptr_h, IdType* kv_len_arr_h,
                                   uint32_t total_num_rows, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, bool causal, bool enable_cuda_graph,
                                   uint32_t sizeof_dtype_o, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec;
  for (uint32_t i = 0; i < batch_size; ++i) {
    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int kv_len = kv_len_arr_h[i];
    if (kv_len < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_len[" << i << "]" << kv_len << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    if (qo_len < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    idx_qo_kv_len_vec.push_back({i, qo_len, kv_len});
  }

  std::sort(idx_qo_kv_len_vec.begin(), idx_qo_kv_len_vec.end(),
            [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
  int cta_tile_q = 128;
  if (head_dim == 64) {
    cta_tile_q = 192;
  }

  int device = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sm90_ctas = 0;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&num_sm90_ctas, cudaDevAttrMultiProcessorCount, device));

  CTACostHeap cta_cost_heap(num_sm90_ctas);
  std::vector<std::vector<IdType>> cta_qo_tile_indices(num_sm90_ctas, std::vector<IdType>()),
      cta_qo_indptr(num_sm90_ctas, std::vector<IdType>()),
      cta_kv_indptr(num_sm90_ctas, std::vector<IdType>()),
      cta_qo_len(num_sm90_ctas, std::vector<IdType>()),
      cta_kv_len(num_sm90_ctas, std::vector<IdType>()),
      cta_head_indices(num_sm90_ctas, std::vector<IdType>());

  int max_num_works_per_head = ceil_div(total_num_rows, cta_tile_q) + batch_size - 1;
  plan_info.same_schedule_for_all_heads = max_num_works_per_head > 4096;

  for (int qo_head_idx = 0;
       qo_head_idx < (plan_info.same_schedule_for_all_heads ? 1 : num_qo_heads); ++qo_head_idx) {
    for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) {
      int num_qo_tiles = ceil_div(qo_len, cta_tile_q);
      for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
        auto [cta_idx, accum_cost] = cta_cost_heap.pop();
        // NOTE(Zihao): our current FA3 implementation do not fuse query and group heads
        // so the group_size in cost_function is always 1
        cta_cost_heap.insert(
            {cta_idx,
             accum_cost + cost_function(cta_tile_q,
                                        causal
                                            ? kv_len - (num_qo_tiles - qo_tile_idx - 1) * cta_tile_q
                                            : kv_len,
                                        /*group_size=*/1)});
        cta_qo_tile_indices[cta_idx].push_back(qo_tile_idx);
        cta_qo_indptr[cta_idx].push_back(qo_indptr_h[i]);
        cta_qo_len[cta_idx].push_back(qo_len);
        cta_kv_indptr[cta_idx].push_back(kv_indptr_h[i]);
        cta_kv_len[cta_idx].push_back(kv_len);
        cta_head_indices[cta_idx].push_back(qo_head_idx);
      }
    }
  }

  std::vector<IdType> work_indptr_vec(num_sm90_ctas + 1, 0);
  for (uint32_t i = 0; i < num_sm90_ctas; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cta_qo_tile_indices[i].size();
  }
  int total_num_works = work_indptr_vec.back();
  auto qo_tile_indices_vec = flatten(cta_qo_tile_indices, total_num_works);
  auto qo_indptr_vec = flatten(cta_qo_indptr, total_num_works);
  auto kv_indptr_vec = flatten(cta_kv_indptr, total_num_works);
  auto qo_len_vec = flatten(cta_qo_len, total_num_works);
  auto kv_len_vec = flatten(cta_kv_len, total_num_works);
  auto head_indices_vec = flatten(cta_head_indices, total_num_works);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  int max_total_num_works;

  if (enable_cuda_graph) {
    max_total_num_works = plan_info.same_schedule_for_all_heads
                              ? max_num_works_per_head
                              : max_num_works_per_head * num_qo_heads;
  } else {
    max_total_num_works = total_num_works;
  }

  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_qo_tile_indices");
  plan_info.qo_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_qo_offset");
  plan_info.kv_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_kv_offset");
  plan_info.qo_len_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works,
                                                               16, "batch_prefill_sm90_qo_len");
  plan_info.kv_len_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works,
                                                               16, "batch_prefill_sm90_kv_len");
  plan_info.head_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_head_indices");
  plan_info.work_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * (num_sm90_ctas + 1), 16, "batch_prefill_sm90_work_indptr");

  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* qo_offset_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_indptr_offset);
  IdType* kv_offset_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_offset);
  IdType* qo_len_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_len_offset);
  IdType* kv_len_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_len_offset);
  IdType* head_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.head_indices_offset);
  IdType* work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);

  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(qo_indptr_vec.begin(), qo_indptr_vec.end(), qo_offset_h);
  std::copy(kv_indptr_vec.begin(), kv_indptr_vec.end(), kv_offset_h);
  std::copy(qo_len_vec.begin(), qo_len_vec.end(), qo_len_h);
  std::copy(kv_len_vec.begin(), kv_len_vec.end(), kv_len_h);
  std::copy(head_indices_vec.begin(), head_indices_vec.end(), head_indices_h);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), work_indptr_h);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_SCHEDULER_CUH_
