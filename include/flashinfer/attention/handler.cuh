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
#ifndef FLASHINFER_ATTENTION_HANDLER_CUH_
#define FLASHINFER_ATTENTION_HANDLER_CUH_

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../allocator.h"
#include "../page.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "logits_post_hook.cuh"
#include "warp_layout.cuh"

namespace flashinfer {

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__
                                                  typename AttentionVariant::ParamsT params);

/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 *   after we partition Paged KV-Cache into multiple chunks on KV sequence length
 *   dimension.
 * \tparam IdType A template type indicates the index data type
 * \param max_grid_size The maximum grid size of the kernel
 * \param num_kv_heads The number of KV heads
 * \param num_pages The number of pages per request in the batch
 * \param max_num_pages_per_batch_lb The pre-set lower bound of maximum number of
 *   pages per batch, default to 1
 * \return (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 *   the new batch size after the partition.
 */
template <typename IdType>
std::pair<uint32_t, uint32_t> PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    const uint32_t max_grid_size, const uint32_t num_kv_heads, const std::vector<IdType>& num_pages,
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
    if (new_batch_size * num_kv_heads > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (const IdType& elem : num_pages) {
    new_batch_size += ceil_div(std::max(elem, 1), low);
  }
  return {low, new_batch_size};
}

inline std::tuple<bool, uint32_t, uint32_t> PrefillBinarySearchKVChunkSize(
    const uint32_t max_grid_size, const uint32_t num_kv_heads,
    const std::vector<int64_t>& packed_qo_len_arr, const std::vector<int64_t>& kv_len_arr,
    const uint32_t qo_chunk_size, const uint32_t min_kv_chunk_size = 1) {
  int64_t low = min_kv_chunk_size, high = 0;
  int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 0;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }
  high = max_kv_len;
  int64_t new_batch_size;
  while (low < high) {
    int64_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size +=
          ceil_div(packed_qo_len_arr[i], qo_chunk_size) * ceil_div(kv_len_arr[i], mid);
    }
    if (new_batch_size * num_kv_heads > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                      ceil_div(std::max(int(kv_len_arr[i]), 1), low);
  }
  return {low < max_kv_len, low, new_batch_size};
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeOut A template type indicates the output data type
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
          typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t batch_size, typename AttentionVariant::IdType* kv_indptr_h,
    const uint32_t num_qo_heads, const uint32_t page_size, bool enable_cuda_graph,
    cudaStream_t stream) {
  using DTypeKV = typename AttentionVariant::DTypeKV;
  using IdType = typename AttentionVariant::IdType;
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
    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                          vec_size, bdx, bdy, bdz, AttentionVariant>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * num_kv_heads >= max_grid_size) {
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
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
              max_grid_size, num_kv_heads, num_pages, std::max(128 / page_size, 1U));
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

/*!
 * \brief Partition Paged KV-Cache into multiple chunks on KV sequence length
 * \tparam IdType A template type indicates the index data type
 * \param old_batch_size The batch size of the old Paged KV-Cache
 * \param old_page_indptr_h The host-side page indptr of the old Paged KV-Cache
 * \param old_last_page_len_h The host-side last page offset of the old Paged KV-Cache
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_paged_kv_d The device-side new Paged KV-Cache
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename IdType>
std::tuple<std::vector<IdType>, std::vector<IdType>, std::vector<IdType>> DecodeSplitKVIndptr(
    IdType* indptr_h, uint32_t batch_size, uint32_t kv_chunk_size) {
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

  return {request_indices, kv_tile_indices, o_indptr};
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
      throw std::invalid_argument(err_msg.str());
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

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant>
cudaError_t DecodePlan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                       void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
                       DecodePlanInfo& plan_info, typename AttentionVariant::IdType* indptr_h,
                       typename AttentionVariant::IdType* last_page_len_h, uint32_t batch_size,
                       uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t page_size,
                       bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeOut = typename AttentionVariant::DTypeO;
  using IdType = typename AttentionVariant::IdType;
  bool split_kv;
  uint32_t max_grid_size, kv_chunk_size_in_pages, new_batch_size;
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    auto work_estimation_func =
        BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE,
                                                            AttentionVariant>;
    FLASHINFER_CUDA_CALL(work_estimation_func(split_kv, max_grid_size, kv_chunk_size_in_pages,
                                              new_batch_size, batch_size, indptr_h, num_qo_heads,
                                              page_size, enable_cuda_graph, stream));
    size_t padded_batch_size;
    plan_info.enable_cuda_graph = enable_cuda_graph;
    plan_info.split_kv = split_kv;
    padded_batch_size = (enable_cuda_graph) ? (split_kv ? max_grid_size / num_kv_heads : batch_size)
                                            : new_batch_size;
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
          num_qo_heads * padded_batch_size * HEAD_DIM * sizeof(DTypeOut), 16, "batch_decode_tmp_v");
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
  });
  return cudaSuccess;
}

template <typename IdType>
std::tuple<bool, uint32_t, uint32_t, WarpLayout, uint32_t, uint32_t, std::vector<IdType>,
           std::vector<IdType>, std::vector<IdType>, std::vector<IdType>, std::vector<IdType>>
PrefillSplitQOKVIndptr(IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t batch_size,
                       uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                       uint32_t page_size, uint32_t max_grid_size) {
  std::vector<IdType> request_indices, qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr;
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  uint32_t total_num_rows = qo_indptr_h[batch_size];

  // step 1: compute qo_chunk_size
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  int64_t sum_packed_qo_len = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    sum_packed_qo_len += packed_qo_len_arr[i];
  }
  int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
  WarpLayout warp_layout;
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    warp_layout = WarpLayout::k4x1x2;  // (num_warps_x = 4, num_warps_z = 1, num_frags_x = 2)
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        warp_layout = WarpLayout::k4x1x1;  // (num_warps_x = 4, num_warps_z = 1, num_frags_x = 1)
      } else {
        // avg_packed_qo_len <= 16
        warp_layout = WarpLayout::k1x4x1;  // (num_warps_x = 1, num_warps_z = 4, num_frags_x = 1)
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4x1 layout
      warp_layout = WarpLayout::k4x1x1;
    }
  }
  const uint32_t qo_chunk_size = get_num_rows_per_cta(warp_layout);

  // step 2: determine kv_chunk_size
  auto [split_kv, kv_chunk_size, new_batch_size] = PrefillBinarySearchKVChunkSize(
      max_grid_size, num_kv_heads, packed_qo_len_arr, kv_len_arr, qo_chunk_size,
      /*min_kv_chunk_size=*/std::max((128 / page_size), 1U));

  // step 3: split qo_indptr and kv_indptr
  uint32_t total_num_tiles_q = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int64_t packed_qo_len = packed_qo_len_arr[request_idx],
            kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    int64_t num_tiles_q = ceil_div(packed_qo_len, qo_chunk_size),
            num_tiles_kv = ceil_div(kv_len, kv_chunk_size);
    total_num_tiles_q += num_tiles_q;
    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
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

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return {split_kv,
          total_num_tiles_q,
          new_batch_size,
          warp_layout,
          kv_chunk_size,
          total_num_rows,
          std::move(request_indices),
          std::move(qo_tile_indices),
          std::move(kv_tile_indices),
          std::move(merge_indptr),
          std::move(o_indptr)};
}

struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t warp_layout_code;
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
        warp_layout_code(0),
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
            warp_layout_code,
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
    if (vec.size() != 14) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 14, but got " << vec.size();
      throw std::invalid_argument(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    warp_layout_code = vec[2];
    request_indices_offset = vec[3];
    qo_tile_indices_offset = vec[4];
    kv_tile_indices_offset = vec[5];
    merge_indptr_offset = vec[6];
    o_indptr_offset = vec[7];
    kv_chunk_size_ptr_offset = vec[8];
    v_offset = vec[9];
    s_offset = vec[10];
    block_valid_mask_offset = vec[11];
    enable_cuda_graph = vec[12];
    split_kv = vec[13];
  }
};

template <typename DTypeOut, typename IdType>
cudaError_t PrefillPlan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                        void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
                        PrefillPlanInfo& plan_info, IdType* qo_indptr_h, IdType* kv_indptr_h,
                        uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                        uint32_t head_dim, uint32_t page_size, bool enable_cuda_graph,
                        cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  uint32_t split_max_batch_size = max_grid_size / num_kv_heads;

  // step 2: determine kv_chunk_size
  auto [split_kv, total_num_tiles_q, new_batch_size, warp_layout,
        kv_chunk_size, total_num_rows, request_indices_vec, qo_tile_indices_vec,
        kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec] =
      PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, batch_size, num_qo_heads, num_kv_heads,
                             head_dim, page_size, max_grid_size);
  const uint32_t qo_tile_size = get_num_rows_per_cta(warp_layout);
  plan_info.warp_layout_code = static_cast<int64_t>(warp_layout);
  plan_info.total_num_rows = total_num_rows;

  plan_info.enable_cuda_graph = enable_cuda_graph;
  size_t padded_batch_size =
      enable_cuda_graph ? std::max(split_max_batch_size, total_num_tiles_q) : new_batch_size;
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
        num_qo_heads * split_max_batch_size * qo_tile_size * head_dim * sizeof(DTypeOut), 16,
        "batch_prefill_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * split_max_batch_size * qo_tile_size * sizeof(float), 16,
        "batch_prefill_tmp_s");
    plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * (plan_info.total_num_rows + 1), 16, "batch_prefill_merge_indptr");
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");
    IdType* merge_indptr_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
    IdType* block_valid_mask_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
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

}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_HANDLER_CUH_
