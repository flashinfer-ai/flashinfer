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

template <LogitsPostHook logits_post_hook, PosEncodingMode pos_encoding_mode,
          uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, PageStorage page_storage, typename DTypeQ, typename DTypeKV,
          typename DTypeOut, typename IdType>
__global__ void BatchDecodeWithPagedKVCacheKernel(
    DTypeQ* __restrict__ q, IdType* __restrict__ q_offset,
    paged_kv_t<page_storage, DTypeKV, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* __restrict__ o,
    float* __restrict__ lse, bool* __restrict__ block_valid_mask, bool partition_kv,
    int maybe_window_left, float logits_soft_cap, float sm_scale, float rope_rcp_scale,
    float rope_rcp_theta);

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
 * \tparam page_storage Whether to store indices or pointers of each active page
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
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage,
          LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode POS_ENCODING_MODE, typename DTypeQ,
          typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t batch_size, IdType* kv_indptr_h, const uint32_t num_qo_heads,
    const uint32_t page_size, bool enable_cuda_graph, cudaStream_t stream) {
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t num_stages_smem = 2U;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;
  const uint32_t smem_size =
      2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
      std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

  auto kernel =
      BatchDecodeWithPagedKVCacheKernel<LOGITS_POST_HOOK, POS_ENCODING_MODE, num_stages_smem,
                                        tile_size_per_bdx, vec_size, bdx, bdy, bdz, page_storage,
                                        DTypeQ, DTypeKV, DTypeOut, IdType>;
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
    new_batch_size = batch_size;
  } else {
    // compute max_num_pages_per_batch and new_batch_size
    std::vector<IdType> num_pages(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
    }
    std::tie(max_num_pages_per_batch, new_batch_size) =
        PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, num_kv_heads, num_pages,
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
cudaError_t PartitionPagedKVCacheComputeAuxiliaryInfo(
    const uint32_t max_num_pages_per_batch, const uint32_t old_batch_size,
    const uint32_t padded_batch_size, const uint32_t page_size, IdType* old_indptr_h,
    IdType* old_last_page_len_h, IdType* new_page_indptr_h, IdType* new_last_page_len_h,
    IdType* chunk_indptr_h, IdType* batch_idx_map_h, IdType* chunk_start_pos_h,
    IdType* seq_lens_before_partition_h, bool* block_valid_mask_h, void* device_buffer,
    void* host_buffer, size_t num_bytes_to_copy, cudaStream_t stream = nullptr) {
  std::vector<IdType> new_page_indptr_vec, new_last_page_len_vec, chunk_indptr_vec,
      batch_idx_map_vec, chunk_start_pos_vec, seq_lens_before_partition_vec;
  std::vector<bool> block_valid_mask_vec;

  new_page_indptr_vec.push_back(0);
  chunk_indptr_vec.push_back(0);

  for (uint32_t batch_idx = 0; batch_idx < old_batch_size; batch_idx++) {
    uint32_t num_chunks =
        ceil_div(old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx], max_num_pages_per_batch);
    chunk_indptr_vec.push_back(chunk_indptr_vec.back() + std::max(num_chunks, 1U));
    if (num_chunks == 0) {
      new_page_indptr_vec.push_back(old_indptr_h[batch_idx]);
      new_last_page_len_vec.push_back(0);
      if (block_valid_mask_h != nullptr) {
        block_valid_mask_vec.push_back(true);
      }
      batch_idx_map_vec.push_back(batch_idx);
      chunk_start_pos_vec.push_back(0);
      seq_lens_before_partition_vec.push_back(0);
    } else {
      uint32_t seq_len_before_partition =
          (old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx] - 1) * page_size +
          old_last_page_len_h[batch_idx];
      for (uint32_t j = 0; j < num_chunks; ++j) {
        bool is_last = (j + 1) == num_chunks;
        new_page_indptr_vec.push_back(
            min(old_indptr_h[batch_idx] + (j + 1) * max_num_pages_per_batch,
                old_indptr_h[batch_idx + 1]));
        new_last_page_len_vec.push_back(is_last ? old_last_page_len_h[batch_idx] : page_size);
        if (block_valid_mask_h != nullptr) {
          block_valid_mask_vec.push_back(true);
        }
        batch_idx_map_vec.push_back(batch_idx);
        chunk_start_pos_vec.push_back(j * max_num_pages_per_batch * page_size);
        seq_lens_before_partition_vec.push_back(seq_len_before_partition);
      }
    }
  }
  IdType last_page_indptr = new_page_indptr_vec.back();
  while (new_page_indptr_vec.size() < padded_batch_size + 1) {
    new_page_indptr_vec.push_back(last_page_indptr);
  }
  std::copy(new_page_indptr_vec.begin(), new_page_indptr_vec.end(), new_page_indptr_h);
  std::copy(new_last_page_len_vec.begin(), new_last_page_len_vec.end(), new_last_page_len_h);
  std::copy(chunk_indptr_vec.begin(), chunk_indptr_vec.end(), chunk_indptr_h);
  std::copy(batch_idx_map_vec.begin(), batch_idx_map_vec.end(), batch_idx_map_h);
  std::copy(chunk_start_pos_vec.begin(), chunk_start_pos_vec.end(), chunk_start_pos_h);
  std::copy(seq_lens_before_partition_vec.begin(), seq_lens_before_partition_vec.end(),
            seq_lens_before_partition_h);
  if (block_valid_mask_h != nullptr) {
    std::copy(block_valid_mask_vec.begin(), block_valid_mask_vec.end(), block_valid_mask_h);
  }

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(device_buffer, host_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

class BatchDecodeHandler {
 public:
  template <typename DType>
  DType* GetTempV() const {
    return (DType*)tmp_v_;
  }
  float* GetTempS() const { return tmp_s_; }
  template <typename IdType>
  IdType* GetNewIndPtr() const {
    return (IdType*)new_indptr_;
  }
  template <typename IdType>
  IdType* GetNewLastPageLen() const {
    return (IdType*)new_last_page_len_;
  }
  template <typename IdType>
  IdType* GetChunkIndPtr() const {
    return (IdType*)chunk_indptr_;
  }
  template <typename IdType>
  IdType* GetBatchIdxMap() const {
    return (IdType*)batch_idx_map_;
  }
  template <typename IdType>
  IdType* GetChunkStartPos() const {
    return (IdType*)chunk_start_pos_;
  }
  template <typename IdType>
  IdType* GetSeqLengthsBeforePartition() const {
    return (IdType*)seq_lengths_before_partition_;
  }

  uint32_t GetPaddedBatchSize() const { return padded_batch_size_; }

  bool* GetBlockValidMask() const { return block_valid_mask_; }

  template <uint32_t HEAD_DIM, PageStorage page_storage, LogitsPostHook LOGITS_POST_HOOK,
            PosEncodingMode POS_ENCODING_MODE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
            typename IdType>
  cudaError_t BeginForwardDispatched(void* float_buffer, size_t float_workspace_size_in_bytes,
                                     void* int_buffer, size_t int_workspace_size_in_bytes,
                                     IdType* indptr_h, IdType* last_page_len_h, uint32_t batch_size,
                                     uint32_t num_qo_heads, uint32_t num_kv_heads,
                                     uint32_t page_size) {
    batch_size_before_partition_ = batch_size;
    bool split_kv;
    uint32_t max_grid_size, max_num_pages_per_batch, new_batch_size;
    DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
      auto work_estimation_func =
          BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE, HEAD_DIM, page_storage,
                                                              LOGITS_POST_HOOK, POS_ENCODING_MODE,
                                                              DTypeQ, DTypeKV, DTypeOut, IdType>;
      FLASHINFER_CUDA_CALL(
          work_estimation_func(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size,
                               batch_size, indptr_h, num_qo_heads, page_size,
                               /*enable_cuda_graph=*/IsCUDAGraphEnabled(), stream_));
      batch_size_after_partition_ = new_batch_size;
      if (IsCUDAGraphEnabled()) {
        if (batch_size != fixed_batch_size_) {
          std::ostringstream err_msg;
          err_msg << "The running batch size " << batch_size
                  << " is not compatible with the fixed batch size " << fixed_batch_size_
                  << " initialized for CUDAGraph";
          throw std::runtime_error(err_msg.str());
        }
        size_t padded_batch_size = max_grid_size / num_kv_heads;
        if (split_kv) {
          padded_batch_size_ = padded_batch_size;
          AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
          tmp_v_ = float_allocator.aligned_alloc<void>(
              num_qo_heads * padded_batch_size * HEAD_DIM * sizeof(DTypeOut), 16,
              "batch_decode_tmp_v");
          tmp_s_ = float_allocator.aligned_alloc<float>(
              num_qo_heads * padded_batch_size * sizeof(float), 16, "batch_decode_tmp_s");
          AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
          new_indptr_ = int_allocator.aligned_alloc<void>((padded_batch_size + 1) * sizeof(IdType),
                                                          16, "batch_decode_new_indptr");

          void* new_indptr_h_ = page_locked_buffer_;
          new_last_page_len_ = int_allocator.aligned_alloc<void>(
              padded_batch_size * sizeof(IdType), 16, "batch_decode_new_last_page_len");
          void* new_last_page_len_h_ =
              (char*)page_locked_buffer_ + ((char*)new_last_page_len_ - (char*)new_indptr_);
          chunk_indptr_ = int_allocator.aligned_alloc<void>(
              (padded_batch_size + 1) * sizeof(IdType), 16, "batch_decode_chunk_indptr");
          void* chunk_indptr_h_ =
              (char*)page_locked_buffer_ + ((char*)chunk_indptr_ - (char*)new_indptr_);
          batch_idx_map_ = int_allocator.aligned_alloc<void>(padded_batch_size * sizeof(IdType), 16,
                                                             "batch_decode_batch_idx_map");
          void* batch_idx_map_h_ =
              (char*)page_locked_buffer_ + ((char*)batch_idx_map_ - (char*)new_indptr_);
          chunk_start_pos_ = int_allocator.aligned_alloc<void>(padded_batch_size * sizeof(IdType),
                                                               16, "batch_decode_chunk_start_pos");
          void* chunk_start_pos_h_ =
              (char*)page_locked_buffer_ + ((char*)chunk_start_pos_ - (char*)new_indptr_);
          seq_lengths_before_partition_ = int_allocator.aligned_alloc<void>(
              padded_batch_size * sizeof(IdType), 16, "batch_decode_seq_lengths_before_partition");
          void* seq_lengths_before_partition_h_ =
              (char*)page_locked_buffer_ +
              ((char*)seq_lengths_before_partition_ - (char*)new_indptr_);
          block_valid_mask_ = int_allocator.aligned_alloc<bool>(
              padded_batch_size * sizeof(bool), 16, "batch_decode_block_valid_mask");
          bool* block_valid_mask_h_ =
              (bool*)page_locked_buffer_ + ((bool*)block_valid_mask_ - (bool*)new_indptr_);
          std::fill(block_valid_mask_h_, block_valid_mask_h_ + padded_batch_size, 0);

          size_t num_bytes_to_copy = (char*)int_allocator.ptr - (char*)new_indptr_;
          FLASHINFER_CUDA_CALL(PartitionPagedKVCacheComputeAuxiliaryInfo(
              max_num_pages_per_batch, batch_size, padded_batch_size, page_size, indptr_h,
              last_page_len_h, (IdType*)new_indptr_h_, (IdType*)new_last_page_len_h_,
              (IdType*)chunk_indptr_h_, (IdType*)batch_idx_map_h_, (IdType*)chunk_start_pos_h_,
              (IdType*)seq_lengths_before_partition_h_, block_valid_mask_h_,
              /*device_buffer=*/new_indptr_,
              /*host_buffer=*/page_locked_buffer_, num_bytes_to_copy, stream_));
        } else {
          block_valid_mask_ = nullptr;
          padded_batch_size_ = batch_size;
        }
      } else {
        // NOTE(Zihao): we don't use block_valid_mask when CUDAGraph is disabled.
        block_valid_mask_ = nullptr;
        // do not pad the batch size when not using CUDAGraph
        padded_batch_size_ = batch_size_after_partition_;
        if (split_kv) {
          AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
          tmp_v_ = float_allocator.aligned_alloc<void>(
              num_qo_heads * new_batch_size * HEAD_DIM * sizeof(DTypeOut), 16,
              "batch_decode_tmp_v");
          tmp_s_ = float_allocator.aligned_alloc<float>(
              num_qo_heads * new_batch_size * sizeof(float), 16, "batch_decode_tmp_s");
          AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
          new_indptr_ = int_allocator.aligned_alloc<void>(
              (batch_size_after_partition_ + 1) * sizeof(IdType), 16, "batch_decode_new_indptr");
          void* new_indptr_h_ = page_locked_buffer_;
          new_last_page_len_ = int_allocator.aligned_alloc<void>(
              batch_size_after_partition_ * sizeof(IdType), 16, "batch_decode_new_last_page_len");
          void* new_last_page_len_h_ =
              (char*)page_locked_buffer_ + ((char*)new_last_page_len_ - (char*)new_indptr_);
          chunk_indptr_ = int_allocator.aligned_alloc<void>(
              (batch_size_before_partition_ + 1) * sizeof(IdType), 16, "batch_decode_chunk_indptr");
          void* chunk_indptr_h_ =
              (char*)page_locked_buffer_ + ((char*)chunk_indptr_ - (char*)new_indptr_);
          batch_idx_map_ = int_allocator.aligned_alloc<void>(
              batch_size_after_partition_ * sizeof(IdType), 16, "batch_decode_batch_idx_map");
          void* batch_idx_map_h_ =
              (char*)page_locked_buffer_ + ((char*)batch_idx_map_ - (char*)new_indptr_);
          chunk_start_pos_ = int_allocator.aligned_alloc<void>(
              batch_size_after_partition_ * sizeof(IdType), 16, "batch_decode_chunk_start_pos");
          void* chunk_start_pos_h_ =
              (char*)page_locked_buffer_ + ((char*)chunk_start_pos_ - (char*)new_indptr_);
          seq_lengths_before_partition_ =
              int_allocator.aligned_alloc<void>(batch_size_after_partition_ * sizeof(IdType), 16,
                                                "batch_decode_seq_lengths_before_partition");
          void* seq_lengths_before_partition_h_ =
              (char*)page_locked_buffer_ +
              ((char*)seq_lengths_before_partition_ - (char*)new_indptr_);
          size_t num_bytes_to_copy = (char*)int_allocator.ptr - (char*)new_indptr_;
          FLASHINFER_CUDA_CALL(PartitionPagedKVCacheComputeAuxiliaryInfo(
              max_num_pages_per_batch, batch_size, batch_size_after_partition_, page_size, indptr_h,
              last_page_len_h, (IdType*)new_indptr_h_, (IdType*)new_last_page_len_h_,
              (IdType*)chunk_indptr_h_, (IdType*)batch_idx_map_h_, (IdType*)chunk_start_pos_h_,
              (IdType*)seq_lengths_before_partition_h_,
              /*block_valid_mask_h=*/nullptr,
              /*device_buffer=*/new_indptr_,
              /*host_buffer=*/page_locked_buffer_, num_bytes_to_copy, stream_));
        }
      }
    });
    forward_started_ = true;
    return cudaSuccess;
  }

  cudaError_t EndForward() {
    forward_started_ = false;
    padded_batch_size_ = 0;
    batch_size_before_partition_ = 0;
    batch_size_after_partition_ = 0;
    block_valid_mask_ = nullptr;
    tmp_v_ = nullptr;
    tmp_s_ = nullptr;
    new_indptr_ = nullptr;
    new_last_page_len_ = nullptr;
    chunk_indptr_ = nullptr;
    batch_idx_map_ = nullptr;
    chunk_start_pos_ = nullptr;
    seq_lengths_before_partition_ = nullptr;
    return cudaSuccess;
  }

  bool IsForwardStarted() const { return forward_started_; }

  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  uint32_t GetBatchSizeBeforePartition() const { return batch_size_before_partition_; }

  uint32_t GetBatchSizeAfterPartition() const { return batch_size_after_partition_; }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  /*!
   * \brief Constructor of BatchDecodeHandler
   * \param enable_cuda_graph A boolean indicates whether to enable CUDA graph
   * \param batch_size If enable_cuda_graph is true, we must specify a fixed batch_size
   */
  BatchDecodeHandler(bool enable_cuda_graph = false, uint32_t batch_size = 0)
      : batch_size_after_partition_(0U),
        tmp_v_(nullptr),
        tmp_s_(nullptr),
        block_valid_mask_(nullptr),
        new_indptr_(nullptr),
        new_last_page_len_(nullptr),
        chunk_indptr_(nullptr),
        batch_idx_map_(nullptr),
        chunk_start_pos_(nullptr),
        seq_lengths_before_partition_(nullptr),
        forward_started_(false),
        cuda_graph_enabled_(enable_cuda_graph),
        fixed_batch_size_(batch_size),
        stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchDecodeHandler() {
    EndForward();
    cudaFreeHost(page_locked_buffer_);
  }

  bool IsCUDAGraphEnabled() const { return cuda_graph_enabled_; }

 protected:
  uint32_t batch_size_before_partition_;
  uint32_t batch_size_after_partition_;
  void* page_locked_buffer_;
  void* tmp_v_;
  float* tmp_s_;
  bool* block_valid_mask_;
  void* new_indptr_;
  void* new_last_page_len_;
  void* chunk_indptr_;
  void* batch_idx_map_;
  void* chunk_start_pos_;
  void* seq_lengths_before_partition_;
  bool forward_started_;
  bool cuda_graph_enabled_;
  uint32_t padded_batch_size_;
  uint32_t fixed_batch_size_;
  cudaStream_t stream_;
};

template <typename IdType>
cudaError_t PrefillSplitQOKVIndptr(bool& split_kv, uint32_t& split_max_batch_size,
                                   uint32_t& total_num_tiles_q, uint32_t& new_batch_size,
                                   WarpLayout& warp_layout, uint32_t& kv_chunk_size,
                                   uint32_t& total_num_rows, std::vector<IdType>& request_indices,
                                   std::vector<IdType>& qo_tile_indices,
                                   std::vector<IdType>& kv_tile_indices,
                                   std::vector<IdType>& merge_indptr, std::vector<IdType>& o_indptr,
                                   IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size) {
  request_indices.clear();
  qo_tile_indices.clear();
  kv_tile_indices.clear();
  merge_indptr.clear();
  o_indptr.clear();
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  total_num_rows = qo_indptr_h[batch_size];

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  split_max_batch_size = max_grid_size / num_kv_heads;

  // step 1: compute qo_chunk_size
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  int64_t sum_packed_qo_len = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    sum_packed_qo_len += packed_qo_len_arr[i];
  }
  int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    warp_layout = WarpLayout::k4x1x2;  // (num_warps_x = 4, num_warps_z = 1, num_frags_x = 2)
  } else {
    if (avg_packed_qo_len > 16) {
      warp_layout = WarpLayout::k4x1x1;  // (num_warps_x = 4, num_warps_z = 1, num_frags_x = 1)
    } else {
      // avg_packed_qo_len <= 16
      warp_layout = WarpLayout::k1x4x1;  // (num_warps_x = 1, num_warps_z = 4, num_frags_x = 1)
    }
  }
  const uint32_t qo_chunk_size = get_num_rows_per_cta(warp_layout);

  // step 2: determine kv_chunk_size
  std::tie(split_kv, kv_chunk_size, new_batch_size) = PrefillBinarySearchKVChunkSize(
      max_grid_size, num_kv_heads, packed_qo_len_arr, kv_len_arr, qo_chunk_size,
      /*min_kv_chunk_size=*/std::max((128 / page_size), 1U));

  // step 3: split qo_indptr and kv_indptr
  total_num_tiles_q = 0;
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

  return cudaSuccess;
}

class BatchPrefillHandler {
 public:
  template <typename IdType>
  IdType* GetRequestIndices() const {
    return (IdType*)request_indices_;
  }

  template <typename IdType>
  IdType* GetQOTileIndices() const {
    return (IdType*)qo_tile_indices_;
  }

  template <typename IdType>
  IdType* GetKVTileIndices() const {
    return (IdType*)kv_tile_indices_;
  }

  template <typename IdType>
  IdType* GetMergeIndptr() const {
    return (IdType*)merge_indptr_;
  }

  template <typename IdType>
  IdType* GetOIndptr() const {
    return (IdType*)o_indptr_;
  }

  template <typename IdType>
  IdType* GetKVChunkSizePtr() const {
    return (IdType*)kv_chunk_size_ptr_;
  }

  template <typename DType>
  DType* GetTempV() const {
    return (DType*)tmp_v_;
  }

  bool* GetBlockValidMask() const { return block_valid_mask_; }

  float* GetTempS() const { return tmp_s_; }

  uint32_t GetPaddedBatchSize() const { return padded_batch_size_; }

  WarpLayout GetWarpLayout() const { return warp_layout_; }

  uint32_t GetTotalNumRows() const { return total_num_rows_; }

  bool IsForwardStarted() const { return request_indices_ != nullptr; }

  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  template <typename DTypeOut, typename IdType>
  cudaError_t BeginForward(void* float_buffer, size_t float_workspace_size_in_bytes,
                           void* int_buffer, size_t int_workspace_size_in_bytes,
                           IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t batch_size,
                           uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                           uint32_t page_size) {
    if (num_qo_heads % num_kv_heads != 0) {
      std::ostringstream err_msg;
      err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
              << num_kv_heads;
      throw std::invalid_argument(err_msg.str());
    }
    bool split_kv;
    uint32_t split_max_batch_size, new_batch_size, total_num_tiles_q, kv_chunk_size;
    std::vector<IdType> request_indices_vec, qo_tile_indices_vec, kv_tile_indices_vec,
        merge_indptr_vec, o_indptr_vec;
    FLASHINFER_CUDA_CALL(PrefillSplitQOKVIndptr(
        split_kv, split_max_batch_size, total_num_tiles_q, new_batch_size, warp_layout_,
        kv_chunk_size, total_num_rows_, request_indices_vec, qo_tile_indices_vec,
        kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec, qo_indptr_h, kv_indptr_h, batch_size,
        num_qo_heads, num_kv_heads, head_dim, page_size));
    const uint32_t qo_tile_size = get_num_rows_per_cta(warp_layout_);

    if (IsCUDAGraphEnabled()) {
      padded_batch_size_ = std::max(split_max_batch_size, total_num_tiles_q);
      AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
      request_indices_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * padded_batch_size_, 16,
                                                           "batch_prefill_request_indices");
      void* request_indices_h_ = page_locked_buffer_;
      qo_tile_indices_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * padded_batch_size_, 16,
                                                           "batch_prefill_qo_tile_indices");
      void* qo_tile_indices_h_ =
          (char*)page_locked_buffer_ + ((char*)qo_tile_indices_ - (char*)request_indices_);
      kv_tile_indices_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * padded_batch_size_, 16,
                                                           "batch_prefill_kv_tile_indices");
      void* kv_tile_indices_h_ =
          (char*)page_locked_buffer_ + ((char*)kv_tile_indices_ - (char*)request_indices_);
      o_indptr_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * (batch_size + 1), 16,
                                                    "batch_prefill_o_indptr");
      void* o_indptr_h_ = (char*)page_locked_buffer_ + ((char*)o_indptr_ - (char*)request_indices_);
      kv_chunk_size_ptr_ =
          int_allocator.aligned_alloc<void>(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");
      void* kv_chunk_size_ptr_h_ =
          (char*)page_locked_buffer_ + ((char*)kv_chunk_size_ptr_ - (char*)request_indices_);
      *(IdType*)kv_chunk_size_ptr_h_ = kv_chunk_size;
      if (total_num_tiles_q < split_max_batch_size) {
        // need merge_indptr
        merge_indptr_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * (total_num_rows_ + 1),
                                                          16, "batch_prefill_merge_indptr");
        void* merge_indptr_h_ =
            (char*)page_locked_buffer_ + ((char*)merge_indptr_ - (char*)request_indices_);
        std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), (IdType*)merge_indptr_h_);
        block_valid_mask_ = int_allocator.aligned_alloc<bool>(sizeof(bool) * padded_batch_size_, 16,
                                                              "batch_prefill_block_valid_mask");
        bool* block_valid_mask_h_ =
            (bool*)page_locked_buffer_ + ((bool*)block_valid_mask_ - (bool*)request_indices_);
        for (uint32_t i = 0; i < padded_batch_size_; ++i) {
          block_valid_mask_h_[i] = i < new_batch_size;
        }
      } else {
        // total_num_tiles_q >= split_max_batch_size, we don't need to perform the second round at
        // all.
        merge_indptr_ = nullptr;
        block_valid_mask_ = nullptr;
      }
      std::copy(request_indices_vec.begin(), request_indices_vec.end(),
                (IdType*)request_indices_h_);
      std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(),
                (IdType*)qo_tile_indices_h_);
      std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(),
                (IdType*)kv_tile_indices_h_);
      std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), (IdType*)o_indptr_h_);

      size_t num_bytes_to_copy = (char*)int_allocator.ptr - (char*)request_indices_;
      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_, page_locked_buffer_, num_bytes_to_copy,
                                           cudaMemcpyHostToDevice, stream_))

      if (total_num_tiles_q < split_max_batch_size) {
        AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
        tmp_v_ = float_allocator.aligned_alloc<void>(
            num_qo_heads * split_max_batch_size * qo_tile_size * head_dim * sizeof(DTypeOut), 16,
            "batch_prefill_tmp_v");
        tmp_s_ = float_allocator.aligned_alloc<float>(
            num_qo_heads * split_max_batch_size * qo_tile_size * sizeof(float), 16,
            "batch_prefill_tmp_s");
      } else {
        tmp_v_ = nullptr;
        tmp_s_ = nullptr;
      }
    } else {
      padded_batch_size_ = new_batch_size;
      AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
      request_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * request_indices_vec.size(), 16, "batch_prefill_request_indices");
      void* request_indices_h_ = page_locked_buffer_;
      qo_tile_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * qo_tile_indices_vec.size(), 16, "batch_prefill_qo_tile_indices");
      void* qo_tile_indices_h_ =
          (char*)page_locked_buffer_ + ((char*)qo_tile_indices_ - (char*)request_indices_);
      kv_tile_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * kv_tile_indices_vec.size(), 16, "batch_prefill_kv_tile_indices");
      void* kv_tile_indices_h_ =
          (char*)page_locked_buffer_ + ((char*)kv_tile_indices_ - (char*)request_indices_);
      if (split_kv) {
        // need merge_indptr when split_kv is true
        merge_indptr_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * merge_indptr_vec.size(),
                                                          16, "batch_prefill_merge_indptr");
        void* merge_indptr_h_ =
            (char*)page_locked_buffer_ + ((char*)merge_indptr_ - (char*)request_indices_);
        std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), (IdType*)merge_indptr_h_);
      }
      o_indptr_ = int_allocator.aligned_alloc<void>(sizeof(IdType) * o_indptr_vec.size(), 16,
                                                    "batch_prefill_o_indptr");
      void* o_indptr_h_ = (char*)page_locked_buffer_ + ((char*)o_indptr_ - (char*)request_indices_);
      kv_chunk_size_ptr_ =
          int_allocator.aligned_alloc<void>(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");
      void* kv_chunk_size_ptr_h_ =
          (char*)page_locked_buffer_ + ((char*)kv_chunk_size_ptr_ - (char*)request_indices_);
      *(IdType*)kv_chunk_size_ptr_h_ = kv_chunk_size;
      std::copy(request_indices_vec.begin(), request_indices_vec.end(),
                (IdType*)request_indices_h_);
      std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(),
                (IdType*)qo_tile_indices_h_);
      std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(),
                (IdType*)kv_tile_indices_h_);
      std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), (IdType*)o_indptr_h_);
      size_t num_bytes_to_copy = (char*)int_allocator.ptr - (char*)request_indices_;

      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_, page_locked_buffer_, num_bytes_to_copy,
                                           cudaMemcpyHostToDevice, stream_))

      if (split_kv) {
        AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
        tmp_v_ = float_allocator.aligned_alloc<void>(
            num_qo_heads * new_batch_size * qo_tile_size * head_dim * sizeof(DTypeOut), 16,
            "batch_prefill_tmp_v");
        tmp_s_ = float_allocator.aligned_alloc<float>(
            num_qo_heads * new_batch_size * qo_tile_size * sizeof(float), 16,
            "batch_prefill_tmp_s");
      } else {
        tmp_v_ = nullptr;
        tmp_s_ = nullptr;
      }

      block_valid_mask_ = nullptr;
    }
    return cudaSuccess;
  }

  cudaError_t EndForward() {
    forward_started_ = false;
    request_indices_ = nullptr;
    qo_tile_indices_ = nullptr;
    kv_tile_indices_ = nullptr;
    merge_indptr_ = nullptr;
    o_indptr_ = nullptr;
    kv_chunk_size_ptr_ = nullptr;
    tmp_v_ = nullptr;
    tmp_s_ = nullptr;
    block_valid_mask_ = nullptr;
    total_num_rows_ = 0U;
    padded_batch_size_ = 0U;
    warp_layout_ = WarpLayout::k4x1x2;
    return cudaSuccess;
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  bool IsCUDAGraphEnabled() const { return enable_cuda_graph_; }

  BatchPrefillHandler(bool enable_cuda_graph = false)
      : request_indices_(nullptr),
        qo_tile_indices_(nullptr),
        kv_tile_indices_(nullptr),
        merge_indptr_(nullptr),
        o_indptr_(nullptr),
        kv_chunk_size_ptr_(nullptr),
        tmp_v_(nullptr),
        tmp_s_(nullptr),
        block_valid_mask_(nullptr),
        total_num_rows_(0U),
        padded_batch_size_(0U),
        warp_layout_(WarpLayout::k4x1x2),
        forward_started_(false),
        enable_cuda_graph_(enable_cuda_graph),
        stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchPrefillHandler() {
    EndForward();
    cudaFreeHost(page_locked_buffer_);
  }

 protected:
  void* page_locked_buffer_;
  void* request_indices_;
  void* qo_tile_indices_;
  void* kv_tile_indices_;
  void* merge_indptr_;
  void* o_indptr_;
  void* kv_chunk_size_ptr_;
  void* tmp_v_;
  float* tmp_s_;
  bool* block_valid_mask_;
  uint32_t total_num_rows_;
  uint32_t padded_batch_size_;
  WarpLayout warp_layout_;
  bool forward_started_;
  bool enable_cuda_graph_;
  cudaStream_t stream_;
};

}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_HANDLER_CUH_
