/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/decode_attention_decl.cuh>
#include <flashinfer/prefill_attention_decl.cuh>
#include <optional>

#include "flashinfer/attention/logits_post_hook.cuh"
#include "flashinfer/attention/mask.cuh"
#include "utils.h"

namespace flashinfer {

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
  cudaError_t PlanDispatched(void* float_buffer, size_t float_workspace_size_in_bytes,
                             void* int_buffer, size_t int_workspace_size_in_bytes, IdType* indptr_h,
                             IdType* last_page_len_h, uint32_t batch_size, uint32_t num_qo_heads,
                             uint32_t num_kv_heads, uint32_t page_size) {
    Clear();
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
    return cudaSuccess;
  }

  void Clear() {
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
  }

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
        cuda_graph_enabled_(enable_cuda_graph),
        fixed_batch_size_(batch_size),
        stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchDecodeHandler() { cudaFreeHost(page_locked_buffer_); }

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
  bool cuda_graph_enabled_;
  uint32_t padded_batch_size_;
  uint32_t fixed_batch_size_;
  cudaStream_t stream_;
};

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

  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  template <typename DTypeOut, typename IdType>
  cudaError_t Plan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                   size_t int_workspace_size_in_bytes, IdType* qo_indptr_h, IdType* kv_indptr_h,
                   uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                   uint32_t head_dim, uint32_t page_size) {
    Clear();
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

  void Clear() {
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
        enable_cuda_graph_(enable_cuda_graph),
        stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchPrefillHandler() { cudaFreeHost(page_locked_buffer_); }

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
  bool enable_cuda_graph_;
  cudaStream_t stream_;
};

template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheCustomMask(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, uint8_t* custom_mask, DTypeOut* o, DTypeOut* tmp,
    float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_allow_fp16_qk_reduction(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            return SinglePrefillWithKVCacheDispatched<HEAD_DIM, LogitsPostHook::kNone,
                                                      POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION,
                                                      MaskMode::kCustom>(
                q, k, v, custom_mask, o, tmp, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h,
                /*window_left=*/-1,
                /*logits_soft_cap*/ 0.f, sm_scale, rope_scale, rope_theta, stream);
          })})});
  return cudaSuccess;
}

/*!
 * \brief FlashAttention prefill CUDA function for a single request.
 * \tparam DTypeIn The data type of input
 * \tparam DTypeOut The data type of output
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary storage (only used for cooperative kernel).
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param kv_layout The layout of input and output.
 * \param pos_encoding_mode The positional encoding mode.
 * \param allow_fp16_qk_reduction Whether to allow accumulating q*k^T with fp16.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o, DTypeOut* tmp,
                                     float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                     uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
                                     bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
                                     PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                     bool allow_fp16_qk_reduction = false,
                                     std::optional<float> maybe_sm_scale = std::nullopt,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_allow_fp16_qk_reduction(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_head_dim(
              head_dim, HEAD_DIM,
              {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                return SinglePrefillWithKVCacheDispatched<HEAD_DIM, LogitsPostHook::kNone,
                                                          POS_ENCODING_MODE,
                                                          ALLOW_FP16_QK_REDUCTION, MASK_MODE>(
                    q, k, v, /*custom_mask=*/nullptr, o, tmp, lse, num_qo_heads, num_kv_heads,
                    qo_len, kv_len, qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h,
                    /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, DTypeKV* k, DTypeKV* v,
    IdType* kv_indptr, IdType* q_offset, IdType* k_rope_pos_offset, DTypeOut* o, float* lse,
    const uint32_t batch_size, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t head_dim, bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    const float rope_scale = 1.f, const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, 0, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, pos_encoding_mode,
              {DISPATCH_allow_fp16_qk_reduction(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                return BatchPrefillWithRaggedKVCacheWrapperDispatched<
                    HEAD_DIM, LogitsPostHook::kNone, pos_encoding_mode, ALLOW_FP16_QK_REDUCTION,
                    MASK_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
                    handler, q, qo_indptr, k, v, kv_indptr, /*custom_mask=*/nullptr,
                    /*qk_indptr=*/nullptr, q_offset, k_rope_pos_offset, o, lse, num_qo_heads,
                    num_kv_heads, qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h,
                    /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, bool causal = true,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE,
              {DISPATCH_allow_fp16_qk_reduction(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                return BatchPrefillWithPagedKVCacheWrapperDispatched<
                    PAGE_STORAGE, HEAD_DIM, LogitsPostHook::kNone, POS_ENCODING_MODE,
                    ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
                    handler, q, qo_indptr, q_offset, paged_kv,
                    /*custom_mask=*/nullptr,
                    /*qk_indptr=*/nullptr, o, lse, num_qo_heads, /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o, DTypeOut* tmp,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
                                    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
                                    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                    std::optional<float> maybe_sm_scale = std::nullopt,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_head_dim(
      head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        SingleDecodeWithKVCacheDispatched<HEAD_DIM, LogitsPostHook::kNone, POS_ENCODING_MODE>(
            q, k, v, o, tmp, num_qo_heads, num_kv_heads, seq_len, kv_layout,
            /*window_left=*/-1,
            /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
      })});
  return cudaSuccess;
}

template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheNoSplitKV(
    DTypeQ* q, IdType* q_offset, paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, float* lse, uint32_t num_qo_heads,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_head_dim(
      head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        return BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PAGE_STORAGE, LogitsPostHook::kNone,
                                                     POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeOut,
                                                     IdType>(
            q, q_offset, paged_kv, kv_partition_info, o, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, lse,
            /*block_valid_mask=*/nullptr, /*padded_batch_size=*/paged_kv.batch_size, num_qo_heads,
            /*window_left=*/-1,
            /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
      })});

  return cudaSuccess;
}

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the temporary buffer
 *   for cooperative kernels.
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeQ The data type of query tensor.
 * \tparam DTypeKV The data type of key-value tensor.
 * \tparam DTypeOut The data type of output tensor.
 * \tparam IdType The data type of index tensor.
 * \param handler The handler for the batch decode forward request.
 * \param q The input tensor.
 * \param paged_kv The paged key-value tensor.
 * \param o The output tensor.
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of heads.
 * \param pos_encoding_mode The positional encoding mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 */
template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler* handler, DTypeQ* q, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_head_dim(paged_kv.head_dim, HEAD_DIM,
                    {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                      return BatchDecodeWithPagedKVCacheWrapperDispatched<
                          PAGE_STORAGE, HEAD_DIM, LogitsPostHook::kNone, POS_ENCODING_MODE, DTypeQ,
                          DTypeKV, DTypeOut, IdType>(
                          handler, q, q_offset, paged_kv, o, lse, num_qo_heads,
                          /*window_left=*/-1,
                          /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
                    })});
  return cudaSuccess;
}

template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeHandlerPlan(BatchDecodeHandler* handler, void* float_buffer,
                                   size_t float_workspace_size_in_bytes, void* int_buffer,
                                   size_t int_workspace_size_in_bytes, IdType* indptr_h,
                                   IdType* last_page_len_h, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, PosEncodingMode pos_encoding_mode) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }
  DISPATCH_head_dim(head_dim, HEAD_DIM, {
    DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
      return handler->PlanDispatched<HEAD_DIM, PAGE_STORAGE, LogitsPostHook::kNone,
                                     POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
          float_buffer, float_workspace_size_in_bytes, int_buffer, int_workspace_size_in_bytes,
          indptr_h, last_page_len_h, batch_size, num_qo_heads, num_kv_heads, page_size);
    });
  });
}

}  // namespace flashinfer
