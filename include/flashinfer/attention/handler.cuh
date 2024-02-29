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
#ifndef FLASHINFER_HANDLER_CUH_
#define FLASHINFER_HANDLER_CUH_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../rope.cuh"
#include "../utils.cuh"
#include "decode.cuh"

namespace flashinfer {

struct AlignedAlloactor {
  void* ptr;
  size_t space;
  AlignedAlloactor(void* buf, size_t space) : ptr(buf), space(space) {}
  template <typename T>
  T* aligned_alloc(size_t size, size_t alignment) {
    if (std::align(alignment, size, ptr, space)) {
      T* result = reinterpret_cast<T*>(ptr);
      ptr = (char*)ptr + size;
      space -= size;
      return result;
    } else {
      throw std::runtime_error("RuntimeError: Out of workspace memory in AlignedAlloactor");
    }
    return nullptr;
  }
};

class BatchDecodeHandler {
 public:
  template <typename DType>
  DType* GetTempFloatBuffer() const {
    return (DType*)float_buffer_;
  }
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

  template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
            typename IdType>
  cudaError_t BeginForward(void* buffer, size_t workspace_size_in_bytes, IdType* indptr,
                           IdType* last_page_len, uint32_t batch_size, uint32_t num_qo_heads,
                           uint32_t num_kv_heads, uint32_t head_dim, uint32_t page_size,
                           RotaryMode rotary_mode) {
    batch_size_before_partition_ = batch_size;
    uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
    auto work_estimation_func =
        BatchDecodeWithPagedKVCacheWorkEstimation<page_storage, kv_layout, DTypeIn, DTypeOut,
                                                  IdType>;
    FLASHINFER_CUDA_CALL(work_estimation_func(
        tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size, batch_size, indptr,
        num_qo_heads, num_kv_heads, head_dim, page_size, rotary_mode, stream_));
    batch_size_after_partition_ = new_batch_size;
    if (tmp_size > 0) {
      AlignedAlloactor allocator(buffer, workspace_size_in_bytes);
      float_buffer_ = allocator.aligned_alloc<void*>(tmp_size, 16);
      new_indptr_ =
          allocator.aligned_alloc<void*>((batch_size_after_partition_ + 1) * sizeof(IdType), 16);
      new_last_page_len_ =
          allocator.aligned_alloc<void*>(batch_size_after_partition_ * sizeof(IdType), 16);
      chunk_indptr_ =
          allocator.aligned_alloc<void*>((batch_size_before_partition_ + 1) * sizeof(IdType), 16);
      batch_idx_map_ =
          allocator.aligned_alloc<void*>(batch_size_after_partition_ * sizeof(IdType), 16);
      chunk_start_pos_ =
          allocator.aligned_alloc<void*>(batch_size_after_partition_ * sizeof(IdType), 16);
      seq_lengths_before_partition_ =
          allocator.aligned_alloc<void*>(batch_size_after_partition_ * sizeof(IdType), 16);
      FLASHINFER_CUDA_CALL(PartitionPagedKVCacheComputeAuxiliaryInfo(
          max_num_pages_per_batch, batch_size, page_size, indptr, last_page_len,
          GetNewIndPtr<IdType>(), GetNewLastPageLen<IdType>(), GetChunkIndPtr<IdType>(),
          GetBatchIdxMap<IdType>(), GetChunkStartPos<IdType>(),
          GetSeqLengthsBeforePartition<IdType>(), stream_));
    }
    forward_started_ = true;
    return cudaSuccess;
  }

  cudaError_t EndForward() {
    forward_started_ = false;
    batch_size_before_partition_ = 0;
    batch_size_after_partition_ = 0;
    float_buffer_ = nullptr;
    new_indptr_ = nullptr;
    new_last_page_len_ = nullptr;
    chunk_indptr_ = nullptr;
    batch_idx_map_ = nullptr;
    chunk_start_pos_ = nullptr;
    seq_lengths_before_partition_ = nullptr;
    return cudaSuccess;
  }

  bool IsForwardStarted() const { return forward_started_; }

  uint32_t GetBatchSizeBeforePartition() const { return batch_size_before_partition_; }

  uint32_t GetBatchSizeAfterPartition() const { return batch_size_after_partition_; }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  BatchDecodeHandler()
      : batch_size_after_partition_(0U),
        float_buffer_(nullptr),
        new_indptr_(nullptr),
        new_last_page_len_(nullptr),
        chunk_indptr_(nullptr),
        batch_idx_map_(nullptr),
        chunk_start_pos_(nullptr),
        seq_lengths_before_partition_(nullptr),
        forward_started_(false),
        stream_(nullptr) {}
  ~BatchDecodeHandler() { EndForward(); }

 private:
  uint32_t batch_size_before_partition_;
  uint32_t batch_size_after_partition_;
  void* float_buffer_;
  void* new_indptr_;
  void* new_last_page_len_;
  void* chunk_indptr_;
  void* batch_idx_map_;
  void* chunk_start_pos_;
  void* seq_lengths_before_partition_;
  bool forward_started_;
  cudaStream_t stream_;
};

class BatchPrefillHandler {
 public:
  template <typename IdType>
  IdType* GetRequestIndices() const {
    return (IdType*)request_indices_;
  }

  template <typename IdType>
  IdType* GetTileIndices() const {
    return (IdType*)tile_indices_;
  }

  uint32_t GetNumFragsX() const { return num_frags_x_; }

  uint32_t GetNumQOTiles() const { return num_qo_tiles_; }

  bool IsForwardStarted() const { return request_indices_ != nullptr; }

  template <typename IdType>
  cudaError_t BeginForward(void* buffer, size_t workspace_size_in_bytes, IdType* qo_indptr,
                           uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                           uint32_t head_dim) {
    if (num_qo_heads % num_kv_heads != 0) {
      std::ostringstream err_msg;
      err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
              << num_kv_heads;
      throw std::invalid_argument(err_msg.str());
    }
    uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
    std::vector<IdType> request_indices_h, tile_indices_h;
    std::tie(num_frags_x_, num_qo_tiles_, request_indices_h, tile_indices_h) =
        split_qo_indptr(qo_indptr, batch_size, gqa_group_size, head_dim, stream_);
    AlignedAlloactor allocator(buffer, workspace_size_in_bytes);
    request_indices_ =
        allocator.aligned_alloc<void*>(sizeof(IdType) * request_indices_h.size(), 16);
    tile_indices_ = allocator.aligned_alloc<void*>(sizeof(IdType) * tile_indices_h.size(), 16);
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_, request_indices_h.data(),
                                         sizeof(IdType) * request_indices_h.size(),
                                         cudaMemcpyHostToDevice, stream_));
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(tile_indices_, tile_indices_h.data(),
                                         sizeof(IdType) * tile_indices_h.size(),
                                         cudaMemcpyHostToDevice, stream_));
    return cudaSuccess;
  }

  cudaError_t EndForward() {
    forward_started_ = false;
    num_frags_x_ = 0U;
    num_qo_tiles_ = 0U;
    request_indices_ = nullptr;
    tile_indices_ = nullptr;
    return cudaSuccess;
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  BatchPrefillHandler()
      : request_indices_(nullptr),
        tile_indices_(nullptr),
        num_frags_x_(0U),
        num_qo_tiles_(0U),
        forward_started_(false),
        stream_(nullptr) {}
  ~BatchPrefillHandler() { EndForward(); }

 private:
  void* request_indices_;
  void* tile_indices_;
  uint32_t num_frags_x_;
  uint32_t num_qo_tiles_;
  bool forward_started_;
  cudaStream_t stream_;
};

}  // namespace flashinfer
#endif  // FLASHINFER_HANDLER_CUH_
