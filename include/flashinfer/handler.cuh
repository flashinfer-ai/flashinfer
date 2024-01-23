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

#include "decode.cuh"
#include "prefill.cuh"
#include "rope.cuh"
#include "utils.cuh"

namespace flashinfer {

class BatchDecodeHandler {
 public:
  template <typename DType>
  DType* GetTempFloatBuffer() const {
    return (DType*)float_buffer_;
  }
  template <typename IdType>
  IdType* GetNewIndPtr() const {
    return (IdType*)int_buffer_;
  }
  template <typename IdType>
  IdType* GetNewLastPageLen() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + batch_size_after_partition_ + 1;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetChunkIndPtr() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 2 * batch_size_after_partition_ + 1;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetBatchIdxMap() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 2 * batch_size_after_partition_ +
             batch_size_before_partition_ + 2;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetChunkStartPos() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 3 * batch_size_after_partition_ +
             batch_size_before_partition_ + 2;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetSeqLengthsBeforePartition() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 4 * batch_size_after_partition_ +
             batch_size_before_partition_ + 2;
    } else {
      return nullptr;
    }
  }

  template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
            typename IdType>
  cudaError_t BeginForward(IdType* indptr, IdType* last_page_len, uint32_t batch_size,
                           uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                           uint32_t page_size, RotaryMode rotary_mode) {
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
      FLASHINFER_CUDA_CALL(cudaMallocAsync(&float_buffer_, tmp_size, stream_));
      FLASHINFER_CUDA_CALL(cudaMallocAsync(
          &int_buffer_, sizeof(IdType) * (5 * new_batch_size + batch_size_before_partition_ + 2),
          stream_));
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
    if (float_buffer_ != nullptr) {
      FLASHINFER_CUDA_CALL(cudaFreeAsync(float_buffer_, stream_));
      float_buffer_ = nullptr;
    }
    if (int_buffer_ != nullptr) {
      FLASHINFER_CUDA_CALL(cudaFreeAsync(int_buffer_, stream_));
      int_buffer_ = nullptr;
    }
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
        int_buffer_(nullptr),
        forward_started_(false),
        stream_(nullptr) {}
  ~BatchDecodeHandler() { EndForward(); }

 private:
  uint32_t batch_size_before_partition_;
  uint32_t batch_size_after_partition_;
  void* float_buffer_;
  void* int_buffer_;
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
  cudaError_t BeginForward(IdType* qo_indptr, uint32_t batch_size, uint32_t num_qo_heads,
                           uint32_t num_kv_heads) {
    if (num_qo_heads % num_kv_heads != 0) {
      std::ostringstream err_msg;
      err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
              << num_kv_heads;
      throw std::invalid_argument(err_msg.str());
    }
    uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
    std::vector<IdType> request_indices_h, tile_indices_h;
    std::tie(num_frags_x_, num_qo_tiles_, request_indices_h, tile_indices_h) =
        split_qo_indptr(qo_indptr, batch_size, gqa_group_size, stream_);
    FLASHINFER_CUDA_CALL(cudaMalloc(&request_indices_, sizeof(IdType) * request_indices_h.size()));
    FLASHINFER_CUDA_CALL(cudaMalloc(&tile_indices_, sizeof(IdType) * tile_indices_h.size()));
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
    if (request_indices_ != nullptr) {
      FLASHINFER_CUDA_CALL(cudaFreeAsync(request_indices_, stream_));
      request_indices_ = nullptr;
    }
    if (tile_indices_ != nullptr) {
      FLASHINFER_CUDA_CALL(cudaFreeAsync(tile_indices_, stream_));
      tile_indices_ = nullptr;
    }
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

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the temporary buffer
 *   for cooperative kernels.
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam kv_layout The layout of last 3 dimensions in KV-Cache
 * \tparam DTypeIn The data type of input tensor.
 * \tparam DTypeOut The data type of output tensor.
 * \tparam IdType The data type of index tensor.
 * \param handler The handler for the batch decode forward request.
 * \param q The input tensor.
 * \param paged_kv The paged key-value tensor.
 * \param o The output tensor.
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of heads.
 * \param rotary_mode The rotary mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 * \note This wrapper function should be only called after we call BeginForward function in the
 *   BatchDecodeHandler.
 */
template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler* handler, DTypeIn* q,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> new_paged_kv = paged_kv;
  kv_partition_info_t<IdType> kv_partition_info;
  DTypeOut* tmp = handler->GetTempFloatBuffer<DTypeOut>();
  if (handler->IsForwardStarted()) {
    if (tmp != nullptr) {
      // create auxiliary information for cooperative kernels
      new_paged_kv.batch_size = handler->GetBatchSizeAfterPartition();
      new_paged_kv.indptr = handler->GetNewIndPtr<IdType>();
      new_paged_kv.last_page_len = handler->GetNewLastPageLen<IdType>();
      kv_partition_info.batch_size_before_partition = handler->GetBatchSizeBeforePartition();
      kv_partition_info.chunk_indptr = handler->GetChunkIndPtr<IdType>();
      kv_partition_info.batch_idx_map = handler->GetBatchIdxMap<IdType>();
      kv_partition_info.chunk_start_pos = handler->GetChunkStartPos<IdType>();
      kv_partition_info.seq_lens_before_partition = handler->GetSeqLengthsBeforePartition<IdType>();
    }
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchDecodeHandler's BeginForward() before calling "
               "BatchDecodeWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }
  return BatchDecodeWithPagedKVCache<page_storage, kv_layout, DTypeIn, DTypeOut, IdType>(
      q, new_paged_kv, kv_partition_info, o, tmp, lse, num_qo_heads, rotary_mode, rope_scale,
      rope_theta, stream);
}

template <PageStorage page_storage, QKVLayout kv_layout, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          RotaryMode ROTARY_MODE, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  float* tmp = nullptr;
  IdType* request_indices = nullptr;
  IdType* tile_indices = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t num_qo_tiles = 0U;
  if (handler->IsForwardStarted()) {
    request_indices = handler->GetRequestIndices<IdType>();
    tile_indices = handler->GetTileIndices<IdType>();
    num_frags_x = handler->GetNumFragsX();
    num_qo_tiles = handler->GetNumQOTiles();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }

  SWITCH_NUM_FRAGS_X(
      num_frags_x, NUM_FRAGS_X, {SWITCH_PAGE_SIZE(paged_kv.page_size, PAGE_SIZE, {
        if constexpr (PAGE_SIZE == 0) {
          return BatchPrefillWithPagedKVCacheFallbackDispatched<
              page_storage, kv_layout, NUM_FRAGS_X, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
              ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
              q, request_indices, tile_indices, qo_indptr, paged_kv, o, tmp, lse, num_qo_tiles,
              rope_scale, rope_theta, stream);
        } else {
          return BatchPrefillWithPagedKVCacheDispatched<
              page_storage, kv_layout, NUM_FRAGS_X, PAGE_SIZE, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
              ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
              q, request_indices, tile_indices, qo_indptr, paged_kv, o, tmp, lse, num_qo_tiles,
              rope_scale, rope_theta, stream);
        }
      })});
  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, bool causal = true, RotaryMode rotary_mode = RotaryMode::kNone,
    bool allow_fp16_qk_reduction = false, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  SWITCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {SWITCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {SWITCH_CAUSAL(causal, CAUSAL,
                         {SWITCH_ROTARY_MODE(
                             rotary_mode, ROTARY_MODE,
                             {SWITCH_ALLOW_FP16_QK_REDUCTION(
                                 allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                                   return BatchPrefillWithPagedKVCacheWrapperDispatched<
                                       page_storage, kv_layout, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
                                       ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
                                       handler, q, qo_indptr, paged_kv, o, lse, num_qo_heads,
                                       rope_scale, rope_theta, stream);
                                 })})})})});
  return cudaSuccess;
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT, RotaryMode ROTARY_MODE,
          bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, DTypeOut* o, float* lse, const uint32_t batch_size,
    const uint32_t num_kv_heads, const float rope_scale = 1.f, const float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  float* tmp = nullptr;
  IdType* request_indices = nullptr;
  IdType* tile_indices = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t num_qo_tiles = 0U;
  if (handler->IsForwardStarted()) {
    request_indices = handler->GetRequestIndices<IdType>();
    tile_indices = handler->GetTileIndices<IdType>();
    num_frags_x = handler->GetNumFragsX();
    num_qo_tiles = handler->GetNumQOTiles();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithRaggedKVWrapperCache()";
    throw std::runtime_error(err_msg.str());
  }

  SWITCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithRaggedKVCacheDispatched<NUM_FRAGS_X, GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                   ROTARY_MODE, ALLOW_FP16_QK_REDUCTION, CAUSAL,
                                                   DTypeIn, DTypeOut, IdType>(
        q, request_indices, tile_indices, qo_indptr, k, v, kv_indptr, o, tmp, lse, batch_size,
        num_qo_tiles, num_kv_heads, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, DTypeOut* o, float* lse, const uint32_t batch_size,
    const uint32_t num_qo_heads, const uint32_t num_kv_heads, const uint32_t head_dim,
    bool causal = true, RotaryMode rotary_mode = RotaryMode::kNone,
    bool allow_fp16_qk_reduction = false, const float rope_scale = 1.f,
    const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  constexpr QKVLayout KV_LAYOUT = QKVLayout::kNHD;
  SWITCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {SWITCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {SWITCH_CAUSAL(causal, CAUSAL,
                         {SWITCH_ROTARY_MODE(
                             rotary_mode, ROTARY_MODE,
                             {SWITCH_ALLOW_FP16_QK_REDUCTION(
                                 allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                                   return BatchPrefillWithRaggedKVCacheWrapperDispatched<
                                       GROUP_SIZE, HEAD_DIM, KV_LAYOUT, ROTARY_MODE,
                                       ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
                                       handler, q, qo_indptr, k, v, kv_indptr, o, lse, batch_size,
                                       num_kv_heads, rope_scale, rope_theta, stream);
                                 })})})})});
  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_HANDLER_CUH_
