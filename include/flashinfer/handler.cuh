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
  float* GetTempFloatBuffer() const { return float_buffer_; }
  template <typename IdType>
  IdType* GetNewIndPtr() const {
    return (IdType*)int_buffer_;
  }
  template <typename IdType>
  IdType* GetNewLastPageLen() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + new_batch_size_ + 1;
    } else {
      return nullptr;
    }
  }
  // cooperative_aux_info starts with cooperative_indptr
  template <typename IdType>
  IdType* GetCooperativeAuxInfo() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 2 * new_batch_size_ + 1;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetCooperativeIndPtr() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 2 * new_batch_size_ + 1;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetBatchIndexMap() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 3 * new_batch_size_ + 2;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetChunkStartPos() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 4 * new_batch_size_ + 2;
    } else {
      return nullptr;
    }
  }
  template <typename IdType>
  IdType* GetSeqLengthsBeforeSplit() const {
    if (int_buffer_ != nullptr) {
      return ((IdType*)int_buffer_) + 5 * new_batch_size_ + 2;
    } else {
      return nullptr;
    }
  }

  template <PageStorage page_storage, typename DTypeIn, typename DTypeOut, typename IdType>
  void BeginForward(IdType* indptr, IdType* last_page_len, uint32_t batch_size,
                    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                    uint32_t page_size, RotaryMode rotary_mode, bool return_lse) {
    uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
    SWITCH_RETURN_LSE(return_lse, RETURN_LSE, {
      BatchDecodeWithPagedKVCacheWorkEstimation<RETURN_LSE, page_storage, DTypeIn, DTypeOut,
                                                IdType>(
          tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size, batch_size, indptr,
          num_qo_heads, num_kv_heads, head_dim, page_size, rotary_mode, stream_);
    });
    new_batch_size_ = new_batch_size;
    if (tmp_size > 0) {
      cudaMallocAsync(&float_buffer_, sizeof(float) * tmp_size, stream_);
      cudaMallocAsync(&int_buffer_, sizeof(IdType) * (6 * new_batch_size + 2), stream_);
      SplitPagedCacheKVComputeAuxiliaryInfo(
          max_num_pages_per_batch, batch_size, page_size, indptr, last_page_len,
          GetNewIndPtr<IdType>(), GetNewLastPageLen<IdType>(), GetCooperativeIndPtr<IdType>(),
          GetBatchIndexMap<IdType>(), GetChunkStartPos<IdType>(),
          GetSeqLengthsBeforeSplit<IdType>(), stream_);
    }
    forward_started_ = true;
  }

  void EndForward() {
    forward_started_ = false;
    new_batch_size_ = 0;
    if (float_buffer_ != nullptr) {
      cudaFreeAsync(float_buffer_, stream_);
      float_buffer_ = nullptr;
    }
    if (int_buffer_ != nullptr) {
      cudaFreeAsync(int_buffer_, stream_);
      int_buffer_ = nullptr;
    }
  }

  bool IsForwardStarted() const { return forward_started_; }

  uint32_t GetNewBatchSize() const { return new_batch_size_; }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  BatchDecodeHandler()
      : new_batch_size_(0U),
        float_buffer_(nullptr),
        int_buffer_(nullptr),
        forward_started_(false),
        stream_(nullptr) {}
  ~BatchDecodeHandler() { EndForward(); }

 private:
  uint32_t new_batch_size_;
  float* float_buffer_;
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
  void BeginForward(IdType* qo_indptr, uint32_t batch_size, uint32_t gqa_group_size) {
    std::vector<IdType> request_indices_h, tile_indices_h;
    std::tie(num_frags_x_, num_qo_tiles_, request_indices_h, tile_indices_h) =
        split_qo_indptr(qo_indptr, batch_size, gqa_group_size, stream_);
    cudaMalloc(&request_indices_, sizeof(IdType) * request_indices_h.size());
    cudaMalloc(&tile_indices_, sizeof(IdType) * tile_indices_h.size());
    cudaMemcpyAsync(request_indices_, request_indices_h.data(),
                    sizeof(IdType) * request_indices_h.size(), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(tile_indices_, tile_indices_h.data(), sizeof(IdType) * tile_indices_h.size(),
                    cudaMemcpyHostToDevice, stream_);
  }

  void EndForward() {
    forward_started_ = false;
    num_frags_x_ = 0U;
    num_qo_tiles_ = 0U;
    if (request_indices_ != nullptr) {
      cudaFreeAsync(request_indices_, stream_);
      request_indices_ = nullptr;
    }
    if (tile_indices_ != nullptr) {
      cudaFreeAsync(tile_indices_, stream_);
      tile_indices_ = nullptr;
    }
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
template <PageStorage page_storage, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(BatchDecodeHandler* handler, DTypeIn* q,
                                               paged_kv_t<page_storage, DTypeIn, IdType> paged_kv,
                                               DTypeOut* o, float* lse, uint32_t num_qo_heads,
                                               RotaryMode rotary_mode = RotaryMode::kNone,
                                               float rope_scale = 1.f, float rope_theta = 1e4,
                                               cudaStream_t stream = nullptr) {
  paged_kv_t<page_storage, DTypeIn, IdType> new_paged_kv = paged_kv;
  float* tmp = handler->GetTempFloatBuffer();
  if (handler->IsForwardStarted()) {
    if (tmp != nullptr) {
      // create auxiliary information for cooperative kernels
      new_paged_kv.batch_size = handler->GetNewBatchSize();
      new_paged_kv.indptr = handler->GetNewIndPtr<IdType>();
      new_paged_kv.last_page_len = handler->GetNewLastPageLen<IdType>();
      new_paged_kv.cooperative_aux_info = handler->GetCooperativeAuxInfo<IdType>();
    }
  } else {
    std::cerr << "Please call BatchDecodeHandler's BeginForward() before calling "
                 "BatchDecodeWithPagedKVCacheWrapper()"
              << std::endl;
    abort();
  }
  return BatchDecodeWithPagedKVCache<page_storage, DTypeIn, DTypeOut, IdType>(
      q, new_paged_kv, o, tmp, lse, num_qo_heads, rotary_mode, rope_scale, rope_theta, stream);
}

template <PageStorage page_storage, bool RETURN_LSE, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          RotaryMode ROTARY_MODE, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, paged_kv_t<page_storage, DTypeIn, IdType> paged_kv,
    IdType* qo_indptr, DTypeOut* o, float* lse, uint32_t num_qo_heads, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
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
    std::cerr << "Please call BatchPrefillHandler's BeginForward() before calling "
                 "BatchPrefillWithPagedKVCacheWrapper()"
              << std::endl;
    abort();
  }

  SWITCH_NUM_FRAGS_X(
      num_frags_x, NUM_FRAGS_X, {SWITCH_PAGE_SIZE(paged_kv.page_size, PAGE_SIZE, {
        if constexpr (PAGE_SIZE == 0) {
          return BatchPrefillWithPagedKVCacheFallbackDispatched<
              page_storage, RETURN_LSE, NUM_FRAGS_X, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
              ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
              q, request_indices, tile_indices, qo_indptr, paged_kv, o, tmp, lse, num_qo_tiles,
              rope_scale, rope_theta, stream);
        } else {
          return BatchPrefillWithPagedKVCacheDispatched<
              page_storage, RETURN_LSE, NUM_FRAGS_X, PAGE_SIZE, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
              ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
              q, request_indices, tile_indices, qo_indptr, paged_kv, o, tmp, lse, num_qo_tiles,
              rope_scale, rope_theta, stream);
        }
      })});
  return cudaSuccess;
}

template <bool RETURN_LSE, uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout LAYOUT,
          RotaryMode ROTARY_MODE, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, DTypeOut* o, float* lse, const uint32_t batch_size,
    const uint32_t num_kv_heads, const float rope_scale, const float rope_theta,
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
    std::cerr << "Please call BatchPrefillHandler's BeginForward() before calling "
                 "BatchPrefillWithRaggedKVWrapperCache()"
              << std::endl;
    abort();
  }

  SWITCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithRaggedKVCacheDispatched<RETURN_LSE, NUM_FRAGS_X, GROUP_SIZE, HEAD_DIM,
                                                   LAYOUT, ROTARY_MODE, ALLOW_FP16_QK_REDUCTION,
                                                   CAUSAL, DTypeIn, DTypeOut, IdType>(
        q, request_indices, tile_indices, qo_indptr, k, v, kv_indptr, o, tmp, lse, batch_size,
        num_qo_tiles, num_kv_heads, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_HANDLER_CUH_
