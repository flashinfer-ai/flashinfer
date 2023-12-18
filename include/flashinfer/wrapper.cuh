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
#ifndef FLASHINFER_WRAPPER_CUH_
#define FLASHINFER_WRAPPER_CUH_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "decode.cuh"
#include "rope.cuh"

namespace flashinfer {

template <PageStorage page_storage, typename DTypeIn, typename DTypeOut, typename IdType>
class BatchDecodeBufferManager {
 public:
  float* get_float_buffer() const { return float_buffer_; }
  IdType* new_indptr() const { return int_buffer_; }
  IdType* new_last_page_len() const {
    if (int_buffer_ != nullptr) {
      return int_buffer_ + new_batch_size_ + 1;
    } else {
      return nullptr;
    }
  }
  // cooperative_aux_info starts with cooperative_indptr
  IdType* cooperative_aux_info() const {
    if (int_buffer_ != nullptr) {
      return int_buffer_ + 2 * new_batch_size_ + 1;
    } else {
      return nullptr;
    }
  }
  IdType* cooperative_indptr() const {
    if (int_buffer_ != nullptr) {
      return int_buffer_ + 2 * new_batch_size_ + 1;
    } else {
      return nullptr;
    }
  }
  IdType* batch_idx_map() const {
    if (int_buffer_ != nullptr) {
      return int_buffer_ + 3 * new_batch_size_ + 2;
    } else {
      return nullptr;
    }
  }
  IdType* chunk_start() const {
    if (int_buffer_ != nullptr) {
      return int_buffer_ + 4 * new_batch_size_ + 2;
    } else {
      return nullptr;
    }
  }
  IdType* seq_lens_before_split() const {
    if (int_buffer_ != nullptr) {
      return int_buffer_ + 5 * new_batch_size_ + 2;
    } else {
      return nullptr;
    }
  }

  void begin_forward(const paged_kv_t<page_storage, DTypeIn, IdType>& paged_kv, bool return_lse,
                     uint32_t num_qo_heads, RotaryMode rotary_mode) {
    uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
    SWITCH_RETURN_LSE(return_lse, RETURN_LSE, {
      BatchDecodeWithPagedKVCacheWorkEstimation<RETURN_LSE, page_storage, DTypeIn, DTypeOut,
                                                IdType>(
          tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size, paged_kv, num_qo_heads,
          rotary_mode, stream_);
    });
    new_batch_size_ = new_batch_size;
    if (tmp_size > 0) {
      cudaMallocAsync(&float_buffer_, sizeof(float) * tmp_size, stream_);
      cudaMallocAsync(&int_buffer_, sizeof(IdType) * (6 * new_batch_size + 2), stream_);
      SplitPagedCacheKVComputeAuxiliaryInfo(
          max_num_pages_per_batch, paged_kv, new_indptr(), new_last_page_len(),
          cooperative_indptr(), batch_idx_map(), chunk_start(), seq_lens_before_split(), stream_);
    }
    forward_started_ = true;
  }

  void end_forward() {
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

  bool forward_started() const { return forward_started_; }

  uint32_t get_new_batch_size() const { return new_batch_size_; }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  BatchDecodeBufferManager()
      : new_batch_size_(0U),
        float_buffer_(nullptr),
        int_buffer_(nullptr),
        forward_started_(false),
        stream_(nullptr) {}
  ~BatchDecodeBufferManager() { end_forward(); }

 private:
  uint32_t new_batch_size_;
  float* float_buffer_;
  IdType* int_buffer_;
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
 * \param buf_manager The buffer manager.
 * \param q The input tensor.
 * \param paged_kv The paged key-value tensor.
 * \param o The output tensor.
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of heads.
 * \param rotary_mode The rotary mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 * \note This wrapper function should be only called after we call begin_forward function in the
 *   BatchDecodeBufferManager.
 */
template <PageStorage page_storage, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(
    BatchDecodeBufferManager<page_storage, DTypeIn, DTypeOut, IdType>* buf_manager, DTypeIn* q,
    paged_kv_t<page_storage, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  paged_kv_t<page_storage, DTypeIn, IdType> new_paged_kv = paged_kv;
  if (buf_manager->forward_started()) {
    new_paged_kv.batch_size = buf_manager->get_new_batch_size();
    new_paged_kv.indptr = buf_manager->new_indptr();
    new_paged_kv.last_page_len = buf_manager->new_last_page_len();
    new_paged_kv.cooperative_aux_info = buf_manager->cooperative_aux_info();
  } else {
    std::cerr << "Please call BatchDecodeBufferManager's begin_forward() before calling "
                 "BatchDecodeWithPagedKVCache()"
              << std::endl;
    abort();
  }
  return BatchDecodeWithPagedKVCache<page_storage, DTypeIn, DTypeOut, IdType>(
      q, new_paged_kv, o, buf_manager->get_float_buffer(), lse, num_qo_heads, rotary_mode,
      rope_scale, rope_theta, stream);
}

}  // namespace flashinfer

#endif  // FLASHINFER_WRAPPER_CUH_
