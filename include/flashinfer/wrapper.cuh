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
  struct Value {
    uint32_t new_batch_size;
    float* float_buffer;
    IdType* int_buffer;
    Value() {
      new_batch_size = 0U;
      float_buffer = nullptr;
      int_buffer = nullptr;
    }
    IdType* new_indptr() const { return int_buffer; }
    IdType* new_last_page_offset() const {
      if (int_buffer != nullptr) {
        return int_buffer + new_batch_size + 1;
      } else {
        return nullptr;
      }
    }
    // cooperative_aux_info starts with cooperative_indptr
    IdType* cooperative_aux_info() const {
      if (int_buffer != nullptr) {
        return int_buffer + 2 * new_batch_size + 1;
      } else {
        return nullptr;
      }
    }
    IdType* cooperative_indptr() const {
      if (int_buffer != nullptr) {
        return int_buffer + 2 * new_batch_size + 1;
      } else {
        return nullptr;
      }
    }
    IdType* batch_idx_map() const {
      if (int_buffer != nullptr) {
        return int_buffer + 3 * new_batch_size + 2;
      } else {
        return nullptr;
      }
    }
    IdType* chunk_start() const {
      if (int_buffer != nullptr) {
        return int_buffer + 4 * new_batch_size + 2;
      } else {
        return nullptr;
      }
    }
    IdType* seq_lens_before_split() const {
      if (int_buffer != nullptr) {
        return int_buffer + 5 * new_batch_size + 2;
      } else {
        return nullptr;
      }
    }
  };

  Value Get(const paged_kv_t<page_storage, DTypeIn, IdType>& paged_kv, uint32_t num_qo_heads,
            RotaryMode rotary_mode) {
    if (paged_kv.layer_idx == 0) {
      FreeValue();
      uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
      BatchDecodeWithPagedKVCacheWorkEstimation<page_storage, DTypeIn, DTypeOut, IdType>(
          tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size, paged_kv, num_qo_heads,
          rotary_mode, stream_);
      value_.new_batch_size = new_batch_size;
      if (tmp_size > 0) {
        cudaMallocAsync(&value_.float_buffer, sizeof(float) * tmp_size, stream_);
        cudaMallocAsync(&value_.int_buffer, sizeof(IdType) * (6 * new_batch_size + 2), stream_);
        SplitPagedCacheKVComputeAuxiliaryInfo(
            max_num_pages_per_batch, paged_kv, value_.new_indptr(), value_.new_last_page_offset(),
            value_.cooperative_indptr(), value_.batch_idx_map(), value_.chunk_start(),
            value_.seq_lens_before_split(), stream_);
      }
    }
    return value_;
  }

  void FreeValue() {
    value_.new_batch_size = 0;
    if (value_.float_buffer != nullptr) {
      cudaFreeAsync(value_.float_buffer, stream_);
      value_.float_buffer = nullptr;
    }
    if (value_.int_buffer != nullptr) {
      cudaFreeAsync(value_.int_buffer, stream_);
      value_.int_buffer = nullptr;
    }
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  BatchDecodeBufferManager() : stream_(nullptr) {}
  ~BatchDecodeBufferManager() { FreeValue(); }

 private:
  Value value_;
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
 * \param num_qo_heads The number of heads.
 * \param rotary_mode The rotary mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 * \note This wrapper function only computes the temporary buffer size and allocates the
 *   temporary buffer when layer_idx is 0.
 */
template <PageStorage page_storage, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(
    BatchDecodeBufferManager<page_storage, DTypeIn, DTypeOut, IdType>* buf_manager, DTypeIn* q,
    paged_kv_t<page_storage, DTypeIn, IdType> paged_kv, DTypeOut* o, uint32_t num_qo_heads,
    RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  auto value = buf_manager->Get(paged_kv, num_qo_heads, rotary_mode);
  paged_kv_t<page_storage, DTypeIn, IdType> new_paged_kv = paged_kv;
  if (value.float_buffer != nullptr) {
    new_paged_kv.batch_size = value.new_batch_size;
    new_paged_kv.indptr = value.new_indptr();
    new_paged_kv.last_page_offset = value.new_last_page_offset();
    new_paged_kv.cooperative_aux_info = value.cooperative_aux_info();
  }
  return BatchDecodeWithPagedKVCache<page_storage, DTypeIn, DTypeOut, IdType>(
      q, new_paged_kv, o, value.float_buffer, num_qo_heads, rotary_mode, rope_scale, rope_theta,
      stream);
}

}  // namespace flashinfer

#endif  // FLASHINFER_WRAPPER_CUH_
