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
#ifndef FLASHINFER_BUFFER_MANAGER_CUH_
#define FLASHINFER_BUFFER_MANAGER_CUH_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "decode.cuh"
#include "rope.cuh"

namespace flashinfer {

namespace {

/*!
 * \brief Combine hash values.
 * \note Adopted from boost::hash_combine.
 */
template <class T>
void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace

template <typename DTypeIn, typename DTypeOut, typename IdType>
class BatchDecodeBufferManager {
 public:
  struct Key {
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t page_size;
    uint32_t head_dim;
    uint32_t batch_size;
    IdType* indptr;
    IdType* indices;
    IdType* last_page_offset;
    RotaryMode rotary_mode;
  };

  struct KeyHasher {
    uint64_t operator()(const Key& key) const {
      uint64_t seed(1111);
      hash_combine(seed, key.num_qo_heads);
      hash_combine(seed, key.num_kv_heads);
      hash_combine(seed, key.page_size);
      hash_combine(seed, key.head_dim);
      hash_combine(seed, key.batch_size);
      hash_combine(seed, key.indptr);
      hash_combine(seed, key.indices);
      hash_combine(seed, key.last_page_offset);
      hash_combine(seed, uint32_t(key.rotary_mode));
      return seed;
    }
  };

  struct KeyEqualFn {
    bool operator()(const Key& lhs, const Key& rhs) const {
      return lhs.num_qo_heads == rhs.num_qo_heads && lhs.num_kv_heads == rhs.num_kv_heads &&
             lhs.page_size == rhs.page_size && lhs.head_dim == rhs.head_dim &&
             lhs.batch_size == rhs.batch_size && lhs.indptr == rhs.indptr &&
             lhs.indices == rhs.indices && lhs.last_page_offset == rhs.last_page_offset &&
             lhs.rotary_mode == rhs.rotary_mode;
    }
  };

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

  uint32_t GetAllocatedBufferSize() const { return allocated_buffer_size_; }

  Value Get(const paged_kv_t<DTypeIn, IdType>& paged_kv, uint32_t num_qo_heads,
            RotaryMode rotary_mode) {
    Key key{num_qo_heads,      paged_kv.num_heads,        paged_kv.page_size,
            paged_kv.head_dim, paged_kv.batch_size,       paged_kv.indptr,
            paged_kv.indices,  paged_kv.last_page_offset, rotary_mode};
    Value value;
    if (cached_tmp_buffer_.find(key) == cached_tmp_buffer_.end()) {
      uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
      BatchDecodeWithPagedKVCacheWorkEstimation<DTypeIn, DTypeOut, IdType>(
          tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size, paged_kv, num_qo_heads,
          rotary_mode, stream_);
      value.new_batch_size = new_batch_size;
      if (tmp_size > 0) {
        cudaMallocAsync(&value.float_buffer, sizeof(float) * tmp_size, stream_);
        cudaMallocAsync(&value.int_buffer, sizeof(IdType) * (6 * new_batch_size + 2), stream_);
        allocated_buffer_size_ +=
            sizeof(float) * tmp_size + sizeof(IdType) * (6 * new_batch_size + 2);
        SplitPagedCacheKVComputeAuxiliaryInfo(
            max_num_pages_per_batch, paged_kv, value.new_indptr(), value.new_last_page_offset(),
            value.cooperative_indptr(), value.batch_idx_map(), value.chunk_start(),
            value.seq_lens_before_split(), stream_);
      }
      cached_tmp_buffer_[key] = value;
      return value;
    } else {
      return cached_tmp_buffer_[key];
    }
  }

  void ClearCache() {
    allocated_buffer_size_ = 0U;
    for (const auto& [k, v] : cached_tmp_buffer_) {
      if (v.float_buffer != nullptr) {
        cudaFreeAsync(v.float_buffer, stream_);
      }
      if (v.int_buffer != nullptr) {
        cudaFreeAsync(v.int_buffer, stream_);
      }
    }
    cached_tmp_buffer_.clear();
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  BatchDecodeBufferManager() : allocated_buffer_size_(0U), cached_tmp_buffer_{}, stream_(nullptr) {}

 private:
  uint32_t allocated_buffer_size_;
  std::unordered_map<Key, Value, KeyHasher, KeyEqualFn> cached_tmp_buffer_;
  cudaStream_t stream_;
};

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(
    BatchDecodeBufferManager<DTypeIn, DTypeOut, IdType>* buf_manager, DTypeIn* q,
    paged_kv_t<DTypeIn, IdType> paged_kv, DTypeOut* o, uint32_t num_qo_heads,
    RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  auto value = buf_manager->Get(paged_kv, num_qo_heads, rotary_mode);
  paged_kv_t new_paged_kv = paged_kv;
  new_paged_kv.batch_size = value.new_batch_size;
  new_paged_kv.indptr = value.new_indptr();
  new_paged_kv.last_page_offset = value.new_last_page_offset();
  new_paged_kv.cooperative_indptr = value.cooperative_indptr();
  new_paged_kv.batch_idx_map = value.batch_idx_map();
  new_paged_kv.chunk_start = value.chunk_start();
  new_paged_kv.seq_lens_before_split = value.seq_lens_before_split();
  return BatchDecodeWithPagedKVCache<DTypeIn, DTypeOut, IdType>(
      q, new_paged_kv, o, value.float_buffer, num_qo_heads, rotary_mode, rope_scale, rope_theta,
      stream);
}

}  // namespace flashinfer

#endif  // FLASHINFER_BUFFER_MANAGER_CUH_
