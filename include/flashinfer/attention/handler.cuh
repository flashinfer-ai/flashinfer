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
#include <unordered_map>
#include <vector>

#include "../page.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"

namespace flashinfer {

template <bool partition_kv, PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem,
          uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
__global__ void BatchDecodeWithPagedKVCacheKernel(
    DTypeIn* __restrict__ q, IdType* __restrict__ q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* __restrict__ o,
    DTypeOut* __restrict__ tmp, float* __restrict__ lse, float sm_scale, float rope_rcp_scale,
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

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeIn A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param tmp_size The estimated temporary buffer size, return 0 if not use partition-kv kernel
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param pos_encoding_mode The positional encoding mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage, QKVLayout kv_layout,
          PosEncodingMode POS_ENCODING_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatched(
    uint32_t& tmp_size, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t batch_size, IdType* kv_indptr, const uint32_t num_qo_heads,
    const uint32_t page_size, cudaStream_t stream) {
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
  constexpr uint32_t num_stages_smem = 2U;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;
  const uint32_t smem_size =
      2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeIn) +
      std::max(tile_size_per_bdx * num_threads * sizeof(DTypeIn*), 2 * bdy * bdz * sizeof(float));

  auto partition_kv_kernel = BatchDecodeWithPagedKVCacheKernel<
      /*partition_kv=*/true, POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size, bdx,
      bdy, bdz, page_storage, kv_layout, DTypeIn, DTypeOut, IdType>;
  int num_blocks_per_sm = 0;
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, partition_kv_kernel, num_threads, smem_size));
  max_grid_size = num_blocks_per_sm * num_sm;
  if (batch_size * num_kv_heads >= max_grid_size) {
    // do not use partition-kv kernel
    tmp_size = 0;
    new_batch_size = batch_size;
  } else {
    // compute max_num_pages_per_batch and new_batch_size
    std::vector<IdType> page_indptr_h(batch_size + 1), num_pages(batch_size);
    if (is_device_ptr(kv_indptr)) {
      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(page_indptr_h.data(), kv_indptr,
                                           sizeof(IdType) * (batch_size + 1),
                                           cudaMemcpyDeviceToHost, stream));
    } else {
      page_indptr_h.assign(kv_indptr, kv_indptr + batch_size + 1);
    }
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      num_pages[batch_idx] = page_indptr_h[batch_idx + 1] - page_indptr_h[batch_idx];
    }
    std::tie(max_num_pages_per_batch, new_batch_size) =
        PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, num_kv_heads, num_pages,
                                                            128 / page_size);
    if (new_batch_size == batch_size) {
      // do not use partition-kv kernel for short sequence
      tmp_size = 0;
    } else {
      tmp_size = num_qo_heads * new_batch_size * (HEAD_DIM * sizeof(DTypeOut) + 2 * sizeof(float));
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
    const uint32_t max_num_pages_per_batch, const uint32_t old_batch_size, const uint32_t page_size,
    IdType* old_indptr, IdType* old_last_page_len, IdType* new_indptr_d,
    IdType* new_last_page_len_d, IdType* chunk_indptr_d, IdType* batch_idx_map_d,
    IdType* chunk_start_d, IdType* seq_lens_before_partition_d, cudaStream_t stream = nullptr) {
  std::vector<IdType> new_page_indptr_h{0}, new_last_page_len_h, chunk_indptr_h{0}, batch_idx_map_h,
      chunk_start_pos_h, seq_lens_before_partition_h;

  std::vector<IdType> old_indptr_h(old_batch_size + 1), old_last_page_len_h(old_batch_size);
  if (is_device_ptr(old_indptr)) {
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(old_indptr_h.data(), old_indptr,
                                         sizeof(IdType) * (old_batch_size + 1),
                                         cudaMemcpyDeviceToHost, stream));
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(old_last_page_len_h.data(), old_last_page_len,
                                         sizeof(IdType) * old_batch_size, cudaMemcpyDeviceToHost,
                                         stream));
    FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
  } else {
    old_indptr_h.assign(old_indptr, old_indptr + old_batch_size + 1);
    old_last_page_len_h.assign(old_last_page_len, old_last_page_len + old_batch_size);
  }

  for (uint32_t batch_idx = 0; batch_idx < old_batch_size; batch_idx++) {
    uint32_t num_chunks =
        ceil_div(old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx], max_num_pages_per_batch);
    chunk_indptr_h.push_back(chunk_indptr_h.back() + num_chunks);
    if (num_chunks == 0) {
      new_page_indptr_h.push_back(old_indptr_h[batch_idx]);
      new_last_page_len_h.push_back(0);
      batch_idx_map_h.push_back(batch_idx);
      chunk_start_pos_h.push_back(0);
      seq_lens_before_partition_h.push_back(0);
    } else {
      uint32_t seq_len_before_partition =
          (old_indptr_h[batch_idx + 1] - old_indptr_h[batch_idx] - 1) * page_size +
          old_last_page_len_h[batch_idx];
      for (uint32_t j = 0; j < num_chunks; ++j) {
        bool is_last = (j + 1) == num_chunks;
        new_page_indptr_h.push_back(min(old_indptr_h[batch_idx] + (j + 1) * max_num_pages_per_batch,
                                        old_indptr_h[batch_idx + 1]));
        new_last_page_len_h.push_back(is_last ? old_last_page_len_h[batch_idx] : page_size);
        batch_idx_map_h.push_back(batch_idx);
        chunk_start_pos_h.push_back(j * max_num_pages_per_batch * page_size);
        seq_lens_before_partition_h.push_back(seq_len_before_partition);
      }
    }
  }

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(new_indptr_d, new_page_indptr_h.data(),
                                       sizeof(IdType) * new_page_indptr_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(new_last_page_len_d, new_last_page_len_h.data(),
                                       sizeof(IdType) * new_last_page_len_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(chunk_indptr_d, chunk_indptr_h.data(),
                                       sizeof(IdType) * chunk_indptr_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(batch_idx_map_d, batch_idx_map_h.data(),
                                       sizeof(IdType) * batch_idx_map_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(chunk_start_d, chunk_start_pos_h.data(),
                                       sizeof(IdType) * chunk_start_pos_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      seq_lens_before_partition_d, seq_lens_before_partition_h.data(),
      sizeof(IdType) * seq_lens_before_partition_h.size(), cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

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

  template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage, QKVLayout kv_layout,
            PosEncodingMode POS_ENCODING_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
  cudaError_t BeginForwardDispatched(void* buffer, size_t workspace_size_in_bytes, IdType* indptr,
                                     IdType* last_page_len, uint32_t batch_size,
                                     uint32_t num_qo_heads, uint32_t page_size) {
    batch_size_before_partition_ = batch_size;
    uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
    auto work_estimation_func =
        BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE, HEAD_DIM, page_storage,
                                                            kv_layout, POS_ENCODING_MODE, DTypeIn,
                                                            DTypeOut, IdType>;
    FLASHINFER_CUDA_CALL(work_estimation_func(tmp_size, max_grid_size, max_num_pages_per_batch,
                                              new_batch_size, batch_size, indptr, num_qo_heads,
                                              page_size, stream_));
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

  template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
            typename IdType>
  cudaError_t BeginForward(void* buffer, size_t workspace_size_in_bytes, IdType* indptr,
                           IdType* last_page_len, uint32_t batch_size, uint32_t num_qo_heads,
                           uint32_t num_kv_heads, uint32_t head_dim, uint32_t page_size,
                           PosEncodingMode pos_encoding_mode) {
    if (num_qo_heads % num_kv_heads != 0) {
      std::ostringstream err_msg;
      err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
              << num_kv_heads;
      throw std::invalid_argument(err_msg.str());
    }
    DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, {
          return BeginForwardDispatched<GROUP_SIZE, HEAD_DIM, page_storage, kv_layout,
                                        POS_ENCODING_MODE, DTypeIn, DTypeOut, IdType>(
              buffer, workspace_size_in_bytes, indptr, last_page_len, batch_size, num_qo_heads,
              page_size);
        });
      });
    });
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
