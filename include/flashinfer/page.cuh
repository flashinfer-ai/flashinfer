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
#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

#include <vector>

#include "fastdiv.cuh"
#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

enum class PageStorage {
  kIndices = 0U,  // Store the pointer to the buffer allocated for paged kv-cache, and indices of
                  // each active offset.
  kPointer = 1U,  // Store the pointers to each active page.
};

/*!
 * \brief The auxiliary information about kv sequence partitioning
 */
template <typename IdType>
struct kv_partition_info_t {
  uint32_t batch_size_before_partition;
  IdType* chunk_indptr;
  IdType* batch_idx_map;
  IdType* chunk_start_pos;
  IdType* seq_lens_before_partition;

  __host__ __device__ __forceinline__ kv_partition_info_t(uint32_t batch_size_before_partition,
                                                          IdType* chunk_indptr,
                                                          IdType* batch_idx_map,
                                                          IdType* chunk_start_pos,
                                                          IdType* seq_lens_before_partition)
      : batch_size_before_partition(batch_size_before_partition),
        chunk_indptr(chunk_indptr),
        batch_idx_map(batch_idx_map),
        chunk_start_pos(chunk_start_pos),
        seq_lens_before_partition(seq_lens_before_partition) {}

  __host__ __device__ __forceinline__ kv_partition_info_t()
      : batch_size_before_partition(0),
        chunk_indptr(nullptr),
        batch_idx_map(nullptr),
        chunk_start_pos(nullptr),
        seq_lens_before_partition(nullptr) {}
};

/*!
 * \brief Paged key-value cache
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimensions in KV-Cache.
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
template <PageStorage page_storage, typename DType, typename IdType>
struct paged_kv_t {
  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;

  // The flattened key-value cache, used when page_storage == kIndices
  // Internal layout:
  // [max_num_pages, num_heads, page_size, head_dim] if layout == HND
  // [max_num_pages, page_size, num_heads, head_dim] if layout == NHD
  DType* k_data;
  DType* v_data;
  // [nnz_pages] The page indices array, used when page_storage == kIndices
  IdType* indices;
  // [nnz_pages] The page pointers array, used when page_storage == kPointer
  DType** kv_ptrs;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  __host__ __device__ __forceinline__ paged_kv_t()
      : num_heads(0),
        page_size(0),
        head_dim(0),
        batch_size(0),
        stride_page(0),
        stride_n(0),
        stride_h(0),
        k_data(nullptr),
        v_data(nullptr),
        indices(nullptr),
        kv_ptrs(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param kv_data The flattened key-value cache
   * \param k_data The flattened key cache
   * \param v_data The flattened value cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   * \note This constructor should only be used when page_storage == kIndices
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* kv_data,
                                      DType* k_data, DType* v_data, IdType* indices, IdType* indptr,
                                      IdType* last_page_len, IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    bool kv_defined = kv_data != nullptr;
    if (kv_defined) {
      stride_page = 2 * num_heads * page_size * head_dim;
      this->k_data = kv_data;
      this->v_data = kv_data + num_heads * page_size * head_dim;
    } else {
      stride_page = num_heads * page_size * head_dim;
      this->k_data = k_data;
      this->v_data = v_data;
    }
    stride_n = layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
    stride_h = layout == QKVLayout::kHND ? page_size * head_dim : head_dim;
  }

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The flattened key cache
   * \param v_data The flattened value cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   * \note This constructor should only be used when page_storage == kIndices
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* k_data,
                                      DType* v_data, IdType* indices, IdType* indptr,
                                      IdType* last_page_len, IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        k_data(k_data),
        v_data(v_data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = num_heads * page_size * head_dim;
    stride_n = layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
    stride_h = layout == QKVLayout::kHND ? page_size * head_dim : head_dim;
  }

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param kv_data The flattened key-value cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   * \note This constructor should only be used when page_storage == kIndices
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* kv_data,
                                      IdType* indices, IdType* indptr, IdType* last_page_len,
                                      IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        k_data(kv_data),
        v_data(kv_data + num_heads * page_size * head_dim),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = 2 * num_heads * page_size * head_dim;
    stride_n = layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
    stride_h = layout == QKVLayout::kHND ? page_size * head_dim : head_dim;
  }

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param kv_ptrs The array of pointers to each active kv page
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   * \note This constructor should only be used when page_storage == kIndices
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType** kv_ptrs,
                                      IdType* indptr, IdType* last_page_len,
                                      IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        kv_ptrs(kv_ptrs),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = 2 * num_heads * page_size * head_dim;
    stride_n = layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
    stride_h = layout == QKVLayout::kHND ? page_size * head_dim : head_dim;
  }

  __host__ __device__ __forceinline__ int64_t kv_ptr_delta() const {
    return page_storage == PageStorage::kPointer
               ? num_heads * page_size * head_dim
               : (int64_t(v_data) - int64_t(k_data)) / sizeof(DType);
  }

  /*!
   * \brief Compute the offset of element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   * \note This function should only be used when page_storage == kIndices
   */
  __host__ __device__ __forceinline__ size_t get_elem_offset(size_t page_idx, size_t head_idx,
                                                             size_t entry_idx,
                                                             size_t feat_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  /*!
   * \brief Compute the offset of element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t get_elem_offset_in_page(size_t head_idx,
                                                                     size_t entry_idx,
                                                                     size_t feat_idx) const {
    return head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  __device__ __forceinline__ DType* get_k_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      return k_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return kv_ptrs[page_iter] + get_elem_offset_in_page(head_idx, entry_idx, feat_idx);
    }
  }

  __device__ __forceinline__ DType* protective_get_k_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      if (page_iter < last_indptr) {
        return k_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
      } else {
        return k_data;
      }
    } else {
      if (page_iter < last_indptr) {
        return kv_ptrs[page_iter] + get_elem_offset_in_page(head_idx, entry_idx, feat_idx);
      } else {
        return *kv_ptrs;
      }
    }
  }

  __device__ __forceinline__ DType* get_v_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      return v_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return (kv_ptrs[page_iter] + kv_ptr_delta()) +
             get_elem_offset_in_page(head_idx, entry_idx, feat_idx);
    }
  }

  __device__ __forceinline__ DType* protective_get_v_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      if (page_iter < last_indptr) {
        return v_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
      } else {
        return v_data;
      }
    } else {
      if (page_iter < last_indptr) {
        return (kv_ptrs[page_iter] + kv_ptr_delta()) +
               get_elem_offset_in_page(head_idx, entry_idx, feat_idx);
      } else {
        return *kv_ptrs;
      }
    }
  }
};

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, PageStorage page_storage, typename DType,
          typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(paged_kv_t<page_storage, DType, IdType> paged_kv,
                                               DType* __restrict__ key, DType* __restrict__ value) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_len[batch_idx];

  uint32_t page_iter = paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size;
  uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;

  DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  vec_t<DType, vec_size>::memcpy(
      k_ptr, key + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);

  vec_t<DType, vec_size>::memcpy(
      v_ptr, value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 */
template <uint32_t head_dim, uint32_t vec_size, PageStorage page_storage, typename DType,
          typename IdType>
__global__ void AppendPagedKVCachePrefillKernel(paged_kv_t<page_storage, DType, IdType> paged_kv,
                                                DType* __restrict__ key, DType* __restrict__ value,
                                                IdType* __restrict__ append_indptr) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_len[batch_idx];
  uint32_t append_seq_len = append_indptr[batch_idx + 1] - append_indptr[batch_idx];
  uint32_t append_start = seq_len - append_seq_len;

#pragma unroll 2
  for (uint32_t j = 0; j < append_seq_len; ++j) {
    uint32_t page_seq_idx = j + append_start;
    uint32_t page_iter = paged_kv.indptr[batch_idx] + page_seq_idx / paged_kv.page_size;
    uint32_t entry_idx = page_seq_idx % paged_kv.page_size;

    DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        k_ptr,
        key + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);

    vec_t<DType, vec_size>::memcpy(
        v_ptr,
        value + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);
  }
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<page_storage, DType, IdType> paged_kv, DType* key,
                                     DType* value, cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    // NOTE(Zihao): could be slow for small batch size, will optimize later
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, page_storage, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Append new keys/values to the paged key-value cache
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, typename DType, typename IdType>
cudaError_t AppendPagedKVCache(paged_kv_t<page_storage, DType, IdType> paged_kv, DType* key,
                               DType* value, IdType* append_indptr, cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    // NOTE(Zihao): could be slow for small batch size, will optimize later
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCachePrefillKernel<HEAD_DIM, vec_size, page_storage, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value, (void*)&append_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_
