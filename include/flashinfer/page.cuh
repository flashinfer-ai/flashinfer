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
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
struct paged_kv_t {
  uint32_t num_heads;
  uint32_t page_size;
  uint32_t head_dim;
  uint32_t batch_size;

  // The flattened key-value cache, used when page_storage == kIndices
  // Internal layout:
  // [max_num_pages, 2, num_heads, page_size, head_dim] if layout == HND
  // [max_num_pages, 2, page_size, num_heads, head_dim] if layout == NHD
  DType* data;
  // [nnz_pages] The page indices array, used when page_storage == kIndices
  IdType* indices;
  // [nnz_pages] The page pointers array, used when page_storage == kPointer
  DType** ptrs;

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
        data(nullptr),
        indices(nullptr),
        ptrs(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param data The flattened key-value cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   * \note This constructor should only be used when page_storage == kIndices
   */
  __host__ __device__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size,
                                                 uint32_t head_dim, uint32_t batch_size,
                                                 DType* data, IdType* indices, IdType* indptr,
                                                 IdType* last_page_len,
                                                 IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        data(data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param ptrs The array of pointers to each active page
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   * \note This constructor should only be used when page_storage == kIndices
   */
  __host__ __device__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size,
                                                 uint32_t head_dim, uint32_t batch_size,
                                                 DType** ptrs, IdType* indptr,
                                                 IdType* last_page_len,
                                                 IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        ptrs(ptrs),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {}

  /*!
   * \brief Compute the offset of k element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   * \note This function should only be used when page_storage == kIndices
   */
  __host__ __device__ __forceinline__ size_t get_k_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx,
                                                               size_t feat_idx) const {
    return layout == QKVLayout::kHND
               ? ((page_idx * 2 * num_heads + head_idx) * page_size + entry_idx) * head_dim +
                     feat_idx
               : ((page_idx * 2 * page_size + entry_idx) * num_heads + head_idx) * head_dim +
                     feat_idx;
  }

  /*!
   * \brief Compute the offset of k element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t get_k_elem_offset_in_page(size_t head_idx,
                                                                       size_t entry_idx,
                                                                       size_t feat_idx) const {
    return layout == QKVLayout::kHND ? (head_idx * page_size + entry_idx) * head_dim + feat_idx
                                     : (entry_idx * num_heads + head_idx) * head_dim + feat_idx;
  }

  /*!
   * \brief Compute the offset of v element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   * \note This function should only be used when page_storage == kIndices
   */
  __host__ __device__ __forceinline__ size_t get_v_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx,
                                                               size_t feat_idx) const {
    return layout == QKVLayout::kHND
               ? (((page_idx * 2 + 1) * num_heads + head_idx) * page_size + entry_idx) * head_dim +
                     feat_idx
               : (((page_idx * 2 + 1) * page_size + entry_idx) * num_heads + head_idx) * head_dim +
                     feat_idx;
  }

  /*!
   * \brief Compute the offset of v element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t get_v_elem_offset_in_page(size_t head_idx,
                                                                       size_t entry_idx,
                                                                       size_t feat_idx) const {
    return layout == QKVLayout::kHND
               ? ((num_heads + head_idx) * page_size + entry_idx) * head_dim + feat_idx
               : ((page_size + entry_idx) * num_heads + head_idx) * head_dim + feat_idx;
  }

  __host__ __device__ __forceinline__ uint32_t kv_offset_delta() const {
    return num_heads * page_size * head_dim;
  }

  __device__ __forceinline__ DType* get_k_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      return data + get_k_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return ptrs[page_iter] + get_k_elem_offset_in_page(head_idx, entry_idx, feat_idx);
    }
  }

  __device__ __forceinline__ DType* protective_get_k_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      if (page_iter < last_indptr) {
        return data + get_k_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
      } else {
        return data;
      }
    } else {
      if (page_iter < last_indptr) {
        return ptrs[page_iter] + get_k_elem_offset_in_page(head_idx, entry_idx, feat_idx);
      } else {
        return *ptrs;
      }
    }
  }

  __device__ __forceinline__ DType* get_v_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      return data + get_v_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return ptrs[page_iter] + get_v_elem_offset_in_page(head_idx, entry_idx, feat_idx);
    }
  }

  __device__ __forceinline__ DType* protective_get_v_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    if constexpr (page_storage == PageStorage::kIndices) {
      if (page_iter < last_indptr) {
        return data + get_v_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
      } else {
        return data;
      }
    } else {
      if (page_iter < last_indptr) {
        return ptrs[page_iter] + get_v_elem_offset_in_page(head_idx, entry_idx, feat_idx);
      } else {
        return *ptrs;
      }
    }
  }
};

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, PageStorage page_storage, QKVLayout layout,
          typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(
    paged_kv_t<page_storage, layout, DType, IdType> paged_kv, DType* __restrict__ key,
    DType* __restrict__ value) {
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
  DType* v_ptr = k_ptr + paged_kv.kv_offset_delta();
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
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 */
template <uint32_t head_dim, uint32_t vec_size, PageStorage page_storage, QKVLayout layout,
          typename DType, typename IdType>
__global__ void AppendPagedKVCachePrefillKernel(
    paged_kv_t<page_storage, layout, DType, IdType> paged_kv, DType* __restrict__ key,
    DType* __restrict__ value, IdType* __restrict__ append_indptr) {
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
    DType* v_ptr = k_ptr + paged_kv.kv_offset_delta();
    vec_t<DType, vec_size>::memcpy(
        k_ptr,
        key + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);

    vec_t<DType, vec_size>::memcpy(
        v_ptr,
        value + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);
  }
}

/*!
 * \brief CUDA kernel to convert the paged key-value cache to a ragged tensor
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param kv_indptr The indptr array of the ragged tensor
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t head_dim, uint32_t vec_size, PageStorage page_storage, QKVLayout layout,
          typename DType, typename IdType>
__global__ void PagedKVCacheToRaggedTensorKernel(
    paged_kv_t<page_storage, layout, DType, IdType> paged_kv, DType* __restrict__ key,
    DType* __restrict__ value, IdType* __restrict__ kv_indptr) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

#pragma unroll 2
  for (uint32_t j = 0; j < kv_indptr[batch_idx + 1] - kv_indptr[batch_idx]; ++j) {
    uint32_t page_iter = paged_kv.indptr[batch_idx] + j / paged_kv.page_size;
    uint32_t entry_idx = j % paged_kv.page_size;

    DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    DType* v_ptr = k_ptr + paged_kv.kv_offset_delta();
    vec_t<DType, vec_size>::memcpy(
        key + ((kv_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size,
        k_ptr);
    vec_t<DType, vec_size>::memcpy(
        value + ((kv_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size,
        v_ptr);
  }
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
                                     DType* key, DType* value, cudaStream_t stream = nullptr) {
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
    auto kernel =
        AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, page_storage, layout, DType, IdType>;
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
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
cudaError_t AppendPagedKVCache(paged_kv_t<page_storage, layout, DType, IdType> paged_kv, DType* key,
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
    auto kernel =
        AppendPagedKVCachePrefillKernel<HEAD_DIM, vec_size, page_storage, layout, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value, (void*)&append_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Compute the index pointers of the ragged tensor converted from paged key-value
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param kv_indptr The indptr array of the ragged tensor (output)
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
cudaError_t PagedKVCacheToRaggedTensorComputeIndptr(
    paged_kv_t<page_storage, layout, DType, IdType> paged_kv, std::vector<IdType>& kv_indptr_host,
    cudaStream_t stream = nullptr) {
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t page_size = paged_kv.page_size;
  std::vector<IdType> paged_kv_indptr_host(batch_size + 1), paged_kv_last_page_len_host(batch_size);
  kv_indptr_host.resize(batch_size + 1);

  if (is_device_ptr(paged_kv.indptr)) {
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(paged_kv_indptr_host.data(), paged_kv.indptr,
                                         sizeof(IdType) * (batch_size + 1), cudaMemcpyDeviceToHost,
                                         stream));
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(paged_kv_last_page_len_host.data(), paged_kv.last_page_len,
                                         sizeof(IdType) * batch_size, cudaMemcpyDeviceToHost,
                                         stream));
    FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
  } else {
    paged_kv_indptr_host.assign(paged_kv.indptr, paged_kv.indptr + batch_size + 1);
    paged_kv_last_page_len_host.assign(paged_kv.last_page_len, paged_kv.last_page_len + batch_size);
  }

  kv_indptr_host[0] = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    kv_indptr_host[i + 1] =
        kv_indptr_host[i] +
        (paged_kv_indptr_host[i + 1] - paged_kv_indptr_host[i] - 1) * page_size +
        paged_kv_last_page_len_host[i];
  }

  return cudaSuccess;
}

/*!
 * \brief Convert the paged key-value cache to a ragged tensor
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param kv_indptr The indptr array of the ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <PageStorage page_storage, QKVLayout layout, typename DType, typename IdType>
cudaError_t PagedKVCacheToRaggedTensor(paged_kv_t<page_storage, layout, DType, IdType> paged_kv,
                                       DType* key, DType* value, IdType* kv_indptr,
                                       cudaStream_t stream = nullptr) {
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t num_heads = paged_kv.num_heads;
  const uint32_t page_size = paged_kv.page_size;

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DType), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    // NOTE(Zihao): could be slow for small batch size, will optimize later
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel =
        PagedKVCacheToRaggedTensorKernel<HEAD_DIM, vec_size, page_storage, layout, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value, (void*)&kv_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_
