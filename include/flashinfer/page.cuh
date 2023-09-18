#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, head_dim]
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  size_t num_layers;
  size_t layer_idx;
  size_t num_heads;
  size_t page_size;
  size_t head_dim;
  size_t batch_size;
  // [max_num_pages * num_layers * 2 * num_heads * page_size * head_dim]
  // The flattened key-value cache
  DType* data;
  // [batch_size + 1] The page indptr array, with the first element 0
  IdType* indptr;
  // [nnz_pages] The page indices array
  IdType* indices;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_offset;
  /*!
   * \brief Construct a paged key-value cache
   * \param num_layers The number of layers
   * \param layer_idx The index of the layer
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param data The flattened key-value cache
   * \param indptr The page indptr array
   * \param indices The page indices array
   * \param last_page_offset The offset of the last page for each request in the batch
   */
  __host__ __device__ __forceinline__ paged_kv_t(size_t num_layers, size_t layer_idx,
                                                 size_t num_heads, size_t page_size,
                                                 size_t head_dim, size_t batch_size, DType* data,
                                                 IdType* indptr, IdType* indices,
                                                 IdType* last_page_offset)
      : num_layers(num_layers),
        layer_idx(layer_idx),
        num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        data(data),
        indptr(indptr),
        indices(indices),
        last_page_offset(last_page_offset) {}

  __host__ __device__ __forceinline__ size_t get_k_offset(size_t page_idx, size_t head_idx,
                                                          size_t entry_idx, size_t feat_idx) {
    return (((page_idx * num_layers + layer_idx) * 2 * num_heads + head_idx) * page_size +
            entry_idx) *
               head_dim +
           feat_idx;
  }

  __host__ __device__ __forceinline__ size_t get_v_offset(size_t page_idx, size_t head_idx,
                                                          size_t entry_idx, size_t feat_idx) {
    return ((((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head_idx) * page_size +
            entry_idx) *
               head_dim +
           feat_idx;
  }
};

template <size_t head_dim, size_t vec_size, size_t bdx, size_t bdy, typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(paged_kv_t<DType, IdType> paged_kv,
                                               DType* __restrict__ key, DType* __restrict__ value) {
  size_t tx = threadIdx.x, ty = threadIdx.y;
  size_t num_heads = paged_kv.num_heads;
  size_t batch_idx = blockIdx.x / (num_heads / bdy);
  size_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;

  size_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_offset[batch_idx];

  size_t page_idx =
      paged_kv.indices[paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size];
  size_t entry_idx = (seq_len - 1) % paged_kv.page_size;

  vec_t<DType, vec_size>::memcpy(
      paged_kv.data + paged_kv.get_k_offset(page_idx, head_idx, entry_idx, tx * vec_size),
      key + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);

  vec_t<DType, vec_size>::memcpy(
      paged_kv.data + paged_kv.get_v_offset(page_idx, head_idx, entry_idx, tx * vec_size),
      value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

template <size_t head_dim, size_t vec_size, size_t bdx, size_t bdy, typename DType, typename IdType>
__global__ void AppendPagedKVCachePrefillKernel(paged_kv_t<DType, IdType> paged_kv,
                                                DType* __restrict__ key, DType* __restrict__ value,
                                                IdType* __restrict__ append_indptr) {
  size_t tx = threadIdx.x, ty = threadIdx.y;
  size_t num_heads = paged_kv.num_heads;
  size_t batch_idx = blockIdx.x / (num_heads / bdy);
  size_t head_idx = (blockIdx.x % (num_heads / bdy)) * bdy + ty;

  size_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_offset[batch_idx];
  size_t append_seq_len = append_indptr[batch_idx + 1] - append_indptr[batch_idx];
  size_t append_start = seq_len - append_seq_len;

#pragma unroll 2
  for (size_t j = 0; j < append_seq_len; ++j) {
    size_t page_seq_idx = j + append_start;
    size_t page_idx =
        paged_kv.indices[paged_kv.indptr[batch_idx] + page_seq_idx / paged_kv.page_size];
    size_t entry_idx = page_seq_idx % paged_kv.page_size;

    vec_t<DType, vec_size>::memcpy(
        paged_kv.data + paged_kv.get_k_offset(page_idx, head_idx, entry_idx, tx * vec_size),
        key + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);

    vec_t<DType, vec_size>::memcpy(
        paged_kv.data + paged_kv.get_v_offset(page_idx, head_idx, entry_idx, tx * vec_size),
        value + ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size);
  }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                     cudaStream_t stream = nullptr, size_t dev_id = 0) {
  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  size_t head_dim = paged_kv.head_dim;
  size_t batch_size = paged_kv.batch_size;
  size_t num_heads = paged_kv.num_heads;
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr size_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    constexpr size_t bdx = HEAD_DIM / vec_size;
    constexpr size_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCachePrefill(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                      IdType* append_indptr, cudaStream_t stream = nullptr,
                                      size_t dev_id = 0) {
  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  size_t head_dim = paged_kv.head_dim;
  size_t batch_size = paged_kv.batch_size;
  size_t num_heads = paged_kv.num_heads;
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr size_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    constexpr size_t bdx = HEAD_DIM / vec_size;
    constexpr size_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCachePrefillKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value, (void*)&append_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_