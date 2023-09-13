#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \note layout: [num_pages, num_layers, 2, num_heads, page_size, head_dim]
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  size_t num_pages;
  size_t num_layers;
  size_t layer_idx;
  size_t num_heads;
  size_t page_size;
  size_t head_dim;
  size_t batch_size;
  // [num_pages * num_layers * 2 * num_heads * page_size * head_dim] The flattened key-value cache
  DType* data;
  // [batch_size + 1] The page indptr array, with the first element 0
  IdType* indptr;
  // [nnz_pages] The page indices array
  IdType* indices;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_offset;
  __host__ __device__ __forceinline__ paged_kv_t(size_t num_pages, size_t num_layers,
                                                 size_t layer_idx, size_t num_heads,
                                                 size_t page_size, size_t head_dim,
                                                 size_t batch_size, DType* data, IdType* indptr,
                                                 IdType* indices, IdType* last_page_offset)
      : num_pages(num_pages),
        num_layers(num_layers),
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

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_