#ifndef FLASHINFER_PAGE_CUH_
#define FLAHSINFER_PAGE_CUH_

namespace flashinfer {

/*!
 * \brief Paged key-value cache, layout: [num_pages, num_layers, 2, num_heads, page_size, head_dim]
 */
template <typename T>
struct paged_kv_t {
  size_t num_pages;
  size_t num_layers;
  size_t layer_idx;
  size_t num_heads;
  size_t page_size;
  size_t head_dim;
  T *data;
  __host__ __device__ __forceinline__ paged_kv_t(size_t num_pages, size_t num_layers,
                                                 size_t layer_idx, size_t num_heads,
                                                 size_t page_size, size_t head_dim, T *data)
      : num_pages(num_pages),
        num_layers(num_layers),
        layer_idx(layer_idx),
        num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        data(data) {}

  __device__ __forceinline__ size_t get_k_offset(size_t page_idx, size_t head_idx, size_t kv_idx,
                                                 size_t feat_idx) {
    return page_idx * num_layers * 2 * num_heads * page_size * head_dim +
           layer_idx * 2 * num_heads * page_size * head_dim + head_idx * page_size * head_dim +
           kv_idx * head_dim + feat_idx;
  }

  __device__ __forceinline__ size_t get_v_offset(size_t page_idx, size_t head_idx, size_t kv_idx,
                                                 size_t feat_idx) {
    return page_idx * num_layers * 2 * num_heads * page_size * head_dim +
           (layer_idx * 2 + 1) * num_heads * page_size * head_dim +
           head_idx * page_size * head_dim + kv_idx * head_dim + feat_idx;
  }
};

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_