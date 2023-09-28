#ifndef FLASHINFER_LAYOUT_CUH_
#define FLASHINFER_LAYOUT_CUH_

#include <string>

namespace flashinfer {

/*!
 * \brief The Layout of QKV matrices
 */
enum class QKVLayout {
  // [seq_len, num_heads, head_dim]
  kNHD = 0U,
  // [num_heads, head_dim, seq_len]
  kHND = 1U,
};

template <QKVLayout layout>
__host__ __device__ __forceinline__ size_t get_elem_offset_impl(size_t elem_idx, size_t head_idx,
                                                                size_t feat_idx, size_t seq_len,
                                                                size_t num_heads, size_t head_dim) {
  if constexpr (layout == QKVLayout::kHND) {
    return (head_idx * seq_len + elem_idx) * head_dim + feat_idx;
  } else {
    return (elem_idx * num_heads + head_idx) * head_dim + feat_idx;
  }
}

template <QKVLayout layout>
struct tensor_info_t {
  size_t qo_len;
  size_t kv_len;
  size_t num_heads;
  size_t head_dim;
  __host__ __device__ __forceinline__ tensor_info_t(size_t qo_len, size_t kv_len, size_t num_heads,
                                                    size_t head_dim)
      : qo_len(qo_len), kv_len(kv_len), num_heads(num_heads), head_dim(head_dim) {}

  __host__ __device__ __forceinline__ size_t get_qo_elem_offset(size_t query_idx, size_t head_idx,
                                                                size_t feat_idx) const {
    return get_elem_offset_impl<layout>(query_idx, head_idx, feat_idx, qo_len, num_heads, head_dim);
  }

  __host__ __device__ __forceinline__ size_t get_kv_elem_offset(size_t kv_idx, size_t head_idx,
                                                                size_t feat_idx) const {
    return get_elem_offset_impl<layout>(kv_idx, head_idx, feat_idx, kv_len, num_heads, head_dim);
  }
};

/*!
 * \brief Convert QKVLayout to string
 * \param qkv_layout The QKVLayout to convert
 */
inline std::string QKVLayoutToString(const QKVLayout &qkv_layout) {
  switch (qkv_layout) {
    case QKVLayout::kNHD:
      return "NHD";
    case QKVLayout::kHND:
      return "HND";
    default:
      return "Unknown";
  }
}

}  // namespace flashinfer
#endif  // FLASHINFER_LAYOUT_CUH_