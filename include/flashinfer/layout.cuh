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

template <QKVLayout qkv_layout>
__device__ __forceinline__ size_t get_kv_offset(size_t pos, size_t head_idx, size_t feat_idx,
                                                size_t seq_len, size_t num_heads, size_t head_dim) {
  if constexpr (qkv_layout == QKVLayout::kHND) {
    return (head_idx * seq_len + pos) * head_dim + feat_idx;
  } else {
    return (pos * num_heads + head_idx) * head_dim + feat_idx;
  }
}

}  // namespace flashinfer
#endif  // FLASHINFER_LAYOUT_CUH_