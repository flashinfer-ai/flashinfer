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
  // [num_heads, seq_len, head_dim]
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

template <QKVLayout layout, size_t head_dim>
__host__ __device__ __forceinline__ size_t get_elem_offset_impl(size_t elem_idx, size_t head_idx,
                                                                size_t feat_idx, size_t seq_len,
                                                                size_t num_heads) {
  if constexpr (layout == QKVLayout::kHND) {
    return (head_idx * seq_len + elem_idx) * head_dim + feat_idx;
  } else {
    return (elem_idx * num_heads + head_idx) * head_dim + feat_idx;
  }
}

template <QKVLayout layout, uint32_t head_dim>
__host__ __device__ __forceinline__ uint32_t get_n_stride_impl(uint32_t num_heads) {
  return layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
}

template <QKVLayout layout, uint32_t head_dim>
__host__ __device__ __forceinline__ uint32_t get_h_stride_impl(uint32_t seq_len) {
  return layout == QKVLayout::kNHD ? head_dim : seq_len * head_dim;
}

template <QKVLayout kv_layout, uint32_t head_dim>
struct tensor_info_t {
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  __host__ __device__ __forceinline__ tensor_info_t(uint32_t qo_len, uint32_t kv_len,
                                                    uint32_t num_qo_heads, uint32_t num_kv_heads)
      : qo_len(qo_len), kv_len(kv_len), num_qo_heads(num_qo_heads), num_kv_heads(num_kv_heads) {}

  __host__ __device__ __forceinline__ size_t get_qo_elem_offset(uint32_t qo_idx,
                                                                uint32_t qo_head_idx,
                                                                uint32_t feat_idx) const {
    return get_elem_offset_impl<QKVLayout::kNHD, head_dim>(qo_idx, qo_head_idx, feat_idx, qo_len,
                                                           num_qo_heads);
  }

  __host__ __device__ __forceinline__ size_t get_kv_elem_offset(uint32_t kv_idx,
                                                                uint32_t kv_head_idx,
                                                                uint32_t feat_idx) const {
    return get_elem_offset_impl<kv_layout, head_dim>(kv_idx, kv_head_idx, feat_idx, kv_len,
                                                     num_kv_heads);
  }

  __host__ __device__ __forceinline__ uint32_t get_group_size() const {
    return num_qo_heads / num_kv_heads;
  }

  __host__ __device__ __forceinline__ uint32_t get_qo_n_stride() const {
    return get_n_stride_impl<QKVLayout::kNHD, head_dim>(num_qo_heads);
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_n_stride() const {
    return get_n_stride_impl<kv_layout, head_dim>(num_kv_heads);
  }

  __host__ __device__ __forceinline__ uint32_t get_qo_h_stride() const {
    return get_h_stride_impl<QKVLayout::kNHD, head_dim>(qo_len);
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_h_stride() const {
    return get_h_stride_impl<kv_layout, head_dim>(kv_len);
  }
};

/*!
 * \brief Convert QKVLayout to string
 * \param layout The QKVLayout to convert
 */
inline std::string QKVLayoutToString(const QKVLayout& layout) {
  switch (layout) {
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
