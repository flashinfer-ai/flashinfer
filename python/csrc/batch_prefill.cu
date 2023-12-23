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
#include <flashinfer.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor batch_prefill_with_paged_kv_cache(
    torch::Tensor q, torch::Tensor q_indptr, torch::Tensor kv_data, torch::Tensor kv_indptr,
    torch::Tensor kv_indices, torch::Tensor kv_last_page_len, unsigned int page_size, bool causal,
    unsigned int layout, unsigned int rotary_mode, bool allow_fp16_qk_reduction, float rope_scale,
    float rope_theta) {
  CHECK_INPUT(q);                 // [sum(extend_len), num_qo_heads, head_dim]
  CHECK_INPUT(q_indptr);          // [batch_size + 1]
  CHECK_INPUT(kv_data);           // [max_num_pages, 2, num_kv_heads, page_size, head_dim]
  CHECK_INPUT(kv_indptr);         // [batch_size + 1]
  CHECK_INPUT(kv_indices);        // [sum(seq_len)]
  CHECK_INPUT(kv_last_page_len);  // [batch_size]
  CHECK_DIM(3, q);
  CHECK_DIM(1, q_indptr);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, kv_last_page_len);
  CHECK_EQ(q_indptr.size(0), kv_indptr.size(0));
  CHECK_EQ(kv_indptr.size(0), kv_last_page_len.size(0) + 1);
  if (kv_data.dim() == 5) {
    CHECK_EQ(q.size(2), kv_data.size(4));
  } else {
    CHECK_DIM(4, kv_data);
    CHECK_EQ(1, page_size);
    CHECK_EQ(q.size(2), kv_data.size(3));
  }

  unsigned int batch_size = q_indptr.size(0) - 1;
  unsigned int head_dim = q.size(2);
  unsigned int num_kv_heads, num_qo_heads;
  QKVLayout qkv_layout = static_cast<QKVLayout>(layout);
  if (qkv_layout == QKVLayout::kNHD) {
    num_kv_heads = kv_data.size(kv_data.dim() - 2);
    num_qo_heads = q.size(1);
  } else {
    num_kv_heads = kv_data.size(2);
    num_qo_heads = q.size(0);
  }

  auto o = torch::empty_like(q, q.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    paged_kv_t<PageStorage::kIndices, c_type, int32_t> paged_kv(
        num_kv_heads, page_size, head_dim, batch_size, static_cast<c_type*>(kv_data.data_ptr()),
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        static_cast<int32_t*>(kv_last_page_len.data_ptr()));
    cudaError_t status =
        BatchPrefillWithPagedKVCache<PageStorage::kIndices, c_type, c_type, int32_t>(
            static_cast<c_type*>(q.data_ptr()), paged_kv,
            static_cast<int32_t*>(q_indptr.data_ptr()), static_cast<c_type*>(o.data_ptr()),
            /*tmp=*/nullptr,
            /*lse=*/nullptr, num_qo_heads, causal, RotaryMode(rotary_mode),
            allow_fp16_qk_reduction);
    TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error code ",
                status);
    return true;
  });

  TORCH_CHECK(success, "BatchPrefillWithPagedKVCache failed to dispatch with dtype ",
              q.scalar_type());

  return o;
}
