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
    torch::Tensor kv_indices, torch::Tensor kv_last_page_len, bool causal, unsigned int layout,
    unsigned int rotary_mode, bool allow_fp16_qk_reduction, float rope_scale, float rope_theta) {
  CHECK_INPUT(q);         // [sum(extend_len), num_qo_heads, head_dim]
  CHECK_INPUT(q_indptr);  // [batch_size + 1]
  // [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
  // [max_num_pages, 2, page_size, num_kv_heads, head_dim] for HND
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);         // [batch_size + 1]
  CHECK_INPUT(kv_indices);        // [sum(seq_len)]
  CHECK_INPUT(kv_last_page_len);  // [batch_size]
  CHECK_DIM(3, q);
  CHECK_DIM(5, kv_data)
  CHECK_DIM(1, q_indptr);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, kv_last_page_len);
  CHECK_EQ(q_indptr.size(0), kv_indptr.size(0));
  CHECK_EQ(kv_indptr.size(0), kv_last_page_len.size(0) + 1);
  // TODO(Zihao): more index data types.
  CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_last_page_len.scalar_type(), torch::kInt32);
  CHECK_EQ(q.size(2), kv_data.size(4));
  QKVLayout kv_layout = QKVLayout(layout);

  unsigned int page_size, num_kv_heads;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = kv_data.size(2);
    page_size = kv_data.size(3);
  } else {
    page_size = kv_data.size(2);
    num_kv_heads = kv_data.size(3);
  }
  unsigned int batch_size = q_indptr.size(0) - 1;
  unsigned int head_dim = q.size(2);
  unsigned int num_qo_heads = q.size(1);

  auto o = torch::empty_like(q, q.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size, static_cast<c_type*>(kv_data.data_ptr()),
          static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
          static_cast<int32_t*>(kv_last_page_len.data_ptr()));

      cudaError_t status =
          BatchPrefillWithPagedKVCache<PageStorage::kIndices, KV_LAYOUT, c_type, c_type, int32_t>(
              static_cast<c_type*>(q.data_ptr()), static_cast<int32_t*>(q_indptr.data_ptr()),
              paged_kv, static_cast<c_type*>(o.data_ptr()),
              /*tmp=*/nullptr,
              /*lse=*/nullptr, num_qo_heads, causal, RotaryMode(rotary_mode),
              allow_fp16_qk_reduction, rope_scale, rope_theta);
      TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error code ",
                  cudaGetErrorString(status));
    });
    return true;
  });

  TORCH_CHECK(success, "BatchPrefillWithPagedKVCache failed to dispatch with dtype ",
              q.scalar_type());

  return o;
}

void BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward(torch::Tensor workspace_buffer,
                                                              torch::Tensor qo_indptr,
                                                              unsigned int batch_size,
                                                              unsigned int num_qo_heads,
                                                              unsigned int num_kv_heads) {
  // NOTE(Zihao): not necessary to be a CUDA tensor
  CHECK_CONTIGUOUS(qo_indptr);
  CHECK_CONTIGUOUS(workspace_buffer);
  CHECK_EQ(num_qo_heads % num_kv_heads, 0);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, workspace_buffer);

  // TODO(Zihao): support dispatching to different index data types.
  CHECK_EQ(qo_indptr.scalar_type(), torch::kInt32);
  size_t workspace_size_in_bytes = workspace_buffer.size(0) * workspace_buffer.element_size();

  cudaError_t status = handler_.BeginForward(
      static_cast<void*>(workspace_size_in_bytes.data_ptr()), workspace_size_in_bytes,
      static_cast<int32_t*>(qo_indptr.data_ptr()), batch_size, num_qo_heads, num_kv_heads);
  TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}

void BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward() { handler_.EndForward(); }

std::vector<torch::Tensor> BatchPrefillWithPagedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor paged_kv_data,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, bool causal, unsigned int rotary_mode,
    bool allow_fp16_qk_reduction, float rope_scale, float rope_theta, bool return_lse) {
  CHECK_INPUT(q);
  CHECK_INPUT(qo_indptr);
  CHECK_INPUT(paged_kv_data);
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  CHECK_DIM(3, q);          // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr);  // (B + 1,)
  // [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
  // [max_num_pages, 2, page_size, num_kv_heads, head_dim] for HND
  CHECK_DIM(5, paged_kv_data);
  CHECK_DIM(1, paged_kv_indptr);         // (B + 1,)
  CHECK_DIM(1, paged_kv_indices);        // (nnz_kv,)
  CHECK_DIM(1, paged_kv_last_page_len);  // (B,)
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (kv_layout_ == QKVLayout::kHND) {
    num_kv_heads = paged_kv_data.size(2);
    page_size = paged_kv_data.size(3);
  } else {
    page_size = paged_kv_data.size(2);
    num_kv_heads = paged_kv_data.size(3);
  }
  CHECK_EQ(num_qo_heads % num_kv_heads, 0);
  CHECK_EQ(qo_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_last_page_len.size(0), batch_size);
  CHECK_EQ(paged_kv_data.size(1), 2);
  CHECK_EQ(paged_kv_data.size(4), head_dim);
  // TODO(Zihao): support dispatching to different index data types.
  CHECK_EQ(qo_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_last_page_len.scalar_type(), torch::kInt32);

  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty_like(q, q.options());
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options()).to(torch::kFloat32);
  }

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    SWITCH_LAYOUT(kv_layout_, KV_LAYOUT, {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type*>(paged_kv_data.data_ptr()),
          static_cast<int32_t*>(paged_kv_indices.data_ptr()),
          static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
          static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
      cudaError_t status = BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, KV_LAYOUT,
                                                               c_type, c_type, int32_t>(
          &handler_, static_cast<c_type*>(q.data_ptr()),
          static_cast<int32_t*>(qo_indptr.data_ptr()), paged_kv, static_cast<c_type*>(o.data_ptr()),
          /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr, num_qo_heads, causal,
          RotaryMode(rotary_mode), allow_fp16_qk_reduction, rope_scale, rope_theta,
          /*stream=*/nullptr);
      TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error code ",
                  cudaGetErrorString(status));
    });
    return true;
  });
  TORCH_CHECK(success, "BatchPrefillWithPagedKVCache failed to dispatch with dtype ",
              q.scalar_type());

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
