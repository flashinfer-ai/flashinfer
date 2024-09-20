"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

batch_decode_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/handler.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

using ParamsT = BatchDecodeParams<PageStorage::kIndices, {{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;
using AttentionVariant = ComposedAttention<ParamsT, get_variant_code(/*use_custom_mask=*/false, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor indptr,
    torch::Tensor last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
    unsigned int pos_encoding_mode, float logits_soft_cap, bool enable_cuda_graph, torch::Tensor empty_q_data,
    torch::Tensor empty_kv_data) {
  CHECK_INPUT(float_workspace_buffer);
  CHECK_INPUT(int_workspace_buffer);
  // NOTE(zihao): not necessary to be CUDA tensor
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(last_page_len);
  CHECK_DIM(1, indptr);
  CHECK_DIM(1, last_page_len);
  CHECK_DIM(1, float_workspace_buffer);
  CHECK_DIM(1, int_workspace_buffer);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  indptr = indptr.to(torch::kCPU);
  last_page_len = last_page_len.to(torch::kCPU);

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  DecodePlanInfo plan_info;

  cudaError_t status = DecodePlan<{{ head_dim }}, {{ pos_encoding_mode }}, AttentionVariant>(
      static_cast<void*>(float_workspace_buffer.data_ptr()),
      float_workspace_size_in_bytes,
      static_cast<void*>(int_workspace_buffer.data_ptr()),
      static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
      int_workspace_size_in_bytes,
      plan_info,
      static_cast<{{ dtype_idx }}*>(indptr.data_ptr()),
      static_cast<{{ dtype_idx }}*>(last_page_len.data_ptr()),
      batch_size, num_qo_heads, num_kv_heads, page_size, enable_cuda_graph, /*stream=*/torch_current_stream);

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
  
  return plan_info.ToVector();
}

std::vector<torch::Tensor> BatchDecodeWithPagedKVCacheRun(
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec,
    torch::Tensor q, std::optional<torch::Tensor> paged_kv_cache,
    std::optional<torch::Tensor> paged_k_cache, std::optional<torch::Tensor> paged_v_cache,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    std::optional<torch::Tensor> alibi_slopes,
    unsigned int kv_layout_code, int window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  CHECK_INPUT(q);
  bool paged_kv_defined = paged_kv_cache.has_value();
  if (paged_kv_defined) {
    CHECK_INPUT(paged_kv_cache.value());
  } else {
    CHECK_INPUT(paged_k_cache.value());
    CHECK_INPUT(paged_v_cache.value());
  }
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  auto device = q.device();
  if (paged_kv_defined) {
    CHECK_EQ(paged_kv_cache->device(), device);
  } else {
    CHECK_EQ(paged_k_cache->device(), device);
    CHECK_EQ(paged_v_cache->device(), device);
  }
  CHECK_EQ(paged_kv_indices.device(), device);
  CHECK_EQ(paged_kv_indptr.device(), device);
  CHECK_EQ(paged_kv_last_page_len.device(), device);
  CHECK_DIM(3, q);                       // (B, H_qo, D)
  CHECK_DIM(1, paged_kv_last_page_len);  // (B,)
  CHECK_DIM(1, paged_kv_indptr);         // (B+1,)
  CHECK_DIM(1, paged_kv_indices);        // (nnz,)
  if (paged_kv_defined) {
    // (num_max_pages, 2, H_kv, page_size, head_dim) for HND
    // (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
    CHECK_DIM(5, paged_kv_cache.value());
  } else {
    // (num_max_pages, H_kv, page_size, head_dim) for HND
    // (num_max_pages, page_size, H_kv, head_dim) for NHD
    CHECK_DIM(4, paged_k_cache.value());
    CHECK_DIM(4, paged_v_cache.value());
  }
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (paged_kv_defined) {
    CHECK_EQ(paged_kv_cache->size(1), 2);
    CHECK_EQ(paged_kv_cache->size(4), head_dim);
    if (kv_layout == QKVLayout::kHND) {
      num_kv_heads = paged_kv_cache->size(2);
      page_size = paged_kv_cache->size(3);
    } else {
      page_size = paged_kv_cache->size(2);
      num_kv_heads = paged_kv_cache->size(3);
    }
  } else {
    CHECK_EQ(paged_k_cache->size(3), head_dim);
    CHECK_EQ(paged_v_cache->size(3), head_dim);
    if (kv_layout == QKVLayout::kHND) {
      num_kv_heads = paged_k_cache->size(1);
      page_size = paged_k_cache->size(2);
    } else {
      page_size = paged_k_cache->size(1);
      num_kv_heads = paged_k_cache->size(2);
    }
  }
  CHECK_GE(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_last_page_len.size(0), batch_size);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q);
  torch::Tensor lse;
  if (return_lse) {
    lse = torch::empty({batch_size, num_qo_heads}, q.options().dtype((torch::kFloat32)));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  bool alibi_slopes_defined = alibi_slopes.has_value();

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  paged_kv_t<PageStorage::kIndices, {{ dtype_kv }}, {{ dtype_idx }}> paged_kv(
      num_kv_heads, page_size, head_dim,
      plan_info.split_kv ? batch_size: plan_info.batch_size_after_partition,
      kv_layout,
      static_cast<{{ dtype_kv }}*>(paged_kv_cache.has_value() ? paged_kv_cache->data_ptr()
                                                          : nullptr),
      static_cast<{{ dtype_kv }} *>(paged_k_cache.has_value() ? paged_k_cache->data_ptr()
                                                          : nullptr),
      static_cast<{{ dtype_kv }}*>(paged_v_cache.has_value() ? paged_v_cache->data_ptr()
                                                          : nullptr),
      static_cast<{{ dtype_idx }}*>(paged_kv_indices.data_ptr()),
      plan_info.split_kv ?
      GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.new_indptr_offset):
      static_cast<{{ dtype_idx }}*>(paged_kv_indptr.data_ptr()),
      plan_info.split_kv ?
      GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.new_last_page_len_offset):
      static_cast<{{ dtype_idx }}*>(paged_kv_last_page_len.data_ptr()));
  ParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()),
    /*q_offset=*/nullptr, paged_kv, static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/(return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr),
    alibi_slopes_defined ? static_cast<float*>(alibi_slopes->data_ptr()): nullptr,
    num_qo_heads, window_left, logits_soft_cap, sm_scale, 1.f / rope_scale, 1.f / rope_theta);
  
  {{ dtype_o }}* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<{{ dtype_o }}>(float_buffer, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
    params.kv_partition_info.batch_size_before_partition = plan_info.batch_size_before_partition;
    params.kv_partition_info.chunk_indptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.chunk_indptr_offset);
    params.kv_partition_info.batch_idx_map = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.batch_idx_map_offset);
    params.kv_partition_info.chunk_start_pos = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.chunk_start_pos_offset);
    params.kv_partition_info.seq_lens_before_partition = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.seq_lengths_before_partition_offset);
  }
  params.block_valid_mask = GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
  params.padded_batch_size = plan_info.padded_batch_size;
  
  cudaError_t status = BatchDecodeWithPagedKVCacheDispatched<
      {{ head_dim }}, {{ pos_encoding_mode }}, AttentionVariant>(
      params, tmp_v, tmp_s, /*stream=*/torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchDecodeWithPagedKVCachePlan);
  m.def("run", &BatchDecodeWithPagedKVCacheRun);
}
"""