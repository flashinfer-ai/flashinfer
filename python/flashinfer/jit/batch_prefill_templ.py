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

batch_prefill_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/handler.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

{% set use_custom_mask = "true" if mask_mode == 0 else "false" %}
using RaggedParamsT = BatchPrefillRaggedParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;
using RaggedAttentionVariant = ComposedAttention<RaggedParamsT, get_variant_code({{ use_custom_mask }}, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;
using PagedParamsT = BatchPrefillPagedParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;
using PagedAttentionVariant = ComposedAttention<PagedParamsT, get_variant_code({{ use_custom_mask }}, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    unsigned int batch_size,
    unsigned int num_qo_heads,
    unsigned int num_kv_heads,
    bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  
  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  qo_indptr = qo_indptr.to(torch::kCPU);
  kv_indptr = kv_indptr.to(torch::kCPU);

  PrefillPlanInfo plan_info;

  cudaError_t status = PrefillPlan<{{ dtype_o }}, {{ dtype_idx }}>(
    float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
    int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
    int_workspace_size_in_bytes,
    plan_info, qo_indptr.data_ptr<{{ dtype_idx }}>(), kv_indptr.data_ptr<{{ dtype_idx }}>(),
    batch_size, num_qo_heads, num_kv_heads, {{ head_dim }}, /*page_size=*/1, enable_cuda_graph,
    torch_current_stream);

  TORCH_CHECK(status == cudaSuccess, "Failed to plan prefill with error: ", cudaGetErrorString(status));

  return plan_info.ToVector();
}

std::vector<torch::Tensor> BatchPrefillWithRaggedKVCacheRun(
  torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
  std::vector<int64_t> plan_info_vec,
  torch::Tensor q, torch::Tensor k, torch::Tensor v,
  std::optional<torch::Tensor> maybe_custom_mask,
  torch::Tensor qo_indptr, torch::Tensor kv_indptr,
  std::optional<torch::Tensor> maybe_qk_indptr,
  unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
  float rope_scale, float rope_theta, bool return_lse) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k.size(1) : k.size(0);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }

  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  int64_t nnz_qo = q.size(0);
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  RaggedParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
    static_cast<{{ dtype_kv }}*>(v.data_ptr()),
    {% if mask_mode == "MaskMode::kCustom" %}static_cast<uint8_t*>(maybe_packed_custom_mask->data_ptr()){% else %}nullptr{% endif %},
    static_cast<{{ dtype_idx }}*>(qo_indptr.data_ptr()),
    static_cast<{{ dtype_idx }}*>(kv_indptr.data_ptr()),
    {% if mask_mode == "MaskMode::kCustom" %}static_cast<{{ dtype_idx }}*>(maybe_qk_indptr->data_ptr()){% else %}nullptr{% endif %},
    static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
    {% if use_alibi == "true" %}static_cast<float*>(maybe_alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
    num_qo_heads, num_kv_heads, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h,
    window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);

  {{ dtype_o }}* tmp_v = nullptr;
  float* tmp_s = nullptr;

  params.request_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.merge_indptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.merge_indptr_offset);
  params.o_indptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<{{ dtype_o }}>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask = GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.total_num_rows = plan_info.total_num_rows;
  params.padded_batch_size = plan_info.padded_batch_size;

  WarpLayout warp_layout = WarpLayout(plan_info.warp_layout_code);
  cudaError_t status = cudaSuccess;

  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    status = BatchPrefillWithRaggedKVCacheDispatched<
      WARP_LAYOUT, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, {{ mask_mode }}, RaggedAttentionVariant>(
        params, tmp_v, tmp_s, torch_current_stream);
  });

  TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithRaggedKVCache failed with error ", cudaGetErrorString(status));

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

std::vector<torch::Tensor> BatchPrefillWithPagedKVCacheRun(
  torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
  std::vector<int64_t> plan_info_vec,
  torch::Tensor q,
  std::optional<torch::Tensor> paged_kv_cache,
  std::optional<torch::Tensor> paged_k_cache,
  std::optional<torch::Tensor> paged_v_cache,
  std::optional<torch::Tensor> maybe_custom_mask,
  torch::Tensor qo_indptr,
  torch::Tensor paged_kv_indptr,
  torch::Tensor paged_kv_indices,
  torch::Tensor paged_kv_last_page_len,
  std::optional<torch::Tensor> maybe_qk_indptr,
  unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
  float rope_scale, float rope_theta, bool return_lse) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  bool paged_kv_defined = paged_kv_cache.has_value();
  auto device = q.device();
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  if (paged_kv_defined) {
    if (kv_layout == QKVLayout::kHND) {
      num_kv_heads = paged_kv_cache->size(2);
      page_size = paged_kv_cache->size(3);
    } else {
      page_size = paged_kv_cache->size(2);
      num_kv_heads = paged_kv_cache->size(3);
    }
  } else {
    if (kv_layout == QKVLayout::kHND) {
      num_kv_heads = paged_k_cache->size(1);
      page_size = paged_k_cache->size(2);
    } else {
      page_size = paged_k_cache->size(1);
      num_kv_heads = paged_k_cache->size(2);
    }
  }

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  int64_t nnz_qo = q.size(0);
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  paged_kv_t<{{ dtype_kv }}, {{ dtype_idx }}> paged_kv(
    num_kv_heads, page_size, {{ head_dim }},
    batch_size, kv_layout,
    static_cast<{{ dtype_kv }}*>(paged_kv_cache.has_value() ? paged_kv_cache->data_ptr()
                                                        : nullptr),
    static_cast<{{ dtype_kv }} *>(paged_k_cache.has_value() ? paged_k_cache->data_ptr()
                                                        : nullptr),
    static_cast<{{ dtype_kv }}*>(paged_v_cache.has_value() ? paged_v_cache->data_ptr()
                                                        : nullptr),
    static_cast<{{ dtype_idx }}*>(paged_kv_indices.data_ptr()),
    static_cast<{{ dtype_idx }}*>(paged_kv_indptr.data_ptr()),
    static_cast<{{ dtype_idx }}*>(paged_kv_last_page_len.data_ptr()));

  PagedParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()), paged_kv,
    {% if mask_mode == "MaskMode::kCustom" %}static_cast<uint8_t*>(maybe_packed_custom_mask->data_ptr()){% else %}nullptr{% endif %},
    static_cast<{{ dtype_idx }}*>(qo_indptr.data_ptr()),
    {% if mask_mode == "MaskMode::kCustom" %}static_cast<{{ dtype_idx }}*>(maybe_qk_indptr->data_ptr()){% else %}nullptr{% endif %},
    static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
    {% if use_alibi == "true" %}static_cast<float*>(maybe_alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
    num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);

  {{ dtype_o }}* tmp_v = nullptr;
  float* tmp_s = nullptr;

  params.request_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.merge_indptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.merge_indptr_offset);
  params.o_indptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<{{ dtype_o }}>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask = GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.total_num_rows = plan_info.total_num_rows;
  params.padded_batch_size = plan_info.padded_batch_size;

  WarpLayout warp_layout = WarpLayout(plan_info.warp_layout_code);
  cudaError_t status = cudaSuccess;

  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    status = BatchPrefillWithPagedKVCacheDispatched<
      WARP_LAYOUT, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, {{ mask_mode }}, PagedAttentionVariant>(
        params, tmp_v, tmp_s, torch_current_stream);
  });

  TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ", cudaGetErrorString(status));

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchPrefillWithKVCachePlan);
  m.def("ragged_run", &BatchPrefillWithRaggedKVCacheRun);
  m.def("paged_run", &BatchPrefillWithPagedKVCacheRun);
}
"""
