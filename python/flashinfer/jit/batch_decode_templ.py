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
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode_params.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

{% set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %}
using ParamsT = BatchDecodeParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;
using AttentionVariant = ComposedAttention<ParamsT, get_variant_code(/*use_custom_mask=*/false, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor indptr,
    unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  TORCH_CHECK(indptr.device() == torch::kCPU, "indptr must be on CPU");

  DecodePlanInfo plan_info;

  cudaError_t status = DecodePlan<{{ head_dim }}, {{ pos_encoding_mode }}, AttentionVariant>(
      static_cast<void*>(float_workspace_buffer.data_ptr()),
      float_workspace_size_in_bytes,
      static_cast<void*>(int_workspace_buffer.data_ptr()),
      static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
      int_workspace_size_in_bytes,
      plan_info,
      static_cast<{{ dtype_idx }}*>(indptr.data_ptr()),
      batch_size, num_qo_heads, num_kv_heads, page_size, enable_cuda_graph, /*stream=*/torch_current_stream);

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
  
  return plan_info.ToVector();
}

std::vector<torch::Tensor> BatchDecodeWithPagedKVCacheRun(
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec,
    torch::Tensor q,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    std::optional<torch::Tensor> alibi_slopes,
    unsigned int kv_layout_code, int window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  auto device = q.device();
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q);
  torch::Tensor lse;
  if (return_lse) {
    lse = torch::empty({batch_size, num_qo_heads}, q.options().dtype((torch::kFloat32)));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());
  
  const auto q_stride_n = q.stride(0), q_stride_h = q.stride(1);

  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  paged_kv_t<{{ dtype_kv }}, {{ dtype_idx }}> paged_kv(
      num_kv_heads, page_size, {{ head_dim }},
      batch_size, kv_layout,
      static_cast<{{ dtype_kv }}*>(paged_k_cache.data_ptr()),
      static_cast<{{ dtype_kv }}*>(paged_v_cache.data_ptr()),
      kv_cache_strides,
      static_cast<{{ dtype_idx }}*>(paged_kv_indices.data_ptr()),
      static_cast<{{ dtype_idx }}*>(paged_kv_indptr.data_ptr()),
      static_cast<{{ dtype_idx }}*>(paged_kv_last_page_len.data_ptr()));
  ParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()),
    /*q_offset=*/nullptr, paged_kv, static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/(return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr),
    {% if use_alibi == "true" %}static_cast<float*>(alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
    num_qo_heads, q_stride_n, q_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);
  
  {{ dtype_o }}* tmp_v = nullptr;
  float* tmp_s = nullptr;
  params.request_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.request_indices_offset);
  params.kv_tile_indices = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr = GetPtrFromBaseOffset<{{ dtype_idx }}>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<{{ dtype_o }}>(float_buffer, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask = GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
    }
  }
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
