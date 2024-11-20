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

batch_decode_mla_templ = [
    r"""
#include <optional>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode_params.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

using ParamsT = BatchDecodeParamsMLA<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;
using AttentionVariant = ComposedAttention<ParamsT, get_variant_code(/*use_custom_mask=*/false, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, /*use_alibi*/false)>;

std::vector<int64_t> BatchDecodeWithPagedKVCachePlanMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer,
    at::Tensor indptr,
    unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int page_size,
    bool enable_cuda_graph,
    int64_t cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  DecodePlanInfo plan_info;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  auto work_estimation_func =
      BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA<{{ head_dim_ckv }}, {{ head_dim_kpe }}, AttentionVariant>;
  cudaError_t status = DecodePlan<{{ head_dim_ckv }}, flashinfer::PosEncodingMode::kRoPELlama, AttentionVariant>(
      static_cast<void*>(float_workspace_buffer.data_ptr()),
      float_workspace_size_in_bytes,
      static_cast<void*>(int_workspace_buffer.data_ptr()),
      static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
      int_workspace_size_in_bytes,
      plan_info,
      static_cast<{{ dtype_idx }}*>(indptr.data_ptr()),
      batch_size, num_qo_heads, page_size, enable_cuda_graph, /*stream=*/stream,
      work_estimation_func);

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCachePlanMLA failed with error ",
              cudaGetErrorString(status));

  return plan_info.ToVector();
}

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec,
    at::Tensor q_nope,
    at::Tensor q_pe,
    at::Tensor paged_ckv_cache,
    at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len,
    at::Tensor o,
    float sm_scale,
    int window_left,
    float logits_soft_cap, float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
    int64_t cuda_stream) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);

  auto device = q_nope.device();
  int64_t batch_size = q_nope.size(0);
  int64_t num_qo_heads = q_nope.size(1);
  int64_t page_size = paged_ckv_cache.size(1);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size, lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q.size(1));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  paged_kv_mla_t<{{ dtype_kv }}, {{ dtype_idx }}> paged_kv(
      page_size, {{ head_dim_ckv }}, {{ head_dim_kpe }},
      batch_size,
      static_cast<{{ dtype_kv }}*>(paged_ckv_cache.data_ptr()),
      paged_ckv_cache.strides().data(),
      static_cast<{{ dtype_kv }}*>(paged_kpe_cache.data_ptr()),
      paged_kpe_cache.strides().data(),
      static_cast<{{ dtype_idx }}*>(paged_kv_indices.data_ptr()),
      static_cast<{{ dtype_idx }}*>(paged_kv_indptr.data_ptr()),
      static_cast<{{ dtype_idx }}*>(paged_kv_last_page_len.data_ptr()));
  ParamsT params(
    static_cast<{{ dtype_q }}*>(q_nope.data_ptr()), static_cast<{{ dtype_q }}*>(q_pe.data_ptr()),
    /*q_offset=*/nullptr, paged_kv, static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
    num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);

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

  cudaError_t status = BatchDecodeWithPagedKVCacheDispatchedMLA<
      {{ head_dim_ckv }}, {{ head_dim_kpe }}, AttentionVariant>(
      params, tmp_v, tmp_s, /*stream=*/torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}
""",
    r"""#include "pytorch_extension_utils.h"

std::vector<int64_t> BatchDecodeWithPagedKVCachePlanMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer,
    at::Tensor indptr,
    unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int page_size,
    bool enable_cuda_graph,
    int64_t cuda_stream);

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec,
    at::Tensor q_nope,
    at::Tensor q_pe,
    at::Tensor paged_ckv_cache,
    at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len,
    at::Tensor o,
    float sm_scale,
    int window_left,
    float logits_soft_cap, float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
    int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchDecodeWithPagedKVCachePlanMLA);
  m.def("run", &BatchDecodeWithPagedKVCacheRunMLA);
}
""",
]
