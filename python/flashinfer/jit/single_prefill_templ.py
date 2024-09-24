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

single_prefill_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

{% set use_custom_mask = "true" if mask_mode == 0 else "false" %}
using ParamsT = SinglePrefillParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}>;
using AttentionVariant = ComposedAttention<ParamsT, get_variant_code({{ use_custom_mask }}, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;

std::vector<torch::Tensor> single_prefill_with_kv_cache(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, std::optional<torch::Tensor> maybe_packed_custom_mask,
    torch::Tensor tmp, std::optional<torch::Tensor> maybe_alibi_slopes, unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, bool return_lse) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(v.device(), device);
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_SHAPE(k, v);
  CHECK_EQ(q.stride(2), 1);
  CHECK_EQ(k.stride(2), 1);
  CHECK_EQ(v.stride(2), 1);
  CHECK_EQ(q.size(2), k.size(2));
  {% if mask_mode == "MaskMode::kCustom" %}
    CHECK_INPUT(maybe_packed_custom_mask.value());
    CHECK_EQ(maybe_packed_custom_mask->device(), device);
    CHECK_DIM(1, maybe_packed_custom_mask.value());
    // packed_custom_mask must be uint8
    TORCH_CHECK(maybe_packed_custom_mask->scalar_type() == torch::kUInt8,
                "packed_custom_mask must be uint8");
  {% endif %}
  unsigned int head_dim = q.size(2);
  unsigned int kv_len, qo_len, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q.size(0);
  num_qo_heads = q.size(1);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {
    kv_len = k.size(1);
    num_kv_heads = k.size(0);
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({qo_len, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  ParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
    static_cast<{{ dtype_kv }}*>(v.data_ptr()),
    {% if mask_mode == "MaskMode::kCustom" %}static_cast<uint8_t*>(maybe_packed_custom_mask->data_ptr()){% else %}nullptr{% endif %},
    static_cast<{{ dtype_kv }}*>(o.data_ptr()),
    /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
    {% if use_alibi == "true" %}static_cast<float*>(maybe_alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
    num_qo_heads, num_kv_heads, qo_len, kv_len, q_stride_n, q_stride_h,
    kv_stride_n, kv_stride_h, head_dim, window_left, logits_soft_cap, sm_scale,
    rope_scale, rope_theta);
  
  cudaError_t status =
      SinglePrefillWithKVCacheDispatched<{{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, {{ mask_mode }}, AttentionVariant>(
            params, static_cast<{{ dtype_o }}*>(tmp.data_ptr()), torch_current_stream);
  TORCH_CHECK(status == cudaSuccess,
             "SinglePrefillWithKVCache kernel launch failed, error: " +
              std::string(cudaGetErrorString(status)));

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
}
"""