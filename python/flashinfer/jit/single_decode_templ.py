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

single_decode_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode_params.cuh>
#include "pytorch_extension_utils.h"

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, std::optional<torch::Tensor> alibi_slopes,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(tmp);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(v.device(), device);
  CHECK_EQ(tmp.device(), device);
  CHECK_DIM(2, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_SHAPE(k, v);
  CHECK_EQ(q.size(1), k.size(2));
  CHECK_EQ(v.scalar_type(), k.scalar_type());
  unsigned int num_qo_heads = q.size(0);
  unsigned int head_dim = q.size(1);
  unsigned int kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
  } else {
    num_kv_heads = k.size(0);
    kv_len = k.size(1);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q);
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  using ParamsT = SingleDecodeParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}>;
  using AttentionVariant = ComposedAttention<ParamsT, get_variant_code(/*use_custom_mask=*/false, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;
  ParamsT params(
      static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
      static_cast<{{ dtype_kv }}*>(v.data_ptr()), static_cast<{{ dtype_o }}*>(o.data_ptr()),
      {% if use_alibi == "true" %}static_cast<float*>(alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
      kv_len, num_qo_heads, num_kv_heads, kv_layout, head_dim, window_left,
      logits_soft_cap, sm_scale, rope_scale, rope_theta);
  
  cudaError_t status = SingleDecodeWithKVCacheDispatched<{{ head_dim }}, {{ pos_encoding_mode }}, AttentionVariant>(
      params, static_cast<{{ dtype_o }}*>(tmp.data_ptr()), torch_current_stream);
  TORCH_CHECK(status == cudaSuccess,
              "SingleDecodeWithKVCache kernel launch failed, error: " +
                  std::string(cudaGetErrorString(status)));

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
}
"""
