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

customizable_single_decode_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/decode.cuh>
#include "pytorch_extension_utils.h"

struct SingleDecodeParams {
  using DTypeQ = {{ dtype_q }};
  using DTypeKV = {{ dtype_kv }};
  using DTypeO = {{ dtype_o }};
  using IdType = int32_t;
  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  float* lse;
  {{ additional_params_decl }}
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t head_dim;
  int32_t window_left;
  uint32_t kv_chunk_size;

  __device__ __host__ SingleDecodeParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o,
                                         uint32_t seq_len,
                                         uint32_t num_qo_heads, uint32_t num_kv_heads,
                                         QKVLayout kv_layout, uint32_t head_dim,
                                         int32_t window_left{{ additional_params }})
      : q(q),
        k(k),
        v(v),
        o(o),
        lse(nullptr),
        kv_len(seq_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(num_qo_heads * head_dim),
        q_stride_h(head_dim),
        kv_stride_n((kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim),
        kv_stride_h((kv_layout == QKVLayout::kNHD) ? head_dim : seq_len * head_dim),
        head_dim(head_dim),
        window_left(window_left),
        kv_chunk_size(0){{ additional_params_init }} {}

  __host__ __device__ __forceinline__ size_t get_q_elem_offset(uint32_t qo_idx,
                                                               uint32_t qo_head_idx,
                                                               uint32_t feat_idx) const {
    return get_elem_offset_impl(qo_idx, qo_head_idx, feat_idx, q_stride_n, q_stride_h);
  }

  __host__ __device__ __forceinline__ size_t get_o_elem_offset(uint32_t qo_idx,
                                                               uint32_t qo_head_idx,
                                                               uint32_t feat_idx) const {
    return get_elem_offset_impl(qo_idx, qo_head_idx, feat_idx, num_qo_heads * head_dim, head_dim);
  }

  __host__ __device__ __forceinline__ size_t get_kv_elem_offset(uint32_t kv_idx,
                                                                uint32_t kv_head_idx,
                                                                uint32_t feat_idx) const {
    return get_elem_offset_impl(kv_idx, kv_head_idx, feat_idx, kv_stride_n, kv_stride_h);
  }

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

{{ variant_decl }}


torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp,
                                          unsigned int layout, int window_left{{ additional_func_params }}) {
  auto device = q.device();
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
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q);

  using ParamsT = SingleDecodeParams;
  using AttentionVariant = {{ variant_name }}<ParamsT>;
  ParamsT params(
      static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
      static_cast<{{ dtype_kv }}*>(v.data_ptr()), static_cast<{{ dtype_o }}*>(o.data_ptr()),
      kv_len, num_qo_heads, num_kv_heads, kv_layout, head_dim, window_left{{ additional_params_data }});
  
  cudaError_t status = SingleDecodeWithKVCacheDispatched<{{ head_dim }}, PosEncodingMode::kNone, AttentionVariant>(
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

single_decode_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode_params.cuh>
#include "pytorch_extension_utils.h"

{% set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %}
torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, std::optional<torch::Tensor> alibi_slopes,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta) {
  auto device = q.device();
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
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q);

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
