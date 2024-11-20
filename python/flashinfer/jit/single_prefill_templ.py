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

customizable_single_prefill_templ = [
    r"""
#include <optional>
#include <flashinfer/attention/prefill.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;


struct SinglePrefillParams {
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
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t head_dim;
  int32_t window_left;

  bool partition_kv;

  __host__ SinglePrefillParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o,
                               float* lse, uint32_t num_qo_heads,
                               uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
                               uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
                               uint32_t kv_stride_h, uint32_t head_dim, int32_t window_left{{ additional_params }})
      : q(q),
        k(k),
        v(v),
        o(o),
        lse(lse),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        qo_len(qo_len),
        kv_len(kv_len),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        kv_stride_n(kv_stride_n),
        kv_stride_h(kv_stride_h),
        head_dim(head_dim),
        window_left(window_left),
        partition_kv(false){{ additional_params_init }} {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return qo_len;
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};


{{ variant_decl }}

at::Tensor single_prefill_with_kv_cache(
    unsigned int mask_mode_code, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor tmp, at::Tensor o, unsigned int layout, int32_t window_left,
    std::optional<at::Tensor> maybe_lse{{ additional_func_params }},
    int64_t cuda_stream) {
  auto device = q.device();
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

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  using ParamsT = SinglePrefillParams;
  using AttentionVariant = {{ variant_name }}<ParamsT>;
  ParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
    static_cast<{{ dtype_kv }}*>(v.data_ptr()),
    static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
    num_qo_heads, num_kv_heads, qo_len, kv_len, q_stride_n, q_stride_h,
    kv_stride_n, kv_stride_h, head_dim, window_left{{ additional_params_data }});

  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    cudaError_t status =
        SinglePrefillWithKVCacheDispatched<{{ head_dim }}, PosEncodingMode::kNone, false, MASK_MODE, AttentionVariant>(
              params, static_cast<{{ dtype_o }}*>(tmp.data_ptr()), stream);
    TORCH_CHECK(status == cudaSuccess,
              "SinglePrefillWithKVCache kernel launch failed, error: " +
                std::string(cudaGetErrorString(status)));
  });

  return o;
}
""",
    r"""#include "pytorch_extension_utils.h"

at::Tensor single_prefill_with_kv_cache(
    unsigned int mask_mode_code, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor tmp, at::Tensor o, unsigned int layout, int32_t window_left,
    std::optional<at::Tensor> maybe_lse{{ additional_func_params }},
    int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
}
""",
]

single_prefill_templ = [
    r"""
#include <optional>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

{% set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %}
using ParamsT = SinglePrefillParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}>;

void single_prefill_with_kv_cache(
    unsigned int mask_mode_code,
    at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> maybe_packed_custom_mask,
    at::Tensor tmp, std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor o,
    unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
    int64_t cuda_stream) {
  auto device = q.device();
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
  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  ParamsT params(
    static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
    static_cast<{{ dtype_kv }}*>(v.data_ptr()),
    /*custom_mask=*/(maybe_packed_custom_mask ? static_cast<uint8_t*>(maybe_packed_custom_mask->data_ptr()) : nullptr),
    static_cast<{{ dtype_o }}*>(o.data_ptr()),
    /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
    {% if use_alibi == "true" %}static_cast<float*>(maybe_alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
    num_qo_heads, num_kv_heads, qo_len, kv_len, q_stride_n, q_stride_h,
    kv_stride_n, kv_stride_h, head_dim, window_left, logits_soft_cap, sm_scale,
    rope_scale, rope_theta);

  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;
    using AttentionVariant = ComposedAttention<ParamsT, get_variant_code(use_custom_mask, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;
    cudaError_t status =
        SinglePrefillWithKVCacheDispatched<{{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, MASK_MODE, AttentionVariant>(
              params, static_cast<{{ dtype_o }}*>(tmp.data_ptr()), stream);
    TORCH_CHECK(status == cudaSuccess,
              "SinglePrefillWithKVCache kernel launch failed, error: " +
                std::string(cudaGetErrorString(status)));
  });
}
""",
    r"""#include "pytorch_extension_utils.h"

void single_prefill_with_kv_cache(
    unsigned int mask_mode_code,
    at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> maybe_packed_custom_mask,
    at::Tensor tmp, std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor o,
    unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
    int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
}
""",
]
