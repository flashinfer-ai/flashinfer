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

single_prefill_sm90_suffix = [
    "_kernel_mask_0.cu",
    "_kernel_mask_1.cu",
    "_kernel_mask_2.cu",
    ".cu",
    "_pybind.cc",
]


single_prefill_sm90_func = r"""void single_prefill_with_kv_cache_sm90(
    unsigned int mask_mode_code,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    std::optional<at::Tensor> maybe_packed_custom_mask,
    std::optional<at::Tensor> maybe_alibi_slopes,
    at::Tensor o,
    unsigned int layout,
    int32_t window_left,
    float logits_soft_cap,
    float sm_scale,
    float rope_scale,
    float rope_theta,
    std::optional<at::Tensor> maybe_lse,
    {{ additional_func_params }}
    int64_t cuda_stream)"""


def single_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return f""" // single_prefill_sm90 template instantiation
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>

namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;

using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
using AttentionVariant = std::conditional_t<
    {{{{use_logits_soft_cap}}}},
    LogitsSoftCap<Params>,
    StandardAttention<Params>>;

template cudaError_t SinglePrefillWithKVCacheDispatched
    <{{{{ head_dim }}}}, {mask_mode}, /*USE_SWA=*/true, AttentionVariant>(
    Params& params, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched
    <{{{{ head_dim }}}}, {mask_mode}, /*USE_SWA=*/false, AttentionVariant>(
    Params& params, cudaStream_t stream);

}}  // namespace flashinfer
"""


single_prefill_sm90_templ = [
    single_prefill_sm90_inst_templ("MaskMode::kNone"),
    single_prefill_sm90_inst_templ("MaskMode::kCausal"),
    single_prefill_sm90_inst_templ("MaskMode::kCustom"),
    f"""// _run.cu
#include <optional>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

{single_prefill_sm90_func} {{
  unsigned int head_dim = q.size(2);
  unsigned int num_qo_heads = q.size(1);
  unsigned int qo_len = q.size(0);

  auto q_scalar_type = q.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
  using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
  using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;

  using SinglePrefillParams = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
  SinglePrefillParams params;
  params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
  params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
  params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
  params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
  params.lse_ptr = maybe_lse ? (static_cast<float*>(maybe_lse->data_ptr())) : nullptr;
  params.q_stride_n = q.stride(0);
  params.q_stride_h = q.stride(1);
  params.o_stride_n = o.stride(0);
  params.o_stride_h = o.stride(1);
  if (kv_layout == QKVLayout::kNHD) {{
    params.k_stride_n = k.stride(0);
    params.k_stride_h = k.stride(1);
    params.v_stride_n = v.stride(0);
    params.v_stride_h = v.stride(1);
  }} else {{
    params.k_stride_h = k.stride(0);
    params.k_stride_n = k.stride(1);
    params.v_stride_h = v.stride(0);
    params.v_stride_n = v.stride(1);
  }}
  params.qo_len = q.size(0);
  params.kv_len = k.size(0);
  params.head_dim = head_dim;
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = k.size(1);
  params.causal = mask_mode == MaskMode::kCausal;
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = window_left;
  params.logits_soft_cap = logits_soft_cap;
  params.sm_scale_log2 = sm_scale * math::log2e;

  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {{
    using AttentionVariant =
        std::conditional_t<{{{{ use_logits_soft_cap }}}},
                           LogitsSoftCap<SinglePrefillParams>,
                           StandardAttention<SinglePrefillParams>>;
    cudaError_t status =
        SinglePrefillWithKVCacheDispatched
            <{{{{ head_dim }}}}, MASK_MODE, {{{{ use_sliding_window }}}}, AttentionVariant>
            (params, stream);
    TORCH_CHECK(status == cudaSuccess,
                "SinglePrefillWithKVCacheDispatched failed with error: ",
                cudaGetErrorString(status));
  }});
}}
""",
    f"""// _pybind.cc
#include "pytorch_extension_utils.h"

{single_prefill_sm90_func};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("run", &single_prefill_with_kv_cache_sm90,
        "Single-request prefill attention with KV-Cache operator");
}}
""",
]


customizable_single_prefill_sm90_func = r"""void single_prefill_with_kv_cache_sm90(
    unsigned int mask_mode_code,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor buffer,
    at::Tensor o,
    unsigned int layout,
    int32_t window_left,
    std::optional<at::Tensor> maybe_lse,
    {{ additional_func_params }}
    int64_t cuda_stream)"""


customizable_single_prefill_sm90_params_templ = r"""
struct SinglePrefillParams {
  using DTypeQ = cutlass_dtype_t<{{ dtype_q }}>;
  using DTypeKV = cutlass_dtype_t<{{ dtype_kv }}>;
  using DTypeO = cutlass_dtype_t<{{ dtype_o }}>;
  using IdType = cutlass_dtype_t<int32_t>;

  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  // Additional params
  {{ additional_params_decl }};

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;

  int qo_len;
  int kv_len;
  int head_dim;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int window_left;

  bool causal;

  // these are bad arguments. we should remove them from default in prefill_sm90.cuh.
  float logits_soft_cap = 0.;
  float sm_scale_log2 = 0.;

  struct AdditionalParams {};
};
"""


def customizable_single_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return f"""// single_prefill_sm90 template instantiation
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>

namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;

{customizable_single_prefill_sm90_params_templ}

{{{{ variant_decl }}}}

using AttentionVariant = {{{{ variant_name }}}}<SinglePrefillParams>;

template cudaError_t SinglePrefillWithKVCacheDispatched
    <{{{{ head_dim }}}}, {mask_mode}, /*USE_SWA=*/true, AttentionVariant>
    (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched
    <{{{{ head_dim }}}}, {mask_mode}, /*USE_SWA=*/false, AttentionVariant>
    (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

}}  // namespace flashinfer
"""


customizable_single_prefill_sm90_suffix = single_prefill_sm90_suffix


customizable_single_prefill_sm90_templ = [
    customizable_single_prefill_sm90_inst_templ("MaskMode::kNone"),
    customizable_single_prefill_sm90_inst_templ("MaskMode::kCausal"),
    customizable_single_prefill_sm90_inst_templ("MaskMode::kCustom"),
    f"""// _run.cu
#include <optional>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include "pytorch_extension_utils.h"

namespace flashinfer {{

{customizable_single_prefill_sm90_params_templ}

{{{{ variant_decl }}}}

}};  // namespace flashinfer

using namespace flashinfer;

{customizable_single_prefill_sm90_func} {{
  unsigned int head_dim = q.size(2);
  unsigned int num_qo_heads = q.size(1);
  unsigned int qo_len = q.size(0);

  auto q_scalar_type = q.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
  using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
  using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;

  SinglePrefillParams params;
  params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
  params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
  params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
  params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
  params.lse_ptr = maybe_lse ? (static_cast<float*>(maybe_lse->data_ptr())) : nullptr;
  params.q_stride_n = q.stride(0);
  params.q_stride_h = q.stride(1);
  params.o_stride_n = o.stride(0);
  params.o_stride_h = o.stride(1);
  if (kv_layout == QKVLayout::kNHD) {{
    params.k_stride_n = k.stride(0);
    params.k_stride_h = k.stride(1);
    params.v_stride_n = v.stride(0);
    params.v_stride_h = v.stride(1);
  }} else {{
    params.k_stride_h = k.stride(0);
    params.k_stride_n = k.stride(1);
    params.v_stride_h = v.stride(0);
    params.v_stride_n = v.stride(1);
  }}
  params.qo_len = q.size(0);
  params.kv_len = k.size(0);
  params.head_dim = head_dim;
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = k.size(1);
  params.causal = mask_mode == MaskMode::kCausal;
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = window_left;
  {{{{ additional_params_setter }}}};

  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {{
    using AttentionVariant = {{{{ variant_name }}}}<SinglePrefillParams>;
    cudaError_t status =
        SinglePrefillWithKVCacheDispatched
            <{{{{ head_dim }}}}, MASK_MODE, {{{{ use_sliding_window }}}}, AttentionVariant>
            (params, stream);
    TORCH_CHECK(status == cudaSuccess,
                "SinglePrefillWithKVCacheDispatched failed with error: ",
                cudaGetErrorString(status));
  }});
}}
""",
    f"""// _pybind.cc
#include "pytorch_extension_utils.h"

{customizable_single_prefill_sm90_func};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("run", &single_prefill_with_kv_cache_sm90,
        "Single-request prefill attention with KV-Cache operator");
}}
""",
]
