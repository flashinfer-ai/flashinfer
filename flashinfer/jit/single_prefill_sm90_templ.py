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
    *[f"_kernel_mask_{mask_mode}.cu" for mask_mode in [0, 1, 2]],
    ".cu",
    "_pybind.cc",
]


def single_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return (
        r"""#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>

namespace flashinfer {

using DTypeQ = cutlass_dtype_t<{{dtype_q}}>;
using DTypeKV = cutlass_dtype_t<{{dtype_kv}}>;
using DTypeO = cutlass_dtype_t<{{dtype_o}}>;

using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
using AttentionVariant = std::conditional_t<{{use_logits_soft_cap}}, LogitsSoftCap, StandardAttention>;

template cudaError_t SinglePrefillWithKVCacheDispatched<{{ head_dim }},"""
        f"{mask_mode}"
        r""", /*USE_SWA=*/false, AttentionVariant>(
    Params& params,
    cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<{{ head_dim }},"""
        f"{mask_mode}"
        r""", /*USE_SWA=*/true, AttentionVariant>(
    Params& params,
    cudaStream_t stream);

}  // namespace flashinfer
"""
    )


single_prefill_sm90_templ = [
    *[
        single_prefill_sm90_inst_templ(mask_mode)
        for mask_mode in ["MaskMode::kNone", "MaskMode::kCausal", "MaskMode::kCustom"]
    ],
    r"""#include <optional>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t SinglePrefillWithKVCacheDispatched(SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
                                               cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void single_prefill_with_kv_cache_sm90(unsigned int mask_mode_code, at::Tensor q, at::Tensor k,
                                       at::Tensor v,
                                       std::optional<at::Tensor> maybe_packed_custom_mask,
                                       std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor o,
                                       unsigned int layout, int32_t window_left,
                                       float logits_soft_cap, float sm_scale, float rope_scale,
                                       float rope_theta, std::optional<at::Tensor> maybe_lse,
                                       int64_t cuda_stream) {
  unsigned int head_dim = q.size(2);
  unsigned int num_qo_heads = q.size(1);
  unsigned int qo_len = q.size(0);

  auto q_scalar_type = q.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  using DTypeQ = cutlass_dtype_t<{{dtype_q}}>;
  using DTypeKV = cutlass_dtype_t<{{dtype_kv}}>;
  using DTypeO = cutlass_dtype_t<{{dtype_o}}>;

  SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> params;
  params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
  params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
  params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
  params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
  params.lse_ptr = maybe_lse ? (static_cast<float*>(maybe_lse->data_ptr())) : nullptr;
  params.q_stride_n = q.stride(0);
  params.q_stride_h = q.stride(1);
  params.o_stride_n = o.stride(0);
  params.o_stride_h = o.stride(1);
  if (kv_layout == QKVLayout::kNHD) {
    params.k_stride_n = k.stride(0);
    params.k_stride_h = k.stride(1);
    params.v_stride_n = v.stride(0);
    params.v_stride_h = v.stride(1);
  } else {
    params.k_stride_h = k.stride(0);
    params.k_stride_n = k.stride(1);
    params.v_stride_h = v.stride(0);
    params.v_stride_n = v.stride(1);
  }
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
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    using AttentionVariant =
        std::conditional_t<{{ use_logits_soft_cap }}, LogitsSoftCap, StandardAttention>;
    cudaError_t status =
        SinglePrefillWithKVCacheDispatched<{{ head_dim }}, MASK_MODE, {{ use_sliding_window }}, AttentionVariant>(
            params, stream);
    TORCH_CHECK(status == cudaSuccess,
                "single_prefill_with_kv_cache_sm90 failed with error: " +
                    std::string(cudaGetErrorString(status)));
  });
}
""",
    r"""#include "pytorch_extension_utils.h"

void single_prefill_with_kv_cache_sm90(unsigned int mask_mode_code, at::Tensor q, at::Tensor k,
                                       at::Tensor v,
                                       std::optional<at::Tensor> maybe_packed_custom_mask,
                                       std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor o,
                                       unsigned int layout, int32_t window_left,
                                       float logits_soft_cap, float sm_scale, float rope_scale,
                                       float rope_theta, std::optional<at::Tensor> maybe_lse,
                                       int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_prefill_with_kv_cache_sm90,
        "Single-request prefill attention with KV-Cache operator");
}
""",
]
