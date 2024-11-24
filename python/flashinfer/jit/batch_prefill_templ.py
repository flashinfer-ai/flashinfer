"""
    Copyright(c) 2024 by FlashInfer team.

    Licensed under the Apache License,
    Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

batch_prefill_suffix = [
    "_plan.cu",
    *[f"_ragged_kernel_mask_{mask_mode}.cu" for mask_mode in [0, 1, 2]],
    "_ragged_run.cu",
    *[f"_paged_kernel_mask_{mask_mode}.cu" for mask_mode in [0, 1, 2]],
    "_paged_run.cu",
    "_pybind.cc",
]


def ragged_prefill_inst_templ(mask_mode: str) -> str:
    return (
        r"""#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>

namespace flashinfer {
  {
    % set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %
  }
  using RaggedParamsT =
      BatchPrefillRaggedParams<{{dtype_q}}, {{dtype_kv}}, {{dtype_o}}, {{dtype_idx}}>;
  constexpr bool use_custom_mask =
      ""
      "
      + mask_mode +
      r
      ""
      " == MaskMode::kCustom;
      using RaggedAttentionVariant =
          ComposedAttention<RaggedParamsT,
                            get_variant_code(use_custom_mask, {{use_sliding_window}},
                                             {{use_logits_soft_cap}}, {{use_alibi}})>;

template
cudaError_t BatchPrefillWithRaggedKVCacheDispatched</*cta_tile_q=*/16, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, """
        + mask_mode
        + r""", RaggedAttentionVariant>(
    typename RaggedAttentionVariant::ParamsT params,
    typename RaggedAttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);

template
cudaError_t BatchPrefillWithRaggedKVCacheDispatched</*cta_tile_q=*/64, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, """
        + mask_mode
        + r""", RaggedAttentionVariant>(
    typename RaggedAttentionVariant::ParamsT params,
    typename RaggedAttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);

template
cudaError_t BatchPrefillWithRaggedKVCacheDispatched</*cta_tile_q=*/128, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, """
        + mask_mode
        + r""", RaggedAttentionVariant>(
    typename RaggedAttentionVariant::ParamsT params,
    typename RaggedAttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);
}
"""
    )


def paged_prefill_inst_templ(mask_mode: str) -> str:
    return (
        r"""#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>

namespace flashinfer {
  {
    % set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %
  }
  using PagedParamsT =
      BatchPrefillPagedParams<{{dtype_q}}, {{dtype_kv}}, {{dtype_o}}, {{dtype_idx}}>;
  constexpr bool use_custom_mask =
      ""
      "
      + mask_mode +
      r
      ""
      " == MaskMode::kCustom;
      using PagedAttentionVariant =
          ComposedAttention<PagedParamsT, get_variant_code(use_custom_mask, {{use_sliding_window}},
                                                           {{use_logits_soft_cap}}, {{use_alibi}})>;

template
cudaError_t BatchPrefillWithPagedKVCacheDispatched</*cta_tile_q=*/16, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, """
        + mask_mode
        + r""", PagedAttentionVariant>(
    typename PagedAttentionVariant::ParamsT params,
    typename PagedAttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);

template
cudaError_t BatchPrefillWithPagedKVCacheDispatched</*cta_tile_q=*/64, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, """
        + mask_mode
        + r""", PagedAttentionVariant>(
    typename PagedAttentionVariant::ParamsT params,
    typename PagedAttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);

template
cudaError_t BatchPrefillWithPagedKVCacheDispatched</*cta_tile_q=*/128, {{ head_dim }}, {{ pos_encoding_mode }}, {{ use_fp16_qk_reduction }}, """
        + mask_mode
        + r""", PagedAttentionVariant>(
    typename PagedAttentionVariant::ParamsT params,
    typename PagedAttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);
}
"""
    )


batch_prefill_templ = [
    r"""#include <flashinfer/attention/scheduler.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer,
    at::Tensor qo_indptr,
    at::Tensor kv_indptr,
    unsigned int batch_size,
    unsigned int num_qo_heads,
    unsigned int num_kv_heads,
    unsigned int page_size,
    bool enable_cuda_graph, int64_t cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  PrefillPlanInfo plan_info;

  cudaError_t status = PrefillPlan<{{dtype_idx}}>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<{{dtype_idx}}>(),
      kv_indptr.data_ptr<{{dtype_idx}}>(), batch_size, num_qo_heads, num_kv_heads, {{head_dim}},
      page_size, enable_cuda_graph, sizeof({{dtype_o}}), stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan prefill with error: ", cudaGetErrorString(status));

  return plan_info.ToVector();
}
""",
    *[
        ragged_prefill_inst_templ(mask_mode)
        for mask_mode in ["MaskMode::kNone", "MaskMode::kCausal", "MaskMode::kCustom"]
    ],
    r"""
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

{
  % set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %}
using RaggedParamsT = BatchPrefillRaggedParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;

namespace flashinfer {
  template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
            bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant>
  cudaError_t BatchPrefillWithRaggedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                      typename AttentionVariant::DTypeO * tmp_v,
                                                      float* tmp_s, cudaStream_t stream);

};

void BatchPrefillWithRaggedKVCacheRun(
  unsigned int mask_mode_code,
  at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
  std::vector<int64_t> plan_info_vec,
  at::Tensor q, at::Tensor k, at::Tensor v,
  std::optional<at::Tensor> maybe_custom_mask,
  std::optional<at::Tensor> maybe_alibi_slopes,
  at::Tensor qo_indptr, at::Tensor kv_indptr,
  std::optional<at::Tensor> maybe_qk_indptr,
  at::Tensor o,
  unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
  float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
  int64_t cuda_stream) {
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

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  RaggedParamsT params(
      static_cast<{{dtype_q}}*>(q.data_ptr()), static_cast<{{dtype_kv}}*>(k.data_ptr()),
      static_cast<{{dtype_kv}}*>(v.data_ptr()),
      /*custom_mask=*/
      (maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()) : nullptr),
      static_cast<{{dtype_idx}}*>(qo_indptr.data_ptr()),
      static_cast<{{dtype_idx}}*>(kv_indptr.data_ptr()),
      /*qk_indptr=*/
      (maybe_qk_indptr ? static_cast<{{dtype_idx}}*>(maybe_qk_indptr->data_ptr()) : nullptr),
      /*q_offset=*/nullptr, /*k_rope_pos_offset=*/nullptr, static_cast<{{dtype_o}}*>(o.data_ptr()),
      /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
      { % if use_alibi == "true" % } static_cast<float*>(maybe_alibi_slopes->data_ptr()) {
        % else %
      } nullptr { % endif % },
      num_qo_heads, num_kv_heads, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, window_left,
      logits_soft_cap, sm_scale, rope_scale, rope_theta);

  {{dtype_o}}* tmp_v = nullptr;
  float* tmp_s = nullptr;

  params.request_indices =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    params.merge_indptr =
        GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.merge_indptr_offset);
    tmp_v = GetPtrFromBaseOffset<{{dtype_o}}>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;
  params.max_total_num_rows = plan_info.total_num_rows;
  if (plan_info.enable_cuda_graph) {
    params.total_num_rows =
        GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
  }

  cudaError_t status = cudaSuccess;

  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;
    using RaggedAttentionVariant =
        ComposedAttention<RaggedParamsT, get_variant_code(use_custom_mask, {{use_sliding_window}},
                                                          {{use_logits_soft_cap}}, {{use_alibi}})>;
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
      status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
          CTA_TILE_Q, {{head_dim}}, {{pos_encoding_mode}}, {{use_fp16_qk_reduction}}, MASK_MODE,
          RaggedAttentionVariant>(params, tmp_v, tmp_s, stream);
    });
  });

  TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithRaggedKVCache failed with error ",
              cudaGetErrorString(status));
}
""",
    *[
        paged_prefill_inst_templ(mask_mode)
        for mask_mode in ["MaskMode::kNone", "MaskMode::kCausal", "MaskMode::kCustom"]
    ],
    r"""#include <optional>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

{
  % set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %}
using PagedParamsT = BatchPrefillPagedParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;

namespace flashinfer {
  template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
            bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant>
  cudaError_t BatchPrefillWithPagedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                     typename AttentionVariant::DTypeO * tmp_v,
                                                     float* tmp_s, cudaStream_t stream);

};

void BatchPrefillWithPagedKVCacheRun(
  unsigned int mask_mode_code,
  at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
  std::vector<int64_t> plan_info_vec,
  at::Tensor q,
  at::Tensor paged_k_cache,
  at::Tensor paged_v_cache,
  std::optional<at::Tensor> maybe_custom_mask,
  std::optional<at::Tensor> maybe_alibi_slopes,
  at::Tensor qo_indptr,
  at::Tensor paged_kv_indptr,
  at::Tensor paged_kv_indices,
  at::Tensor paged_kv_last_page_len,
  std::optional<at::Tensor> maybe_qk_indptr,
  at::Tensor o,
  unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
  float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
  int64_t cuda_stream) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  auto device = q.device();
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  paged_kv_t<{{dtype_kv}}, {{dtype_idx}}> paged_kv(
      num_kv_heads, page_size, {{head_dim}}, batch_size, kv_layout,
      static_cast<{{dtype_kv}}*>(paged_k_cache.data_ptr()),
      static_cast<{{dtype_kv}}*>(paged_v_cache.data_ptr()), kv_cache_strides,
      static_cast<{{dtype_idx}}*>(paged_kv_indices.data_ptr()),
      static_cast<{{dtype_idx}}*>(paged_kv_indptr.data_ptr()),
      static_cast<{{dtype_idx}}*>(paged_kv_last_page_len.data_ptr()));

  PagedParamsT params(
      static_cast<{{dtype_q}}*>(q.data_ptr()), paged_kv,
      /*custom_mask=*/
      (maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()) : nullptr),
      static_cast<{{dtype_idx}}*>(qo_indptr.data_ptr()),
      /*qk_indptr=*/
      (maybe_qk_indptr ? static_cast<{{dtype_idx}}*>(maybe_qk_indptr->data_ptr()) : nullptr),
      /*q_offset=*/nullptr, static_cast<{{dtype_o}}*>(o.data_ptr()),
      /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
      { % if use_alibi == "true" % } static_cast<float*>(maybe_alibi_slopes->data_ptr()) {
        % else %
      } nullptr { % endif % },
      num_qo_heads, q_stride_n, q_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale,
      rope_theta);

  {{dtype_o}}* tmp_v = nullptr;
  float* tmp_s = nullptr;

  params.request_indices =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    params.merge_indptr =
        GetPtrFromBaseOffset<{{dtype_idx}}>(int_buffer_ptr, plan_info.merge_indptr_offset);
    tmp_v = GetPtrFromBaseOffset<{{dtype_o}}>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;
  params.max_total_num_rows = plan_info.total_num_rows;
  if (plan_info.enable_cuda_graph) {
    params.total_num_rows =
        GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
  }

  cudaError_t status = cudaSuccess;

  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;
    using PagedAttentionVariant =
        ComposedAttention<PagedParamsT, get_variant_code(use_custom_mask, {{use_sliding_window}},
                                                         {{use_logits_soft_cap}}, {{use_alibi}})>;
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
      status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
          CTA_TILE_Q, {{head_dim}}, {{pos_encoding_mode}}, {{use_fp16_qk_reduction}}, MASK_MODE,
          PagedAttentionVariant>(params, tmp_v, tmp_s, stream);
    });
  });
  TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}
""",
    r"""#include "pytorch_extension_utils.h"

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer,
    at::Tensor qo_indptr,
    at::Tensor kv_indptr,
    unsigned int batch_size,
    unsigned int num_qo_heads,
    unsigned int num_kv_heads,
    unsigned int page_size,
    bool enable_cuda_graph, int64_t cuda_stream);

void BatchPrefillWithRaggedKVCacheRun(
  unsigned int mask_mode_code,
  at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
  std::vector<int64_t> plan_info_vec,
  at::Tensor q, at::Tensor k, at::Tensor v,
  std::optional<at::Tensor> maybe_custom_mask,
  std::optional<at::Tensor> maybe_alibi_slopes,
  at::Tensor qo_indptr, at::Tensor kv_indptr,
  std::optional<at::Tensor> maybe_qk_indptr,
  at::Tensor o,
  unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
  float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
  int64_t cuda_stream);

void BatchPrefillWithPagedKVCacheRun(
  unsigned int mask_mode_code,
  at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
  std::vector<int64_t> plan_info_vec,
  at::Tensor q,
  at::Tensor paged_k_cache,
  at::Tensor paged_v_cache,
  std::optional<at::Tensor> maybe_custom_mask,
  std::optional<at::Tensor> maybe_alibi_slopes,
  at::Tensor qo_indptr,
  at::Tensor paged_kv_indptr,
  at::Tensor paged_kv_indices,
  at::Tensor paged_kv_last_page_len,
  std::optional<at::Tensor> maybe_qk_indptr,
  at::Tensor o,
  unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
  float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse,
  int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchPrefillWithKVCachePlan);
  m.def("ragged_run", &BatchPrefillWithRaggedKVCacheRun);
  m.def("paged_run", &BatchPrefillWithPagedKVCacheRun);
}
""",
]
