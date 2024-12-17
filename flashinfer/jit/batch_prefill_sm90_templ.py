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

batch_prefill_sm90_suffix = [
    "_plan.cu",
    *[f"_ragged_kernel_mask_{mask_mode}.cu" for mask_mode in [0, 1, 2]],
    "_ragged_run.cu",
    *[f"_paged_kernel_mask_{mask_mode}.cu" for mask_mode in [0, 1, 2]],
    "_paged_run.cu",
    "_pybind.cc",
]


def ragged_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return (
        r"""#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

namespace flashinfer {

using DTypeQ = cutlass_dtype_t<{{dtype_q}}>;
using DTypeKV = cutlass_dtype_t<{{dtype_kv}}>;
using DTypeO = cutlass_dtype_t<{{dtype_o}}>;
using IdType = cutlass_dtype_t<{{dtype_idx}}>;

using RaggedParams =
    BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using AttentionVariant = std::conditional_t<{{use_logits_soft_cap}}, LogitsSoftCap, StandardAttention>;

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<{{ head_dim }}, """
        + mask_mode
        + r""", /*USE_SWA=*/true, AttentionVariant>(
    RaggedParams& params,
    cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<{{ head_dim }}, """
        + mask_mode
        + r""", /*USE_SWA=*/false, AttentionVariant>(
    RaggedParams& params,
    cudaStream_t stream);

}"""
    )


def paged_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return (
        r"""#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

namespace flashinfer {

using DTypeQ = cutlass_dtype_t<{{dtype_q}}>;
using DTypeKV = cutlass_dtype_t<{{dtype_kv}}>;
using DTypeO = cutlass_dtype_t<{{dtype_o}}>;
using IdType = cutlass_dtype_t<{{dtype_idx}}>;

using PagedParams = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using AttentionVariant = std::conditional_t<{{use_logits_soft_cap}}, LogitsSoftCap, StandardAttention>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<{{ head_dim }}, """
        + mask_mode
        + r""", /*USE_SWA=*/true, AttentionVariant>(
    PagedParams& params,
    cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<{{ head_dim }}, """
        + mask_mode
        + r""", /*USE_SWA=*/false, AttentionVariant>(
    PagedParams& params,
    cudaStream_t stream);

}"""
    )


batch_prefill_sm90_templ = [
    r"""#include <flashinfer/attention/scheduler.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

std::vector<int64_t> BatchPrefillWithKVCacheSM90Plan(
    bool causal, at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer, at::Tensor page_locked_int_workspace_buffer,
    at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len_arr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph, int64_t cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  PrefillPlanSM90Info plan_info;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  cudaError_t status = PrefillSM90Plan(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<{{ dtype_idx }}>(),
      kv_indptr.data_ptr<{{ dtype_idx }}>(), kv_len_arr.data_ptr<{{ dtype_idx }}>(), batch_size, num_qo_heads,
      num_kv_heads, {{ head_dim }}, page_size, causal, enable_cuda_graph, sizeof({{dtype_o}}), stream);

  TORCH_CHECK(status == cudaSuccess,
              "PrefillSM90Plan failed with error: ", cudaGetErrorString(status));

  return plan_info.ToVector();
}
""",
    *[
        ragged_prefill_sm90_inst_templ(mask_mode)
        for mask_mode in ["MaskMode::kNone", "MaskMode::kCausal", "MaskMode::kCustom"]
    ],
    r"""
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>& params, cudaStream_t stream);

};  // namespace flashinfer

using namespace flashinfer;

using DTypeQ = cutlass_dtype_t<{{ dtype_q }}>;
using DTypeKV = cutlass_dtype_t<{{ dtype_kv }}>;
using DTypeO = cutlass_dtype_t<{{ dtype_o }}>;
using IdType = cutlass_dtype_t<{{ dtype_idx }}>;

using RaggedParams = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

void BatchPrefillWithRaggedKVCacheSM90Run(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes,
    at::Tensor qo_indptr, at::Tensor kv_indptr, std::optional<at::Tensor> maybe_qk_indptr,
    at::Tensor o, unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(plan_info_vec);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  auto q_scalar_type = q.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  RaggedParams params;

  params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
  params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
  params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
  params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
  params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
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
  params.nnz_qo = q.size(0);
  params.nnz_kv = k.size(0);
  params.head_dim = {{ head_dim }};
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = k.size(1);
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = window_left;
  params.logits_soft_cap = logits_soft_cap;
  params.sm_scale_log2 = sm_scale * math::log2e;
  params.causal = mask_mode_code == 1;
  params.qo_tile_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
  params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
  params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
  params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
  params.head_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
  params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);

  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    using AttentionVariant =
        std::conditional_t<{{ use_logits_soft_cap }}, LogitsSoftCap, StandardAttention>;
    cudaError_t status =
        BatchPrefillWithRaggedKVCacheDispatched<{{ head_dim }}, MASK_MODE, {{ use_sliding_window }},
                                                AttentionVariant>(params, stream);
    TORCH_CHECK(status == cudaSuccess,
                "BatchPrefillWithRaggedKVCacheSM90Run failed with error: ",
                cudaGetErrorString(status));
  });
}
""",
    *[
        paged_prefill_sm90_inst_templ(mask_mode)
        for mask_mode in ["MaskMode::kNone", "MaskMode::kCausal", "MaskMode::kCustom"]
    ],
    r"""#include <cutlass/numeric_types.h>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>& params, cudaStream_t stream);

};  // namespace flashinfer

using namespace flashinfer;

using DTypeQ = cutlass_dtype_t<{{ dtype_q }}>;
using DTypeKV = cutlass_dtype_t<{{ dtype_kv }}>;
using DTypeO = cutlass_dtype_t<{{ dtype_o }}>;
using IdType = cutlass_dtype_t<{{ dtype_idx }}>;

using PagedParams = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

void BatchPrefillWithPagedKVCacheSM90Run(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, std::optional<at::Tensor> maybe_custom_mask,
    std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    std::optional<at::Tensor> maybe_qk_indptr, at::Tensor o, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<at::Tensor> maybe_lse, int64_t cuda_stream) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(plan_info_vec);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  unsigned int num_kv_heads, page_size;
  unsigned int head_dim = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  auto q_scalar_type = q.scalar_type();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  PagedParams params;

  params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
  params.k_ptr = static_cast<DTypeKV*>(paged_k_cache.data_ptr());
  params.v_ptr = static_cast<DTypeKV*>(paged_v_cache.data_ptr());
  params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
  params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
  params.q_stride_n = q.stride(0);
  params.q_stride_h = q.stride(1);
  params.o_stride_n = o.stride(0);
  params.o_stride_h = o.stride(1);
  if (kv_layout == QKVLayout::kNHD) {
    // (num_pages, page_size, num_heads, head_dim)
    params.k_stride_n = paged_k_cache.stride(1);
    params.k_stride_h = paged_k_cache.stride(2);
    params.v_stride_n = paged_v_cache.stride(1);
    params.v_stride_h = paged_v_cache.stride(2);
  } else {
    // (num_pages, num_heads, page_size, head_dim)
    params.k_stride_h = paged_k_cache.stride(1);
    params.k_stride_n = paged_k_cache.stride(2);
    params.v_stride_h = paged_v_cache.stride(1);
    params.v_stride_n = paged_v_cache.stride(2);
  }
  params.nnz_qo = q.size(0);
  params.head_dim = head_dim;
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = num_kv_heads;
  params.group_size = params.num_qo_heads / num_kv_heads;
  params.page_size = page_size;
  params.window_left = window_left;
  params.logits_soft_cap = logits_soft_cap;
  params.sm_scale_log2 = sm_scale * math::log2e;
  params.causal = mask_mode_code == 1;
  params.qo_tile_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
  params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
  params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
  params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
  params.head_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
  params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
  params.kv_indices = static_cast<IdType*>(paged_kv_indices.data_ptr());

  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
    using AttentionVariant =
        std::conditional_t<{{ use_logits_soft_cap }}, LogitsSoftCap, StandardAttention>;
    cudaError_t status =
        BatchPrefillWithPagedKVCacheDispatched<{{ head_dim }}, MASK_MODE, {{ use_sliding_window }},
                                                AttentionVariant>(params, stream);
    TORCH_CHECK(status == cudaSuccess,
                "BatchPrefillWithPagedKVCacheSM90Run failed with error: ",
                cudaGetErrorString(status));
  });
}
""",
    r"""#include "pytorch_extension_utils.h"

std::vector<int64_t> BatchPrefillWithKVCacheSM90Plan(
    bool causal, at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer, at::Tensor page_locked_int_workspace_buffer,
    at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len_arr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph, int64_t cuda_stream);

void BatchPrefillWithRaggedKVCacheSM90Run(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes,
    at::Tensor qo_indptr, at::Tensor kv_indptr, std::optional<at::Tensor> maybe_qk_indptr,
    at::Tensor o, unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

void BatchPrefillWithPagedKVCacheSM90Run(
    unsigned int mask_mode_code, at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, std::optional<at::Tensor> maybe_custom_mask,
    std::optional<at::Tensor> maybe_alibi_slopes, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    std::optional<at::Tensor> maybe_qk_indptr, at::Tensor o, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchPrefillWithKVCacheSM90Plan);
  m.def("ragged_run", &BatchPrefillWithRaggedKVCacheSM90Run);
  m.def("paged_run", &BatchPrefillWithPagedKVCacheSM90Run);
}
""",
]