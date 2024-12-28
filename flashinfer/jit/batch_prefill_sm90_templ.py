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

batch_prefill_plan_func = r"""std::vector<int64_t> BatchPrefillWithKVCacheSM90Plan(
    bool causal,
    at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer,
    at::Tensor qo_indptr,
    at::Tensor kv_indptr,
    at::Tensor kv_len_arr,
    unsigned int total_num_rows,
    unsigned int batch_size,
    unsigned int num_qo_heads,
    unsigned int num_kv_heads,
    unsigned int page_size,
    bool enable_cuda_graph,
    int64_t cuda_stream)"""


batch_prefill_plan_impl = f"""// _plan.cu
#include <flashinfer/attention/scheduler.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

{batch_prefill_plan_func} {{

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.numel() * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.numel() * int_workspace_buffer.element_size();

  PrefillPlanSM90Info plan_info;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  cudaError_t status = PrefillSM90Plan(
      float_workspace_buffer.data_ptr(),
      float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(),
      page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes,
      plan_info,
      qo_indptr.data_ptr<{{{{ dtype_idx }}}}>(),
      kv_indptr.data_ptr<{{{{ dtype_idx }}}}>(),
      kv_len_arr.data_ptr<{{{{ dtype_idx }}}}>(),
      total_num_rows,
      batch_size,
      num_qo_heads,
      num_kv_heads,
      {{{{ head_dim }}}},
      page_size,
      causal,
      enable_cuda_graph,
      sizeof({{{{dtype_o}}}}),
      stream);

  TORCH_CHECK(status == cudaSuccess,
              "PrefillSM90Plan failed with error: ", cudaGetErrorString(status));

  return plan_info.ToVector();
}}
"""


batch_prefill_sm90_suffix = [
    "_plan.cu",
    "_ragged_kernel_mask_0.cu",
    "_ragged_kernel_mask_1.cu",
    "_ragged_kernel_mask_2.cu",
    "_ragged_run.cu",
    "_paged_kernel_mask_0.cu",
    "_paged_kernel_mask_r.cu",
    "_paged_kernel_mask_2.cu",
    "_paged_run.cu",
    "_pybind.cc",
]


batch_prefill_ragged_func_templ = r"""void BatchPrefillWithRaggedKVCacheSM90Run(
    unsigned int mask_mode_code,
    at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    std::optional<at::Tensor> maybe_custom_mask,
    std::optional<at::Tensor> maybe_alibi_slopes,
    at::Tensor qo_indptr,
    at::Tensor kv_indptr,
    std::optional<at::Tensor> maybe_qk_indptr,
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


def ragged_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return f"""// batch_prefill_ragged_sm90 template inst
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

using RaggedParams = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using AttentionVariant = std::conditional_t<
    {{{{use_logits_soft_cap}}}}, LogitsSoftCap<RaggedParams>, StandardAttention<RaggedParams>>;

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/true,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
     AttentionVariant>(RaggedParams& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/true,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
     AttentionVariant>(RaggedParams& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/false,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
     AttentionVariant>(RaggedParams& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/false,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
     AttentionVariant>(RaggedParams& params, cudaStream_t stream);
}}"""


batch_prefill_paged_func_templ = r"""void BatchPrefillWithPagedKVCacheSM90Run(
    unsigned int mask_mode_code,
    at::Tensor float_workspace_buffer,
    at::Tensor int_workspace_buffer,
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
    unsigned int layout,
    int32_t window_left,
    float logits_soft_cap,
    float sm_scale,
    float rope_scale,
    float rope_theta,
    std::optional<at::Tensor> maybe_lse,
    {{ additional_func_params }}
    int64_t cuda_stream)"""


def paged_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return f"""// batch_prefill_paged_sm90 template inst
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

using PagedParams = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using AttentionVariant = std::conditional_t<
    {{{{use_logits_soft_cap}}}}, LogitsSoftCap<PagedParams>, StandardAttention<PagedParams>>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/true,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
     AttentionVariant>(PagedParams& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/true,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
     AttentionVariant>(PagedParams& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/false,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
     AttentionVariant>(PagedParams& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/false,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
     AttentionVariant>(PagedParams& params, cudaStream_t stream);

}}"""


batch_prefill_sm90_templ = [
    batch_prefill_plan_impl,
    ragged_prefill_sm90_inst_templ("MaskMode::kNone"),
    ragged_prefill_sm90_inst_templ("MaskMode::kCausal"),
    ragged_prefill_sm90_inst_templ("MaskMode::kCustom"),
    f"""// _ragged_run.cu
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

using RaggedParams = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

{batch_prefill_ragged_func_templ} {{
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(plan_info_vec);

  if (maybe_lse) {{
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }}

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
  params.nnz_qo = q.size(0);
  params.nnz_kv = k.size(0);
  params.head_dim = {{{{ head_dim }}}};
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

  bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {{
    DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {{
      using AttentionVariant =
          std::conditional_t<{{{{ use_logits_soft_cap }}}},
                             LogitsSoftCap<RaggedParams>,
                             StandardAttention<RaggedParams>>;
      cudaError_t status =
          BatchPrefillWithRaggedKVCacheDispatched
              <{{{{ head_dim }}}},
               MASK_MODE,
               {{{{ use_sliding_window }}}},
               SAME_SCHEDULER_FOR_ALL_HEADS,
               AttentionVariant>(params, stream);
      TORCH_CHECK(status == cudaSuccess,
                  "BatchPrefillWithRaggedKVCacheSM90Run failed with error: ",
                  cudaGetErrorString(status));
      return true;
    }});
  }});
}}
""",
    paged_prefill_sm90_inst_templ("MaskMode::kNone"),
    paged_prefill_sm90_inst_templ("MaskMode::kCausal"),
    paged_prefill_sm90_inst_templ("MaskMode::kCustom"),
    f"""// _paged_run.cu
#include <cutlass/numeric_types.h>
#include <flashinfer/attention/hopper/params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

using PagedParams = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

{batch_prefill_paged_func_templ} {{
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(plan_info_vec);

  if (maybe_lse) {{
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }}
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  unsigned int num_kv_heads, page_size;
  unsigned int head_dim = q.size(2);
  if (kv_layout == QKVLayout::kHND) {{
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  }} else {{
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }}

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
  if (kv_layout == QKVLayout::kNHD) {{
    // (num_pages, page_size, num_heads, head_dim)
    params.k_stride_n = paged_k_cache.stride(1);
    params.k_stride_h = paged_k_cache.stride(2);
    params.v_stride_n = paged_v_cache.stride(1);
    params.v_stride_h = paged_v_cache.stride(2);
  }} else {{
    // (num_pages, num_heads, page_size, head_dim)
    params.k_stride_h = paged_k_cache.stride(1);
    params.k_stride_n = paged_k_cache.stride(2);
    params.v_stride_h = paged_v_cache.stride(1);
    params.v_stride_n = paged_v_cache.stride(2);
  }}
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

  bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {{
    DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {{
      using AttentionVariant =
          std::conditional_t<{{{{ use_logits_soft_cap }}}},
                             LogitsSoftCap<PagedParams>,
                             StandardAttention<PagedParams>>;
      cudaError_t status =
          BatchPrefillWithPagedKVCacheDispatched
              <{{{{ head_dim }}}},
               MASK_MODE,
               {{{{ use_sliding_window }}}},
               SAME_SCHEDULER_FOR_ALL_HEADS,
               AttentionVariant>(params, stream);
      TORCH_CHECK(status == cudaSuccess,
                  "BatchPrefillWithPagedKVCacheSM90Run failed with error: ",
                  cudaGetErrorString(status));
      return true;
    }});
  }});
}}
""",
    f"""// _pybind.cc
#include "pytorch_extension_utils.h"

{batch_prefill_plan_func};

{batch_prefill_ragged_func_templ};

{batch_prefill_paged_func_templ};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("plan", &BatchPrefillWithKVCacheSM90Plan);
  m.def("ragged_run", &BatchPrefillWithRaggedKVCacheSM90Run);
  m.def("paged_run", &BatchPrefillWithPagedKVCacheSM90Run);
}}
""",
]


# stuffs beyond this line are not tested


customizable_batch_prefill_ragged_params_templ = r"""
struct BatchPrefillRaggedParams {
  using DTypeQ = cutlass_dtype_t<{{ dtype_q }}>;
  using DTypeKV = cutlass_dtype_t<{{ dtype_kv }}>;
  using DTypeO = cutlass_dtype_t<{{ dtype_o }}>;
  using IdType = cutlass_dtype_t<{{ dtype_idx }}>;

  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  // Additional params
  {{ additional_params_decl }};

  IdType* qo_tile_indices;
  IdType* qo_indptr;
  IdType* kv_indptr;
  IdType* qo_lens;
  IdType* kv_lens;
  IdType* head_indices;
  IdType* work_indptr;

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;
  int64_t nnz_qo;
  int64_t nnz_kv;

  int head_dim;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int window_left;

  float logits_soft_cap;
  float sm_scale_log2;
  bool causal;

  struct AdditionalParams {};
};
"""


def customizable_ragged_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return f"""// customizable_ragged_prefill_sm90_inst
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

{customizable_batch_prefill_ragged_params_templ}

{{{{ variant_decl }}}}

using AttentionVariant = {{{{ variant_name }}}}<BatchPrefillRaggedParams>;

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
  <{{{{ head_dim }}}},
    /*mask_mode*/{mask_mode},
    /*USE_SWA=*/true,
    /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
    AttentionVariant>
  (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
  <{{{{ head_dim }}}},
    /*mask_mode*/{mask_mode},
    /*USE_SWA=*/true,
    /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
    AttentionVariant>
  (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
  <{{{{ head_dim }}}},
   /*mask_mode*/{mask_mode},
   /*USE_SWA=*/false,
   /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
   AttentionVariant>
  (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched
  <{{{{ head_dim }}}},
    /*mask_mode*/{mask_mode},
    /*USE_SWA=*/false,
    /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
    AttentionVariant>
  (typename AttentionVariant::ParamsT& params, cudaStream_t stream);
}}  // namespace flashinfer
"""


customizable_batch_prefill_paged_params_templ = r"""
struct BatchPrefillPagedParams {
  using DTypeQ = cutlass_dtype_t<{{ dtype_q }}>;
  using DTypeKV = cutlass_dtype_t<{{ dtype_kv }}>;
  using DTypeO = cutlass_dtype_t<{{ dtype_o }}>;
  using IdType = cutlass_dtype_t<{{ dtype_idx }}>;

  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  // Additional params
  {{ additional_params_decl }};

  IdType* qo_tile_indices;
  IdType* qo_indptr;
  IdType* kv_indptr;
  IdType* kv_indices;
  IdType* qo_lens;
  IdType* kv_lens;
  IdType* head_indices;
  IdType* work_indptr;

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;
  int64_t nnz_qo;

  int head_dim;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int page_size;
  int window_left;

  float logits_soft_cap;
  float sm_scale_log2;
  bool causal;

  struct AdditionalParams {};
};
"""


def customizable_paged_prefill_sm90_inst_templ(mask_mode: str) -> str:
    return f"""#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

{customizable_batch_prefill_paged_params_templ}

{{{{ variant_decl }}}}

using AttentionVariant = {{{{ variant_name }}}}<BatchPrefillPagedParams>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/true,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
     AttentionVariant>
    (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/true,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
     AttentionVariant>
    (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/false,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true,
     AttentionVariant>
    (typename AttentionVariant::ParamsT& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{{{{ head_dim }}}},
     {mask_mode},
     /*USE_SWA=*/false,
     /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false,
     AttentionVariant>
    (typename AttentionVariant::ParamsT& params, cudaStream_t stream);
}}
"""


customizable_batch_prefill_sm90_suffix = [
    "_plan.cu",
    "_ragged_kernel_mask_0.cu",
    "_ragged_kernel_mask_1.cu",
    "_ragged_kernel_mask_2.cu",
    "_ragged_run.cu",
    "_paged_kernel_mask_0.cu",
    "_paged_kernel_mask_1.cu",
    "_paged_kernel_mask_2.cu",
    "_paged_run.cu",
    "_pybind.cu",
]


customizable_batch_prefill_sm90_templ = [
    batch_prefill_plan_impl,
    customizable_ragged_prefill_sm90_inst_templ("MaskMode::kNone"),
    customizable_ragged_prefill_sm90_inst_templ("MaskMode::kCausal"),
    customizable_ragged_prefill_sm90_inst_templ("MaskMode::kCustom"),
    f"""// _ragged_run.cu
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {{

{customizable_batch_prefill_ragged_params_templ}

{{{{ variant_decl }}}}

}};  // namespace flashinfer

using namespace flashinfer;

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

using RaggedParams = BatchPrefillRaggedParams;

{batch_prefill_ragged_func_templ} {{
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(plan_info_vec);

  if (maybe_lse) {{
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }}

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  auto q_scalar_type = q.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  RaggedParams params;

  params.q_ptr = static_cast<cutlass_dtype_t<{{{{ dtype_q }}}}>*>(q.data_ptr());
  params.k_ptr = static_cast<cutlass_dtype_t<{{{{ dtype_kv }}}}>*>(k.data_ptr());
  params.v_ptr = static_cast<cutlass_dtype_t<{{{{ dtype_kv }}}}>*>(v.data_ptr());
  params.o_ptr = static_cast<cutlass_dtype_t<{{{{ dtype_o }}}}>*>(o.data_ptr());
  params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
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
  params.nnz_qo = q.size(0);
  params.nnz_kv = k.size(0);
  params.head_dim = {{{{ head_dim }}}};
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = k.size(1);
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = window_left;
  params.logits_soft_cap = logits_soft_cap;
  params.sm_scale_log2 = sm_scale * math::log2e;
  params.causal = mask_mode_code == 1;
  params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
  params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
  params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
  params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
  params.head_indices = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
  params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
  {{{{ additional_params_setter }}}};

  bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {{
    DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {{
      using AttentionVariant = {{{{ variant_name }}}}<RaggedParams>;
      cudaError_t status =
          BatchPrefillWithRaggedKVCacheDispatched
              <{{{{ head_dim }}}},
               MASK_MODE,
               {{{{ use_sliding_window }}}},
               false,
               AttentionVariant>(params, stream);
      TORCH_CHECK(status == cudaSuccess,
                  "BatchPrefillWithRaggedKVCacheSM90Run failed with error: ",
                  cudaGetErrorString(status));
      return true;
    }});
  }});
}}
""",
    customizable_paged_prefill_sm90_inst_templ("MaskMode::kNone"),
    customizable_paged_prefill_sm90_inst_templ("MaskMode::kCausal"),
    customizable_paged_prefill_sm90_inst_templ("MaskMode::kCustom"),
    f"""// _paged_run.cu
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <cutlass/numeric_types.h>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {{

{customizable_batch_prefill_paged_params_templ}

{{{{ variant_decl }}}}

}};  // namespace flashinfer

using namespace flashinfer;

using DTypeQ = cutlass_dtype_t<{{{{ dtype_q }}}}>;
using DTypeKV = cutlass_dtype_t<{{{{ dtype_kv }}}}>;
using DTypeO = cutlass_dtype_t<{{{{ dtype_o }}}}>;
using IdType = cutlass_dtype_t<{{{{ dtype_idx }}}}>;

using PagedParams = BatchPrefillPagedParams;

{batch_prefill_paged_func_templ} {{
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(plan_info_vec);

  if (maybe_lse) {{
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }}
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  unsigned int num_kv_heads, page_size;
  unsigned int head_dim = q.size(2);
  if (kv_layout == QKVLayout::kHND) {{
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  }} else {{
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }}

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
  if (kv_layout == QKVLayout::kNHD) {{
    // (num_pages, page_size, num_heads, head_dim)
    params.k_stride_n = paged_k_cache.stride(1);
    params.k_stride_h = paged_k_cache.stride(2);
    params.v_stride_n = paged_v_cache.stride(1);
    params.v_stride_h = paged_v_cache.stride(2);
  }} else {{
    // (num_pages, num_heads, page_size, head_dim)
    params.k_stride_h = paged_k_cache.stride(1);
    params.k_stride_n = paged_k_cache.stride(2);
    params.v_stride_h = paged_v_cache.stride(1);
    params.v_stride_n = paged_v_cache.stride(2);
  }}
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
  params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
  params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
  params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
  params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
  params.head_indices = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
  params.work_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
  params.kv_indices = static_cast<IdType*>(paged_kv_indices.data_ptr());
  {{{{ additional_params_setter }}}};

  bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {{
    DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {{
      using AttentionVariant = {{{{ variant_name }}}}<PagedParams>;
      cudaError_t status = BatchPrefillWithPagedKVCacheDispatched
          <{{{{ head_dim }}}},
           MASK_MODE,
           {{{{ use_sliding_window }}}},
           false,
           AttentionVariant>(params, stream);
      TORCH_CHECK(status == cudaSuccess,
                  "BatchPrefillWithPagedKVCacheSM90Run failed with error: ",
                  cudaGetErrorString(status));
      return true;
    }});
  }});
}}
""",
    f"""// _pybind.cu
#include "pytorch_extension_utils.h"

{batch_prefill_plan_func};

{batch_prefill_ragged_func_templ};

{batch_prefill_paged_func_templ};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("plan", &BatchPrefillWithKVCacheSM90Plan);
  m.def("ragged_run", &BatchPrefillWithRaggedKVCacheSM90Run);
  m.def("paged_run", &BatchPrefillWithPagedKVCacheSM90Run);
}}
""",
]
