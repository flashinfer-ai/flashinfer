/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>

#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {

template <WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                   typename AttentionVariant::DTypeO* tmp_v,
                                                   float* tmp_s, cudaStream_t stream);

template <WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                    typename AttentionVariant::DTypeO* tmp_v,
                                                    float* tmp_s, cudaStream_t stream);

}  // namespace flashinfer

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    unsigned int head_dim, torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer, torch::Tensor qo_indptr,
    torch::Tensor kv_indptr, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  qo_indptr = qo_indptr.to(torch::kCPU);
  kv_indptr = kv_indptr.to(torch::kCPU);

  PrefillPlanInfo plan_info;

  using IdType = int32_t;

  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
      kv_indptr.data_ptr<IdType>(), batch_size, num_qo_heads, num_kv_heads, head_dim, page_size,
      enable_cuda_graph, /*sizeof_dtype_o=*/2, torch_current_stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan prefill with error: ", cudaGetErrorString(status));

  return plan_info.ToVector();
}

std::vector<torch::Tensor> BatchPrefillWithRaggedKVCacheRun(
    unsigned int mask_mode_code, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, std::vector<int64_t> plan_info_vec, torch::Tensor q,
    torch::Tensor k, torch::Tensor v, std::optional<torch::Tensor> maybe_custom_mask,
    std::optional<torch::Tensor> maybe_alibi_slopes, torch::Tensor qo_indptr,
    torch::Tensor kv_indptr, std::optional<torch::Tensor> maybe_qk_indptr, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    bool return_lse) {
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

  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  int64_t nnz_qo = q.size(0);
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const bool use_logits_soft_cap = logits_soft_cap > 0.f;
  using IdType = int32_t;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, q_type, kv_type, [&] {
    using DTypeQ = q_type;
    using DTypeKV = kv_type;
    using DTypeO = DTypeQ;
    return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
      return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
        return DISPATCH_LOGITS_SOFT_CAP(use_logits_soft_cap, USE_LOGITS_SOFT_CAP, [&] {
          using RaggedParamsT = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
          using RaggedAttentionVariant =
              ComposedAttention<RaggedParamsT,
                                get_variant_code(/*use_custom_mask=*/MASK_MODE == MaskMode::kCustom,
                                                 /*use_sliding_window=*/true, USE_LOGITS_SOFT_CAP,
                                                 /*use_alibi_slopes=*/false)>;

          RaggedParamsT params(
              static_cast<DTypeQ*>(q.data_ptr()), static_cast<DTypeKV*>(k.data_ptr()),
              static_cast<DTypeKV*>(v.data_ptr()),
              maybe_custom_mask.has_value() ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr())
                                            : nullptr,
              static_cast<IdType*>(qo_indptr.data_ptr()),
              static_cast<IdType*>(kv_indptr.data_ptr()),
              maybe_qk_indptr.has_value() ? static_cast<IdType*>(maybe_qk_indptr->data_ptr())
                                          : nullptr,
              /*q_offset=*/nullptr,
              /*k_rope_pos_offset=*/nullptr, static_cast<DTypeO*>(o.data_ptr()),
              /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
              /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, q_stride_n, q_stride_h,
              kv_stride_n, kv_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale,
              rope_theta);

          DTypeO* tmp_v = nullptr;
          float* tmp_s = nullptr;

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
          params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
          if (plan_info.split_kv) {
            params.merge_indptr =
                GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
            }
          }
          params.total_num_rows = plan_info.total_num_rows;
          params.padded_batch_size = plan_info.padded_batch_size;

          WarpLayout warp_layout = WarpLayout(plan_info.warp_layout_code);
          cudaError_t status = cudaSuccess;

          DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
            status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
                WARP_LAYOUT, HEAD_DIM, POS_ENCODING_MODE,
                /*use_fp16_qk_reduction=*/false, MASK_MODE, RaggedAttentionVariant>(
                params, tmp_v, tmp_s, torch_current_stream);
          });

          TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithRaggedKVCache failed with error ",
                      cudaGetErrorString(status));
          return true;
        });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

std::vector<torch::Tensor> BatchPrefillWithPagedKVCacheRun(
    unsigned int mask_mode_code, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, std::vector<int64_t> plan_info_vec, torch::Tensor q,
    torch::Tensor paged_k_cache, torch::Tensor paged_v_cache,
    std::optional<torch::Tensor> maybe_custom_mask, std::optional<torch::Tensor> maybe_alibi_slopes,
    torch::Tensor qo_indptr, torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, std::optional<torch::Tensor> maybe_qk_indptr,
    unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, bool return_lse) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  auto device = q.device();
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  int64_t nnz_qo = q.size(0);
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  using IdType = int32_t;
  bool use_logits_soft_cap = logits_soft_cap > 0.f;
  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, q_type, kv_type, [&] {
    using DTypeQ = q_type;
    using DTypeKV = kv_type;
    using DTypeO = DTypeQ;
    return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
      return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
        return DISPATCH_LOGITS_SOFT_CAP(use_logits_soft_cap, USE_LOGITS_SOFT_CAP, [&] {
          paged_kv_t<DTypeKV, IdType> paged_kv(
              num_kv_heads, page_size, HEAD_DIM, batch_size, kv_layout,
              static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache.data_ptr()),
              static_cast<IdType*>(paged_kv_indices.data_ptr()),
              static_cast<IdType*>(paged_kv_indptr.data_ptr()),
              static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));

          using PagedParamsT = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
          using PagedAttentionVariant =
              ComposedAttention<PagedParamsT,
                                get_variant_code(/*use_custom_mask=*/MASK_MODE == MaskMode::kCustom,
                                                 /*use_sliding_window=*/true, USE_LOGITS_SOFT_CAP,
                                                 /*use_alibi_slopes=*/false)>;

          PagedParamsT params(
              static_cast<DTypeQ*>(q.data_ptr()), paged_kv,
              maybe_custom_mask.has_value() ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr())
                                            : nullptr,
              static_cast<IdType*>(qo_indptr.data_ptr()),
              maybe_qk_indptr.has_value() ? static_cast<IdType*>(maybe_qk_indptr->data_ptr())
                                          : nullptr,
              /*q_offset=*/nullptr, static_cast<DTypeO*>(o.data_ptr()),
              /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
              /*alibi_slopes=*/nullptr, num_qo_heads, window_left, logits_soft_cap, sm_scale,
              rope_scale, rope_theta);

          DTypeO* tmp_v = nullptr;
          float* tmp_s = nullptr;

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
          params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
          if (plan_info.split_kv) {
            params.merge_indptr =
                GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
            }
          }
          params.total_num_rows = plan_info.total_num_rows;
          params.padded_batch_size = plan_info.padded_batch_size;

          WarpLayout warp_layout = WarpLayout(plan_info.warp_layout_code);
          cudaError_t status = cudaSuccess;

          DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
            status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
                WARP_LAYOUT, HEAD_DIM, POS_ENCODING_MODE,
                /*use_fp16_qk_reduction=*/false, MASK_MODE, PagedAttentionVariant>(
                params, tmp_v, tmp_s, torch_current_stream);
          });

          TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                      cudaGetErrorString(status));
          return true;
        });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
