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
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "pod_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE_P, uint32_t CTA_TILE_Q_P,
          uint32_t CTA_TILE_Q_D, MaskMode MASK_MODE_D, typename PrefillAttentionVariant,
          typename DecodeAttentionVariant, typename PrefillParams, typename DecodeParams>
cudaError_t PODWithKVCacheTensorDispatched(PrefillParams prefill_params, DecodeParams decode_params,
                                           typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                           bool enable_pdl, cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor PODWithKVCachePlan(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                              at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr_p,
                              at::Tensor kv_indptr_p, at::Tensor kv_len_arr_p,
                              uint32_t total_num_rows_p, uint32_t batch_size_p,
                              at::Tensor qo_indptr_d, at::Tensor kv_indptr_d,
                              uint32_t total_num_rows_d, uint32_t batch_size_d,
                              uint32_t num_qo_heads_p, uint32_t num_kv_heads, uint32_t head_dim_qk,
                              uint32_t head_dim_vo, uint32_t page_size, bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  PODPlanInfo plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t status =
      PODPlan<IdType>(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
                      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
                      int_workspace_size_in_bytes, plan_info, qo_indptr_p.data_ptr<IdType>(),
                      kv_indptr_p.data_ptr<IdType>(), total_num_rows_p, batch_size_p,
                      qo_indptr_d.data_ptr<IdType>(), kv_indptr_d.data_ptr<IdType>(),
                      total_num_rows_d, batch_size_d, num_qo_heads_p, num_kv_heads, head_dim_qk,
                      head_dim_vo, page_size, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan prefill with error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void PODWithKVCacheTensorRun(
    // Shared params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
    at::Tensor qo_indptr, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    int64_t layout,
    // Prefill params
    at::Tensor q_p, int64_t mask_mode_code_p, int64_t window_left_p,
    std::optional<at::Tensor> maybe_custom_mask_p, std::optional<at::Tensor> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor q_d, int64_t mask_mode_code_d, int64_t window_left_d,
    std::optional<at::Tensor> maybe_custom_mask_d, std::optional<at::Tensor> maybe_mask_indptr_d,
    std::optional<at::Tensor> maybe_alibi_slopes_d, double logits_soft_cap_d, double sm_scale_d,
    double rope_rcp_scale_d, double rope_rcp_theta_d, bool enable_pdl) {
  PODPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  auto device = q_d.device();
  uint32_t batch_size = paged_kv_indptr.size(0) - 1;
  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer_d.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer_d.data_ptr());
  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  // Prefill setup
  uint32_t head_dim_qk = q_p.size(2);
  uint32_t qo_len, num_qo_heads_p;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q_p.size(0) + q_d.size(0);
  num_qo_heads_p = q_p.size(1);
  uint32_t q_stride_n_p = q_p.stride(0), q_stride_h_p = q_p.stride(1);
  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == qo_len, lse.size(0), qo_len);
    TORCH_CHECK(lse.size(1) == num_qo_heads_p, lse.size(1), q_p.size(1));
  }

  const MaskMode mask_mode_p = static_cast<MaskMode>(mask_mode_code_p);

  auto q_scalar_type = q_p.scalar_type();

  // Decode setup (Tensor decode = batched prefill)
  uint32_t num_qo_heads = q_d.size(1);
  TORCH_CHECK(num_qo_heads_p == num_qo_heads,
              "POD currently requires same # Query heads for prefill and decode");

  uint32_t num_kv_heads_d, num_kv_heads, page_size;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    num_kv_heads_d = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    num_kv_heads = paged_k_cache.size(2);
    num_kv_heads_d = paged_k_cache.size(2);
    page_size = paged_k_cache.size(1);
  }
  TORCH_CHECK(num_kv_heads == num_kv_heads_d,
              "POD currently requires same # KV heads for prefill and decode; Prefill: ",
              num_kv_heads, ", Decode: ", num_kv_heads_d);

  const MaskMode mask_mode_d = static_cast<MaskMode>(mask_mode_code_d);

  // get q_stride_n and q_stride_h
  const auto q_stride_n_d = q_d.stride(0);
  const auto q_stride_h_d = q_d.stride(1);

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer_d.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK, USE_SLIDING_WINDOW_P,
      USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, [&] {
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        PrefillParams prefill_params;
        {
          // Make params a reference to prefill_params to set values
          PrefillParams& params = prefill_params;
          params.q = static_cast<DTypeQ*>(q_p.data_ptr());
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
          params.o = static_cast<DTypeO*>(o.data_ptr());
          params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
          params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
          params.q_stride_n = q_stride_n_p;
          params.q_stride_h = q_stride_h_p;
          params.window_left = window_left_p;
          params.paged_kv.num_heads = num_kv_heads;
          params.num_qo_heads = num_qo_heads;

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
          params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
          if (plan_info.split_kv) {
            params.merge_indptr =
                GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
            }
          }
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset_p);
          params.padded_batch_size = plan_info.padded_batch_size_p;
          params.maybe_custom_mask = maybe_custom_mask_p
                                         ? static_cast<uint8_t*>(maybe_custom_mask_p->data_ptr())
                                         : nullptr;
          params.maybe_alibi_slopes = maybe_alibi_slopes_p
                                          ? static_cast<float*>(maybe_alibi_slopes_p->data_ptr())
                                          : nullptr;
          params.logits_soft_cap = logits_soft_cap_p;
          params.sm_scale = sm_scale_p;
          params.rope_rcp_scale = rope_rcp_scale_p;
          params.rope_rcp_theta = rope_rcp_theta_p;
          params.max_total_num_rows = plan_info.total_num_rows;
          if (plan_info.enable_cuda_graph) {
            params.total_num_rows =
                GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
          }
          params.partition_kv = plan_info.split_kv;
          if (plan_info.split_kv) {
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
            }
          }
        }

        DecodeParams decode_params;
        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        {
          DecodeParams& params = decode_params;
          params.q = static_cast<DTypeQ*>(q_d.data_ptr());
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
          params.o = static_cast<DTypeO*>(o.data_ptr());
          params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
          params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
          params.q_stride_n = q_stride_n_d;
          params.q_stride_h = q_stride_h_d;
          params.window_left = window_left_d;
          params.paged_kv.num_heads = num_kv_heads;
          params.num_qo_heads = num_qo_heads;

          params.request_indices = prefill_params.request_indices;
          params.qo_tile_indices = prefill_params.qo_tile_indices;
          params.kv_tile_indices = prefill_params.kv_tile_indices;
          params.o_indptr = prefill_params.o_indptr;
          params.kv_chunk_size_ptr = prefill_params.kv_chunk_size_ptr;

          params.partition_kv = plan_info.split_kv;
          if (plan_info.split_kv) {
            params.merge_indptr = prefill_params.merge_indptr;
            // These should be assigned from plan info, not from prefill_params
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask = prefill_params.block_valid_mask;
            }
          }
          params.padded_batch_size = plan_info.padded_batch_size_d;
          params.max_total_num_rows = plan_info.total_num_rows;

          params.maybe_mask_indptr = maybe_mask_indptr_d
                                         ? static_cast<int32_t*>(maybe_mask_indptr_d->data_ptr())
                                         : nullptr;
          params.maybe_alibi_slopes = maybe_alibi_slopes_d
                                          ? static_cast<float*>(maybe_alibi_slopes_d->data_ptr())
                                          : nullptr;
          params.logits_soft_cap = logits_soft_cap_d;
          params.sm_scale = sm_scale_d;
          params.rope_rcp_scale = rope_rcp_scale_d;
          params.rope_rcp_theta = rope_rcp_theta_d;

          if (plan_info.enable_cuda_graph) {
            params.total_num_rows = prefill_params.total_num_rows;
          }
        }

        constexpr bool use_custom_mask_p = MASK_MODE_P == MaskMode::kCustom;
        using PrefillAttentionVariant =
            DefaultAttention</*use_custom_mask=*/use_custom_mask_p, USE_SLIDING_WINDOW_P,
                             USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;
        constexpr bool use_custom_mask_d = MASK_MODE_D == MaskMode::kCustom;
        using DecodeAttentionVariant =
            DefaultAttention</*use_custom_mask=*/use_custom_mask_d, USE_SLIDING_WINDOW_D,
                             USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;
        // DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        constexpr size_t CTA_TILE_Q_P = plan_info.cta_tile_q_p;
        constexpr size_t CTA_TILE_Q_D = plan_info.cta_tile_q_d;
        cudaError_t status = flashinfer::PODWithKVCacheTensorDispatched<
            HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION, MASK_MODE_P,
            CTA_TILE_Q_P, CTA_TILE_Q_D, MASK_MODE_D, PrefillAttentionVariant,
            DecodeAttentionVariant>(prefill_params, decode_params, tmp_v, tmp_s, enable_pdl,
                                    stream);
        TORCH_CHECK(status == cudaSuccess, "PODWithKVCache kernel launch failed, error: " +
                                               std::string(cudaGetErrorString(status)));
        //});
      });
}
