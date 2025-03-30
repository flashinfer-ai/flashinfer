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
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE_P, uint32_t CTA_TILE_Q,
          MaskMode MASK_MODE_D, typename PrefillAttentionVariant, typename DecodeAttentionVariant,
          typename PrefillParams, typename DecodeParams>
cudaError_t PODWithKVCacheTensorDispatched(PrefillParams prefill_params,
                                           typename PrefillParams::DTypeO* tmp,
                                           DecodeParams decode_params,
                                           typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                           cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void pod_with_kv_cache_tensor(
    // Prefill params
    at::Tensor q_p, at::Tensor k_p, at::Tensor v_p, at::Tensor tmp_p, at::Tensor o_p,
    std::optional<at::Tensor> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, std::optional<at::Tensor> maybe_custom_mask_p,
    std::optional<at::Tensor> maybe_alibi_slopes_p, double logits_soft_cap_p, double sm_scale_p,
    double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec, at::Tensor q_d, at::Tensor paged_k_cache_d,
    at::Tensor paged_v_cache_d, at::Tensor qo_indptr_d, at::Tensor paged_kv_indptr_d,
    at::Tensor paged_kv_indices_d, at::Tensor paged_kv_last_page_len_d, at::Tensor o_d,
    std::optional<at::Tensor> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, std::optional<at::Tensor> maybe_custom_mask_d,
    std::optional<at::Tensor> maybe_mask_indptr_d, std::optional<at::Tensor> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d) {
  // Prefill setup
  unsigned int head_dim_qk = q_p.size(2);
  unsigned int kv_len_p, qo_len_p, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout_p = static_cast<QKVLayout>(layout_p);
  qo_len_p = q_p.size(0);
  num_qo_heads = q_p.size(1);
  uint32_t q_stride_n_p = q_p.stride(0), q_stride_h_p = q_p.stride(1), k_stride_n_p, k_stride_h_p,
           v_stride_n_p, v_stride_h_p;
  if (kv_layout_p == QKVLayout::kNHD) {
    kv_len_p = k_p.size(0);
    num_kv_heads = k_p.size(1);
    k_stride_n_p = k_p.stride(0);
    k_stride_h_p = k_p.stride(1);
    v_stride_n_p = v_p.stride(0);
    v_stride_h_p = v_p.stride(1);
  } else {
    kv_len_p = k_p.size(1);
    num_kv_heads = k_p.size(0);
    k_stride_h_p = k_p.stride(0);
    k_stride_n_p = k_p.stride(1);
    v_stride_h_p = v_p.stride(0);
    v_stride_n_p = v_p.stride(1);
  }
  if (maybe_lse_p) {
    const auto& lse = *maybe_lse_p;
    TORCH_CHECK(lse.size(0) == qo_len_p, lse.size(0), q_p.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q_p.size(1));
  }

  const MaskMode mask_mode_p = static_cast<MaskMode>(mask_mode_code_p);

  auto q_scalar_type = q_p.scalar_type();
  auto kv_scalar_type = k_p.scalar_type();

  // Decode setup (Tensor decode = batched prefill)
  PrefillPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout_d = static_cast<QKVLayout>(layout_d);
  auto device = q_d.device();
  int64_t batch_size = paged_kv_indptr_d.size(0) - 1;
  int64_t num_qo_heads_d = q_d.size(1);

  TORCH_CHECK(num_qo_heads == num_qo_heads_d,
              "POD currently requires same # Query heads for prefill and decode");

  int64_t num_kv_heads_d, page_size_d;
  uint32_t head_dim_qk_d = q_d.size(2);
  if (kv_layout_d == QKVLayout::kHND) {
    num_kv_heads_d = paged_k_cache_d.size(1);
    page_size_d = paged_k_cache_d.size(2);
  } else {
    page_size_d = paged_k_cache_d.size(1);
    num_kv_heads_d = paged_k_cache_d.size(2);
  }
  TORCH_CHECK(num_kv_heads == num_kv_heads_d,
              "POD currently requires same # KV heads for prefill and decode; Prefill: ",
              num_kv_heads, ", Decode: ", num_kv_heads_d);

  if (maybe_lse_d) {
    const auto& lse = *maybe_lse_d;
    TORCH_CHECK(lse.size(0) == q_d.size(0), lse.size(0), q_d.size(0));
    TORCH_CHECK(lse.size(1) == q_d.size(1), lse.size(1), q_d.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer_d.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer_d.data_ptr());

  const MaskMode mask_mode_d = static_cast<MaskMode>(mask_mode_code_d);
  auto q_scalar_type_d = q_d.scalar_type();
  auto kv_scalar_type_d = paged_k_cache_d.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n_d = q_d.stride(0);
  const auto q_stride_h_d = q_d.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides_d = nullptr;
  auto k_strides_d = paged_k_cache_d.strides();
  auto v_strides_d = paged_v_cache_d.strides();
  TORCH_CHECK(k_strides_d == v_strides_d, "k/v strides must be identical");
  kv_cache_strides_d = k_strides_d.data();

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer_d.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK, USE_SLIDING_WINDOW_P,
      USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, [&] {
        PrefillParams prefill_params;
        {
          // Make params a reference to prefill_params to set values
          PrefillParams& params = prefill_params;
          params.q = static_cast<DTypeQ*>(q_p.data_ptr());
          params.k = static_cast<DTypeKV*>(k_p.data_ptr());
          params.v = static_cast<DTypeKV*>(v_p.data_ptr());
          params.o = static_cast<DTypeO*>(o_p.data_ptr());
          params.lse = maybe_lse_p ? static_cast<float*>(maybe_lse_p->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.num_kv_heads = num_kv_heads;
          params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
          params.qo_len = qo_len_p;
          params.kv_len = kv_len_p;
          params.q_stride_n = q_stride_n_p;
          params.q_stride_h = q_stride_h_p;
          params.k_stride_n = k_stride_n_p;
          params.k_stride_h = k_stride_h_p;
          params.v_stride_n = v_stride_n_p;
          params.v_stride_h = v_stride_h_p;

          params.window_left = window_left_p;
          params.partition_kv = false;

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
        }

        DecodeParams decode_params;
        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        {
          DecodeParams& params = decode_params;
          params.q = static_cast<DTypeQ*>(q_d.data_ptr());
          paged_kv_t<DTypeKV, IdType> paged_kv(
              num_kv_heads, page_size_d, HEAD_DIM_VO, batch_size, kv_layout_d,
              static_cast<DTypeKV*>(paged_k_cache_d.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache_d.data_ptr()), kv_cache_strides_d,
              static_cast<IdType*>(paged_kv_indices_d.data_ptr()),
              static_cast<IdType*>(paged_kv_indptr_d.data_ptr()),
              static_cast<IdType*>(paged_kv_last_page_len_d.data_ptr()));
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr_d.data_ptr());
          params.o = static_cast<DTypeO*>(o_d.data_ptr());

          params.lse = maybe_lse_d ? static_cast<float*>(maybe_lse_d->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
          params.q_stride_n = q_stride_n_d;
          params.q_stride_h = q_stride_h_d;
          params.window_left = window_left_d;

          params.request_indices = nullptr;
          params.qo_tile_indices = nullptr;
          params.kv_tile_indices = nullptr;
          params.merge_indptr = nullptr;
          params.o_indptr = nullptr;
          params.kv_chunk_size_ptr = nullptr;
          params.block_valid_mask = nullptr;
          params.total_num_rows = nullptr;
          params.max_total_num_rows = 0;
          params.padded_batch_size = 0;
          params.partition_kv = false;

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
          params.padded_batch_size = plan_info.padded_batch_size;
          params.max_total_num_rows = plan_info.total_num_rows;
          if (plan_info.enable_cuda_graph) {
            params.total_num_rows =
                GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
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
        constexpr size_t CTA_TILE_Q = 16;
        cudaError_t status = flashinfer::PODWithKVCacheTensorDispatched<
            HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION, MASK_MODE_P,
            CTA_TILE_Q, MASK_MODE_D, PrefillAttentionVariant, DecodeAttentionVariant>(
            prefill_params, static_cast<DTypeO*>(tmp_p.data_ptr()), decode_params, tmp_v, tmp_s,
            stream);
        TORCH_CHECK(status == cudaSuccess, "PODWithKVCache kernel launch failed, error: " +
                                               std::string(cudaGetErrorString(status)));
        //});
      });
}
