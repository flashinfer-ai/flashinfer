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

#include "pod_config.inc"
#include "tvm_ffi_utils.h"

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
using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> PODWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr_p, TensorView kv_indptr_p,
    int64_t total_num_rows_p, int64_t batch_size_p, TensorView qo_indptr_d, TensorView kv_indptr_d,
    int64_t total_num_rows_d, int64_t batch_size_d, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_vo, int64_t page_size, bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  PODPlanInfo plan_info;

  cudaSetDevice(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  cudaError_t status = PODPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr_p.data_ptr()),
      static_cast<IdType*>(kv_indptr_p.data_ptr()), total_num_rows_p, batch_size_p,
      static_cast<IdType*>(qo_indptr_d.data_ptr()), static_cast<IdType*>(kv_indptr_d.data_ptr()),
      total_num_rows_d, batch_size_d, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
      page_size, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan prefill with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}

void PODWithKVCacheTensorRun(
    // Shared params
    TensorView float_workspace_buffer_d, TensorView int_workspace_buffer_d,
    Array<int64_t> plan_info_vec, TensorView paged_k_cache, TensorView paged_v_cache,
    TensorView qo_indptr, TensorView paged_kv_indptr, TensorView paged_kv_indices,
    TensorView paged_kv_last_page_len, TensorView o, Optional<TensorView> maybe_lse, int64_t layout,
    // Prefill params
    TensorView q_p, int64_t mask_mode_code_p, int64_t window_left_p,
    Optional<TensorView> maybe_custom_mask_p, Optional<TensorView> maybe_alibi_slopes_p,
    double logits_soft_cap_p, double sm_scale_p, double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    TensorView q_d, int64_t mask_mode_code_d, int64_t window_left_d,
    Optional<TensorView> maybe_custom_mask_d, Optional<TensorView> maybe_mask_indptr_d,
    Optional<TensorView> maybe_alibi_slopes_d, double logits_soft_cap_d, double sm_scale_d,
    double rope_rcp_scale_d, double rope_rcp_theta_d, bool enable_pdl) {
  PODPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  uint32_t batch_size = paged_kv_indptr.size(0) - 1;
  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer_d.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer_d.data_ptr());

  // Prefill setup
  uint32_t head_dim_qk = q_p.size(2);
  uint32_t qo_len, num_qo_heads_p;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q_p.size(0) + q_d.size(0);
  num_qo_heads_p = q_p.size(1);
  uint32_t q_stride_n_p = q_p.stride(0), q_stride_h_p = q_p.stride(1);
  if (maybe_lse.has_value()) {
    const auto& lse = maybe_lse.value();
    TVM_FFI_ICHECK_EQ(lse.size(0), qo_len);
    TVM_FFI_ICHECK_EQ(lse.size(1), num_qo_heads_p);
  }

  const MaskMode mask_mode_p = static_cast<MaskMode>(mask_mode_code_p);

  // Decode setup (Tensor decode = batched prefill)
  uint32_t num_qo_heads = q_d.size(1);
  TVM_FFI_ICHECK_EQ(num_qo_heads_p, num_qo_heads)
      << "POD currently requires same # Query heads for prefill and decode";

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
  TVM_FFI_ICHECK_EQ(num_kv_heads, num_kv_heads_d)
      << "POD currently requires same # KV heads for prefill and decode; Prefill: " << num_kv_heads
      << ", Decode: " << num_kv_heads_d;

  const MaskMode mask_mode_d = static_cast<MaskMode>(mask_mode_code_d);

  // get q_stride_n and q_stride_h
  const auto q_stride_n_d = q_d.stride(0);
  const auto q_stride_h_d = q_d.stride(1);

  cudaSetDevice(float_workspace_buffer_d.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer_d.device());

  DISPATCH_context(
      MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK, USE_SLIDING_WINDOW_P,
      USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, [&] {
        // Compute kv_cache_strides from tensor strides
        // paged_kv_t expects [stride_page, stride_n, stride_h] where:
        // - stride_page is stride(0)
        // - stride_n and stride_h depend on layout
        int64_t kv_strides[3];
        kv_strides[0] = paged_k_cache.stride(0);  // stride_page
        if (kv_layout == QKVLayout::kHND) {
          kv_strides[1] = paged_k_cache.stride(1);  // stride_h
          kv_strides[2] = paged_k_cache.stride(2);  // stride_n
        } else {
          kv_strides[1] = paged_k_cache.stride(1);  // stride_n
          kv_strides[2] = paged_k_cache.stride(2);  // stride_h
        }
        TVM_FFI_ICHECK(paged_k_cache.stride(0) == paged_v_cache.stride(0) &&
                       paged_k_cache.stride(1) == paged_v_cache.stride(1) &&
                       paged_k_cache.stride(2) == paged_v_cache.stride(2))
            << "k/v strides must be identical";

        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        IdType* q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());

        // debug indices
        PrefillParams prefill_params;
        {
          // Make params a reference to prefill_params to set values
          PrefillParams& params = prefill_params;
          params.q = static_cast<DTypeQ*>(q_p.data_ptr());
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());

          params.o = static_cast<DTypeO*>(o.data_ptr());
          params.lse =
              maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
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
          params.maybe_custom_mask =
              maybe_custom_mask_p.has_value()
                  ? static_cast<uint8_t*>(maybe_custom_mask_p.value().data_ptr())
                  : nullptr;
          params.maybe_alibi_slopes =
              maybe_alibi_slopes_p.has_value()
                  ? static_cast<float*>(maybe_alibi_slopes_p.value().data_ptr())
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
          params.lse =
              maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
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

          params.maybe_mask_indptr =
              maybe_mask_indptr_d.has_value()
                  ? static_cast<int32_t*>(maybe_mask_indptr_d.value().data_ptr())
                  : nullptr;
          params.maybe_alibi_slopes =
              maybe_alibi_slopes_d.has_value()
                  ? static_cast<float*>(maybe_alibi_slopes_d.value().data_ptr())
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
        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q_p, CTA_TILE_Q_P, {
          TVM_FFI_ICHECK(plan_info.cta_tile_q_d == 16)
              << "Decode tile size should be 16 for POD. Check planner.";
          constexpr size_t CTA_TILE_Q_D = 16;
          cudaError_t status = flashinfer::PODWithKVCacheTensorDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION, MASK_MODE_P,
              CTA_TILE_Q_P, CTA_TILE_Q_D, MASK_MODE_D, PrefillAttentionVariant,
              DecodeAttentionVariant>(prefill_params, decode_params, tmp_v, tmp_s, enable_pdl,
                                      stream);
          TVM_FFI_ICHECK(status == cudaSuccess)
              << "PODWithKVCache kernel launch failed, error: " << cudaGetErrorString(status);
        });
      });
}
