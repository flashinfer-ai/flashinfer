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
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "batch_prefill_config.inc"
#include "pytorch_extension_utils.h"
#include "pytorch_conversion_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
    bool enable_cuda_graph, int64_t head_dim_qk, int64_t head_dim_vo, bool causal,
    int64_t cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  PrefillPlanInfo plan_info;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
      kv_indptr.data_ptr<IdType>(), total_num_rows, batch_size, num_qo_heads, num_kv_heads,
      head_dim_qk, head_dim_vo, page_size, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan prefill with error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    int64_t mask_mode_code, int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS,
    int64_t cuda_stream) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q.size(1);
  int64_t head_dim_qk = q.size(2);
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k.size(1) : k.size(0);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), k_stride_n, k_stride_h, v_stride_n,
           v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k.stride(0);
    k_stride_h = k.stride(1);
    v_stride_n = v.stride(0);
    v_stride_h = v.stride(1);
  } else {
    k_stride_h = k.stride(0);
    k_stride_n = k.stride(1);
    v_stride_h = v.stride(0);
    v_stride_n = v.stride(1);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.kv_indptr = static_cast<IdType*>(kv_indptr.data_ptr());
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.k_stride_n = k_stride_n;
        params.k_stride_h = k_stride_h;
        params.v_stride_n = v_stride_n;
        params.v_stride_h = v_stride_h;
        params.window_left = window_left;

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

        ADDITIONAL_PARAMS_SETTER

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
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              RaggedParams>(params, tmp_v, tmp_s, stream);
        });

        TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithRaggedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
}

void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  auto device = q.device();
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);
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

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());

        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

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

        ADDITIONAL_PARAMS_SETTER

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
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, stream);
        });

        TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
}
