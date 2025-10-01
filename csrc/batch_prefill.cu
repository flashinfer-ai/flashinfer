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

#include "batch_prefill_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchPrefillWithKVCachePlan(
    Tensor float_workspace_buffer, Tensor int_workspace_buffer,
    Tensor page_locked_int_workspace_buffer, Tensor qo_indptr, Tensor kv_indptr, Tensor kv_len_arr,
    int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk, int64_t head_dim_vo,
    bool causal, int64_t window_left, int64_t fixed_split_size, bool disable_split_kv) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * get_element_size(int_workspace_buffer);

  PrefillPlanInfo plan_info;

  cudaSetDevice(float_workspace_buffer->device.device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer->device);
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer->data, float_workspace_size_in_bytes, int_workspace_buffer->data,
      page_locked_int_workspace_buffer->data, int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data), static_cast<IdType*>(kv_indptr->data), total_num_rows,
      batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      enable_cuda_graph,
      /*sizeof_dtype_o=*/2, window_left, fixed_split_size, disable_split_kv, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan prefill with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheRun(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                      Array<int64_t> plan_info_vec, Tensor q, Tensor k, Tensor v,
                                      Tensor qo_indptr, Tensor kv_indptr, Tensor o,
                                      Optional<Tensor> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout, int64_t window_left,
                                      bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q->shape[1];
  int64_t head_dim_qk = q->shape[2];
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k->shape[1] : k->shape[0];
  uint32_t q_stride_n = q->strides[0], q_stride_h = q->strides[1], k_stride_n, k_stride_h,
           v_stride_n, v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k->strides[0];
    k_stride_h = k->strides[1];
    v_stride_n = v->strides[0];
    v_stride_h = v->strides[1];
  } else {
    k_stride_h = k->strides[0];
    k_stride_n = k->strides[1];
    v_stride_h = v->strides[0];
    v_stride_n = v->strides[1];
  }

  if (maybe_lse.has_value()) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse->shape[0], q->shape[0]);
    TVM_FFI_ICHECK_EQ(lse->shape[1], q->shape[1]);
  }

  void* float_buffer_ptr = float_workspace_buffer->data;
  void* int_buffer_ptr = int_workspace_buffer->data;

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  cudaSetDevice(float_workspace_buffer->device.device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer->device);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q = static_cast<DTypeQ*>(q->data);
        params.k = static_cast<DTypeKV*>(k->data);
        params.v = static_cast<DTypeKV*>(v->data);
        params.o = static_cast<DTypeO*>(o->data);
        params.lse = maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value()->data) : nullptr;
        params.q_indptr = static_cast<IdType*>(qo_indptr->data);
        params.kv_indptr = static_cast<IdType*>(kv_indptr->data);
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
              RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchPrefillWithRaggedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}

void BatchPrefillWithPagedKVCacheRun(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                     Array<int64_t> plan_info_vec, Tensor q, Tensor paged_k_cache,
                                     Tensor paged_v_cache, Tensor qo_indptr, Tensor paged_kv_indptr,
                                     Tensor paged_kv_indices, Tensor paged_kv_last_page_len,
                                     Tensor o, Optional<Tensor> maybe_lse, int64_t mask_mode_code,
                                     int64_t layout, int64_t window_left,
                                     bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t batch_size = paged_kv_indptr->shape[0] - 1;
  int64_t num_qo_heads = q->shape[1];
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q->shape[2];
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache->shape[1];
    page_size = paged_k_cache->shape[2];
  } else {
    page_size = paged_k_cache->shape[1];
    num_kv_heads = paged_k_cache->shape[2];
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse->shape[0], q->shape[0]);
    TVM_FFI_ICHECK_EQ(lse->shape[1], q->shape[1]);
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer->data);
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer->data);

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q->strides[0];
  const auto q_stride_h = q->strides[1];

  // get kv_cache_strides
  const int64_t* kv_cache_strides = paged_k_cache.strides().data();
  TVM_FFI_ICHECK_EQ(paged_k_cache->ndim, paged_v_cache->ndim);
  for (int i = 0; i < paged_k_cache->ndim; ++i) {
    TVM_FFI_ICHECK_EQ(paged_k_cache->strides[i], paged_v_cache->strides[i])
        << "k/v strides differs at " << i;
  }

  cudaSetDevice(float_workspace_buffer->device.device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer->device);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q = static_cast<DTypeQ*>(q->data);
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache->data), static_cast<DTypeKV*>(paged_v_cache->data),
            kv_cache_strides, static_cast<IdType*>(paged_kv_indices->data),
            static_cast<IdType*>(paged_kv_indptr->data),
            static_cast<IdType*>(paged_kv_last_page_len->data));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr->data);
        params.o = static_cast<DTypeO*>(o->data);

        params.lse = maybe_lse ? static_cast<float*>(maybe_lse.value()->data) : nullptr;
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
              PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchPrefillWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
