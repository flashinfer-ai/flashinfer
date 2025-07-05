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
#include "tvm_binding_utils.h"

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

IntTuple BatchPrefillWithKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    IntTuple kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  PrefillPlanInfo plan_info;

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = PrefillPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      enable_cuda_graph,
      /*sizeof_dtype_o=*/2, stream);

  CHECK(status == cudaSuccess) << "Failed to plan prefill with error: "
                               << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

void BatchPrefillWithRaggedKVCacheRun(DLTensor* float_workspace_buffer,
                                      DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
                                      DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* qo_indptr,
                                      DLTensor* kv_indptr, DLTensor* q_rope_offset,
                                      DLTensor* k_rope_offset, DLTensor* o, DLTensor* lse,
                                      int64_t mask_mode_code, int64_t pos_encoding_mode_code,
                                      int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS,
                                      TVMStreamHandle cuda_stream) {
  PrefillPlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q->shape[1];
  int64_t head_dim_qk = q->shape[2];
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k->shape[1] : k->shape[0];
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  int64_t k_strides[3] = {k->strides ? k->strides[0] : k->shape[1] * k->shape[2],  //
                          k->strides ? k->strides[1] : k->shape[2],                //
                          k->strides ? k->strides[2] : 1};
  int64_t v_strides[3] = {v->strides ? v->strides[0] : v->shape[1] * v->shape[2],  //
                          v->strides ? v->strides[1] : v->shape[2],                //
                          v->strides ? v->strides[2] : 1};
  uint32_t q_stride_n = q_strides[0], q_stride_h = q_strides[1];
  uint32_t k_stride_n, k_stride_h, v_stride_n, v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k_strides[0];
    k_stride_h = k_strides[1];
    v_stride_n = v_strides[0];
    v_stride_h = v_strides[1];
  } else {
    k_stride_h = k_strides[0];
    k_stride_n = k_strides[1];
    v_stride_h = v_strides[0];
    v_stride_n = v_strides[1];
  }

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(k->dtype);

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.k = static_cast<DTypeKV*>(k->data) + k->byte_offset / sizeof(DTypeKV);
        params.v = static_cast<DTypeKV*>(v->data) + v->byte_offset / sizeof(DTypeKV);
        params.o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.q_indptr =
            static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType);
        params.kv_indptr =
            static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType);
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
        params.maybe_q_rope_offset = q_rope_offset != nullptr
                                         ? static_cast<IdType*>(q_rope_offset->data) +
                                               q_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.maybe_k_rope_offset = k_rope_offset != nullptr
                                         ? static_cast<IdType*>(k_rope_offset->data) +
                                               k_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
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
              RaggedParams>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
        });

        CHECK(status == cudaSuccess)
            << "BatchPrefillWithRaggedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}

void BatchPrefillWithPagedKVCacheRun(DLTensor* float_workspace_buffer,
                                     DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
                                     DLTensor* q, DLTensor* paged_kv_cache, DLTensor* qo_indptr,
                                     DLTensor* paged_kv_indptr, DLTensor* paged_kv_indices,
                                     DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset,
                                     DLTensor* paged_kv_rope_pos_offset, DLTensor* o, DLTensor* lse,
                                     int64_t mask_mode_code, int64_t pos_encoding_mode_code,
                                     int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS,
                                     TVMStreamHandle cuda_stream) {
  PrefillPlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t batch_size = paged_kv_indptr->shape[0] - 1;
  int64_t num_qo_heads = q->shape[1];
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q->shape[2];
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_kv_cache->shape[2];
    page_size = paged_kv_cache->shape[3];
  } else {
    page_size = paged_kv_cache->shape[2];
    num_kv_heads = paged_kv_cache->shape[3];
  }

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);
  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(paged_kv_cache->dtype);

  // get q_stride_n and q_stride_h
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  const auto q_stride_n = q_strides[0];
  const auto q_stride_h = q_strides[1];

  // get kv_cache_strides
  int64_t kv_cache_strides[4] = {
      paged_kv_cache->strides ? paged_kv_cache->strides[0]
                              : paged_kv_cache->shape[1] * paged_kv_cache->shape[2] *
                                    paged_kv_cache->shape[3] * paged_kv_cache->shape[4],
      paged_kv_cache->strides ? paged_kv_cache->strides[2]
                              : paged_kv_cache->shape[3] * paged_kv_cache->shape[4],    //
      paged_kv_cache->strides ? paged_kv_cache->strides[3] : paged_kv_cache->shape[4],  //
      paged_kv_cache->strides ? paged_kv_cache->strides[4] : 1};
  int64_t v_offset = paged_kv_cache->strides ? paged_kv_cache->strides[1]
                                             : paged_kv_cache->shape[2] * paged_kv_cache->shape[3] *
                                                   paged_kv_cache->shape[4];

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_kv_cache->data) +
                paged_kv_cache->byte_offset / sizeof(DTypeKV),
            static_cast<DTypeKV*>(paged_kv_cache->data) +
                paged_kv_cache->byte_offset / sizeof(DTypeKV) + v_offset,
            kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices->data) +
                paged_kv_indices->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_indptr->data) +
                paged_kv_indptr->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_last_page_len->data) +
                paged_kv_last_page_len->byte_offset / sizeof(IdType),
            paged_kv_rope_pos_offset != nullptr
                ? static_cast<IdType*>(paged_kv_rope_pos_offset->data) +
                      paged_kv_rope_pos_offset->byte_offset / sizeof(IdType)
                : nullptr);
        params.paged_kv = paged_kv;
        params.q_indptr =
            static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType);
        params.o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);

        params.lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.maybe_q_rope_offset = q_rope_offset != nullptr
                                         ? static_cast<IdType*>(q_rope_offset->data) +
                                               q_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
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
              PagedParams>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
        });

        CHECK(status == cudaSuccess)
            << "BatchPrefillWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
