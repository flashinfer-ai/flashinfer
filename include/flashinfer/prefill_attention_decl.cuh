/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_PREFILL_ATTENTION_DECL_CUH_
#define FLASHINFER_PREFILL_ATTENTION_DECL_CUH_

#include <cuda_runtime.h>

#include "attention/handler.cuh"
#include "attention/logits_post_hook.cuh"
#include "attention/mask.cuh"
#include "flashinfer/attention/warp_layout.cuh"
#include "layout.cuh"
#include "page.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"

namespace flashinfer {

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* custom_mask, DTypeOut* o, DTypeOut* tmp, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n, uint32_t kv_stride_h,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream);

template <WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeQ* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* q_indptr, DTypeKV* k, DTypeKV* v, IdType* kv_indptr, uint8_t* custom_mask,
    IdType* qk_indptr, IdType* q_offset, IdType* k_rope_pos_offset, IdType* o_indptr, DTypeOut* o,
    DTypeOut* tmp_v, float* tmp_s, float* lse, IdType* merge_indptr, bool* block_valid_mask,
    IdType* kv_chunk_size_ptr, uint32_t total_num_rows, uint32_t num_qo_heads,
    uint32_t padded_batch_size, uint32_t num_kv_heads, uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h, int32_t window_left, float logits_soft_cap,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream = nullptr);

template <PageStorage page_storage, WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeQ* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* q_indptr, IdType* q_offset, paged_kv_t<page_storage, DTypeKV, IdType> paged_kv,
    uint8_t* custom_mask, IdType* qk_indptr, IdType* o_indptr, DTypeOut* o, DTypeOut* tmp_v,
    float* tmp_s, float* lse, IdType* merge_indptr, bool* block_valid_mask,
    IdType* kv_chunk_size_ptr, uint32_t total_num_rows, uint32_t num_qo_heads,
    uint32_t padded_batch_size, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream);

template <PageStorage PAGE_STORAGE, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* q_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv, uint8_t* custom_mask, IdType* qk_indptr,
    DTypeOut* o, float* lse, uint32_t num_qo_heads, int32_t window_left, float logits_soft_cap,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream) {
  DTypeOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  IdType *request_indices = nullptr, *qo_tile_indices = nullptr, *kv_tile_indices = nullptr,
         *o_indptr = nullptr, *merge_indptr = nullptr, *kv_chunk_size_ptr = nullptr;
  bool* block_valid_mask = nullptr;
  WarpLayout warp_layout;
  uint32_t padded_batch_size = 0U;
  uint32_t total_num_rows = 0U;
  if (handler->IsForwardStarted()) {
    tmp_v = handler->GetTempV<DTypeOut>();
    tmp_s = handler->GetTempS();
    request_indices = handler->GetRequestIndices<IdType>();
    qo_tile_indices = handler->GetQOTileIndices<IdType>();
    kv_tile_indices = handler->GetKVTileIndices<IdType>();
    block_valid_mask = handler->GetBlockValidMask();
    o_indptr = handler->GetOIndptr<IdType>();
    merge_indptr = handler->GetMergeIndptr<IdType>();
    kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
    warp_layout = handler->GetWarpLayout();
    padded_batch_size = handler->GetPaddedBatchSize();
    total_num_rows = handler->GetTotalNumRows();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    return BatchPrefillWithPagedKVCacheDispatched<
        PAGE_STORAGE, WARP_LAYOUT, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE,
        ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
        q, request_indices, qo_tile_indices, kv_tile_indices, q_indptr, q_offset, paged_kv,
        custom_mask, qk_indptr, o_indptr, o, tmp_v, tmp_s, lse, merge_indptr, block_valid_mask,
        kv_chunk_size_ptr, total_num_rows, num_qo_heads, padded_batch_size, window_left,
        logits_soft_cap, sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* q_indptr, DTypeKV* k, DTypeKV* v,
    IdType* kv_indptr, uint8_t* custom_mask, IdType* qk_indptr, IdType* q_offset,
    IdType* k_rope_pos_offset, DTypeOut* o, float* lse, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
    uint32_t kv_stride_h, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream) {
  DTypeOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  IdType *request_indices = nullptr, *qo_tile_indices = nullptr, *kv_tile_indices = nullptr,
         *o_indptr = nullptr, *merge_indptr = nullptr, *kv_chunk_size_ptr = nullptr;
  bool* block_valid_mask = nullptr;
  WarpLayout warp_layout;
  uint32_t padded_batch_size = 0U;
  uint32_t total_num_rows = 0U;
  if (handler->IsForwardStarted()) {
    tmp_v = handler->GetTempV<DTypeOut>();
    tmp_s = handler->GetTempS();
    request_indices = handler->GetRequestIndices<IdType>();
    qo_tile_indices = handler->GetQOTileIndices<IdType>();
    kv_tile_indices = handler->GetKVTileIndices<IdType>();
    block_valid_mask = handler->GetBlockValidMask();
    o_indptr = handler->GetOIndptr<IdType>();
    merge_indptr = handler->GetMergeIndptr<IdType>();
    kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
    warp_layout = handler->GetWarpLayout();
    padded_batch_size = handler->GetPaddedBatchSize();
    total_num_rows = handler->GetTotalNumRows();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithRaggedKVWrapperCache()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    return BatchPrefillWithRaggedKVCacheDispatched<WARP_LAYOUT, HEAD_DIM, LOGITS_POST_HOOK,
                                                   POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION,
                                                   MASK_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
        q, request_indices, qo_tile_indices, kv_tile_indices, q_indptr, k, v, kv_indptr,
        custom_mask, qk_indptr, q_offset, k_rope_pos_offset, o_indptr, o, tmp_v, tmp_s, lse,
        merge_indptr, block_valid_mask, kv_chunk_size_ptr, total_num_rows, num_qo_heads,
        padded_batch_size, num_kv_heads, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h,
        window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_ATTENTION_DECL_CUH_
