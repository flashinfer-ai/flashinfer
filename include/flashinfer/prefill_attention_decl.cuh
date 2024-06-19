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
#include "layout.cuh"
#include "page.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"

namespace flashinfer {

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT,
          PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v,
                                               uint8_t* custom_mask, DTypeOut* o, DTypeOut* tmp,
                                               float* lse, uint32_t num_qo_heads,
                                               uint32_t num_kv_heads, uint32_t qo_len,
                                               uint32_t kv_len, float sm_scale, float rope_scale,
                                               float rope_theta, cudaStream_t stream);

template <uint32_t num_frags_x, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* kv_lens, IdType* q_indptr, DTypeIn* k, DTypeIn* v, IdType* kv_indptr,
    uint8_t* custom_mask, IdType* qk_indptr, IdType* q_offset, IdType* k_rope_pos_offset,
    IdType* o_indptr, DTypeOut* o, DTypeOut* tmp_v, float* tmp_s, float* lse, IdType* merge_indptr,
    bool* block_valid_mask, const uint32_t batch_size, const uint32_t num_qo_heads,
    const uint32_t kv_chunk_size, const uint32_t padded_batch_size, const uint32_t num_kv_heads,
    const float sm_scale, const float rope_scale, const float rope_theta,
    cudaStream_t stream = nullptr);

template <PageStorage page_storage, uint32_t num_frags_x, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* kv_lens, IdType* q_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, uint8_t* custom_mask,
    IdType* qk_indptr, IdType* o_indptr, DTypeOut* o, DTypeOut* tmp_v, float* tmp_s, float* lse,
    IdType* merge_indptr, bool* block_valid_mask, uint32_t num_qo_heads, uint32_t kv_chunk_size,
    uint32_t padded_batch_size, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream);

template <PageStorage PAGE_STORAGE, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* q_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeIn, IdType> paged_kv, uint8_t* custom_mask,
    IdType* qk_indptr, IdType* o_indptr, DTypeOut* o, float* lse, uint32_t num_qo_heads,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream) {
  DTypeOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  IdType *kv_lens = nullptr, request_indices = nullptr, qo_tile_indices = nullptr,
         *kv_tile_indices = nullptr, *o_merge_indptr, *merge_indptr = nullptr;
  bool* block_valid_mask = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t padded_batch_size = 0U;
  uint32_t kv_chunk_size = 0U;
  if (handler->IsForwardStarted()) {
    tmp_v = handler->GetTempV<DTypeOut>();
    tmp_s = handler->GetTempS();
    kv_lens = handler->GetKVLen<IdType>();
    request_indices = handler->GetRequestIndices<IdType>();
    qo_tile_indices = handler->GetQOTileIndices<IdType>();
    kv_tile_indices = handler->GetKVTileIndices<IdType>();
    block_valid_mask = handler->GetBlockValidMask();
    o_indptr = handler->GetOIndptr<IdType>();
    merge_indptr = handler->GetMergeIndptr<IdType>();
    kv_chunk_size = handler->GetKVChunkSize();
    num_frags_x = handler->GetNumFragsX();
    padded_batch_size = handler->GetPaddedBatchSize();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithPagedKVCacheDispatched<
        PAGE_STORAGE, NUM_FRAGS_X, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT, POS_ENCODING_MODE,
        ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeIn, DTypeOut, IdType>(
        q, request_indices, qo_tile_indices, kv_tile_indices, kv_lens, q_indptr, q_offset, paged_kv,
        custom_mask, qk_indptr, o_indptr, o, tmp_v, tmp_s, lse, merge_indptr, block_valid_mask,
        num_qo_heads, kv_chunk_size, padded_batch_size, sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT,
          PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* q_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, uint8_t* custom_mask, IdType* qk_indptr, IdType* q_offset,
    IdType* k_rope_pos_offset, IdType* o_indptr, DTypeOut* o, float* lse, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads, float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream) {
  DTypeOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  IdType *kv_lens = nullptr, request_indices = nullptr, qo_tile_indices = nullptr,
         *kv_tile_indices = nullptr, *o_merge_indptr, *merge_indptr = nullptr;
  bool* block_valid_mask = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t padded_batch_size = 0U;
  uint32_t kv_chunk_size = 0U;
  if (handler->IsForwardStarted()) {
    tmp_v = handler->GetTempV<DTypeOut>();
    tmp_s = handler->GetTempS();
    kv_lens = handler->GetKVLen<IdType>();
    request_indices = handler->GetRequestIndices<IdType>();
    qo_tile_indices = handler->GetQOTileIndices<IdType>();
    kv_tile_indices = handler->GetKVTileIndices<IdType>();
    block_valid_mask = handler->GetBlockValidMask();
    o_indptr = handler->GetOIndptr<IdType>();
    merge_indptr = handler->GetMergeIndptr<IdType>();
    kv_chunk_size = handler->GetKVChunkSize();
    num_frags_x = handler->GetNumFragsX();
    padded_batch_size = handler->GetPaddedBatchSize();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithRaggedKVWrapperCache()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithRaggedKVCacheDispatched<
        NUM_FRAGS_X, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT, POS_ENCODING_MODE,
        ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeIn, DTypeOut, IdType>(
        q, request_indices, qo_tile_indices, kv_tile_indices, kv_lens, q_indptr, k, v, kv_indptr,
        custom_mask, qk_indptr, q_offset, k_rope_pos_offset, o_indptr, o, tmp_v, tmp_s, lse,
        merge_indptr, block_valid_mask, batch_size, num_qo_heads, kv_chunk_size, padded_batch_size,
        num_kv_heads, sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_ATTENTION_DECL_CUH_
