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
                                               float* custom_mask, DTypeOut* o, float* tmp,
                                               float* lse, uint32_t num_qo_heads,
                                               uint32_t num_kv_heads, uint32_t qo_len,
                                               uint32_t kv_len, float sm_scale, float rope_scale,
                                               float rope_theta, cudaStream_t stream);

template <uint32_t NUM_FRAGS_X, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, DTypeIn* k,
    DTypeIn* v, IdType* kv_indptr, float* custom_mask, IdType* qk_indptr, IdType* q_offset,
    IdType* k_rope_pos_offset, DTypeOut* o, float* tmp, float* lse, uint32_t batch_size,
    uint32_t num_qo_tiles, uint32_t num_qo_heads, uint32_t num_kv_heads, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream = nullptr);

template <PageStorage PAGE_STORAGE, uint32_t NUM_FRAGS_X, uint32_t PAGE_SIZE, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeIn, IdType> paged_kv, float* custom_mask,
    IdType* qk_indptr, DTypeOut* o, float* tmp, float* lse, uint32_t num_qo_tiles,
    uint32_t num_qo_heads, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template <PageStorage PAGE_STORAGE, uint32_t PAGE_SIZE, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeIn, IdType> paged_kv, float* custom_mask,
    IdType* qk_indptr, DTypeOut* o, float* lse, uint32_t num_qo_heads, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream) {
  float* tmp = nullptr;
  IdType* request_indices = nullptr;
  IdType* tile_indices = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t num_qo_tiles = 0U;
  if (handler->IsForwardStarted()) {
    request_indices = handler->GetRequestIndices<IdType>();
    tile_indices = handler->GetTileIndices<IdType>();
    num_frags_x = handler->GetNumFragsX();
    num_qo_tiles = handler->GetNumQOTiles();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithPagedKVCacheDispatched<
        PAGE_STORAGE, NUM_FRAGS_X, PAGE_SIZE, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT,
        POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeIn, DTypeOut, IdType>(
        q, request_indices, tile_indices, qo_indptr, q_offset, paged_kv, custom_mask, qk_indptr, o,
        tmp, lse, num_qo_heads, num_qo_tiles, sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT,
          PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, float* custom_mask, IdType* qk_indptr, IdType* q_offset,
    IdType* k_rope_pos_offset, DTypeOut* o, float* lse, uint32_t batch_size, uint32_t num_qo_heads,
    uint32_t num_kv_heads, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream) {
  float* tmp = nullptr;
  IdType* request_indices = nullptr;
  IdType* tile_indices = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t num_qo_tiles = 0U;
  if (handler->IsForwardStarted()) {
    request_indices = handler->GetRequestIndices<IdType>();
    tile_indices = handler->GetTileIndices<IdType>();
    num_frags_x = handler->GetNumFragsX();
    num_qo_tiles = handler->GetNumQOTiles();
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
        q, request_indices, tile_indices, qo_indptr, k, v, kv_indptr, custom_mask, qk_indptr,
        q_offset, k_rope_pos_offset, o, tmp, lse, batch_size, num_qo_heads, num_qo_tiles,
        num_kv_heads, sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_ATTENTION_DECL_CUH_
