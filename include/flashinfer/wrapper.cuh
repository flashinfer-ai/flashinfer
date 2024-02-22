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
#ifndef FLASHINFER_WRAPPER_CUH_
#define FLASHINFER_WRAPPER_CUH_

#include "decode.cuh"
#include "handler.cuh"
#include "prefill.cuh"

namespace flashinfer {

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the temporary buffer
 *   for cooperative kernels.
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam kv_layout The layout of last 3 dimensions in KV-Cache
 * \tparam DTypeIn The data type of input tensor.
 * \tparam DTypeOut The data type of output tensor.
 * \tparam IdType The data type of index tensor.
 * \param handler The handler for the batch decode forward request.
 * \param q The input tensor.
 * \param paged_kv The paged key-value tensor.
 * \param o The output tensor.
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of heads.
 * \param rotary_mode The rotary mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 * \note This wrapper function should be only called after we call BeginForward function in the
 *   BatchDecodeHandler.
 */
template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler* handler, DTypeIn* q, IdType* q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> new_paged_kv = paged_kv;
  kv_partition_info_t<IdType> kv_partition_info;
  DTypeOut* tmp = handler->GetTempFloatBuffer<DTypeOut>();
  if (handler->IsForwardStarted()) {
    if (tmp != nullptr) {
      // create auxiliary information for cooperative kernels
      new_paged_kv.batch_size = handler->GetBatchSizeAfterPartition();
      new_paged_kv.indptr = handler->GetNewIndPtr<IdType>();
      new_paged_kv.last_page_len = handler->GetNewLastPageLen<IdType>();
      kv_partition_info.batch_size_before_partition = handler->GetBatchSizeBeforePartition();
      kv_partition_info.chunk_indptr = handler->GetChunkIndPtr<IdType>();
      kv_partition_info.batch_idx_map = handler->GetBatchIdxMap<IdType>();
      kv_partition_info.chunk_start_pos = handler->GetChunkStartPos<IdType>();
      kv_partition_info.seq_lens_before_partition = handler->GetSeqLengthsBeforePartition<IdType>();
    }
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchDecodeHandler's BeginForward() before calling "
               "BatchDecodeWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }
  return BatchDecodeWithPagedKVCache<page_storage, kv_layout, DTypeIn, DTypeOut, IdType>(
      q, q_rope_position, new_paged_kv, kv_partition_info, o, tmp, lse, num_qo_heads, rotary_mode,
      rope_scale, rope_theta, stream);
}

template <PageStorage page_storage, QKVLayout kv_layout, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          RotaryMode ROTARY_MODE, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
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

  DISPATCH_NUM_FRAGS_X(
      num_frags_x, NUM_FRAGS_X, {DISPATCH_PAGE_SIZE(paged_kv.page_size, PAGE_SIZE, {
        if constexpr (PAGE_SIZE == 0) {
          return BatchPrefillWithPagedKVCacheFallbackDispatched<
              page_storage, kv_layout, NUM_FRAGS_X, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
              ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
              q, request_indices, tile_indices, qo_indptr, q_rope_position, paged_kv, o, tmp, lse,
              num_qo_tiles, rope_scale, rope_theta, stream);
        } else {
          return BatchPrefillWithPagedKVCacheDispatched<
              page_storage, kv_layout, NUM_FRAGS_X, PAGE_SIZE, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
              ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
              q, request_indices, tile_indices, qo_indptr, q_rope_position, paged_kv, o, tmp, lse,
              num_qo_tiles, rope_scale, rope_theta, stream);
        }
      })});
  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, bool causal = true, RotaryMode rotary_mode = RotaryMode::kNone,
    bool allow_fp16_qk_reduction = false, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_CAUSAL(
              causal, CAUSAL,
              {DISPATCH_ROTARY_MODE(
                  rotary_mode, ROTARY_MODE,
                  {DISPATCH_ALLOW_FP16_QK_REDUCTION(
                      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                        return BatchPrefillWithPagedKVCacheWrapperDispatched<
                            page_storage, kv_layout, GROUP_SIZE, HEAD_DIM, ROTARY_MODE,
                            ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
                            handler, q, qo_indptr, q_rope_position, paged_kv, o, lse, rope_scale,
                            rope_theta, stream);
                      })})})})});
  return cudaSuccess;
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT, RotaryMode ROTARY_MODE,
          bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, IdType* q_rope_position, IdType* k_rope_pos_offset, DTypeOut* o, float* lse,
    const uint32_t batch_size, const uint32_t num_kv_heads, const float rope_scale,
    const float rope_theta, cudaStream_t stream) {
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
    return BatchPrefillWithRaggedKVCacheDispatched<NUM_FRAGS_X, GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                   ROTARY_MODE, ALLOW_FP16_QK_REDUCTION, CAUSAL,
                                                   DTypeIn, DTypeOut, IdType>(
        q, request_indices, tile_indices, qo_indptr, k, v, kv_indptr, q_rope_position,
        k_rope_pos_offset, o, tmp, lse, batch_size, num_qo_tiles, num_kv_heads, rope_scale,
        rope_theta, stream);
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, DTypeOut* o, float* lse, const uint32_t batch_size,
    const uint32_t num_qo_heads, const uint32_t num_kv_heads, const uint32_t head_dim,
    bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    RotaryMode rotary_mode = RotaryMode::kNone, bool allow_fp16_qk_reduction = false,
    const float rope_scale = 1.f, const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  DISPATCH_LAYOUT(
      kv_layout, KV_LAYOUT,
      {DISPATCH_GQA_GROUP_SIZE(
          num_qo_heads / num_kv_heads, GROUP_SIZE,
          {DISPATCH_HEAD_DIM(
              head_dim, HEAD_DIM,
              {DISPATCH_CAUSAL(
                  causal, CAUSAL,
                  {DISPATCH_ROTARY_MODE(
                      rotary_mode, ROTARY_MODE,
                      {DISPATCH_ALLOW_FP16_QK_REDUCTION(
                          allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                            return BatchPrefillWithRaggedKVCacheWrapperDispatched<
                                GROUP_SIZE, HEAD_DIM, KV_LAYOUT, ROTARY_MODE,
                                ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
                                handler, q, qo_indptr, k, v, kv_indptr, /*q_rope_position=*/nullptr,
                                /*k_rope_pos_offset=*/nullptr, o, lse, batch_size, num_kv_heads,
                                rope_scale, rope_theta, stream);
                          })})})})})});
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_HANDLER_CUH_
