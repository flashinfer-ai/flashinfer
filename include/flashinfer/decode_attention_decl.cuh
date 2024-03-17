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
#ifndef FLASHINFER_DECODE_ATTENTION_DECL_CUH_
#define FLASHINFER_DECODE_ATTENTION_DECL_CUH_

#include <cuda_runtime.h>

#include <optional>

#include "attention/handler.cuh"
#include "layout.cuh"
#include "page.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"

namespace flashinfer {

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o,
                                              DTypeOut* tmp, uint32_t num_kv_heads,
                                              uint32_t seq_len, float sm_scale, float rope_scale,
                                              float rope_theta, cudaStream_t stream);

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage, QKVLayout kv_layout,
          PosEncodingMode POS_ENCODING_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(
    DTypeIn* q, IdType* q_offset, paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, DTypeOut* tmp, float* lse,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT,
          PosEncodingMode POS_ENCODING_MODE, typename DTypeIn, typename DTypeOut>
cudaError_t BatchDecodeWithPaddedKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o,
                                                   DTypeOut* tmp, float* lse, uint32_t batch_size,
                                                   uint32_t padded_kv_len, uint32_t num_qo_heads,
                                                   float sm_scale, float rope_scale,
                                                   float rope_theta, cudaStream_t stream);

template <typename DTypeIn, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, DTypeOut* tmp,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
                                    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
                                    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                    std::optional<float> maybe_sm_scale = std::nullopt,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_POS_ENCODING_MODE(
              pos_encoding_mode, POS_ENCODING_MODE, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                SingleDecodeWithKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                  POS_ENCODING_MODE>(q, k, v, o, tmp, num_kv_heads,
                                                                     seq_len, sm_scale, rope_scale,
                                                                     rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t BatchDecodeWithPaddedKVCache(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o,
                                         DTypeOut* tmp, float* lse, uint32_t batch_size,
                                         uint32_t padded_kv_len, uint32_t num_qo_heads,
                                         uint32_t num_kv_heads, uint32_t head_dim,
                                         QKVLayout kv_layout = QKVLayout::kNHD,
                                         PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                         std::optional<float> maybe_sm_scale = std::nullopt,
                                         float rope_scale = 1.f, float rope_theta = 1e4,
                                         cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM,
          {DISPATCH_POS_ENCODING_MODE(
              pos_encoding_mode, POS_ENCODING_MODE, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                return BatchDecodeWithPaddedKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                              POS_ENCODING_MODE, DTypeIn, DTypeOut>(
                    q, k, v, o, tmp, lse, batch_size, padded_kv_len, num_qo_heads, sm_scale,
                    rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCache(
    DTypeIn* q, IdType* q_offset, paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, DTypeOut* tmp, float* lse,
    uint32_t num_qo_heads, PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          head_dim, HEAD_DIM, {DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, {
            return BatchDecodeWithPagedKVCacheDispatched<GROUP_SIZE, HEAD_DIM, page_storage,
                                                         kv_layout, POS_ENCODING_MODE, DTypeIn,
                                                         DTypeOut, IdType>(
                q, q_offset, paged_kv, kv_partition_info, o, tmp, lse, sm_scale, rope_scale,
                rope_theta, stream);
          })})});

  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout KV_LAYOUT, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          PosEncodingMode POS_ENCODING_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapperDispatched(
    BatchDecodeHandler* handler, DTypeIn* q, IdType* q_offset,
    paged_kv_t<page_storage, KV_LAYOUT, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream) {
  paged_kv_t<page_storage, KV_LAYOUT, DTypeIn, IdType> new_paged_kv = paged_kv;
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

  return BatchDecodeWithPagedKVCacheDispatched<GROUP_SIZE, HEAD_DIM, page_storage, KV_LAYOUT,
                                               POS_ENCODING_MODE, DTypeIn, DTypeOut, IdType>(
      q, q_offset, new_paged_kv, kv_partition_info, o, tmp, lse, sm_scale, rope_scale, rope_theta,
      stream);
}

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
 * \param pos_encoding_mode The positional encoding mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 * \note This wrapper function should be only called after we call BeginForward function in the
 *   BatchDecodeHandler.
 */
template <PageStorage page_storage, QKVLayout KV_LAYOUT, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler* handler, DTypeIn* q, IdType* q_offset,
    paged_kv_t<page_storage, KV_LAYOUT, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(
          paged_kv.head_dim, HEAD_DIM,
          {DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, {
            return BatchDecodeWithPagedKVCacheWrapperDispatched<page_storage, KV_LAYOUT, GROUP_SIZE,
                                                                HEAD_DIM, POS_ENCODING_MODE,
                                                                DTypeIn, DTypeOut, IdType>(
                handler, q, q_offset, paged_kv, o, lse, sm_scale, rope_scale, rope_theta, stream);
          })})});
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_ATTENTION_DECL_CUH_
