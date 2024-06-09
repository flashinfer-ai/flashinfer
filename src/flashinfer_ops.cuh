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
#include <flashinfer/decode_attention_decl.cuh>
#include <flashinfer/prefill_attention_decl.cuh>
#include <optional>

#include "utils.h"

namespace flashinfer {

/*!
 * \brief FlashAttention prefill CUDA function for a single request.
 * \tparam DTypeIn The data type of input
 * \tparam DTypeOut The data type of output
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary storage (only used for cooperative kernel).
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param kv_layout The layout of input and output.
 * \param pos_encoding_mode The positional encoding mode.
 * \param allow_fp16_qk_reduction Whether to allow accumulating q*k^T with fp16.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, float* tmp,
                                     float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                     uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
                                     bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
                                     PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                     bool allow_fp16_qk_reduction = false,
                                     std::optional<float> maybe_sm_scale = std::nullopt,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  DISPATCH_allow_fp16_qk_reduction(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_group_size(
          group_size, GROUP_SIZE,
          {DISPATCH_mask_mode(
              mask_mode, MASK_MODE,
              {DISPATCH_head_dim(head_dim, HEAD_DIM,
                                 {DISPATCH_pos_encoding_mode(
                                     pos_encoding_mode, POS_ENCODING_MODE,
                                     {DISPATCH_kv_layout(kv_layout, KV_LAYOUT, {
                                       return SinglePrefillWithKVCacheDispatched<
                                           GROUP_SIZE, HEAD_DIM, KV_LAYOUT, POS_ENCODING_MODE,
                                           ALLOW_FP16_QK_REDUCTION, MASK_MODE>(
                                           q, k, v, /*custom_mask=*/nullptr, o, tmp, lse,
                                           num_kv_heads, qo_len, kv_len, sm_scale, rope_scale,
                                           rope_theta, stream);
                                     })})})})})});
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, IdType* q_offset, IdType* k_rope_pos_offset, DTypeOut* o, float* lse,
    const uint32_t batch_size, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t head_dim, bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    const float rope_scale = 1.f, const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  DISPATCH_kv_layout(
      kv_layout, KV_LAYOUT,
      {DISPATCH_group_size(
          num_qo_heads / num_kv_heads, GROUP_SIZE,
          {DISPATCH_head_dim(
              head_dim, HEAD_DIM,
              {DISPATCH_mask_mode(
                  mask_mode, MASK_MODE,
                  {DISPATCH_pos_encoding_mode(
                      pos_encoding_mode, pos_encoding_mode,
                      {DISPATCH_allow_fp16_qk_reduction(
                          allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                            return BatchPrefillWithRaggedKVCacheWrapperDispatched<
                                GROUP_SIZE, HEAD_DIM, KV_LAYOUT, pos_encoding_mode,
                                ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeIn, DTypeOut, IdType>(
                                handler, q, qo_indptr, k, v, kv_indptr, /*custom_mask=*/nullptr,
                                /*qk_indptr=*/nullptr, q_offset, k_rope_pos_offset, o, lse,
                                batch_size, num_kv_heads, sm_scale, rope_scale, rope_theta, stream);
                          })})})})})});
  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, bool causal = true,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  DISPATCH_group_size(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM,
          {DISPATCH_mask_mode(mask_mode, MASK_MODE,
                              {DISPATCH_pos_encoding_mode(
                                  pos_encoding_mode, pos_encoding_mode,
                                  {DISPATCH_allow_fp16_qk_reduction(
                                      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
                                      {DISPATCH_page_size(paged_kv.page_size, PAGE_SIZE, {
                                        return BatchPrefillWithPagedKVCacheWrapperDispatched<
                                            page_storage, kv_layout, PAGE_SIZE, GROUP_SIZE,
                                            HEAD_DIM, pos_encoding_mode, ALLOW_FP16_QK_REDUCTION,
                                            MASK_MODE, DTypeIn, DTypeOut, IdType>(
                                            handler, q, qo_indptr, q_offset, paged_kv,
                                            /*custom_mask=*/nullptr, /*qk_indptr=*/nullptr, o, lse,
                                            sm_scale, rope_scale, rope_theta, stream);
                                      })})})})})});
  return cudaSuccess;
}

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

  DISPATCH_group_size(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE, {DISPATCH_kv_layout(kv_layout, KV_LAYOUT, {
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

  DISPATCH_group_size(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE, {DISPATCH_kv_layout(kv_layout, KV_LAYOUT, {
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

  DISPATCH_group_size(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            return BatchDecodeWithPagedKVCacheDispatched<GROUP_SIZE, HEAD_DIM, page_storage,
                                                         kv_layout, POS_ENCODING_MODE, DTypeIn,
                                                         DTypeOut, IdType>(
                q, q_offset, paged_kv, kv_partition_info, o, tmp,
                (float*)tmp + batch_size * num_qo_heads * head_dim,
                lse, std::nullopt, sm_scale,
                rope_scale, rope_theta, stream);
          })})});

  return cudaSuccess;
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

  DISPATCH_group_size(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {DISPATCH_head_dim(
          paged_kv.head_dim, HEAD_DIM,
          {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            return BatchDecodeWithPagedKVCacheWrapperDispatched<page_storage, KV_LAYOUT, GROUP_SIZE,
                                                                HEAD_DIM, POS_ENCODING_MODE,
                                                                DTypeIn, DTypeOut, IdType>(
                handler, q, q_offset, paged_kv, o, lse, sm_scale, rope_scale, rope_theta, stream);
          })})});
  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeHandlerBeginForward(BatchDecodeHandler* handler, void* buffer,
                                           size_t workspace_size_in_bytes, IdType* indptr,
                                           IdType* last_page_len, uint32_t batch_size,
                                           uint32_t num_qo_heads, uint32_t num_kv_heads,
                                           uint32_t head_dim, uint32_t page_size,
                                           PosEncodingMode pos_encoding_mode) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }
  DISPATCH_group_size(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    DISPATCH_head_dim(head_dim, HEAD_DIM, {
      DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        return handler->BeginForwardDispatched<GROUP_SIZE, HEAD_DIM, page_storage, kv_layout,
                                               POS_ENCODING_MODE, DTypeIn, DTypeOut, IdType>(
            buffer, workspace_size_in_bytes, indptr, last_page_len, batch_size, num_qo_heads,
            page_size);
      });
    });
  });
}

}  // namespace flashinfer
