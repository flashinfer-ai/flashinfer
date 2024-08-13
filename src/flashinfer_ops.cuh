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

#include "flashinfer/attention/logits_post_hook.cuh"
#include "flashinfer/attention/mask.cuh"
#include "utils.h"

namespace flashinfer {

template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheCustomMask(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, uint8_t* custom_mask, DTypeOut* o, DTypeOut* tmp,
    float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_allow_fp16_qk_reduction(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            return SinglePrefillWithKVCacheDispatched<HEAD_DIM, LogitsPostHook::kNone,
                                                      POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION,
                                                      MaskMode::kCustom>(
                q, k, v, custom_mask, o, tmp, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h,
                /*window_left=*/-1,
                /*logits_soft_cap*/ 0.f, sm_scale, rope_scale, rope_theta, stream);
          })})});
  return cudaSuccess;
}

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
template <typename DTypeQ, typename DTypeKV, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o, DTypeOut* tmp,
                                     float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                     uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
                                     bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
                                     PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                     bool allow_fp16_qk_reduction = false,
                                     std::optional<float> maybe_sm_scale = std::nullopt,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_allow_fp16_qk_reduction(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_head_dim(
              head_dim, HEAD_DIM,
              {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                return SinglePrefillWithKVCacheDispatched<HEAD_DIM, LogitsPostHook::kNone,
                                                          POS_ENCODING_MODE,
                                                          ALLOW_FP16_QK_REDUCTION, MASK_MODE>(
                    q, k, v, /*custom_mask=*/nullptr, o, tmp, lse, num_qo_heads, num_kv_heads,
                    qo_len, kv_len, qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h,
                    /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, DTypeKV* k, DTypeKV* v,
    IdType* kv_indptr, IdType* q_offset, IdType* k_rope_pos_offset, DTypeOut* o, float* lse,
    const uint32_t batch_size, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t head_dim, bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    const float rope_scale = 1.f, const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, 0, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, pos_encoding_mode,
              {DISPATCH_allow_fp16_qk_reduction(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                return BatchPrefillWithRaggedKVCacheWrapperDispatched<
                    HEAD_DIM, LogitsPostHook::kNone, pos_encoding_mode, ALLOW_FP16_QK_REDUCTION,
                    MASK_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
                    handler, q, qo_indptr, k, v, kv_indptr, /*custom_mask=*/nullptr,
                    /*qk_indptr=*/nullptr, q_offset, k_rope_pos_offset, o, lse, num_qo_heads,
                    num_kv_heads, qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h,
                    /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, bool causal = true,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE,
              {DISPATCH_allow_fp16_qk_reduction(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, {
                return BatchPrefillWithPagedKVCacheWrapperDispatched<
                    PAGE_STORAGE, HEAD_DIM, LogitsPostHook::kNone, POS_ENCODING_MODE,
                    ALLOW_FP16_QK_REDUCTION, MASK_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
                    handler, q, qo_indptr, q_offset, paged_kv,
                    /*custom_mask=*/nullptr,
                    /*qk_indptr=*/nullptr, o, lse, num_qo_heads, /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
              })})})});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeOut>
cudaError_t SingleDecodeWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o, DTypeOut* tmp,
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

  DISPATCH_head_dim(
      head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        SingleDecodeWithKVCacheDispatched<HEAD_DIM, LogitsPostHook::kNone, POS_ENCODING_MODE>(
            q, k, v, o, tmp, num_qo_heads, num_kv_heads, seq_len, kv_layout,
            /*window_left=*/-1,
            /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
      })});
  return cudaSuccess;
}

template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheNoSplitKV(
    DTypeQ* q, IdType* q_offset, paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, float* lse, uint32_t num_qo_heads,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
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

  DISPATCH_head_dim(
      head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        return BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PAGE_STORAGE, LogitsPostHook::kNone,
                                                     POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeOut,
                                                     IdType>(
            q, q_offset, paged_kv, kv_partition_info, o, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, lse,
            /*block_valid_mask=*/nullptr, /*padded_batch_size=*/paged_kv.batch_size, num_qo_heads,
            /*window_left=*/-1,
            /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
      })});

  return cudaSuccess;
}

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the temporary buffer
 *   for cooperative kernels.
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeQ The data type of query tensor.
 * \tparam DTypeKV The data type of key-value tensor.
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
template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler* handler, DTypeQ* q, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, DTypeKV, IdType> paged_kv, DTypeOut* o, float* lse,
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

  DISPATCH_head_dim(paged_kv.head_dim, HEAD_DIM,
                    {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                      return BatchDecodeWithPagedKVCacheWrapperDispatched<
                          PAGE_STORAGE, HEAD_DIM, LogitsPostHook::kNone, POS_ENCODING_MODE, DTypeQ,
                          DTypeKV, DTypeOut, IdType>(
                          handler, q, q_offset, paged_kv, o, lse, num_qo_heads,
                          /*window_left=*/-1,
                          /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta, stream);
                    })});
  return cudaSuccess;
}

template <PageStorage PAGE_STORAGE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchDecodeHandlerBeginForward(BatchDecodeHandler* handler, void* float_buffer,
                                           size_t float_workspace_size_in_bytes, void* int_buffer,
                                           size_t int_workspace_size_in_bytes, IdType* indptr_h,
                                           IdType* last_page_len_h, uint32_t batch_size,
                                           uint32_t num_qo_heads, uint32_t num_kv_heads,
                                           uint32_t head_dim, uint32_t page_size,
                                           PosEncodingMode pos_encoding_mode) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }
  DISPATCH_head_dim(head_dim, HEAD_DIM, {
    DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
      return handler->BeginForwardDispatched<HEAD_DIM, PAGE_STORAGE, LogitsPostHook::kNone,
                                             POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeOut, IdType>(
          float_buffer, float_workspace_size_in_bytes, int_buffer, int_workspace_size_in_bytes,
          indptr_h, last_page_len_h, batch_size, num_qo_heads, num_kv_heads, page_size);
    });
  });
}

}  // namespace flashinfer
