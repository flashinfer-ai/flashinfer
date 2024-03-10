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
#include "layout.cuh"
#include "page.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"

namespace flashinfer {

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL,
          typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o,
                                               float* tmp, float* lse, uint32_t num_kv_heads,
                                               uint32_t qo_len, uint32_t kv_len, float sm_scale,
                                               float rope_scale, float rope_theta,
                                               cudaStream_t stream);

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
  DISPATCH_ALLOW_FP16_QK_REDUCTION(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_GQA_GROUP_SIZE(
          group_size, GROUP_SIZE,
          {DISPATCH_CAUSAL(
              causal, CAUSAL,
              {DISPATCH_HEAD_DIM(
                  head_dim, HEAD_DIM,
                  {DISPATCH_POS_ENCODING_MODE(
                      pos_encoding_mode, pos_encoding_mode, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                        SinglePrefillWithKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                           pos_encoding_mode,
                                                           ALLOW_FP16_QK_REDUCTION, CAUSAL>(
                            q, k, v, o, tmp, lse, num_kv_heads, qo_len, kv_len, sm_scale,
                            rope_scale, rope_theta, stream);
                      })})})})})});
  return cudaSuccess;
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, IdType* q_offset, IdType* k_rope_pos_offset, DTypeOut* o, float* lse,
    uint32_t batch_size, uint32_t num_kv_heads, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream);

template <PageStorage page_storage, QKVLayout kv_layout, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCache(
    DTypeIn* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* tmp,
    float* lse, uint32_t num_qo_heads, bool causal = true,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));

  uint32_t num_frags_x, num_qo_tiles;
  std::vector<IdType> request_indices_h, tile_indices_h;
  std::tie(num_frags_x, num_qo_tiles, request_indices_h, tile_indices_h) =
      split_qo_indptr(qo_indptr, batch_size, group_size, head_dim, stream);

  IdType* request_indices_d;
  IdType* tile_indices_d;

  FLASHINFER_CUDA_CALL(
      cudaMallocAsync(&request_indices_d, sizeof(IdType) * request_indices_h.size(), stream));
  FLASHINFER_CUDA_CALL(
      cudaMallocAsync(&tile_indices_d, sizeof(IdType) * tile_indices_h.size(), stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_d, request_indices_h.data(),
                                       sizeof(IdType) * request_indices_h.size(),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(tile_indices_d, tile_indices_h.data(),
                                       sizeof(IdType) * tile_indices_h.size(),
                                       cudaMemcpyHostToDevice, stream));

  DISPATCH_NUM_FRAGS_X(
      num_frags_x, NUM_FRAGS_X,
      {DISPATCH_ALLOW_FP16_QK_REDUCTION(
          allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
          {DISPATCH_GQA_GROUP_SIZE(
              group_size, GROUP_SIZE,
              {DISPATCH_CAUSAL(
                  causal, CAUSAL,
                  {DISPATCH_HEAD_DIM(
                      head_dim, HEAD_DIM,
                      {DISPATCH_POS_ENCODING_MODE(
                          pos_encoding_mode, pos_encoding_mode,
                          {DISPATCH_PAGE_SIZE(paged_kv.page_size, PAGE_SIZE,
                                              {
                                                return BatchPrefillWithPagedKVCacheDispatched<
                                                    page_storage, kv_layout, NUM_FRAGS_X, PAGE_SIZE,
                                                    GROUP_SIZE, HEAD_DIM, pos_encoding_mode,
                                                    ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn,
                                                    DTypeOut, IdType>(
                                                    q, request_indices_d, tile_indices_d, qo_indptr,
                                                    q_offset, paged_kv, o, tmp, lse, num_qo_tiles,
                                                    sm_scale, rope_scale, rope_theta, stream);
                                              })

                          })})})})})});

  FLASHINFER_CUDA_CALL(cudaFreeAsync(request_indices_d, stream));
  FLASHINFER_CUDA_CALL(cudaFreeAsync(tile_indices_d, stream));
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_ATTENTION_DECL_CUH_