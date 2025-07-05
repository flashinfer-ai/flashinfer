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
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <optional>

#include "batch_decode_config.inc"
#include "tvm_binding_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

IntTuple BatchDecodeWithPagedKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t pos_encoding_mode_code, int64_t window_left, int64_t head_dim_qk, int64_t head_dim_vo,
    DataType q_scalar_type, DataType kv_scalar_type, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  DecodePlanInfo plan_info;

  CHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
              GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
          cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
              static_cast<char*>(float_workspace_buffer->data) +
                  float_workspace_buffer->byte_offset,
              float_workspace_size_in_bytes,
              static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
              static_cast<char*>(page_locked_int_workspace_buffer->data) +
                  page_locked_int_workspace_buffer->byte_offset,
              int_workspace_size_in_bytes, plan_info,
              static_cast<IdType*>(indptr->data) + indptr->byte_offset / sizeof(IdType), batch_size,
              num_qo_heads, page_size, enable_cuda_graph,
              /*stream=*/stream, work_estimation_func);

          CHECK(status == cudaSuccess)
              << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
          return true;
        });
      });

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

void BatchDecodeWithPagedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* paged_kv_indptr, DLTensor* paged_kv_indices,
    DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset, DLTensor* paged_kv_rope_pos_offset,
    DLTensor* o, DLTensor* lse, int64_t pos_encoding_mode_code, int64_t kv_layout_code,
    int64_t window_left ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream) {
  DecodePlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  int64_t batch_size = q->shape[0];
  int64_t num_qo_heads = q->shape[1];
  int64_t num_kv_heads, page_size;

  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_kv_cache->shape[2];
    page_size = paged_kv_cache->shape[3];
  } else {
    page_size = paged_kv_cache->shape[2];
    num_kv_heads = paged_kv_cache->shape[3];
  }
  uint32_t head_dim_qk = q->shape[2];
  uint32_t head_dim_vo = paged_kv_cache->shape[4];

  CHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  // get q_scalar_type and kv_scalar_type
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
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_QK, batch_size, kv_layout,
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
            static_cast<IdType*>(paged_kv_rope_pos_offset->data) +
                paged_kv_rope_pos_offset->byte_offset / sizeof(IdType));

        Params params;
        params.q = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.paged_kv = paged_kv;
        params.o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.padded_batch_size = 0;
        params.num_qo_heads = num_qo_heads;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.decode_maybe_q_rope_offset =
            static_cast<IdType*>(q_rope_offset->data) + q_rope_offset->byte_offset / sizeof(IdType);
        params.window_left = window_left;
        params.request_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;

        cudaError_t status =
            flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM_QK, POS_ENCODING_MODE,
                                                              AttentionVariant>(
                params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
        CHECK(status == cudaSuccess)
            << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
