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

#include "attention/handler.cuh"
#include "attention/logits_post_hook.cuh"
#include "layout.cuh"
#include "page.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"

namespace flashinfer {

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode pos_encoding_mode, typename DTypeQ, typename DTypeKV,
          typename DTypeOut>
cudaError_t SingleDecodeWithKVCacheDispatched(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o,
                                              DTypeOut* tmp, uint32_t num_kv_heads,
                                              uint32_t seq_len, float sm_scale, float rope_scale,
                                              float rope_theta, cudaStream_t stream);

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE,
          typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(
    DTypeQ* q, IdType* q_offset, paged_kv_t<page_storage, KV_LAYOUT, DTypeKV, IdType> paged_kv,
    kv_partition_info_t<IdType> kv_partition_info, DTypeOut* o, DTypeOut* tmp_v, float* tmp_s,
    float* lse, bool* block_valid_mask, uint32_t padded_batch_size, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream);

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut>
cudaError_t BatchDecodeWithPaddedKVCacheDispatched(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o,
                                                   DTypeOut* tmp, float* lse, uint32_t batch_size,
                                                   uint32_t padded_kv_len, uint32_t num_qo_heads,
                                                   float sm_scale, float rope_scale,
                                                   float rope_theta, cudaStream_t stream);

template <PageStorage PAGE_STORAGE, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT, PosEncodingMode POS_ENCODING_MODE,
          typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapperDispatched(
    BatchDecodeHandler* handler, DTypeQ* q, IdType* q_offset,
    paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeKV, IdType> paged_kv, DTypeOut* o, float* lse,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream) {
  paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeKV, IdType> new_paged_kv = paged_kv;
  kv_partition_info_t<IdType> kv_partition_info;
  DTypeOut* tmp_v = handler->GetTempV<DTypeOut>();
  float* tmp_s = handler->GetTempS<float>();

  if (handler->IsForwardStarted()) {
    if (tmp_v != nullptr) {
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

  return BatchDecodeWithPagedKVCacheDispatched<GROUP_SIZE, HEAD_DIM, PAGE_STORAGE, LOGITS_POST_HOOK,
                                               KV_LAYOUT, POS_ENCODING_MODE, DTypeQ, DTypeKV,
                                               DTypeOut, IdType>(
      q, q_offset, new_paged_kv, kv_partition_info, o, tmp_v, tmp_s, lse,
      handler->GetBlockValidMask(), handler->GetPaddedBatchSize(), sm_scale, rope_scale, rope_theta,
      stream);
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_ATTENTION_DECL_CUH_
