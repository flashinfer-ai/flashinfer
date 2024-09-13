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

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode pos_encoding_mode,
          typename DTypeQ, typename DTypeKV, typename DTypeOut>
cudaError_t SingleDecodeWithKVCacheDispatched(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeOut* o, DTypeOut* tmp, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t seq_len, QKVLayout kv_layout, int32_t window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                  typename AttentionVariant::DTypeOut* tmp_v,
                                                  float* tmp_s, cudaStream_t stream);

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheWrapperDispatched(BatchDecodeHandler* handler,
                                                         typename AttentionVariant::ParamsT params,
                                                         cudaStream_t stream) {
  using DTypeOut = typename AttentionVariant::DTypeO;
  using IdType = typename AttentionVariant::IdType;
  auto new_paged_kv = params.paged_kv;
  DTypeOut* tmp_v = handler->GetTempV<DTypeOut>();
  float* tmp_s = handler->GetTempS();

  if (tmp_v != nullptr) {
    // create auxiliary information for cooperative kernels
    new_paged_kv.batch_size = handler->GetBatchSizeAfterPartition();
    new_paged_kv.indptr = handler->GetNewIndPtr<IdType>();
    new_paged_kv.last_page_len = handler->GetNewLastPageLen<IdType>();
    params.paged_kv = new_paged_kv;
    params.kv_partition_info.batch_size_before_partition = handler->GetBatchSizeBeforePartition();
    params.kv_partition_info.chunk_indptr = handler->GetChunkIndPtr<IdType>();
    params.kv_partition_info.batch_idx_map = handler->GetBatchIdxMap<IdType>();
    params.kv_partition_info.chunk_start_pos = handler->GetChunkStartPos<IdType>();
    params.kv_partition_info.seq_lens_before_partition =
        handler->GetSeqLengthsBeforePartition<IdType>();
  }
  params.block_valid_mask = handler->GetBlockValidMask();
  params.padded_batch_size = handler->GetPaddedBatchSize();

  return BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>(
      params, tmp_v, tmp_s, stream);
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_ATTENTION_DECL_CUH_
