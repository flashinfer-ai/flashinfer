/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/fastdiv.cuh>
#include <optional>

#include "batch_mla_config.inc"
#include "tvm_binding_utils.h"

using namespace flashinfer;

void BatchMLAPagedAttentionRun(DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
                               IntTuple plan_info_vec, DLTensor* q, DLTensor* kv_cache,
                               DLTensor* kv_indices, DLTensor* o, DLTensor* lse,
                               int64_t mask_mode_code, int64_t num_heads, int64_t page_size,
                               double sm_scale, TVMStreamHandle cuda_stream) {
  // q: [n, num_heads, head_dim_ckv + head_dim_kpe]
  // kv_cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe]
  MLAPlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(kv_cache->dtype);

  // get q_strides
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  unsigned int q_stride_n = q_strides[0];
  unsigned int q_stride_h = q_strides[1];

  int64_t kv_cache_strides[3] = {
      kv_cache->strides ? kv_cache->strides[0] : kv_cache->shape[1] * kv_cache->shape[2],  //
      kv_cache->strides ? kv_cache->strides[1] : kv_cache->shape[2],                       //
      kv_cache->strides ? kv_cache->strides[2] : 1};
  unsigned int kv_stride_page = kv_cache_strides[0];
  unsigned int kv_stride_n = kv_cache_strides[1];

  int64_t pe_offset = HEAD_DIM_CKV;

  int64_t o_strides[3] = {o->strides ? o->strides[0] : o->shape[1] * o->shape[2],  //
                          o->strides ? o->strides[1] : o->shape[2],                //
                          o->strides ? o->strides[2] : 1};
  unsigned int o_stride_n = o_strides[0];
  unsigned int o_stride_h = o_strides[1];

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE, Params, [&] {
        Params params;

        params.q_nope = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.q_pe = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ) + pe_offset;
        params.ckv =
            static_cast<DTypeKV*>(kv_cache->data) + kv_cache->byte_offset / sizeof(DTypeKV);
        params.kpe = static_cast<DTypeKV*>(kv_cache->data) +
                     kv_cache->byte_offset / sizeof(DTypeKV) + pe_offset;

        params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.partial_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.partial_indptr_offset);
        params.kv_indices =
            static_cast<IdType*>(kv_indices->data) + kv_indices->byte_offset / sizeof(IdType);
        params.q_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_len_offset);
        params.kv_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.q_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_start_offset);
        params.kv_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_start_offset);
        params.kv_end = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_end_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.merge_packed_offset_start = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_packed_offset_start_offset);
        params.merge_packed_offset_end =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_packed_offset_end_offset);
        params.merge_partial_packed_offset_start = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_partial_packed_offset_start_offset);
        params.merge_partial_packed_offset_end = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_partial_packed_offset_end_offset);
        params.merge_partial_stride =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_partial_stride_offset);
        params.final_o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.final_lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.partial_o =
            GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.partial_o_offset);
        params.partial_lse =
            GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.partial_lse_offset);

        params.num_heads = uint_fastdiv(num_heads);
        params.block_size = uint_fastdiv(page_size);

        params.q_nope_stride_n = q_stride_n;
        params.q_nope_stride_h = q_stride_h;
        params.q_pe_stride_n = q_stride_n;
        params.q_pe_stride_h = q_stride_h;
        params.ckv_stride_page = kv_stride_page;
        params.ckv_stride_n = kv_stride_n;
        params.kpe_stride_page = kv_stride_page;
        params.kpe_stride_n = kv_stride_n;
        params.o_stride_n = o_stride_n;
        params.o_stride_h = o_stride_h;

        params.sm_scale = sm_scale;

        cudaError_t status = mla::BatchMLAPagedAttention<MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE>(
            params, plan_info.num_blks_x, plan_info.num_blks_y, stream);

        CHECK(status == cudaSuccess) << "Failed to run MLA, error: " << cudaGetErrorString(status);
      });
}
