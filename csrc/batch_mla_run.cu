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
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchMLAPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                               at::Tensor plan_info_vec, at::Tensor q_nope, at::Tensor q_pe,
                               at::Tensor ckv_cache, at::Tensor kpe_cache, at::Tensor kv_indices,
                               at::Tensor o, std::optional<at::Tensor> maybe_lse,
                               int64_t mask_mode_code, int64_t num_heads, int64_t page_size,
                               double sm_scale) {
  // q_nope: [n, num_heads, head_dim_ckv]
  // q_pe: [n, num_heads, head_dim_kpe]
  // ckv_cache: [num_pages, page_size, head_dim_ckv]
  // kpe_cache: [num_pages, page_size, head_dim_kpe]
  MLAPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  auto device = q_nope.device();

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  auto q_scalar_type = q_nope.scalar_type();
  auto kv_scalar_type = ckv_cache.scalar_type();

  unsigned int q_nope_stride_n = q_nope.stride(0);
  unsigned int q_nope_stride_h = q_nope.stride(1);
  unsigned int q_pe_stride_n = q_pe.stride(0);
  unsigned int q_pe_stride_h = q_pe.stride(1);
  unsigned int ckv_stride_page = ckv_cache.stride(0);
  unsigned int ckv_stride_n = ckv_cache.stride(1);
  unsigned int kpe_stride_page = kpe_cache.stride(0);
  unsigned int kpe_stride_n = kpe_cache.stride(1);
  unsigned int o_stride_n = o.stride(0);
  unsigned int o_stride_h = o.stride(1);

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE, Params, [&] {
        Params params;

        params.q_nope = static_cast<DTypeQ*>(q_nope.data_ptr());
        params.q_pe = static_cast<DTypeQ*>(q_pe.data_ptr());
        params.ckv = static_cast<DTypeKV*>(ckv_cache.data_ptr());
        params.kpe = static_cast<DTypeKV*>(kpe_cache.data_ptr());

        params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.partial_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.partial_indptr_offset);
        params.kv_indices = static_cast<IdType*>(kv_indices.data_ptr());
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
        params.final_o = static_cast<DTypeO*>(o.data_ptr());
        params.final_lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.partial_o =
            GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.partial_o_offset);
        params.partial_lse =
            GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.partial_lse_offset);

        params.num_heads = uint_fastdiv(num_heads);
        params.block_size = uint_fastdiv(page_size);

        params.q_nope_stride_n = q_nope_stride_n;
        params.q_nope_stride_h = q_nope_stride_h;
        params.q_pe_stride_n = q_pe_stride_n;
        params.q_pe_stride_h = q_pe_stride_h;
        params.ckv_stride_page = ckv_stride_page;
        params.ckv_stride_n = ckv_stride_n;
        params.kpe_stride_page = kpe_stride_page;
        params.kpe_stride_n = kpe_stride_n;
        params.o_stride_n = o_stride_n;
        params.o_stride_h = o_stride_h;

        params.sm_scale = sm_scale;

        cudaError_t status = mla::BatchMLAPagedAttention<MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE>(
            params, plan_info.num_blks_x, plan_info.num_blks_y, stream);

        TORCH_CHECK(status == cudaSuccess,
                    "Failed to run MLA, error: ", cudaGetErrorString(status));
      });
}
