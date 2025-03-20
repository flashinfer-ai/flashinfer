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
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "pod_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {
template <uint32_t CTA_TILE_Q_P, uint32_t CTA_TILE_Q_D, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename PrefillParams, typename DecodeParams>
cudaError_t PODWithPagedKVCacheDispatched(PrefillParams prefill_params, DecodeParams decode_params,
                                          typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                          cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor PODWithPagedKVCachePlan(at::Tensor float_workspace_buffer,
                                   at::Tensor int_workspace_buffer,
                                   at::Tensor page_locked_int_workspace_buffer,
                                   at::Tensor qo_indptr, at::Tensor kv_indptr,
                                   at::Tensor kv_last_page_len, int64_t total_num_rows,
                                   int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
                                   int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
                                   int64_t head_dim_vo, bool causal, int64_t cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  PoDPlanInfo plan_info;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status =
      PoDPlan<IdType>(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
                      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
                      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
                      kv_indptr.data_ptr<IdType>(), kv_last_page_len.data_ptr<IdType>(),
                      total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk,
                      head_dim_vo, page_size, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan PoD Attention with error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void PODWithPagedKVCacheRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
                            at::Tensor paged_v_cache, at::Tensor paged_kv_indices, at::Tensor o,
                            std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                            int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS,
                            int64_t cuda_stream) {
  PoDPlanInfo pod_plan_info;
  pod_plan_info.FromVector(tensor_to_vec(plan_info_vec));

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto device = q.device();

  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());
  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params_p, params_d;
        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        auto _configureParams = [&](PagedParams& params, const PrefillPlanInfo& plan_info,
                                    int64_t batch_size) {
          params.q = static_cast<DTypeQ*>(q.data_ptr());
          params.o = static_cast<DTypeO*>(o.data_ptr());
          params.q_start_ptr = nullptr;
          params.qo_len_ptr = nullptr;
          params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
          params.q_stride_n = q_stride_n;
          params.q_stride_h = q_stride_h;
          params.window_left = window_left;

          params.request_indices = nullptr;
          params.qo_tile_indices = nullptr;
          params.kv_tile_indices = nullptr;
          params.merge_indptr = nullptr;
          params.o_indptr = nullptr;
          params.kv_chunk_size_ptr = nullptr;
          params.block_valid_mask = nullptr;
          params.total_num_rows = nullptr;
          params.max_total_num_rows = 0;
          params.padded_batch_size = 0;
          params.partition_kv = false;

          ADDITIONAL_PARAMS_SETTER

          paged_kv_t<DTypeKV, IdType> paged_kv(
              num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
              static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
              static_cast<IdType*>(paged_kv_indices.data_ptr()), nullptr, nullptr);

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
          params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
          params.q_start_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_start_ptr_offset);
          params.qo_len_ptr =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.q_len_ptr_offset);
          paged_kv.indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_start_ptr_offset);
          paged_kv.len_ptr =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.kv_len_ptr_offset);
          paged_kv.last_page_len =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_last_page_offset);
          params.paged_kv = paged_kv;

          if (plan_info.split_kv) {
            params.partition_kv = true;  // used in prefill kernel
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
            }
          }
          params.padded_batch_size = plan_info.padded_batch_size;
          params.max_total_num_rows = plan_info.total_num_rows;
          if (plan_info.enable_cuda_graph) {
            params.total_num_rows =
                GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
          }
        };

        _configureParams(params_p, pod_plan_info.plan_info_p, pod_plan_info.batch_size_vec_p);
        _configureParams(params_d, pod_plan_info.plan_info_d, pod_plan_info.batch_size_vec_d);

        if (pod_plan_info.plan_info_p.split_kv || pod_plan_info.plan_info_d.split_kv) {
          params_p.merge_indptr = GetPtrFromBaseOffset<IdType>(
              int_buffer_ptr, pod_plan_info.plan_info_p.merge_indptr_offset);
          params_d.merge_indptr = GetPtrFromBaseOffset<IdType>(
              int_buffer_ptr, pod_plan_info.plan_info_d.merge_indptr_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(pod_plan_info.plan_info_p.cta_tile_q, CTA_TILE_Q_P, {
          DISPATCH_CTA_TILE_Q(pod_plan_info.plan_info_d.cta_tile_q, CTA_TILE_Q_D, {
            status = flashinfer::PODWithPagedKVCacheDispatched<
                CTA_TILE_Q_P, CTA_TILE_Q_D, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
                USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant, PagedParams, PagedParams>(
                params_p, params_d, tmp_v, tmp_s, stream);
          });
        });

        TORCH_CHECK(status == cudaSuccess, "PODWithPagedKVCacheDispatched failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
}
