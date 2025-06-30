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
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "batch_attention_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t BatchPagedAttentionPersistent(const Params params_1, const Params params_2,
                                          const uint32_t num_blks_x, const uint32_t num_blks_y,
                                          const cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

at::Tensor BatchPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                   at::Tensor int_workspace_buffer,
                                   at::Tensor page_locked_int_workspace_buffer,
                                   at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor kv_len,
                                   int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads,
                                   int64_t head_dim_o, bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  HolisticPlanInfo<2> plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  cudaError_t status = TwoStageHolisticPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
      kv_indptr.data_ptr<IdType>(), kv_len.data_ptr<IdType>(), batch_size, num_qo_heads,
      num_kv_heads, head_dim_o, causal, stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan persistent paged attention, error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor plan_info_vec, at::Tensor q, at::Tensor k_cache,
                            at::Tensor v_cache, at::Tensor kv_indices, at::Tensor o,
                            std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                            int64_t layout_code, int64_t num_qo_heads, int64_t num_kv_heads,
                            int64_t page_size,
                            double sm_scale ADDITIONAL_FUNC_PARAMS PROFILER_FUNC_PARAMS) {
  HolisticPlanInfo<2> plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  auto device = q.device();

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k_cache.scalar_type();

  // NOTE (Yilong): assume both q and o are NHD
  unsigned int q_stride_n = q.stride(0);
  unsigned int q_stride_h = q.stride(1);

  // layout only constraint paged KV
  const QKVLayout kv_layout = static_cast<QKVLayout>(layout_code);
  unsigned int k_stride_page = k_cache.stride(0);
  unsigned int v_stride_page = v_cache.stride(0);
  unsigned int k_stride_n, k_stride_h, v_stride_n, v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_h = k_cache.stride(2);
    k_stride_n = k_cache.stride(1);
    v_stride_h = v_cache.stride(2);
    v_stride_n = v_cache.stride(1);
  } else {
    k_stride_h = k_cache.stride(1);
    k_stride_n = k_cache.stride(2);
    v_stride_h = v_cache.stride(1);
    v_stride_n = v_cache.stride(2);
  }

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      AttentionVariant, PersistentParams, [&] {
        PersistentParams params[2];

        for (int i = 0; i < 2; i++) {
          params[i].q = static_cast<DTypeQ*>(q.data_ptr());
          params[i].k = static_cast<DTypeKV*>(k_cache.data_ptr());
          params[i].v = static_cast<DTypeKV*>(v_cache.data_ptr());

          params[i].q_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].q_indptr_offset);
          params[i].kv_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].kv_indptr_offset);
          params[i].partial_indptr = GetPtrFromBaseOffset<IdType>(
              int_buffer_ptr, plan_info.tasks[i].partial_indptr_offset);
          params[i].kv_indices = static_cast<int*>(kv_indices.data_ptr());
          params[i].q_len =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].q_len_offset);
          params[i].kv_len =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].kv_len_offset);
          params[i].q_start =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].q_start_offset);
          params[i].kv_start =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].kv_start_offset);
          params[i].kv_end =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].kv_end_offset);
          params[i].kv_head_idx_arr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].kv_head_idx_offset);
          params[i].work_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].work_indptr_offset);
          params[i].len_kv_chunk =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.tasks[i].len_kv_chunk_offset);

          params[i].final_o = static_cast<DTypeO*>(o.data_ptr());
          params[i].final_lse =
              maybe_lse.has_value() ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
          params[i].partial_o =
              GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.partial_o_offset);
          params[i].partial_lse =
              GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.partial_lse_offset);

          // for state reduction
          params[i].merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          params[i].merge_o_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_o_indices_offset);
          params[i].num_packed_qo_len =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.num_qo_len_offset);

          params[i].num_kv_heads = num_kv_heads;
          params[i].gqa_group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
          params[i].page_size = uint_fastdiv(page_size);

          params[i].q_stride_n = q_stride_n;
          params[i].q_stride_h = q_stride_h;
          params[i].k_stride_page = k_stride_page;
          params[i].k_stride_h = k_stride_h;
          params[i].k_stride_n = k_stride_n;
          params[i].v_stride_page = v_stride_page;
          params[i].v_stride_h = v_stride_h;
          params[i].v_stride_n = v_stride_n;

          params[i].sm_scale = sm_scale;

          ADDITIONAL_PARAMS_SETTER
          PROFILER_PARAMS_SETTER
        }

        cudaError_t status = BatchPagedAttentionPersistent<128, 16, HEAD_DIM_QK, HEAD_DIM_VO,
                                                           MASK_MODE, AttentionVariant>(
            params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
        TORCH_CHECK(status == cudaSuccess, "Failed to run persistent paged attention, error: ",
                    cudaGetErrorString(status));
        return true;
      });
}
