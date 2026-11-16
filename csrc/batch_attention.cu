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

#include "batch_attention_config.inc"
#include "tvm_ffi_utils.h"

namespace flashinfer {

using tvm::ffi::Array;
using tvm::ffi::Optional;

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE, bool EQUAL_KV_STRIDES, typename AttentionVariant, typename Params>
cudaError_t BatchPagedAttentionPersistent(const Params params_1, const Params params_2,
                                          const uint32_t num_blks_x, const uint32_t num_blks_y,
                                          const cudaStream_t stream);

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE, bool EQUAL_KV_STRIDES, typename AttentionVariant, typename Params>
cudaError_t BatchPagedAttentionPersistentResidentBlocks(int* blocks_per_sm, size_t* smem_size);
}  // namespace flashinfer

using namespace flashinfer;

Array<int64_t> BatchPagedAttentionPlan(TensorView float_workspace_buffer,
                                       TensorView int_workspace_buffer,
                                       TensorView page_locked_int_workspace_buffer,
                                       TensorView qo_indptr, TensorView kv_indptr,
                                       TensorView kv_len, int64_t batch_size, int64_t num_qo_heads,
                                       int64_t num_kv_heads, int64_t head_dim_o, bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  HolisticPlanInfo<2> plan_info;

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  // The persistent kernel is launched cooperatively, so the plan's grid has to fit what
  // this device can keep resident. That depends on the shared memory this module's kernel
  // requests, which in turn depends on the head dim and dtype the module was built for and
  // on the device's per-SM budget, so it is measured for the actual kernel rather than
  // inferred from the head dim.
  int resident_ctas_per_sm = 0;
  size_t kernel_smem_size = 0;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  // The residency helper is instantiated for the stride variant this module launches. The
  // stride split does not change the shared memory request, so the runtime variant can
  // measure either; it asks for the equal-stride kernel.
#if BATCH_ATTENTION_KV_STRIDE_MODE == BATCH_ATTENTION_KV_STRIDE_MODE_INDEPENDENT
  constexpr bool kResidencyEqualKvStrides = false;
#else
  constexpr bool kResidencyEqualKvStrides = true;
#endif
  cudaError_t status = cudaErrorInvalidValue;
  DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,
                   POS_ENCODING_MODE, AttentionVariant, PersistentParams, [&] {
                     TVM_FFI_ICHECK(head_dim_o == HEAD_DIM_VO)
                         << "BatchAttention planned head_dim_vo=" << head_dim_o
                         << " against a module built for head_dim_vo=" << HEAD_DIM_VO;
                     status = BatchPagedAttentionPersistentResidentBlocks<
                         /*CTA_TILE_Q_1=*/128, /*CTA_TILE_Q_2=*/16, HEAD_DIM_QK, HEAD_DIM_VO,
                         MASK_MODE, kResidencyEqualKvStrides, AttentionVariant, PersistentParams>(
                         &resident_ctas_per_sm, &kernel_smem_size);
                   });
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to query persistent paged attention residency, error: "
      << cudaGetErrorString(status);
  TVM_FFI_ICHECK(resident_ctas_per_sm > 0)
      << "The persistent paged attention kernel for head_dim_qk=" << HEAD_DIM_QK
      << " head_dim_vo=" << HEAD_DIM_VO << " requests " << kernel_smem_size
      << " bytes of shared memory, which does not fit a single CTA on this device's "
      << "shared memory per SM. Use BatchPrefillWithPagedKVCacheWrapper(backend='fa2') "
      << "or BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True) instead.";

  status = TwoStageHolisticPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
      static_cast<IdType*>(kv_indptr.data_ptr()), static_cast<IdType*>(kv_len.data_ptr()),
      batch_size, num_qo_heads, num_kv_heads, head_dim_o, causal,
      static_cast<uint32_t>(resident_ctas_per_sm), stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan persistent paged attention, error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}

void BatchPagedAttentionRun(TensorView float_workspace_buffer, TensorView int_workspace_buffer,
                            Array<int64_t> plan_info_vec, TensorView q, TensorView k_cache,
                            TensorView v_cache, TensorView kv_indices, TensorView o,
                            Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                            int64_t layout_code, int64_t num_qo_heads, int64_t num_kv_heads,
                            int64_t page_size,
                            double v_scale,  // must use double due to pytorch binding
                            double sm_scale,
                            double logits_soft_cap ADDITIONAL_FUNC_PARAMS PROFILER_FUNC_PARAMS) {
  HolisticPlanInfo<2> plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  // NOTE (Yilong): assume both q and o are NHD
  unsigned int q_stride_n = q.stride(0);
  unsigned int q_stride_h = q.stride(1);

  // layout only constraint paged KV
  const QKVLayout kv_layout = static_cast<QKVLayout>(layout_code);
  // Compare the original strides before narrowing to the kernel parameter types.
  [[maybe_unused]] const bool equal_kv_strides = k_cache.stride(0) == v_cache.stride(0) &&
                                                 k_cache.stride(1) == v_cache.stride(1) &&
                                                 k_cache.stride(2) == v_cache.stride(2);
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

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      AttentionVariant, PersistentParams, [&] {
        PersistentParams params[2];
        IdType* len_kv_chunk =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.len_kv_chunk_offset);
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
          params[i].len_kv_chunk = len_kv_chunk + i;

          params[i].final_o = static_cast<DTypeO*>(o.data_ptr());
          params[i].final_lse =
              maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
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
          params[i].k_sf_stride_page = 0;
          params[i].k_sf_stride_h = 0;
          params[i].k_sf_stride_n = 0;
          params[i].v_sf_stride_page = 0;
          params[i].v_sf_stride_h = 0;
          params[i].v_sf_stride_n = 0;

          params[i].sm_scale = sm_scale;
          params[i].v_scale = v_scale;
          params[i].logits_soft_cap = logits_soft_cap;
          // NOTE(Wenxuan) directly using the additional_params_decl from generate_additional_params
          // will be problematic because of the params[i]
          ADDITIONAL_PARAMS_SETTER
          PROFILER_PARAMS_SETTER
        }

        cudaError_t status;
#if BATCH_ATTENTION_KV_STRIDE_MODE == BATCH_ATTENTION_KV_STRIDE_MODE_RUNTIME
        if (equal_kv_strides) {
          status = BatchPagedAttentionPersistent<128, 16, HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, true,
                                                 AttentionVariant>(
              params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
        } else {
          status = BatchPagedAttentionPersistent<128, 16, HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE,
                                                 false, AttentionVariant>(
              params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
        }
#elif BATCH_ATTENTION_KV_STRIDE_MODE == BATCH_ATTENTION_KV_STRIDE_MODE_EQUAL
        TVM_FFI_ICHECK(equal_kv_strides)
            << "The equal-stride BatchAttention primary requires identical K/V data strides "
               "for page, token, and head dimensions; route unequal layouts through the "
               "independent module.";
        status = BatchPagedAttentionPersistent<128, 16, HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, true,
                                               AttentionVariant>(
            params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
#elif BATCH_ATTENTION_KV_STRIDE_MODE == BATCH_ATTENTION_KV_STRIDE_MODE_INDEPENDENT
        status = BatchPagedAttentionPersistent<128, 16, HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, false,
                                               AttentionVariant>(
            params[0], params[1], plan_info.num_blks_x, plan_info.num_blks_y, stream);
#else
#error "Unsupported BATCH_ATTENTION_KV_STRIDE_MODE"
#endif
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Failed to run persistent paged attention, error: " << cudaGetErrorString(status);
        return true;
      });
}
