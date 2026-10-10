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

#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>

#include "batch_prefill_sm90_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params& params, bool enable_pdl,
                                                    cudaStream_t stream);

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                   cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace flashinfer {

// The merge kernels in cascade.cuh vectorize over CUDA half types, not cutlass ones.
template <typename T>
struct SplitKVMergeDType {
  using type = T;
};
template <>
struct SplitKVMergeDType<cutlass::half_t> {
  using type = half;
};
template <>
struct SplitKVMergeDType<cutlass::bfloat16_t> {
  using type = nv_bfloat16;
};

// Wire the split-KV plan into the kernel params and check the output layout the merge expects.
template <typename Params>
void SetSplitKVParams(Params& params, const PrefillPlanSM90Info& plan_info, void* float_buffer_ptr,
                      void* int_buffer_ptr, ffi::TensorView o, int64_t head_dim_vo) {
  params.num_kv_chunks = plan_info.num_kv_chunks;
  if (plan_info.num_kv_chunks > 1) {
    using IdType = typename Params::IdType;
    using DTypeO = typename Params::DTypeO;
    params.kv_chunk_indices =
        GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_indices_offset);
    params.tmp_o = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.tmp_o_offset);
    params.tmp_lse = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.tmp_lse_offset);
    TVM_FFI_ICHECK(o.stride(1) == head_dim_vo && o.stride(0) == o.size(1) * head_dim_vo)
        << "split-KV prefill requires a contiguous output tensor";
  }
}

// Merge the per-chunk partial outputs written by a split-KV launch into o / lse.
template <typename Params>
cudaError_t MergeSplitKVStates(const Params& params, int64_t head_dim_vo, cudaStream_t stream) {
  if (params.num_kv_chunks <= 1) {
    return cudaSuccess;
  }
  using DTypeO = typename SplitKVMergeDType<typename Params::DTypeO>::type;
  return MergeStates(
      reinterpret_cast<DTypeO*>(params.tmp_o), params.tmp_lse,
      reinterpret_cast<DTypeO*>(params.o_ptr), params.lse_ptr,
      static_cast<uint32_t>(params.num_kv_chunks), static_cast<uint32_t>(params.nnz_qo),
      static_cast<uint32_t>(params.num_qo_heads), static_cast<uint32_t>(head_dim_vo), stream);
}

}  // namespace flashinfer

Array<int64_t> BatchPrefillWithKVCacheSM90Plan(
    ffi::TensorView float_workspace_buffer, ffi::TensorView int_workspace_buffer,
    ffi::TensorView page_locked_int_workspace_buffer, ffi::TensorView qo_indptr,
    ffi::TensorView kv_indptr, ffi::TensorView kv_len_arr, int64_t total_num_rows,
    int64_t batch_size, int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
    bool enable_cuda_graph, int64_t head_dim_qk, int64_t head_dim_vo, bool causal,
    int64_t window_left, bool disable_split_kv) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  flashinfer::PrefillPlanSM90Info plan_info;

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  cudaError_t status = PrefillSM90Plan(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
      static_cast<IdType*>(kv_indptr.data_ptr()), static_cast<IdType*>(kv_len_arr.data_ptr()),
      total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      causal, enable_cuda_graph,
      /*sizeof_dtype_o=*/sizeof(DTypeO), stream,
      /*allow_split_kv=*/!disable_split_kv && window_left < 0);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "PrefillSM90Plan failed with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheSM90Run(
    ffi::TensorView float_workspace_buffer, ffi::TensorView int_workspace_buffer,
    Array<int64_t> plan_info_vec, ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView qo_indptr, ffi::TensorView kv_indptr, ffi::TensorView o,
    Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse.size(0), q.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  int64_t head_dim_qk = q.size(2);
  int64_t head_dim_vo = v.size(2);

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_swa = window_left != -1;

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
        params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
        params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
        params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
        params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.q_stride_n = q.stride(0);
        params.q_stride_h = q.stride(1);
        params.o_stride_n = o.stride(0);
        params.o_stride_h = o.stride(1);
        if (kv_layout == QKVLayout::kNHD) {
          params.k_stride_n = k.stride(0);
          params.k_stride_h = k.stride(1);
          params.v_stride_n = v.stride(0);
          params.v_stride_h = v.stride(1);
        } else {
          params.k_stride_h = k.stride(0);
          params.k_stride_n = k.stride(1);
          params.v_stride_h = v.stride(0);
          params.v_stride_n = v.stride(1);
        }
        params.nnz_qo = q.size(0);
        params.nnz_kv = k.size(0);
        params.num_qo_heads = q.size(1);
        params.num_kv_heads = k.size(1);
        params.group_size = params.num_qo_heads / params.num_kv_heads;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.batch_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.batch_indices_offset);

        ADDITIONAL_PARAMS_SETTER
        SetSplitKVParams(params, plan_info, float_buffer_ptr, int_buffer_ptr, o, head_dim_vo);

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status = BatchPrefillWithRaggedKVCacheDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
              AttentionVariant>(params, enable_pdl, stream);

          TVM_FFI_ICHECK(status == cudaSuccess)
              << "BatchPrefillWithRaggedKVCacheSM90Run failed with error: "
              << cudaGetErrorString(status);
          return true;
        });
        cudaError_t merge_status = MergeSplitKVStates(params, head_dim_vo, stream);
        TVM_FFI_ICHECK(merge_status == cudaSuccess)
            << "BatchPrefillWithRaggedKVCacheSM90Run merge failed with error: "
            << cudaGetErrorString(merge_status);
      });
}

void BatchPrefillWithPagedKVCacheSM90Run(
    ffi::TensorView float_workspace_buffer, ffi::TensorView int_workspace_buffer,
    Array<int64_t> plan_info_vec, ffi::TensorView q, ffi::TensorView paged_k_cache,
    ffi::TensorView paged_v_cache, ffi::TensorView qo_indptr, ffi::TensorView paged_kv_indptr,
    ffi::TensorView paged_kv_indices, ffi::TensorView paged_kv_last_page_len, ffi::TensorView o,
    Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse.size(0), q.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), q.size(1));
  }
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t num_kv_heads, page_size;
  int64_t head_dim_qk = q.size(2);
  int64_t head_dim_vo = paged_v_cache.size(3);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_swa = window_left != -1;

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
        params.k_ptr = static_cast<DTypeKV*>(paged_k_cache.data_ptr());
        params.v_ptr = static_cast<DTypeKV*>(paged_v_cache.data_ptr());
        params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
        params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.q_stride_n = q.stride(0);
        params.q_stride_h = q.stride(1);
        params.o_stride_n = o.stride(0);
        params.o_stride_h = o.stride(1);
        if (kv_layout == QKVLayout::kNHD) {
          // (num_pages, page_size, num_heads, head_dim)
          params.k_stride_n = paged_k_cache.stride(1);
          params.k_stride_h = paged_k_cache.stride(2);
          params.v_stride_n = paged_v_cache.stride(1);
          params.v_stride_h = paged_v_cache.stride(2);
          // For sparse paged KV cache, store the stride between pages
          params.k_page_stride = paged_k_cache.stride(0);
          params.v_page_stride = paged_v_cache.stride(0);
        } else {
          // (num_pages, num_heads, page_size, head_dim)
          params.k_stride_h = paged_k_cache.stride(1);
          params.k_stride_n = paged_k_cache.stride(2);
          params.v_stride_h = paged_v_cache.stride(1);
          params.v_stride_n = paged_v_cache.stride(2);
          // For sparse paged KV cache, store the stride between pages
          params.k_page_stride = paged_k_cache.stride(0);
          params.v_page_stride = paged_v_cache.stride(0);
        }
        // Sparse mainloop assumes K and V have same strides for efficiency
        TVM_FFI_ICHECK_EQ(params.k_page_stride, params.v_page_stride)
            << "K and V must have same page stride for sparse attention";
        TVM_FFI_ICHECK_EQ(params.k_stride_n, params.v_stride_n)
            << "K and V must have same stride_n for sparse attention";
        params.nnz_qo = q.size(0);
        params.num_qo_heads = q.size(1);
        params.num_kv_heads = num_kv_heads;
        params.group_size = params.num_qo_heads / num_kv_heads;
        params.page_size = page_size;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.batch_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.batch_indices_offset);
        params.kv_indices = static_cast<IdType*>(paged_kv_indices.data_ptr());

        ADDITIONAL_PARAMS_SETTER
        SetSplitKVParams(params, plan_info, float_buffer_ptr, int_buffer_ptr, o, head_dim_vo);

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status = BatchPrefillWithPagedKVCacheDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
              AttentionVariant>(params, enable_pdl, stream);

          TVM_FFI_ICHECK(status == cudaSuccess)
              << "BatchPrefillWithPagedKVCacheSM90Run failed with error: "
              << cudaGetErrorString(status);
          return true;
        });
        cudaError_t merge_status = MergeSplitKVStates(params, head_dim_vo, stream);
        TVM_FFI_ICHECK(merge_status == cudaSuccess)
            << "BatchPrefillWithPagedKVCacheSM90Run merge failed with error: "
            << cudaGetErrorString(merge_status);
      });
}
