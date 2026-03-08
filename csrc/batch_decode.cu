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
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include "batch_decode_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchDecodeWithPagedKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    TensorView empty_q_data, TensorView empty_kv_data) {
  CHECK_INPUT_TYPE(indptr, dl_int32);

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  DecodePlanInfo plan_info;

  TVM_FFI_ICHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
              GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
          cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
              static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
              static_cast<void*>(int_workspace_buffer.data_ptr()),
              static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
              int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
              batch_size, num_qo_heads, page_size, enable_cuda_graph,
              /*stream=*/stream, work_estimation_func);

          TVM_FFI_ICHECK(status == cudaSuccess)
              << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
          return true;
        });
      });

  return Array(plan_info.ToVector());
}

void BatchDecodeWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                    TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                    TensorView q, TensorView paged_k_cache,
                                    TensorView paged_v_cache, TensorView paged_kv_indptr,
                                    TensorView paged_kv_indices, TensorView paged_kv_last_page_len,
                                    TensorView o, Optional<TensorView> maybe_lse,
                                    int64_t kv_layout_code, int64_t window_left,
                                    bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  CHECK_INPUT_TYPE(paged_kv_indptr, dl_int32);
  CHECK_INPUT_TYPE(paged_kv_indices, dl_int32);
  CHECK_INPUT_TYPE(paged_kv_last_page_len, dl_int32);

  DecodePlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;

  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }
  uint32_t head_dim_qk = q.size(2);
  uint32_t head_dim_vo = paged_v_cache.size(3);

  TVM_FFI_ICHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  if (maybe_lse.has_value()) {
    const auto& lse = maybe_lse.value();
    TVM_FFI_ICHECK_EQ(lse.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(lse.size(1), num_qo_heads);
  }

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TVM_FFI_ICHECK_EQ(k_strides.size(), v_strides.size());
  for (int i = 0; i < k_strides.size(); ++i) {
    TVM_FFI_ICHECK_EQ(k_strides[i], v_strides[i]);
  }
  kv_cache_strides = k_strides.data();

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_QK, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));

        Params params;
        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.paged_kv = paged_kv;
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.padded_batch_size = 0;
        params.num_qo_heads = num_qo_heads;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
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
                                                              AttentionVariant>(params, tmp_v,
                                                                                tmp_s, enable_pdl,
                                                                                /*stream=*/stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
