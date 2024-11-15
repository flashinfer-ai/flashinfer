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
#include <torch/extension.h>

#include <flashinfer/attention/decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <optional>

#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                  typename AttentionVariant::DTypeO* tmp_v,
                                                  float* tmp_s, cudaStream_t stream);

}  // namespace flashinfer

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    bool use_logits_soft_cap, unsigned int head_dim, torch::Tensor empty_q_data,
    torch::Tensor empty_kv_data, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor indptr, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  auto device = float_workspace_buffer.device();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device));
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  TORCH_CHECK(indptr.device() == torch::kCPU, "indptr must be on CPU");

  DecodePlanInfo plan_info;

  using IdType = int32_t;
  // check indptr has idtype int32
  TORCH_CHECK(indptr.scalar_type() == torch::kInt32, "indptr must be int32");
  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;

  auto q_scalar_type = empty_q_data.scalar_type();
  auto kv_scalar_type = empty_kv_data.scalar_type();

  DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, q_type, kv_type, [&] {
    using DTypeQ = q_type;
    using DTypeKV = kv_type;
    using DTypeO = DTypeQ;
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_LOGITS_SOFT_CAP(use_logits_soft_cap, USE_LOGITS_SOFT_CAP, [&] {
        using ParamsT = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
        using AttentionVariant =
            ComposedAttention<ParamsT, get_variant_code(/*use_custom_mask=*/false,
                                                        /*use_sliding_window=*/true,
                                                        USE_LOGITS_SOFT_CAP, /*use_alibi=*/false)>;
        DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
              GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>;
          cudaError_t status = DecodePlan<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>(
              static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
              static_cast<void*>(int_workspace_buffer.data_ptr()),
              static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
              int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
              batch_size, num_qo_heads, page_size, enable_cuda_graph,
              /*stream=*/torch_current_stream, work_estimation_func);

          TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
                      cudaGetErrorString(status));
          return true;
        });
      });
    });
  });

  return plan_info.ToVector();
}

torch::Tensor BatchDecodeWithPagedKVCacheRun(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, torch::Tensor q, torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache, torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, std::optional<torch::Tensor> alibi_slopes,
    unsigned int kv_layout_code, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<torch::Tensor> maybe_lse) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  auto device = q.device();
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
  uint32_t head_dim = q.size(2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(device));
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q);
  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size, lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q.size(1));
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be float32");
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  using IdType = int32_t;
  constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;

  // get q_scalar_type and kv_scalar_type
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

  DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, q_type, kv_type, [&] {
    using DTypeQ = q_type;
    using DTypeKV = kv_type;
    using DTypeO = DTypeQ;
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_LOGITS_SOFT_CAP(logits_soft_cap > 0, USE_LOGITS_SOFT_CAP, [&] {
        using ParamsT = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
        using AttentionVariant =
            ComposedAttention<ParamsT, get_variant_code(/*use_custom_mask=*/false,
                                                        /*use_sliding_window=*/true,
                                                        USE_LOGITS_SOFT_CAP, /*use_alibi=*/false)>;

        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        ParamsT params(static_cast<DTypeQ*>(q.data_ptr()),
                       /*q_offset=*/nullptr, paged_kv, static_cast<DTypeO*>(o.data_ptr()),
                       /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
                       /*alibi_slopes=*/nullptr, num_qo_heads, q_stride_n, q_stride_h, window_left,
                       logits_soft_cap, sm_scale, rope_scale, rope_theta);

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
            flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                              AttentionVariant>(
                params, tmp_v, tmp_s, /*stream=*/torch_current_stream);
        TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
    });
  });

  return o;
}
