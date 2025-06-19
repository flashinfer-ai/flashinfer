
#include <flashinfer/attention/decode_mla_cute_sm80.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <optional>

#include "mla_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

at::Tensor BatchDecodeWithPagedKVCachePlanMLA(at::Tensor float_workspace_buffer,
                                              at::Tensor int_workspace_buffer,
                                              at::Tensor page_locked_int_workspace_buffer,
                                              at::Tensor indptr, int64_t batch_size,
                                              int64_t num_qo_heads, int64_t page_size,
                                              bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  DecodePlanInfo plan_info;
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMlaCuteSM80<
      HEAD_DIM_CKV, HEAD_DIM_KPE, QO_TILE_LEN, AttentionVariant, Params>;
  cudaError_t status =
      DecodePlan<HEAD_DIM_CKV, flashinfer::PosEncodingMode::kNone, AttentionVariant, Params>(
          static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
          static_cast<void*>(int_workspace_buffer.data_ptr()),
          static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
          int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
          batch_size, num_qo_heads, page_size, enable_cuda_graph, /*stream=*/stream,
          work_estimation_func);

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCachePlanMLA failed with error ",
              cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q_nope, at::Tensor q_pe, at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, double sm_scale, int64_t window_left, double logits_soft_cap, double rope_scale,
    double rope_theta, std::optional<at::Tensor> maybe_lse,
    bool enable_pdl  // fake placeholder, sm80 does not support pdl
) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  auto device = q_nope.device();
  int64_t batch_size = q_nope.size(0);
  int64_t num_qo_heads = q_nope.size(1);
  int64_t page_size = paged_ckv_cache.size(1);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size, lse.size(0), q_nope.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q_nope.size(1));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  paged_kv_mla_t<DTypeKV, IdType> paged_kv(
      page_size, HEAD_DIM_CKV, HEAD_DIM_KPE, batch_size,
      static_cast<DTypeKV*>(paged_ckv_cache.data_ptr()), paged_ckv_cache.strides().data(),
      static_cast<DTypeKV*>(paged_kpe_cache.data_ptr()), paged_kpe_cache.strides().data(),
      static_cast<IdType*>(paged_kv_indices.data_ptr()),
      static_cast<IdType*>(paged_kv_indptr.data_ptr()),
      static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
  Params params(static_cast<DTypeQ*>(q_nope.data_ptr()), static_cast<DTypeQ*>(q_pe.data_ptr()),
                /*q_offset=*/nullptr, paged_kv, static_cast<DTypeO*>(o.data_ptr()),
                /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
                num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);

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

  const c10::cuda::OptionalCUDAGuard device_guard(paged_ckv_cache.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t status = BatchDecodeWithPagedKVCacheDispatchedMlaCuteSM80<HEAD_DIM_CKV, HEAD_DIM_KPE,
                                                                        QO_TILE_LEN, Params>(
      params, tmp_v, tmp_s, /*stream=*/stream);
  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}
