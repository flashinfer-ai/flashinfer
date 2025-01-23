#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <optional>

#include "mla_config.inc"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q_nope, at::Tensor q_pe,
    at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o, float sm_scale,
    int window_left, float logits_soft_cap, float rope_scale, float rope_theta,
    std::optional<at::Tensor> maybe_lse, int64_t cuda_stream) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(plan_info_vec);

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

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status =
      BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE, AttentionVariant,
                                               Params>(params, tmp_v, tmp_s, /*stream=*/stream);
  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}
