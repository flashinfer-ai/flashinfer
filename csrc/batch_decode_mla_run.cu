#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>

#include "mla_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

void BatchDecodeWithPagedKVCacheRunMLA(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                                       Array<int64_t> plan_info_vec, Tensor q_nope, Tensor q_pe,
                                       Tensor paged_ckv_cache, Tensor paged_kpe_cache,
                                       Tensor paged_kv_indptr, Tensor paged_kv_indices,
                                       Tensor paged_kv_last_page_len, Tensor o, double sm_scale,
                                       int64_t window_left, double logits_soft_cap,
                                       double rope_scale, double rope_theta,
                                       Optional<Tensor> maybe_lse, bool enable_pdl) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));

  int64_t batch_size = q_nope->shape[0];
  int64_t num_qo_heads = q_nope->shape[1];
  int64_t page_size = paged_ckv_cache->shape[1];

  if (maybe_lse.has_value()) {
    const auto& lse = maybe_lse.value();
    TVM_FFI_ICHECK_EQ(lse->shape[0], batch_size);
    TVM_FFI_ICHECK_EQ(lse->shape[1], num_qo_heads);
  }

  TVM_FFI_ICHECK_GE(logits_soft_cap, 0.f) << "logits_soft_cap must be non-negative";

  void* float_buffer = static_cast<void*>(float_workspace_buffer->data);
  void* int_buffer = static_cast<void*>(int_workspace_buffer->data);

  cudaSetDevice(q_nope->device.device_id);
  const cudaStream_t stream = get_stream(q_nope->device);

  paged_kv_mla_t<DTypeKV, IdType> paged_kv(
      page_size, HEAD_DIM_CKV, HEAD_DIM_KPE, batch_size,
      static_cast<DTypeKV*>(paged_ckv_cache->data), paged_ckv_cache.strides().data(),
      static_cast<DTypeKV*>(paged_kpe_cache->data), paged_kpe_cache.strides().data(),
      static_cast<IdType*>(paged_kv_indices->data), static_cast<IdType*>(paged_kv_indptr->data),
      static_cast<IdType*>(paged_kv_last_page_len->data));
  Params params(static_cast<DTypeQ*>(q_nope->data), static_cast<DTypeQ*>(q_pe->data),
                /*q_offset=*/nullptr, paged_kv, static_cast<DTypeO*>(o->data),
                /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse.value()->data) : nullptr),
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

  cudaError_t status =
      BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE, AttentionVariant,
                                               Params>(params, tmp_v, tmp_s, enable_pdl,
                                                       /*stream=*/stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "BatchDecodeWithPagedKVCache failed with error: " << cudaGetErrorString(status);
}
