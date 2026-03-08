#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>

#include "mla_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;

Array<int64_t> BatchDecodeWithPagedKVCachePlanMLA(TensorView float_workspace_buffer,
                                                  TensorView int_workspace_buffer,
                                                  TensorView page_locked_int_workspace_buffer,
                                                  TensorView indptr, int64_t batch_size,
                                                  int64_t num_qo_heads, int64_t page_size,
                                                  bool enable_cuda_graph) {
  CHECK_INPUT_TYPE(indptr, dl_int32);

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  DecodePlanInfo plan_info;

  auto work_estimation_func =
      BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE,
                                                             AttentionVariant, Params>;
  cudaError_t status =
      DecodePlan<HEAD_DIM_CKV, flashinfer::PosEncodingMode::kRoPELlama, AttentionVariant, Params>(
          static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
          static_cast<void*>(int_workspace_buffer.data_ptr()),
          static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
          int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
          batch_size, num_qo_heads, page_size, enable_cuda_graph, /*stream=*/stream,
          work_estimation_func);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "BatchDecodeWithPagedKVCachePlanMLA failed with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}
