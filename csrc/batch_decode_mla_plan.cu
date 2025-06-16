#include <flashinfer/attention/decode.cuh>
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
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  DecodePlanInfo plan_info;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

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

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCachePlanMLA failed with error ",
              cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}
