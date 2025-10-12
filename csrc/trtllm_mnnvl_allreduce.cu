#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce;

using tvm::ffi::Optional;

#define DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(dtype, c_type, ...)             \
  [&] {                                                                             \
    switch (encode_dlpack_dtype(dtype)) {                                           \
      case float32_code: {                                                          \
        using c_type = float;                                                       \
        return __VA_ARGS__();                                                       \
      }                                                                             \
      case float16_code: {                                                          \
        using c_type = half;                                                        \
        return __VA_ARGS__();                                                       \
      }                                                                             \
      case bfloat16_code: {                                                         \
        using c_type = __nv_bfloat16;                                               \
        return __VA_ARGS__();                                                       \
      }                                                                             \
      default:                                                                      \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                                  \
            << "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE."; \
    }                                                                               \
  }()

void trtllm_mnnvl_all_reduce(TensorView in, int64_t multicast_buffer_ptr, int64_t buffer_ptrs_dev,
                             int64_t buffer_M, TensorView buffer_flags_mnnvl, int64_t nranks,
                             int64_t rank, bool wait_for_results, bool launch_with_pdl,
                             Optional<TensorView> out) {
  cudaSetDevice(in->device.device_id);
  auto stream = get_stream(in->device);

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(in->dtype, c_type, [&] {
    // Extract parameters from tensors
    int64_t num_tokens = in->shape[0];
    int64_t token_dim = in->shape[1];

    // Validate input parameters
    TVM_FFI_ICHECK_EQ(token_dim % (sizeof(float2) / sizeof(c_type)), 0)
        << "token_dim must be divisible by " << sizeof(float2) / sizeof(c_type);
    TVM_FFI_ICHECK(nranks >= 2 && nranks <= 64)
        << "nranks must be between 2 and 64, got " << nranks;
    TVM_FFI_ICHECK(rank >= 0 && rank < nranks)
        << "rank must be between 0 and nranks-1, got " << rank;
    TVM_FFI_ICHECK(out.has_value() || !wait_for_results)
        << "out tensor must be provided if wait_for_results is true";

    // Create the parameters struct
    AllReduceParams<c_type> params;
    params.nranks = nranks;
    params.rank = rank;
    params.buffer_M = buffer_M;
    params.num_tokens = num_tokens;
    params.token_dim = token_dim;
    params.buffer_ptrs_dev = reinterpret_cast<void**>(buffer_ptrs_dev);
    params.multicast_ptr = reinterpret_cast<void*>(multicast_buffer_ptr);
    params.buffer_flags = buffer_flags_mnnvl->data;
    params.wait_for_results = wait_for_results;
    params.launch_with_pdl = launch_with_pdl;
    params.input = in->data;
    params.output = out.has_value() ? out.value()->data : nullptr;
    params.stream = stream;

    auto status = twoshot_allreduce_dispatch_world_size<c_type>(params);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "twoshot_allreduce_dispatch_world_size failed with error code "
        << cudaGetErrorString(status);
  });
}

void trtllm_mnnvl_rmsnorm(int64_t multicast_buffer_ptr, TensorView prenorm_output,
                          TensorView normed_output, TensorView gamma, double epsilon,
                          TensorView residual, TensorView buffer_flags, bool launch_with_pdl) {
  cudaSetDevice(prenorm_output->device.device_id);
  auto stream = get_stream(prenorm_output->device);

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(prenorm_output->dtype, c_type, [&] {
    // Create the parameters struct
    RMSNormParams<c_type> params;
    params.residual_output = prenorm_output->data;
    params.output = normed_output->data;
    params.input = reinterpret_cast<void const*>(multicast_buffer_ptr);
    params.gamma = gamma->data;
    params.epsilon = epsilon;
    params.residual = residual->data;
    params.buffer_flags = reinterpret_cast<uint32_t*>(buffer_flags->data);
    params.batch = normed_output->shape[0];
    params.hidden_dim = normed_output->shape[1];
    params.stream = stream;
    params.launch_with_pdl = launch_with_pdl;
    auto status = twoshot_rmsnorm_dispatch_hidden_dim<c_type>(params);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "twoshot_rmsnorm_dispatch_hidden_dim failed with error code "
        << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_all_reduce, trtllm_mnnvl_all_reduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_rmsnorm, trtllm_mnnvl_rmsnorm);
