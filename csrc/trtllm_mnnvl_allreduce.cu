#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "pytorch_extension_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce;

#define DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(scalar_type, c_type, ...)                    \
  [&] {                                                                                          \
    switch (scalar_type) {                                                                       \
      case at::ScalarType::Float: {                                                              \
        using c_type = float;                                                                    \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
      case at::ScalarType::Half: {                                                               \
        using c_type = half;                                                                     \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
      case at::ScalarType::BFloat16: {                                                           \
        using c_type = __nv_bfloat16;                                                            \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
      default:                                                                                   \
        TORCH_CHECK(false, "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE: ", \
                    scalar_type);                                                                \
    }                                                                                            \
  }()

void trtllm_mnnvl_all_reduce(at::Tensor& in, int64_t multicast_buffer_ptr, int64_t buffer_ptrs_dev,
                             int64_t buffer_M, at::Tensor& buffer_flags_mnnvl, int64_t nranks,
                             int64_t rank, bool wait_for_results, bool launch_with_pdl,
                             std::optional<at::Tensor> out) {
  const c10::cuda::OptionalCUDAGuard device_guard(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(in.scalar_type(), c_type, [&] {
    // Extract parameters from tensors
    int64_t num_tokens = in.size(0);
    int64_t token_dim = in.size(1);

    // Validate input parameters
    TORCH_CHECK(token_dim % (sizeof(float2) / sizeof(c_type)) == 0,
                "token_dim must be divisible by ", sizeof(float2) / sizeof(c_type));
    TORCH_CHECK(nranks >= 2 && nranks <= 64, "nranks must be between 2 and 64, got ", nranks);
    TORCH_CHECK(rank >= 0 && rank < nranks, "rank must be between 0 and nranks-1, got ", rank);
    TORCH_CHECK(out.has_value() || !wait_for_results,
                "out tensor must be provided if wait_for_results is true");

    // Create the parameters struct
    AllReduceParams<c_type> params;
    params.nranks = nranks;
    params.rank = rank;
    params.buffer_M = buffer_M;
    params.num_tokens = num_tokens;
    params.token_dim = token_dim;
    params.buffer_ptrs_dev = reinterpret_cast<void**>(buffer_ptrs_dev);
    params.multicast_ptr = reinterpret_cast<void*>(multicast_buffer_ptr);
    params.buffer_flags = buffer_flags_mnnvl.data_ptr();
    params.wait_for_results = wait_for_results;
    params.launch_with_pdl = launch_with_pdl;
    params.input = in.data_ptr();
    params.output = out.has_value() ? out.value().data_ptr() : nullptr;
    params.stream = stream.stream();

    auto status = twoshot_allreduce_dispatch_world_size<c_type>(params);
    TORCH_CHECK(status == cudaSuccess,
                "twoshot_allreduce_dispatch_world_size failed with error code ",
                cudaGetErrorString(status));
  });
}

void trtllm_mnnvl_rmsnorm(int64_t multicast_buffer_ptr, at::Tensor& prenorm_output,
                          at::Tensor& normed_output, at::Tensor const& gamma, double epsilon,
                          at::Tensor const& residual, at::Tensor& buffer_flags,
                          bool launch_with_pdl) {
  const c10::cuda::OptionalCUDAGuard device_guard(prenorm_output.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(prenorm_output.scalar_type(), c_type, [&] {
    // Create the parameters struct
    RMSNormParams<c_type> params;
    params.residual_output = prenorm_output.data_ptr();
    params.output = normed_output.data_ptr();
    params.input = reinterpret_cast<void const*>(multicast_buffer_ptr);
    params.gamma = gamma.data_ptr();
    params.epsilon = epsilon;
    params.residual = residual.data_ptr();
    params.buffer_flags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr());
    params.batch = normed_output.size(0);
    params.hidden_dim = normed_output.size(1);
    params.stream = stream.stream();
    params.launch_with_pdl = launch_with_pdl;
    auto status = twoshot_rmsnorm_dispatch_hidden_dim<c_type>(params);
    TORCH_CHECK(status == cudaSuccess,
                "twoshot_rmsnorm_dispatch_hidden_dim failed with error code ",
                cudaGetErrorString(status));
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_mnnvl_all_reduce", &trtllm_mnnvl_all_reduce);
  m.def("trtllm_mnnvl_rmsnorm", &trtllm_mnnvl_rmsnorm);
}
