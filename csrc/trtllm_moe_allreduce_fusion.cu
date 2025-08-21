#include <string>

#include "flashinfer/comm/trtllm_moe_allreduce_fusion.cuh"
#include "pytorch_extension_utils.h"

using namespace flashinfer::trtllm_moe_allreduce_fusion;

#define DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(scalar_type, c_type, ...)                           \
  [&] {                                                                                           \
    switch (scalar_type) {                                                                        \
      case at::ScalarType::Half: {                                                                \
        using c_type = half;                                                                      \
        return __VA_ARGS__();                                                                     \
      }                                                                                           \
      case at::ScalarType::BFloat16: {                                                            \
        using c_type = __nv_bfloat16;                                                             \
        return __VA_ARGS__();                                                                     \
      }                                                                                           \
      default:                                                                                    \
        TORCH_CHECK(false,                                                                        \
                    "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE: ", scalar_type); \
    }                                                                                             \
  }()

void trtllm_moe_allreduce_fusion(
    int64_t world_size, int64_t world_rank, int64_t token_num, int64_t hidden_size,
    at::Tensor& workspace_ptrs, bool launch_with_pdl, at::Tensor& residual_in,
    at::Tensor& rms_gamma, double rms_eps, double scale_factor,
    int64_t moe_reduction_device_num_experts, at::Tensor& moe_reduction_scale_input,
    at::Tensor& moe_reduction_active_experts_token_input, at::Tensor& moe_reduction_token_input,
    std::optional<int64_t> layout_code, std::optional<at::Tensor> moe_allreduce_out,
    std::optional<at::Tensor> residual_out, std::optional<at::Tensor> norm_out,
    std::optional<at::Tensor> quant_out, std::optional<at::Tensor> scale_out) {
  const c10::cuda::OptionalCUDAGuard device_guard(
      moe_reduction_active_experts_token_input.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(
      moe_reduction_active_experts_token_input.scalar_type(), c_type, [&] {
        MoeReductionAllReduceFusionParams<c_type> params;
        params.nranks = world_size;
        params.rank = world_rank;
        params.size = token_num * hidden_size;
        params.hidden_dim = hidden_size;
        params.workspace = reinterpret_cast<void**>(workspace_ptrs.data_ptr());

        params.moe_allreduce_out =
            moe_allreduce_out.has_value()
                ? reinterpret_cast<void*>(moe_allreduce_out.value().data_ptr())
                : nullptr;
        params.residual_in = reinterpret_cast<void*>(residual_in.data_ptr());
        params.residual_out = residual_out.has_value()
                                  ? reinterpret_cast<void*>(residual_out.value().data_ptr())
                                  : nullptr;
        params.norm_out =
            norm_out.has_value() ? reinterpret_cast<void*>(norm_out.value().data_ptr()) : nullptr;
        params.quant_out =
            quant_out.has_value() ? reinterpret_cast<void*>(quant_out.value().data_ptr()) : nullptr;
        params.scale_out =
            scale_out.has_value() ? reinterpret_cast<void*>(scale_out.value().data_ptr()) : nullptr;
        params.rms_gamma = reinterpret_cast<void*>(rms_gamma.data_ptr());
        params.rms_eps = static_cast<float>(rms_eps);
        params.scale_factor = static_cast<float>(scale_factor);
        params.layout = layout_code.has_value()
                            ? static_cast<QuantizationSFLayout>(layout_code.value())
                            : QuantizationSFLayout::SWIZZLED_128x4;
        params.stream = stream;

        params.moe_reduction_device_num_experts = moe_reduction_device_num_experts;
        params.moe_reduction_scale_input =
            reinterpret_cast<float*>(moe_reduction_scale_input.data_ptr());
        params.moe_reduction_active_experts_token_input =
            reinterpret_cast<void*>(moe_reduction_active_experts_token_input.data_ptr());
        params.moe_reduction_token_input =
            reinterpret_cast<void*>(moe_reduction_token_input.data_ptr());

        auto status = moereduction_allreduce_fusion_op(params, launch_with_pdl);
        TORCH_CHECK(status == cudaSuccess,
                    "moereduction_allreduce_fusion_op failed with error code ",
                    cudaGetErrorString(status));
      });
}

void trtllm_moe_finalize_allreduce_fusion(
    at::Tensor const& allreduce_in, at::Tensor const& residual_in, at::Tensor const& norm_weight,
    at::Tensor const& expanded_idx_to_permuted_idx, at::Tensor& norm_out, at::Tensor& residual_out,
    bool launch_with_pdl, at::Tensor& workspace, int64_t const world_rank, int64_t const world_size,
    double const eps, std::optional<at::Tensor> const& shared_expert_output,
    std::optional<at::Tensor> const& expert_scale_factor) {
  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(residual_in.scalar_type(), c_type, [&] {
    MoeFinalizeAllReduceFusionParams<c_type> params;

    int hidden_dim = residual_in.size(-1);
    int top_k = expanded_idx_to_permuted_idx.size(-1);

    params.quant_out = nullptr;
    params.scale_out = nullptr;

    params.nranks = static_cast<int>(world_size);
    params.rank = static_cast<int>(world_rank);
    // size: num_token * hidden_dim
    params.size = residual_in.numel();
    params.hidden_dim = hidden_dim;

    // workspace: AR scratch space
    params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
    params.rms_gamma = norm_weight.data_ptr();
    params.rms_eps = static_cast<float>(eps);
    params.residual_in = residual_in.data_ptr();
    params.stream = at::cuda::getCurrentCUDAStream(norm_weight.get_device());

    // MOE Reduction specific params
    params.top_k = top_k;
    params.allreduce_in = allreduce_in.data_ptr();
    params.expert_scale_factor =
        expert_scale_factor.has_value() ? expert_scale_factor.value().data_ptr() : nullptr;
    TORCH_CHECK(expanded_idx_to_permuted_idx.scalar_type() == at::ScalarType::Int,
                "expanded_idx_to_permuted_idx must be int32");
    params.expanded_idx_to_permuted_idx =
        static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    params.shared_expert_output =
        shared_expert_output.has_value() ? shared_expert_output.value().data_ptr() : nullptr;

    // output tensors
    params.norm_out = norm_out.mutable_data_ptr();
    params.residual_out = residual_out.mutable_data_ptr();

    auto status = moefinalize_allreduce_fusion_op(params, launch_with_pdl);
    TORCH_CHECK(status == cudaSuccess, "moefinalize_allreduce_fusion_op failed with error code ",
                cudaGetErrorString(status));
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_moe_allreduce_fusion", &trtllm_moe_allreduce_fusion);
  m.def("trtllm_moe_finalize_allreduce_fusion", &trtllm_moe_finalize_allreduce_fusion);
}
