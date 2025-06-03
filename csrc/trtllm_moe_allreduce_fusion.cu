#include <string>

#include "flashinfer/comm/trtllm_moe_allreduce_fusion.cuh"
#include "pytorch_extension_utils.h"

using namespace flashinfer::trtllm_moe_allreduce_fusion;

void trtllm_moe_allreduce_fusion(
    at::Tensor& allreduce_in, int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_size, int64_t workspace_ptr, bool launch_with_pdl, at::Tensor& residual_in,
    at::Tensor& rms_gamma, double rms_eps, double scale_factor,
    int64_t moe_reduction_device_num_experts, at::Tensor& moe_reduction_scale_input,
    at::Tensor& moe_reduction_active_experts_token_input, at::Tensor& moe_reduction_token_input,
    std::optional<int64_t> layout_code, std::optional<at::Tensor>& residual_out,
    std::optional<at::Tensor>& norm_out, std::optional<at::Tensor>& quant_out,
    std::optional<at::Tensor>& scale_out) {
  const c10::cuda::OptionalCUDAGuard device_guard(allreduce_in.device());
  MoeReductionAllReduceFusionParams<half> params;
  params.nranks = world_size;
  params.rank = world_rank;
  params.size = token_num * hidden_size;
  params.hidden_dim = hidden_size;
  params.workspace = reinterpret_cast<void**>(workspace_ptr);

  params.allreduce_in = reinterpret_cast<void*>(allreduce_in.data_ptr());
  params.residual_in = reinterpret_cast<void*>(residual_in.data_ptr());
  params.residual_out =
      residual_out.has_value() ? reinterpret_cast<void*>(residual_out.value().data_ptr()) : nullptr;
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
                      ? static_cast<FP4QuantizationSFLayout>(layout_code.value())
                      : FP4QuantizationSFLayout::SWIZZLED;
  params.stream = at::cuda::getCurrentCUDAStream();

  params.moe_reduction_device_num_experts = moe_reduction_device_num_experts;
  params.moe_reduction_scale_input = reinterpret_cast<float*>(moe_reduction_scale_input.data_ptr());
  params.moe_reduction_active_experts_token_input =
      reinterpret_cast<void*>(moe_reduction_active_experts_token_input.data_ptr());
  params.moe_reduction_token_input = reinterpret_cast<void*>(moe_reduction_token_input.data_ptr());

  auto status = moereduction_allreduce_fusion_op(params, launch_with_pdl);
  TORCH_CHECK(status == cudaSuccess, "moereduction_allreduce_fusion_op failed with error code ",
              cudaGetErrorString(status));
}
