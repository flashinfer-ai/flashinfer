#include <string>

#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"
#include "pytorch_extension_utils.h"

using namespace flashinfer::trtllm_allreduce_fusion;

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
      case at::ScalarType::Float: {                                                               \
        using c_type = float;                                                                     \
        return __VA_ARGS__();                                                                     \
      }                                                                                           \
      default:                                                                                    \
        TORCH_CHECK(false,                                                                        \
                    "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE: ", scalar_type); \
    }                                                                                             \
  }()

void trtllm_allreduce_fusion(
    at::Tensor& allreduce_in, int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_size, at::Tensor& workspace_ptrs, bool launch_with_pdl, bool use_oneshot,
    bool trigger_completion_at_end, bool fp32_acc, int64_t pattern_code,
    std::optional<at::Tensor> allreduce_out, std::optional<at::Tensor> residual_in,
    std::optional<at::Tensor> residual_out, std::optional<at::Tensor> norm_out,
    std::optional<at::Tensor> quant_out, std::optional<at::Tensor> scale_out,
    std::optional<at::Tensor> rms_gamma, std::optional<double> rms_eps,
    std::optional<at::Tensor> scale_factor, std::optional<int64_t> layout_code) {
  const c10::cuda::OptionalCUDAGuard device_guard(allreduce_in.device());
  // todo(Yingyi): add dispatch for float and bfloat16

  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(allreduce_in.scalar_type(), c_type, [&] {
    AllReduceFusionParams<c_type> params;
    params.nranks = world_size;
    params.rank = world_rank;
    params.size = token_num * hidden_size;
    params.hidden_dim = hidden_size;
    params.workspace = reinterpret_cast<void**>(workspace_ptrs.data_ptr());

    // todo(Yingyi): update optional params
    // todo(Yingyi): add params check with pattern
    params.allreduce_in = reinterpret_cast<void*>(allreduce_in.data_ptr());
    params.allreduce_out = allreduce_out.has_value()
                               ? reinterpret_cast<void*>(allreduce_out.value().data_ptr())
                               : nullptr;
    params.residual_in =
        residual_in.has_value() ? reinterpret_cast<void*>(residual_in.value().data_ptr()) : nullptr;
    params.residual_out = residual_out.has_value()
                              ? reinterpret_cast<void*>(residual_out.value().data_ptr())
                              : nullptr;
    params.norm_out =
        norm_out.has_value() ? reinterpret_cast<void*>(norm_out.value().data_ptr()) : nullptr;
    params.quant_out =
        quant_out.has_value() ? reinterpret_cast<void*>(quant_out.value().data_ptr()) : nullptr;
    params.scale_out =
        scale_out.has_value() ? reinterpret_cast<void*>(scale_out.value().data_ptr()) : nullptr;
    params.rms_gamma =
        rms_gamma.has_value() ? reinterpret_cast<void*>(rms_gamma.value().data_ptr()) : nullptr;
    params.rms_eps = rms_eps.has_value() ? static_cast<float>(rms_eps.value()) : 0.0f;
    params.scale_factor = scale_factor.has_value()
                              ? reinterpret_cast<float*>(scale_factor.value().data_ptr())
                              : nullptr;
    params.use_oneshot = use_oneshot;
    params.layout = layout_code.has_value() ? static_cast<QuantizationSFLayout>(layout_code.value())
                                            : QuantizationSFLayout::SWIZZLED_128x4;
    params.pattern = static_cast<AllReduceFusionPattern>(pattern_code);
    params.trigger_completion_at_end = trigger_completion_at_end;
    params.stream = at::cuda::getCurrentCUDAStream();

    auto status = allreduce_fusion_op(params, launch_with_pdl, fp32_acc);
    TORCH_CHECK(status == cudaSuccess, "allreduce_fusion_op failed with error code",
                cudaGetErrorString(status));
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_allreduce_fusion", &trtllm_allreduce_fusion);
}
