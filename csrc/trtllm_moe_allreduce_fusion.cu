#include <string>

#include "flashinfer/comm/trtllm_moe_allreduce_fusion.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_moe_allreduce_fusion;

using tvm::ffi::Optional;

#define DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(dtype, c_type, ...)             \
  [&] {                                                                       \
    switch (encode_dlpack_dtype(dtype)) {                                     \
      case float16_code: {                                                    \
        using c_type = half;                                                  \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case bfloat16_code: {                                                   \
        using c_type = __nv_bfloat16;                                         \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                            \
            << "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE."; \
    }                                                                         \
  }()

void trtllm_moe_allreduce_fusion(
    int64_t world_size, int64_t world_rank, int64_t token_num, int64_t hidden_size,
    TensorView workspace_ptrs, bool launch_with_pdl, TensorView residual_in, TensorView rms_gamma,
    double rms_eps, double scale_factor, int64_t moe_reduction_device_num_experts,
    TensorView moe_reduction_scale_input, TensorView moe_reduction_active_experts_token_input,
    TensorView moe_reduction_token_input, Optional<int64_t> layout_code,
    Optional<TensorView> moe_allreduce_out, Optional<TensorView> residual_out,
    Optional<TensorView> norm_out, Optional<TensorView> quant_out, Optional<TensorView> scale_out) {
  ffi::CUDADeviceGuard device_guard(moe_reduction_active_experts_token_input.device().device_id);
  auto stream = get_stream(moe_reduction_active_experts_token_input.device());

  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(
      moe_reduction_active_experts_token_input.dtype(), c_type, [&] {
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
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "moereduction_allreduce_fusion_op failed with error code "
            << cudaGetErrorString(status);
      });
}

void trtllm_moe_finalize_allreduce_fusion(
    TensorView allreduce_in, TensorView residual_in, TensorView norm_weight,
    TensorView expanded_idx_to_permuted_idx, TensorView norm_out, TensorView residual_out,
    bool launch_with_pdl, TensorView workspace, int64_t const world_rank, int64_t const world_size,
    double const eps, Optional<TensorView> shared_expert_output,
    Optional<TensorView> expert_scale_factor) {
  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(residual_in.dtype(), c_type, [&] {
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
    params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
    params.rms_gamma = norm_weight.data_ptr();
    params.rms_eps = static_cast<float>(eps);
    params.residual_in = residual_in.data_ptr();
    params.stream = get_stream(norm_weight.device());

    // MOE Reduction specific params
    params.top_k = top_k;
    params.allreduce_in = allreduce_in.data_ptr();
    params.expert_scale_factor =
        expert_scale_factor.has_value() ? expert_scale_factor.value().data_ptr() : nullptr;
    TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.dtype(), dl_int32)
        << "expanded_idx_to_permuted_idx must be int32";
    params.expanded_idx_to_permuted_idx =
        static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    params.shared_expert_output =
        shared_expert_output.has_value() ? shared_expert_output.value().data_ptr() : nullptr;

    // output tensors
    params.norm_out = norm_out.data_ptr();
    params.residual_out = residual_out.data_ptr();

    auto status = moefinalize_allreduce_fusion_op(params, launch_with_pdl);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "moefinalize_allreduce_fusion_op failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_allreduce_fusion, trtllm_moe_allreduce_fusion);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_finalize_allreduce_fusion,
                              trtllm_moe_finalize_allreduce_fusion);
