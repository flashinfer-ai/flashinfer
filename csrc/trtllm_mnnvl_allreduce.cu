#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce;

using flashinfer::QuantizationSFLayout;
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

void trtllm_mnnvl_allreduce_fusion(TensorView input, int64_t multicast_buffer_ptr,
                                   int64_t buffer_ptrs_dev, int64_t buffer_ptr_local,
                                   TensorView buffer_flags_mnnvl, int64_t nranks, int64_t rank,
                                   bool rmsnorm_fusion, bool launch_with_pdl, bool use_oneshot,
                                   Optional<TensorView> output, Optional<TensorView> residual_out,
                                   Optional<TensorView> residual_in, Optional<TensorView> gamma,
                                   Optional<double> epsilon, Optional<double> weight_bias,
                                   Optional<int64_t> quant_type, Optional<TensorView> quant_out,
                                   Optional<TensorView> sf_out, Optional<TensorView> output_scale,
                                   Optional<int64_t> layout_code) {
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(input.dtype(), c_type, [&] {
    // Extract parameters from tensors
    int64_t num_tokens = input.size(0);
    int64_t token_dim = input.size(1);
    auto quant_type_enum =
        quant_type.has_value() ? static_cast<QuantType>(quant_type.value()) : QuantType::kNone;
    auto sf_layout = QuantizationSFLayout::SWIZZLED_128x4;
    if (layout_code.has_value()) {
      auto const sf_layout_code = layout_code.value();
      TVM_FFI_ICHECK(sf_layout_code == static_cast<int64_t>(QuantizationSFLayout::LINEAR) ||
                     sf_layout_code == static_cast<int64_t>(QuantizationSFLayout::SWIZZLED_128x4))
          << "MNNVL quantization fusion supports SWIZZLED_128x4 or LINEAR scale layouts";
      sf_layout = static_cast<QuantizationSFLayout>(sf_layout_code);
    }

    // Validate input parameters
    TVM_FFI_ICHECK_EQ(token_dim % (sizeof(float4) / sizeof(c_type)), 0)
        << "token_dim must be divisible by " << sizeof(float4) / sizeof(c_type);
    TVM_FFI_ICHECK(output.has_value() || quant_type_enum != QuantType::kNone)
        << "output must be provided unless quantization fusion is enabled";
    if (output.has_value()) {
      TVM_FFI_ICHECK(output.value().size(0) == input.size(0) &&
                     output.value().size(1) == input.size(1))
          << "output shape mismatch: expected (" << input.size(0) << ", " << input.size(1)
          << ") but got (" << output.value().size(0) << ", " << output.value().size(1) << ")";
    }
    TVM_FFI_ICHECK(nranks >= 2 && nranks <= 64)
        << "nranks must be between 2 and 64, got " << nranks;
    TVM_FFI_ICHECK(rank >= 0 && rank < nranks)
        << "rank must be between 0 and nranks-1, got " << rank;
    TVM_FFI_ICHECK((residual_in.has_value() && residual_out.has_value() && gamma.has_value() &&
                    epsilon.has_value()) ||
                   !rmsnorm_fusion)
        << "residual_in, residual_out, gamma, and epsilon must be provided if rmsnorm_fusion is "
           "true";
    TVM_FFI_ICHECK(quant_type_enum == QuantType::kNone || rmsnorm_fusion)
        << "MNNVL quantization fusion requires rmsnorm_fusion=true";
    TVM_FFI_ICHECK(quant_type_enum == QuantType::kNone ||
                   quant_type_enum == QuantType::kDynamicFP8 ||
                   (output_scale.has_value() &&
                    encode_dlpack_dtype(output_scale.value().dtype()) == float32_code))
        << "output_scale must be provided for static MNNVL quantization fusion and must be float32";
    TVM_FFI_ICHECK(quant_type_enum == QuantType::kNone || quant_out.has_value())
        << "quant_out must be provided when quantization fusion is enabled";

    if (rmsnorm_fusion) {
      TVM_FFI_ICHECK(residual_in.value().size(0) == num_tokens &&
                     residual_in.value().size(1) == token_dim)
          << "residual_in shape mismatch: expected (" << input.size(0) << ", " << input.size(1)
          << ") but got (" << residual_in.value().size(0) << ", " << residual_in.value().size(1)
          << ")";
      TVM_FFI_ICHECK(residual_out.value().size(0) == num_tokens &&
                     residual_out.value().size(1) == token_dim)
          << "residual_out shape mismatch: expected (" << input.size(0) << ", " << input.size(1)
          << ") but got (" << residual_out.value().size(0) << ", " << residual_out.value().size(1)
          << ")";
      TVM_FFI_ICHECK(gamma.value().size(0) == token_dim)
          << "gamma must have the same shape as token dimension (" << token_dim << ") but got ("
          << gamma.value().size(0) << ")";
    }
    switch (quant_type_enum) {
      case QuantType::kNone:
        break;
      case QuantType::kFP8:
        TVM_FFI_ICHECK(quant_out.value().size(0) == num_tokens &&
                       quant_out.value().size(1) == token_dim)
            << "quant_out shape mismatch for FP8: expected (" << num_tokens << ", " << token_dim
            << ") but got (" << quant_out.value().size(0) << ", " << quant_out.value().size(1)
            << ")";
        TVM_FFI_ICHECK(encode_dlpack_dtype(quant_out.value().dtype()) == float8_e4m3fn_code)
            << "quant_out for FP8 must have dtype float8_e4m3fn";
        break;
      case QuantType::kFP4:
        TVM_FFI_ICHECK(sizeof(c_type) == 2)
            << "NVFP4 MNNVL quantization fusion is only supported for FP16/BF16 inputs";
        TVM_FFI_ICHECK(token_dim % 16 == 0)
            << "NVFP4 MNNVL quantization fusion requires token_dim divisible by 16";
        TVM_FFI_ICHECK(quant_out.value().size(0) == num_tokens &&
                       quant_out.value().size(1) == token_dim / 2)
            << "quant_out shape mismatch for FP4: expected (" << num_tokens << ", " << token_dim / 2
            << ") but got (" << quant_out.value().size(0) << ", " << quant_out.value().size(1)
            << ")";
        TVM_FFI_ICHECK(encode_dlpack_dtype(quant_out.value().dtype()) == uint8_code ||
                       encode_dlpack_dtype(quant_out.value().dtype()) ==
                           encode_dlpack_dtype(dl_float4_e2m1fn_x2))
            << "quant_out for FP4 must have dtype uint8 or float4_e2m1fn_x2";
        TVM_FFI_ICHECK(sf_out.has_value())
            << "sf_out must be provided for NVFP4 MNNVL quantization fusion";
        TVM_FFI_ICHECK(encode_dlpack_dtype(sf_out.value().dtype()) == float8_e4m3fn_code)
            << "sf_out for FP4 must have dtype float8_e4m3fn";
        TVM_FFI_ICHECK(sf_out.value().numel() >= num_tokens * token_dim / 16)
            << "sf_out is too small for FP4: expected at least " << num_tokens * token_dim / 16
            << " elements but got " << sf_out.value().numel();
        break;
      case QuantType::kDynamicFP8:
        TVM_FFI_ICHECK(quant_out.value().size(0) == num_tokens &&
                       quant_out.value().size(1) == token_dim)
            << "quant_out shape mismatch for dynamic FP8: expected (" << num_tokens << ", "
            << token_dim << ") but got (" << quant_out.value().size(0) << ", "
            << quant_out.value().size(1) << ")";
        TVM_FFI_ICHECK(encode_dlpack_dtype(quant_out.value().dtype()) == float8_e4m3fn_code)
            << "quant_out for dynamic FP8 must have dtype float8_e4m3fn";
        TVM_FFI_ICHECK(sf_out.has_value())
            << "scale_out must be provided for dynamic FP8 MNNVL quantization fusion";
        TVM_FFI_ICHECK(sf_out.value().size(0) == num_tokens && sf_out.value().size(1) == 1)
            << "scale_out shape mismatch for dynamic FP8: expected (" << num_tokens
            << ", 1) but got (" << sf_out.value().size(0) << ", " << sf_out.value().size(1) << ")";
        TVM_FFI_ICHECK(encode_dlpack_dtype(sf_out.value().dtype()) == float32_code)
            << "scale_out for dynamic FP8 must have dtype float32";
        break;
      default:
        TVM_FFI_LOG_AND_THROW(NotImplementedError)
            << "Unsupported MNNVL quantization type " << static_cast<int>(quant_type_enum);
    }

    // Create the parameters struct
    AllReduceFusionParams params;

    // Aux Information
    params.nRanks = nranks;
    params.rank = rank;
    params.numTokens = num_tokens;
    params.tokenDim = token_dim;
    params.bufferPtrsDev = reinterpret_cast<void**>(buffer_ptrs_dev);
    params.bufferPtrLocal = reinterpret_cast<void*>(buffer_ptr_local);
    params.multicastPtr = reinterpret_cast<void*>(multicast_buffer_ptr);
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags_mnnvl.data_ptr());
    params.rmsNormFusion = rmsnorm_fusion;
    params.launchWithPdl = launch_with_pdl;
    params.sfLayout = sf_layout;
    params.quantType = quant_type_enum;

    // input data
    params.input = const_cast<void const*>(input.data_ptr());
    params.residualIn =
        residual_in.has_value() ? const_cast<void const*>(residual_in.value().data_ptr()) : nullptr;
    params.gamma = gamma.has_value() ? const_cast<void const*>(gamma.value().data_ptr()) : nullptr;
    params.epsilon = epsilon.has_value() ? epsilon.value() : 1e-5;
    params.weightBias = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
    params.outputScale = output_scale.has_value()
                             ? reinterpret_cast<float*>(output_scale.value().data_ptr())
                             : nullptr;

    // output data
    params.output = output.has_value() ? const_cast<void*>(output.value().data_ptr()) : nullptr;
    params.residualOut =
        residual_out.has_value() ? const_cast<void*>(residual_out.value().data_ptr()) : nullptr;
    params.quantOut =
        quant_out.has_value() ? reinterpret_cast<void*>(quant_out.value().data_ptr()) : nullptr;
    params.scalingFactorOut =
        sf_out.has_value() ? reinterpret_cast<void*>(sf_out.value().data_ptr()) : nullptr;
    params.stream = stream;

    cudaError_t status;
    if (use_oneshot) {
      status = oneshotAllreduceFusionDispatch<c_type>(params);
    } else {
      status = twoshotAllreduceFusionDispatch<c_type>(params);
    }
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "trtllm_mnnvl_allreduce_fusion failed with error code " << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mnnvl_allreduce_fusion, trtllm_mnnvl_allreduce_fusion);
