#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_mnnvl_allreduce;

using flashinfer::QuantizationSFLayout;
using flashinfer::trtllm_mnnvl_allreduce::QuantType;
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

void trtllm_mnnvl_allreduce_fusion(
    // Primary I/O
    TensorView input,             // Input shard to be reduced
    Optional<TensorView> output,  // Output result; w/ rmsnorm fusion, it is the normed output; This
                                  // tensor can be empty if quant fusion is enabled and the normed
                                  // result is not needed.

    // Communication infrastructure
    int64_t multicast_buffer_ptr,   // Multicast pointer of the communication buffer
    int64_t buffer_ptrs_dev,        // Device array of unicast pointer
    int64_t buffer_ptr_local,       // Pointer to the local buffer
    TensorView buffer_flags_mnnvl,  // Buffer flag tensor
    // Distributed configuration
    int64_t nranks, int64_t rank,
    // Kernel control flags
    bool use_oneshot, bool launch_with_pdl,

    // RMSNorm fusion
    bool rmsnorm_fusion,  // Enable RMSNorm fusion;
    Optional<TensorView> residual_in,
    Optional<TensorView> residual_out,  // Pre-normed result
    Optional<TensorView> gamma, Optional<double> epsilon,

    // Quantization
    Optional<int64_t> quant_type,       // Quantization type: kFP8, kFP4, kNone
    Optional<TensorView> quant_out,     // Quantized result
    Optional<TensorView> sf_out,        // Scaling factor result
    Optional<TensorView> output_scale,  // This is an INPUT argument that specify the scale applied
                                        // to the quant output
    Optional<int64_t> layout_code       // Scaling factor layout
) {
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  DISPATCH_FLOATING_TYPES_FOR_MNNVL_ALLREDUCE(input.dtype(), c_type, [&] {
    // Extract parameters from tensors
    int64_t num_tokens = input.size(0);
    int64_t token_dim = input.size(1);

    // Convert to enum for validation check
    auto quant_type_enum =
        quant_type.has_value() ? static_cast<QuantType>(quant_type.value()) : QuantType::kNone;

    // Validate input parameters
    TVM_FFI_ICHECK_EQ(token_dim % (sizeof(float4) / sizeof(c_type)), 0)
        << "token_dim must be divisible by " << sizeof(float4) / sizeof(c_type);
    // The output can be null only if we are fusing quantization
    TVM_FFI_ICHECK(quant_type_enum != QuantType::kNone || output.has_value())
        << "Output tensor must be provided when quantization fusion is disabled";
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
    TVM_FFI_ICHECK(quant_type_enum != QuantType::kFP4 || (rmsnorm_fusion && sizeof(c_type) == 2))
        << "NVFP4 Quant fusion is only supported with RMSNorm fusion and FP16/BF16 dtype.";
    TVM_FFI_ICHECK(quant_type_enum == QuantType::kNone ||
                   (output_scale.has_value() &&
                    encode_dlpack_dtype(output_scale.value().dtype()) == float32_code))
        << "output_scale must be provided when quant_type_enum != QuantType::kNone and must be "
           "float32 dtype";
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
      switch (quant_type_enum) {
        case QuantType::kFP8:
          TVM_FFI_ICHECK(quant_out.has_value() && quant_out.value().size(0) == num_tokens &&
                         quant_out.value().size(1) == token_dim)
              << "quant_out shape mismatch: expected (" << num_tokens << ", " << token_dim
              << ") but got (" << quant_out.value().size(0) << ", " << quant_out.value().size(1)
              << ")";
          break;
        case QuantType::kFP4:
          // FP4 packs 2 elements per byte, assuming input tensor is of uint8 or FP4X2 dtype, so
          // quant_out has half the token_dim
          TVM_FFI_ICHECK(quant_out.has_value() && quant_out.value().size(0) == num_tokens &&
                         quant_out.value().size(1) == token_dim / 2)
              << "quant_out shape mismatch for FP4: expected (" << num_tokens << ", "
              << token_dim / 2 << ") but got (" << quant_out.value().size(0) << ", "
              << quant_out.value().size(1) << ")";
          // We only check the sf out size to be large enough.
          TVM_FFI_ICHECK(sf_out.has_value() &&
                         sf_out.value().numel() >= (num_tokens * token_dim / 16))
              << "sf_out size mismatch for FP4: expected at least " << num_tokens * token_dim / 16
              << " elements but got " << sf_out.value().numel();
          break;
      }
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
    params.sfLayout = layout_code.has_value()
                          ? static_cast<QuantizationSFLayout>(layout_code.value())
                          : QuantizationSFLayout::SWIZZLED_128x4;
    params.quantType = quant_type_enum;

    // input data
    params.input = const_cast<void const*>(input.data_ptr());
    params.residualIn =
        residual_in.has_value() ? const_cast<void const*>(residual_in.value().data_ptr()) : nullptr;
    params.gamma = gamma.has_value() ? const_cast<void const*>(gamma.value().data_ptr()) : nullptr;
    params.epsilon = epsilon.has_value() ? epsilon.value() : 1e-5;
    params.outputScale = output_scale.has_value()
                             ? reinterpret_cast<float*>(output_scale.value().data_ptr())
                             : nullptr;

    // output data
    params.output =
        output.has_value() ? reinterpret_cast<void*>(output.value().data_ptr()) : nullptr;
    params.residualOut = residual_out.has_value()
                             ? reinterpret_cast<void*>(residual_out.value().data_ptr())
                             : nullptr;
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
