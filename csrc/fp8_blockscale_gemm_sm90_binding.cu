
#include <tvm/ffi/extra/module.h>
#include "tvm_ffi_utils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace kernels = tensorrt_llm::kernels::fp8_blockscale_gemm;

using tvm::ffi::Function;
using tvm::ffi::Optional;

/**
 * @brief TVM FFI wrapper for TensorRT-LLM's FP8 block-scale GEMM kernel.
 * 
 * This class provides a Python-accessible interface to the CUTLASS-based FP8 GEMM
 * implementation with block-wise quantization. It supports two execution modes:
 * 
 * 1. Pre-quantized FP8: Both inputs are FP8 with external scale factors
 * 2. Internal quantization: BF16 inputs are quantized to FP8 internally
 * 
 * The kernel automatically selects between normal and swapAB execution based on
 * the M dimension for optimal performance.
 * 
 * @note Requires NVIDIA Hopper (SM90) architecture and CUDA 12.8+
 */
class Fp8BlockScaleGemmRunner : public tvm::ffi::ModuleObj {
 public:
  /**
   * @brief Constructor initializes the CUTLASS FP8 GEMM runner.
   * 
   * Template parameters <BF16, BF16, BF16> allow the kernel to:
   * - Accept FP8 inputs with external scales (pre-quantized path)
   * - Accept BF16 inputs for internal quantization to FP8
   * - Produce BF16 output (accumulation happens in FP32 internally)
   */
  Fp8BlockScaleGemmRunner() {
    runner_ = std::make_unique<kernels::CutlassFp8BlockScaleGemmRunner<
        __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();
  }

  ~Fp8BlockScaleGemmRunner() = default;

  const char* type_key() const {
    return "flashinfer.Fp8BlockScaleGemmRunner";
  }

  const char* kind() const final {
    return "fp8_blockscale_gemm_runner";
  }

  Optional<Function> GetFunction(const tvm::ffi::String& name) {
    if (name == "gemm") {
      return Function::FromTyped(
          [this](TensorView input, TensorView weight, TensorView output,
                 Optional<TensorView> scales_a, Optional<TensorView> scales_b) {
            runGemm(input, weight, output, scales_a, scales_b);
          });
    } else if (name == "get_workspace_size") {
      return Function::FromTyped(
          [this](int64_t shape_m, int64_t shape_n, int64_t shape_k) -> int64_t {
            return getWorkspaceSize(shape_m, shape_n, shape_k);
          });
    } else if (name == "configure_workspace") {
      return Function::FromTyped(
          [this](TensorView workspace) {
            configureWorkspace(workspace);
          });
    }
    return Function(nullptr);
  }

 private:
  void runGemm(const TensorView& input, const TensorView& weight, const TensorView& output,
              const Optional<TensorView>& scales_a, const Optional<TensorView>& scales_b) {
    // Get CUDA stream from TVM runtime
    auto stream = get_stream(input.device());

    // Extract tensor info
    auto input_ptr = input.data_ptr();
    auto weight_ptr = weight.data_ptr();
    auto output_ptr = output.data_ptr();

    // Validate tensor dimensions
    TVM_FFI_ICHECK(input.ndim() == 2) << "Input must be 2D (M, K), got " << input.ndim() << "D";
    TVM_FFI_ICHECK(weight.ndim() == 2) << "Weight must be 2D (N, K), got " << weight.ndim() << "D";
    TVM_FFI_ICHECK(output.ndim() == 2) << "Output must be 2D (M, N), got " << output.ndim() << "D";

    int shape_m = input.size(0);
    int shape_k = input.size(1);
    int shape_n = weight.size(0);

    TVM_FFI_ICHECK_EQ(weight.size(1), shape_k) 
        << "Weight K dimension must match input K. Expected " << shape_k 
        << ", got " << weight.size(1);
    TVM_FFI_ICHECK_EQ(output.size(0), shape_m) 
        << "Output M dimension must match input M. Expected " << shape_m 
        << ", got " << output.size(0);
    TVM_FFI_ICHECK_EQ(output.size(1), shape_n) 
        << "Output N dimension must match weight N. Expected " << shape_n 
        << ", got " << output.size(1);

    // Validate K is divisible by block size (128)
    constexpr int BLOCK_SIZE = 128;
    TVM_FFI_ICHECK_EQ(shape_k % BLOCK_SIZE, 0) 
        << "K dimension must be divisible by block size (" << BLOCK_SIZE 
        << "), got K=" << shape_k;

    // Validate workspace is configured
    TVM_FFI_ICHECK(workspace_ != nullptr) 
        << "Workspace not configured. Call configure_workspace() before gemm()";

    // Get scales if provided
    float const* scales_a_ptr = nullptr;
    float const* scales_b_ptr = nullptr;
    
    if (scales_a.has_value()) {
      const auto& scale_a_view = scales_a.value();
      TVM_FFI_ICHECK_EQ(scale_a_view.dtype().code, kDLFloat) 
          << "input_scale must be float32";
      TVM_FFI_ICHECK_EQ(scale_a_view.dtype().bits, 32) 
          << "input_scale must be float32";
      TVM_FFI_ICHECK_EQ(scale_a_view.ndim(), 2) 
          << "input_scale must be 2D (M, K//128)";
      TVM_FFI_ICHECK_EQ(scale_a_view.size(0), shape_m) 
          << "input_scale M dimension mismatch. Expected " << shape_m 
          << ", got " << scale_a_view.size(0);
      TVM_FFI_ICHECK_EQ(scale_a_view.size(1), shape_k / BLOCK_SIZE) 
          << "input_scale K dimension mismatch. Expected " << (shape_k / BLOCK_SIZE) 
          << ", got " << scale_a_view.size(1);
      scales_a_ptr = reinterpret_cast<float const*>(scale_a_view.data_ptr());
    }
    
    if (scales_b.has_value()) {
      const auto& scale_b_view = scales_b.value();
      TVM_FFI_ICHECK_EQ(scale_b_view.dtype().code, kDLFloat) 
          << "weight_scale must be float32";
      TVM_FFI_ICHECK_EQ(scale_b_view.dtype().bits, 32) 
          << "weight_scale must be float32";
      TVM_FFI_ICHECK_EQ(scale_b_view.ndim(), 2) 
          << "weight_scale must be 2D (N, K//128)";
      TVM_FFI_ICHECK_EQ(scale_b_view.size(0), shape_n) 
          << "weight_scale N dimension mismatch. Expected " << shape_n 
          << ", got " << scale_b_view.size(0);
      TVM_FFI_ICHECK_EQ(scale_b_view.size(1), shape_k / BLOCK_SIZE) 
          << "weight_scale K dimension mismatch. Expected " << (shape_k / BLOCK_SIZE) 
          << ", got " << scale_b_view.size(1);
      scales_b_ptr = reinterpret_cast<float const*>(scale_b_view.data_ptr());
    }

    // Check input types - FP8 uses special dtype codes
    bool input_is_fp8 = (input.dtype().code == kDLFloat8_e4m3fn || 
                         input.dtype().code == kDLFloat8_e5m2);
    bool weight_is_fp8 = (weight.dtype().code == kDLFloat8_e4m3fn || 
                          weight.dtype().code == kDLFloat8_e5m2);
    
    // Dispatch to appropriate kernel path
    if (input_is_fp8 && weight_is_fp8) {
      // Path 1: Both inputs are FP8 - use pre-quantized FP8 GEMM
      TVM_FFI_ICHECK(scales_a_ptr != nullptr && scales_b_ptr != nullptr)
          << "FP8 inputs require scale factors. Provide both input_scale and weight_scale.";
      
      // Validate output dtype is BF16
      TVM_FFI_ICHECK_EQ(output.dtype().code, kDLBfloat) 
          << "Output must be BF16 for FP8 inputs";
      TVM_FFI_ICHECK_EQ(output.dtype().bits, 16) 
          << "Output must be BF16 for FP8 inputs";
      
      auto input_fp8 = reinterpret_cast<__nv_fp8_e4m3 const*>(input_ptr);
      auto weight_fp8 = reinterpret_cast<__nv_fp8_e4m3 const*>(weight_ptr);
      auto output_bf16 = reinterpret_cast<__nv_bfloat16*>(output_ptr);
      
      int ld_a = shape_k;  // Leading dimension for row-major input
      int ld_b = shape_k;  // Leading dimension for row-major weight
      int ld_d = shape_n;  // Leading dimension for row-major output
      
      runner_->gemm(input_fp8, ld_a, weight_fp8, ld_b, output_bf16, ld_d,
                    shape_m, shape_n, shape_k, scales_a_ptr, scales_b_ptr, stream);
    } else if (!input_is_fp8 && !weight_is_fp8) {
      // Path 2: Both inputs are BF16 - use internal quantization
      TVM_FFI_ICHECK(scales_a_ptr == nullptr && scales_b_ptr == nullptr) 
          << "BF16 inputs use internal quantization. Do not provide scales. "
          << "For external scales, use FP8 inputs.";
      
      // Validate input/weight dtypes are BF16
      TVM_FFI_ICHECK_EQ(input.dtype().code, kDLBfloat) 
          << "Input must be BF16 for internal quantization path";
      TVM_FFI_ICHECK_EQ(input.dtype().bits, 16) 
          << "Input must be BF16 for internal quantization path";
      TVM_FFI_ICHECK_EQ(weight.dtype().code, kDLBfloat) 
          << "Weight must be BF16 for internal quantization path";
      TVM_FFI_ICHECK_EQ(weight.dtype().bits, 16) 
          << "Weight must be BF16 for internal quantization path";
      
      // Call internal quantization path (note: different argument order!)
      runner_->gemm(output_ptr, input_ptr, weight_ptr, shape_m, shape_n, shape_k,
                    stream, scales_a_ptr, scales_b_ptr);
    } else {
      // Path 3: Mixed dtypes - not supported
      TVM_FFI_ICHECK(false) 
          << "Mixed FP8/BF16 inputs not supported. Both input and weight must be "
          << "either FP8 (with scales) or BF16 (internal quantization).";
    }
  }

  int64_t getWorkspaceSize(int64_t shape_m, int64_t shape_n, int64_t shape_k) {
    // Use getWorkspaceSizeBase to ensure internal state is properly initialized
    // This is critical for BF16 internal quantization path
    // num_problems=1 for single GEMM, top_k=1 for regular gemm (not MOE)
    size_t workspace_size = runner_->getWorkspaceSizeBase(shape_m, shape_n, shape_k, /*num_problems=*/1);
    return workspace_size;
  }

  void configureWorkspace(const TensorView& workspace) {
    auto workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
    runner_->configureWorkspace(workspace_ptr);
  }

  std::unique_ptr<kernels::CutlassFp8BlockScaleGemmRunnerInterface> runner_;
};

tvm::ffi::Module init() {
  auto ptr = tvm::ffi::make_object<Fp8BlockScaleGemmRunner>();
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
