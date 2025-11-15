
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
using tvm::ffi::TensorView;

#ifdef FLASHINFER_ENABLE_FP8_E4M3
inline bool is_fp8_e4m3fn(DLDataType dtype) {
  return encode_dlpack_dtype(dtype) == float8_e4m3fn_code;
}
#else
inline bool is_fp8_e4m3fn(DLDataType) { return false; }
#endif

/**
 * @brief FP8 Block-Scale GEMM binding for SM90
 * 
 * Supports:
 * - BF16 + BF16 → BF16
 * - BF16 + FP8 → BF16 
 * 
 * @note Output is BF16
 */
class Fp8BlockScaleGemmRunner : public tvm::ffi::ModuleObj {
 public:
  Fp8BlockScaleGemmRunner() {
    // Instantiate runners
    runner_bf16_bf16_ = std::make_unique<kernels::CutlassFp8BlockScaleGemmRunner<
        __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();
    
    runner_bf16_fp8_ = std::make_unique<kernels::CutlassFp8BlockScaleGemmRunner<
        __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>();
  }

  ~Fp8BlockScaleGemmRunner() = default;

  const char* type_key() const { return "flashinfer.Fp8BlockScaleGemmRunner"; }
  const char* kind() const final { return "fp8_blockscale_gemm_runner"; }

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
      return Function::FromTyped([this](TensorView workspace) {
            configureWorkspace(workspace);
          });
    }
    return Function(nullptr);
  }

 private:
  /**
   * @brief Runtime dtype dispatch
   */
  kernels::CutlassFp8BlockScaleGemmRunnerInterface* selectRunner(
      bool input_is_fp8, bool weight_is_fp8) {
    
    if (!input_is_fp8 && !weight_is_fp8) {
      return runner_bf16_bf16_.get();
    } else if (!input_is_fp8 && weight_is_fp8) {
      return runner_bf16_fp8_.get();
    } else {
      return nullptr; 
    }
  }

  void runGemm(const TensorView& input, const TensorView& weight, const TensorView& output,
              const Optional<TensorView>& scales_a, const Optional<TensorView>& scales_b) {
    auto stream = get_stream(input.device());

    // Extract tensor info
    auto input_ptr = input.data_ptr();
    auto weight_ptr = weight.data_ptr();
    auto output_ptr = output.data_ptr();
    
    int shape_m = input.size(0);
    int shape_k = input.size(1);
    int shape_n = weight.size(0);
    
    // Sanity checks (defense against Python bugs)
    TVM_FFI_ICHECK(input_ptr != nullptr) << "input is null";
    TVM_FFI_ICHECK(weight_ptr != nullptr) << "weight is null";
    TVM_FFI_ICHECK(output_ptr != nullptr) << "output is null";
    TVM_FFI_ICHECK(shape_k == weight.size(1)) << "K dimension mismatch";
    
    // Determine dtypes for runner selection
    bool input_is_fp8 = is_fp8_e4m3fn(input.dtype());
    bool weight_is_fp8 = is_fp8_e4m3fn(weight.dtype());
    
    // Extract scale pointers (nullptr if not provided)
    float const* scales_a_ptr = scales_a.has_value() 
        ? reinterpret_cast<float const*>(scales_a.value().data_ptr()) 
        : nullptr;
    float const* scales_b_ptr = scales_b.has_value() 
        ? reinterpret_cast<float const*>(scales_b.value().data_ptr()) 
        : nullptr;
    
    // Select appropriate runner
    auto* runner = selectRunner(input_is_fp8, weight_is_fp8);
    TVM_FFI_ICHECK(runner != nullptr) << "Unsupported dtype combination";
    
    // Ensure workspace is configured (defensive check)
    TVM_FFI_ICHECK(workspace_ != nullptr) << "Workspace not configured. Call configure_workspace first.";
    
    // Call kernel
    runner->gemm(output_ptr, input_ptr, weight_ptr, shape_m, shape_n, shape_k,
                 stream, scales_a_ptr, scales_b_ptr);
  }

  int64_t getWorkspaceSize(int64_t shape_m, int64_t shape_n, int64_t shape_k) {
    size_t max_size = 0;
    
    max_size = std::max(max_size, 
        runner_bf16_bf16_->getWorkspaceSizeBase(shape_m, shape_n, shape_k, 1));
    max_size = std::max(max_size, 
        runner_bf16_fp8_->getWorkspaceSizeBase(shape_m, shape_n, shape_k, 1));
    
    return max_size;
  }

  void configureWorkspace(const TensorView& workspace) {
    auto workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
    workspace_ = workspace_ptr;
    
    runner_bf16_bf16_->configureWorkspace(workspace_ptr);
    runner_bf16_fp8_->configureWorkspace(workspace_ptr);
  }

  std::unique_ptr<kernels::CutlassFp8BlockScaleGemmRunner<
      __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>> runner_bf16_bf16_;
  std::unique_ptr<kernels::CutlassFp8BlockScaleGemmRunner<
      __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>> runner_bf16_fp8_;
  
  char* workspace_ = nullptr;
};

tvm::ffi::Module init() {
  auto ptr = tvm::ffi::make_object<Fp8BlockScaleGemmRunner>();
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);