
#include <tvm/ffi/extra/module.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tvm_ffi_utils.h"

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
 * - BF16 + FP8 → BF16 (weight-only quantization)
 * - FP8 + FP8 → BF16 (W8A8 full quantization)
 *
 * @note Output is always BF16
 */
class Fp8BlockScaleGemmRunner : public tvm::ffi::ModuleObj {
 public:
  Fp8BlockScaleGemmRunner() {
    // Instantiate runners for all supported combinations
    runner_bf16_bf16_ = std::make_unique<
        kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();

    runner_bf16_fp8_ = std::make_unique<
        kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>();

    runner_fp8_fp8_ = std::make_unique<
        kernels::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>>();
  }

  ~Fp8BlockScaleGemmRunner() = default;

  const char* type_key() const { return "flashinfer.Fp8BlockScaleGemmRunner"; }
  const char* kind() const final { return "fp8_blockscale_gemm_runner"; }

  Optional<Function> GetFunction(const tvm::ffi::String& name) {
    if (name == "run_gemm") {
      return Function::FromTyped([this](TensorView input, TensorView weight, TensorView output,
                                        Optional<TensorView> scales_a,
                                        Optional<TensorView> scales_b) {
        runGemm(input, weight, output, scales_a, scales_b);
      });
    } else if (name == "fp8_quantize_1x128") {
      return Function::FromTyped([this](TensorView input, TensorView outValueE4M3,
                                        TensorView outScaleFP8SF, bool use_ue8m0) {
        fp8_quantize_1x128(input, outValueE4M3, outScaleFP8SF, use_ue8m0);
      });
    } else if (name == "get_workspace_size") {
      return Function::FromTyped(
          [this](int64_t shape_m, int64_t shape_n, int64_t shape_k) -> int64_t {
            return getWorkspaceSize(shape_m, shape_n, shape_k);
          });
    } else if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configureWorkspace(workspace); });
    }
    return Function(nullptr);
  }

 private:
  /**
   * @brief Runtime dtype dispatch
   */
  kernels::CutlassFp8BlockScaleGemmRunnerInterface* selectRunner(bool input_is_fp8,
                                                                 bool weight_is_fp8) {
    if (!input_is_fp8 && !weight_is_fp8) {
      return runner_bf16_bf16_.get();
    } else if (!input_is_fp8 && weight_is_fp8) {
      return runner_bf16_fp8_.get();
    } else if (input_is_fp8 && weight_is_fp8) {
      return runner_fp8_fp8_.get();  // W8A8
    } else {
      // FP8 input + BF16 weight is not supported by TensorRT-LLM
      return nullptr;
    }
  }

  void runGemm(const TensorView& input, const TensorView& weight, const TensorView& output,
               const Optional<TensorView>& scales_a, const Optional<TensorView>& scales_b) {
    auto stream = get_stream(input.device());

    auto input_ptr = input.data_ptr();
    auto weight_ptr = weight.data_ptr();
    auto output_ptr = output.data_ptr();

    int64_t shape_m = input.size(0);
    int64_t shape_k = input.size(1);
    int64_t shape_n = weight.size(0);

    TVM_FFI_ICHECK(input_ptr != nullptr) << "input is null";
    TVM_FFI_ICHECK(weight_ptr != nullptr) << "weight is null";
    TVM_FFI_ICHECK(output_ptr != nullptr) << "output is null";
    TVM_FFI_ICHECK(shape_k == weight.size(1)) << "K dimension mismatch";
    TVM_FFI_ICHECK(shape_k % 16 == 0) << "N must be a multiple of 16, (K=" << shape_k << ")";
    TVM_FFI_ICHECK(shape_n % 16 == 0) << "N must be a multiple of 16, (N=" << shape_n << ")";

    // Determine dtypes for runner selection
    bool input_is_fp8 = is_fp8_e4m3fn(input.dtype());
    bool weight_is_fp8 = is_fp8_e4m3fn(weight.dtype());

    // Validate scale requirements
    if (input_is_fp8) {
      TVM_FFI_ICHECK(scales_a.has_value() && scales_a.value().data_ptr() != nullptr)
          << "scales_a is required for FP8 input";
    }

    if (weight_is_fp8) {
      TVM_FFI_ICHECK(scales_b.has_value() && scales_b.value().data_ptr() != nullptr)
          << "scales_b is required for FP8 weight";
      // Validate scale shape: should be (N, K/128) for per-token or (N/128, K/128) for per-block
      int64_t expected_scale_k = (shape_k + 127) / 128;
      int64_t scale_dim0 = scales_b.value().size(0);
      int64_t scale_dim1 = scales_b.value().size(1);

      bool is_per_token = (scale_dim0 == shape_n && scale_dim1 == expected_scale_k);
      bool is_per_block = (scale_dim0 == (shape_n + 127) / 128 && scale_dim1 == expected_scale_k);

      TVM_FFI_ICHECK(is_per_token || is_per_block)
          << "scales_b shape mismatch: expected (" << shape_n << ", " << expected_scale_k
          << ") for per-token or (" << ((shape_n + 127) / 128) << ", " << expected_scale_k
          << ") for per-block, got (" << scale_dim0 << ", " << scale_dim1 << ")";
    }

    // Extract scale pointers
    float const* scales_a_ptr = scales_a.has_value()
                                    ? reinterpret_cast<float const*>(scales_a.value().data_ptr())
                                    : nullptr;
    float const* scales_b_ptr = scales_b.has_value()
                                    ? reinterpret_cast<float const*>(scales_b.value().data_ptr())
                                    : nullptr;

    // Select appropriate runner
    auto* runner = selectRunner(input_is_fp8, weight_is_fp8);
    TVM_FFI_ICHECK(runner != nullptr) << "Unsupported dtype combination";
    TVM_FFI_ICHECK(workspace_ != nullptr)
        << "Workspace not configured. Call configure_workspace first.";

    // TensorRT-LLM has two gemm() methods:
    // 1. gemm(void*, ...) - for internal quantization (BF16 inputs)
    // 2. gemm(__nv_fp8_e4m3*, int, __nv_fp8_e4m3*, int, ...) - for pre-quantized FP8 inputs
    if (input_is_fp8 && weight_is_fp8) {
      // W8A8: Use the pre-quantized FP8 path
      auto* fp8_input = reinterpret_cast<__nv_fp8_e4m3*>(input_ptr);
      auto* fp8_weight = reinterpret_cast<__nv_fp8_e4m3*>(weight_ptr);
      auto* bf16_output = reinterpret_cast<__nv_bfloat16*>(output_ptr);

      runner->gemm(fp8_input, shape_k,    // input with leading dimension
                   fp8_weight, shape_k,   // weight with leading dimension
                   bf16_output, shape_n,  // output with leading dimension
                   shape_m, shape_n, shape_k, scales_a_ptr, scales_b_ptr, stream);
    } else {
      // BF16+BF16 or BF16+FP8: Use internal quantization path
      runner->gemm(output_ptr, input_ptr, weight_ptr, shape_m, shape_n, shape_k, stream,
                   scales_a_ptr, scales_b_ptr);
    }
  }

  void fp8_quantize_1x128(const TensorView input, TensorView valueE4M3, TensorView scaleFP8SF,
                          bool use_ue8m0) {
    auto data_shape = input.sizes();
    TVM_FFI_ICHECK_EQ(data_shape.size(), 2) << "input should be 2D tensor.";

    auto const m = data_shape[0];
    auto const n = data_shape[1];

    TVM_FFI_ICHECK_LE(m, std::numeric_limits<int32_t>::max()) << "M must be within int32";
    TVM_FFI_ICHECK_LE(n, std::numeric_limits<int32_t>::max()) << "N must be within int32";
    TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be divisible by 16";

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = get_stream(input.device());

    runner_bf16_fp8_->fp8CS1x128(act_buffer, act_scale_buffer,
                                 reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()), n, m,
                                 stream);
  }

  int64_t getWorkspaceSize(int64_t shape_m, int64_t shape_n, int64_t shape_k) {
    size_t max_size = 0;

    max_size =
        std::max(max_size, runner_bf16_bf16_->getWorkspaceSizeBase(shape_m, shape_n, shape_k, 1));
    max_size =
        std::max(max_size, runner_bf16_fp8_->getWorkspaceSizeBase(shape_m, shape_n, shape_k, 1));
    max_size =
        std::max(max_size, runner_fp8_fp8_->getWorkspaceSizeBase(shape_m, shape_n, shape_k, 1));

    return max_size;
  }

  void configureWorkspace(const TensorView& workspace) {
    auto workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
    workspace_ = workspace_ptr;

    runner_bf16_bf16_->configureWorkspace(workspace_ptr);
    runner_bf16_fp8_->configureWorkspace(workspace_ptr);
    runner_fp8_fp8_->configureWorkspace(workspace_ptr);
  }

  std::unique_ptr<
      kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>
      runner_bf16_bf16_;
  std::unique_ptr<
      kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>
      runner_bf16_fp8_;
  std::unique_ptr<
      kernels::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>>
      runner_fp8_fp8_;

  char* workspace_ = nullptr;
};

tvm::ffi::Module init() {
  auto ptr = tvm::ffi::make_object<Fp8BlockScaleGemmRunner>();
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
