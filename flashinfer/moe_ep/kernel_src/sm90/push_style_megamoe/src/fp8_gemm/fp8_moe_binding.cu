/*
 * Copyright (c) 2026 FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "fp8_moe_launcher.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::sm90_push_fp8 {

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

inline void check_shape_scalars(char const* name, int64_t shape_n, int64_t shape_k) {
  TVM_FFI_ICHECK(shape_n > 0 && shape_k > 0)
      << name << ": N and K must be positive, got N=" << shape_n << " K=" << shape_k;
}

inline size_t check_offsets(char const* name, TensorView offsets, TensorView input) {
  CHECK_INPUT(offsets);
  auto const dtype = offsets.dtype();
  TVM_FFI_ICHECK(dtype.code == kDLInt && dtype.bits == 64 && dtype.lanes == 1)
      << name << ": offsets must be int64";
  TVM_FFI_ICHECK(offsets.ndim() == 1 && offsets.size(0) >= 2)
      << name << ": offsets must be a 1D tensor with at least two entries";
  CHECK_DEVICE(offsets, input);
  return static_cast<size_t>(offsets.size(0) - 1);
}

__global__ void offsets_preflight_kernel(int64_t const* offsets, int num_problems,
                                         int64_t row_capacity, int64_t input_capacity) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t prev = offsets[0];
  bool bad = prev != 0;
  for (int group = 1; group <= num_problems && !bad; ++group) {
    int64_t const current = offsets[group];
    bad = current < prev;
    prev = current;
  }
  if (bad || prev > row_capacity || prev > input_capacity) {
    printf(
        "sm90_push_fp8_moe: bad offsets (first != 0, decreasing, or final=%lld exceeds "
        "capacity min(D=%lld, A=%lld))\n",
        static_cast<long long>(prev), static_cast<long long>(row_capacity),
        static_cast<long long>(input_capacity));
    asm volatile("trap;");
  }
}

class Fp8MoeGemmRunner : public tvm::ffi::ModuleObj {
 public:
  char const* type_key() const { return "flashinfer.Sm90PushFp8MoeGemmRunner"; }
  char const* kind() const final { return "sm90_push_fp8_moe_gemm_runner"; }

  Optional<Function> GetFunction(tvm::ffi::String const& name) final {
    if (name == "get_moe_workspace_size") {
      return Function::FromTyped([this](int64_t expected_m, int64_t max_rows, int64_t shape_n,
                                        int64_t shape_k, int64_t num_problems, bool a_is_fp8,
                                        bool b_is_fp8) -> int64_t {
        return get_workspace_size(expected_m, max_rows, shape_n, shape_k, num_problems, a_is_fp8,
                                  b_is_fp8);
      });
    }
    if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configure_workspace(workspace); });
    }
    if (name == "get_deepgemm_cache_dir") {
      return Function::FromTyped([]() -> tvm::ffi::String {
        return tvm::ffi::String(deep_gemm::jit::getCacheDir().string());
      });
    }
    if (name == "get_deepgemm_nvcc_compiler") {
      return Function::FromTyped(
          []() -> tvm::ffi::String { return tvm::ffi::String(deep_gemm::jit::getNvccCompiler()); });
    }
    if (name == "is_deepgemm_jit_enabled") {
      return Function::FromTyped([]() -> bool {
        return tensorrt_llm::kernels::fp8_blockscale_gemm::getDeepGemmEnabled();
      });
    }
    if (name == "is_moe_gemm_jit_cache_ready") {
      return Function::FromTyped([this](int64_t shape_n, int64_t shape_k) -> bool {
        check_jit_shape("is_moe_gemm_jit_cache_ready", shape_n, shape_k);
        return fp8_moe_gemm_jit_cache_ready(expected_m_, static_cast<int>(shape_n),
                                            static_cast<int>(shape_k),
                                            static_cast<int>(num_problems_));
      });
    }
    if (name == "is_moe_gemm_fc1_fused_jit_cache_ready") {
      return Function::FromTyped([this](int64_t shape_n, int64_t shape_k) -> bool {
        check_jit_shape("is_moe_gemm_fc1_fused_jit_cache_ready", shape_n, shape_k);
        return fp8_moe_fc1_fused_jit_cache_ready(expected_m_, static_cast<int>(shape_n),
                                                 static_cast<int>(shape_k),
                                                 static_cast<int>(num_problems_));
      });
    }
    if (name == "moe_gemm") {
      return Function::FromTyped([this](TensorView output, TensorView input, TensorView weight,
                                        TensorView offsets, int64_t shape_n, int64_t shape_k,
                                        Optional<TensorView> scales_a,
                                        Optional<TensorView> scales_b, bool trusted_offsets) {
        run_moe_gemm(output, input, weight, offsets, shape_n, shape_k, scales_a, scales_b,
                     trusted_offsets);
      });
    }
    if (name == "moe_gemm_fc1_fused") {
      return Function::FromTyped(
          [this](TensorView output_fp8, TensorView output_scales, TensorView input,
                 TensorView weight, TensorView offsets, int64_t shape_n_interleaved,
                 int64_t shape_k, TensorView scales_a, TensorView scales_b, bool trusted_offsets) {
            run_fc1_fused(output_fp8, output_scales, input, weight, offsets, shape_n_interleaved,
                          shape_k, scales_a, scales_b, trusted_offsets);
          });
    }
    return Function(nullptr);
  }

 private:
  int64_t get_workspace_size(int64_t expected_m, int64_t max_rows, int64_t shape_n, int64_t shape_k,
                             int64_t num_problems, bool a_is_fp8, bool b_is_fp8) {
    TVM_FFI_ICHECK(expected_m > 0 && max_rows > 0 && shape_n > 0 && shape_k > 0 && num_problems > 0)
        << "get_moe_workspace_size: expected_m, max_rows, N, K, and num_problems must be positive";
    TVM_FFI_ICHECK(a_is_fp8 && b_is_fp8)
        << "get_moe_workspace_size: the private SM90 push runner accepts FP8 A and FP8 B";
    TVM_FFI_ICHECK_LE(num_problems, std::numeric_limits<int>::max())
        << "get_moe_workspace_size: num_problems exceeds int32";
    int64_t const scale_padding = num_problems * int64_t{31};
    TVM_FFI_ICHECK_LE(expected_m, std::numeric_limits<int>::max())
        << "get_moe_workspace_size: expected_m exceeds int32";
    TVM_FFI_ICHECK_LE(max_rows, std::numeric_limits<int>::max() - scale_padding)
        << "get_moe_workspace_size: padded max_rows exceeds int32";
    TVM_FFI_ICHECK_LE(shape_n, std::numeric_limits<int>::max())
        << "get_moe_workspace_size: N exceeds int32";
    TVM_FFI_ICHECK_LE(shape_k, std::numeric_limits<int>::max())
        << "get_moe_workspace_size: K exceeds int32";
    TVM_FFI_ICHECK_EQ(shape_n % 128, 0) << "get_moe_workspace_size: N must be divisible by 128";
    TVM_FFI_ICHECK_EQ(shape_k % 128, 0) << "get_moe_workspace_size: K must be divisible by 128";

    expected_m_ = expected_m;
    max_rows_ = ceil_div(max_rows, int64_t{4}) * 4;
    padded_rows_ = deep_gemm::compute_padded_offset(max_rows, num_problems);
    max_shape_n_ = shape_n;
    max_shape_k_ = shape_k;
    num_problems_ = num_problems;
    queried_ = true;
    configured_ = false;
    workspace_bytes_ = 0;
    return 0;
  }

  void configure_workspace(TensorView workspace) {
    TVM_FFI_ICHECK(queried_)
        << "configure_workspace: call get_moe_workspace_size before configuring the runner";
    CHECK_INPUT(workspace);
    auto const dtype = workspace.dtype();
    TVM_FFI_ICHECK(dtype.code == kDLUInt && dtype.bits == 8 && dtype.lanes == 1)
        << "configure_workspace: workspace must be uint8";
    workspace_device_ = workspace.device();
    workspace_bytes_ = workspace.numel();
    configured_ = true;
  }

  void check_jit_shape(char const* name, int64_t shape_n, int64_t shape_k) const {
    TVM_FFI_ICHECK(queried_) << name << ": query this runner before inspecting its JIT cache";
    TVM_FFI_ICHECK(shape_n > 0 && shape_k > 0) << name << ": N and K must be positive";
    TVM_FFI_ICHECK_LE(shape_n, max_shape_n_) << name << ": N exceeds the workspace query";
    TVM_FFI_ICHECK_LE(shape_k, max_shape_k_) << name << ": K exceeds the workspace query";
    TVM_FFI_ICHECK_LE(shape_n, std::numeric_limits<int>::max()) << name << ": N exceeds int32";
    TVM_FFI_ICHECK_LE(shape_k, std::numeric_limits<int>::max()) << name << ": K exceeds int32";
  }

  void check_contract(char const* name, size_t num_problems, int64_t shape_n, int64_t shape_k,
                      TensorView input) const {
    TVM_FFI_ICHECK(queried_ && configured_)
        << name << ": query and configure this runner before launching GEMM";
    TVM_FFI_ICHECK_EQ(static_cast<int64_t>(num_problems), num_problems_)
        << name << ": runtime group count differs from the workspace query";
    TVM_FFI_ICHECK_LE(shape_n, max_shape_n_) << name << ": N exceeds the workspace query";
    TVM_FFI_ICHECK_LE(shape_k, max_shape_k_) << name << ": K exceeds the workspace query";
    TVM_FFI_ICHECK(workspace_device_.device_type == input.device().device_type &&
                   workspace_device_.device_id == input.device().device_id)
        << name << ": workspace and input must be on the same CUDA device";
  }

  static void check_weight_shape(char const* name, TensorView weight, size_t num_problems,
                                 int64_t shape_n, int64_t shape_k) {
    if (weight.ndim() == 3) {
      TVM_FFI_ICHECK(weight.size(0) == static_cast<int64_t>(num_problems) &&
                     weight.size(1) == shape_n && weight.size(2) == shape_k)
          << name << ": 3D weight must be (G, N, K)";
    } else {
      CHECK_DIM(2, weight);
      TVM_FFI_ICHECK(weight.size(0) == static_cast<int64_t>(num_problems) * shape_n &&
                     weight.size(1) == shape_k)
          << name << ": 2D weight must be (G*N, K)";
    }
  }

  static void check_fp8_inputs(char const* name, TensorView input, TensorView weight,
                               int64_t shape_k) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_DIM(2, input);
    TVM_FFI_ICHECK(is_fp8_e4m3fn(input.dtype()) && is_fp8_e4m3fn(weight.dtype()))
        << name << ": input and weight must be float8_e4m3fn";
    TVM_FFI_ICHECK_EQ(input.size(1), shape_k) << name << ": input K does not match shape_k";
    CHECK_DEVICE(weight, input);
  }

  void run_moe_gemm(TensorView output, TensorView input, TensorView weight, TensorView offsets,
                    int64_t shape_n, int64_t shape_k, Optional<TensorView> scales_a,
                    Optional<TensorView> scales_b, bool trusted_offsets) {
    check_shape_scalars("moe_gemm", shape_n, shape_k);
    check_fp8_inputs("moe_gemm", input, weight, shape_k);
    size_t const num_problems = check_offsets("moe_gemm", offsets, input);
    check_weight_shape("moe_gemm", weight, num_problems, shape_n, shape_k);
    check_contract("moe_gemm", num_problems, shape_n, shape_k, input);

    CHECK_INPUT(output);
    CHECK_DIM(2, output);
    CHECK_DEVICE(output, input);
    TVM_FFI_ICHECK(output.dtype() == dl_bfloat16) << "moe_gemm: output must be bfloat16";
    TVM_FFI_ICHECK_EQ(output.size(1), shape_n) << "moe_gemm: output N does not match shape_n";
    TVM_FFI_ICHECK_GE(input.size(0), max_rows_)
        << "moe_gemm: input rows do not cover the frozen TMA declaration";
    TVM_FFI_ICHECK_GE(output.size(0), max_rows_)
        << "moe_gemm: output rows do not cover the frozen TMA declaration";
    TVM_FFI_ICHECK(scales_a.has_value() && scales_b.has_value())
        << "moe_gemm: FP8 input and weight require scales_a and scales_b";
    auto scale_a = scales_a.value();
    auto scale_b = scales_b.value();
    CHECK_INPUT(scale_a);
    CHECK_INPUT(scale_b);
    CHECK_DEVICE(scale_a, input);
    CHECK_DEVICE(scale_b, input);
    TVM_FFI_ICHECK(scale_a.dtype().code == kDLFloat && scale_a.dtype().bits == 32)
        << "moe_gemm: scales_a must be float32";
    TVM_FFI_ICHECK(scale_b.dtype().code == kDLFloat && scale_b.dtype().bits == 32)
        << "moe_gemm: scales_b must be float32";
    TVM_FFI_ICHECK_GE(scale_a.numel(), (shape_k / 128) * padded_rows_)
        << "moe_gemm: scales_a does not cover the frozen padded row stride";
    TVM_FFI_ICHECK_GE(scale_b.numel(),
                      static_cast<int64_t>(num_problems) * (shape_n / 128) * (shape_k / 128))
        << "moe_gemm: scales_b is smaller than (G, N/128, K/128)";

    auto const stream = get_stream(input.device());
    if (!trusted_offsets) {
      offsets_preflight_kernel<<<1, 32, 0, stream>>>(
          static_cast<int64_t const*>(offsets.data_ptr()), static_cast<int>(num_problems),
          output.size(0), std::min(input.size(0), max_rows_));
      auto const status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(status, cudaSuccess)
          << "moe_gemm: offsets preflight launch failed: " << cudaGetErrorString(status);
    }
    fp8_moe_gemm(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr()),
                 static_cast<float*>(scale_a.data_ptr()),
                 reinterpret_cast<__nv_fp8_e4m3*>(weight.data_ptr()),
                 static_cast<float*>(scale_b.data_ptr()),
                 reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                 static_cast<int64_t const*>(offsets.data_ptr()), static_cast<int>(num_problems),
                 expected_m_, max_rows_, padded_rows_, static_cast<int>(shape_n),
                 static_cast<int>(shape_k), stream);
  }

  void run_fc1_fused(TensorView output_fp8, TensorView output_scales, TensorView input,
                     TensorView weight, TensorView offsets, int64_t shape_n_interleaved,
                     int64_t shape_k, TensorView scales_a, TensorView scales_b,
                     bool trusted_offsets) {
    check_shape_scalars("moe_gemm_fc1_fused", shape_n_interleaved, shape_k);
    TVM_FFI_ICHECK_EQ(shape_n_interleaved % 256, 0)
        << "moe_gemm_fc1_fused: interleaved N must be divisible by 256";
    check_fp8_inputs("moe_gemm_fc1_fused", input, weight, shape_k);
    size_t const num_problems = check_offsets("moe_gemm_fc1_fused", offsets, input);
    check_weight_shape("moe_gemm_fc1_fused", weight, num_problems, shape_n_interleaved, shape_k);
    check_contract("moe_gemm_fc1_fused", num_problems, shape_n_interleaved, shape_k, input);

    CHECK_INPUT(output_fp8);
    CHECK_INPUT(output_scales);
    CHECK_INPUT(scales_a);
    CHECK_INPUT(scales_b);
    CHECK_DIM(2, output_fp8);
    CHECK_DEVICE(output_fp8, input);
    CHECK_DEVICE(output_scales, input);
    CHECK_DEVICE(scales_a, input);
    CHECK_DEVICE(scales_b, input);
    auto const output_dtype = output_fp8.dtype();
    TVM_FFI_ICHECK((output_dtype.code == kDLUInt && output_dtype.bits == 8) ||
                   is_fp8_e4m3fn(output_dtype))
        << "moe_gemm_fc1_fused: output must be uint8 or float8_e4m3fn";
    int64_t const shape_i = shape_n_interleaved / 2;
    TVM_FFI_ICHECK_EQ(output_fp8.size(1), shape_i)
        << "moe_gemm_fc1_fused: output width must equal N/2";
    TVM_FFI_ICHECK_GE(input.size(0), max_rows_)
        << "moe_gemm_fc1_fused: input rows do not cover the frozen TMA declaration";
    TVM_FFI_ICHECK_GE(output_fp8.size(0), input.size(0))
        << "moe_gemm_fc1_fused: output rows must cover the input row capacity";
    for (auto const& item : {std::pair{"output_scales", output_scales},
                             std::pair{"scales_a", scales_a}, std::pair{"scales_b", scales_b}}) {
      TVM_FFI_ICHECK(item.second.dtype().code == kDLFloat && item.second.dtype().bits == 32)
          << "moe_gemm_fc1_fused: " << item.first << " must be float32";
    }
    TVM_FFI_ICHECK_GE(output_scales.numel(), (shape_i / 128) * padded_rows_)
        << "moe_gemm_fc1_fused: output scales do not cover the padded row stride";
    TVM_FFI_ICHECK_GE(scales_a.numel(), (shape_k / 128) * padded_rows_)
        << "moe_gemm_fc1_fused: scales_a does not cover the padded row stride";
    TVM_FFI_ICHECK_GE(scales_b.numel(), static_cast<int64_t>(num_problems) *
                                            (shape_n_interleaved / 128) * (shape_k / 128))
        << "moe_gemm_fc1_fused: scales_b is smaller than (G, 2I/128, K/128)";

    auto const stream = get_stream(input.device());
    if (!trusted_offsets) {
      offsets_preflight_kernel<<<1, 32, 0, stream>>>(
          static_cast<int64_t const*>(offsets.data_ptr()), static_cast<int>(num_problems),
          output_fp8.size(0), std::min(input.size(0), max_rows_));
      auto const status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(status, cudaSuccess)
          << "moe_gemm_fc1_fused: offsets preflight launch failed: " << cudaGetErrorString(status);
    }
    fp8_moe_fc1_fused(reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr()),
                      static_cast<float*>(scales_a.data_ptr()),
                      reinterpret_cast<__nv_fp8_e4m3*>(weight.data_ptr()),
                      static_cast<float*>(scales_b.data_ptr()),
                      reinterpret_cast<__nv_fp8_e4m3*>(output_fp8.data_ptr()), output_fp8.size(0),
                      static_cast<float*>(output_scales.data_ptr()),
                      static_cast<int64_t const*>(offsets.data_ptr()),
                      static_cast<int>(num_problems), expected_m_, max_rows_, padded_rows_,
                      static_cast<int>(shape_n_interleaved), static_cast<int>(shape_k), stream);
  }

  bool queried_ = false;
  bool configured_ = false;
  int64_t workspace_bytes_ = 0;
  int64_t expected_m_ = 0;
  int64_t max_rows_ = 0;
  int64_t padded_rows_ = 0;
  int64_t max_shape_n_ = 0;
  int64_t max_shape_k_ = 0;
  int64_t num_problems_ = 0;
  DLDevice workspace_device_{kDLCPU, 0};
};

tvm::ffi::Module init_moe() { return tvm::ffi::Module(tvm::ffi::make_object<Fp8MoeGemmRunner>()); }

void set_deepgemm_jit_include_dirs(tvm::ffi::Array<tvm::ffi::String> include_dirs) {
  std::vector<std::filesystem::path> dirs;
  dirs.reserve(include_dirs.size());
  for (auto const& dir : include_dirs) dirs.emplace_back(std::string(dir));
  deep_gemm::jit::Compiler::setIncludeDirs(dirs);
}

}  // namespace flashinfer::sm90_push_fp8

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_moe, flashinfer::sm90_push_fp8::init_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_deepgemm_jit_include_dirs,
                              flashinfer::sm90_push_fp8::set_deepgemm_jit_include_dirs);
