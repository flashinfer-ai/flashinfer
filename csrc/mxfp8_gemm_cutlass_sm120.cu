/*
 * Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/mxfp8_gemm_cutlass_template_sm120.h"
#include "tvm_ffi_utils.h"

using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassMxfp8GemmRunnerSm120;
using flashinfer::gemm::CutlassTileConfigSM120;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;

namespace flashinfer {
namespace gemm {
template class CutlassMxfp8GemmRunnerSm120<__nv_bfloat16>;
template class CutlassMxfp8GemmRunnerSm120<half>;
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace detail {

inline const std::vector<CutlassGemmConfig>& GetMxfp8GemmConfigsSm120() {
  static const std::vector<CutlassGemmConfig> kGlobalConfigs = []() {
    CutlassMxfp8GemmRunnerSm120<__nv_bfloat16> gemmRunner;
    return gemmRunner.getConfigs();
  }();
  return kGlobalConfigs;
}

}  // namespace detail

namespace {

CutlassGemmConfig getMxfp8GemmConfigSm120(int64_t tactic) {
  const auto& globalConfigs = detail::GetMxfp8GemmConfigsSm120();
  TVM_FFI_ICHECK(tactic >= 0 && tactic < static_cast<int64_t>(globalConfigs.size()))
      << "tactic must be between 0 and " << globalConfigs.size();
  return globalConfigs[tactic];
}

template <typename T>
void runGemmSm120(TensorView out, TensorView mat1, TensorView mat2, TensorView mat1Scale,
                  TensorView mat2Scale, int64_t m, int64_t n, int64_t k, int64_t batch_count,
                  CutlassGemmConfig const& gemmConfig, TensorView workspace_buffer) {
  CutlassMxfp8GemmRunnerSm120<T> gemmRunner;

  int64_t const required_workspace_size = gemmRunner.getWorkspaceSize(m, n, k, batch_count);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);

  auto runKernel = [&](void* workspace) {
    gemmRunner.gemm(out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(), mat1Scale.data_ptr(),
                    mat2Scale.data_ptr(), m, n, k, batch_count, gemmConfig,
                    reinterpret_cast<char*>(workspace), required_workspace_size,
                    get_stream(mat1.device()));
  };

  if (provided_workspace_size < required_workspace_size) {
    Tensor new_workspace =
        alloc_tensor({required_workspace_size}, DLDataType{kDLInt, 8, 1}, mat1.device());
    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

constexpr auto FLOAT8_E4M3FN = dl_float8_e4m3fn;
constexpr auto SF_DTYPE = dl_uint8;

// mat1: [B, M, K], FLOAT8_E4M3FN
// mat2: [B, N, K], FLOAT8_E4M3FN (passed as transposed, TensorView sees [N, K])
// out: [B, M, N], fp16/bf16
// mat1Scale/mat2Scale: SF_DTYPE (UE8M0), sfVecSize is always 32
void mxfp8_gemm_sm120_impl(TensorView mat1, TensorView mat2, TensorView mat1Scale,
                           TensorView mat2Scale, TensorView out, TensorView workspace_buffer,
                           int64_t tactic) {
  CHECK_INPUT_AND_TYPE(mat1, FLOAT8_E4M3FN);
  CHECK_INPUT_AND_TYPE(mat2, FLOAT8_E4M3FN);
  CHECK_INPUT_AND_TYPE(mat1Scale, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(mat2Scale, SF_DTYPE);

  int64_t m, n, k, b;
  if (mat1.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 2) << "mat2 must be a matrix";
    TVM_FFI_ICHECK_EQ(mat1.size(1), mat2.size(1))
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(0) << "x" << mat1.size(1)
        << " and " << mat2.size(0) << "x" << mat2.size(1) << ")";
    m = mat1.size(0);
    n = mat2.size(0);
    k = mat2.size(1);
    b = 1;
  } else if (mat1.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 3) << "mat2 must be a batch of matrices";
    TVM_FFI_ICHECK_EQ(mat1.size(0), mat2.size(0)) << "batch size mismatch";
    TVM_FFI_ICHECK_EQ(mat1.size(2), mat2.size(2)) << "K dimension mismatch";
    m = mat1.size(1);
    n = mat2.size(1);
    k = mat2.size(2);
    b = mat1.size(0);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "mat1 must be a matrix or a batch of matrices";
  }

  // One F8_128x4 swizzled block per batch (tile_atom_to_shape_SFA/SFB with L = b), 3D or flat 1D.
  constexpr int64_t sfVecSize = 32;
  auto pad_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  const int64_t k_scales = pad_up((k + sfVecSize - 1) / sfVecSize, 4);
  auto check_scale = [&](TensorView const& scale, int64_t rows, const char* name) {
    const int64_t rows_pad = pad_up(rows, 128);
    if (scale.ndim() == 3) {
      TVM_FFI_ICHECK(scale.size(0) == b && scale.size(1) == rows_pad && scale.size(2) == k_scales)
          << "mxfp8_gemm_sm120: " << name << " shape mismatch, expected [" << b << ", " << rows_pad
          << ", " << k_scales << "], got [" << scale.size(0) << ", " << scale.size(1) << ", "
          << scale.size(2) << "]";
    } else {
      TVM_FFI_ICHECK_EQ(scale.ndim(), 1)
          << "SM120 MXFP8 only supports the F8_128x4 swizzled scale format as a 1D or 3D "
             "[batch, rows_pad, k_pad] tensor. Use SfLayout.layout_128x4 when calling "
             "mxfp8_quantize.";
      TVM_FFI_ICHECK_EQ(scale.size(0), b * rows_pad * k_scales)
          << "mxfp8_gemm_sm120: " << name << " size mismatch, expected " << b * rows_pad * k_scales
          << ", got " << scale.size(0);
    }
  };
  check_scale(mat1Scale, m, "mat1Scale");
  check_scale(mat2Scale, n, "mat2Scale");

  constexpr int alignment = 32;
  TVM_FFI_ICHECK_EQ(k % alignment, 0)
      << "Expected k to be divisible by " << alignment << ", but got mat1 shape: (" << mat1.size(0)
      << "x" << mat1.size(1) << "), k: " << k << ".";
  TVM_FFI_ICHECK_EQ(n % alignment, 0)
      << "Expected n to be divisible by " << alignment << ", but got mat2 shape: (" << mat2.size(0)
      << "x" << mat2.size(1) << ").";

  // Validate out dimensions
  std::vector<int64_t> out_shape =
      mat1.ndim() == 2 ? std::vector<int64_t>{m, n} : std::vector<int64_t>{b, m, n};
  TVM_FFI_ICHECK_EQ(out.ndim(), static_cast<int64_t>(out_shape.size()))
      << "out must have " << out_shape.size() << " dimensions, but got " << out.ndim();
  for (size_t i = 0; i < out_shape.size(); ++i) {
    TVM_FFI_ICHECK_EQ(out.size(i), out_shape[i])
        << "out shape mismatch at dimension " << i << ": expected " << out_shape[i] << ", got "
        << out.size(i);
  }

  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getMxfp8GemmConfigSm120(tactic);

  switch (encode_dlpack_dtype(out.dtype())) {
    case float16_code:
      runGemmSm120<half>(out, mat1, mat2, mat1Scale, mat2Scale, m, n, k, b, config,
                         workspace_buffer);
      break;
    case bfloat16_code:
      runGemmSm120<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, m, n, k, b, config,
                                  workspace_buffer);
      break;
    default:
      TVM_FFI_ICHECK(false) << "out_dtype must be one of fp16/bf16.";
  }
}

}  // namespace

void mxfp8_gemm_sm120(TensorView mat1, TensorView mat2, TensorView mat1Scale, TensorView mat2Scale,
                      TensorView out, TensorView workspace_buffer, int64_t tactic) {
  mxfp8_gemm_sm120_impl(mat1, mat2, mat1Scale, mat2Scale, out, workspace_buffer, tactic);
}

int64_t mxfp8_gemm_tactic_num_sm120() {
  return static_cast<int64_t>(detail::GetMxfp8GemmConfigsSm120().size());
}

}  // namespace torch_ext

// Export under the standard names so _create_cutlass_mxfp8_gemm_module can load this module
// using the same interface as the SM100 module (each arch has its own .so, no symbol conflicts).
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_gemm, torch_ext::mxfp8_gemm_sm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_gemm_tactic_num, torch_ext::mxfp8_gemm_tactic_num_sm120);
