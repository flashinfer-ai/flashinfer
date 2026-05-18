/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "flashinfer/gemm/mxfp8_gemm_cutlass.h"
#include "flashinfer/gemm/mxfp8_gemm_cutlass_template.h"
#include "tvm_ffi_utils.h"

using flashinfer::gemm::ClusterShape;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassMxfp8GemmRunner;
using flashinfer::gemm::CutlassMxfp8GemmRunnerInterface;
using flashinfer::gemm::CutlassTileConfigSM100;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;
using flashinfer::gemm::MXFP8GemmType;

namespace flashinfer {
namespace gemm {
template class CutlassMxfp8GemmRunner<__nv_bfloat16, MXFP8GemmType::W8A8_MXFP8_MXFP8>;
template class CutlassMxfp8GemmRunner<half, MXFP8GemmType::W8A8_MXFP8_MXFP8>;
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace {

CutlassGemmConfig getMxfp8GemmConfig(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  auto getCutlassMxfp8GemmConfigs = []() {
    CutlassMxfp8GemmRunner<__nv_bfloat16, MXFP8GemmType::W8A8_MXFP8_MXFP8> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static std::vector<CutlassGemmConfig> globalConfigs = getCutlassMxfp8GemmConfigs();
  TVM_FFI_ICHECK(tactic >= 0 && tactic < globalConfigs.size())
      << "tactic must be between 0 and " << globalConfigs.size();
  return globalConfigs[tactic];
}

template <typename T>
void runGemm(TensorView out, TensorView mat1, TensorView mat2, TensorView mat1Scale,
             TensorView mat2Scale, int64_t m, int64_t n, int64_t k, int64_t batch_count,
             CutlassGemmConfig const& gemmConfig, TensorView workspace_buffer) {
  CutlassMxfp8GemmRunner<T, MXFP8GemmType::W8A8_MXFP8_MXFP8> gemmRunner;

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

constexpr auto FLOAT8_E4M3FN = dl_float8_e4m3fn;  // float8_e4m3fn
constexpr auto SF_DTYPE = dl_uint8;               // uint8_t

// mat1: [B, M, K], FLOAT8_E4M3FN
// mat2: [B, N, K], FLOAT8_E4M3FN (passed as transposed, TensorView sees [N, K])
// out: [B, M, N], fp16/bf16
// mat1Scale/mat2Scale: SF_DTYPE (UE8M0), sfVecSize is always 32
// B = 1 for GEMM op as a special case
void mxfp8_bmm_impl(TensorView mat1, TensorView mat2, TensorView mat1Scale, TensorView mat2Scale,
                    TensorView out, TensorView workspace_buffer, int64_t tactic) {
  CHECK_INPUT_AND_TYPE(mat1, FLOAT8_E4M3FN);
  CHECK_INPUT_AND_TYPE(mat2, FLOAT8_E4M3FN);

  CHECK_INPUT_AND_TYPE(mat1Scale, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(mat2Scale, SF_DTYPE);

  int64_t m, n, k, b;
  // Scale validation for swizzled (1D) and non-swizzled (2D) layouts.
  if (mat1.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 2) << "mat2 must be a matrix";
    // mat2 is passed as b.T, but TensorView reads underlying storage as [N, K]
    // mat1 is [M, K]
    // Check: mat1.size(1) == mat2.size(1) (both should be K)
    TVM_FFI_ICHECK_EQ(mat1.size(1), mat2.size(1))
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(0) << "x" << mat1.size(1)
        << " and " << mat2.size(0) << "x" << mat2.size(1) << ")";
    m = mat1.size(0);
    n = mat2.size(0);  // mat2 is [N, K] in storage
    k = mat2.size(1);  // mat2 is [N, K] in storage
    b = 1;
  } else if (mat1.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 3) << "mat2 must be a batch of matrices";
    TVM_FFI_ICHECK_EQ(mat1.size(0), mat2.size(0)) << "mat1 and mat2 must have the same batch size ("
                                                  << mat1.size(0) << " and " << mat2.size(0) << ")";
    // mat2 is passed as b.T, but TensorView reads underlying storage as [B, N, K]
    // mat1 is [B, M, K]
    // Check: mat1.size(2) == mat2.size(2) (both should be K)
    TVM_FFI_ICHECK_EQ(mat1.size(2), mat2.size(2))
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(1) << "x" << mat1.size(2)
        << " and " << mat2.size(1) << "x" << mat2.size(2) << ")";
    m = mat1.size(1);
    n = mat2.size(1);  // mat2 is [B, N, K] in storage
    k = mat2.size(2);  // mat2 is [B, N, K] in storage
    b = mat1.size(0);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "mat1 must be a matrix or a batch of matrices";
  }

  constexpr int64_t sfVecSize = 32;  // MXFP8 block size
  auto scale_len = [&](int64_t dim) { return (dim + sfVecSize - 1) / sfVecSize; };
  auto swizzled_len = [&](int64_t rows, int64_t cols) {
    auto pad_up = [](int64_t value, int64_t multiple) {
      return (value + multiple - 1) / multiple * multiple;
    };
    int64_t padded_rows = pad_up(rows, 128);
    int64_t padded_cols = pad_up(cols, 4);
    return padded_rows * padded_cols;
  };

  if (mat1.ndim() == 2) {
    const int64_t k_scales = scale_len(k);
    if (mat1Scale.ndim() == 1) {
      int64_t expected = swizzled_len(m, k_scales);
      TVM_FFI_ICHECK_EQ(mat1Scale.size(0), expected)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << expected << ", got "
          << mat1Scale.size(0);
    } else {
      TVM_FFI_ICHECK_EQ(mat1Scale.ndim(), 2)
          << "mxfp8_bmm_impl: mat1Scale must be 1D (swizzled) or 2D (non-swizzled), got "
          << mat1Scale.ndim();
      TVM_FFI_ICHECK_EQ(mat1Scale.size(0), m)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << m << ", got "
          << mat1Scale.size(0);
      TVM_FFI_ICHECK_EQ(mat1Scale.size(1), k_scales)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << k_scales << ", got "
          << mat1Scale.size(1);
    }

    if (mat2Scale.ndim() == 1) {
      int64_t expected = swizzled_len(n, k_scales);
      TVM_FFI_ICHECK_EQ(mat2Scale.size(0), expected)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << expected << ", got "
          << mat2Scale.size(0);
    } else {
      TVM_FFI_ICHECK_EQ(mat2Scale.ndim(), 2)
          << "mxfp8_bmm_impl: mat2Scale must be 1D (swizzled) or 2D (non-swizzled), got "
          << mat2Scale.ndim();
      TVM_FFI_ICHECK_EQ(mat2Scale.size(0), n)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << n << ", got "
          << mat2Scale.size(0);
      TVM_FFI_ICHECK_EQ(mat2Scale.size(1), k_scales)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << k_scales << ", got "
          << mat2Scale.size(1);
    }
  } else {
    const int64_t k_scales = scale_len(k);
    if (mat1Scale.ndim() == 1) {
      int64_t expected = swizzled_len(b * m, k_scales);
      TVM_FFI_ICHECK_EQ(mat1Scale.size(0), expected)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << expected << ", got "
          << mat1Scale.size(0);
    } else if (mat1Scale.ndim() == 2) {
      TVM_FFI_ICHECK_EQ(mat1Scale.size(1), k_scales)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << k_scales << ", got "
          << mat1Scale.size(1);
      TVM_FFI_ICHECK_EQ(mat1Scale.size(0), b * m)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << (b * m) << ", got "
          << mat1Scale.size(0);
    } else {
      TVM_FFI_ICHECK_EQ(mat1Scale.ndim(), 3)
          << "mxfp8_bmm_impl: mat1Scale must be 1D (swizzled), 2D (flattened), or 3D "
             "(batched), got "
          << mat1Scale.ndim();
      TVM_FFI_ICHECK_EQ(mat1Scale.size(0), b)
          << "mxfp8_bmm_impl: mat1Scale batch size mismatch, expected " << b << ", got "
          << mat1Scale.size(0);
      TVM_FFI_ICHECK_EQ(mat1Scale.size(1), m)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << m << ", got "
          << mat1Scale.size(1);
      TVM_FFI_ICHECK_EQ(mat1Scale.size(2), k_scales)
          << "mxfp8_bmm_impl: mat1Scale size mismatch, expected " << k_scales << ", got "
          << mat1Scale.size(2);
    }

    if (mat2Scale.ndim() == 1) {
      int64_t expected = swizzled_len(b * n, k_scales);
      TVM_FFI_ICHECK_EQ(mat2Scale.size(0), expected)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << expected << ", got "
          << mat2Scale.size(0);
    } else if (mat2Scale.ndim() == 2) {
      TVM_FFI_ICHECK_EQ(mat2Scale.size(1), k_scales)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << k_scales << ", got "
          << mat2Scale.size(1);
      TVM_FFI_ICHECK_EQ(mat2Scale.size(0), b * n)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << (b * n) << ", got "
          << mat2Scale.size(0);
    } else {
      TVM_FFI_ICHECK_EQ(mat2Scale.ndim(), 3)
          << "mxfp8_bmm_impl: mat2Scale must be 1D (swizzled), 2D (flattened), or 3D "
             "(batched), got "
          << mat2Scale.ndim();
      TVM_FFI_ICHECK_EQ(mat2Scale.size(0), b)
          << "mxfp8_bmm_impl: mat2Scale batch size mismatch, expected " << b << ", got "
          << mat2Scale.size(0);
      TVM_FFI_ICHECK_EQ(mat2Scale.size(1), n)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << n << ", got "
          << mat2Scale.size(1);
      TVM_FFI_ICHECK_EQ(mat2Scale.size(2), k_scales)
          << "mxfp8_bmm_impl: mat2Scale size mismatch, expected " << k_scales << ", got "
          << mat2Scale.size(2);
    }
  }

  // No heuristic for now, we rely on the autotuner to select the best tactic.
  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getMxfp8GemmConfig(m, n, k, tactic);

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
  TVM_FFI_ICHECK_EQ(out.ndim(), out_shape.size())
      << "out must have " << out_shape.size() << " dimensions, but got " << out.ndim();
  for (int i = 0; i < out_shape.size(); ++i) {
    TVM_FFI_ICHECK_EQ(out.size(i), out_shape[i])
        << "out shape mismatch at dimension " << i << ": expected " << out_shape[i] << ", got "
        << out.size(i);
  }

  switch (encode_dlpack_dtype(out.dtype())) {
    case float16_code:
      runGemm<half>(out, mat1, mat2, mat1Scale, mat2Scale, m, n, k, b, config, workspace_buffer);
      break;
    case bfloat16_code:
      runGemm<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, m, n, k, b, config,
                             workspace_buffer);
      break;
    default:
      TVM_FFI_ICHECK(false) << "out_dtype must be one of fp16/bf16.";
  }
}

}  // namespace

void mxfp8_gemm(TensorView mat1, TensorView mat2, TensorView mat1Scale, TensorView mat2Scale,
                TensorView out, TensorView workspace_buffer, int64_t tactic) {
  mxfp8_bmm_impl(mat1, mat2, mat1Scale, mat2Scale, out, workspace_buffer, tactic);
}

int64_t mxfp8_gemm_tactic_num() {
  auto getCutlassConfigs = []() {
    CutlassMxfp8GemmRunner<__nv_bfloat16, MXFP8GemmType::W8A8_MXFP8_MXFP8> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static int64_t totalTactics = getCutlassConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_gemm, torch_ext::mxfp8_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mxfp8_gemm_tactic_num, torch_ext::mxfp8_gemm_tactic_num);
