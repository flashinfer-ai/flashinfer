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
#include "flashinfer/gemm/fp8_gemm_cutlass.h"
#include "flashinfer/gemm/fp8_gemm_cutlass_template.h"
#include "tvm_ffi_utils.h"

using flashinfer::gemm::ClusterShape;
using flashinfer::gemm::CutlassFp8GemmRunner;
using flashinfer::gemm::CutlassFp8GemmRunnerInterface;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM100;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;

namespace flashinfer {
namespace gemm {
template class CutlassFp8GemmRunner<__nv_bfloat16>;
template class CutlassFp8GemmRunner<half>;
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace {

CutlassGemmConfig getFp8GemmConfig(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  auto getCutlassFp8GemmConfigs = []() {
    CutlassFp8GemmRunner<__nv_bfloat16> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static std::vector<CutlassGemmConfig> globalConfigs = getCutlassFp8GemmConfigs();
  TVM_FFI_ICHECK(tactic >= 0 && tactic < globalConfigs.size())
      << "tactic must be between 0 and " << globalConfigs.size();
  return globalConfigs[tactic];
}

template <typename T>
void runGemm(TensorView out, TensorView mat1, TensorView mat2, TensorView scale_a,
             TensorView scale_b, int64_t m, int64_t n, int64_t k, int64_t b,
             CutlassGemmConfig const& gemmConfig, TensorView workspace_buffer) {
  CutlassFp8GemmRunner<T> gemmRunner;

  int64_t const required_workspace_size = gemmRunner.getWorkspaceSize(m, n, k);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);

  auto runKernel = [&](void* workspace) {
    gemmRunner.gemm(
        static_cast<__nv_fp8_e4m3*>(mat1.data_ptr()), static_cast<__nv_fp8_e4m3*>(mat2.data_ptr()),
        static_cast<float*>(scale_a.data_ptr()), static_cast<float*>(scale_b.data_ptr()),
        out.data_ptr(), m, n, k, b, gemmConfig, static_cast<char*>(workspace),
        required_workspace_size, get_stream(mat1.device()));
  };

  if (provided_workspace_size < required_workspace_size) {
    Tensor new_workspace =
        alloc_tensor({required_workspace_size}, DLDataType{kDLInt, 8, 1}, mat1.device());

    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

void fp8_bmm_impl(TensorView mat1, TensorView mat2, TensorView scale_a, TensorView scale_b,
                  TensorView out, TensorView workspace_buffer, int64_t tactic) {
  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scale_a);
  CHECK_INPUT(scale_b);

  int mat2_k_scale = 1;

  int64_t m, n, k, b;
  if (mat1.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 2) << "mat2 must be a matrix";
    TVM_FFI_ICHECK_EQ(mat1.size(1), mat2.size(1) * mat2_k_scale)
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(0) << "x" << mat1.size(1)
        << " and " << mat2.size(0) << "x" << mat2.size(1) << ")";
    m = mat1.size(0);
    n = mat2.size(0);
    k = mat2.size(1);
    b = 1;
  } else if (mat1.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 3) << "mat2 must be a batch of matrices";
    TVM_FFI_ICHECK_EQ(mat1.size(0), mat2.size(0)) << "mat1 and mat2 must have the same batch size ("
                                                  << mat1.size(0) << " and " << mat2.size(0) << ")";
    TVM_FFI_ICHECK_EQ(mat1.size(2), mat2.size(2) * mat2_k_scale)
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(1) << "x" << mat1.size(2)
        << " and " << mat2.size(1) << "x" << mat2.size(2) << ")";
    m = mat1.size(1);
    n = mat2.size(1);
    k = mat2.size(2);
    b = mat1.size(0);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "mat1 must be a matrix or a batch of matrices";
  }

  // No heuristic for now, we rely on the autotuner to select the best tactic.
  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getFp8GemmConfig(m, n, k, tactic);

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
      runGemm<half>(out, mat1, mat2, scale_a, scale_b, m, n, k, b, config, workspace_buffer);
      break;
    case bfloat16_code:
      runGemm<__nv_bfloat16>(out, mat1, mat2, scale_a, scale_b, m, n, k, b, config,
                             workspace_buffer);
      break;
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "out_dtype must be one of fp16/bf16.";
  }
}

}  // namespace

void fp8_gemm(TensorView mat1, TensorView mat2, TensorView scale_a, TensorView scale_b,
              TensorView out, TensorView workspace_buffer, int64_t tactic) {
  fp8_bmm_impl(mat1, mat2, scale_a, scale_b, out, workspace_buffer, tactic);
}

int64_t fp8_gemm_tactic_num() {
  auto getCutlassConfigs = []() {
    CutlassFp8GemmRunner<__nv_bfloat16> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static int64_t totalTactics = getCutlassConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm, torch_ext::fp8_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm_tactic_num, torch_ext::fp8_gemm_tactic_num);
