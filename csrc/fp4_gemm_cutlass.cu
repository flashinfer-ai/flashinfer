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
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_cutlass.h"
#include "flashinfer/gemm/fp4_gemm_cutlass_template.h"
#include "pytorch_extension_utils.h"

using flashinfer::gemm::ClusterShape;
using flashinfer::gemm::CutlassFp4GemmRunner;
using flashinfer::gemm::CutlassFp4GemmRunnerInterface;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM100;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::FP4GemmType;
using flashinfer::gemm::MainloopScheduleType;

namespace flashinfer {
namespace gemm {
template class CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>;
template class CutlassFp4GemmRunner<half, FP4GemmType::W4A4_NVFP4_NVFP4>;
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace {

CutlassGemmConfig getFp4GemmConfig(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  auto getCutlassFp4GemmConfigs = []() {
    CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static std::vector<CutlassGemmConfig> globalConfigs = getCutlassFp4GemmConfigs();
  TORCH_CHECK(tactic >= 0 && tactic < globalConfigs.size(), "tactic must be between 0 and ",
              globalConfigs.size());
  return globalConfigs[tactic];
}

template <typename T>
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2,
             at::Tensor const& mat1Scale, at::Tensor const& mat2Scale,
             at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k, int64_t batch_count,
             CutlassGemmConfig const& gemmConfig, at::Tensor workspace_buffer) {
  CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4> gemmRunner;

  int64_t const required_workspace_size = gemmRunner.getWorkspaceSize(m, n, k, batch_count);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * workspace_buffer.element_size();

  auto runKernel = [&](void* workspace) {
    gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(),
                    mat1Scale.const_data_ptr(), mat2Scale.const_data_ptr(),
                    globalScale.data_ptr<float>(), m, n, k, batch_count, gemmConfig,
                    reinterpret_cast<char*>(workspace), required_workspace_size,
                    at::cuda::getCurrentCUDAStream(mat1.get_device()));
  };

  if (provided_workspace_size < required_workspace_size) {
    at::Tensor new_workspace = at::detail::empty_cuda(
        {required_workspace_size}, at::ScalarType::Char, mat1.device(), std::nullopt);

    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;  // uint8_t
constexpr auto SF_DTYPE = at::ScalarType::Byte;       // uint8_t

// mat1: [B, M, K / 2], FLOAT4_E2M1X2 or [B, M, K], FLOAT8_E4M3FN
// mat2: [B, N, K / 2], FLOAT4_E2M1X2
// out: [B, M, N], fp16/bf16/fp32
// mat1Scale: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// mat2Scale: ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// globalScale: [1], 1 / (((448 * 6) / mat1.abs().max()) * ((448 * 6) / mat2.abs().max()))
// B = 1 for GEMM op as a special case
at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
                        at::Tensor const& mat2Scale, at::Tensor const& globalScale, at::Tensor out,
                        at::Tensor workspace_buffer, int64_t tactic) {
  CHECK_INPUT_AND_TYPE(mat1, FLOAT4_E2M1X2);
  CHECK_INPUT_AND_TYPE(mat2, FLOAT4_E2M1X2);

  int mat2_k_scale = 1;

  CHECK_INPUT_AND_TYPE(mat1Scale, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(mat2Scale, SF_DTYPE);

  CHECK_INPUT_AND_TYPE(globalScale, at::ScalarType::Float);

  int64_t m, n, k, b;
  if (mat1.dim() == 2) {
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1] * mat2_k_scale,
                "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
                mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
    m = mat1.sizes()[0];
    n = mat2.sizes()[0];
    k = mat2.sizes()[1] * 2;
    b = 1;
  } else if (mat1.dim() == 3) {
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batch of matrices");
    TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 must have the same batch size (",
                mat1.sizes()[0], " and ", mat2.sizes()[0], ")");
    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2] * mat2_k_scale,
                "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[1], "x",
                mat1.sizes()[2], " and ", mat2.sizes()[1], "x", mat2.sizes()[2], ")");
    m = mat1.sizes()[1];
    n = mat2.sizes()[1];
    k = mat2.sizes()[2] * 2;
    b = mat1.sizes()[0];
  } else {
    C10_THROW_ERROR(NotImplementedError, "mat1 must be a matrix or a batch of matrices");
  }

  // No heuristic for now, we rely on the autotuner to select the best tactic.
  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getFp4GemmConfig(m, n, k, tactic);

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment,
              ", but got mat1 shape: (", mat1.sizes()[0], "x", mat1.sizes()[1], "), k: ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got mat2 shape: (", mat2.sizes()[0], "x", mat2.sizes()[1], ").");

  // Validate out dimensions
  std::vector<int64_t> out_shape =
      mat1.dim() == 2 ? std::vector<int64_t>{m, n} : std::vector<int64_t>{b, m, n};
  TORCH_CHECK(out.dim() == out_shape.size(), "out must have ", out_shape.size(),
              " dimensions, but got ", out.dim());
  for (int i = 0; i < out_shape.size(); ++i) {
    TORCH_CHECK(out.sizes()[i] == out_shape[i], "out shape mismatch at dimension ", i,
                ": expected ", out_shape[i], ", got ", out.sizes()[i]);
  }

  c10::ScalarType out_dtype = out.scalar_type();

  switch (out_dtype) {
    case at::ScalarType::Half:
      runGemm<half>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config,
                    workspace_buffer);
      break;
    case at::ScalarType::BFloat16:
      runGemm<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config,
                             workspace_buffer);
      break;
    default:
      TORCH_CHECK(false, "out_dtype must be one of fp16/bf16.");
  }
  return out;
}

}  // namespace

at::Tensor fp4_gemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
                    at::Tensor const& mat2Scale, at::Tensor const& globalScale, at::Tensor out,
                    at::Tensor workspace_buffer, int64_t tactic) {
  return fp4_bmm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, out, workspace_buffer, tactic);
}

int64_t fp4_gemm_tactic_num() {
  auto getCutlassConfigs = []() {
    CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static int64_t totalTactics = getCutlassConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_gemm", &torch_ext::fp4_gemm);
  m.def("fp4_gemm_tactic_num", &torch_ext::fp4_gemm_tactic_num);
}
