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
#include "flashinfer/gemm/fp8_gemm_cutlass.h"
#include "flashinfer/gemm/fp8_gemm_cutlass_template.h"
#include "pytorch_extension_utils.h"

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
  TORCH_CHECK(tactic >= 0 && tactic < globalConfigs.size(), "tactic must be between 0 and ",
              globalConfigs.size());
  return globalConfigs[tactic];
}

template <typename T>
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2,
             at::Tensor const& scale_a, at::Tensor const& scale_b, int64_t m, int64_t n, int64_t k,
             int64_t b, CutlassGemmConfig const& gemmConfig, at::Tensor workspace_buffer) {
  CutlassFp8GemmRunner<T> gemmRunner;

  int64_t const required_workspace_size = gemmRunner.getWorkspaceSize(m, n, k);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * workspace_buffer.element_size();

  auto runKernel = [&](void* workspace) {
    gemmRunner.gemm(reinterpret_cast<__nv_fp8_e4m3 const*>(mat1.const_data_ptr()),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(mat2.const_data_ptr()),
                    reinterpret_cast<float const*>(scale_a.const_data_ptr()),
                    reinterpret_cast<float const*>(scale_b.const_data_ptr()), out.data_ptr(), m, n,
                    k, b, gemmConfig, reinterpret_cast<char*>(workspace), required_workspace_size,
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

at::Tensor fp8_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& scale_a,
                        at::Tensor const& scale_b, at::Tensor out, at::Tensor workspace_buffer,
                        int64_t tactic) {
  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scale_a);
  CHECK_INPUT(scale_b);

  int mat2_k_scale = 1;

  int64_t m, n, k, b;
  if (mat1.dim() == 2) {
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1] * mat2_k_scale,
                "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
                mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
    m = mat1.sizes()[0];
    n = mat2.sizes()[0];
    k = mat2.sizes()[1];
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
    k = mat2.sizes()[2];
    b = mat1.sizes()[0];
  } else {
    C10_THROW_ERROR(NotImplementedError, "mat1 must be a matrix or a batch of matrices");
  }

  // No heuristic for now, we rely on the autotuner to select the best tactic.
  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getFp8GemmConfig(m, n, k, tactic);

  // Validate out dimensions
  std::vector<int64_t> out_shape =
      mat1.dim() == 2 ? std::vector<int64_t>{m, n} : std::vector<int64_t>{b, m, n};
  TORCH_CHECK(out.dim() == out_shape.size(), "out must have ", out_shape.size(),
              " dimensions, but got ", out.dim());
  for (int i = 0; i < out_shape.size(); ++i) {
    TORCH_CHECK(out.sizes()[i] == out_shape[i], "out shape mismatch at dimension ", i,
                ": expected ", out_shape[i], ", got ", out.sizes()[i]);
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Half:
      runGemm<half>(out, mat1, mat2, scale_a, scale_b, m, n, k, b, config, workspace_buffer);
      break;
    case at::ScalarType::BFloat16:
      runGemm<__nv_bfloat16>(out, mat1, mat2, scale_a, scale_b, m, n, k, b, config,
                             workspace_buffer);
      break;
    default:
      TORCH_CHECK(false, "out_dtype must be one of fp16/bf16.");
  }
  return out;
}

}  // namespace

at::Tensor fp8_gemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& scale_a,
                    at::Tensor const& scale_b, at::Tensor out, at::Tensor workspace_buffer,
                    int64_t tactic) {
  return fp8_bmm_impl(mat1, mat2, scale_a, scale_b, out, workspace_buffer, tactic);
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

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemm", &torch_ext::fp8_gemm);
  m.def("fp8_gemm_tactic_num", &torch_ext::fp8_gemm_tactic_num);
}
