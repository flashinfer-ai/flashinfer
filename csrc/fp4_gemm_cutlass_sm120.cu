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
// Use SM120-specific dispatch template (includes fp4_gemm_cutlass.h)
#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"
#include "pytorch_extension_utils.h"

using flashinfer::gemm::ClusterShape;
using flashinfer::gemm::CutlassFp4GemmRunner;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM120;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::FP4GemmType;
using flashinfer::gemm::MainloopScheduleType;

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

at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
                        at::Tensor const& mat2Scale, at::Tensor const& globalScale, at::Tensor out,
                        at::Tensor workspace_buffer, int64_t tactic) {
  // Validate inputs
  TORCH_CHECK(mat1.dtype() == FLOAT4_E2M1X2, "mat1 must be FLOAT4_E2M1X2 (uint8)");
  TORCH_CHECK(mat2.dtype() == FLOAT4_E2M1X2, "mat2 must be FLOAT4_E2M1X2 (uint8)");
  TORCH_CHECK(mat1Scale.dtype() == SF_DTYPE, "mat1Scale must be SF_DTYPE (uint8)");
  TORCH_CHECK(mat2Scale.dtype() == SF_DTYPE, "mat2Scale must be SF_DTYPE (uint8)");
  TORCH_CHECK(globalScale.dtype() == at::ScalarType::Float, "globalScale must be float");
  TORCH_CHECK(mat1.is_cuda(), "mat1 must be on CUDA device");
  TORCH_CHECK(mat2.is_cuda(), "mat2 must be on CUDA device");
  TORCH_CHECK(mat1Scale.is_cuda(), "mat1Scale must be on CUDA device");
  TORCH_CHECK(mat2Scale.is_cuda(), "mat2Scale must be on CUDA device");
  TORCH_CHECK(globalScale.is_cuda(), "globalScale must be on CUDA device");
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA device");
  TORCH_CHECK(workspace_buffer.is_cuda(), "workspace_buffer must be on CUDA device");

  // Check device consistency
  TORCH_CHECK(mat1.device() == mat2.device() && mat1.device() == mat1Scale.device() &&
                  mat1.device() == mat2Scale.device() && mat1.device() == globalScale.device() &&
                  mat1.device() == out.device() && mat1.device() == workspace_buffer.device(),
              "All tensors must be on the same device");

  // Get dimensions
  int64_t b = 1;
  int64_t m, k_packed, n;

  if (mat1.dim() == 2) {
    m = mat1.size(0);
    k_packed = mat1.size(1);
  } else if (mat1.dim() == 3) {
    b = mat1.size(0);
    m = mat1.size(1);
    k_packed = mat1.size(2);
  } else {
    TORCH_CHECK(false, "mat1 must be 2D or 3D tensor");
  }

  if (mat2.dim() == 2) {
    n = mat2.size(0);
    TORCH_CHECK(mat2.size(1) == k_packed, "mat2.size(1) must match mat1.size(-1)");
  } else if (mat2.dim() == 3) {
    TORCH_CHECK(mat2.size(0) == b, "Batch dimensions must match");
    n = mat2.size(1);
    TORCH_CHECK(mat2.size(2) == k_packed, "mat2.size(2) must match mat1.size(-1)");
  } else {
    TORCH_CHECK(false, "mat2 must be 2D or 3D tensor");
  }

  // k_packed stores 2 FP4 values per byte
  int64_t k = k_packed * 2;

  TORCH_CHECK(globalScale.numel() == 1, "globalScale must be a scalar tensor");

  // Configure the kernel
  CutlassGemmConfig config =
      (tactic >= 0) ? getFp4GemmConfig(m, n, k, tactic)
                    : CutlassGemmConfig(CutlassTileConfigSM120::CtaShape128x128x128B,
                                        MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                                        ClusterShape::ClusterShape_1x1x1);

  // Validate output dimensions
  std::vector<int64_t> out_shape =
      (b > 1) ? std::vector<int64_t>{b, m, n} : std::vector<int64_t>{m, n};
  TORCH_CHECK(out.dim() == out_shape.size(), "out must have ", out_shape.size(), " dimensions");
  for (size_t i = 0; i < out_shape.size(); ++i) {
    TORCH_CHECK(out.sizes()[i] == out_shape[i], "out.size(", i, "): expected ", out_shape[i],
                ", got ", out.sizes()[i]);
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
  static const int64_t totalTactics =
      CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>{}.getConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_gemm", &torch_ext::fp4_gemm);
  m.def("fp4_gemm_tactic_num", &torch_ext::fp4_gemm_tactic_num);
}
