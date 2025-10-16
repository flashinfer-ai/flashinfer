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
// Use SM120-specific dispatch template (includes fp4_gemm_cutlass.h)
#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"
#include "tvm_ffi_utils.h"

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
  TVM_FFI_ICHECK(tactic >= 0 && tactic < globalConfigs.size())
      << "tactic must be between 0 and " << globalConfigs.size();
  return globalConfigs[tactic];
}

template <typename T>
void runGemm(TensorView out, TensorView mat1, TensorView mat2, TensorView mat1Scale,
             TensorView mat2Scale, TensorView globalScale, int64_t m, int64_t n, int64_t k,
             int64_t batch_count, CutlassGemmConfig const& gemmConfig,
             TensorView workspace_buffer) {
  CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4> gemmRunner;

  int64_t const required_workspace_size = gemmRunner.getWorkspaceSize(m, n, k, batch_count);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);

  auto runKernel = [&](void* workspace) {
    gemmRunner.gemm(out.data_ptr(), mat1.data_ptr(), mat2.data_ptr(), mat1Scale.data_ptr(),
                    mat2Scale.data_ptr(), static_cast<float*>(globalScale.data_ptr()), m, n, k,
                    batch_count, gemmConfig, reinterpret_cast<char*>(workspace),
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

constexpr auto FLOAT4_E2M1X2 = dl_uint8;  // uint8_t
constexpr auto SF_DTYPE = dl_uint8;       // uint8_t

void fp4_bmm_impl(TensorView mat1, TensorView mat2, TensorView mat1Scale, TensorView mat2Scale,
                  TensorView globalScale, TensorView out, TensorView workspace_buffer,
                  int64_t tactic) {
  // Validate inputs
  TVM_FFI_ICHECK_EQ(mat1.dtype(), FLOAT4_E2M1X2) << "mat1 must be FLOAT4_E2M1X2 (uint8)";
  TVM_FFI_ICHECK_EQ(mat2.dtype(), FLOAT4_E2M1X2) << "mat2 must be FLOAT4_E2M1X2 (uint8)";
  TVM_FFI_ICHECK_EQ(mat1Scale.dtype(), SF_DTYPE) << "mat1Scale must be SF_DTYPE (uint8)";
  TVM_FFI_ICHECK_EQ(mat2Scale.dtype(), SF_DTYPE) << "mat2Scale must be SF_DTYPE (uint8)";
  TVM_FFI_ICHECK_EQ(globalScale.dtype(), dl_float32) << "globalScale must be float";
  TVM_FFI_ICHECK_EQ(mat1.device().device_type, kDLCUDA) << "mat1 must be on CUDA device";
  TVM_FFI_ICHECK_EQ(mat2.device().device_type, kDLCUDA) << "mat2 must be on CUDA device";
  TVM_FFI_ICHECK_EQ(mat1Scale.device().device_type, kDLCUDA) << "mat1Scale must be on CUDA device";
  TVM_FFI_ICHECK_EQ(mat2Scale.device().device_type, kDLCUDA) << "mat2Scale must be on CUDA device";
  TVM_FFI_ICHECK_EQ(globalScale.device().device_type, kDLCUDA)
      << "globalScale must be on CUDA device";
  TVM_FFI_ICHECK_EQ(out.device().device_type, kDLCUDA) << "out must be on CUDA device";
  TVM_FFI_ICHECK_EQ(workspace_buffer.device().device_type, kDLCUDA)
      << "workspace_buffer must be on CUDA device";

  // Check device consistency
  CHECK_DEVICE(mat1, mat2);
  CHECK_DEVICE(mat1, mat1Scale);
  CHECK_DEVICE(mat1, mat2Scale);
  CHECK_DEVICE(mat1, globalScale);
  CHECK_DEVICE(mat1, out);
  CHECK_DEVICE(mat1, workspace_buffer);

  // Get dimensions
  int64_t b = 1;
  int64_t m, k_packed, n;

  if (mat1.ndim() == 2) {
    m = mat1.size(0);
    k_packed = mat1.size(1);
  } else if (mat1.ndim() == 3) {
    b = mat1.size(0);
    m = mat1.size(1);
    k_packed = mat1.size(2);
  } else {
    TVM_FFI_ICHECK(false) << "mat1 must be 2D or 3D tensor";
  }

  if (mat2.ndim() == 2) {
    n = mat2.size(0);
    TVM_FFI_ICHECK_EQ(mat2.size(1), k_packed) << "mat2.size(1) must match mat1.size(-1)";
  } else if (mat2.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(mat2.size(0), b) << "Batch dimensions must match";
    n = mat2.size(1);
    TVM_FFI_ICHECK_EQ(mat2.size(2), k_packed) << "mat2.size(2) must match mat1.size(-1)";
  } else {
    TVM_FFI_ICHECK(false) << "mat2 must be 2D or 3D tensor";
  }

  // k_packed stores 2 FP4 values per byte
  int64_t k = k_packed * 2;

  TVM_FFI_ICHECK_EQ(globalScale.numel(), 1) << "globalScale must be a scalar tensor";

  // Configure the kernel
  CutlassGemmConfig config =
      (tactic >= 0) ? getFp4GemmConfig(m, n, k, tactic)
                    : CutlassGemmConfig(CutlassTileConfigSM120::CtaShape128x128x128B,
                                        MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                                        ClusterShape::ClusterShape_1x1x1);

  // Validate output dimensions
  std::vector<int64_t> out_shape =
      (b > 1) ? std::vector<int64_t>{b, m, n} : std::vector<int64_t>{m, n};
  TVM_FFI_ICHECK_EQ(out.ndim(), out_shape.size())
      << "out must have " << out_shape.size() << " dimensions";
  for (size_t i = 0; i < out_shape.size(); ++i) {
    TVM_FFI_ICHECK_EQ(out.size(i), out_shape[i])
        << "out.size(" << i << "): expected " << out_shape[i] << ", got " << out.size(i);
  }

  switch (encode_dlpack_dtype(out.dtype())) {
    case float16_code:
      runGemm<half>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config,
                    workspace_buffer);
      break;
    case bfloat16_code:
      runGemm<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, b, config,
                             workspace_buffer);
      break;
    default:
      TVM_FFI_ICHECK(false) << "out_dtype must be one of fp16/bf16.";
  }
}

}  // namespace

void fp4_gemm(TensorView mat1, TensorView mat2, TensorView mat1Scale, TensorView mat2Scale,
              TensorView globalScale, TensorView out, TensorView workspace_buffer, int64_t tactic) {
  fp4_bmm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, out, workspace_buffer, tactic);
}

int64_t fp4_gemm_tactic_num() {
  static const int64_t totalTactics =
      CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>{}.getConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_gemm, torch_ext::fp4_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_gemm_tactic_num, torch_ext::fp4_gemm_tactic_num);
