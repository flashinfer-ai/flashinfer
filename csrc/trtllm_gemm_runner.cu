/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>

#include <string>

#include "flashinfer/trtllm/common.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/Enums.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

namespace {
static thread_local gemm::gemm::GemmInterface::ModuleCache globalTrtllmGenGemmModuleCache;
}  // namespace

int64_t select_kernel(int32_t M, int32_t N, int32_t K, const gemm::gemm::GemmInterface& interface) {
  static constexpr const char* KERNEL_NAME_HIGH_N_K_RATIO =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_et64x8_m64x8x32_cga1x1x1_16dp256b_s6_TN_transOut_"
      "noShflA_dsFp8_schedP_sm100a";

  static constexpr const char* KERNEL_NAME_LOW_N_K_RATIO =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_"
      "transOut_noShflA_dsFp8_schedS_sm100a";

  static constexpr const char* KERNEL_NAME_LARGE_N =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_"
      "transOut_noShflA_dsFp8_schedP_sm100a";

  static constexpr const char* KERNEL_NAME_DEFAULT =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_et64x16_m64x16x32_cga1x1x1_16dp256b_s6_TN_"
      "transOut_noShflA_dsFp8_schedS_sm100a";

  double const n_k_ratio = static_cast<double>(N) / static_cast<double>(K);

  std::string kernel_name;
  if (n_k_ratio >= 32) {
    kernel_name = KERNEL_NAME_HIGH_N_K_RATIO;
  } else if (n_k_ratio <= 2.0) {
    kernel_name = KERNEL_NAME_LOW_N_K_RATIO;
  } else if (N >= 20000) {
    kernel_name = KERNEL_NAME_LARGE_N;
  } else {
    kernel_name = KERNEL_NAME_DEFAULT;
  }

  auto const& configs = interface.getGemmConfigs();
  size_t const num_configs = interface.getNumGemmConfigs();

  for (size_t i = 0; i < num_configs; ++i) {
    if (std::string(configs[i].mFunctionName) == kernel_name) {
      return static_cast<int64_t>(i);
    }
  }

  TORCH_CHECK(false, "Kernel not found");
}

void trtllm_gemm(at::Tensor workspace_buffer, at::Tensor a, at::Tensor b, at::Tensor a_scale,
                 at::Tensor b_scale, at::Tensor out) {
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  TORCH_CHECK(a.device() == out.device(), "a and out must be on the same device");
  TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA device");
  int32_t m = a.size(0);
  int32_t k = a.size(1);
  int32_t n = b.size(0);
  TORCH_CHECK(b.size(1) == k, "Matrix dimensions don't match for multiplication");
  TORCH_CHECK(out.size(0) == m && out.size(1) == n, "Output tensor has wrong dimensions");

  auto interface = gemm::gemm::GemmInterface();
  const int64_t config_index = select_kernel(m, n, k, interface);

  auto const& config = interface.getGemmConfigs()[config_index];

  // output is transposed -> operands are swapped
  gemm::gemm::GemmData gemmData;
  gemmData.mProblemDimensions.mM = n;
  gemmData.mProblemDimensions.mN = m;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mInputBuffers.mPtrA = b.data_ptr();
  gemmData.mInputBuffers.mPtrSfA = b_scale.data_ptr();
  gemmData.mInputBuffers.mPtrB = a.data_ptr();
  gemmData.mInputBuffers.mPtrSfB = a_scale.data_ptr();
  gemmData.mOutputBuffers.mPtrC = out.data_ptr();

  size_t const required_workspace_size = interface.getWorkspaceSizeInBytes(config, gemmData);
  size_t const provided_workspace_size = workspace_buffer.numel() * workspace_buffer.element_size();
  TORCH_CHECK(provided_workspace_size >= required_workspace_size,
              "Workspace buffer is too small. Required: ", required_workspace_size,
              " bytes, provided: ", provided_workspace_size, " bytes");

  const int device_index = a.device().index();
  const int32_t multiProcessorCount = [device_index]() {
    static thread_local int32_t cached_multi_processor_count = -1;
    static thread_local int cached_device_index = -1;

    if (device_index == cached_device_index && cached_multi_processor_count != -1) {
      return cached_multi_processor_count;
    } else {
      int32_t count;
      cudaError_t cudaStatus =
          cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
      TORCH_CHECK(cudaStatus == cudaSuccess,
                  "Failed to get device attribute: ", cudaGetErrorString(cudaStatus));
      cached_multi_processor_count = count;
      cached_device_index = device_index;
      return count;
    }
  }();

  TORCH_CHECK(interface.run(config, workspace_buffer.data_ptr(), gemmData,
                            static_cast<void*>(at::cuda::getCurrentCUDAStream(a.device().index())),
                            multiProcessorCount, true, globalTrtllmGenGemmModuleCache) == 0,
              "Error occurred when running GEMM!");
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

}  // namespace flashinfer

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("trtllm_gemm", flashinfer::trtllm_gemm); }
