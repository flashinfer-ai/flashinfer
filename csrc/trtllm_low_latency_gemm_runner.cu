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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm_ffi_utils.h>

#include <vector>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/common.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/Enums.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/SfLayoutDecl.h"

namespace {
static thread_local gemm::gemm::GemmInterface::ModuleCache globalTrtllmLowLatencyGemmModuleCache;
}  // namespace

namespace flashinfer {

using tvm::ffi::Array;
using tvm::ffi::Optional;

struct TrtllmLowLatencyGemmRunnerOptions {
  gemm::trtllm::gen::Dtype eltType;
  gemm::trtllm::gen::Dtype outputType;
};

gemm::gemm::GemmData createGemmData(int64_t m, int64_t n, int64_t k) {
  gemm::gemm::GemmData gemmData{};

  // Dims
  gemmData.mProblemDimensions.mM = n;
  gemmData.mProblemDimensions.mN = m;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;

  return gemmData;
}

/**
 * Very rough heuristic for selecting a kernel. Prefer using auto-tuning.
 */
int64_t select_kernel(int32_t m, int32_t n, int32_t k, const gemm::gemm::GemmInterface& interface) {
  static constexpr const char* KERNEL_MMAN_8_TILEK_128_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s7_et128x8_m128x8x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_8_TILEK_128_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s7_et128x8_m128x8x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_8_TILEK_256_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x256_s4_et128x8_m128x8x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_8_TILEK_256_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x256_s4_et128x8_m128x8x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_16_TILEK_128_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128_s7_et128x16_m128x16x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_16_TILEK_128_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128_s7_et128x16_m128x16x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_16_TILEK_256_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x256_s5_et128x16_m128x16x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_16_TILEK_256_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x256_s5_et128x16_m128x16x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_32_TILEK_128_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128_s9_et128x32_m128x32x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_32_TILEK_128_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128_s9_et128x32_m128x32x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_32_TILEK_256_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x256_s5_et128x32_m128x32x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_32_TILEK_256_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x256_s5_et128x32_m128x32x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_64_TILEK_128_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s7_et128x64_m128x64x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_64_TILEK_128_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s7_et128x64_m128x64x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_64_TILEK_256_SPLITK_2 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x64x256_s3_et128x64_m128x64x32_cga1x1x2_16dp256b_splitK2_BN_"
      "transOut_schedS_sm100f";
  static constexpr const char* KERNEL_MMAN_64_TILEK_256_SPLITK_3 =
      "Gemm_Bfloat16_E4m3E4m3_Fp32_t128x64x256_s3_et128x64_m128x64x32_cga1x1x3_16dp256b_splitK3_BN_"
      "transOut_schedS_sm100f";

  std::string kernel_name;
  if (m <= 8) {
    kernel_name = KERNEL_MMAN_8_TILEK_128_SPLITK_2;
  } else if (m <= 16) {
    kernel_name = KERNEL_MMAN_16_TILEK_128_SPLITK_2;
  } else if (m <= 32) {
    kernel_name = KERNEL_MMAN_32_TILEK_128_SPLITK_2;
  } else {
    kernel_name = KERNEL_MMAN_64_TILEK_128_SPLITK_2;
  }

  auto const& configs = interface.getGemmConfigs();
  size_t const num_configs = interface.getNumGemmConfigs();

  for (size_t i = 0; i < num_configs; ++i) {
    if (std::string(configs[i].mFunctionName) == kernel_name) {
      return static_cast<int64_t>(i);
    }
  }

  TVM_FFI_LOG_AND_THROW(RuntimeError)
      << "No kernel was found heuristically for the given problem size";
}

int64_t getWorkspaceSizeInBytes(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  auto gemm = gemm::gemm::GemmInterface();

  if (tactic == -1) {
    tactic = select_kernel(m, n, k, gemm);
  }

  auto const configs = gemm.getGemmConfigs();
  FLASHINFER_CHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs(),
                   "Invalid tactic in getWorkspaceSizeInBytes");
  auto const config = configs[tactic];

  auto const gemmData = createGemmData(m, n, k);

  return gemm.getWorkspaceSizeInBytes(config, gemmData);
}

class TrtllmLowLatencyGemmRunner {
 public:
  explicit TrtllmLowLatencyGemmRunner(TrtllmLowLatencyGemmRunnerOptions const& options)
      : mOptions(options) {
    // Select a GEMM kernel config to use
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
      auto const configOptions = configs[i].mOptions;

      if (configOptions.mDtypeA == mOptions.eltType &&
          configOptions.mDtypeC == mOptions.outputType &&
          configOptions.mTransposeMmaOutput == true &&
          configOptions.mLayoutA == gemm::gemm::MatrixLayout::BlockMajorK &&
          configOptions.mUseShuffledMatrixA) {
        mPassingConfigIndices.push_back(i);
      }
    }

    FLASHINFER_CHECK(
        mPassingConfigIndices.size() > 0,
        "No valid low latency TRTLLM-GEN GEMM kernel was found for the given data types.");
  }

  void run(int64_t m, int64_t n, int64_t k, void const* a, void const* b, void* c, void* cScale,
           void* workspace, CUstream stream, int32_t device_index, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    TVM_FFI_ICHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs()) << "Invalid tactic id in run";
    auto const& config = configs[tactic];

    gemm::gemm::GemmData gemmData = createGemmData(m, n, k);

    // Inputs
    gemmData.mInputBuffers.mPtrA = b;
    gemmData.mInputBuffers.mPtrB = a;
    gemmData.mInputBuffers.mPtrScaleC = cScale;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;

    TVM_FFI_ICHECK(gemm.isValidConfig(config, gemmData))
        << "The selected tactic points to a TRTLLM-GEN low latency GEMM kernel that is not valid "
           "for "
           "the given problem size.";

    int32_t const multiProcessorCount = [device_index]() {
      static thread_local int32_t cached_multi_processor_count = -1;
      static thread_local int cached_device_index = -1;

      if (device_index == cached_device_index && cached_multi_processor_count != -1) {
        return cached_multi_processor_count;
      } else {
        int32_t count;
        cudaError_t cudaStatus =
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
        TVM_FFI_ICHECK(cudaStatus == cudaSuccess)
            << "Failed to get device attribute: " << cudaGetErrorString(cudaStatus);
        cached_multi_processor_count = count;
        cached_device_index = device_index;
        return count;
      }
    }();

    TVM_FFI_ICHECK(gemm.run(config, workspace, gemmData, static_cast<void*>(stream),
                            multiProcessorCount, true, globalTrtllmLowLatencyGemmModuleCache) == 0)
        << "Error occurred when running low latency TRTLLM-GEN GEMM!";
  }

  std::vector<int64_t> getValidTactics(int64_t m, int64_t n, int64_t k) const {
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    auto const gemmData = createGemmData(m, n, k);

    std::vector<int64_t> validTactics{};
    for (auto const& configIndex : mPassingConfigIndices) {
      auto const& config = configs[configIndex];
      if (gemm.isValidConfig(config, gemmData)) {
        validTactics.push_back(configIndex);
      }
    }
    return validTactics;
  }

 private:
  TrtllmLowLatencyGemmRunnerOptions mOptions;
  std::vector<int64_t> mPassingConfigIndices;
};

void trtllm_low_latency_gemm(TensorView workspace_buffer, TensorView a, TensorView b,
                             TensorView globalScale, TensorView out, int64_t tactic) {
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, out);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(out);
  CHECK_INPUT(workspace_buffer);
  CHECK_DIM(2, a);
  TVM_FFI_ICHECK(b.ndim() == 3) << "b must be a block layout matrix (3D tensor with "
                                   "dims [N/BLOCK_SIZE, K, BLOCK_SIZE])";
  TVM_FFI_ICHECK_EQ(a.dtype(), b.dtype());
  TVM_FFI_ICHECK(a.dtype() == dl_float8_e4m3fn) << "a must be a Float8 tensor";

  int32_t m = a.size(0);
  int32_t k = a.size(1);
  int32_t n = b.size(1);
  auto const blockSize = b.size(2);
  auto const kFromB = b.size(0) * blockSize;
  TVM_FFI_ICHECK(kFromB == a.size(1)) << "Matrix dimensions don't match for multiplication";
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == n) << "Output tensor has wrong dimensions";

  if (tactic == -1) {
    tactic = select_kernel(m, n, k, gemm::gemm::GemmInterface());
  }

  auto runner =
      flashinfer::TrtllmLowLatencyGemmRunner(flashinfer::TrtllmLowLatencyGemmRunnerOptions{
          .eltType = gemm::trtllm::gen::Dtype::E4m3,
          .outputType = gemm::trtllm::gen::Dtype::Bfloat16,
      });

  auto stream = get_stream(a.device());

  int64_t const required_workspace_size = getWorkspaceSizeInBytes(m, n, k, tactic);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  if (provided_workspace_size < required_workspace_size) {
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "The size of the provided workspace to the TRTLLM-GEN low latency GEMM is too small. "
           "Please use the provided workspace sizing function to pre-allocate an adequate "
           "workspace.";
  }

  runner.run(m, n, k, a.data_ptr(), b.data_ptr(), out.data_ptr(), globalScale.data_ptr(),
             workspace_buffer.data_ptr(), stream, a.device().device_id, tactic);
}

enum class Dtype : int64_t {
  E2m1 = 0,
  E4m3 = 1,
  Bfloat16 = 2,
};

Array<int64_t> trtllm_low_latency_gemm_tactics(int64_t m, int64_t n, int64_t k, int64_t input_dtype,
                                               int64_t output_dtype) {
  TVM_FFI_ICHECK(input_dtype == static_cast<int64_t>(Dtype::E4m3)) << "Unsupported input dtype";
  TVM_FFI_ICHECK_EQ(output_dtype, static_cast<int64_t>(Dtype::Bfloat16))
      << "Unsupported output dtype";

  auto runner =
      flashinfer::TrtllmLowLatencyGemmRunner(flashinfer::TrtllmLowLatencyGemmRunnerOptions{
          .eltType = gemm::trtllm::gen::Dtype::E4m3,
          .outputType = gemm::trtllm::gen::Dtype::Bfloat16,
      });

  return runner.getValidTactics(m, n, k);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

}  // namespace flashinfer

// Exposes low latency optimized GEMMs that require some pre-processing of the inputs.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_low_latency_gemm, flashinfer::trtllm_low_latency_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_low_latency_gemm_tactics,
                              flashinfer::trtllm_low_latency_gemm_tactics);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_workspace_size_in_bytes, flashinfer::getWorkspaceSizeInBytes);
