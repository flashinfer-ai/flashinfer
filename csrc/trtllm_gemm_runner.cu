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

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/common.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/Enums.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/GemmInterface.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/gemm/trtllmGen_gemm_export/trtllm/gen/SfLayoutDecl.h"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace {
static thread_local gemm::gemm::GemmInterface::ModuleCache globalTrtllmGenGemmModuleCache;
}  // namespace

namespace flashinfer {

struct TrtllmGenGemmRunnerOptions {
  gemm::trtllm::gen::Dtype eltType;
  gemm::trtllm::gen::Dtype outputType;
  bool transposeMmaOutput{false};
  gemm::trtllm::gen::SfLayout sfLayoutB;
};

int64_t select_kernel_fp8(int32_t M, int32_t N, int32_t K,
                          const gemm::gemm::GemmInterface& interface) {
  static constexpr const char* KERNEL_NAME_HIGH_N_K_RATIO =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_cga1x1x1_16dp256b_rM_TN_"
      "transOut_"
      "noShflA_dsFp8_schedP2x2x1x3_sm100f";

  static constexpr const char* KERNEL_NAME_LOW_N_K_RATIO =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_rM_TN_"
      "transOut_noShflA_dsFp8_schedS_sm100f";

  static constexpr const char* KERNEL_NAME_LARGE_N =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_cga1x1x1_16dp256b_rM_TN_"
      "transOut_noShflA_dsFp8_schedP2x2x1x3_sm100f";

  static constexpr const char* KERNEL_NAME_DEFAULT =
      "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_cga1x1x1_16dp256b_rM_TN_"
      "transOut_noShflA_dsFp8_schedS_sm100f";

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

  TVM_FFI_ICHECK(false) << "Kernel not found";
}

class TrtllmGenGemmRunner {
 public:
  explicit TrtllmGenGemmRunner(TrtllmGenGemmRunnerOptions const& options) : mOptions(options) {
    // Select a GEMM kernel config to use
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    mPassingConfigIndices.clear();

    for (size_t i = 0; i < gemm.getNumGemmConfigs(); ++i) {
      auto const options = configs[i].mOptions;

      if (options.mDtypeA == mOptions.eltType && options.mDtypeC == mOptions.outputType &&
          options.mTransposeMmaOutput == mOptions.transposeMmaOutput &&
          options.mSfLayoutB == mOptions.sfLayoutB) {
        mPassingConfigIndices.push_back(i);
      }
    }

    FLASHINFER_CHECK(
        mPassingConfigIndices.size() > 0,
        "No valid tactic found for the given options (precision, transpose, sf layout)");
  }

  int64_t getWorkspaceSizeInBytes(int64_t m, int64_t n, int64_t k, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    FLASHINFER_CHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs(),
                     "Invalid tactic in getWorkspaceSizeInBytes");
    auto const config = configs[tactic];

    gemm::gemm::GemmData gemmData;
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    return gemm.getWorkspaceSizeInBytes(config, gemmData);
  }

  void run(int64_t m, int64_t n, int64_t k, void const* a, void const* aScale, void const* b,
           void const* bScale, void* c, void* cScale, void* cScalePtr, void* workspace,
           CUstream stream, int32_t device_index, int64_t tactic) {
    auto gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();
    TVM_FFI_ICHECK(tactic >= 0 && tactic < gemm.getNumGemmConfigs()) << "Invalid tactic id in run";
    auto const& config = configs[tactic];
    TVM_FFI_ICHECK(config.mOptions.mSfLayoutB == mOptions.sfLayoutB) << "Invalid sf layout in run";

    gemm::gemm::GemmData gemmData;
    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    // Inputs
    gemmData.mInputBuffers.mPtrA = mOptions.transposeMmaOutput ? b : a;
    gemmData.mInputBuffers.mPtrSfA = mOptions.transposeMmaOutput ? bScale : aScale;
    gemmData.mInputBuffers.mPtrB = mOptions.transposeMmaOutput ? a : b;
    gemmData.mInputBuffers.mPtrSfB = mOptions.transposeMmaOutput ? aScale : bScale;
    gemmData.mInputBuffers.mPtrScaleC = cScale;

    // Outputs
    gemmData.mOutputBuffers.mPtrC = c;
    gemmData.mOutputBuffers.mPtrSfC = cScalePtr;

    TVM_FFI_ICHECK(gemm.isValidConfig(config, gemmData)) << "unsupported tactic id in run";

    const int32_t multiProcessorCount = [device_index]() {
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
                            multiProcessorCount, true, globalTrtllmGenGemmModuleCache) == 0)
        << "Error occurred when running GEMM!";
  }

  std::vector<int64_t> getValidTactics(int64_t m, int64_t n, int64_t k) const {
    auto const gemm = gemm::gemm::GemmInterface();
    auto const configs = gemm.getGemmConfigs();

    gemm::gemm::GemmData gemmData;
    // Dims
    gemmData.mProblemDimensions.mM = mOptions.transposeMmaOutput ? n : m;
    gemmData.mProblemDimensions.mN = mOptions.transposeMmaOutput ? m : n;
    gemmData.mProblemDimensions.mK = k;
    gemmData.mProblemDimensions.mRank = 0;
    gemmData.mProblemDimensions.mWorldSize = 1;

    std::vector<int64_t> sortedIndices = mPassingConfigIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&configs](int64_t idx0, int64_t idx1) {
      auto const& optionsA = configs[idx0].mOptions;
      auto const& optionsB = configs[idx1].mOptions;

      // Sort by tileK sizes first
      if (optionsA.mTileK != optionsB.mTileK) {
        return optionsA.mTileK > optionsB.mTileK;
      }

      // Then by splitK sizes
      if (optionsA.mNumSlicesForSplitK != optionsB.mNumSlicesForSplitK) {
        return optionsA.mNumSlicesForSplitK > optionsB.mNumSlicesForSplitK;
      }

      // Then by unroll loop 2x for mma
      if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma) {
        return optionsA.mUseUnrollLoop2xForMma;
      }

      return false;
    });

    bool findLoop2xMma = false;
    std::vector<int64_t> validTactics;
    for (auto const& configIndex : sortedIndices) {
      auto const& config = configs[configIndex];
      if (gemm.isValidConfig(config, gemmData)) {
        validTactics.push_back(configIndex);

        // when loop2x mma is found, only add the tactic that has loop2x mma
        if (!findLoop2xMma) {
          if (config.mOptions.mUseUnrollLoop2xForMma) {
            findLoop2xMma = true;
          }
        } else {
          if (!config.mOptions.mUseUnrollLoop2xForMma) {
            break;
          }
        }
      }
    }
    return validTactics;
  }

  int64_t selectHeuristic(int64_t m, int64_t n, int64_t k) const {
    if (mOptions.eltType == gemm::trtllm::gen::Dtype::E4m3) {
      return select_kernel_fp8(m, n, k, gemm::gemm::GemmInterface());
    } else if (mOptions.eltType == gemm::trtllm::gen::Dtype::E2m1) {
      auto sortedIndices = getValidTactics(m, n, k);
      TVM_FFI_ICHECK(!sortedIndices.empty()) << "No valid tactic found";

      // the getValidTactics is sorted by priority, so the first one is the best one
      return sortedIndices[0];
    } else {
      TVM_FFI_ICHECK(false) << "Unsupported eltType";
    }
  }

 private:
  TrtllmGenGemmRunnerOptions mOptions;
  std::vector<int64_t> mPassingConfigIndices;
};

using tvm::ffi::Array;
using tvm::ffi::Optional;

void trtllm_gemm(TensorView workspace_buffer, TensorView a, TensorView b, TensorView a_scale,
                 TensorView b_scale, Optional<TensorView> globalScale, TensorView out,
                 bool use_8x4_sf_layout, int64_t tactic) {
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, out);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(out);
  CHECK_INPUT(workspace_buffer);
  TVM_FFI_ICHECK_EQ(workspace_buffer.ndim(), 1);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  TVM_FFI_ICHECK_EQ(a.dtype(), b.dtype());
  TVM_FFI_ICHECK(a.dtype() == dl_float8_e4m3fn || a.dtype() == dl_uint8)
      << "a must be a Float8 or Byte(e2m1) tensor";
  bool is_fp8 = a.dtype() == dl_float8_e4m3fn;
  if (is_fp8) {
    TVM_FFI_ICHECK(!globalScale.has_value()) << "globalScale must be a none tensor";
  } else {
    CHECK_INPUT(a_scale);
    CHECK_INPUT(b_scale);
    if (globalScale.has_value()) {
      CHECK_INPUT(globalScale.value());
    }
  }

  int32_t m = a.size(0);
  int32_t k = is_fp8 ? a.size(1) : a.size(1) * 2;
  int32_t n = b.size(0);
  TVM_FFI_ICHECK_EQ(b.size(1), a.size(1)) << "Matrix dimensions don't match for multiplication";
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == n) << "Output tensor has wrong dimensions";

  auto runner = flashinfer::TrtllmGenGemmRunner(flashinfer::TrtllmGenGemmRunnerOptions{
      .eltType = is_fp8 ? gemm::trtllm::gen::Dtype::E4m3 : gemm::trtllm::gen::Dtype::E2m1,
      .outputType = gemm::trtllm::gen::Dtype::Bfloat16,
      .transposeMmaOutput = true,
      .sfLayoutB = use_8x4_sf_layout ? gemm::trtllm::gen::SfLayout::R8c4
                                     : gemm::trtllm::gen::SfLayout::R128c4,
  });

  if (tactic == -1) {
    tactic = runner.selectHeuristic(m, n, k);
  }

  auto stream = get_stream(a.device());

  auto runKernel = [&](void* workspace) {
    runner.run(m, n, k, a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(),
               out.data_ptr(), globalScale.has_value() ? globalScale.value().data_ptr() : nullptr,
               nullptr, workspace, stream, a.device().device_id, tactic);
  };

  int64_t const required_workspace_size = runner.getWorkspaceSizeInBytes(m, n, k, tactic);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  if (provided_workspace_size < required_workspace_size) {
    Tensor new_workspace = alloc_tensor({required_workspace_size}, dl_int8, a.device());
    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

enum class Dtype : int64_t {
  E2m1 = 0,
  E4m3 = 1,
  Bfloat16 = 2,
};

Array<int64_t> trtllm_gemm_tactics(int64_t m, int64_t n, int64_t k, int64_t input_dtype,
                                   int64_t output_dtype, bool use_8x4_sf_layout) {
  TVM_FFI_ICHECK(input_dtype == static_cast<int64_t>(Dtype::E4m3) ||
                 input_dtype == static_cast<int64_t>(Dtype::E2m1))
      << "Unsupported input dtype";
  TVM_FFI_ICHECK_EQ(output_dtype, static_cast<int64_t>(Dtype::Bfloat16))
      << "Unsupported output dtype";

  auto runner = flashinfer::TrtllmGenGemmRunner(flashinfer::TrtllmGenGemmRunnerOptions{
      .eltType = input_dtype == static_cast<int64_t>(Dtype::E4m3) ? gemm::trtllm::gen::Dtype::E4m3
                                                                  : gemm::trtllm::gen::Dtype::E2m1,
      .outputType = gemm::trtllm::gen::Dtype::Bfloat16,
      .transposeMmaOutput = true,
      .sfLayoutB = use_8x4_sf_layout ? gemm::trtllm::gen::SfLayout::R8c4
                                     : gemm::trtllm::gen::SfLayout::R128c4,
  });

  return runner.getValidTactics(m, n, k);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_gemm, flashinfer::trtllm_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_gemm_tactics, flashinfer::trtllm_gemm_tactics);
