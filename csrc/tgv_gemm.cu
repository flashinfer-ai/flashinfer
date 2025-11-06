/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "cuda_runtime.h"
#include "flashinfer/cutlass_utils.cuh"
#include "flashinfer/gemm/tgv_gemm.cuh"
#include "flashinfer/gemm/tgv_gemm_configs.h"
#include "tvm_ffi_utils.h"

// CUTLASS type includes
#include <cutlass/numeric_types.h>

using tvm::ffi::Optional;

#define SUPPORTED_TGV_GEMM_CONFIGS \
  TGV_GEMM_CONFIG(64, 8, 6)        \
  TGV_GEMM_CONFIG(64, 8, 8)        \
  TGV_GEMM_CONFIG(64, 8, 10)       \
  TGV_GEMM_CONFIG(64, 8, 12)       \
  TGV_GEMM_CONFIG(64, 16, 6)       \
  TGV_GEMM_CONFIG(64, 16, 8)       \
  TGV_GEMM_CONFIG(64, 16, 10)      \
  TGV_GEMM_CONFIG(64, 32, 6)       \
  TGV_GEMM_CONFIG(64, 32, 8)       \
  TGV_GEMM_CONFIG(64, 64, 6)       \
  TGV_GEMM_CONFIG(128, 16, 6)

#define TGV_GEMM_CONFIG(CTA_M, CTA_N, DMA_STAGE)                                                 \
  if (cta_m == CTA_M && cta_n == CTA_N && dma_stage == DMA_STAGE) {                              \
    *func_ptr = &flashinfer::gemm::tgv_gemm_host<TypeA, TypeB, TypeC, AccType, TypeBias, CTA_M,  \
                                                 CTA_N, 128, DMA_STAGE, UmmaMajorA, UmmaMajorB>; \
    return;                                                                                      \
  }

template <typename TypeA, typename TypeB, typename TypeC, typename AccType, typename TypeBias>
using GemmFuncPtr = void (*)(TypeA*, TypeB*, TypeC*, TypeBias*, int, int, int, int, int, int, int,
                             int, int, int, int, int, int, bool, int, cudaStream_t);

template <typename TypeA, typename TypeB, typename TypeC, typename AccType, typename TypeBias,
          cute::UMMA::Major UmmaMajorA, cute::UMMA::Major UmmaMajorB>
void dispatch_kernel(int cta_m, int cta_n, int cta_k, int dma_stage,
                     GemmFuncPtr<TypeA, TypeB, TypeC, AccType, TypeBias>* func_ptr) {
  SUPPORTED_TGV_GEMM_CONFIGS

  TVM_FFI_ICHECK(false) << "Unsupported tile configuration: cta_m=" << std::to_string(cta_m)
                        << ", cta_n=" << std::to_string(cta_n)
                        << ", cta_k=" << std::to_string(cta_k);
}
#undef TGV_GEMM_CONFIG

namespace torch_ext {

namespace {
// Use the shared function from the header file

using flashinfer::gemm::getAllTgvConfigs;
using flashinfer::gemm::TGVGemmConfig;

TGVGemmConfig getTgvGemmConfig(int64_t tactic) {
  auto globalConfigs = getAllTgvConfigs();

  TVM_FFI_ICHECK(tactic >= 0 && tactic < globalConfigs.size())
      << "tactic must be between 0 and " << globalConfigs.size();
  return globalConfigs[tactic];
}

template <typename input_type, typename output_type>
void tgv_gemm_impl(input_type* mat1_ptr, input_type* mat2_ptr, output_type* output_ptr,
                   output_type* bias_ptr, int M, int N, int K, int stride_A_M, int stride_A_K,
                   int stride_A_L, int stride_B_N, int stride_B_K, int stride_B_L, int stride_C_M,
                   int stride_C_N, int stride_C_L, int cta_m, int cta_n, int dma_stage, bool pdl,
                   cudaStream_t stream) {
  // Kernel config constants
  using TypeA = input_type;
  using TypeB = input_type;
  using TypeC = output_type;
  using AccType = float;
  using TypeBias = TypeC;
  // only supports K major now
  static constexpr cute::UMMA::Major UmmaMajorA = cute::UMMA::Major::K;
  static constexpr cute::UMMA::Major UmmaMajorB = cute::UMMA::Major::K;
  static constexpr int CTA_K = 128;  // Fixed for now

  // Function pointer for the selected template instantiation
  GemmFuncPtr<TypeA, TypeB, TypeC, AccType, TypeBias> func_ptr = nullptr;

  dispatch_kernel<TypeA, TypeB, TypeC, AccType, TypeBias, UmmaMajorA, UmmaMajorB>(
      cta_m, cta_n, CTA_K, dma_stage, &func_ptr);

  // Call the selected function
  func_ptr(mat1_ptr, mat2_ptr, output_ptr, bias_ptr, M, N, K, 1, stride_A_M, stride_A_K, stride_A_L,
           stride_B_N, stride_B_K, stride_B_L, stride_C_M, stride_C_N, stride_C_L, pdl, -1,
           stream);  // pdl_count=-1 for gemm
}

}  // namespace

void tgv_gemm(TensorView mat1, TensorView mat2, Optional<TensorView> bias, int64_t tactic,
              TensorView out, bool pdl) {
  // Input validation
  TVM_FFI_ICHECK_EQ(mat1.device().device_type, kDLCUDA) << "mat1 tensor must be on CUDA";
  TVM_FFI_ICHECK_EQ(mat2.device().device_type, kDLCUDA) << "mat2 tensor must be on CUDA";
  TVM_FFI_ICHECK_EQ(mat1.ndim(), 2) << "mat1 tensor must be 2D (M, K)";
  TVM_FFI_ICHECK_EQ(mat2.ndim(), 2) << "mat2 tensor must be 2D (K, N)";
  TVM_FFI_ICHECK_EQ(mat1.size(1), mat2.size(0)) << "mat1.K must match mat2.K";
  TVM_FFI_ICHECK_EQ(mat1.dtype(), mat2.dtype()) << "mat1 and mat2 must have the same dtype";

  // No heuristic for now, we use 64x8 with 8 DMA stages as the default tactic.
  if (tactic == -1) {
    tactic = 1;
  }
  auto config = getTgvGemmConfig(tactic);

  // Get tile parameters from config
  int cta_m, cta_n, dma_stage;
  config.getTileParams(cta_m, cta_n, dma_stage);

  // Validate DMA_Stage
  TVM_FFI_ICHECK(dma_stage == 6 || dma_stage == 8 || dma_stage == 10 || dma_stage == 12)
      << "dma_stage must be one of: 6, 8, 10, 12";

  // Validate tile sizes
  TVM_FFI_ICHECK(cta_m == 64 || cta_m == 128) << "cta_m must be one of: 64, 128";

  // Get dimensions
  int M = mat1.size(0);
  int K = mat1.size(1);
  int N = mat2.size(1);

  int64_t element_size = get_element_size(mat1);
  TVM_FFI_ICHECK(int64_t(M) * N * element_size < std::numeric_limits<int32_t>::max())
      << "TMA plane stride (M * N * element_size) exceeds INT32_MAX; tensor too large for TMA";
  TVM_FFI_ICHECK(int64_t(M) * K * element_size < std::numeric_limits<int32_t>::max())
      << "TMA plane stride (M * K * element_size) exceeds INT32_MAX; mat1 too large for TMA";
  TVM_FFI_ICHECK(int64_t(N) * K * element_size < std::numeric_limits<int32_t>::max())
      << "TMA plane stride (N * K * element_size) exceeds INT32_MAX; mat2 too large for TMA";

  // validity check for bias
  if (bias.has_value()) {
    TVM_FFI_ICHECK_EQ(bias.value().device().device_type, kDLCUDA) << "Bias tensor must be on CUDA";
    TVM_FFI_ICHECK_EQ(bias.value().ndim(), 1) << "Bias tensor must be 1D (M,)";
    TVM_FFI_ICHECK_EQ(bias.value().size(0), M) << "Bias tensor must have M elements";
    TVM_FFI_ICHECK_EQ(bias.value().dtype(), mat1.dtype())
        << "Bias tensor must have the same dtype as input matrices";
    TVM_FFI_ICHECK_EQ(bias.value().stride(0), 1) << "Bias tensor must be M contiguous";
  }

  // Create output tensor [N, M] row major
  TVM_FFI_ICHECK_EQ(out.size(0), N);
  TVM_FFI_ICHECK_EQ(out.size(1), M);
  TVM_FFI_ICHECK_EQ(out.dtype(), mat1.dtype());

  // manually calculate the L stride
  // A [M, K] row major
  int stride_A_M = mat1.stride(0);
  int stride_A_K = mat1.stride(1);
  int stride_A_L = M * K;
  // B [K, N] column major
  int stride_B_N = mat2.stride(1);
  int stride_B_K = mat2.stride(0);
  int stride_B_L = N * K;
  // original C [N, M] row major
  int stride_C_M = out.stride(1);
  int stride_C_N = out.stride(0);
  int stride_C_L = M * N;

  // Get CUDA stream
  cudaStream_t stream = get_stream(out.device());

  // Dispatch based on dtype
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(mat1.dtype(), c_type, [&] {
    using cutlass_input_type = flashinfer::cutlass_dtype_t<c_type>;
    using cutlass_output_type = flashinfer::cutlass_dtype_t<c_type>;

    cutlass_input_type* mat1_ptr = static_cast<cutlass_input_type*>(mat1.data_ptr());
    cutlass_input_type* mat2_ptr = static_cast<cutlass_input_type*>(mat2.data_ptr());
    cutlass_output_type* output_ptr = static_cast<cutlass_output_type*>(out.data_ptr());
    cutlass_output_type* bias_ptr =
        bias.has_value() ? static_cast<cutlass_output_type*>(bias.value().data_ptr()) : nullptr;

    tgv_gemm_impl<cutlass_input_type, cutlass_output_type>(
        mat1_ptr, mat2_ptr, output_ptr, bias_ptr, M, N, K, stride_A_M, stride_A_K, stride_A_L,
        stride_B_N, stride_B_K, stride_B_L, stride_C_M, stride_C_N, stride_C_L, cta_m, cta_n,
        dma_stage, pdl, stream);
    return true;
  });
}

// Keep backward compatibility functions
void bf16_gemm(TensorView mat1, TensorView mat2, std::optional<TensorView> bias, int64_t tactic,
               TensorView out, bool pdl) {
  // Check that inputs are bfloat16 for backward compatibility
  TVM_FFI_ICHECK_EQ(mat1.dtype(), dl_bfloat16) << "mat1 tensor must be bfloat16";
  TVM_FFI_ICHECK_EQ(mat2.dtype(), dl_bfloat16) << "mat2 tensor must be bfloat16";
  tgv_gemm(mat1, mat2, bias, tactic, out, pdl);
}

int64_t tgv_gemm_tactic_num() {
  static int64_t totalTactics = getAllTgvConfigs().size();
  return totalTactics;
}

int64_t bf16_gemm_tactic_num() { return tgv_gemm_tactic_num(); }

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tgv_gemm, torch_ext::tgv_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tgv_gemm_tactic_num, torch_ext::tgv_gemm_tactic_num);
