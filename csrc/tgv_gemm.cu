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

#include "cuda_runtime.h"
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

#include "flashinfer/gemm/tgv_gemm.cuh"
#include "flashinfer/gemm/tgv_gemm_configs.h"
#include "pytorch_extension_utils.h"

// CUTLASS type includes
#include <cutlass/numeric_types.h>

#define SUPPORTED_TGV_GEMM_CONFIGS \
    TGV_GEMM_CONFIG(64, 8, 6)  \
    TGV_GEMM_CONFIG(64, 8, 8) \
    TGV_GEMM_CONFIG(64, 8, 10) \
    TGV_GEMM_CONFIG(64, 8, 12) \
    TGV_GEMM_CONFIG(64, 16, 6) \
    TGV_GEMM_CONFIG(64, 16, 8) \
    TGV_GEMM_CONFIG(64, 16, 10) \
    TGV_GEMM_CONFIG(64, 32, 6) \
    TGV_GEMM_CONFIG(64, 32, 8) \
    TGV_GEMM_CONFIG(64, 64, 6) \
    TGV_GEMM_CONFIG(128, 16, 6) 
    

#define TGV_GEMM_CONFIG(CTA_M, CTA_N, DMA_STAGE) \
    if (cta_m == CTA_M && cta_n == CTA_N && dma_stage == DMA_STAGE) { \
        *func_ptr = &flashinfer::gemm::tgv_gemm_host<TypeA, TypeB, TypeC, AccType, TypeBias, CTA_M, CTA_N, 128, DMA_STAGE, UmmaMajorA, UmmaMajorB>; \
        return; \
    }

template <typename TypeA, typename TypeB, typename TypeC,
          typename AccType, typename TypeBias>
using GemmFuncPtr = void (*)(TypeA*, TypeB*, TypeC*, TypeBias*,
                             int, int, int, int,
                             int, int, int,
                             int, int, int,
                             int, int, int,
                             bool, int, cudaStream_t);
							 
template<typename TypeA, typename TypeB, typename TypeC, typename AccType, typename TypeBias, 
         cute::UMMA::Major UmmaMajorA, cute::UMMA::Major UmmaMajorB>
void dispatch_kernel(int cta_m, int cta_n, int cta_k, int dma_stage, 
                     GemmFuncPtr<TypeA, TypeB, TypeC, AccType, TypeBias>* func_ptr) {
    SUPPORTED_TGV_GEMM_CONFIGS
        
    TORCH_CHECK(false, "Unsupported tile configuration: cta_m=" + std::to_string(cta_m) + 
                ", cta_n=" + std::to_string(cta_n) + ", cta_k=" + std::to_string(cta_k)); 
}
#undef TGV_GEMM_CONFIG				

namespace torch_ext {

namespace {
// Use the shared function from the header file


using flashinfer::gemm::TGVGemmConfig;
using flashinfer::gemm::getAllTgvConfigs;

TGVGemmConfig getBf16GemmConfig(int64_t tactic) {
  auto globalConfigs = getAllTgvConfigs();
  
  TORCH_CHECK(tactic >= 0 && tactic < globalConfigs.size(), "tactic must be between 0 and ",
              globalConfigs.size());
  return globalConfigs[tactic];
}


at::Tensor bf16_gemm_impl(at::Tensor const& mat1, at::Tensor const& mat2, 
                          std::optional<at::Tensor> bias, int64_t tactic, bool pdl) {
  
  // No heuristic for now, we use 128x8 with 6 DMA stages as the default tactic.
  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getBf16GemmConfig(tactic);

  // Input validation
  TORCH_CHECK(mat1.is_cuda(), "mat1 tensor must be on CUDA");
  TORCH_CHECK(mat2.is_cuda(), "mat2 tensor must be on CUDA");
  TORCH_CHECK(mat1.dim() == 2, "mat1 tensor must be 2D (M, K)");
  TORCH_CHECK(mat2.dim() == 2, "mat2 tensor must be 2D (K, N)");
  TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat1.K must match mat2.K");

  // Check data types - only support bfloat16
  TORCH_CHECK(mat1.scalar_type() == at::ScalarType::BFloat16, "mat1 tensor must be bfloat16");
  TORCH_CHECK(mat2.scalar_type() == at::ScalarType::BFloat16, "mat2 tensor must be bfloat16");

  // Get tile parameters from config
  int cta_m, cta_n, dma_stage;
  config.getTileParams(cta_m, cta_n, dma_stage);
  
  // Validate DMA_Stage
  TORCH_CHECK(dma_stage == 6 || dma_stage == 8 || dma_stage == 10 || dma_stage == 12, 
              "dma_stage must be one of: 6, 8, 10, 12");

  // Validate tile sizes
  TORCH_CHECK(cta_m == 64 || cta_m == 128, "cta_m must be one of: 64, 128");
  static constexpr int CTA_K = 128; // Fixed for now

  // Get dimensions
  int M = mat1.size(0);
  int K = mat1.size(1);
  int N = mat2.size(1);

  // Create output tensor [N, M] row major
  at::Tensor C = at::detail::empty_cuda({N, M}, at::ScalarType::BFloat16, mat1.device(), std::nullopt);

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
  int stride_C_M = C.stride(1);
  int stride_C_N = C.stride(0);
  int stride_C_L = M * N;

  // validity check for bias
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
    TORCH_CHECK(bias.value().dim() == 1, "Bias tensor must be 1D (M,)");
    TORCH_CHECK(bias.value().size(0) == M, "Bias tensor must have M elements");
    TORCH_CHECK(bias.value().scalar_type() == at::ScalarType::BFloat16, "Bias tensor must be bfloat16");
    TORCH_CHECK(bias.value().stride(0) == 1, "Bias tensor must be M contiguous");
  }

  // Get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Kernel config constants
  using TypeA = cutlass::bfloat16_t;
  using TypeB = cutlass::bfloat16_t;
  using TypeC = cutlass::bfloat16_t;
  using AccType = float;
  using TypeBias = TypeC;
  // only supports K major now
  static constexpr cute::UMMA::Major UmmaMajorA = cute::UMMA::Major::K;
  static constexpr cute::UMMA::Major UmmaMajorB = cute::UMMA::Major::K;

  auto bias_ptr = bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr;

  // Function pointer for the selected template instantiation
  GemmFuncPtr<TypeA, TypeB, TypeC, AccType, TypeBias> func_ptr = nullptr;

  dispatch_kernel<TypeA, TypeB, TypeC, AccType, TypeBias, UmmaMajorA, UmmaMajorB>(cta_m, cta_n, CTA_K, dma_stage, &func_ptr);

  // Call the selected function
  func_ptr(reinterpret_cast<TypeA*>(mat1.data_ptr<at::BFloat16>()), 
          reinterpret_cast<TypeB*>(mat2.data_ptr<at::BFloat16>()), 
          reinterpret_cast<TypeC*>(C.data_ptr<at::BFloat16>()), 
          reinterpret_cast<TypeBias*>(bias_ptr),
          M, N, K, 1, 
          stride_A_M, stride_A_K, stride_A_L,
          stride_B_N, stride_B_K, stride_B_L,
          stride_C_M, stride_C_N, stride_C_L,
          pdl, -1, stream); // pdl_count=-1 for gemm

  // original C is [N, M] row major
  // after transpose, it's [M, N] column major
  // the storage is unchanged, only the logical coordinates are changed
  return C.t();
}

}  // namespace

at::Tensor bf16_gemm(at::Tensor const& mat1, at::Tensor const& mat2, 
                     std::optional<at::Tensor> bias, int64_t tactic, bool pdl) {
  return bf16_gemm_impl(mat1, mat2, bias, tactic, pdl);
}

int64_t bf16_gemm_tactic_num() {
  static int64_t totalTactics = getAllTgvConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext


TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_gemm", &torch_ext::bf16_gemm);
    m.def("bf16_gemm_tactic_num", &torch_ext::bf16_gemm_tactic_num);
}
