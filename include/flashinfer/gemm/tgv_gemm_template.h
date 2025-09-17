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

#ifndef FLASHINFER_TGV_GEMM_TEMPLATE_H_
#define FLASHINFER_TGV_GEMM_TEMPLATE_H_

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "flashinfer/gemm/tgv_gemm.cuh"

namespace flashinfer {
namespace gemm {

#define INSTANTIATE_TGV_GEMM_HOST(CTA_M, CTA_N, DMA_Stage, dtype, out_dtype, UmmaMajorA,           \
                                  UmmaMajorB)                                                      \
  template void tgv_gemm_host<dtype, dtype, out_dtype, float, out_dtype, CTA_M, CTA_N, 128,        \
                              DMA_Stage, UmmaMajorA, UmmaMajorB>(                                  \
      dtype * device_ptr_A, dtype * device_ptr_B, out_dtype * device_ptr_C,                        \
      out_dtype * device_ptr_Bias, int Gemm_M, int Gemm_N, int Gemm_K, int Gemm_L, int stride_A_M, \
      int stride_A_K, int stride_A_L, int stride_B_N, int stride_B_K, int stride_B_L,              \
      int stride_C_M, int stride_C_N, int stride_C_L, bool pdl, int pdl_count,                     \
      cudaStream_t stream);

#define INSTANTIATE_TGV_GEMM_HOST_BF16_BF16(CTA_M, CTA_N, DMA_Stage)                           \
  INSTANTIATE_TGV_GEMM_HOST(CTA_M, CTA_N, DMA_Stage, cutlass::bfloat16_t, cutlass::bfloat16_t, \
                            cute::UMMA::Major::K, cute::UMMA::Major::K)

#define INSTANTIATE_TGV_GEMM_HOST_FP16_FP16(CTA_M, CTA_N, DMA_Stage)                   \
  INSTANTIATE_TGV_GEMM_HOST(CTA_M, CTA_N, DMA_Stage, cutlass::half_t, cutlass::half_t, \
                            cute::UMMA::Major::K, cute::UMMA::Major::K)

}  // namespace gemm
}  // namespace flashinfer

#endif
