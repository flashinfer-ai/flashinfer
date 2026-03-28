/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_MM_BF16_CUBLASLT_CUH_
#define FLASHINFER_GEMM_MM_BF16_CUBLASLT_CUH_

#include <cublasLt.h>
#include <cuda_bf16.h>

#include <array>

#include "bmm_fp8.cuh"

namespace flashinfer {
namespace mm_bf16_cublaslt {

using bmm_fp8::CuBlasLtMatmulDescriptor;
using bmm_fp8::CuBlasLtMatmulPreference;
using bmm_fp8::CuBlasLtMatrixLayout;

static constexpr int kMaxAlgorithms = 100;

/*!
 * \brief Set up cuBLASLt descriptors for BF16 GEMM in row-major convention.
 *
 * Python mm_bf16 passes mat1 (m,k) row-major and mat2 (n,k) row-major
 * (after b.transpose(-2,-1)). We want out (m,n) row-major.
 *
 * cuBLASLt is column-major, so we use the standard trick:
 *   out^T = mat2 @ mat1^T   (all column-major)
 *
 * Memory layouts:
 *   mat2 row-major (n,k) = col-major (k,n) ld=k  → cuBLASLt "A", TRANSA=T → (n,k)
 *   mat1 row-major (m,k) = col-major (k,m) ld=k  → cuBLASLt "B", TRANSB=N → (k,m)
 *   out  row-major (m,n) = col-major (n,m) ld=n   → cuBLASLt "D"
 *
 * Result: (n,k)×(k,m) = (n,m) col-major = (m,n) row-major ✓
 */
struct GemmDescriptors {
  CuBlasLtMatmulDescriptor matmul_desc;
  CuBlasLtMatrixLayout a_layout;  // mat2
  CuBlasLtMatrixLayout b_layout;  // mat1
  CuBlasLtMatrixLayout d_layout;  // out

  GemmDescriptors(int m, int n, int k, cudaDataType_t d_type)
      : matmul_desc(CUBLAS_COMPUTE_32F, CUDA_R_32F),
        a_layout(CUDA_R_16BF, n, k, k, /*t=*/true),
        b_layout(CUDA_R_16BF, k, m, k),
        d_layout(d_type, n, m, n) {
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
  }
};

/*!
 * \brief Get the number of available cuBLASLt algorithms for a BF16 GEMM.
 */
inline int get_algorithm_count(int m, int n, int k, cudaDataType_t d_type,
                               size_t workspace_size_in_bytes, cublasLtHandle_t lt_handle) {
  GemmDescriptors desc(m, n, k, d_type);

  CuBlasLtMatmulPreference preference;
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size_in_bytes);

  std::array<cublasLtMatmulHeuristicResult_t, kMaxAlgorithms> results;
  int returned_count = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle, desc.matmul_desc.descriptor(), desc.a_layout.descriptor(),
      desc.b_layout.descriptor(), desc.d_layout.descriptor(), desc.d_layout.descriptor(),
      preference.descriptor(), kMaxAlgorithms, results.data(), &returned_count);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return 0;
  }
  return returned_count;
}

/*!
 * \brief Run a BF16 GEMM using a specific cuBLASLt algorithm (tactic).
 *
 * \param mat1  Pointer to mat1 data, row-major (m, k)
 * \param mat2  Pointer to mat2 data, row-major (n, k) — after b.transpose(-2,-1) in Python
 * \param out   Pointer to output data, row-major (m, n)
 * \param tactic  Algorithm index; -1 means use the top heuristic (index 0).
 */
inline cublasStatus_t run(const __nv_bfloat16* mat1, const __nv_bfloat16* mat2, void* out, int m,
                          int n, int k, cudaDataType_t d_type, void* workspace,
                          size_t workspace_size_in_bytes, cublasLtHandle_t lt_handle,
                          cudaStream_t stream, int tactic) {
  GemmDescriptors desc(m, n, k, d_type);

  CuBlasLtMatmulPreference preference;
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size_in_bytes);

  int algo_idx = (tactic < 0) ? 0 : tactic;
  int request_count = algo_idx + 1;
  if (request_count > kMaxAlgorithms) {
    request_count = kMaxAlgorithms;
  }

  std::array<cublasLtMatmulHeuristicResult_t, kMaxAlgorithms> results;
  int returned_count = 0;
  cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle, desc.matmul_desc.descriptor(), desc.a_layout.descriptor(),
      desc.b_layout.descriptor(), desc.d_layout.descriptor(), desc.d_layout.descriptor(),
      preference.descriptor(), request_count, results.data(), &returned_count);
  if (heur_status != CUBLAS_STATUS_SUCCESS || returned_count <= algo_idx) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Note: mat2 is cuBLASLt "A", mat1 is cuBLASLt "B" (swap for row-major output)
  FLASHINFER_CUBLAS_CALL(
      cublasLtMatmul(lt_handle, desc.matmul_desc.descriptor(), &alpha, mat2,
                     desc.a_layout.descriptor(), mat1, desc.b_layout.descriptor(), &beta, nullptr,
                     desc.d_layout.descriptor(), out, desc.d_layout.descriptor(),
                     &results[algo_idx].algo, workspace, workspace_size_in_bytes, stream));
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace mm_bf16_cublaslt
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_MM_BF16_CUBLASLT_CUH_
