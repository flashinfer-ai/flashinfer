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
#include <cstring>

#include "bmm_fp8.cuh"

namespace flashinfer {
namespace mm_bf16_cublaslt {

using bmm_fp8::CuBlasLtMatmulDescriptor;
using bmm_fp8::CuBlasLtMatmulPreference;
using bmm_fp8::CuBlasLtMatrixLayout;

static constexpr int kMaxAlgorithms = 100;
using bmm_fp8::kAlgoBytes;

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
 * \brief Query heuristics once and serialize all cublasLtMatmulAlgo_t structs into a buffer.
 *
 * Each algo occupies kAlgoBytes (64) contiguous bytes.  The buffer can be cached
 * and later passed to run_with_algo() to skip the heuristic lookup entirely.
 *
 * \param algo_buf  Output buffer, must hold at least max_algos * kAlgoBytes bytes.
 * \param max_algos Maximum number of algorithms to retrieve.
 * \return Number of algorithms written to algo_buf.
 */
inline int get_algorithms(int m, int n, int k, cudaDataType_t d_type,
                          size_t workspace_size_in_bytes, cublasLtHandle_t lt_handle,
                          void* algo_buf, int max_algos) {
  GemmDescriptors desc(m, n, k, d_type);

  CuBlasLtMatmulPreference preference;
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size_in_bytes);

  int request_count = (max_algos > kMaxAlgorithms) ? kMaxAlgorithms : max_algos;
  std::array<cublasLtMatmulHeuristicResult_t, kMaxAlgorithms> results;
  int returned_count = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle, desc.matmul_desc.descriptor(), desc.a_layout.descriptor(),
      desc.b_layout.descriptor(), desc.d_layout.descriptor(), desc.d_layout.descriptor(),
      preference.descriptor(), request_count, results.data(), &returned_count);
  if (status != CUBLAS_STATUS_SUCCESS) return 0;

  auto* out = static_cast<uint8_t*>(algo_buf);
  for (int i = 0; i < returned_count; ++i) {
    std::memcpy(out + i * kAlgoBytes, &results[i].algo, kAlgoBytes);
  }
  return returned_count;
}

/*!
 * \brief Run a BF16 GEMM using a pre-resolved algorithm — zero heuristic overhead.
 *
 * \param algo_buf  Buffer of serialized cublasLtMatmulAlgo_t structs (from get_algorithms).
 * \param algo_idx  Index into algo_buf selecting which algorithm to use.
 */
inline cublasStatus_t run_with_algo(const __nv_bfloat16* mat1, const __nv_bfloat16* mat2, void* out,
                                    int m, int n, int k, cudaDataType_t d_type, void* workspace,
                                    size_t workspace_size_in_bytes, cublasLtHandle_t lt_handle,
                                    cudaStream_t stream, const void* algo_buf, int algo_idx) {
  GemmDescriptors desc(m, n, k, d_type);

  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo, static_cast<const uint8_t*>(algo_buf) + algo_idx * kAlgoBytes, kAlgoBytes);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  FLASHINFER_CUBLAS_CALL(cublasLtMatmul(
      lt_handle, desc.matmul_desc.descriptor(), &alpha, mat2, desc.a_layout.descriptor(), mat1,
      desc.b_layout.descriptor(), &beta, nullptr, desc.d_layout.descriptor(), out,
      desc.d_layout.descriptor(), &algo, workspace, workspace_size_in_bytes, stream));
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace mm_bf16_cublaslt
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_MM_BF16_CUBLASLT_CUH_
