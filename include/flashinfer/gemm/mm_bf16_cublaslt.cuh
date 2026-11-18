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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

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

  GemmDescriptors(int m, int n, int k, cudaDataType_t d_type, const __nv_bfloat16* bias = nullptr)
      : matmul_desc(CUBLAS_COMPUTE_32F, CUDA_R_32F),
        a_layout(CUDA_R_16BF, n, k, k, /*t=*/true),
        b_layout(CUDA_R_16BF, k, m, k),
        d_layout(d_type, n, m, n) {
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
    if (bias != nullptr) {
      matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
      matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
      matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUDA_R_16BF);
    }
  }
};

/*!
 * \brief Query heuristics once and serialize all cublasLtMatmulAlgo_t structs into a buffer.
 *
 * Each algo occupies kAlgoBytes (64) contiguous bytes; any of them can later be passed
 * to run_with_descriptor(), including for a different M.
 *
 * \param algo_buf  Output buffer, must hold at least max_algos * kAlgoBytes bytes.
 * \param max_algos Maximum number of algorithms to retrieve.
 * \return Number of algorithms written to algo_buf.
 */
inline int get_algorithms(int m, int n, int k, cudaDataType_t d_type, const __nv_bfloat16* bias,
                          size_t workspace_size_in_bytes, cublasLtHandle_t lt_handle,
                          void* algo_buf, int max_algos) {
  GemmDescriptors desc(m, n, k, d_type, bias);

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
 * \brief Run a BF16 GEMM with a caller-provided cuBLASLt algorithm descriptor.
 *
 * \param algo_desc  A serialized cublasLtMatmulAlgo_t (kAlgoBytes), e.g. one returned by
 *                   get_algorithms() for another M, or null for the heuristic default. See
 *                   bmm_fp8::resolve_algo() for how it is validated for this problem.
 * \param used_descriptor  Set to whether \p algo_desc ran (false when the heuristic default ran).
 */
inline cublasStatus_t run_with_descriptor(const __nv_bfloat16* mat1, const __nv_bfloat16* mat2,
                                          void* out, const __nv_bfloat16* bias, int m, int n, int k,
                                          cudaDataType_t d_type, void* workspace,
                                          size_t workspace_size_in_bytes,
                                          cublasLtHandle_t lt_handle, cudaStream_t stream,
                                          int device_id, const void* algo_desc,
                                          bool* used_descriptor) {
  GemmDescriptors desc(m, n, k, d_type, bias);

  const auto bias_address = reinterpret_cast<uintptr_t>(bias);
  const int64_t bias_alignment =
      bias_address == 0 ? 0 : std::min<uintptr_t>(bias_address & -bias_address, 256);
  const std::string key = bmm_fp8::ResolvedAlgoCache::make_key(
      algo_desc, {bmm_fp8::kBf16ProblemTag, device_id, m, n, k, d_type, bias_alignment,
                  static_cast<int64_t>(workspace_size_in_bytes)});
  bmm_fp8::ResolvedAlgo resolved;
  FLASHINFER_CUBLAS_CALL(bmm_fp8::resolved_algo_cache().resolve(
      key, lt_handle, desc.matmul_desc.descriptor(), desc.a_layout.descriptor(),
      desc.b_layout.descriptor(), desc.d_layout.descriptor(), workspace_size_in_bytes, algo_desc,
      &resolved));
  *used_descriptor = resolved.from_descriptor;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  FLASHINFER_CUBLAS_CALL(cublasLtMatmul(
      lt_handle, desc.matmul_desc.descriptor(), &alpha, mat2, desc.a_layout.descriptor(), mat1,
      desc.b_layout.descriptor(), &beta, nullptr, desc.d_layout.descriptor(), out,
      desc.d_layout.descriptor(), &resolved.algo, workspace, workspace_size_in_bytes, stream));
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace mm_bf16_cublaslt
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_MM_BF16_CUBLASLT_CUH_
