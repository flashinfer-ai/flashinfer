/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_BMM_FP8_CUH_
#define FLASHINFER_BMM_FP8_CUH_

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <stdexcept>
#include <type_traits>

namespace flashinfer {

namespace bmm_fp8 {

template <typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const { return descriptor_.get(); }
  T* descriptor() { return descriptor_.get(); }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor
    : public CuBlasLtDescriptor<cublasLtMatmulDescOpaque_t, &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(cublasComputeType_t compute_type, cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatrixLayout
    : public CuBlasLtDescriptor<cublasLtMatrixLayoutOpaque_t, &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld,
                       bool t = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutCreate(&raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatrixLayoutAttribute_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<cublasLtMatmulPreferenceOpaque_t,
                                                           &cublasLtMatmulPreferenceDestroy> {
 public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulPreferenceAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(
        ::cublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

template <typename T>
cudaDataType_t get_cuda_data_type() {
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return CUDA_R_8F_E4M3;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return CUDA_R_8F_E5M2;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return CUDA_R_16BF;
  } else if constexpr (std::is_same_v<T, half>) {
    return CUDA_R_16F;
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

template <typename AT, typename BT, typename DT>
void bmm_fp8_internal_cublaslt(const AT* A, const BT* B, DT* D, int batch_size, int m, int n,
                               int k) {
  auto matmul_desp = CuBlasLtMatmulDescriptor(CUBLAS_COMPUTE_32F, CUDA_R_32F);
  matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
  matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
  int8_t fast_accum = 1;
  matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);

  cudaDataType_t a_type = get_cuda_data_type<AT>();
  cudaDataType_t b_type = get_cuda_data_type<BT>();
  cudaDataType_t d_type = get_cuda_data_type<DT>();
  if (std::is_same_v<AT, __nv_fp8_e5m2> && std::is_same_v<BT, __nv_fp8_e5m2>) {
    throw std::runtime_error("Unsupported combination: both A and B are e5m2");
  }

  auto a_desp = CuBlasLtMatrixLayout(a_type, m, k, k, true);
  auto b_desp = CuBlasLtMatrixLayout(b_type, k, n, k);
  auto d_desp = CuBlasLtMatrixLayout(d_type, m, n, m);

  if (batch_size > 1) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_d = m * n;
    a_desp.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    a_desp.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
    b_desp.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    b_desp.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);
    d_desp.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    d_desp.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_d);
  }

  CuBlasLtMatmulPreference preference;
  size_t workspace_size = 1024 * 1024;  // 1 MiB
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size);
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspace_size);
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_result = 0;
  auto lt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt_handle, matmul_desp.descriptor(), a_desp.descriptor(), b_desp.descriptor(),
      d_desp.descriptor(), d_desp.descriptor(), preference.descriptor(), 1, &heuristic_result,
      &returned_result));
  if (returned_result == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(
      lt_handle, matmul_desp.descriptor(), &alpha, A, a_desp.descriptor(), B, b_desp.descriptor(),
      &beta, nullptr, d_desp.descriptor(), D, d_desp.descriptor(), &heuristic_result.algo,
      workspace.mutable_get(), workspace_size, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, at::cuda::blas::_cublasGetErrorEnum(status));
}

template void bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_bfloat16* D, int batch_size, int m, int n,
    int k);

template void bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e4m3, half>(const __nv_fp8_e4m3* A,
                                                                            const __nv_fp8_e4m3* B,
                                                                            half* D, int batch_size,
                                                                            int m, int n, int k);

template void bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_bfloat16>(
    const __nv_fp8_e4m3* A, const __nv_fp8_e5m2* B, __nv_bfloat16* D, int batch_size, int m, int n,
    int k);

template void bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e5m2, half>(const __nv_fp8_e4m3* A,
                                                                            const __nv_fp8_e5m2* B,
                                                                            half* D, int batch_size,
                                                                            int m, int n, int k);

template void bmm_fp8_internal_cublaslt<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_bfloat16>(
    const __nv_fp8_e5m2* A, const __nv_fp8_e4m3* B, __nv_bfloat16* D, int batch_size, int m, int n,
    int k);

template void bmm_fp8_internal_cublaslt<__nv_fp8_e5m2, __nv_fp8_e4m3, half>(const __nv_fp8_e5m2* A,
                                                                            const __nv_fp8_e4m3* B,
                                                                            half* D, int batch_size,
                                                                            int m, int n, int k);

}  // namespace bmm_fp8
}  // namespace flashinfer

#endif  // FLASHINFER_BMM_FP8_CUH_
