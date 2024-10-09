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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <flashinfer/gemm/bmm_fp8.cuh>

#include "pytorch_extension_utils.h"

namespace flashinfer {
namespace bmm_fp8 {

template cublasStatus_t bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>(
    void* workspace, size_t workspace_size_in_bytes, const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    __nv_bfloat16* D, int batch_size, int m, int n, int k, const float* A_scale,
    const float* B_scale, cublasLtHandle_t lt_handle, cudaStream_t stream);

template cublasStatus_t bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e4m3, half>(
    void* workspace, size_t workspace_size_in_bytes, const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    half* D, int batch_size, int m, int n, int k, const float* A_scale, const float* B_scale,
    cublasLtHandle_t lt_handle, cudaStream_t stream);

template cublasStatus_t bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_bfloat16>(
    void* workspace, size_t workspace_size_in_bytes, const __nv_fp8_e4m3* A, const __nv_fp8_e5m2* B,
    __nv_bfloat16* D, int batch_size, int m, int n, int k, const float* A_scale,
    const float* B_scale, cublasLtHandle_t lt_handle, cudaStream_t stream);

template cublasStatus_t bmm_fp8_internal_cublaslt<__nv_fp8_e4m3, __nv_fp8_e5m2, half>(
    void* workspace, size_t workspace_size_in_bytes, const __nv_fp8_e4m3* A, const __nv_fp8_e5m2* B,
    half* D, int batch_size, int m, int n, int k, const float* A_scale, const float* B_scale,
    cublasLtHandle_t lt_handle, cudaStream_t stream);

template cublasStatus_t bmm_fp8_internal_cublaslt<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_bfloat16>(
    void* workspace, size_t workspace_size_in_bytes, const __nv_fp8_e5m2* A, const __nv_fp8_e4m3* B,
    __nv_bfloat16* D, int batch_size, int m, int n, int k, const float* A_scale,
    const float* B_scale, cublasLtHandle_t lt_handle, cudaStream_t stream);

template cublasStatus_t bmm_fp8_internal_cublaslt<__nv_fp8_e5m2, __nv_fp8_e4m3, half>(
    void* workspace, size_t workspace_size_in_bytes, const __nv_fp8_e5m2* A, const __nv_fp8_e4m3* B,
    half* D, int batch_size, int m, int n, int k, const float* A_scale, const float* B_scale,
    cublasLtHandle_t lt_handle, cudaStream_t stream);

}  // namespace bmm_fp8
}  // namespace flashinfer

void bmm_fp8(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& D,
             torch::Tensor& A_scale, torch::Tensor& B_scale) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 3, "Expected 3D tensor for A");
  TORCH_CHECK(B.dim() == 3, "Expected 3D tensor for B");
  TORCH_CHECK(D.dim() == 3, "Expected 3D tensor for D");
  TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0), "Batch sizes must match");
  TORCH_CHECK(A.size(2) == B.size(1), "Incompatible matrix sizes");
  TORCH_CHECK(A.size(1) == D.size(1) && B.size(2) == D.size(2),
              "Result tensor has incorrect shape");
  TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fn || A.scalar_type() == torch::kFloat8_e5m2,
              "A must be Float8_e4m3fn or Float8_e5m2");
  TORCH_CHECK(B.scalar_type() == torch::kFloat8_e4m3fn || B.scalar_type() == torch::kFloat8_e5m2,
              "B must be Float8_e4m3fn or Float8_e5m2");
  TORCH_CHECK(D.scalar_type() == torch::kBFloat16 || D.scalar_type() == torch::kHalf,
              "D must be BFloat16 or Half");

  TORCH_CHECK(A_scale.scalar_type() == torch::kFloat32 && B_scale.scalar_type() == torch::kFloat32,
              "A_scale and B_scale must be Float32");

  auto batch_size = A.size(0);
  auto m = A.size(1);
  auto k = A.size(2);
  auto n = B.size(2);

  // Per the cublas documentation, the recommended workspace buffer size for hopper is 32MB.
  // https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
  // create an empty buffer of 32MB, with data type uint8 and on the same device as A
  auto workspace_buffer = torch::empty(
      {32 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
  auto lt_handle = reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasHandle());
  auto stream = at::cuda::getCurrentCUDAStream();

  // PyTorch is row major by default. cuBLASLt is column major by default.
  // We need row major D as expected.
  // A ^ T * B = D, so D ^ T = B ^ T * A
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(B.scalar_type(), b_type, [&] {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(A.scalar_type(), a_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(D.scalar_type(), d_type, [&] {
        auto status = flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(
            workspace_buffer.data_ptr(), workspace_buffer.numel(),
            static_cast<b_type*>(B.data_ptr()), static_cast<a_type*>(A.data_ptr()),
            static_cast<d_type*>(D.data_ptr()), batch_size, n, m, k,
            static_cast<float*>(B_scale.data_ptr()), static_cast<float*>(A_scale.data_ptr()),
            lt_handle, stream);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "bmm_fp8_internal_cublaslt failed: ",
                    cublasGetStatusString(status));
        return true;
      });
    });
  });
}
