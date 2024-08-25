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

#include <flashinfer/bmm_fp8.cuh>

#include "flashinfer_ops.h"

using namespace flashinfer;

void bmm_fp8(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& D) {
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
  TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fn, "A must be Float8_e4m3fn");
  TORCH_CHECK(B.scalar_type() == torch::kFloat8_e4m3fn, "B must be Float8_e4m3fn");
  TORCH_CHECK(D.scalar_type() == torch::kBFloat16, "D must be BFloat16");

  auto batch_size = A.size(0);
  auto m = A.size(1);
  auto k = A.size(2);
  auto n = B.size(2);

  // PyTorch is row major by default. cuBLASLt is column major by default.
  // We need row major D as expected.
  // A ^ T * B = D, so D ^ T = B ^ T * A
  if (D.scalar_type() == at::ScalarType::BFloat16) {
    flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(
        static_cast<__nv_fp8_e4m3*>(B.data_ptr()), static_cast<__nv_fp8_e4m3*>(A.data_ptr()),
        static_cast<__nv_bfloat16*>(D.data_ptr()), batch_size, n, m, k);
  }
}
