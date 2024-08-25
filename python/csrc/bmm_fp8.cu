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

void bmm_fp8(const torch::Tensor& input, const torch::Tensor& weight, torch::Tensor& result) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
  TORCH_CHECK(result.is_cuda(), "Result must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 3, "Expected 3D tensor for input");
  TORCH_CHECK(weight.dim() == 3, "Expected 3D tensor for weight");
  TORCH_CHECK(result.dim() == 3, "Expected 3D tensor for result");
  TORCH_CHECK(input.size(0) == weight.size(0) && input.size(0) == result.size(0),
              "Batch sizes must match");
  TORCH_CHECK(input.size(2) == weight.size(1), "Incompatible matrix sizes");
  TORCH_CHECK(input.size(1) == result.size(1) && weight.size(2) == result.size(2),
              "Result tensor has incorrect shape");
  TORCH_CHECK(input.scalar_type() == torch::kFloat8_e4m3fn, "input must be Float8_e4m3fn");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn, "weight must be Float8_e4m3fn");
  TORCH_CHECK(result.scalar_type() == torch::kBFloat16, "Result must be BFloat16");

  auto batch_size = input.size(0);
  auto m = input.size(1);
  auto k = input.size(2);
  auto n = weight.size(2);

  if (result.scalar_type() == at::ScalarType::BFloat16) {
    flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(static_cast<__nv_fp8_e4m3*>(weight.data_ptr()),
                                                   static_cast<__nv_fp8_e4m3*>(input.data_ptr()),
                                                   static_cast<__nv_bfloat16*>(result.data_ptr()),
                                                   batch_size, n, m, k);
  }
}
