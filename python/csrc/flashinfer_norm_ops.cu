/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "pytorch_extension_utils.h"

void rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double eps,
             int64_t cuda_stream);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
                       int64_t cuda_stream);

void gemma_rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double eps,
                   int64_t cuda_stream);

void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight,
                             double eps, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "Root mean square normalization");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add root mean square normalization");
  m.def("gemma_rmsnorm", &gemma_rmsnorm, "Gemma Root mean square normalization");
  m.def("gemma_fused_add_rmsnorm", &gemma_fused_add_rmsnorm,
        "Gemma Fused add root mean square normalization");
}
