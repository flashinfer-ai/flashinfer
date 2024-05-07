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
#include <flashinfer/norm.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_DIM(2, x);  // x: (batch_size, hidden_size)
  CHECK_DIM(1, w);  // w: (hidden_size)
  CHECK_EQ(x.size(1), w.size(0));
  unsigned int batch_size = x.size(0);
  unsigned int hidden_size = x.size(1);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto y = torch::empty_like(x);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(x.scalar_type(), c_type, [&] {
    cudaError_t status = norm::RMSNorm(
        static_cast<c_type*>(x.data_ptr()), static_cast<c_type*>(w.data_ptr()),
        static_cast<c_type*>(y.data_ptr()), batch_size, hidden_size, eps, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "RMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
  return y;
}