/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/gemm/gemm_groupwise_sm100.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(input_dtype, output_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                                  \
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, c_type_out, [&] {                    \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(input_dtype, c_type_in,                           \
                                                 [&] { return __VA_ARGS__(); });                   \
    });                                                                                            \
  }()

void CutlassGemmGroupwiseScaledSM100(at::Tensor float_workspace_buffer, at::Tensor A, at::Tensor B,
                                     at::Tensor SFA, at::Tensor SFB, at::Tensor C) {
  unsigned int batch_size = A.size(0);
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(A.scalar_type(), C.scalar_type(), c_type_in, c_type_out, [&] {
    using cutlass_t_in = cutlass_dtype_t<c_type_in>;
    using cutlass_t_out = cutlass_dtype_t<c_type_out>;
    auto status = flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100(
        static_cast<float*>(float_workspace_buffer.data_ptr()),
        float_workspace_buffer.element_size() * float_workspace_buffer.size(0),
        static_cast<cutlass_t_in*>(A.data_ptr()), static_cast<cutlass_t_in*>(B.data_ptr()),
        static_cast<float*>(SFA.data_ptr()), static_cast<float*>(SFB.data_ptr()),
        static_cast<cutlass_t_out*>(C.data_ptr()), A.size(0), B.size(0), A.size(1), 1, stream);
    return true;
  });
}
