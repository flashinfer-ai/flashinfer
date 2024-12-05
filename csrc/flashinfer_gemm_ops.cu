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

void bmm_fp8(at::Tensor A, at::Tensor B, at::Tensor D, at::Tensor A_scale, at::Tensor B_scale,
             at::Tensor workspace_buffer, int64_t cublas_handle, int64_t cuda_stream);

void CutlassSegmentGEMM(at::Tensor workspace_buffer, at::Tensor all_problems, at::Tensor x_ptr,
                        at::Tensor w_ptr, at::Tensor y_ptr, at::Tensor x_ld, at::Tensor w_ld,
                        at::Tensor y_ld, at::Tensor empty_x_data, bool weight_column_major,
                        int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_segment_gemm", &CutlassSegmentGEMM, "Cutlass Segment GEMM");
  m.def("bmm_fp8", &bmm_fp8, "BMM FP8");
}
