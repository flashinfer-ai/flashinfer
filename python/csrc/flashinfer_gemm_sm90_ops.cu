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
#include <torch/extension.h>

void CutlassSegmentGEMMSM90(torch::Tensor float_workspace_buffer,
                                     torch::Tensor int_workspace_buffer, torch::Tensor all_problems,
                                     torch::Tensor x_ptr, torch::Tensor w_ptr, torch::Tensor y_ptr,
                                     torch::Tensor x_stride, torch::Tensor weight_stride,
                                     torch::Tensor y_stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_segment_gemm_sm90", &CutlassSegmentGEMMSM90,
        "Cutlass Segment GEMM operator for SM90");
}
