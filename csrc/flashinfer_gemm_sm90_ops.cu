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
#include "pytorch_extension_utils.h"

void CutlassSegmentGEMMSM90(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor all_problems, at::Tensor x_ptr, at::Tensor w_ptr,
                            at::Tensor y_ptr, at::Tensor x_stride, at::Tensor weight_stride,
                            at::Tensor y_stride, at::Tensor empty_x_data, bool weight_column_major,
                            std::vector<int64_t> plan_info_vec, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_segment_gemm_sm90", &CutlassSegmentGEMMSM90,
        "Cutlass Segment GEMM operator for SM90");
}
