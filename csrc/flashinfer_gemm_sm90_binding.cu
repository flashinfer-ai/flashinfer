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
#include "tvm_ffi_utils.h"

void CutlassSegmentGEMMSM90(Tensor float_workspace_buffer, Tensor int_workspace_buffer,
                            Tensor all_problems, Tensor x_ptr, Tensor w_ptr, Tensor y_ptr,
                            Tensor x_stride, Tensor weight_stride, Tensor y_stride,
                            Tensor empty_x_data, Tensor empty_y_data, bool weight_column_major);

// "Cutlass Segment GEMM operator for SM90"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cutlass_segment_gemm_sm90, CutlassSegmentGEMMSM90);
