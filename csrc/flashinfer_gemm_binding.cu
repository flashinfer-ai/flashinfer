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

#include "tvm_ffi_utils.h"

void bmm_fp8(TensorView A, TensorView B, TensorView D, TensorView A_scale, TensorView B_scale,
             TensorView workspace_buffer, int64_t cublas_handle);

void CutlassSegmentGEMM(TensorView workspace_buffer, TensorView all_problems, TensorView x_ptr,
                        TensorView w_ptr, TensorView y_ptr, TensorView x_ld, TensorView w_ld,
                        TensorView y_ld, TensorView empty_x_data, bool weight_column_major);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cutlass_segment_gemm, CutlassSegmentGEMM);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp8, bmm_fp8);
