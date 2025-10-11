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
#include <string>

#include "tvm_ffi_utils.h"

void CutlassGroupGemmFP8GroupwiseScaledSM120(
    TensorView int_workspace_buffer, TensorView float_workspace_buffer, TensorView A, TensorView B,
    TensorView SFA, TensorView SFB, TensorView D, TensorView m_indptr, int64_t n, int64_t k,
    int64_t scale_granularity_m, int64_t scale_granularity_n, int64_t scale_granularity_k,
    std::string scale_major_mode);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_fp8_nt_groupwise, CutlassGroupGemmFP8GroupwiseScaledSM120);
