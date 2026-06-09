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

void CutlassGemmMxfp8GroupwiseScaledCuteSM120(TensorView a, TensorView b, TensorView a_scale,
                                              TensorView b_scale, TensorView out,
                                              int64_t scale_granularity_m,
                                              int64_t scale_granularity_n,
                                              int64_t scale_granularity_k);

void CutlassBatchGemmMxfp8GroupwiseScaledCuteSM120(
    TensorView a, TensorView b, TensorView a_scale, TensorView b_scale, TensorView out, int64_t lda,
    int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t ldd, int64_t stride_d,
    int64_t scale_granularity_m, int64_t scale_granularity_n, int64_t scale_granularity_k);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_mxfp8_nt_groupwise, CutlassGemmMxfp8GroupwiseScaledCuteSM120);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_gemm_mxfp8_nt_groupwise,
                              CutlassBatchGemmMxfp8GroupwiseScaledCuteSM120);
