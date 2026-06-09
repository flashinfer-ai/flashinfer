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

void CutlassGroupGemmMxfp8GroupwiseScaledCuteSM120ZeroPadding(
    TensorView a, TensorView b, TensorView a_scale, TensorView b_scale, TensorView m_indptr,
    TensorView out, std::string scale_major_mode, int64_t scale_granularity_m,
    int64_t scale_granularity_n, int64_t scale_granularity_k);

void CutlassGroupGemmMxfp8GroupwiseScaledCuteSM120Main(
    TensorView a, TensorView b, TensorView a_scale, TensorView b_scale, TensorView m_indices,
    TensorView out, bool use_psum_layout, int64_t scale_granularity_m, int64_t scale_granularity_n,
    int64_t scale_granularity_k);

void CutlassGroupGemmMxfp8GroupwiseScaledCuteSM120Masked(TensorView a, TensorView b,
                                                         TensorView a_scale, TensorView b_scale,
                                                         TensorView masked_m, TensorView out,
                                                         int64_t scale_granularity_m,
                                                         int64_t scale_granularity_n,
                                                         int64_t scale_granularity_k);

void QuantizeMxfp8ForZeroPaddingCuteSM120(TensorView input, TensorView token_offset,
                                          TensorView out_fp8, TensorView out_scale_raw,
                                          int64_t granK);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_mxfp8_nt_groupwise_zero_padding,
                              CutlassGroupGemmMxfp8GroupwiseScaledCuteSM120ZeroPadding);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_mxfp8_nt_groupwise,
                              CutlassGroupGemmMxfp8GroupwiseScaledCuteSM120Main);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_mxfp8_nt_groupwise_masked,
                              CutlassGroupGemmMxfp8GroupwiseScaledCuteSM120Masked);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_mxfp8_for_zero_padding,
                              QuantizeMxfp8ForZeroPaddingCuteSM120);
