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

using tvm::ffi::Optional;

void packbits(TensorView x, const std::string& bitorder, TensorView y);

void segment_packbits(TensorView x, TensorView input_indptr, TensorView output_indptr,
                      const std::string& bitorder, TensorView y);

void per_token_group_quant_8bit(TensorView input, TensorView output_q, TensorView output_s,
                                double eps, int64_t group_size, int64_t groups_per_row,
                                int64_t scale_stride, bool column_major, bool scale_ue8m0);

void csf_encode_rows(TensorView scales, TensorView fixed, TensorView row_exceptions);

void csf_encode_exceptions(TensorView scales, TensorView fixed, TensorView row_offsets,
                           TensorView exceptions);

void csf_expand(TensorView fixed, TensorView exceptions, TensorView exc_offsets,
                Optional<TensorView> token_selected_experts, TensorView out, int64_t layout,
                bool swap_row_halves, int64_t local_expert_offset);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(packbits, packbits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(segment_packbits, segment_packbits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(per_token_group_quant_8bit, per_token_group_quant_8bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(csf_encode_rows, csf_encode_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(csf_encode_exceptions, csf_encode_exceptions);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(csf_expand, csf_expand);
