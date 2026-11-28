/* SPDX-License-Identifier: Apache-2.0 */
#include "tvm_ffi_utils.h"
void apply_mrope(TensorView q, TensorView k, TensorView positions, TensorView cache,
                 int64_t head_size, int64_t st, int64_t sh, int64_t sw, bool axis_interleaved);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_mrope, apply_mrope);
void validate_mrope(TensorView q, TensorView k, TensorView positions, TensorView cache,
                    int64_t head_size, int64_t st, int64_t sh, int64_t sw, bool axis_interleaved);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(validate_mrope, validate_mrope);
