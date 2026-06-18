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
#pragma once
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

// Helper to validate sampling parameters
inline void check_tensor_param(const Optional<TensorView>& maybe_param, const TensorView& tensor) {
  if (maybe_param.has_value()) {
    const TensorView& param = maybe_param.value();
    if (param.ndim() == 0) {
      TVM_FFI_THROW(ValueError)
          << "Expected a 1D tensor of shape (batch_size,) or scalar for the sampling parameter, "
          << "but got a 0-dimensional tensor.";
    } else if (param.ndim() > 1) {
      TVM_FFI_THROW(ValueError) << "Expected a 1D tensor or scalar for the sampling parameter, "
                                << "but got a " << param.ndim() << "D tensor.";
    } else if (param.size(0) != tensor.size(0)) {
      TVM_FFI_THROW(ValueError) << "Sampling parameter tensor batch size mismatch: "
                                << "expected length " << tensor.size(0)
                                << " to match the reference tensor batch size, "
                                << "but got length " << param.size(0) << ".";
    }
  }
}
