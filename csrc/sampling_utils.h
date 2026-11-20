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

#include <cstdint>
#include <numeric>

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

// Widest vector load of the sampling kernels, in bytes.
constexpr uint64_t kSamplingVectorLoadBytes = 16;

// The sampling kernels read each row of `tensor` with vector loads of
// gcd(kSamplingVectorLoadBytes / element size, vocab_size) elements. Rows are vocab_size
// elements apart, so every load is aligned exactly when the first one is. A contiguous view
// that starts at an unaligned storage offset passes the other input checks and makes the
// kernel fault with cudaErrorMisalignedAddress, an asynchronous error that leaves the CUDA
// context unusable. Refuse such a view before anything is launched.
inline void check_vector_load_alignment(const TensorView& tensor, const char* name) {
  const uint64_t element_bytes = tensor.dtype().bits / 8;
  const uint64_t vocab_size = static_cast<uint64_t>(tensor.size(tensor.ndim() - 1));
  const uint64_t load_bytes =
      std::gcd(kSamplingVectorLoadBytes / element_bytes, vocab_size) * element_bytes;
  const uint64_t address = reinterpret_cast<uint64_t>(tensor.data_ptr());
  if (address % load_bytes != 0) {
    TVM_FFI_THROW(ValueError) << name << " must be " << load_bytes
                              << "-byte aligned for a last dimension of " << vocab_size
                              << ", but its data pointer is " << address % load_bytes
                              << " bytes past such a boundary. Pass a copy (for example with "
                              << ".clone()).";
  }
}

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
