/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/sampling.cuh>

#include "sampling_utils.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

void top_p_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_p_arr, double top_p_val) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_p_arr, probs);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());
  cudaError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopPRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                        TensorView row_states_buffer) {
  CHECK_INPUT(probs);
  CHECK_INPUT(row_states_buffer);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_k_arr, probs);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  cudaError_t status;
  auto dtype = probs.dtype();

  // Use radix-based top-k with dtype dispatch for FP32/FP16/BF16
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKRenormProbMultiCTA<c_type, int>(
        static_cast<c_type*>(probs.data_ptr()), static_cast<c_type*>(renorm_probs.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value().data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, static_cast<sampling::RadixRowState*>(row_states_buffer.data_ptr()),
        stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_mask_logits(TensorView logits, TensorView mask_logits,
                       Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                       TensorView row_states_buffer) {
  CHECK_INPUT(logits);
  CHECK_INPUT(row_states_buffer);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  check_tensor_param(maybe_top_k_arr, logits);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());

  cudaError_t status;
  auto dtype = logits.dtype();

  // Use radix-based top-k with auto-selection (single-CTA for small vocab, multi-CTA for large
  // vocab)
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKMaskLogitsMultiCTA<c_type, int>(
        static_cast<c_type*>(logits.data_ptr()), static_cast<c_type*>(mask_logits.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value().data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, static_cast<sampling::RadixRowState*>(row_states_buffer.data_ptr()),
        stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKMaskLogits failed with error code " << cudaGetErrorString(status);
}
