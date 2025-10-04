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

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

void top_p_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_p_arr, double top_p_val) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs->shape[0];
  unsigned int vocab_size = probs->shape[1];
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs->data), static_cast<float*>(renorm_probs->data),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value()->data) : nullptr, batch_size,
      top_p_val, vocab_size, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopPRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_k_arr, int64_t top_k_val) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs->shape[0];
  unsigned int vocab_size = probs->shape[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(probs->data), static_cast<float*>(renorm_probs->data),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value()->data) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_mask_logits(TensorView logits, TensorView mask_logits,
                       Optional<TensorView> maybe_top_k_arr, int64_t top_k_val) {
  CHECK_INPUT(logits);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits->shape[0];
  unsigned int vocab_size = logits->shape[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  cudaSetDevice(logits->device.device_id);
  auto stream = get_stream(logits->device);
  cudaError_t status = sampling::TopKMaskLogits<float>(
      static_cast<float*>(logits->data), static_cast<float*>(mask_logits->data),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value()->data) : nullptr, batch_size,
      top_k_val, vocab_size, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKMaskLogits failed with error code " << cudaGetErrorString(status);
}
