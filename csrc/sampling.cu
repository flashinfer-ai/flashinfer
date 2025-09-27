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

void softmax(Tensor workspace_buffer, Tensor logits, Tensor output,
             Optional<Tensor> maybe_temperature_arr, double temperature_val, bool enable_pdl) {
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits->shape[0];
  unsigned int vocab_size = logits->shape[1];

  bool has_temperature_arr = maybe_temperature_arr.has_value();

  cudaSetDevice(logits->device.device_id);
  auto stream = get_stream(logits->device);
  cudaError_t status = sampling::OnlineSoftmax<float>(
      static_cast<float*>(logits->data), static_cast<float*>(output->data), batch_size, vocab_size,
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr.value()->data) : nullptr,
      temperature_val, workspace_buffer->data,
      get_element_size(workspace_buffer) * workspace_buffer->shape[0], enable_pdl, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "OnlineSoftmax failed with error code " << cudaGetErrorString(status);
}

void sampling_from_logits(Tensor logits, Tensor output, Optional<Tensor> maybe_indices,
                          bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  CHECK_INPUT(logits);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = logits->shape[1];

  cudaSetDevice(logits->device.device_id);
  auto stream = get_stream(logits->device);
  cudaError_t status = sampling::SamplingFromLogits(
      static_cast<float*>(logits->data), static_cast<int*>(output->data),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value()->data) : nullptr,
      batch_size, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "SamplingFromLogits failed with error code " << cudaGetErrorString(status);
}

void sampling_from_probs(Tensor probs, Tensor output, Optional<Tensor> maybe_indices,
                         bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = probs->shape[1];

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::SamplingFromProb(
      static_cast<float*>(probs->data), static_cast<int*>(output->data),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value()->data) : nullptr,
      batch_size, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "SamplingFromProbs failed with error code " << cudaGetErrorString(status);
}

void top_p_sampling_from_probs(Tensor probs, Tensor output, Optional<Tensor> maybe_indices,
                               Optional<Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = probs->shape[1];
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs->data), static_cast<int*>(output->data),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value()->data) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value()->data) : nullptr, batch_size,
      top_p_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopPSamplingFromProbs failed with error code " << cudaGetErrorString(status);
}

void top_k_sampling_from_probs(Tensor probs, Tensor output, Optional<Tensor> maybe_indices,
                               Optional<Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = probs->shape[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs->data), static_cast<int*>(output->data),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value()->data) : nullptr,
      has_top_k_arr ? static_cast<float*>(maybe_top_k_arr.value()->data) : nullptr, batch_size,
      top_k_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKSamplingFromProbs failed with error code " << cudaGetErrorString(status);
}

void min_p_sampling_from_probs(Tensor probs, Tensor output, Optional<Tensor> maybe_indices,
                               Optional<Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = probs->shape[1];
  bool has_min_p_arr = maybe_min_p_arr.has_value();

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::MinPSamplingFromProb<float, int>(
      static_cast<float*>(probs->data),
      has_min_p_arr ? static_cast<float*>(maybe_min_p_arr.value()->data) : nullptr,
      static_cast<int*>(output->data),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value()->data) : nullptr,
      batch_size, min_p_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "MinPSamplingFromProb failed with error code " << cudaGetErrorString(status);
}

void top_k_top_p_sampling_from_probs(Tensor probs, Tensor output, Optional<Tensor> maybe_indices,
                                     Optional<Tensor> maybe_top_k_arr, double top_k_val,
                                     Optional<Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = probs->shape[1];
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaSetDevice(probs->device.device_id);
  auto stream = get_stream(probs->device);
  cudaError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs->data),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value()->data) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value()->data) : nullptr,
      static_cast<int*>(output->data),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value()->data) : nullptr,
      batch_size, top_k_val, top_p_val, vocab_size, deterministic, philox_seed, philox_offset,
      stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKTopPSamplingFromProbs failed with error code " << cudaGetErrorString(status);
}

void chain_speculative_sampling(Tensor draft_probs, Tensor draft_token_ids, Tensor target_probs,
                                Tensor output_token_ids, Tensor output_accepted_token_num,
                                Tensor output_emitted_draft_token_num, bool deterministic,
                                uint64_t philox_seed, uint64_t philox_offset) {
  CHECK_INPUT(draft_probs);
  CHECK_INPUT(draft_token_ids);
  CHECK_INPUT(target_probs);
  CHECK_DEVICE(draft_token_ids, draft_probs);
  CHECK_DEVICE(target_probs, draft_probs);
  CHECK_DIM(3, draft_probs);      // draft_probs: (batch_size, num_speculate_tokens, vocab_size)
  CHECK_DIM(2, draft_token_ids);  // draft_token_ids: (batch_size, num_speculate_tokens)
  CHECK_DIM(3, target_probs);  // target_probs: (batch_size, num_speculate_tokens + 1, vocab_size)
  unsigned int batch_size = draft_probs->shape[0];
  unsigned int num_speculate_tokens = draft_probs->shape[1];
  unsigned int vocab_size = draft_probs->shape[2];
  TVM_FFI_ICHECK_EQ(batch_size, draft_token_ids->shape[0]);
  TVM_FFI_ICHECK_EQ(batch_size, target_probs->shape[0]);
  TVM_FFI_ICHECK_EQ(num_speculate_tokens + 1, target_probs->shape[1]);
  TVM_FFI_ICHECK_EQ(vocab_size, target_probs->shape[2]);
  TVM_FFI_ICHECK_EQ(batch_size, output_accepted_token_num->shape[0]);
  TVM_FFI_ICHECK_EQ(batch_size, output_emitted_draft_token_num->shape[0]);

  cudaSetDevice(draft_probs->device.device_id);
  auto stream = get_stream(draft_probs->device);
  cudaError_t status = sampling::ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs->data), static_cast<int*>(draft_token_ids->data),
      static_cast<float*>(target_probs->data), static_cast<int*>(output_token_ids->data),
      static_cast<int*>(output_accepted_token_num->data),
      static_cast<int*>(output_emitted_draft_token_num->data), batch_size, num_speculate_tokens,
      vocab_size, deterministic, philox_seed, philox_offset, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "ChainSpeculativeSampling failed with error code " << cudaGetErrorString(status);
}
