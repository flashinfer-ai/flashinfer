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
#include <ATen/Utils.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <flashinfer/sampling.cuh>
#include <mutex>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void softmax(at::Tensor workspace_buffer, at::Tensor logits, at::Tensor output,
             std::optional<at::Tensor> maybe_temperature_arr, double temperature_val,
             bool enable_pdl) {
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);

  bool has_temperature_arr = maybe_temperature_arr.has_value();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::OnlineSoftmax<float>(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(output.data_ptr()), batch_size,
      vocab_size,
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr->data_ptr()) : nullptr,
      temperature_val, workspace_buffer.data_ptr(),
      workspace_buffer.element_size() * workspace_buffer.size(0), enable_pdl, stream);
  TORCH_CHECK(status == cudaSuccess,
              "OnlineSoftmax failed with error code " + std::string(cudaGetErrorString(status)));
}

void sampling_from_logits(at::Tensor logits, at::Tensor output,
                          std::optional<at::Tensor> maybe_indices, bool deterministic,
                          std::optional<at::Generator> gen_) {
  CHECK_INPUT(logits);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = logits.size(1);

  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(batch_size * vocab_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::SamplingFromLogits(
      static_cast<float*>(logits.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TORCH_CHECK(status == cudaSuccess, "SamplingFromLogits failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void sampling_from_probs(at::Tensor probs, at::Tensor output,
                         std::optional<at::Tensor> maybe_indices, bool deterministic,
                         std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);

  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(batch_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::SamplingFromProb(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TORCH_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(32 * batch_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(32 * batch_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      has_top_k_arr ? static_cast<float*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TORCH_CHECK(status == cudaSuccess, "TopKSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void min_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_min_p_arr = maybe_min_p_arr.has_value();
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(batch_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::MinPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()),
      has_min_p_arr ? static_cast<float*>(maybe_min_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, min_p_val, vocab_size, deterministic, philox_seed, philox_offset, stream);
  TORCH_CHECK(status == cudaSuccess, "MinPSamplingFromProb failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                                     std::optional<at::Tensor> maybe_indices,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, std::optional<at::Generator> gen_) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  auto device = probs.device();
  CHECK_EQ(output.device(), device);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(32 * batch_size);
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data_ptr()) : nullptr,
      batch_size, top_k_val, top_p_val, vocab_size, deterministic, philox_seed, philox_offset,
      stream);
  TORCH_CHECK(status == cudaSuccess, "TopKTopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor target_probs, at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num, bool deterministic,
                                std::optional<at::Generator> gen_) {
  CHECK_INPUT(draft_probs);
  CHECK_INPUT(draft_token_ids);
  CHECK_INPUT(target_probs);
  auto device = draft_probs.device();
  CHECK_EQ(draft_token_ids.device(), device);
  CHECK_EQ(target_probs.device(), device);
  CHECK_DIM(3, draft_probs);      // draft_probs: (batch_size, num_speculate_tokens, vocab_size)
  CHECK_DIM(2, draft_token_ids);  // draft_token_ids: (batch_size, num_speculate_tokens)
  CHECK_DIM(3, target_probs);  // target_probs: (batch_size, num_speculate_tokens + 1, vocab_size)
  unsigned int batch_size = draft_probs.size(0);
  unsigned int num_speculate_tokens = draft_probs.size(1);
  unsigned int vocab_size = draft_probs.size(2);
  CHECK_EQ(batch_size, draft_token_ids.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_speculate_tokens + 1, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, output_accepted_token_num.size(0));
  CHECK_EQ(batch_size, output_emitted_draft_token_num.size(0));
  uint64_t philox_seed, philox_offset;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState rng_engine_inputs =
      gen->philox_cuda_state(batch_size * (num_speculate_tokens + 1));
  philox_seed = rng_engine_inputs.seed_.val;
  philox_offset = rng_engine_inputs.offset_.val;

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = sampling::ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs.data_ptr()), static_cast<int*>(draft_token_ids.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()), static_cast<int*>(output_token_ids.data_ptr()),
      static_cast<int*>(output_accepted_token_num.data_ptr()),
      static_cast<int*>(output_emitted_draft_token_num.data_ptr()), batch_size,
      num_speculate_tokens, vocab_size, deterministic, philox_seed, philox_offset, stream);

  TORCH_CHECK(status == cudaSuccess, "ChainSpeculativeSampling failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}
