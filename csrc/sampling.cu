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

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                         bool deterministic, int64_t cuda_stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(1, uniform_samples);  // uniform_samples: (batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(0));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::SamplingFromProb(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), batch_size, vocab_size, deterministic, stream);
  TORCH_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_p_arr,
                               double top_p_val, bool deterministic, int64_t cuda_stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_p_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_p_rounds = uniform_samples.size(0);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, max_top_p_rounds, deterministic, stream);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_k_arr,
                               unsigned int top_k_val, bool deterministic, int64_t cuda_stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_k_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_k_rounds = uniform_samples.size(0);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_k_arr ? static_cast<float*>(maybe_top_k_arr->data_ptr()) : nullptr, batch_size,
      top_k_val, vocab_size, max_top_k_rounds, deterministic, stream);
  TORCH_CHECK(status == cudaSuccess, "TopKSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void min_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, int64_t cuda_stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(1, uniform_samples);  // uniform_samples: (batch_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  CHECK_EQ(uniform_samples.size(0), batch_size);
  bool has_min_p_arr = maybe_min_p_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::MinPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      has_min_p_arr ? static_cast<float*>(maybe_min_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(samples.data_ptr()), batch_size, min_p_val, vocab_size, deterministic,
      stream);
  TORCH_CHECK(status == cudaSuccess, "MinPSamplingFromProb failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples,
                                     at::Tensor samples, at::Tensor success,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, int64_t cuda_stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_rounds, batch_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_rounds = uniform_samples.size(0);
  CHECK_EQ(uniform_samples.size(1), batch_size);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      has_top_k_arr ? static_cast<int*>(maybe_top_k_arr->data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr,
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()), batch_size,
      top_k_val, top_p_val, vocab_size, max_rounds, deterministic, stream);
  TORCH_CHECK(status == cudaSuccess, "TopKTopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor uniform_samples, at::Tensor target_probs,
                                at::Tensor output_token_ids, at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_token_num, bool deterministic,
                                int64_t cuda_stream) {
  CHECK_INPUT(draft_probs);
  CHECK_INPUT(draft_token_ids);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(target_probs);
  auto device = draft_probs.device();
  CHECK_EQ(draft_token_ids.device(), device);
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_EQ(target_probs.device(), device);
  CHECK_DIM(3, draft_probs);      // draft_probs: (batch_size, num_speculate_tokens, vocab_size)
  CHECK_DIM(2, draft_token_ids);  // draft_token_ids: (batch_size, num_speculate_tokens)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (batch_size, num_speculate_tokens + 1)
  CHECK_DIM(3, target_probs);  // target_probs: (batch_size, num_speculate_tokens + 1, vocab_size)
  unsigned int batch_size = draft_probs.size(0);
  unsigned int num_speculate_tokens = draft_probs.size(1);
  unsigned int vocab_size = draft_probs.size(2);
  CHECK_EQ(batch_size, draft_token_ids.size(0));
  CHECK_EQ(batch_size, uniform_samples.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_speculate_tokens + 1, uniform_samples.size(1));
  CHECK_EQ(num_speculate_tokens + 1, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, output_accepted_token_num.size(0));
  CHECK_EQ(batch_size, output_emitted_token_num.size(0));

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs.data_ptr()), static_cast<int*>(draft_token_ids.data_ptr()),
      static_cast<float*>(uniform_samples.data_ptr()), static_cast<float*>(target_probs.data_ptr()),
      static_cast<int*>(output_token_ids.data_ptr()),
      static_cast<int*>(output_accepted_token_num.data_ptr()),
      static_cast<int*>(output_emitted_token_num.data_ptr()), batch_size, num_speculate_tokens,
      vocab_size, deterministic, stream);

  TORCH_CHECK(status == cudaSuccess, "ChainSpeculativeSampling failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}
