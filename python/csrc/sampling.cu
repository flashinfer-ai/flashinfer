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

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor sampling_from_probs(torch::Tensor probs, torch::Tensor uniform_samples,
                                  bool deterministic) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(1, uniform_samples);  // uniform_samples: (batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(0));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));

  cudaError_t status = sampling::SamplingFromProb(static_cast<float*>(probs.data_ptr()),
                                                  static_cast<float*>(uniform_samples.data_ptr()),
                                                  static_cast<int*>(samples.data_ptr()), batch_size,
                                                  vocab_size, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
  return samples;
}

std::vector<torch::Tensor> top_p_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                                     double top_p_val, bool deterministic) {
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
  auto top_p_arr = maybe_top_p_arr.value_or(torch::empty({0}, torch::dtype(torch::kFloat32)));
  if (has_top_p_arr) {
    CHECK_INPUT(top_p_arr);
    CHECK_DIM(1, top_p_arr);  // top_p_arr: (batch_size,)
    CHECK_EQ(top_p_arr.size(0), batch_size);
    CHECK_EQ(top_p_arr.device(), device);
  }
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);
  top_p_arr = top_p_arr.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(device));

  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_p_arr ? static_cast<float*>(top_p_arr.data_ptr()) : nullptr, batch_size, top_p_val,
      vocab_size, max_top_p_rounds, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

std::vector<torch::Tensor> top_k_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                                     unsigned int top_k_val, bool deterministic) {
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
  auto top_k_arr = maybe_top_k_arr.value_or(torch::empty({0}, torch::dtype(torch::kInt32)));
  if (has_top_k_arr) {
    CHECK_INPUT(top_k_arr);
    CHECK_DIM(1, top_k_arr);  // top_k_arr: (batch_size,)
    CHECK_EQ(top_k_arr.size(0), batch_size);
    CHECK_EQ(top_k_arr.device(), device);
  }
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);
  top_k_arr = top_k_arr.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(device));

  cudaError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_k_arr ? static_cast<float*>(top_k_arr.data_ptr()) : nullptr, batch_size, top_k_val,
      vocab_size, max_top_k_rounds, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopKSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

std::vector<torch::Tensor> min_p_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_min_p_arr,
                                                     double min_p_val, bool deterministic) {
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
  bool has_min_p_arr = maybe_min_p_arr.has_value();
  auto min_p_arr = maybe_min_p_arr.value_or(torch::empty({0}, torch::dtype(torch::kFloat32)));
  if (has_min_p_arr) {
    CHECK_INPUT(min_p_arr);
    CHECK_DIM(1, min_p_arr);  // min_p_arr: (batch_size,)
    CHECK_EQ(min_p_arr.size(0), batch_size);
    CHECK_EQ(min_p_arr.device(), device);
  }
  min_p_arr = min_p_arr.to(torch::kFloat32);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(device));

  cudaError_t status = sampling::MinPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      has_min_p_arr ? static_cast<float*>(min_p_arr.data_ptr()) : nullptr,
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()), batch_size,
      min_p_val, vocab_size, max_rounds, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "MinPSamplingFromProb failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

std::vector<torch::Tensor> top_k_top_p_sampling_from_probs(
    torch::Tensor probs, torch::Tensor uniform_samples,
    std::optional<torch::Tensor> maybe_top_k_arr, double top_k_val,
    std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val, bool deterministic) {
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
  auto top_k_arr = maybe_top_k_arr.value_or(torch::empty({0}, torch::dtype(torch::kInt32)));
  if (has_top_k_arr) {
    CHECK_INPUT(top_k_arr);
    CHECK_DIM(1, top_k_arr);  // top_k_arr: (batch_size,)
    CHECK_EQ(top_k_arr.size(0), batch_size);
    CHECK_EQ(top_k_arr.device(), device);
  }
  top_k_arr = top_k_arr.to(torch::kInt32);
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  auto top_p_arr = maybe_top_p_arr.value_or(torch::empty({0}, torch::dtype(torch::kFloat32)));
  if (has_top_p_arr) {
    CHECK_INPUT(top_p_arr);
    CHECK_DIM(1, top_p_arr);  // top_p_arr: (batch_size,)
    CHECK_EQ(top_p_arr.size(0), batch_size);
    CHECK_EQ(top_p_arr.device(), device);
  }
  top_p_arr = top_p_arr.to(torch::kFloat32);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(device));

  cudaError_t status = sampling::TopKTopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      has_top_k_arr ? static_cast<int*>(top_k_arr.data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(top_p_arr.data_ptr()) : nullptr,
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()), batch_size,
      top_k_val, top_p_val, vocab_size, max_rounds, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopKTopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

torch::Tensor top_p_renorm_prob(torch::Tensor probs, std::optional<torch::Tensor> maybe_top_p_arr,
                                double top_p_val) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  auto top_p_arr = maybe_top_p_arr.value_or(torch::empty({0}, torch::dtype(torch::kFloat32)));
  if (has_top_p_arr) {
    CHECK_INPUT(top_p_arr);
    CHECK_DIM(1, top_p_arr);  // top_p_arr: (batch_size,)
    CHECK_EQ(top_p_arr.size(0), batch_size);
    CHECK_EQ(top_p_arr.device(), device);
  }
  top_p_arr = top_p_arr.to(torch::kFloat32);
  probs = probs.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto renorm_probs =
      torch::empty({batch_size, vocab_size}, torch::dtype(torch::kFloat32).device(device));

  cudaError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_p_arr ? static_cast<float*>(top_p_arr.data_ptr()) : nullptr, batch_size, top_p_val,
      vocab_size, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess,
              "TopPRenormProb failed with error code " + std::string(cudaGetErrorString(status)));
  return renorm_probs;
}

torch::Tensor top_k_renorm_prob(torch::Tensor probs, std::optional<torch::Tensor> maybe_top_k_arr,
                                unsigned int top_k_val) {
  CHECK_INPUT(probs);
  auto device = probs.device();
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  auto top_k_arr = maybe_top_k_arr.value_or(torch::empty({0}, torch::dtype(torch::kInt32)));
  if (has_top_k_arr) {
    CHECK_INPUT(top_k_arr);
    CHECK_DIM(1, top_k_arr);  // top_k_arr: (batch_size,)
    CHECK_EQ(top_k_arr.size(0), batch_size);
    CHECK_EQ(top_k_arr.device(), device);
  }
  top_k_arr = top_k_arr.to(torch::kInt32);
  probs = probs.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto renorm_probs =
      torch::empty({batch_size, vocab_size}, torch::dtype(torch::kFloat32).device(device));

  cudaError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
      has_top_k_arr ? static_cast<int*>(top_k_arr.data_ptr()) : nullptr, batch_size, top_k_val,
      vocab_size, torch_current_stream);

  TORCH_CHECK(status == cudaSuccess,
              "TopKRenormProb failed with error code " + std::string(cudaGetErrorString(status)));
  return renorm_probs;
}

torch::Tensor top_k_mask_logits(torch::Tensor logits, std::optional<torch::Tensor> maybe_top_k_arr,
                                unsigned int top_k_val) {
  CHECK_INPUT(logits);
  auto device = logits.device();
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  auto top_k_arr = maybe_top_k_arr.value_or(torch::empty({0}, torch::dtype(torch::kInt32)));
  if (has_top_k_arr) {
    CHECK_INPUT(top_k_arr);
    CHECK_DIM(1, top_k_arr);  // top_k_arr: (batch_size,)
    CHECK_EQ(top_k_arr.size(0), batch_size);
    CHECK_EQ(top_k_arr.device(), device);
  }
  top_k_arr = top_k_arr.to(torch::kInt32);
  logits = logits.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto mask_logits =
      torch::empty({batch_size, vocab_size}, torch::dtype(torch::kFloat32).device(device));

  cudaError_t status = sampling::TopKMaskLogits<float>(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(mask_logits.data_ptr()),
      has_top_k_arr ? static_cast<int*>(top_k_arr.data_ptr()) : nullptr, batch_size, top_k_val,
      vocab_size, torch_current_stream);

  TORCH_CHECK(status == cudaSuccess,
              "TopKMaskLogits failed with error code " + std::string(cudaGetErrorString(status)));
  return mask_logits;
}

std::vector<torch::Tensor> chain_speculative_sampling(
    torch::Tensor draft_probs, torch::Tensor draft_token_ids, torch::Tensor uniform_samples,
    torch::Tensor target_probs, std::optional<torch::Tensor> maybe_output_accepted_token_num,
    std::optional<torch::Tensor> maybe_output_emitted_token_num, bool deterministic) {
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

  draft_probs = draft_probs.to(torch::kFloat32);
  draft_token_ids = draft_token_ids.to(torch::kInt32);
  uniform_samples = uniform_samples.to(torch::kFloat32);
  target_probs = target_probs.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto output_token_ids = torch::empty({batch_size, num_speculate_tokens + 1},
                                       torch::dtype(torch::kInt32).device(device));

  bool has_output_accepted_token_num = maybe_output_accepted_token_num.has_value();
  bool has_output_emitted_token_num = maybe_output_emitted_token_num.has_value();
  auto output_accepted_token_num = maybe_output_accepted_token_num.value_or(
      torch::zeros({batch_size}, torch::dtype(torch::kInt32).device(device)));
  auto output_emitted_token_num = maybe_output_emitted_token_num.value_or(
      torch::zeros({batch_size}, torch::dtype(torch::kInt32).device(device)));
  if (has_output_accepted_token_num) {
    CHECK_EQ(has_output_emitted_token_num, true);
    CHECK_EQ(batch_size, output_accepted_token_num.size(0));
    CHECK_EQ(batch_size, output_emitted_token_num.size(0));
  }

  cudaError_t status = sampling::ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs.data_ptr()), static_cast<int*>(draft_token_ids.data_ptr()),
      static_cast<float*>(uniform_samples.data_ptr()), static_cast<float*>(target_probs.data_ptr()),
      static_cast<int*>(output_token_ids.data_ptr()),
      static_cast<int*>(output_accepted_token_num.data_ptr()),
      static_cast<int*>(output_emitted_token_num.data_ptr()), batch_size, num_speculate_tokens,
      vocab_size, deterministic, torch_current_stream);

  TORCH_CHECK(status == cudaSuccess, "ChainSpeculativeSampling failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {output_token_ids, output_accepted_token_num, output_emitted_token_num};
}
