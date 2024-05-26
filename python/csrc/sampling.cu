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

torch::Tensor sampling_from_probs(torch::Tensor probs, torch::Tensor uniform_samples) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(1, uniform_samples);  // uniform_samples: (batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(0));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(probs.device()));

  cudaError_t status = sampling::SamplingFromProb(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), batch_size, vocab_size, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
  return samples;
}

std::vector<torch::Tensor> top_p_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples, double top_p) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_p_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_p_rounds = uniform_samples.size(0);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(probs.device()));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(probs.device()));

  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()), top_p,
      batch_size, vocab_size, max_top_p_rounds, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

std::vector<torch::Tensor> top_k_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     unsigned int top_k) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_k_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_k_rounds = uniform_samples.size(0);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(probs.device()));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(probs.device()));

  cudaError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()), top_k,
      batch_size, vocab_size, max_top_k_rounds, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopKSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

torch::Tensor top_p_renorm_prob(torch::Tensor probs, double top_p, double eps) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  probs = probs.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto renorm_probs =
      torch::empty({batch_size, vocab_size}, torch::dtype(torch::kFloat32).device(probs.device()));

  cudaError_t status = sampling::TopPRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()), top_p,
      eps, batch_size, vocab_size, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess,
              "TopPRenormProb failed with error code " + std::string(cudaGetErrorString(status)));
  return renorm_probs;
}

torch::Tensor top_k_renorm_prob(torch::Tensor probs, unsigned int top_k, double eps) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  probs = probs.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto renorm_probs =
      torch::empty({batch_size, vocab_size}, torch::dtype(torch::kFloat32).device(probs.device()));

  cudaError_t status = sampling::TopKRenormProb<float>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()), top_k,
      eps, batch_size, vocab_size, torch_current_stream);

  TORCH_CHECK(status == cudaSuccess,
              "TopKRenormProb failed with error code " + std::string(cudaGetErrorString(status)));
  return renorm_probs;
}
