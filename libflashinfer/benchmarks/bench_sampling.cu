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
#include <thrust/device_vector.h>

#include <flashinfer/sampling.cuh>
#include <nvbench/nvbench.cuh>

#include "utils.h"

using namespace flashinfer;

template <typename T>
void bench_sampling_with_probability(nvbench::state& state) {
  size_t batch_size = state.get_int64("batch_size");
  size_t vocab_size = state.get_int64("vocab_size");
  bool deterministic = state.get_int64("determinisic");

  std::vector<T> probs_h(batch_size * vocab_size);
  std::vector<T> uniform_samples_h(batch_size);
  utils::vec_uniform_<T>(uniform_samples_h, 0, 1);
  utils::vec_uniform_<T>(probs_h, 0, 1);

  // normalize the probs_h
  for (uint32_t i = 0; i < batch_size; ++i) {
    T sum = 0;
    for (uint32_t j = 0; j < vocab_size; ++j) {
      sum += probs_h[i * vocab_size + j];
    }
    for (uint32_t j = 0; j < vocab_size; ++j) {
      probs_h[i * vocab_size + j] /= sum;
    }
  }

  thrust::device_vector<T> probs_d(probs_h);
  thrust::device_vector<T> uniform_samples_d(uniform_samples_h);
  thrust::device_vector<int32_t> output_d(batch_size);

  state.add_global_memory_reads<T>(batch_size * vocab_size, "Read");
  state.add_global_memory_writes<int32_t>(batch_size, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = sampling::SamplingFromProb<T>(
        thrust::raw_pointer_cast(probs_d.data()),
        thrust::raw_pointer_cast(uniform_samples_d.data()),
        thrust::raw_pointer_cast(output_d.data()), batch_size, vocab_size, deterministic);
    timer.stop();
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
  });
}

template <typename T>
void bench_top_p_sampling_with_probability(nvbench::state& state) {
  size_t batch_size = state.get_int64("batch_size");
  size_t vocab_size = state.get_int64("vocab_size");
  bool deterministic = state.get_int64("determinisic");
  double p = state.get_float64("p");
  constexpr uint32_t max_top_p_rounds = 32;

  std::vector<T> probs_h(batch_size * vocab_size);
  std::vector<T> uniform_samples_h(max_top_p_rounds * batch_size);
  utils::vec_uniform_<T>(uniform_samples_h, 0, 1);
  utils::vec_uniform_<T>(probs_h, 0, 1);

  // normalize the probs_h
  for (uint32_t i = 0; i < batch_size; ++i) {
    T sum = 0;
    for (uint32_t j = 0; j < vocab_size; ++j) {
      sum += probs_h[i * vocab_size + j];
    }
    for (uint32_t j = 0; j < vocab_size; ++j) {
      probs_h[i * vocab_size + j] /= sum;
    }
  }

  thrust::device_vector<T> probs_d(probs_h);
  thrust::device_vector<T> uniform_samples_d(uniform_samples_h);
  thrust::device_vector<int32_t> output_d(batch_size);
  thrust::device_vector<bool> success_d(batch_size);

  state.add_global_memory_reads<T>(batch_size * vocab_size, "Read");
  state.add_global_memory_writes<int32_t>(batch_size, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = sampling::TopPSamplingFromProb<T, int32_t>(
        thrust::raw_pointer_cast(probs_d.data()),
        thrust::raw_pointer_cast(uniform_samples_d.data()),
        thrust::raw_pointer_cast(output_d.data()), thrust::raw_pointer_cast(success_d.data()),
        /*top_p_arr=*/nullptr, batch_size, p, vocab_size, max_top_p_rounds, deterministic);
    timer.stop();
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
  });
}

template <typename T>
void bench_top_k_sampling_with_probability(nvbench::state& state) {
  size_t batch_size = state.get_int64("batch_size");
  size_t vocab_size = state.get_int64("vocab_size");
  size_t k = state.get_int64("k");
  bool deterministic = state.get_int64("determinisic");
  constexpr uint32_t max_top_k_rounds = 32;

  std::vector<T> probs_h(batch_size * vocab_size);
  std::vector<T> uniform_samples_h(max_top_k_rounds * batch_size);
  utils::vec_uniform_<T>(uniform_samples_h, 0, 1);
  utils::vec_uniform_<T>(probs_h, 0, 1);

  // normalize the probs_h
  for (uint32_t i = 0; i < batch_size; ++i) {
    T sum = 0;
    for (uint32_t j = 0; j < vocab_size; ++j) {
      sum += probs_h[i * vocab_size + j];
    }
    for (uint32_t j = 0; j < vocab_size; ++j) {
      probs_h[i * vocab_size + j] /= sum;
    }
  }

  thrust::device_vector<T> probs_d(probs_h);
  thrust::device_vector<T> uniform_samples_d(uniform_samples_h);
  thrust::device_vector<int32_t> output_d(batch_size);
  thrust::device_vector<bool> success_d(batch_size);

  state.add_global_memory_reads<T>(batch_size * vocab_size, "Read");
  state.add_global_memory_writes<int32_t>(batch_size, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = sampling::TopKSamplingFromProb<T, int32_t>(
        thrust::raw_pointer_cast(probs_d.data()),
        thrust::raw_pointer_cast(uniform_samples_d.data()),
        thrust::raw_pointer_cast(output_d.data()), thrust::raw_pointer_cast(success_d.data()),
        /*top_k_arr=*/nullptr, batch_size, k, vocab_size, max_top_k_rounds, deterministic);
    timer.stop();
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
  });
}

auto bench_sampling_with_probability_f32 = bench_sampling_with_probability<float>;
NVBENCH_BENCH(bench_sampling_with_probability_f32)
    .set_name("bench_sampling_with_probability_f32")
    .add_int64_axis("batch_size", {16, 32, 128, 512, 2048})
    .add_int64_axis("vocab_size", {32000, 32001, 32002, 128000, 256000})
    .add_int64_axis("determinisic", {0, 1});

auto bench_top_p_sampling_with_probability_f32 = bench_top_p_sampling_with_probability<float>;
NVBENCH_BENCH(bench_top_p_sampling_with_probability_f32)
    .set_name("bench_top_p_sampling_with_probability_f32")
    .add_int64_axis("batch_size", {16, 32, 128, 512, 2048})
    .add_int64_axis("vocab_size", {32000, 32001, 32002, 128000, 256000})
    .add_float64_axis("p", {0.1, 0.5, 0.9, 1.0})
    .add_int64_axis("determinisic", {0, 1});

auto bench_top_k_sampling_with_probability_f32 = bench_top_k_sampling_with_probability<float>;
NVBENCH_BENCH(bench_top_k_sampling_with_probability_f32)
    .set_name("bench_top_k_sampling_with_probability_f32")
    .add_int64_axis("batch_size", {16, 32, 128, 512, 2048})
    .add_int64_axis("vocab_size", {32000, 32001, 32002, 128000, 256000})
    .add_int64_axis("k", {16, 32, 128, 1024})
    .add_int64_axis("determinisic", {0, 1});
