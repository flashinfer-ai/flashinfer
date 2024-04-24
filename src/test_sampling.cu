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
#include <gtest/gtest.h>

#include <cstdint>
#include <flashinfer/sampling.cuh>
#include <random>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <typename T, typename IdType>
void _TestTopKSamplingFromProb(size_t batch_size, uint32_t k, size_t vocab_size) {
  std::vector<T> probs_h(batch_size * vocab_size);
  float p = float(k) * 0.1;
  utils::vec_fill_<T>(probs_h, (1 - p) / float((vocab_size - k)));
  std::vector<int32_t> all_token_ids(vocab_size);
  std::iota(all_token_ids.begin(), all_token_ids.end(), 0);
  std::vector<std::set<int32_t>> high_prob_token_ids_sets;
  for (uint32_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> high_prob_token_ids;
    std::set<int32_t> high_prob_token_ids_set;
    std::sample(all_token_ids.begin(), all_token_ids.end(), std::back_inserter(high_prob_token_ids),
                k, std::mt19937{std::random_device{}()});
    high_prob_token_ids_set.insert(high_prob_token_ids.begin(), high_prob_token_ids.end());
    high_prob_token_ids_sets.emplace_back(high_prob_token_ids_set);
    for (uint32_t j = 0; j < k; ++j) {
      probs_h[i * vocab_size + high_prob_token_ids[j]] = 0.1;
    }
  }

  thrust::device_vector<T> probs_d(probs_h);
  thrust::device_vector<bool> success_d(batch_size);
  thrust::device_vector<IdType> sampled_ids_d(batch_size);
  const int32_t num_samples = 1000;
  const uint32_t max_top_p_rounds = 32;
  std::vector<int32_t> counter(batch_size * vocab_size);
  utils::vec_fill_<int32_t>(counter, 0);
  for (uint32_t draw = 0; draw < num_samples; ++draw) {
    std::vector<T> uniform_samples_h(batch_size * max_top_p_rounds);
    utils::vec_uniform_<T>(uniform_samples_h, 0, 1);
    thrust::device_vector<T> uniform_samples_d(uniform_samples_h);

    auto status = sampling::TopKSamplingFromProb<max_top_p_rounds, T, IdType>(
        thrust::raw_pointer_cast(probs_d.data()),
        thrust::raw_pointer_cast(uniform_samples_d.data()),
        thrust::raw_pointer_cast(sampled_ids_d.data()), thrust::raw_pointer_cast(success_d.data()),
        k, batch_size, vocab_size);

    EXPECT_EQ(status, cudaSuccess) << "TopKSamplingFromProb kernel launch failed, error message: "
                                   << cudaGetErrorString(status);

    thrust::host_vector<bool> success_h(success_d);
    for (uint32_t i = 0; i < batch_size; ++i) {
      EXPECT_TRUE(success_h[i]) << "TopKSamplingFromProb failed for batch " << i;
    }

    thrust::host_vector<IdType> sampled_ids_h(sampled_ids_d);
    for (uint32_t i = 0; i < batch_size; ++i) {
      counter[i * vocab_size + sampled_ids_h[i]]++;
    }
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = 0; j < vocab_size; ++j) {
      if (counter[i * vocab_size + j] > 0) {
        EXPECT_TRUE(high_prob_token_ids_sets[i].find(j) != high_prob_token_ids_sets[i].end())
            << "high_prob_token_ids_sets[" << i << "] does not contain " << j << std::endl;
      }
    }
  }

  std::cout << "batch_size: " << batch_size << ", k: " << k << ", vocab_size: " << vocab_size
            << ", accuracy test passed." << std::endl;
}

template <typename T, typename IdType>
void _TestTopPSamplingFromProb(size_t batch_size, uint32_t k, size_t vocab_size) {
  std::vector<T> probs_h(batch_size * vocab_size);
  float p = float(k) * 0.1;
  utils::vec_fill_<T>(probs_h, (1 - p) / float((vocab_size - k)));
  std::vector<int32_t> all_token_ids(vocab_size);
  std::iota(all_token_ids.begin(), all_token_ids.end(), 0);
  std::vector<std::set<int32_t>> high_prob_token_ids_sets;
  for (uint32_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> high_prob_token_ids;
    std::set<int32_t> high_prob_token_ids_set;
    std::sample(all_token_ids.begin(), all_token_ids.end(), std::back_inserter(high_prob_token_ids),
                k, std::mt19937{std::random_device{}()});
    high_prob_token_ids_set.insert(high_prob_token_ids.begin(), high_prob_token_ids.end());
    high_prob_token_ids_sets.emplace_back(high_prob_token_ids_set);
    for (uint32_t j = 0; j < k; ++j) {
      probs_h[i * vocab_size + high_prob_token_ids[j]] = 0.1;
    }
  }

  thrust::device_vector<T> probs_d(probs_h);
  thrust::device_vector<bool> success_d(batch_size);
  thrust::device_vector<IdType> sampled_ids_d(batch_size);
  const int32_t num_samples = 1000;
  const uint32_t max_top_p_rounds = 32;
  std::vector<int32_t> counter(batch_size * vocab_size);
  utils::vec_fill_<int32_t>(counter, 0);
  for (uint32_t draw = 0; draw < num_samples; ++draw) {
    std::vector<T> uniform_samples_h(batch_size * max_top_p_rounds);
    utils::vec_uniform_<T>(uniform_samples_h, 0, 1);
    thrust::device_vector<T> uniform_samples_d(uniform_samples_h);

    auto status = sampling::TopPSamplingFromProb<max_top_p_rounds, T, IdType>(
        thrust::raw_pointer_cast(probs_d.data()),
        thrust::raw_pointer_cast(uniform_samples_d.data()),
        thrust::raw_pointer_cast(sampled_ids_d.data()), thrust::raw_pointer_cast(success_d.data()),
        p, batch_size, vocab_size);

    EXPECT_EQ(status, cudaSuccess) << "TopPSamplingFromProb kernel launch failed, error message: "
                                   << cudaGetErrorString(status);

    thrust::host_vector<bool> success_h(success_d);
    for (uint32_t i = 0; i < batch_size; ++i) {
      EXPECT_TRUE(success_h[i]) << "TopPSamplingFromProb failed for batch " << i;
    }

    thrust::host_vector<IdType> sampled_ids_h(sampled_ids_d);
    for (uint32_t i = 0; i < batch_size; ++i) {
      counter[i * vocab_size + sampled_ids_h[i]]++;
    }
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = 0; j < vocab_size; ++j) {
      if (counter[i * vocab_size + j] > 0) {
        EXPECT_TRUE(high_prob_token_ids_sets[i].find(j) != high_prob_token_ids_sets[i].end())
            << "high_prob_token_ids_sets[" << i << "] does not contain " << j << std::endl;
      }
    }
  }

  std::cout << "batch_size: " << batch_size << ", p: " << p << ", vocab_size: " << vocab_size
            << ", accuracy test passed." << std::endl;
}

template <typename T, typename IdType>
void _TestSamplingFromProb(size_t batch_size, size_t vocab_size) {
  std::vector<IdType> sampled_ids_ref_h(batch_size);
  std::vector<T> probs_h(batch_size * vocab_size);
  std::vector<T> uniform_samples_h(batch_size);
  utils::vec_randint_<int32_t>(sampled_ids_ref_h, 0, vocab_size - 1);
  utils::vec_uniform_<T>(uniform_samples_h, 0, 1);
  utils::vec_zero_<T>(probs_h);

  for (uint32_t i = 0; i < batch_size; ++i) {
    probs_h[i * vocab_size + sampled_ids_ref_h[i]] = 1;
  }

  thrust::device_vector<T> probs_d(probs_h);
  thrust::device_vector<T> uniform_samples_d(uniform_samples_h);
  thrust::device_vector<IdType> sampled_ids_d(batch_size);

  auto status = sampling::SamplingFromProb<T>(
      thrust::raw_pointer_cast(probs_d.data()), thrust::raw_pointer_cast(uniform_samples_d.data()),
      thrust::raw_pointer_cast(sampled_ids_d.data()), batch_size, vocab_size);
  EXPECT_EQ(status, cudaSuccess) << "SamplingFromProb kernel launch failed, error message: "
                                 << cudaGetErrorString(status);

  thrust::host_vector<IdType> sampled_ids_h(sampled_ids_d);

  for (uint32_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ(sampled_ids_h[i], sampled_ids_ref_h[i])
        << "sampled_ids_h[" << i << "] != sampled_ids_ref_h[" << i << "]";
  }

  std::cout << "batch_size: " << batch_size << ", vocab_size: " << vocab_size
            << ", accuracy test passed." << std::endl;
}

template <typename T, typename IdType>
void TestSamplingFromProb() {
  for (size_t batch_size : {1, 7, 333}) {
    for (size_t d : {24, 97, 1002, 4096, 32000, 128000}) {
      _TestSamplingFromProb<T, IdType>(batch_size, d);
    }
  }
}

template <typename T, typename IdType>
void TestTopKSamplingFromProb() {
  for (size_t batch_size : {1, 7, 333}) {
    for (size_t k : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
      for (size_t d : {24, 97, 1002, 4096, 32000, 128000}) {
        _TestTopKSamplingFromProb<T, IdType>(batch_size, k, d);
      }
    }
  }
}

template <typename T, typename IdType>
void TestTopPSamplingFromProb() {
  for (size_t batch_size : {1, 7, 333}) {
    for (size_t k : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
      for (size_t d : {24, 4096, 1002, 32000, 128000}) {
        _TestTopPSamplingFromProb<T, IdType>(batch_size, k, d);
      }
    }
  }
}

TEST(FlashInferCorrectnessTests, TestTopKSamplingFromProbFP32) {
  TestTopKSamplingFromProb<float, int32_t>();
}

TEST(FlashInferCorrectnessTests, TestTopPSamplingFromProbFP32) {
  TestTopPSamplingFromProb<float, int32_t>();
}

TEST(FlashInferCorrectnessTests, TestSamplingFromProbFP32) {
  TestSamplingFromProb<float, int32_t>();
}
