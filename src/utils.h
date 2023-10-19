/*
 * Copyright (c) 2023 by FlashInfer team.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

namespace utils {

template <typename T>
void vec_normal_(std::vector<T>& vec, float mean = 0.f, float std = 1.f) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = T(d(gen));
  }
}

template <typename T>
void vec_zero_(std::vector<T>& vec) {
  std::fill(vec.begin(), vec.end(), T(0));
}

template <typename T>
void vec_fill_(std::vector<T>& vec, T val) {
  std::fill(vec.begin(), vec.end(), val);
}

template <typename T>
void vec_randint_(std::vector<T>& vec, int low, int high) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution d{low, high};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = T(d(gen));
  }
}

template <typename T>
size_t vec_bytes(const T& vec) {
  return vec.size() * sizeof(typename T::value_type);
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

}  // namespace utils
