/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <ATen/ATen.h>
#include <c10/util/Exception.h>

namespace {

inline at::Tensor vec_to_tensor(const std::vector<int64_t>& vec) {
  return at::tensor(vec);
}

inline std::vector<int64_t> tensor_to_vec(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, "tensor must be int64");

  const size_t size = tensor.numel();
  const int64_t* first = tensor.cpu().contiguous().const_data_ptr<int64_t>();
  const int64_t* last = first + size;
  return std::vector(first, last);
}

}