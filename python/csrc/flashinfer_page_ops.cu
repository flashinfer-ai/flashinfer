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
#include <torch/extension.h>

void append_paged_kv_cache(torch::Tensor append_key, torch::Tensor append_value,
                           torch::Tensor append_indptr, std::optional<torch::Tensor> paged_kv_cache,
                           std::optional<torch::Tensor> paged_k_cache,
                           std::optional<torch::Tensor> paged_v_cache, torch::Tensor kv_indices,
                           torch::Tensor kv_indptr, torch::Tensor kv_last_page_len,
                           unsigned int layout);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append_paged_kv_cache", &append_paged_kv_cache, "Append paged KV-Cache operator");
}
