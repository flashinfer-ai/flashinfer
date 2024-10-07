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

std::vector<torch::Tensor> merge_state(torch::Tensor v_a, torch::Tensor s_a, torch::Tensor v_b,
                                       torch::Tensor s_b);

void merge_state_in_place(torch::Tensor v, torch::Tensor s, torch::Tensor v_other,
                          torch::Tensor s_other, std::optional<torch::Tensor> mask = std::nullopt);

std::vector<torch::Tensor> merge_states(torch::Tensor v, torch::Tensor s);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_state", &merge_state, "Merge two self-attention states");
  m.def("merge_state_in_place", &merge_state_in_place,
        "Merge another self-attention state in-place.");
  m.def("merge_states", &merge_states, "Merge multiple self-attention states");
}
