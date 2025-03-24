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
#include "pytorch_extension_utils.h"

void merge_state(at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b,
                 at::Tensor v_merged, at::Tensor s_merged);

void merge_state_in_place(at::Tensor v, at::Tensor s, at::Tensor v_other, at::Tensor s_other,
                          std::optional<at::Tensor> mask);

void merge_states(at::Tensor v, at::Tensor s, at::Tensor v_merged, at::Tensor s_merged);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Merge two self-attention states
  m.def("merge_state", merge_state);
  // Merge another self-attention state in-place.
  m.def("merge_state_in_place", merge_state_in_place);
  // "Merge multiple self-attention states"
  m.def("merge_states", merge_states);
}
