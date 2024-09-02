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

#include "flashinfer_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sampling_from_probs", &sampling_from_probs, "Sample from probabilities");
  m.def("top_k_sampling_from_probs", &top_k_sampling_from_probs,
        "Top-k sampling from probabilities");
  m.def("min_p_sampling_from_probs", &min_p_sampling_from_probs,
        "Min-p sampling from probabilities");
  m.def("top_p_sampling_from_probs", &top_p_sampling_from_probs,
        "Top-p sampling from probabilities");
  m.def("top_k_top_p_sampling_from_probs", &top_k_top_p_sampling_from_probs,
        "Top-k and top-p sampling from probabilities");
  m.def("top_k_renorm_probs", &top_k_renorm_probs, "Renormalize probabilities by top-k mask");
  m.def("top_p_renorm_probs", &top_p_renorm_probs, "Renormalize probabilities by top-p mask");
  m.def("top_k_mask_logits", &top_k_mask_logits, "Mask logits by top-k mask");
  m.def("chain_speculative_sampling", &chain_speculative_sampling,
        "Speculative sampling from sequence of probabilities");
}
