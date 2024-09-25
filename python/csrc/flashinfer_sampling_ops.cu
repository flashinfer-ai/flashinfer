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

torch::Tensor sampling_from_probs(torch::Tensor probs, torch::Tensor uniform_samples,
                                  bool deterministic);

std::vector<torch::Tensor> top_p_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                                     double top_p_val, bool deterministic);

std::vector<torch::Tensor> top_k_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                                     unsigned int top_k_val, bool deterministic);

std::vector<torch::Tensor> min_p_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_min_p_arr,
                                                     double min_p_val, bool deterministic);

std::vector<torch::Tensor> top_k_top_p_sampling_from_probs(
    torch::Tensor probs, torch::Tensor uniform_samples,
    std::optional<torch::Tensor> maybe_top_k_arr, double top_k_val,
    std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val, bool deterministic);

torch::Tensor top_p_renorm_probs(torch::Tensor probs, std::optional<torch::Tensor> maybe_top_p_arr,
                                 double top_p_val);

torch::Tensor top_k_renorm_probs(torch::Tensor probs, std::optional<torch::Tensor> maybe_top_k_arr,
                                 unsigned int top_k_val);

torch::Tensor top_k_mask_logits(torch::Tensor logits, std::optional<torch::Tensor> maybe_top_k_arr,
                                unsigned int top_k_val);

std::vector<torch::Tensor> chain_speculative_sampling(
    torch::Tensor draft_probs, torch::Tensor draft_token_ids, torch::Tensor uniform_samples,
    torch::Tensor target_probs, std::optional<torch::Tensor> maybe_output_accepted_token_num,
    std::optional<torch::Tensor> maybe_output_emitted_token_num, bool deterministic);

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
