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

void softmax(at::Tensor workspace_buffer, at::Tensor logits, at::Tensor output,
             std::optional<at::Tensor> maybe_temperature_arr, double temperature_val,
             bool enable_pdl);

void sampling_from_probs(at::Tensor probs, at::Tensor output,
                         std::optional<at::Tensor> maybe_indices, bool deterministic,
                         std::optional<at::Generator> gen);

void sampling_from_logits(at::Tensor logits, at::Tensor output,
                          std::optional<at::Tensor> maybe_indices, bool deterministic,
                          std::optional<at::Generator> gen);

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, std::optional<at::Generator> gen);

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, std::optional<at::Generator> gen);

void min_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, std::optional<at::Generator> gen);

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                                     std::optional<at::Tensor> maybe_indices,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, std::optional<at::Generator> gen);

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val);

void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val);

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val);

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor target_probs, at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num, bool deterministic,
                                std::optional<at::Generator> gen);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Softmax
  m.def("softmax", softmax);
  // Sample from probabilities
  m.def("sampling_from_probs", sampling_from_probs);
  // Sample from logits
  m.def("sampling_from_logits", sampling_from_logits);
  // Top-k sampling from probabilities
  m.def("top_k_sampling_from_probs", top_k_sampling_from_probs);
  // Min-p sampling from probabilities
  m.def("min_p_sampling_from_probs", min_p_sampling_from_probs);
  // Top-p sampling from probabilities
  m.def("top_p_sampling_from_probs", top_p_sampling_from_probs);
  // Top-k and top-p sampling from probabilities
  m.def("top_k_top_p_sampling_from_probs", top_k_top_p_sampling_from_probs);
  // Renormalize probabilities by top-k mask
  m.def("top_k_renorm_probs", top_k_renorm_probs);
  // Renormalize probabilities by top-p mask
  m.def("top_p_renorm_probs", top_p_renorm_probs);
  // Mask logits by top-k mask
  m.def("top_k_mask_logits", top_k_mask_logits);
  // Speculative sampling from sequence of probabilities
  m.def("chain_speculative_sampling", chain_speculative_sampling);
}
