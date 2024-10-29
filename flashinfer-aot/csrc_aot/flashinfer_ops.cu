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
                           torch::Tensor append_indptr, torch::Tensor paged_k_cache,
                           torch::Tensor paged_v_cache, torch::Tensor kv_indices,
                           torch::Tensor kv_indptr, torch::Tensor kv_last_page_len,
                           unsigned int layout);

std::vector<torch::Tensor> merge_state(torch::Tensor v_a, torch::Tensor s_a, torch::Tensor v_b,
                                       torch::Tensor s_b);

void merge_state_in_place(torch::Tensor v, torch::Tensor s, torch::Tensor v_other,
                          torch::Tensor s_other, std::optional<torch::Tensor> mask = std::nullopt);

std::vector<torch::Tensor> merge_states(torch::Tensor v, torch::Tensor s);

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

torch::Tensor chain_speculative_sampling(torch::Tensor draft_probs, torch::Tensor draft_token_ids,
                                         torch::Tensor uniform_samples, torch::Tensor target_probs,
                                         torch::Tensor output_accepted_token_num,
                                         torch::Tensor output_emitted_token_num,
                                         bool deterministic);

void rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double eps);

void fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
                       double eps);

void gemma_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double eps);

void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
                             double eps);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

std::vector<torch::Tensor> apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                                      torch::Tensor k_rope, torch::Tensor indptr,
                                      torch::Tensor offsets, bool interleave, float rope_scale,
                                      float rope_theta);

std::vector<torch::Tensor> apply_llama31_rope(torch::Tensor q, torch::Tensor k,
                                              torch::Tensor q_rope, torch::Tensor k_rope,
                                              torch::Tensor indptr, torch::Tensor offsets,
                                              bool interleave, float rope_scale, float rope_theta,
                                              float low_freq_factor, float high_freq_factor,
                                              float old_context_length);

std::vector<torch::Tensor> apply_rope_pos_ids(torch::Tensor q, torch::Tensor k,
                                              torch::Tensor q_rope, torch::Tensor k_rope,
                                              torch::Tensor pos_ids, bool interleave,
                                              float rope_scale, float rope_theta);

std::vector<torch::Tensor> apply_llama31_rope_pos_ids(torch::Tensor q, torch::Tensor k,
                                                      torch::Tensor q_rope, torch::Tensor k_rope,
                                                      torch::Tensor pos_ids, bool interleave,
                                                      float rope_scale, float rope_theta,
                                                      float low_freq_factor, float high_freq_factor,
                                                      float old_context_length);

torch::Tensor packbits(torch::Tensor x, const std::string& bitorder);

torch::Tensor segment_packbits(torch::Tensor x, torch::Tensor input_indptr,
                               torch::Tensor output_indptr, const std::string& bitorder);

void bmm_fp8(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& D,
             torch::Tensor& A_scale, torch::Tensor& B_scale);

torch::Tensor CutlassSegmentGEMM(torch::Tensor workspace_buffer, torch::Tensor seg_indptr,
                                 torch::Tensor weight_indices, torch::Tensor x,
                                 torch::Tensor weight, unsigned int batch_size,
                                 bool weight_column_major);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append_paged_kv_cache", &append_paged_kv_cache, "Append paged KV-Cache operator");
  m.def("merge_state", &merge_state, "Merge two self-attention states");
  m.def("merge_state_in_place", &merge_state_in_place,
        "Merge another self-attention state in-place.");
  m.def("merge_states", &merge_states, "Merge multiple self-attention states");
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
  m.def("rmsnorm", &rmsnorm, "Root mean square normalization");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add root mean square normalization");
  m.def("gemma_rmsnorm", &gemma_rmsnorm, "Gemma Root mean square normalization");
  m.def("gemma_fused_add_rmsnorm", &gemma_fused_add_rmsnorm,
        "Gemma Fused add root mean square normalization");
  m.def("silu_and_mul", &silu_and_mul, "Fused SiLU and Mul");
  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Fused GeLU Tanh and Mul");
  m.def("gelu_and_mul", &gelu_and_mul, "Fused GeLU and Mul");
  m.def("apply_rope", &apply_rope, "Apply RoPE");
  m.def("apply_llama31_rope", &apply_llama31_rope, "Apply Llama 3.1 style RoPE");
  m.def("apply_rope_pos_ids", &apply_rope_pos_ids, "Apply RoPE with positional ids");
  m.def("apply_llama31_rope_pos_ids", &apply_llama31_rope_pos_ids,
        "Apply Llama 3.1 style RoPE with positional ids");
  m.def("packbits", &packbits, "GPU packbits operator");
  m.def("segment_packbits", &segment_packbits, "GPU segment packbits operator");
  m.def("cutlass_segment_gemm", &CutlassSegmentGEMM, "Cutlass Segment GEMM operator");
  m.def("bmm_fp8", &bmm_fp8, "BMM FP8");
}
