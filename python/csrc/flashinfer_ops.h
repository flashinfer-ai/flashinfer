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
#include <torch/extension.h>

#include <flashinfer/group_gemm/handler.cuh>
#include <flashinfer/layout.cuh>
#include <memory>

void append_paged_kv_cache(torch::Tensor append_key, torch::Tensor append_value,
                           torch::Tensor append_indptr, std::optional<torch::Tensor> paged_kv_cache,
                           std::optional<torch::Tensor> paged_k_cache,
                           std::optional<torch::Tensor> paged_v_cache, torch::Tensor kv_indices,
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

torch::Tensor top_p_renorm_prob(torch::Tensor probs, std::optional<torch::Tensor> maybe_top_p_arr,
                                double top_p_val);

torch::Tensor top_k_renorm_prob(torch::Tensor probs, std::optional<torch::Tensor> maybe_top_k_arr,
                                unsigned int top_k_val);

torch::Tensor top_k_mask_logits(torch::Tensor logits, std::optional<torch::Tensor> maybe_top_k_arr,
                                unsigned int top_k_val);

std::vector<torch::Tensor> chain_speculative_sampling(
    torch::Tensor draft_probs, torch::Tensor draft_token_ids, torch::Tensor uniform_samples,
    torch::Tensor target_probs, std::optional<torch::Tensor> maybe_output_accepted_token_num,
    std::optional<torch::Tensor> maybe_output_emitted_token_num, bool deterministic);

torch::Tensor rmsnorm(torch::Tensor input, torch::Tensor weight, double eps);

void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
                       double eps);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void apply_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                        torch::Tensor offsets, bool interleave, float rope_scale, float rope_theta);

void apply_llama31_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                                torch::Tensor offsets, bool interleave, float rope_scale,
                                float rope_theta, float low_freq_factor, float high_freq_factor,
                                float old_context_length);

std::vector<torch::Tensor> apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                                      torch::Tensor offsets, bool interleave, float rope_scale,
                                      float rope_theta);

std::vector<torch::Tensor> apply_llama31_rope(torch::Tensor q, torch::Tensor k,
                                              torch::Tensor indptr, torch::Tensor offsets,
                                              bool interleave, float rope_scale, float rope_theta,
                                              float low_freq_factor, float high_freq_factor,
                                              float old_context_length);

torch::Tensor packbits(torch::Tensor x, const std::string& bitorder);

torch::Tensor segment_packbits(torch::Tensor x, torch::Tensor input_indptr,
                               torch::Tensor output_indptr, const std::string& bitorder);

class CutlassSegmentGEMMPyTorchWrapper {
 public:
  void RegisterWorkspaceBuffer(torch::Tensor workspace_buffer);

  torch::Tensor Forward(torch::Tensor seg_indptr, torch::Tensor weight_indices, torch::Tensor x,
                        torch::Tensor weight, unsigned int batch_size, bool weight_column_major);

  CutlassSegmentGEMMPyTorchWrapper(torch::Tensor workspace_buffer)
      : handler_(std::make_shared<flashinfer::group_gemm::CutlassSegmentGEMMHandler>()) {
    RegisterWorkspaceBuffer(workspace_buffer);
  }

 private:
  std::shared_ptr<flashinfer::group_gemm::CutlassSegmentGEMMHandler> handler_;
};
