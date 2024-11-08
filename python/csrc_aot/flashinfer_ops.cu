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

//========== activation ==========

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

//========== cascade ==========

std::vector<torch::Tensor> merge_state(torch::Tensor v_a, torch::Tensor s_a, torch::Tensor v_b,
                                       torch::Tensor s_b);

void merge_state_in_place(torch::Tensor v, torch::Tensor s, torch::Tensor v_other,
                          torch::Tensor s_other, std::optional<torch::Tensor> mask = std::nullopt);

std::vector<torch::Tensor> merge_states(torch::Tensor v, torch::Tensor s);

//========== decode ==========

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp,
                                          std::optional<torch::Tensor> alibi_slopes,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta);

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    bool use_logits_soft_cap, unsigned int head_dim, torch::Tensor empty_q_data,
    torch::Tensor empty_kv_data, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor indptr, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph);

torch::Tensor BatchDecodeWithPagedKVCacheRun(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, torch::Tensor q, torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache, torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, std::optional<torch::Tensor> alibi_slopes,
    unsigned int kv_layout_code, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<torch::Tensor> maybe_lse);

//========== gemm ==========

void bmm_fp8(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& D,
             torch::Tensor& A_scale, torch::Tensor& B_scale);

void CutlassSegmentGEMM(torch::Tensor workspace_buffer, torch::Tensor all_problems,
                        torch::Tensor x_ptr, torch::Tensor w_ptr, torch::Tensor y_ptr,
                        torch::Tensor x_ld, torch::Tensor w_ld, torch::Tensor y_ld,
                        torch::Tensor empty_x_data, bool weight_column_major);

//========== norm ==========

void rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double eps);

void fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
                       double eps);

void gemma_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double eps);

void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
                             double eps);

//========== page ==========

void append_paged_kv_cache(torch::Tensor append_key, torch::Tensor append_value,
                           torch::Tensor batch_indices, torch::Tensor positions,
                           torch::Tensor paged_k_cache, torch::Tensor paged_v_cache,
                           torch::Tensor kv_indices, torch::Tensor kv_indptr,
                           torch::Tensor kv_last_page_len, unsigned int layout);

//========== prefill ==========

torch::Tensor single_prefill_with_kv_cache(
    unsigned int mask_mode_code, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    std::optional<torch::Tensor> maybe_packed_custom_mask, torch::Tensor tmp,
    std::optional<torch::Tensor> maybe_alibi_slopes, unsigned int layout, int32_t window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<torch::Tensor> maybe_lse);

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    unsigned int head_dim, torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer, torch::Tensor qo_indptr,
    torch::Tensor kv_indptr, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int page_size, bool enable_cuda_graph);

torch::Tensor BatchPrefillWithRaggedKVCacheRun(
    unsigned int mask_mode_code, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, std::vector<int64_t> plan_info_vec, torch::Tensor q,
    torch::Tensor k, torch::Tensor v, std::optional<torch::Tensor> maybe_custom_mask,
    std::optional<torch::Tensor> maybe_alibi_slopes, torch::Tensor qo_indptr,
    torch::Tensor kv_indptr, std::optional<torch::Tensor> maybe_qk_indptr, unsigned int layout,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    std::optional<torch::Tensor> maybe_lse);

torch::Tensor BatchPrefillWithPagedKVCacheRun(
    unsigned int mask_mode_code, torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer, std::vector<int64_t> plan_info_vec, torch::Tensor q,
    torch::Tensor paged_k_cache, torch::Tensor paged_v_cache,
    std::optional<torch::Tensor> maybe_custom_mask, std::optional<torch::Tensor> maybe_alibi_slopes,
    torch::Tensor qo_indptr, torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, std::optional<torch::Tensor> maybe_qk_indptr,
    unsigned int layout, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, std::optional<torch::Tensor> maybe_lse);

//========== quantization ==========

torch::Tensor packbits(torch::Tensor x, const std::string& bitorder);

torch::Tensor segment_packbits(torch::Tensor x, torch::Tensor input_indptr,
                               torch::Tensor output_indptr, const std::string& bitorder);

//========== rope ==========

void apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope, torch::Tensor k_rope,
                torch::Tensor indptr, torch::Tensor offsets, bool interleave, float rope_scale,
                float rope_theta);

void apply_llama31_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                        torch::Tensor k_rope, torch::Tensor indptr, torch::Tensor offsets,
                        bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
                        float high_freq_factor, float old_context_length);

void apply_rope_pos_ids(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                        torch::Tensor k_rope, torch::Tensor pos_ids, bool interleave,
                        float rope_scale, float rope_theta);

void apply_llama31_rope_pos_ids(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope,
                                torch::Tensor k_rope, torch::Tensor pos_ids, bool interleave,
                                float rope_scale, float rope_theta, float low_freq_factor,
                                float high_freq_factor, float old_context_length);

//========== sampling ==========

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

//========== pybind11 ==========

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // activation
  m.def("silu_and_mul", &silu_and_mul, "Fused SiLU and Mul");
  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Fused GeLU Tanh and Mul");
  m.def("gelu_and_mul", &gelu_and_mul, "Fused GeLU and Mul");

  // cascade
  m.def("merge_state", &merge_state, "Merge two self-attention states");
  m.def("merge_state_in_place", &merge_state_in_place,
        "Merge another self-attention state in-place.");
  m.def("merge_states", &merge_states, "Merge multiple self-attention states");

  // decode
  m.def("single_decode_with_kv_cache", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
  m.def("batch_decode_with_paged_kv_cache_plan", &BatchDecodeWithPagedKVCachePlan);
  m.def("batch_decode_with_paged_kv_cache_run", &BatchDecodeWithPagedKVCacheRun);

  // gemm
  m.def("bmm_fp8", &bmm_fp8, "BMM FP8");
  m.def("cutlass_segment_gemm", &CutlassSegmentGEMM, "Cutlass Segment GEMM operator");

  // norm
  m.def("rmsnorm", &rmsnorm, "Root mean square normalization");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add root mean square normalization");
  m.def("gemma_rmsnorm", &gemma_rmsnorm, "Gemma Root mean square normalization");
  m.def("gemma_fused_add_rmsnorm", &gemma_fused_add_rmsnorm,
        "Gemma Fused add root mean square normalization");

  // page
  m.def("append_paged_kv_cache", &append_paged_kv_cache, "Append paged KV-Cache operator");

  // prefill
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
  m.def("batch_prefill_with_kv_cache_plan", &BatchPrefillWithKVCachePlan);
  m.def("batch_prefill_with_ragged_kv_cache_run", &BatchPrefillWithRaggedKVCacheRun);
  m.def("batch_prefill_with_paged_kv_cache_run", &BatchPrefillWithPagedKVCacheRun);

  // quantization
  m.def("packbits", &packbits, "GPU packbits operator");
  m.def("segment_packbits", &segment_packbits, "GPU segment packbits operator");

  // rope
  m.def("apply_rope", &apply_rope, "Apply RoPE");
  m.def("apply_llama31_rope", &apply_llama31_rope, "Apply Llama 3.1 style RoPE");
  m.def("apply_rope_pos_ids", &apply_rope_pos_ids, "Apply RoPE with positional ids");
  m.def("apply_llama31_rope_pos_ids", &apply_llama31_rope_pos_ids,
        "Apply Llama 3.1 style RoPE with positional ids");

  // sampling
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
