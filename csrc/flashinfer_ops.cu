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
#include "aot_default_additional_params.h"
#include "pytorch_extension_utils.h"

//========== activation ==========

void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);
void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);
void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream);

//========== cascade ==========

void merge_state(at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b,
                 at::Tensor v_merged, at::Tensor s_merged, int64_t cuda_stream);

void merge_state_in_place(at::Tensor v, at::Tensor s, at::Tensor v_other, at::Tensor s_other,
                          std::optional<at::Tensor> mask, int64_t cuda_stream);

void merge_states(at::Tensor v, at::Tensor s, at::Tensor v_merged, at::Tensor s_merged,
                  int64_t cuda_stream);

//========== decode ==========

void single_decode_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                 at::Tensor o, int64_t layout,
                                 int64_t window_left SINGLE_DECODE_ADDITIONAL_FUNC_PARAMS,
                                 int64_t cuda_stream);

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
    bool enable_cuda_graph, int64_t window_left, double logits_soft_cap, int64_t head_dim_qk,
    int64_t head_dim_vo, at::Tensor empty_q_data, at::Tensor empty_kv_data, int64_t cuda_stream);

void BatchDecodeWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    int64_t kv_layout_code, int64_t window_left BATCH_DECODE_ADDITIONAL_FUNC_PARAMS,
    int64_t cuda_stream);

//========== gemm ==========

void bmm_fp8(at::Tensor A, at::Tensor B, at::Tensor D, at::Tensor A_scale, at::Tensor B_scale,
             at::Tensor workspace_buffer, int64_t cublas_handle, int64_t cuda_stream);

void CutlassSegmentGEMM(at::Tensor workspace_buffer, at::Tensor all_problems, at::Tensor x_ptr,
                        at::Tensor w_ptr, at::Tensor y_ptr, at::Tensor x_ld, at::Tensor w_ld,
                        at::Tensor y_ld, at::Tensor empty_x_data, bool weight_column_major,
                        int64_t cuda_stream);

//========== norm ==========

void rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double eps,
             int64_t cuda_stream);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
                       int64_t cuda_stream);

void gemma_rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double eps,
                   int64_t cuda_stream);

void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight,
                             double eps, int64_t cuda_stream);

//========== page ==========

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t layout, int64_t cuda_stream);

void block_sparse_indices_to_vector_sparse_offsets(at::Tensor block_sparse_indices,
                                                   at::Tensor block_sparse_indptr,
                                                   at::Tensor vector_sparse_offsets,
                                                   at::Tensor vector_sparse_indptr,
                                                   at::Tensor kv_len_arr, int64_t stride_block,
                                                   int64_t stride_n, int64_t batch_size,
                                                   int64_t block_size, int64_t cuda_stream);

//========== prefill ==========

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  int64_t mask_mode_code, int64_t layout,
                                  int64_t window_left SINGLE_PREFILL_ADDITIONAL_FUNC_PARAMS,
                                  int64_t cuda_stream);

at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
    bool enable_cuda_graph, int64_t head_dim_qk, int64_t head_dim_vo, bool causal,
    int64_t cuda_stream);

void BatchPrefillWithRaggedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    int64_t mask_mode_code, int64_t layout,
    int64_t window_left BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream);

void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream);

//========== quantization ==========

void packbits(at::Tensor x, const std::string& bitorder, at::Tensor y, int64_t cuda_stream);

void segment_packbits(at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr,
                      const std::string& bitorder, at::Tensor y, int64_t cuda_stream);

//========== rope ==========

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta, int64_t cuda_stream);

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, int64_t rotary_dim,
                        bool interleave, double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length, int64_t cuda_stream);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, int64_t cuda_stream);

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length,
                                int64_t cuda_stream);

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids, bool interleave, int64_t cuda_stream);

//========== sampling ==========

void sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                         bool deterministic, int64_t cuda_stream);

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_p_arr,
                               double top_p_val, bool deterministic, int64_t cuda_stream);

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_k_arr,
                               int64_t top_k_val, bool deterministic, int64_t cuda_stream);

void min_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, int64_t cuda_stream);

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples,
                                     at::Tensor samples, at::Tensor success,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, int64_t cuda_stream);

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                        int64_t cuda_stream);

void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                        int64_t cuda_stream);

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                       int64_t cuda_stream);

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor uniform_samples, at::Tensor target_probs,
                                at::Tensor output_token_ids, at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_token_num, bool deterministic,
                                int64_t cuda_stream);

//========== Torch Library ==========

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // activation
  // Fused SiLU and Mul
  m.def("silu_and_mul", silu_and_mul);
  // Fused GeLU Tanh and Mul
  m.def("gelu_tanh_and_mul", gelu_tanh_and_mul);
  // Fused GeLU and Mul
  m.def("gelu_and_mul", gelu_and_mul);

  // cascade
  // Merge two self-attention states
  m.def("merge_state", merge_state);
  // Merge another self-attention state in-place.
  m.def("merge_state_in_place", merge_state_in_place);
  // "Merge multiple self-attention states"
  m.def("merge_states", merge_states);

  // decode
  // "Single-request decode with KV-Cache operator"
  m.def("single_decode_with_kv_cache", single_decode_with_kv_cache);
  m.def("batch_decode_with_paged_kv_cache_plan", BatchDecodeWithPagedKVCachePlan);
  m.def("batch_decode_with_paged_kv_cache_run", BatchDecodeWithPagedKVCacheRun);

  // gemm
  // BMM FP8
  m.def("bmm_fp8", bmm_fp8);
  // Cutlass Segment GEMM operator
  m.def("cutlass_segment_gemm", CutlassSegmentGEMM);

  // norm
  // Root mean square normalization
  m.def("rmsnorm", rmsnorm);
  // Fused add root mean square normalization
  m.def("fused_add_rmsnorm", fused_add_rmsnorm);
  // Gemma Root mean square normalization
  m.def("gemma_rmsnorm", gemma_rmsnorm);
  // Gemma Fused add root mean square normalization
  m.def("gemma_fused_add_rmsnorm", gemma_fused_add_rmsnorm);

  // page
  // Append paged KV-Cache operator
  m.def("append_paged_kv_cache", append_paged_kv_cache);
  // Precompute block sparse offsets
  m.def("block_sparse_indices_to_vector_sparse_offsets",
        block_sparse_indices_to_vector_sparse_offsets);

  // prefill
  // Single-request prefill attention with KV-Cache operator
  m.def("single_prefill_with_kv_cache", single_prefill_with_kv_cache);
  m.def("batch_prefill_with_kv_cache_plan", BatchPrefillWithKVCachePlan);
  m.def("batch_prefill_with_ragged_kv_cache_run", BatchPrefillWithRaggedKVCacheRun);
  m.def("batch_prefill_with_paged_kv_cache_run", BatchPrefillWithPagedKVCacheRun);

  // quantization
  // GPU packbits operator
  m.def("packbits", packbits);
  // GPU segment packbits operator
  m.def("segment_packbits", segment_packbits);

  // rope
  // "Apply RoPE"
  m.def("apply_rope", apply_rope);
  // "Apply Llama 3.1 style RoPE"
  m.def("apply_llama31_rope", apply_llama31_rope);
  // "Apply RoPE with positional ids"
  m.def("apply_rope_pos_ids", apply_rope_pos_ids);
  // "Apply Llama 3.1 style RoPE with positional ids"
  m.def("apply_llama31_rope_pos_ids", apply_llama31_rope_pos_ids);
  // "Apply RoPE with positional ids and cosine/sine cache"
  m.def("apply_rope_pos_ids_cos_sin_cache", apply_rope_pos_ids_cos_sin_cache);

  // sampling
  // Sample from probabilities
  m.def("sampling_from_probs", sampling_from_probs);
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
