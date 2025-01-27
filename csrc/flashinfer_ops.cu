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
                                 at::Tensor o, unsigned int layout,
                                 int window_left SINGLE_DECODE_ADDITIONAL_FUNC_PARAMS,
                                 int64_t cuda_stream);

std::vector<int64_t> BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph, int window_left, float logits_soft_cap, unsigned int head_dim,
    at::Tensor empty_q_data, at::Tensor empty_kv_data, int64_t cuda_stream);

void BatchDecodeWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices,
    at::Tensor paged_kv_last_page_len, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    unsigned int kv_layout_code, int window_left BATCH_DECODE_ADDITIONAL_FUNC_PARAMS,
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
                           unsigned int layout, int64_t cuda_stream);

void block_sparse_indices_to_vector_sparse_offsets(at::Tensor block_sparse_indices,
                                                   at::Tensor block_sparse_indptr,
                                                   at::Tensor vector_sparse_offsets,
                                                   at::Tensor vector_sparse_indptr,
                                                   at::Tensor kv_len_arr, unsigned int stride_block,
                                                   unsigned int stride_n, unsigned int batch_size,
                                                   unsigned int block_size, int64_t cuda_stream);

//========== prefill ==========

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  unsigned int mask_mode_code, unsigned int layout,
                                  int32_t window_left SINGLE_PREFILL_ADDITIONAL_FUNC_PARAMS,
                                  int64_t cuda_stream);

std::vector<int64_t> BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, unsigned total_num_rows, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int page_size,
    bool enable_cuda_graph, unsigned int head_dim, bool causal, int64_t cuda_stream);

void BatchPrefillWithRaggedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    unsigned int mask_mode_code, unsigned int layout,
    int32_t window_left BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream);

void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q, at::Tensor paged_k_cache,
    at::Tensor paged_v_cache, at::Tensor qo_indptr, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o,
    std::optional<at::Tensor> maybe_lse, unsigned int mask_mode_code, unsigned int layout,
    int32_t window_left BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream);

//========== quantization ==========

void packbits(at::Tensor x, const std::string& bitorder, at::Tensor y, int64_t cuda_stream);

void segment_packbits(at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr,
                      const std::string& bitorder, at::Tensor y, int64_t cuda_stream);

//========== rope ==========

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, unsigned int rotary_dim, bool interleave, float rope_scale,
                float rope_theta, int64_t cuda_stream);

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, unsigned int rotary_dim,
                        bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
                        float high_freq_factor, float old_context_length, int64_t cuda_stream);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                        float rope_scale, float rope_theta, int64_t cuda_stream);

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                                float rope_scale, float rope_theta, float low_freq_factor,
                                float high_freq_factor, float old_context_length,
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
                               unsigned int top_k_val, bool deterministic, int64_t cuda_stream);

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
                        std::optional<at::Tensor> maybe_top_k_arr, unsigned int top_k_val,
                        int64_t cuda_stream);

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, unsigned int top_k_val,
                       int64_t cuda_stream);

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor uniform_samples, at::Tensor target_probs,
                                at::Tensor output_token_ids, at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_token_num, bool deterministic,
                                int64_t cuda_stream);

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
  m.def("block_sparse_indices_to_vector_sparse_offsets",
        &block_sparse_indices_to_vector_sparse_offsets, "Precompute block sparse offsets");

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
  m.def("apply_rope_pos_ids_cos_sin_cache", &apply_rope_pos_ids_cos_sin_cache,
        "Apply RoPE with positional ids and cosine/sine cache");

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
