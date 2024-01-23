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

#include <flashinfer.cuh>

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, unsigned int rotary_mode,
                                          unsigned int layout, float sm_scale, float rope_scale,
                                          float rope_theta);

torch::Tensor single_prefill_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                           torch::Tensor tmp, bool causal, unsigned int layout,
                                           unsigned int rotary_mode, bool allow_fp16_qk_reduction,
                                           float rope_scale, float rope_theta);

std::vector<torch::Tensor> single_prefill_with_kv_cache_return_lse(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor tmp, bool causal,
    unsigned int layout, unsigned int rotary_mode, bool allow_fp16_qk_reduction, float rope_scale,
    float rope_theta);

std::vector<torch::Tensor> merge_state(torch::Tensor v_a, torch::Tensor s_a, torch::Tensor v_b,
                                       torch::Tensor s_b);

std::vector<torch::Tensor> merge_states(torch::Tensor v, torch::Tensor s);

torch::Tensor batch_decode_with_padded_kv_cache(torch::Tensor q, torch::Tensor k_padded,
                                                torch::Tensor v_padded, unsigned int layout,
                                                unsigned int rotary_mode, float sm_scale,
                                                float rope_scale, float rope_theta);

std::vector<torch::Tensor> batch_decode_with_padded_kv_cache_return_lse(
    torch::Tensor q, torch::Tensor k_padded, torch::Tensor v_padded, unsigned int layout,
    unsigned int rotary_mode, float sm_scale, float rope_scale, float rope_theta);

torch::Tensor batch_prefill_with_paged_kv_cache(
    torch::Tensor q, torch::Tensor q_indptr, torch::Tensor kv_data, torch::Tensor kv_indptr,
    torch::Tensor kv_indices, torch::Tensor kv_last_page_len, bool causal, unsigned int layout,
    unsigned int rotary_mode, bool allow_fp16_qk_reduction, float rope_scale, float rope_theta);

class BatchDecodeWithPagedKVCachePyTorchWrapper {
 public:
  static BatchDecodeWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
    return BatchDecodeWithPagedKVCachePyTorchWrapper(layout);
  }
  void BeginForward(torch::Tensor indptr, torch::Tensor last_page_len, unsigned int batch_size,
                    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
                    unsigned int page_size, unsigned int rotary_mode, torch::Tensor empty_data);
  void EndForward();
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor paged_kv_data,
                                     torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len, unsigned int rotary_mode,
                                     float rope_scale, float rope_theta, bool return_lse);

 private:
  BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchDecodeHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};

class BatchPrefillWithPagedKVCachePyTorchWrapper {
 public:
  static BatchPrefillWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
    return BatchPrefillWithPagedKVCachePyTorchWrapper(layout);
  }
  void BeginForward(torch::Tensor qo_indptr, unsigned int batch_size, unsigned int num_qo_heads,
                    unsigned int num_kv_heads);
  void EndForward();
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor qo_indptr,
                                     torch::Tensor paged_kv_data, torch::Tensor paged_kv_indptr,
                                     torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len, bool causal,
                                     unsigned int rotary_mode, bool allow_fp16_qk_reduction,
                                     float rope_scale, float rope_theta, bool return_lse);

 private:
  BatchPrefillWithPagedKVCachePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchPrefillHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};
