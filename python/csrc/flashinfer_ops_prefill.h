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

#include <flashinfer/attention/handler.cuh>
#include <flashinfer/layout.cuh>
#include <memory>

std::vector<torch::Tensor> single_prefill_with_kv_cache(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor tmp, bool causal,
    unsigned int layout, unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
    int window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    bool return_lse);

std::vector<torch::Tensor> single_prefill_with_kv_cache_custom_mask(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor packed_custom_mask,
    torch::Tensor tmp, unsigned int layout, unsigned int pos_encoding_mode,
    bool allow_fp16_qk_reduction, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, bool return_lse);

class BatchPrefillWithPagedKVCachePyTorchWrapper {
 public:
  void BeginForward(torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
                    torch::Tensor qo_indptr, torch::Tensor page_kv_indptr, unsigned int batch_size,
                    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
                    unsigned page_size, torch::Tensor empty_q_data);
  void EndForward();
  bool IsCUDAGraphEnabled() const { return handler_->IsCUDAGraphEnabled(); }
  void UpdatePageLockedBufferSize(uint32_t int_workspace_size_in_bytes);
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor qo_indptr,
                                     std::optional<torch::Tensor> paged_kv_cache,
                                     std::optional<torch::Tensor> paged_k_cache,
                                     std::optional<torch::Tensor> paged_v_cache,
                                     torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len, bool causal,
                                     unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
                                     int window_left, float logits_soft_cap, float sm_scale,
                                     float rope_scale, float rope_theta, bool return_lse);
  std::vector<torch::Tensor> ForwardCustomMask(
      torch::Tensor q, torch::Tensor qo_indptr, std::optional<torch::Tensor> paged_kv_cache,
      std::optional<torch::Tensor> paged_k_cache, std::optional<torch::Tensor> paged_v_cache,
      torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
      torch::Tensor paged_kv_last_page_len, torch::Tensor packed_custom_mask,
      torch::Tensor qk_indptr, unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
      int window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
      bool return_lse);
  BatchPrefillWithPagedKVCachePyTorchWrapper(unsigned int layout, bool enable_cuda_graph)
      : kv_layout_(flashinfer::QKVLayout(layout)),
        handler_(std::make_shared<flashinfer::BatchPrefillHandler>(enable_cuda_graph)) {}

 private:
  std::shared_ptr<flashinfer::BatchPrefillHandler> handler_;
  flashinfer::QKVLayout kv_layout_;
};

class BatchPrefillWithRaggedKVCachePyTorchWrapper {
 public:
  void BeginForward(torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
                    torch::Tensor qo_indptr, torch::Tensor kv_indptr, unsigned int batch_size,
                    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
                    torch::Tensor empty_q_data);
  void EndForward();
  bool IsCUDAGraphEnabled() const { return handler_->IsCUDAGraphEnabled(); }
  void UpdatePageLockedBufferSize(uint32_t int_workspace_size_in_bytes);
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor k,
                                     torch::Tensor v, torch::Tensor kv_indptr, bool causal,
                                     unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
                                     int window_left, float logits_soft_cap, float sm_scale,
                                     float rope_scale, float rope_theta, bool return_lse);
  std::vector<torch::Tensor> ForwardCustomMask(
      torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor k, torch::Tensor v,
      torch::Tensor kv_indptr, torch::Tensor packed_custom_mask, torch::Tensor qk_indptr,
      unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction, int window_left,
      float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse);
  BatchPrefillWithRaggedKVCachePyTorchWrapper(unsigned int layout, bool enable_cuda_graph)
      : kv_layout_(flashinfer::QKVLayout(layout)),
        handler_(std::make_shared<flashinfer::BatchPrefillHandler>(enable_cuda_graph)) {}

 private:
  std::shared_ptr<flashinfer::BatchPrefillHandler> handler_;
  flashinfer::QKVLayout kv_layout_;
};
