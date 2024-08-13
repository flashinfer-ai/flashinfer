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

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, unsigned int pos_encoding_mode,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta);

class BatchDecodeWithPagedKVCachePyTorchWrapper {
 public:
  void BeginForward(torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
                    torch::Tensor indptr, torch::Tensor last_page_len, unsigned int batch_size,
                    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
                    unsigned int page_size, unsigned int pos_encoding_mode, float logits_soft_cap,
                    torch::Tensor empty_q_data, torch::Tensor empty_kv_data);
  void EndForward();
  void UpdatePageLockedBufferSize(uint32_t int_workspace_size_in_bytes);
  bool IsCUDAGraphEnabled() const { return handler_->IsCUDAGraphEnabled(); }
  std::vector<torch::Tensor> Forward(torch::Tensor q, std::optional<torch::Tensor> paged_kv_cache,
                                     std::optional<torch::Tensor> paged_k_cache,
                                     std::optional<torch::Tensor> paged_v_cache,
                                     torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len,
                                     unsigned int pos_encoding_mode, int window_left,
                                     float logits_soft_cap, float sm_scale, float rope_scale,
                                     float rope_theta, bool return_lse);
  BatchDecodeWithPagedKVCachePyTorchWrapper(
      std::shared_ptr<flashinfer::BatchDecodeHandler> handler_ptr, flashinfer::QKVLayout kv_layout)
      : handler_(handler_ptr), kv_layout_(kv_layout) {}
  BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout, bool enable_cuda_graph,
                                            unsigned int fixed_batch_size)
      : kv_layout_(flashinfer::QKVLayout(layout)),
        handler_(std::make_shared<flashinfer::BatchDecodeHandler>(enable_cuda_graph,
                                                                  fixed_batch_size)) {}

 protected:
  std::shared_ptr<flashinfer::BatchDecodeHandler> handler_;
  flashinfer::QKVLayout kv_layout_;
};
