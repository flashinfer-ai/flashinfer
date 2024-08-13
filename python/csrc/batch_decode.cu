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
#include <flashinfer/decode_attention_decl.cuh>

#include "flashinfer_ops_decode.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer, torch::Tensor indptr,
    torch::Tensor last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
    unsigned int pos_encoding_mode, float logits_soft_cap, torch::Tensor empty_q_data,
    torch::Tensor empty_kv_data) {
  CHECK_INPUT(float_workspace_buffer);
  CHECK_INPUT(int_workspace_buffer);
  // NOTE(zihao): not necessary to be CUDA tensor
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(last_page_len);
  CHECK_DIM(1, indptr);
  CHECK_DIM(1, last_page_len);
  CHECK_DIM(1, float_workspace_buffer);
  CHECK_DIM(1, int_workspace_buffer);
  CHECK_EQ(indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(indptr.scalar_type(), torch::kInt32);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  handler_->SetCUDAStream(torch_current_stream);
  indptr = indptr.to(torch::kCPU);
  last_page_len = last_page_len.to(torch::kCPU);

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = empty_q_data.scalar_type();
  auto kv_scalar_type = empty_kv_data.scalar_type();

  if (q_scalar_type == kv_scalar_type) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q_scalar_type, qkv_type, [&] {
      return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
        return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
          return DISPATCH_pos_encoding_mode(
              PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                cudaError_t status =
                    handler_->BeginForwardDispatched<HEAD_DIM, PageStorage::kIndices,
                                                     LOGITS_POST_HOOK, POS_ENCODING_MODE, qkv_type,
                                                     qkv_type, qkv_type, int32_t>(
                        static_cast<void*>(float_workspace_buffer.data_ptr()),
                        float_workspace_size_in_bytes,
                        static_cast<void*>(int_workspace_buffer.data_ptr()),
                        int_workspace_size_in_bytes, static_cast<int32_t*>(indptr.data_ptr()),
                        static_cast<int32_t*>(last_page_len.data_ptr()), batch_size, num_qo_heads,
                        num_kv_heads, page_size);
                TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
                            cudaGetErrorString(status));
                return true;
              });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(kv_scalar_type, kv_type, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
            return DISPATCH_pos_encoding_mode(
                PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                  cudaError_t status =
                      handler_->BeginForwardDispatched<HEAD_DIM, PageStorage::kIndices,
                                                       LOGITS_POST_HOOK, POS_ENCODING_MODE, q_type,
                                                       kv_type, q_type, int32_t>(
                          static_cast<void*>(float_workspace_buffer.data_ptr()),
                          float_workspace_size_in_bytes,
                          static_cast<void*>(int_workspace_buffer.data_ptr()),
                          int_workspace_size_in_bytes, static_cast<int32_t*>(indptr.data_ptr()),
                          static_cast<int32_t*>(last_page_len.data_ptr()), batch_size, num_qo_heads,
                          num_kv_heads, page_size);
                  TORCH_CHECK(status == cudaSuccess,
                              "BatchDecodeWithPagedKVCache failed with error ",
                              cudaGetErrorString(status));
                  return true;
                });
          });
        });
      });
    });
  }
}

void BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward() { handler_->EndForward(); }

void BatchDecodeWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize(
    unsigned int int_workspace_size_in_bytes) {
  handler_->UpdatePageLockedBufferSize(int_workspace_size_in_bytes);
}

std::vector<torch::Tensor> BatchDecodeWithPagedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, std::optional<torch::Tensor> paged_kv_cache,
    std::optional<torch::Tensor> paged_k_cache, std::optional<torch::Tensor> paged_v_cache,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, unsigned int pos_encoding_mode, int window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
  CHECK_INPUT(q);
  bool paged_kv_defined = paged_kv_cache.has_value();
  if (paged_kv_defined) {
    CHECK_INPUT(paged_kv_cache.value());
  } else {
    CHECK_INPUT(paged_k_cache.value());
    CHECK_INPUT(paged_v_cache.value());
  }
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  auto device = q.device();
  if (paged_kv_defined) {
    CHECK_EQ(paged_kv_cache->device(), device);
  } else {
    CHECK_EQ(paged_k_cache->device(), device);
    CHECK_EQ(paged_v_cache->device(), device);
  }
  CHECK_EQ(paged_kv_indices.device(), device);
  CHECK_EQ(paged_kv_indptr.device(), device);
  CHECK_EQ(paged_kv_last_page_len.device(), device);
  CHECK_DIM(3, q);                       // (B, H_qo, D)
  CHECK_DIM(1, paged_kv_last_page_len);  // (B,)
  CHECK_DIM(1, paged_kv_indptr);         // (B+1,)
  CHECK_DIM(1, paged_kv_indices);        // (nnz,)
  if (paged_kv_defined) {
    // (num_max_pages, 2, H_kv, page_size, head_dim) for HND
    // (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
    CHECK_DIM(5, paged_kv_cache.value());
  } else {
    // (num_max_pages, H_kv, page_size, head_dim) for HND
    // (num_max_pages, page_size, H_kv, head_dim) for NHD
    CHECK_DIM(4, paged_k_cache.value());
    CHECK_DIM(4, paged_v_cache.value());
  }
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (paged_kv_defined) {
    CHECK_EQ(paged_kv_cache->size(1), 2);
    CHECK_EQ(paged_kv_cache->size(4), head_dim);
    if (kv_layout_ == QKVLayout::kHND) {
      num_kv_heads = paged_kv_cache->size(2);
      page_size = paged_kv_cache->size(3);
    } else {
      page_size = paged_kv_cache->size(2);
      num_kv_heads = paged_kv_cache->size(3);
    }
  } else {
    CHECK_EQ(paged_k_cache->size(3), head_dim);
    CHECK_EQ(paged_v_cache->size(3), head_dim);
    if (kv_layout_ == QKVLayout::kHND) {
      num_kv_heads = paged_k_cache->size(1);
      page_size = paged_k_cache->size(2);
    } else {
      page_size = paged_k_cache->size(1);
      num_kv_heads = paged_k_cache->size(2);
    }
  }
  CHECK_GE(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_last_page_len.size(0), batch_size);
  // TODO(Zihao): support dispatching to different data types
  CHECK_EQ(paged_kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_last_page_len.scalar_type(), torch::kInt32);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q);
  torch::Tensor lse;
  if (return_lse) {
    lse = torch::empty({batch_size, num_qo_heads}, q.options().dtype((torch::kFloat32)));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type =
      paged_kv_defined ? paged_kv_cache->scalar_type() : paged_k_cache->scalar_type();

  if (q_scalar_type == kv_scalar_type) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q_scalar_type, qkv_type, [&] {
      return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_pos_encoding_mode(
              PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                paged_kv_t<PageStorage::kIndices, qkv_type, int32_t> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, kv_layout_,
                    static_cast<qkv_type*>(paged_kv_cache.has_value() ? paged_kv_cache->data_ptr()
                                                                      : nullptr),
                    static_cast<qkv_type*>(paged_k_cache.has_value() ? paged_k_cache->data_ptr()
                                                                     : nullptr),
                    static_cast<qkv_type*>(paged_v_cache.has_value() ? paged_v_cache->data_ptr()
                                                                     : nullptr),
                    static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                    static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                    static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
                cudaError_t status = BatchDecodeWithPagedKVCacheWrapperDispatched<
                    PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE, qkv_type,
                    qkv_type, qkv_type, int32_t>(
                    handler_.get(), static_cast<qkv_type*>(q.data_ptr()),
                    /*q_offset=*/nullptr, paged_kv, static_cast<qkv_type*>(o.data_ptr()),
                    /*lse=*/(return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr),
                    num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta,
                    /*stream=*/torch_current_stream);
                TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
                            cudaGetErrorString(status));
                return true;
              });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(kv_scalar_type, kv_type, [&] {
        return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
          return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
            return DISPATCH_pos_encoding_mode(
                PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                  paged_kv_t<PageStorage::kIndices, kv_type, int32_t> paged_kv(
                      num_kv_heads, page_size, head_dim, batch_size, kv_layout_,
                      static_cast<kv_type*>(paged_kv_cache.has_value() ? paged_kv_cache->data_ptr()
                                                                       : nullptr),
                      static_cast<kv_type*>(paged_k_cache.has_value() ? paged_k_cache->data_ptr()
                                                                      : nullptr),
                      static_cast<kv_type*>(paged_v_cache.has_value() ? paged_v_cache->data_ptr()
                                                                      : nullptr),
                      static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                      static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                      static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
                  cudaError_t status = BatchDecodeWithPagedKVCacheWrapperDispatched<
                      PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE, q_type,
                      kv_type, q_type, int32_t>(
                      handler_.get(), static_cast<q_type*>(q.data_ptr()),
                      /*q_offset=*/nullptr, paged_kv, static_cast<q_type*>(o.data_ptr()),
                      /*lse=*/(return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr),
                      num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta,
                      /*stream=*/torch_current_stream);
                  TORCH_CHECK(status == cudaSuccess,
                              "BatchDecodeWithPagedKVCache failed with error ",
                              cudaGetErrorString(status));
                  return true;
                });
          });
        });
      });
    });
  }

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
