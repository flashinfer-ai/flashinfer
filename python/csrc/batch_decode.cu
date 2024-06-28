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

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

std::vector<torch::Tensor> batch_decode_with_padded_kv_cache(
    torch::Tensor q, torch::Tensor k_padded, torch::Tensor v_padded, unsigned int layout,
    unsigned int pos_encoding_mode, float logits_soft_cap, float sm_scale, float rope_scale,
    float rope_theta, bool return_lse) {
  CHECK_INPUT(q);
  CHECK_INPUT(k_padded);
  CHECK_INPUT(v_padded);
  CHECK_DIM(3, q);
  CHECK_DIM(4, k_padded);
  CHECK_DIM(4, v_padded);
  CHECK_SHAPE(k_padded, v_padded);
  CHECK_EQ(q.size(0), k_padded.size(0));
  CHECK_EQ(q.size(2), k_padded.size(3));
  CHECK_EQ(v_padded.scalar_type(), k_padded.scalar_type());
  unsigned int batch_size = q.size(0);
  unsigned int num_qo_heads = q.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int padded_kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    padded_kv_len = k_padded.size(1);
    num_kv_heads = k_padded.size(2);
  } else {
    padded_kv_len = k_padded.size(2);
    num_kv_heads = k_padded.size(1);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  auto o = torch::empty_like(
      q, q.options().dtype(is_float8_tensor(q) ? torch::kFloat16 : q.scalar_type()));
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({batch_size, num_qo_heads}, q.options()).to(torch::kFloat32);
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  if (is_float8_tensor(q)) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(q.scalar_type(), q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_COMBINED_FP8(k_padded.scalar_type(), kv_type, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_pos_encoding_mode(
              PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
                  return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
                    nv_half* tmp = nullptr;
                    cudaError_t status =
                        BatchDecodeWithPaddedKVCacheDispatched<HEAD_DIM, LOGITS_POST_HOOK,
                                                               KV_LAYOUT, POS_ENCODING_MODE, q_type,
                                                               kv_type, nv_half>(
                            static_cast<q_type*>(q.data_ptr()),
                            static_cast<kv_type*>(k_padded.data_ptr()),
                            static_cast<kv_type*>(v_padded.data_ptr()),
                            static_cast<nv_half*>(o.data_ptr()),
                            /*tmp=*/tmp,
                            /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                            batch_size, padded_kv_len, num_qo_heads, num_kv_heads, logits_soft_cap,
                            sm_scale, rope_scale, rope_theta, torch_current_stream);
                    TORCH_CHECK(status == cudaSuccess,
                                "BatchDecodeWithPaddedKVCache failed with error code ", status);
                    return true;
                  });
                });
              });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_COMBINED_FP8(k_padded.scalar_type(), kv_type, [&] {
        q_type* tmp = nullptr;
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_pos_encoding_mode(
              PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
                  return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
                    cudaError_t status =
                        BatchDecodeWithPaddedKVCacheDispatched<HEAD_DIM, LOGITS_POST_HOOK,
                                                               KV_LAYOUT, POS_ENCODING_MODE, q_type,
                                                               kv_type, q_type>(
                            static_cast<q_type*>(q.data_ptr()),
                            static_cast<kv_type*>(k_padded.data_ptr()),
                            static_cast<kv_type*>(v_padded.data_ptr()),
                            static_cast<q_type*>(o.data_ptr()),
                            /*tmp=*/tmp,
                            /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                            batch_size, padded_kv_len, num_qo_heads, num_kv_heads, logits_soft_cap,
                            sm_scale, rope_scale, rope_theta, torch_current_stream);
                    TORCH_CHECK(status == cudaSuccess,
                                "BatchDecodeWithPaddedKVCache failed with error code ", status);
                    return true;
                  });
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

void BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward(
    torch::Tensor workspace_buffer, torch::Tensor indptr, torch::Tensor last_page_len,
    unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int page_size, unsigned int pos_encoding_mode,
    float logits_soft_cap, torch::Tensor empty_q_data, torch::Tensor empty_kv_data) {
  // NOTE(zihao): not necessary to be CUDA tensor
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(last_page_len);
  CHECK_CONTIGUOUS(workspace_buffer);
  CHECK_DIM(1, indptr);
  CHECK_DIM(1, last_page_len);
  CHECK_DIM(1, workspace_buffer);
  CHECK_EQ(indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(indptr.scalar_type(), torch::kInt32);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  size_t workspace_size_in_bytes = workspace_buffer.size(0) * workspace_buffer.element_size();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  handler_->SetCUDAStream(torch_current_stream);
  indptr = indptr.to(torch::kCPU);
  last_page_len = last_page_len.to(torch::kCPU);

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  if (is_float8_tensor(empty_q_data)) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(empty_q_data.scalar_type(), q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_COMBINED_FP8(
          empty_kv_data.scalar_type(), kv_type, [&] {
            return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
              return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
                return DISPATCH_kv_layout(kv_layout_, KV_LAYOUT, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        cudaError_t status = handler_->BeginForwardDispatched<
                            HEAD_DIM, PageStorage::kIndices, LOGITS_POST_HOOK, KV_LAYOUT,
                            POS_ENCODING_MODE, q_type, kv_type, nv_half, int32_t>(
                            static_cast<void*>(workspace_buffer.data_ptr()),
                            workspace_size_in_bytes, static_cast<int32_t*>(indptr.data_ptr()),
                            static_cast<int32_t*>(last_page_len.data_ptr()), batch_size,
                            num_qo_heads, num_kv_heads, page_size);
                        TORCH_CHECK(status == cudaSuccess,
                                    "BatchDecodeWithPagedKVCache failed with error ",
                                    cudaGetErrorString(status));
                        return true;
                      });
                });
              });
            });
          });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(empty_q_data.scalar_type(), q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_COMBINED_FP8(
          empty_kv_data.scalar_type(), kv_type, [&] {
            return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
              return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
                return DISPATCH_kv_layout(kv_layout_, KV_LAYOUT, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        cudaError_t status = handler_->BeginForwardDispatched<
                            HEAD_DIM, PageStorage::kIndices, LOGITS_POST_HOOK, KV_LAYOUT,
                            POS_ENCODING_MODE, q_type, kv_type, q_type, int32_t>(
                            static_cast<void*>(workspace_buffer.data_ptr()),
                            workspace_size_in_bytes, static_cast<int32_t*>(indptr.data_ptr()),
                            static_cast<int32_t*>(last_page_len.data_ptr()), batch_size,
                            num_qo_heads, num_kv_heads, page_size);
                        TORCH_CHECK(status == cudaSuccess,
                                    "BatchDecodeWithPagedKVCache failed with error ",
                                    cudaGetErrorString(status));
                        return true;
                      });
                });
              });
            });
          });
    });
  }
}

void BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward() { handler_->EndForward(); }

void BatchDecodeWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize(
    unsigned int max_workspace_size_in_bytes) {
  handler_->UpdatePageLockedBufferSize(max_workspace_size_in_bytes);
}

std::vector<torch::Tensor> BatchDecodeWithPagedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor paged_kv_data, torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices, torch::Tensor paged_kv_last_page_len,
    unsigned int pos_encoding_mode, float logits_soft_cap, float sm_scale, float rope_scale,
    float rope_theta, bool return_lse) {
  CHECK_INPUT(q);
  CHECK_INPUT(paged_kv_data);
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  CHECK_DIM(3, q);                       // (B, H_qo, D)
  CHECK_DIM(1, paged_kv_last_page_len);  // (B,)
  CHECK_DIM(1, paged_kv_indptr);         // (B+1,)
  CHECK_DIM(1, paged_kv_indices);        // (nnz,)
  // (num_max_pages, 2, H_kv, page_size, head_dim) for HND
  // (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
  CHECK_DIM(5, paged_kv_data);
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (kv_layout_ == QKVLayout::kHND) {
    num_kv_heads = paged_kv_data.size(2);
    page_size = paged_kv_data.size(3);
  } else {
    page_size = paged_kv_data.size(2);
    num_kv_heads = paged_kv_data.size(3);
  }
  CHECK_EQ(paged_kv_data.size(1), 2);
  CHECK_EQ(paged_kv_data.size(4), head_dim);
  CHECK_GE(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_last_page_len.size(0), batch_size);
  // TODO(Zihao): support dispatching to different data types
  CHECK_EQ(paged_kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_last_page_len.scalar_type(), torch::kInt32);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  torch::Tensor o = torch::empty_like(
      q, q.options().dtype(is_float8_tensor(q) ? torch::kFloat16 : q.scalar_type()));
  torch::Tensor lse;
  if (return_lse) {
    lse = torch::empty({batch_size, num_qo_heads}, q.options()).to(torch::kFloat32);
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  if (is_float8_tensor(q)) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(q.scalar_type(), q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_COMBINED_FP8(
          paged_kv_data.scalar_type(), kv_type, [&] {
            return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
              return DISPATCH_kv_layout(kv_layout_, KV_LAYOUT, [&] {
                return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        paged_kv_t<PageStorage::kIndices, KV_LAYOUT, kv_type, int32_t> paged_kv(
                            num_kv_heads, page_size, head_dim, batch_size,
                            static_cast<kv_type*>(paged_kv_data.data_ptr()),
                            static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                            static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                            static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
                        cudaError_t status = BatchDecodeWithPagedKVCacheWrapperDispatched<
                            PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT,
                            POS_ENCODING_MODE, q_type, kv_type, nv_half, int32_t>(
                            handler_.get(), static_cast<q_type*>(q.data_ptr()),
                            /*q_offset=*/nullptr, paged_kv, static_cast<nv_half*>(o.data_ptr()),
                            /*lse=*/(return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr),
                            num_qo_heads, logits_soft_cap, sm_scale, rope_scale, rope_theta,
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
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_COMBINED_FP8(
          paged_kv_data.scalar_type(), kv_type, [&] {
            return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
              return DISPATCH_kv_layout(kv_layout_, KV_LAYOUT, [&] {
                return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        paged_kv_t<PageStorage::kIndices, KV_LAYOUT, kv_type, int32_t> paged_kv(
                            num_kv_heads, page_size, head_dim, batch_size,
                            static_cast<kv_type*>(paged_kv_data.data_ptr()),
                            static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                            static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                            static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
                        cudaError_t status = BatchDecodeWithPagedKVCacheWrapperDispatched<
                            PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, KV_LAYOUT,
                            POS_ENCODING_MODE, q_type, kv_type, q_type, int32_t>(
                            handler_.get(), static_cast<q_type*>(q.data_ptr()),
                            /*q_offset=*/nullptr, paged_kv, static_cast<q_type*>(o.data_ptr()),
                            /*lse=*/(return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr),
                            num_qo_heads, logits_soft_cap, sm_scale, rope_scale, rope_theta,
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
    });
  }

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
