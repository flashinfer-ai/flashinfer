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
#include <flashinfer/prefill_attention_decl.cuh>

#include "flashinfer_ops_prefill.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor qo_indptr, torch::Tensor paged_kv_indptr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
    unsigned int page_size, torch::Tensor empty_q_data) {
  CHECK_INPUT(float_workspace_buffer);
  CHECK_INPUT(int_workspace_buffer);
  // NOTE(Zihao): not necessary to be a CUDA tensor
  CHECK_CONTIGUOUS(qo_indptr);
  CHECK_CONTIGUOUS(paged_kv_indptr);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, paged_kv_indptr);
  CHECK_DIM(1, float_workspace_buffer);
  CHECK_DIM(1, int_workspace_buffer);
  CHECK_EQ(qo_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1);
  qo_indptr = qo_indptr.to(torch::dtype(torch::kInt32).device(torch::kCPU));
  paged_kv_indptr = paged_kv_indptr.to(torch::dtype(torch::kInt32).device(torch::kCPU));
  auto device = float_workspace_buffer.device();
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  handler_->SetCUDAStream(torch_current_stream);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(empty_q_data.scalar_type(), q_type, [&] {
    cudaError_t status = handler_->BeginForward<q_type, int32_t>(
        static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
        static_cast<void*>(int_workspace_buffer.data_ptr()), int_workspace_size_in_bytes,
        static_cast<int32_t*>(qo_indptr.data_ptr()),
        static_cast<int32_t*>(paged_kv_indptr.data_ptr()), batch_size, num_qo_heads, num_kv_heads,
        head_dim, page_size);
    TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                cudaGetErrorString(status));
    return true;
  });
}

void BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward() { handler_->EndForward(); }

void BatchPrefillWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize(
    unsigned int int_workspace_size_in_bytes) {
  handler_->UpdatePageLockedBufferSize(int_workspace_size_in_bytes);
}

std::vector<torch::Tensor> BatchPrefillWithPagedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor qo_indptr, std::optional<torch::Tensor> paged_kv_cache,
    std::optional<torch::Tensor> paged_k_cache, std::optional<torch::Tensor> paged_v_cache,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, bool causal, unsigned int pos_encoding_mode,
    bool allow_fp16_qk_reduction, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, bool return_lse) {
  bool paged_kv_defined = paged_kv_cache.has_value();
  CHECK_INPUT(q);
  CHECK_INPUT(qo_indptr);
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
  CHECK_EQ(device, qo_indptr.device());
  if (paged_kv_defined) {
    CHECK_EQ(device, paged_kv_cache->device());
  } else {
    CHECK_EQ(device, paged_k_cache->device());
    CHECK_EQ(device, paged_v_cache->device());
  }
  CHECK_EQ(device, paged_kv_indptr.device());
  CHECK_EQ(device, paged_kv_indices.device());
  CHECK_EQ(device, paged_kv_last_page_len.device());
  CHECK_DIM(3, q);          // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr);  // (B + 1,)

  if (paged_kv_defined) {
    // [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
    // [max_num_pages, 2, page_size, num_kv_heads, head_dim] for HND
    CHECK_DIM(5, paged_kv_cache.value());
  } else {
    // [max_num_pages, num_kv_heads, page_size, head_dim] for HND
    // [max_num_pages, page_size, num_kv_heads, head_dim] for HND
    CHECK_DIM(4, paged_k_cache.value());
    CHECK_DIM(4, paged_v_cache.value());
  }

  CHECK_DIM(1, paged_kv_indptr);         // (B + 1,)
  CHECK_DIM(1, paged_kv_indices);        // (nnz_kv,)
  CHECK_DIM(1, paged_kv_last_page_len);  // (B,)
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
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
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_GE(qo_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_last_page_len.size(0), batch_size);
  qo_indptr = qo_indptr.to(torch::kInt32);
  paged_kv_indptr = paged_kv_indptr.to(torch::kInt32);
  paged_kv_indices = paged_kv_indices.to(torch::kInt32);
  paged_kv_last_page_len = paged_kv_last_page_len.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }
  MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type =
      paged_kv_defined ? paged_kv_cache->scalar_type() : paged_k_cache->scalar_type();

  if (q_scalar_type == kv_scalar_type) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, c_type, [&] {
      return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
        paged_kv_t<PageStorage::kIndices, c_type, int32_t> paged_kv(
            num_kv_heads, page_size, head_dim, batch_size, kv_layout_,
            static_cast<c_type*>(paged_kv_cache.has_value() ? paged_kv_cache->data_ptr() : nullptr),
            static_cast<c_type*>(paged_k_cache.has_value() ? paged_k_cache->data_ptr() : nullptr),
            static_cast<c_type*>(paged_v_cache.has_value() ? paged_v_cache->data_ptr() : nullptr),
            static_cast<int32_t*>(paged_kv_indices.data_ptr()),
            static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
            static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
            return DISPATCH_allow_fp16_qk_reduction(
                allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        cudaError_t status = BatchPrefillWithPagedKVCacheWrapperDispatched<
                            PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE,
                            ALLOW_FP16_QK_REDUCTION, MASK_MODE, c_type, c_type, c_type, int32_t>(
                            handler_.get(), static_cast<c_type*>(q.data_ptr()),
                            static_cast<int32_t*>(qo_indptr.data_ptr()),
                            /*q_offset=*/nullptr, paged_kv,
                            /*custom_mask=*/nullptr,
                            /*qk_indptr=*/nullptr, static_cast<c_type*>(o.data_ptr()),
                            /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                            num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale,
                            rope_theta,
                            /*stream=*/torch_current_stream);
                        TORCH_CHECK(status == cudaSuccess,
                                    "BatchPrefillWithPagedKVCache failed with error code ",
                                    cudaGetErrorString(status));
                        return true;
                      });
                });
          });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(kv_scalar_type, kv_type, [&] {
        return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
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
          return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
            return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
              return DISPATCH_allow_fp16_qk_reduction(
                  allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                    return DISPATCH_pos_encoding_mode(
                        PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                          cudaError_t status = BatchPrefillWithPagedKVCacheWrapperDispatched<
                              PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE,
                              ALLOW_FP16_QK_REDUCTION, MASK_MODE, q_type, kv_type, q_type, int32_t>(
                              handler_.get(), static_cast<q_type*>(q.data_ptr()),
                              static_cast<int32_t*>(qo_indptr.data_ptr()),
                              /*q_offset=*/nullptr, paged_kv,
                              /*custom_mask=*/nullptr,
                              /*qk_indptr=*/nullptr, static_cast<q_type*>(o.data_ptr()),
                              /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                              num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale,
                              rope_theta,
                              /*stream=*/torch_current_stream);
                          TORCH_CHECK(status == cudaSuccess,
                                      "BatchPrefillWithPagedKVCache failed with error code ",
                                      cudaGetErrorString(status));
                          return true;
                        });
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

std::vector<torch::Tensor> BatchPrefillWithPagedKVCachePyTorchWrapper::ForwardCustomMask(
    torch::Tensor q, torch::Tensor qo_indptr, std::optional<torch::Tensor> paged_kv_cache,
    std::optional<torch::Tensor> paged_k_cache, std::optional<torch::Tensor> paged_v_cache,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, torch::Tensor custom_mask, torch::Tensor qk_indptr,
    unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction, int window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
  bool paged_kv_defined = paged_kv_cache.has_value();
  CHECK_INPUT(q);
  CHECK_INPUT(qo_indptr);
  if (paged_kv_defined) {
    CHECK_INPUT(paged_kv_cache.value());
  } else {
    CHECK_INPUT(paged_k_cache.value());
    CHECK_INPUT(paged_v_cache.value());
  }
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  CHECK_INPUT(custom_mask);
  CHECK_INPUT(qk_indptr);
  auto device = q.device();
  CHECK_EQ(device, qo_indptr.device());
  if (paged_kv_defined) {
    CHECK_EQ(device, paged_kv_cache->device());
  } else {
    CHECK_EQ(device, paged_k_cache->device());
    CHECK_EQ(device, paged_v_cache->device());
  }
  CHECK_EQ(device, paged_kv_indptr.device());
  CHECK_EQ(device, paged_kv_indices.device());
  CHECK_EQ(device, paged_kv_last_page_len.device());
  CHECK_EQ(device, custom_mask.device());
  CHECK_EQ(device, qk_indptr.device());
  CHECK_DIM(3, q);          // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr);  // (B + 1,)

  if (paged_kv_defined) {
    // [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
    // [max_num_pages, 2, page_size, num_kv_heads, head_dim] for NHD
    CHECK_DIM(5, paged_kv_cache.value());
  } else {
    // [max_num_pages, num_kv_heads, page_size, head_dim] for HND
    // [max_num_pages, page_size, num_kv_heads, head_dim] for NHD
    CHECK_DIM(4, paged_k_cache.value());
    CHECK_DIM(4, paged_v_cache.value());
  }
  CHECK_DIM(1, paged_kv_indptr);         // (B + 1,)
  CHECK_DIM(1, paged_kv_indices);        // (nnz_kv,)
  CHECK_DIM(1, paged_kv_last_page_len);  // (B,)
  CHECK_DIM(1, custom_mask);             // (nnz_qk,)
  CHECK_DIM(1, qk_indptr);               // (B + 1,)
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
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
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_GE(qo_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_GE(paged_kv_last_page_len.size(0), batch_size);
  CHECK_GE(qk_indptr.size(0), batch_size + 1);
  qo_indptr = qo_indptr.to(torch::kInt32);
  paged_kv_indptr = paged_kv_indptr.to(torch::kInt32);
  paged_kv_indices = paged_kv_indices.to(torch::kInt32);
  paged_kv_last_page_len = paged_kv_last_page_len.to(torch::kInt32);
  qk_indptr = qk_indptr.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }
  constexpr MaskMode MASK_MODE = MaskMode::kCustom;
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type =
      paged_kv_defined ? paged_kv_cache->scalar_type() : paged_k_cache->scalar_type();

  if (q_scalar_type == kv_scalar_type) {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, c_type, [&] {
      return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
        paged_kv_t<PageStorage::kIndices, c_type, int32_t> paged_kv(
            num_kv_heads, page_size, head_dim, batch_size, kv_layout_,
            static_cast<c_type*>(paged_kv_cache.has_value() ? paged_kv_cache->data_ptr() : nullptr),
            static_cast<c_type*>(paged_k_cache.has_value() ? paged_k_cache->data_ptr() : nullptr),
            static_cast<c_type*>(paged_v_cache.has_value() ? paged_v_cache->data_ptr() : nullptr),
            static_cast<int32_t*>(paged_kv_indices.data_ptr()),
            static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
            static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_allow_fp16_qk_reduction(
              allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                return DISPATCH_pos_encoding_mode(
                    PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                      cudaError_t status = BatchPrefillWithPagedKVCacheWrapperDispatched<
                          PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE,
                          ALLOW_FP16_QK_REDUCTION, MASK_MODE, c_type, c_type, c_type, int32_t>(
                          handler_.get(), static_cast<c_type*>(q.data_ptr()),
                          static_cast<int32_t*>(qo_indptr.data_ptr()),
                          /*q_offset=*/nullptr, paged_kv,
                          static_cast<uint8_t*>(custom_mask.data_ptr()),
                          static_cast<int32_t*>(qk_indptr.data_ptr()),
                          static_cast<c_type*>(o.data_ptr()),
                          /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                          num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale,
                          rope_theta,
                          /*stream=*/torch_current_stream);
                      TORCH_CHECK(status == cudaSuccess,
                                  "BatchPrefillWithPagedKVCache failed with error code ",
                                  cudaGetErrorString(status));
                      return true;
                    });
              });
        });
      });
    });
  } else {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, q_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(kv_scalar_type, kv_type, [&] {
        return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
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
          return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
            return DISPATCH_allow_fp16_qk_reduction(
                allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                        cudaError_t status = BatchPrefillWithPagedKVCacheWrapperDispatched<
                            PageStorage::kIndices, HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE,
                            ALLOW_FP16_QK_REDUCTION, MASK_MODE, q_type, kv_type, q_type, int32_t>(
                            handler_.get(), static_cast<q_type*>(q.data_ptr()),
                            static_cast<int32_t*>(qo_indptr.data_ptr()),
                            /*q_offset=*/nullptr, paged_kv,
                            static_cast<uint8_t*>(custom_mask.data_ptr()),
                            static_cast<int32_t*>(qk_indptr.data_ptr()),
                            static_cast<q_type*>(o.data_ptr()),
                            /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                            num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale,
                            rope_theta,
                            /*stream=*/torch_current_stream);
                        TORCH_CHECK(status == cudaSuccess,
                                    "BatchPrefillWithPagedKVCache failed with error code ",
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

void BatchPrefillWithRaggedKVCachePyTorchWrapper::BeginForward(
    torch::Tensor float_workspace_buffer, torch::Tensor int_workspace_buffer,
    torch::Tensor qo_indptr, torch::Tensor kv_indptr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int num_kv_heads, unsigned int head_dim,
    torch::Tensor empty_q_data) {
  CHECK_INPUT(float_workspace_buffer);
  CHECK_INPUT(int_workspace_buffer);
  // NOTE(Zihao): not necessary to be a CUDA tensor
  CHECK_CONTIGUOUS(qo_indptr);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, float_workspace_buffer);
  CHECK_DIM(1, int_workspace_buffer);
  CHECK_EQ(qo_indptr.size(0), batch_size + 1);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  qo_indptr = qo_indptr.to(torch::dtype(torch::kInt32).device(torch::kCPU));
  kv_indptr = kv_indptr.to(torch::dtype(torch::kInt32).device(torch::kCPU));
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();
  auto device = float_workspace_buffer.device();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  handler_->SetCUDAStream(torch_current_stream);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(empty_q_data.scalar_type(), q_type, [&] {
    cudaError_t status = handler_->BeginForward<q_type, int32_t>(
        static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
        static_cast<void*>(int_workspace_buffer.data_ptr()), int_workspace_size_in_bytes,
        static_cast<int32_t*>(qo_indptr.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, head_dim,
        /*page_size=*/1);
    TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                cudaGetErrorString(status));
    return true;
  });
}

void BatchPrefillWithRaggedKVCachePyTorchWrapper::EndForward() { handler_->EndForward(); }

void BatchPrefillWithRaggedKVCachePyTorchWrapper::UpdatePageLockedBufferSize(
    unsigned int int_workspace_size_in_bytes) {
  handler_->UpdatePageLockedBufferSize(int_workspace_size_in_bytes);
}

std::vector<torch::Tensor> BatchPrefillWithRaggedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor k, torch::Tensor v,
    torch::Tensor kv_indptr, bool causal, unsigned int pos_encoding_mode,
    bool allow_fp16_qk_reduction, int window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, bool return_lse) {
  CHECK_INPUT(qo_indptr);
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_INPUT(kv_indptr);
  auto device = q.device();
  CHECK_EQ(device, qo_indptr.device());
  CHECK_EQ(device, k.device());
  CHECK_EQ(device, v.device());
  CHECK_EQ(device, kv_indptr.device());
  CHECK_DIM(3, q);          // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr);  // (B + 1,)
  CHECK_DIM(3, k);          // (nnz_kv, H_kv, D) if NHD else (H_kv, nnz_kv, D)
  CHECK_DIM(3, v);          // (nnz_kv, H_kv, D) if NHD else (H_kv, nnz_kv, D)
  CHECK_DIM(1, kv_indptr);  // (B + 1,)
  CHECK_EQ(q.scalar_type(), k.scalar_type());
  CHECK_EQ(q.scalar_type(), v.scalar_type());
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  CHECK_GE(kv_indptr.size(0), batch_size + 1);
  int64_t num_kv_heads = (kv_layout_ == QKVLayout::kNHD) ? k.size(1) : k.size(0);
  CHECK_EQ(q.stride(2), 1);
  CHECK_EQ(k.stride(2), 1);
  CHECK_EQ(v.stride(2), 1);
  CHECK_EQ(k.size(0), v.size(0));
  CHECK_EQ(k.size(1), v.size(1));
  CHECK_EQ(k.size(2), v.size(2));
  CHECK_EQ(k.size(2), head_dim);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout_ == QKVLayout::kNHD) {
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }
  qo_indptr = qo_indptr.to(torch::kInt32);
  kv_indptr = kv_indptr.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  TORCH_CHECK(q_scalar_type == kv_scalar_type,
              "q and k must have the same scalar type, but got q: ", q_scalar_type,
              " and k: ", kv_scalar_type);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, c_type, [&] {
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
        return DISPATCH_allow_fp16_qk_reduction(
            allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
              return DISPATCH_pos_encoding_mode(
                  PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                    return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
                      cudaError_t status = BatchPrefillWithRaggedKVCacheWrapperDispatched<
                          HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION,
                          MASK_MODE, c_type, c_type, c_type, int32_t>(
                          handler_.get(), static_cast<c_type*>(q.data_ptr()),
                          static_cast<int32_t*>(qo_indptr.data_ptr()),
                          static_cast<c_type*>(k.data_ptr()), static_cast<c_type*>(v.data_ptr()),
                          static_cast<int32_t*>(kv_indptr.data_ptr()),
                          /*custom_mask=*/nullptr, /*qk_indptr=*/nullptr,
                          /*q_offset=*/nullptr, /*k_rope_pos_offset=*/nullptr,
                          static_cast<c_type*>(o.data_ptr()),
                          /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                          num_qo_heads, num_kv_heads, q_stride_n, q_stride_h, kv_stride_n,
                          kv_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale,
                          rope_theta,
                          /*stream=*/torch_current_stream);
                      TORCH_CHECK(status == cudaSuccess,
                                  "BatchPrefillWithRaggedKVCache failed with error ",
                                  cudaGetErrorString(status));
                      return true;
                    });
                  });
            });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

std::vector<torch::Tensor> BatchPrefillWithRaggedKVCachePyTorchWrapper::ForwardCustomMask(
    torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor k, torch::Tensor v,
    torch::Tensor kv_indptr, torch::Tensor custom_mask, torch::Tensor qk_indptr,
    unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction, int window_left,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
  CHECK_INPUT(qo_indptr);
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(custom_mask);
  CHECK_INPUT(qk_indptr);
  auto device = q.device();
  CHECK_EQ(device, qo_indptr.device());
  CHECK_EQ(device, k.device());
  CHECK_EQ(device, v.device());
  CHECK_EQ(device, kv_indptr.device());
  CHECK_EQ(device, custom_mask.device());
  CHECK_EQ(device, qk_indptr.device());
  CHECK_DIM(3, q);            // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr);    // (B + 1,)
  CHECK_DIM(3, k);            // (nnz_kv, H_kv, D) if NHD else (H_kv, nnz_kv, D)
  CHECK_DIM(3, v);            // (nnz_kv, H_kv, D) if NHD else (H_kv, nnz_kv, D)
  CHECK_DIM(1, kv_indptr);    // (B + 1,)
  CHECK_DIM(1, custom_mask);  // (nnz_qk,)
  CHECK_DIM(1, qk_indptr);    // (B + 1,)
  CHECK_EQ(q.scalar_type(), k.scalar_type());
  CHECK_EQ(q.scalar_type(), v.scalar_type());
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  CHECK_GE(kv_indptr.size(0), batch_size + 1);
  CHECK_GE(qk_indptr.size(0), batch_size + 1);
  int64_t num_kv_heads = (kv_layout_ == QKVLayout::kNHD) ? k.size(1) : k.size(0);
  CHECK_EQ(q.stride(2), 1);
  CHECK_EQ(k.stride(2), 1);
  CHECK_EQ(v.stride(2), 1);
  CHECK_EQ(k.size(0), v.size(0));
  CHECK_EQ(k.size(1), v.size(1));
  CHECK_EQ(k.size(2), v.size(2));
  CHECK_EQ(k.size(2), head_dim);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout_ == QKVLayout::kNHD) {
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }
  qo_indptr = qo_indptr.to(torch::kInt32);
  kv_indptr = kv_indptr.to(torch::kInt32);
  qk_indptr = qk_indptr.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options().dtype((torch::kFloat32)));
  }

  constexpr MaskMode MASK_MODE = MaskMode::kCustom;
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();
  TORCH_CHECK(q_scalar_type == kv_scalar_type,
              "q and k must have the same scalar type, but got q: ", q_scalar_type,
              " and k: ", kv_scalar_type);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_allow_fp16_qk_reduction(
          allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
            return DISPATCH_pos_encoding_mode(
                PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                  return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
                    cudaError_t status = BatchPrefillWithRaggedKVCacheWrapperDispatched<
                        HEAD_DIM, LOGITS_POST_HOOK, POS_ENCODING_MODE, ALLOW_FP16_QK_REDUCTION,
                        MASK_MODE, c_type, c_type, c_type, int32_t>(
                        handler_.get(), static_cast<c_type*>(q.data_ptr()),
                        static_cast<int32_t*>(qo_indptr.data_ptr()),
                        static_cast<c_type*>(k.data_ptr()), static_cast<c_type*>(v.data_ptr()),
                        static_cast<int32_t*>(kv_indptr.data_ptr()),
                        static_cast<uint8_t*>(custom_mask.data_ptr()),
                        static_cast<int32_t*>(qk_indptr.data_ptr()),
                        /*q_offset=*/nullptr, /*k_rope_pos_offset=*/nullptr,
                        static_cast<c_type*>(o.data_ptr()),
                        /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                        num_qo_heads, num_kv_heads, q_stride_n, q_stride_h, kv_stride_n,
                        kv_stride_h, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta,
                        /*stream=*/torch_current_stream);
                    TORCH_CHECK(status == cudaSuccess,
                                "BatchPrefillWithRaggedKVCache failed with error ",
                                cudaGetErrorString(status));
                    return true;
                  });
                });
          });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
