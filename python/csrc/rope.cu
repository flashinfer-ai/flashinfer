/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/pos_enc.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void apply_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                        torch::Tensor offsets, bool interleave, float rope_scale,
                        float rope_theta) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotaryInPlace(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotaryInPlace failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}

void apply_llama31_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                                torch::Tensor offsets, bool interleave, float rope_scale,
                                float rope_theta, float low_freq_factor, float high_freq_factor,
                                float old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyLlama31RotaryInPlace(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor,
        old_context_length, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyLlama31RotaryInPlace failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}

std::vector<torch::Tensor> apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                                      torch::Tensor offsets, bool interleave, float rope_scale,
                                      float rope_theta) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);
  // NOTE(Zihao): empty_like do not copy strides so it's okay to use it here.
  auto q_rope = torch::empty_like(q);
  auto k_rope = torch::empty_like(k);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotary(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotary failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });

  return {q_rope, k_rope};
}

std::vector<torch::Tensor> apply_llama31_rope(torch::Tensor q, torch::Tensor k,
                                              torch::Tensor indptr, torch::Tensor offsets,
                                              bool interleave, float rope_scale, float rope_theta,
                                              float low_freq_factor, float high_freq_factor,
                                              float old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);

  // NOTE(Zihao): empty_like do not copy strides so it's okay to use it here.
  auto q_rope = torch::empty_like(q);
  auto k_rope = torch::empty_like(k);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyLlama31Rotary(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor,
        old_context_length, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyLlama31Rotary failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });

  return {q_rope, k_rope};
}
