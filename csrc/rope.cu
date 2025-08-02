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

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
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
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  CHECK_EQ(indptr.scalar_type(), offsets.scalar_type());

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(indptr.scalar_type(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotary(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(indptr.data_ptr()), static_cast<c_idtype*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
          k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n,
          k_rope_stride_h, interleave, rope_scale, rope_theta, stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotary failed with error code " +
                                             std::string(cudaGetErrorString(status)));
      return true;
    });
  });
}

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(pos_ids);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotaryPosIds(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
          head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n,
          q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta,
          stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotaryPosIds failed with error code " +
                                             std::string(cudaGetErrorString(status)));
      return true;
    });
  });
}

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids, bool interleave) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(cos_sin_cache.device(), device);
  CHECK_EQ(pos_ids.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  // cos_sin_cache: (max_seq_len, R)
  // First half of R is cos, second half is sin
  CHECK_DIM(2, cos_sin_cache);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int rotary_dim = cos_sin_cache.size(1);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()), static_cast<c_idtype*>(pos_ids.data_ptr()),
          nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
          k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
          interleave, stream);
      TORCH_CHECK(status == cudaSuccess,
                  "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " +
                      std::string(cudaGetErrorString(status)));
      return true;
    });
  });
}

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length) {
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
  CHECK_EQ(indptr.scalar_type(), offsets.scalar_type());
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  CHECK_EQ(indptr.scalar_type(), offsets.scalar_type());

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(indptr.scalar_type(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyLlama31Rotary(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(indptr.data_ptr()), static_cast<c_idtype*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
          k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n,
          k_rope_stride_h, interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor,
          old_context_length, stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyLlama31Rotary failed with error code " +
                                             std::string(cudaGetErrorString(status)));
      return true;
    });
  });
}

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(pos_ids);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyLlama31RotaryPosIds(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
          head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n,
          q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta,
          low_freq_factor, high_freq_factor, old_context_length, stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyLlama31RotaryPosIds failed with error code " +
                                             std::string(cudaGetErrorString(status)));
      return true;
    });
  });
}

void mla_rope_quantize(at::Tensor q_rope_in, at::Tensor k_rope_in, at::Tensor q_nope_in,
                       at::Tensor k_nope_in, at::Tensor q_rope_out, at::Tensor k_rope_out,
                       at::Tensor q_nope_out, at::Tensor k_nope_out, at::Tensor cos_sin_cache,
                       at::Tensor pos_ids, double quant_scale_q, double quant_scale_kv,
                       bool interleave) {
  CHECK_LAST_DIM_CONTIGUOUS(q_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS(k_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS(q_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS(q_nope_out);
  CHECK_LAST_DIM_CONTIGUOUS(k_nope_out);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);

  CHECK_EQ(q_rope_in.size(-1), 64);
  CHECK_EQ(k_rope_in.size(-1), 64);
  CHECK_EQ(q_nope_in.size(-1), 512);
  CHECK_EQ(k_nope_in.size(-1), 512);
  CHECK_EQ(q_rope_out.size(-1), 64);
  CHECK_EQ(k_rope_out.size(-1), 64);
  CHECK_EQ(q_nope_out.size(-1), 512);
  CHECK_EQ(k_nope_out.size(-1), 512);
  auto scalar_type_in = q_rope_in.scalar_type();
  TORCH_CHECK(scalar_type_in == k_rope_in.scalar_type());
  TORCH_CHECK(scalar_type_in == q_nope_in.scalar_type());
  TORCH_CHECK(scalar_type_in == k_nope_in.scalar_type());
  auto quant_type_out = q_rope_out.scalar_type();
  TORCH_CHECK(quant_type_out == k_rope_out.scalar_type());
  TORCH_CHECK(quant_type_out == q_nope_out.scalar_type());
  TORCH_CHECK(quant_type_out == k_nope_out.scalar_type());

  CHECK_DIM(3, q_rope_in);   // q_rope_in: (nnz, H_Q, 64)
  CHECK_DIM(3, q_nope_in);   // q_nope_in: (nnz, H_Q, 512)
  CHECK_DIM(2, k_rope_in);   // k_rope_in: (nnz, 64)
  CHECK_DIM(2, k_nope_in);   // k_nope_in: (nnz, 512)
  CHECK_DIM(3, q_rope_out);  // q_rope_out: (nnz, H_Q, 64)
  CHECK_DIM(3, q_nope_out);  // q_nope_out: (nnz, H_Q, 512)
  CHECK_DIM(2, k_rope_out);  // k_rope_out: (nnz, 64)
  CHECK_DIM(2, k_nope_out);  // k_nope_out: (nnz, 512)
  uint32_t nnz = q_rope_in.size(0);
  CHECK_EQ(q_nope_in.size(0), nnz);
  CHECK_EQ(k_nope_in.size(0), nnz);
  CHECK_EQ(q_rope_out.size(0), nnz);
  CHECK_EQ(k_rope_out.size(0), nnz);
  CHECK_EQ(q_nope_out.size(0), nnz);
  CHECK_EQ(k_nope_out.size(0), nnz);
  uint32_t num_heads = q_rope_in.size(1);
  CHECK_EQ(q_rope_in.size(1), num_heads);
  CHECK_EQ(q_nope_in.size(1), num_heads);
  CHECK_EQ(q_rope_out.size(1), num_heads);
  CHECK_EQ(q_nope_out.size(1), num_heads);

  const uint32_t q_rope_in_stride_n = q_rope_in.stride(0);
  const uint32_t q_rope_in_stride_h = q_rope_in.stride(1);
  const uint32_t q_nope_in_stride_n = q_nope_in.stride(0);
  const uint32_t q_nope_in_stride_h = q_nope_in.stride(1);
  const uint32_t q_rope_out_stride_n = q_rope_out.stride(0);
  const uint32_t q_rope_out_stride_h = q_rope_out.stride(1);
  const uint32_t q_nope_out_stride_n = q_nope_out.stride(0);
  const uint32_t q_nope_out_stride_h = q_nope_out.stride(1);
  const uint32_t k_rope_in_stride = k_rope_in.stride(0);
  const uint32_t k_nope_in_stride = k_nope_in.stride(0);
  const uint32_t k_rope_out_stride = k_rope_out.stride(0);
  const uint32_t k_nope_out_stride = k_nope_out.stride(0);

  const c10::cuda::OptionalCUDAGuard device_guard(q_rope_in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(scalar_type_in, c_type, [&] {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(quant_type_out, c_quant_type, [&] {
      return DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pos_ids.scalar_type(), c_idtype, [&] {
        cudaError_t status = MLARopeQuantize(
            static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
            static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
            static_cast<c_quant_type*>(q_rope_out.data_ptr()),
            static_cast<c_quant_type*>(k_rope_out.data_ptr()),
            static_cast<c_quant_type*>(q_nope_out.data_ptr()),
            static_cast<c_quant_type*>(k_nope_out.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_heads, q_rope_in_stride_n,
            q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h, q_rope_out_stride_n,
            q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h, k_rope_in_stride,
            k_nope_in_stride, k_rope_out_stride, k_nope_out_stride, quant_scale_q, quant_scale_kv,
            interleave, stream);
        TORCH_CHECK(status == cudaSuccess,
                    "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " +
                        std::string(cudaGetErrorString(status)));
        return true;
      });
    });
  });
}
