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

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Tensor;

void apply_rope(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope, TensorView indptr,
                TensorView offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  CHECK_DEVICE(q, k);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  TVM_FFI_ICHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  TVM_FFI_ICHECK_EQ(indptr.dtype(), offsets.dtype());

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(indptr.dtype(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotary(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(indptr.data_ptr()), static_cast<c_idtype*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
          k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n,
          k_rope_stride_h, interleave, rope_scale, rope_theta, stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "BatchQKApplyRotary failed with error code " << cudaGetErrorString(status);
      return true;
    });
  });
}

void apply_rope_pos_ids(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope,
                        TensorView pos_ids, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_INPUT(pos_ids);

  CHECK_DEVICE(q, k);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
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

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(pos_ids.dtype(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotaryPosIds(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
          head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n,
          q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta,
          stream);

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "BatchQKApplyRotaryPosIds failed with error code " << cudaGetErrorString(status);
      return true;
    });
  });
}

void apply_rope_pos_ids_cos_sin_cache(TensorView q, TensorView k, TensorView q_rope,
                                      TensorView k_rope, TensorView cos_sin_cache,
                                      TensorView pos_ids, bool interleave) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, cos_sin_cache);
  CHECK_DEVICE(q, pos_ids);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  // cos_sin_cache: (max_seq_len, R)
  // First half of R is cos, second half is sin
  CHECK_DIM(2, cos_sin_cache);
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
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

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(pos_ids.dtype(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()), static_cast<c_idtype*>(pos_ids.data_ptr()),
          nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
          k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
          interleave, stream);

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "BatchQKApplyRotaryPosIdsCosSinCache failed with error code "
          << cudaGetErrorString(status);
      return true;
    });
  });
}

void apply_llama31_rope(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope,
                        TensorView indptr, TensorView offsets, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  CHECK_DEVICE(q, k);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  TVM_FFI_ICHECK_EQ(indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(indptr.dtype(), offsets.dtype());
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  TVM_FFI_ICHECK_EQ(indptr.dtype(), offsets.dtype());

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(indptr.dtype(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyLlama31Rotary(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(indptr.data_ptr()), static_cast<c_idtype*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
          k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n,
          k_rope_stride_h, interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor,
          old_context_length, stream);

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "BatchQKApplyLlama31Rotary failed with error code " << cudaGetErrorString(status);
      return true;
    });
  });
}

void apply_llama31_rope_pos_ids(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope,
                                TensorView pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(pos_ids);

  CHECK_DEVICE(q, k);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK_EQ(q.size(2), k.size(2));
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

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(pos_ids.dtype(), c_idtype, [&] {
      cudaError_t status = BatchQKApplyLlama31RotaryPosIds(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
          head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n,
          q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta,
          low_freq_factor, high_freq_factor, old_context_length, stream);

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "BatchQKApplyLlama31RotaryPosIds failed with error code "
          << cudaGetErrorString(status);
      return true;
    });
  });
}

/*!
 * TVM FFI binding for RoPE + quantization kernel
 *
 * Validates tensor shapes, dimensions, and data types, then dispatches to the templated
 * RopeQuantize CUDA kernel implementation.
 */
void rope_quantize(TensorView q_rope_in, TensorView k_rope_in, TensorView q_nope_in,
                   TensorView k_nope_in, TensorView q_rope_out, TensorView k_rope_out,
                   TensorView q_nope_out, TensorView k_nope_out, TensorView cos_sin_cache,
                   TensorView pos_ids, double quant_scale_q, double quant_scale_kv,
                   bool interleave) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_nope_out);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_nope_out);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);

  // Extract dimensions from tensor shapes (flexible)
  uint32_t rope_dim = q_rope_in.size(-1);
  uint32_t no_rope_dim = q_nope_in.size(-1);

  // Validate rope and no_rope dimensions are consistent
  TVM_FFI_ICHECK_EQ(k_rope_in.size(-1), rope_dim);
  TVM_FFI_ICHECK_EQ(k_nope_in.size(-1), no_rope_dim);
  TVM_FFI_ICHECK_EQ(q_rope_out.size(-1), rope_dim);
  TVM_FFI_ICHECK_EQ(k_rope_out.size(-1), rope_dim);
  TVM_FFI_ICHECK_EQ(q_nope_out.size(-1), no_rope_dim);
  TVM_FFI_ICHECK_EQ(k_nope_out.size(-1), no_rope_dim);
  TVM_FFI_ICHECK_EQ(q_rope_in.dtype(), k_rope_in.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_in.dtype(), q_nope_in.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_in.dtype(), k_nope_in.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_out.dtype(), k_rope_out.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_out.dtype(), q_nope_out.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_out.dtype(), k_nope_out.dtype());

  // Validate supported input data types (float16 or bfloat16)
  TVM_FFI_ICHECK(q_rope_in.dtype() == dl_float16 || q_rope_in.dtype() == dl_bfloat16)
      << "Input dtype must be float16 or bfloat16";

  // Validate supported output quantization data types (float8_e4m3fn or float8_e5m2)
  TVM_FFI_ICHECK(q_rope_out.dtype() == dl_float8_e4m3fn || q_rope_out.dtype() == dl_float8_e5m2)
      << "Output dtype must be float8_e4m3fn or float8_e5m2";

  // Q tensors are always 3D: (nnz, num_qo_heads, rope_dim/no_rope_dim)
  CHECK_DIM(3, q_rope_in);
  CHECK_DIM(3, q_nope_in);
  CHECK_DIM(3, q_rope_out);
  CHECK_DIM(3, q_nope_out);

  // K tensors can be 2D (MLA) or 3D (GQA/MHA)
  uint32_t num_kv_heads;
  if (k_rope_in.ndim() == 2) {
    // MLA case: k_rope_in: (nnz, rope_dim), k_nope_in: (nnz, no_rope_dim)
    CHECK_DIM(2, k_rope_in);
    CHECK_DIM(2, k_nope_in);
    CHECK_DIM(2, k_rope_out);
    CHECK_DIM(2, k_nope_out);
    num_kv_heads = 1;  // Shared K/V head
  } else {
    // GQA/MHA case: k_rope_in: (nnz, num_kv_heads, rope_dim)
    CHECK_DIM(3, k_rope_in);
    CHECK_DIM(3, k_nope_in);
    CHECK_DIM(3, k_rope_out);
    CHECK_DIM(3, k_nope_out);
    num_kv_heads = k_rope_in.size(1);
  }
  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);

  // Validate consistent dimensions across all tensors
  TVM_FFI_ICHECK_EQ(q_nope_in.size(0), nnz);
  TVM_FFI_ICHECK_EQ(k_rope_in.size(0), nnz);
  TVM_FFI_ICHECK_EQ(k_nope_in.size(0), nnz);
  TVM_FFI_ICHECK_EQ(q_rope_out.size(0), nnz);
  TVM_FFI_ICHECK_EQ(k_rope_out.size(0), nnz);
  TVM_FFI_ICHECK_EQ(q_nope_out.size(0), nnz);
  TVM_FFI_ICHECK_EQ(k_nope_out.size(0), nnz);

  // Validate Q tensor head dimensions are consistent
  TVM_FFI_ICHECK_EQ(q_nope_in.size(1), num_qo_heads);
  TVM_FFI_ICHECK_EQ(q_rope_out.size(1), num_qo_heads);
  TVM_FFI_ICHECK_EQ(q_nope_out.size(1), num_qo_heads);

  // Validate K tensor head dimensions (if 3D)
  if (k_rope_in.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(k_nope_in.size(1), num_kv_heads);
    TVM_FFI_ICHECK_EQ(k_rope_out.size(1), num_kv_heads);
    TVM_FFI_ICHECK_EQ(k_nope_out.size(1), num_kv_heads);
  }

  const uint32_t q_rope_in_stride_n = q_rope_in.stride(0);
  const uint32_t q_rope_in_stride_h = q_rope_in.stride(1);
  const uint32_t q_nope_in_stride_n = q_nope_in.stride(0);
  const uint32_t q_nope_in_stride_h = q_nope_in.stride(1);
  const uint32_t q_rope_out_stride_n = q_rope_out.stride(0);
  const uint32_t q_rope_out_stride_h = q_rope_out.stride(1);
  const uint32_t q_nope_out_stride_n = q_nope_out.stride(0);
  const uint32_t q_nope_out_stride_h = q_nope_out.stride(1);

  // K tensor strides depend on dimensionality
  uint32_t k_rope_in_stride, k_nope_in_stride, k_rope_out_stride, k_nope_out_stride;
  uint32_t k_rope_in_stride_h, k_nope_in_stride_h, k_rope_out_stride_h, k_nope_out_stride_h;

  if (k_rope_in.ndim() == 2) {
    // 2D K tensors (MLA): only have batch stride
    k_rope_in_stride = k_rope_in.stride(0);
    k_nope_in_stride = k_nope_in.stride(0);
    k_rope_out_stride = k_rope_out.stride(0);
    k_nope_out_stride = k_nope_out.stride(0);
    // For 2D tensors, head stride is the same as batch stride (shared K/V)
    k_rope_in_stride_h = k_rope_in_stride;
    k_nope_in_stride_h = k_nope_in_stride;
    k_rope_out_stride_h = k_rope_out_stride;
    k_nope_out_stride_h = k_nope_out_stride;
  } else {
    // 3D K tensors (GQA/MHA): have both batch and head strides
    k_rope_in_stride = k_rope_in.stride(0);
    k_rope_in_stride_h = k_rope_in.stride(1);
    k_nope_in_stride = k_nope_in.stride(0);
    k_nope_in_stride_h = k_nope_in.stride(1);
    k_rope_out_stride = k_rope_out.stride(0);
    k_rope_out_stride_h = k_rope_out.stride(1);
    k_nope_out_stride = k_nope_out.stride(0);
    k_nope_out_stride_h = k_nope_out.stride(1);
  }

  cudaSetDevice(q_rope_in.device().device_id);
  const cudaStream_t stream = get_stream(q_rope_in.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q_rope_in.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(q_rope_out.dtype(), c_quant_type, [&] {
      return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(pos_ids.dtype(), c_idtype, [&] {
        cudaError_t status = RopeQuantize(
            static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
            static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
            static_cast<c_quant_type*>(q_rope_out.data_ptr()),
            static_cast<c_quant_type*>(k_rope_out.data_ptr()),
            static_cast<c_quant_type*>(q_nope_out.data_ptr()),
            static_cast<c_quant_type*>(k_nope_out.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<c_idtype*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rope_dim,
            no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n,
            q_nope_in_stride_h, q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n,
            q_nope_out_stride_h, k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride,
            k_nope_in_stride_h, k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride,
            k_nope_out_stride_h, quant_scale_q, quant_scale_kv, interleave, stream);

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RopeQuantize failed with error code " << cudaGetErrorString(status);
        return true;
      });
    });
  });
}
