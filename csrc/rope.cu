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

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
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

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
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

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
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

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
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

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
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
                   TensorView pos_ids, double quant_scale_q, double quant_scale_kv, bool interleave,
                   bool enable_pdl) {
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

  ffi::CUDADeviceGuard device_guard(q_rope_in.device().device_id);
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
            k_nope_out_stride_h, quant_scale_q, quant_scale_kv, interleave, enable_pdl, stream);

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RopeQuantize failed with error code " << cudaGetErrorString(status);
        return true;
      });
    });
  });
}

/*!
 * TVM FFI binding for fused RoPE + quantization + paged KV cache append kernel
 *
 * Validates tensor shapes, dimensions, and data types, then dispatches to the templated
 * RopeQuantizeAppendPagedKVCache CUDA kernel implementation.
 */
void rope_quantize_append_paged_kv_cache(
    TensorView q_rope_in, TensorView k_rope_in, TensorView q_nope_in, TensorView k_nope_in,
    TensorView v_in, TensorView q_rope_out, TensorView q_nope_out, TensorView cos_sin_cache,
    TensorView pos_ids,
    // Paged cache tensors
    TensorView k_cache, TensorView v_cache, TensorView ckv_cache, TensorView kpe_cache,
    TensorView kv_indices, TensorView kv_indptr, TensorView batch_indices, TensorView positions,
    int64_t kv_layout_code, int64_t page_size, double quant_scale_q, double quant_scale_kv,
    bool interleave, bool enable_pdl) {
  // Validate inputs
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_rope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_nope_in);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_rope_out);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_nope_out);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(positions);

  // Extract dimensions
  uint32_t rope_dim = q_rope_in.size(-1);
  uint32_t no_rope_dim = q_nope_in.size(-1);
  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);

  // Validate dimensions
  TVM_FFI_ICHECK_EQ(k_rope_in.size(-1), rope_dim);
  TVM_FFI_ICHECK_EQ(k_nope_in.size(-1), no_rope_dim);
  TVM_FFI_ICHECK_EQ(q_rope_out.size(-1), rope_dim);
  TVM_FFI_ICHECK_EQ(q_nope_out.size(-1), no_rope_dim);
  TVM_FFI_ICHECK_EQ(q_rope_in.dtype(), k_rope_in.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_in.dtype(), q_nope_in.dtype());
  TVM_FFI_ICHECK_EQ(q_rope_in.dtype(), k_nope_in.dtype());

  // Validate input/output dtypes
  TVM_FFI_ICHECK(q_rope_in.dtype() == dl_float16 || q_rope_in.dtype() == dl_bfloat16)
      << "Input dtype must be float16 or bfloat16";
  TVM_FFI_ICHECK(q_rope_out.dtype() == dl_float8_e4m3fn || q_rope_out.dtype() == dl_float8_e5m2)
      << "Output dtype must be float8_e4m3fn or float8_e5m2";

  // Q tensors are always 3D
  CHECK_DIM(3, q_rope_in);
  CHECK_DIM(3, q_nope_in);
  CHECK_DIM(3, q_rope_out);
  CHECK_DIM(3, q_nope_out);

  // Detect architecture based on cache presence/layout (not K dimensionality)
  QKVLayout kv_layout = QKVLayout(kv_layout_code);
  bool has_mla_caches = (ckv_cache.data_ptr() != nullptr && kpe_cache.data_ptr() != nullptr);
  bool has_gqa_caches = (k_cache.data_ptr() != nullptr && v_cache.data_ptr() != nullptr);
  bool is_mla = has_mla_caches && !has_gqa_caches;
  uint32_t num_kv_heads;
  uint32_t batch_size = kv_indptr.size(0) - 1;

  // Require 3D K tensors in both paths; for MLA head dim must be 1
  CHECK_DIM(3, k_rope_in);
  CHECK_DIM(3, k_nope_in);
  if (is_mla) {
    num_kv_heads = 1;
    TVM_FFI_ICHECK_EQ(k_rope_in.size(1), 1) << "MLA expects K rope head dim == 1";
    TVM_FFI_ICHECK_EQ(k_nope_in.size(1), 1) << "MLA expects K nope head dim == 1";
    // V can be empty for MLA
    TVM_FFI_ICHECK(v_in.data_ptr() == nullptr || v_in.size(0) == 0)
        << "MLA should not have V input (or it should be empty)";
    // Validate MLA cache tensors are provided
    TVM_FFI_ICHECK(ckv_cache.data_ptr() != nullptr && kpe_cache.data_ptr() != nullptr)
        << "MLA requires ckv_cache and kpe_cache";
    CHECK_DIM(3, ckv_cache);  // (max_pages, page_size, ckv_dim)
    CHECK_DIM(3, kpe_cache);  // (max_pages, page_size, kpe_dim)
    TVM_FFI_ICHECK_EQ(ckv_cache.size(2), no_rope_dim);
    TVM_FFI_ICHECK_EQ(kpe_cache.size(2), rope_dim);
  } else {
    // GQA/MHA validation
    num_kv_heads = k_rope_in.size(1);
    TVM_FFI_ICHECK_EQ(k_nope_in.size(1), num_kv_heads);
    // V is required for GQA/MHA
    CHECK_DIM(3, v_in);
    TVM_FFI_ICHECK_EQ(v_in.size(0), nnz);
    TVM_FFI_ICHECK_EQ(v_in.size(1), num_kv_heads);
    // Validate GQA/MHA cache tensors are provided
    TVM_FFI_ICHECK(k_cache.data_ptr() != nullptr && v_cache.data_ptr() != nullptr)
        << "GQA/MHA requires k_cache and v_cache";
    // Cache must be 4D
    CHECK_DIM(4, k_cache);
    CHECK_DIM(4, v_cache);
  }

  // Extract Q strides
  const uint32_t q_rope_in_stride_n = q_rope_in.stride(0);
  const uint32_t q_rope_in_stride_h = q_rope_in.stride(1);
  const uint32_t q_nope_in_stride_n = q_nope_in.stride(0);
  const uint32_t q_nope_in_stride_h = q_nope_in.stride(1);
  const uint32_t q_rope_out_stride_n = q_rope_out.stride(0);
  const uint32_t q_rope_out_stride_h = q_rope_out.stride(1);
  const uint32_t q_nope_out_stride_n = q_nope_out.stride(0);
  const uint32_t q_nope_out_stride_h = q_nope_out.stride(1);

  // Extract K strides
  uint32_t k_rope_in_stride, k_nope_in_stride;
  uint32_t k_rope_in_stride_h, k_nope_in_stride_h;
  uint32_t v_in_stride = 0, v_in_stride_h = 0;

  k_rope_in_stride = k_rope_in.stride(0);
  k_nope_in_stride = k_nope_in.stride(0);
  k_rope_in_stride_h = k_rope_in.stride(1);
  k_nope_in_stride_h = k_nope_in.stride(1);
  if (!is_mla) {
    v_in_stride = v_in.stride(0);
    v_in_stride_h = v_in.stride(1);
  }

  ffi::CUDADeviceGuard device_guard(q_rope_in.device().device_id);
  const cudaStream_t stream = get_stream(q_rope_in.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q_rope_in.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(q_rope_out.dtype(), c_quant_type, [&] {
      cudaError_t status;

      if (is_mla) {
        // MLA: Construct paged_kv_mla_t struct
        auto ckv_strides = ckv_cache.strides();
        auto kpe_strides = kpe_cache.strides();

        paged_kv_mla_t<c_quant_type, int32_t> paged_kv_mla(
            page_size, no_rope_dim, rope_dim, batch_size,
            static_cast<c_quant_type*>(ckv_cache.data_ptr()), ckv_strides.data(),
            static_cast<c_quant_type*>(kpe_cache.data_ptr()), kpe_strides.data(),
            static_cast<int32_t*>(kv_indices.data_ptr()),
            static_cast<int32_t*>(kv_indptr.data_ptr()),
            nullptr  // last_page_len not needed for this kernel
        );

        status = RopeQuantizeAppendPagedMLACache(
            static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
            static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
            static_cast<c_quant_type*>(q_rope_out.data_ptr()),
            static_cast<c_quant_type*>(q_nope_out.data_ptr()), paged_kv_mla,
            static_cast<int32_t*>(batch_indices.data_ptr()),
            static_cast<int32_t*>(positions.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, rope_dim, no_rope_dim,
            q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
            q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
            k_rope_in_stride, k_nope_in_stride, quant_scale_q, quant_scale_kv, interleave,
            enable_pdl, stream);

      } else {
        // GQA/MHA: Construct paged_kv_t struct
        auto k_strides = k_cache.strides();
        auto v_strides = v_cache.strides();
        uint32_t head_dim = rope_dim + no_rope_dim;

        paged_kv_t<c_quant_type, int32_t> paged_kv(
            num_kv_heads, page_size, head_dim, batch_size, kv_layout,
            static_cast<c_quant_type*>(k_cache.data_ptr()),
            static_cast<c_quant_type*>(v_cache.data_ptr()), k_strides.data(),
            static_cast<int32_t*>(kv_indices.data_ptr()),
            static_cast<int32_t*>(kv_indptr.data_ptr()),
            nullptr  // last_page_len not needed for this kernel
        );

        status = RopeQuantizeAppendPagedKVCache(
            static_cast<c_type*>(q_rope_in.data_ptr()), static_cast<c_type*>(k_rope_in.data_ptr()),
            static_cast<c_type*>(q_nope_in.data_ptr()), static_cast<c_type*>(k_nope_in.data_ptr()),
            static_cast<c_type*>(v_in.data_ptr()),
            static_cast<c_quant_type*>(q_rope_out.data_ptr()),
            static_cast<c_quant_type*>(q_nope_out.data_ptr()), paged_kv,
            static_cast<int32_t*>(batch_indices.data_ptr()),
            static_cast<int32_t*>(positions.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rope_dim,
            no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n,
            q_nope_in_stride_h, q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n,
            q_nope_out_stride_h, k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride,
            k_nope_in_stride_h, v_in_stride, v_in_stride_h, quant_scale_q, quant_scale_kv,
            interleave, enable_pdl, stream);
      }

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "RopeQuantizeAppendPagedKVCache failed with error code " << cudaGetErrorString(status);
      return true;
    });
  });
}
