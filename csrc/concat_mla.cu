/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/concat_mla.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

/*!
 * \brief Concatenate k_nope and k_rope tensors for MLA attention
 *
 * Input:
 *   - k_nope: [num_tokens, num_heads, nope_dim]
 *   - k_rope: [num_tokens, 1, rope_dim]
 * Output:
 *   - k: [num_tokens, num_heads, nope_dim + rope_dim]
 *
 * The k_rope values are broadcast to all heads.
 *
 * \param k Output tensor
 * \param k_nope The nope part of k (per-head)
 * \param k_rope The rope part of k (shared across heads)
 */
void concat_mla_k(TensorView k, TensorView k_nope, TensorView k_rope) {
  // Validate inputs
  CHECK_CUDA(k);
  CHECK_CUDA(k_nope);
  CHECK_CUDA(k_rope);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_LAST_DIM_CONTIGUOUS(k_nope);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope);
  CHECK_DEVICE(k, k_nope);
  CHECK_DEVICE(k, k_rope);

  // Check dimensions
  CHECK_DIM(3, k);       // [num_tokens, num_heads, k_head_dim]
  CHECK_DIM(3, k_nope);  // [num_tokens, num_heads, nope_dim]
  CHECK_DIM(3, k_rope);  // [num_tokens, 1, rope_dim]

  int num_tokens = k.size(0);
  int num_heads = k.size(1);
  int k_head_dim = k.size(2);
  int nope_dim = k_nope.size(2);
  int rope_dim = k_rope.size(2);

  // Validate shapes
  TVM_FFI_ICHECK_EQ(k_nope.size(0), num_tokens) << "k and k_nope must have the same num_tokens";
  TVM_FFI_ICHECK_EQ(k_nope.size(1), num_heads) << "k and k_nope must have the same num_heads";
  TVM_FFI_ICHECK_EQ(k_rope.size(0), num_tokens) << "k and k_rope must have the same num_tokens";
  TVM_FFI_ICHECK_EQ(k_rope.size(1), 1) << "k_rope must have num_heads=1 (broadcast)";
  TVM_FFI_ICHECK_EQ(k_head_dim, nope_dim + rope_dim)
      << "k head_dim must equal nope_dim + rope_dim, got " << k_head_dim << " != " << nope_dim
      << " + " << rope_dim;

  // Validate expected dimensions for optimized kernel
  TVM_FFI_ICHECK_EQ(num_heads, MLA_NUM_LOCAL_HEADS) << "num_heads must be 128 for optimized kernel";
  TVM_FFI_ICHECK_EQ(nope_dim, MLA_QK_NOPE_HEAD_DIM) << "nope_dim must be 128 for optimized kernel";
  TVM_FFI_ICHECK_EQ(rope_dim, MLA_QK_ROPE_HEAD_DIM) << "rope_dim must be 64 for optimized kernel";

  // Validate data types
  TVM_FFI_ICHECK(k.dtype() == k_nope.dtype()) << "k and k_nope must have the same dtype";
  TVM_FFI_ICHECK(k.dtype() == k_rope.dtype()) << "k and k_rope must have the same dtype";

  // Get strides
  int64_t k_stride_0 = k.stride(0);
  int k_stride_1 = k.stride(1);
  int64_t k_nope_stride_0 = k_nope.stride(0);
  int k_nope_stride_1 = k_nope.stride(1);
  int64_t k_rope_stride_0 = k_rope.stride(0);

  ffi::CUDADeviceGuard device_guard(k.device().device_id);
  const cudaStream_t stream = get_stream(k.device());

  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k.dtype(), c_type, [&] {
    cudaError_t status = ConcatMLAK<c_type>(
        static_cast<c_type*>(k.data_ptr()), static_cast<c_type*>(k_nope.data_ptr()),
        static_cast<c_type*>(k_rope.data_ptr()), num_tokens, k_stride_0, k_stride_1,
        k_nope_stride_0, k_nope_stride_1, k_rope_stride_0, stream);

    TVM_FFI_ICHECK(status == cudaSuccess)
        << "ConcatMLAK failed with error: " << cudaGetErrorString(status);
    return true;
  });

  TVM_FFI_ICHECK(success) << "concat_mla_k failed to dispatch with dtype " << k.dtype();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(concat_mla_k, concat_mla_k);
