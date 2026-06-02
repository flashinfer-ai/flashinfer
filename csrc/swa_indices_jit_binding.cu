// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// TVM-FFI binding for the SWA paged slot-id + window-length compute kernel
// (sliding-window layers of DSv4-Flash / GLM-5.1; feeds sparse-MLA indices).

#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace flashinfer::swa_indices {

void launch_compute_swa_indices_and_lens(int32_t* swa_indices, int swa_indices_stride,
                                         int32_t* swa_lens, int window_size,
                                         const int32_t* query_start_loc, const int32_t* seq_lens,
                                         const int32_t* token_to_req_indices,
                                         const bool* is_valid_token, const int32_t* block_table,
                                         int block_table_stride, int block_size, int token_offset,
                                         int num_tokens, cudaStream_t stream);

void ComputeSwaIndicesAndLens(TensorView swa_indices, TensorView swa_lens, int64_t window_size,
                              TensorView query_start_loc, TensorView seq_lens,
                              TensorView token_to_req_indices, TensorView is_valid_token,
                              TensorView block_table, int64_t block_size, int64_t token_offset,
                              int64_t num_tokens) {
  TVM_FFI_ICHECK(swa_indices.dtype().code == kDLInt && swa_indices.dtype().bits == 32)
      << "swa_indices must be int32";
  TVM_FFI_ICHECK(swa_indices.ndim() == 2 || (swa_indices.ndim() == 3 && swa_indices.size(1) == 1))
      << "swa_indices must be [N, W] or [N, 1, W]; got ndim=" << swa_indices.ndim();
  TVM_FFI_ICHECK_EQ(swa_indices.stride(swa_indices.ndim() - 1), 1)
      << "swa_indices innermost stride must be 1";
  TVM_FFI_ICHECK_GE(swa_indices.size(swa_indices.ndim() - 1), window_size)
      << "swa_indices last dim must be >= window_size";

  TVM_FFI_ICHECK(swa_lens.dtype().code == kDLInt && swa_lens.dtype().bits == 32)
      << "swa_lens must be int32";
  TVM_FFI_ICHECK_EQ(swa_lens.ndim(), 1) << "swa_lens must be 1-D";
  TVM_FFI_ICHECK_EQ(swa_lens.stride(0), 1) << "swa_lens must be contiguous";

  TVM_FFI_ICHECK(query_start_loc.dtype().code == kDLInt && query_start_loc.dtype().bits == 32)
      << "query_start_loc must be int32";
  TVM_FFI_ICHECK(seq_lens.dtype().code == kDLInt && seq_lens.dtype().bits == 32)
      << "seq_lens must be int32";
  TVM_FFI_ICHECK(token_to_req_indices.dtype().code == kDLInt &&
                 token_to_req_indices.dtype().bits == 32)
      << "token_to_req_indices must be int32";
  TVM_FFI_ICHECK(is_valid_token.dtype().code == kDLBool) << "is_valid_token must be bool";
  TVM_FFI_ICHECK(block_table.dtype().code == kDLInt && block_table.dtype().bits == 32)
      << "block_table must be int32";
  TVM_FFI_ICHECK_EQ(block_table.ndim(), 2) << "block_table must be 2-D";
  TVM_FFI_ICHECK_EQ(block_table.stride(1), 1) << "block_table innermost stride must be 1";

  TVM_FFI_ICHECK_GE(num_tokens, 0);
  if (num_tokens == 0) return;

  TVM_FFI_ICHECK_GT(window_size, 0) << "window_size must be > 0";
  TVM_FFI_ICHECK_GT(block_size, 0) << "block_size must be > 0";
  TVM_FFI_ICHECK_GE(token_offset, 0) << "token_offset must be >= 0";

  // stride(0) is in elements; kernel takes int32-element row strides.
  const int swa_indices_stride = static_cast<int>(swa_indices.stride(0));
  const int block_table_stride = static_cast<int>(block_table.stride(0));

  cudaStream_t stream = get_stream(swa_indices.device());
  launch_compute_swa_indices_and_lens(
      static_cast<int32_t*>(swa_indices.data_ptr()), swa_indices_stride,
      static_cast<int32_t*>(swa_lens.data_ptr()), static_cast<int>(window_size),
      static_cast<const int32_t*>(query_start_loc.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int32_t*>(token_to_req_indices.data_ptr()),
      static_cast<const bool*>(is_valid_token.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()), block_table_stride,
      static_cast<int>(block_size), static_cast<int>(token_offset), static_cast<int>(num_tokens),
      stream);
}

}  // namespace flashinfer::swa_indices

TVM_FFI_DLL_EXPORT_TYPED_FUNC(compute_swa_indices_and_lens,
                              flashinfer::swa_indices::ComputeSwaIndicesAndLens);
