/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "flashinfer/attention/msa_decode_metadata.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer {

void PrepareMSADecode(TensorView topk, TensorView block_table, TensorView seq_lens,
                      TensorView k_scale, TensorView sparse_pages, TensorView sparse_lens,
                      TensorView qk_scale_log2, int64_t query_len, int64_t num_physical_pages,
                      double sm_scale) {
  for (const auto& tensor : {topk, block_table, seq_lens, sparse_pages, sparse_lens}) {
    CHECK_CUDA(tensor);
    TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_int32);
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, topk.device().device_id);
  }
  CHECK_CONTIGUOUS(sparse_pages);
  CHECK_CONTIGUOUS(sparse_lens);
  for (const auto& tensor : {k_scale, qk_scale_log2}) {
    CHECK_INPUT(tensor);
    TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(tensor.numel(), 1);
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, topk.device().device_id);
  }
  TVM_FFI_ICHECK_EQ(topk.ndim(), 3);
  TVM_FFI_ICHECK_EQ(topk.size(2), 16);
  TVM_FFI_ICHECK_EQ(block_table.ndim(), 2);
  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_GT(query_len, 0);
  TVM_FFI_ICHECK_EQ(topk.size(1), seq_lens.numel() * query_len);
  TVM_FFI_ICHECK_EQ(block_table.size(0), seq_lens.numel());
  TVM_FFI_ICHECK_EQ(sparse_pages.numel(), topk.numel());
  TVM_FFI_ICHECK_EQ(sparse_lens.numel(), topk.size(0) * topk.size(1));
  const int rows = topk.size(0) * topk.size(1);
  PrepareMSADecodeMetadata<<<(rows + 3) / 4, 128, 0, get_stream(topk.device())>>>(
      static_cast<const int32_t*>(topk.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const float*>(k_scale.data_ptr()), static_cast<int32_t*>(sparse_pages.data_ptr()),
      static_cast<int32_t*>(sparse_lens.data_ptr()), static_cast<float*>(qk_scale_log2.data_ptr()),
      topk.size(1), topk.size(0), query_len, block_table.size(1), num_physical_pages, sm_scale,
      topk.stride(0), topk.stride(1), topk.stride(2), block_table.stride(0), block_table.stride(1),
      seq_lens.stride(0));
  const auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, flashinfer::PrepareMSADecode);
