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
#include <cstdint>
#include <flashinfer/fast_topk_clusters_exact.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using namespace flashinfer::sampling;

void fast_topk_clusters_exact(TensorView logits, TensorView indices,
                              Optional<TensorView> output_values, Optional<TensorView> histogram,
                              TensorView cached_overflow, int64_t TopK, int64_t num_cached,
                              int64_t num_clusters, bool pdl_enabled) {
  CHECK_DIM(2, logits);   // input: (batch_size, seq_len)
  CHECK_DIM(2, indices);  // output_indices: (batch_size, top_k)
  const int batch_size = static_cast<int>(logits.size(0));
  const int seq_len = static_cast<int>(logits.size(1));

  int* hist_ptr = nullptr;
  if (histogram.has_value()) {
    hist_ptr = (int*)histogram.value().data_ptr();
  }

  void* values_ptr = nullptr;
  if (output_values.has_value()) {
    values_ptr = (output_values.value().data_ptr());
  }

  const int logit_stride = static_cast<int>(logits.stride(0));
  const int indices_stride = static_cast<int>(indices.stride(0));
  const int n_clusters = static_cast<int>(num_clusters);
  cudaStream_t stream = get_current_stream();
  const int ovf_stride = static_cast<int>(cached_overflow.stride(0)) / (4 * n_clusters);

  auto dtype = logits.dtype();

  auto idx_dtype = indices.dtype();
  TVM_FFI_ICHECK(idx_dtype.code == kDLInt && (idx_dtype.bits == 32 || idx_dtype.bits == 64))
      << "indices must be int32 or int64, got code=" << idx_dtype.code
      << " bits=" << idx_dtype.bits;
  const bool idx_int64 = (idx_dtype.bits == 64);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    if (idx_int64) {
      launch_fast_topk_clusters_exact<c_type, int64_t>(
          static_cast<const c_type*>(logits.data_ptr()), static_cast<int64_t*>(indices.data_ptr()),
          (c_type*)(values_ptr), seq_len, (hist_ptr), static_cast<int*>(cached_overflow.data_ptr()),
          ovf_stride, batch_size, logit_stride, indices_stride, static_cast<int>(num_cached),
          n_clusters, pdl_enabled, static_cast<int>(TopK), stream);
    } else {
      launch_fast_topk_clusters_exact<c_type, int>(
          static_cast<const c_type*>(logits.data_ptr()), static_cast<int*>(indices.data_ptr()),
          (c_type*)(values_ptr), seq_len, (hist_ptr), static_cast<int*>(cached_overflow.data_ptr()),
          ovf_stride, batch_size, logit_stride, indices_stride, static_cast<int>(num_cached),
          n_clusters, pdl_enabled, static_cast<int>(TopK), stream);
    }
    return true;
  });
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "launch_fast_topk_clusters_exact failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_topk_clusters_exact, fast_topk_clusters_exact);

void fast_topk_clusters_exact_page_table_transform(TensorView logits, TensorView indices,
                                                   TensorView seq_lens, TensorView page_table,
                                                   Optional<TensorView> histogram,
                                                   TensorView cached_overflow, int64_t TopK,
                                                   int64_t num_cached, int64_t num_clusters,
                                                   bool pdl_enabled) {
  CHECK_DIM(2, logits);
  CHECK_DIM(2, indices);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(2, page_table);
  const int batch_size = static_cast<int>(logits.size(0));

  const int* hist_ptr = nullptr;
  if (histogram.has_value()) {
    hist_ptr = static_cast<const int*>(histogram.value().data_ptr());
  }

  const int logit_stride = static_cast<int>(logits.stride(0));
  const int indices_stride = static_cast<int>(indices.stride(0));
  const int page_table_stride = static_cast<int>(page_table.stride(0));
  const int n_clusters = static_cast<int>(num_clusters);
  const int ovf_stride = static_cast<int>(cached_overflow.stride(0)) / (4 * n_clusters);
  cudaStream_t stream = get_current_stream();

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(logits.dtype(), c_type, [&] {
    launch_fast_topk_clusters_exact_page_table_transform<c_type>(
        static_cast<const c_type*>(logits.data_ptr()), static_cast<int*>(indices.data_ptr()),
        static_cast<int*>(seq_lens.data_ptr()), static_cast<int*>(page_table.data_ptr()),
        const_cast<int*>(hist_ptr), static_cast<int*>(cached_overflow.data_ptr()), ovf_stride,
        batch_size, logit_stride, indices_stride, page_table_stride, static_cast<int>(num_cached),
        n_clusters, pdl_enabled, static_cast<int>(TopK), stream);
    return true;
  });
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "launch_fast_topk_clusters_exact_page_table_transform failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_topk_clusters_exact_page_table_transform,
                              fast_topk_clusters_exact_page_table_transform);

void fast_topk_clusters_exact_ragged_transform(TensorView logits, TensorView indices,
                                               TensorView seq_lens, TensorView offsets,
                                               Optional<TensorView> histogram,
                                               TensorView cached_overflow, int64_t TopK,
                                               int64_t num_cached, int64_t num_clusters,
                                               bool pdl_enabled) {
  CHECK_DIM(2, logits);
  CHECK_DIM(2, indices);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, offsets);
  const int batch_size = static_cast<int>(logits.size(0));

  const int* hist_ptr = nullptr;
  if (histogram.has_value()) {
    hist_ptr = static_cast<const int*>(histogram.value().data_ptr());
  }

  const int logit_stride = static_cast<int>(logits.stride(0));
  const int indices_stride = static_cast<int>(indices.stride(0));
  const int n_clusters = static_cast<int>(num_clusters);
  const int ovf_stride = static_cast<int>(cached_overflow.stride(0)) / (4 * n_clusters);
  cudaStream_t stream = get_current_stream();

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(logits.dtype(), c_type, [&] {
    launch_fast_topk_clusters_exact_ragged_transform<c_type>(
        static_cast<const c_type*>(logits.data_ptr()), static_cast<int*>(indices.data_ptr()),
        static_cast<int*>(seq_lens.data_ptr()), static_cast<int*>(offsets.data_ptr()),
        const_cast<int*>(hist_ptr), static_cast<int*>(cached_overflow.data_ptr()), ovf_stride,
        batch_size, logit_stride, indices_stride, static_cast<int>(num_cached), n_clusters,
        pdl_enabled, static_cast<int>(TopK), stream);
    return true;
  });
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "launch_fast_topk_clusters_exact_ragged_transform failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_topk_clusters_exact_ragged_transform,
                              fast_topk_clusters_exact_ragged_transform);
