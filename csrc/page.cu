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
#include <flashinfer/page.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Tensor;

void append_paged_kv_cache(TensorView append_key, TensorView append_value, TensorView batch_indices,
                           TensorView positions, TensorView paged_k_cache, TensorView paged_v_cache,
                           TensorView kv_indices, TensorView kv_indptr, TensorView kv_last_page_len,
                           int64_t layout) {
  CHECK_LAST_DIM_CONTIGUOUS(append_key);
  CHECK_LAST_DIM_CONTIGUOUS(append_value);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(positions);
  // NOTE(Zihao): doesn't have to be contiguous
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_v_cache);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_last_page_len);
  CHECK_DIM(3, append_key);
  CHECK_DIM(3, append_value);
  CHECK_DIM(1, batch_indices);
  CHECK_DIM(1, positions);
  CHECK_DIM(4, paged_k_cache);
  CHECK_DIM(4, paged_v_cache);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_last_page_len);
  unsigned int nnz = append_key.size(0);
  unsigned int batch_size = kv_last_page_len.size(0);
  TVM_FFI_ICHECK_EQ(kv_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(batch_indices.size(0), nnz);
  TVM_FFI_ICHECK_EQ(positions.size(0), nnz);
  CHECK_DEVICE(append_key, append_key);
  CHECK_DEVICE(append_value, append_key);
  CHECK_DEVICE(paged_k_cache, append_key);
  CHECK_DEVICE(paged_v_cache, append_key);
  CHECK_DEVICE(kv_indices, append_key);
  CHECK_DEVICE(kv_indptr, append_key);
  CHECK_DEVICE(kv_last_page_len, append_key);

  QKVLayout kv_layout = QKVLayout(layout);

  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_k_cache.size(3);
  if (kv_layout == QKVLayout::kHND) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  // get kv_cache_strides
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  auto k_dim = paged_k_cache.ndim();
  TVM_FFI_ICHECK(std::equal(k_strides.begin(), k_strides.begin() + k_dim, v_strides.begin()))
      << "k/v strides must be identical";

  auto append_k_strides = append_key.strides();
  auto append_k_stride_n = append_k_strides[0];
  auto append_k_stride_h = append_k_strides[1];
  auto append_v_strides = append_value.strides();
  auto append_v_stride_n = append_v_strides[0];
  auto append_v_stride_h = append_v_strides[1];

  TVM_FFI_ICHECK_EQ(append_key.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(append_key.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(append_value.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(append_value.size(2), head_dim);

  ffi::CUDADeviceGuard device_guard(append_key.device().device_id);
  const cudaStream_t stream = get_stream(append_key.device());
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE(paged_k_cache.dtype(), c_type, [&] {
    paged_kv_t<c_type, int32_t> paged_kv(
        num_heads, page_size, head_dim, batch_size, kv_layout,
        static_cast<c_type*>(paged_k_cache.data_ptr()),
        static_cast<c_type*>(paged_v_cache.data_ptr()), k_strides.data(),
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        static_cast<int32_t*>(kv_last_page_len.data_ptr()));
    cudaError_t status =
        AppendPagedKVCache(paged_kv, static_cast<c_type*>(append_key.data_ptr()),
                           static_cast<c_type*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), nnz, append_k_stride_n,
                           append_k_stride_h, append_v_stride_n, append_v_stride_h, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "AppendPagedKVCache failed with error: " << cudaGetErrorString(status);
    return true;
  });

  TVM_FFI_ICHECK(success) << "AppendPagedKVCache failed to dispatch with dtype "
                          << paged_k_cache.dtype();
}

void append_paged_mla_kv_cache(TensorView append_ckv, TensorView append_kpe,
                               TensorView batch_indices, TensorView positions, TensorView ckv_cache,
                               TensorView kpe_cache, TensorView kv_indices, TensorView kv_indptr,
                               TensorView kv_last_page_len) {
  CHECK_LAST_DIM_CONTIGUOUS(append_ckv);
  CHECK_LAST_DIM_CONTIGUOUS(append_kpe);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(positions);
  // NOTE(Zihao): doesn't have to be contiguous
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(ckv_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(kpe_cache);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_last_page_len);
  CHECK_DIM(2, append_ckv);
  CHECK_DIM(2, append_kpe);
  CHECK_DIM(1, batch_indices);
  CHECK_DIM(1, positions);
  CHECK_DIM(3, ckv_cache);
  CHECK_DIM(3, kpe_cache);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_last_page_len);
  unsigned int nnz = append_ckv.size(0);
  unsigned int batch_size = kv_last_page_len.size(0);
  TVM_FFI_ICHECK_EQ(kv_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(batch_indices.size(0), nnz);
  TVM_FFI_ICHECK_EQ(positions.size(0), nnz);
  CHECK_DEVICE(append_ckv, append_ckv);
  CHECK_DEVICE(append_kpe, append_ckv);
  CHECK_DEVICE(ckv_cache, append_ckv);

  CHECK_DEVICE(kv_indices, append_ckv);
  CHECK_DEVICE(kv_indptr, append_ckv);
  CHECK_DEVICE(kv_last_page_len, append_ckv);

  unsigned int page_size, ckv_dim, kpe_dim;
  page_size = ckv_cache.size(1);
  ckv_dim = ckv_cache.size(2);
  kpe_dim = kpe_cache.size(2);

  // get kv_cache_strides
  auto ckv_strides = ckv_cache.strides();
  auto kpe_strides = kpe_cache.strides();

  auto append_ckv_strides = append_ckv.strides();
  auto append_ckv_stride_n = append_ckv_strides[0];
  auto append_kpe_strides = append_kpe.strides();
  auto append_kpe_stride_n = append_kpe_strides[0];

  TVM_FFI_ICHECK_EQ(append_ckv.size(1), ckv_dim);
  TVM_FFI_ICHECK_EQ(append_kpe.size(1), kpe_dim);

  ffi::CUDADeviceGuard device_guard(append_ckv.device().device_id);
  const cudaStream_t stream = get_stream(append_ckv.device());
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE(ckv_cache.dtype(), c_type, [&] {
    paged_kv_mla_t<c_type, int32_t> paged_mla_kv(
        page_size, ckv_dim, kpe_dim, batch_size, static_cast<c_type*>(ckv_cache.data_ptr()),
        ckv_strides.data(), static_cast<c_type*>(kpe_cache.data_ptr()), kpe_strides.data(),
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        static_cast<int32_t*>(kv_last_page_len.data_ptr()));
    cudaError_t status =
        AppendPagedKVMlaCache(paged_mla_kv, static_cast<c_type*>(append_ckv.data_ptr()),
                              static_cast<c_type*>(append_kpe.data_ptr()),
                              static_cast<int32_t*>(batch_indices.data_ptr()),
                              static_cast<int32_t*>(positions.data_ptr()), nnz, append_ckv_stride_n,
                              append_kpe_stride_n, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "AppendPagedKVMlaCache failed with error: " << cudaGetErrorString(status);
    return true;
  });

  TVM_FFI_ICHECK(success) << "AppendPagedKVMlaCache failed to dispatch with dtype "
                          << ckv_cache.dtype();
}
