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
#include <cmath>
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

void nvfp4_quantize_append_paged_kv_cache(TensorView append_key, TensorView append_value,
                                          TensorView batch_indices, TensorView positions,
                                          TensorView paged_k_cache, TensorView paged_v_cache,
                                          TensorView k_scale_cache, TensorView v_scale_cache,
                                          TensorView kv_indices, TensorView kv_indptr,
                                          TensorView kv_last_page_len, double k_scale,
                                          double v_scale, int64_t layout) {
  CHECK_LAST_DIM_CONTIGUOUS(append_key);
  CHECK_LAST_DIM_CONTIGUOUS(append_value);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(positions);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_v_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_scale_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_scale_cache);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_last_page_len);
  CHECK_DIM(3, append_key);
  CHECK_DIM(3, append_value);
  CHECK_DIM(1, batch_indices);
  CHECK_DIM(1, positions);
  CHECK_DIM(4, paged_k_cache);
  CHECK_DIM(4, paged_v_cache);
  CHECK_DIM(4, k_scale_cache);
  CHECK_DIM(4, v_scale_cache);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_last_page_len);
  CHECK_DEVICE(append_key, append_key);
  CHECK_DEVICE(append_value, append_key);
  CHECK_DEVICE(paged_k_cache, append_key);
  CHECK_DEVICE(paged_v_cache, append_key);
  CHECK_DEVICE(k_scale_cache, append_key);
  CHECK_DEVICE(v_scale_cache, append_key);
  CHECK_DEVICE(batch_indices, append_key);
  CHECK_DEVICE(positions, append_key);
  CHECK_DEVICE(kv_indices, append_key);
  CHECK_DEVICE(kv_indptr, append_key);
  CHECK_DEVICE(kv_last_page_len, append_key);

  TVM_FFI_ICHECK(append_key.dtype() == dl_float16 || append_key.dtype() == dl_bfloat16)
      << "append_key must be float16 or bfloat16";
  TVM_FFI_ICHECK(append_value.dtype() == append_key.dtype())
      << "append_key and append_value must have the same dtype";
  TVM_FFI_ICHECK(paged_k_cache.dtype() == dl_uint8 && paged_v_cache.dtype() == dl_uint8)
      << "paged_k_cache and paged_v_cache must be uint8 packed NVFP4 tensors";
  TVM_FFI_ICHECK(k_scale_cache.dtype() == dl_float8_e4m3fn &&
                 v_scale_cache.dtype() == dl_float8_e4m3fn)
      << "k_scale_cache and v_scale_cache must be float8_e4m3fn tensors";
  TVM_FFI_ICHECK(batch_indices.dtype() == dl_int32) << "batch_indices must be int32";
  TVM_FFI_ICHECK(positions.dtype() == dl_int32) << "positions must be int32";
  TVM_FFI_ICHECK(kv_indices.dtype() == dl_int32) << "kv_indices must be int32";
  TVM_FFI_ICHECK(kv_indptr.dtype() == dl_int32) << "kv_indptr must be int32";
  TVM_FFI_ICHECK(kv_last_page_len.dtype() == dl_int32) << "kv_last_page_len must be int32";
  TVM_FFI_ICHECK(std::isfinite(k_scale) && std::isfinite(v_scale) && k_scale > 0.0 && v_scale > 0.0)
      << "k_scale and v_scale must be positive finite global decode scales";

  const unsigned int nnz = append_key.size(0);
  const unsigned int batch_size = kv_last_page_len.size(0);
  TVM_FFI_ICHECK_EQ(kv_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(batch_indices.size(0), nnz);
  TVM_FFI_ICHECK_EQ(positions.size(0), nnz);

  QKVLayout kv_layout = QKVLayout(layout);
  unsigned int num_heads, page_size;
  const unsigned int packed_head_dim = paged_k_cache.size(3);
  const unsigned int scale_dim = k_scale_cache.size(3);
  const unsigned int head_dim = append_key.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  TVM_FFI_ICHECK_EQ(append_key.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(append_value.size(0), nnz);
  TVM_FFI_ICHECK_EQ(append_value.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(append_value.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(packed_head_dim * 2, head_dim);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, head_dim);
  TVM_FFI_ICHECK_EQ(head_dim % 16, 0);

  auto require_same_shape = [](TensorView lhs, TensorView rhs, const char* name) {
    TVM_FFI_ICHECK_EQ(lhs.ndim(), rhs.ndim()) << name << " ndim mismatch";
    for (int i = 0; i < lhs.ndim(); ++i) {
      TVM_FFI_ICHECK_EQ(lhs.size(i), rhs.size(i)) << name << " shape mismatch at dim " << i;
    }
  };
  require_same_shape(paged_k_cache, paged_v_cache, "paged K/V cache");
  require_same_shape(k_scale_cache, v_scale_cache, "K/V scale cache");
  TVM_FFI_ICHECK_EQ(k_scale_cache.size(0), paged_k_cache.size(0));
  if (kv_layout == QKVLayout::kHND) {
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(1), num_heads);
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(2), page_size);
  } else {
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(1), page_size);
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(2), num_heads);
  }

  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  auto k_dim = paged_k_cache.ndim();
  TVM_FFI_ICHECK(std::equal(k_strides.begin(), k_strides.begin() + k_dim, v_strides.begin()))
      << "packed k/v strides must be identical";
  auto k_sf_strides = k_scale_cache.strides();
  auto v_sf_strides = v_scale_cache.strides();
  auto sf_dim = k_scale_cache.ndim();
  TVM_FFI_ICHECK(
      std::equal(k_sf_strides.begin(), k_sf_strides.begin() + sf_dim, v_sf_strides.begin()))
      << "scale k/v strides must be identical";

  const size_t k_sf_stride_page = k_sf_strides[0];
  const size_t k_sf_stride_n = kv_layout == QKVLayout::kHND ? k_sf_strides[2] : k_sf_strides[1];
  const size_t k_sf_stride_h = kv_layout == QKVLayout::kHND ? k_sf_strides[1] : k_sf_strides[2];
  const size_t v_sf_stride_page = v_sf_strides[0];
  const size_t v_sf_stride_n = kv_layout == QKVLayout::kHND ? v_sf_strides[2] : v_sf_strides[1];
  const size_t v_sf_stride_h = kv_layout == QKVLayout::kHND ? v_sf_strides[1] : v_sf_strides[2];
  auto append_k_strides = append_key.strides();
  auto append_v_strides = append_value.strides();
  const size_t append_k_stride_n = append_k_strides[0];
  const size_t append_k_stride_h = append_k_strides[1];
  const size_t append_v_stride_n = append_v_strides[0];
  const size_t append_v_stride_h = append_v_strides[1];

  ffi::CUDADeviceGuard device_guard(append_key.device().device_id);
  const cudaStream_t stream = get_stream(append_key.device());
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(append_key.dtype(), c_type, [&] {
    paged_kv_t<uint8_t, int32_t> paged_kv(
        num_heads, page_size, packed_head_dim, batch_size, kv_layout,
        static_cast<uint8_t*>(paged_k_cache.data_ptr()),
        static_cast<uint8_t*>(paged_v_cache.data_ptr()), k_strides.data(),
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        static_cast<int32_t*>(kv_last_page_len.data_ptr()));
    cudaError_t status = NVFP4QuantizeAppendPagedKVCache(
        paged_kv, static_cast<c_type*>(append_key.data_ptr()),
        static_cast<c_type*>(append_value.data_ptr()),
        static_cast<int32_t*>(batch_indices.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()), nnz, append_k_stride_n, append_k_stride_h,
        append_v_stride_n, append_v_stride_h, static_cast<uint8_t*>(k_scale_cache.data_ptr()),
        static_cast<uint8_t*>(v_scale_cache.data_ptr()), k_sf_stride_page, k_sf_stride_n,
        k_sf_stride_h, v_sf_stride_page, v_sf_stride_n, v_sf_stride_h, static_cast<float>(k_scale),
        static_cast<float>(v_scale), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "NVFP4QuantizeAppendPagedKVCache failed with error: " << cudaGetErrorString(status);
    return true;
  });

  TVM_FFI_ICHECK(success) << "NVFP4QuantizeAppendPagedKVCache failed to dispatch with dtype "
                          << append_key.dtype();
}

void nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
    TensorView append_key, TensorView append_value, TensorView slot_mapping,
    TensorView paged_k_cache, TensorView paged_v_cache, TensorView k_scale_cache,
    TensorView v_scale_cache, TensorView k_scale, TensorView v_scale, int64_t layout) {
  CHECK_LAST_DIM_CONTIGUOUS(append_key);
  CHECK_LAST_DIM_CONTIGUOUS(append_value);
  CHECK_INPUT(slot_mapping);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_v_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_scale_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_scale_cache);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale);
  CHECK_DIM(3, append_key);
  CHECK_DIM(3, append_value);
  CHECK_DIM(1, slot_mapping);
  CHECK_DIM(4, paged_k_cache);
  CHECK_DIM(4, paged_v_cache);
  CHECK_DIM(4, k_scale_cache);
  CHECK_DIM(4, v_scale_cache);
  CHECK_DEVICE(append_key, append_key);
  CHECK_DEVICE(append_value, append_key);
  CHECK_DEVICE(slot_mapping, append_key);
  CHECK_DEVICE(paged_k_cache, append_key);
  CHECK_DEVICE(paged_v_cache, append_key);
  CHECK_DEVICE(k_scale_cache, append_key);
  CHECK_DEVICE(v_scale_cache, append_key);
  CHECK_DEVICE(k_scale, append_key);
  CHECK_DEVICE(v_scale, append_key);

  TVM_FFI_ICHECK(append_key.dtype() == dl_float16 || append_key.dtype() == dl_bfloat16)
      << "append_key must be float16 or bfloat16";
  TVM_FFI_ICHECK(append_value.dtype() == append_key.dtype())
      << "append_key and append_value must have the same dtype";
  TVM_FFI_ICHECK(slot_mapping.dtype() == dl_int32 || slot_mapping.dtype() == dl_int64)
      << "slot_mapping must be int32 or int64";
  TVM_FFI_ICHECK(paged_k_cache.dtype() == dl_uint8 && paged_v_cache.dtype() == dl_uint8)
      << "paged_k_cache and paged_v_cache must be uint8 packed NVFP4 tensors";
  TVM_FFI_ICHECK(k_scale_cache.dtype() == dl_float8_e4m3fn &&
                 v_scale_cache.dtype() == dl_float8_e4m3fn)
      << "k_scale_cache and v_scale_cache must be float8_e4m3fn tensors";
  TVM_FFI_ICHECK(k_scale.dtype() == dl_float32 && v_scale.dtype() == dl_float32)
      << "k_scale and v_scale must be float32 tensors";
  TVM_FFI_ICHECK_EQ(k_scale.numel(), 1) << "k_scale must be a single-element tensor";
  TVM_FFI_ICHECK_EQ(v_scale.numel(), 1) << "v_scale must be a single-element tensor";

  ffi::CUDADeviceGuard device_guard(append_key.device().device_id);
  const cudaStream_t stream = get_stream(append_key.device());

  const unsigned int nnz = slot_mapping.size(0);
  TVM_FFI_ICHECK_GE(append_key.size(0), nnz);
  TVM_FFI_ICHECK_GE(append_value.size(0), nnz);

  QKVLayout kv_layout = QKVLayout(layout);
  unsigned int num_heads, page_size;
  const unsigned int packed_head_dim = paged_k_cache.size(3);
  const unsigned int scale_dim = k_scale_cache.size(3);
  const unsigned int head_dim = append_key.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  TVM_FFI_ICHECK_EQ(append_key.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(append_value.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(append_value.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(packed_head_dim * 2, head_dim);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, head_dim);
  TVM_FFI_ICHECK_EQ(head_dim % 16, 0);

  auto require_same_shape = [](TensorView lhs, TensorView rhs, const char* name) {
    TVM_FFI_ICHECK_EQ(lhs.ndim(), rhs.ndim()) << name << " ndim mismatch";
    for (int i = 0; i < lhs.ndim(); ++i) {
      TVM_FFI_ICHECK_EQ(lhs.size(i), rhs.size(i)) << name << " shape mismatch at dim " << i;
    }
  };
  require_same_shape(paged_k_cache, paged_v_cache, "paged K/V cache");
  require_same_shape(k_scale_cache, v_scale_cache, "K/V scale cache");
  TVM_FFI_ICHECK_EQ(k_scale_cache.size(0), paged_k_cache.size(0));
  if (kv_layout == QKVLayout::kHND) {
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(1), num_heads);
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(2), page_size);
  } else {
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(1), page_size);
    TVM_FFI_ICHECK_EQ(k_scale_cache.size(2), num_heads);
  }

  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  auto k_sf_strides = k_scale_cache.strides();
  auto v_sf_strides = v_scale_cache.strides();
  const size_t k_stride_page = k_strides[0];
  const size_t k_stride_n = kv_layout == QKVLayout::kHND ? k_strides[2] : k_strides[1];
  const size_t k_stride_h = kv_layout == QKVLayout::kHND ? k_strides[1] : k_strides[2];
  const size_t v_stride_page = v_strides[0];
  const size_t v_stride_n = kv_layout == QKVLayout::kHND ? v_strides[2] : v_strides[1];
  const size_t v_stride_h = kv_layout == QKVLayout::kHND ? v_strides[1] : v_strides[2];
  const size_t k_sf_stride_page = k_sf_strides[0];
  const size_t k_sf_stride_n = kv_layout == QKVLayout::kHND ? k_sf_strides[2] : k_sf_strides[1];
  const size_t k_sf_stride_h = kv_layout == QKVLayout::kHND ? k_sf_strides[1] : k_sf_strides[2];
  const size_t v_sf_stride_page = v_sf_strides[0];
  const size_t v_sf_stride_n = kv_layout == QKVLayout::kHND ? v_sf_strides[2] : v_sf_strides[1];
  const size_t v_sf_stride_h = kv_layout == QKVLayout::kHND ? v_sf_strides[1] : v_sf_strides[2];
  auto append_k_strides = append_key.strides();
  auto append_v_strides = append_value.strides();
  const size_t append_k_stride_n = append_k_strides[0];
  const size_t append_k_stride_h = append_k_strides[1];
  const size_t append_v_stride_n = append_v_strides[0];
  const size_t append_v_stride_h = append_v_strides[1];

  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(append_key.dtype(), c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(slot_mapping.dtype(), id_type, [&] {
      cudaError_t status = NVFP4QuantizeAppendPagedKVCacheWithSlotMapping(
          static_cast<c_type*>(append_key.data_ptr()),
          static_cast<c_type*>(append_value.data_ptr()),
          static_cast<id_type*>(slot_mapping.data_ptr()), nnz, num_heads, page_size,
          static_cast<uint32_t>(paged_k_cache.size(0)), packed_head_dim, append_k_stride_n,
          append_k_stride_h, append_v_stride_n, append_v_stride_h,
          static_cast<uint8_t*>(paged_k_cache.data_ptr()),
          static_cast<uint8_t*>(paged_v_cache.data_ptr()),
          static_cast<uint8_t*>(k_scale_cache.data_ptr()),
          static_cast<uint8_t*>(v_scale_cache.data_ptr()), k_stride_page, k_stride_n, k_stride_h,
          v_stride_page, v_stride_n, v_stride_h, k_sf_stride_page, k_sf_stride_n, k_sf_stride_h,
          v_sf_stride_page, v_sf_stride_n, v_sf_stride_h, static_cast<float*>(k_scale.data_ptr()),
          static_cast<float*>(v_scale.data_ptr()), stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "NVFP4QuantizeAppendPagedKVCacheWithSlotMapping failed with error: "
          << cudaGetErrorString(status);
      return true;
    });
  });

  TVM_FFI_ICHECK(success)
      << "NVFP4QuantizeAppendPagedKVCacheWithSlotMapping failed to dispatch with dtype "
      << append_key.dtype();
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
