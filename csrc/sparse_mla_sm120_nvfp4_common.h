/*
 * Copyright (c) 2026 by FlashInfer team.
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

#pragma once

#include <cstddef>
#include <flashinfer/attention/sparse_mla_sm120/model/nvfp4_cache_traits.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

struct PagedLayout {
  int num_pages;
  int page_size;
  size_t page_stride_bytes;
};

inline PagedLayout parse_nvfp4_paged_layout(const TensorView& cache) {
  constexpr int BPT = NVFP4CacheTraits<ModelType::DSV4>::BYTES_PER_TOKEN;
  TVM_FFI_ICHECK_EQ(cache.dtype(), dl_uint8) << "NVFP4 kv_cache must be uint8";
  if (cache.ndim() == 2) {
    const size_t page_bytes = static_cast<size_t>(cache.size(1));
    TVM_FFI_ICHECK_EQ(page_bytes % BPT, 0);
    TVM_FFI_ICHECK_EQ(cache.stride(1), 1) << "kv_cache byte dimension must be contiguous";
    TVM_FFI_ICHECK_GE(cache.stride(0), static_cast<int64_t>(page_bytes))
        << "kv_cache page stride is smaller than its logical payload";
    return {static_cast<int>(cache.size(0)), static_cast<int>(page_bytes / BPT),
            static_cast<size_t>(cache.stride(0))};
  }

  TVM_FFI_ICHECK(cache.ndim() == 3 || cache.ndim() == 4) << "kv_cache must be 2D, 3D, HND, or NHD";
  TVM_FFI_ICHECK_EQ(cache.size(cache.ndim() - 1), BPT)
      << "NVFP4 cache last dimension must be " << BPT;
  int page_dim;
  if (cache.ndim() == 3) {
    page_dim = 1;
  } else if (cache.size(1) == 1) {
    page_dim = 2;
  } else {
    TVM_FFI_ICHECK_EQ(cache.size(2), 1) << "4D cache requires a singleton KV-head axis";
    page_dim = 1;
  }
  const int page_size = static_cast<int>(cache.size(page_dim));
  TVM_FFI_ICHECK_EQ(cache.stride(cache.ndim() - 1), 1)
      << "kv_cache byte dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(cache.stride(page_dim), BPT)
      << "kv_cache entries inside a page must have stride " << BPT;
  TVM_FFI_ICHECK_GE(cache.stride(0), static_cast<int64_t>(page_size) * BPT)
      << "kv_cache page stride is smaller than its logical payload";
  return {static_cast<int>(cache.size(0)), page_size, static_cast<size_t>(cache.stride(0))};
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
