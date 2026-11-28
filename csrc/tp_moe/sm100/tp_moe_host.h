// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
//
// Host helpers shared by the SM100 fused TP MoE entries: TMA descriptor encoding and the workspace
// layout.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "tvm_ffi_utils.h"

namespace flashinfer::tp_moe::sm100::host {

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeTiledFn encode_tiled_fn() {
  static EncodeTiledFn fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TVM_FFI_ICHECK_EQ(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                                       cudaEnableDefault, &q),
                      cudaSuccess);
    TVM_FFI_ICHECK(q == cudaDriverEntryPointSuccess && p) << "cuTensorMapEncodeTiled not found";
    return reinterpret_cast<EncodeTiledFn>(p);
  }();
  return fn;
}

// A tiled tensor map: dims innermost first; strides (bytes) of dims 1.. ; box per dim; element
// strides 1.
inline void encode_tiled(CUtensorMap* map, CUtensorMapDataType dtype, const void* ptr,
                         std::initializer_list<uint64_t> dims,
                         std::initializer_list<uint64_t> strides,
                         std::initializer_list<uint32_t> box, CUtensorMapSwizzle swizzle,
                         CUtensorMapL2promotion promotion) {
  cuuint64_t d[5], s[4];
  cuuint32_t b[5], e[5] = {1, 1, 1, 1, 1};
  int n = 0;
  for (uint64_t v : dims) d[n++] = v;
  int k = 0;
  for (uint64_t v : strides) s[k++] = v;
  int j = 0;
  for (uint32_t v : box) b[j++] = v;
  TVM_FFI_ICHECK(k == n - 1 && j == n);
  const CUresult r = encode_tiled_fn()(
      map, dtype, static_cast<cuuint32_t>(n), const_cast<void*>(ptr), d, s, b, e,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(r, CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed";
}

// Sub-buffers of one workspace allocation, each aligned to 4 KB.
template <int N>
struct WorkspaceLayout {
  size_t offset[N] = {};
  size_t total = 0;
  constexpr explicit WorkspaceLayout(const size_t (&bytes)[N]) {
    size_t at = 0;
    for (int i = 0; i < N; ++i) {
      offset[i] = at;
      at += (bytes[i] + 4095) / 4096 * 4096;
    }
    total = at;
  }
};

}  // namespace flashinfer::tp_moe::sm100::host
