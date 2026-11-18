// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Regime-composed exact QSA indexer for SM107.
//
// Shipped regimes use independently full-suite-correct implementations:
//   * small T: flat identity writer / cp.async scorer
//   * large-B identity rows: request-grouped writer
//   * mid-size B<=4: compact destination-scratch selector
//   * W7-size rows: verified branchless-histogram / early-exit scorer
// Extremely large in-scope requests use the generic recomputation fallback,
// avoiding the bounded shared-score slab used by the fast paths.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;

namespace cross_best_impl {
#define DEVICE_ONLY
#include "kernel_cross_best34.cuh"
#undef DEVICE_ONLY
}  // namespace cross_best_impl

#undef CR
#undef BLOCK_TOPK
#undef PWIDTH
#undef CPAGE
#undef IDX_D
#undef SLOW_CNT
#undef FK_THR
#undef RW
#undef MMA
#undef SROW
#undef NTILE
#undef GMAX
#undef NTHR
#undef NWARP

namespace mid_impl {
#define DEVICE_ONLY
#include "kernel_mid_acdd.cuh"
#undef DEVICE_ONLY
}  // namespace mid_impl

#undef TPB
#undef NWARP
#undef RI4
#undef TOPK
#undef TRIVP
#undef MAXRPC
#undef STILE
#undef VCAP
#undef SCORE_STRIDE
#undef SCAP
#undef NBT
#undef META_I
#undef HPW
#undef KROW
#undef FULLM

namespace a3_impl {
#define DEVICE_ONLY
#include "kernel_a3_w7.cuh"
#undef DEVICE_ONLY
}  // namespace a3_impl

#undef CR
#undef BLOCK_TOPK
#undef PWIDTH
#undef CPAGE
#undef IDX_D
#undef SLOW_CNT
#undef FK_THR
#undef RW
#undef MMA
#undef SROW
#undef NTILE
#undef GMAX
#undef NTHR
#undef NWARP
#undef HBK

namespace split_impl {
#define DEVICE_ONLY
#include "fused_fp8_pack_split_impl.cuh"
#include "fused_fp8_pack_w7fast_impl.cuh"
#undef DEVICE_ONLY
}  // namespace split_impl

#undef CR
#undef KTOP
#undef PWIDTH
#undef ROW_V4
#undef CPAGE
#undef IDXH
#undef IDXD
#undef FASTPOS
#undef NTS
#undef NTK
#undef TM
#undef TN
#undef QPAD
#undef SCSTR
#undef VCAP

constexpr int kPackThreads = 192;
constexpr int kPackWidth = 2052;
constexpr int kPackV4 = 513;

__global__ __launch_bounds__(kPackThreads, 5) void pack_by_request(const int* __restrict__ cu,
                                                                   int* __restrict__ packed,
                                                                   int batch) {
  const int req = static_cast<int>(blockIdx.x);
  if (req >= batch) return;
  const int begin = cu[req];
  const int end = cu[req + 1];
  for (int token = begin; token < end; ++token) {
    const int count = token - begin + 1;
    int4* row = reinterpret_cast<int4*>(packed + static_cast<size_t>(token) * kPackWidth);
    for (int v = static_cast<int>(threadIdx.x); v < kPackV4; v += kPackThreads) {
      const int j = v << 2;
      row[v] = make_int4(j < count ? j : -1, j + 1 < count ? j + 1 : -1, j + 2 < count ? j + 2 : -1,
                         v == kPackV4 - 1 ? count : (j + 3 < count ? j + 3 : -1));
    }
  }
}

static void LaunchGeneric(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cu_seqlens,
                          ffi::TensorView cpage_base, ffi::TensorView packed, cudaStream_t stream) {
  const int total_tokens = static_cast<int>(qi.shape()[0]);
  const int batch = static_cast<int>(cpage_base.shape()[0]);
  const int rows_per_cta = 8;
  const int row_ctas = (total_tokens + rows_per_cta - 1) / rows_per_cta;

  split_impl::qsa_store<<<static_cast<unsigned>(row_ctas), 256, 0, stream>>>(
      static_cast<const int*>(cu_seqlens.data_ptr()), static_cast<int*>(packed.data_ptr()),
      total_tokens, batch, kPackV4, rows_per_cta);

  // This is an upper bound on the sum of request-local post-FASTPOS tiles.
  const int num_tiles = (total_tokens + 15) / 16 + batch;
  split_impl::qsa_topk_fast<<<static_cast<unsigned>(num_tiles), 256, 0, stream>>>(
      static_cast<const split_impl::u8*>(qi.data_ptr()),
      static_cast<const split_impl::u8*>(ck_cache.data_ptr()),
      static_cast<const int*>(cu_seqlens.data_ptr()),
      static_cast<const int*>(cpage_base.data_ptr()), static_cast<int*>(packed.data_ptr()), batch);
  split_impl::qsa_topk_slow<<<static_cast<unsigned>(num_tiles), 256, 0, stream>>>(
      static_cast<const split_impl::u8*>(qi.data_ptr()),
      static_cast<const split_impl::u8*>(ck_cache.data_ptr()),
      static_cast<const int*>(cu_seqlens.data_ptr()),
      static_cast<const int*>(cpage_base.data_ptr()), static_cast<int*>(packed.data_ptr()), batch);
}

void Run(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cu_seqlens,
         ffi::TensorView cpage_base, ffi::TensorView packed) {
  const int total_tokens = static_cast<int>(qi.shape()[0]);
  if (total_tokens <= 0) return;
  const int batch = static_cast<int>(cpage_base.shape()[0]);

  if (total_tokens <= 2051 && batch >= 64) {
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, packed.device().device_id));
    pack_by_request<<<batch, kPackThreads, 0, stream>>>(
        static_cast<const int*>(cu_seqlens.data_ptr()), static_cast<int*>(packed.data_ptr()),
        batch);
    return;
  }

  if (total_tokens > 2051 && batch <= 4 && (total_tokens + batch - 1) / batch <= 2051) {
    mid_impl::Run(qi, ck_cache, cu_seqlens, cpage_base, packed);
    return;
  }

  // The shared-score fast implementations cap a request at 32768 blocks.
  // Above 131071 tokens, use the exact on-demand recomputation fallback.
  if (total_tokens > 131071) {
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, packed.device().device_id));
    LaunchGeneric(qi, ck_cache, cu_seqlens, cpage_base, packed, stream);
    return;
  }

  if (total_tokens > 10000) {
    a3_impl::Run(qi, ck_cache, cu_seqlens, cpage_base, packed);
    return;
  }

  cross_best_impl::Run(qi, ck_cache, cu_seqlens, cpage_base, packed);
}

// The composed root (qsa_topk_indexer.cu) suppresses this export: the generic
// qsa_topk_slow launch above is only scope-correct for the shipped regime
// bounds, and the module must expose a single, scope-safe entry point.
#ifndef QSA_LEGACY_NO_EXPORT
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
#endif
