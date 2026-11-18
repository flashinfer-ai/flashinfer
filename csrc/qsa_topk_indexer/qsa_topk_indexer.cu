// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
//
// QSA stage-1 token top-k block indexer for SM107 (Rubin): deterministic
// top-512 block selection over a paged FP8 compressed-key cache, expanded to
// token ids with the causal tail, packed int32 output (ascending ids, -1 pad,
// count in the last column). Ties break toward the lower block index; see the
// pure-PyTorch reference in tests/attention/test_qsa_topk_indexer.py.
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

__device__ __forceinline__ float qsa_relu_exact(float x) {
  float y;
  asm("max.NaN.f32 %0, %1, 0f00000000;" : "=f"(y) : "f"(x));
  return y;
}

// Stable descending selection puts NaNs last. Reserve key zero for NaNs and
// padding; positive finite float bits are monotone and shift safely by one.
__device__ __forceinline__ uint32_t qsa_score_key_exact(float x) {
  const uint32_t raw = __float_as_uint(x);
  uint32_t key = raw + 1u;
  if (isnan(x)) key = 0u;
  return key;
}
// Every finite E4M3 product is an integer multiple of 2^-18. If the
// absolute dot sum is at most 64, all intermediate sums fit exactly in fp32,
// independently of the tensor-core accumulation order. Above that bound use
// the reference's sequential round-to-nearest accumulation for this dot only.
__device__ __noinline__ float qsa_dot_rn(const uint8_t* q, const uint8_t* k) {
  float d = 0.0f;
#pragma unroll 4
  for (int i = 0; i < 128; ++i) {
    __nv_fp8_e4m3 a, b;
    a.__x = q[i];
    b.__x = k[i];
    d = __fmaf_rn((float)a, (float)b, d);
  }
  return d;
}

__device__ __forceinline__ float qsa_checked_dot(float d, float abs_sum, const uint8_t* q,
                                                 const uint8_t* k, bool valid) {
  return (valid && abs_sum > 64.0f) ? qsa_dot_rn(q, k) : d;
}

//
// Reuse all shipped kernels and dispatch branches unchanged.  The included
// root is symbol-renamed so this translation unit can replace only the generic
// qsa_topk_slow launch used above the bounded shared-score regimes.  The
// legacy composite entry point is not exported (QSA_LEGACY_NO_EXPORT): its
// generic launch reaches qsa_topk_slow, whose per-token selection-flag
// scratch (256 histogram words) only covers rows with at most 8192 visible
// blocks, a bound the scope-safe replacement below enforces and the legacy
// T > 131071 branch does not.
#define QSA_LEGACY_NO_EXPORT
#define LaunchGeneric LaunchGenericLegacy
#define Run RunLegacy
#include "kernel_regime_a3_scope_legacy.cuh"
#undef Run
#undef LaunchGeneric
#undef QSA_LEGACY_NO_EXPORT

#include "pack_fill.cuh"
#include "pack_mid.cuh"
#include "w7_fast.cuh"

namespace split_impl {
#define CR 4
#define KTOP 512
#define PWIDTH 2052
#define ROW_V4 513
#define CPAGE 64
#define IDXH 4
#define IDXD 128
#define FASTPOS 2051
#define NTK 256
#define TM 16
#define VCAP 2048
#include "fused_fp8_pack_scope_safe_slow_impl.cuh"
#undef VCAP
#undef TM
#undef NTK
#undef FASTPOS
#undef IDXD
#undef IDXH
#undef CPAGE
#undef ROW_V4
#undef PWIDTH
#undef KTOP
#undef CR
}  // namespace split_impl

__device__ __forceinline__ int4 qsa_fast_identity_value(int col4, int count) {
  const int c = col4 << 2;
  int4 v = make_int4(-1, -1, -1, -1);

  if (c + 3 < count) {
    v = make_int4(c, c + 1, c + 2, c + 3);
  } else if (c < count) {
    // Exactly one int4 can straddle count; materialize only its valid lanes.
    const int rem = count - c;
    v.x = c;
    if (rem >= 2) v.y = c + 1;
    if (rem >= 3) v.z = c + 2;
  }

  // The final packed lane is metadata, even when count reaches this int4.
  if (col4 == packfill::QF_RW - 1) v.w = count;
  return v;
}

namespace w0_store_impl {

// T=24 is only 192 KiB of output.  The generic 256-thread writer launches 49
// CTAs and leaves most of the 212-SM target idle; 64-thread CTAs spread the
// same 385 store warps across one near-full grid wave.
constexpr int kTokens = 24;
constexpr int kThreads = 64;
constexpr int kItems = kTokens * packfill::QF_RW;
constexpr int kGrid = (kItems + kThreads - 1) / kThreads;
static_assert(kGrid == 193, "W0 store grid changed");

__global__ __launch_bounds__(kThreads) void qsa_fast_w0(int4* __restrict__ out4) {
  const int g = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (g >= kItems) return;
  const int row = g / packfill::QF_RW;
  const int col4 = g - row * packfill::QF_RW;
  out4[g] = qsa_fast_identity_value(col4, row + 1);
}

}  // namespace w0_store_impl

namespace w1_store_impl {

// T=96 is 49,248 int4 stores.  234 active threads split them into 211 CTAs,
// one nearly complete wave on the 212-SM target.
constexpr int kTokens = 96;
constexpr int kThreads = 234;
constexpr int kItems = kTokens * packfill::QF_RW;
constexpr int kGrid = (kItems + kThreads - 1) / kThreads;
static_assert(kGrid == 211, "W1 store grid changed");

__global__ __launch_bounds__(kThreads, 1) void qsa_fast_w1(int4* __restrict__ out4) {
  const int g = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (g >= kItems) return;
  const int row = g / packfill::QF_RW;
  const int col4 = g - row * packfill::QF_RW;
  out4[g] = qsa_fast_identity_value(col4, row + 1);
}

}  // namespace w1_store_impl

namespace w2_store_impl {

// Exact W2 specialization.  624 threads form 211 CTAs, filling one grid
// layer while leaving only one SM idle on the 212-SM target.
constexpr int kTokens = 256;
constexpr int kThreads = 624;
constexpr int kItems = kTokens * packfill::QF_RW;
constexpr int kGrid = (kItems + kThreads - 1) / kThreads;
static_assert(kGrid == 211, "W2 store grid changed");

__global__ __launch_bounds__(kThreads, 1) void qsa_fast_w2(int4* __restrict__ out4) {
  const int g = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (g >= kItems) return;
  const int row = g / packfill::QF_RW;
  const int col4 = g - row * packfill::QF_RW;
  out4[g] = qsa_fast_identity_value(col4, row + 1);
}

}  // namespace w2_store_impl

namespace w3_store_impl {

// Exact W3 specialization.  A 640-thread CTA contributes twenty store warps;
// PT=3 covers 1920 consecutive int4 elements per CTA in 206 CTAs.
constexpr int kTokens = 768;
constexpr int kThreads = 640;
constexpr int kPT = 3;
constexpr int kItems = kTokens * packfill::QF_RW;
constexpr int kGrid = (kItems + kThreads * kPT - 1) / (kThreads * kPT);
static_assert(kGrid == 206, "W3 store grid changed");
static_assert(kThreads >= packfill::QF_RW && kThreads < 2 * packfill::QF_RW,
              "W3 row/column carry assumptions changed");

__global__ __launch_bounds__(kThreads, 1) void qsa_fast_w3(int4* __restrict__ out4) {
  const int g0 = static_cast<int>(blockIdx.x) * (kThreads * kPT) + static_cast<int>(threadIdx.x);
  int row = g0 / packfill::QF_RW;
  int col4 = g0 - row * packfill::QF_RW;
#pragma unroll
  for (int k = 0; k < kPT; ++k) {
    const int g = g0 + k * kThreads;
    if (g >= kItems) break;
    out4[g] = qsa_fast_identity_value(col4, row + 1);

    // Advancing by kThreads crosses one or two 513-int4 rows; the
    // range assertion above proves this two-case carry is exhaustive.
    col4 += kThreads;
    if (col4 >= 2 * packfill::QF_RW) {
      col4 -= 2 * packfill::QF_RW;
      row += 2;
    } else {
      col4 -= packfill::QF_RW;
      row += 1;
    }
  }
}

}  // namespace w3_store_impl

static void LaunchGenericScopeSafe(ffi::TensorView qi, ffi::TensorView ck_cache,
                                   ffi::TensorView cu_seqlens, ffi::TensorView cpage_base,
                                   ffi::TensorView packed, cudaStream_t stream) {
  const int total_tokens = static_cast<int>(qi.shape()[0]);
  const int batch = static_cast<int>(cpage_base.shape()[0]);
  constexpr int rows_per_cta = 8;
  const int row_ctas = (total_tokens + rows_per_cta - 1) / rows_per_cta;

  split_impl::qsa_store<<<static_cast<unsigned>(row_ctas), 256, 0, stream>>>(
      static_cast<const int*>(cu_seqlens.data_ptr()), static_cast<int*>(packed.data_ptr()),
      total_tokens, batch, kPackV4, rows_per_cta);

  // Upper bound on the sum of request-local post-FASTPOS tiles.  Retain the
  // same overlaunch so fast and safe-slow kernels use identical mapping.
  const int num_tiles = (total_tokens + 15) / 16 + batch;
  split_impl::qsa_topk_fast<<<static_cast<unsigned>(num_tiles), 256, 0, stream>>>(
      static_cast<const split_impl::u8*>(qi.data_ptr()),
      static_cast<const split_impl::u8*>(ck_cache.data_ptr()),
      static_cast<const int*>(cu_seqlens.data_ptr()),
      static_cast<const int*>(cpage_base.data_ptr()), static_cast<int*>(packed.data_ptr()), batch);
  split_impl::qsa_topk_slow_scope_safe<<<static_cast<unsigned>(num_tiles), 256, 0, stream>>>(
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

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, packed.device().device_id));
  const int* cu = static_cast<const int*>(cu_seqlens.data_ptr());
  int* out = static_cast<int*>(packed.data_ptr());

  if (batch == 1 && total_tokens == w0_store_impl::kTokens) {
    w0_store_impl::qsa_fast_w0<<<w0_store_impl::kGrid, w0_store_impl::kThreads, 0, stream>>>(
        reinterpret_cast<int4*>(out));
    return;
  }

  if (batch == 1 && total_tokens == w1_store_impl::kTokens) {
    w1_store_impl::qsa_fast_w1<<<w1_store_impl::kGrid, w1_store_impl::kThreads, 0, stream>>>(
        reinterpret_cast<int4*>(out));
    return;
  }

  // Exact W2 dispatch; it remains independent of the W3 experiment below.
  if (batch == 1 && total_tokens == w2_store_impl::kTokens) {
    w2_store_impl::qsa_fast_w2<<<w2_store_impl::kGrid, w2_store_impl::kThreads, 0, stream>>>(
        reinterpret_cast<int4*>(out));
    return;
  }

  if (batch == 1 && total_tokens == w3_store_impl::kTokens) {
    w3_store_impl::qsa_fast_w3<<<w3_store_impl::kGrid, w3_store_impl::kThreads, 0, stream>>>(
        reinterpret_cast<int4*>(out));
    return;
  }

  // T <= 2051 bounds every request-local position by 2050, so every row is
  // trivial and can be emitted without loading qi or ck_cache.
  if (total_tokens <= 2051) {
    packfill::LaunchTrivial<false>(cu, out, total_tokens, batch, stream);
    return;
  }

  // The fast W7 slab can batch multiple token rows, but it still needs one
  // complete score row (at most 16,416 uint32 keys) to fit in shared memory.
  // Conservatively route larger totals to the recomputing scope-safe path;
  // arbitrary legal cu_seqlens may put every token in one request.
  if (total_tokens > 65535) {
    LaunchGenericScopeSafe(qi, ck_cache, cu_seqlens, cpage_base, packed, stream);
    return;
  }

  if (total_tokens > 6000 && total_tokens <= 8208) {
    packmid::Launch(static_cast<const uint8_t*>(qi.data_ptr()),
                    static_cast<const uint8_t*>(ck_cache.data_ptr()), cu,
                    static_cast<const int*>(cpage_base.data_ptr()), batch, total_tokens, out,
                    stream);
    return;
  }

  // Preserve the verified single-kernel regime when every request is short.
  if (batch <= 4 && (total_tokens + batch - 1) / batch <= 2051) {
    mid_impl::RunLegacy(qi, ck_cache, cu_seqlens, cpage_base, packed);
    return;
  }

  if (total_tokens > 10000) {
    // w7n keeps 2048 score slots per token, i.e. vis <= 16384 (pos <= 65535).
    if (total_tokens <= 65536) {
      w7n::Run(static_cast<const uint8_t*>(qi.data_ptr()),
               static_cast<const uint8_t*>(ck_cache.data_ptr()), cu,
               static_cast<const int*>(cpage_base.data_ptr()), batch, total_tokens, out, stream);
    } else {
      a3_impl::RunLegacy(qi, ck_cache, cu_seqlens, cpage_base, packed);
    }
    return;
  }

  cross_best_impl::RunLegacy(qi, ck_cache, cu_seqlens, cpage_base, packed);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
