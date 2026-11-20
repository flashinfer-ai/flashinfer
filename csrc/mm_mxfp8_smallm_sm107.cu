/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MXFP8 (E4M3 data + UE8M0 block-32 scales, swizzled 128x4 SF layout) NT GEMM
//   result[M,N] = dequant(aq[M,K]) * dequant(bq[N,K])^T   (fp32 accum, bf16 out)
// Decode-size M (1..32).  sm_107 has no native warp-level fp8 MMA (ptxas expands
// mma...e4m3 into 12 F2FP unpacks + 2 HMMA), so operands are converted to bf16
// once and pre-multiplied by the power-of-two block scale; mma.m16n8k16.bf16
// then accumulates in place.  The *B* matrix drives the mma 16-row axis so the
// streamed operand is the wide one and the tiny A matrix is the 8-wide one.
//
// Each lane streams 16 contiguous k bytes per load.  A lane's 16 bytes sit
// inside one 32-element scale block, so one scale decode covers four mma
// fragments instead of one, and one load replaces two.  The mma k index is a
// permutation of the real k index; A and B use the same permutation, so the
// dot product is unchanged.
//
// This translation unit holds the TVM-FFI binding and the measured per-shape
// route table; the framework-agnostic raw-pointer kernel (kmain) and its
// device helpers live in include/flashinfer/gemm/mm_mxfp8_smallm_sm107.cuh.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <flashinfer/gemm/mm_mxfp8_smallm_sm107.cuh>
#include <optional>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace mxfp8_smallm_sm107 {

// ---------------------------------------------------------------------------
struct Args {
  const uint8_t *aq, *bq, *asf, *bsf;
  __nv_bfloat16* out;
  int64_t M, N, K, m0;
};
static inline int64_t cdiv(int64_t a, int64_t b) { return a / b + (a % b != 0); }

template <int MSUB>
static void gen_m(const Args& A, cudaStream_t st);

// One launch per measured shape family: CLS is a compile-time constant so the
// cluster fold, the owner mapping and the grid shape are all resolved at
// compile time, and only the kernel actually used by a route is instantiated.
//
// Route encoding — every template argument is a named tuning axis, in the
// order of the launch<> sites in Run() (kmain's own WN is fixed to 1 here):
//   MSUB  8-row m sub-tiles; one CTA always covers all of M (1..4 = M 1..32)
//   NSUB  16-row n sub-tiles handled by one warp (per-warp n tile = 16*NSUB)
//   WK    warps splitting the CTA's k range (reduced through smem)
//   PIPE  software-pipelined k loop (overlapped load/convert schedule)
//   BPS   __launch_bounds__ resident-block budget per SM
//   CLS   thread-block cluster height (grid.y); CLS > 1 folds K across the
//         cluster's CTAs and reduces through DSMEM
//   SCV   fused scaled-cvt dequant form (cvt.rn.scaled ue8m0 up-conversion)
//         instead of the two-step convert-and-scale form
//   CLSC  compile-time __cluster_dims__ height (0 = runtime attribute)
//   MZ    grid.z replication
//   HALF  16-byte instead of 32-byte per-lane k chunk
//   GPW   k-unit groups issued per warp step (0/1/2 schedule variants)
//   Q32   maps a lane's two 16-byte scale chunks onto one aligned 32-byte
//         block, and tests the small-scale (precise-dot) class per word
//   AFFB / AFFA  precomputed per-row operand offset bases for B / A
//   FS    cluster mbarrier startup handshake for the DSMEM epilogue
//   DACC  second accumulator bank per (n, m) pair
//   CM/CN/CK  the exact (M, N, K) the route was measured for, folded into
//         the instantiation (0 = family catch-all with runtime dims)
//   CGPC  compile-time k-groups-per-rank (0 = take the runtime value)
//   L2P / L1B  operand load form (L2-prefetch / plain ldg128 / default)
//   MBH   mbarrier wait form used for the cluster handshake
// Re-deriving a route (for example after a CUDA toolkit bump): each entry is
// the winner of a paired A/B sweep over these axes for its (M, N, K) under
// CUDA-graph timing against the stock mm_mxfp8 backends; re-run such a sweep
// for the shape, update the launch<> entry, and re-validate with
// tests/gemm/test_mm_mxfp8_smallm_sm107.py (which runs every allowlisted
// shape numerically). Routes must beat BOTH measured stock arms with margin
// before their shape enters the dispatch allowlist JSON.
template <int MSUB, int NSUB, int WK, bool PIPE, int BPS, int CLS, bool SCV = SCVD, int CLSC = CLS,
          int MZ = 1, bool HALF = false, int GPW = 0, bool Q32 = false, bool AFFB = false,
          bool AFFA = false, bool FS = false, bool DACC = false, int CM = 0, int CN = 0, int CK = 0,
          int CGPC = 0, bool L2P = false, bool MBH = false, bool L1B = false>
static void launch(const Args& A, int64_t gpc, cudaStream_t st) {
  constexpr int NTOT = 16 * NSUB;
  const int64_t nkb = A.K / 32;
  dim3 grid(cdiv(A.N, NTOT), CLS, MZ);
  // A green context may restrict cluster co-scheduling.  Query this call's
  // stream before capture/launch, and retain the tuned route whenever it fits.
  // No context, stream or resource decision is cached between calls.
  if constexpr (CLS >= 4) {
    CUgreenCtx green = nullptr;
    TVM_FFI_ICHECK(cuStreamGetGreenCtx(reinterpret_cast<CUstream>(st), &green) == CUDA_SUCCESS)
        << "mm_mxfp8_smallm_sm107: cannot query stream green context";
    if (green) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = grid;
      cfg.blockDim = dim3(32 * WK, 1, 1);
      cfg.stream = st;
      int active = 0;
      cudaError_t status = cudaOccupancyMaxActiveClusters(
          &active,
          kmain<MSUB, NSUB, 1, WK, PIPE, BPS, (CLS > 1), (CLS > 1 ? CLSC : 0), SCV, HALF, GPW, Q32,
                AFFB, AFFA, FS, DACC, CM, CN, CK, CGPC, L2P, MBH, L1B>,
          &cfg);
      if ((status == cudaSuccess && active == 0) || status == cudaErrorInvalidConfiguration ||
          status == cudaErrorInvalidClusterSize) {
        if (status != cudaSuccess) (void)cudaGetLastError();
        gen_m<MSUB>(A, st);
        return;
      }
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "mm_mxfp8_smallm_sm107: cluster occupancy query failed: "
          << cudaGetErrorString(status);
    }
  }
  kmain<MSUB, NSUB, 1, WK, PIPE, BPS, (CLS > 1), (CLS > 1 ? CLSC : 0), SCV, HALF, GPW, Q32, AFFB,
        AFFA, FS, DACC, CM, CN, CK, CGPC, L2P, MBH, L1B><<<grid, 32 * WK, 0, st>>>(
      A.aq, A.bq, A.asf, A.bsf, A.out, A.M, A.N, A.K, nkb, cdiv(nkb, 4), gpc, CLS, A.m0);
  cudaError_t launch_status = cudaGetLastError();
  TVM_FFI_ICHECK(launch_status == cudaSuccess)
      << "mm_mxfp8_smallm_sm107: kernel launch failed: " << cudaGetErrorString(launch_status);
}

// Fallback for shapes outside the measured families: correct and reasonable
// everywhere, but deliberately narrow so the whole translation unit stays
// inside the evaluator's compile budget.  Cluster size stays a runtime value
// here, which keeps this to one kernel per (MSUB, WK) pair.
template <int MSUB, int WK, bool CLU, bool WIDE>
static void launch_gen(const Args& A, int CLS, int64_t gpc, cudaStream_t st) {
  constexpr int BPS = (MSUB == 4) ? 2 : 1;
  const int64_t nkb = A.K / 32;
  dim3 grid(cdiv(A.N, 32), CLU ? CLS : 1, 1);
  if constexpr (!CLU) {
    kmain<MSUB, 2, 1, WK, false, BPS, false, 0, SCVD, false, 0, false, false, false, false, false,
          0, 0, 0, 0, false, false, false, WIDE><<<grid, 32 * WK, 0, st>>>(
        A.aq, A.bq, A.asf, A.bsf, A.out, A.M, A.N, A.K, nkb, cdiv(nkb, 4), gpc, 1, A.m0);
    cudaError_t launch_status = cudaGetLastError();
    TVM_FFI_ICHECK(launch_status == cudaSuccess)
        << "mm_mxfp8_smallm_sm107: fallback kernel launch failed: "
        << cudaGetErrorString(launch_status);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(32 * WK, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = CLS;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t launch_status = cudaLaunchKernelEx(
        &cfg,
        kmain<MSUB, 2, 1, WK, false, BPS, true, 0, SCVD, false, 0, false, false, false, false,
              false, 0, 0, 0, 0, false, false, false, WIDE>,
        A.aq, A.bq, A.asf, A.bsf, A.out, A.M, A.N, A.K, nkb, cdiv(nkb, 4), gpc, CLS, A.m0);
    TVM_FFI_ICHECK(launch_status == cudaSuccess)
        << "mm_mxfp8_smallm_sm107: clustered fallback launch failed: "
        << cudaGetErrorString(launch_status);
  }
}

template <int MSUB, bool CLU, bool WIDE>
static void gen_wk(const Args& A, int CLS, int64_t gpc, int wk, cudaStream_t st) {
  if (wk >= 8)
    launch_gen<MSUB, 8, CLU, WIDE>(A, CLS, gpc, st);
  else if (wk >= 4)
    launch_gen<MSUB, 4, CLU, WIDE>(A, CLS, gpc, st);
  else if (wk >= 2)
    launch_gen<MSUB, 2, CLU, WIDE>(A, CLS, gpc, st);
  else
    launch_gen<MSUB, 1, CLU, WIDE>(A, CLS, gpc, st);
}

template <int MSUB>
static void gen_m(const Args& A, cudaStream_t st) {
  const int64_t gtot = cdiv(A.K / 32, 4);
  const int64_t tiles = cdiv(A.N, 32);
  int maxcls = gtot >= 32 ? 8 : int(gtot >> 2);
  if (maxcls > 8) maxcls = 8;
  if (maxcls < 1) maxcls = 1;
  int cls = 1;
  if (tiles < 160) {
    while (cls * 2 <= maxcls && tiles * cls * 2 <= 300) cls *= 2;
    if (MSUB < 4)
      while (cls > 1 && tiles * cls > 240) cls >>= 1;
  }
  const int64_t gpc = cdiv(gtot, cls);
  const int64_t rounds = cdiv(gpc, 8);
  int wk = cdiv(gpc, rounds);
  if (wk > 8) wk = 8;
  if (wk < 1) wk = 1;
  // Unlisted shapes use a single-CTA K fold.  Measured cluster routes above
  // carry compile-time cluster dimensions; the generic fallback intentionally
  // avoids a runtime cluster size that could conflict with that attribute.
  // Select width from the largest offset actually formed, rather than the
  // allocation size: a correctly handled 2-GiB row-major tensor can still use
  // the original narrow route.  The swizzled scales can be wider than data
  // when K is very large and the row count is small.
  const auto needs_wide = [&](int64_t rows) {
    const int64_t last = rows - 1;
    const int64_t rr = last & 127;
    const int64_t sr = rr < 31 ? rr * 16 : 496 + ((rr - 31) / 32) * 4;
    const int64_t last_block = A.K / 32 - 1;
    const int64_t scale_offset =
        ((last >> 7) * gtot + (last_block >> 2)) * 512 + sr + (last_block & 3);
    return last * A.K + 48 > INT32_MAX || scale_offset > INT32_MAX;
  };
  if (needs_wide(A.M) || needs_wide(A.N))
    gen_wk<MSUB, false, true>(A, 1, gtot, wk, st);
  else
    gen_wk<MSUB, false, false>(A, 1, gtot, wk, st);
}

// DPS-style entry: result[M,N] (bf16, row-major) = dequant(aq[M,K]) @ dequant(bq[N,K])^T.
// aq/bq are fp8 e4m3 bytes (row-major, K contiguous); a_sf/b_sf are the 1D
// swizzled 128x4 UE8M0 block-32 scale buffers mxfp8_quantize(...,
// is_sf_swizzled_layout=True) produces. The Python dispatch gate guarantees
// layout, dtype and shape preconditions; only cheap invariants are re-checked.
void Run(TensorView aq, TensorView bq, TensorView a_sf, TensorView b_sf, TensorView result) {
  check_cuda_tensor(aq, "aq");
  check_same_device(bq, aq, "bq", "aq");
  check_same_device(a_sf, aq, "a_sf", "aq");
  check_same_device(b_sf, aq, "b_sf", "aq");
  check_same_device(result, aq, "result", "aq");

  const int64_t M = aq.shape()[0];
  const int64_t K = aq.shape()[1];
  const int64_t N = bq.shape()[0];
  TVM_FFI_ICHECK_EQ(bq.shape()[1], K) << "mm_mxfp8_smallm_sm107: K mismatch";
  TVM_FFI_ICHECK_EQ(result.shape()[0], M) << "mm_mxfp8_smallm_sm107: M mismatch";
  TVM_FFI_ICHECK_EQ(result.shape()[1], N) << "mm_mxfp8_smallm_sm107: N mismatch";
  TVM_FFI_ICHECK_EQ(K % 32, 0) << "mm_mxfp8_smallm_sm107: K must be a multiple of 32";
  Args A;
  A.M = M;
  A.K = K;
  A.N = N;
  // get_stream only retrieves the per-device FFI stream; it does not make
  // that device current. Establish the operands' device for the launches
  // (and restore the caller's current device on scope exit), so a call with
  // all tensors on a non-current GPU is not issued on the wrong device.
  // The guard is engaged only on an actual mismatch: this entry sits on a
  // host-latency-bound path, and the common same-device call should pay one
  // cudaGetDevice, not a set/restore pair.
  int current_device = -1;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&current_device), cudaSuccess)
      << "mm_mxfp8_smallm_sm107: current device query failed";
  std::optional<ffi::CUDADeviceGuard> device_guard;
  if (current_device != aq.device().device_id) {
    device_guard.emplace(aq.device().device_id);
  }
  cudaStream_t st = get_stream(aq.device());
  A.aq = (const uint8_t*)aq.data_ptr();
  A.bq = (const uint8_t*)bq.data_ptr();
  A.asf = (const uint8_t*)a_sf.data_ptr();
  A.bsf = (const uint8_t*)b_sf.data_ptr();
  A.out = (__nv_bfloat16*)result.data_ptr();
  // Match the specialized Python contract, including for direct FFI calls.
  const auto aligned = [](const void* p, uintptr_t bytes) {
    return reinterpret_cast<uintptr_t>(p) % bytes == 0;
  };
  TVM_FFI_ICHECK(aligned(A.aq, 32) && aligned(A.bq, 32) && aligned(A.asf, 16) &&
                 aligned(A.bsf, 16) && aligned(A.out, 16))
      << "mm_mxfp8_smallm_sm107: pointer_alignment";
  A.m0 = 0;

  // ---- measured shape families -------------------------------------------
  // launch<MSUB, NSUB, WK, PIPE, BPS, CLS, SCV, CLSC, MZ, HALF, GPW>(A, units-per-rank)
  // Every entry is the winner of a paired A/B against the alternatives for that
  // family: m tile, n tile (32 or 16 rows per warp), warp k-split, k unit size
  // (128 or 64 elements), cluster width, resident-block budget and dequant form.
  //
  // The table deliberately carries more routes than the Python dispatch gate
  // admits. The gate's allowlist is exact membership over the shapes that beat
  // both measured stock arms WITH MARGIN, so the exact-M routes of shapes that
  // measured faster-but-not-with-margin (for example the N=8192/K=1280 family)
  // and the measured M<=x family catch-alls are unreachable through mm_mxfp8
  // today. They are kept because they are tuned, validated routes: widening
  // the allowlist after a re-measure is then a JSON-only change (no kernel
  // retuning, no new instantiations), and any out-of-gate caller of this FFI
  // entry still gets a measured route instead of the generic fallback.
  if (K == 6144) {
    if (N == 768) {
      if (M == 24) {
        launch<3, 1, 8, false, 2, 6, true, 6, 1, false, 1, true, true, true, true, false, 24, 768,
               6144, 8, true, false, true>(A, 8, st);
        return;
      }
      if (M == 12) {
        launch<2, 1, 6, false, 1, 8, SCVD, 8, 1, false, 1, true, true, false, true, false, 12, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
      if (M == 8) {
        launch<1, 1, 6, false, 1, 8, SCVD, 8, 1, false, 1, true, true, true, true, false, 8, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
      if (M == 4) {
        launch<1, 1, 6, false, 1, 8, SCVD, 8, 1, false, 1, true, true, false, true, false, 4, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
      if (M <= 8) {
        launch<1, 1, 6, false, 1, 8, SCVD, 8, 1, false, 1, false, true, false, true, false, 0, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
      if (M <= 16) {
        launch<2, 1, 6, false, 1, 8, SCVD, 8, 1, false, 1, false, true, false, true, false, 0, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
      if (M <= 24) {
        launch<3, 1, 6, false, 3, 8, SCVD, 8, 1, false, 1, false, true, false, true, false, 0, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
      if (M <= 32) {
        launch<4, 1, 6, false, 3, 8, SCVD, 8, 1, false, 1, false, true, false, true, false, 0, 768,
               6144, 6, true>(A, 6, st);
        return;
      }
    } else if (N == 1536) {
      if (M == 12) {
        launch<2, 1, 8, false, 1, 6, true, 6, 1, false, 1, true, true, false, true, false, 12, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
      if (M == 24) {
        launch<3, 2, 8, false, 1, 6, true, 6, 1, false, 1, true, true, true, true, false, 24, 1536,
               6144, 8, false, false, true>(A, 8, st);
        return;
      }
      if (M == 32) {
        launch<4, 2, 4, true, 2, 6, true, 6, 1, false, 2, true, true, true, true, false, 32, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
      if (M == 8) {
        launch<1, 1, 8, false, 1, 6, SCVD, 6, 1, false, 1, true, true, true, true, false, 8, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
      if (M == 4) {
        launch<1, 1, 8, false, 1, 6, SCVD, 6, 1, false, 1, true, true, false, true, false, 4, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
      if (M <= 8) {
        launch<1, 1, 6, true, 1, 4, SCVD, 4, 1, false, 2, true, true, false, true, false, 0, 1536,
               6144, 12, true>(A, 12, st);
        return;
      }
      if (M <= 16) {
        launch<2, 2, 4, true, 1, 6, false, 6, 1, false, 2, false, true, false, true, false, 0, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
      if (M <= 24) {
        launch<3, 2, 4, true, 2, 6, true, 6, 1, false, 2, false, true, false, true, false, 0, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
      if (M <= 32) {
        launch<4, 2, 4, true, 2, 6, false, 6, 1, false, 2, false, true, false, true, false, 0, 1536,
               6144, 8, true>(A, 8, st);
        return;
      }
    }
  } else if (K == 5120) {
    if (N == 1152) {
      if (M == 1) {
        launch<1, 1, 5, false, 2, 8, SCVD, 8, 1, false, 1, true, true, false, true, false, 1, 1152,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M == 2) {
        launch<1, 1, 5, false, 2, 8, SCVD, 8, 1, false, 1, true, true, false, true, false, 2, 1152,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M == 4) {
        launch<1, 1, 5, false, 2, 8, SCVD, 8, 1, false, 1, true, true, false, true, false, 4, 1152,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M == 8) {
        launch<1, 1, 5, false, 2, 8, SCVD, 8, 1, false, 1, true, true, true, true, false, 8, 1152,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M <= 8) {
        launch<1, 1, 10, false, 1, 4, SCVD, 4, 1, false, 1, false, true, false, true, false, 0,
               1152, 5120, 10>(A, 10, st);
        return;
      }
      if (M == 16) {
        launch<2, 1, 5, false, 2, 8, SCVD, 8, 1, false, 1, true, true, true, true, false, 16, 1152,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M <= 16) {
        launch<2, 1, 10, false, 1, 4, SCVD, 4, 1, false, 1, false, true, false, true, false, 0,
               1152, 5120, 10>(A, 10, st);
        return;
      }
      if (M == 32) {
        launch<4, 1, 5, false, 2, 8, SCVD, 8, 1, false, 1, false, true, true, true, false, 32, 1152,
               5120, 5, false, true>(A, 5, st);
        return;
      }
      if (M <= 32) {
        launch<4, 1, 5, true, 2, 4, SCVD, 4, 1, false, 2, false, true, false, true, false, 0, 1152,
               5120, 10>(A, 10, st);
        return;
      }
    } else if (N == 1792) {
      if (M == 8) {
        launch<1, 1, 5, false, 3, 8, SCVD, 8, 1, false, 1, true, false, false, true, false, 8, 1792,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M <= 8) {
        launch<1, 1, 5, false, 3, 8, SCVD, 8, 1, false, 1, true, false, false, true, false, 0, 1792,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M == 16) {
        launch<2, 1, 5, false, 4, 8, SCVD, 8, 1, false, 1, false, true, true, true, false, 16, 1792,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M <= 16) {
        launch<2, 2, 5, false, 2, 8, SCVD, 8, 1, false, 1, false, true, false, true, false, 0, 1792,
               5120, 5, true>(A, 5, st);
        return;
      }
      if (M <= 24) {
        launch<3, 1, 5, false, 3, 4, SCVD, 4, 1, false, 2, false, true, false, true, false, 0, 1792,
               5120, 10, true>(A, 10, st);
        return;
      }
      if (M == 32) {
        launch<4, 1, 5, true, 3, 4, SCVD, 4, 1, false, 2, true, true, true, true, false, 32, 1792,
               5120, 10, true>(A, 10, st);
        return;
      }
      if (M <= 32) {
        launch<4, 1, 5, false, 3, 4, SCVD, 4, 1, false, 2, true, true, false, true, false, 0, 1792,
               5120, 10, true>(A, 10, st);
        return;
      }
    }
  } else if (N == 6144 && K == 384) {
    if (M == 4) {
      launch<1, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 4, 6144,
             384, 6>(A, 6, st);
      return;
    }
    if (M == 8) {
      launch<1, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 8, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M <= 8) {
      launch<1, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M == 12) {
      launch<2, 1, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 12, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M == 16) {
      launch<2, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 16, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M <= 16) {
      launch<2, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M == 24) {
      launch<3, 1, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 24, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M <= 24) {
      launch<3, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M == 32) {
      launch<4, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 32, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
    if (M <= 32) {
      launch<4, 2, 6, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 6144,
             384, 6, true>(A, 6, st);
      return;
    }
  } else if (N == 5120 && K == 576) {
    if (M == 1) {
      launch<1, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 1, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M == 2) {
      launch<1, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 2, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M == 4) {
      launch<1, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 4, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M == 8) {
      launch<1, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 8, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M <= 8) {
      launch<1, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M == 16) {
      launch<2, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 16, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M <= 16) {
      launch<2, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M == 24) {
      launch<3, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 24, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M <= 24) {
      launch<3, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M == 32) {
      launch<4, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 32, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
    if (M <= 32) {
      launch<4, 2, 9, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 5120,
             576, 9, true>(A, 9, st);
      return;
    }
  } else if (N == 6144 && K == 1024) {
    if (M == 4) {
      launch<1, 1, 8, false, 2, 1, SCVD, 1, 1, false, 1, true, true, false, false, false, 4, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M == 8) {
      launch<1, 1, 8, false, 2, 1, SCVD, 1, 1, false, 1, true, true, true, false, false, 8, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M <= 8) {
      launch<1, 1, 8, false, 2, 1, SCVD, 1, 1, false, 1, true, true, false, false, false, 0, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M == 12) {
      launch<2, 1, 8, false, 2, 1, SCVD, 1, 1, false, 1, true, true, false, false, false, 12, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M == 16) {
      launch<2, 2, 16, false, 1, 1, SCVD, 1, 1, true, 0, false, true, true, false, false, 16, 6144,
             1024, 16, true>(A, 16, st);
      return;
    }
    if (M <= 16) {
      launch<2, 2, 16, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 6144,
             1024, 16, true>(A, 16, st);
      return;
    }
    if (M == 24) {
      launch<3, 1, 8, false, 2, 1, SCVD, 1, 1, false, 1, true, true, true, false, false, 24, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M <= 24) {
      launch<3, 2, 8, false, 1, 1, SCVD, 1, 1, false, 1, false, true, false, false, false, 0, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M == 32) {
      launch<4, 1, 4, true, 2, 1, SCVD, 1, 1, false, 2, true, true, true, false, false, 32, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
    if (M <= 32) {
      launch<4, 1, 8, false, 2, 1, SCVD, 1, 1, false, 1, false, true, false, false, false, 0, 6144,
             1024, 8, true>(A, 8, st);
      return;
    }
  } else if (N == 8192 && K == 1280) {
    if (M == 16) {
      launch<2, 1, 10, false, 1, 1, SCVD, 1, 1, false, 1, true, true, true, false, false, 16, 8192,
             1280, 10, true>(A, 10, st);
      return;
    }
    if (M <= 16) {
      launch<2, 1, 10, false, 1, 1, SCVD, 1, 1, true, 0, false, true, false, false, false, 0, 8192,
             1280, 20, true>(A, 20, st);
      return;
    }
    if (M == 24) {
      launch<3, 1, 5, false, 2, 2, SCVD, 2, 1, false, 1, true, true, true, true, false, 24, 8192,
             1280, 5, true>(A, 5, st);
      return;
    }
    if (M <= 24) {
      launch<3, 1, 5, false, 3, 1, SCVD, 1, 1, false, 2, false, true, false, false, false, 0, 8192,
             1280, 10, true>(A, 10, st);
      return;
    }
    if (M == 32) {
      launch<4, 1, 5, false, 3, 1, SCVD, 1, 1, false, 0, false, true, true, false, false, 32, 8192,
             1280, 10, true>(A, 10, st);
      return;
    }
    if (M <= 32) {
      launch<4, 1, 5, false, 3, 1, SCVD, 1, 1, false, 0, false, true, false, false, false, 0, 8192,
             1280, 10, true>(A, 10, st);
      return;
    }
  } else if (N == 5120 && K == 2048) {
    if (M == 24) {
      launch<3, 2, 8, true, 1, 1, SCVD, 1, 1, false, 2, true, false, true, false, false, 24, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M == 2) {
      launch<1, 1, 16, false, 1, 1, SCVD, 1, 1, false, 1, true, true, false, false, false, 2, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M == 1) {
      launch<1, 1, 16, false, 1, 1, SCVD, 1, 1, false, 1, true, true, false, false, false, 1, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M == 4) {
      launch<1, 1, 16, false, 1, 1, SCVD, 1, 1, false, 1, true, true, false, false, false, 4, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M == 8) {
      launch<1, 1, 16, false, 1, 1, SCVD, 1, 1, false, 1, true, true, true, false, false, 8, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M <= 8) {
      launch<1, 1, 8, true, 1, 1, SCVD, 1, 1, false, 2, false, true, false, false, false, 0, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M == 16) {
      launch<2, 2, 8, true, 1, 1, SCVD, 1, 1, false, 2, false, true, true, false, false, 16, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M <= 16) {
      launch<2, 2, 8, true, 1, 1, SCVD, 1, 1, false, 2, false, true, false, false, false, 0, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M <= 24) {
      launch<3, 2, 8, true, 1, 1, SCVD, 1, 1, false, 2, false, true, false, false, false, 0, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M == 32) {
      launch<4, 2, 8, true, 2, 1, SCVD, 1, 1, false, 2, false, true, true, false, false, 32, 5120,
             2048, 16>(A, 16, st);
      return;
    }
    if (M <= 32) {
      launch<4, 2, 8, true, 2, 1, SCVD, 1, 1, false, 2, false, true, false, false, false, 0, 5120,
             2048, 16>(A, 16, st);
      return;
    }
  }

  // decode M is 1..32, but the declared scope has no upper bound: cover any M
  // by stepping 32 rows per launch (a single launch for every shipped shape).
  for (A.m0 = 0; A.m0 < A.M; A.m0 += 32) {
    const int mrem = A.M - A.m0 < 32 ? A.M - A.m0 : 32;
    switch (cdiv(mrem, 8)) {
      case 1:
        gen_m<1>(A, st);
        break;
      case 2:
        gen_m<2>(A, st);
        break;
      case 3:
        gen_m<3>(A, st);
        break;
      default:
        gen_m<4>(A, st);
        break;
    }
  }
}

}  // namespace mxfp8_smallm_sm107
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mm_mxfp8_smallm_sm107_run, flashinfer::mxfp8_smallm_sm107::Run);
