// Host-side launch glue for the SM90 fp8bs decode-M multi-warpgroup kernel.
// The device code lives in
// include/flashinfer/gemm/fp8bs_gemm_decodem_sm90/mwg.cuh; this TU keeps the
// launch-configuration selection and the fp8bs_mwg_launch entry point
// consumed by the TVM-FFI binding TU.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <cstring>
#include <flashinfer/gemm/fp8bs_gemm_decodem_sm90/mwg.cuh>

#include "fp8bs_gemm_decodem_sm90.h"

namespace mwg {

template <int MT, int NWG, int KU, bool ExactOneRound = false, bool StaggerMT8 = false,
          bool SparseM1 = false, bool HalfQuant = false, bool PipelineMMA = false,
          bool DenseM = false>
static void launch(dim3 grid, cudaStream_t stream, const __nv_bfloat16* x, const uint8_t* w,
                   const float* ws, float eps, __nv_bfloat16* out, int M, int N, int K) {
  constexpr int ATOMS = MT / 8;
  constexpr int NACC = MT / 2;
  constexpr int COMPUTE = NWG * KU * (64 * 128 + ATOMS * 1024 + MT * 4);
  constexpr int REDUCE = (NACC / 2) * NWG * 128 * 8;
  constexpr int SMEM = (COMPUTE > REDUCE ? COMPUTE : REDUCE) + 1024;
  // Set per launch: cudaFuncSetAttribute applies to the current device only,
  // so a once-per-process flag would skip the opt-in on every later device.
  const cudaError_t err = cudaFuncSetAttribute(
      mwg_kernel<MT, NWG, KU, ExactOneRound, StaggerMT8, SparseM1, HalfQuant, PipelineMMA, DenseM>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "mwg shared-memory opt-in: " << cudaGetErrorString(err);
  mwg_kernel<MT, NWG, KU, ExactOneRound, StaggerMT8, SparseM1, HalfQuant, PipelineMMA, DenseM>
      <<<grid, 128 * NWG, SMEM, stream>>>(x, w, ws, eps, out, M, N, K);
}

template <int MT>
static bool launch_cfg(int nwg, int ku, dim3 grid, cudaStream_t stream, const __nv_bfloat16* x,
                       const uint8_t* w, const float* ws, float eps, __nv_bfloat16* out, int M,
                       int N, int K) {
  if (nwg == 8 && ku == 2) {
    launch<MT, 8, 2>(grid, stream, x, w, ws, eps, out, M, N, K);
    return true;
  }
  if (nwg == 6 && ku == 2) {
    launch<MT, 6, 2>(grid, stream, x, w, ws, eps, out, M, N, K);
    return true;
  }
  if (nwg == 4 && ku == 4) {
    launch<MT, 4, 4>(grid, stream, x, w, ws, eps, out, M, N, K);
    return true;
  }
  if (nwg == 4 && ku == 2) {
    launch<MT, 4, 2>(grid, stream, x, w, ws, eps, out, M, N, K);
    return true;
  }
  if (nwg == 8 && ku == 1) {
    launch<MT, 8, 1>(grid, stream, x, w, ws, eps, out, M, N, K);
    return true;
  }
  return false;
}

}  // namespace mwg

// Returns false when the shape is not a good fit for this kernel.
bool fp8bs_mwg_launch(void* out, const void* x, const void* w, const void* ws, int M, int N, int K,
                      float eps, cudaStream_t stream) {
  const int ntiles = (N + 63) / 64;
  if (ntiles < 80 || M > 24 || M < 1) return false;
  const int Sk = K >> 7;
  if (Sk < 1) return false;
  int nwg = 0, ku = 0;
  const int cand_nwg[3] = {8, 6, 4};
  for (int ci = 0; ci < 3 && nwg == 0; ++ci) {
    const int c = cand_nwg[ci];
    if (Sk % c != 0) continue;
    const int per = Sk / c;
    for (int k = 4; k >= 1; --k) {
      if (per % k != 0) continue;
      if (c * k > 16) continue;
      nwg = c;
      ku = k;
      break;
    }
  }
  if (nwg == 0) return false;
  const int mt = M <= 8 ? 8 : (M <= 16 ? 16 : 24);
  const dim3 grid(ntiles, (M + mt - 1) / mt);
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x);
  const auto* wp = reinterpret_cast<const uint8_t*>(w);
  const auto* wsp = reinterpret_cast<const float*>(ws);
  auto* op = reinterpret_cast<__nv_bfloat16*>(out);
  // The three K=2048 serving rows have exactly one KU2 round in each of
  // eight warpgroups.  Specialize every MT so the compiler removes range
  // division, live/padding predicates, scale clamping, and loop-back.
  if (Sk == 16) {
    if (mt == 8) {
      mwg::launch<8, 8, 2, true>(grid, stream, xp, wp, wsp, eps, op, M, N, K);
    } else if (mt == 16) {
      mwg::launch<16, 8, 2, true, true, false, true, false, true>(grid, stream, xp, wp, wsp, eps,
                                                                  op, M, N, K);
    } else {
      if (M % 24 == 0 && (N & 63) == 0)
        mwg::launch<24, 8, 2, true, false, false, true, false, true>(grid, stream, xp, wp, wsp, eps,
                                                                     op, M, N, K);
      else
        mwg::launch<24, 8, 2, true, false, false, true>(grid, stream, xp, wp, wsp, eps, op, M, N,
                                                        K);
    }
    return true;
  }
  // The two MT8 serving rows have exactly one KU2 round per warpgroup.
  if (Sk == 12 && mt == 8) {
    if (N == 8192 && K == 1536 && M == 1) {
      mwg::launch<8, 6, 2, true, true, true>(grid, stream, xp, wp, wsp, eps, op, M, N, K);
    } else if (N == 8192 && K == 1536 && M == 8) {
      mwg::launch<8, 6, 2, true, true, false, false, true>(grid, stream, xp, wp, wsp, eps, op, M, N,
                                                           K);
    } else {
      mwg::launch<8, 6, 2, true>(grid, stream, xp, wp, wsp, eps, op, M, N, K);
    }
    return true;
  }
  if (Sk == 12 && mt == 24) {
    if (M % 24 == 0 && (N & 63) == 0)
      mwg::launch<24, 6, 2, true, false, false, true, false, true>(grid, stream, xp, wp, wsp, eps,
                                                                   op, M, N, K);
    else
      mwg::launch<24, 6, 2, true, false, false, true>(grid, stream, xp, wp, wsp, eps, op, M, N, K);
    return true;
  }
  if (mt == 8) return mwg::launch_cfg<8>(nwg, ku, grid, stream, xp, wp, wsp, eps, op, M, N, K);
  if (mt == 16) return mwg::launch_cfg<16>(nwg, ku, grid, stream, xp, wp, wsp, eps, op, M, N, K);
  return mwg::launch_cfg<24>(nwg, ku, grid, stream, xp, wp, wsp, eps, op, M, N, K);
}
