// Host-side launch glue for the SM90 fp8bs decode-M HMMA kernel. The device
// code lives in include/flashinfer/gemm/fp8bs_gemm_decodem_sm90/hmma.cuh;
// this TU keeps the launch-shape selection and the fp8bs_launch entry point
// consumed by the TVM-FFI binding TU.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstring>
#include <flashinfer/gemm/fp8bs_gemm_decodem_sm90/hmma.cuh>

#include "fp8bs_gemm_decodem_sm90.h"

// ---------------------------------------------------------------------------

template <int MT, int NWARP, int KSHIFT>
static cudaError_t launch_mt(const void* x, const void* w, const void* ws, void* out, int M, int N,
                             int K, int Sk, float eps, int nt, int mb, cudaStream_t stream) {
  static_assert(KSHIFT >= 0 && KSHIFT <= 3);
  constexpr int MP = MT * 8;
  constexpr int WK = NWARP / WN;
  constexpr int NTHREAD = NWARP * 32;
  constexpr int ksplit = 1 << KSHIFT;
  const int ntotal = (Sk + ksplit - 1) >> KSHIFT;
  int cap = (SMEM_BUDGET - 32 * 1024) / (MP * (XROW + 4));
  if (cap < 1) cap = 1;
  int nbuf = ntotal < cap ? ntotal : cap;
  if (nbuf < ntotal) nbuf &= ~1;  // k-groups alternate between warp groups
  if (nbuf < 2) nbuf = 2;
  constexpr int wdt = BN >> KSHIFT;
  size_t smem =
      (size_t)nbuf * MP * (XROW + 4) + (size_t)ksplit * WK * (MP / 2) * (wdt + 4) * sizeof(float2);

  // Set per launch: cudaFuncSetAttribute applies to the current device only,
  // so a once-per-process flag would skip the opt-in on every later device.
  cudaError_t err = cudaFuncSetAttribute((const void*)fp8bs_kernel<MT, NWARP, KSHIFT>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BUDGET);
  if (err != cudaSuccess) return err;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(ksplit, nt, mb);
  cfg.blockDim = dim3(NTHREAD, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = ksplit;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  if constexpr (MT == 3 && NWARP == 16 && KSHIFT == 1) {
    attr[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attr[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
  } else if constexpr (KSHIFT != 0) {
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&cfg, fp8bs_kernel<MT, NWARP, KSHIFT>, (const __nv_bfloat16*)x,
                            (const uint8_t*)w, (const float*)ws, (__nv_bfloat16*)out, M, N, K, Sk,
                            eps, nbuf);
}

template <int MT, int NWARP>
static cudaError_t launch_mt_split(const void* x, const void* w, const void* ws, void* out, int M,
                                   int N, int K, int Sk, float eps, int kshift, int nt, int mb,
                                   cudaStream_t stream) {
  if (kshift == 0)
    return launch_mt<MT, NWARP, 0>(x, w, ws, out, M, N, K, Sk, eps, nt, mb, stream);
  else if (kshift == 1)
    return launch_mt<MT, NWARP, 1>(x, w, ws, out, M, N, K, Sk, eps, nt, mb, stream);
  else if (kshift == 2)
    return launch_mt<MT, NWARP, 2>(x, w, ws, out, M, N, K, Sk, eps, nt, mb, stream);
  else
    return launch_mt<MT, NWARP, 3>(x, w, ws, out, M, N, K, Sk, eps, nt, mb, stream);
}

cudaError_t fp8bs_launch(void* out, const void* x, const void* w, const void* ws, int M, int N,
                         int K, float eps, cudaStream_t stream) {
  // Queried per call: the SM count is a per-device quantity and this entry
  // can serve several devices in one process.
  int numsm = 0;
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  err = cudaDeviceGetAttribute(&numsm, cudaDevAttrMultiProcessorCount, dev);
  if (err != cudaSuccess) return err;
  if (numsm <= 0) numsm = 132;
  const int Sk = K >> 7;
  if (Sk <= 0 || M <= 0 || N <= 0) return cudaSuccess;
  const int nt = (N + BN - 1) / BN;

  const int target = numsm;
  int kshift = 0;
  while (kshift < 3 && (nt << (kshift + 1)) <= target && (1 << (kshift + 1)) <= Sk) ++kshift;
  if (N > 2304 && N <= 3072 && K >= 4096 && Sk >= 8) kshift = 3;
  bool narrow = false;
  // If a split-2 launch leaves at least one eighth of the SMs idle, double
  // the cluster count while halving the block warps.  Total warp work and
  // partial contributors are unchanged, but the grid fills the chip.
  if (kshift == 1 && (nt << 1) <= (7 * target) / 8 && (nt << 2) <= 2 * target && Sk >= 4) {
    ++kshift;
    narrow = true;
  }

  int MT = (M + 7) / 8;
  if (MT > 3) MT = 3;
  const int MP = MT * 8;
  const int mb = (M + MP - 1) / MP;

  // Wide blocks (16 warps, one per SM) pay the quant prologue and the cluster
  // barrier once; at large split-K the k-range per warp gets too short for that
  // to pay off, so fall back to 8 warps.
  const bool wide = (kshift <= 2) && !narrow;
  if (MT == 1)
    return wide ? launch_mt_split<1, 16>(x, w, ws, out, M, N, K, Sk, eps, kshift, nt, mb, stream)
                : launch_mt_split<1, 8>(x, w, ws, out, M, N, K, Sk, eps, kshift, nt, mb, stream);
  else if (MT == 2)
    return wide ? launch_mt_split<2, 16>(x, w, ws, out, M, N, K, Sk, eps, kshift, nt, mb, stream)
                : launch_mt_split<2, 8>(x, w, ws, out, M, N, K, Sk, eps, kshift, nt, mb, stream);
  else
    return wide ? launch_mt_split<3, 16>(x, w, ws, out, M, N, K, Sk, eps, kshift, nt, mb, stream)
                : launch_mt_split<3, 8>(x, w, ws, out, M, N, K, Sk, eps, kshift, nt, mb, stream);
}
