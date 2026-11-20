// TVM-FFI binding for the SM90 decode-M fp8 block-scaled dense GEMM. The
// device code lives in include/flashinfer/gemm/fp8bs_gemm_decodem_sm90/
// decodem.cuh; this TU keeps the host-side launch glue and the exported
// entry point.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <cstring>
#include <flashinfer/gemm/fp8bs_gemm_decodem_sm90/decodem.cuh>

#include "fp8bs_gemm_decodem_sm90.h"

namespace ffi = tvm::ffi;

template <class T>
static inline T* tensor_ptr(ffi::TensorView t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}

template <int MT, int SPLIT, bool KeepX, int KU = 2, bool PipeWeights = false, bool PackMT8 = false,
          bool ExactRound = false, bool PairPipe = false, bool PushDSM = false,
          bool FlatPush = false, bool CompactM2 = false, bool ExplicitSTS = false,
          bool DenseTile = false, bool DenseOut = false, bool PrefetchScale = false,
          bool AsyncPush = false, bool DenseN = false, bool MmaPipe = false, bool Quant8 = false,
          bool PerWarpMbar = false, bool PreWaitScale = false, bool SpreadOwners = false>
static void launch_ku(dim3 grid, bool lb, cudaStream_t stream, const __nv_bfloat16* x,
                      const uint8_t* weight, const float* weight_scale, float eps,
                      __nv_bfloat16* out, int M, int N, int K) {
  auto* kern =
      fp8_blockscale_wgmma_ku_kernel<MT, SPLIT, KU, KeepX, PipeWeights, PackMT8, ExactRound,
                                     PairPipe, PushDSM, FlatPush, CompactM2, ExplicitSTS, DenseTile,
                                     DenseOut, PrefetchScale, AsyncPush, DenseN, MmaPipe, Quant8,
                                     PerWarpMbar, PreWaitScale, SpreadOwners>;
  // The push/flat mailbox is DYNAMIC shared memory (the 12.x ptxas static
  // limit is 48 KiB per function and the KU=4 instantiations exceed it with
  // the mailbox counted as static).  Both attributes are set per launch:
  // cudaFuncSetAttribute applies to the current device only, so a
  // once-per-process flag would skip the opt-in on every later device.
  constexpr size_t PUSH_BYTES =
      sizeof(uint32_t) *
      (FlatPush ? SplitkMailbox<MT, SPLIT, (MT >= 16 || PackMT8)>::WORDS
                : ((PushDSM || AsyncPush) ? (CompactM2 ? SPLIT * 64 : SPLIT * 256) : 1));
  if constexpr (SPLIT > 8) {
    TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
                      cudaSuccess);
  }
  TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(PUSH_BYTES)),
                    cudaSuccess);
  if (lb) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = PUSH_BYTES;
    config.stream = stream;
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attribute.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = &attribute;
    config.numAttrs = 1;
    const cudaError_t err =
        cudaLaunchKernelEx(&config, kern, x, weight, weight_scale, eps, out, M, N, K);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "launch_ku: " << cudaGetErrorString(err);
  } else {
    kern<<<grid, 128, PUSH_BYTES, stream>>>(x, weight, weight_scale, eps, out, M, N, K);
  }
}

template <int MT, bool KeepX>
static void launch_ku_split(int split, dim3 grid, bool lb, cudaStream_t stream,
                            const __nv_bfloat16* x, const uint8_t* weight,
                            const float* weight_scale, float eps, __nv_bfloat16* out, int M, int N,
                            int K) {
  switch (split) {
    case 2:
      launch_ku<MT, 2, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    case 4:
      launch_ku<MT, 4, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    case 6:
      launch_ku<MT, 6, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    case 8:
      launch_ku<MT, 8, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    default:
      launch_ku<MT, 16, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
  }
}

static void launch_ku_all(int mt, int split, bool keepx, bool lb, dim3 grid, cudaStream_t stream,
                          const __nv_bfloat16* x, const uint8_t* weight, const float* weight_scale,
                          float eps, __nv_bfloat16* out, int M, int N, int K) {
  if (mt == 8) {
    if (keepx)
      launch_ku_split<8, true>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
    else
      launch_ku_split<8, false>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                K);
  } else if (mt == 16) {
    if (keepx)
      launch_ku_split<16, true>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                K);
    else
      launch_ku_split<16, false>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                 K);
  } else {
    if (keepx && split == 8)
      launch_ku<24, 8, true, 2, true>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
    else if (keepx)
      launch_ku_split<24, true>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                K);
    else
      launch_ku_split<24, false>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                 K);
  }
}

template <int MT, int SPLIT, bool KeepX = false, bool PackMT8 = false, bool PushDSM = false,
          bool CompactM2 = false, bool AsyncM2 = false, bool DenseN = false, bool ScaleOpt = false>
static void launch_splitk_load_balanced(dim3 grid, cudaStream_t stream, const __nv_bfloat16* x,
                                        const uint8_t* weight, const float* weight_scale, float eps,
                                        __nv_bfloat16* out, int M, int N, int K) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(128, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attribute = {};
  attribute.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attribute.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
  config.attrs = &attribute;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config,
      fp8_blockscale_wgmma_splitk_kernel<MT, SPLIT, KeepX, PackMT8, PushDSM, false, false, false,
                                         CompactM2, ScaleOpt, false, AsyncM2, DenseN>,
      x, weight, weight_scale, eps, out, M, N, K);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "launch_splitk_load_balanced: " << cudaGetErrorString(err);
}

static void RunImpl(ffi::TensorView x, ffi::TensorView weight, ffi::TensorView weight_scale,
                    float eps, ffi::TensorView out) {
  const auto dev = x.device();
  TVM_FFI_ICHECK_EQ(dev.device_type, kDLCUDA);
  // Every attribute set and launch below binds to the current device; run
  // the whole dispatch on the operand's device and restore the caller's.
  ffi::CUDADeviceGuard device_guard(dev.device_id);
  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight_scale.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);

  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(weight.size(0));
  TVM_FFI_ICHECK_EQ(K & 127, 0);
  // K == 0 would leave `out` unwritten on several routes; the stock entry
  // rejects it (null input) and this backend matches that contract.
  TVM_FFI_ICHECK_GT(K, 0);

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const auto* xp = tensor_ptr<const __nv_bfloat16>(x);
  const auto* wp = tensor_ptr<const uint8_t>(weight);
  const auto* wsp = tensor_ptr<const float>(weight_scale);
  auto* op = tensor_ptr<__nv_bfloat16>(out);

  // Wide-N K=2048 rows need more than one CTA per column tile to cover all
  // 132 SMs.  Split-8 clusters replace the single 768/1024-thread MWG CTA.
  if (M == 24 && N == 6144 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, 1);
    launch_ku<24, 8, false, 2, false, false, true, false, true, false, false, true, true, false,
              true, true, false, false, false, false, true>(grid, true, stream, xp, wp, wsp, eps,
                                                            op, M, N, K);
    return;
  }
  if (M == 16 && N == 7168 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, 1);
    launch_ku<16, 8, false, 2, true, false, true, false, true, false, false, true, true, false,
              false, true, false, false, false, true>(grid, true, stream, xp, wp, wsp, eps, op, M,
                                                      N, K);
    return;
  }

  // Preserve c797's tuned dispatch and exact MWG routes for W6 and W9.  W5
  // uses the conventional split-K PushDSM path below.
  const bool use_mwg = (N == 8192 && K == 1536 && (M == 24 || M == 1 || M == 8)) ||
                       (K == 2048 && ((N == 7168 && M == 16) || (N == 6144 && M == 24)));
  if (use_mwg) {
    if (fp8bs_mwg_launch(op, xp, wp, wsp, M, N, K, eps, stream)) return;
  }

  // Exact W10/W11 experiment: retain every existing route, but replace the
  // selected row's pull/conventional reduction with c131's flat owner-push
  // mailbox.  Each specialization gets its own cluster-size opt-in.
  if (M == 12 && N == 2624 && K == 6144) {
    TVM_FFI_ICHECK_EQ(
        cudaFuncSetAttribute(fp8_blockscale_wgmma_cpa_kernel<16, 16, false, false, true, true,
                                                             false, true, true, true, true>,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
        cudaSuccess);
    const dim3 grid((N + 63) / 64 * 16, (M + 15) / 16);
    fp8_blockscale_wgmma_cpa_kernel<16, 16, false, false, true, true, false, true, true, true, true>
        <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  if (M == 4 && N == 2048 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, (M + 7) / 8);
    launch_ku<8, 8, false, 2, true, true, true, false, true, false, false, true, false, false,
              false, true, false, true, false, false, false, true>(grid, true, stream, xp, wp, wsp,
                                                                   eps, op, M, N, K);
    return;
  }

  if (N > 2304 && N <= 3072 && K >= 4096 && M > 16 && M <= 24 && ((K >> 7) % 24 == 0)) {
    const dim3 grid((N + 63) / 64 * 12, (M + 23) / 24);
    if (M % 24 == 0 && (N & 63) == 0)
      launch_ku<24, 12, false, 2, false, false, false, false, true, false, false, true, false, true,
                true, true, true, false, false, false, true>(grid, true, stream, xp, wp, wsp, eps,
                                                             op, M, N, K);
    else
      launch_ku<24, 12, false, 2, false, false, false, false, true, false, false, true>(
          grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  if (M == 2 && N == 7168 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, (M + 7) / 8);
    launch_splitk_load_balanced<8, 8, false, true, true, true, true, true, true>(
        grid, stream, xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  const bool use_hmma = (N > 3072 && K < 2048 && M > 16 && M <= 24) ||
                        (N > 3072 && K >= 2048 && M > 8 && M <= 16) ||
                        (N > 2304 && N <= 3072 && K >= 4096 && M > 16 && M <= 24);
  if (use_hmma) {
    const cudaError_t err = fp8bs_launch(op, xp, wp, wsp, M, N, K, eps, stream);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess)
        << "fp8bs_gemm_decodem_sm90: kernel launch failed: " << cudaGetErrorString(err);
    return;
  }
  if (N > 2304 && N <= 3072 && K >= 4096 && M > 8 && M <= 16 && (K >> 7) >= 16) {
    TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(fp8_blockscale_wgmma_cpa_kernel<16, 16, true, true>,
                                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
                      cudaSuccess);
    const dim3 grid((N + 63) / 64 * 16, (M + 15) / 16);
    fp8_blockscale_wgmma_cpa_kernel<16, 16, true, true>
        <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  const bool use_split8 = N <= 2304 && K >= 4096;
  const bool use_split4 = N <= 2304 && K >= 1536 && K < 4096;
  const bool use_split6 = N > 2304 && N <= 3072 && K >= 4096;
  const bool use_split2 = N > 3072 && M <= 24;
  const bool use_wide_large_k = N > 3072 && K >= 1536;
  const dim3 row8_grid((N + 63) / 64, (M + 7) / 8);
  const bool use_wide_mid_n = N > 2304 && N <= 3072 && K >= 4096;
  if (use_wide_large_k && K >= 2048 && M > 16 && M <= 24) {
    const dim3 grid((N + 63) / 64 * 4, (M + 23) / 24);
    launch_ku<24, 4, true, 2, true>(grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
  } else if (use_wide_large_k && M > 8 && M <= 16) {
    if (K >= 2048) {
      const dim3 grid((N + 63) / 64 * 4, (M + 15) / 16);
      fp8_blockscale_wgmma_splitk_kernel<16, 4, true>
          <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
    } else {
      const dim3 grid((N + 63) / 64 * 6, (M + 15) / 16);
      if (M == 16 && N == 8192 && K == 1536)
        launch_ku<16, 6, false, 2, true, false, false, false, true, false, false, true, false, true,
                  true, true, false, false, true>(grid, true, stream, xp, wp, wsp, eps, op, M, N,
                                                  K);
      else
        launch_ku<16, 6, false, 2, true, false, false, false, true, false, false, true, false,
                  false, true>(grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
    }
  } else if (use_wide_mid_n && M > 8 && M <= 16) {
    const dim3 grid((N + 63) / 64 * 8, (M + 15) / 16);
    fp8_blockscale_wgmma_splitk_kernel<16, 8>
        <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split4 && M <= 8) {
    TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(fp8_blockscale_wgmma_splitk_kernel<8, 16>,
                                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
                      cudaSuccess);
    fp8_blockscale_wgmma_splitk_kernel<8, 16>
        <<<dim3(row8_grid.x * 16, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split4 && M > 16 && M <= 24) {
    // Narrow N leaves few column tiles, so the grid is small; batching two
    // k-groups per shared round doubles the weight vectors in flight per
    // thread and halves the publish fence/barrier count per weight byte.
    const dim3 grid((N + 63) / 64 * 8, (M + 23) / 24);
    if (M == 24 && N == 2048 && K == 2048) {
      launch_ku<24, 8, false, 2, true, false, true, false, true, false, false, true, true, false,
                false, true, false, true, false, true>(grid, true, stream, xp, wp, wsp, eps, op, M,
                                                       N, K);
    } else {
      launch_ku<24, 8, false, 2, true, false, false, false, true, false, false, true>(
          grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
    }
  } else if (use_split2 && K >= 2048 && M <= 8) {
    if (M > 1) {
      launch_splitk_load_balanced<8, 8, true, true>(dim3(row8_grid.x * 8, row8_grid.y), stream, xp,
                                                    wp, wsp, eps, op, M, N, K);
    } else {
      launch_splitk_load_balanced<8, 8, true>(dim3(row8_grid.x * 8, row8_grid.y), stream, xp, wp,
                                              wsp, eps, op, M, N, K);
    }
  } else if (use_split2 && M <= 8) {
    if (M > 1) {
      launch_splitk_load_balanced<8, 6, false, true>(dim3(row8_grid.x * 6, row8_grid.y), stream, xp,
                                                     wp, wsp, eps, op, M, N, K);
    } else {
      launch_splitk_load_balanced<8, 6>(dim3(row8_grid.x * 6, row8_grid.y), stream, xp, wp, wsp,
                                        eps, op, M, N, K);
    }
  } else if (use_split2) {
    fp8_blockscale_wgmma_splitk_kernel<8, 2>
        <<<dim3(row8_grid.x * 2, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split8 && M <= 8) {
    if ((K >> 7) == 14 * 4) {
      if (M == 2) {
        if (N == 2048)
          launch_ku<8, 14, false, 4, false, true, true, true, true, false, true, false, false,
                    false, true, true, true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp,
                                             wp, wsp, eps, op, M, N, K);
        else
          launch_ku<8, 14, false, 4, false, true, true, true, true, false, true, false, false,
                    false, true, true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp, wp,
                                       wsp, eps, op, M, N, K);
      } else {
        launch_ku<8, 14, false, 4, false, true, true, true, true, false, false, false, false, false,
                  true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp, wp, wsp, eps, op, M,
                        N, K);
      }
    } else {
      launch_ku<8, 14, false, 4, false, true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp,
                                              wp, wsp, eps, op, M, N, K);
    }
  } else if (use_split8 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 8>
        <<<dim3(row8_grid.x * 8, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split4 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 4>
        <<<dim3(row8_grid.x * 4, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split6 && M > 16 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 8>
        <<<dim3(row8_grid.x * 8, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split6 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 6>
        <<<dim3(row8_grid.x * 6, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (M <= 24) {
    fp8_blockscale_wgmma_kernel<8><<<row8_grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else {
    const dim3 block(64);
    const dim3 grid((N + 63) / 64, (M + 15) / 16);
    fp8_blockscale_kernel<<<grid, block, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  }
}

void Run(ffi::TensorView x, ffi::TensorView weight, ffi::TensorView weight_scale, float eps,
         ffi::TensorView out) {
  // Scope the post-launch check to THIS call: an earlier, already-handled
  // runtime error left in the host thread's error slot must not be reported
  // as this GEMM's launch failure (sticky device-fault errors re-report on
  // every call either way and are unaffected by this clear).
  (void)cudaGetLastError();
  RunImpl(x, weight, weight_scale, eps, out);
  // Every route launches asynchronously; a refused launch (bad cluster or
  // shared-memory configuration) must surface instead of returning an
  // unwritten `out`.
  const cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess)
      << "fp8bs_gemm_decodem_sm90: kernel launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
