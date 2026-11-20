// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/a003_tuned.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/a003_tuned.cuh>

#include "launch_status.h"

namespace ffi = tvm::ffi;

// ---------------------------------------------------------------- dispatch

namespace a003_tuned_impl {  // reopened; the device code is in the header

template <int NW_N, int NW_K, int NT, int MR, int KU, int ITERS, int MINB>
static inline void launch_bw(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                             __nv_bfloat16* Y, int M, int N, int K, int KB, const dim3& grid,
                             int threads, cudaStream_t s) {
  constexpr int SMEM = red_smem_bytes<NW_N, NW_K, NT, MR>();
  if constexpr (SMEM > 48 * 1024) {
    sm120_launch::check(cudaFuncSetAttribute(gemm_bw<NW_N, NW_K, NT, MR, KU, ITERS, MINB>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM),
                        "cudaFuncSetAttribute");
  }
  gemm_bw<NW_N, NW_K, NT, MR, KU, ITERS, MINB>
      <<<grid, threads, SMEM, s>>>(A, B, As, Bs, Y, M, N, K, KB);
  sm120_launch::check_submission();
}

template <int NW_N, int NW_K, int NT, int KU, int ITERS, int MINB>
static inline void go2(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                       __nv_bfloat16* Y, int M, int N, int K, int KB, int MR, cudaStream_t s) {
  constexpr int CN = NW_N * NT * 8;
  constexpr int TH = NW_N * NW_K * 32;
  const dim3 grid((N + CN - 1) / CN, (unsigned)((M + MR - 1) / MR), 1);
  if (MR == 8)
    launch_bw<NW_N, NW_K, NT, 8, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, TH, s);
  else if (MR == 16)
    launch_bw<NW_N, NW_K, NT, 16, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, TH, s);
  else
    launch_bw<NW_N, NW_K, NT, 32, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, TH, s);
}

template <int NW_N, int NW_K, int NT, int KU, int MINB>
static inline void go(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                      __nv_bfloat16* Y, int M, int N, int K, int KB, int MR, cudaStream_t s) {
  const int iters = KB / (NW_K * KU);
  if (iters == 1)
    go2<NW_N, NW_K, NT, KU, 1, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
  else if (iters == 2)
    go2<NW_N, NW_K, NT, KU, 2, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
  else
    go2<NW_N, NW_K, NT, KU, 0, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
}

void Run(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
         ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  if (M <= 0 || N <= 0 || KB <= 0) return;

  const uint8_t* Ap = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asp = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Yp = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));

  // combo_entry routes only (N=12288, K=2048, M 3..4) here. Measured best
  // tiling for that cell: one n-warp-group, eight k-warps, two n8 subtiles,
  // one k-group in flight, two fixed k steps (KB=16), 8-row m tiles.
  if (N == 12288 && K == 2048 && M <= 4) {
    const dim3 grid(768, 1, 1);  // 16 columns per CTA
    launch_bw<1, 8, 2, 8, 1, 2, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, grid, 256, stream);
    return;
  }

  // Defensive generic path for anything else (one warp owns all of K;
  // correct for any shape, never the measured route).
  const int mr = (M <= 8) ? 8 : ((M <= 16) ? 16 : 32);
  go<1, 1, 1, 1, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, mr, stream);
}

}  // namespace a003_tuned_impl

void RunTuned(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
              ffi::TensorView Y) {
  // Called from combo_entry::Run, which holds the CUDADeviceGuard for the
  // whole dispatch (the separate run_tuned FFI export was unused and pruned).
  a003_tuned_impl::Run(A, B, As, Bs, Y);
}
