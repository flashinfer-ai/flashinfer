// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/a003_kernel.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/a003_kernel.cuh>

namespace ffi = tvm::ffi;

#define LAUNCH(Wv, MRv, SPv, K1)       \
  gemm_fp8_bw_kernel<Wv, MRv, SPv, K1> \
      <<<grid, Wv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, mm, N, K, KB)

#define DISP_SP(Wv, MRv, K1)      \
  do {                            \
    if (msplit) {                 \
      LAUNCH(Wv, MRv, true, K1);  \
    } else {                      \
      LAUNCH(Wv, MRv, false, K1); \
    }                             \
  } while (0)

#define DISP_MR1(Wv, K1)         \
  do {                           \
    if (MR == 8) {               \
      DISP_SP(Wv, 8, K1);        \
    } else if (MR == 16) {       \
      DISP_SP(Wv, 16, K1);       \
    } else {                     \
      LAUNCH(Wv, 32, false, K1); \
    }                            \
  } while (0)

#define DISP_MR(Wv)          \
  do {                       \
    if (kiter == 1) {        \
      DISP_MR1(Wv, 1);       \
    } else if (kiter == 2) { \
      DISP_MR1(Wv, 2);       \
    } else {                 \
      DISP_MR1(Wv, 0);       \
    }                        \
  } while (0)

#define DISP_W()     \
  do {               \
    switch (W) {     \
      case 1:        \
        DISP_MR(1);  \
        break;       \
      case 2:        \
        DISP_MR(2);  \
        break;       \
      case 4:        \
        DISP_MR(4);  \
        break;       \
      case 8:        \
        DISP_MR(8);  \
        break;       \
      default:       \
        DISP_MR(16); \
        break;       \
    }                \
  } while (0)

void RunSpecial(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  if (M <= 0 || N <= 0 || KB <= 0) return;

  const uint8_t* Abase = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asbase = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Ybase = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));

  // Reached only through a003_leader for (N=12288, M 8/16) and
  // (N=9216, M=16); the wide/async and N=64 specials that used to live here
  // served shapes the allowlist excludes and were pruned.
  if (M == 16 && N == 9216 && K == 2048) {
    const dim3 grid((N + 7) >> 3, 1, 1);
    gemm_fp8_bw_kernel<4, 16, false, 0, true, true>
        <<<grid, 128, 0, stream>>>(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB);
    return;
  }

  if (M == 16 && N == 12288 && K == 2048) {
    const dim3 grid((N + 7) >> 3, 1, 1);
    gemm_fp8_bw_kernel<4, 16, false, 0, true>
        <<<grid, 128, 0, stream>>>(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB);
    return;
  }

  const int ntiles = (N + 7) >> 3;
  const int blocks = ntiles;
  int W = 1;
  const int wt = (9024 + blocks - 1) / blocks;
  while (W < wt && W < 16) W <<= 1;
  while (W > 1 && W > KB) W >>= 1;
  if ((M == 2 || M == 4 || M == 8) && N == 12288 && K == 2048) W = 4;
  if (M == 16 && N == 9216 && K == 2048) W = 4;
  if (M == 16 && N == 12288 && K == 2048) W = 4;

  int MR = (M <= 8) ? 8 : ((M <= 16) ? 16 : 32);
  // Halving the m chunk doubles the block count at the price of re-reading the
  // weights; only worth it while that stream is small.
  const bool msplit = (MR > 8) && (blocks < 376) && ((long long)N * (long long)K <= (1536LL << 10));
  if (msplit) MR >>= 1;

  // Fixed one- and two-group forms remove loop control on the common decode
  // schedules.  The unsplit MR16 one-group form measured slower, so it keeps
  // the generic loop.
  int kiter = 0;
  const bool kok = (MR != 16) || msplit || (W <= 8);
  if (W == KB && kok)
    kiter = 1;
  else if (2 * W == KB && kok)
    kiter = 2;
  if (M == 4 && N == 2048 && K == 4096) kiter = 0;

  if (msplit) {
    const dim3 grid(blocks, (unsigned)((M + MR - 1) / MR), 1);
    const uint8_t* Ap = Abase;
    const float* Asp = Asbase;
    __nv_bfloat16* Yp = Ybase;
    const int mm = M;
    DISP_W();
  } else {
    for (int m0 = 0; m0 < M; m0 += 32) {
      const int mm = (M - m0 < 32) ? (M - m0) : 32;
      MR = (mm <= 8) ? 8 : ((mm <= 16) ? 16 : 32);
      const dim3 grid(blocks, 1u, 1u);
      const uint8_t* Ap = Abase + (size_t)m0 * (size_t)K;
      const float* Asp = Asbase + (size_t)m0 * (size_t)KB;
      __nv_bfloat16* Yp = Ybase + (size_t)m0 * (size_t)N;
      DISP_W();
    }
  }
}
