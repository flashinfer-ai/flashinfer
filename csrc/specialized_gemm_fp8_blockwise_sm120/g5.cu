// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/g5.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/g5.cuh>

#include "launch_status.h"

namespace ffi = tvm::ffi;

void LaunchExactN64Rows1(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                         __nv_bfloat16* Y, int M, cudaStream_t stream);
void LaunchExactN64Rows2(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                         __nv_bfloat16* Y, int M, cudaStream_t stream);

namespace g5_impl {  // reopened; the device code is in the header
// N=64 has only 8 n-tiles, so the number of m chunks is the only parallelism
// knob and its optimum moves with the live row count.  At 16 rows a 16-warp
// CTA that covers all 16 k groups in one trip beats the 8-warp two-trip tile
// the shared rule picked; at 2-8 rows a narrower row tile buys the CTAs.
#define G5B(NWv, NWNv, NTv, MRv, KIv, MINBv, SWAPGv, AMv)                     \
  do {                                                                        \
    constexpr int CNv = (NWNv) * (NTv) * 8;                                   \
    const unsigned nt = (unsigned)((N + CNv - 1) / CNv);                      \
    const unsigned mc = (unsigned)((M + (AMv) - 1) / (AMv));                  \
    const dim3 grid(SWAPGv ? mc : nt, SWAPGv ? nt : mc, 1);                   \
    gk<NWv, NWNv, NTv, MRv, KIv, MINBv, SWAPGv, AMv>                          \
        <<<grid, (NWv) * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB); \
    sm120_launch::check_submission();                                         \
    return;                                                                   \
  } while (0)

void RunImpl(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
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

  if (N == 64 && K == 2048) {
    if (M == 2) {
      LaunchExactN64Rows1(Ap, Bp, Asp, Bsp, Yp, M, stream);
      return;
    }
    if (M == 4) {
      LaunchExactN64Rows2(Ap, Bp, Asp, Bsp, Yp, M, stream);
      return;
    }
    if (M == 8) {
      LaunchExactN64Rows1(Ap, Bp, Asp, Bsp, Yp, M, stream);
      return;
    }
  }
  G5B(16, 1, 1, 8, 0, 1, false, 8);
}

}  // namespace g5_impl

void RunG5(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
           ffi::TensorView Y) {
  g5_impl::RunImpl(A, B, As, Bs, Y);
}
