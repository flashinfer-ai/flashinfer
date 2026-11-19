// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/xp.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/xp.cuh>

namespace ffi = tvm::ffi;

namespace xp_impl {  // reopened; the device code is in the header

#define XLAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv, SWAPv, WTv)                  \
  do {                                                                        \
    constexpr int CNv = (NWNv) * (NTv) * 8;                                   \
    const unsigned nt = (unsigned)((N + CNv - 1) / CNv);                      \
    const unsigned mc = (unsigned)((M + (MRv) - 1) / (MRv));                  \
    const dim3 grid((SWAPv) ? mc : nt, (SWAPv) ? nt : mc, 1);                 \
    xpk<NWv, NWNv, NTv, MRv, KIv, MINBv, SWAPv, WTv>                          \
        <<<grid, (NWv) * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB); \
    return;                                                                   \
  } while (0)

// m-chunked launches put the chunk on the fast axis so CTAs that share a
// weight column issue back to back; a single chunk keeps n there instead.
#define XAUTO(NWv, NWNv, NTv, MRv, KIv, MINBv, WTv)         \
  do {                                                      \
    if ((M + (MRv) - 1) / (MRv) > 1) {                      \
      XLAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv, true, WTv);  \
    } else {                                                \
      XLAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv, false, WTv); \
    }                                                       \
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

  if (K == 2048 && N == 1024 && M > 16) {
    const dim3 grid(64, 2, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, xpk<16, 2, 1, 16, 2, 1, false, false>, Ap, Bp, Asp, Bsp, Yp, M, N,
                       K, KB);
    return;
  }
  if (K == 2048) {
    if (N <= 128) {
      if (M <= 4) XAUTO(16, 1, 1, 8, 1, 1, true);
      XAUTO(8, 1, 1, 8, 2, 2, true);
    }
    if (N == 1024) XAUTO(16, 1, 2, 8, 1, 1, false);
    if (N == 9216) XAUTO(8, 1, 1, 8, 2, 2, true);
    if (N >= 12288) XAUTO(8, 1, 2, 8, 2, 2, true);
  }
  if (N == 2048 && K == 512) XLAUNCH(8, 2, 2, 8, 1, 2, false, false);
  if (N == 2048 && K == 4096) XLAUNCH(16, 2, 2, 16, 4, 1, false, false);
  XLAUNCH(16, 1, 2, 8, 0, 1, false, false);
}

}  // namespace xp_impl

void RunXP(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
           ffi::TensorView Y) {
  xp_impl::RunImpl(A, B, As, Bs, Y);
}
