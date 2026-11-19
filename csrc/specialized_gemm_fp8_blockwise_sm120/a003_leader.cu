// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/a003_leader.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/a003_leader.cuh>

namespace ffi = tvm::ffi;

void RunSpecial(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                ffi::TensorView Y);

namespace a003_leader_impl {  // reopened; the device code is in the header

#define LAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv)                                                   \
  do {                                                                                            \
    const dim3 grid((N + NWNv * NTv * 8 - 1) / (NWNv * NTv * 8), (unsigned)((M + MRv - 1) / MRv), \
                    1);                                                                           \
    gk<NWv, NWNv, NTv, MRv, KIv, MINBv>                                                           \
        <<<grid, NWv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);                       \
    return;                                                                                       \
  } while (0)

#define BYMR(NWv, NWNv, NTv, KIv, MINBv)      \
  do {                                        \
    if (mr == 8) {                            \
      LAUNCH(NWv, NWNv, NTv, 8, KIv, MINBv);  \
    } else if (mr == 16) {                    \
      LAUNCH(NWv, NWNv, NTv, 16, KIv, MINBv); \
    } else {                                  \
      LAUNCH(NWv, NWNv, NTv, 32, KIv, MINBv); \
    }                                         \
  } while (0)

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

  // combo_entry routes here only (N=9216, M 9..16) and (N=12288, M 5..16);
  // the exact-shape specials below are the ones reachable from that set.
  if ((N == 12288 && K == 2048 && (M == 8 || M == 16)) || (M == 16 && N == 9216 && K == 2048)) {
    RunSpecial(A, B, As, Bs, Y);
    return;
  }

  const long long bbytes = (long long)N * (long long)K;
  const bool tiny = bbytes <= (3LL << 19);
  const int mr = (M <= 8) ? 8 : ((M <= 16) ? (tiny ? 8 : 16) : (tiny ? 16 : 32));

  if (N <= 128 && KB == 16) LAUNCH(16, 1, 1, 8, 1, 1);
  if (N <= 128) BYMR(8, 1, 1, 0, 2);

  if (bbytes >= (12LL << 20)) {
    if (mr == 32) {
      if (KB == 16) BYMR(8, 1, 4, 2, 2);
      BYMR(8, 1, 4, 0, 2);
    }
    if (M >= 16 && N < 12288) {
      if (KB == 16) BYMR(8, 2, 2, 4, 2);
      BYMR(8, 2, 2, 0, 2);
    }
    if (N >= 12288) {
      if (KB == 16) BYMR(8, 1, 2, 2, 2);
      BYMR(8, 1, 2, 0, 2);
    }
    if (KB == 16) BYMR(8, 1, 1, 2, 2);
    BYMR(8, 1, 1, 0, 2);
  }
  if (bbytes >= (4LL << 20)) {
    if (M == 16 && N == 2048 && KB == 32) LAUNCH(16, 2, 2, 8, 4, 1);
    if (M >= 8) {
      if (KB == 32) BYMR(16, 2, 2, 4, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (KB == 32) BYMR(16, 1, 2, 2, 1);
    BYMR(16, 1, 2, 0, 1);
  }
  if (KB <= 4) {
    if (KB == 4) BYMR(8, 2, 2, 1, 2);
    BYMR(8, 2, 2, 0, 2);
  }
  if (M > 16) {
    if (KB == 16) BYMR(16, 2, 1, 2, 1);
    BYMR(16, 2, 1, 0, 1);
  }
  if (M > 8) {
    if (KB == 16) LAUNCH(16, 1, 2, 8, 1, 1);
    LAUNCH(16, 1, 2, 8, 0, 1);
  }
  if (KB == 16) BYMR(16, 1, 2, 1, 1);
  BYMR(16, 1, 2, 0, 1);
}

}  // namespace a003_leader_impl

void RunA003(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
             ffi::TensorView Y) {
  a003_leader_impl::Run(A, B, As, Bs, Y);
}
