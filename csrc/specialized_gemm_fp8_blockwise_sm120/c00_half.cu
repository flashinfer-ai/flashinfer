// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/c00_half.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/c00_half.cuh>

#include "launch_status.h"

namespace ffi = tvm::ffi;

namespace c00_impl {  // reopened; the device code is in the header

#define LAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv)                             \
  do {                                                                      \
    const dim3 grid((N + (NWNv * NTv * 8) - 1) / (NWNv * NTv * 8),          \
                    (unsigned)((M + MRv - 1) / MRv), 1);                    \
    gk<NWv, NWNv, NTv, MRv, KIv, MINBv>                                     \
        <<<grid, NWv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB); \
    sm120_launch::check_submission();                                       \
    return;                                                                 \
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

  if (M == 16 && N == 1024 && K == 2048) {
    const dim3 grid(64, 2, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 2;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    sm120_launch::launch_ex(&config, gk<16, 1, 2, 8, 1, 1, true, true>, Ap, Bp, Asp, Bsp, Yp, M, N,
                            K, KB);
    return;
  }

  if (M == 4 && N == 1024 && K == 2048) {
    const dim3 grid(64, 1, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    sm120_launch::launch_ex(&config, gk<16, 1, 2, 8, 1, 1>, Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    return;
  }

  if (M == 8 && N == 1024 && K == 2048) {
    const dim3 grid(64, 1, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    sm120_launch::launch_ex(&config, gk<16, 1, 2, 8, 1, 1, true, true>, Ap, Bp, Asp, Bsp, Yp, M, N,
                            K, KB);
    return;
  }

  if (M == 32 && N == 1024 && K == 2048) {
    const dim3 grid(64, 2, 1);
    gk<16, 2, 1, 16, 2, 1, true><<<grid, 512, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    sm120_launch::check_submission();
    return;
  }

  if (M == 32 && N == 2048 && K == 4096) {
    const dim3 grid(64, 2, 1);
    gk<16, 2, 2, 16, 4, 1, true><<<grid, 512, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    sm120_launch::check_submission();
    return;
  }

  // This short-K row benefits slightly from the ordinary cached global path;
  // its 1 MiB weight stream is small enough to reuse in L2 across the two M
  // chunks without the noncoherent load hint.
  if (M == 16 && N == 2048 && K == 512) {
    const dim3 grid(64, 2, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(256, 1, 1);
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
    sm120_launch::launch_ex(&config, gk<8, 2, 2, 8, 1, 2, false, true>, Ap, Bp, Asp, Bsp, Yp, M, N,
                            K, KB);
    return;
  }

  const long long bbytes = (long long)N * (long long)K;
  // A small weight stream stays resident in L2, so splitting m across CTAs to
  // buy parallelism costs a cheap re-read rather than DRAM traffic.
  const bool tiny = bbytes <= (3LL << 19);  // <= 1.5 MB of weights
  const int mr = (M <= 8) ? 8 : ((M <= 16) ? (tiny ? 8 : 16) : (tiny ? 16 : 32));

  if (N <= 128 && KB == 16) LAUNCH(16, 1, 1, 8, 1, 1);
  if (N <= 128) BYMR(8, 1, 1, 0, 2);

  // Activation re-reads cost (N / CTA-n-width) * M * K bytes of L2 traffic, so
  // the CTA n-width grows with the live row count.  NWN n-groups of warps let a
  // CTA cover 2x the columns while the duplicate A request merges in L1.
  if (bbytes >= (12LL << 20)) {  // 9216 / 12288 columns
    if (mr == 32) {
      if (KB == 16) BYMR(8, 1, 4, 2, 2);
      BYMR(8, 1, 4, 0, 2);
    }
    if (M >= 16 && N < 12288) {
      if (KB == 16) BYMR(8, 2, 2, 4, 2);
      BYMR(8, 2, 2, 0, 2);
    }
    if (N >= 12288 && M >= 16) {
      if (KB == 16) BYMR(16, 2, 2, 2, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (N >= 12288) {
      if (KB == 16) BYMR(8, 1, 2, 2, 2);
      BYMR(8, 1, 2, 0, 2);
    }
    if (KB == 16) BYMR(8, 1, 1, 2, 2);
    BYMR(8, 1, 1, 0, 2);
  }
  if (bbytes >= (4LL << 20)) {  // 2048 x 4096
    if (M >= 8) {
      if (KB == 32) BYMR(16, 2, 2, 4, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (KB == 32) BYMR(16, 1, 2, 2, 1);
    BYMR(16, 1, 2, 0, 1);
  }
  if (KB <= 4) {  // 2048 x 512
    if (KB == 4) BYMR(8, 2, 2, 1, 2);
    BYMR(8, 2, 2, 0, 2);
  }
  if (M > 16) {  // 1024 x 2048
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

}  // namespace c00_impl
