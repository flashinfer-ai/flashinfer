// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/r2_narrow.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/r2_narrow.cuh>

namespace ffi = tvm::ffi;

void RunR2Narrow(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                 ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  const uint8_t* Ap = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asp = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Yp = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));
  const dim3 grid((N + 15) / 16, 1, 1);
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
  // The paired path wins at two and eight live rows; four rows remains on the
  // lean path after an isolated A/B was flat.
  if (M == 2 || M == 4 || M == 8) {
    cudaLaunchKernelEx(&config, r2_narrow::gemm_fp8_bw_kernel<16, 8, false, 2, 2, true, true>, Ap,
                       Bp, Asp, Bsp, Yp, M, N, K, KB);
  } else {
    cudaLaunchKernelEx(&config, r2_narrow::gemm_fp8_bw_kernel<16, 8, false, 2, 2>, Ap, Bp, Asp, Bsp,
                       Yp, M, N, K, KB);
  }
}
