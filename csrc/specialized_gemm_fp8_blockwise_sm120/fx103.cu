// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/fx103.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/fx103.cuh>

#include "launch_status.h"

namespace ffi = tvm::ffi;

void RunFX103(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
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

  const dim3 grid(64, 1, 1);
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(1024, 1, 1);
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
  sm120_launch::launch_ex(&config, fx103::fxk<2, 16, 2, 16, 1, 2, 1, true>, Ap, Bp, Asp, Bsp, Yp, M,
                          N, K, KB);
}
