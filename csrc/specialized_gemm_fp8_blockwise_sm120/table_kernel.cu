// TVM-FFI binding/launch TU for the specialized sm120 fp8 blockwise GEMM kernels.
// The framework-independent device code lives in
// include/flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/table_kernel.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/gemm/specialized_gemm_fp8_blockwise_sm120/table_kernel.cuh>

#include "launch_status.h"

namespace ffi = tvm::ffi;

// ---------------------------------------------------------------- dispatch

namespace table_impl {  // reopened; the device code is in the header

// Surface a rejected launch to the FFI caller instead of returning normally
// with Y unwritten: the Python wrapper would otherwise count the dispatch as
// successful and mark the shape warmed for capture. Successful launches are
// never synchronized; for <<<...>>> launches the immediate error is only
// observable through cudaGetLastError().
static inline void check_launch(cudaError_t status) { sm120_launch::check(status, "launch"); }

template <int NW_N, int NW_K, int NT, int MR, int KU, int ITERS, int MINB>
static inline void launch_bw(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                             __nv_bfloat16* Y, int M, int N, int K, int KB, const dim3& grid,
                             cudaStream_t s) {
  constexpr int TH = NW_N * NW_K * 32;
  constexpr int SMEM = red_smem_bytes<NW_N, NW_K, NT, MR>();
  if constexpr (SMEM > 48 * 1024) {
    const cudaError_t status =
        cudaFuncSetAttribute(gemm_bw<NW_N, NW_K, NT, MR, KU, ITERS, MINB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    sm120_launch::check(status, "cudaFuncSetAttribute");
  }
  gemm_bw<NW_N, NW_K, NT, MR, KU, ITERS, MINB><<<grid, TH, SMEM, s>>>(A, B, As, Bs, Y, M, N, K, KB);
  check_launch(cudaGetLastError());
}

template <int NW_N, int NW_K, int NT, int KU, int ITERS, int MINB>
static inline void go2(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                       __nv_bfloat16* Y, int M, int N, int K, int KB, int MR, cudaStream_t s) {
  constexpr int CN = NW_N * NT * 8;
  const dim3 grid((N + CN - 1) / CN, (unsigned)((M + MR - 1) / MR), 1);
  if (MR == 8)
    launch_bw<NW_N, NW_K, NT, 8, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, s);
  else if (MR == 16)
    launch_bw<NW_N, NW_K, NT, 16, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, s);
  else
    launch_bw<NW_N, NW_K, NT, 32, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, s);
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

  if (M <= 8 && N == 2048 && K == 512) {
    const dim3 grid(64, 1, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = red_smem_bytes<4, 4, 1, 8>();
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
    check_launch(cudaLaunchKernelEx(&config, gemm_bw<4, 4, 1, 8, 1, 1, 1>, Ap, Bp, Asp, Bsp, Yp, M,
                                    N, K, KB));
    return;
  }

  if ((M == 32 || (M >= 17 && M <= 31)) && N == 2048 && K == 512) {
    // M=32 uses a 2-CTA cluster; M 17..31 the same tiling without one (the
    // measured route; combo_entry sends M 9..16 of this class to c00).
    if (M != 32) {
      const dim3 grid(64, (unsigned)((M + 15) / 16), 1);
      launch_bw<4, 4, 1, 16, 1, 1, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, grid, stream);
      return;
    }
    const dim3 grid(64, 2, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = red_smem_bytes<4, 4, 1, 16>();
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    check_launch(cudaLaunchKernelEx(&config, gemm_bw<4, 4, 1, 16, 1, 1, 1>, Ap, Bp, Asp, Bsp, Yp, M,
                                    N, K, KB));
    return;
  }

  // combo_entry sends (N=9216, M 5..8) here; measured best is eight k-warps,
  // two n8 subtiles, two fixed k steps, two blocks per SM.
  if (N == 9216 && K == 2048 && M >= 5 && M <= 8) {
    go<1, 8, 2, 1, 2>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, 8, stream);
    return;
  }

  // Remaining reachable cell (N=9216, M=1) measured best on the generic
  // single-warp-K route; it also serves as the dispatcher's safety net
  // (correct for any shape).
  const int mr = (M <= 8) ? 8 : ((M <= 16) ? 16 : 32);
  go<1, 1, 1, 1, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, mr, stream);
}

}  // namespace table_impl

void RunTable(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
              ffi::TensorView Y) {
  table_impl::Run(A, B, As, Bs, Y);
}
