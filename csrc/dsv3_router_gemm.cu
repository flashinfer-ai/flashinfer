#include "flashinfer/gemm/dsv3_router_gemm.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::trtllm_dsv3_router_gemm {
template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemm(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream,
                      bool use_pdl = false) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;
  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = use_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto status = cudaLaunchKernelEx(
      &config, router_gemm_kernel<T, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim>, output,
      mat_a, mat_b);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaLaunchKernelEx failed with error code " << cudaGetErrorString(status);
}

template void invokeRouterGemm<__nv_bfloat16, 1, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 2, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 3, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 4, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 5, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 6, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 7, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 8, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 9, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 10, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 11, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 12, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 13, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 14, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 15, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 16, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);
}  // namespace flashinfer::trtllm_dsv3_router_gemm
