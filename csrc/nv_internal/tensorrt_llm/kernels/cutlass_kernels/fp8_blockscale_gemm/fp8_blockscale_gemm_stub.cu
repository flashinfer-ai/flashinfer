#include "fp8_blockscale_gemm.h"

namespace tensorrt_llm {
namespace kernels {
namespace fp8_blockscale_gemm {

// Explicit instantiation of the template
template class CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::gemm(
    void*, const void*, const void*, int, int, int, cudaStream_t, const float*, const float*) {
  // stub
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::gemm(
    const __nv_fp8_e4m3*, int, const __nv_fp8_e4m3*, int, __nv_bfloat16*, int, int, int, int,
    const float*, const float*, cudaStream_t) {
  // stub
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::moeGemm(
    void*, const void*, const void*, const int64_t*, size_t, size_t, size_t, cudaStream_t,
    const float*, const float*) {}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::strideBatchGemm(
    __nv_bfloat16*, int, int, __nv_fp8_e4m3*, int, int, __nv_fp8_e4m3*, int, int, int, int, int,
    int, cudaStream_t, float*, int, float*) {}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS1x128(
    __nv_fp8_e4m3*, float*, const __nv_bfloat16*, int, int, cudaStream_t) {}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS1x128Reshape(
    __nv_fp8_e4m3*, float*, const __nv_bfloat16*, int, int, int, int, cudaStream_t) {}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS128x128(
    __nv_fp8_e4m3*, float*, const __nv_bfloat16*, int, int, cudaStream_t) {}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3,
                                      __nv_bfloat16>::getWorkspaceSizeBase(size_t, size_t, size_t,
                                                                           size_t) {
  return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3,
                                      __nv_bfloat16>::getWorkspaceSize(size_t, size_t, size_t,
                                                                       size_t, size_t) {
  return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getFP8DataSize(
    int, int, bool) {
  return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getActScaleSize(
    int, int) {
  return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3,
                                      __nv_bfloat16>::getWeightScaleSize(int, int) {
  return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3,
                                      __nv_bfloat16>::getActWorkspaceSize(int, int) {
  return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3,
                                      __nv_bfloat16>::getWeightWorkspaceSize(int, int) {
  return 0;
}

// Add other method stubs if linker demands more

}  // namespace fp8_blockscale_gemm
}  // namespace kernels
}  // namespace tensorrt_llm
