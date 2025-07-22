#include "tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 32,
                                          64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 64,
                                          64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 64,
                                          64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 128,
                                          64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 128,
                                          64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 256,
                                          64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 256,
                                          64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 64, 512,
                                          64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 32, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 64, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 64, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 128, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 128, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 256, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 256, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 512, 64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          32, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          64, 32, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          64, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          128, 32, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          128, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          256, 32, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          256, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 64,
                                          512, 32, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 8, 256, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 8, 256, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 16, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 16, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 32, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 32, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 64, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 64, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 64, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 64, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 128, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 128, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 128, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 128, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 256, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 256, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 256, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 256, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 64, 512, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 64, 512, 128, 1, 2, 1, false);

#endif

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm
