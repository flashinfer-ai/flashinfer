#include "tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128, 64,
                                          64, 2, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          128, 64, 2, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          128, 64, 2, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          256, 64, 2, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          256, 64, 2, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          512, 64, 2, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128, 32,
                                          64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128, 64,
                                          64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128, 64,
                                          64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          128, 64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          128, 64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          256, 64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          256, 64, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, half, half, half, EpilogueOpDefault, NONE, 128,
                                          512, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 64, 2, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 64, 2, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 64, 2, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 64, 2, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 64, 2, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 512, 64, 2, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 32, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 64, 1, 2, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 64, 1, 1, 1, false);

#endif

#if defined(ENABLE_BF16)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 512, 64, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          64, 32, 2, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          128, 32, 2, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          128, 32, 2, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          256, 32, 2, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          256, 32, 2, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          512, 32, 2, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          32, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          64, 32, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          64, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          128, 32, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          128, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          256, 32, 1, 2, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          256, 32, 1, 1, 1, false);

#endif

#if 1

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, float, float, float, EpilogueOpDefault, NONE, 128,
                                          512, 32, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 2, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 2, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 2, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 2, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 2, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 2, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 2, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 2, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 2, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 2, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 512, 128, 2, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 512, 128, 2, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 16, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 16, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 32, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 32, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, half,
                                          EpilogueOpDefault, NONE, 128, 512, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP8)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 512, 128, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 64, 256, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 64, 256, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 128, 256, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 128, 256, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 128, 256, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 128, 256, 1, 1, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 256, 256, 1, 2, 1, false);

#endif

#if defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 256, 256, 1, 2, 1, false);

#endif

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm
