#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 64, 256, 1, 1, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 64, 256, 1, 1, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 128, 256, 1, 2, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 128, 256, 1, 2, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 128, 256, 1, 1, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 128, 256, 1, 1, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, half, EpilogueOpDefault, NONE,
                                          128, 256, 256, 1, 2, 1, false, false);

#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, SafeFP4, SafeFP4, __nv_bfloat16, EpilogueOpDefault,
                                          NONE, 128, 256, 256, 1, 2, 1, false, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, half, EpilogueOpDefault,
                                          NONE, 128, 64, 128, 1, 1, 1, true, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 64, 128, 1, 1, 1, true,
                                          false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, half, EpilogueOpDefault,
                                          NONE, 128, 128, 128, 1, 2, 1, true, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 1, 2, 1, true,
                                          false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, half, EpilogueOpDefault,
                                          NONE, 128, 128, 128, 1, 1, 1, true, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 128, 128, 1, 1, 1, true,
                                          false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, half, EpilogueOpDefault,
                                          NONE, 128, 256, 128, 1, 2, 1, true, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 1, 2, 1, true,
                                          false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, half, EpilogueOpDefault,
                                          NONE, 128, 256, 128, 1, 1, 1, true, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, half, EpilogueOpDefault,
                                          NONE, 256, 256, 128, 2, 1, 1, true, false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 128, 256, 128, 1, 1, 1, true,
                                          false);

#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
                                          EpilogueOpDefault, NONE, 256, 256, 128, 2, 1, 1, true,
                                          false);

#endif

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm
